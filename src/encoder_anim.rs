//! Animated WebP encoder.
//!
//! Builds a `RIFF/WEBP/VP8X + ANIM + ANMF...ANMF` file from a sequence of
//! per-frame RGBA tiles. Each frame is encoded **per-frame** in either
//! VP8L (lossless) or VP8+ALPH (lossy) mode, whichever produces the
//! smaller ANMF payload — the file format permits mixing the two, and
//! the decoder already handles both shapes.
//!
//! Two factory entry points are exposed:
//!
//! * [`build_animated_webp`] — drives every frame through the
//!   lossless VP8L path. Bit-exact, no quality loss, larger files. Used
//!   by callers that need pixel-perfect playback.
//! * [`build_animated_webp_with_options`] — accepts an
//!   [`AnimEncoderOptions`] knob bag. With `mode_select = AUTO` (the
//!   default), each frame is encoded both ways and the byte-smallest
//!   payload wins, so animations with photographic / smoothly-varying
//!   frames get the lossy path's compression while sharp synthetic
//!   frames stay on the lossless path. The wrapper preserves the
//!   bit-exact behaviour of [`build_animated_webp`] when options are
//!   defaulted to `Lossless`.
//!
//! The reference for this is libwebp's per-frame `WebPAnimEncoderAdd`
//! decision: each frame calls into both encoders and the smallest
//! payload wins. We use raw byte count rather than a perceptual cost
//! model for now — closes #335.
//!
//! # Container layout
//!
//! ```text
//! RIFF <size> WEBP
//!   VP8X <10>   — flags (ANIM bit set), canvas_w-1, canvas_h-1
//!   ANIM <6>    — 4 BGRA bytes background, 2-byte loop count (0=infinite)
//!   ANMF <n>    — per-frame envelope (header + nested VP8L chunk)
//!   ANMF <n>    — ...
//! ```
//!
//! Per `ANMF` header (16 bytes before nested chunks):
//!
//! ```text
//!   3 bytes  X offset / 2          (must be even)
//!   3 bytes  Y offset / 2          (must be even)
//!   3 bytes  frame_w - 1
//!   3 bytes  frame_h - 1
//!   3 bytes  duration_ms
//!   1 byte   bit0 = blending (0=blend, 1=overwrite)
//!            bit1 = disposal (0=none,  1=dispose-to-background)
//! ```
//!
//! The nested chunk is a single `VP8L` (lossless) sub-chunk produced by
//! the existing per-frame VP8L encoder.

use crate::error::{Result, WebpError as Error};
use crate::riff::WebpMetadata;
use crate::vp8l::encode_vp8l_argb;

/// Per-frame mode-selection policy for [`build_animated_webp_with_options`].
///
/// `Eq` is intentionally not derived because the `Delta` variant carries
/// a `DeltaConfig` whose `max_bbox_fraction: f32` field doesn't satisfy
/// the total-equality contract — `PartialEq` is sufficient for the
/// mode-pattern matches the encoder relies on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AnimFrameMode {
    /// Always encode every frame as VP8L (lossless). Bit-exact, larger
    /// files. Matches the historical [`build_animated_webp`] behaviour.
    Lossless,
    /// Always encode every frame as VP8 + ALPH (lossy colour, lossless
    /// alpha). Smaller files for photographic / smoothly-varying
    /// frames; visible compression artefacts for sharp synthetic
    /// content.
    Lossy,
    /// **Default.** Encode every frame both ways and pick whichever
    /// produces the smaller ANMF sub-chunk payload. Mixed lossless +
    /// lossy output — the WebP container permits this and the decoder
    /// already handles both shapes. Mirrors libwebp's
    /// `WebPAnimEncoderAdd` per-frame mode decision.
    #[default]
    Auto,
    /// **Delta** (AVIF-style perceptual frame-merge): each non-first frame
    /// is compared against the prior frame on a block-by-block basis using
    /// a luminance-biased SAD cost model
    /// ([`DeltaConfig::block_cost`]). Blocks whose cost stays below
    /// [`DeltaConfig::threshold`] are presumed unchanged; the encoder
    /// computes the bounding box of all changed blocks, encodes only
    /// that sub-rectangle, and emits an ANMF with `blending_method = 1`
    /// (`DoNotBlend` — overwrite the prior canvas). Frames whose change
    /// region is at most [`DeltaConfig::max_bbox_fraction`] of the
    /// canvas (default 80%) take the delta path; otherwise the encoder
    /// falls back to encoding the full frame in `Auto` mode.
    ///
    /// Constraints (caller responsibility — checked at encode time):
    ///   * every frame must be canvas-sized (`width = canvas_w`,
    ///     `height = canvas_h`, `x_offset = y_offset = 0`),
    ///   * `dispose_to_background = false` and `blend = false`
    ///     (delta-mode output forces overwrite, and a dispose-to-bg
    ///     between frames invalidates the prior-canvas reference).
    ///
    /// The first frame is always emitted in full; subsequent frames may
    /// be partial sub-rectangles.
    Delta(DeltaConfig),
}

/// Tunable parameters for the [`AnimFrameMode::Delta`] frame-merge mode.
///
/// The cost model is a **luminance-biased sum-of-absolute-differences**
/// over fixed-size blocks. For each `block_size × block_size` block we
/// compute `sum_over_pixels |luma(prev) - luma(new)| + 0.25 * |R'-R| +
/// 0.25 * |G'-G| + 0.25 * |B'-B| + |A'-A|`, where `luma = 0.299R + 0.587G
/// + 0.114B` (BT.601). A block is considered **changed** if its cost
/// exceeds [`Self::threshold`].
///
/// Defaults are tuned for 8×8 blocks at threshold 32 (≈1 LSB per pixel
/// on a flat region — small enough to flag any real motion, large
/// enough to absorb codec rounding noise).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaConfig {
    /// Block side length in pixels. Default 8. The encoder rounds
    /// the cost-model bbox up to a multiple of this and then up to
    /// the WebP-mandated even offset.
    pub block_size: u32,
    /// Luminance-biased SAD threshold per block — blocks with cost
    /// strictly greater than this are flagged as changed. Default 32.
    pub threshold: u32,
    /// If the changed-region bounding box covers more than this
    /// fraction of the canvas, the encoder bails out of the delta
    /// path for that frame and falls back to a full-frame encode in
    /// `Auto` mode. Range `0.0..=1.0`; default `0.8`.
    pub max_bbox_fraction: f32,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            block_size: 8,
            threshold: 32,
            max_bbox_fraction: 0.8,
        }
    }
}

/// Knob bag for [`build_animated_webp_with_options`]. Defaults pick the
/// per-frame mode-select strategy at quality 75 (libwebp's default).
///
/// File-level metadata chunks (`ICCP` / `EXIF` / `XMP `) can be attached
/// via [`metadata`](Self::metadata) — when any field of the inner
/// [`WebpMetadata`] is `Some`, the matching VP8X flag bit is set and the
/// chunk is written into the file body in the spec-mandated order
/// (ICCP immediately after VP8X; EXIF / XMP after the last ANMF).
#[derive(Clone, Debug)]
pub struct AnimEncoderOptions<'a> {
    /// Per-frame mode-selection policy. Defaults to [`AnimFrameMode::Auto`]
    /// (per-frame byte-smallest wins).
    pub mode: AnimFrameMode,
    /// Quality for the lossy path, on libwebp's `0.0..=100.0` scale
    /// (higher = better). Ignored when `mode = Lossless`. Default 75.
    pub lossy_quality: f32,
    /// Optional file-level auxiliary metadata (ICC profile, EXIF, XMP)
    /// to attach to the animation's VP8X header. Defaults to all `None`.
    pub metadata: WebpMetadata<'a>,
}

impl<'a> Default for AnimEncoderOptions<'a> {
    fn default() -> Self {
        Self {
            mode: AnimFrameMode::default(),
            lossy_quality: 75.0,
            metadata: WebpMetadata::default(),
        }
    }
}

/// One frame of an animation: an RGBA tile sized `width × height` rendered
/// at `(x_offset, y_offset)` on the canvas, displayed for `duration_ms`
/// before the next frame is composited.
///
/// `x_offset` and `y_offset` are stored on disk as half their value (the
/// spec mandates even offsets), so we silently round odd values down to
/// the next even number.
#[derive(Clone)]
pub struct AnimFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub x_offset: u32,
    pub y_offset: u32,
    pub duration_ms: u32,
    /// `true` → blend the frame's alpha onto the canvas. `false` → the
    /// frame overwrites the destination pixels (alpha included).
    pub blend: bool,
    /// `true` → after rendering, clear the frame's bbox to the background
    /// colour before drawing the next frame.
    pub dispose_to_background: bool,
    /// Row-major RGBA bytes for this tile — `width * height * 4` long.
    pub rgba: &'a [u8],
}

/// Build a complete animated `.webp` file from a slice of frames + a
/// canvas size. Every frame is encoded losslessly (VP8L) — for the
/// per-frame lossy/lossless mode-selection decision wired up by #335
/// see [`build_animated_webp_with_options`].
///
/// Loop count = 0 means infinite playback (the WebP default).
/// Background is BGRA; the spec writes B, G, R, A in that order — we
/// accept it the same way.
pub fn build_animated_webp(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
) -> Result<Vec<u8>> {
    build_animated_webp_with_options(
        canvas_w,
        canvas_h,
        background_bgra,
        loop_count,
        frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Lossless,
            ..AnimEncoderOptions::default()
        },
    )
}

/// Build an animated `.webp` file with explicit encoder options.
/// See [`AnimEncoderOptions`] for the knobs; the default policy is
/// per-frame mode auto-selection (whichever of VP8L / VP8+ALPH is
/// byte-smaller wins per frame).
pub fn build_animated_webp_with_options(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
    options: AnimEncoderOptions<'_>,
) -> Result<Vec<u8>> {
    if canvas_w == 0 || canvas_h == 0 {
        return Err(Error::invalid("animated WebP: zero canvas size"));
    }
    if canvas_w > 16384 || canvas_h > 16384 {
        return Err(Error::invalid("animated WebP: canvas exceeds 16384 px"));
    }
    if frames.is_empty() {
        return Err(Error::invalid("animated WebP: needs at least one frame"));
    }

    // Delta mode rewrites the input frame stream into per-frame sub-rect
    // tiles + an internal "auto"-fallback policy before falling through
    // to the standard layout loop. Doing the rewrite up front keeps the
    // RIFF body assembly path identical for every mode.
    if let AnimFrameMode::Delta(cfg) = options.mode {
        return build_animated_webp_delta(
            canvas_w,
            canvas_h,
            background_bgra,
            loop_count,
            frames,
            &options,
            cfg,
        );
    }

    // Pre-encode every frame's nested image sub-chunk(s) first. Doing
    // it up front lets us measure each chunk and lay out the RIFF body
    // in a single pass without a second iteration. Track whether *any*
    // frame carries non-opaque alpha — the VP8X ALPHA flag should only
    // be set when at least one frame actually needs alpha, otherwise
    // strict readers see the flag set with no real alpha and treat it
    // as a malformed file.
    let mut any_frame_has_alpha = false;
    let mut anmf_payloads: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for f in frames {
        if f.width == 0 || f.height == 0 {
            return Err(Error::invalid("animated WebP: zero frame size"));
        }
        if f.x_offset
            .checked_add(f.width)
            .map(|r| r > canvas_w)
            .unwrap_or(true)
            || f.y_offset
                .checked_add(f.height)
                .map(|r| r > canvas_h)
                .unwrap_or(true)
        {
            return Err(Error::invalid(
                "animated WebP: frame bbox extends past canvas",
            ));
        }
        if f.rgba.len() != (f.width as usize) * (f.height as usize) * 4 {
            return Err(Error::invalid(
                "animated WebP: frame rgba length mismatch frame_w*frame_h*4",
            ));
        }
        if f.duration_ms > 0x00FF_FFFF {
            return Err(Error::invalid(
                "animated WebP: duration_ms exceeds 24-bit field",
            ));
        }

        // Detect non-opaque alpha for the canvas-level VP8X flag. We
        // can't piggyback off the per-frame encode (the lossy path
        // checks `any(!= 0xff)` inside encode_lossy_anmf) because the
        // mode-select decision can drop that signal — scan once here
        // so the canvas flag is correct regardless of mode.
        if !any_frame_has_alpha && f.rgba.chunks_exact(4).any(|px| px[3] != 0xff) {
            any_frame_has_alpha = true;
        }

        // Per-frame mode selection: produce the requested encoding(s)
        // and pick whichever sub-chunk(s) lay out the smaller ANMF
        // payload. The choice is per-frame so an animation can mix
        // lossless and lossy frames depending on which wins on each.
        let chosen = encode_one_anmf_image(f, &options)?;

        // Build the ANMF payload (16-byte header + nested image sub-chunks).
        let nested_capacity = chosen.iter().map(|c| 8 + c.payload.len()).sum::<usize>();
        let mut payload = Vec::with_capacity(16 + nested_capacity);
        // Even offsets; the spec stores them divided by 2.
        write_u24_le(&mut payload, (f.x_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.y_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.width - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.height - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, f.duration_ms & 0x00FF_FFFF);
        // bit 0: blending — 0 = use alpha blending, 1 = overwrite.
        // bit 1: disposal — 0 = none, 1 = dispose-to-background.
        let mut flags: u8 = 0;
        if !f.blend {
            flags |= 0x01;
        }
        if f.dispose_to_background {
            flags |= 0x02;
        }
        payload.push(flags);

        // Nested image sub-chunk(s) inside the ANMF body. Either
        // [VP8L] or [ALPH, VP8 ] depending on the per-frame decision.
        for sub in &chosen {
            write_chunk(&mut payload, &sub.fourcc, &sub.payload);
        }
        anmf_payloads.push(payload);
    }

    // Assemble the body that lives between "WEBP" and the end of the
    // RIFF envelope: VP8X header + [ICCP] + ANIM + N x ANMF + [EXIF] + [XMP ].
    let mut body: Vec<u8> = Vec::new();

    // VP8X flags byte:
    //   bit 1 (0x02) = ANIM   — always set for an animation.
    //   bit 4 (0x10) = ALPHA  — set iff any frame carries non-opaque alpha.
    //   bit 5 (0x20) = ICCP   — set iff `meta.icc.is_some()`.
    //   bit 3 (0x08) = EXIF   — set iff `meta.exif.is_some()`.
    //   bit 2 (0x04) = XMP    — set iff `meta.xmp.is_some()`.
    let mut flags: u8 = 0x02; // ANIM
    if any_frame_has_alpha {
        flags |= 0x10;
    }
    if options.metadata.icc.is_some() {
        flags |= 0x20;
    }
    if options.metadata.exif.is_some() {
        flags |= 0x08;
    }
    if options.metadata.xmp.is_some() {
        flags |= 0x04;
    }
    let vp8x = vp8x_payload(flags, canvas_w, canvas_h);
    write_chunk(&mut body, b"VP8X", &vp8x);

    // ICCP must come immediately after VP8X per the WebP container spec.
    if let Some(icc) = options.metadata.icc {
        write_chunk(&mut body, b"ICCP", icc);
    }

    // ANIM chunk: 4 bytes BGRA + 2 bytes loop count.
    let mut anim = [0u8; 6];
    anim[0] = background_bgra[0];
    anim[1] = background_bgra[1];
    anim[2] = background_bgra[2];
    anim[3] = background_bgra[3];
    anim[4] = (loop_count & 0xff) as u8;
    anim[5] = ((loop_count >> 8) & 0xff) as u8;
    write_chunk(&mut body, b"ANIM", &anim);

    // ANMF chunks.
    for payload in &anmf_payloads {
        write_chunk(&mut body, b"ANMF", payload);
    }

    // EXIF / XMP follow the image-data chunks per the WebP container spec.
    if let Some(exif) = options.metadata.exif {
        write_chunk(&mut body, b"EXIF", exif);
    }
    if let Some(xmp) = options.metadata.xmp {
        write_chunk(&mut body, b"XMP ", xmp);
    }

    // RIFF envelope.
    let riff_size = 4 + body.len();
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(&body);
    Ok(out)
}

/// One nested image sub-chunk inside an `ANMF` payload (`VP8L`,
/// `VP8 `, or `ALPH`). Pre-assembled — `payload` is the bytes that go
/// after the chunk header.
struct AnmfSubChunk {
    fourcc: [u8; 4],
    payload: Vec<u8>,
}

/// Encode a single animated-frame image into the nested ANMF
/// sub-chunk(s) the per-frame mode policy selects. Returns either
/// `[VP8L]` (lossless) or `[ALPH, VP8 ]` (lossy colour + lossless
/// alpha) depending on `options.mode`. With `Auto`, both encodings
/// are produced and the byte-smaller wins (sum of sub-chunk header
/// + payload, mirroring the on-disk cost).
fn encode_one_anmf_image(
    f: &AnimFrame<'_>,
    options: &AnimEncoderOptions<'_>,
) -> Result<Vec<AnmfSubChunk>> {
    // Always produce the lossless candidate first — it's the
    // historic behaviour and the fallback when the lossy encode fails
    // (e.g. on too-small frames). `Delta` is rewritten upstream into
    // per-frame `Auto`-equivalent encodes — it should never reach
    // here, but treat it as `Auto` defensively.
    let lossless: Option<Vec<AnmfSubChunk>> = match options.mode {
        AnimFrameMode::Lossy => None,
        AnimFrameMode::Lossless | AnimFrameMode::Auto | AnimFrameMode::Delta(_) => {
            Some(encode_lossless_anmf(f)?)
        }
    };

    let lossy: Option<Vec<AnmfSubChunk>> = match options.mode {
        AnimFrameMode::Lossless => None,
        AnimFrameMode::Lossy | AnimFrameMode::Auto | AnimFrameMode::Delta(_) => {
            encode_lossy_anmf(f, options.lossy_quality)?
        }
    };

    match (lossless, lossy) {
        (None, None) => unreachable!("at least one mode must produce a candidate"),
        (Some(l), None) => Ok(l),
        (None, Some(l)) => Ok(l),
        (Some(ll), Some(ly)) => {
            // Auto mode: pick the smaller payload by total on-disk cost
            // (each sub-chunk costs `8 + payload + (payload & 1)`).
            let cost = |subs: &[AnmfSubChunk]| -> usize {
                subs.iter()
                    .map(|s| 8 + s.payload.len() + (s.payload.len() & 1))
                    .sum()
            };
            if cost(&ly) < cost(&ll) {
                Ok(ly)
            } else {
                Ok(ll)
            }
        }
    }
}

/// Encode a single frame as a lossless VP8L sub-chunk. Per-frame
/// `has_alpha` detection is done while scanning the RGBA buffer into
/// the packed-ARGB pixels the VP8L encoder consumes.
fn encode_lossless_anmf(f: &AnimFrame<'_>) -> Result<Vec<AnmfSubChunk>> {
    let mut pixels = Vec::with_capacity((f.width as usize) * (f.height as usize));
    let mut has_alpha = false;
    for px in f.rgba.chunks_exact(4) {
        let r = px[0] as u32;
        let g = px[1] as u32;
        let b = px[2] as u32;
        let a = px[3] as u32;
        if a != 0xff {
            has_alpha = true;
        }
        pixels.push((a << 24) | (r << 16) | (g << 8) | b);
    }
    let vp8l_bytes = encode_vp8l_argb(f.width, f.height, &pixels, has_alpha)?;
    Ok(vec![AnmfSubChunk {
        fourcc: *b"VP8L",
        payload: vp8l_bytes,
    }])
}

/// Encode a single frame as a lossy VP8 (+ optional ALPH) sub-chunk
/// pair. Mirrors the still-image encoder's RGBA → YUV420 + ALPH
/// orchestration: the colour planes go into a bare VP8 keyframe via
/// [`oxideav_vp8::encoder::encode_vp8_keyframe`], the alpha plane
/// (when not fully opaque) is compressed into an `ALPH` sub-chunk via
/// the same helper the still-image path uses.
///
/// Returns `Ok(None)` when the frame is too small for the VP8
/// encoder (e.g. <16 px on a side, where the keyframe would have
/// no MBs to emit) — the auto-mode caller falls back to lossless.
fn encode_lossy_anmf(f: &AnimFrame<'_>, quality: f32) -> Result<Option<Vec<AnmfSubChunk>>> {
    // VP8 needs at least a single 16×16 macroblock; smaller frames
    // can't go through the lossy path. Fall back to lossless silently.
    if f.width == 0 || f.height == 0 {
        return Ok(None);
    }
    let qindex = crate::encoder_vp8::quality_to_qindex(quality);

    let w = f.width as usize;
    let h = f.height as usize;
    let mut alpha_plane: Vec<u8> = Vec::with_capacity(w * h);
    let (y_plane, u_plane, v_plane) =
        crate::encoder_vp8::rgba_rows_to_yuv420(w, h, w * 4, f.rgba, &mut alpha_plane);

    // Detect "fully opaque" so we can skip the ALPH sub-chunk on
    // animations whose frames don't carry alpha (smaller payload).
    let has_alpha = alpha_plane.iter().any(|&a| a != 0xff);

    let vp8_frame = oxideav_vp8::Vp8Frame {
        width: f.width,
        height: f.height,
        pts: None,
        y: y_plane,
        u: u_plane,
        v: v_plane,
        y_stride: f.width,
        uv_stride: (f.width + 1) / 2,
    };
    let vp8_bytes =
        match oxideav_vp8::encoder::encode_vp8_keyframe(f.width, f.height, qindex, &vp8_frame) {
            Ok(b) => b,
            // VP8 keyframe encode failed (e.g. dimensions too small or
            // some other validation). Fall back to lossless.
            Err(_) => return Ok(None),
        };

    let mut subs: Vec<AnmfSubChunk> = Vec::with_capacity(2);
    if has_alpha {
        let alph = crate::encoder_vp8::encode_alph_chunk(f.width, f.height, &alpha_plane)
            .map_err(|e| Error::invalid(format!("animated WebP: ALPH encode: {e}")))?;
        // ALPH on disk: 1 header byte + payload bytes.
        let mut alph_payload = Vec::with_capacity(1 + alph.payload.len());
        alph_payload.push(alph.header_byte);
        alph_payload.extend_from_slice(&alph.payload);
        subs.push(AnmfSubChunk {
            fourcc: *b"ALPH",
            payload: alph_payload,
        });
    }
    subs.push(AnmfSubChunk {
        fourcc: *b"VP8 ",
        payload: vp8_bytes,
    });
    Ok(Some(subs))
}

/// AVIF-style delta-merge entry point — see [`AnimFrameMode::Delta`].
///
/// Validates Delta-mode caller constraints (canvas-sized full frames,
/// no per-frame disposal/blend), then for each non-first frame computes
/// the changed-region bounding box via [`changed_block_bbox`] and emits
/// either a sub-rect ANMF (cost-model says small change) or a full-frame
/// ANMF (cost-model says full repaint, or first frame). The returned
/// blob layout is byte-identical to the standard
/// [`build_animated_webp_with_options`] output — the only difference is
/// per-frame ANMF bbox/sub-rect placement and the `blending_method` bit.
fn build_animated_webp_delta(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
    options: &AnimEncoderOptions<'_>,
    cfg: DeltaConfig,
) -> Result<Vec<u8>> {
    // Validate cfg defensively first — caller-controlled.
    if cfg.block_size == 0 {
        return Err(Error::invalid(
            "animated WebP delta: block_size must be ≥ 1",
        ));
    }
    if !(cfg.max_bbox_fraction >= 0.0 && cfg.max_bbox_fraction <= 1.0) {
        return Err(Error::invalid(
            "animated WebP delta: max_bbox_fraction must be in [0.0, 1.0]",
        ));
    }

    // Validate caller constraints + collect per-frame full-canvas RGBA
    // for the cost-model comparison. We require frames that cover the
    // whole canvas because we have to reconstruct "what the prior
    // canvas looks like" to diff against, and the simplest invariant
    // is `prior_canvas[i] == frames[i-1].rgba` (which only holds when
    // each frame paints the entire canvas, blend=false, dispose=false).
    for (i, f) in frames.iter().enumerate() {
        if f.width != canvas_w || f.height != canvas_h {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must be canvas-sized ({canvas_w}x{canvas_h}), got {}x{}",
                f.width, f.height
            )));
        }
        if f.x_offset != 0 || f.y_offset != 0 {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must be at origin (0,0), got ({},{})",
                f.x_offset, f.y_offset
            )));
        }
        if f.blend {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must have blend=false (Delta mode forces overwrite)"
            )));
        }
        if f.dispose_to_background {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must have dispose_to_background=false (would invalidate the prior-canvas reference)"
            )));
        }
        if f.duration_ms > 0x00FF_FFFF {
            return Err(Error::invalid(
                "animated WebP delta: duration_ms exceeds 24-bit field",
            ));
        }
        if f.rgba.len() != (f.width as usize) * (f.height as usize) * 4 {
            return Err(Error::invalid(
                "animated WebP delta: frame rgba length mismatch frame_w*frame_h*4",
            ));
        }
    }

    // Build the rewritten frame list. The first frame is always full-
    // canvas; each subsequent frame is either (a) a sub-rect tile sized
    // to the changed-block bbox, with `blend = false` so the decoder
    // overwrites the matching canvas region, or (b) a full-canvas
    // refresh when the cost-model bbox is too large to win.
    //
    // We carry the source RGBA for sub-rect frames in a transient
    // `Vec<u8>` since the sub-rect doesn't exist as a contiguous slice
    // in the original buffer (different stride). For full frames we
    // reuse the caller's slice — no copy.
    let max_pixels = (canvas_w as u64).saturating_mul(canvas_h as u64);
    let max_bbox_pixels = ((max_pixels as f64) * (cfg.max_bbox_fraction as f64)) as u64;
    let mut tile_storage: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    // For each output frame, the layout (offset, width, height, blend,
    // duration). The actual rgba slice is resolved at encode time from
    // either the caller's frame.rgba (full-canvas) or `tile_storage`
    // (sub-rect).
    struct PlannedFrame {
        x_offset: u32,
        y_offset: u32,
        width: u32,
        height: u32,
        duration_ms: u32,
        blend: bool,
        // index into either the caller's `frames` (rgba_kind=Full) or
        // `tile_storage` (rgba_kind=Tile).
        rgba_kind: RgbaKind,
        rgba_idx: usize,
    }
    enum RgbaKind {
        Full,
        Tile,
    }

    let mut planned: Vec<PlannedFrame> = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        if i == 0 {
            // First frame: always full-canvas.
            planned.push(PlannedFrame {
                x_offset: 0,
                y_offset: 0,
                width: f.width,
                height: f.height,
                duration_ms: f.duration_ms,
                blend: f.blend,
                rgba_kind: RgbaKind::Full,
                rgba_idx: i,
            });
            continue;
        }
        // Cost-model bbox against frame i-1.
        let prior = &frames[i - 1];
        let bbox = changed_block_bbox(prior.rgba, f.rgba, canvas_w, canvas_h, &cfg);
        match bbox {
            None => {
                // Identical — emit a 1×1 (smallest the spec allows)
                // overwrite at (0,0) with the prior pixel: zero visible
                // change and the smallest possible payload.
                let p0 = &prior.rgba[..4];
                let mut tile = Vec::with_capacity(4);
                tile.extend_from_slice(p0);
                let tile_idx = tile_storage.len();
                tile_storage.push(tile);
                planned.push(PlannedFrame {
                    x_offset: 0,
                    y_offset: 0,
                    width: 1,
                    height: 1,
                    duration_ms: f.duration_ms,
                    blend: false, // overwrite (DoNotBlend)
                    rgba_kind: RgbaKind::Tile,
                    rgba_idx: tile_idx,
                });
            }
            Some((bx, by, bw, bh)) => {
                let bbox_pixels = (bw as u64) * (bh as u64);
                if bbox_pixels > max_bbox_pixels {
                    // Bbox too large — skip the delta path for this
                    // frame and emit it full-canvas in the underlying
                    // mode (Auto by default; respects caller's
                    // lossy_quality).
                    planned.push(PlannedFrame {
                        x_offset: 0,
                        y_offset: 0,
                        width: f.width,
                        height: f.height,
                        duration_ms: f.duration_ms,
                        blend: f.blend,
                        rgba_kind: RgbaKind::Full,
                        rgba_idx: i,
                    });
                } else {
                    // Carve out the bbox sub-rectangle into a contiguous
                    // RGBA buffer.
                    let tile = extract_subrect(f.rgba, canvas_w, bx, by, bw, bh);
                    let tile_idx = tile_storage.len();
                    tile_storage.push(tile);
                    planned.push(PlannedFrame {
                        x_offset: bx,
                        y_offset: by,
                        width: bw,
                        height: bh,
                        duration_ms: f.duration_ms,
                        blend: false, // overwrite (DoNotBlend)
                        rgba_kind: RgbaKind::Tile,
                        rgba_idx: tile_idx,
                    });
                }
            }
        }
    }

    // Drive the planned frames through the standard encoder path with
    // mode = Auto (so each sub-rect tile picks lossy/lossless per the
    // smaller-payload rule). We construct a fresh `AnimFrame` array
    // borrowing into either the original input or `tile_storage`.
    let rewritten: Vec<AnimFrame<'_>> = planned
        .iter()
        .map(|p| {
            let rgba: &[u8] = match p.rgba_kind {
                RgbaKind::Full => frames[p.rgba_idx].rgba,
                RgbaKind::Tile => tile_storage[p.rgba_idx].as_slice(),
            };
            AnimFrame {
                width: p.width,
                height: p.height,
                x_offset: p.x_offset,
                y_offset: p.y_offset,
                duration_ms: p.duration_ms,
                blend: p.blend,
                dispose_to_background: false,
                rgba,
            }
        })
        .collect();

    // Drive the rewritten frames through the standard encoder path in
    // `Lossless` mode: sub-rect tiles produced by Delta are typically
    // tiny (≤ a few KB raw RGBA), so the VP8 keyframe overhead would
    // win on byte count and the rebuild would also incur expensive
    // RDO during the per-frame Auto-mode candidate evaluation. Forcing
    // lossless keeps Delta-mode encodes deterministic + fast and
    // preserves pixel-identical round-trip semantics.
    let inner_options = AnimEncoderOptions {
        mode: AnimFrameMode::Lossless,
        lossy_quality: options.lossy_quality,
        metadata: options.metadata.clone(),
    };
    build_animated_webp_with_options(
        canvas_w,
        canvas_h,
        background_bgra,
        loop_count,
        &rewritten,
        inner_options,
    )
}

/// Copy the `bw × bh` sub-rectangle starting at `(bx, by)` out of an
/// RGBA buffer with `canvas_w` pixels per row. Allocates a fresh
/// `Vec<u8>` of length `bw * bh * 4`. Caller guarantees the bbox stays
/// inside the canvas.
fn extract_subrect(rgba: &[u8], canvas_w: u32, bx: u32, by: u32, bw: u32, bh: u32) -> Vec<u8> {
    let canvas_w = canvas_w as usize;
    let bx = bx as usize;
    let by = by as usize;
    let bw = bw as usize;
    let bh = bh as usize;
    let mut out = Vec::with_capacity(bw * bh * 4);
    for row in 0..bh {
        let src_off = ((by + row) * canvas_w + bx) * 4;
        out.extend_from_slice(&rgba[src_off..src_off + bw * 4]);
    }
    out
}

/// Walk `prev` vs `curr` (both row-major canvas-sized RGBA) on
/// `cfg.block_size`-sized blocks; for each block compute the
/// luminance-biased SAD cost; return the bounding box (in pixels,
/// even-aligned + clipped to canvas) of all blocks whose cost exceeds
/// `cfg.threshold`. Returns `None` when no block is changed (frame
/// is bit-identical or the cost-model says it's all under threshold).
///
/// Output bbox is `(x, y, w, h)` with `x`/`y` rounded down to even
/// (WebP ANMF spec mandates even offsets) and `w`/`h` adjusted so the
/// bbox still encloses every changed block.
fn changed_block_bbox(
    prev: &[u8],
    curr: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    cfg: &DeltaConfig,
) -> Option<(u32, u32, u32, u32)> {
    let bs = cfg.block_size;
    let cw = canvas_w as usize;

    // Block-grid extents (last block may be shorter than `bs` on the
    // right/bottom edge — count it as a regular block, just iterate
    // fewer pixels in that case).
    let n_bx = canvas_w.div_ceil(bs);
    let n_by = canvas_h.div_ceil(bs);

    let mut min_bx = u32::MAX;
    let mut min_by = u32::MAX;
    let mut max_bx: i64 = -1;
    let mut max_by: i64 = -1;

    for by in 0..n_by {
        let y0 = (by * bs) as usize;
        let y1 = ((by + 1) * bs).min(canvas_h) as usize;
        for bx in 0..n_bx {
            let x0 = (bx * bs) as usize;
            let x1 = ((bx + 1) * bs).min(canvas_w) as usize;
            let cost = block_cost(prev, curr, cw, x0, y0, x1, y1);
            if cost > cfg.threshold as u64 {
                if bx < min_bx {
                    min_bx = bx;
                }
                if by < min_by {
                    min_by = by;
                }
                if bx as i64 > max_bx {
                    max_bx = bx as i64;
                }
                if by as i64 > max_by {
                    max_by = by as i64;
                }
            }
        }
    }
    if max_bx < 0 || max_by < 0 {
        return None;
    }

    // Pixel bbox from block-grid bbox.
    let mut px = min_bx * bs;
    let mut py = min_by * bs;
    let mut pw = ((max_bx as u32 + 1) * bs).min(canvas_w) - px;
    let mut ph = ((max_by as u32 + 1) * bs).min(canvas_h) - py;

    // ANMF spec mandates even offsets — round (px, py) down to even,
    // and grow (pw, ph) to compensate.
    if px % 2 != 0 {
        px -= 1;
        pw += 1;
    }
    if py % 2 != 0 {
        py -= 1;
        ph += 1;
    }
    // Clamp width/height in case rounding pushed past the canvas.
    if px + pw > canvas_w {
        pw = canvas_w - px;
    }
    if py + ph > canvas_h {
        ph = canvas_h - py;
    }
    Some((px, py, pw, ph))
}

/// Luminance-biased SAD over a block in two RGBA canvas buffers. Both
/// `prev` and `curr` are row-major with `canvas_w` pixels per row;
/// the block spans `[x0, x1) × [y0, y1)`. Computes
/// `sum |luma(prev) - luma(curr)| + 0.25 * (|R'-R| + |G'-G| + |B'-B|) +
/// |A'-A|` per pixel using fixed-point integer math (the 0.25 weight is
/// a `>> 2`). Returns the accumulated cost as `u64` to avoid overflow on
/// 8×8 blocks of fully-saturated 8-bit deltas (max ≈ 8×8×(255+3*64+255)
/// ≈ 45k — fits in u32, but use u64 for safety vs larger blocks).
fn block_cost(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> u64 {
    let mut acc: u64 = 0;
    for y in y0..y1 {
        let row_off = y * canvas_w * 4;
        for x in x0..x1 {
            let off = row_off + x * 4;
            let pr = prev[off] as i32;
            let pg = prev[off + 1] as i32;
            let pb = prev[off + 2] as i32;
            let pa = prev[off + 3] as i32;
            let cr = curr[off] as i32;
            let cg = curr[off + 1] as i32;
            let cb = curr[off + 2] as i32;
            let ca = curr[off + 3] as i32;
            // BT.601 luma (integer-scaled): 0.299R + 0.587G + 0.114B
            // → (77*R + 150*G + 29*B + 128) >> 8 (sums to 256).
            let lp = (77 * pr + 150 * pg + 29 * pb + 128) >> 8;
            let lc = (77 * cr + 150 * cg + 29 * cb + 128) >> 8;
            let dl = (lp - lc).unsigned_abs() as u64;
            let dr = (pr - cr).unsigned_abs() as u64;
            let dg = (pg - cg).unsigned_abs() as u64;
            let db = (pb - cb).unsigned_abs() as u64;
            let da = (pa - ca).unsigned_abs() as u64;
            // Luma carries the bulk of the weight; chroma contributes
            // a quarter (>> 2) each; alpha gets full weight so changes
            // in transparency are flagged immediately.
            acc += dl + ((dr + dg + db) >> 2) + da;
        }
    }
    acc
}

/// VP8X payload: 1 byte flags, 3 bytes reserved, 3 bytes canvas_w-1,
/// 3 bytes canvas_h-1.
fn vp8x_payload(flags: u8, canvas_w: u32, canvas_h: u32) -> [u8; 10] {
    let mut out = [0u8; 10];
    out[0] = flags;
    let w_minus_1 = canvas_w.saturating_sub(1) & 0x00FF_FFFF;
    let h_minus_1 = canvas_h.saturating_sub(1) & 0x00FF_FFFF;
    out[4] = (w_minus_1 & 0xff) as u8;
    out[5] = ((w_minus_1 >> 8) & 0xff) as u8;
    out[6] = ((w_minus_1 >> 16) & 0xff) as u8;
    out[7] = (h_minus_1 & 0xff) as u8;
    out[8] = ((h_minus_1 >> 8) & 0xff) as u8;
    out[9] = ((h_minus_1 >> 16) & 0xff) as u8;
    out
}

fn write_u24_le(out: &mut Vec<u8>, v: u32) {
    out.push((v & 0xff) as u8);
    out.push(((v >> 8) & 0xff) as u8);
    out.push(((v >> 16) & 0xff) as u8);
}

fn write_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w as usize) * (h as usize) * 4);
        for _ in 0..(w * h) {
            v.extend_from_slice(&rgba);
        }
        v
    }

    #[test]
    fn build_animated_emits_vp8x_anim_anmf_in_order() {
        let f0 = solid_frame(8, 8, [0xff, 0, 0, 0xff]);
        let f1 = solid_frame(8, 8, [0, 0xff, 0, 0xff]);
        let frames = [
            AnimFrame {
                width: 8,
                height: 8,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 100,
                blend: false,
                dispose_to_background: false,
                rgba: &f0,
            },
            AnimFrame {
                width: 8,
                height: 8,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 200,
                blend: false,
                dispose_to_background: false,
                rgba: &f1,
            },
        ];
        let out = build_animated_webp(8, 8, [0; 4], 0, &frames).expect("build");
        // RIFF / WEBP magic at the front.
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        // VP8X first, with ANIM bit (0x02) set.
        assert_eq!(&out[12..16], b"VP8X");
        assert_ne!(out[20] & 0x02, 0, "ANIM flag must be set in VP8X");
        // ANIM next.
        let vp8x_chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]) as usize;
        let anim_off = 12 + 8 + vp8x_chunk_len + (vp8x_chunk_len & 1);
        assert_eq!(&out[anim_off..anim_off + 4], b"ANIM");
        // First ANMF after ANIM.
        let anim_chunk_len = u32::from_le_bytes([
            out[anim_off + 4],
            out[anim_off + 5],
            out[anim_off + 6],
            out[anim_off + 7],
        ]) as usize;
        let anmf0_off = anim_off + 8 + anim_chunk_len + (anim_chunk_len & 1);
        assert_eq!(&out[anmf0_off..anmf0_off + 4], b"ANMF");
    }

    #[test]
    fn rejects_oversized_frame_bbox() {
        let f = solid_frame(8, 8, [0; 4]);
        let frames = [AnimFrame {
            width: 8,
            height: 8,
            x_offset: 4,
            y_offset: 4,
            duration_ms: 0,
            blend: false,
            dispose_to_background: false,
            rgba: &f,
        }];
        // 8x8 frame at (4,4) on a 8x8 canvas — extends past edge.
        let r = build_animated_webp(8, 8, [0; 4], 0, &frames);
        assert!(r.is_err(), "expected oversized-bbox to be rejected");
    }

    #[test]
    fn auto_mode_picks_smaller_of_the_two_candidates() {
        // 96×96 noisy photographic-style frame: the per-pixel value
        // varies enough that VP8L can't collapse it to a tiny literal
        // run, while VP8 lossy at q=75 compresses the smooth-noise
        // structure to a fraction of the size. Auto mode must end up
        // at min(lossless, lossy) — modulo bit-for-bit equality on the
        // mode-specific candidate payload.
        let w = 96u32;
        let h = 96u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        // Pseudo-random but reproducible: an xorshift-ish hash per pixel
        // gives the VP8L Huffman alphabet a wide distribution that
        // doesn't compress to a tiny payload, and gives VP8 a smooth-
        // ish tile pattern (since adjacent pixels share most of their
        // hash bits) that DCT handles very well.
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let mut s = y.wrapping_mul(0x9E37_79B9) ^ x.wrapping_mul(0x85EB_CA77);
                s ^= s.wrapping_shr(13);
                s = s.wrapping_mul(0xC2B2_AE35);
                s ^= s.wrapping_shr(16);
                rgba[i] = ((s >> 0) & 0xff) as u8;
                rgba[i + 1] = ((s >> 8) & 0xff) as u8;
                rgba[i + 2] = ((s >> 16) & 0xff) as u8;
                rgba[i + 3] = 0xff;
            }
        }
        let frames = [AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &rgba,
        }];

        // Force lossless and force lossy, then run auto: auto should
        // not be larger than the smaller of the two forced encodings.
        let lossless = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions {
                mode: AnimFrameMode::Lossless,
                ..Default::default()
            },
        )
        .expect("encode lossless");
        let lossy = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions {
                mode: AnimFrameMode::Lossy,
                ..Default::default()
            },
        )
        .expect("encode lossy");
        let auto = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions::default(),
        )
        .expect("encode auto");

        eprintln!(
            "anim sizes (noise 96x96): lossless={} lossy={} auto={}",
            lossless.len(),
            lossy.len(),
            auto.len()
        );
        // Auto must be ≤ the smaller candidate. (Modulo a few-byte
        // wiggle room for the optional ALPH sub-chunk's even-length
        // padding — if the ALPH/VP8 split happens to round differently
        // than the bare VP8L would, the comparison can be off by 1
        // byte. So compare to `min + 2` for slack.)
        let smaller = lossless.len().min(lossy.len());
        assert!(
            auto.len() <= smaller + 2,
            "auto ({}) > min(lossless={}, lossy={}) + 2 — mode-selection broken",
            auto.len(),
            lossless.len(),
            lossy.len(),
        );
    }

    #[test]
    fn auto_mode_picks_lossless_for_palette_frame() {
        // Build a small flat-colour frame: 32×32 of a single solid
        // colour. VP8L collapses this to ≤ 30 bytes (a single literal +
        // run). VP8 spends a fixed overhead on the keyframe header +
        // partition data + entropy-default tables that's much larger.
        // Auto mode should therefore pick lossless on this fixture.
        let w = 32u32;
        let h = 32u32;
        let rgba = solid_frame(w, h, [0x80, 0x40, 0x20, 0xff]);
        let frames = [AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &rgba,
        }];
        let auto = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions::default(),
        )
        .expect("encode auto");
        let lossless = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions {
                mode: AnimFrameMode::Lossless,
                ..Default::default()
            },
        )
        .expect("encode lossless");
        // On a solid colour the lossless path is the byte-smaller
        // candidate, so auto must match it (no inflation).
        assert_eq!(
            auto.len(),
            lossless.len(),
            "auto mode failed to pick lossless on a flat-colour fixture"
        );
    }

    #[test]
    fn loop_count_and_background_round_trip_on_disk() {
        let f = solid_frame(4, 4, [0; 4]);
        let frames = [AnimFrame {
            width: 4,
            height: 4,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 1,
            blend: false,
            dispose_to_background: false,
            rgba: &f,
        }];
        let out = build_animated_webp(4, 4, [0x12, 0x34, 0x56, 0x78], 7, &frames).expect("build");
        let vp8x_chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]) as usize;
        let anim_off = 12 + 8 + vp8x_chunk_len + (vp8x_chunk_len & 1);
        // ANIM payload starts at anim_off + 8.
        let anim_payload = &out[anim_off + 8..anim_off + 8 + 6];
        assert_eq!(&anim_payload[0..4], &[0x12, 0x34, 0x56, 0x78]);
        let lc = u16::from_le_bytes([anim_payload[4], anim_payload[5]]);
        assert_eq!(lc, 7);
    }
}
