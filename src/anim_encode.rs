//! Animation *encoder* — the published-0.1.5 `build_animated_webp` surface,
//! rebuilt clean-room on top of the in-crate §3.7 VP8L lossless encoder and
//! the §2.7.1.1 `ANIM` / `ANMF` container framing.
//!
//! Where [`crate::anim`] / [`crate::anmf`] *parse* the global and per-frame
//! animation headers, and [`crate::build`] frames a single still image, this
//! module assembles a **multi-frame** animated `.webp` from a list of
//! caller-supplied frames:
//!
//! ```text
//! RIFF | WEBP | VP8X(A flag) | [ICCP] | ANIM | ANMF… ANMF | [EXIF] | [XMP ]
//! ```
//!
//! Each `ANMF` carries the §2.7.1.1 Figure 9 16-byte header (frame X / Y /
//! width / height / duration plus the `Reserved|B|D` info byte) followed by
//! its "Frame Data" — a padded §2.3 sub-RIFF holding a single §2.6 `VP8L`
//! chunk for the [`AnimFrameMode::Lossless`] path. The bitstream itself is
//! produced by [`crate::vp8l_encode::encode_vp8l_argb_with`], so the encoded
//! file decodes back through [`crate::decode_webp`] (animation path) to the
//! exact input pixels.
//!
//! ## Auto / Delta encoding (round 127)
//!
//! [`AnimFrameMode::Lossless`] always emits the full caller-supplied frame
//! at its `(x, y)` offset as a single VP8L keyframe. [`AnimFrameMode::Delta`]
//! and [`AnimFrameMode::Auto`] take advantage of the §2.7.1.1
//! `B = 1` (overwrite) / `D = 0` (no dispose) ANMF semantics to encode only
//! the **dirty rectangle** of each frame against the previous canvas:
//!
//! * `Delta` always emits the dirty-rect sub-frame; pixels outside the
//!   dirty rect remain whatever the previous frame left on the canvas, so
//!   the §2.7.1.1 compositing rules naturally reconstruct the caller's
//!   intended canvas on decode. The sub-frame carries the **post-composite**
//!   pixels (the frame as drawn per its `blend` method), which keeps
//!   `AlphaBlend` frames bit-exact. Two cases fall back to a full keyframe
//!   with the caller's flags honoured verbatim: the first frame (no
//!   previous canvas to diff against) and any frame with
//!   `dispose == Background` (the §2.7.1.1 clear applies to the frame's
//!   declared rect, which a smaller dirty-rect ANMF cannot express).
//! * `Auto` evaluates both the dirty-rect sub-frame and the full-canvas
//!   keyframe and emits whichever produces a smaller VP8L bitstream
//!   (subject to the same two full-keyframe fallbacks).
//!
//! Both modes are **lossless** — every encoded byte round-trips through
//! [`crate::decode_webp`] to the exact caller-provided pixels, the same as
//! `Lossless`. The [`DeltaConfig`] / [`DownsampleKernel`] knobs are
//! preserved for API-shape compatibility but the dirty-rect algorithm
//! does not consult them yet (they were originally intended for a
//! lossy-aware MS-SSIM quality gate).

use crate::anmf::{BlendingMethod, DisposalMethod};
use crate::build::{self, Vp8xFlags};
use crate::container::fourcc;
use crate::vp8l_encode;
use crate::{Error, WebpError, WebpMetadata};

/// §2.7.1.1 Figure 9 fixed `ANMF` header length (5 × uint24 + 1 info byte).
const ANMF_HEADER_LEN: usize = 16;

/// §2.7.1.1 Figure 8 fixed `ANIM` payload length (uint32 bg + uint16 loop).
const ANIM_PAYLOAD_LEN: usize = 6;

/// How a single animation frame's pixels are compressed into its `ANMF`
/// "Frame Data" bitstream subchunk.
///
/// Reproduces the published-0.1.5 variant set. All three modes are wired
/// in this build; the round-127 `Auto` and `Delta` paths encode the
/// caller's full-canvas frame as a **lossless dirty-rect sub-frame**
/// against the previous canvas (the original lossy keyframe vs. inter-frame
/// delta choice is deferred until the `oxideav-vp8` lossy encoder is ready,
/// at which point `Auto` will also evaluate a lossy candidate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimFrameMode {
    /// Evaluate both the full-canvas VP8L keyframe and the dirty-rect VP8L
    /// sub-frame and emit whichever is smaller. Falls back to the
    /// full-canvas keyframe for the first frame and for frames whose
    /// dirty rect happens to cover the whole canvas.
    #[default]
    Auto,
    /// Always emit the dirty-rect sub-frame (the §2.7.1.1 `B = 1` / `D = 0`
    /// overwrite-no-dispose path). First frame is always the full canvas.
    Delta,
    /// Encode the frame as a standalone §2.6 `VP8L` lossless keyframe.
    Lossless,
}

/// A single animation frame to encode.
///
/// `pixels` is `width * height * 4` interleaved 8-bit `[R, G, B, A]` bytes in
/// scan-line order — the same flat layout [`crate::WebpFrame::rgba`] decodes
/// to. `x` / `y` place the frame's upper-left corner on the canvas (must be
/// even per §2.7.1.1, since the on-disk field is the coordinate / 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimFrame {
    /// `width * height * 4` interleaved `[R, G, B, A]` bytes, scan order.
    pub pixels: Vec<u8>,
    /// Frame width in pixels (≥ 1).
    pub width: u32,
    /// Frame height in pixels (≥ 1).
    pub height: u32,
    /// X coordinate of the frame's upper-left corner on the canvas. Must be
    /// even (§2.7.1.1 stores `x / 2`).
    pub x: u32,
    /// Y coordinate of the frame's upper-left corner on the canvas. Must be
    /// even (§2.7.1.1 stores `y / 2`).
    pub y: u32,
    /// Display duration in 1-millisecond units (the §2.7.1.1 `Frame
    /// Duration` field).
    pub duration: u32,
    /// §2.7.1.1 blending method (`B` bit).
    pub blend: BlendingMethod,
    /// §2.7.1.1 disposal method (`D` bit).
    pub dispose: DisposalMethod,
    /// Per-frame compression mode.
    pub mode: AnimFrameMode,
}

impl AnimFrame {
    /// Construct a top-left, **overwrite-blended**, non-disposed lossless
    /// frame from a flat RGBA buffer — the common case.
    ///
    /// `BlendingMethod::Overwrite` (§2.7.1.1 `B = 1`) is the default so a
    /// full-canvas frame round-trips byte-for-byte through
    /// [`crate::decode_webp`]'s canvas-compositing path. Callers that want
    /// §2.7.1.1 alpha-blending of a translucent sub-frame onto the existing
    /// canvas must build the struct literally and set `blend:
    /// BlendingMethod::AlphaBlend`.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>, duration: u32) -> Self {
        Self {
            pixels,
            width,
            height,
            x: 0,
            y: 0,
            duration,
            blend: BlendingMethod::Overwrite,
            dispose: DisposalMethod::None,
            mode: AnimFrameMode::Lossless,
        }
    }
}

/// Multi-scale SSIM downsample kernel selector for the (blocked) delta path.
///
/// Re-exposed for published-API shape compatibility. Has no effect on the
/// lossless path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DownsampleKernel {
    /// Plain box (average) downsample.
    #[default]
    Box,
    /// Gaussian-weighted downsample.
    Gaussian,
}

/// Tuning knobs for the inter-frame delta path.
///
/// Re-exposed for published-API shape compatibility. The fields feed the
/// (still blocked) [`AnimFrameMode::Delta`] / [`AnimFrameMode::Auto`] paths;
/// they have no effect on the lossless path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaConfig {
    /// Maximum number of disjoint dirty-rectangle components to keep when
    /// diffing two frames before falling back to a full keyframe.
    pub max_components: usize,
    /// Optional byte threshold below which an inner sub-rectangle delta is
    /// preferred. `None` disables the heuristic.
    pub auto_inner_threshold_bytes: Option<usize>,
    /// Which kernel to use when downsampling for the MS-SSIM quality gate.
    pub msssim_downsample_kernel: DownsampleKernel,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            max_components: 8,
            auto_inner_threshold_bytes: None,
            msssim_downsample_kernel: DownsampleKernel::Box,
        }
    }
}

impl DeltaConfig {
    /// Override the maximum dirty-rectangle component count (builder form).
    pub fn max_components_override(mut self, n: usize) -> Self {
        self.max_components = n;
        self
    }

    /// Set the inner-rectangle byte threshold (builder form).
    pub fn auto_inner_threshold_bytes(mut self, bytes: Option<usize>) -> Self {
        self.auto_inner_threshold_bytes = bytes;
        self
    }

    /// Select the MS-SSIM downsample kernel (builder form).
    pub fn msssim_downsample_kernel(mut self, kernel: DownsampleKernel) -> Self {
        self.msssim_downsample_kernel = kernel;
        self
    }
}

/// Options for [`build_animated_webp_with_options`].
///
/// `loop_count` is the §2.7.1.1 `ANIM` loop count (`0` = loop forever).
/// `background_rgba` is the `ANIM` background colour as `[R, G, B, A]`. The
/// borrowed [`WebpMetadata`] carries optional ICC / Exif / XMP payloads to
/// embed in the §2.7 chunk order. `delta` tunes the (blocked) delta path.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnimEncoderOptions<'a> {
    /// §2.7.1.1 `ANIM` loop count. `0` means "loop infinitely".
    pub loop_count: u16,
    /// §2.7.1.1 `ANIM` background colour as `[R, G, B, A]`.
    pub background_rgba: [u8; 4],
    /// File-level metadata (ICC / Exif / XMP) to embed, borrowed.
    pub metadata: WebpMetadata<'a>,
    /// Tuning for the (blocked) inter-frame delta path.
    pub delta: DeltaConfig,
}

/// Build an animated `.webp` from `frames`, defaulting all encoder options
/// (infinite loop, transparent-black background, no metadata).
///
/// Convenience wrapper over [`build_animated_webp_with_options`]; see that
/// function for the full semantics.
pub fn build_animated_webp(frames: &[AnimFrame]) -> Result<Vec<u8>, WebpError> {
    build_animated_webp_with_options(frames, &AnimEncoderOptions::default())
}

/// Assemble a complete animated `RIFF/WEBP` file from `frames` per
/// RFC 9649 §2.7.1.1.
///
/// Output layout:
///
/// ```text
/// RIFF | WEBP | VP8X(A[,L][,I][,E][,X]) | [ICCP] | ANIM | ANMF… | [EXIF] | [XMP ]
/// ```
///
/// The §2.7.1 `VP8X` canvas is sized to cover every frame
/// (`max(frame.x + frame.width)` × `max(frame.y + frame.height)`). The `A`
/// (animation) flag is always set; `L` (alpha) is set when any frame carries
/// a non-opaque pixel; `I` / `E` / `X` follow the supplied metadata.
///
/// Each frame's [`AnimFrameMode::Lossless`] pixels are encoded to a §2.6
/// `VP8L` chunk via [`crate::vp8l_encode::encode_vp8l_argb_with`] and
/// wrapped in the `ANMF` "Frame Data" sub-RIFF as-is.
///
/// **Round 127** (corrected round 279): [`AnimFrameMode::Delta`] encodes
/// only the dirty rectangle — the bounding box of canvas pixels the frame
/// actually changes once composited per its `blend` method — as the `ANMF`
/// sub-frame, carrying the post-composite pixels with `B = 1` (overwrite)
/// and `D = 0` (no dispose). The first frame, any frame with
/// `dispose == Background`, and any frame whose dirty rect happens to span
/// its whole declared rect, fall back to a full keyframe with the caller's
/// flags honoured verbatim. [`AnimFrameMode::Auto`] evaluates both
/// candidates and picks the smaller bitstream. Both modes round-trip
/// byte-for-byte through [`crate::decode_webp`]'s canvas compositor for
/// every blend/dispose combination.
///
/// An empty `frames` slice, a frame whose `pixels` length disagrees with
/// `width * height * 4`, or an odd `x` / `y` offset is
/// [`WebpError::InvalidData`].
pub fn build_animated_webp_with_options(
    frames: &[AnimFrame],
    opts: &AnimEncoderOptions<'_>,
) -> Result<Vec<u8>, WebpError> {
    if frames.is_empty() {
        return Err(WebpError::InvalidData);
    }

    // §2.7.1.1: canvas must cover every frame rectangle.
    let mut canvas_width = 0u32;
    let mut canvas_height = 0u32;
    let mut any_alpha = false;

    for f in frames {
        if f.width == 0 || f.height == 0 {
            return Err(WebpError::InvalidData);
        }
        // §2.7.1.1 stores Frame X / Frame Y as coord/2, so only even
        // offsets are representable.
        if f.x & 1 != 0 || f.y & 1 != 0 {
            return Err(WebpError::InvalidData);
        }
        let expected = (f.width as usize)
            .checked_mul(f.height as usize)
            .and_then(|n| n.checked_mul(4));
        if expected != Some(f.pixels.len()) {
            return Err(WebpError::InvalidData);
        }
        let right = f.x.checked_add(f.width).ok_or(WebpError::InvalidData)?;
        let bottom = f.y.checked_add(f.height).ok_or(WebpError::InvalidData)?;
        canvas_width = canvas_width.max(right);
        canvas_height = canvas_height.max(bottom);
        if f.pixels.chunks_exact(4).any(|px| px[3] != 0xff) {
            any_alpha = true;
        }
    }

    let meta = &opts.metadata;

    // §2.7.1 VP8X flag octet — animation always; alpha/metadata as present.
    let flags = Vp8xFlags {
        has_iccp: meta.icc.is_some(),
        has_alpha: any_alpha,
        has_exif: meta.exif.is_some(),
        has_xmp: meta.xmp.is_some(),
        has_animation: true,
    };
    let vp8x_payload = build::build_vp8x_chunk(canvas_width, canvas_height, flags).map_err(to_w)?;

    let mut body = Vec::new();
    let mut push = |fourcc, payload: &[u8]| -> Result<(), WebpError> {
        let chunk = build::build_chunk(fourcc, payload).map_err(to_w)?;
        body.extend_from_slice(&chunk);
        Ok(())
    };

    // §2.7 chunk order: VP8X, ICCP, ANIM, ANMF…, EXIF, XMP.
    push(fourcc::VP8X, &vp8x_payload)?;
    if let Some(icc) = meta.icc {
        push(fourcc::ICCP, icc)?;
    }
    push(fourcc::ANIM, &build_anim_payload(opts))?;

    // Track the canvas state the decoder will see right *before* this
    // iteration's frame is drawn. Initialised to the ANIM bg colour to
    // mirror the decoder's §2.7.1.1 "canvas is cleared at the start" rule.
    let mut prev_canvas = build_initial_canvas(canvas_width, canvas_height, opts.background_rgba);
    let bg_rgba = opts.background_rgba;
    // Per the decoder: before drawing each frame after the first, the
    // *previous* frame's dispose method is applied to *its* rect.
    let mut prev_disposal: Option<(u32, u32, u32, u32, DisposalMethod)> = None;

    for (idx, f) in frames.iter().enumerate() {
        // Apply previous frame's dispose to the predicted-canvas tracker.
        if let Some((px, py, pw, ph, DisposalMethod::Background)) = prev_disposal {
            fill_canvas_rect_in_place(&mut prev_canvas, canvas_width, px, py, pw, ph, bg_rgba);
        }
        let anmf_payload = build_anmf_payload_with_prev(f, canvas_width, &prev_canvas, idx == 0)?;
        push(fourcc::ANMF, &anmf_payload)?;
        // Update the canvas tracker with this frame's drawn pixels
        // (matching the decoder's blend method).
        composite_frame_onto_canvas(&mut prev_canvas, canvas_width, f);
        // Remember this frame's rect + dispose for the next iteration.
        prev_disposal = Some((f.x, f.y, f.width, f.height, f.dispose));
    }

    if let Some(exif) = meta.exif {
        push(fourcc::EXIF, exif)?;
    }
    if let Some(xmp) = meta.xmp {
        push(fourcc::XMP, xmp)?;
    }

    // §2.4 file framing: "RIFF" | File Size (= body + 4 for "WEBP") | "WEBP".
    let file_size = (body.len() as u64) + 4;
    if file_size > u64::from(u32::MAX) {
        return Err(WebpError::InvalidData);
    }
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&fourcc::RIFF);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&fourcc::WEBP);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Build a fresh canvas filled with `bg` per §2.7.1.1 — what the decoder
/// initialises the canvas to before drawing the first frame.
fn build_initial_canvas(width: u32, height: u32, bg: [u8; 4]) -> Vec<u8> {
    let pixels = (width as usize) * (height as usize);
    let mut canvas = Vec::with_capacity(pixels * 4);
    for _ in 0..pixels {
        canvas.extend_from_slice(&bg);
    }
    canvas
}

/// Fill the sub-rectangle `(x, y, w, h)` of `canvas` with `rgba` — mirrors
/// the decoder's `fill_canvas_rect` when applying a previous frame's
/// `DisposalMethod::Background`.
fn fill_canvas_rect_in_place(
    canvas: &mut [u8],
    canvas_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    rgba: [u8; 4],
) {
    let cw_bytes = (canvas_w as usize) * 4;
    for row in 0..(h as usize) {
        let off = ((y as usize) + row) * cw_bytes + (x as usize) * 4;
        for col in 0..(w as usize) {
            canvas[off + col * 4] = rgba[0];
            canvas[off + col * 4 + 1] = rgba[1];
            canvas[off + col * 4 + 2] = rgba[2];
            canvas[off + col * 4 + 3] = rgba[3];
        }
    }
}

/// Composite a frame onto `canvas` exactly the way the decoder will, so
/// the next iteration's dirty-rect diff is computed against the same
/// reference state. Honours the frame's `blend` method — `Overwrite`
/// (the default and the one Auto/Delta forces) copies the rect bytes
/// verbatim; `AlphaBlend` runs the §2.7.1.1 8-bit alpha-blending formula.
fn composite_frame_onto_canvas(canvas: &mut [u8], canvas_w: u32, f: &AnimFrame) {
    let cw = canvas_w as usize;
    let cw_bytes = cw * 4;
    let fw = f.width as usize;
    let fh = f.height as usize;
    let fx = f.x as usize;
    let fy = f.y as usize;
    match f.blend {
        BlendingMethod::Overwrite => {
            let src_stride = fw * 4;
            for row in 0..fh {
                let src_off = row * src_stride;
                let dst_off = (fy + row) * cw_bytes + fx * 4;
                canvas[dst_off..dst_off + src_stride]
                    .copy_from_slice(&f.pixels[src_off..src_off + src_stride]);
            }
        }
        BlendingMethod::AlphaBlend => {
            for row in 0..fh {
                for col in 0..fw {
                    let src_off = (row * fw + col) * 4;
                    let dst_off = (fy + row) * cw_bytes + (fx + col) * 4;
                    let sa = f.pixels[src_off + 3] as u32;
                    if sa == 255 {
                        canvas[dst_off..dst_off + 4]
                            .copy_from_slice(&f.pixels[src_off..src_off + 4]);
                    } else if sa == 0 {
                        // src fully transparent → leave dst.
                    } else {
                        let sr = f.pixels[src_off] as u32;
                        let sg = f.pixels[src_off + 1] as u32;
                        let sb = f.pixels[src_off + 2] as u32;
                        let dr = canvas[dst_off] as u32;
                        let dg = canvas[dst_off + 1] as u32;
                        let db = canvas[dst_off + 2] as u32;
                        let da = canvas[dst_off + 3] as u32;
                        let dst_factor = (da * (255 - sa) + 127) / 255;
                        let out_a = sa + dst_factor;
                        let out_r = (sr * sa + dr * dst_factor + out_a / 2)
                            .checked_div(out_a)
                            .unwrap_or(0);
                        let out_g = (sg * sa + dg * dst_factor + out_a / 2)
                            .checked_div(out_a)
                            .unwrap_or(0);
                        let out_b = (sb * sa + db * dst_factor + out_a / 2)
                            .checked_div(out_a)
                            .unwrap_or(0);
                        canvas[dst_off] = out_r.min(255) as u8;
                        canvas[dst_off + 1] = out_g.min(255) as u8;
                        canvas[dst_off + 2] = out_b.min(255) as u8;
                        canvas[dst_off + 3] = out_a.min(255) as u8;
                    }
                }
            }
        }
    }
}

/// Dirty-rect (bounding box of changed pixels) between the post-composite
/// `drawn` canvas and the `prev` canvas, restricted to `f`'s declared
/// rect (the only region a frame may touch) and expressed in canvas
/// coordinates. Diffing the *drawn* state — rather than the raw source
/// pixels — makes the rect correct for `AlphaBlend` frames, whose drawn
/// pixels differ from their source pixels. Returns `None` when no canvas
/// pixel changes — Delta/Auto then emit a degenerate 2×2 same-pixels rect
/// (the smallest representable ANMF) to preserve duration timing.
fn dirty_rect_canvas_coords(
    f: &AnimFrame,
    canvas_w: u32,
    prev: &[u8],
    drawn: &[u8],
) -> Option<DirtyRect> {
    let cw = canvas_w as usize;
    let cw_bytes = cw * 4;
    let fw = f.width as usize;
    let fh = f.height as usize;
    let fx = f.x as usize;
    let fy = f.y as usize;

    let mut min_x = usize::MAX;
    let mut min_y = usize::MAX;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut any_diff = false;

    for row in 0..fh {
        let dst_row_off = (fy + row) * cw_bytes + fx * 4;
        for col in 0..fw {
            let s = &drawn[dst_row_off + col * 4..dst_row_off + col * 4 + 4];
            let d = &prev[dst_row_off + col * 4..dst_row_off + col * 4 + 4];
            if s != d {
                any_diff = true;
                let cx = fx + col;
                let cy = fy + row;
                if cx < min_x {
                    min_x = cx;
                }
                if cy < min_y {
                    min_y = cy;
                }
                if cx > max_x {
                    max_x = cx;
                }
                if cy > max_y {
                    max_y = cy;
                }
            }
        }
    }

    if !any_diff {
        return None;
    }

    // §2.7.1.1 stores Frame X / Frame Y as coord/2, so the dirty rect's
    // top-left must be aligned to even coordinates. Round down to the
    // nearest even.
    let aligned_min_x = min_x & !1;
    let aligned_min_y = min_y & !1;

    Some(DirtyRect {
        x: aligned_min_x as u32,
        y: aligned_min_y as u32,
        w: ((max_x + 1) - aligned_min_x) as u32,
        h: ((max_y + 1) - aligned_min_y) as u32,
    })
}

/// Extract a sub-rectangle of a full-canvas RGBA buffer covering `rect`
/// (in canvas coordinates). Returns the flat RGBA buffer the VP8L encoder
/// will consume. `rect` must lie fully inside the canvas.
fn extract_subrect_from_canvas(canvas: &[u8], canvas_w: u32, rect: DirtyRect) -> Vec<u8> {
    let cw_bytes = (canvas_w as usize) * 4;
    let rx = rect.x as usize;
    let ry = rect.y as usize;
    let rw = rect.w as usize;
    let rh = rect.h as usize;
    let mut out = Vec::with_capacity(rw * rh * 4);
    for row in 0..rh {
        let src_off = (ry + row) * cw_bytes + rx * 4;
        out.extend_from_slice(&canvas[src_off..src_off + rw * 4]);
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct DirtyRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// Build the 6-byte §2.7.1.1 Figure 8 `ANIM` payload: BGRA background colour
/// (the `[R,G,B,A]` option re-ordered to on-disk `[B,G,R,A]`) + LE u16 loop
/// count.
fn build_anim_payload(opts: &AnimEncoderOptions<'_>) -> Vec<u8> {
    let [r, g, b, a] = opts.background_rgba;
    let mut p = Vec::with_capacity(ANIM_PAYLOAD_LEN);
    // §2.7.1.1: on-disk byte order is [Blue, Green, Red, Alpha].
    p.push(b);
    p.push(g);
    p.push(r);
    p.push(a);
    p.extend_from_slice(&opts.loop_count.to_le_bytes());
    p
}

/// Build a single `ANMF` chunk payload, taking the previous-canvas state
/// into account for [`AnimFrameMode::Auto`] / [`AnimFrameMode::Delta`].
///
/// For [`AnimFrameMode::Lossless`], the frame is always encoded as a full
/// keyframe at its declared `(x, y, width, height)` — the caller's
/// `blend`/`dispose` flags are honoured verbatim.
///
/// For [`AnimFrameMode::Delta`], the dirty rect — the bounding box of
/// canvas pixels the frame actually *changes*, i.e. the diff between the
/// post-composite canvas (`prev` with the frame drawn per its `blend`
/// method) and `prev` itself — is encoded as the ANMF sub-frame carrying
/// the **post-composite** pixels with `B = 1` / `D = 0`. Diffing and
/// emitting the drawn state (rather than the raw source pixels) is what
/// keeps an `AlphaBlend` frame bit-exact: the decoder overwrites the rect
/// with the already-blended pixels and lands on the same canvas the
/// caller's blend would have produced.
///
/// Two cases fall back to a full keyframe with the caller's flags
/// honoured verbatim (same emission as `Lossless`):
///
/// * `is_first_frame` — no `prev` to diff against;
/// * `dispose == Background` — a dirty-rect sub-frame cannot carry the
///   §2.7.1.1 "clear the frame's rect to the background colour" rule for
///   the *caller's* full rect (the emitted rect is smaller), so the
///   dispose flag must travel on a full-rect ANMF to keep the decoder's
///   canvas in lockstep with the reference state the next frame is
///   diffed against.
///
/// For [`AnimFrameMode::Auto`], both candidates are encoded and the
/// smaller VP8L bitstream wins (subject to the same two fallbacks).
fn build_anmf_payload_with_prev(
    f: &AnimFrame,
    canvas_w: u32,
    prev: &[u8],
    is_first_frame: bool,
) -> Result<Vec<u8>, WebpError> {
    match f.mode {
        AnimFrameMode::Lossless => emit_full_anmf(f, f.x, f.y, f.width, f.height, &f.pixels),
        AnimFrameMode::Delta => {
            if is_first_frame || f.dispose == DisposalMethod::Background {
                emit_full_anmf(f, f.x, f.y, f.width, f.height, &f.pixels)
            } else {
                let drawn = drawn_canvas(prev, canvas_w, f);
                let rect =
                    dirty_rect_canvas_coords(f, canvas_w, prev, &drawn).unwrap_or(DirtyRect {
                        // Identical-to-previous: emit a 2×2 overwrite of
                        // the (unchanged) drawn-canvas pixels at the
                        // frame's corner — duration is preserved and
                        // re-writing the same pixels is a no-op for the
                        // decoder's compositor.
                        x: f.x,
                        y: f.y,
                        w: 2.min(f.width),
                        h: 2.min(f.height),
                    });
                let sub_rgba = extract_subrect_from_canvas(&drawn, canvas_w, rect);
                emit_dirty_anmf(f, rect, &sub_rgba)
            }
        }
        AnimFrameMode::Auto => {
            // Always evaluate the full-frame candidate (and use it for the
            // first frame / Background-dispose fallbacks regardless).
            let full = emit_full_anmf(f, f.x, f.y, f.width, f.height, &f.pixels)?;
            if is_first_frame || f.dispose == DisposalMethod::Background {
                return Ok(full);
            }
            let drawn = drawn_canvas(prev, canvas_w, f);
            let Some(rect) = dirty_rect_canvas_coords(f, canvas_w, prev, &drawn) else {
                // Frame leaves the canvas untouched — a 2×2 same-pixels
                // delta is smaller than any non-trivial full frame.
                let degen_rect = DirtyRect {
                    x: f.x,
                    y: f.y,
                    w: 2.min(f.width),
                    h: 2.min(f.height),
                };
                let sub_rgba = extract_subrect_from_canvas(&drawn, canvas_w, degen_rect);
                let delta = emit_dirty_anmf(f, degen_rect, &sub_rgba)?;
                return Ok(if delta.len() < full.len() {
                    delta
                } else {
                    full
                });
            };
            // If the dirty rect covers the whole declared frame rect and
            // the frame is plain-overwrite, the full candidate carries the
            // identical pixels — no win from a sub-frame.
            if f.blend == BlendingMethod::Overwrite
                && rect.w == f.width
                && rect.h == f.height
                && rect.x == f.x
                && rect.y == f.y
            {
                return Ok(full);
            }
            let sub_rgba = extract_subrect_from_canvas(&drawn, canvas_w, rect);
            let delta = emit_dirty_anmf(f, rect, &sub_rgba)?;
            Ok(if delta.len() < full.len() {
                delta
            } else {
                full
            })
        }
    }
}

/// Clone `prev` and composite `f` onto it per its `blend` method — the
/// §2.7.1.1 canvas state the decoder must display after this frame.
fn drawn_canvas(prev: &[u8], canvas_w: u32, f: &AnimFrame) -> Vec<u8> {
    let mut drawn = prev.to_vec();
    composite_frame_onto_canvas(&mut drawn, canvas_w, f);
    drawn
}

/// Encode `pixels` (`w*h*4` flat RGBA) as a §2.6 `VP8L` chunk wrapped in
/// the §2.7.1.1 Figure 9 16-byte ANMF header at `(x, y, w, h)` with the
/// caller's blend/dispose/duration. Used by both the full-keyframe and
/// dirty-rect emission paths.
fn emit_full_anmf(
    f: &AnimFrame,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    pixels: &[u8],
) -> Result<Vec<u8>, WebpError> {
    let argb = rgba_to_argb(pixels);
    let has_alpha = pixels.chunks_exact(4).any(|px| px[3] != 0xff);
    let bitstream = vp8l_encode::encode_vp8l_argb_with(&argb, w, h, has_alpha)
        .map_err(Error::from)
        .map_err(WebpError::from)?;
    let frame_data = build::build_chunk(fourcc::VP8L, &bitstream).map_err(to_w)?;
    Ok(build_anmf_header_then_data(
        x,
        y,
        w,
        h,
        f.duration,
        f.blend,
        f.dispose,
        &frame_data,
    ))
}

/// Emit an ANMF carrying only the dirty `rect` sub-frame, forced to
/// `B = 1` (overwrite) and `D = 0` (no dispose) so the decoder's
/// compositor reconstructs the caller's intended canvas bit-exactly:
/// `sub_rgba` is the **post-composite** drawn state, so a plain overwrite
/// of the rect lands exactly on the canvas the caller's `blend` produces.
/// Only reached when the caller's `dispose` is `None` (a `Background`
/// dispose travels on the full-keyframe fallback instead — see
/// [`build_anmf_payload_with_prev`]), so `D = 0` *is* the caller's flag.
fn emit_dirty_anmf(f: &AnimFrame, rect: DirtyRect, sub_rgba: &[u8]) -> Result<Vec<u8>, WebpError> {
    let argb = rgba_to_argb(sub_rgba);
    let has_alpha = sub_rgba.chunks_exact(4).any(|px| px[3] != 0xff);
    let bitstream = vp8l_encode::encode_vp8l_argb_with(&argb, rect.w, rect.h, has_alpha)
        .map_err(Error::from)
        .map_err(WebpError::from)?;
    let frame_data = build::build_chunk(fourcc::VP8L, &bitstream).map_err(to_w)?;
    Ok(build_anmf_header_then_data(
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        f.duration,
        BlendingMethod::Overwrite,
        DisposalMethod::None,
        &frame_data,
    ))
}

/// Splice the 16-byte §2.7.1.1 Figure 9 header in front of the per-frame
/// Frame Data sub-RIFF and return the complete ANMF chunk payload.
#[allow(clippy::too_many_arguments)]
fn build_anmf_header_then_data(
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    duration: u32,
    blend: BlendingMethod,
    dispose: DisposalMethod,
    frame_data: &[u8],
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(ANMF_HEADER_LEN + frame_data.len());
    push_u24_le(&mut payload, x / 2);
    push_u24_le(&mut payload, y / 2);
    push_u24_le(&mut payload, w - 1);
    push_u24_le(&mut payload, h - 1);
    push_u24_le(&mut payload, duration & 0x00FF_FFFF);
    let b_bit = match blend {
        BlendingMethod::AlphaBlend => 0,
        BlendingMethod::Overwrite => 1,
    };
    let d_bit = match dispose {
        DisposalMethod::None => 0,
        DisposalMethod::Background => 1,
    };
    payload.push((b_bit << 1) | d_bit);
    payload.extend_from_slice(frame_data);
    payload
}

/// Push the low 24 bits of `v` as three little-endian bytes.
fn push_u24_le(out: &mut Vec<u8>, v: u32) {
    out.push((v & 0xFF) as u8);
    out.push(((v >> 8) & 0xFF) as u8);
    out.push(((v >> 16) & 0xFF) as u8);
}

/// Repack interleaved `[R, G, B, A]` bytes into packed ARGB
/// (`(a<<24)|(r<<16)|(g<<8)|b`) — the layout the VP8L encoder consumes.
fn rgba_to_argb(rgba: &[u8]) -> Vec<u32> {
    rgba.chunks_exact(4)
        .map(|px| {
            let (r, g, b, a) = (px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32);
            (a << 24) | (r << 16) | (g << 8) | b
        })
        .collect()
}

/// Collapse a [`crate::build::BuildError`] into the published coarse
/// [`WebpError::InvalidData`].
fn to_w(_e: build::BuildError) -> WebpError {
    WebpError::InvalidData
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&color);
        }
        v
    }

    #[test]
    fn empty_frames_is_invalid_data() {
        assert_eq!(build_animated_webp(&[]), Err(WebpError::InvalidData));
    }

    #[test]
    fn auto_and_delta_modes_emit_valid_files_round_127() {
        // Round 127: Auto + Delta are wired up against the §2.7.1.1
        // overwrite-no-dispose path. Selecting them no longer returns
        // Unsupported; the encoded file is structurally valid and the
        // container walker accepts it.
        for mode in [AnimFrameMode::Auto, AnimFrameMode::Delta] {
            let mut f = AnimFrame::new(2, 2, solid_rgba(2, 2, [1, 2, 3, 255]), 100);
            f.mode = mode;
            let file = build_animated_webp(&[f]).unwrap_or_else(|e| {
                panic!("mode {mode:?} must build a valid file in round 127, got {e:?}")
            });
            assert_eq!(&file[0..4], b"RIFF");
            assert_eq!(&file[8..12], b"WEBP");
            let c = crate::container::parse(&file).expect("parseable container");
            assert!(c.first_chunk_with_fourcc(fourcc::ANMF).is_some());
        }
    }

    #[test]
    fn dirty_rect_shrinks_anmf_payload_for_localised_change() {
        // Build a 16×16 canvas-aligned frame pair where only a 4×4 block
        // in the centre differs. A Delta-mode emit must produce a
        // strictly smaller ANMF chunk than a Lossless full-frame emit.
        let w = 16u32;
        let h = 16u32;
        let mut f0 = AnimFrame::new(w, h, solid_rgba(w, h, [200, 100, 50, 255]), 80);
        f0.mode = AnimFrameMode::Lossless;
        let mut f1_pixels = solid_rgba(w, h, [200, 100, 50, 255]);
        for row in 6..10 {
            for col in 6..10 {
                let off = (row * w as usize + col) * 4;
                f1_pixels[off] = 0;
                f1_pixels[off + 1] = 0;
                f1_pixels[off + 2] = 0;
                f1_pixels[off + 3] = 255;
            }
        }
        let mut f1_lossless = AnimFrame::new(w, h, f1_pixels.clone(), 80);
        f1_lossless.mode = AnimFrameMode::Lossless;
        let mut f1_delta = AnimFrame::new(w, h, f1_pixels.clone(), 80);
        f1_delta.mode = AnimFrameMode::Delta;

        let file_lossless = build_animated_webp(&[f0.clone(), f1_lossless]).unwrap();
        let file_delta = build_animated_webp(&[f0, f1_delta]).unwrap();

        assert!(
            file_delta.len() < file_lossless.len(),
            "delta-mode file ({} bytes) must beat lossless ({} bytes) on a 4×4 change",
            file_delta.len(),
            file_lossless.len(),
        );
    }

    #[test]
    fn auto_mode_picks_dirty_rect_on_localised_change() {
        // Same 4×4-change setup as above; Auto must pick the smaller
        // (delta) candidate.
        let w = 16u32;
        let h = 16u32;
        let mut f0 = AnimFrame::new(w, h, solid_rgba(w, h, [200, 100, 50, 255]), 80);
        f0.mode = AnimFrameMode::Lossless;
        let mut f1_pixels = solid_rgba(w, h, [200, 100, 50, 255]);
        for row in 6..10 {
            for col in 6..10 {
                let off = (row * w as usize + col) * 4;
                f1_pixels[off] = 0;
            }
        }
        let mut f1_auto = AnimFrame::new(w, h, f1_pixels.clone(), 80);
        f1_auto.mode = AnimFrameMode::Auto;
        let mut f1_lossless = AnimFrame::new(w, h, f1_pixels, 80);
        f1_lossless.mode = AnimFrameMode::Lossless;

        let file_auto = build_animated_webp(&[f0.clone(), f1_auto]).unwrap();
        let file_lossless = build_animated_webp(&[f0, f1_lossless]).unwrap();

        assert!(
            file_auto.len() <= file_lossless.len(),
            "auto-mode never regresses vs lossless ({} vs {} bytes)",
            file_auto.len(),
            file_lossless.len(),
        );
    }

    #[test]
    fn dirty_rect_canvas_coords_covers_only_the_changed_pixels() {
        let w = 8u32;
        let h = 8u32;
        let prev = solid_rgba(w, h, [0, 0, 0, 0]);
        let mut pixels = solid_rgba(w, h, [0, 0, 0, 0]);
        pixels[(3 * 8 + 5) * 4] = 0xff;
        pixels[(4 * 8 + 5) * 4 + 1] = 0xee;
        let f = AnimFrame::new(w, h, pixels, 0);
        let drawn = drawn_canvas(&prev, w, &f);
        let rect = dirty_rect_canvas_coords(&f, w, &prev, &drawn).expect("change exists");
        // Single-pixel changes at (5,3) and (5,4); after even-alignment of
        // the top-left, the rect spans x ∈ [4, 5], y ∈ [2, 4].
        assert_eq!(rect.x % 2, 0);
        assert_eq!(rect.y % 2, 0);
        assert!(rect.x <= 5 && rect.x + rect.w > 5);
        assert!(rect.y <= 3 && rect.y + rect.h > 4);
    }

    #[test]
    fn dirty_rect_is_none_on_identical_frames() {
        let w = 4u32;
        let h = 4u32;
        let pixels = solid_rgba(w, h, [1, 2, 3, 255]);
        let prev = pixels.clone();
        let f = AnimFrame::new(w, h, pixels, 0);
        let drawn = drawn_canvas(&prev, w, &f);
        assert!(dirty_rect_canvas_coords(&f, w, &prev, &drawn).is_none());
    }

    #[test]
    fn pixel_length_mismatch_is_invalid_data() {
        let mut f = AnimFrame::new(2, 2, solid_rgba(2, 2, [0, 0, 0, 255]), 0);
        f.pixels.truncate(4);
        assert_eq!(build_animated_webp(&[f]), Err(WebpError::InvalidData));
    }

    #[test]
    fn odd_offset_is_invalid_data() {
        let mut f = AnimFrame::new(2, 2, solid_rgba(2, 2, [0, 0, 0, 255]), 0);
        f.x = 1;
        assert_eq!(build_animated_webp(&[f]), Err(WebpError::InvalidData));
    }

    #[test]
    fn output_begins_with_riff_webp_and_is_parseable() {
        let f = AnimFrame::new(4, 4, solid_rgba(4, 4, [10, 20, 30, 255]), 100);
        let file = build_animated_webp(&[f]).expect("build animated webp");
        assert_eq!(&file[0..4], b"RIFF");
        assert_eq!(&file[8..12], b"WEBP");
        // The container walker must accept it.
        let c = crate::container::parse(&file).expect("parseable container");
        // VP8X then ANIM then ANMF must all be present.
        assert!(c.first_chunk_with_fourcc(fourcc::VP8X).is_some());
        assert!(c.first_chunk_with_fourcc(fourcc::ANIM).is_some());
        assert!(c.first_chunk_with_fourcc(fourcc::ANMF).is_some());
    }

    #[test]
    fn delta_config_builders_chain() {
        let cfg = DeltaConfig::default()
            .max_components_override(3)
            .auto_inner_threshold_bytes(Some(512))
            .msssim_downsample_kernel(DownsampleKernel::Gaussian);
        assert_eq!(cfg.max_components, 3);
        assert_eq!(cfg.auto_inner_threshold_bytes, Some(512));
        assert_eq!(cfg.msssim_downsample_kernel, DownsampleKernel::Gaussian);
    }
}
