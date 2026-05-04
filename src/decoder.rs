//! Top-level WebP decoder that coordinates VP8 lossy, VP8L lossless, and
//! the extended format's alpha/animation paths.
//!
//! The demuxer hands the decoder per-frame packets in an internal `OWEB`
//! envelope carrying the VP8/VP8L bytes + optional ALPH payload + a
//! render bbox against the canvas. The decoder:
//!
//! 1. Decodes the image chunk into a tile-sized RGBA buffer.
//!    * VP8 → run the keyframe through `oxideav-vp8`, then YUV→RGB convert.
//!    * VP8L → run [`crate::vp8l::decode`] directly (native RGBA).
//! 2. If an ALPH chunk is present, decodes it and overlays its alpha
//!    plane on the RGB tile.
//! 3. Composites the tile onto an internal RGBA canvas following the
//!    ANMF blend/disposal rules.
//! 4. Emits one `VideoFrame` per packet with the canvas state at that
//!    point in the animation.

#[cfg(feature = "registry")]
use std::collections::VecDeque;

#[cfg(feature = "registry")]
use oxideav_core::Decoder;
#[cfg(feature = "registry")]
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase, VideoFrame, VideoPlane};
use oxideav_vp8::decode_vp8 as decode_vp8_frame;

#[cfg(feature = "registry")]
use crate::demux::{decode_frame_payload, DecodedAlph};
use crate::demux::{AlphChunk, ImagePayload, ParsedFrame, WebpFileMetadata};
use crate::error::{Result, WebpError as Error};
use crate::vp8l;

/// Public helper — decode an entire `.webp` file sitting in `buf` and
/// return all frames as RGBA `WebpFrame`s plus any auxiliary metadata
/// (`ICCP` / `EXIF` / `XMP `) carried by the container.
///
/// Standalone (no `oxideav-core`) entry point: walks the parsed
/// container directly without going through the framework's
/// `Demuxer` / `Decoder` traits, so it works whether or not the
/// `registry` feature is enabled.
pub fn decode_webp(buf: &[u8]) -> Result<WebpImage> {
    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WEBP" {
        return Err(Error::invalid("WebP: bad RIFF/WEBP magic"));
    }
    let riff_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let end = (8 + riff_size).min(buf.len());
    let body = &buf[12..end];
    let parsed = crate::demux::parse_webp_body(body)?;
    let (canvas_w, canvas_h) = parsed.canvas;
    let mut canvas = vec![0u8; (canvas_w as usize) * (canvas_h as usize) * 4];
    let mut frames = Vec::with_capacity(parsed.frames.len());
    for f in &parsed.frames {
        let tile_rgba = decode_parsed_frame_to_rgba(f)?;
        composite(
            &mut canvas,
            canvas_w,
            canvas_h,
            &tile_rgba,
            f.x_offset,
            f.y_offset,
            f.width,
            f.height,
            f.blend_with_previous,
        );
        frames.push(WebpFrame {
            width: canvas_w,
            height: canvas_h,
            duration_ms: f.duration_ms,
            rgba: canvas.clone(),
        });
        if f.dispose_to_background {
            let x0 = f.x_offset as usize;
            let y0 = f.y_offset as usize;
            let x1 = (x0 + f.width as usize).min(canvas_w as usize);
            let y1 = (y0 + f.height as usize).min(canvas_h as usize);
            let w = canvas_w as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * w + x) * 4;
                    canvas[i] = 0;
                    canvas[i + 1] = 0;
                    canvas[i + 2] = 0;
                    canvas[i + 3] = 0;
                }
            }
        }
    }
    Ok(WebpImage {
        width: canvas_w,
        height: canvas_h,
        frames,
        metadata: parsed.metadata,
    })
}

/// Decode one `ParsedFrame` (image + optional ALPH) into a tightly-
/// packed RGBA tile sized `frame.width * frame.height * 4`. Used by
/// the standalone [`decode_webp`] path, which never sees the framework
/// `Decoder` / `Packet` envelope.
fn decode_parsed_frame_to_rgba(f: &ParsedFrame) -> Result<Vec<u8>> {
    let (image_bytes, is_vp8l) = match &f.image {
        ImagePayload::Vp8(b) => (b.as_slice(), false),
        ImagePayload::Vp8l(b) => (b.as_slice(), true),
    };
    let tile_rgba = if is_vp8l {
        let img = vp8l::decode(image_bytes)?;
        img.to_rgba()
    } else {
        decode_vp8_to_rgba(image_bytes, f.width, f.height)?
    };
    let tile_rgba = if let Some(alph) = &f.alph {
        overlay_alpha_chunk(tile_rgba, f.width, f.height, alph)?
    } else if !is_vp8l {
        set_alpha_opaque(tile_rgba)
    } else {
        tile_rgba
    };
    Ok(tile_rgba)
}

/// Variant of [`overlay_alpha`] that takes the owned [`AlphChunk`]
/// (used by the standalone walk over [`ParsedFrame`]) instead of the
/// [`DecodedAlph`] borrow the Decoder-trait path uses. Same result.
fn overlay_alpha_chunk(
    mut rgba: Vec<u8>,
    width: u32,
    height: u32,
    alph: &AlphChunk,
) -> Result<Vec<u8>> {
    let alpha = decode_alpha_plane_chunk(width, height, alph)?;
    if alpha.len() != (width as usize) * (height as usize) {
        return Err(Error::invalid("WebP: alpha plane size mismatch"));
    }
    for (i, &a) in alpha.iter().enumerate() {
        rgba[i * 4 + 3] = a;
    }
    Ok(rgba)
}

fn decode_alpha_plane_chunk(width: u32, height: u32, alph: &AlphChunk) -> Result<Vec<u8>> {
    let mut plane = match alph.compression {
        0 => alph.data.clone(),
        1 => {
            let mut synth = Vec::with_capacity(alph.data.len() + 5);
            synth.push(0x2f);
            let w = width.saturating_sub(1) & 0x3fff;
            let h = height.saturating_sub(1) & 0x3fff;
            let packed = w | (h << 14);
            synth.extend_from_slice(&packed.to_le_bytes());
            synth.extend_from_slice(&alph.data);
            let img = vp8l::decode(&synth)?;
            img.pixels.iter().map(|p| ((p >> 8) & 0xff) as u8).collect()
        }
        _ => return Err(Error::invalid("WebP: unknown ALPH compression")),
    };
    unfilter_alpha(&mut plane, width as usize, height as usize, alph.filtering);
    Ok(plane)
}

/// Convenience struct returned by [`decode_webp`].
#[derive(Debug, Clone)]
pub struct WebpImage {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<WebpFrame>,
    /// Auxiliary container-level metadata (ICC / EXIF / XMP). All three
    /// fields are `None` for files that don't carry the matching chunk
    /// — including every simple-layout (no `VP8X` header) `.webp`.
    pub metadata: WebpFileMetadata,
}

#[derive(Debug, Clone)]
pub struct WebpFrame {
    pub width: u32,
    pub height: u32,
    pub duration_ms: u32,
    pub rgba: Vec<u8>,
}

/// Factory used for the `webp_vp8l` codec id — decodes a bare VP8L
/// bitstream (no RIFF/WebP wrapper). Useful for consumers that have
/// already stripped the container and want the codec registry to give
/// them a standalone decoder.
#[cfg(feature = "registry")]
pub fn make_vp8l_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    let w = params.width.unwrap_or(0);
    let h = params.height.unwrap_or(0);
    Ok(Box::new(Vp8lStandalone {
        codec_id: params.codec_id.clone(),
        width: w,
        height: h,
        queued: VecDeque::new(),
        pending_pts: None,
        pending_tb: TimeBase::new(1, 1000),
    }))
}

#[cfg(feature = "registry")]
struct Vp8lStandalone {
    codec_id: CodecId,
    width: u32,
    height: u32,
    queued: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
}

#[cfg(feature = "registry")]
impl Decoder for Vp8lStandalone {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }
    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        let img = vp8l::decode(&packet.data)?;
        if self.width == 0 {
            self.width = img.width;
        }
        if self.height == 0 {
            self.height = img.height;
        }
        let rgba = img.to_rgba();
        let vf = VideoFrame {
            pts: self.pending_pts,
            planes: vec![VideoPlane {
                stride: (img.width as usize) * 4,
                data: rgba,
            }],
        };
        self.queued.push_back(vf);
        Ok(())
    }
    fn receive_frame(&mut self) -> oxideav_core::Result<Frame> {
        self.queued
            .pop_front()
            .map(Frame::Video)
            .ok_or(oxideav_core::Error::NeedMore)
    }
    fn flush(&mut self) -> oxideav_core::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "registry")]
pub struct WebpDecoder {
    codec_id: CodecId,
    canvas_w: u32,
    canvas_h: u32,
    canvas: Vec<u8>, // RGBA
    queued: VecDeque<VideoFrame>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    first_frame: bool,
    /// When `true`, single-frame VP8+ALPH inputs emit a `Yuva420P`
    /// frame (4 planes: Y, U, V, A) instead of compositing into the
    /// RGBA canvas. Default is `false` so existing callers keep the
    /// historical `Rgba` output. VP8L stays on the RGBA path either
    /// way (it's natively RGBA inside the bitstream — converting it
    /// back to YUVA would lose information for negligible benefit) and
    /// animated files always composite onto the RGBA canvas (cross-
    /// frame disposal/blend semantics need a unified pixel format).
    prefer_yuva420p: bool,
}

#[cfg(feature = "registry")]
impl WebpDecoder {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            codec_id: CodecId::new(crate::demux::WEBP_CODEC_ID),
            canvas_w: w,
            canvas_h: h,
            canvas: vec![0; (w as usize) * (h as usize) * 4],
            queued: VecDeque::new(),
            pending_pts: None,
            pending_tb: TimeBase::new(1, 1000),
            first_frame: true,
            prefer_yuva420p: false,
        }
    }

    /// Build a decoder that emits `Yuva420P` frames whenever the input
    /// is a single-frame VP8 + ALPH file (the natural shape for
    /// lossy-with-alpha WebP). For VP8L and animated files this still
    /// composites into the RGBA canvas — see [`Self::prefer_yuva420p`]
    /// for the rationale.
    ///
    /// Skips the YUV→RGB conversion the VP8 chunk normally goes through
    /// (and the matching alpha-overlay step), so callers that feed the
    /// frame back into a YUV-native pipeline (further encoding, video
    /// compositing, GPU upload paths that prefer YUV) avoid the
    /// roundtrip.
    pub fn new_yuva420p(w: u32, h: u32) -> Self {
        let mut dec = Self::new(w, h);
        dec.prefer_yuva420p = true;
        dec
    }

    /// Toggle the `Yuva420P` output mode after construction. See
    /// [`Self::new_yuva420p`] for what it does.
    pub fn set_prefer_yuva420p(&mut self, prefer: bool) {
        self.prefer_yuva420p = prefer;
    }
}

#[cfg(feature = "registry")]
impl Decoder for WebpDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        let payload = decode_frame_payload(&packet.data)?;
        if self.canvas_w == 0 || self.canvas_h == 0 {
            self.canvas_w = payload.canvas.0;
            self.canvas_h = payload.canvas.1;
            self.canvas = vec![0; (self.canvas_w as usize) * (self.canvas_h as usize) * 4];
        }

        let _ = self.first_frame;

        // Yuva420P fast path: a non-animated VP8 chunk with an ALPH
        // sidecar that lines up with the canvas (i.e. a still file in
        // the VP8X + ALPH + VP8 layout). Emits a 4-plane Yuva420P frame
        // straight from the VP8 decoder + the alpha plane, skipping
        // YUV→RGB conversion + the canvas composite. VP8L and ANMF
        // animation always fall through to the RGBA path — see
        // `prefer_yuva420p` for the reasoning.
        if self.prefer_yuva420p
            && !payload.is_vp8l
            && payload.x_offset == 0
            && payload.y_offset == 0
            && payload.width == self.canvas_w
            && payload.height == self.canvas_h
        {
            if let Some(alph) = payload.alph.as_ref() {
                let yuva = decode_vp8_alph_to_yuva420p(
                    payload.image,
                    payload.width,
                    payload.height,
                    alph,
                    self.pending_pts,
                )?;
                self.queued.push_back(yuva);
                self.first_frame = false;
                return Ok(());
            }
        }

        // Decode the image chunk.
        let tile_rgba = if payload.is_vp8l {
            let img = vp8l::decode(payload.image)?;
            img.to_rgba()
        } else {
            decode_vp8_to_rgba(payload.image, payload.width, payload.height)?
        };

        // Apply alpha chunk, if present. VP8L already carries alpha in
        // its RGBA output. VP8 is RGB only, so ALPH (if any) overwrites
        // the alpha channel here.
        let tile_rgba = if let Some(alph) = &payload.alph {
            overlay_alpha(tile_rgba, payload.width, payload.height, alph)?
        } else if !payload.is_vp8l {
            // VP8 without ALPH: opaque image.
            set_alpha_opaque(tile_rgba)
        } else {
            tile_rgba
        };

        // Composite onto the canvas.
        composite(
            &mut self.canvas,
            self.canvas_w,
            self.canvas_h,
            &tile_rgba,
            payload.x_offset,
            payload.y_offset,
            payload.width,
            payload.height,
            payload.blend_with_previous,
        );

        let vf = VideoFrame {
            pts: self.pending_pts,
            planes: vec![VideoPlane {
                stride: (self.canvas_w as usize) * 4,
                data: self.canvas.clone(),
            }],
        };
        self.queued.push_back(vf);

        // Post-frame disposal.
        if payload.dispose_to_background {
            let x0 = payload.x_offset as usize;
            let y0 = payload.y_offset as usize;
            let x1 = (x0 + payload.width as usize).min(self.canvas_w as usize);
            let y1 = (y0 + payload.height as usize).min(self.canvas_h as usize);
            let w = self.canvas_w as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * w + x) * 4;
                    self.canvas[i] = 0;
                    self.canvas[i + 1] = 0;
                    self.canvas[i + 2] = 0;
                    self.canvas[i + 3] = 0;
                }
            }
        }

        self.first_frame = false;
        Ok(())
    }

    fn receive_frame(&mut self) -> oxideav_core::Result<Frame> {
        self.queued
            .pop_front()
            .map(Frame::Video)
            .ok_or(oxideav_core::Error::NeedMore)
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> oxideav_core::Result<()> {
        // WebP animation composites each ANMF frame onto a persistent RGBA
        // canvas — that canvas IS the cross-frame state. Wiping it forces
        // the next frame to composite onto a clean background, matching
        // the semantics of starting playback from the seek target.
        // `first_frame` returns true so a post-seek frame with
        // blend_with_previous doesn't mix into the pre-seek canvas.
        if self.canvas_w > 0 && self.canvas_h > 0 {
            self.canvas = vec![0; (self.canvas_w as usize) * (self.canvas_h as usize) * 4];
        } else {
            self.canvas.clear();
        }
        self.queued.clear();
        self.pending_pts = None;
        self.first_frame = true;
        Ok(())
    }
}

/// BT.601 YUV→RGB conversion matching libwebp's `dsp/yuv.h` *exactly*,
/// including its two-stage truncating fixed-point arithmetic. RFC 9649
/// §2.5 specifies "Recommendation 601 SHOULD be used"; libwebp's
/// implementation choice is the de-facto interop reference.
///
/// libwebp computes each channel as
///
/// ```text
///     VP8Clip8(MultHi(y, kY)  + MultHi(c, kC)  + cst)
///     where MultHi(a, b)  = (a * b) >> 8     // YUV_FIX2 = 6, but
///                                            // first stage shifts 8
///     and Clip8(v)        = clamp(v >> 6, 0, 255)
/// ```
///
/// The two-stage right-shift (`>> 8` per multiply, `>> 6` on the sum)
/// is **not** algebraically equivalent to a single
/// `(KY*y + KC*c + …) >> 14`. The intermediate truncation loses a few
/// low bits on every term, and the bias matters for bit-exact interop
/// with the libwebp-encoded `expected.png` ground truth. Folding the
/// shifts into one biased every output channel ~1 LSB high — which is
/// exactly the off-by-one pattern the lossy_corpus integration test
/// reported pre-fix (q1 pixel #2 actual=[173,235,214] vs
/// expected=[172,234,213] in lockstep across q1/q75/q100/with-alpha).
///
/// The integer offsets `-14234 / +8708 / -17685` come straight from
/// libwebp's `VP8YUVToR/G/B` and absorb the Y=16 / chroma=128 biases
/// after MultHi truncation.
const KY_SCALE: i32 = 19077;
const KV_TO_R: i32 = 26149;
const KU_TO_G: i32 = 6419;
const KV_TO_G: i32 = 13320;
const KU_TO_B: i32 = 33050;
const KR_OFFSET: i32 = -14234;
const KG_OFFSET: i32 = 8708;
const KB_OFFSET: i32 = -17685;

#[inline]
fn mult_hi(v: i32, coeff: i32) -> i32 {
    (v * coeff) >> 8
}

/// Mirror of libwebp's `VP8Clip8`: an unsigned-range check on the
/// pre-shift value (all 26 bits beyond `YUV_MASK2` zero) is the cheap
/// path that just `>> 6`'s; otherwise saturate.
#[inline]
fn clip8(v: i32) -> u8 {
    const YUV_MASK2: i32 = (256 << 6) - 1;
    if (v & !YUV_MASK2) == 0 {
        (v >> 6) as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

#[inline]
fn yuv_to_r(y: i32, v: i32) -> u8 {
    clip8(mult_hi(y, KY_SCALE) + mult_hi(v, KV_TO_R) + KR_OFFSET)
}

#[inline]
fn yuv_to_g(y: i32, u: i32, v: i32) -> u8 {
    clip8(mult_hi(y, KY_SCALE) - mult_hi(u, KU_TO_G) - mult_hi(v, KV_TO_G) + KG_OFFSET)
}

#[inline]
fn yuv_to_b(y: i32, u: i32) -> u8 {
    clip8(mult_hi(y, KY_SCALE) + mult_hi(u, KU_TO_B) + KB_OFFSET)
}

/// Bilinear-fancy chroma upsample for a pair of luma rows, mirroring
/// libwebp's `UPSAMPLE_FUNC` macro (`src/dsp/upsampling.c`). For each
/// 2×2 luma block we synthesise 4 (u, v) pairs from the surrounding
/// 2×2 chroma corners with weights `(9,3,3,1) / 16`, packing u and v
/// into a single u32 so a single `>> 1` etc. operates on both
/// channels at once. Edge pixels (column 0 and column w-1 when even)
/// fall back to the 2-tap `(3*near + far + 2) / 4` form libwebp uses
/// at the row endpoints.
///
/// `top_y` / `bottom_y` are luma rows (length `w`). `top_u/top_v` and
/// `cur_u/cur_v` are the *upper* and *lower* chroma rows that
/// straddle this luma row pair (length `(w+1)/2`). For the first
/// luma row pair libwebp passes the same chroma row as both top and
/// cur, mirroring at the image boundary; this routine doesn't care
/// — the caller is responsible for that mirroring.
///
/// `bottom_y` may be `None` when the image height is odd and we're
/// drawing only the last luma row alone; in that case `bottom_dst`
/// is unused.
#[allow(clippy::too_many_arguments)]
fn upsample_rgba_line_pair(
    top_y: &[u8],
    bottom_y: Option<&[u8]>,
    top_u: &[u8],
    top_v: &[u8],
    cur_u: &[u8],
    cur_v: &[u8],
    top_dst: &mut [u8],
    mut bottom_dst: Option<&mut [u8]>,
    len: usize,
) {
    debug_assert!(len > 0);
    debug_assert_eq!(top_y.len(), len);
    debug_assert_eq!(top_dst.len(), len * 4);

    // Pack u/v into a single u32 (u in low 16 bits, v in high 16 bits)
    // so the 9/3/3/1 averaging op processes both channels in a single
    // shift / add — exactly libwebp's LOAD_UV trick.
    #[inline]
    fn load_uv(u: u8, v: u8) -> u32 {
        (u as u32) | ((v as u32) << 16)
    }
    #[inline]
    fn write_pixel(dst: &mut [u8], y: i32, u: u8, v: u8) {
        let u = u as i32;
        let v = v as i32;
        dst[0] = yuv_to_r(y, v);
        dst[1] = yuv_to_g(y, u, v);
        dst[2] = yuv_to_b(y, u);
        dst[3] = 0xff;
    }

    let last_pixel_pair = (len - 1) >> 1;
    let mut tl_uv = load_uv(top_u[0], top_v[0]);
    let mut l_uv = load_uv(cur_u[0], cur_v[0]);

    // Edge pixel at column 0 — libwebp uses (3*tl_uv + l_uv + 2) / 4
    // for the top-row endpoint. The +0x00020002 broadcasts the +2
    // rounding constant across both packed channels.
    {
        let uv0 = (3 * tl_uv + l_uv + 0x0002_0002) >> 2;
        write_pixel(
            &mut top_dst[0..4],
            top_y[0] as i32,
            (uv0 & 0xff) as u8,
            ((uv0 >> 16) & 0xff) as u8,
        );
    }
    if let (Some(by), Some(bd)) = (bottom_y, &mut bottom_dst) {
        let uv0 = (3 * l_uv + tl_uv + 0x0002_0002) >> 2;
        write_pixel(
            &mut bd[0..4],
            by[0] as i32,
            (uv0 & 0xff) as u8,
            ((uv0 >> 16) & 0xff) as u8,
        );
    }

    for x in 1..=last_pixel_pair {
        let t_uv = load_uv(top_u[x], top_v[x]);
        let uv = load_uv(cur_u[x], cur_v[x]);
        let avg = tl_uv + t_uv + l_uv + uv + 0x0008_0008;
        let diag_12 = (avg + 2 * (t_uv + l_uv)) >> 3;
        let diag_03 = (avg + 2 * (tl_uv + uv)) >> 3;

        let uv0 = (diag_12 + tl_uv) >> 1;
        let uv1 = (diag_03 + t_uv) >> 1;
        let i_lo = 2 * x - 1;
        let i_hi = 2 * x;
        write_pixel(
            &mut top_dst[i_lo * 4..i_lo * 4 + 4],
            top_y[i_lo] as i32,
            (uv0 & 0xff) as u8,
            ((uv0 >> 16) & 0xff) as u8,
        );
        write_pixel(
            &mut top_dst[i_hi * 4..i_hi * 4 + 4],
            top_y[i_hi] as i32,
            (uv1 & 0xff) as u8,
            ((uv1 >> 16) & 0xff) as u8,
        );

        if let (Some(by), Some(bd)) = (bottom_y, &mut bottom_dst) {
            let uv0 = (diag_03 + l_uv) >> 1;
            let uv1 = (diag_12 + uv) >> 1;
            write_pixel(
                &mut bd[i_lo * 4..i_lo * 4 + 4],
                by[i_lo] as i32,
                (uv0 & 0xff) as u8,
                ((uv0 >> 16) & 0xff) as u8,
            );
            write_pixel(
                &mut bd[i_hi * 4..i_hi * 4 + 4],
                by[i_hi] as i32,
                (uv1 & 0xff) as u8,
                ((uv1 >> 16) & 0xff) as u8,
            );
        }
        tl_uv = t_uv;
        l_uv = uv;
    }

    // Trailing pixel when `len` is even — same 2-tap edge-case as the
    // column-0 endpoint.
    if (len & 1) == 0 {
        let uv0 = (3 * tl_uv + l_uv + 0x0002_0002) >> 2;
        let last = len - 1;
        write_pixel(
            &mut top_dst[last * 4..last * 4 + 4],
            top_y[last] as i32,
            (uv0 & 0xff) as u8,
            ((uv0 >> 16) & 0xff) as u8,
        );
        if let (Some(by), Some(bd)) = (bottom_y, &mut bottom_dst) {
            let uv0 = (3 * l_uv + tl_uv + 0x0002_0002) >> 2;
            write_pixel(
                &mut bd[last * 4..last * 4 + 4],
                by[last] as i32,
                (uv0 & 0xff) as u8,
                ((uv0 >> 16) & 0xff) as u8,
            );
        }
    }
}

fn decode_vp8_to_rgba(bytes: &[u8], frame_w: u32, frame_h: u32) -> Result<Vec<u8>> {
    let vf = decode_vp8_frame(bytes)?;
    // VP8 always produces Yuv420P; pixel format is no longer carried per
    // frame and is asserted at the codec-parameters level upstream.
    let w = frame_w as usize;
    let h = frame_h as usize;
    // VP8's standalone `Vp8Frame` already returns tight-stride cropped
    // planes (stride == width / chroma-width), so the top-left region
    // is the whole plane.
    let y_stride = vf.y_stride as usize;
    let uv_stride = vf.uv_stride as usize;
    let mut out = vec![0u8; w * h * 4];

    if w == 0 || h == 0 {
        return Ok(out);
    }

    // Width-1 fast path: the libwebp fancy-upsample 2-tap endpoint
    // collapses to `(3*x + x + 2)/4 == x`, so all four edge cases are
    // bit-identical to point sampling for w=1. Skip the row-pair
    // machinery entirely.
    if w == 1 {
        for j in 0..h {
            let y_val = vf.y[j * y_stride] as i32;
            let u_val = vf.u[(j / 2) * uv_stride] as i32;
            let v_val = vf.v[(j / 2) * uv_stride] as i32;
            let idx = j * 4;
            out[idx] = yuv_to_r(y_val, v_val);
            out[idx + 1] = yuv_to_g(y_val, u_val, v_val);
            out[idx + 2] = yuv_to_b(y_val, u_val);
            out[idx + 3] = 0xff;
        }
        return Ok(out);
    }

    // Mirror libwebp's `EmitFancyRGB` (src/dec/io_dec.c) called once on
    // the whole frame:
    //
    //   1. Paint row 0 alone with chroma row 0 mirrored as both top and
    //      cur (the row-0 special case at the start of EmitFancyRGB).
    //   2. Loop while `y + 2 < h`: each iteration y → y+2 paints rows
    //      (y+1, y+2) using chroma rows (y/2, y/2 + 1). On the first
    //      iteration that's rows (1, 2) using chroma (0, 1); on the
    //      next, rows (3, 4) using chroma (1, 2); etc.
    //   3. Tail: for even h, the last luma row (h-1) hasn't been
    //      painted yet. Emit it as a top-only call with the last chroma
    //      row mirrored. For odd h, the loop already painted it.
    let cw = w.div_ceil(2);
    let chroma_h = h.div_ceil(2);
    debug_assert!(chroma_h >= 1);

    // (1) Row 0 with mirrored chroma row 0.
    {
        let top_y = &vf.y[0..w];
        let cur_u_row = &vf.u[0..cw];
        let cur_v_row = &vf.v[0..cw];
        let top_dst = &mut out[0..w * 4];
        upsample_rgba_line_pair(
            top_y, None, cur_u_row, cur_v_row, cur_u_row, cur_v_row, top_dst, None, w,
        );
    }

    // (2) Main loop: y goes 0, 2, 4, … and after each step we paint
    //     rows (y-1, y) — i.e. (1,2), (3,4), … — with chroma rows
    //     (y/2 - 1, y/2). Stop when y + 2 >= h.
    let mut y = 0usize;
    while y + 2 < h {
        y += 2;
        let top_chroma_row = (y / 2) - 1;
        let cur_chroma_row = y / 2;
        let top_y = &vf.y[(y - 1) * y_stride..(y - 1) * y_stride + w];
        let bottom_y = &vf.y[y * y_stride..y * y_stride + w];
        let top_u_row = &vf.u[top_chroma_row * uv_stride..top_chroma_row * uv_stride + cw];
        let top_v_row = &vf.v[top_chroma_row * uv_stride..top_chroma_row * uv_stride + cw];
        let cur_u_row = &vf.u[cur_chroma_row * uv_stride..cur_chroma_row * uv_stride + cw];
        let cur_v_row = &vf.v[cur_chroma_row * uv_stride..cur_chroma_row * uv_stride + cw];

        let row_offset_top = (y - 1) * w * 4;
        let (_, after) = out.split_at_mut(row_offset_top);
        let (top_dst, after_top) = after.split_at_mut(w * 4);
        let (bottom_dst, _) = after_top.split_at_mut(w * 4);
        upsample_rgba_line_pair(
            top_y,
            Some(bottom_y),
            top_u_row,
            top_v_row,
            cur_u_row,
            cur_v_row,
            top_dst,
            Some(bottom_dst),
            w,
        );
    }

    // (3) Tail: for even h, the last luma row (h-1) is still un-painted
    //     (the loop's last iteration painted (h-3, h-2)). For odd h, the
    //     loop already covered it. Mirror the last chroma row.
    if h >= 2 && (h & 1) == 0 {
        let last = h - 1;
        let top_y = &vf.y[last * y_stride..last * y_stride + w];
        let last_chroma_row = chroma_h - 1;
        let cur_u_row = &vf.u[last_chroma_row * uv_stride..last_chroma_row * uv_stride + cw];
        let cur_v_row = &vf.v[last_chroma_row * uv_stride..last_chroma_row * uv_stride + cw];
        let top_dst = &mut out[last * w * 4..(last + 1) * w * 4];
        upsample_rgba_line_pair(
            top_y, None, cur_u_row, cur_v_row, cur_u_row, cur_v_row, top_dst, None, w,
        );
    }

    Ok(out)
}

/// Decode a VP8 chunk + ALPH sidecar into a 4-plane `Yuva420P`
/// `VideoFrame`. The Y/U/V planes come from the VP8 decoder verbatim
/// (cropped to the frame's logical size — `Vp8Frame` already returns
/// tight-stride cropped planes, so this path is a straight clone),
/// and the alpha plane is a fresh full-resolution byte buffer.
///
/// Used only for non-animated single-frame VP8+ALPH inputs when the
/// decoder is in [`WebpDecoder::new_yuva420p`] mode. Skips the
/// YUV→RGB→consumer roundtrip the default RGBA path forces.
#[cfg(feature = "registry")]
fn decode_vp8_alph_to_yuva420p(
    bytes: &[u8],
    frame_w: u32,
    frame_h: u32,
    alph: &DecodedAlph<'_>,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let vf = decode_vp8_frame(bytes)?;
    let w = frame_w as usize;
    let h = frame_h as usize;
    let cw = w / 2 + (w & 1);
    let _ch = h / 2 + (h & 1);

    // `Vp8Frame` already carries tight-stride cropped planes, so we
    // can move the Vec buffers straight through without a re-pack.
    let y_plane = vf.y;
    let u_plane = vf.u;
    let v_plane = vf.v;

    let alpha = decode_alpha_plane(frame_w, frame_h, alph)?;
    if alpha.len() != w * h {
        return Err(Error::invalid("WebP: alpha plane size mismatch (Yuva420P)"));
    }

    Ok(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: w,
                data: y_plane,
            },
            VideoPlane {
                stride: cw,
                data: u_plane,
            },
            VideoPlane {
                stride: cw,
                data: v_plane,
            },
            VideoPlane {
                stride: w,
                data: alpha,
            },
        ],
    })
}

fn set_alpha_opaque(mut rgba: Vec<u8>) -> Vec<u8> {
    for i in (3..rgba.len()).step_by(4) {
        rgba[i] = 0xff;
    }
    rgba
}

/// Decode an ALPH payload and overlay its alpha plane onto an RGBA tile.
///
/// Spec §5.2.3: the alpha plane can be raw (`compression=0`), filtered,
/// or VP8L-compressed (`compression=1`, carrying a stripped single-green
/// VP8L bitstream whose green channel holds the alpha values).
#[cfg(feature = "registry")]
fn overlay_alpha(
    mut rgba: Vec<u8>,
    width: u32,
    height: u32,
    alph: &DecodedAlph<'_>,
) -> Result<Vec<u8>> {
    let alpha = decode_alpha_plane(width, height, alph)?;
    if alpha.len() != (width as usize) * (height as usize) {
        return Err(Error::invalid("WebP: alpha plane size mismatch"));
    }
    for (i, &a) in alpha.iter().enumerate() {
        rgba[i * 4 + 3] = a;
    }
    Ok(rgba)
}

#[cfg(feature = "registry")]
fn decode_alpha_plane(width: u32, height: u32, alph: &DecodedAlph<'_>) -> Result<Vec<u8>> {
    let mut plane = match alph.compression {
        0 => alph.data.to_vec(),
        1 => {
            // VP8L wrapper: synthesise a VP8L header in front of the
            // payload so we can reuse `vp8l::decode`. The header has no
            // alpha bit and 0 version, and the signature byte 0x2f.
            let mut synth = Vec::with_capacity(alph.data.len() + 5);
            synth.push(0x2f);
            let w = width.saturating_sub(1) & 0x3fff;
            let h = height.saturating_sub(1) & 0x3fff;
            let packed = w | (h << 14);
            synth.extend_from_slice(&packed.to_le_bytes());
            synth.extend_from_slice(alph.data);
            let img = vp8l::decode(&synth)?;
            img.pixels.iter().map(|p| ((p >> 8) & 0xff) as u8).collect()
        }
        _ => return Err(Error::invalid("WebP: unknown ALPH compression")),
    };
    unfilter_alpha(&mut plane, width as usize, height as usize, alph.filtering);
    Ok(plane)
}

fn unfilter_alpha(plane: &mut [u8], w: usize, h: usize, filt: u8) {
    match filt {
        0 => {}
        1 => {
            // Horizontal.
            for y in 0..h {
                for x in 1..w {
                    let i = y * w + x;
                    let left = plane[i - 1] as u16;
                    plane[i] = ((plane[i] as u16 + left) & 0xff) as u8;
                }
            }
        }
        2 => {
            // Vertical.
            for y in 1..h {
                for x in 0..w {
                    let i = y * w + x;
                    let top = plane[i - w] as u16;
                    plane[i] = ((plane[i] as u16 + top) & 0xff) as u8;
                }
            }
        }
        3 => {
            // Gradient (Paeth-like: predictor = clip(L + T - TL)).
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let pred = if y == 0 && x == 0 {
                        0
                    } else if y == 0 {
                        plane[i - 1] as i32
                    } else if x == 0 {
                        plane[i - w] as i32
                    } else {
                        let l = plane[i - 1] as i32;
                        let t = plane[i - w] as i32;
                        let tl = plane[i - w - 1] as i32;
                        (l + t - tl).clamp(0, 255)
                    };
                    plane[i] = ((plane[i] as i32 + pred) & 0xff) as u8;
                }
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn composite(
    canvas: &mut [u8],
    canvas_w: u32,
    canvas_h: u32,
    tile: &[u8],
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    blend: bool,
) {
    let cw = canvas_w as usize;
    for j in 0..h as usize {
        let cy = y as usize + j;
        if cy >= canvas_h as usize {
            break;
        }
        for i in 0..w as usize {
            let cx = x as usize + i;
            if cx >= canvas_w as usize {
                break;
            }
            let src = &tile[(j * w as usize + i) * 4..(j * w as usize + i) * 4 + 4];
            let dst_idx = (cy * cw + cx) * 4;
            if blend && src[3] < 0xff {
                let sa = src[3] as u32;
                let ia = 255 - sa;
                for c in 0..3 {
                    let s = src[c] as u32;
                    let d = canvas[dst_idx + c] as u32;
                    canvas[dst_idx + c] = ((s * sa + d * ia + 127) / 255) as u8;
                }
                let da = canvas[dst_idx + 3] as u32;
                canvas[dst_idx + 3] = (sa + ((da * ia) + 127) / 255) as u8;
            } else {
                canvas[dst_idx..dst_idx + 4].copy_from_slice(src);
            }
        }
    }
}
