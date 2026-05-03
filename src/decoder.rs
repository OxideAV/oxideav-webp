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

/// 14-bit fixed-point BT.601 YUV→RGB constants matching libwebp's
/// `dsp/yuv.h` (the reference implementation that produces the
/// `expected.png` ground truth in the lossy-corpus fixtures). RFC 9649
/// §2.5 specifies "Recommendation 601 SHOULD be used"; libwebp's
/// implementation choice is the de-facto interop reference.
///
/// The constants represent (1.164, 1.596, 0.391, 0.813, 2.018) in
/// 14-bit fixed point. Despite YUV being limited-range nominally
/// (Y in [16, 235], chroma in [16, 240]), libwebp does NOT pre-scale
/// the input — the kCst rounding constants absorb the 16/128 biases
/// directly so the formulas operate on raw u8 samples.
///
/// Why we deviate from the textbook 8-bit `(298 * (Y-16) + …) >> 8`
/// formula: the 8-bit constants represent slightly different ratios
/// (298/256 = 1.16406 vs libwebp's 19077/16384 = 1.16461; 409/256 =
/// 1.59766 vs 26149/16384 = 1.59601). Combined with different
/// rounding (+128>>8 vs +8192>>14 absorbed into kCst), every RGB
/// channel ends up biased ~1 LSB high — which is exactly the off-by-
/// one downward pattern observed against libwebp's expected.png in
/// the lossy_corpus integration test (q1/q75/q100/with-alpha all
/// reported "actual = expected + 1" in lockstep).
const KY_SCALE: i32 = 19077;
const KV_TO_R: i32 = 26149;
const KU_TO_G: i32 = 6419;
const KV_TO_G: i32 = 13320;
const KU_TO_B: i32 = 33050;
const YUV_HALF2: i32 = 1 << 13;
const KR_CST: i32 = -KY_SCALE * 16 - KV_TO_R * 128 + YUV_HALF2;
const KG_CST: i32 = -KY_SCALE * 16 + KU_TO_G * 128 + KV_TO_G * 128 + YUV_HALF2;
const KB_CST: i32 = -KY_SCALE * 16 - KU_TO_B * 128 + YUV_HALF2;

#[inline]
fn yuv_to_r(y: i32, v: i32) -> u8 {
    ((KY_SCALE * y + KV_TO_R * v + KR_CST) >> 14).clamp(0, 255) as u8
}

#[inline]
fn yuv_to_g(y: i32, u: i32, v: i32) -> u8 {
    ((KY_SCALE * y - KU_TO_G * u - KV_TO_G * v + KG_CST) >> 14).clamp(0, 255) as u8
}

#[inline]
fn yuv_to_b(y: i32, u: i32) -> u8 {
    ((KY_SCALE * y + KU_TO_B * u + KB_CST) >> 14).clamp(0, 255) as u8
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
    for j in 0..h {
        for i in 0..w {
            // 4:2:0 chroma subsampling — one (u, v) sample per 2×2 luma.
            let y_val = vf.y[j * y_stride + i] as i32;
            let u_val = vf.u[(j / 2) * uv_stride + (i / 2)] as i32;
            let v_val = vf.v[(j / 2) * uv_stride + (i / 2)] as i32;
            let idx = (j * w + i) * 4;
            out[idx] = yuv_to_r(y_val, v_val);
            out[idx + 1] = yuv_to_g(y_val, u_val, v_val);
            out[idx + 2] = yuv_to_b(y_val, u_val);
            out[idx + 3] = 0xff;
        }
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
