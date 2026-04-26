//! `oxideav_core::Encoder` adapter that produces a full `.webp` file
//! using the VP8 lossy path.
//!
//! Two input pixel formats are accepted:
//!
//! * **`Yuv420P`** — the native VP8 format. We feed it directly to
//!   [`oxideav_vp8::encoder::encode_keyframe`] and emit a simple-file
//!   `RIFF/WEBP/VP8 ` container.
//! * **`Rgba`** — VP8 itself is RGB-only, but the WebP container adds
//!   alpha support via a separate `ALPH` chunk (§5.2.3 of the WebP
//!   spec). When given an RGBA frame we convert the RGB plane to
//!   YUV420P for the VP8 keyframe, encode the alpha plane as a
//!   VP8L-compressed green-only bitstream, and emit an extended
//!   `RIFF/WEBP/VP8X + ALPH + VP8 ` container. The VP8X header
//!   advertises the ALPHA flag + canvas size so any compliant reader
//!   picks up the sidecar.
//!
//! Registered under the crate-level codec id [`crate::CODEC_ID_VP8`]
//! (`"webp_vp8"`), a sibling of the existing `webp_vp8l` lossless id.
//! The corresponding read path is the WebP container demuxer —
//! callers wanting to decode the output can feed the bytes directly
//! to [`crate::decode_webp`], which handles both simple and extended
//! layouts with or without ALPH.
//!
//! Scope (v2):
//!   * single-frame still images only (no animated `ANMF` chunks);
//!   * RGB→YUV conversion uses the BT.601 limited-range coefficients
//!     (matches the decoder's inverse matrix);
//!   * ALPH compression is always VP8L-based (type 1, no filtering,
//!     no pre-processing). Uncompressed / filtered raw alpha (type 0)
//!     is decodable but not produced here.

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Rational, Result,
    TimeBase, VideoFrame, VideoPlane,
};

use oxideav_vp8::encoder::{encode_keyframe, DEFAULT_QINDEX};

use crate::riff::{build_webp_file, AlphChunkBytes, ImageKind, WebpMetadata};
use crate::vp8l::encode_vp8l_argb;
use crate::CODEC_ID_VP8;

/// Factory used by [`crate::register_codecs`] for the `webp_vp8` codec id.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, DEFAULT_QINDEX)
}

/// Build a VP8-lossy WebP encoder with an explicit qindex (0..=127).
/// Lower values produce higher quality at the cost of file size.
pub fn make_encoder_with_qindex(params: &CodecParameters, qindex: u8) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("VP8 WebP encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("VP8 WebP encoder: missing height"))?;
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(Error::invalid(format!(
            "VP8 WebP encoder: dimensions {width}x{height} out of range (1..=16383)"
        )));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if pix != PixelFormat::Yuv420P && pix != PixelFormat::Rgba {
        return Err(Error::unsupported(format!(
            "VP8 WebP encoder: pixel format {pix:?} not supported — feed Yuv420P or Rgba"
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(1, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(CODEC_ID_VP8);
    output_params.pixel_format = Some(pix);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.frame_rate = Some(frame_rate);

    let time_base = TimeBase::new(1, 1000);

    Ok(Box::new(Vp8WebpEncoder {
        output_params,
        width,
        height,
        qindex: qindex.min(127),
        input_format: pix,
        time_base,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct Vp8WebpEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    qindex: u8,
    input_format: PixelFormat,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for Vp8WebpEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("VP8 WebP encoder: video frames only")),
        };
        // Frame dims and pixel format are stream-level (set on the
        // encoder at construction); the pipeline upstream is responsible
        // for matching `output_params`. Dispatch on the encoder's
        // configured input format.
        let bytes = match self.input_format {
            PixelFormat::Yuv420P => {
                let vp8 = encode_keyframe(self.width, self.height, self.qindex, v)?;
                build_webp_file(
                    ImageKind::Vp8Lossy,
                    &vp8,
                    self.width,
                    self.height,
                    None,
                    &WebpMetadata::default(),
                )
            }
            PixelFormat::Rgba => encode_rgba_lossy(self.width, self.height, self.qindex, v)?,
            other => {
                return Err(Error::unsupported(format!(
                    "VP8 WebP encoder: frame format {other:?} unsupported"
                )))
            }
        };
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = pkt.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Encode an RGBA frame as VP8 lossy + ALPH sidecar + VP8X extended
/// header. Returns a complete `.webp` file.
fn encode_rgba_lossy(width: u32, height: u32, qindex: u8, v: &VideoFrame) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if v.planes.is_empty() {
        return Err(Error::invalid("VP8 WebP encoder: RGBA frame has no planes"));
    }
    let plane = &v.planes[0];
    if plane.stride < w * 4 {
        return Err(Error::invalid(
            "VP8 WebP encoder: RGBA stride too small for frame width",
        ));
    }

    // Split the input into RGB planes (we convert to YUV below) and a
    // side alpha plane.
    let mut alpha = Vec::with_capacity(w * h);
    let (y, u, v_chroma) = rgba_rows_to_yuv420(w, h, plane.stride, &plane.data, &mut alpha);
    let yuv_frame = VideoFrame {
        pts: v.pts,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: w / 2 + (w & 1),
                data: u,
            },
            VideoPlane {
                stride: w / 2 + (w & 1),
                data: v_chroma,
            },
        ],
    };
    let vp8_bytes = encode_keyframe(width, height, qindex, &yuv_frame)?;

    // Encode the alpha plane as a VP8L green-only bitstream, then strip
    // the 5-byte header (the VP8X/ALPH decoder synthesises an identical
    // header when parsing).
    let alph_payload = encode_alpha_plane_as_vp8l(width, height, &alpha)?;
    let alph = AlphChunkBytes {
        // header byte: compression=1 (VP8L), filtering=0, pre=0, reserved=0.
        header_byte: 1,
        payload: alph_payload,
    };

    Ok(build_webp_file(
        ImageKind::Vp8Lossy,
        &vp8_bytes,
        width,
        height,
        Some(&alph),
        &WebpMetadata::default(),
    ))
}

/// Convert a row-major RGBA buffer into BT.601 limited-range YUV 4:2:0
/// planes. The `alpha` output is filled with the alpha channel bytes in
/// row-major order — one byte per source pixel.
///
/// This mirrors the decoder's YUV→RGB path so a round-trip through the
/// VP8 codec preserves as much colour fidelity as possible for the
/// smooth test pattern used in the integration tests.
fn rgba_rows_to_yuv420(
    w: usize,
    h: usize,
    stride: usize,
    rgba: &[u8],
    alpha: &mut Vec<u8>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w / 2 + (w & 1);
    let ch = h / 2 + (h & 1);
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];

    // First pass: Y + alpha from every pixel.
    for j in 0..h {
        let row_start = j * stride;
        for i in 0..w {
            let px = &rgba[row_start + i * 4..row_start + i * 4 + 4];
            let r = px[0] as i32;
            let g = px[1] as i32;
            let b = px[2] as i32;
            alpha.push(px[3]);
            // BT.601 limited-range, matching the decoder's YUV→RGB
            // inverse matrix: Y = 0.257 R + 0.504 G + 0.098 B + 16.
            let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_plane[j * w + i] = y.clamp(0, 255) as u8;
        }
    }

    // Second pass: U/V averaged over 2×2 blocks.
    for cy in 0..ch {
        for cx in 0..cw {
            let mut u_sum = 0i32;
            let mut v_sum = 0i32;
            let mut n = 0i32;
            for dy in 0..2 {
                let jj = cy * 2 + dy;
                if jj >= h {
                    break;
                }
                for dx in 0..2 {
                    let ii = cx * 2 + dx;
                    if ii >= w {
                        break;
                    }
                    let px = &rgba[jj * stride + ii * 4..jj * stride + ii * 4 + 4];
                    let r = px[0] as i32;
                    let g = px[1] as i32;
                    let b = px[2] as i32;
                    // U = -0.148 R - 0.291 G + 0.439 B + 128.
                    // V =  0.439 R - 0.368 G - 0.071 B + 128.
                    u_sum += (-38 * r - 74 * g + 112 * b + 128) >> 8;
                    v_sum += (112 * r - 94 * g - 18 * b + 128) >> 8;
                    n += 1;
                }
            }
            let u = (u_sum / n) + 128;
            let v = (v_sum / n) + 128;
            u_plane[cy * cw + cx] = u.clamp(0, 255) as u8;
            v_plane[cy * cw + cx] = v.clamp(0, 255) as u8;
        }
    }

    (y_plane, u_plane, v_plane)
}

/// Compress an 8-bit alpha plane into the "header-less" VP8L bitstream
/// used in `ALPH` chunks with `compression=1`. The decoder synthesises
/// a 5-byte VP8L header (signature + dimensions + alpha/version = 0)
/// before handing the bytes to [`crate::vp8l::decode`], so we produce a
/// full VP8L stream here and drop the leading 5 bytes.
///
/// The alpha values go into the green channel of an ARGB pixel buffer
/// (R=B=0, A=0xff). The ALPH decoder extracts `((p >> 8) & 0xff)` —
/// matching exactly what we write.
fn encode_alpha_plane_as_vp8l(width: u32, height: u32, alpha: &[u8]) -> Result<Vec<u8>> {
    debug_assert_eq!(alpha.len(), (width as usize) * (height as usize));
    let mut pixels = Vec::with_capacity(alpha.len());
    for &a in alpha {
        let g = a as u32;
        pixels.push(0xff00_0000 | (g << 8));
    }
    let full_bitstream = encode_vp8l_argb(width, height, &pixels, false)?;
    // The synthesised header the decoder prepends is 5 bytes:
    // signature (1) + 14-bit width-1 + 14-bit height-1 + 1-bit alpha
    // flag (0) + 3-bit version (0) → 32 bits of packed field, written
    // LE as 4 bytes. 1 + 4 = 5. Strip them.
    if full_bitstream.len() <= 5 {
        return Err(Error::invalid(
            "VP8 WebP encoder: VP8L alpha bitstream too short to strip header",
        ));
    }
    Ok(full_bitstream[5..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riff_wrapper_layout_even_payload() {
        // Simple-file layout should be byte-identical to what the
        // pre-RIFF-refactor helper produced for a plain VP8 payload.
        let payload = vec![0xAAu8; 10];
        let out = build_webp_file(
            ImageKind::Vp8Lossy,
            &payload,
            16,
            16,
            None,
            &WebpMetadata::default(),
        );
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        assert_eq!(&out[12..16], b"VP8 ");
        let riff_size = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        assert_eq!(riff_size, 22);
        let chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]);
        assert_eq!(chunk_len, 10);
        assert_eq!(&out[20..30], &payload[..]);
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn riff_wrapper_layout_odd_payload_pads() {
        let payload = vec![0x55u8; 11];
        let out = build_webp_file(
            ImageKind::Vp8Lossy,
            &payload,
            16,
            16,
            None,
            &WebpMetadata::default(),
        );
        let riff_size = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        assert_eq!(riff_size, 24);
        assert_eq!(out.len(), 32);
        assert_eq!(out[31], 0x00);
    }
}
