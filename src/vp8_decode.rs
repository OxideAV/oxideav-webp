//! §2.5 `VP8 ` (lossy) bitstream decode → packed RGBA.
//!
//! The `VP8 ` chunk payload is handed to the `oxideav-vp8` sibling crate's
//! [`oxideav_vp8::decode_vp8`] entry point, which returns the fully
//! reconstructed, loop-filtered I420 key-frame
//! ([`oxideav_vp8::Vp8DecodedFrame`]). This module converts that 4:2:0
//! picture to packed 8-bit `[R, G, B, A]` with the exact limited-range
//! Rec. ITU-R BT.601 kernel in the `yuv` module (RFC 9649 §2.5 /
//! RFC 6386 §9.2) and nearest-neighbour chroma upsampling.
//!
//! The contract path ([`crate::decode`]) keeps the planes **native**
//! ([`crate::PixelFormat::Yuv420P`]) and converts on request through
//! [`crate::WebpImage::to_rgba8`]; this module is the one-shot packed
//! convenience the benches isolate. No container walking happens here —
//! the caller hands over the extracted `VP8 ` bitstream; alpha is filled
//! opaque (the `ALPH` plane is layered on by the caller).

use oxideav_vp8::{decode_vp8, DecodeError, Vp8DecodedFrame};

/// Decode a §2.5 `VP8 ` lossy bitstream (the `VP8 ` chunk payload, RFC
/// 6386 §9.1 frame tag included) to `(width, height, rgba)` — packed
/// `[R, G, B, A]` bytes, alpha `0xff`.
pub fn decode_lossy_rgba(bitstream: &[u8]) -> Result<(u32, u32, Vec<u8>), DecodeError> {
    let frame = decode_vp8(bitstream)?;
    let (w, h) = (frame.width, frame.height);
    let rgba = yuv420_to_rgba(&frame);
    Ok((w, h, rgba))
}

/// Convert a decoded I420 [`Vp8DecodedFrame`] to packed 8-bit RGBA with
/// the limited-range BT.601 kernel (see the `yuv` module).
///
/// The luma plane is `width × height`; the chroma planes are
/// `⌈width/2⌉ × ⌈height/2⌉`, so pixel `(x, y)` reads its chroma from
/// `(x/2, y/2)`. Kept `pub` for `benches/lossy_decode.rs`.
pub fn yuv420_to_rgba(frame: &Vp8DecodedFrame) -> Vec<u8> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let uv = w.div_ceil(2) * h.div_ceil(2);
    crate::yuv::yuv420_to_rgba(
        w,
        h,
        &frame.y[..w * h],
        &frame.u[..uv],
        &frame.v[..uv],
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuv420_to_rgba_produces_flat_buffer_with_opaque_alpha() {
        // Neutral chroma → grey through the limited-range kernel:
        // Y′ 16 → 0, 235 → 255, 126 → 128.
        let frame = Vp8DecodedFrame {
            width: 2,
            height: 2,
            y: vec![16, 235, 126, 16],
            u: vec![128],
            v: vec![128],
        };
        let rgba = yuv420_to_rgba(&frame);
        assert_eq!(rgba.len(), 2 * 2 * 4);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0xff]);
        assert_eq!(&rgba[4..8], &[255, 255, 255, 0xff]);
        assert_eq!(&rgba[8..12], &[128, 128, 128, 0xff]);
        assert_eq!(&rgba[12..16], &[0, 0, 0, 0xff]);
    }

    #[test]
    fn yuv420_to_rgba_handles_odd_dimensions() {
        // 3x1 luma → uv_w = 2; pixels 0,1 share u[0], pixel 2 uses u[1].
        let frame = Vp8DecodedFrame {
            width: 3,
            height: 1,
            y: vec![126, 126, 126],
            u: vec![128, 128],
            v: vec![128, 128],
        };
        let rgba = yuv420_to_rgba(&frame);
        assert_eq!(
            rgba,
            vec![128, 128, 128, 0xff, 128, 128, 128, 0xff, 128, 128, 128, 0xff]
        );
    }

    #[test]
    fn yuv420_to_rgba_matches_the_image_path() {
        // The one-shot packed path and `WebpImage::to_rgba8` share one
        // kernel; pin that they agree byte-for-byte.
        for &(w, h) in &[(1u32, 1u32), (3, 3), (5, 4), (17, 9), (32, 31)] {
            let (wu, hu) = (w as usize, h as usize);
            let uv = wu.div_ceil(2) * hu.div_ceil(2);
            let frame = Vp8DecodedFrame {
                width: w,
                height: h,
                y: (0..wu * hu).map(|i| ((i * 37 + 11) & 0xff) as u8).collect(),
                u: (0..uv).map(|i| ((i * 53 + 200) & 0xff) as u8).collect(),
                v: (0..uv).map(|i| ((i * 71 + 7) & 0xff) as u8).collect(),
            };
            let img = crate::WebpImage::from_yuv420(
                w,
                h,
                frame.y.clone(),
                frame.u.clone(),
                frame.v.clone(),
            )
            .unwrap();
            assert_eq!(yuv420_to_rgba(&frame), img.to_rgba8(), "{w}x{h}");
        }
    }
}
