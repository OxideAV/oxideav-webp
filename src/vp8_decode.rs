//! §2.5 `VP8 ` (lossy) bitstream decode → interleaved RGBA.
//!
//! Round 124 wires the `VP8 ` lossy path that the round-6
//! [`crate::vp8_chunk::WebpLossyChunk`] handle previously only
//! *routed*: the chunk payload is handed to the `oxideav-vp8`
//! sibling crate's [`oxideav_vp8::decode_vp8`] entry point, which
//! returns a fully reconstructed, loop-filtered I420 key-frame
//! ([`oxideav_vp8::Vp8DecodedFrame`]). This module converts that
//! YUV 4:2:0 picture to the crate-wide interleaved 8-bit `[R, G, B, A]`
//! surface, applying nearest-neighbour chroma up-sampling and the
//! ITU-R BT.601 full-range YCbCr→RGB matrix that RFC 6386 §9.2 names
//! as VP8's color space ("YUV color space similar to the YCrCb color
//! space defined in [ITU-R BT.601]"; RFC 9649 §10 likewise cites
//! BT.601).
//!
//! This module performs **no container walking** — the caller hands
//! it the already-extracted `VP8 ` bitstream slice and the visible
//! dimensions. Alpha is filled opaque (`0xff`); a §2.7.1.2 `ALPH`
//! chunk's decoded plane is layered on by the caller in
//! [`crate::decode_webp_image`] (`VP8 ` + `ALPH` extended-lossy).

use oxideav_vp8::{decode_vp8, Vp8DecodedFrame};

/// Decode a §2.5 `VP8 ` lossy bitstream to interleaved RGBA.
///
/// `bitstream` is the full `VP8 ` chunk payload (the
/// [`crate::vp8_chunk::WebpLossyChunk::bitstream`] slice — the RFC 6386
/// §9.1 frame tag at offset 0 included). Returns `width * height * 4`
/// tightly packed `[R, G, B, A]` bytes in scan-line order, alpha set
/// opaque, together with the visible dimensions reported by the VP8
/// key-frame header.
pub fn decode_lossy_rgba(bitstream: &[u8]) -> Result<(u32, u32, Vec<u8>), oxideav_vp8::Vp8Error> {
    let frame = decode_vp8(bitstream)?;
    let (w, h) = (frame.width, frame.height);
    let rgba = yuv420_to_rgba(&frame);
    Ok((w, h, rgba))
}

/// Convert a decoded I420 [`Vp8DecodedFrame`] to interleaved 8-bit RGBA.
///
/// The luma plane is full-resolution (`width * height`); the two
/// chroma planes are sub-sampled to `((width+1)/2) * ((height+1)/2)`,
/// so a pixel `(x, y)` reads its chroma from `(x/2, y/2)` —
/// nearest-neighbour up-sampling, the simplest spec-conformant choice
/// (RFC 6386 §2 leaves the up-sampling kernel to the decoder; only the
/// 4:2:0 sub-sampling geometry is normative). The YCbCr→RGB matrix is
/// the BT.601 full-range form RFC 6386 §9.2 cites.
fn yuv420_to_rgba(frame: &Vp8DecodedFrame) -> Vec<u8> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let uv_w = w.div_ceil(2);

    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        let y_row = y * w;
        let uv_row = (y / 2) * uv_w;
        for x in 0..w {
            let yv = frame.y[y_row + x];
            let cu = frame.u[uv_row + (x / 2)];
            let cv = frame.v[uv_row + (x / 2)];
            let [r, g, b] = ycbcr_to_rgb(yv, cu, cv);
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(0xff);
        }
    }
    rgba
}

/// ITU-R BT.601 full-range YCbCr → RGB for a single pixel.
///
/// RFC 6386 §9.2 specifies VP8's color space as "YUV color space
/// similar to the YCrCb color space defined in [ITU-R BT.601]". VP8 /
/// WebP carry full-range (0..255) samples — no 16..235 luma head-room —
/// so the conversion is the un-scaled BT.601 form:
///
/// ```text
///   R = Y                 + 1.402   * (Cr - 128)
///   G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
///   B = Y + 1.772   * (Cb - 128)
/// ```
///
/// Computed in fixed point (16-bit fractional) and clamped to `0..=255`
/// per RFC 6386 §9.2's pixel-value-clamping requirement.
fn ycbcr_to_rgb(y: u8, cb: u8, cv: u8) -> [u8; 3] {
    // BT.601 full-range coefficients in Q16 fixed point.
    //   1.402    -> 91881
    //   0.344136 -> 22554
    //   0.714136 -> 46802
    //   1.772    -> 116130
    const HALF: i32 = 1 << 15; // rounding bias for the >> 16

    let yi = y as i32;
    let d = cb as i32 - 128; // Cb - 128
    let e = cv as i32 - 128; // Cr - 128

    let r = ((yi << 16) + 91_881 * e + HALF) >> 16;
    let g = ((yi << 16) - 22_554 * d - 46_802 * e + HALF) >> 16;
    let b = ((yi << 16) + 116_130 * d + HALF) >> 16;

    [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
}

/// Clamp an `i32` into the `0..=255` byte range (RFC 6386 §9.2).
#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycbcr_neutral_chroma_is_grey() {
        // Cb = Cr = 128 (neutral) → R = G = B = Y.
        for y in [0u8, 1, 64, 127, 128, 200, 255] {
            assert_eq!(ycbcr_to_rgb(y, 128, 128), [y, y, y]);
        }
    }

    #[test]
    fn ycbcr_pure_primaries_round_to_expected() {
        // Y=128, max Cr → strong red, near-zero green/blue.
        let red = ycbcr_to_rgb(128, 128, 255);
        assert_eq!(red[0], 255); // R saturates (128 + 1.402*127 ≈ 306 → 255)
        assert!(red[1] < 60, "green low for max-Cr, got {}", red[1]);
        assert_eq!(red[2], 128); // B unchanged by Cr at neutral Cb

        // Y=128, max Cb → strong blue.
        let blue = ycbcr_to_rgb(128, 255, 128);
        assert_eq!(blue[2], 255); // B saturates
        assert_eq!(blue[0], 128); // R unchanged by Cb at neutral Cr
        assert!(blue[1] < 100, "green low for max-Cb, got {}", blue[1]);
    }

    #[test]
    fn ycbcr_clamps_out_of_range() {
        // White luma + max Cr can't exceed 255.
        assert_eq!(ycbcr_to_rgb(255, 128, 255)[0], 255);
        // Black luma + min Cr can't go below 0.
        assert_eq!(ycbcr_to_rgb(0, 128, 0)[0], 0);
    }

    #[test]
    fn yuv420_to_rgba_produces_flat_buffer_with_opaque_alpha() {
        // Hand-build a 2x2 grey I420 frame (neutral chroma) and confirm
        // the conversion length + alpha invariants. One chroma sample
        // covers the whole 2x2 luma block.
        let frame = Vp8DecodedFrame {
            width: 2,
            height: 2,
            y: vec![10, 20, 30, 40],
            u: vec![128],
            v: vec![128],
        };
        let rgba = yuv420_to_rgba(&frame);
        assert_eq!(rgba.len(), 2 * 2 * 4);
        // Neutral chroma → grey; alpha opaque.
        assert_eq!(&rgba[0..4], &[10, 10, 10, 0xff]);
        assert_eq!(&rgba[4..8], &[20, 20, 20, 0xff]);
        assert_eq!(&rgba[8..12], &[30, 30, 30, 0xff]);
        assert_eq!(&rgba[12..16], &[40, 40, 40, 0xff]);
    }

    #[test]
    fn yuv420_to_rgba_handles_odd_dimensions() {
        // 3x1 luma → uv_w = (3+1)/2 = 2; pixels 0,1 share u[0], pixel 2
        // uses u[1]. Confirms the (x/2) chroma index + opaque alpha.
        let frame = Vp8DecodedFrame {
            width: 3,
            height: 1,
            y: vec![100, 110, 120],
            u: vec![128, 128],
            v: vec![128, 128],
        };
        let rgba = yuv420_to_rgba(&frame);
        assert_eq!(rgba.len(), 3 * 4);
        assert_eq!(&rgba[0..4], &[100, 100, 100, 0xff]);
        assert_eq!(&rgba[4..8], &[110, 110, 110, 0xff]);
        assert_eq!(&rgba[8..12], &[120, 120, 120, 0xff]);
    }
}
