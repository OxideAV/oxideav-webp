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
//!
//! The error surface is `oxideav-vp8`'s published
//! [`oxideav_vp8::DecodeError`]. The crate also defines a `Vp8Error`
//! umbrella enum (on vp8 master, commit `d85d244`) that the
//! published surface wants a
//! `From<oxideav_vp8::Vp8Error> for WebpError` adapter against, but that
//! type is **not yet on crates.io** (it landed after the v0.2.0 tag), so
//! the adapter is deferred until vp8 publishes a release carrying it.
//! See `lib.rs` for the temporary `From<DecodeError>` adapters used in
//! the meantime.

use oxideav_vp8::{decode_vp8, DecodeError, Vp8DecodedFrame};

/// Decode a §2.5 `VP8 ` lossy bitstream to interleaved RGBA.
///
/// `bitstream` is the full `VP8 ` chunk payload (the
/// [`crate::vp8_chunk::WebpLossyChunk::bitstream`] slice — the RFC 6386
/// §9.1 frame tag at offset 0 included). Returns `width * height * 4`
/// tightly packed `[R, G, B, A]` bytes in scan-line order, alpha set
/// opaque, together with the visible dimensions reported by the VP8
/// key-frame header.
///
/// The error surface is `oxideav-vp8`'s [`oxideav_vp8::DecodeError`] —
/// the published 0.2.0 decoder error. (The crate's `Vp8Error` umbrella
/// is not yet on crates.io; the published `From<oxideav_vp8::Vp8Error>`
/// adapter is deferred until vp8 publishes it — see the module-level
/// note.)
pub fn decode_lossy_rgba(bitstream: &[u8]) -> Result<(u32, u32, Vec<u8>), DecodeError> {
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
///
/// This is the per-pixel lossy-decode reconstruction hot loop owned by
/// this crate (the entropy decode + inverse transform are the sibling
/// `oxideav-vp8` decoder's; everything this crate runs after the I420
/// picture comes back happens here). It is `pub` so the
/// `benches/lossy_decode.rs` harness can isolate it from the sibling
/// decode — see `BENCHMARKS.md`. Exposing it does not change decoded
/// bytes: it is the same function `decode_lossy_rgba` calls.
pub fn yuv420_to_rgba(frame: &Vp8DecodedFrame) -> Vec<u8> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let uv_w = w.div_ceil(2);

    let mut rgba = vec![0u8; w * h * 4];
    let y_plane = &frame.y[..w * h];
    let u_plane = &frame.u[..uv_w * h.div_ceil(2)];
    let v_plane = &frame.v[..uv_w * h.div_ceil(2)];

    for y in 0..h {
        let y_row = &y_plane[y * w..y * w + w];
        let uv_base = (y / 2) * uv_w;
        let u_row = &u_plane[uv_base..uv_base + uv_w];
        let v_row = &v_plane[uv_base..uv_base + uv_w];
        let out_row = &mut rgba[y * w * 4..y * w * 4 + w * 4];

        // §9.2 nearest-neighbour 4:2:0: the two luma pixels (2k, 2k+1)
        // sub-sampled to chroma column k share one (Cb, Cr) sample. The
        // matrix's chroma terms depend only on (Cb-128, Cr-128), so they
        // are computed once per chroma column and reused across the pair —
        // only the luma offset `Y << 16` differs between the two pixels.
        // The per-pixel arithmetic (add the luma offset, +HALF, >> 16,
        // clamp) is byte-for-byte identical to `ycbcr_to_rgb`.
        let mut k = 0usize;
        let mut x = 0usize;
        while x + 1 < w {
            let (cr_off, cg_off, cb_off) = chroma_offsets(u_row[k], v_row[k]);

            let p0 = x * 4;
            let y0 = y_row[x] as i32;
            out_row[p0] = clamp_u8(((y0 << 16) + cr_off) >> 16);
            out_row[p0 + 1] = clamp_u8(((y0 << 16) + cg_off) >> 16);
            out_row[p0 + 2] = clamp_u8(((y0 << 16) + cb_off) >> 16);
            out_row[p0 + 3] = 0xff;

            let p1 = (x + 1) * 4;
            let y1 = y_row[x + 1] as i32;
            out_row[p1] = clamp_u8(((y1 << 16) + cr_off) >> 16);
            out_row[p1 + 1] = clamp_u8(((y1 << 16) + cg_off) >> 16);
            out_row[p1 + 2] = clamp_u8(((y1 << 16) + cb_off) >> 16);
            out_row[p1 + 3] = 0xff;

            k += 1;
            x += 2;
        }
        // Odd-width tail: one luma pixel maps to the last chroma column.
        if x < w {
            let (cr_off, cg_off, cb_off) = chroma_offsets(u_row[k], v_row[k]);
            let p = x * 4;
            let yv = y_row[x] as i32;
            out_row[p] = clamp_u8(((yv << 16) + cr_off) >> 16);
            out_row[p + 1] = clamp_u8(((yv << 16) + cg_off) >> 16);
            out_row[p + 2] = clamp_u8(((yv << 16) + cb_off) >> 16);
            out_row[p + 3] = 0xff;
        }
    }
    rgba
}

/// The Q16 chroma contributions (`+HALF` rounding bias folded in) shared by
/// both luma pixels of a 4:2:0 pair. Returns the R, G, B offsets to which the
/// per-pixel `Y << 16` term is added before the `>> 16` and clamp. Splitting
/// these out of `ycbcr_to_rgb` lets `yuv420_to_rgba` evaluate them once per
/// chroma sample instead of once per output pixel — the result of
/// `(Y << 16) + offset >> 16` is identical to the original per-pixel form.
#[inline]
fn chroma_offsets(cb: u8, cv: u8) -> (i32, i32, i32) {
    const HALF: i32 = 1 << 15; // rounding bias for the >> 16
    let d = cb as i32 - 128; // Cb - 128
    let e = cv as i32 - 128; // Cr - 128
    (
        91_881 * e + HALF,
        -22_554 * d - 46_802 * e + HALF,
        116_130 * d + HALF,
    )
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
///
/// The production [`yuv420_to_rgba`] loop folds this into the
/// [`chroma_offsets`] pair-hoist (one chroma evaluation per 4:2:0 column,
/// reused across both luma pixels) rather than calling per pixel; this
/// single-pixel form is retained as the byte-for-byte reference oracle the
/// `tests` module checks the hoisted loop against.
#[cfg(test)]
fn ycbcr_to_rgb(y: u8, cb: u8, cv: u8) -> [u8; 3] {
    // BT.601 full-range coefficients in Q16 fixed point.
    //   1.402    -> 91881
    //   0.344136 -> 22554
    //   0.714136 -> 46802
    //   1.772    -> 116130
    let yi = (y as i32) << 16;
    let (cr_off, cg_off, cb_off) = chroma_offsets(cb, cv);

    let r = (yi + cr_off) >> 16;
    let g = (yi + cg_off) >> 16;
    let b = (yi + cb_off) >> 16;

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

    /// Reference per-pixel reconstruction: the pre-round-290 shape, kept
    /// here as an independent oracle so the chroma-pair-hoisted
    /// [`yuv420_to_rgba`] is proven byte-for-byte identical to evaluating
    /// the full `ycbcr_to_rgb` matrix once per output pixel.
    fn yuv420_to_rgba_reference(frame: &Vp8DecodedFrame) -> Vec<u8> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let uv_w = w.div_ceil(2);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            let y_row = y * w;
            let uv_row = (y / 2) * uv_w;
            for x in 0..w {
                let [r, g, b] = ycbcr_to_rgb(
                    frame.y[y_row + x],
                    frame.u[uv_row + (x / 2)],
                    frame.v[uv_row + (x / 2)],
                );
                rgba.extend_from_slice(&[r, g, b, 0xff]);
            }
        }
        rgba
    }

    #[test]
    fn yuv420_to_rgba_matches_per_pixel_reference_across_dimensions() {
        // Deterministic non-neutral chroma across a spread of even/odd
        // dimensions; the optimized pair-hoisted loop must equal the
        // per-pixel reference byte-for-byte (the bit-identity guarantee).
        for &(w, h) in &[
            (1u32, 1u32),
            (2, 2),
            (3, 1),
            (1, 3),
            (3, 3),
            (5, 4),
            (16, 16),
            (17, 9),
            (32, 31),
        ] {
            let (wu, hu) = (w as usize, h as usize);
            let uv_w = wu.div_ceil(2);
            let uv_h = hu.div_ceil(2);
            let y: Vec<u8> = (0..wu * hu).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
            let u: Vec<u8> = (0..uv_w * uv_h)
                .map(|i| ((i * 53 + 200) & 0xff) as u8)
                .collect();
            let v: Vec<u8> = (0..uv_w * uv_h)
                .map(|i| ((i * 71 + 7) & 0xff) as u8)
                .collect();
            let frame = Vp8DecodedFrame {
                width: w,
                height: h,
                y,
                u,
                v,
            };
            assert_eq!(
                yuv420_to_rgba(&frame),
                yuv420_to_rgba_reference(&frame),
                "mismatch at {}x{}",
                w,
                h
            );
        }
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
