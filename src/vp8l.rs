//! Published-API `oxideav_webp::vp8l` module — the §3.4 / §4–§6 VP8L
//! lossless surface grouped under its qualified path.
//!
//! Per `API-COMPAT-0.1.2.md` consumers reach the lossless encode /
//! decode entry points either at the crate root
//! (`oxideav_webp::encode_vp8l_argb`) or via this module
//! (`oxideav_webp::vp8l::encode_vp8l_argb`).
//!
//! The sub-modules ([`bit_reader`], [`huffman`], [`encoder`],
//! [`transform`]) are re-export shims over the in-crate file layout
//! (`vp8l_stream`, `vp8l_prefix`, `vp8l_encode`, `vp8l_transform`) so
//! the published-0.1.2 qualified paths line up.

/// §3.4 VP8L image-header signature byte. The first byte of a bare
/// VP8L bitstream is always `0x2F`; the bytes that follow are the
/// 14-bit `width - 1`, the 14-bit `height - 1`, the `alpha_is_used`
/// flag, and the 3-bit `version_number` field.
pub const VP8L_SIGNATURE: u8 = 0x2F;

/// Decode a bare §3.4 VP8L bitstream to a [`Vp8lImage`].
///
/// `buf` is the **chunk payload** — the bytes starting at the 5-byte
/// VP8L image header, **not** a complete `RIFF/WEBP` file. For a full
/// `.webp`, use [`crate::decode_webp`] (which routes the VP8L chunk
/// here internally).
pub fn decode(buf: &[u8]) -> Result<Vp8lImage, crate::WebpError> {
    // The bare-bitstream entry point matches the published 0.1.2 shape:
    // it does **not** walk the RIFF/WEBP container — `buf` is the VP8L
    // chunk payload (image-header + image stream). The 5-byte header
    // carries width / height / alpha_is_used; we then run the full §4
    // inverse-transform chain over the §5/§6 entropy-coded body.
    if buf.is_empty() || buf[0] != VP8L_SIGNATURE {
        return Err(crate::WebpError::InvalidData);
    }
    // Reuse the in-crate VP8L chunk header reader for width / height /
    // alpha-is-used extraction.
    let chunk = crate::vp8l_chunk::WebpLosslessChunk::from_payload(buf)
        .map_err(|_| crate::WebpError::InvalidData)?;
    let width = chunk.width();
    let height = chunk.height();
    let has_alpha = chunk.alpha_is_used();
    let image = crate::vp8l_transform::decode_lossless(chunk.bitstream(), width, height)
        .map_err(|_| crate::WebpError::InvalidData)?;
    Ok(Vp8lImage {
        width,
        height,
        pixels: image.pixels().to_vec(),
        has_alpha,
    })
}

// The published 0.1.2 `vp8l::encode_vp8l_argb` returns
// `Result<Vec<u8>, WebpError>` (the coarse published error type), not
// the rich internal [`crate::vp8l_encode::EncodeError`]. Re-export the
// crate-root wrappers that already do that conversion.
pub use crate::{encode_vp8l_argb, encode_vp8l_argb_with, encode_vp8l_argb_with_metadata};

/// A fully decoded VP8L bitstream: dimensions, ARGB pixels in scan
/// order, and the §3.4 `alpha_is_used` header bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vp8lImage {
    /// Image width in pixels (1..=16384, the §3.4 14-bit limit + 1).
    pub width: u32,
    /// Image height in pixels (1..=16384).
    pub height: u32,
    /// `width * height` packed ARGB values, scan-line order. Each
    /// pixel is `(alpha << 24) | (red << 16) | (green << 8) | blue`.
    pub pixels: Vec<u32>,
    /// §3.4 `alpha_is_used` flag from the image header.
    pub has_alpha: bool,
}

impl Vp8lImage {
    /// Repack [`Self::pixels`] into interleaved 8-bit `[R, G, B, A]`
    /// bytes — `oxideav_core::PixelFormat::Rgba`, row-major, no stride
    /// padding.
    ///
    /// Round-170 rewrite (see `BENCHMARKS.md`):
    ///
    /// * **Scalar path (default):** writes four bytes per pixel into a
    ///   pre-sized buffer via `chunks_exact_mut(4)`, eliminating the
    ///   `Vec::push` bounds check + capacity growth that the old
    ///   one-byte-at-a-time loop incurred. The compiler also auto-
    ///   vectorises the resulting strided byte stores.
    /// * **`simd` feature (nightly only):** opt-in `std::simd` shuffle
    ///   path that processes 4 ARGB pixels per iteration via a 16-byte
    ///   shuffle of the underlying `[A, R, G, B]` channel layout into
    ///   `[R, G, B, A]`. Byte-identical to the scalar path; gated by
    ///   the crate's `simd` cargo feature.
    pub fn to_rgba(&self) -> Vec<u8> {
        #[cfg(feature = "simd")]
        {
            self.to_rgba_simd()
        }
        #[cfg(not(feature = "simd"))]
        {
            self.to_rgba_scalar()
        }
    }

    /// Stable scalar ARGB-`u32` → packed `[R, G, B, A]` byte repack.
    ///
    /// Reachable as the SIMD-path fallback when the `simd` feature is
    /// off, and as the direct entry point for tests that need to
    /// pin the scalar path independently of the build feature set.
    pub fn to_rgba_scalar(&self) -> Vec<u8> {
        let n = self.pixels.len();
        let mut out = vec![0u8; n * 4];
        for (chunk, &argb) in out.chunks_exact_mut(4).zip(self.pixels.iter()) {
            // [R, G, B, A] order, matching `oxideav_core::PixelFormat::Rgba`.
            chunk[0] = (argb >> 16) as u8;
            chunk[1] = (argb >> 8) as u8;
            chunk[2] = argb as u8;
            chunk[3] = (argb >> 24) as u8;
        }
        out
    }

    /// `std::simd` (nightly, `simd` feature) ARGB → packed `[R, G, B, A]`
    /// byte repack. Bit-identical to [`Self::to_rgba_scalar`].
    ///
    /// Processes four ARGB pixels (16 bytes) per iteration with a
    /// single byte-lane shuffle. The shuffle index list maps the
    /// little-endian on-wire ARGB byte layout
    /// `[B0, G0, R0, A0, B1, G1, R1, A1, …]` to the published
    /// `[R0, G0, B0, A0, R1, G1, B1, A1, …]` output layout.
    #[cfg(feature = "simd")]
    pub fn to_rgba_simd(&self) -> Vec<u8> {
        use core::simd::{simd_swizzle, u8x16};
        let n = self.pixels.len();
        let mut out = vec![0u8; n * 4];
        // ARGB `u32` in memory (little-endian) is the byte sequence
        // [B, G, R, A]. We want [R, G, B, A]. Per 4-pixel block of
        // 16 bytes the mapping is:
        //   in:  B0 G0 R0 A0 | B1 G1 R1 A1 | B2 G2 R2 A2 | B3 G3 R3 A3
        //   out: R0 G0 B0 A0 | R1 G1 B1 A1 | R2 G2 B2 A2 | R3 G3 B3 A3
        // i.e. within each 4-byte group swap byte 0 (B) with byte 2 (R).
        // (Indices are into the 16-element source vector.)
        //
        // Reinterpret the `&[u32]` pixel buffer as `&[u8]` so we can
        // pull 16 bytes per iteration in one load, then write 16 bytes
        // per iteration in one store. SAFETY: `[u32]` is layout-
        // compatible with `[u8; 4]` on every target the rest of the
        // crate already supports (the same channel-extraction code in
        // `to_rgba_scalar` relies on this), and the slice length is
        // exactly `n * 4` bytes.
        let pix_bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(self.pixels.as_ptr() as *const u8, n * 4) };
        let main_len = (n / 4) * 16;
        let mut src_iter = pix_bytes[..main_len].chunks_exact(16);
        let mut dst_iter = out[..main_len].chunks_exact_mut(16);
        for (src, dst) in (&mut src_iter).zip(&mut dst_iter) {
            let v = u8x16::from_slice(src);
            let shuffled: u8x16 =
                simd_swizzle!(v, [2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15]);
            dst.copy_from_slice(shuffled.as_array());
        }
        // Tail (0..3 pixels): scalar repack.
        let tail_pixels = &self.pixels[(n / 4) * 4..];
        for (chunk, &argb) in out[main_len..].chunks_exact_mut(4).zip(tail_pixels.iter()) {
            chunk[0] = (argb >> 16) as u8;
            chunk[1] = (argb >> 8) as u8;
            chunk[2] = argb as u8;
            chunk[3] = (argb >> 24) as u8;
        }
        out
    }
}

/// One §6.2 prefix-code group: the five canonical Huffman codes that
/// jointly decode a single pixel (`G` + `R` + `B` + `A` + `distance`).
///
/// Re-exposed as an opaque published-API handle on the
/// `oxideav_webp::vp8l` surface. Construction / inspection is left to
/// the in-crate decoder + encoder paths.
#[derive(Debug, Default)]
pub struct HuffmanGroup {
    _private: (),
}

impl HuffmanGroup {
    /// Construct a fresh, empty group. Useful as a placeholder when
    /// driving the API from a test harness; the real decode / encode
    /// paths populate the group internally.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

// ───────────────────── sub-module re-exports ─────────────────────

/// `oxideav_webp::vp8l::bit_reader` — re-export of the §4 / §5
/// bitstream reader.
pub mod bit_reader {
    pub use crate::vp8l_stream::{BitReader, BitReaderEof};
}

/// `oxideav_webp::vp8l::huffman` — re-export of the §6.2 prefix-code
/// reader plus the high-level [`super::HuffmanGroup`] handle.
pub mod huffman {
    pub use super::HuffmanGroup;
    pub use crate::vp8l_prefix::{PrefixCode, PrefixError};
}

/// `oxideav_webp::vp8l::transform` — re-export of the §4
/// inverse-transform chain plus the bare-bitstream lossless decoder.
pub mod transform {
    pub use crate::vp8l_transform::{
        decode_lossless, decode_lossless_headerless, inverse_color, inverse_color_indexing,
        inverse_color_table, inverse_predictor, inverse_subtract_green,
    };
}

/// `oxideav_webp::vp8l::encoder` — re-export of the §3.5 / §3.7 VP8L
/// lossless encoder (bare bitstream entry points).
///
/// Returns the coarse published [`crate::WebpError`] (not the rich
/// internal `EncodeError`) so the contract `Result<Vec<u8>, WebpError>`
/// shape resolves. The richer error is reachable via the
/// [`EncodeError`] re-export on this module.
pub mod encoder {
    pub use crate::vp8l_encode::EncodeError;
    pub use crate::{encode_vp8l_argb, encode_vp8l_argb_with};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp8l_encode::encode_vp8l_argb_with;

    #[test]
    fn signature_constant_matches_rfc_9649_3_4() {
        assert_eq!(VP8L_SIGNATURE, 0x2F);
    }

    #[test]
    fn vp8l_image_to_rgba_round_trip_shape() {
        let img = Vp8lImage {
            width: 2,
            height: 1,
            pixels: vec![0xff_aa_bb_cc, 0xff_11_22_33],
            has_alpha: false,
        };
        let rgba = img.to_rgba();
        assert_eq!(rgba.len(), 2 * 4);
        // First pixel: alpha=0xff, r=0xaa, g=0xbb, b=0xcc → [aa, bb, cc, ff].
        assert_eq!(&rgba[0..4], &[0xaa, 0xbb, 0xcc, 0xff]);
        assert_eq!(&rgba[4..8], &[0x11, 0x22, 0x33, 0xff]);
    }

    #[test]
    fn bare_decode_round_trips_through_published_path() {
        // Encode a 2x2 ARGB image to a bare VP8L bitstream, then decode
        // it back via the published `vp8l::decode`. The round-tripped
        // pixels must match byte-for-byte.
        let (w, h) = (2u32, 2u32);
        let argb = vec![
            0xff_00_00_00u32,
            0xff_ff_00_00,
            0xff_00_ff_00,
            0xff_00_00_ff,
        ];
        let bare = encode_vp8l_argb_with(&argb, w, h, false).expect("encode");
        let img = decode(&bare).expect("decode");
        assert_eq!(img.width, w);
        assert_eq!(img.height, h);
        assert_eq!(img.pixels, argb);
    }

    #[test]
    fn bare_decode_rejects_bad_signature() {
        let err = decode(&[0x00, 0x00, 0x00, 0x00, 0x00]).expect_err("bad sig");
        assert_eq!(err, crate::WebpError::InvalidData);
    }

    #[test]
    fn to_rgba_scalar_matches_published_layout() {
        // Round-170 scalar rewrite must produce byte-identical output
        // to the legacy per-pixel push loop, on a buffer wide enough
        // to exercise the chunk_exact_mut(4) path's main body and the
        // (zero-length) tail.
        let pixels: Vec<u32> = (0..32u32).map(|i| 0xff00_0000 | (i * 0x010101)).collect();
        let img = Vp8lImage {
            width: 32,
            height: 1,
            pixels,
            has_alpha: false,
        };
        let bytes = img.to_rgba_scalar();
        assert_eq!(bytes.len(), 32 * 4);
        // Spot-check pixel 0 and 31:
        // pixel 0  = 0xff_00_00_00 → [R=0x00, G=0x00, B=0x00, A=0xff]
        // pixel 31 = 0xff_1f_1f_1f → [R=0x1f, G=0x1f, B=0x1f, A=0xff]
        assert_eq!(&bytes[0..4], &[0x00, 0x00, 0x00, 0xff]);
        assert_eq!(&bytes[31 * 4..32 * 4], &[0x1f, 0x1f, 0x1f, 0xff]);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn to_rgba_simd_matches_scalar_byte_for_byte() {
        // Round-170 contract: turning on the `simd` cargo feature must
        // not change a single output byte. Test on a 67-pixel buffer
        // (a non-multiple of 4) so the SIMD path's tail handler is
        // also exercised.
        let pixels: Vec<u32> = (0..67u32)
            .map(|i| (i.wrapping_mul(0x6789_abcd)) | 0xff00_0000)
            .collect();
        let img = Vp8lImage {
            width: 67,
            height: 1,
            pixels,
            has_alpha: false,
        };
        let scalar = img.to_rgba_scalar();
        let simd = img.to_rgba_simd();
        assert_eq!(scalar, simd, "SIMD path must be byte-identical to scalar");
        // Also: the default `to_rgba()` dispatcher resolves to the SIMD
        // path when the feature is on, so it must also match scalar.
        assert_eq!(img.to_rgba(), scalar);
    }

    #[test]
    fn average2_swar_matches_per_channel_truncating_divide_reference() {
        // The round-170 SWAR rewrite of `vp8l_transform::average2` is
        // a guard rail of the inverse-predictor decode path. Drive it
        // through a public `predict_*` path by encoding+decoding a
        // single-block §4.1 image; if the SWAR identity drifts, the
        // round-trip's red+green+blue bytes won't match.
        let argb = vec![0xff_aa_bb_ccu32, 0xff_55_44_33];
        let bare = crate::vp8l_encode::encode_vp8l_argb_with(&argb, 2, 1, false).expect("encode");
        let img = decode(&bare).expect("decode");
        assert_eq!(img.pixels, argb);
    }
}
