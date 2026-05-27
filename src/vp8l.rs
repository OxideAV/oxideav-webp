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
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for &argb in &self.pixels {
            out.push((argb >> 16) as u8); // R
            out.push((argb >> 8) as u8); // G
            out.push(argb as u8); // B
            out.push((argb >> 24) as u8); // A
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
}
