//! Typed parser for the `ALPH` chunk **info byte** per RFC 9649
//! §2.7.1.2 (Figure 10).
//!
//! The §2.3 walker in [`crate::container`] surfaces an `ALPH` chunk as
//! an opaque payload whose first byte packs four 2-bit fields:
//!
//! ```text
//!  0 1 2 3 4 5 6 7
//! +-+-+-+-+-+-+-+-+
//! |Rsv| P | F | C |
//! +-+-+-+-+-+-+-+-+
//! ```
//!
//! * `Rsv` — Reserved, 2 bits. MUST be 0; readers MUST ignore.
//! * `P`   — Preprocessing, 2 bits. 0 = none, 1 = level reduction.
//!   Other values are informational (decoders are not required to act
//!   on this hint).
//! * `F`   — Filtering method, 2 bits. 0 = none, 1 = horizontal,
//!   2 = vertical, 3 = gradient.
//! * `C`   — Compression method, 2 bits. 0 = uncompressed raw,
//!   1 = WebP lossless format. Other values are not defined by RFC
//!   9649 §2.7.1.2.
//!
//! This module decodes the info byte into a typed [`AlphHeader`]; it
//! also decodes the full Alpha Bitstream that follows
//! ([`decode_alpha`]) into a width × height plane of 8-bit alpha
//! values, covering both compression methods and all four §2.7.1.2
//! filtering methods.
//!
//! ## Alpha bitstream decode ([`decode_alpha`])
//!
//! Per RFC 9649 §2.7.1.2, the alpha bitstream is either:
//!
//! * **Compression method 0** — raw, uncompressed 8-bit alpha values
//!   in scan order, of length `width * height`.
//! * **Compression method 1** — a §3 WebP-lossless *image-stream* of
//!   implicit dimensions `width x height` (no 5-byte image header).
//!   Once decoded into ARGB, "the transparency information must be
//!   extracted from the green channel of the ARGB quadruplet."
//!
//! After de-compression, the §2.7.1.2 inverse filter
//! (none / horizontal / vertical / gradient) is applied over the
//! reconstructed plane: each output alpha is
//! `(predictor + decompressed) % 256`, with the per-method predictor
//! and the documented left-most / top-most edge cases.
//!
//! ## Bit layout anchor
//!
//! The RFC's ASCII-art `|Rsv|P|F|C|` reads MSB-first within the byte,
//! giving:
//!
//! | bit (LSB=0) | field |
//! |-------------|-------|
//! | 7..6        | Rsv   |
//! | 5..4        | P     |
//! | 3..2        | F     |
//! | 1..0        | C     |
//!
//! Cross-checked against `docs/image/webp/fixtures/lossy-with-alpha-128x128/trace.txt`
//! which reports `header_byte=0x01 method=1 filter=0 pre_processing=0` for a
//! libwebp-encoded fixture — only the C nibble's LSB is set, matching
//! `compression = 1` (lossless) with everything else 0.

use core::fmt;

/// Compression method (`C`) per RFC 9649 §2.7.1.2.
///
/// The spec enumerates `0` (no compression) and `1` (WebP lossless
/// format). Higher values are not defined; we preserve them in
/// [`Self::Reserved`] so callers can refuse on encounter without the
/// parser itself imposing that policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphCompression {
    /// 0: No compression — the alpha bitstream is raw 8-bit values in
    /// scan order, of length `width * height`.
    None,
    /// 1: Lossless — the alpha bitstream is a §3 VP8L image-stream
    /// with implicit dimensions `width x height` (no header).
    Lossless,
    /// 2 or 3 — undefined by §2.7.1.2.
    Reserved(u8),
}

impl AlphCompression {
    fn from_bits(c: u8) -> Self {
        match c & 0b11 {
            0 => Self::None,
            1 => Self::Lossless,
            other => Self::Reserved(other),
        }
    }
}

/// Filtering method (`F`) per RFC 9649 §2.7.1.2.
///
/// The four values are exhaustive within the 2-bit field; the spec
/// defines a prediction rule for each (None / A / B / clip(A+B-C)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphFiltering {
    /// 0: predictor = 0 for every pixel (no filter).
    None,
    /// 1: predictor = A (the pixel to the left).
    Horizontal,
    /// 2: predictor = B (the pixel above).
    Vertical,
    /// 3: predictor = clip(A + B - C) — the gradient predictor.
    Gradient,
}

impl AlphFiltering {
    fn from_bits(f: u8) -> Self {
        match f & 0b11 {
            0 => Self::None,
            1 => Self::Horizontal,
            2 => Self::Vertical,
            3 => Self::Gradient,
            _ => unreachable!("masked to 2 bits"),
        }
    }
}

/// Preprocessing hint (`P`) per RFC 9649 §2.7.1.2.
///
/// Only `0` and `1` are named in the spec; the other two 2-bit values
/// are reserved. §2.7.1.2: "Decoders are not required to use this
/// information in any specified way." — i.e. this is purely
/// informational metadata, not a refusal trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphPreprocessing {
    /// 0: No preprocessing was applied.
    None,
    /// 1: Level reduction was applied prior to compression.
    LevelReduction,
    /// 2 or 3 — undefined by §2.7.1.2.
    Reserved(u8),
}

impl AlphPreprocessing {
    fn from_bits(p: u8) -> Self {
        match p & 0b11 {
            0 => Self::None,
            1 => Self::LevelReduction,
            other => Self::Reserved(other),
        }
    }
}

/// Errors raised by the §2.7.1.2 ALPH info-byte parser and the
/// [`decode_alpha`] bitstream decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlphError {
    /// The ALPH payload is empty — at minimum one info byte is
    /// required per §2.7.1.2 Figure 10, even if the alpha bitstream
    /// itself is zero-length (which §2.7.1.2 does not forbid).
    EmptyPayload,
    /// `width * height` overflowed `usize` (or `u32`), so the plane
    /// cannot be addressed on this platform.
    DimensionsOverflow {
        /// The implicit width passed by the caller.
        width: u32,
        /// The implicit height passed by the caller.
        height: u32,
    },
    /// Compression method 0 (raw) but the alpha bitstream length does
    /// not equal `width * height` (§2.7.1.2: "a byte sequence of
    /// length = width * height").
    RawLengthMismatch {
        /// The expected `width * height` byte count.
        expected: usize,
        /// The actual number of bytes available in the bitstream.
        actual: usize,
    },
    /// Compression method `C` was `2` or `3` — undefined by §2.7.1.2.
    UnsupportedCompression(u8),
    /// The compression-method-1 §3 VP8L image-stream failed to decode.
    Vp8l(crate::vp8l_decode::DecodeError),
}

impl fmt::Display for AlphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPayload => {
                f.write_str("ALPH payload missing the §2.7.1.2 info byte (payload length 0)")
            }
            Self::DimensionsOverflow { width, height } => write!(
                f,
                "ALPH alpha-plane dimensions {width}x{height} overflow the addressable range"
            ),
            Self::RawLengthMismatch { expected, actual } => write!(
                f,
                "ALPH raw (method 0) bitstream length {actual} != width*height {expected}"
            ),
            Self::UnsupportedCompression(c) => write!(
                f,
                "ALPH compression method {c} is undefined by §2.7.1.2 (only 0 and 1 exist)"
            ),
            Self::Vp8l(e) => write!(f, "ALPH method-1 VP8L image-stream decode: {e}"),
        }
    }
}

impl std::error::Error for AlphError {}

impl From<crate::vp8l_decode::DecodeError> for AlphError {
    fn from(e: crate::vp8l_decode::DecodeError) -> Self {
        Self::Vp8l(e)
    }
}

/// Decoded §2.7.1.2 `ALPH` info byte plus the offset at which the
/// alpha bitstream begins inside the chunk payload.
///
/// Constructed via [`AlphHeader::parse`]. The actual alpha bitstream
/// (raw or VP8L-compressed) is **not** decoded — this layer's job is
/// to surface the 2-bit `Rsv` / `P` / `F` / `C` decomposition. The
/// payload after byte 0 — `payload[1..]` — is the §2.7.1.2 "Alpha
/// bitstream" of `Chunk Size - 1` bytes; callers that need it should
/// slice the chunk payload at [`Self::bitstream_offset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphHeader {
    /// `C` field — compression method (§2.7.1.2).
    pub compression: AlphCompression,
    /// `F` field — filtering method (§2.7.1.2).
    pub filtering: AlphFiltering,
    /// `P` field — preprocessing hint (§2.7.1.2).
    pub preprocessing: AlphPreprocessing,
    /// `Rsv` field — raw 2-bit value from bits 7..6 of the info byte.
    /// §2.7.1.2 says "MUST be 0. Readers MUST ignore this field." —
    /// we surface the raw value for observability without rejecting.
    pub reserved: u8,
    /// Raw info byte, preserved for round-trip and trace assertions.
    pub info_byte: u8,
}

impl AlphHeader {
    /// Parse the `ALPH` chunk payload's §2.7.1.2 info byte.
    ///
    /// `payload` is the whole §2.3 chunk payload (i.e. the slice
    /// returned by [`crate::container::WebpChunk::payload`] for a
    /// chunk whose FourCC is [`crate::container::fourcc::ALPH`]). Only
    /// the first byte is consumed by this layer; the remainder is the
    /// alpha bitstream callers must hand off to a later VP8L or raw
    /// decode pass.
    pub fn parse(payload: &[u8]) -> Result<Self, AlphError> {
        let info = *payload.first().ok_or(AlphError::EmptyPayload)?;

        // §2.7.1.2 Figure 10: byte 0 packs Rsv|P|F|C, MSB-first.
        let reserved = (info >> 6) & 0b11;
        let p_bits = (info >> 4) & 0b11;
        let f_bits = (info >> 2) & 0b11;
        let c_bits = info & 0b11;

        Ok(Self {
            compression: AlphCompression::from_bits(c_bits),
            filtering: AlphFiltering::from_bits(f_bits),
            preprocessing: AlphPreprocessing::from_bits(p_bits),
            reserved,
            info_byte: info,
        })
    }

    /// Offset (within the ALPH chunk payload) at which the alpha
    /// bitstream begins. Always 1 per §2.7.1.2 — the info byte is
    /// followed immediately by the bitstream.
    pub const fn bitstream_offset(&self) -> usize {
        1
    }
}

/// `clip(v)` per §2.7.1.2: 0 if `v < 0`, 255 if `v > 255`, else `v`.
#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Decode a complete `ALPH` chunk payload to a `width * height` plane of
/// 8-bit alpha values, in scan order.
///
/// `payload` is the **whole** §2.3 ALPH chunk payload (the §2.7.1.2 info
/// byte followed by the alpha bitstream). `width` / `height` are the
/// implicit alpha-plane dimensions — for a still image these are the
/// `VP8X` canvas dimensions (or the `VP8 ` keyframe dimensions); for an
/// animation frame they are the `ANMF` frame dimensions.
///
/// The decode follows §2.7.1.2 in two stages:
///
/// 1. **De-compression** (`C` field): method 0 copies the raw bytes;
///    method 1 decodes the headerless §3 VP8L image-stream and lifts the
///    alpha values out of the **green** channel of each ARGB pixel.
/// 2. **Inverse filtering** (`F` field): the per-pixel predictor
///    (none / A / B / clip(A+B-C)) is added to the de-compressed value
///    modulo 256, with the §2.7.1.2 left-most / top-most edge cases.
///
/// Returns the reconstructed alpha plane (`width * height` bytes). The
/// §2.7.1.2 preprocessing (`P`) hint is informational and is **not**
/// applied here (the spec: "Decoders are not required to use this
/// information in any specified way.").
pub fn decode_alpha(payload: &[u8], width: u32, height: u32) -> Result<Vec<u8>, AlphError> {
    let header = AlphHeader::parse(payload)?;

    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(AlphError::DimensionsOverflow { width, height })?;

    // The alpha bitstream proper is everything after the info byte.
    let bitstream = &payload[header.bitstream_offset()..];

    // Stage 1 — de-compression into the raw (still-filtered) plane.
    let filtered: Vec<u8> = match header.compression {
        AlphCompression::None => {
            if bitstream.len() != count {
                return Err(AlphError::RawLengthMismatch {
                    expected: count,
                    actual: bitstream.len(),
                });
            }
            bitstream.to_vec()
        }
        AlphCompression::Lossless => {
            // §2.7.1.2: a headerless §3 image-stream of implicit
            // dimensions width x height; the alpha values live in the
            // GREEN channel of the decoded ARGB quadruplets.
            let image =
                crate::vp8l_transform::decode_lossless_headerless(bitstream, width, height)?;
            image
                .pixels()
                .iter()
                .map(|argb| (argb >> 8) as u8)
                .collect()
        }
        AlphCompression::Reserved(c) => return Err(AlphError::UnsupportedCompression(c)),
    };

    // A zero-area plane has nothing to filter.
    if count == 0 {
        return Ok(filtered);
    }

    // Stage 2 — inverse filter into the final plane.
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; count];

    // §2.7.1.2: alpha = (predictor + X) % 256. Indexing helpers below use
    // the already-reconstructed `out` plane for the A / B / C neighbours.
    let idx = |x: usize, y: usize| y * w + x;

    for y in 0..h {
        for x in 0..w {
            let xv = filtered[idx(x, y)] as i32;

            // §2.7.1.2 Figure 11 neighbours over the *reconstructed*
            // plane: A = left, B = above, C = above-left.
            let predictor: i32 = match (x, y) {
                // Top-left always predicts 0, for every filter method.
                (0, 0) => 0,
                _ => match header.filtering {
                    AlphFiltering::None => 0,
                    AlphFiltering::Horizontal => {
                        if x == 0 {
                            // Left-most (0, y>0): predicted by (0, y-1).
                            out[idx(0, y - 1)] as i32
                        } else {
                            out[idx(x - 1, y)] as i32
                        }
                    }
                    AlphFiltering::Vertical => {
                        if y == 0 {
                            // Top-most (x>0, 0): predicted by (x-1, 0).
                            out[idx(x - 1, 0)] as i32
                        } else {
                            out[idx(x, y - 1)] as i32
                        }
                    }
                    AlphFiltering::Gradient => {
                        if x == 0 {
                            // Left-most (0, y>0): predicted by (0, y-1).
                            out[idx(0, y - 1)] as i32
                        } else if y == 0 {
                            // Top-most (x>0, 0): predicted by (x-1, 0).
                            out[idx(x - 1, 0)] as i32
                        } else {
                            let a = out[idx(x - 1, y)] as i32;
                            let b = out[idx(x, y - 1)] as i32;
                            let c = out[idx(x - 1, y - 1)] as i32;
                            clip(a + b - c) as i32
                        }
                    }
                },
            };

            out[idx(x, y)] = ((predictor + xv) & 0xff) as u8;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compose an ALPH info byte from its four 2-bit fields, MSB-first.
    fn info(rsv: u8, p: u8, f: u8, c: u8) -> u8 {
        ((rsv & 0b11) << 6) | ((p & 0b11) << 4) | ((f & 0b11) << 2) | (c & 0b11)
    }

    #[test]
    fn empty_payload_is_rejected_with_named_error() {
        // §2.7.1.2 Figure 10 mandates one info byte at minimum.
        assert_eq!(AlphHeader::parse(&[]), Err(AlphError::EmptyPayload));
    }

    #[test]
    fn all_zero_info_decodes_to_none_none_none_zero() {
        // info = 0x00 → C=0, F=0, P=0, Rsv=0. The simplest legal ALPH.
        let h = AlphHeader::parse(&[0x00]).unwrap();
        assert_eq!(h.compression, AlphCompression::None);
        assert_eq!(h.filtering, AlphFiltering::None);
        assert_eq!(h.preprocessing, AlphPreprocessing::None);
        assert_eq!(h.reserved, 0);
        assert_eq!(h.info_byte, 0);
        assert_eq!(h.bitstream_offset(), 1);
    }

    #[test]
    fn compression_field_decodes_all_four_values() {
        // C nibble at bits 1..0.
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 0)]).unwrap().compression,
            AlphCompression::None
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 1)]).unwrap().compression,
            AlphCompression::Lossless
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 2)]).unwrap().compression,
            AlphCompression::Reserved(2)
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 3)]).unwrap().compression,
            AlphCompression::Reserved(3)
        );
    }

    #[test]
    fn filtering_field_decodes_all_four_methods() {
        // F nibble at bits 3..2. All four are named in §2.7.1.2.
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 0)]).unwrap().filtering,
            AlphFiltering::None
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 1, 0)]).unwrap().filtering,
            AlphFiltering::Horizontal
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 2, 0)]).unwrap().filtering,
            AlphFiltering::Vertical
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 3, 0)]).unwrap().filtering,
            AlphFiltering::Gradient
        );
    }

    #[test]
    fn preprocessing_field_decodes_both_named_values_plus_reserved() {
        // P nibble at bits 5..4. §2.7.1.2 names 0 + 1.
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::None
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 1, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::LevelReduction
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 2, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::Reserved(2)
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 3, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::Reserved(3)
        );
    }

    #[test]
    fn reserved_field_surfaces_raw_two_bit_value_without_rejection() {
        // §2.7.1.2: "MUST be 0. Readers MUST ignore this field." So a
        // non-zero Rsv must parse, with the raw value carried through.
        for rsv in 0u8..=3 {
            let h = AlphHeader::parse(&[info(rsv, 0, 0, 0)]).unwrap();
            assert_eq!(h.reserved, rsv, "Rsv={rsv}");
            // Named fields stay clean.
            assert_eq!(h.compression, AlphCompression::None);
            assert_eq!(h.filtering, AlphFiltering::None);
            assert_eq!(h.preprocessing, AlphPreprocessing::None);
        }
    }

    #[test]
    fn fields_decode_independently_across_a_full_combination() {
        // Hand-pick a byte where every nibble is non-zero & distinct:
        // Rsv=2, P=3, F=1, C=2  →  10 11 01 10  =  0xB6
        let h = AlphHeader::parse(&[0xB6]).unwrap();
        assert_eq!(h.reserved, 0b10);
        assert_eq!(h.preprocessing, AlphPreprocessing::Reserved(0b11));
        assert_eq!(h.filtering, AlphFiltering::Horizontal);
        assert_eq!(h.compression, AlphCompression::Reserved(0b10));
        assert_eq!(h.info_byte, 0xB6);
    }

    #[test]
    fn fixture_lossy_with_alpha_info_byte_decodes_to_lossless_no_filter_no_pre() {
        // docs/image/webp/fixtures/lossy-with-alpha-128x128/trace.txt
        //   ALPH method=1 filter=0 pre_processing=0 header_byte=0x01
        let h = AlphHeader::parse(&[0x01]).unwrap();
        assert_eq!(h.compression, AlphCompression::Lossless);
        assert_eq!(h.filtering, AlphFiltering::None);
        assert_eq!(h.preprocessing, AlphPreprocessing::None);
        assert_eq!(h.reserved, 0);
        assert_eq!(h.info_byte, 0x01);
    }

    #[test]
    fn bitstream_offset_is_always_one_past_the_info_byte() {
        // §2.7.1.2 "Alpha bitstream: _Chunk Size_ bytes - 1" — i.e.
        // payload[1..] for any payload that survives parse().
        let h = AlphHeader::parse(&[0x01, 0xAA, 0xBB]).unwrap();
        assert_eq!(h.bitstream_offset(), 1);
    }

    #[test]
    fn trailing_bytes_are_not_consumed_by_the_info_byte_parse() {
        // Extra bytes (the actual bitstream) must NOT change the
        // decoded info-byte fields; the parser only reads byte 0.
        let baseline = AlphHeader::parse(&[0x01]).unwrap();
        let with_tail = AlphHeader::parse(&[0x01, 0xFF, 0x00, 0x55, 0xAA]).unwrap();
        assert_eq!(baseline, with_tail);
    }

    // ---- decode_alpha: §2.7.1.2 bitstream decode ----

    /// Build an ALPH payload with filter method `f` and compression
    /// method 0 (raw): the info byte followed by the residual stream.
    fn raw_alph(f: u8, residual: &[u8]) -> Vec<u8> {
        let mut v = vec![info(0, 0, f, 0)];
        v.extend_from_slice(residual);
        v
    }

    #[test]
    fn decode_raw_uncompressed_no_filter_is_identity() {
        // §2.7.1.2 method 0 + filter 0: alpha = (0 + X) % 256 = X.
        let residual = [10u8, 5, 250, 3, 100, 200];
        let payload = raw_alph(0, &residual);
        let plane = decode_alpha(&payload, 3, 2).unwrap();
        assert_eq!(plane, residual.to_vec());
    }

    #[test]
    fn decode_raw_length_mismatch_is_rejected() {
        // method 0 requires exactly width*height residual bytes.
        let payload = raw_alph(0, &[1, 2, 3]); // 3 bytes for a 2x2 (=4) plane.
        assert_eq!(
            decode_alpha(&payload, 2, 2),
            Err(AlphError::RawLengthMismatch {
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn decode_unsupported_compression_method_is_rejected() {
        // C = 2 → Reserved(2) → UnsupportedCompression.
        let payload = vec![info(0, 0, 0, 2), 0, 0, 0, 0];
        assert_eq!(
            decode_alpha(&payload, 2, 2),
            Err(AlphError::UnsupportedCompression(2))
        );
    }

    #[test]
    fn decode_horizontal_filter_inverse() {
        // §2.7.1.2 method 1 (horizontal): predictor = A (left); the
        // left-most pixel (0, y>0) uses (0, y-1); (0,0) uses 0.
        //   X     = [10,  5, 250,   3, 100, 200]   (3x2)
        //   out   = [10, 15,   9,  13, 113,  57]
        let residual = [10u8, 5, 250, 3, 100, 200];
        let payload = raw_alph(1, &residual);
        let plane = decode_alpha(&payload, 3, 2).unwrap();
        assert_eq!(plane, vec![10, 15, 9, 13, 113, 57]);
    }

    #[test]
    fn decode_vertical_filter_inverse() {
        // §2.7.1.2 method 2 (vertical): predictor = B (above); the
        // top-most pixel (x>0, 0) uses (x-1, 0); (0,0) uses 0.
        //   X   = [10,  5, 250,  3, 100, 200]   (3x2)
        //   out = [10, 15,   9, 13, 115, 209]
        let residual = [10u8, 5, 250, 3, 100, 200];
        let payload = raw_alph(2, &residual);
        let plane = decode_alpha(&payload, 3, 2).unwrap();
        assert_eq!(plane, vec![10, 15, 9, 13, 115, 209]);
    }

    #[test]
    fn decode_gradient_filter_inverse() {
        // §2.7.1.2 method 3 (gradient): predictor = clip(A+B-C) for
        // interior pixels; left-most uses above, top-most uses left,
        // (0,0) uses 0.
        //   X   = [10,  5,  7,   3, 100,  50,  20,   8,   9]   (3x3)
        //   out = [10, 15, 22,  13, 118, 175,  33, 146, 212]
        let residual = [10u8, 5, 7, 3, 100, 50, 20, 8, 9];
        let payload = raw_alph(3, &residual);
        let plane = decode_alpha(&payload, 3, 3).unwrap();
        assert_eq!(plane, vec![10, 15, 22, 13, 118, 175, 33, 146, 212]);
    }

    #[test]
    fn decode_modulo_256_wraps_into_0_255() {
        // §2.7.1.2: "modulo-256 arithmetic to wrap the [256..511] range
        // into the [0..255] one." Horizontal, single row.
        //   X   = [200, 200]   →   out = [200, (200+200)%256 = 144]
        let payload = raw_alph(1, &[200, 200]);
        let plane = decode_alpha(&payload, 2, 1).unwrap();
        assert_eq!(plane, vec![200, 144]);
    }

    #[test]
    fn decode_gradient_clip_clamps_predictor() {
        // Force clip() to clamp high: A=255, B=255, C=0 → A+B-C=510 →
        // clip=255. Build a 2x2 whose reconstruction reaches that.
        //   X   = [255, 0, 0, 5]   (2x2)
        //   (0,0)=255; (1,0) top-most pred=255 → 255; (0,1) left-most
        //   pred=out(0,0)=255 → 255; (1,1) interior A=255 B=255 C=255 →
        //   clip(255)=255 → (255+5)%256 = 4.
        let payload = raw_alph(3, &[255, 0, 0, 5]);
        let plane = decode_alpha(&payload, 2, 2).unwrap();
        assert_eq!(plane, vec![255, 255, 255, 4]);
    }

    #[test]
    fn decode_zero_area_plane_is_empty() {
        // A 0xN or Nx0 plane decodes to an empty raw plane (length 0).
        let payload = raw_alph(0, &[]);
        assert_eq!(decode_alpha(&payload, 0, 4).unwrap(), Vec::<u8>::new());
        assert_eq!(decode_alpha(&payload, 4, 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn decode_empty_payload_is_rejected() {
        assert_eq!(decode_alpha(&[], 1, 1), Err(AlphError::EmptyPayload));
    }
}
