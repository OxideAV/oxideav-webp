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
//! This module decodes the info byte into a typed [`AlphHeader`]; the
//! Alpha Bitstream that follows (raw bytes vs. embedded VP8L
//! image-stream) is **not** decoded here. RFC 9649 §2.7.1.2 calls the
//! remainder "_Chunk Size_ bytes - 1" — the caller can borrow it via
//! [`AlphHeader::bitstream_offset`] after a successful parse.
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

/// Errors raised by the §2.7.1.2 ALPH info-byte parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlphError {
    /// The ALPH payload is empty — at minimum one info byte is
    /// required per §2.7.1.2 Figure 10, even if the alpha bitstream
    /// itself is zero-length (which §2.7.1.2 does not forbid).
    EmptyPayload,
}

impl fmt::Display for AlphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPayload => {
                f.write_str("ALPH payload missing the §2.7.1.2 info byte (payload length 0)")
            }
        }
    }
}

impl std::error::Error for AlphError {}

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
}
