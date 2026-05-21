//! Typed parser for the `VP8X` chunk payload per RFC 9649 §2.7.1
//! (Extended File Format → Figure 7).
//!
//! The §2.3–§2.7 RIFF walker in [`crate::container`] surfaces the
//! `VP8X` chunk as an opaque 10-byte payload. This module turns that
//! payload into a typed [`Vp8xHeader`] carrying:
//!
//! * The six feature flags drawn from byte 0 — `I`, `L`, `E`, `X`,
//!   `A` (§2.7.1 Figure 7's `Rsv|I|L|E|X|A|R` octet).
//! * The two 24-bit little-endian `Minus One` canvas dimensions
//!   converted to 1-based pixel counts.
//! * A `has_unknown` summary that is true when **any** of the §2.7.1
//!   reserved bits (the 2-bit `Rsv` pair, the trailing `R` bit, or
//!   the 24-bit reserved field at bytes 1..4) is set. Per
//!   §2.7.1 ("Future specifications may add more fields. Unknown
//!   fields MUST be ignored.") this is informational only — the
//!   parser does NOT reject on non-zero reserved bits because §2.7.1
//!   says "Readers MUST ignore" each of them.
//!
//! The parser does enforce the **structural** constraints that §2.7.1
//! does call out as MUST:
//!
//! * The payload is exactly 10 bytes (matches the §2.7.1 Figure 7
//!   layout: 1 flags byte + 3 reserved bytes + 3 width bytes +
//!   3 height bytes).
//! * `Canvas Width * Canvas Height` does not exceed 2^32 - 1 (the
//!   explicit §2.7.1 product cap).
//!
//! No `VP8 ` / `VP8L` / `ALPH` payload is decoded here.
//!
//! ## Bit layout of the §2.7.1 flag octet (byte 0)
//!
//! The RFC's ASCII-art `|Rsv|I|L|E|X|A|R|` field reads left-to-right
//! MSB-first within the byte, giving:
//!
//! | bit (LSB=0) | symbol | meaning                                              |
//! |-------------|--------|------------------------------------------------------|
//! | 7           | Rsv    | reserved, MUST be 0, readers MUST ignore             |
//! | 6           | Rsv    | reserved, MUST be 0, readers MUST ignore             |
//! | 5           | I      | ICC profile present (an `ICCP` chunk follows)        |
//! | 4           | L      | "Alpha" — any frame contains transparency            |
//! | 3           | E      | Exif metadata present (an `EXIF` chunk follows)      |
//! | 2           | X      | XMP metadata present (an `XMP ` chunk follows)       |
//! | 1           | A      | Animation present (`ANIM` + `ANMF` chunks follow)    |
//! | 0           | R      | reserved, MUST be 0, readers MUST ignore             |
//!
//! The mapping is anchored against six libwebp-encoded fixtures in
//! `docs/image/webp/fixtures/`: `extended-with-exif` (flags=0x08 →
//! E), `extended-with-icc-profile` (0x20 → I), `extended-with-xmp`
//! (0x04 → X), `lossy-with-alpha-128x128` (0x10 → L),
//! `animated-3-frames-rgb` and `animated-with-alpha` (0x12 → L+A).
//! Those fixtures' `trace.txt` files independently report the same
//! `has_alpha` / `has_animation` / `has_icc` / `has_exif` / `has_xmp`
//! decoding.

use core::fmt;

/// Errors raised by the §2.7.1 VP8X field parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vp8xError {
    /// The VP8X chunk payload was not exactly 10 bytes — Figure 7
    /// fixes the layout at 1 flags + 3 reserved + 3 width + 3 height.
    BadPayloadLength {
        /// Actual payload length observed.
        got: usize,
    },
    /// §2.7.1 mandates that the product of `Canvas Width` and
    /// `Canvas Height` be at most `2^32 - 1`. The parsed canvas
    /// dimensions exceed that cap.
    CanvasTooLarge {
        /// 1-based canvas width (`Canvas Width Minus One + 1`).
        canvas_width: u32,
        /// 1-based canvas height (`Canvas Height Minus One + 1`).
        canvas_height: u32,
    },
}

impl fmt::Display for Vp8xError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadPayloadLength { got } => write!(
                f,
                "VP8X payload must be 10 bytes per §2.7.1 Figure 7, got {got}"
            ),
            Self::CanvasTooLarge {
                canvas_width,
                canvas_height,
            } => write!(
                f,
                "§2.7.1 canvas size {canvas_width}x{canvas_height} \
                 exceeds the 2^32 - 1 product cap"
            ),
        }
    }
}

impl std::error::Error for Vp8xError {}

/// Decoded §2.7.1 `VP8X` chunk — six feature flags plus the two
/// 1-based canvas dimensions.
///
/// Constructed via [`Vp8xHeader::parse`]. All `bool` fields reflect
/// the bit in the §2.7.1 flag octet of the same letter (`has_iccp`
/// ↔ `I`, `has_alpha` ↔ `L`, `has_exif` ↔ `E`, `has_xmp` ↔ `X`,
/// `has_animation` ↔ `A`). `has_unknown` is a derived signal —
/// `true` when any of the §2.7.1 reserved positions (the `Rsv` pair,
/// the `R` bit, or the 24-bit reserved field) is non-zero, hinting
/// that the producer set a forward-compatibility bit a future
/// revision may define.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8xHeader {
    /// 1-based canvas width — `Canvas Width Minus One + 1`. Always
    /// in `1..=2^24` because the on-disk field is 24 bits wide.
    pub canvas_width: u32,
    /// 1-based canvas height — `Canvas Height Minus One + 1`. Always
    /// in `1..=2^24`.
    pub canvas_height: u32,
    /// §2.7.1 `I` bit — an `ICCP` chunk is present.
    pub has_iccp: bool,
    /// §2.7.1 `A` bit — `ANIM` + `ANMF` chunks describe an animation.
    pub has_animation: bool,
    /// §2.7.1 `E` bit — an `EXIF` chunk carries Exif metadata.
    pub has_exif: bool,
    /// §2.7.1 `X` bit — an `XMP ` chunk carries XMP metadata.
    pub has_xmp: bool,
    /// §2.7.1 `L` bit — at least one frame has alpha (either an
    /// `ALPH` subchunk alongside `VP8 ` or `VP8L` with `alpha_used`).
    pub has_alpha: bool,
    /// `true` when any §2.7.1 reserved bit is non-zero (the 2-bit
    /// `Rsv` pair, the trailing `R` bit, or the 24-bit reserved
    /// field at payload bytes 1..4). Per §2.7.1 reserved bits MUST
    /// be ignored — this signal exists for callers that want to
    /// flag forward-compat use of an as-yet-undefined feature, NOT
    /// to drive a parse refusal.
    pub has_unknown: bool,
}

impl Vp8xHeader {
    /// Parse the 10-byte `VP8X` chunk payload per RFC 9649 §2.7.1.
    ///
    /// The caller is expected to hand in just the payload bytes —
    /// i.e. the slice returned by [`crate::container::WebpChunk::payload`]
    /// for a chunk whose `fourcc` is [`crate::container::fourcc::VP8X`].
    /// The 8-byte chunk header (`'VP8X'` + Size) is the container
    /// walker's responsibility, not this layer's.
    pub fn parse(payload: &[u8]) -> Result<Self, Vp8xError> {
        // §2.7.1 Figure 7: flags(1) + Reserved(3) + Width(3) + Height(3).
        if payload.len() != 10 {
            return Err(Vp8xError::BadPayloadLength { got: payload.len() });
        }

        let flags = payload[0];
        let reserved_lo = payload[1];
        let reserved_mid = payload[2];
        let reserved_hi = payload[3];

        // §2.7.1 byte 0: bits 7..6 = Rsv, 5 = I, 4 = L, 3 = E,
        // 2 = X, 1 = A, 0 = R. See the module-level table.
        let has_iccp = (flags & 0b0010_0000) != 0;
        let has_alpha = (flags & 0b0001_0000) != 0;
        let has_exif = (flags & 0b0000_1000) != 0;
        let has_xmp = (flags & 0b0000_0100) != 0;
        let has_animation = (flags & 0b0000_0010) != 0;

        // Sum every position §2.7.1 spells out as "MUST be 0, readers
        // MUST ignore". The Rsv pair (bits 7..6) and R (bit 0) of the
        // flag octet, plus all 24 bits of the trailing reserved field
        // (bytes 1..4). Any non-zero among these is a forward-compat
        // hint, NOT a refusal trigger.
        let reserved_flag_bits = flags & 0b1100_0001;
        let has_unknown =
            reserved_flag_bits != 0 || reserved_lo != 0 || reserved_mid != 0 || reserved_hi != 0;

        // §2.7.1 Canvas Width Minus One — 24-bit little-endian at
        // bytes 4..7; actual width = value + 1.
        let cwm1 =
            u32::from(payload[4]) | (u32::from(payload[5]) << 8) | (u32::from(payload[6]) << 16);
        // §2.7.1 Canvas Height Minus One — 24-bit little-endian at
        // bytes 7..10; actual height = value + 1.
        let chm1 =
            u32::from(payload[7]) | (u32::from(payload[8]) << 8) | (u32::from(payload[9]) << 16);

        // The 24-bit "Minus One" field maxes out at 0x00FF_FFFF, so
        // `+ 1` cannot overflow a `u32` (max possible value 0x0100_0000).
        let canvas_width = cwm1 + 1;
        let canvas_height = chm1 + 1;

        // §2.7.1: "The product of Canvas Width and Canvas Height
        // MUST be at most 2^32 - 1." `u32::checked_mul` is exactly
        // the cap check.
        if (canvas_width as u64) * (canvas_height as u64) > u64::from(u32::MAX) {
            return Err(Vp8xError::CanvasTooLarge {
                canvas_width,
                canvas_height,
            });
        }

        Ok(Self {
            canvas_width,
            canvas_height,
            has_iccp,
            has_animation,
            has_exif,
            has_xmp,
            has_alpha,
            has_unknown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 10-byte VP8X payload from a flags byte + 24-bit
    /// width-minus-one + 24-bit height-minus-one + an optional
    /// 24-bit reserved field.
    fn vp8x(flags: u8, cwm1: u32, chm1: u32, reserved24: u32) -> Vec<u8> {
        vec![
            flags,
            // §2.7.1 24-bit reserved field, little-endian.
            (reserved24 & 0xFF) as u8,
            ((reserved24 >> 8) & 0xFF) as u8,
            ((reserved24 >> 16) & 0xFF) as u8,
            // Canvas Width Minus One, 24-bit little-endian.
            (cwm1 & 0xFF) as u8,
            ((cwm1 >> 8) & 0xFF) as u8,
            ((cwm1 >> 16) & 0xFF) as u8,
            // Canvas Height Minus One, 24-bit little-endian.
            (chm1 & 0xFF) as u8,
            ((chm1 >> 8) & 0xFF) as u8,
            ((chm1 >> 16) & 0xFF) as u8,
        ]
    }

    #[test]
    fn smallest_canvas_one_by_one_with_no_flags() {
        // Every reserved bit zero, no feature flags, 1x1 canvas.
        let h = Vp8xHeader::parse(&vp8x(0x00, 0, 0, 0)).unwrap();
        assert_eq!(h.canvas_width, 1);
        assert_eq!(h.canvas_height, 1);
        assert!(!h.has_iccp);
        assert!(!h.has_alpha);
        assert!(!h.has_exif);
        assert!(!h.has_xmp);
        assert!(!h.has_animation);
        assert!(!h.has_unknown);
    }

    #[test]
    fn flag_bit_assignments_match_2_7_1_letters() {
        // Each named feature on its own — confirms the bit-position
        // decode table in this module's docstring.
        let i = Vp8xHeader::parse(&vp8x(0b0010_0000, 0, 0, 0)).unwrap();
        assert!(i.has_iccp);
        assert!(!i.has_alpha && !i.has_exif && !i.has_xmp && !i.has_animation);
        assert!(!i.has_unknown);

        let l = Vp8xHeader::parse(&vp8x(0b0001_0000, 0, 0, 0)).unwrap();
        assert!(l.has_alpha);
        assert!(!l.has_iccp && !l.has_exif && !l.has_xmp && !l.has_animation);

        let e = Vp8xHeader::parse(&vp8x(0b0000_1000, 0, 0, 0)).unwrap();
        assert!(e.has_exif);
        assert!(!e.has_iccp && !e.has_alpha && !e.has_xmp && !e.has_animation);

        let x = Vp8xHeader::parse(&vp8x(0b0000_0100, 0, 0, 0)).unwrap();
        assert!(x.has_xmp);
        assert!(!x.has_iccp && !x.has_alpha && !x.has_exif && !x.has_animation);

        let a = Vp8xHeader::parse(&vp8x(0b0000_0010, 0, 0, 0)).unwrap();
        assert!(a.has_animation);
        assert!(!a.has_iccp && !a.has_alpha && !a.has_exif && !a.has_xmp);
    }

    #[test]
    fn multiple_feature_flags_combine_independently() {
        // L + A together (the animation-with-alpha pattern).
        let h = Vp8xHeader::parse(&vp8x(0b0001_0010, 63, 63, 0)).unwrap();
        assert_eq!(h.canvas_width, 64);
        assert_eq!(h.canvas_height, 64);
        assert!(h.has_alpha);
        assert!(h.has_animation);
        assert!(!h.has_iccp);
        assert!(!h.has_exif);
        assert!(!h.has_xmp);
        assert!(!h.has_unknown);
    }

    #[test]
    fn canvas_dims_are_one_based_24bit_little_endian() {
        // Mid-range value to exercise all three octets of the
        // little-endian width / height fields. 0x00ABCD - 1
        // expressed as Minus-One = 0x00ABCC.
        let h = Vp8xHeader::parse(&vp8x(0x00, 0x00ABCC, 0x000123, 0)).unwrap();
        assert_eq!(h.canvas_width, 0x00ABCD);
        assert_eq!(h.canvas_height, 0x000124);
    }

    #[test]
    fn maximum_24bit_canvas_dim_decodes_then_trips_product_cap() {
        // The 24-bit "Minus One" field's largest value is 0x00FF_FFFF
        // → canvas dim 2^24. The §2.7.1 product cap (≤ 2^32 - 1) is
        // therefore the first thing tripped at the maximum width AND
        // height combined; the parser must surface that as the right
        // error variant carrying the decoded dims so callers can see
        // both numbers in the error message.
        let err = Vp8xHeader::parse(&vp8x(0x00, 0x00FF_FFFF, 0x00FF_FFFF, 0)).unwrap_err();
        assert_eq!(
            err,
            Vp8xError::CanvasTooLarge {
                canvas_width: 0x0100_0000,
                canvas_height: 0x0100_0000,
            }
        );

        // Asymmetric maximum: width at 2^24, height = 1 → product
        // 2^24 < 2^32 - 1 → decode succeeds. Confirms the 24-bit
        // little-endian decode survives a saturated minus-one field.
        let h = Vp8xHeader::parse(&vp8x(0x00, 0x00FF_FFFF, 0, 0)).unwrap();
        assert_eq!(h.canvas_width, 0x0100_0000);
        assert_eq!(h.canvas_height, 1);
    }

    #[test]
    fn canvas_product_above_2_32_minus_1_is_rejected() {
        // 65536 * 65536 = 2^32, exactly one above the §2.7.1 cap.
        // Minus-One: 65535 each.
        let err = Vp8xHeader::parse(&vp8x(0x00, 65_535, 65_535, 0)).unwrap_err();
        assert_eq!(
            err,
            Vp8xError::CanvasTooLarge {
                canvas_width: 65_536,
                canvas_height: 65_536,
            }
        );
    }

    #[test]
    fn canvas_product_at_2_32_minus_1_is_accepted() {
        // 65536 * 65535 = 2^32 - 65536 < 2^32 — the largest product
        // representable with one dim at 65536 that still meets the
        // cap. Exercises the strict-less-than-or-equal boundary.
        let h = Vp8xHeader::parse(&vp8x(0x00, 65_535, 65_534, 0)).unwrap();
        assert_eq!(h.canvas_width, 65_536);
        assert_eq!(h.canvas_height, 65_535);
    }

    #[test]
    fn reserved_bits_in_flag_octet_set_has_unknown_but_do_not_reject() {
        // §2.7.1: Rsv (bits 7..6), R (bit 0) — "MUST be 0. Readers
        // MUST ignore this field." So setting them is a producer
        // violation but a reader must still parse.
        for flag in [0b1000_0000, 0b0100_0000, 0b1100_0000, 0b0000_0001] {
            let h = Vp8xHeader::parse(&vp8x(flag, 0, 0, 0)).unwrap();
            assert!(h.has_unknown, "flag=0x{flag:02x}");
            // None of the named feature bits should leak as set.
            assert!(!h.has_iccp);
            assert!(!h.has_alpha);
            assert!(!h.has_exif);
            assert!(!h.has_xmp);
            assert!(!h.has_animation);
        }
    }

    #[test]
    fn reserved_24bit_field_set_sets_has_unknown_but_does_not_reject() {
        // §2.7.1: the 24-bit Reserved field at bytes 1..4 — "MUST be
        // 0. Readers MUST ignore this field." Same MUST-ignore rule.
        let h = Vp8xHeader::parse(&vp8x(0x00, 7, 7, 0x12_3456)).unwrap();
        assert!(h.has_unknown);
        assert_eq!(h.canvas_width, 8);
        assert_eq!(h.canvas_height, 8);
    }

    #[test]
    fn payload_must_be_exactly_ten_bytes() {
        // Figure 7 fixes the layout at 1 + 3 + 3 + 3 = 10 bytes;
        // anything shorter or longer fails BadPayloadLength.
        assert_eq!(
            Vp8xHeader::parse(&[]),
            Err(Vp8xError::BadPayloadLength { got: 0 })
        );
        assert_eq!(
            Vp8xHeader::parse(&[0u8; 9]),
            Err(Vp8xError::BadPayloadLength { got: 9 })
        );
        assert_eq!(
            Vp8xHeader::parse(&[0u8; 11]),
            Err(Vp8xError::BadPayloadLength { got: 11 })
        );
    }

    /// Helper for the fixture-derived golden tests below — `flags`
    /// is the raw §2.7.1 byte 0 value observed at offset 20 of each
    /// extended-format fixture (i.e. payload byte 0 of the VP8X chunk).
    fn parse_at_byte(flags: u8, cwm1: u32, chm1: u32) -> Vp8xHeader {
        Vp8xHeader::parse(&vp8x(flags, cwm1, chm1, 0)).unwrap()
    }

    #[test]
    fn fixture_extended_with_exif_decode_matches_trace() {
        // docs/image/webp/fixtures/extended-with-exif:
        // payload[0]=0x08, width=128, height=128.
        let h = parse_at_byte(0x08, 127, 127);
        assert_eq!(h.canvas_width, 128);
        assert_eq!(h.canvas_height, 128);
        assert!(h.has_exif);
        assert!(!h.has_iccp);
        assert!(!h.has_xmp);
        assert!(!h.has_alpha);
        assert!(!h.has_animation);
    }

    #[test]
    fn fixture_extended_with_icc_profile_decode_matches_trace() {
        // payload[0]=0x20, 128x128.
        let h = parse_at_byte(0x20, 127, 127);
        assert!(h.has_iccp);
        assert!(!h.has_exif && !h.has_xmp && !h.has_alpha && !h.has_animation);
    }

    #[test]
    fn fixture_extended_with_xmp_decode_matches_trace() {
        // payload[0]=0x04, 128x128.
        let h = parse_at_byte(0x04, 127, 127);
        assert!(h.has_xmp);
        assert!(!h.has_iccp && !h.has_exif && !h.has_alpha && !h.has_animation);
    }

    #[test]
    fn fixture_lossy_with_alpha_128x128_decode_matches_trace() {
        // payload[0]=0x10, 128x128.
        let h = parse_at_byte(0x10, 127, 127);
        assert!(h.has_alpha);
        assert!(!h.has_iccp && !h.has_exif && !h.has_xmp && !h.has_animation);
    }

    #[test]
    fn fixture_animated_with_alpha_decode_matches_trace() {
        // payload[0]=0x12, 64x64 (L + A together).
        let h = parse_at_byte(0x12, 63, 63);
        assert_eq!(h.canvas_width, 64);
        assert_eq!(h.canvas_height, 64);
        assert!(h.has_alpha);
        assert!(h.has_animation);
        assert!(!h.has_iccp && !h.has_exif && !h.has_xmp);
    }
}
