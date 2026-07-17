//! Typed parser for the per-frame `ANMF` chunk header per RFC 9649
//! §2.7.1.1 (Figure 9).
//!
//! Where [`crate::anim`] decodes the global animation parameters that
//! apply once to the whole file, `ANMF` carries the **per-frame**
//! parameters that prefix each frame's payload. Figure 9 fixes the
//! first 16 bytes of an `ANMF` chunk payload as:
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                        Frame X                |             ...
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ...          Frame Y            |   Frame Width Minus One     ...
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ...             |           Frame Height Minus One              |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                 Frame Duration                |  Reserved |B|D|
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! :                         Frame Data                            :
//! ```
//!
//! Field-by-field per §2.7.1.1:
//!
//! * **Frame X** — 24-bit `uint24` (little-endian). The X coordinate
//!   of the upper-left corner is `Frame X * 2`. (§2.7.1.1: "is Frame
//!   X * 2".)
//! * **Frame Y** — 24-bit `uint24` (little-endian). The Y coordinate
//!   of the upper-left corner is `Frame Y * 2`.
//! * **Frame Width Minus One** — 24-bit `uint24` (little-endian).
//!   The actual frame width is `1 + Frame Width Minus One`.
//! * **Frame Height Minus One** — 24-bit `uint24` (little-endian).
//!   The actual frame height is `1 + Frame Height Minus One`.
//! * **Frame Duration** — 24-bit `uint24` (little-endian) in
//!   1-millisecond units.
//! * **Reserved** — 6 bits, MUST be 0, readers MUST ignore.
//! * **Blending method (B)** — 1 bit: `0` = alpha-blend with the
//!   previous canvas, `1` = overwrite the rectangle.
//! * **Disposal method (D)** — 1 bit: `0` = leave canvas as-is,
//!   `1` = dispose to ANIM's background color before the next frame.
//!
//! The total header is exactly 16 bytes. The rest of the `ANMF`
//! payload — `Chunk Size - 16` bytes per §2.7.1.1 — is the per-frame
//! "Frame Data" sub-RIFF (an optional `ALPH` plus a `VP8 ` / `VP8L`
//! bitstream, plus any unknown chunks). This module does **not**
//! decode that — it stays purely structural.
//!
//! ## Bit-layout anchor for the info byte
//!
//! Figure 9's ASCII art reads `|  Reserved |B|D|` MSB-first within
//! the byte. So in LSB-numbered bit positions:
//!
//! | bit (LSB=0) | field |
//! |-------------|-------|
//! | 7..2        | Reserved (6 bits) |
//! | 1           | B (Blending)     |
//! | 0           | D (Disposal)     |
//!
//! Cross-checked against
//! `docs/image/webp/fixtures/animated-with-alpha/trace.txt`:
//!
//! ```text
//! ANMF_FRAME frame_idx=1 ... duration=100 dispose=0 blend=1 flags_byte=0x02
//! ```
//!
//! That is, `0x02 = 0000_0010` resolves to B=1, D=0, Reserved=0 —
//! matching `blend=1 dispose=0`. The same byte appears for frames
//! 2 and 3 of that fixture. The `animated-3-frames-rgb` fixture
//! adds a `flags_byte=0x00` (B=0, D=0) for frames 2 and 3.
//!
//! ## Little-endian uint24 rationale
//!
//! RFC 9649 §2.3 establishes that all multi-byte fields in the
//! container are little-endian (the §2.3 chunk `Size` field is
//! `uint32` LE, and §2.7.1 / §2.7.1.1 / §2.7.1.2 follow). The
//! `uint24` fields in Figure 9 are stored as three consecutive bytes
//! in LSB→MSB order. The trace files cross-check this: for the
//! `animated-with-alpha` fixture, frame 1 reports
//! `x_offset=0 y_offset=0 width=64 height=64 duration=100`, which
//! matches three-byte little-endian decodes of `00 00 00`, `00 00 00`,
//! `3F 00 00`, `3F 00 00`, `64 00 00` (with width/height being
//! `width-1 = 63 = 0x3F` and duration `100 = 0x64`).

use core::fmt;

/// Blending method (`B`) per RFC 9649 §2.7.1.1.
///
/// Determines how the current frame's transparent pixels combine
/// with the previous canvas during compositing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendingMethod {
    /// 0: After disposing of the previous frame, render the current
    /// frame on the canvas using alpha-blending. If the current
    /// frame has no alpha channel, assume `alpha = 255` (effectively
    /// replacing the rectangle).
    AlphaBlend,
    /// 1: After disposing of the previous frame, render the current
    /// frame on the canvas by overwriting the rectangle covered by
    /// the current frame.
    Overwrite,
}

impl BlendingMethod {
    /// Decode the 1-bit `B` field. The single masked bit selects one
    /// of the two §2.7.1.1 cases.
    const fn from_bit(b: u8) -> Self {
        match b & 0b1 {
            0 => Self::AlphaBlend,
            1 => Self::Overwrite,
            _ => unreachable!(),
        }
    }
}

/// Disposal method (`D`) per RFC 9649 §2.7.1.1.
///
/// Determines what the canvas looks like after the current frame
/// has been displayed and before the next frame is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisposalMethod {
    /// 0: Do not dispose. Leave the canvas as-is.
    None,
    /// 1: Dispose to the background color — fill the rectangle on
    /// the canvas covered by the current frame with the background
    /// color from the `ANIM` chunk (Figure 8).
    Background,
}

impl DisposalMethod {
    /// Decode the 1-bit `D` field.
    const fn from_bit(d: u8) -> Self {
        match d & 0b1 {
            0 => Self::None,
            1 => Self::Background,
            _ => unreachable!(),
        }
    }
}

/// Errors raised by the §2.7.1.1 ANMF header parser.
#[derive(Debug, Clone, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub enum AnmfError {
    /// The chunk payload is shorter than the 16-byte fixed header
    /// laid out in Figure 9 (5 × 24-bit fields + 1 info byte). The
    /// remainder is "Frame Data" of `Chunk Size - 16` bytes, so a
    /// payload of exactly 16 bytes is legal (just with an empty
    /// `Frame Data` sub-RIFF — unusual but not forbidden by the
    /// header layout per se).
    PayloadTooShort {
        /// Actual payload length observed.
        got: usize,
    },
}

impl fmt::Display for AnmfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PayloadTooShort { got } => write!(
                f,
                "ANMF payload must be at least 16 bytes per §2.7.1.1 Figure 9, got {got}"
            ),
        }
    }
}

impl std::error::Error for AnmfError {}

/// Decoded §2.7.1.1 `ANMF` chunk header — the 16-byte per-frame
/// prefix that precedes the frame's "Frame Data" sub-RIFF.
///
/// Constructed via [`AnmfHeader::parse`]. Fields are stored in the
/// **resolved** form §2.7.1.1 specifies:
///
/// * `x` is `Frame X * 2` (the canvas-pixel coordinate, not the
///   on-disk `uint24`).
/// * `y` is `Frame Y * 2`.
/// * `width` is `1 + Frame Width Minus One`.
/// * `height` is `1 + Frame Height Minus One`.
/// * `duration_ms` is the literal `Frame Duration` (already in ms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub struct AnmfHeader {
    /// X coordinate of the upper-left corner of the frame, in
    /// canvas pixels. Equal to `Frame X * 2` per §2.7.1.1.
    pub x: u32,
    /// Y coordinate of the upper-left corner of the frame, in
    /// canvas pixels. Equal to `Frame Y * 2` per §2.7.1.1.
    pub y: u32,
    /// Frame width in pixels. Equal to `1 + Frame Width Minus One`
    /// per §2.7.1.1. Always ≥ 1.
    pub width: u32,
    /// Frame height in pixels. Equal to `1 + Frame Height Minus One`
    /// per §2.7.1.1. Always ≥ 1.
    pub height: u32,
    /// Time to wait before displaying the next frame, in
    /// 1-millisecond units, per §2.7.1.1. The interpretation of
    /// `0` (and often `<= 10`) is implementation-defined; this
    /// parser surfaces the literal value.
    pub duration_ms: u32,
    /// Decoded `B` bit.
    pub blend: BlendingMethod,
    /// Decoded `D` bit.
    pub dispose: DisposalMethod,
    /// Raw 6-bit `Reserved` field from the info byte (bits 7..2).
    /// §2.7.1.1: "MUST be 0. Readers MUST ignore this field." We
    /// surface the raw value for observability without rejecting.
    pub reserved: u8,
    /// Raw info byte (the 16th byte of the header), preserved for
    /// round-trip and trace assertions.
    pub info_byte: u8,
}

impl AnmfHeader {
    /// Fixed byte length of the §2.7.1.1 ANMF header preceding the
    /// `Frame Data` sub-RIFF. 5 × 3 bytes for the uint24 fields,
    /// plus one info byte.
    pub const HEADER_LEN: usize = 16;

    /// Parse the first 16 bytes of an `ANMF` chunk payload per
    /// RFC 9649 §2.7.1.1.
    ///
    /// `payload` is the §2.3 chunk payload — the slice returned by
    /// [`crate::container::WebpChunk::payload`] for a chunk whose
    /// FourCC is [`crate::container::fourcc::ANMF`]. Only the first
    /// 16 bytes are consumed by this layer; the rest is the
    /// per-frame "Frame Data" sub-RIFF (an optional `ALPH` plus
    /// `VP8 ` or `VP8L`), which is not decoded here.
    pub fn parse(payload: &[u8]) -> Result<Self, AnmfError> {
        if payload.len() < Self::HEADER_LEN {
            return Err(AnmfError::PayloadTooShort { got: payload.len() });
        }

        // §2.7.1.1: all five 24-bit fields are little-endian uint24.
        let frame_x = read_u24_le(&payload[0..3]);
        let frame_y = read_u24_le(&payload[3..6]);
        let frame_w_minus_one = read_u24_le(&payload[6..9]);
        let frame_h_minus_one = read_u24_le(&payload[9..12]);
        let duration_ms = read_u24_le(&payload[12..15]);

        // §2.7.1.1 Figure 9 byte 15: 6 reserved bits (MSB-aligned),
        // then B, then D.
        let info = payload[15];
        let reserved = (info >> 2) & 0b0011_1111;
        let blend = BlendingMethod::from_bit(info >> 1);
        let dispose = DisposalMethod::from_bit(info);

        Ok(Self {
            // §2.7.1.1: "The X coordinate of the upper left corner of
            // the frame is Frame X * 2." Same for Y. The doubling is
            // structural — only even canvas coordinates are
            // representable.
            x: frame_x * 2,
            y: frame_y * 2,
            // §2.7.1.1: "1-based width" / "1-based height".
            width: frame_w_minus_one + 1,
            height: frame_h_minus_one + 1,
            duration_ms,
            blend,
            dispose,
            reserved,
            info_byte: info,
        })
    }

    /// Offset (within the ANMF chunk payload) at which the per-frame
    /// "Frame Data" sub-RIFF begins. Always 16 per §2.7.1.1 — the
    /// 16-byte header is followed immediately by the chunk list.
    pub const fn frame_data_offset(&self) -> usize {
        Self::HEADER_LEN
    }
}

/// Decode a 3-byte little-endian uint24 into a u32. The high byte
/// of the returned u32 is always 0.
const fn read_u24_le(b: &[u8]) -> u32 {
    (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an ANMF header payload from its field components.
    /// `frame_x` / `frame_y` are the on-disk uint24 values (not the
    /// `*2` resolved coordinates). `width_minus_one` /
    /// `height_minus_one` likewise. `flags_byte` is the literal
    /// 16th byte (`Reserved|B|D`).
    fn anmf(
        frame_x: u32,
        frame_y: u32,
        width_minus_one: u32,
        height_minus_one: u32,
        duration_ms: u32,
        flags_byte: u8,
    ) -> Vec<u8> {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&u24_le(frame_x));
        v.extend_from_slice(&u24_le(frame_y));
        v.extend_from_slice(&u24_le(width_minus_one));
        v.extend_from_slice(&u24_le(height_minus_one));
        v.extend_from_slice(&u24_le(duration_ms));
        v.push(flags_byte);
        v
    }

    fn u24_le(v: u32) -> [u8; 3] {
        [
            (v & 0xFF) as u8,
            ((v >> 8) & 0xFF) as u8,
            ((v >> 16) & 0xFF) as u8,
        ]
    }

    /// Build the §2.7.1.1 info byte from its three fields.
    fn flags(reserved: u8, b: u8, d: u8) -> u8 {
        ((reserved & 0b0011_1111) << 2) | ((b & 0b1) << 1) | (d & 0b1)
    }

    #[test]
    fn payload_under_sixteen_bytes_is_rejected() {
        // §2.7.1.1 Figure 9 fixes the header at 16 bytes.
        assert_eq!(
            AnmfHeader::parse(&[]),
            Err(AnmfError::PayloadTooShort { got: 0 })
        );
        assert_eq!(
            AnmfHeader::parse(&[0u8; 15]),
            Err(AnmfError::PayloadTooShort { got: 15 })
        );
    }

    #[test]
    fn exactly_sixteen_bytes_is_accepted_with_empty_frame_data() {
        // §2.7.1.1: Frame Data is `Chunk Size - 16` bytes. A 16-byte
        // payload has zero frame data — legal per the header layout.
        let h = AnmfHeader::parse(&[0u8; 16]).unwrap();
        assert_eq!(h.x, 0);
        assert_eq!(h.y, 0);
        // width/height = (width_minus_one) + 1 — never zero.
        assert_eq!(h.width, 1);
        assert_eq!(h.height, 1);
        assert_eq!(h.duration_ms, 0);
        assert_eq!(h.blend, BlendingMethod::AlphaBlend);
        assert_eq!(h.dispose, DisposalMethod::None);
        assert_eq!(h.reserved, 0);
        assert_eq!(h.info_byte, 0);
        assert_eq!(h.frame_data_offset(), 16);
    }

    #[test]
    fn frame_x_and_frame_y_are_doubled_per_section_2_7_1_1() {
        // §2.7.1.1: "The X coordinate of the upper left corner of
        // the frame is Frame X * 2." Same for Y. So an on-disk
        // Frame X of 3 must surface as x = 6.
        let h = AnmfHeader::parse(&anmf(3, 5, 0, 0, 0, 0)).unwrap();
        assert_eq!(h.x, 6);
        assert_eq!(h.y, 10);
    }

    #[test]
    fn frame_width_and_height_are_one_based_per_section_2_7_1_1() {
        // §2.7.1.1: width = 1 + Frame Width Minus One.
        // On-disk 63 → width 64; on-disk 0 → width 1.
        let h = AnmfHeader::parse(&anmf(0, 0, 63, 127, 0, 0)).unwrap();
        assert_eq!(h.width, 64);
        assert_eq!(h.height, 128);
        let h_min = AnmfHeader::parse(&anmf(0, 0, 0, 0, 0, 0)).unwrap();
        assert_eq!(h_min.width, 1);
        assert_eq!(h_min.height, 1);
    }

    #[test]
    fn frame_duration_is_a_24_bit_little_endian_uint() {
        // §2.7.1.1: Frame Duration is uint24 LE in ms. Pick a value
        // whose three bytes are distinct: 0x12_3456 → 0x56 0x34 0x12.
        let mut p = vec![0u8; 16];
        p[12] = 0x56;
        p[13] = 0x34;
        p[14] = 0x12;
        let h = AnmfHeader::parse(&p).unwrap();
        assert_eq!(h.duration_ms, 0x12_3456);
    }

    #[test]
    fn uint24_fields_decode_in_little_endian_byte_order() {
        // Pin every uint24 field independently with byte patterns
        // 0xAA_BBCC / 0xDD_EEFF / etc. so a byte-order swap shows up.
        let mut p = vec![0u8; 16];
        // Frame X = 0x33_2211 (LE bytes 11 22 33)
        p[0..3].copy_from_slice(&[0x11, 0x22, 0x33]);
        // Frame Y = 0x66_5544 (LE bytes 44 55 66)
        p[3..6].copy_from_slice(&[0x44, 0x55, 0x66]);
        // Frame W-1 = 0x99_8877 (LE bytes 77 88 99) → width = +1
        p[6..9].copy_from_slice(&[0x77, 0x88, 0x99]);
        // Frame H-1 = 0xCC_BBAA (LE bytes AA BB CC) → height = +1
        p[9..12].copy_from_slice(&[0xAA, 0xBB, 0xCC]);
        // Duration = 0x000064 (LE bytes 64 00 00) = 100 ms.
        p[12..15].copy_from_slice(&[0x64, 0x00, 0x00]);
        let h = AnmfHeader::parse(&p).unwrap();
        // x = Frame X * 2.
        assert_eq!(h.x, 0x33_2211 * 2);
        assert_eq!(h.y, 0x66_5544 * 2);
        assert_eq!(h.width, 0x99_8877 + 1);
        assert_eq!(h.height, 0xCC_BBAA + 1);
        assert_eq!(h.duration_ms, 100);
    }

    #[test]
    fn info_byte_disposal_bit_is_the_lsb() {
        // §2.7.1.1 Figure 9 ASCII art reads `Reserved|B|D` MSB-first.
        // D occupies bit 0 (the LSB).
        let h0 = AnmfHeader::parse(&anmf(0, 0, 0, 0, 0, 0b0000_0000)).unwrap();
        assert_eq!(h0.dispose, DisposalMethod::None);
        let h1 = AnmfHeader::parse(&anmf(0, 0, 0, 0, 0, 0b0000_0001)).unwrap();
        assert_eq!(h1.dispose, DisposalMethod::Background);
    }

    #[test]
    fn info_byte_blending_bit_is_bit_one() {
        // B occupies bit 1.
        let h0 = AnmfHeader::parse(&anmf(0, 0, 0, 0, 0, 0b0000_0000)).unwrap();
        assert_eq!(h0.blend, BlendingMethod::AlphaBlend);
        let h1 = AnmfHeader::parse(&anmf(0, 0, 0, 0, 0, 0b0000_0010)).unwrap();
        assert_eq!(h1.blend, BlendingMethod::Overwrite);
    }

    #[test]
    fn info_byte_reserved_field_surfaces_six_high_bits_without_rejection() {
        // §2.7.1.1: "MUST be 0. Readers MUST ignore this field." So a
        // non-zero Reserved must parse, with the raw value preserved.
        // Reserved occupies bits 7..2.
        let raw = flags(0b10_1010, 0, 0);
        let h = AnmfHeader::parse(&anmf(0, 0, 0, 0, 0, raw)).unwrap();
        assert_eq!(h.reserved, 0b10_1010);
        assert_eq!(h.blend, BlendingMethod::AlphaBlend);
        assert_eq!(h.dispose, DisposalMethod::None);
        assert_eq!(h.info_byte, raw);
    }

    #[test]
    fn info_byte_all_three_subfields_decode_independently() {
        // Reserved = 0b11_0011, B = 1, D = 1.
        let raw = flags(0b11_0011, 1, 1);
        let h = AnmfHeader::parse(&anmf(0, 0, 0, 0, 0, raw)).unwrap();
        assert_eq!(h.reserved, 0b11_0011);
        assert_eq!(h.blend, BlendingMethod::Overwrite);
        assert_eq!(h.dispose, DisposalMethod::Background);
    }

    #[test]
    fn frame_data_offset_is_always_sixteen() {
        // §2.7.1.1: Frame Data is `Chunk Size - 16`.
        let h = AnmfHeader::parse(&[0u8; 16]).unwrap();
        assert_eq!(h.frame_data_offset(), 16);
        // Trailing bytes (the frame data) don't change the offset.
        let mut p = vec![0u8; 64];
        p[6] = 63; // width-1 = 63 → width 64
        p[9] = 63; // height-1 = 63 → height 64
        let h_long = AnmfHeader::parse(&p).unwrap();
        assert_eq!(h_long.frame_data_offset(), 16);
    }

    #[test]
    fn trailing_frame_data_bytes_are_not_consumed_by_header_parse() {
        // The header parser must read exactly 16 bytes; anything
        // after is the per-frame sub-RIFF (out of scope here).
        let base = anmf(0, 0, 63, 63, 100, 0x02);
        let with_tail = {
            let mut v = base.clone();
            v.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
            v
        };
        assert_eq!(
            AnmfHeader::parse(&base).unwrap(),
            AnmfHeader::parse(&with_tail).unwrap()
        );
    }

    #[test]
    fn fixture_animated_with_alpha_frame_one_decodes_to_trace_values() {
        // docs/image/webp/fixtures/animated-with-alpha/trace.txt:
        //   ANMF_FRAME frame_idx=1 x_offset=0 y_offset=0 width=64
        //              height=64 duration=100 dispose=0 blend=1
        //              flags_byte=0x02
        //
        // On-disk uint24 values:
        //   Frame X      = 0
        //   Frame Y      = 0
        //   Width-1      = 63 (→ width 64)
        //   Height-1     = 63 (→ height 64)
        //   Duration     = 100
        //   flags        = 0x02 (B=1, D=0, Reserved=0)
        let bytes = anmf(0, 0, 63, 63, 100, 0x02);
        let h = AnmfHeader::parse(&bytes).unwrap();
        assert_eq!(h.x, 0);
        assert_eq!(h.y, 0);
        assert_eq!(h.width, 64);
        assert_eq!(h.height, 64);
        assert_eq!(h.duration_ms, 100);
        assert_eq!(h.blend, BlendingMethod::Overwrite);
        assert_eq!(h.dispose, DisposalMethod::None);
        assert_eq!(h.reserved, 0);
        assert_eq!(h.info_byte, 0x02);
    }

    #[test]
    fn fixture_animated_3_frames_rgb_frame_two_decodes_to_trace_values() {
        // docs/image/webp/fixtures/animated-3-frames-rgb/trace.txt
        // frame 2:
        //   ANMF_FRAME frame_idx=2 x_offset=0 y_offset=0 width=64
        //              height=64 duration=100 dispose=0 blend=0
        //              flags_byte=0x00
        let bytes = anmf(0, 0, 63, 63, 100, 0x00);
        let h = AnmfHeader::parse(&bytes).unwrap();
        assert_eq!(h.x, 0);
        assert_eq!(h.y, 0);
        assert_eq!(h.width, 64);
        assert_eq!(h.height, 64);
        assert_eq!(h.duration_ms, 100);
        assert_eq!(h.blend, BlendingMethod::AlphaBlend);
        assert_eq!(h.dispose, DisposalMethod::None);
    }

    #[test]
    fn max_uint24_values_for_each_field() {
        // The largest representable on-disk values for each uint24.
        // x = (2^24 - 1) * 2 = 33_554_430 — still fits in u32.
        // width = (2^24 - 1) + 1 = 16_777_216.
        let max = 0xFF_FFFF_u32;
        let bytes = anmf(max, max, max, max, max, 0x00);
        let h = AnmfHeader::parse(&bytes).unwrap();
        assert_eq!(h.x, 0xFF_FFFF * 2);
        assert_eq!(h.y, 0xFF_FFFF * 2);
        assert_eq!(h.width, 0xFF_FFFF + 1);
        assert_eq!(h.height, 0xFF_FFFF + 1);
        assert_eq!(h.duration_ms, 0xFF_FFFF);
    }

    #[test]
    fn duration_zero_is_accepted_and_preserved_literally() {
        // §2.7.1.1: "interpretation of the Frame Duration of 0 ...
        // is defined by the implementation." The parser must not
        // reject 0 — that's the caller's policy.
        let bytes = anmf(0, 0, 0, 0, 0, 0x00);
        let h = AnmfHeader::parse(&bytes).unwrap();
        assert_eq!(h.duration_ms, 0);
    }

    #[test]
    fn header_len_const_is_sixteen() {
        // Anchor the public constant so a future refactor doesn't
        // silently change the contract.
        assert_eq!(AnmfHeader::HEADER_LEN, 16);
    }
}
