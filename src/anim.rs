//! Typed parser for the `ANIM` chunk payload per RFC 9649 §2.7.1.1
//! (Figure 8).
//!
//! ANIM carries the **global animation parameters** that apply to
//! every `ANMF` frame in the file:
//!
//! ```text
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                       Background Color                        |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |          Loop Count           |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! * `Background Color` — 32 bits, **uint32** stored as four bytes in
//!   `[Blue, Green, Red, Alpha]` order. §2.7.1.1 explicitly calls out
//!   the byte order; this module surfaces the four components in a
//!   separated `BackgroundColor` struct so callers don't have to
//!   second-guess endianness.
//! * `Loop Count` — 16 bits, **uint16** little-endian (consistent with
//!   every other multi-byte field in RFC 9649). `0` means "loop
//!   infinitely" per §2.7.1.1.
//!
//! The ANIM chunk's `Size` is fixed at 6 bytes by Figure 8; the
//! parser rejects any other length.
//!
//! ## Cross-check
//!
//! `docs/image/webp/fixtures/animated-with-alpha/trace.txt`:
//!
//! ```text
//! ANIM    bgcolor=0xffffffff   loop_count=0
//! ```
//!
//! That is, the on-disk bytes `ff ff ff ff` decode to B=R=G=A=255 and
//! `00 00` decodes to `loop_count=0` (infinite). Same trace appears in
//! `animated-3-frames-rgb`.

use core::fmt;

/// Background color from §2.7.1.1 — the `Background Color` field,
/// laid out on disk in `[Blue, Green, Red, Alpha]` byte order.
///
/// §2.7.1.1: "This color MAY be used to fill the unused space on the
/// canvas around the frames, as well as the transparent pixels of the
/// first frame." It may carry a non-opaque alpha even when the
/// `VP8X.L` alpha flag is unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackgroundColor {
    /// Blue channel (on-disk byte 0).
    pub blue: u8,
    /// Green channel (on-disk byte 1).
    pub green: u8,
    /// Red channel (on-disk byte 2).
    pub red: u8,
    /// Alpha channel (on-disk byte 3). 255 = opaque.
    pub alpha: u8,
}

impl BackgroundColor {
    /// Pack the four components as a `uint32` in **on-disk** byte
    /// order — i.e. the same little-endian-loaded value the trace
    /// records as `bgcolor=0xXXXXXXXX`.
    ///
    /// For `ff ff ff ff` on disk this returns `0xffff_ffff`; for the
    /// degenerate "all zero" payload it returns `0x0000_0000`. The
    /// integer's byte layout (LSB→MSB) is exactly `[B, G, R, A]`.
    pub const fn as_u32_le(&self) -> u32 {
        (self.blue as u32)
            | ((self.green as u32) << 8)
            | ((self.red as u32) << 16)
            | ((self.alpha as u32) << 24)
    }
}

/// Errors raised by the §2.7.1.1 ANIM parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimError {
    /// The ANIM payload was not exactly 6 bytes — Figure 8 fixes the
    /// layout at 4-byte background + 2-byte loop count.
    BadPayloadLength {
        /// Actual payload length observed.
        got: usize,
    },
}

impl fmt::Display for AnimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadPayloadLength { got } => write!(
                f,
                "ANIM payload must be 6 bytes per §2.7.1.1 Figure 8, got {got}"
            ),
        }
    }
}

impl std::error::Error for AnimError {}

/// Decoded §2.7.1.1 `ANIM` chunk — global animation parameters.
///
/// Constructed via [`AnimHeader::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimHeader {
    /// `Background Color` field, broken into BGRA components.
    pub background_color: BackgroundColor,
    /// `Loop Count` field. `0` means "loop infinitely" per §2.7.1.1.
    pub loop_count: u16,
}

impl AnimHeader {
    /// Parse the 6-byte `ANIM` chunk payload per RFC 9649 §2.7.1.1.
    ///
    /// `payload` is the slice returned by
    /// [`crate::container::WebpChunk::payload`] for a chunk whose
    /// FourCC is [`crate::container::fourcc::ANIM`].
    pub fn parse(payload: &[u8]) -> Result<Self, AnimError> {
        if payload.len() != 6 {
            return Err(AnimError::BadPayloadLength { got: payload.len() });
        }
        // §2.7.1.1: BGRA byte order for the background color uint32.
        let background_color = BackgroundColor {
            blue: payload[0],
            green: payload[1],
            red: payload[2],
            alpha: payload[3],
        };
        // §2.7.1.1: 16-bit Loop Count (RFC 9649 multi-byte fields are
        // little-endian throughout — see §2.3 "all data is stored
        // little-endian unless explicitly noted").
        let loop_count = u16::from_le_bytes([payload[4], payload[5]]);
        Ok(Self {
            background_color,
            loop_count,
        })
    }

    /// `true` when `Loop Count == 0`, which §2.7.1.1 defines as
    /// "infinite playback".
    pub const fn loops_forever(&self) -> bool {
        self.loop_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 6-byte ANIM payload from its component fields.
    fn anim(b: u8, g: u8, r: u8, a: u8, loop_count: u16) -> Vec<u8> {
        let mut v = Vec::with_capacity(6);
        v.push(b);
        v.push(g);
        v.push(r);
        v.push(a);
        v.extend_from_slice(&loop_count.to_le_bytes());
        v
    }

    #[test]
    fn payload_must_be_exactly_six_bytes() {
        // Figure 8 fixes the layout at 4 + 2 = 6 bytes.
        assert_eq!(
            AnimHeader::parse(&[]),
            Err(AnimError::BadPayloadLength { got: 0 })
        );
        assert_eq!(
            AnimHeader::parse(&[0u8; 5]),
            Err(AnimError::BadPayloadLength { got: 5 })
        );
        assert_eq!(
            AnimHeader::parse(&[0u8; 7]),
            Err(AnimError::BadPayloadLength { got: 7 })
        );
    }

    #[test]
    fn all_zero_payload_decodes_to_transparent_black_infinite_loop() {
        // bgcolor=0x00000000 (B=G=R=A=0, fully transparent black),
        // loop_count=0 (infinite). The simplest legal ANIM.
        let h = AnimHeader::parse(&[0u8; 6]).unwrap();
        assert_eq!(h.background_color.blue, 0);
        assert_eq!(h.background_color.green, 0);
        assert_eq!(h.background_color.red, 0);
        assert_eq!(h.background_color.alpha, 0);
        assert_eq!(h.background_color.as_u32_le(), 0);
        assert_eq!(h.loop_count, 0);
        assert!(h.loops_forever());
    }

    #[test]
    fn background_color_byte_order_is_bgra() {
        // Distinct values for each channel — pinning down which byte
        // index maps to which component.
        let h = AnimHeader::parse(&anim(0x10, 0x20, 0x30, 0x40, 5)).unwrap();
        assert_eq!(h.background_color.blue, 0x10);
        assert_eq!(h.background_color.green, 0x20);
        assert_eq!(h.background_color.red, 0x30);
        assert_eq!(h.background_color.alpha, 0x40);
        // Trace-style uint32 with the on-disk order in LSB→MSB:
        // 0x40 30 20 10.
        assert_eq!(h.background_color.as_u32_le(), 0x4030_2010);
    }

    #[test]
    fn loop_count_is_little_endian_u16() {
        // 0x0102 stored as `02 01` little-endian.
        let h = AnimHeader::parse(&[0, 0, 0, 0, 0x02, 0x01]).unwrap();
        assert_eq!(h.loop_count, 0x0102);
        assert!(!h.loops_forever());
    }

    #[test]
    fn maximum_loop_count_decodes_and_is_not_infinite() {
        // 0xFFFF is the largest finite loop count; only 0 is infinite.
        let h = AnimHeader::parse(&[0, 0, 0, 0, 0xFF, 0xFF]).unwrap();
        assert_eq!(h.loop_count, 0xFFFF);
        assert!(!h.loops_forever());
    }

    #[test]
    fn fixture_animated_with_alpha_decodes_to_white_opaque_infinite() {
        // docs/image/webp/fixtures/animated-with-alpha/trace.txt
        //   ANIM bgcolor=0xffffffff loop_count=0
        //
        // On-disk: ff ff ff ff 00 00.
        let h = AnimHeader::parse(&[0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]).unwrap();
        assert_eq!(h.background_color.blue, 0xFF);
        assert_eq!(h.background_color.green, 0xFF);
        assert_eq!(h.background_color.red, 0xFF);
        assert_eq!(h.background_color.alpha, 0xFF);
        assert_eq!(h.background_color.as_u32_le(), 0xFFFF_FFFF);
        assert_eq!(h.loop_count, 0);
        assert!(h.loops_forever());
    }
}
