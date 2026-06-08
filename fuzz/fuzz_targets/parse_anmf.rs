#![no_main]

//! Parse arbitrary fuzz-supplied bytes through the §2.7.1.1 ANMF chunk
//! header parser entry point `oxideav_webp::anmf::AnmfHeader::parse`.
//!
//! The §2 RIFF walker in `decode_webp` reaches `AnmfHeader::parse` only
//! with payload slices the container layer has already validated as
//! `ANMF` chunk bodies, so the cross-product reachable through the
//! existing `decode.rs` / `extract_metadata.rs` / `roundtrip_animated.rs`
//! harnesses exercises this parser only along the "well-formed RIFF"
//! code path. `AnmfHeader::parse` is also a public standalone surface —
//! `pub fn parse(payload: &[u8]) -> Result<Self, AnmfError>` exported
//! through `pub mod anmf` — that downstream callers can invoke against
//! any byte slice they obtained from a different demuxer (or, for that
//! matter, an attacker-controlled buffer of arbitrary length). This
//! harness widens fuzz coverage onto that direct entry point.
//!
//! The contract under test, per RFC 9649 §2.7.1.1 (Figure 9):
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow on the `Frame X * 2` doubling or the
//!   `Frame W/H Minus One + 1` resolution, no out-of-bounds index when
//!   the payload is shorter or longer than the 16-byte header.
//! * If the call returns `Ok(hdr)`, every field is internally
//!   consistent with the §2.7.1.1 Figure 9 byte layout:
//!     * `x` equals `Frame X * 2`, with `Frame X` re-derived from the
//!       little-endian 24-bit field at bytes 0..3. Same for `y` from
//!       bytes 3..6.
//!     * `width` equals `Frame W Minus One + 1`, with the on-disk
//!       value re-derived from the little-endian 24-bit field at bytes
//!       6..9. Same for `height` from bytes 9..12.
//!     * `duration_ms` equals the little-endian 24-bit field at bytes
//!       12..15 verbatim (no transformation).
//!     * `info_byte` equals payload[15] verbatim.
//!     * `reserved` equals `(info_byte >> 2) & 0b0011_1111` — the 6-bit
//!       reserved field at info-byte bits 7..2 per §2.7.1.1.
//!     * `blend` decodes the §2.7.1.1 B bit (bit 1 of info_byte):
//!       0 = AlphaBlend, 1 = Overwrite.
//!     * `dispose` decodes the §2.7.1.1 D bit (bit 0 of info_byte):
//!       0 = None, 1 = Background.
//!     * `frame_data_offset() == 16` — the §2.7.1.1 fixed-header length
//!       that separates the header from the per-frame "Frame Data"
//!       sub-RIFF.
//!     * `width >= 1` and `height >= 1` — the `+ 1` resolution makes
//!       zero unrepresentable on disk.
//!     * `x <= (2^24 - 1) * 2` and `y <= (2^24 - 1) * 2` — `Frame X`
//!       and `Frame Y` are 24-bit on-disk, so doubled they stay
//!       strictly within `u32`.
//! * If the call returns `Err(PayloadTooShort { got })`, then `got`
//!   equals the supplied slice length and that length is strictly less
//!   than 16. A payload of exactly 16 bytes is the minimum legal
//!   length (Frame Data sub-RIFF empty), and any longer payload also
//!   parses successfully — the surplus bytes are the §2.7.1.1 Frame
//!   Data sub-RIFF that this layer deliberately does not consume.
//!
//! Every branch of the contract is observable to libFuzzer: a panic
//! short-circuits, and every assertion below is a real §2.7.1.1
//! Figure 9 carrier violation if it ever fires.
//!
//! ## Iteration cost bound
//!
//! `AnmfHeader::parse` is a fixed-cost branch chain plus five 3-byte
//! little-endian uint24 reads plus three `u8` bit-extracts plus two
//! `u32` arithmetic ops (`Frame X * 2`, `Frame W-1 + 1`). There is no
//! allocation, no loop sized by the input, and no recursion. A single
//! fuzz iteration is microseconds regardless of input length, so the
//! harness can attempt arbitrary payload sizes without iteration-cost
//! concerns.
//!
//! ## Input layout
//!
//! The entire fuzz buffer is forwarded verbatim as the §2.7.1.1 ANMF
//! payload candidate. Inputs shorter than 16 bytes hit the
//! `PayloadTooShort` path; inputs ≥ 16 bytes cover the full §2.7.1.1
//! Figure 9 5 × uint24 + info-byte cross-product (with the surplus
//! bytes after byte 15 being the §2.7.1.1 Frame Data sub-RIFF the
//! header parser does not touch).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::anmf::{AnmfError, AnmfHeader, BlendingMethod, DisposalMethod};

fuzz_target!(|data: &[u8]| {
    match AnmfHeader::parse(data) {
        Ok(hdr) => {
            // §2.7.1.1 Figure 9: success implies the payload had at
            // least the 16-byte fixed header. Surplus bytes are the
            // Frame Data sub-RIFF; this parser is deliberately
            // structural and does not consume them.
            assert!(
                data.len() >= AnmfHeader::HEADER_LEN,
                "§2.7.1.1 AnmfHeader::parse returned Ok on a payload shorter than 16 bytes",
            );

            // §2.7.1.1: every 24-bit "Minus One" field resolved through
            // `+ 1` yields at least 1, so width / height can never be
            // zero on a successful parse.
            assert!(
                hdr.width >= 1,
                "§2.7.1.1 width must be >= 1 (Frame W Minus One + 1), got {}",
                hdr.width,
            );
            assert!(
                hdr.height >= 1,
                "§2.7.1.1 height must be >= 1 (Frame H Minus One + 1), got {}",
                hdr.height,
            );

            // §2.7.1.1: `Frame X` and `Frame Y` are 24-bit on-disk, so
            // the `* 2` resolved coordinates max out at (2^24 - 1) * 2
            // = 33_554_430. The bound proves the doubling did not
            // overflow `u32`.
            const MAX_DOUBLED_UINT24: u32 = ((1u32 << 24) - 1) * 2;
            assert!(
                hdr.x <= MAX_DOUBLED_UINT24,
                "§2.7.1.1 x out of [0, (2^24-1)*2]: {}",
                hdr.x,
            );
            assert!(
                hdr.y <= MAX_DOUBLED_UINT24,
                "§2.7.1.1 y out of [0, (2^24-1)*2]: {}",
                hdr.y,
            );

            // §2.7.1.1: the resolved width / height max out at
            // (2^24 - 1) + 1 = 2^24 = 16_777_216.
            const MAX_RESOLVED_DIM: u32 = 1u32 << 24;
            assert!(
                hdr.width <= MAX_RESOLVED_DIM,
                "§2.7.1.1 width out of [1, 2^24]: {}",
                hdr.width,
            );
            assert!(
                hdr.height <= MAX_RESOLVED_DIM,
                "§2.7.1.1 height out of [1, 2^24]: {}",
                hdr.height,
            );

            // §2.7.1.1: duration_ms is a uint24 LE, so it lands in
            // [0, 2^24 - 1] = [0, 16_777_215].
            assert!(
                hdr.duration_ms <= 0x00FF_FFFF,
                "§2.7.1.1 duration_ms exceeds the uint24 range: {}",
                hdr.duration_ms,
            );

            // §2.7.1.1: Frame Data sub-RIFF begins at byte 16 of the
            // chunk payload — every parse outcome reports the same
            // fixed offset.
            assert_eq!(
                hdr.frame_data_offset(),
                AnmfHeader::HEADER_LEN,
                "§2.7.1.1 frame_data_offset must be 16",
            );

            // Re-derive every header field directly from the input
            // bytes the same way the parser does and assert
            // field-for-field equality. Mismatch on any of these is a
            // bit-layout regression in the §2.7.1.1 Figure 9 decode.
            let frame_x = read_u24_le(&data[0..3]);
            let frame_y = read_u24_le(&data[3..6]);
            let frame_w_m1 = read_u24_le(&data[6..9]);
            let frame_h_m1 = read_u24_le(&data[9..12]);
            let duration = read_u24_le(&data[12..15]);
            let info = data[15];

            // §2.7.1.1: "The X coordinate of the upper left corner of
            // the frame is Frame X * 2." Same for Y. The doubling is
            // structural — only even canvas coordinates representable.
            assert_eq!(
                hdr.x,
                frame_x * 2,
                "§2.7.1.1 x does not match `Frame X * 2`",
            );
            assert_eq!(
                hdr.y,
                frame_y * 2,
                "§2.7.1.1 y does not match `Frame Y * 2`",
            );

            // §2.7.1.1: 1-based width / height.
            assert_eq!(
                hdr.width,
                frame_w_m1 + 1,
                "§2.7.1.1 width does not match `Frame W Minus One + 1`",
            );
            assert_eq!(
                hdr.height,
                frame_h_m1 + 1,
                "§2.7.1.1 height does not match `Frame H Minus One + 1`",
            );

            // §2.7.1.1: Frame Duration is a literal uint24 LE.
            assert_eq!(
                hdr.duration_ms, duration,
                "§2.7.1.1 duration_ms does not match the bytes 12..15 uint24 LE",
            );

            // §2.7.1.1: info_byte is preserved verbatim for round-trip
            // and trace assertions.
            assert_eq!(
                hdr.info_byte, info,
                "§2.7.1.1 info_byte does not match payload[15]",
            );

            // §2.7.1.1 Figure 9: Reserved occupies bits 7..2 of the
            // info byte (MSB-first within the byte, so LSB positions
            // 7..2 in the conventional bit numbering).
            let expected_reserved = (info >> 2) & 0b0011_1111;
            assert_eq!(
                hdr.reserved, expected_reserved,
                "§2.7.1.1 reserved field mismatch (info_byte={info:#04x})",
            );

            // §2.7.1.1 Figure 9: B occupies bit 1 of the info byte.
            let expected_blend = match (info >> 1) & 0b1 {
                0 => BlendingMethod::AlphaBlend,
                1 => BlendingMethod::Overwrite,
                _ => unreachable!(),
            };
            assert_eq!(
                hdr.blend, expected_blend,
                "§2.7.1.1 blend (B / bit 1) mismatch (info_byte={info:#04x})",
            );

            // §2.7.1.1 Figure 9: D occupies bit 0 of the info byte
            // (the LSB).
            let expected_dispose = match info & 0b1 {
                0 => DisposalMethod::None,
                1 => DisposalMethod::Background,
                _ => unreachable!(),
            };
            assert_eq!(
                hdr.dispose, expected_dispose,
                "§2.7.1.1 dispose (D / bit 0) mismatch (info_byte={info:#04x})",
            );
        }
        Err(AnmfError::PayloadTooShort { got }) => {
            // §2.7.1.1 Figure 9: PayloadTooShort is raised iff the
            // input is strictly shorter than 16 bytes, and `got`
            // reports the actual length the parser observed.
            assert_eq!(
                got,
                data.len(),
                "§2.7.1.1 PayloadTooShort.got must equal the input slice length",
            );
            assert!(
                got < AnmfHeader::HEADER_LEN,
                "§2.7.1.1 PayloadTooShort must not be raised on a >= 16-byte payload",
            );
        }
    }
});

/// Decode a 3-byte little-endian uint24 into a u32. Mirrors the
/// `read_u24_le` helper in `src/anmf.rs`; redefined here so the fuzz
/// harness does not need a `pub(crate)`-only function.
fn read_u24_le(b: &[u8]) -> u32 {
    u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16)
}
