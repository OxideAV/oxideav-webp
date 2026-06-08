#![no_main]

//! Parse arbitrary fuzz-supplied bytes through the §2.7.1 VP8X chunk
//! parser entry point `oxideav_webp::vp8x::Vp8xHeader::parse`.
//!
//! The §2 RIFF walker in `decode_webp` reaches `Vp8xHeader::parse` only
//! with a payload slice the container layer has already validated as a
//! `VP8X` chunk body — meaning the existing `decode.rs` /
//! `extract_metadata.rs` harnesses exercise this parser only along the
//! "well-formed RIFF" code path. `Vp8xHeader::parse` is also a public
//! standalone surface — `pub fn parse(payload: &[u8]) -> Result<Self,
//! Vp8xError>` exported through `pub mod vp8x` and the convenience
//! wrapper [`oxideav_webp::parse_vp8x_header`] — that downstream
//! callers can invoke against any byte slice they obtained from a
//! different demuxer (or, for that matter, an attacker-controlled
//! buffer of arbitrary length). This harness widens fuzz coverage onto
//! that direct entry point.
//!
//! The contract under test, per RFC 9649 §2.7.1 (Figure 7):
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow on the `(canvas_width as u64) * (canvas_height
//!   as u64)` product cap check, no out-of-bounds index when the
//!   payload is shorter or longer than the §2.7.1 Figure 7 ten bytes.
//! * If the call returns `Ok(hdr)`, every field is internally
//!   consistent with the §2.7.1 Figure 7 byte layout:
//!     * `canvas_width ∈ [1, 2^24]` and `canvas_height ∈ [1, 2^24]`
//!       because the on-disk fields are 24-bit "Minus One" values
//!       (`field + 1` ranges over `[1, 2^24]`).
//!     * `canvas_width as u64 * canvas_height as u64 <= u32::MAX as
//!       u64` — the explicit §2.7.1 product cap; success implies the
//!       cap was respected.
//!     * The five named feature-flag bools (`has_iccp`, `has_alpha`,
//!       `has_exif`, `has_xmp`, `has_animation`) match the §2.7.1
//!       byte-0 bit positions (I=5, L=4, E=3, X=2, A=1) exactly.
//!     * `has_unknown` is true iff any §2.7.1 reserved position is
//!       non-zero — the 2-bit `Rsv` pair (byte 0 bits 7..6), the `R`
//!       bit (byte 0 bit 0), or any bit of the 24-bit reserved field
//!       at bytes 1..4. Per §2.7.1 reserved bits MUST be ignored, so
//!       `has_unknown` does not drive a parse refusal — but the
//!       summary signal must still match the actual reserved-bit
//!       contents.
//! * If the call returns `Err(BadPayloadLength { got })`, then `got`
//!   equals the supplied slice length and that length is not 10.
//! * If the call returns `Err(CanvasTooLarge { canvas_width,
//!   canvas_height })`, then the product `canvas_width as u64 *
//!   canvas_height as u64` strictly exceeds `u32::MAX as u64` —
//!   the only condition under which §2.7.1 mandates refusal.
//!
//! Every branch of the contract is observable to libFuzzer: a panic
//! short-circuits, and every assertion below is a real §2.7.1
//! carrier violation if it ever fires.
//!
//! ## Iteration cost bound
//!
//! `Vp8xHeader::parse` is a fixed-cost branch chain plus a single
//! `u64` multiply — there is no allocation, no loop sized by the
//! input, and no recursion. A single fuzz iteration is microseconds
//! regardless of input length, so the harness can attempt arbitrary
//! payload sizes without iteration-cost concerns.
//!
//! ## Input layout
//!
//! The entire fuzz buffer is forwarded verbatim as the §2.7.1 VP8X
//! payload candidate. Inputs shorter or longer than 10 bytes hit the
//! `BadPayloadLength` path; 10-byte inputs cover the full Figure 7
//! flag-octet / reserved-field / canvas-dimension cross-product.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8x::{Vp8xError, Vp8xHeader};

fuzz_target!(|data: &[u8]| {
    match Vp8xHeader::parse(data) {
        Ok(hdr) => {
            // §2.7.1 Figure 7: success implies the payload was exactly
            // 10 bytes — every other length lands on BadPayloadLength.
            assert_eq!(
                data.len(),
                10,
                "§2.7.1 Vp8xHeader::parse returned Ok on a non-10-byte payload",
            );

            // §2.7.1 24-bit "Minus One" canvas dimensions land in
            // [1, 2^24] after `+ 1`.
            assert!(
                (1..=(1u32 << 24)).contains(&hdr.canvas_width),
                "§2.7.1 canvas_width out of [1, 2^24]: {}",
                hdr.canvas_width,
            );
            assert!(
                (1..=(1u32 << 24)).contains(&hdr.canvas_height),
                "§2.7.1 canvas_height out of [1, 2^24]: {}",
                hdr.canvas_height,
            );

            // §2.7.1 explicit product cap: success implies the
            // (width * height) ≤ 2^32 - 1 check passed.
            let product = u64::from(hdr.canvas_width) * u64::from(hdr.canvas_height);
            assert!(
                product <= u64::from(u32::MAX),
                "§2.7.1 canvas product {product} exceeds the 2^32 - 1 cap",
            );

            // Re-derive every header field directly from the input
            // bytes the same way the parser does and assert
            // field-for-field equality. Mismatch on any of these is a
            // bit-layout regression in the §2.7.1 Figure 7 decode.
            let flags = data[0];
            let reserved_lo = data[1];
            let reserved_mid = data[2];
            let reserved_hi = data[3];

            // §2.7.1 byte-0 flag-bit positions, from the module-level
            // table: I=5, L=4, E=3, X=2, A=1, R=0, Rsv=7..6.
            assert_eq!(
                hdr.has_iccp,
                (flags & 0b0010_0000) != 0,
                "§2.7.1 has_iccp (bit 5 / `I`) mismatch",
            );
            assert_eq!(
                hdr.has_alpha,
                (flags & 0b0001_0000) != 0,
                "§2.7.1 has_alpha (bit 4 / `L`) mismatch",
            );
            assert_eq!(
                hdr.has_exif,
                (flags & 0b0000_1000) != 0,
                "§2.7.1 has_exif (bit 3 / `E`) mismatch",
            );
            assert_eq!(
                hdr.has_xmp,
                (flags & 0b0000_0100) != 0,
                "§2.7.1 has_xmp (bit 2 / `X`) mismatch",
            );
            assert_eq!(
                hdr.has_animation,
                (flags & 0b0000_0010) != 0,
                "§2.7.1 has_animation (bit 1 / `A`) mismatch",
            );

            // `has_unknown` is true iff any §2.7.1 reserved bit is
            // non-zero — Rsv (bits 7..6) + R (bit 0) of the flag
            // octet plus all 24 bits of the trailing reserved field.
            let reserved_flag_bits = flags & 0b1100_0001;
            let expected_unknown = reserved_flag_bits != 0
                || reserved_lo != 0
                || reserved_mid != 0
                || reserved_hi != 0;
            assert_eq!(
                hdr.has_unknown, expected_unknown,
                "§2.7.1 has_unknown mismatch (reserved flag bits {reserved_flag_bits:#04x}, \
                 reserved field [{reserved_lo:#04x}, {reserved_mid:#04x}, {reserved_hi:#04x}])",
            );

            // §2.7.1 Canvas Width / Height Minus One — 24-bit
            // little-endian at bytes 4..7 and 7..10 respectively;
            // actual dimension = value + 1.
            let cwm1 = u32::from(data[4]) | (u32::from(data[5]) << 8) | (u32::from(data[6]) << 16);
            let chm1 = u32::from(data[7]) | (u32::from(data[8]) << 8) | (u32::from(data[9]) << 16);
            assert_eq!(
                hdr.canvas_width,
                cwm1 + 1,
                "§2.7.1 canvas_width does not match `Canvas Width Minus One + 1`",
            );
            assert_eq!(
                hdr.canvas_height,
                chm1 + 1,
                "§2.7.1 canvas_height does not match `Canvas Height Minus One + 1`",
            );
        }
        Err(Vp8xError::BadPayloadLength { got }) => {
            // §2.7.1 Figure 7: BadPayloadLength is raised iff the
            // input is not exactly 10 bytes, and `got` reports the
            // actual length the parser observed.
            assert_eq!(
                got,
                data.len(),
                "§2.7.1 BadPayloadLength.got must equal the input slice length",
            );
            assert_ne!(
                got, 10,
                "§2.7.1 BadPayloadLength must not be raised on a 10-byte payload",
            );
        }
        Err(Vp8xError::CanvasTooLarge {
            canvas_width,
            canvas_height,
        }) => {
            // §2.7.1 product-cap path: refusal implies the product
            // strictly exceeded 2^32 - 1. Both reported dimensions
            // are 1-based (`+ 1`-ed from the on-disk Minus-One
            // fields) and therefore land in [1, 2^24].
            assert_eq!(
                data.len(),
                10,
                "§2.7.1 CanvasTooLarge implies a well-formed 10-byte payload",
            );
            assert!(
                (1..=(1u32 << 24)).contains(&canvas_width),
                "§2.7.1 reported canvas_width out of [1, 2^24]: {canvas_width}",
            );
            assert!(
                (1..=(1u32 << 24)).contains(&canvas_height),
                "§2.7.1 reported canvas_height out of [1, 2^24]: {canvas_height}",
            );
            let product = u64::from(canvas_width) * u64::from(canvas_height);
            assert!(
                product > u64::from(u32::MAX),
                "§2.7.1 CanvasTooLarge raised but product {product} fits in u32",
            );
        }
    }
});
