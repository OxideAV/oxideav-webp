#![no_main]

//! Parse arbitrary fuzz-supplied bytes through the §2.7.1.1 ANIM chunk
//! parser entry point `oxideav_webp::anim::AnimHeader::parse`.
//!
//! The §2 RIFF walker in `decode_webp` reaches `AnimHeader::parse` only
//! with a payload slice the container layer has already validated as an
//! `ANIM` chunk body, so the cross-product reachable through the
//! existing `decode.rs` / `extract_metadata.rs` / `roundtrip_animated.rs`
//! harnesses exercises this parser only along the "well-formed RIFF"
//! code path. `AnimHeader::parse` is also a public standalone surface —
//! `pub fn parse(payload: &[u8]) -> Result<Self, AnimError>` exported
//! through `pub mod anim` and the convenience wrapper
//! [`oxideav_webp::parse_anim_header`] — that downstream callers can
//! invoke against any byte slice they obtained from a different demuxer
//! (or, for that matter, an attacker-controlled buffer of arbitrary
//! length). This harness widens fuzz coverage onto that direct entry
//! point.
//!
//! The contract under test, per RFC 9649 §2.7.1.1 (Figure 8):
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow on the §2.7.1.1 BGRA-packed `as_u32_le` helper,
//!   no out-of-bounds index when the payload is shorter or longer than
//!   the 6 bytes Figure 8 fixes for the ANIM chunk.
//! * If the call returns `Ok(hdr)`, every field is internally
//!   consistent with the §2.7.1.1 Figure 8 byte layout:
//!     * `data.len() == 6` — the ANIM chunk payload is fixed-length at
//!       4-byte `Background Color` + 2-byte `Loop Count`, and any
//!       successful parse implies the input matched that length exactly.
//!     * `background_color.blue == data[0]`,
//!       `background_color.green == data[1]`,
//!       `background_color.red == data[2]`, and
//!       `background_color.alpha == data[3]` — §2.7.1.1 fixes the
//!       on-disk byte order at `[Blue, Green, Red, Alpha]`, so a
//!       successful parse must surface those four components verbatim
//!       at the named struct positions.
//!     * `background_color.as_u32_le()` packs the four channels in
//!       LSB→MSB `[B, G, R, A]` order — i.e. equals the little-endian
//!       `u32` reload of bytes 0..4. This pins down the §2.7.1.1
//!       "uint32 stored in BGRA byte order" carrier.
//!     * `loop_count` equals `u16::from_le_bytes([data[4], data[5]])`
//!       — the only multi-byte field besides the background color, and
//!       §2.3 stipulates RFC 9649 multi-byte fields are little-endian.
//!     * `loops_forever()` is true iff `loop_count == 0`. §2.7.1.1
//!       defines a zero loop count as "loop infinitely"; every non-zero
//!       value (up to 0xFFFF) means a finite repeat count.
//! * If the call returns `Err(BadPayloadLength { got })`, then `got`
//!   equals the supplied slice length and that length is not 6 — the
//!   only refusal trigger §2.7.1.1 Figure 8 fixes.
//!
//! Every branch of the contract is observable to libFuzzer: a panic
//! short-circuits, and every assertion below is a real §2.7.1.1
//! Figure 8 carrier violation if it ever fires.
//!
//! ## Iteration cost bound
//!
//! `AnimHeader::parse` is a fixed-cost branch chain (one length test)
//! plus four `u8` reads plus one 2-byte little-endian `u16` reload —
//! there is no allocation, no loop sized by the input, and no
//! recursion. A single fuzz iteration is microseconds regardless of
//! input length, so the harness can attempt arbitrary payload sizes
//! without iteration-cost concerns.
//!
//! ## Input layout
//!
//! The entire fuzz buffer is forwarded verbatim as the §2.7.1.1 ANIM
//! payload candidate. Inputs shorter or longer than 6 bytes hit the
//! `BadPayloadLength` refusal path; 6-byte inputs cover the full
//! §2.7.1.1 Figure 8 BGRA × loop-count cross-product (4 × 8-bit
//! background channels + 1 × 16-bit loop count = 2^48 distinct
//! payloads, all bit-patterns legal).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::anim::{AnimError, AnimHeader};

/// The §2.7.1.1 Figure 8 ANIM chunk payload length: 4-byte
/// `Background Color` + 2-byte `Loop Count`. Redefined locally so the
/// harness does not depend on a constant the crate does not export.
const ANIM_PAYLOAD_LEN: usize = 6;

fuzz_target!(|data: &[u8]| {
    match AnimHeader::parse(data) {
        Ok(hdr) => {
            // §2.7.1.1 Figure 8: success implies the payload was
            // exactly 6 bytes — every other length lands on
            // BadPayloadLength.
            assert_eq!(
                data.len(),
                ANIM_PAYLOAD_LEN,
                "§2.7.1.1 AnimHeader::parse returned Ok on a non-6-byte payload",
            );

            // §2.7.1.1: BGRA byte order for the background color
            // uint32. Re-derive each channel directly from the input
            // bytes and assert field-for-field equality. Mismatch on
            // any of these is a Figure 8 byte-layout regression.
            assert_eq!(
                hdr.background_color.blue, data[0],
                "§2.7.1.1 background_color.blue must equal payload[0]",
            );
            assert_eq!(
                hdr.background_color.green, data[1],
                "§2.7.1.1 background_color.green must equal payload[1]",
            );
            assert_eq!(
                hdr.background_color.red, data[2],
                "§2.7.1.1 background_color.red must equal payload[2]",
            );
            assert_eq!(
                hdr.background_color.alpha, data[3],
                "§2.7.1.1 background_color.alpha must equal payload[3]",
            );

            // §2.7.1.1: the BGRA uint32 packs the four channels in
            // LSB→MSB `[B, G, R, A]` order — i.e. the little-endian
            // `u32` reload of bytes 0..4.
            let expected_bg_u32 = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            assert_eq!(
                hdr.background_color.as_u32_le(),
                expected_bg_u32,
                "§2.7.1.1 background_color.as_u32_le must equal LE u32 of payload[0..4]",
            );

            // §2.7.1.1: 16-bit Loop Count, little-endian (RFC 9649 §2.3
            // multi-byte field convention).
            let expected_loop = u16::from_le_bytes([data[4], data[5]]);
            assert_eq!(
                hdr.loop_count, expected_loop,
                "§2.7.1.1 loop_count must equal LE u16 of payload[4..6]",
            );

            // §2.7.1.1: a `Loop Count` of 0 means "loop infinitely";
            // every other value (including 0xFFFF) is finite.
            assert_eq!(
                hdr.loops_forever(),
                hdr.loop_count == 0,
                "§2.7.1.1 loops_forever must be (loop_count == 0)",
            );
        }
        Err(AnimError::BadPayloadLength { got }) => {
            // §2.7.1.1 Figure 8: BadPayloadLength is raised iff the
            // input is not exactly 6 bytes long, and `got` reports the
            // actual length the parser observed.
            assert_eq!(
                got,
                data.len(),
                "§2.7.1.1 BadPayloadLength.got must equal the input slice length",
            );
            assert_ne!(
                got, ANIM_PAYLOAD_LEN,
                "§2.7.1.1 BadPayloadLength must not be raised on a 6-byte payload",
            );
        }
    }
});
