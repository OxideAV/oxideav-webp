#![no_main]

//! Parse arbitrary fuzz-supplied bytes through the §2.7.1.2 ALPH chunk
//! info-byte parser entry point `oxideav_webp::alph::AlphHeader::parse`.
//!
//! The §2 RIFF walker in `decode_webp` reaches `AlphHeader::parse` only
//! with payload slices the container layer has already validated as
//! `ALPH` chunk bodies, and the existing `decode_alph` harness drives
//! the full §2.7.1.2 decode (`decode_alpha`) with constrained
//! `(width, height) ∈ [1, 64]²` so its alpha-bitstream traversal stays
//! bounded. `AlphHeader::parse` is also a public standalone surface —
//! `pub fn parse(payload: &[u8]) -> Result<Self, AlphError>` exported
//! through `pub mod alph` and the convenience wrapper
//! [`oxideav_webp::parse_alph_header`] — that downstream callers can
//! invoke against any byte slice they obtained from a different demuxer
//! (or, for that matter, an attacker-controlled buffer of arbitrary
//! length, including the empty slice). This harness widens fuzz
//! coverage onto that direct entry point with no dimension constraints
//! and no implied alpha-bitstream consumption: only the §2.7.1.2
//! Figure 10 info-byte bitfield decomposition is under test here.
//!
//! The contract under test, per RFC 9649 §2.7.1.2 (Figure 10):
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow, no out-of-bounds index when the payload is
//!   empty or arbitrarily long. The §2.7.1.2 info byte is a single
//!   octet so the parser only ever inspects `payload[0]`; any extra
//!   bytes after byte 0 are the "Alpha bitstream" the spec describes,
//!   which `AlphHeader::parse` deliberately ignores (its job is
//!   bitfield decomposition only).
//! * If the call returns `Ok(hdr)`, every field is internally
//!   consistent with the §2.7.1.2 Figure 10 MSB-first byte layout
//!   `Rsv|P|F|C`:
//!     * `payload.len() >= 1` — success implies at least one byte was
//!       observed (the info byte).
//!     * `hdr.info_byte == payload[0]` — the raw info byte is
//!       surfaced verbatim.
//!     * `hdr.reserved == (payload[0] >> 6) & 0b11` — §2.7.1.2 places
//!       `Rsv` at bits 7..6 of the info byte.
//!     * The `P` field at bits 5..4 is `(payload[0] >> 4) & 0b11`, and
//!       its typed value matches:
//!         * 0 → `AlphPreprocessing::None`
//!         * 1 → `AlphPreprocessing::LevelReduction`
//!         * 2 | 3 → `AlphPreprocessing::Reserved(v)`
//!     * The `F` field at bits 3..2 is `(payload[0] >> 2) & 0b11`, and
//!       its typed value matches:
//!         * 0 → `AlphFiltering::None`
//!         * 1 → `AlphFiltering::Horizontal`
//!         * 2 → `AlphFiltering::Vertical`
//!         * 3 → `AlphFiltering::Gradient`
//!     * The `C` field at bits 1..0 is `payload[0] & 0b11`, and its
//!       typed value matches:
//!         * 0 → `AlphCompression::None`
//!         * 1 → `AlphCompression::Lossless`
//!         * 2 | 3 → `AlphCompression::Reserved(v)`
//!     * `hdr.bitstream_offset() == 1` — §2.7.1.2 fixes the info byte
//!       at position 0 and the alpha bitstream immediately after, so
//!       the offset is a per-spec constant.
//! * If the call returns `Err(AlphError::EmptyPayload)`, then
//!   `payload.is_empty()` — empty payload is the only refusal the
//!   §2.7.1.2 info-byte parser raises (every other §2.7.1.2 refusal
//!   originates in [`oxideav_webp::alph::decode_alpha`]'s downstream
//!   bitstream-decode stages, not the info-byte parser itself).
//!
//! Every branch of the contract is observable to libFuzzer: a panic
//! short-circuits, and every assertion below is a real §2.7.1.2
//! Figure 10 carrier violation if it ever fires.
//!
//! ## Iteration cost bound
//!
//! `AlphHeader::parse` is a fixed-cost branch chain (one length test)
//! plus four 2-bit field extracts from a single byte — there is no
//! allocation, no loop sized by the input, and no recursion. A single
//! fuzz iteration is microseconds regardless of input length, so the
//! harness can attempt arbitrary payload sizes (including the empty
//! slice and slices longer than any conceivable §2.7.1.2 ALPH chunk)
//! without iteration-cost concerns.
//!
//! ## Input layout
//!
//! The entire fuzz buffer is forwarded verbatim as the §2.7.1.2 ALPH
//! payload candidate. The empty slice hits the `EmptyPayload` refusal
//! path; every other length lets `payload[0]` cover the full
//! §2.7.1.2 Figure 10 Rsv × P × F × C cross-product (4 × 4 × 4 × 4 =
//! 256 distinct info bytes, every bit-pattern legal: the spec
//! mandates readers IGNORE `Rsv`, and the "undefined" values of the
//! `C` and `P` fields are explicitly surfaced through their
//! `Reserved(_)` variants rather than raising an error at this layer).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::alph::{
    AlphCompression, AlphError, AlphFiltering, AlphHeader, AlphPreprocessing,
};

fuzz_target!(|data: &[u8]| {
    match AlphHeader::parse(data) {
        Ok(hdr) => {
            // §2.7.1.2 Figure 10: success implies the payload had at
            // least one byte (the info byte). The empty slice is the
            // single EmptyPayload trigger.
            assert!(
                !data.is_empty(),
                "§2.7.1.2 AlphHeader::parse returned Ok on an empty payload",
            );

            // §2.7.1.2 Figure 10: byte 0 is the info byte, surfaced
            // verbatim through `info_byte`.
            let info = data[0];
            assert_eq!(
                hdr.info_byte, info,
                "§2.7.1.2 info_byte must equal payload[0]",
            );

            // §2.7.1.2 Figure 10: Rsv at bits 7..6 of the info byte,
            // MSB-first byte layout.
            let expected_reserved = (info >> 6) & 0b11;
            assert_eq!(
                hdr.reserved, expected_reserved,
                "§2.7.1.2 reserved must equal (info_byte >> 6) & 0b11",
            );

            // §2.7.1.2 Figure 10: P at bits 5..4.
            let p_bits = (info >> 4) & 0b11;
            match (p_bits, hdr.preprocessing) {
                (0, AlphPreprocessing::None) => {}
                (1, AlphPreprocessing::LevelReduction) => {}
                (v @ (2 | 3), AlphPreprocessing::Reserved(rv)) => {
                    assert_eq!(
                        rv, v,
                        "§2.7.1.2 AlphPreprocessing::Reserved must carry the raw P bits",
                    );
                }
                (v, got) => panic!(
                    "§2.7.1.2 P bits = {v} did not decode to the expected variant (got {got:?})",
                ),
            }

            // §2.7.1.2 Figure 10: F at bits 3..2.
            let f_bits = (info >> 2) & 0b11;
            match (f_bits, hdr.filtering) {
                (0, AlphFiltering::None) => {}
                (1, AlphFiltering::Horizontal) => {}
                (2, AlphFiltering::Vertical) => {}
                (3, AlphFiltering::Gradient) => {}
                (v, got) => panic!(
                    "§2.7.1.2 F bits = {v} did not decode to the expected variant (got {got:?})",
                ),
            }

            // §2.7.1.2 Figure 10: C at bits 1..0.
            let c_bits = info & 0b11;
            match (c_bits, hdr.compression) {
                (0, AlphCompression::None) => {}
                (1, AlphCompression::Lossless) => {}
                (v @ (2 | 3), AlphCompression::Reserved(rv)) => {
                    assert_eq!(
                        rv, v,
                        "§2.7.1.2 AlphCompression::Reserved must carry the raw C bits",
                    );
                }
                (v, got) => panic!(
                    "§2.7.1.2 C bits = {v} did not decode to the expected variant (got {got:?})",
                ),
            }

            // §2.7.1.2: the alpha bitstream begins immediately after
            // the info byte, so the offset is a fixed constant 1.
            assert_eq!(
                hdr.bitstream_offset(),
                1,
                "§2.7.1.2 bitstream_offset must always be 1 (info byte is fixed at position 0)",
            );
        }
        Err(AlphError::EmptyPayload) => {
            // §2.7.1.2: EmptyPayload is raised iff the payload slice
            // is empty (zero bytes available for the info byte).
            assert!(
                data.is_empty(),
                "§2.7.1.2 EmptyPayload must not be raised on a non-empty payload",
            );
        }
        Err(other) => {
            // §2.7.1.2 Figure 10: the info-byte parser only raises
            // EmptyPayload. Every other AlphError variant
            // (DimensionsOverflow / RawLengthMismatch /
            // UnsupportedCompression / Vp8l) is produced exclusively
            // by `decode_alpha`'s bitstream-decode stages, never by
            // `AlphHeader::parse`. Surface a refusal mismatch if any
            // other variant ever escapes the parser.
            panic!("§2.7.1.2 AlphHeader::parse raised a non-EmptyPayload error: {other:?}");
        }
    }
});
