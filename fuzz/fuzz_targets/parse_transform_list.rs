#![no_main]

//! Parse arbitrary fuzz-supplied bits through the §4 VP8L transform-list
//! reader entry point `oxideav_webp::vp8l_stream::TransformList::read`.
//!
//! The §3 image-header peek lands the §2.6 lossless bitstream at the
//! start of the §4 transform-presence loop:
//!
//! ```text
//! while (ReadBits(1)) {              // transform present
//!   enum TransformType t = ReadBits(2);
//!   ...                              // per-type fixed fields, then a
//!                                    //   §5 entropy body (or none, for
//!                                    //   SUBTRACT_GREEN)
//! }
//! ```
//!
//! `TransformList::read` decodes the §4 *leading fixed fields* of each
//! present transform (§4.1 `size_bits = ReadBits(3) + 2` for predictor /
//! §4.2 color; §4.3 SUBTRACT_GREEN carries no data; §4.4
//! `color_table_size = ReadBits(8) + 1` plus the derived pixel-bundling
//! `width_bits`), stopping either at the terminating `0` presence bit
//! (structurally-complete list, no §5 body the reader cannot skip) or at
//! the byte/bit boundary where the first §5 entropy-coded body begins.
//! §4 also says "each transform is allowed to be used only once": a
//! repeat of any of the four `TransformType` values raises
//! `DuplicateTransform`. The function is a public standalone surface
//! exported through `pub mod vp8l_stream` (and the convenience wrapper
//! `oxideav_webp::read_vp8l_transform_list`) that downstream callers can
//! invoke against any byte slice they obtained from a different demuxer.
//!
//! This harness widens fuzz coverage onto that direct entry point with
//! no §3 image-header constraint and no implied §5 entropy consumption:
//! only the §4 transform-presence loop bitfield decomposition is under
//! test here. Sibling harnesses already cover the orthogonal parsers —
//! `parse_vp8x` (§2.7.1 Figure 7 octet), `parse_anmf` (§2.7.1.1 Figure 9
//! header), `parse_anim` (§2.7.1.1 Figure 8 carrier), `parse_alph`
//! (§2.7.1.2 Figure 10 info byte), `decode_alph` (§2.7.1.2 alpha plane),
//! `extract_metadata` (§2 RIFF walk for ICCP/EXIF/XMP), `decode` (full
//! §2 RIFF + §3..§5 entry), and `roundtrip_animated` / `roundtrip_lossless`
//! (encode→decode equality oracles).
//!
//! The contract under test, per RFC 9649 §4:
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow, no out-of-bounds index when the input is empty
//!   or arbitrarily long. Every read goes through
//!   [`oxideav_webp::vp8l_stream::BitReader`] whose EOF path raises
//!   `BitReaderEof`, never an underflow.
//! * If the call returns `Ok(list)`, every recorded transform is
//!   internally consistent with §4:
//!     * `list.transforms().len() <= 4` — §4 limits the read to one
//!       entry per `TransformType` (Predictor / Color / SubtractGreen /
//!       ColorIndexing), so the list cannot exceed four entries.
//!     * Every entry's `transform_type` tag matches its variant: the
//!       discriminant exposed by [`Transform::transform_type`] equals
//!       the §4 `TransformType` value the parser dispatched on.
//!     * No `TransformType` value is repeated across the list — §4 says
//!       "each transform is allowed to be used only once", and the
//!       parser raises `DuplicateTransform` rather than silently
//!       accepting one.
//!     * §4.1 / §4.2 `size_bits` lies in `2..=9` — the field is
//!       `ReadBits(3) + 2`, so its decoded range is `[2, 2+7] = [2, 9]`.
//!     * §4.4 `color_table_size` lies in `1..=256` — the field is
//!       `ReadBits(8) + 1`, so its decoded range is `[1, 1+255]`.
//!     * §4.4 pixel-bundling `width_bits` matches the §4.4 threshold
//!       table on `color_table_size`: 3 for `<= 2`, 2 for `<= 4`, 1 for
//!       `<= 16`, 0 otherwise.
//!     * `list.body_bit_position()` does not exceed the slice's total
//!       bit length — `BitReader` clamps every read at the slice end,
//!       so a successful `Ok` cannot have walked past it.
//!     * If `list.stopped_at_entropy_body()`, then the last entry has
//!       `has_entropy_body()` (the parser stopped *at* a §5 body, not
//!       past one); if `!stopped_at_entropy_body()`, the terminating
//!       `0` presence bit was consumed and no entry needs to follow.
//! * If the call returns `Err(TransformListError::Eof(_))`, the EOF is
//!   real: the captured `bit_pos + available` does not exceed the
//!   slice's bit length.
//! * If the call returns `Err(TransformListError::DuplicateTransform
//!   { transform_type })`, the rejection means at least two of the §4
//!   `ReadBits(2)` `TransformType` values along the parsed prefix
//!   matched the reported `transform_type`. (The harness records the
//!   raised variant but does not need to redo the parse to verify it;
//!   the parser raises `DuplicateTransform` only after observing the
//!   second presence + type pair.)
//!
//! Every assertion below is a real §4 carrier violation if it ever
//! fires; a panic short-circuits to libFuzzer.
//!
//! ## Iteration cost bound
//!
//! `TransformList::read` is a bounded loop: at most four iterations
//! before the §4 `DuplicateTransform` refusal triggers (one entry per
//! `TransformType`), and each iteration reads at most `1 + 2 + 8 = 11`
//! bits. Total bit reads in a successful parse are bounded by
//! `4 * 11 + 1 = 45` bits regardless of input length, and the
//! terminating presence bit is the only way the loop exits short of a
//! §5 body or a duplicate refusal. A single fuzz iteration is
//! microseconds regardless of input length.
//!
//! ## Input layout
//!
//! The entire fuzz buffer is forwarded verbatim as a §4 transform-list
//! byte sequence, with the [`BitReader`] positioned at bit 0 (no §3
//! image-header skip). The empty slice immediately hits the EOF refusal
//! path on the first `ReadBits(1)`; non-empty slices let the §4 Figure
//! ‟transform-presence loop" walk into the four `TransformType` arms
//! based on the LSB-first bit pattern of the fuzz buffer. This widens
//! coverage onto the per-arm fixed-field reads (`ReadBits(3) + 2` /
//! `ReadBits(8) + 1`), the §4 duplicate-detection refusal, the §4.4
//! `width_bits` derivation, and the §5-body boundary recording.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_stream::{
    BitReader, Transform, TransformList, TransformListError, TransformType,
};

fuzz_target!(|data: &[u8]| {
    let mut reader = BitReader::new(data);
    let total_bits = data.len() * 8;
    match TransformList::read(&mut reader) {
        Ok(list) => {
            let transforms = list.transforms();

            // §4: only four `TransformType` values exist and each may
            // appear at most once, so the list cannot exceed four
            // entries.
            assert!(
                transforms.len() <= 4,
                "§4 TransformList::read returned more than four entries: {}",
                transforms.len(),
            );

            // §4: every `TransformType` value appears at most once across
            // the list. The parser raises DuplicateTransform if a second
            // appearance is observed, so a successful Ok must have no
            // duplicates.
            let mut seen = [false; 4];
            for t in transforms {
                let idx = t.transform_type() as usize;
                assert!(
                    !seen[idx],
                    "§4 TransformList::read kept a duplicate {:?} entry in Ok output",
                    t.transform_type(),
                );
                seen[idx] = true;
            }

            // §4: every entry's `transform_type()` tag matches its
            // variant + the §4.1 / §4.2 / §4.4 fixed-field ranges hold.
            for t in transforms {
                match *t {
                    Transform::Predictor { size_bits } => {
                        assert_eq!(
                            t.transform_type(),
                            TransformType::Predictor,
                            "§4.1 Predictor entry's transform_type() must be Predictor",
                        );
                        // §4.1: size_bits = ReadBits(3) + 2 → [2, 9].
                        assert!(
                            (2..=9).contains(&size_bits),
                            "§4.1 Predictor size_bits out of range: {size_bits}",
                        );
                        // §4.1: Predictor carries a §5 entropy body.
                        assert!(
                            t.has_entropy_body(),
                            "§4.1 Predictor must report has_entropy_body() == true",
                        );
                    }
                    Transform::Color { size_bits } => {
                        assert_eq!(
                            t.transform_type(),
                            TransformType::Color,
                            "§4.2 Color entry's transform_type() must be Color",
                        );
                        // §4.2: size_bits = ReadBits(3) + 2 → [2, 9].
                        assert!(
                            (2..=9).contains(&size_bits),
                            "§4.2 Color size_bits out of range: {size_bits}",
                        );
                        // §4.2: Color carries a §5 entropy body.
                        assert!(
                            t.has_entropy_body(),
                            "§4.2 Color must report has_entropy_body() == true",
                        );
                    }
                    Transform::SubtractGreen => {
                        assert_eq!(
                            t.transform_type(),
                            TransformType::SubtractGreen,
                            "§4.3 SubtractGreen entry's transform_type() must be SubtractGreen",
                        );
                        // §4.3: SubtractGreen is the only no-body
                        // transform.
                        assert!(
                            !t.has_entropy_body(),
                            "§4.3 SubtractGreen must report has_entropy_body() == false",
                        );
                    }
                    Transform::ColorIndexing {
                        color_table_size,
                        width_bits,
                    } => {
                        assert_eq!(
                            t.transform_type(),
                            TransformType::ColorIndexing,
                            "§4.4 ColorIndexing entry's transform_type() must be ColorIndexing",
                        );
                        // §4.4: color_table_size = ReadBits(8) + 1 →
                        // [1, 256].
                        assert!(
                            (1..=256).contains(&color_table_size),
                            "§4.4 ColorIndexing color_table_size out of range: {color_table_size}",
                        );
                        // §4.4: pixel-bundling width_bits threshold
                        // table.
                        let expected_width_bits = if color_table_size <= 2 {
                            3
                        } else if color_table_size <= 4 {
                            2
                        } else if color_table_size <= 16 {
                            1
                        } else {
                            0
                        };
                        assert_eq!(
                            width_bits, expected_width_bits,
                            "§4.4 ColorIndexing width_bits must follow the threshold table on \
                             color_table_size = {color_table_size}",
                        );
                        // §4.4: ColorIndexing carries a §5 entropy body
                        // (the color table).
                        assert!(
                            t.has_entropy_body(),
                            "§4.4 ColorIndexing must report has_entropy_body() == true",
                        );
                    }
                }
            }

            // §4: the body bit position cannot exceed the slice's bit
            // length — `BitReader::read_bits` raises EOF before walking
            // past the end, so a successful Ok must be in-range.
            assert!(
                list.body_bit_position() <= total_bits,
                "§4 body_bit_position {} exceeded slice bit length {}",
                list.body_bit_position(),
                total_bits,
            );

            // §4: `stopped_at_entropy_body` is consistent with the last
            // entry's `has_entropy_body()`. If the parser stopped at a
            // §5 body, the last entry must be a body-bearing transform;
            // if it consumed the terminating `0` presence bit, either
            // the list is empty or the last entry is SubtractGreen
            // (the only no-body transform).
            if list.stopped_at_entropy_body() {
                let last = transforms.last().expect(
                    "§4 stopped_at_entropy_body == true requires at least one transform on record",
                );
                assert!(
                    last.has_entropy_body(),
                    "§4 stopped_at_entropy_body == true requires the last entry to carry a §5 body",
                );
            } else if let Some(last) = transforms.last() {
                assert!(
                    !last.has_entropy_body(),
                    "§4 stopped_at_entropy_body == false implies the loop consumed the terminating \
                     `0` presence bit, so the last entry must be SubtractGreen (no §5 body)",
                );
            }
        }
        Err(TransformListError::Eof(eof)) => {
            // §4 / §2: the BitReader EOF coordinate must be in range —
            // the position the failing read started at plus the bits it
            // could see must not exceed the slice's bit length.
            assert!(
                eof.bit_pos.saturating_add(eof.available) <= total_bits,
                "§4 BitReaderEof reported bit_pos {} + available {} > slice bit length {}",
                eof.bit_pos,
                eof.available,
                total_bits,
            );
            // The reader only raises EOF if the demand strictly exceeded
            // what was left.
            assert!(
                eof.wanted > eof.available,
                "§4 BitReaderEof wanted {} <= available {} should not have raised EOF",
                eof.wanted,
                eof.available,
            );
        }
        Err(TransformListError::DuplicateTransform { transform_type }) => {
            // §4: the raised type must be one of the four legal
            // `TransformType` values. `TransformType::from_bits` only
            // ever returns one of the four, so this is structurally
            // total — but it pins the contract.
            match transform_type {
                TransformType::Predictor
                | TransformType::Color
                | TransformType::SubtractGreen
                | TransformType::ColorIndexing => {}
            }
        }
    }
});
