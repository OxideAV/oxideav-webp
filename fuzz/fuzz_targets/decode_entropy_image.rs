#![no_main]

//! Decode arbitrary fuzz-supplied bits through the §6.2.2 *entropy
//! image* decode path standalone entry point
//! `oxideav_webp::vp8l_decode::decode_entropy_image`.
//!
//! When a VP8L §5.1 ARGB image sets the §6.2.2 meta-prefix bit, the
//! decoder reads `prefix_bits = ReadBits(3) + 2`, derives a block grid
//! of `DIV_ROUND_UP(image_width, 1 << prefix_bits) ×
//! DIV_ROUND_UP(image_height, 1 << prefix_bits)` blocks, and then
//! decodes the §6.2.2 *entropy image* — itself a §7.3
//! `entropy-coded-image` (a §5.2.3 `color-cache-info` bit, one §6.2
//! prefix-code group, and the §5.2 LZ77 / color-cache data) of exactly
//! `prefix_image_width × prefix_image_height` pixels — folding each
//! decoded entropy pixel's red+green channels into one 16-bit
//! meta-prefix code per block:
//!
//! ```text
//! meta_prefix_code = (entropy_pixel >> 8) & 0xffff
//! ```
//!
//! `decode_entropy_image` is the bitstream-level producer of the
//! `MetaPrefixIndex` table: given a `BitReader` positioned at the start
//! of the entropy image (just past the `prefix_bits` field), the
//! recorded `prefix_bits`, and the §6.2.2-derived
//! `(prefix_image_width, prefix_image_height)` block dimensions, it
//! decodes the entropy-coded sub-image, folds its pixels into the
//! per-block meta-codes, and returns the assembled
//! [`oxideav_webp::vp8l_decode::MetaPrefixIndex`].
//!
//! Sibling harnesses cover the layers around this primitive —
//! `parse_meta_prefix` (§5.2.3 color-cache info + §6.2.2 preamble up to
//! the entropy-image boundary; it stops *at* this image, never decodes
//! it), `meta_prefix_index` (the standalone constructor
//! `MetaPrefixIndex::from_parts` + per-pixel `meta_code_for`, fed
//! already-decoded meta-codes), `decode` (full §2 RIFF + §3..§5 entry
//! point), and `roundtrip_lossless` (encode → decode equality oracle).
//! **None** of them drives the §6.2.2 entropy-image bitstream decode
//! itself across an attacker-controlled `(prefix_bits,
//! prefix_image_width, prefix_image_height, bitstream)` cross-product.
//! This nineteenth harness does.
//!
//! The contract under test, per RFC 9649 §6.2.2 + §7.3:
//!
//! * `decode_entropy_image` always returns a `Result` — no panic, no
//!   debug-build integer overflow, no out-of-bounds index when the
//!   bitstream is empty, truncated, or arbitrarily long, and no
//!   allocation sized by a header field the §5 / §6.2 readers did not
//!   bound. Every read goes through
//!   [`oxideav_webp::vp8l_stream::BitReader`] whose EOF path raises
//!   `Eof`, never an underflow.
//! * If the call returns `Ok(index)`:
//!     * The accessors echo the carrier triple verbatim:
//!       `prefix_bits()`, `block_width() == prefix_image_width`,
//!       `block_height() == prefix_image_height`.
//!     * §7.3: the entropy-coded sub-image carries exactly one pixel
//!       per block, so `meta_codes().len() == prefix_image_width *
//!       prefix_image_height`.
//!     * §6.2.2 "Interpretation of Meta Prefix Codes":
//!       `num_prefix_groups() == max(meta_codes) + 1`.
//!     * §6.2.2 fold: each meta-code is the `(entropy_pixel >> 8) &
//!       0xffff` red+green fold of the matching entropy-image pixel.
//!       This is cross-checked against an **independent** decode of the
//!       same bytes through the public sibling
//!       `vp8l_decode::decode_entropy_coded_image`, which decodes the
//!       same §7.3 block and exposes the raw ARGB pixels — the harness
//!       refolds them and asserts byte-equality with the meta-codes
//!       `decode_entropy_image` produced, and asserts both readers
//!       advanced to the same bit position.
//!     * §6.2.2 carrier asymmetry: `decode_entropy_image` records
//!       `prefix_bits` verbatim (an opaque carrier — it never re-derives
//!       a block size from it here), so an `Ok` index may carry any
//!       `prefix_bits ∈ [0, 15]`. Rebuilding through the validated
//!       constructor `MetaPrefixIndex::from_parts` reproduces the index
//!       identically when `prefix_bits ∈ [2, 9]` (the §6.2.2
//!       `ReadBits(3) + 2` wire window) and is refused with
//!       `InvalidPrefixBits` otherwise — both arms cross-checked.
//!     * Determinism: replaying the same bytes + carrier triple yields
//!       an identical index advanced to an identical bit position.
//! * If the call returns `Err(_)`, the harness asserts only that it
//!   returned (no panic). The granular §5.2 / §6.2 refusal modes are
//!   cross-checked by the sibling `decode` / `parse_meta_prefix`
//!   harnesses through their own entry points; here the degenerate
//!   `prefix_image_width == 0 || prefix_image_height == 0` refusal is
//!   additionally pinned to the §6.2.2 `EmptyEntropyImage` variant.
//!
//! Every assertion below is a real §6.2.2 / §7.3 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — `prefix_bits`, masked to `[0, 15]`. Only `[2, 9]` is
//!   §6.2.2 wire-reachable (`ReadBits(3) + 2`), but `decode_entropy_image`
//!   itself takes `prefix_bits` as an opaque `u8` carrier (it only
//!   records it into the index, never re-derives a block size from it
//!   inside this function), so the full byte range is forwarded to
//!   stress the `MetaPrefixIndex` carrier without constraining the
//!   §7.3 sub-image decode.
//! * Byte `1` — `prefix_image_width`, modulo 9 (`[0, 8]`; 0 reaches the
//!   §6.2.2 degenerate-dimension `EmptyEntropyImage` refusal).
//! * Byte `2` — `prefix_image_height`, modulo 9 (likewise).
//! * Bytes `[3..]` — the §7.3 `entropy-coded-image` bit sequence read
//!   by both `decode_entropy_image` and the cross-check
//!   `decode_entropy_coded_image`. A short or empty tail raises `Eof` on
//!   the first color-cache-info bit.
//!
//! ## Iteration cost bound
//!
//! The entropy-coded sub-image is at most `8 × 8 = 64` pixels, so the
//! §5.2 / §6.2 decode loop is bounded by 64 emitted pixels plus the
//! per-group §3.7 prefix-code-table reads (the 19-symbol code-length
//! alphabet plus the per-code alphabets). A single fuzz iteration
//! completes in microseconds to milliseconds regardless of input
//! length; `BitReader` indexes by `usize` across the slice so every
//! read is clamped at the slice end.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_decode::{
    decode_entropy_coded_image, decode_entropy_image, DecodeError, MetaPrefixIndex,
    MetaPrefixIndexError,
};
use oxideav_webp::vp8l_stream::BitReader;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    // §6.2.2 carrier triple. `prefix_bits` is forwarded as an opaque u8
    // (the function records it without re-deriving a block size, so the
    // full range exercises the `MetaPrefixIndex` carrier); the block
    // dimensions stay small so the §7.3 sub-image decode is bounded, and
    // 0 reaches the degenerate-dimension refusal.
    let prefix_bits = data[0] & 0x0f;
    let prefix_image_width = u32::from(data[1] % 9);
    let prefix_image_height = u32::from(data[2] % 9);

    let payload = &data[3..];
    let total_bits = payload.len() * 8;

    let mut reader = BitReader::new(payload);
    let result = decode_entropy_image(
        &mut reader,
        prefix_bits,
        prefix_image_width,
        prefix_image_height,
    );

    match result {
        Ok(index) => {
            // §6.2.2 success is only reachable with both block
            // dimensions ≥ 1 (a zero dimension short-circuits to the
            // `EmptyEntropyImage` refusal before any pixel is decoded).
            assert!(
                prefix_image_width >= 1 && prefix_image_height >= 1,
                "§6.2.2 decode_entropy_image success implies a nonempty {prefix_image_width}x{prefix_image_height} grid",
            );

            // The accessors echo the carrier triple verbatim.
            assert_eq!(
                index.prefix_bits(),
                prefix_bits,
                "§6.2.2 prefix_bits must be recorded verbatim",
            );
            assert_eq!(
                index.block_width(),
                prefix_image_width,
                "§6.2.2 block_width must equal prefix_image_width",
            );
            assert_eq!(
                index.block_height(),
                prefix_image_height,
                "§6.2.2 block_height must equal prefix_image_height",
            );

            // §7.3: the entropy-coded sub-image carries exactly one
            // pixel per block — one meta-code per `(prefix_image_width *
            // prefix_image_height)` block.
            let expected_codes = (prefix_image_width as usize) * (prefix_image_height as usize);
            assert_eq!(
                index.meta_codes().len(),
                expected_codes,
                "§7.3 entropy image carries one meta-code per block",
            );

            // §6.2.2 "Interpretation of Meta Prefix Codes":
            // num_prefix_groups = max(meta-codes) + 1.
            let shadow_max = index.meta_codes().iter().copied().max().unwrap_or(0) as usize;
            assert_eq!(
                index.num_prefix_groups(),
                shadow_max + 1,
                "§6.2.2 num_prefix_groups must be max(entropy image) + 1",
            );

            // §6.2.2 fold cross-check against an independent decode of
            // the same bytes through the public sibling
            // `decode_entropy_coded_image`. It decodes the same §7.3
            // block and exposes the raw ARGB pixels; the harness refolds
            // each pixel's red+green channels and asserts byte-equality
            // with the meta-codes `decode_entropy_image` produced, and
            // asserts both readers advanced to the same bit position.
            let mut ref_reader = BitReader::new(payload);
            let ref_image = decode_entropy_coded_image(
                &mut ref_reader,
                prefix_image_width,
                prefix_image_height,
            )
            .expect("§7.3 re-decode of a successful entropy image must also succeed");
            assert_eq!(
                ref_reader.bit_position(),
                reader.bit_position(),
                "§7.3 both readers must advance to the same bit position",
            );
            assert_eq!(
                ref_image.pixels().len(),
                expected_codes,
                "§7.3 re-decoded entropy image must carry one pixel per block",
            );
            let refolded: Vec<u16> = ref_image
                .pixels()
                .iter()
                .map(|&argb| ((argb >> 8) & 0xffff) as u16)
                .collect();
            assert_eq!(
                index.meta_codes(),
                &refolded[..],
                "§6.2.2 meta-codes must be the (entropy_pixel >> 8) & 0xffff fold of the decoded pixels",
            );

            // §6.2.2 carrier asymmetry: `decode_entropy_image` records
            // `prefix_bits` verbatim from the round-106 header (it never
            // re-derives a block size from it inside this function), so
            // an `Ok` index can carry any `prefix_bits ∈ [0, 15]`. The
            // standalone `from_parts` constructor, by contrast, enforces
            // the §6.2.2 `prefix_bits = ReadBits(3) + 2` wire window
            // `[2, 9]`. Cross-check both arms of the asymmetry:
            let from_parts = MetaPrefixIndex::from_parts(
                index.prefix_bits(),
                index.block_width(),
                index.block_height(),
                index.meta_codes().to_vec(),
            );
            if (2..=9).contains(&index.prefix_bits()) {
                // Inside the wire window the round-trip must reproduce
                // the index identically.
                let round_tripped = from_parts.expect(
                    "§6.2.2 accessor round-trip through from_parts must succeed for prefix_bits in [2, 9]",
                );
                assert_eq!(
                    round_tripped, index,
                    "§6.2.2 index must rebuild identically from its own accessors",
                );
            } else {
                // Outside the wire window `from_parts` must refuse with
                // `InvalidPrefixBits` echoing the recorded value — the
                // documented §6.2.2 carrier asymmetry between the
                // bitstream path and the validated constructor.
                match from_parts {
                    Err(MetaPrefixIndexError::InvalidPrefixBits { prefix_bits: pb }) => {
                        assert_eq!(
                            pb,
                            index.prefix_bits(),
                            "§6.2.2 from_parts InvalidPrefixBits must echo the recorded prefix_bits",
                        );
                    }
                    other => panic!(
                        "§6.2.2 from_parts must reject prefix_bits {} (outside [2, 9]) with InvalidPrefixBits, got {other:?}",
                        index.prefix_bits(),
                    ),
                }
            }

            // Determinism: replaying the same bytes + carrier triple
            // yields an identical index advanced to an identical bit
            // position.
            let mut replay_reader = BitReader::new(payload);
            let replay = decode_entropy_image(
                &mut replay_reader,
                prefix_bits,
                prefix_image_width,
                prefix_image_height,
            )
            .expect("§6.2.2 replay of a successful decode must also succeed");
            assert_eq!(
                replay, index,
                "§6.2.2 decode_entropy_image must be deterministic over the same bytes",
            );
            assert_eq!(
                replay_reader.bit_position(),
                reader.bit_position(),
                "§6.2.2 replay must advance the reader identically",
            );

            // BitReader clamps every read at the slice end, so a
            // successful decode cannot have walked past the slice.
            assert!(
                reader.bit_position() <= total_bits,
                "§7.3 successful decode advanced to bit {} beyond slice bit length {total_bits}",
                reader.bit_position(),
            );
        }
        Err(DecodeError::EmptyEntropyImage {
            prefix_image_width: w,
            prefix_image_height: h,
        }) => {
            // §6.2.2 degenerate-dimension refusal: reached iff at least
            // one block dimension is zero, with the offending dimensions
            // echoed verbatim.
            assert_eq!(
                (w, h),
                (prefix_image_width, prefix_image_height),
                "§6.2.2 EmptyEntropyImage must echo the carrier dimensions",
            );
            assert!(
                prefix_image_width == 0 || prefix_image_height == 0,
                "§6.2.2 EmptyEntropyImage must imply a zero-block {prefix_image_width}x{prefix_image_height} grid",
            );
        }
        Err(_) => {
            // §5.2 / §6.2: a bitstream-level refusal (truncated input,
            // prefix-code parse failure, out-of-range green symbol,
            // color-cache or backward-reference fault). The granular
            // refusal modes are cross-checked by the sibling `decode` /
            // `parse_meta_prefix` harnesses through their own entry
            // points; here the contract under test is only that the
            // call returned a `Result` rather than panicking.
        }
    }
});
