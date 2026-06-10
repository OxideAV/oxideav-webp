#![no_main]

//! Build arbitrary fuzz-supplied §6.2.2 meta-prefix indexes through the
//! standalone validated constructor
//! `oxideav_webp::vp8l_decode::MetaPrefixIndex::from_parts` and drive
//! the §6.2.2 per-pixel group selection `meta_code_for` across every
//! block of the resulting grid.
//!
//! When a VP8L §5.1 ARGB image sets the §6.2.2 meta-prefix bit, the
//! decoder reads `prefix_bits = ReadBits(3) + 2`, derives a block grid
//! of `DIV_ROUND_UP(image_width, 1 << prefix_bits) ×
//! DIV_ROUND_UP(image_height, 1 << prefix_bits)` blocks, decodes the
//! entropy image, and folds each entropy pixel's red+green channels
//! into one 16-bit meta-prefix code per block
//! (`(entropy_pixel >> 8) & 0xffff`). Every subsequent pixel decode
//! starts by resolving its prefix-code group through the §6.2.2
//! position formula:
//!
//! ```text
//! int position =
//!     (y >> prefix_bits) * prefix_image_width + (x >> prefix_bits);
//! int meta_prefix_code = (entropy_image[position] >> 8) & 0xffff;
//! ```
//!
//! Sibling harnesses cover the layers around this primitive —
//! `parse_meta_prefix` (§5.2.3 color-cache info + §6.2.2 preamble up to
//! the entropy-image boundary), `decode` (full §2 RIFF + §3..§5 entry
//! point), `roundtrip_lossless` (encode → decode equality oracle) — but
//! **none** of them drives the §6.2.2 block-lookup table itself across
//! the full `(prefix_bits, block_width, block_height, meta_codes)`
//! cross-product. This eighteenth harness does, cross-checking every
//! outcome against the §6.2.2 spec rules with a shadow model.
//!
//! The contract under test, per RFC 9649 §6.2.2:
//!
//! * `from_parts` always returns a `Result` — no panic, no arithmetic
//!   overflow in the `block_width * block_height` expected-count
//!   product.
//! * `prefix_bits ∉ [2, 9]` ⇒ `InvalidPrefixBits { prefix_bits }`
//!   echoing the call (the on-wire field is `ReadBits(3) + 2`, so only
//!   that window is bitstream-reachable).
//! * Otherwise `block_width == 0 || block_height == 0` ⇒
//!   `EmptyIndex { block_width, block_height }` echoing the call (the
//!   §6.2.2 `DIV_ROUND_UP` derivation of an ≥1×≥1 image never yields a
//!   zero grid).
//! * Otherwise `meta_codes.len() != block_width * block_height` ⇒
//!   `CodeCountMismatch` with `expected == block_width * block_height`
//!   and `got == meta_codes.len()` (one code per block, scan-line
//!   order).
//! * On success the accessors echo the parts verbatim;
//!   `num_prefix_groups()` equals `max(meta_codes) + 1` (§6.2.2
//!   "Interpretation of Meta Prefix Codes"); and for every block
//!   `(bx, by)` of the grid, `meta_code_for(x, y)` at each corner of
//!   the block's `(1 << prefix_bits)`-pixel-square covered area equals
//!   `meta_codes[by * block_width + bx]` per the §6.2.2 position
//!   formula.
//! * The constructor is deterministic — the same parts rebuild an
//!   equal index, and an index rebuilt from a successful index's own
//!   accessors equals it.
//!
//! Every assertion below is a real §6.2.2 carrier violation if it ever
//! fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — `prefix_bits`, masked to `[0, 15]` so roughly half the
//!   draws land inside the §6.2.2 `[2, 9]` window and half exercise the
//!   rejection.
//! * Byte `1` — `block_width`, modulo 33 (`[0, 32]`; 0 reaches the
//!   `EmptyIndex` refusal).
//! * Byte `2` — `block_height`, modulo 33 (likewise).
//! * Byte `3` — count skew: when nonzero, the supplied `meta_codes`
//!   length is shifted off `block_width * block_height` by
//!   `(skew % 5) - 2` entries (saturating at 0) so the
//!   `CodeCountMismatch` refusal is routinely reached alongside the
//!   exact-count success path.
//! * Bytes `[4..]` — repeated little-endian u16 words supplying the
//!   meta-prefix codes verbatim (cycled if fewer words than blocks are
//!   supplied; an empty tail falls back to 0).
//!
//! ## Iteration cost bound
//!
//! The grid is capped at 32 × 32 = 1024 blocks and the per-block
//! cross-check probes 4 corners, so an iteration costs at most ~4 K
//! constant-time lookups — well inside the libFuzzer budget.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_decode::{MetaPrefixIndex, MetaPrefixIndexError};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let prefix_bits = data[0] & 0x0f;
    let block_width = (data[1] % 33) as u32;
    let block_height = (data[2] % 33) as u32;
    let skew = data[3];

    // Supply the meta-prefix codes verbatim from the fuzz tail, cycled
    // to the requested count.
    let words: Vec<u16> = data[4..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let exact = (block_width as usize) * (block_height as usize);
    let count = if skew == 0 {
        exact
    } else {
        // Shift the supplied count off the §6.2.2 expectation by
        // [-2, +2] entries (0 shift re-lands on the success path).
        (exact as i64 + (skew % 5) as i64 - 2).max(0) as usize
    };
    let meta_codes: Vec<u16> = (0..count)
        .map(|i| {
            if words.is_empty() {
                0
            } else {
                words[i % words.len()]
            }
        })
        .collect();

    let result =
        MetaPrefixIndex::from_parts(prefix_bits, block_width, block_height, meta_codes.clone());

    match result {
        Ok(index) => {
            // §6.2.2 success implies every carrier invariant held.
            assert!(
                (2..=9).contains(&prefix_bits),
                "§6.2.2 success implies prefix_bits {prefix_bits} in [2, 9]",
            );
            assert!(
                block_width >= 1 && block_height >= 1,
                "§6.2.2 success implies a nonempty {block_width}x{block_height} grid",
            );
            assert_eq!(
                meta_codes.len(),
                exact,
                "§6.2.2 success implies one code per block",
            );

            // Accessors echo the parts verbatim.
            assert_eq!(index.prefix_bits(), prefix_bits);
            assert_eq!(index.block_width(), block_width);
            assert_eq!(index.block_height(), block_height);
            assert_eq!(index.meta_codes(), &meta_codes[..]);

            // §6.2.2 "Interpretation of Meta Prefix Codes":
            // num_prefix_groups = max(entropy image) + 1.
            let shadow_max = meta_codes.iter().copied().max().unwrap_or(0) as usize;
            assert_eq!(
                index.num_prefix_groups(),
                shadow_max + 1,
                "§6.2.2 num_prefix_groups must be max(entropy image) + 1",
            );

            // §6.2.2 position formula, probed at the four corners of
            // every block's covered pixel area:
            //   position = (y >> prefix_bits) * block_width
            //            + (x >> prefix_bits).
            let block = 1u32 << prefix_bits;
            for by in 0..block_height {
                for bx in 0..block_width {
                    let want = meta_codes[(by * block_width + bx) as usize];
                    for &x in &[bx * block, bx * block + (block - 1)] {
                        for &y in &[by * block, by * block + (block - 1)] {
                            assert_eq!(
                                index.meta_code_for(x, y),
                                want,
                                "§6.2.2 meta_code_for({x}, {y}) must select block ({bx}, {by})",
                            );
                        }
                    }
                }
            }

            // Determinism: the same parts rebuild an equal index, and
            // the index's own accessors round-trip through from_parts.
            let rebuilt = MetaPrefixIndex::from_parts(
                prefix_bits,
                block_width,
                block_height,
                meta_codes.clone(),
            )
            .expect("§6.2.2 replay of a successful build must also succeed");
            assert_eq!(rebuilt, index, "§6.2.2 rebuild must be identical");
            let round_tripped = MetaPrefixIndex::from_parts(
                index.prefix_bits(),
                index.block_width(),
                index.block_height(),
                index.meta_codes().to_vec(),
            )
            .expect("§6.2.2 accessor round-trip must succeed");
            assert_eq!(round_tripped, index, "§6.2.2 round-trip must be identical");
        }
        Err(MetaPrefixIndexError::InvalidPrefixBits { prefix_bits: pb }) => {
            assert_eq!(pb, prefix_bits, "§6.2.2 refusal echoes prefix_bits");
            assert!(
                !(2..=9).contains(&prefix_bits),
                "§6.2.2 InvalidPrefixBits must imply prefix_bits {prefix_bits} outside [2, 9]",
            );
        }
        Err(MetaPrefixIndexError::EmptyIndex {
            block_width: bw,
            block_height: bh,
        }) => {
            assert_eq!((bw, bh), (block_width, block_height));
            // Check precedence: the prefix_bits gate passed first.
            assert!(
                (2..=9).contains(&prefix_bits),
                "§6.2.2 EmptyIndex implies the prefix_bits gate passed",
            );
            assert!(
                block_width == 0 || block_height == 0,
                "§6.2.2 EmptyIndex must imply a zero-block {block_width}x{block_height} grid",
            );
        }
        Err(MetaPrefixIndexError::CodeCountMismatch {
            block_width: bw,
            block_height: bh,
            expected,
            got,
        }) => {
            assert_eq!((bw, bh), (block_width, block_height));
            assert_eq!(
                expected,
                block_width as u64 * block_height as u64,
                "§6.2.2 CodeCountMismatch reports the block count as expected",
            );
            assert_eq!(got, meta_codes.len());
            // Check precedence: both earlier gates passed.
            assert!(
                (2..=9).contains(&prefix_bits) && block_width >= 1 && block_height >= 1,
                "§6.2.2 CodeCountMismatch implies the prefix_bits + grid gates passed",
            );
            assert_ne!(
                got as u64, expected,
                "§6.2.2 CodeCountMismatch must imply a count off the §6.2.2 expectation",
            );
        }
    }
});
