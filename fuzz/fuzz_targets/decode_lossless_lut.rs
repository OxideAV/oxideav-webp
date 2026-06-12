#![no_main]

//! Decode adversarial VP8L bitstreams through the §4 + §5 + §6 full
//! lossless decode path `oxideav_webp::vp8l_transform::{decode_lossless,
//! decode_lossless_headerless}` at carrier dimensions wide enough that
//! the round-284 §6.2.1 read-symbol fast path actually runs hot — the
//! 256-entry primary lookup table (codes with ≥ 32 used symbols, code
//! length ≤ 8 resolved by one peek + one load) *and* its > 8-bit
//! continuation walk, plus the word-load `BitReader::read_bits` /
//! `peek_bits` / `advance_bits` carrying every fixed-width field.
//!
//! The round-273 `decode_lossless` sibling pins the same entry points
//! but clamps the carrier into `[1, 8]` (≤ 64 pixels), so a typical
//! iteration reads a handful of symbols per prefix code — the lookup
//! table is built (the gate is on the *code's* used-symbol count, not
//! the image size) but barely exercised, and near-EOF fallbacks at
//! every cursor phase are rare. This harness widens the carrier into
//! `[1, 64]` (≤ 4096 pixels, still microseconds per iteration) and its
//! corpus is seeded from the VP8L chunk payloads of the committed
//! fixture corpus — real streams whose §6.2 prefix-code groups carry
//! 100+-symbol codes with 9..15-bit tails, which a coverage-guided
//! mutator then corrupts adversarially (truncations landing mid-code at
//! every bit phase, code-length tables rewritten across the
//! `MIN_LOOKUP_USED` gate, LZ77/cache symbols steered into the long-code
//! continuation rows).
//!
//! The contract under test is the round-273 one, now load-bearing for
//! the fast path:
//!
//! * Both entry points always return a `Result` — no panic, no
//!   debug-build overflow, no out-of-bounds index, no allocation an
//!   unvalidated field sized. In particular the primary-table fast path
//!   must never advance the cursor past the slice (its near-EOF table
//!   outcomes are replayed through the per-bit walk) and the word-load
//!   `BitReader` must clamp every tail load.
//! * `Ok(image)` echoes the carrier (`width()` / `height()`, even after
//!   a §4.4 color-indexing transform un-bundles the subsampled internal
//!   width) and carries exactly `width * height` pixels.
//! * Determinism: replaying the same bytes + carrier yields a
//!   byte-identical pixel buffer — on these wide carriers that replay
//!   re-runs the very same lookup-table hits and continuation walks, so
//!   any nondeterminism in the table build or the cursor bookkeeping
//!   fires here.
//!
//! The symbol-level fast-path-vs-reference equivalence is pinned by the
//! `read_symbol_lut_diff` differential sibling; this harness covers the
//! assembled pipeline those symbols feed (transform sub-images, color
//! cache, LZ77 windows, inverse-transform chain) at realistic scale.
//!
//! ## Input layout
//!
//! * Byte `0` — `width`, clamped into `[1, 64]` (`data[0] % 64 + 1`).
//! * Byte `1` — `height`, clamped into `[1, 64]` likewise.
//! * Bytes `[2..]` — the VP8L chunk-payload bytes, read by
//!   `decode_lossless` past the §3.4 5-byte image-header (transform
//!   list at bit 40) and by `decode_lossless_headerless` from bit 0
//!   (the §2.7.1.2 ALPH form).
//!
//! ## Iteration cost bound
//!
//! At most `64 × 64 = 4096` main-image pixels plus bounded §7.3
//! sub-images and at most four §4 transforms; every read is clamped at
//! the slice end. A single iteration completes in microseconds to a few
//! milliseconds regardless of input length.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_transform::{decode_lossless, decode_lossless_headerless};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Carrier dimensions wide enough to keep the §6.2.1 lookup-table
    // fast path hot (hundreds-to-thousands of read_symbol calls per
    // accepted stream) while staying ≤ 4096 pixels per decode.
    let width = u32::from(data[0] % 64) + 1;
    let height = u32::from(data[1] % 64) + 1;

    let payload = &data[2..];

    // ---- §4 headerful driver: the form `decode_webp` routes the VP8L
    // chunk through (BitReader past the §3.4 5-byte image-header). ----
    if let Ok(image) = decode_lossless(payload, width, height) {
        assert_eq!(
            image.width(),
            width,
            "§4 decoded lossless width must equal the carrier width",
        );
        assert_eq!(
            image.height(),
            height,
            "§4 decoded lossless height must equal the carrier height",
        );
        assert_eq!(
            image.pixels().len(),
            (width as usize) * (height as usize),
            "§4 decoded lossless image must carry width*height pixels",
        );
        let replay = decode_lossless(payload, width, height)
            .expect("§4 replay of a successful headerful decode must also succeed");
        assert_eq!(
            replay.pixels(),
            image.pixels(),
            "§4 decode_lossless must be deterministic over the same bytes",
        );
    }

    // ---- §2.7.1.2 / §3 headerless driver: the ALPH form, same bytes
    // from bit 0, so every prefix-code read lands at a shifted bit
    // phase relative to the headerful pass. ----
    if let Ok(image) = decode_lossless_headerless(payload, width, height) {
        assert_eq!(
            image.width(),
            width,
            "§3 decoded headerless width must equal the carrier width",
        );
        assert_eq!(
            image.height(),
            height,
            "§3 decoded headerless height must equal the carrier height",
        );
        assert_eq!(
            image.pixels().len(),
            (width as usize) * (height as usize),
            "§3 decoded headerless image must carry width*height pixels",
        );
        let replay = decode_lossless_headerless(payload, width, height)
            .expect("§3 replay of a successful headerless decode must also succeed");
        assert_eq!(
            replay.pixels(),
            image.pixels(),
            "§3 decode_lossless_headerless must be deterministic over the same bytes",
        );
    }
});
