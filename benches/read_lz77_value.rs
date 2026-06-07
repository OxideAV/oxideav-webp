//! Criterion bench — decoder-side §3.6.2.2 `read_lz77_value` LZ77
//! prefix-code-to-value expansion across the §3.6.2.2 Table 4 prefix
//! ranges.
//!
//! `read_lz77_value` is the per-symbol second half of the §3.6.2.2
//! length / distance decode path: given a §3.6.2.2 prefix code
//! `[0..40)` already pulled from the prefix-code group's GREEN /
//! DISTANCE Huffman tree, it returns the decoded length-or-distance
//! value by either folding the prefix code (fast path: `prefix_code <
//! 4`) or by reading `extra_bits = (prefix_code - 2) >> 1` bits from
//! the bit reader and folding them with the §3.6.2.2 offset formula
//! `offset = (2 + (prefix_code & 1)) << extra_bits`.
//!
//! Inside `decode_one_symbol` (`vp8l_decode.rs`) the function is
//! invoked **twice per LZ77 match**: once for the length prefix code
//! and once for the distance prefix code. Every backward-reference
//! pixel run in a §5.2 single-group ARGB decode therefore costs at
//! least two `read_lz77_value` calls plus the §3.6.2.2.1 distance
//! mapping. The round-170 profile attributed a non-trivial share of
//! decode self-time to the surrounding `decode_one_symbol` body
//! *through* this function, so the per-prefix-code cost is a natural
//! next entry in the §3 decode per-pass inventory alongside the §3.7
//! encoder builder inventory landed at rounds 250 / 251.
//!
//! The §3.6.2.2 Table 4 splits naturally into three regimes that
//! exercise different microarchitectural paths through the function,
//! and this bench samples each of them:
//!
//! * *Fast path* — `prefix_code ∈ [0, 4)` — the function returns
//!   `prefix_code + 1` without touching the reader. This is the only
//!   branch a value-1..=4 LZ77 reference ever exercises, and per
//!   §3.6.2.2 Table 4 it covers the value range `1..=4` (one prefix
//!   code per value). Sampled with `prefix_code = 2` (the midpoint).
//! * *Short extra* — `prefix_code ∈ [4, 16)` — `extra_bits` runs
//!   `1..=7`, so the reader is hit but the per-call work stays small.
//!   This is the dominant regime for typical natural-image LZ77 runs
//!   (lengths up to 256 and the close-neighbour distances inside the
//!   §3.6.2.2.1 distance map). Sampled with `prefix_code = 10`
//!   (`extra_bits = 4`, covers values `34..=49`).
//! * *Long extra* — `prefix_code ∈ [16, 40)` — `extra_bits` runs
//!   `7..=18`. The reader is hit hard and the §3.6.2.2 offset
//!   formula's `<< extra_bits` reaches 18. This regime is
//!   distance-only (the §3.6.2.2 note caps length prefixes at 24, so
//!   only DISTANCE ever reaches `prefix_code = 39`). Sampled with
//!   `prefix_code = 30` (`extra_bits = 14`, covers values
//!   `32769..=49152`) and `prefix_code = 39` (`extra_bits = 18`, the
//!   maximum, covers values `786433..=1048576`).
//!
//! The per-call work is tiny — a few bit-shifts plus the §3.6.2.2
//! reader hit when `prefix_code >= 4` — so the bench measures one
//! call per iteration over a fresh-each-iteration `BitReader<'_>`.
//! The bit slice is pre-allocated outside `b.iter` and the reader is
//! constructed inside `b.iter` to keep the per-iteration setup
//! reader-construction-only. The reader-construction cost is the
//! same across every cell, so cross-cell deltas come exclusively
//! from the `read_lz77_value` body.
//!
//! A future rewrite (branchless prefix decode, table-driven
//! offset/extra-bits lookup, or §3.6.2.2-table inlining via const
//! arrays indexed by `prefix_code`) has direct A/B numbers at every
//! representative regime here.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench read_lz77_value -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_decode::read_lz77_value;
use oxideav_webp::vp8l_stream::BitReader;

/// Deterministic LCG fill matching the constants used across the §3.x
/// / §4.x per-pass benches so the extra-bits byte stream here is
/// reproducible across runs and hosts and cross-bench comparable to
/// the rest of the inventory.
fn lcg_step(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed
}

/// Fill `len` bytes with the same deterministic LCG used across the
/// §3.x / §4.x per-pass benches. The byte stream is what the reader
/// will pull `extra_bits` from each iteration.
fn lcg_bytes(len: usize) -> Vec<u8> {
    let mut seed: u32 = 0x9e37_79b9;
    let mut bytes = Vec::with_capacity(len);
    while bytes.len() < len {
        let w = lcg_step(&mut seed);
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

/// Bench `read_lz77_value(reader, prefix_code)` over a fresh reader
/// each iteration. `extra_bits` is implicit in `prefix_code` per
/// §3.6.2.2: `extra_bits = (prefix_code - 2) >> 1` when
/// `prefix_code >= 4`, otherwise 0.
///
/// The bit slice is built once outside `b.iter` and is long enough
/// that the reader never EOFs across the inner loop (32 KiB ⇒ 262144
/// bits, more than 14000 calls at the §3.6.2.2 maximum of 18 extra
/// bits per call).
fn bench_prefix_code(c: &mut Criterion, prefix_code: u32, label: &str) {
    let bytes = lcg_bytes(32 * 1024);
    let name = format!("read_lz77_value_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            // Per-iteration reader construction matches the inner
            // decode loop's reader cursor moving forward one §3.6.2.2
            // call at a time. We rebuild the reader each iteration so
            // the bit cursor doesn't drift past the slice — the
            // §3.6.2.2 body cost is what we're measuring, not the
            // per-iteration reader-restart cost (it's the same across
            // every cell, so cross-cell deltas isolate to the §3.6.2.2
            // body work).
            let mut reader = BitReader::new(black_box(&bytes));
            let value = read_lz77_value(&mut reader, black_box(prefix_code)).unwrap();
            black_box(value);
        })
    });
}

// Fast path: `prefix_code < 4` ⇒ 0 extra bits, returns `prefix_code +
// 1` without touching the reader. Per §3.6.2.2 Table 4 this covers
// the value range `1..=4` (one prefix code per value).
fn bench_fastpath_prefix2(c: &mut Criterion) {
    bench_prefix_code(c, 2, "fastpath_prefix2");
}

// Short extra: `prefix_code = 10` ⇒ `extra_bits = 4`, covers value
// range `34..=49` per §3.6.2.2 Table 4. Typical natural-image LZ77
// length / close-neighbour distance regime.
fn bench_short_extra_prefix10(c: &mut Criterion) {
    bench_prefix_code(c, 10, "short_extra_prefix10");
}

// Long extra: `prefix_code = 30` ⇒ `extra_bits = 14`, covers value
// range `32769..=49152` per §3.6.2.2 Table 4. Distance-only regime
// (the §3.6.2.2 note caps length prefixes at 24).
fn bench_long_extra_prefix30(c: &mut Criterion) {
    bench_prefix_code(c, 30, "long_extra_prefix30");
}

// Maximum extra: `prefix_code = 39` ⇒ `extra_bits = 18`, the largest
// §3.6.2.2 extra-bits count, covers value range `786433..=1048576`
// per §3.6.2.2 Table 4. Distance-only regime and the §3.6.2.2 hard
// upper bound.
fn bench_max_extra_prefix39(c: &mut Criterion) {
    bench_prefix_code(c, 39, "max_extra_prefix39");
}

criterion_group!(
    benches,
    bench_fastpath_prefix2,
    bench_short_extra_prefix10,
    bench_long_extra_prefix30,
    bench_max_extra_prefix39,
);
criterion_main!(benches);
