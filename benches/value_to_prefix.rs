//! Criterion bench — encoder-side §5.2.2 `value_to_prefix` LZ77
//! length-or-distance value-to-prefix split, the exact inverse of the
//! decoder-side §3.6.2.2 `read_lz77_value` benched at round 252.
//!
//! `value_to_prefix` is the per-symbol first half of the §5.2.2
//! length / distance encode path: given a 1-based length-or-distance
//! `value` (≥ 1), it returns the `(prefix_code, extra_bits,
//! extra_value)` triple needed to emit the symbol. The §5.2.2 emit
//! sequence then writes the prefix code through the GREEN (length) or
//! DISTANCE (distance) Huffman code and the `extra_value` as
//! `extra_bits` raw LSB-first bits. The function is the strict
//! reverse of the §3.6.2.2 decoder formula:
//!
//! ```text
//! if prefix_code < 4 { value = prefix_code + 1 }
//! else {
//!     extra_bits = (prefix_code - 2) >> 1
//!     offset = (2 + (prefix_code & 1)) << extra_bits
//!     value = offset + extra_value + 1
//! }
//! ```
//!
//! Inside `encode_argb_literals` (`vp8l_encode.rs`) the function is
//! invoked **twice per emitted LZ77 match**: once for the length and
//! once for the §5.2.2 distance map's small-code output. Every
//! backward-reference run an encoder picks therefore costs at least
//! two `value_to_prefix` calls plus the surrounding §5.2.2 Huffman
//! emit. It is also called once more during cost estimation in the
//! `try_lz77_at` cost-model path (`vp8l_encode.rs` ≈ line 6488) when
//! the encoder is choosing between candidate match lengths — so for a
//! match-heavy encode the per-call count is multiples-of-three the
//! match count, not twice the match count. The per-call cost is
//! therefore the natural encoder-side next entry in the §5 encode
//! per-pass inventory, mirroring the decode-side round-252 §3.6.2.2
//! `read_lz77_value` bench.
//!
//! The §5.2.2 value range splits naturally into four regimes that
//! exercise different microarchitectural paths through the function,
//! mirroring the round-252 decoder-side cell layout so the two
//! benches' numbers are directly comparable cell-for-cell:
//!
//! * *Fast path* — `value ∈ [1, 4]` — the function returns `(value -
//!   1, 0, 0)` without touching the `leading_zeros` / shift /
//!   multiply chain. This is the only branch a length-1..=4 / close-
//!   neighbour-distance match ever exercises, and per §5.2.2 it
//!   covers prefix codes `0..=3` (one prefix code per value). Sampled
//!   with `value = 3` (midpoint, mirrors the decode-side
//!   `prefix_code = 2` cell which decodes to value 3).
//! * *Short extra* — `value ∈ [5, 64]` — `extra_bits` runs `1..=4`,
//!   so the `leading_zeros` is hit but the per-call work stays small.
//!   This is the dominant regime for typical natural-image LZ77 runs
//!   (lengths up to ~64 and the close-neighbour distances in the
//!   §5.2.2 distance map). Sampled with `value = 40` (`extra_bits =
//!   4`, prefix 10, mirrors the decode-side `prefix_code = 10` cell
//!   which decodes to values `34..=49`).
//! * *Long extra* — `value ∈ [65, 65536]` — `extra_bits` runs `5..=14`.
//!   The `leading_zeros` is hit hard and the §5.2.2 offset formula's
//!   `<< extra_bits` reaches 14. This regime is distance-only (the
//!   §5.2.2 note caps length prefixes at 24, so only DISTANCE ever
//!   reaches `prefix_code = 30`). Sampled with `value = 40000`
//!   (`extra_bits = 14`, prefix 30, mirrors the decode-side
//!   `prefix_code = 30` cell which decodes to values `32769..=49152`).
//! * *Max extra* — `value ∈ [524289, 1048576]` — `extra_bits = 18`,
//!   the §5.2.2 maximum and the §3.6.2.2 hard upper bound. Sampled
//!   with `value = 900_000` (prefix 39, mirrors the decode-side
//!   `prefix_code = 39` cell which decodes to values
//!   `786433..=1048576`).
//!
//! The per-call work is tiny — a few bit-shifts plus an integer
//! multiply plus a `u32::leading_zeros` — so the bench amortises
//! Criterion's per-iteration overhead by running an inner loop of
//! 1024 calls per `b.iter` body over a pre-filled `u32` value stream.
//! Every value in the stream is the cell's representative value, so
//! the inner-loop body exercises one §5.2.2 regime at a time. The
//! returned tuple is XOR-accumulated into a single `u32` so the
//! optimiser cannot drop any individual call, and `black_box` on both
//! the input value and the accumulator guards against constant-
//! folding and dead-store elimination of the inner work. The per-
//! iteration value count and the inner loop body are identical across
//! every cell, so cross-cell deltas come exclusively from the §5.2.2
//! body cost at the cell's `value`.
//!
//! A future rewrite (branchless prefix encode, table-driven `(prefix,
//! extra_bits, offset)` lookup for the small-value range, or §5.2.2-
//! table inlining via const arrays indexed by `value.leading_zeros`)
//! has direct A/B numbers at every representative regime here.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench value_to_prefix -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::value_to_prefix;

/// Per-`b.iter` inner-loop length. Sized to amortise Criterion's
/// per-iteration overhead well past the few-shifts-plus-multiply-plus-
/// `leading_zeros` per-call cost — at 1024 calls the inner body
/// dominates the timing even at the smallest §5.2.2 `value`.
const INNER_CALLS: usize = 1024;

/// Bench `value_to_prefix(value)` over an inner loop of `INNER_CALLS`
/// repetitions of the cell's representative `value`. The XOR
/// accumulator on the returned tuple keeps the optimiser from
/// hoisting a single call out of the inner loop, and `black_box` on
/// the input value blocks constant-folding the entire body.
fn bench_value(c: &mut Criterion, value: u32, label: &str) {
    let name = format!("value_to_prefix_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            // XOR-accumulate every returned `(prefix, extra_bits,
            // extra_value)` triple so the optimiser cannot hoist a
            // single call out of the inner loop. Mix the three fields
            // through a wrapping XOR so each lane of the returned
            // tuple has to be materialised — folding only `prefix`
            // would let the compiler discard the `extra_bits` /
            // `extra_value` computation. `black_box` on the input
            // value blocks constant-folding the entire body, and
            // `black_box` on the accumulator at the end blocks dead-
            // store elimination of the loop body.
            let mut acc: u32 = 0;
            for _ in 0..INNER_CALLS {
                let (p, e, x) = value_to_prefix(black_box(value));
                acc ^= p ^ e ^ x;
            }
            black_box(acc);
        })
    });
}

// Fast path: `value ∈ [1, 4]` ⇒ returns `(value - 1, 0, 0)` without
// touching the `leading_zeros` / shift chain. Sampled with `value =
// 3` (midpoint). Mirrors the decode-side round-252 `prefix_code = 2`
// fast-path cell which decodes to value 3 — the two cells are
// directly comparable.
fn bench_fastpath_value3(c: &mut Criterion) {
    bench_value(c, 3, "fastpath_value3");
}

// Short extra: `value = 40` ⇒ `extra_bits = 4`, `prefix_code = 10`.
// Mirrors the decode-side round-252 `prefix_code = 10` cell which
// decodes to values `34..=49`. Typical natural-image LZ77 length /
// close-neighbour distance regime.
fn bench_short_extra_value40(c: &mut Criterion) {
    bench_value(c, 40, "short_extra_value40");
}

// Long extra: `value = 40_000` ⇒ `extra_bits = 14`, `prefix_code =
// 30`. Mirrors the decode-side round-252 `prefix_code = 30` cell
// which decodes to values `32769..=49152`. Distance-only regime (the
// §5.2.2 note caps length prefixes at 24).
fn bench_long_extra_value40000(c: &mut Criterion) {
    bench_value(c, 40_000, "long_extra_value40000");
}

// Max extra: `value = 900_000` ⇒ `extra_bits = 18`, `prefix_code =
// 39`, the §5.2.2 maximum extra-bits count. Mirrors the decode-side
// round-252 `prefix_code = 39` cell which decodes to values
// `786433..=1048576`. Distance-only regime and the §5.2.2 hard upper
// bound on encodable values.
fn bench_max_extra_value900000(c: &mut Criterion) {
    bench_value(c, 900_000, "max_extra_value900000");
}

criterion_group!(
    benches,
    bench_fastpath_value3,
    bench_short_extra_value40,
    bench_long_extra_value40000,
    bench_max_extra_value900000,
);
criterion_main!(benches);
