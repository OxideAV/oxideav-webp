//! Criterion bench — decoder-side §3.6.2.3 `ColorCache::hash` color-cache
//! multiplicative-hash slot index across the §3.6.2.3 `code_bits`
//! `[1..11]` allowed range.
//!
//! `ColorCache::hash` is the per-pixel slot-index function that turns an
//! emitted ARGB color into the §3.6.2.3 color-cache array index. RFC
//! 9649 §3.6.2.3 specifies it verbatim:
//!
//! ```text
//! (0x1e35a7bd * color) >> (32 - color_cache_code_bits)
//! ```
//!
//! Inside the §3.6.2 per-pixel decode loop (`vp8l_decode.rs`) the
//! function is called twice per emitted pixel when the §3.6.2.3 cache
//! is enabled: once by `ColorCache::insert` (every literal / LZ77-copied
//! / cache-resolved pixel is inserted into the cache in stream order
//! per §3.6.2.3 "the state of the color cache is maintained by
//! inserting every pixel ... into the cache in the order they appear
//! in the stream"), and indirectly once more inside the encoder mirror
//! `vp8l_encode::EncoderColorCache::hash` during encode. Per-call cost
//! therefore scales linearly with the per-image pixel count whenever
//! the §3.6.2.3 cache is enabled, which is the common case for natural
//! images. The round-170 profile attributed a non-trivial share of the
//! per-pixel decode self-time to this surrounding insert path, so the
//! per-call cost is a natural next entry in the §3 decode per-pass
//! inventory alongside the round-252 §3.6.2.2 `read_lz77_value` bench
//! and the round-250 / round-251 §3.7.2 encoder-builder benches.
//!
//! The §3.6.2.3 `code_bits` range is `[1..11]`. The right-shift in the
//! hash formula is `32 - code_bits`, so as `code_bits` grows the
//! resulting slot index occupies more bits and the cache array grows
//! from 2 entries (`code_bits = 1`) up to 2048 entries (`code_bits =
//! 11`). The per-call work is identical at every `code_bits` value — a
//! `u32` multiply plus a right-shift plus a `usize` cast — so the
//! bench expectation is a flat per-call cost across the four sampled
//! cells. The deliverable is the A/B reference for a future const-
//! folding / inlining rewrite at the call site, where `code_bits` is
//! often known statically per decode group; an A/B that reveals an
//! unexpected per-`code_bits` slope here would surface a missing
//! optimization opportunity in the surrounding decode loop.
//!
//! The four sampled cells cover the `[1..11]` range at its endpoints
//! and two interior points:
//!
//! * `code_bits = 1` — minimum allowed (`>> 31`, 2-slot cache).
//! * `code_bits = 4` — the typical small-cache regime, common in
//!   tightly-bounded palette / line-art image data.
//! * `code_bits = 8` — the natural-image regime where the cache holds
//!   256 recently-seen ARGB values.
//! * `code_bits = 11` — maximum allowed (`>> 21`, 2048-slot cache),
//!   the §3.6.2.3 hard upper bound.
//!
//! The per-call work is tiny — a single multiply plus a shift — so the
//! bench amortizes Criterion's per-iteration overhead by running an
//! inner loop of 1024 calls per `b.iter` body over a pre-allocated
//! deterministic LCG-filled `u32` ARGB stream. The inner loop accumulates
//! every slot index through a wrapping XOR so the optimiser cannot
//! drop any individual call. The ARGB input stream and the loop count
//! are identical across every cell, so cross-cell deltas come
//! exclusively from the `hash` body cost at the cell's `code_bits`.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench color_cache_hash -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_decode::ColorCache;

/// Deterministic LCG fill matching the constants used across the §3.x
/// / §4.x per-pass benches so the ARGB input stream here is
/// reproducible across runs and hosts and cross-bench comparable to
/// the rest of the inventory.
fn lcg_step(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed
}

/// Pre-allocate `len` deterministic LCG-generated ARGB samples. The
/// LCG matches the rest of the §3.x / §4.x per-pass bench inventory.
fn lcg_argb(len: usize) -> Vec<u32> {
    let mut seed: u32 = 0x9e37_79b9;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(lcg_step(&mut seed));
    }
    out
}

/// Per-`b.iter` inner-loop length. Sized to amortise Criterion's
/// per-iteration overhead well past the single-multiply-plus-shift
/// per-call cost — at 1024 calls the inner body dominates the timing
/// even at the smallest §3.6.2.3 `code_bits`.
const INNER_CALLS: usize = 1024;

/// Bench `ColorCache::hash(argb)` over a fresh ARGB stream each
/// iteration, with an inner loop of `INNER_CALLS` calls per iteration.
/// The XOR accumulator keeps the optimiser from dropping any single
/// call.
fn bench_code_bits(c: &mut Criterion, code_bits: u32, label: &str) {
    let cache = ColorCache::new(code_bits);
    let argbs = lcg_argb(INNER_CALLS);
    let name = format!("color_cache_hash_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            // XOR-accumulate every slot index so the optimiser cannot
            // hoist a single call out of the inner loop. The
            // accumulator type matches `ColorCache::hash`'s return
            // (`usize`); `black_box` on both inputs and output guards
            // against constant-folding and dead-store elimination of
            // the inner work.
            let mut acc: usize = 0;
            for &argb in argbs.iter() {
                let slot = black_box(&cache).hash(black_box(argb));
                acc ^= slot;
            }
            black_box(acc);
        })
    });
}

// Minimum allowed `code_bits` per §3.6.2.3: 1. The hash right-shifts
// by 31, producing a 2-slot index in `{0, 1}`.
fn bench_code_bits_1(c: &mut Criterion) {
    bench_code_bits(c, 1, "code_bits_1");
}

// Small-cache regime: 16-slot cache. Common in tightly-bounded
// palette / line-art image data.
fn bench_code_bits_4(c: &mut Criterion) {
    bench_code_bits(c, 4, "code_bits_4");
}

// Natural-image regime: 256-slot cache. Models a §3.6.2.3 cache
// holding the 256 most-recently-seen ARGB values.
fn bench_code_bits_8(c: &mut Criterion) {
    bench_code_bits(c, 8, "code_bits_8");
}

// Maximum allowed `code_bits` per §3.6.2.3: 11. The hash right-shifts
// by 21, producing a 2048-slot index and the §3.6.2.3 hard upper
// bound on cache size.
fn bench_code_bits_11(c: &mut Criterion) {
    bench_code_bits(c, 11, "code_bits_11");
}

criterion_group!(
    benches,
    bench_code_bits_1,
    bench_code_bits_4,
    bench_code_bits_8,
    bench_code_bits_11,
);
criterion_main!(benches);
