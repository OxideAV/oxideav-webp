//! Criterion bench — encoder-side §3.7.2 `build_code_lengths` Huffman
//! code-length builder across the §3.7.1 / §3.7.2 prefix-code-group
//! alphabets.
//!
//! `build_code_lengths` is the encoder's per-symbol code-length
//! assignment pass for a single prefix code: given a frequency table
//! over an alphabet, it returns the per-symbol Huffman lengths capped
//! at `MAX_CODE_LENGTH` with the §3.7.2 completeness invariant (Kraft
//! sum exactly 1). It is invoked five times per §3.7.1 prefix code
//! group (GREEN+length, RED, BLUE, ALPHA, DISTANCE) and again for the
//! §3.7.2.1.2 normal-form *code-length-of-code-lengths* sub-pass, so
//! its self-time scales linearly with the per-meta prefix code-group
//! count of an encoded image.
//!
//! The §3.7.1 prefix-code-group alphabet sizes that occur in the wild:
//!
//! * RED / BLUE / ALPHA — 256 (8-bit channel literals).
//! * DISTANCE — 40 (§3.6.2.2 distance prefix codes).
//! * GREEN — `256 + 24 + color_cache_size`, where `color_cache_size =
//!   1 << color_cache_bits` and `color_cache_bits` is in `0..=11` per
//!   §3.6.2.3. The two endpoints sampled here are `bits = 0` → 281
//!   (`color_cache_size = 1`) and `bits = 11` → 2328 (the largest GREEN
//!   alphabet the §3.6.2.3 color cache lets the encoder reach).
//!
//! The §3.7.2 builder cost has two regimes worth distinguishing in the
//! A/B reference set:
//!
//! 1. *Dense* frequency tables (every symbol used at least once) —
//!    realistic of a natural-image meta prefix code group where the
//!    literal channels are nearly always saturated. The heap-pop /
//!    heap-push loop runs `n - 1` times where `n` = alphabet size,
//!    and the dominant cost is the heap maintenance plus the post-pass
//!    leaf-depth walk.
//! 2. *Sparse* frequency tables (a small subset of symbols used,
//!    skewed Zipf-ish) — realistic of a DISTANCE table where only a
//!    handful of distance prefix codes ever fire, or a GREEN table
//!    where the §3.6.2.3 color cache code range is barely populated.
//!    The early-out paths (`used.len() == 0` / `1`) and the small-`n`
//!    heap loop dominate.
//!
//! This bench samples both regimes for each representative alphabet so
//! a future heap rewrite (`std::collections::BinaryHeap` vs the
//! hand-rolled sift loop, or a radix-bucket replacement for the heap
//! entirely) has direct A/B numbers at every realistic alphabet size.
//!
//! The §3.7.2 builder has had no per-pass bench since the encoder
//! shipped — the round-170 profile attribution at `BENCHMARKS.md`
//! ranked it on the encode hot-path through `canonical_codes` (rank 4,
//! ~40 samples), but the §3.7.2 length-build pass itself was sampled
//! only as part of surrounding closure-body work. This bench closes
//! the §3 entropy per-pass inventory gap that the §4.x transform
//! inventory has had since round 249.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench build_code_lengths -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::build_code_lengths;

/// Deterministic LCG fill matching the constants used across the §4.x
/// per-pass benches so a frequency stream here is reproducible across
/// runs and hosts and cross-bench comparable to the rest of the
/// inventory.
fn lcg_step(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed
}

/// Build a *dense* frequency table over `alphabet`: every symbol is
/// used with a frequency in `1..=255` drawn from the LCG. This models
/// the §3.7.2 dense-table regime (literal channels in a natural-image
/// meta prefix code group).
fn build_freqs_dense(alphabet: usize) -> Vec<u32> {
    let mut seed: u32 = 0x1357_9bdf;
    let mut freqs = Vec::with_capacity(alphabet);
    for _ in 0..alphabet {
        // 1..=255 frequency keeps every symbol live without letting any
        // one symbol's count dominate the heap's tie-breakers.
        let f = (lcg_step(&mut seed) % 255) + 1;
        freqs.push(f);
    }
    freqs
}

/// Build a *sparse* Zipf-ish frequency table over `alphabet`: only
/// `sqrt(alphabet)` symbols are used, with frequencies shaped 1/(k+1).
/// This models the §3.7.2 sparse-table regime (a DISTANCE table where
/// few prefix codes fire, or a GREEN table whose §3.6.2.3 color cache
/// range is barely populated).
fn build_freqs_sparse(alphabet: usize) -> Vec<u32> {
    let used = ((alphabet as f64).sqrt() as usize).max(2);
    let mut seed: u32 = 0x2468_ace0;
    let mut freqs = vec![0u32; alphabet];
    for k in 0..used {
        // Pick a slot pseudo-randomly so the live symbols are scattered
        // across the alphabet rather than packed in the prefix.
        let slot = (lcg_step(&mut seed) as usize) % alphabet;
        // 1/(k+1) shape, scaled so the first slot dominates without
        // pushing any code past MAX_CODE_LENGTH (the cap pass is
        // measured implicitly when it does fire).
        let f = (1024 / (k as u32 + 1)).max(1);
        freqs[slot] = freqs[slot].saturating_add(f);
    }
    freqs
}

fn bench_dense(c: &mut Criterion, alphabet: usize, label: &str) {
    let freqs = build_freqs_dense(alphabet);
    let name = format!("build_code_lengths_dense_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            // `build_code_lengths` takes `&[u32]` and allocates its
            // own working vectors per call, so no per-iteration buffer
            // hygiene is needed beyond `black_box`'ing the inputs and
            // the result to defeat any whole-call CSE.
            let lengths = build_code_lengths(black_box(&freqs));
            black_box(lengths);
        })
    });
}

fn bench_sparse(c: &mut Criterion, alphabet: usize, label: &str) {
    let freqs = build_freqs_sparse(alphabet);
    let name = format!("build_code_lengths_sparse_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let lengths = build_code_lengths(black_box(&freqs));
            black_box(lengths);
        })
    });
}

// Distance alphabet (§3.6.2.2): 40 symbols.
fn bench_dense_distance40(c: &mut Criterion) {
    bench_dense(c, 40, "distance40");
}
fn bench_sparse_distance40(c: &mut Criterion) {
    bench_sparse(c, 40, "distance40");
}

// 8-bit literal channels (RED / BLUE / ALPHA): 256 symbols.
fn bench_dense_literal256(c: &mut Criterion) {
    bench_dense(c, 256, "literal256");
}
fn bench_sparse_literal256(c: &mut Criterion) {
    bench_sparse(c, 256, "literal256");
}

// GREEN alphabet, smallest cache: 256 + 24 + 1 = 281 symbols.
// (`color_cache_bits = 0` ⇒ `color_cache_size = 1`.)
fn bench_dense_green281(c: &mut Criterion) {
    bench_dense(c, 281, "green281");
}
fn bench_sparse_green281(c: &mut Criterion) {
    bench_sparse(c, 281, "green281");
}

// GREEN alphabet, largest cache (`color_cache_bits = 11`):
// 256 + 24 + 2048 = 2328 symbols.
fn bench_dense_green2328(c: &mut Criterion) {
    bench_dense(c, 2328, "green2328");
}
fn bench_sparse_green2328(c: &mut Criterion) {
    bench_sparse(c, 2328, "green2328");
}

criterion_group!(
    benches,
    bench_dense_distance40,
    bench_sparse_distance40,
    bench_dense_literal256,
    bench_sparse_literal256,
    bench_dense_green281,
    bench_sparse_green281,
    bench_dense_green2328,
    bench_sparse_green2328,
);
criterion_main!(benches);
