//! Criterion bench — encoder-side §3.7.2 `canonical_codes` canonical-
//! code-value assignment pass across the §3.7.1 prefix-code-group
//! alphabets.
//!
//! `canonical_codes` is the second per-symbol pass in the §3.7.2
//! length-then-code Huffman build: given the per-symbol code lengths
//! produced by `build_code_lengths`, it returns the canonical code
//! values that the decoder's [`crate::vp8l_prefix::PrefixCode`]
//! reconstructs from those same lengths (DEFLATE canonical rule:
//! symbols ordered by `(length, value)`, codes assigned sequentially,
//! read most-significant-bit-first within a code).
//!
//! Just like `build_code_lengths`, `canonical_codes` is invoked five
//! times per §3.7.1 prefix code group (GREEN + length, RED, BLUE,
//! ALPHA, DISTANCE) and again for the §3.7.2.1.2 normal-form
//! *code-length-of-code-lengths* sub-pass, so its self-time scales
//! linearly with the per-meta prefix code-group count of an encoded
//! image. The round-170 encoder profile attributed rank 4 of self-time
//! to the surrounding closure body *through* `canonical_codes`, so the
//! per-pass cost of this exact symbol is one of the two natural
//! optimization targets in the §3 entropy domain (the other being
//! `build_code_lengths`, sampled by `benches/build_code_lengths.rs`
//! as of round 250).
//!
//! The §3.7.1 prefix-code-group alphabet sizes are the same four
//! sampled by `benches/build_code_lengths.rs`:
//!
//! * RED / BLUE / ALPHA — 256 (8-bit channel literals).
//! * DISTANCE — 40 (§3.6.2.2 distance prefix codes).
//! * GREEN — `256 + 24 + color_cache_size`. Endpoints:
//!   `color_cache_bits = 0` → 281 and `color_cache_bits = 11` → 2328.
//!
//! The implementation walks `1..=MAX_CODE_LENGTH` outer and the full
//! `lengths` slice inner — an explicit `O(MAX_CODE_LENGTH · N)` pass
//! that ignores the active-symbol count. So unlike the §3.7.2 length
//! builder, the dense / sparse split here is expected to be *small*
//! (the inner loop runs over every slot regardless of whether the
//! length is zero), and the four alphabet sizes are the dominant axis.
//! This bench samples both regimes anyway, so a future rewrite (e.g.
//! a single-pass bucket sort by length that skips zero-length symbols)
//! has direct A/B numbers at every realistic alphabet size *and* gets
//! credit for any dense / sparse asymmetry it introduces.
//!
//! The length tables fed to `canonical_codes` are the exact ones
//! `build_code_lengths` would produce on the same dense / sparse
//! frequency inputs used in `benches/build_code_lengths.rs`, so the
//! `canonical_codes_{dense,sparse}_{...}` numbers cell-for-cell match
//! a real per-prefix-code-group call in the encode pipeline. The
//! `build_code_lengths` call is *outside* the measured interval — only
//! the `canonical_codes` self-time is sampled.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench canonical_codes -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::{build_code_lengths, canonical_codes};

/// Deterministic LCG fill matching the constants used across the §3.x
/// / §4.x per-pass benches so the length tables here are reproducible
/// across runs and hosts and cross-bench comparable to the rest of the
/// inventory.
fn lcg_step(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed
}

/// Build a *dense* frequency table over `alphabet`: every symbol is
/// used with a frequency in `1..=255` drawn from the LCG. Matches the
/// dense path in `benches/build_code_lengths.rs`.
fn build_freqs_dense(alphabet: usize) -> Vec<u32> {
    let mut seed: u32 = 0x1357_9bdf;
    let mut freqs = Vec::with_capacity(alphabet);
    for _ in 0..alphabet {
        let f = (lcg_step(&mut seed) % 255) + 1;
        freqs.push(f);
    }
    freqs
}

/// Build a *sparse* Zipf-ish frequency table over `alphabet`: only
/// `sqrt(alphabet)` symbols are used, with frequencies shaped
/// `1/(k+1)`. Matches the sparse path in
/// `benches/build_code_lengths.rs`.
fn build_freqs_sparse(alphabet: usize) -> Vec<u32> {
    let used = ((alphabet as f64).sqrt() as usize).max(2);
    let mut seed: u32 = 0x2468_ace0;
    let mut freqs = vec![0u32; alphabet];
    for k in 0..used {
        let slot = (lcg_step(&mut seed) as usize) % alphabet;
        let f = (1024 / (k as u32 + 1)).max(1);
        freqs[slot] = freqs[slot].saturating_add(f);
    }
    freqs
}

/// Build the §3.7.2 length table the encoder would feed to
/// `canonical_codes` after the dense-regime `build_code_lengths`
/// call. The build is outside `b.iter` and runs exactly once per
/// bench cell.
fn lengths_dense(alphabet: usize) -> Vec<u8> {
    let freqs = build_freqs_dense(alphabet);
    build_code_lengths(&freqs)
}

/// Same shape, sparse regime.
fn lengths_sparse(alphabet: usize) -> Vec<u8> {
    let freqs = build_freqs_sparse(alphabet);
    build_code_lengths(&freqs)
}

fn bench_dense(c: &mut Criterion, alphabet: usize, label: &str) {
    let lengths = lengths_dense(alphabet);
    let name = format!("canonical_codes_dense_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            // `canonical_codes` allocates its own working / return
            // vectors per call, so no per-iteration buffer hygiene is
            // needed beyond `black_box`'ing the inputs and the result
            // to defeat any whole-call CSE.
            let codes = canonical_codes(black_box(&lengths));
            black_box(codes);
        })
    });
}

fn bench_sparse(c: &mut Criterion, alphabet: usize, label: &str) {
    let lengths = lengths_sparse(alphabet);
    let name = format!("canonical_codes_sparse_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let codes = canonical_codes(black_box(&lengths));
            black_box(codes);
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
