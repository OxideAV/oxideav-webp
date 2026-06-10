//! Criterion bench — decoder-side §6.2.1 `PrefixCode::from_code_lengths`
//! canonical-table build across the §3.7.1 prefix-code-group alphabets.
//!
//! `from_code_lengths` is the decoder mirror of the encoder's §3.7.2
//! length-then-code pair sampled by `benches/build_code_lengths.rs`
//! (round 250) and `benches/canonical_codes.rs` (round 251): given the
//! per-symbol code lengths recovered from the bitstream, it counts
//! symbols per length, validates the §6.2.1 Kraft completeness rule
//! (`sum 2^-len == 1`, with the single-leaf-node exception), assigns
//! canonical code values in `(length, value)` order, and materialises
//! the per-length decode rows that `read_symbol` walks.
//!
//! Like its encoder mirrors, it runs once per prefix code per §6.2
//! prefix code group (GREEN + length, RED, BLUE, ALPHA, DISTANCE) plus
//! once for the inner code-length code of every normal-form §6.2.1
//! read, so its self-time scales with the per-meta prefix code-group
//! count of the decoded image. The round-170 decode profile attributed
//! rank 4 of decode self-time (~2%) to this exact symbol — the last
//! pass in the §3.7.2 / §6.2.1 length-then-code chain that still had
//! no per-pass bench.
//!
//! The §3.7.1 prefix-code-group alphabet sizes are the same four
//! sampled by the round-250 / round-251 encoder benches:
//!
//! * RED / BLUE / ALPHA — 256 (8-bit channel literals).
//! * DISTANCE — 40 (§3.6.2.2 distance prefix codes).
//! * GREEN — `256 + 24 + color_cache_size`. Endpoints:
//!   `color_cache_bits = 0` → 281 and `color_cache_bits = 11` → 2328.
//!
//! The implementation makes two full passes over the `code_lengths`
//! slice (the count/validate pass, then one rescan per *used* length
//! inside the assignment loop), so the expected scaling is
//! `O(used_lengths · N)` — between `canonical_codes`'s fixed
//! `O(MAX_CODE_LENGTH · N)` and a single-pass bucket sort's `O(N)`.
//! Dense tables use most of the 15 length slots; sparse tables use
//! only a handful, so unlike `canonical_codes` a moderate dense /
//! sparse asymmetry is expected here, and a future single-rescan
//! bucket-sort rewrite would show up on the dense side first.
//!
//! The length tables fed to `from_code_lengths` are the exact ones
//! `build_code_lengths` produces on the same dense / sparse frequency
//! inputs used in rounds 250 / 251 (same LCG constants), so each
//! `prefix_from_code_lengths_{dense,sparse}_{...}` cell is directly
//! comparable to its `build_code_lengths` / `canonical_codes`
//! counterpart. The `build_code_lengths` call is *outside* the
//! measured interval; so is the per-iteration `Vec` clone
//! (`iter_batched` setup), because `from_code_lengths` takes the
//! length table by value — only the canonical-table build itself is
//! sampled.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench prefix_from_code_lengths -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use oxideav_webp::vp8l_encode::build_code_lengths;
use oxideav_webp::vp8l_prefix::PrefixCode;

/// Deterministic LCG fill matching the constants used across the §3.x
/// / §4.x per-pass benches so the length tables here are byte-for-byte
/// the ones the round-250 / round-251 encoder benches operate on.
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

/// The §3.7.2 length table a real decode would hand to
/// `from_code_lengths` after reading the normal-form code-length
/// stream — produced here by the encoder-side builder on the dense
/// frequency regime, exactly once per bench cell, outside `b.iter`.
fn lengths_dense(alphabet: usize) -> Vec<u8> {
    let freqs = build_freqs_dense(alphabet);
    build_code_lengths(&freqs)
}

/// Same shape, sparse regime.
fn lengths_sparse(alphabet: usize) -> Vec<u8> {
    let freqs = build_freqs_sparse(alphabet);
    build_code_lengths(&freqs)
}

fn bench_lengths(c: &mut Criterion, lengths: Vec<u8>, name: &str) {
    c.bench_function(name, |b| {
        // `from_code_lengths` takes ownership of the length table, so
        // the per-iteration clone lives in `iter_batched` setup and
        // stays outside the measured interval.
        b.iter_batched(
            || lengths.clone(),
            |owned| {
                let code = PrefixCode::from_code_lengths(black_box(owned))
                    .expect("encoder-produced length tables are Kraft-complete");
                black_box(code);
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_dense(c: &mut Criterion, alphabet: usize, label: &str) {
    let name = format!("prefix_from_code_lengths_dense_{}", label);
    bench_lengths(c, lengths_dense(alphabet), &name);
}

fn bench_sparse(c: &mut Criterion, alphabet: usize, label: &str) {
    let name = format!("prefix_from_code_lengths_sparse_{}", label);
    bench_lengths(c, lengths_sparse(alphabet), &name);
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
