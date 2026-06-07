//! Criterion bench — §4.4 `inverse_color_table` across a range of
//! palette sizes.
//!
//! `inverse_color_table` is the palette subtraction-decode pass
//! described under §4.4 of the in-tree clean-room §-numbering (the
//! "Color Indexing Transform" section): the color table is stored
//! subtraction-coded, and every final color is recovered by adding
//! the previous color's components into the current entry's
//! components, mod 256, on each of A / R / G / B.
//!
//! That makes this pass a tight `len - 1` iteration sequential
//! dependency over the palette array — palette `[i]` cannot be
//! recovered until palette `[i - 1]` has been recovered — and the
//! palette length spans 1..=256 per the spec (the size byte holds
//! `color_table_size - 1`).
//!
//! The inverse counterpart of this pass — `inverse_color_indexing`,
//! which expands the per-pixel index lookup into final ARGB pixels
//! — has had a per-bundle-tier bench since round 210. The §4.4
//! per-palette subtraction-decode pass itself was the last
//! `pub fn` in `vp8l_transform` that had no per-pass bench. This
//! bench closes that inventory gap so any future SWAR / lane-
//! parallel rewrite (the per-lane wrapping-add is a natural target
//! for byte-wise SIMD) has an A/B reference number to measure
//! against.
//!
//! The palette is filled with a deterministic LCG so every per-lane
//! wrap path is exercised. We run three sizes covering the bundling
//! tiers the spec defines under §4.4:
//!
//! * `palette_2`   → `width_bits = 3` tier (smallest palette length)
//! * `palette_16`  → boundary between `width_bits = 1` and `0` tiers
//! * `palette_256` → `width_bits = 0` tier (maximum palette length)
//!
//! The §4.4 subtraction-decode itself walks every entry the same way
//! regardless of which bundling tier the indexing pass will later
//! use — the per-tier widths above are sampled here only as a
//! representative cross-section of palette lengths.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench inverse_color_table -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_transform::inverse_color_table;

fn build_palette(size: usize) -> Vec<u32> {
    // Deterministic LCG fill (same constants the §4.x decoder benches
    // and the encoder-side `predictor_subtract` / `apply_subtract_green`
    // benches use) so this bench's numbers are reproducible across runs
    // and cross-pass comparable to the rest of the §4.x inventory.
    let mut seed: u32 = 0x1357_9bdf;
    let mut v = Vec::with_capacity(size);
    for _ in 0..size {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(seed);
    }
    v
}

fn bench_palette(c: &mut Criterion, palette_size: usize) {
    let palette = build_palette(palette_size);
    let label = format!("inverse_color_table_palette{}", palette_size);
    c.bench_function(&label, |b| {
        b.iter(|| {
            // Clone into a fresh working buffer each iteration so the
            // in-place subtraction-decode pass starts from the same
            // input every time and a future SWAR rewrite cannot win
            // simply by caching deltas across iterations.
            let mut t = palette.clone();
            inverse_color_table(black_box(&mut t));
            black_box(t);
        })
    });
}

fn bench_palette_2(c: &mut Criterion) {
    bench_palette(c, 2);
}
fn bench_palette_16(c: &mut Criterion) {
    bench_palette(c, 16);
}
fn bench_palette_256(c: &mut Criterion) {
    bench_palette(c, 256);
}

criterion_group!(
    benches,
    bench_palette_2,
    bench_palette_16,
    bench_palette_256
);
criterion_main!(benches);
