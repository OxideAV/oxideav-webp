//! Criterion bench — §4.3 `inverse_subtract_green`.
//!
//! `inverse_subtract_green` walks an ARGB buffer in place and adds the
//! green byte into both the red and the blue lanes (mod 256). It's the
//! last of the four §4.x transforms (`inverse_predictor`, §4.1;
//! `inverse_color`, §4.2; `inverse_subtract_green`, §4.3;
//! `inverse_color_indexing`, §4.4) and the only one that was still
//! un-benched after the round-170 / 180 / 194 / 207 / 210 progression.
//! This bench closes the inventory gap so any future SWAR / lane-
//! parallel pass on `inverse_subtract_green` has a baseline number to
//! A/B against, the same way the per-mode `inverse_predictor` bench
//! and the per-`size_bits` `inverse_color` bench enabled their
//! respective round-194 / round-207 optimizations.
//!
//! The §4.3 transform has no parameters — the operation is the same
//! regardless of image dimensions. We run one fixed-size bench on a
//! 256×256 deterministic LCG-filled ARGB buffer (the same shape used
//! by the §4.2 and §4.4 benches) so cross-pass numbers are comparable.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench inverse_subtract_green -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_transform::inverse_subtract_green;

const W: usize = 256;
const H: usize = 256;

fn build_pixels() -> Vec<u32> {
    // Deterministic LCG fill so the bench is reproducible across runs
    // and across hosts. Same constants as the §4.2 / §4.4 benches.
    let mut seed: u32 = 0x1357_9bdf;
    let n = W * H;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(seed);
    }
    v
}

fn bench_inverse_subtract_green(c: &mut Criterion) {
    let pixels = build_pixels();
    c.bench_function("inverse_subtract_green_256x256", |b| {
        b.iter(|| {
            let mut px = pixels.clone();
            inverse_subtract_green(black_box(&mut px));
            black_box(px);
        })
    });
}

criterion_group!(benches, bench_inverse_subtract_green);
criterion_main!(benches);
