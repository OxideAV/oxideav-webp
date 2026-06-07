//! Criterion bench — encoder-side §4.3 `apply_subtract_green` over a
//! 256×256 ARGB buffer.
//!
//! `apply_subtract_green` is the forward §4.3 transform: it walks an
//! in-memory ARGB buffer and replaces every red / blue lane with the
//! mod-256 difference against that pixel's green lane (alpha and green
//! pass through untouched). The decoder's mirror is
//! `vp8l_transform::inverse_subtract_green`, which adds the green
//! channel back into red and blue mod 256. A roundtrip through both
//! is byte-exact (asserted by
//! `vp8l_encode::tests::apply_subtract_green_is_inverse_of_inverse_subtract_green`).
//!
//! The decoder-side mirror has had a per-pass criterion bench since
//! round 217 (`benches/inverse_subtract_green.rs`); this bench closes
//! the matching encoder-side inventory gap so any future SWAR /
//! `std::simd` rewrite of the forward §4.3 pass has the same kind of
//! A/B reference the round-217 `inverse_subtract_green` bench gave
//! the inverse pass.
//!
//! The §4.3 transform has no parameters — the operation is the same
//! regardless of image dimensions. We run one fixed-size bench on a
//! 256×256 deterministic LCG-filled ARGB buffer (identical seed +
//! constants to the §4.x decoder-side benches and the round-224
//! `predictor_subtract` bench) so cross-pass numbers are directly
//! comparable.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench apply_subtract_green -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::apply_subtract_green;

const W: usize = 256;
const H: usize = 256;

fn build_pixels() -> Vec<u32> {
    // Deterministic LCG fill (same constants as the §4.x decoder
    // benches and the encoder-side `predictor_subtract` bench) so
    // numbers are reproducible across runs and hosts and directly
    // comparable across the §4.1 / §4.2 / §4.3 / §4.4 inventory.
    let mut seed: u32 = 0x1357_9bdf;
    let n = W * H;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(seed);
    }
    v
}

fn bench_apply_subtract_green(c: &mut Criterion) {
    let pixels = build_pixels();
    c.bench_function("apply_subtract_green_256x256", |b| {
        b.iter(|| {
            // Clone into a fresh working buffer each iteration so the
            // in-place pass starts from the same input every time and
            // a SWAR rewrite cannot win simply by caching residuals
            // across iterations.
            let mut px = pixels.clone();
            apply_subtract_green(black_box(&mut px));
            black_box(px);
        })
    });
}

criterion_group!(benches, bench_apply_subtract_green);
criterion_main!(benches);
