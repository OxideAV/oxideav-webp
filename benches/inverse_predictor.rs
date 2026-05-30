//! Criterion bench — §4.1 `inverse_predictor` with a worst-case mode
//! mix.
//!
//! The existing `lossless_decode` bench feeds a 256×256 gradient that
//! the encoder's mode chooser typically resolves with subtract-green +
//! the lightweight predictor modes 0–4. Predictor modes 12 / 13 (which
//! call `ClampAddSubtractFull` / `ClampAddSubtractHalf` — the §4.1
//! arithmetic-heavy predictors) are rarely exercised by that fixture.
//!
//! This bench builds a 256×256 ARGB buffer plus a predictor image that
//! forces **mode 12 everywhere** (`size_bits = 0`, one predictor per
//! pixel, all modes = 12) so the inner-loop spends as much time as
//! possible in the `clamp_add_subtract_full` SWAR path the round-194
//! rewrite targets.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench inverse_predictor -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_transform::inverse_predictor;

const W: u32 = 256;
const H: u32 = 256;

fn build_residuals() -> Vec<u32> {
    // Deterministic LCG fill so the bench is reproducible and the
    // residuals cover a wide range of per-channel saturation cases
    // (i.e. `clamp_add_subtract_full` actually hits the clamp on
    // some lanes, not just the no-op pass-through interior).
    let mut seed: u32 = 0x1234_5678;
    let n = (W as usize) * (H as usize);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(seed);
    }
    v
}

fn predictor_image(mode: u8) -> Vec<u32> {
    // size_bits = 0 → 1 predictor per pixel → predictor image is W×H.
    // The mode lives in the green channel.
    let g = (mode as u32) << 8;
    vec![g; (W as usize) * (H as usize)]
}

fn bench_mode_12(c: &mut Criterion) {
    let residuals = build_residuals();
    let pred = predictor_image(12);
    c.bench_function("inverse_predictor_mode12_256x256", |b| {
        b.iter(|| {
            let mut px = residuals.clone();
            inverse_predictor(black_box(&mut px), W, H, black_box(&pred), W, 0);
            black_box(px);
        })
    });
}

fn bench_mode_13(c: &mut Criterion) {
    let residuals = build_residuals();
    let pred = predictor_image(13);
    c.bench_function("inverse_predictor_mode13_256x256", |b| {
        b.iter(|| {
            let mut px = residuals.clone();
            inverse_predictor(black_box(&mut px), W, H, black_box(&pred), W, 0);
            black_box(px);
        })
    });
}

fn bench_mode_11(c: &mut Criterion) {
    // Predictor mode 11 → `Select(L, T, TL)`. Round 194 simplified the
    // body from "build a 4-channel `estimate` then take 8 per-channel
    // abs differences" to "take 4 per-channel abs differences twice
    // (Manhattan(t, tl) and Manhattan(l, tl))" after observing the
    // `estimate` term cancels in the per-channel subtractions.
    let residuals = build_residuals();
    let pred = predictor_image(11);
    c.bench_function("inverse_predictor_mode11_256x256", |b| {
        b.iter(|| {
            let mut px = residuals.clone();
            inverse_predictor(black_box(&mut px), W, H, black_box(&pred), W, 0);
            black_box(px);
        })
    });
}

criterion_group!(benches, bench_mode_11, bench_mode_12, bench_mode_13);
criterion_main!(benches);
