//! Criterion bench — encoder-side `predictor_subtract` over a
//! 256×256 ARGB buffer.
//!
//! `predictor_subtract` is the per-channel mod-256 residual builder
//! that mirrors the decoder's §4.1 `add_pred`. It runs for every
//! pixel of every candidate predictor mode during the encoder's mode
//! chooser sweep, so the round-170 encode-side profile attributed its
//! call site (the predictor + residual path) to the #1 self-time slot.
//!
//! This bench feeds a deterministic LCG-filled ARGB buffer and a
//! second LCG-filled prediction buffer through `predictor_subtract`
//! once per pixel — 65 536 calls per iteration — and reports the
//! aggregate residual XOR so the optimizer cannot fold the loop body
//! away. The buffer shape matches the §4.x decoder-side benches so
//! per-pass numbers across the four §4.1 / §4.2 / §4.3 / §4.4
//! decoder transforms and the encoder-side `predictor_subtract` are
//! visually comparable.
//!
//! Closes the per-pass bench inventory gap for the encoder-side
//! residual builder (mirror of the decoder's `add_pred`, which has
//! been on the inverse-transform bench shelf since round 217 closed
//! `inverse_subtract_green` coverage). Round-224 measurement on the
//! current closure-of-four body: ~34 µs. A biased-SWAR rewrite was
//! tried this round and measured +18.4% — see the function's doc-
//! comment for the lane-bias underflow-prevention details and the
//! reason it regresses on AArch64 NEON auto-vectorisation. This
//! bench is the A/B reference any future `std::simd` rewrite of
//! `predictor_subtract` (mirroring the `to_rgba_simd` precedent
//! under the `simd` feature) must measure against.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench predictor_subtract -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::predictor_subtract;

const W: usize = 256;
const H: usize = 256;

fn lcg_buffer(seed_init: u32) -> Vec<u32> {
    // Deterministic LCG fill (same constants as the §4.x decoder
    // benches) so the bench is reproducible across runs and hosts.
    let mut seed: u32 = seed_init;
    let n = W * H;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(seed);
    }
    v
}

fn bench_predictor_subtract(c: &mut Criterion) {
    let original = lcg_buffer(0x1357_9bdf);
    let pred = lcg_buffer(0x2468_ace0);
    c.bench_function("predictor_subtract_256x256", |b| {
        b.iter(|| {
            // Aggregate XOR over per-pixel residuals so the loop body
            // has an observable output and is not optimized away.
            let mut acc: u32 = 0;
            for (o, p) in original.iter().zip(pred.iter()) {
                acc ^= predictor_subtract(black_box(*o), black_box(*p));
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_predictor_subtract);
criterion_main!(benches);
