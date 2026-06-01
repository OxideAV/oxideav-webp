//! Criterion bench — §4.2 `inverse_color` across a range of `size_bits`.
//!
//! `inverse_color` walks a 256×256 ARGB buffer and, for every pixel,
//! looks up its block's `ColorTransformElement` from a sub-resolution
//! color image and applies three per-channel signed deltas:
//!
//! * `new_red  = red  + delta(green_to_red,  green)`
//! * `new_blue = blue + delta(green_to_blue, green) + delta(red_to_blue, new_red)`
//!
//! The per-block CTE is constant inside each `1 << size_bits` block, so
//! the extract / mask work is hoistable out of the inner pixel loop.
//! This bench parameterises `size_bits` ∈ {0, 3, 5, 7} so a future
//! hoist-the-CTE rewrite can be measured against the per-pixel form.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench inverse_color -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_transform::inverse_color;

const W: u32 = 256;
const H: u32 = 256;

fn build_pixels() -> Vec<u32> {
    // Deterministic LCG fill so the bench is reproducible.
    let mut seed: u32 = 0x1357_9bdf;
    let n = (W as usize) * (H as usize);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(seed);
    }
    v
}

fn color_image(size_bits: u8) -> (Vec<u32>, u32) {
    // §4.2 sub-resolution color image: width = ceil(W / (1 << size_bits)).
    let block = 1u32 << size_bits;
    let tw = W.div_ceil(block);
    let th = H.div_ceil(block);
    // Per §4.2 pixel encoding: red = red_to_blue, green = green_to_blue,
    // blue = green_to_red. Fill with a varying CTE so the inner-loop
    // arithmetic actually exercises the signed-delta path.
    let n = (tw as usize) * (th as usize);
    let mut v = Vec::with_capacity(n);
    let mut seed: u32 = 0xabcd_ef01;
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        // Keep alpha at 0xff so the CTE word has a valid layout; the
        // r/g/b bytes are the three signed delta coefficients.
        v.push(0xff00_0000 | (seed & 0x00ff_ffff));
    }
    (v, tw)
}

fn bench_size_bits(c: &mut Criterion, size_bits: u8) {
    let pixels = build_pixels();
    let (cimg, tw) = color_image(size_bits);
    let label = format!("inverse_color_256x256_sb{}", size_bits);
    c.bench_function(&label, |b| {
        b.iter(|| {
            let mut px = pixels.clone();
            inverse_color(black_box(&mut px), W, H, black_box(&cimg), tw, size_bits);
            black_box(px);
        })
    });
}

fn bench_sb0(c: &mut Criterion) {
    bench_size_bits(c, 0);
}
fn bench_sb3(c: &mut Criterion) {
    bench_size_bits(c, 3);
}
fn bench_sb5(c: &mut Criterion) {
    bench_size_bits(c, 5);
}
fn bench_sb7(c: &mut Criterion) {
    bench_size_bits(c, 7);
}

criterion_group!(benches, bench_sb0, bench_sb3, bench_sb5, bench_sb7);
criterion_main!(benches);
