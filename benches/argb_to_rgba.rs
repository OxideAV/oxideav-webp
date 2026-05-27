//! Criterion bench — [`oxideav_webp::vp8l::Vp8lImage::to_rgba`]
//! ARGB-`u32` → packed `[R, G, B, A]` byte repack.
//!
//! Constructs a 256×256 `Vp8lImage` and measures `to_rgba` only,
//! isolating the per-pixel byte shuffle from any decode work. This
//! is the cleanest target for a SIMD shuffle rewrite.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench argb_to_rgba -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l::Vp8lImage;

fn build_image_256() -> Vp8lImage {
    let (w, h) = (256u32, 256u32);
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let a = 0xffu32;
            let r = x & 0xff;
            let g = y & 0xff;
            let b = (x ^ y) & 0xff;
            pixels.push((a << 24) | (r << 16) | (g << 8) | b);
        }
    }
    Vp8lImage {
        width: w,
        height: h,
        pixels,
        has_alpha: false,
    }
}

fn bench_argb_to_rgba(c: &mut Criterion) {
    let img = build_image_256();
    c.bench_function("argb_to_rgba", |b| {
        b.iter(|| {
            let out = black_box(&img).to_rgba();
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_argb_to_rgba);
criterion_main!(benches);
