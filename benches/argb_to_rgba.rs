//! Criterion bench — [`oxideav_webp::vp8l::Vp8lImage::to_rgba`]
//! ARGB-`u32` → packed `[R, G, B, A]` byte repack.
//!
//! Constructs a 256×256 `Vp8lImage` and measures `to_rgba` only,
//! isolating the per-pixel byte shuffle from any decode work. This
//! is the cleanest target for a SIMD shuffle rewrite.
//!
//! Two extra A/B cells (`repack_push_loop` / `repack_chunks_exact`)
//! isolate the byte-repack *form* itself over the same 256×256 ARGB
//! buffer the crate's private `decode_lossless_image` conversion runs
//! on. They mirror, in this harness only, the two shapes the conversion
//! has taken: the original `Vec::push` one-byte-at-a-time loop, and the
//! pre-sized `chunks_exact_mut(4)` form (the round-170 `to_rgba_scalar`
//! shape, now adopted by the private converter too). Both produce
//! byte-identical output; the cells size the gap so the converter's
//! switch to the pre-sized form has a recorded before/after.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench argb_to_rgba -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l::Vp8lImage;

/// Original `Vec::push` repack form (pre-pack-rewrite baseline). Pushes
/// four bytes per pixel into a `with_capacity` buffer — one bounds /
/// capacity check per byte.
fn repack_push_loop(pixels: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() * 4);
    for &argb in pixels {
        out.push((argb >> 16) as u8);
        out.push((argb >> 8) as u8);
        out.push(argb as u8);
        out.push((argb >> 24) as u8);
    }
    out
}

/// Pre-sized `chunks_exact_mut(4)` repack form. Allocates the exact
/// output up front and writes each pixel's four channels into a
/// fixed-width chunk, which the compiler auto-vectorises.
fn repack_chunks_exact(pixels: &[u32]) -> Vec<u8> {
    let mut out = vec![0u8; pixels.len() * 4];
    for (chunk, &argb) in out.chunks_exact_mut(4).zip(pixels.iter()) {
        chunk[0] = (argb >> 16) as u8;
        chunk[1] = (argb >> 8) as u8;
        chunk[2] = argb as u8;
        chunk[3] = (argb >> 24) as u8;
    }
    out
}

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

fn bench_repack_forms(c: &mut Criterion) {
    let pixels = build_image_256().pixels;
    // Sanity: both forms agree byte-for-byte before timing them.
    assert_eq!(repack_push_loop(&pixels), repack_chunks_exact(&pixels));

    c.bench_function("repack_push_loop", |b| {
        b.iter(|| black_box(repack_push_loop(black_box(&pixels))))
    });
    c.bench_function("repack_chunks_exact", |b| {
        b.iter(|| black_box(repack_chunks_exact(black_box(&pixels))))
    });
}

criterion_group!(benches, bench_argb_to_rgba, bench_repack_forms);
criterion_main!(benches);
