//! Criterion bench — §4.4 `inverse_color_indexing` across the four
//! palette sizes that drive the four bundling levels.
//!
//! §4.4 of RFC 9649 maps each output pixel to `color_table[index]`,
//! where the index is read from the green channel of a *bundled*
//! input image: at `width_bits = 1` two indices share one byte
//! (4 bits each), at `width_bits = 2` four share (2 bits), at
//! `width_bits = 3` eight share (1 bit), and at `width_bits = 0`
//! every output pixel has its own packed byte. The pre-round-210
//! body recomputes `y * packed_w + (x / count)`, `y * orig_width +
//! x`, and `(x % count) * bits` for every output pixel; the round-
//! 210 rewrite hoists those out of the x-loop and walks the row in
//! `count`-pixel bundles.
//!
//! This bench parameterises the palette size to cover all four
//! `width_bits` cases on a 256x256 output:
//!
//! * `palette_2`   → `width_bits = 3`, 1 bit / index, 8 outputs / byte
//! * `palette_4`   → `width_bits = 2`, 2 bits / index, 4 outputs / byte
//! * `palette_16`  → `width_bits = 1`, 4 bits / index, 2 outputs / byte
//! * `palette_256` → `width_bits = 0`, one byte / output (no bundle)
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench inverse_color_indexing -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_transform::inverse_color_indexing;

const W: u32 = 256;
const H: u32 = 256;

fn build_palette(size: usize) -> Vec<u32> {
    // Deterministic LCG fill so the bench is reproducible. Alpha pinned
    // to 0xff so the palette holds opaque pixels (the `index ->
    // color_table[index]` lookup doesn't care about the alpha channel
    // anyway; only the test of "did we land the right entry" matters).
    let mut seed: u32 = 0xfeed_face;
    let mut v = Vec::with_capacity(size);
    for _ in 0..size {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(0xff00_0000 | (seed & 0x00ff_ffff));
    }
    v
}

fn build_packed(palette_size: usize) -> Vec<u32> {
    // Round-210 bench: build a packed image whose green channel
    // contains a deterministic LCG fill so every sub-index slot is
    // exercised. Width is the §4.4 packed width
    // (`DIV_ROUND_UP(W, 1 << width_bits)`).
    let width_bits = width_bits_for(palette_size);
    let count = 1u32 << width_bits;
    let packed_w = W.div_ceil(count) as usize;
    let n = packed_w * (H as usize);
    let mut seed: u32 = 0x1357_9bdf;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        // Spec only reads the green channel; pack the LCG word into
        // the green lane so each byte covers `count` output pixels.
        let g = (seed >> 8) & 0xff;
        v.push(g << 8);
    }
    v
}

fn width_bits_for(palette_size: usize) -> u8 {
    // Mirrors `color_indexing_width_bits` (private in the crate).
    if palette_size <= 2 {
        3
    } else if palette_size <= 4 {
        2
    } else if palette_size <= 16 {
        1
    } else {
        0
    }
}

fn bench_palette(c: &mut Criterion, palette_size: usize) {
    let palette = build_palette(palette_size);
    let packed = build_packed(palette_size);
    let label = format!("inverse_color_indexing_256x256_palette{}", palette_size);
    c.bench_function(&label, |b| {
        b.iter(|| {
            let out = inverse_color_indexing(black_box(&packed), W, H, black_box(&palette));
            black_box(out);
        })
    });
}

fn bench_palette_2(c: &mut Criterion) {
    bench_palette(c, 2);
}
fn bench_palette_4(c: &mut Criterion) {
    bench_palette(c, 4);
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
    bench_palette_4,
    bench_palette_16,
    bench_palette_256,
);
criterion_main!(benches);
