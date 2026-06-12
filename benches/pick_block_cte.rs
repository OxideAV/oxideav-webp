//! Criterion bench — §3.5.2 encoder color-transform-element chooser
//! (`pick_block_cte`) full-image block walk.
//!
//! `pick_block_cte` is the per-block kernel behind
//! `encode_with_color_transform`'s `build_color_image` pass: for each
//! `(1 << size_bits)`-square block it gathers the block's `(r, g, b)`
//! triples and greedily sweeps the 25-entry `CTE_AXIS_CANDIDATES`
//! table per axis (75 cost evaluations per block, each a per-pixel
//! folded-magnitude sum with an early-out prune). The round-280
//! post-change encode profile placed `encode_with_color_transform`
//! at rank 2 (~9% self-time), and this chooser walk is that
//! function's per-pixel-heavy stage — analogous to the §4.1
//! predictor chooser that round 280 despecialised.
//!
//! The bench drives the exact walk `build_color_image` performs at
//! the encoder-default `size_bits = 4`: a 16×16 grid of 16×16-pixel
//! blocks over a 256×256 ARGB image, 256 `pick_block_cte` calls per
//! iteration. The input is a deterministic LCG image with genuinely
//! correlated channels (red ≈ green/2, blue ≈ green/3 + red/4, plus
//! per-channel noise), so the per-axis sweeps land on non-zero CTEs
//! and the `cost >= best` early-out prunes the way it does on
//! natural content — neither the all-prune degenerate case (solid
//! color) nor the never-prune one (uniform random, where every
//! candidate costs the same in expectation).
//!
//! The folded per-block results are mixed into an accumulator so the
//! optimizer cannot fold the walk away.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench pick_block_cte -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::pick_block_cte;

/// Image side (pixels). 256×256 matches the `lossless_encode` /
/// `predictor_subtract` bench shapes so per-pass numbers stay
/// visually comparable across the encoder bench shelf.
const SIDE: usize = 256;

/// Encoder-default §3.5.2 block size: `size_bits = 4` → 16×16 blocks
/// (`DEFAULT_COLOR_TRANSFORM_SIZE_BITS`).
const BLOCK: usize = 16;

/// Build a 256×256 ARGB buffer with correlated channels:
/// `g` is a diagonal gradient plus ±8 LCG noise, `r ≈ g/2` and
/// `b ≈ g/3 + r/4` each plus ±8 LCG noise. The §3.5.2 fixed-point
/// slopes the chooser should recover are therefore ≈16 (green→red),
/// ≈11 (green→blue), and ≈8 (red→blue) — all interior entries of the
/// candidate table.
fn correlated_argb_256() -> Vec<u32> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut noise = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // ±8 noise from the high bits.
        ((state >> 60) as i32) - 8
    };
    let mut out = Vec::with_capacity(SIDE * SIDE);
    for y in 0..SIDE {
        for x in 0..SIDE {
            let g = (((x + y) / 2) as i32 + noise()) & 0xff;
            let r = (g / 2 + noise()) & 0xff;
            let b = (g / 3 + r / 4 + noise()) & 0xff;
            out.push(0xff00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32);
        }
    }
    out
}

fn bench_pick_block_cte(c: &mut Criterion) {
    let pixels = correlated_argb_256();
    c.bench_function("pick_block_cte_walk_256x256", |b| {
        b.iter(|| {
            let px = black_box(&pixels[..]);
            let mut acc = 0u32;
            for by in 0..SIDE / BLOCK {
                for bx in 0..SIDE / BLOCK {
                    let (gtr, gtb, rtb) =
                        pick_block_cte(px, SIDE, SIDE, bx * BLOCK, by * BLOCK, BLOCK, BLOCK);
                    acc = acc
                        .rotate_left(5)
                        .wrapping_add(((gtr as u32) << 16) | ((gtb as u32) << 8) | rtb as u32);
                }
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_pick_block_cte);
criterion_main!(benches);
