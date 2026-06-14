//! Criterion bench — encoder-side §6.2.2 entropy-image block-clustering
//! heuristic (`cluster_blocks_by_histogram_distance`).
//!
//! This is the per-image kernel behind `encode_with_meta_prefix`'s
//! multi-meta-prefix candidate. The RFC 9649 §6.2.2 *entropy image*
//! itself is a decoder construct (it stores which of the
//! `num_prefix_groups` prefix-code groups applies to each
//! `1 << prefix_bits`-aligned block); the *encoder* must decide that
//! partition, and this crate does so with a coarse-RGB-histogram
//! Lloyd's-k-means clusterer:
//!
//! 1. `histogram_block_features` — one pass over every pixel, binning
//!    each channel into `256 >> 4 = 16` bins → a 48-dim count vector
//!    per block (rank-1 by pixel count: O(width × height)).
//! 2. `seed_cluster_centroids` — deterministic farthest-point seeding,
//!    O(num_groups × block_count² × feat_dim) in the worst case.
//! 3. Lloyd's assignment/update loop — up to 8 passes, each
//!    O(block_count × num_groups × feat_dim).
//! 4. Compaction onto a contiguous `0..used-1` group range.
//!
//! Until now the clusterer was only sized by subtraction inside the
//! `lossless_encode` end-to-end number (the chooser invokes
//! `encode_with_meta_prefix` once per `(prefix_bits, num_groups,
//! cache)` candidate). This bench isolates it so the relative weight
//! of the per-pixel feature pass versus the per-block Lloyd loop is
//! visible directly, giving any future cut (e.g. a feature-pass SIMD
//! bin update, or a centroid-average cache so the assignment step
//! stops re-dividing the running sums every pass) A/B numbers per
//! content regime and per `num_groups`.
//!
//! Three altitudes:
//!
//! * **content** — fixed `256×256`, default `prefix_bits = 4`
//!   (16×16 blocks → a 16×16 = 256-block entropy image), `num_groups
//!   = 4`, across three contents:
//!   * `bimodal` — two distinguishable halves (the clusterer does its
//!     full assignment work and Lloyd's converges to a real split),
//!   * `gradient` — a smooth diagonal ramp (neighbouring blocks differ
//!     slightly so every pass reassigns a few blocks),
//!   * `uniform` — a solid colour (seeding finds < 2 distinguishable
//!     centroids and returns the degenerate single-group early-out;
//!     measures the floor — the feature pass plus the failed seed).
//! * **num_groups** — the `bimodal` content at `num_groups ∈ {2,3,4}`
//!   to expose the seeding's `O(num_groups × block_count²)` term.
//! * **size** — the `bimodal` content at `128/256/384` px square
//!   (block counts 64 / 256 / 576) to expose feature-pass vs.
//!   Lloyd-loop scaling as the entropy image grows.
//!
//! No committed fixtures — every input is synthesised in-bench. The
//! returned `Vec<u16>` is folded into a `black_box` accumulator so the
//! optimiser cannot elide the call. Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench meta_prefix_cluster -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxideav_webp::vp8l_encode::cluster_blocks_by_histogram_distance;

/// Encoder-default §6.2.2 entropy-image block size: `prefix_bits = 4`
/// → `1 << 4 = 16`-pixel-square blocks.
const PREFIX_BITS: u8 = 4;

/// Pack `(r, g, b)` (alpha forced opaque) into the crate's ARGB `u32`.
#[inline]
fn argb(r: u8, g: u8, b: u8) -> u32 {
    0xff00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

/// Two distinguishable halves: left = a low-RGB region, right = a
/// high-RGB region, each with ±8 LCG noise so blocks within a half
/// share a histogram mode but are not byte-identical. The clusterer
/// should split cleanly down the vertical midline regardless of the
/// `prefix_bits`-block grid, and Lloyd's converges in a few passes.
fn bimodal_argb(side: usize) -> Vec<u32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut noise = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 60) as i32) - 8
    };
    let mut out = Vec::with_capacity(side * side);
    for _y in 0..side {
        for x in 0..side {
            let base = if x < side / 2 { 40 } else { 200 };
            let r = (base + noise()).clamp(0, 255) as u8;
            let g = (base + noise()).clamp(0, 255) as u8;
            let b = (base + noise()).clamp(0, 255) as u8;
            out.push(argb(r, g, b));
        }
    }
    out
}

/// Smooth diagonal ramp: `(x + y)` mapped across the full 0..=255 range
/// per channel (green steepest, red half, blue third). Neighbouring
/// blocks differ by a small but non-zero histogram shift, so each
/// Lloyd pass reassigns a handful of boundary blocks rather than
/// converging on the first pass.
fn gradient_argb(side: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(side * side);
    let span = (2 * side).max(1);
    for y in 0..side {
        for x in 0..side {
            let t = ((x + y) * 255 / span) as i32;
            let r = (t / 2).clamp(0, 255) as u8;
            let g = t.clamp(0, 255) as u8;
            let b = (t / 3).clamp(0, 255) as u8;
            out.push(argb(r, g, b));
        }
    }
    out
}

/// Solid mid-grey: every block's histogram is identical, so seeding
/// finds no second distinguishable centroid and the clusterer returns
/// the degenerate single-group early-out. Measures the floor cost
/// (the full feature pass plus the failed seed scan).
fn uniform_argb(side: usize) -> Vec<u32> {
    vec![argb(128, 128, 128); side * side]
}

/// Fold a `Vec<u16>` group map into a `u32` so `black_box` cannot drop
/// the clustering work.
#[inline]
fn fold_codes(codes: &[u16]) -> u32 {
    let mut acc = 0u32;
    for &c in codes {
        acc = acc.rotate_left(3).wrapping_add(c as u32 + 1);
    }
    acc
}

fn bench_meta_prefix_cluster(c: &mut Criterion) {
    // --- Altitude 1: content regime (fixed 256×256, num_groups = 4) ---
    const SIDE: u32 = 256;
    const NG: u32 = 4;
    let contents: [(&str, Vec<u32>); 3] = [
        ("bimodal", bimodal_argb(SIDE as usize)),
        ("gradient", gradient_argb(SIDE as usize)),
        ("uniform", uniform_argb(SIDE as usize)),
    ];
    {
        let mut group = c.benchmark_group("meta_prefix_cluster_content_256");
        for (name, pixels) in &contents {
            group.bench_with_input(BenchmarkId::from_parameter(name), pixels, |b, px| {
                b.iter(|| {
                    let codes = cluster_blocks_by_histogram_distance(
                        black_box(px),
                        SIDE,
                        SIDE,
                        PREFIX_BITS,
                        NG,
                    );
                    black_box(fold_codes(&codes))
                })
            });
        }
        group.finish();
    }

    // --- Altitude 2: num_groups sweep (bimodal 256×256) ---
    {
        let pixels = bimodal_argb(SIDE as usize);
        let mut group = c.benchmark_group("meta_prefix_cluster_groups_256");
        for ng in [2u32, 3, 4] {
            group.bench_with_input(BenchmarkId::from_parameter(ng), &ng, |b, &ng| {
                b.iter(|| {
                    let codes = cluster_blocks_by_histogram_distance(
                        black_box(&pixels),
                        SIDE,
                        SIDE,
                        PREFIX_BITS,
                        ng,
                    );
                    black_box(fold_codes(&codes))
                })
            });
        }
        group.finish();
    }

    // --- Altitude 3: image-size sweep (bimodal, num_groups = 4) ---
    {
        let mut group = c.benchmark_group("meta_prefix_cluster_size");
        for side in [128u32, 256, 384] {
            let pixels = bimodal_argb(side as usize);
            group.bench_with_input(BenchmarkId::from_parameter(side), &side, |b, &side| {
                b.iter(|| {
                    let codes = cluster_blocks_by_histogram_distance(
                        black_box(&pixels),
                        side,
                        side,
                        PREFIX_BITS,
                        NG,
                    );
                    black_box(fold_codes(&codes))
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_meta_prefix_cluster);
criterion_main!(benches);
