//! Criterion bench — §5.2.2 LZ77 hash-chain matcher *chain-depth*
//! scenarios (`Lz77Matcher::find` + `insert`).
//!
//! The round-283 encode profile placed `Lz77Matcher::find` + `insert`
//! at rank 3 (~10% combined self-time) and the round-281 / round-283
//! follow-ups flagged it as the next encoder candidate after the
//! chooser walk, noting it is "still unbenched in isolation — the
//! public-entry `lz77_match` bench amortises it against tokenisation."
//! The matcher itself is package-private, so this bench (like
//! `lz77_match`) drives it through the public
//! `encode_argb_literals_with_width` entry — but where `lz77_match`
//! uses a single repeating tile, these cells deliberately *vary the
//! hash-chain depth* the matcher walks, so a future chain-walk cut
//! (the bench-first step the round-281 follow-up calls for) has A/B
//! numbers at each depth regime instead of one blended figure.
//!
//! Chain depth is governed by how many earlier positions hash to the
//! same bucket as the current one. The §5.2.2 matcher inserts every
//! position into a hash chain keyed on a few leading pixels; the
//! shorter and more uniform the content's period, the more positions
//! pile into one bucket and the deeper `find` walks before it accepts
//! or exhausts a candidate. The cells span that axis:
//!
//! * **`deep_period2`** — a 2-pixel A/B alternation over a 8192-pixel
//!   tile. Almost every position shares one of two hash buckets, so
//!   the chains are maximally deep: the worst case for `find`'s walk
//!   and the regime a depth-cap or better-match early-out would help
//!   most.
//! * **`deep_period4`** — a 4-pixel period: still very deep chains but
//!   four buckets instead of two, a half-step toward natural content.
//! * **`medium_period64`** — a 64-pixel period (one full bench row at
//!   width 64). Chains are moderate; matches are long but the walk to
//!   find them is short — the "easy long match" regime.
//! * **`shallow_unique`** — a deterministic LCG image with essentially
//!   unique pixels. Hash buckets stay near-empty, `find` almost always
//!   walks one or zero candidates and rejects: the literal-dominated
//!   lower bound where the matcher's overhead is pure insert + miss.
//! * **`natural_gradient`** — a smooth diagonal gradient plus light
//!   noise: realistic mixed content where some runs match and the rest
//!   are near-misses, the blended cost a photographic tile pays.
//!
//! All cells are the same 8192-pixel length so the per-iteration
//! insert count is constant and cross-cell deltas isolate to the
//! `find` walk depth. Each is encoded at width 64 so the §5.2.2
//! distance-map encoder also exercises the row-stride distance codes.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench lz77_chain -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::encode_argb_literals_with_width;

/// Pixels per tile. 8192 = 128 rows × 64 px; large enough that the
/// hash chains reach their steady-state depth on the repetitive cells.
const TILE: usize = 8192;

/// Encode width — one bench row is 64 px, matching the `lz77_match` /
/// distance-map shapes elsewhere in the encoder bench shelf.
const WIDTH: usize = 64;

/// Deterministic LCG used by the unique / gradient cells.
fn lcg_step(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

/// A tile that cycles a `period`-pixel pattern of distinct opaque
/// colors. Smaller `period` ⇒ deeper hash chains.
fn periodic_tile(period: usize) -> Vec<u32> {
    let mut pattern = Vec::with_capacity(period);
    let mut seed = 0x1357_9bdf_2468_aceeu64;
    for _ in 0..period {
        let v = lcg_step(&mut seed);
        pattern.push(0xff00_0000 | (v as u32 & 0x00ff_ffff));
    }
    (0..TILE).map(|i| pattern[i % period]).collect()
}

/// A near-unique tile: every pixel a fresh LCG draw. Hash buckets stay
/// shallow; the matcher mostly inserts and rejects.
fn unique_tile() -> Vec<u32> {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    (0..TILE)
        .map(|_| 0xff00_0000 | (lcg_step(&mut seed) as u32 & 0x00ff_ffff))
        .collect()
}

/// A smooth diagonal gradient plus ±4 LCG noise — realistic
/// photographic-ish content with a mix of short matches and near-misses.
fn gradient_tile() -> Vec<u32> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    (0..TILE)
        .map(|i| {
            let x = (i % WIDTH) as u32;
            let y = (i / WIDTH) as u32;
            let base = (x + y) & 0xff;
            let noise = (lcg_step(&mut seed) >> 60) as u32 & 0x7; // 0..=7
            let g = base.wrapping_add(noise) & 0xff;
            let r = (g / 2).wrapping_add(noise) & 0xff;
            let b = (g / 3).wrapping_add(r / 4) & 0xff;
            0xff00_0000 | (r << 16) | (g << 8) | b
        })
        .collect()
}

fn bench_tile(c: &mut Criterion, tile: Vec<u32>, label: &str) {
    let name = format!("lz77_chain_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let out = encode_argb_literals_with_width(black_box(&tile), WIDTH as u32);
            black_box(out)
        })
    });
}

fn bench_deep_period2(c: &mut Criterion) {
    bench_tile(c, periodic_tile(2), "deep_period2");
}

fn bench_deep_period4(c: &mut Criterion) {
    bench_tile(c, periodic_tile(4), "deep_period4");
}

fn bench_medium_period64(c: &mut Criterion) {
    bench_tile(c, periodic_tile(64), "medium_period64");
}

fn bench_shallow_unique(c: &mut Criterion) {
    bench_tile(c, unique_tile(), "shallow_unique");
}

fn bench_natural_gradient(c: &mut Criterion) {
    bench_tile(c, gradient_tile(), "natural_gradient");
}

criterion_group!(
    benches,
    bench_deep_period2,
    bench_deep_period4,
    bench_medium_period64,
    bench_shallow_unique,
    bench_natural_gradient,
);
criterion_main!(benches);
