//! Criterion bench — encoder-side §5.2.2 `pixel_distance_to_distance_code`,
//! the distance-code chooser run once per emitted LZ77 backward reference.
//!
//! Every LZ77 `Copy { length, distance }` token an encoder picks must map
//! its pixel `distance` to a §5.2.2 *distance code* before the value enters
//! the DISTANCE prefix-code through [`value_to_prefix`]. The chooser
//! ([`oxideav_webp::vp8l_encode::pixel_distance_to_distance_code`]) returns
//! the **smaller** of two encodings:
//!
//! * the legacy scan-line code `D + 120`, always valid; and
//! * any §5.2.2 *distance-map* code `c ∈ 1..=120` whose neighbour offset
//!   `(xi, yi)` reconstructs to `max(xi + yi * image_width, 1) == D` for
//!   the image's width — these small codes (`1..=120`) feed `value_to_prefix`
//!   through low-prefix slots, entering the DISTANCE Huffman tree with the
//!   highest frequencies and shortest emitted lengths.
//!
//! Choosing the map code is what keeps natural-image vertical matches
//! (distance == one row) cheap on the wire. But the chooser has **no
//! early-out**: it linearly scans all 120 `DISTANCE_MAP` entries on every
//! call, recomputing `xi + yi * W` and the clamp-to-1, regardless of where
//! (or whether) a match lands. Inside the encode path the function is
//! called at least twice per match — once in `count_frequencies` to build
//! the DISTANCE frequency table, once again in the emit loop — so a
//! match-heavy lossless encode pays this 120-iteration scan on the order of
//! twice per backward reference. The `lz77_match` / `lz77_chain` benches
//! time the matcher that *finds* a run and `value_to_prefix` times the
//! per-symbol prefix split, but the distance→code mapping between them had
//! no isolated harness; this bench fills that gap.
//!
//! Four cells fix `image_width = 256` (a typical lossless tile width) and
//! vary `distance` across the regimes that reach different points of the
//! 120-entry scan and different `candidate < best` update counts:
//!
//! * **`dist1_rle`** — `distance = 1`. The tightest, most common run (a
//!   flat-colour RLE fill). Several map entries clamp to 1 (the `(1, 0)`
//!   entry reconstructs to `1` directly, and every `xi + yi * W < 1`
//!   negative-offset entry clamps to 1), so the chooser performs multiple
//!   `mapped == distance` hits and `candidate < best` comparisons before
//!   settling on the smallest.
//! * **`dist_row_above`** — `distance = 256` (== `image_width`). The
//!   `(0, 1)` entry at index 0 reconstructs to `0 + 1 * 256 = 256`, so the
//!   very first scanned entry matches and yields code `1` — the single
//!   cheapest distance code, the natural-image "pixel directly above"
//!   match. `best` is set on iteration 0 and never beaten, but the scan
//!   still runs to completion (no early-out).
//! * **`dist_small_neighbor`** — `distance = 2`. A close horizontal
//!   neighbour: the `(2, 0)` entry reconstructs to `2`, matching mid-scan;
//!   exercises a match that lands away from index 0 with the scan
//!   continuing past it.
//! * **`dist_large_nomatch`** — `distance = 70_000`. Larger than any
//!   `xi + yi * 256` the map can reach, so **no** entry matches: the
//!   chooser runs the full 120-iteration scan, takes zero `candidate <
//!   best` branches, and returns the scan-line fallback `70_000 + 120`.
//!   This is the worst case for the no-early-out scan — all cost, no map
//!   benefit — and the A/B target for any future map-lookup short-circuit.
//!
//! The per-call work is a 120-iteration integer scan, larger than a single
//! `value_to_prefix` call but still small relative to Criterion's
//! per-iteration overhead, so each `b.iter` body runs an inner loop of
//! `INNER_CALLS` chooser calls over the cell's representative `distance`.
//! The returned codes are XOR-accumulated so the optimiser cannot hoist a
//! single call out of the loop, and `black_box` on both the inputs and the
//! accumulator blocks constant-folding and dead-store elimination.
//!
//! No behaviour is measured beyond the `pub` encoder API; the chooser is
//! a pure function of `(distance, image_width)` and returns the same code
//! every call, so the bench is a stable per-regime cost probe.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench distance_code -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::pixel_distance_to_distance_code;

/// Per-`b.iter` inner-loop length. Sized so the 120-iteration scan
/// dominates Criterion's per-iteration overhead at every cell's
/// representative `distance`.
const INNER_CALLS: usize = 1024;

/// A typical lossless tile width; fixed across every cell so the only
/// axis that varies is `distance`.
const IMAGE_WIDTH: u32 = 256;

/// Bench `pixel_distance_to_distance_code(distance, IMAGE_WIDTH)` over an
/// inner loop of `INNER_CALLS` repetitions of the cell's representative
/// `distance`. The XOR accumulator on the returned code keeps the optimiser
/// from hoisting a single call out of the inner loop, and `black_box` on
/// both inputs blocks constant-folding the whole body.
fn bench_distance(c: &mut Criterion, distance: usize, label: &str) {
    let name = format!("distance_code_{}", label);
    c.bench_function(&name, |b| {
        b.iter(|| {
            let mut acc: u32 = 0;
            for _ in 0..INNER_CALLS {
                acc ^= pixel_distance_to_distance_code(black_box(distance), black_box(IMAGE_WIDTH));
            }
            black_box(acc);
        })
    });
}

// `distance = 1` — flat-colour RLE fill, multiple clamp-to-1 map hits.
fn bench_dist1_rle(c: &mut Criterion) {
    bench_distance(c, 1, "dist1_rle");
}

// `distance = image_width = 256` — the `(0, 1)` "pixel above" match at
// index 0, code 1, the single cheapest distance code.
fn bench_dist_row_above(c: &mut Criterion) {
    bench_distance(c, IMAGE_WIDTH as usize, "dist_row_above");
}

// `distance = 2` — close horizontal neighbour matched mid-scan.
fn bench_dist_small_neighbor(c: &mut Criterion) {
    bench_distance(c, 2, "dist_small_neighbor");
}

// `distance = 70_000` — no map entry matches at width 256: full
// 120-iteration scan, scan-line fallback, worst case for the no-early-out
// loop.
fn bench_dist_large_nomatch(c: &mut Criterion) {
    bench_distance(c, 70_000, "dist_large_nomatch");
}

criterion_group!(
    benches,
    bench_dist1_rle,
    bench_dist_row_above,
    bench_dist_small_neighbor,
    bench_dist_large_nomatch,
);
criterion_main!(benches);
