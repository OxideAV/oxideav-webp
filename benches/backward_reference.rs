//! Criterion bench — decoder-side §5.2.2 `apply_backward_reference`, the
//! LZ77 copy-back that assembles one backward-reference run into the
//! decoded ARGB pixel buffer.
//!
//! Every §5.2 lossless decode that contains an LZ77 length symbol routes
//! through this function once per run: `decode_one_symbol` reads the
//! length `L` and the scan-line distance `D`, then calls
//! `apply_backward_reference` to append `L` pixels copied from `D`
//! positions back. The two existing LZ77 benches (`lz77_match`,
//! `lz77_chain`) measure the **encoder**'s hash-chain matcher — the
//! search that *finds* a run; the **decoder**'s copy-back that *replays*
//! the chosen run had no isolated harness. This bench fills that gap.
//!
//! The §5.2.2 walk is `for i in 0..L { pixels.push(pixels[src + i]) }`,
//! so its cost splits sharply by the relation between `D` (`dist`) and
//! `L` (`length`) — the cells isolate the distinct regimes:
//!
//! * **`nonoverlap` (`dist >= length`)** — the source region is fully
//!   materialised before the copy starts, so each read sees a settled
//!   pixel: the region-copy lower bound.
//! * **`overlap_partial` (`dist < length`, `dist = 4`)** — the run reads
//!   pixels it is itself emitting; the just-copied 4-pixel window
//!   repeats. The self-referential read-after-append regime.
//! * **`rle_dist1` (`dist == 1`)** — maximal overlap: every appended
//!   pixel is a copy of the immediately preceding one, i.e. a single-
//!   pixel run-length fill (a flat-colour span). The tightest dependency
//!   chain the walk can produce.
//! * **`manyruns` (mixed short runs)** — a sequence of many small runs
//!   at varied `dist`/`length`, replaying the per-call entry/guard cost
//!   (underflow + overflow checks) that a real fragmented stream pays
//!   far more often than a few long runs.
//!
//! ## Construction (public API only)
//!
//! Each cell seeds a `Vec<u32>` with a deterministic LCG-filled literal
//! prefix (the already-emitted pixels a real decode would have produced
//! before the first run), built once outside `b.iter`. Only the cheap
//! `clone` of that seed buffer lives in the `iter_batched` setup, outside
//! the measured interval, so each iteration starts from the same state
//! (the function appends, so the buffer must be reset between runs).
//! `total_pixels` is set large enough that the overflow guard never
//! fires, matching a valid stream.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench backward_reference -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use oxideav_webp::vp8l_decode::apply_backward_reference;

/// One backward-reference run: §5.2.2 length `L` and scan-line pixel
/// distance `D`.
#[derive(Clone, Copy)]
struct Run {
    length: usize,
    dist: usize,
}

/// Deterministic LCG fill (same constants as the §4.x / §6.2 benches) so
/// the literal prefix is reproducible across runs and hosts.
fn lcg_buffer(seed_init: u32, n: usize) -> Vec<u32> {
    let mut seed: u32 = seed_init;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        v.push(seed);
    }
    v
}

/// Drive `runs` through `apply_backward_reference` against a freshly
/// `clone`d seed buffer. The seed (the literal prefix) is built once
/// outside the timed loop; only its `clone` is in the `iter_batched`
/// setup, so the measured interval is the §5.2.2 copy-back itself.
fn bench_cell(c: &mut Criterion, seed: &[u32], runs: &[Run], total_pixels: usize, label: &str) {
    let name = format!("apply_backward_reference_{}", label);
    c.bench_function(&name, |b| {
        b.iter_batched(
            || seed.to_vec(),
            |mut pixels| {
                for run in runs {
                    let r = apply_backward_reference(
                        &mut pixels,
                        black_box(run.length),
                        black_box(run.dist),
                        total_pixels,
                    )
                    .expect("hand-built runs stay within bounds");
                    black_box(r.end);
                }
                black_box(pixels.len())
            },
            BatchSize::SmallInput,
        )
    });
}

// A 256-pixel literal prefix is more than enough source for every cell's
// distances; the per-cell run plans append on top of it.
const PREFIX: usize = 256;

fn bench_nonoverlap(c: &mut Criterion) {
    // 32 runs, each copying a 64-pixel region from `dist == 64` back —
    // `dist >= length`, so every source pixel is already settled.
    let seed = lcg_buffer(0x1357_9bdf, PREFIX);
    let runs: Vec<Run> = (0..32)
        .map(|_| Run {
            length: 64,
            dist: 64,
        })
        .collect();
    bench_cell(c, &seed, &runs, PREFIX + 32 * 64, "nonoverlap_d64_l64");
}

fn bench_overlap_partial(c: &mut Criterion) {
    // 32 runs, each a 64-pixel copy from `dist == 4`: the 4-pixel window
    // repeats 16× — the self-referential read-after-append regime.
    let seed = lcg_buffer(0x2468_ace0, PREFIX);
    let runs: Vec<Run> = (0..32)
        .map(|_| Run {
            length: 64,
            dist: 4,
        })
        .collect();
    bench_cell(c, &seed, &runs, PREFIX + 32 * 64, "overlap_partial_d4_l64");
}

fn bench_rle_dist1(c: &mut Criterion) {
    // 32 runs, each a 64-pixel `dist == 1` fill: maximal overlap, every
    // appended pixel copies its immediate predecessor (flat-colour span).
    let seed = lcg_buffer(0x0f1e_2d3c, PREFIX);
    let runs: Vec<Run> = (0..32)
        .map(|_| Run {
            length: 64,
            dist: 1,
        })
        .collect();
    bench_cell(c, &seed, &runs, PREFIX + 32 * 64, "rle_dist1_l64");
}

fn bench_manyruns(c: &mut Criterion) {
    // 512 short runs cycling small length/dist pairs — the entry/guard
    // cost a fragmented stream pays far more often than a few long runs.
    let seed = lcg_buffer(0x7a6b_5c4d, PREFIX);
    let plans = [
        Run { length: 3, dist: 1 },
        Run { length: 4, dist: 2 },
        Run { length: 5, dist: 4 },
        Run { length: 2, dist: 7 },
    ];
    let runs: Vec<Run> = (0..512).map(|i| plans[i % plans.len()]).collect();
    let total: usize = PREFIX + runs.iter().map(|r| r.length).sum::<usize>();
    bench_cell(c, &seed, &runs, total, "manyruns_512_short");
}

criterion_group!(
    benches,
    bench_nonoverlap,
    bench_overlap_partial,
    bench_rle_dist1,
    bench_manyruns,
);
criterion_main!(benches);
