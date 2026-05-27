//! Criterion bench — VP8L §5.2.2 LZ77 hash-chain matcher.
//!
//! The matcher is package-private (`Lz77Matcher` lives inside
//! `vp8l_encode`), so this bench drives it through the public
//! [`oxideav_webp::vp8l_encode::encode_argb_literals_with_width`]
//! entry point, which always runs the LZ77 + entropy stage. The
//! bench input is a 4096-pixel (4 KiB ARGB-words → 16 KiB) tile of
//! a repeating 8-pixel pattern, which guarantees many backward
//! references for the matcher to find.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench lz77_match -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_encode::encode_argb_literals_with_width;

/// Build a 4096-pixel ARGB tile with a repeating 8-pixel period so the
/// matcher emits long backward-reference runs.
fn lz77_tile_4k() -> Vec<u32> {
    let pattern = [
        0xff00_0000u32,
        0xff80_4020,
        0xffaa_bbcc,
        0xff00_ff00,
        0xff10_2030,
        0xff40_5060,
        0xff70_8090,
        0xffff_ffff,
    ];
    let mut out = Vec::with_capacity(4096);
    for i in 0..4096 {
        out.push(pattern[i % pattern.len()]);
    }
    out
}

fn bench_lz77_match(c: &mut Criterion) {
    let tile = lz77_tile_4k();
    c.bench_function("vp8l_lz77_match", |b| {
        b.iter(|| {
            // 64-pixel wide so the §5.2.2 distance-map encoder also
            // exercises the row-stride distance codes.
            let out = encode_argb_literals_with_width(black_box(&tile), 64);
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_lz77_match);
criterion_main!(benches);
