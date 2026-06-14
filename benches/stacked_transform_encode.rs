//! Criterion bench — VP8L (lossless) full RIFF/WEBP encode, separated by
//! the three distinct content regimes the RFC 9649 §3.5 **stacked-transform
//! chains** target.
//!
//! Rounds 302–306 grew the §3.5 stacked candidates the lossless encoder
//! evaluates (§4.4 color-indexing → §4.1 predictor; §4.2 cross-color → §4.1
//! predictor; §4.2 cross-color → §4.3 subtract-green → §4.1 predictor) and
//! widened the predictor-sub-image cost-model sweep each chain runs (L1
//! proxy / Shannon entropy / sub-image-aware entropy across a four-weight
//! lambda set straddling the empirically-observed residual-vs-§7.2-sub-image
//! cost crossover). Every one of those rounds reasoned about an
//! "empirically-observed crossover" without a committed harness that
//! exercises the chooser on inputs where each stacked chain is the regime it
//! was added for. This bench supplies that A/B target.
//!
//! The encoder evaluates every candidate and keeps the byte-shortest stream,
//! so the encode *time* scales with how many candidates a given input
//! activates. The three inputs here are shaped so that a different stacked
//! chain is the one expected to win each one, giving future cost-model rounds
//! a per-regime before/after comparison for both encode time and (read off
//! the produced `Vec<u8>` length) output size:
//!
//! * `palette_indexed` — a small fixed-palette tile (icon / line-art-like).
//!   Activates the §4.4 color-indexing path and its round-302 color-indexing
//!   → predictor stacked chain.
//! * `photo_decorrelated` — a smoothly-varying image whose red / blue
//!   channels are strongly correlated to green. Activates the §4.2 cross-color
//!   transform and its round-303/304 color → predictor and
//!   color → subtract-green → predictor stacked chains.
//! * `smooth_gradient` — a high-spatial-coherence gradient that drives the
//!   §4.1 predictor sub-image cost model across the lambda-sweep crossover the
//!   stacked chains tune.
//!
//! All inputs are built once outside `b.iter`, so the measured time is
//! encode-only. Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench stacked_transform_encode -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::encode_webp_lossless;

/// 128×128 RGBA tile drawn from a fixed 8-colour palette in
/// spatially-coherent horizontal bands. The small distinct-colour count makes
/// the §4.4 color-indexing transform feasible (a `collect_palette` probe
/// succeeds), and the long same-index runs make the round-302 color-indexing
/// → §4.1 predictor stacked chain a live candidate over the bundled-index
/// image.
fn palette_indexed_rgba() -> (Vec<u8>, u32, u32) {
    let (w, h) = (128u32, 128u32);
    const PALETTE: [[u8; 4]; 8] = [
        [0x00, 0x00, 0x00, 0xff],
        [0xff, 0xff, 0xff, 0xff],
        [0xc0, 0x10, 0x10, 0xff],
        [0x10, 0xc0, 0x10, 0xff],
        [0x10, 0x10, 0xc0, 0xff],
        [0xc0, 0xc0, 0x10, 0xff],
        [0x10, 0xc0, 0xc0, 0xff],
        [0xc0, 0x10, 0xc0, 0xff],
    ];
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        // Bands of varying height so index runs are long but the band
        // boundaries still give the predictor real work between runs.
        let band = ((y / 9) as usize + (y / 23) as usize) % PALETTE.len();
        for _x in 0..w {
            buf.extend_from_slice(&PALETTE[band]);
        }
    }
    (buf, w, h)
}

/// 192×192 RGBA image whose red and blue channels are strong, smoothly
/// varying functions of the green channel — exactly the inter-channel
/// correlation the §4.2 cross-color transform models, with a residual
/// uniform green-correlated component left for the §4.3 subtract-green pass
/// and spatial coherence for the §4.1 predictor. Drives the round-303/304
/// color → predictor and color → subtract-green → predictor stacked chains.
fn photo_decorrelated_rgba() -> (Vec<u8>, u32, u32) {
    let (w, h) = (192u32, 192u32);
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            // Smooth green ramp across the diagonal plus a low-frequency
            // ripple; red / blue are affine functions of green so the
            // cross-color multipliers carry real mass.
            let g = (((x + y) * 2 / 3) % 256) as u8;
            let gv = g as i32;
            let r = ((gv * 7 / 8 + 12) & 0xff) as u8;
            let b = ((gv * 5 / 6 + 30) & 0xff) as u8;
            buf.push(r);
            buf.push(g);
            buf.push(b);
            buf.push(0xff);
        }
    }
    (buf, w, h)
}

/// 256×256 high-coherence RGBA gradient. `(x, y)` → channels that vary
/// slowly and smoothly across the canvas, so a single dominant §4.1
/// predictor mode covers wide regions and the sub-image-aware lambda sweep
/// the stacked chains run reaches its residual-vs-sub-image cost crossover.
fn smooth_gradient_rgba() -> (Vec<u8>, u32, u32) {
    let (w, h) = (256u32, 256u32);
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            buf.push((x / 2) as u8);
            buf.push((y / 2) as u8);
            buf.push(((x + y) / 4) as u8);
            buf.push(0xff);
        }
    }
    (buf, w, h)
}

fn bench_stacked_transform_encode(c: &mut Criterion) {
    let (palette, pw, ph) = palette_indexed_rgba();
    c.bench_function("stacked_encode_palette_indexed", |b| {
        b.iter(|| {
            let out = encode_webp_lossless(black_box(&palette), pw, ph).expect("encode");
            black_box(out)
        })
    });

    let (photo, fw, fh) = photo_decorrelated_rgba();
    c.bench_function("stacked_encode_photo_decorrelated", |b| {
        b.iter(|| {
            let out = encode_webp_lossless(black_box(&photo), fw, fh).expect("encode");
            black_box(out)
        })
    });

    let (gradient, gw, gh) = smooth_gradient_rgba();
    c.bench_function("stacked_encode_smooth_gradient", |b| {
        b.iter(|| {
            let out = encode_webp_lossless(black_box(&gradient), gw, gh).expect("encode");
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_stacked_transform_encode);
criterion_main!(benches);
