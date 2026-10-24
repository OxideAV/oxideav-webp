//! Criterion bench — VP8L (lossless) full RIFF/WEBP encode.
//!
//! Drives the public [`oxideav_webp::encode_webp_lossless`] entry
//! point on two synthetic inputs:
//!
//! * `gradient_256` — a 256×256 RGBA gradient (high spatial coherence,
//!   exercises the predictor + LZ77 + entropy coder on smooth tones).
//! * `natural_128` — a 128×128 RGBA tile derived from the existing
//!   `tests/data/lossless-32x32-rgba.webp` fixture (decoded once, tiled
//!   4×4 to 128×128), exercising a more realistic colour distribution.
//!
//! Both benches build their input once outside `b.iter`, so the
//! measured time is encode-only. Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench lossless_encode -- --quick
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use oxideav_webp::{EncodeOptions, decode_rgba8, encode_rgba8};

/// Build a 256×256 RGBA gradient. `(x, y)` → `(x, y, (x ^ y), 0xff)`.
fn gradient_rgba_256() -> Vec<u8> {
    let (w, h) = (256u32, 256u32);
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            buf.push(x as u8);
            buf.push(y as u8);
            buf.push((x ^ y) as u8);
            buf.push(0xff);
        }
    }
    buf
}

/// Decode the committed 32×32 natural-image fixture and tile it 4×4 to
/// produce a 128×128 RGBA buffer. Falls back to the 256×256 gradient
/// truncated to 128×128 if the fixture cannot be decoded for any reason
/// (defensive — the fixture is part of the crate's CI corpus).
fn natural_rgba_128() -> Vec<u8> {
    const FIXTURE: &[u8] = include_bytes!("../tests/data/lossless-32x32-rgba.webp");
    let img = match decode_rgba8(FIXTURE) {
        Ok(img) => img,
        Err(_) => {
            let g = gradient_rgba_256();
            let mut out = Vec::with_capacity(128 * 128 * 4);
            for y in 0..128usize {
                let src_row = &g[y * 256 * 4..y * 256 * 4 + 128 * 4];
                out.extend_from_slice(src_row);
            }
            return out;
        }
    };
    let src = &img.data;
    let sw = img.width as usize;
    let sh = img.height as usize;
    // Tile to 128×128 (4× repeat each axis for a 32×32 source).
    let tw = 128usize;
    let th = 128usize;
    let mut out = Vec::with_capacity(tw * th * 4);
    for y in 0..th {
        let sy = y % sh;
        for x in 0..tw {
            let sx = x % sw;
            let off = (sy * sw + sx) * 4;
            out.extend_from_slice(&src[off..off + 4]);
        }
    }
    out
}

fn bench_lossless_encode(c: &mut Criterion) {
    let gradient = gradient_rgba_256();
    c.bench_function("lossless_encode_rgba_256", |b| {
        b.iter(|| {
            let out = encode_rgba8(256, 256, black_box(&gradient), &EncodeOptions::default())
                .expect("encode");
            black_box(out)
        })
    });

    let natural = natural_rgba_128();
    c.bench_function("lossless_encode_natural_128", |b| {
        b.iter(|| {
            let out = encode_rgba8(128, 128, black_box(&natural), &EncodeOptions::default())
                .expect("encode");
            black_box(out)
        })
    });
}

/// A deterministic 512 x 384 photo-like RGBA image (gradients, a soft
/// blob and per-channel noise): no long exact repeats, strong spatial
/// correlation — the content the effort levels differ most on.
fn photo_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut px = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let d = ((fx - 0.5).powi(2) + (fy - 0.4).powi(2)).sqrt();
            let blob = (1.0 - (d * 3.0).min(1.0)) * 60.0;
            let n = |s: u32| ((state >> s) & 0xF) as f32 - 8.0;
            let r = (40.0 + 180.0 * fx + blob + n(0)).clamp(0.0, 255.0) as u8;
            let g = (60.0 + 150.0 * fy + blob + n(4)).clamp(0.0, 255.0) as u8;
            let b = (200.0 - 120.0 * fx * fy + n(8)).clamp(0.0, 255.0) as u8;
            px.extend_from_slice(&[r, g, b, 255]);
        }
    }
    px
}

/// `EncodeOptions::method` cells (#45): the fastest level, the default
/// and the exhaustive search on the same photo-like image.
fn bench_lossless_methods(c: &mut Criterion) {
    let (w, h) = (512u32, 384u32);
    let photo = photo_rgba(w, h);
    let mut group = c.benchmark_group("lossless_encode_photo_512x384");
    group.sample_size(10);
    for method in [0u8, 4, 6] {
        let opts = EncodeOptions::default().with_method(method);
        group.bench_function(format!("method_{method}"), |b| {
            b.iter(|| black_box(encode_rgba8(w, h, black_box(&photo), &opts).expect("encode")))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lossless_encode, bench_lossless_methods);
criterion_main!(benches);
