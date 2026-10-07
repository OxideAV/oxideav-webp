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
//! * `photo_256`: a 256×256 photo-like RGBA image (smooth gradients, a
//!   soft blob and small per-channel noise), the input the standalone
//!   speed comparison against the reference encoder uses.
//!
//! The `*_method6` cells encode the same inputs at `EncodeOptions::method`
//! 6, the exhaustive search that was the only lossless path before the
//! single-pass default; the other cells measure the default.
//!
//! Every bench builds its input once outside `b.iter`, so the
//! measured time is encode-only. Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench lossless_encode -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::{decode_rgba8, encode_rgba8, EncodeOptions};

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

/// Build a deterministic photo-like `w × h` RGBA image: gradients, a soft
/// bright blob and xorshift noise of ±8 per channel.
fn photo_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let d = ((fx - 0.5).powi(2) + (fy - 0.4).powi(2)).sqrt();
            let blob = (1.0 - (d * 3.0).min(1.0)) * 60.0;
            let nr = (state & 0xF) as f32 - 8.0;
            let ng = ((state >> 4) & 0xF) as f32 - 8.0;
            let nb = ((state >> 8) & 0xF) as f32 - 8.0;
            let r = (40.0 + 180.0 * fx + blob + nr).clamp(0.0, 255.0) as u8;
            let g = (60.0 + 150.0 * fy + blob + ng).clamp(0.0, 255.0) as u8;
            let b = (200.0 - 120.0 * fx * fy + nb).clamp(0.0, 255.0) as u8;
            px.extend_from_slice(&[r, g, b, 255]);
        }
    }
    px
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

    let photo = photo_rgba(256, 256);
    c.bench_function("lossless_encode_photo_256", |b| {
        b.iter(|| {
            let out = encode_rgba8(256, 256, black_box(&photo), &EncodeOptions::default())
                .expect("encode");
            black_box(out)
        })
    });

    let exhaustive = EncodeOptions::default().with_method(6);
    c.bench_function("lossless_encode_rgba_256_method6", |b| {
        b.iter(|| {
            let out = encode_rgba8(256, 256, black_box(&gradient), &exhaustive).expect("encode");
            black_box(out)
        })
    });
    c.bench_function("lossless_encode_natural_128_method6", |b| {
        b.iter(|| {
            let out = encode_rgba8(128, 128, black_box(&natural), &exhaustive).expect("encode");
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_lossless_encode);
criterion_main!(benches);
