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

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::{decode_webp, encode_webp_lossless};

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
    let img = match decode_webp(FIXTURE) {
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
    let frame = &img.frames[0];
    let src = &frame.rgba;
    let sw = frame.width as usize;
    let sh = frame.height as usize;
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
            let out = encode_webp_lossless(black_box(&gradient), 256, 256).expect("encode");
            black_box(out)
        })
    });

    let natural = natural_rgba_128();
    c.bench_function("lossless_encode_natural_128", |b| {
        b.iter(|| {
            let out = encode_webp_lossless(black_box(&natural), 128, 128).expect("encode");
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_lossless_encode);
criterion_main!(benches);
