//! Criterion bench — full-file VP8L (lossless) decode.
//!
//! Encodes a 256×256 RGBA gradient once at startup, then measures
//! the round-trip decode of that buffer via the public
//! [`oxideav_webp::decode_webp`] entry point — i.e. RIFF walk +
//! VP8L bitstream decode + per-pixel ARGB→RGBA repack.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench lossless_decode -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::{decode_webp, encode_webp_lossless};

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

fn bench_lossless_decode(c: &mut Criterion) {
    let rgba = gradient_rgba_256();
    let webp = encode_webp_lossless(&rgba, 256, 256).expect("encode");
    c.bench_function("lossless_decode_argb_256", |b| {
        b.iter(|| {
            let img = decode_webp(black_box(&webp)).expect("decode");
            black_box(img)
        })
    });
}

criterion_group!(benches, bench_lossless_decode);
criterion_main!(benches);
