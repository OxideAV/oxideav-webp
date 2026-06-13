//! Criterion bench — the §2.5 `VP8 ` (lossy) decode path this crate owns.
//!
//! The lossy bitstream's entropy decode + inverse DCT + intra
//! prediction + loop filter live in the sibling `oxideav-vp8` decoder,
//! not here. What **this** crate runs on the lossy path is the stage
//! after the reconstructed I420 key-frame comes back: the §2.5
//! [`oxideav_webp::vp8_decode::decode_lossy_rgba`] wrapper, whose
//! per-pixel cost is the [`oxideav_webp::vp8_decode::yuv420_to_rgba`]
//! reconstruction loop — nearest-neighbour 4:2:0 chroma up-sampling plus
//! the RFC 6386 §9.2 BT.601 full-range YCbCr→RGB matrix, evaluated once
//! per output pixel (RFC 9649 §10 cites the same color space). That loop
//! had no isolated harness; this bench fills the gap and ranks the lossy
//! decode hot path the crate can actually act on.
//!
//! The cells separate the three altitudes a lossy decode passes through:
//!
//! * **`decode_webp_lossy_e2e`** — the full public
//!   [`oxideav_webp::decode_webp`] over the committed
//!   `lossy-with-alpha-128x128.webp` fixture: §2.3 RIFF walk + §2.7.1.2
//!   extended layout + the sibling VP8 decode + the §2.7.1.2 `ALPH`
//!   plane layering + this crate's YCbCr→RGB conversion — what a caller
//!   of the public API actually pays for one lossy still image.
//! * **`decode_lossy_rgba_extracted`** — the §2.5
//!   [`oxideav_webp::vp8_decode::decode_lossy_rgba`] over the *already
//!   extracted* `VP8 ` chunk bitstream (container walk removed): the
//!   sibling decode plus this crate's conversion, isolated from the RIFF
//!   layer. Subtracting this from the e2e cell sizes the container +
//!   `ALPH` overhead; subtracting the `yuv420_to_rgba_*` cells from this
//!   one sizes the sibling decode itself.
//! * **`yuv420_to_rgba_*`** — the pure crate-owned conversion loop in
//!   isolation, over a synthetic neutral-chroma I420 frame at three
//!   sizes (16×16, 128×128, 256×256). Same-shaped input every iteration,
//!   no decode, no allocation churn beyond the output buffer: the
//!   cleanest A/B target for a SIMD or fixed-point-matrix rewrite of the
//!   conversion. The 128×128 cell matches the fixture's dimensions so
//!   its number is directly comparable to the share of
//!   `decode_lossy_rgba_extracted` the conversion accounts for.
//!
//! No behavior is measured beyond the public/`pub` API. The synthetic
//! frames carry neutral chroma so the conversion path is exercised
//! end-to-end (every pixel runs the full matrix + clamp) without
//! depending on any particular decoded content.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench lossy_decode -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_vp8::Vp8DecodedFrame;
use oxideav_webp::container::{self, fourcc};
use oxideav_webp::vp8_chunk::WebpLossyChunk;
use oxideav_webp::vp8_decode::{decode_lossy_rgba, yuv420_to_rgba};

/// The committed 128×128 extended-lossy fixture (`VP8 ` + `ALPH`). Same
/// bytes the `fixture_walks` / `vp8_lossy_roundtrip` integration tests
/// decode; included here so the bench needs no external file.
const LOSSY_128: &[u8] = include_bytes!("../tests/data/lossy-with-alpha-128x128.webp");

/// Build a synthetic I420 key-frame of `w × h` luma with neutral chroma.
///
/// Luma is a deterministic gradient (so the matrix sees a spread of
/// inputs, not a single constant the optimiser could fold); chroma is
/// the 4:2:0 sub-sampled neutral plane (`128`), sized `⌈w/2⌉ × ⌈h/2⌉` to
/// match what [`yuv420_to_rgba`] indexes. The result is shaped exactly
/// like a real decoded `Vp8DecodedFrame`.
fn synthetic_frame(w: u32, h: u32) -> Vp8DecodedFrame {
    let (wu, hu) = (w as usize, h as usize);
    let uv_w = wu.div_ceil(2);
    let uv_h = hu.div_ceil(2);
    let mut y = Vec::with_capacity(wu * hu);
    for row in 0..hu {
        for col in 0..wu {
            y.push(((row * 3 + col * 5) & 0xff) as u8);
        }
    }
    Vp8DecodedFrame {
        width: w,
        height: h,
        y,
        u: vec![128u8; uv_w * uv_h],
        v: vec![128u8; uv_w * uv_h],
    }
}

/// Extract the raw `VP8 ` bitstream slice from a full extended-lossy
/// WebP file, so the bench can drive [`decode_lossy_rgba`] without the
/// RIFF walk inside the measured interval.
fn extract_vp8_bitstream(file: &[u8]) -> Vec<u8> {
    let cont = container::parse(file).expect("fixture is a well-formed RIFF container");
    let chunk = cont
        .first_chunk_with_fourcc(fourcc::VP8)
        .expect("fixture carries a VP8 lossy chunk");
    let lossy = WebpLossyChunk::from_chunk(file, chunk).expect("fixture VP8 header is well-formed");
    lossy.bitstream().to_vec()
}

fn bench_decode_webp_lossy_e2e(c: &mut Criterion) {
    // Sanity: the fixture decodes before we time it.
    let probe = oxideav_webp::decode_webp(LOSSY_128).expect("lossy fixture decodes");
    assert_eq!((probe.width, probe.height), (128, 128));
    c.bench_function("decode_webp_lossy_e2e", |b| {
        b.iter(|| {
            let img = oxideav_webp::decode_webp(black_box(LOSSY_128)).expect("decode");
            black_box(img)
        })
    });
}

fn bench_decode_lossy_rgba_extracted(c: &mut Criterion) {
    let bitstream = extract_vp8_bitstream(LOSSY_128);
    let (w, h, _) = decode_lossy_rgba(&bitstream).expect("extracted VP8 bitstream decodes");
    assert_eq!((w, h), (128, 128));
    c.bench_function("decode_lossy_rgba_extracted", |b| {
        b.iter(|| {
            let out = decode_lossy_rgba(black_box(&bitstream)).expect("decode");
            black_box(out)
        })
    });
}

fn bench_yuv420_to_rgba(c: &mut Criterion) {
    for (w, h) in [(16u32, 16u32), (128, 128), (256, 256)] {
        let frame = synthetic_frame(w, h);
        let name = format!("yuv420_to_rgba_{}x{}", w, h);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let rgba = yuv420_to_rgba(black_box(&frame));
                black_box(rgba)
            })
        });
    }
}

criterion_group!(
    benches,
    bench_decode_webp_lossy_e2e,
    bench_decode_lossy_rgba_extracted,
    bench_yuv420_to_rgba,
);
criterion_main!(benches);
