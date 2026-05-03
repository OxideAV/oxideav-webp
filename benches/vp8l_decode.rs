//! Criterion bench harness for the VP8L (lossless) decode hot paths.
//!
//! Each bench feeds the decoder a real libwebp-encoded fixture from the
//! workspace `docs/image/webp/fixtures/` corpus (mirrored under
//! `tests/fixtures/lossless_corpus/`). The fixtures are picked to
//! exercise the four main code paths inside `vp8l::decode`:
//!
//! * `decode_natural_image` — `lossless-128x128-natural`. 128×128 RGB
//!   image carrying both predictor and cross-color transforms; the
//!   broadest single-fixture exercise of the pipeline (LZ77 + literal
//!   green path + meta-Huffman per-tile group selection +
//!   subtract-green + cross-color + predictor).
//! * `decode_color_cache_stress` — `lossless-color-cache-stress`. Image
//!   constructed (per the corpus README) to stress the colour-cache
//!   alphabet expansion and the cache hash path.
//! * `decode_palette` — `lossless-color-indexing-paletted`. Heavy
//!   colour-indexing transform exercise; the inner pixel loop hits the
//!   bit-packed palette unpack branch.
//! * `decode_simple_32x32` — `lossless-32x32-rgba`. Smaller frame for
//!   high-iteration low-noise measurement; useful when comparing
//!   per-pixel inner-loop costs.
//!
//! There's also a `huffman_decode_throughput` micro-bench feeding a
//! synthetic prefix-coded byte stream straight at `HuffmanTree::decode`
//! — this isolates the entropy stage from transform / pixel-loop work
//! so the Huffman LUT optimisation can be measured cleanly.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_webp::vp8l::bit_reader::BitReader;
use oxideav_webp::vp8l::huffman::HuffmanTree;

// ── Fixture bytes (compiled in at build time) ────────────────────────────
//
// `decode` operates on the raw VP8L chunk payload, *not* on the wrapping
// RIFF container. The natural code path through `vp8l::decode` skips the
// RIFF header — but the fixture files on disk are full `.webp` files.
// We extract the VP8L chunk payload at bench start (once) and hand the
// inner bytes to `vp8l::decode` for each iteration.

const NATURAL_WEBP: &[u8] =
    include_bytes!("../tests/fixtures/lossless_corpus/lossless-128x128-natural/input.webp");
const CACHE_WEBP: &[u8] =
    include_bytes!("../tests/fixtures/lossless_corpus/lossless-color-cache-stress/input.webp");
const PALETTE_WEBP: &[u8] = include_bytes!(
    "../tests/fixtures/lossless_corpus/lossless-color-indexing-paletted/input.webp"
);
const SIMPLE_WEBP: &[u8] =
    include_bytes!("../tests/fixtures/lossless_corpus/lossless-32x32-rgba/input.webp");

/// Pull the `VP8L` chunk's payload bytes out of a RIFF/WEBP file. Panics
/// on a malformed input — bench fixtures are vetted at copy-in time, so
/// any failure here is a corruption-in-tree bug, not a runtime concern.
fn extract_vp8l(buf: &[u8]) -> &[u8] {
    // RIFF header: "RIFF" + u32 size + "WEBP".
    assert!(buf.len() >= 12 && &buf[0..4] == b"RIFF" && &buf[8..12] == b"WEBP");
    let mut i = 12;
    while i + 8 <= buf.len() {
        let id = &buf[i..i + 4];
        let sz = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
        let body = &buf[i + 8..i + 8 + sz];
        if id == b"VP8L" {
            return body;
        }
        // Chunks are 2-byte aligned per RIFF.
        i += 8 + sz + (sz & 1);
    }
    panic!("no VP8L chunk found in fixture");
}

fn bench_decode(c: &mut Criterion) {
    // Each fixture decode benchmark; record throughput in bytes of the
    // *compressed* VP8L payload so criterion's "thrpt" line is in
    // payload-bytes/s — easier to reason about across fixtures than
    // pixel counts (which depend on the predictor mode mix).
    let cases: [(&str, &[u8]); 4] = [
        ("decode_natural_image", extract_vp8l(NATURAL_WEBP)),
        ("decode_color_cache_stress", extract_vp8l(CACHE_WEBP)),
        ("decode_palette", extract_vp8l(PALETTE_WEBP)),
        ("decode_simple_32x32", extract_vp8l(SIMPLE_WEBP)),
    ];

    let mut group = c.benchmark_group("vp8l_decode");
    for (name, payload) in cases.iter() {
        group.throughput(Throughput::Bytes(payload.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let img = oxideav_webp::vp8l::decode(black_box(payload)).expect("decode");
                black_box(img.pixels.len());
            });
        });
    }
    group.finish();
}

// ── Huffman micro-bench ──────────────────────────────────────────────────
//
// Build a small canonical Huffman tree (a literal-green-style alphabet
// where the most-frequent symbol is 1 bit and others stretch out to 8
// bits) and feed a long pseudo-random bit stream through `decode` in a
// tight loop. This is the cleanest measurement of the per-symbol cost
// of `HuffmanTree::decode` — the function we expect to LUT-ify.
//
// Construction (Kraft-equality canonical Huffman):
//   * Symbol 0 (the green-channel "common literal") gets length 2.
//   * Symbols 1..33 get length 6 (32 codes use 1/2 of codespace).
//   * Symbols 33..65 get length 7 (32 codes use 1/4 of codespace).
//   * Sum: 1/4 + 32/64 + 32/128 = 0.25 + 0.5 + 0.25 = 1.0.
// The bit stream is a fixed PRNG-seeded byte array — no randomness per
// iteration, so the bench is deterministic.

fn build_skewed_tree() -> HuffmanTree {
    let mut lens = vec![0u8; 65];
    lens[0] = 2;
    for s in lens.iter_mut().take(33).skip(1) {
        *s = 6;
    }
    for s in lens.iter_mut().take(65).skip(33) {
        *s = 7;
    }
    HuffmanTree::from_code_lengths_for_bench(&lens)
}

fn xorshift_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.truncate(n);
    out
}

fn bench_huffman(c: &mut Criterion) {
    let tree = build_skewed_tree();
    // 64 KiB of pseudo-random bits — long enough that the loop dominates
    // the bench, short enough to fit in L2.
    let bits = xorshift_bytes(0xC0FFEE_DEAD_BEEFu64, 64 * 1024);
    // Number of symbols to decode per iteration. With the skewed tree the
    // average symbol length sits around 4 bits, so 64 KiB ≈ 130 K symbols.
    // We cap at 100 K and stop early on EOF (bit-reader returns zeros
    // gracefully) so each iteration is the same fixed work.
    const N: usize = 100_000;

    let mut group = c.benchmark_group("vp8l_huffman_decode");
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("skewed_64sym", |b| {
        b.iter(|| {
            let mut br = BitReader::new(&bits);
            let mut sum: u32 = 0;
            for _ in 0..N {
                let s = tree.decode(&mut br).unwrap();
                sum = sum.wrapping_add(s as u32);
            }
            black_box(sum);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_decode, bench_huffman);
criterion_main!(benches);
