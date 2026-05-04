//! Integration test: decode the docs/image/webp/fixtures/lossless-*
//! corpus and assert bit-exact equality with the libwebp-encoded
//! `expected.png` ground truth.
//!
//! Mirrors the shape of `lossy_corpus.rs` but for the VP8L (lossless)
//! path: every fixture is `Tier::BitExact` because lossless decode
//! is, by definition, bit-exact. Any pixel divergence is a hard
//! failure and a real bug — there is no "perceptual quality" wiggle
//! room here.
//!
//! The seven fixtures cover:
//!
//! * `lossless-1x1` — the trivial single-pixel case (no predictor /
//!   transform exercised).
//! * `lossless-32x32-rgb` — opaque RGB, no alpha plane.
//! * `lossless-32x32-rgba` — alpha channel exercised.
//! * `lossless-128x128-natural` — photo-like content stressing the
//!   subtract-green + predictor + LZ77 + Huffman pipeline.
//! * `lossless-color-cache-stress` — colour-cache hit path.
//! * `lossless-color-indexing-paletted` — palette / colour-indexing
//!   transform path.
//! * `lossless-cross-color-active` — cross-colour transform exercised.
//!
//! Each `expected.png` file in this corpus is libwebp-cwebp output
//! decoded back to PNG with `dwebp`, so every test here doubles as a
//! libwebp interop assertion: if our decoder agrees with libwebp on
//! the bits, our entropy coder + transforms + colour-cache walker are
//! all spec-correct.

use oxideav_png::decode_png_to_frame;
use oxideav_webp::decode_webp;

struct Fixture {
    name: &'static str,
    webp: &'static [u8],
    png: &'static [u8],
}

const LOSSLESS_1X1: Fixture = Fixture {
    name: "lossless-1x1",
    webp: include_bytes!("fixtures/lossless_corpus/lossless-1x1/input.webp"),
    png: include_bytes!("fixtures/lossless_corpus/lossless-1x1/expected.png"),
};

const LOSSLESS_32X32_RGB: Fixture = Fixture {
    name: "lossless-32x32-rgb",
    webp: include_bytes!("fixtures/lossless_corpus/lossless-32x32-rgb/input.webp"),
    png: include_bytes!("fixtures/lossless_corpus/lossless-32x32-rgb/expected.png"),
};

const LOSSLESS_32X32_RGBA: Fixture = Fixture {
    name: "lossless-32x32-rgba",
    webp: include_bytes!("fixtures/lossless_corpus/lossless-32x32-rgba/input.webp"),
    png: include_bytes!("fixtures/lossless_corpus/lossless-32x32-rgba/expected.png"),
};

const LOSSLESS_128X128_NATURAL: Fixture = Fixture {
    name: "lossless-128x128-natural",
    webp: include_bytes!("fixtures/lossless_corpus/lossless-128x128-natural/input.webp"),
    png: include_bytes!("fixtures/lossless_corpus/lossless-128x128-natural/expected.png"),
};

const LOSSLESS_COLOR_CACHE_STRESS: Fixture = Fixture {
    name: "lossless-color-cache-stress",
    webp: include_bytes!("fixtures/lossless_corpus/lossless-color-cache-stress/input.webp"),
    png: include_bytes!("fixtures/lossless_corpus/lossless-color-cache-stress/expected.png"),
};

const LOSSLESS_COLOR_INDEXING_PALETTED: Fixture = Fixture {
    name: "lossless-color-indexing-paletted",
    webp: include_bytes!("fixtures/lossless_corpus/lossless-color-indexing-paletted/input.webp"),
    png: include_bytes!("fixtures/lossless_corpus/lossless-color-indexing-paletted/expected.png"),
};

const LOSSLESS_CROSS_COLOR_ACTIVE: Fixture = Fixture {
    name: "lossless-cross-color-active",
    webp: include_bytes!("fixtures/lossless_corpus/lossless-cross-color-active/input.webp"),
    png: include_bytes!("fixtures/lossless_corpus/lossless-cross-color-active/expected.png"),
};

const FIXTURES: &[Fixture] = &[
    LOSSLESS_1X1,
    LOSSLESS_32X32_RGB,
    LOSSLESS_32X32_RGBA,
    LOSSLESS_128X128_NATURAL,
    LOSSLESS_COLOR_CACHE_STRESS,
    LOSSLESS_COLOR_INDEXING_PALETTED,
    LOSSLESS_CROSS_COLOR_ACTIVE,
];

/// Decode an `expected.png` reference into a tight RGBA buffer. Mirrors
/// the helper in `lossy_corpus.rs` — kept as a free function so each
/// test crate can be its own translation unit (cargo doesn't share
/// integration-test code by default).
fn decode_reference_png(png: &[u8]) -> (u32, u32, Vec<u8>) {
    let vf = decode_png_to_frame(png, None).expect("expected.png must decode");
    assert_eq!(
        vf.planes.len(),
        1,
        "PNG decoder must return a single packed plane"
    );
    let plane = &vf.planes[0];
    let (w, h, bpp) = parse_png_ihdr(png);
    let stride = (w as usize) * bpp;
    assert_eq!(
        plane.stride, stride,
        "PNG plane stride {} disagrees with IHDR {}×{}×{}bpp",
        plane.stride, w, h, bpp
    );
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    match bpp {
        3 => {
            for px in plane.data.chunks_exact(3) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        4 => {
            rgba.extend_from_slice(&plane.data);
        }
        other => panic!("PNG: unexpected bpp {other} in lossless fixture corpus"),
    }
    assert_eq!(rgba.len(), (w * h * 4) as usize);
    (w, h, rgba)
}

fn parse_png_ihdr(png: &[u8]) -> (u32, u32, usize) {
    assert!(png.len() >= 26, "PNG too short for IHDR");
    assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n", "PNG signature mismatch");
    assert_eq!(&png[12..16], b"IHDR", "first chunk must be IHDR");
    let w = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let h = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    let bit_depth = png[24];
    let colour_type = png[25];
    assert_eq!(bit_depth, 8, "lossless-corpus PNGs are all 8-bit");
    let bpp = match colour_type {
        2 => 3, // RGB
        6 => 4, // RGBA
        other => panic!("unexpected PNG colour_type {other} in lossless fixture corpus"),
    };
    (w, h, bpp)
}

fn run_one(fix: &Fixture) {
    eprintln!("--- fixture: {} ---", fix.name);
    eprintln!(
        "    input.webp = {} B   expected.png = {} B",
        fix.webp.len(),
        fix.png.len()
    );

    let (w, h, expected) = decode_reference_png(fix.png);
    eprintln!("    expected: {}×{} ({} RGBA bytes)", w, h, expected.len());

    let img = decode_webp(fix.webp).unwrap_or_else(|e| {
        panic!("{}: decode_webp failed: {e}", fix.name);
    });
    assert!(
        !img.frames.is_empty(),
        "{}: decode_webp returned 0 frames",
        fix.name
    );
    assert_eq!(
        (img.width, img.height),
        (w, h),
        "{}: dimensions mismatch (webp={}×{} png={}×{})",
        fix.name,
        img.width,
        img.height,
        w,
        h
    );
    let actual = &img.frames[0].rgba;
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: RGBA buffer length mismatch (actual={} expected={})",
        fix.name,
        actual.len(),
        expected.len()
    );

    if actual != &expected {
        // Find the first divergent pixel and surface it in the panic
        // message so the failure log is debuggable without re-running.
        for (idx, (a, e)) in actual
            .chunks_exact(4)
            .zip(expected.chunks_exact(4))
            .enumerate()
        {
            if a != e {
                let x = idx % w as usize;
                let y = idx / w as usize;
                panic!(
                    "{}: pixel divergence at ({x},{y}) idx={idx}: \
                     actual={:?} expected={:?}",
                    fix.name, a, e
                );
            }
        }
        unreachable!("buffers differ but no per-pixel divergence found");
    }
    eprintln!("    BIT-EXACT match");
}

#[test]
fn lossless_corpus_pixel_correctness() {
    for fix in FIXTURES {
        run_one(fix);
    }
}
