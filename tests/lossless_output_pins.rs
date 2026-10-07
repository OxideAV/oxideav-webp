//! Pins of the lossless (`VP8L`) encoder's output bytes.
//!
//! Every pin is the length and FNV-1a 64 fingerprint of a file the v0.3.1
//! encoder wrote, including lossy files whose alpha plane (`ALPH`) the
//! lossless encoder codes. That encoder's exhaustive search is now
//! `EncodeOptions::method` 6, so the pins are taken at method 6, which
//! must keep reproducing them byte for byte.

mod common;

use oxideav_webp::{
    decode_all, encode_animation, encode_animation_frames, encode_rgba8, AnimFrame, AnimFrameMode,
    EncodeOptions, Metadata,
};

/// `(fixture, frame, length, FNV-1a 64)` of `encode_rgba8` over every
/// frame of every committed fixture, decoded to RGBA.
const FIXTURE_PINS: &[(&str, usize, usize, u64)] = &[
    ("animated-3-frames-rgb.webp", 0, 114, 0x393aff27a8d6dc83),
    ("animated-3-frames-rgb.webp", 1, 112, 0x97197d9c9b6dfb59),
    ("animated-3-frames-rgb.webp", 2, 112, 0xf7f9c0d86de48000),
    ("animated-with-alpha.webp", 0, 114, 0x393aff27a8d6dc83),
    ("animated-with-alpha.webp", 1, 114, 0x403186f83a680620),
    ("animated-with-alpha.webp", 2, 114, 0xb8ca0305ab9ed210),
    ("extended-with-exif.webp", 0, 12852, 0x5562d8d53fa2c459),
    (
        "extended-with-icc-profile.webp",
        0,
        12852,
        0x5562d8d53fa2c459,
    ),
    ("extended-with-xmp.webp", 0, 12852, 0x5562d8d53fa2c459),
    ("lossless-128x128-natural.webp", 0, 644, 0x76bacb83a65950f4),
    ("lossless-1x1.webp", 0, 32, 0x049968a41030e11c),
    ("lossless-32x32-rgb.webp", 0, 46, 0xd033732bc6efabe7),
    ("lossless-32x32-rgba.webp", 0, 58, 0x5f4ac39ee6592166),
    (
        "lossless-color-cache-stress.webp",
        0,
        158,
        0x8a080c28c974f0da,
    ),
    (
        "lossless-color-indexing-paletted.webp",
        0,
        92,
        0x7e134d71277c3164,
    ),
    (
        "lossless-cross-color-active.webp",
        0,
        52,
        0x057bda5f00a26dc7,
    ),
    ("lossy-1x1.webp", 0, 32, 0xd9f4418f2b8f76e2),
    ("lossy-near-lossless-q40.webp", 0, 8192, 0xb873cf9d2ec09844),
    (
        "lossy-with-alpha-128x128.webp",
        0,
        15302,
        0x37d7c15f52fb1d11,
    ),
];

/// The options that reproduce the v0.3.1 encoder: the exhaustive search.
fn pinned() -> EncodeOptions {
    EncodeOptions::default().with_method(6)
}

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn assert_pin(what: &str, out: &[u8], len: usize, hash: u64) {
    assert_eq!(
        (out.len(), common::fnv1a64(out)),
        (len, hash),
        "{what}: lossless output changed"
    );
}

#[test]
fn every_fixture_frame_encodes_to_its_pinned_bytes() {
    for &(name, frame, len, hash) in FIXTURE_PINS {
        let frames = decode_all(&fixture(name)).expect("fixture decodes");
        let image = &frames[frame].image;
        let rgba = image.to_rgba8();
        let out = encode_rgba8(image.width(), image.height(), &rgba, &pinned()).expect("encode");
        assert_pin(&format!("{name} frame {frame}"), &out, len, hash);
    }
}

#[test]
fn synthetic_images_encode_to_their_pinned_bytes() {
    let mut gradient = Vec::new();
    for y in 0..64u32 {
        for x in 0..64u32 {
            gradient.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, ((x ^ y) * 4) as u8, 255]);
        }
    }
    let cases: [(&str, u32, Vec<u8>, usize, u64); 3] = [
        (
            "photo 64",
            64,
            common::photo_rgba(64, 64),
            6964,
            0xd6465eb22d3b33b5,
        ),
        (
            "photo 96",
            96,
            common::photo_rgba(96, 96),
            15492,
            0xbee378c23db30649,
        ),
        ("gradient 64", 64, gradient, 106, 0x7e0b06c8c122cf07),
    ];
    for (name, side, rgba, len, hash) in cases {
        let out = encode_rgba8(side, side, &rgba, &pinned()).expect("encode");
        assert_pin(name, &out, len, hash);
    }
}

/// `(fixture, frame, length, FNV-1a 64)` of `encode_rgba8` at quality 80
/// over every fixture frame with a non-opaque pixel: lossy files whose
/// `ALPH` plane the lossless encoder codes.
const ALPHA_PLANE_PINS: &[(&str, usize, usize, u64)] = &[
    ("animated-with-alpha.webp", 1, 676, 0x3592dd3085494c6d),
    ("animated-with-alpha.webp", 2, 750, 0xf911162ad628de4e),
    ("lossless-32x32-rgba.webp", 0, 200, 0x86a60121cd80db82),
    ("lossy-with-alpha-128x128.webp", 0, 2924, 0x775f3dc1cb8da518),
];

#[test]
fn lossy_alpha_planes_encode_to_their_pinned_bytes() {
    for &(name, frame, len, hash) in ALPHA_PLANE_PINS {
        let frames = decode_all(&fixture(name)).expect("fixture decodes");
        let image = &frames[frame].image;
        let rgba = image.to_rgba8();
        let opts = pinned().with_quality(80.0);
        let out = encode_rgba8(image.width(), image.height(), &rgba, &opts).expect("encode");
        assert_pin(&format!("{name} frame {frame}, lossy"), &out, len, hash);
    }
}

/// A 4-frame 48 x 48 timeline: a red square stepping 8 pixels per frame
/// over a textured background.
fn moving_square(mode: AnimFrameMode) -> Vec<AnimFrame> {
    (0..4u32)
        .map(|i| {
            let mut px = Vec::with_capacity(48 * 48 * 4);
            for y in 0..48u32 {
                for x in 0..48u32 {
                    let inside = x >= 8 * i && x < 8 * i + 12 && (10..22).contains(&y);
                    if inside {
                        px.extend_from_slice(&[255, 40, 40, 255]);
                    } else {
                        px.extend_from_slice(&[
                            (x * 5) as u8,
                            (y * 5) as u8,
                            ((x ^ y) * 3) as u8,
                            255,
                        ]);
                    }
                }
            }
            let mut f = AnimFrame::new(48, 48, px, 50);
            f.mode = mode;
            f
        })
        .collect()
}

#[test]
fn animations_encode_to_their_pinned_bytes() {
    for (name, len, hash) in [
        (
            "animated-3-frames-rgb.webp",
            418usize,
            0x9deec555cd626950u64,
        ),
        ("animated-with-alpha.webp", 422, 0x08b87752979e0c4d),
    ] {
        let frames = decode_all(&fixture(name)).expect("fixture decodes");
        let out = encode_animation(&frames, &pinned()).expect("animation");
        assert_pin(name, &out, len, hash);
    }
    for (mode, len, hash) in [
        (AnimFrameMode::Lossless, 1260usize, 0x6644da39088d76f8u64),
        (AnimFrameMode::Delta, 1038, 0xd2b92d083d100a65),
        (AnimFrameMode::Auto, 1038, 0xd2b92d083d100a65),
    ] {
        let out = encode_animation_frames(&moving_square(mode), &Metadata::default(), &pinned())
            .expect("animation");
        assert_pin(&format!("{mode:?} timeline"), &out, len, hash);
    }
}
