//! The lossless (`VP8L`) encoder's `method` knob.
//!
//! * Methods `0..=5` (default `4`) take the single-pass path: the
//!   transforms and the colour cache are chosen from histogram cost
//!   estimates and the image is encoded once.
//! * Method `6` runs the exhaustive search; anything above behaves as `6`.
//! * The default must stay within a measured bound of the exhaustive
//!   output's size on every still image the output pins use, plus a
//!   128 x 128 photo.

mod common;

use oxideav_webp::{decode_all, decode_rgba8, encode_rgba8, EncodeOptions};

#[test]
fn method_defaults_to_4_and_with_method_sets_it() {
    assert_eq!(EncodeOptions::default().method, 4);
    assert_eq!(EncodeOptions::new().with_method(6).method, 6);
    assert_eq!(EncodeOptions::new().with_method(0).method, 0);
}

#[test]
fn methods_above_6_behave_as_6() {
    let rgba = common::photo_rgba(24, 20);
    let six =
        encode_rgba8(24, 20, &rgba, &EncodeOptions::default().with_method(6)).expect("method 6");
    for method in [7u8, 200, u8::MAX] {
        let out = encode_rgba8(24, 20, &rgba, &EncodeOptions::default().with_method(method))
            .expect("encode");
        assert_eq!(out, six, "method {method} must match method 6");
    }
}

#[test]
fn every_method_round_trips() {
    // Odd sizes exercise partial predictor blocks and the right-edge column.
    let (w, h) = (37u32, 29u32);
    let mut rgba = common::photo_rgba(w, h);
    // Give some pixels non-opaque alpha so the alpha channel is coded too.
    for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
        if i % 7 == 0 {
            px[3] = (i % 251) as u8;
        }
    }
    for method in 0..=6u8 {
        let opts = EncodeOptions::default().with_method(method);
        let file = encode_rgba8(w, h, &rgba, &opts).expect("encode");
        let back = decode_rgba8(&file).expect("decode");
        assert_eq!(back.data, rgba, "method {method} round trip");
    }
}

/// The payload of the first `fourcc` chunk of a RIFF file.
fn chunk<'a>(file: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 12;
    while pos + 8 <= file.len() {
        let size = u32::from_le_bytes(file[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &file[pos..pos + 4] == fourcc {
            return Some(&file[pos + 8..pos + 8 + size]);
        }
        pos += 8 + size + (size & 1);
    }
    None
}

/// With a quality set, `method` sets the effort of the lossless-coded
/// alpha plane: the `ALPH` chunk carries the plane coded by the lossless
/// encoder at that method (alpha in the green channel of an opaque image,
/// the stream after its 5-byte image header), or the raw plane when that
/// is not smaller.
#[test]
fn method_reaches_the_alpha_plane_of_a_lossy_encode() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/lossy-with-alpha-128x128.webp"
    );
    let frames = decode_all(&std::fs::read(path).expect("fixture")).expect("decode");
    let image = &frames[0].image;
    let (w, h) = (image.width(), image.height());
    let rgba = image.to_rgba8();
    let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
    let carrier: Vec<u8> = alpha.iter().flat_map(|&a| [0, a, 0, 255]).collect();
    for method in [4u8, 6] {
        let opts = EncodeOptions::default().with_method(method);
        let lossy = encode_rgba8(w, h, &rgba, &opts.clone().with_quality(80.0)).expect("lossy");
        let alph = chunk(&lossy, b"ALPH").expect("ALPH chunk");
        let lossless = encode_rgba8(w, h, &carrier, &opts).expect("lossless carrier");
        let stream = &chunk(&lossless, b"VP8L").expect("VP8L chunk")[5..];
        let mut expected = Vec::new();
        if stream.len() < alpha.len() {
            expected.push(0x01);
            expected.extend_from_slice(stream);
        } else {
            expected.push(0x00);
            expected.extend_from_slice(&alpha);
        }
        assert_eq!(alph, &expected[..], "method {method}");
    }
}

/// Largest file-size ratio of the default single-pass path (method 4) to
/// the exhaustive search (method 6) over [`SINGLE_PASS_SIZES`], plus a small
/// margin. Measured worst: 1.087 (+8.7%), `lossless-32x32-rgb.webp` at 50
/// bytes against 46; most inputs come out the same size.
const SIZE_BOUND: f64 = 1.09;

/// Default (method-4) file size of every still image the output pins use,
/// plus the photo at 128 x 128 (23 inputs): each fixture frame
/// (`name#frame`) and each synthetic image. Growth of more than 1% over
/// the recorded size (rounded down, so the small images must not grow at
/// all), or of more than 9% over method 6, fails. Smaller is always fine.
const SINGLE_PASS_SIZES: &[(&str, usize)] = &[
    ("animated-3-frames-rgb.webp#0", 114),
    ("animated-3-frames-rgb.webp#1", 112),
    ("animated-3-frames-rgb.webp#2", 120),
    ("animated-with-alpha.webp#0", 114),
    ("animated-with-alpha.webp#1", 114),
    ("animated-with-alpha.webp#2", 114),
    ("extended-with-exif.webp#0", 12852),
    ("extended-with-icc-profile.webp#0", 12852),
    ("extended-with-xmp.webp#0", 12852),
    ("lossless-128x128-natural.webp#0", 658),
    ("lossless-1x1.webp#0", 32),
    ("lossless-32x32-rgb.webp#0", 50),
    ("lossless-32x32-rgba.webp#0", 58),
    ("lossless-color-cache-stress.webp#0", 158),
    ("lossless-color-indexing-paletted.webp#0", 94),
    ("lossless-cross-color-active.webp#0", 54),
    ("lossy-1x1.webp#0", 32),
    ("lossy-near-lossless-q40.webp#0", 8214),
    ("lossy-with-alpha-128x128.webp#0", 15368),
    ("photo 64", 6964),
    ("photo 96", 15492),
    ("gradient 64", 106),
    ("photo 128", 27394),
];

/// Every input of [`SINGLE_PASS_SIZES`] as `(name, width, height, rgba)`.
fn size_gate_inputs() -> Vec<(String, u32, u32, Vec<u8>)> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("fixture directory")
        .map(|e| {
            e.expect("entry")
                .file_name()
                .into_string()
                .expect("utf-8 name")
        })
        .collect();
    names.sort();
    let mut inputs = Vec::new();
    for name in names {
        let bytes = std::fs::read(format!("{dir}/{name}")).expect("fixture");
        for (i, frame) in decode_all(&bytes)
            .expect("fixture decodes")
            .iter()
            .enumerate()
        {
            let image = &frame.image;
            inputs.push((
                format!("{name}#{i}"),
                image.width(),
                image.height(),
                image.to_rgba8(),
            ));
        }
    }
    for side in [64u32, 96] {
        inputs.push((
            format!("photo {side}"),
            side,
            side,
            common::photo_rgba(side, side),
        ));
    }
    let mut gradient = Vec::new();
    for y in 0..64u32 {
        for x in 0..64u32 {
            gradient.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, ((x ^ y) * 4) as u8, 255]);
        }
    }
    inputs.push(("gradient 64".to_string(), 64, 64, gradient));
    inputs.push((
        "photo 128".to_string(),
        128,
        128,
        common::photo_rgba(128, 128),
    ));
    inputs
}

#[test]
fn default_size_stays_within_its_measured_bound_of_exhaustive() {
    let inputs = size_gate_inputs();
    assert_eq!(
        inputs.len(),
        SINGLE_PASS_SIZES.len(),
        "one recorded size per input"
    );
    let mut worst = (0.0f64, String::new());
    for (name, w, h, rgba) in inputs {
        let fast = encode_rgba8(w, h, &rgba, &EncodeOptions::default()).expect("default encode");
        let best = encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_method(6))
            .expect("method 6 encode");
        let ratio = fast.len() as f64 / best.len() as f64;
        if ratio > worst.0 {
            worst = (ratio, name.clone());
        }
        let &(_, recorded) = SINGLE_PASS_SIZES
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("{name}: no recorded size"));
        assert!(
            fast.len() <= recorded + recorded / 100,
            "{name}: the default wrote {} bytes, recorded {recorded}",
            fast.len()
        );
        assert!(
            ratio <= SIZE_BOUND,
            "{name}: the default's {} bytes are {:+.2}% against method 6's {}",
            fast.len(),
            (ratio - 1.0) * 100.0,
            best.len()
        );
        assert_eq!(decode_rgba8(&fast).expect("decode").data, rgba, "{name}");
    }
    eprintln!(
        "worst size ratio: {:+.2}% on {}",
        (worst.0 - 1.0) * 100.0,
        worst.1
    );
}
