//! The lossless (`VP8L`) encoder's `method` knob.
//!
//! * Methods `0..=5` take the single-pass path: the transform stack is
//!   chosen from histogram cost estimates and the image is encoded once.
//! * Method `6` (the default) runs the exhaustive search; anything above
//!   behaves as `6`.

mod common;

use oxideav_webp::{decode_all, decode_rgba8, encode_rgba8, EncodeOptions};

#[test]
fn method_defaults_to_6_and_with_method_sets_it() {
    assert_eq!(EncodeOptions::default().method, 6);
    assert_eq!(EncodeOptions::new().with_method(4).method, 4);
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
