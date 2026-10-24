//! `EncodeOptions::method` on the lossy (`VP8 `) path (#47).
//!
//! Methods `0..=2` turn the VP8 trellis quantiser off (faster, a few
//! percent more bytes); `3..=6` keep the quality-mapped trellis, so the
//! default (`4`) writes exactly what earlier versions wrote (pinned in
//! `lossy_output_pins.rs`). The alpha plane follows the lossless levels
//! and, on larger images, is coded on a second thread next to the VP8
//! encode.

mod common;

use oxideav_webp::{EncodeOptions, decode, encode_rgba8};

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

fn rgb_of(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect()
}

#[test]
fn lossy_methods_split_into_trellis_off_and_trellis_on() {
    let (w, h) = (160u32, 112u32);
    let rgba = common::photo_rgba(w, h);
    let enc = |m: u8| {
        encode_rgba8(
            w,
            h,
            &rgba,
            &EncodeOptions::default().with_quality(80.0).with_method(m),
        )
        .expect("lossy encode")
    };
    let files: Vec<Vec<u8>> = (0..=6).map(enc).collect();
    for m in 1..=2 {
        assert_eq!(
            files[m], files[0],
            "methods 0..=2 share the trellis-off encode"
        );
    }
    for m in 3..=6 {
        assert_eq!(files[m], files[4], "methods 3..=6 share the default encode");
    }
    assert_ne!(files[0], files[4], "trellis off changes the bitstream");
    let default = encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_quality(80.0))
        .expect("lossy encode");
    assert_eq!(default, files[4], "the default lossy encode is method 4");

    let src = rgb_of(&rgba);
    for (m, file) in [(0usize, &files[0]), (4, &files[4])] {
        let back = rgb_of(&decode(file).expect("decode").to_rgba8());
        let p = psnr(&back, &src);
        assert!(p > 30.0, "method {m}: PSNR {p:.2} dB");
    }
}

/// An alpha plane big enough for the concurrent `ALPH` encode stays
/// exact at every level, and the file decodes like the sequential path
/// would write it (same bytes on repeat).
#[test]
fn concurrent_alpha_plane_is_exact_and_deterministic() {
    let (w, h) = (128u32, 96u32);
    let mut rgba = common::photo_rgba(w, h);
    for (i, px) in rgba.chunks_mut(4).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let d = (x as i32 - 64).pow(2) + (y as i32 - 48).pow(2);
        px[3] = if d < 1600 { 255 } else { (x * 2) as u8 };
    }
    let alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
    for m in [0u8, 4, 6] {
        let opts = EncodeOptions::default().with_quality(75.0).with_method(m);
        let a = encode_rgba8(w, h, &rgba, &opts).expect("encode");
        let b = encode_rgba8(w, h, &rgba, &opts).expect("encode");
        assert_eq!(a, b, "method {m}: output is deterministic");
        let back: Vec<u8> = decode(&a)
            .expect("decode")
            .to_rgba8()
            .chunks(4)
            .map(|p| p[3])
            .collect();
        assert_eq!(back, alpha, "method {m}: alpha plane exact");
    }
}
