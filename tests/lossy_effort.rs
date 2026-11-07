//! `EncodeOptions::method` on the lossy (`VP8 `) path (#47).
//!
//! Methods `3..=4` write the single-pass encode of earlier versions (the
//! default, pinned in `lossy_output_pins.rs`); `5` fits the RFC 6386
//! §13.4 token-probability updates to the picture (never larger than
//! `4`); `6` also picks the §9.4 loop-filter level by reconstruction
//! error. `0..=2` turn the trellis quantiser off at quality 90 and above
//! and encode like `4` below it. The alpha plane follows the lossless
//! levels and, on larger images, is coded on a second thread next to the
//! VP8 encode.

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

/// Every method's lossy encode at `quality`, methods `0..=6`.
fn encode_every_method(w: u32, h: u32, rgba: &[u8], quality: f32) -> Vec<Vec<u8>> {
    (0..=6u8)
        .map(|m| {
            encode_rgba8(
                w,
                h,
                rgba,
                &EncodeOptions::default()
                    .with_quality(quality)
                    .with_method(m),
            )
            .expect("lossy encode")
        })
        .collect()
}

#[test]
fn lossy_method_ladder() {
    let (w, h) = (160u32, 112u32);
    let rgba = common::photo_rgba(w, h);
    let src = rgb_of(&rgba);
    let default = encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_quality(80.0))
        .expect("lossy encode");

    // Below quality 90 the low levels keep the trellis: 0..=4 are one
    // encode, the default.
    let q80 = encode_every_method(w, h, &rgba, 80.0);
    for m in 0..=3 {
        assert_eq!(q80[m], q80[4], "q80: method {m} encodes like 4");
    }
    assert_eq!(default, q80[4], "the default lossy encode is method 4");
    // 5 fits the §13.4 probabilities: a different wire, never larger.
    assert_ne!(q80[5], q80[4], "q80: method 5 refits the probabilities");
    assert!(q80[5].len() <= q80[4].len(), "q80: method 5 larger than 4");
    // 6 adds the loop-filter search: same coefficients as 5 here, so the
    // same size class, and a reconstruction at least as close.
    let p = |file: &[u8]| psnr(&rgb_of(&decode(file).expect("decode").to_rgba8()), &src);
    assert!(
        p(&q80[6]) >= p(&q80[5]) - 0.05,
        "q80: method 6 {:.2} dB against method 5 {:.2} dB",
        p(&q80[6]),
        p(&q80[5])
    );
    for (m, file) in q80.iter().enumerate() {
        let db = p(file);
        assert!(db > 30.0, "q80 method {m}: PSNR {db:.2} dB");
    }

    // From quality 90, 0..=2 turn the trellis off.
    let q95 = encode_every_method(w, h, &rgba, 95.0);
    for m in 1..=2 {
        assert_eq!(q95[m], q95[0], "q95: methods 0..=2 share one encode");
    }
    assert_eq!(q95[3], q95[4], "q95: method 3 encodes like 4");
    assert_ne!(q95[0], q95[4], "q95: trellis off changes the bitstream");
    for (m, file) in q95.iter().enumerate() {
        let db = p(file);
        assert!(db > 33.0, "q95 method {m}: PSNR {db:.2} dB");
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
    for m in [0u8, 4, 5, 6] {
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
