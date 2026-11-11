//! `EncodeOptions::method` and `threads` on the lossy (`VP8 `) path (#47).
//!
//! Methods `0`, `1`, `2` are the three reduced macroblock mode searches;
//! `3..=4` write the exhaustive single-pass encode of earlier versions
//! (the default, pinned in `lossy_output_pins.rs`); `5` fits the RFC 6386
//! §13.4 token-probability updates to the picture (never larger than
//! `4`); `6` also picks the §9.4 loop-filter level by reconstruction
//! error. The bytes are the same for every thread budget. The alpha
//! plane is always exact: at methods `0..=2` it goes through the quick
//! `ALPH` coder (a §2.7.1.2 filter and a single-candidate lossless
//! stream), above them through the lossless search at the same level;
//! on larger images it is coded on its own threads next to the VP8
//! encode.

mod common;

use oxideav_webp::{DecodeOptions, EncodeOptions, decode, decode_with, encode_rgba8};

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

    // 0, 1, 2 are three different reduced searches; 3 and 4 are one
    // encode, the default.
    let q80 = encode_every_method(w, h, &rgba, 80.0);
    for m in 0..=2 {
        assert_ne!(q80[m], q80[4], "q80: method {m} is a reduced search");
        assert_ne!(q80[m], q80[m + 1], "q80: methods {m} and {} differ", m + 1);
    }
    assert_eq!(q80[3], q80[4], "q80: method 3 encodes like 4");
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

    // The same ladder at a fine quantiser.
    let q95 = encode_every_method(w, h, &rgba, 95.0);
    assert_eq!(q95[3], q95[4], "q95: method 3 encodes like 4");
    assert_ne!(q95[0], q95[4], "q95: method 0 is a reduced search");
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
    for m in 0..=6u8 {
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

/// The thread budget never changes a lossy file: every method, with and
/// without alpha, on a picture whose VP8 rows run as a wavefront, and
/// the fastest and default methods on one large enough for the RGB
/// conversion to split into bands too.
#[test]
fn lossy_bytes_do_not_depend_on_the_thread_budget() {
    for (w, h, methods) in [
        (208u32, 136u32, &[0u8, 1, 2, 3, 4, 5, 6][..]),
        (656, 420, &[0, 4][..]),
    ] {
        let opaque = common::photo_rgba(w, h);
        let mut translucent = opaque.clone();
        for (i, px) in translucent.chunks_mut(4).enumerate() {
            px[3] = (i % 251) as u8;
        }
        for (name, rgba) in [("opaque", &opaque), ("alpha", &translucent)] {
            if name == "alpha" && methods.len() < 7 {
                // The alpha plane is its own (lossless) encode; the
                // small picture covers it.
                continue;
            }
            for &method in methods {
                let opts = EncodeOptions::default()
                    .with_quality(80.0)
                    .with_method(method);
                let serial =
                    encode_rgba8(w, h, rgba, &opts.clone().with_threads(1)).expect("encode");
                for threads in [2usize, 5, 0] {
                    let file = encode_rgba8(w, h, rgba, &opts.clone().with_threads(threads))
                        .expect("encode");
                    assert!(
                        file == serial,
                        "{w}x{h} {name}, method {method}: {threads} threads changed the bytes"
                    );
                }
            }
        }
    }
}

/// `with_effort` is `with_method` under the registry option's name, and
/// the RGB entry point writes what the image entry point writes.
#[test]
fn effort_alias_and_entry_points_agree() {
    let (w, h) = (96u32, 64u32);
    let rgba = common::photo_rgba(w, h);
    let rgb = rgb_of(&rgba);
    for level in [0u8, 2, 4] {
        let by_method = EncodeOptions::default()
            .with_quality(70.0)
            .with_method(level);
        let by_effort = EncodeOptions::default()
            .with_quality(70.0)
            .with_effort(level);
        assert_eq!(by_method, by_effort);
        let from_rgba = encode_rgba8(w, h, &rgba, &by_method).expect("encode");
        let from_rgb = oxideav_webp::encode_rgb8(w, h, &rgb, &by_method).expect("encode");
        let image = oxideav_webp::WebpImage::from_rgb8(w, h, rgb.clone()).expect("image");
        let from_image = oxideav_webp::encode(&image, &by_method).expect("encode");
        assert_eq!(
            from_rgba, from_rgb,
            "level {level}: opaque RGBA writes the RGB file"
        );
        assert_eq!(
            from_rgb, from_image,
            "level {level}: slice and image entry points"
        );
    }
}

/// `with_alpha_method` sets the alpha plane's effort alone: the `VP8 `
/// chunk is the one `method` writes, the `ALPH` chunk the one the alpha
/// level writes, and the plane is exact.
#[test]
fn alpha_method_is_independent_of_the_picture_method() {
    let (w, h) = (120u32, 88u32);
    let mut rgba = common::photo_rgba(w, h);
    for (i, px) in rgba.chunks_mut(4).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let d = (x as i32 - 60).pow(2) + (y as i32 - 44).pow(2);
        px[3] = (255 - (d / 12).min(255)) as u8;
    }
    let chunk = |file: &[u8], tag: &[u8; 4]| -> Vec<u8> {
        let at = file
            .windows(4)
            .position(|c| c == tag)
            .unwrap_or_else(|| panic!("no {} chunk", String::from_utf8_lossy(tag)));
        let len = u32::from_le_bytes(file[at + 4..at + 8].try_into().unwrap()) as usize;
        file[at + 8..at + 8 + len].to_vec()
    };
    let encode = |opts: &EncodeOptions| encode_rgba8(w, h, &rgba, opts).expect("encode");
    let base = EncodeOptions::default().with_quality(80.0);
    let default = encode(&base);
    let quick_everything = encode(&base.clone().with_method(2));
    let quick_alpha = encode(&base.clone().with_alpha_method(2));
    assert_eq!(
        chunk(&quick_alpha, b"VP8 "),
        chunk(&default, b"VP8 "),
        "the picture keeps the default bytes"
    );
    assert_eq!(
        chunk(&quick_alpha, b"ALPH"),
        chunk(&quick_everything, b"ALPH"),
        "the alpha plane is the quick coder's"
    );
    assert_ne!(chunk(&quick_alpha, b"ALPH"), chunk(&default, b"ALPH"));
    // The other way round: a quick picture with a searched alpha plane.
    let searched_alpha = encode(&base.clone().with_method(2).with_alpha_effort(4));
    assert_eq!(chunk(&searched_alpha, b"ALPH"), chunk(&default, b"ALPH"));
    assert_eq!(
        chunk(&searched_alpha, b"VP8 "),
        chunk(&quick_everything, b"VP8 ")
    );
    // Naming the level `method` already implies changes nothing.
    assert_eq!(encode(&base.clone().with_alpha_method(4)), default);
    let alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
    for file in [&default, &quick_alpha, &searched_alpha] {
        let back: Vec<u8> = decode(file)
            .expect("decode")
            .to_rgba8()
            .chunks(4)
            .map(|p| p[3])
            .collect();
        assert_eq!(back, alpha);
    }
    // A lossless encode has no separate alpha plane to tune.
    let lossless = EncodeOptions::default();
    assert_eq!(
        encode(&lossless.clone().with_alpha_method(0)),
        encode(&lossless)
    );
}

/// `DecodeOptions::threads`: the alpha plane decoded next to the VP8
/// frame gives the image the serial decode gives, and a damaged `ALPH`
/// chunk the same error.
#[test]
fn threaded_decode_matches_the_serial_decode() {
    let (w, h) = (176u32, 144u32);
    let mut rgba = common::photo_rgba(w, h);
    for (i, px) in rgba.chunks_mut(4).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        px[3] = (x * 3 / 2) as u8 ^ (y / 4) as u8;
    }
    for method in [0u8, 4] {
        let opts = EncodeOptions::default()
            .with_quality(80.0)
            .with_method(method);
        let mut file = encode_rgba8(w, h, &rgba, &opts).expect("encode");
        let serial = decode(&file).expect("decode");
        assert_eq!(serial.format, oxideav_webp::PixelFormat::Yuva420P);
        for threads in [2usize, 8, 0] {
            let threaded = decode_with(&file, &DecodeOptions::default().with_threads(threads))
                .expect("threaded decode");
            assert!(threaded == serial, "method {method}, {threads} threads");
        }
        // An `ALPH` info byte naming a reserved compression method.
        let at = file
            .windows(4)
            .position(|c| c == b"ALPH")
            .expect("ALPH chunk");
        file[at + 8] = (file[at + 8] & !3) | 2;
        let serial_error = decode(&file).expect_err("reserved compression");
        let threaded_error = decode_with(&file, &DecodeOptions::default().with_threads(4))
            .expect_err("reserved compression");
        assert_eq!(serial_error.to_string(), threaded_error.to_string());
    }
}
