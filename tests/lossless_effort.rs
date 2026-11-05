//! `EncodeOptions::method`: the lossless effort level (#45).
//!
//! Every level writes an exactly lossless file; method 6 is the
//! exhaustive search (its bytes are pinned by `lossless_output_pins.rs`),
//! levels 0..=5 screen fewer candidates with a cheaper parse. These tests
//! pin the round trip at every level over varied content, bound how much
//! larger the default level writes than method 6, and check that the
//! level reaches the alpha plane of a lossy encode and animation frames.

mod common;

use oxideav_webp::{
    AnimFrame, AnimFrameMode, EncodeOptions, Metadata, decode, decode_all, encode_animation_frames,
    encode_rgba8,
};

fn fixture_frames() -> Vec<(String, u32, u32, Vec<u8>)> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("tests/data")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "webp"))
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|p| {
            let frames = decode_all(&std::fs::read(&p).expect("read fixture")).expect("decode");
            let img = &frames[0].image;
            (
                p.file_name().expect("name").to_string_lossy().into_owned(),
                img.width(),
                img.height(),
                img.to_rgba8(),
            )
        })
        .collect()
}

/// Inputs beyond the fixtures: photo-like, a gradient with alpha, a
/// 7-colour palette image and odd tiny sizes (no predictor block fits).
fn synthetic_inputs() -> Vec<(String, u32, u32, Vec<u8>)> {
    let mut out = vec![(
        "photo 96x80".to_string(),
        96,
        80,
        common::photo_rgba(96, 80),
    )];
    let (w, h) = (45u32, 33u32);
    let mut grad = Vec::new();
    let mut pal = Vec::new();
    for y in 0..h {
        for x in 0..w {
            grad.extend_from_slice(&[
                (x * 5) as u8,
                (y * 7) as u8,
                ((x + y) * 3) as u8,
                (x * 3 + y) as u8,
            ]);
            let c = ((x / 5 + y / 4) % 7) as u8;
            pal.extend_from_slice(&[c * 36, 255 - c * 30, c * 11, 255]);
        }
    }
    out.push(("gradient+alpha 45x33".into(), w, h, grad));
    out.push(("palette 45x33".into(), w, h, pal));
    out.push(("1x1".into(), 1, 1, vec![9, 8, 7, 6]));
    out.push((
        "7x3".into(),
        7,
        3,
        (0..7 * 3 * 4).map(|i| (i * 37) as u8).collect(),
    ));
    out
}

#[test]
fn every_method_round_trips_exactly() {
    for (name, w, h, rgba) in fixture_frames().into_iter().chain(synthetic_inputs()) {
        for method in 0..=6u8 {
            let opts = EncodeOptions::default().with_method(method);
            let file = encode_rgba8(w, h, &rgba, &opts).expect("encode");
            let back = decode(&file).expect("decode").to_rgba8();
            assert!(back == rgba, "{name}: method {method} is not lossless");
        }
    }
}

/// The default level stays close to the exhaustive search: within 10 %
/// plus 8 bytes of method 6 on every input (measured: at most +8.7 %,
/// 50 bytes against 46 on a 32 x 32 fixture; most within 1 %).
#[test]
fn default_method_size_stays_near_method_6() {
    for (name, w, h, rgba) in fixture_frames().into_iter().chain(synthetic_inputs()) {
        let default = encode_rgba8(w, h, &rgba, &EncodeOptions::default()).expect("encode");
        let max =
            encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_method(6)).expect("encode");
        let bound = max.len() + max.len() / 10 + 8;
        assert!(
            default.len() <= bound,
            "{name}: default {} bytes against method 6 {} (bound {bound})",
            default.len(),
            max.len()
        );
    }
}

#[test]
fn method_option_defaults_to_4_and_clamps_to_6() {
    assert_eq!(EncodeOptions::default().method, 4);
    assert_eq!(EncodeOptions::new().with_method(0).method, 0);
    assert_eq!(EncodeOptions::new().with_method(6).method, 6);
    assert_eq!(EncodeOptions::new().with_method(200).method, 6);
}

/// A lossy encode's `ALPH` plane is a lossless image: it follows the
/// method too, and stays exact at every level.
#[test]
fn alpha_plane_follows_the_method_and_stays_exact() {
    let (w, h) = (64u32, 48u32);
    let mut rgba = common::photo_rgba(w, h);
    for (i, px) in rgba.chunks_mut(4).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        px[3] = ((x * 4) ^ (y * 5)) as u8;
    }
    let alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
    let mut sizes = Vec::new();
    for method in [0u8, 4, 6] {
        let opts = EncodeOptions::default()
            .with_quality(80.0)
            .with_method(method);
        let file = encode_rgba8(w, h, &rgba, &opts).expect("lossy encode");
        let back: Vec<u8> = decode(&file)
            .expect("decode")
            .to_rgba8()
            .chunks(4)
            .map(|p| p[3])
            .collect();
        assert_eq!(back, alpha, "method {method}: alpha plane not exact");
        sizes.push(file.len());
    }
    assert_ne!(
        sizes[0], sizes[2],
        "the method did not reach the alpha plane"
    );
}

/// Animation frames are lossless images: every level decodes back to
/// the source frames.
#[test]
fn animation_frames_follow_the_method_and_stay_exact() {
    let frames: Vec<AnimFrame> = (0..3u32)
        .map(|i| {
            let mut px = common::photo_rgba(40, 32);
            for (k, p) in px.chunks_mut(4).enumerate() {
                if (k as u32 % 40) / 8 == i {
                    p[0] = 250;
                }
            }
            let mut f = AnimFrame::new(40, 32, px, 40);
            f.mode = AnimFrameMode::Lossless;
            f
        })
        .collect();
    for method in [0u8, 3, 6] {
        let opts = EncodeOptions::default().with_method(method);
        let file =
            encode_animation_frames(&frames, &Metadata::default(), &opts).expect("animation");
        let back = decode_all(&file).expect("decode");
        assert_eq!(back.len(), frames.len());
        for (b, f) in back.iter().zip(&frames) {
            assert_eq!(b.image.to_rgba8(), f.pixels, "method {method}");
        }
    }
}

/// `EncodeOptions::threads` never changes the output: every level, on
/// content with and without a palette and with alpha, writes the same
/// bytes serially and on several worker budgets (the parallel stages
/// fold their results in a fixed order).
#[test]
fn output_is_identical_for_every_thread_budget() {
    let photo = common::photo_rgba(96, 72);
    let mut alpha = common::photo_rgba(64, 48);
    for (k, p) in alpha.chunks_mut(4).enumerate() {
        p[3] = (k * 7 % 256) as u8;
    }
    let mut pal = Vec::new();
    for i in 0..48 * 40u32 {
        let c = ((i % 48) / 6 + (i / 48) / 5) % 5;
        pal.extend_from_slice(&[(c * 50) as u8, (c * 20) as u8, 255 - (c * 40) as u8, 255]);
    }
    let inputs = [(96u32, 72u32, &photo), (64, 48, &alpha), (48, 40, &pal)];
    for method in 0..=6u8 {
        for &(w, h, px) in &inputs {
            let serial = encode_rgba8(
                w,
                h,
                px,
                &EncodeOptions::default().with_method(method).with_threads(1),
            )
            .expect("serial encode");
            for threads in [2usize, 3, 8, 0] {
                let opts = EncodeOptions::default()
                    .with_method(method)
                    .with_threads(threads);
                let par = encode_rgba8(w, h, px, &opts).expect("parallel encode");
                assert_eq!(par, serial, "method {method} {w}x{h} threads {threads}");
            }
            assert_eq!(decode(&serial).expect("decode").to_rgba8(), *px);
        }
    }
    // The alpha plane of a lossy encode, coded next to the VP8 encode
    // when two or more workers are allowed.
    let mut big = common::photo_rgba(80, 80);
    for (k, p) in big.chunks_mut(4).enumerate() {
        p[3] = if k % 80 < 40 { 255 } else { (k / 80 * 3) as u8 };
    }
    let lossy = |threads| {
        encode_rgba8(
            80,
            80,
            &big,
            &EncodeOptions::default()
                .with_quality(75.0)
                .with_threads(threads),
        )
        .expect("lossy encode")
    };
    assert_eq!(lossy(1), lossy(2));
}

/// `encode_rgb8` / `encode_rgba8` read the caller's buffer in place on
/// the lossless path: the bytes and every rejection match encoding the
/// equivalent `WebpImage` (zero dimension, short buffer, oversize side;
/// a longer buffer is accepted and its tail ignored).
#[test]
fn packed_lossless_entry_points_match_the_image_path() {
    use oxideav_webp::{WebpImage, encode, encode_rgb8};
    let opts = EncodeOptions::default().with_threads(1);
    let rgba = common::photo_rgba(37, 21);
    let rgb: Vec<u8> = rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let via_image = |w, h, data: &[u8], bpp| {
        let img = if bpp == 4 {
            WebpImage::from_rgba8(w, h, data.to_vec())
        } else {
            WebpImage::from_rgb8(w, h, data.to_vec())
        };
        img.and_then(|img| encode(&img, &opts))
            .map_err(|e| e.to_string())
    };
    let packed = |w, h, data: &[u8], bpp| {
        if bpp == 4 {
            encode_rgba8(w, h, data, &opts)
        } else {
            encode_rgb8(w, h, data, &opts)
        }
        .map_err(|e| e.to_string())
    };
    let mut long = rgba.clone();
    long.extend_from_slice(&[9; 11]);
    let cases: Vec<(u32, u32, &[u8], usize)> = vec![
        (37, 21, &rgba, 4),
        (37, 21, &rgb, 3),
        (37, 21, &long, 4),
        (0, 21, &rgba, 4),
        (37, 0, &rgb, 3),
        (37, 22, &rgba, 4),
        (38, 21, &rgb, 3),
        (16_384, 1, &[0u8; 16_384 * 4], 4),
    ];
    for (w, h, data, bpp) in cases {
        assert_eq!(
            packed(w, h, data, bpp),
            via_image(w, h, data, bpp),
            "{w}x{h} bpp {bpp}"
        );
    }
}
