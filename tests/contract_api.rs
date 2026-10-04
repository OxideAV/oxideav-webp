//! The workspace image-crate API contract, exercised standalone (no
//! `oxideav-core`) over the in-crate fixture corpus.
//!
//! Every root item the contract names — `probe` / `info` / `decode` /
//! `decode_with` / `decode_rgb8` / `decode_rgba8` / `decode_all` /
//! `decode_from` / `encode` / `encode_rgb8` / `encode_rgba8` / `encode_to`,
//! `WebpImage` + `RgbImage` / `RgbaImage`, `PixelFormat`, `ImageInfo`,
//! `Frame`, `EncodeOptions`, `DecodeOptions`, `Error` — is compiled
//! against here, and the behaviour rules (native layout, exact `to_rgba8`,
//! lossless round trip, limits before allocation, hostile input never
//! panics) are pinned.

use std::io::Cursor;
use std::time::Duration;

use oxideav_webp::{
    decode, decode_all, decode_all_with, decode_from, decode_rgb8, decode_rgba8, decode_with,
    encode, encode_animation, encode_rgb8, encode_rgba8, encode_to, info, probe, ColorInfo,
    ColorRange, DecodeOptions, EncodeOptions, Error, Frame, ImageInfo, Metadata, PixelFormat,
    Plane, RgbImage, RgbaImage, WebpError, WebpImage,
};

const LOSSLESS_1X1: &[u8] = include_bytes!("data/lossless-1x1.webp");
const LOSSLESS_32X32_RGB: &[u8] = include_bytes!("data/lossless-32x32-rgb.webp");
const LOSSLESS_32X32_RGBA: &[u8] = include_bytes!("data/lossless-32x32-rgba.webp");
const LOSSLESS_NATURAL: &[u8] = include_bytes!("data/lossless-128x128-natural.webp");
const LOSSLESS_PALETTED: &[u8] = include_bytes!("data/lossless-color-indexing-paletted.webp");
const LOSSLESS_CACHE: &[u8] = include_bytes!("data/lossless-color-cache-stress.webp");
const LOSSLESS_CROSS: &[u8] = include_bytes!("data/lossless-cross-color-active.webp");
const NEAR_LOSSLESS: &[u8] = include_bytes!("data/lossy-near-lossless-q40.webp");
const LOSSY_1X1: &[u8] = include_bytes!("data/lossy-1x1.webp");
const LOSSY_ALPHA: &[u8] = include_bytes!("data/lossy-with-alpha-128x128.webp");
const EXT_EXIF: &[u8] = include_bytes!("data/extended-with-exif.webp");
const EXT_ICC: &[u8] = include_bytes!("data/extended-with-icc-profile.webp");
const EXT_XMP: &[u8] = include_bytes!("data/extended-with-xmp.webp");
const ANIM_RGB: &[u8] = include_bytes!("data/animated-3-frames-rgb.webp");
const ANIM_ALPHA: &[u8] = include_bytes!("data/animated-with-alpha.webp");

const ALL: &[(&str, &[u8])] = &[
    ("lossless-1x1", LOSSLESS_1X1),
    ("lossless-32x32-rgb", LOSSLESS_32X32_RGB),
    ("lossless-32x32-rgba", LOSSLESS_32X32_RGBA),
    ("lossless-128x128-natural", LOSSLESS_NATURAL),
    ("lossless-color-indexing-paletted", LOSSLESS_PALETTED),
    ("lossless-color-cache-stress", LOSSLESS_CACHE),
    ("lossless-cross-color-active", LOSSLESS_CROSS),
    ("lossy-near-lossless-q40", NEAR_LOSSLESS),
    ("lossy-1x1", LOSSY_1X1),
    ("lossy-with-alpha-128x128", LOSSY_ALPHA),
    ("extended-with-exif", EXT_EXIF),
    ("extended-with-icc-profile", EXT_ICC),
    ("extended-with-xmp", EXT_XMP),
    ("animated-3-frames-rgb", ANIM_RGB),
    ("animated-with-alpha", ANIM_ALPHA),
];

// ───────────────────────────── signatures ────────────────────────────────

#[test]
fn contract_root_signatures_compile() {
    let _: fn(&[u8]) -> bool = probe;
    let _: fn(&[u8]) -> Result<ImageInfo, Error> = info;
    let _: fn(&[u8]) -> Result<WebpImage, Error> = decode;
    let _: fn(&[u8], &DecodeOptions) -> Result<WebpImage, Error> = decode_with;
    let _: fn(&[u8]) -> Result<RgbImage, Error> = decode_rgb8;
    let _: fn(&[u8]) -> Result<RgbaImage, Error> = decode_rgba8;
    let _: fn(&[u8]) -> Result<Vec<Frame>, Error> = decode_all;
    let _: fn(Cursor<Vec<u8>>) -> Result<WebpImage, Error> = decode_from::<Cursor<Vec<u8>>>;
    let _: fn(&WebpImage, &EncodeOptions) -> Result<Vec<u8>, Error> = encode;
    type RawEncode = fn(u32, u32, &[u8], &EncodeOptions) -> Result<Vec<u8>, Error>;
    let _: RawEncode = encode_rgb8;
    let _: RawEncode = encode_rgba8;
    let _: fn(&WebpImage, &EncodeOptions, Vec<u8>) -> Result<(), Error> = encode_to::<Vec<u8>>;
    // `Error` is the alias of the one crate error; `PixelFormat` of the
    // crate's pixel-format enum.
    let e: Error = WebpError::invalid("x");
    assert!(e.is_invalid_data());
    let _: PixelFormat = oxideav_webp::WebpPixelFormat::Rgba;
    // Options are `Default` + `with_*`.
    let _ = DecodeOptions::default()
        .with_max_width(Some(1))
        .with_max_height(None)
        .with_max_pixels(Some(1))
        .with_max_bytes(None)
        .with_strict(true);
    let _ = EncodeOptions::default().with_quality(80.0).with_lossless();
}

#[test]
fn error_variants_and_traits() {
    fn assert_error<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error::<Error>();
    let variants = [
        WebpError::invalid("a"),
        WebpError::unsupported("b"),
        WebpError::limit("c"),
        WebpError::Io(std::io::Error::other("d")),
        WebpError::Eof,
        WebpError::NeedMore,
    ];
    for v in &variants {
        assert!(!v.to_string().is_empty());
        assert_eq!(v.clone(), *v);
    }
    assert!(matches!(variants[0], WebpError::InvalidData(_)));
    assert!(matches!(variants[1], WebpError::Unsupported(_)));
    assert!(matches!(variants[2], WebpError::LimitExceeded(_)));
    assert!(matches!(variants[3], WebpError::Io(_)));
}

// ─────────────────────────────── probe / info ────────────────────────────

#[test]
fn probe_accepts_every_fixture_and_rejects_non_webp() {
    for (name, bytes) in ALL {
        assert!(probe(bytes), "{name}");
    }
    assert!(!probe(b""));
    assert!(!probe(b"RIFF\0\0\0\0WEB"));
    assert!(!probe(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"));
    assert!(!probe(b"RIFF\x24\0\0\0WAVEfmt "));
}

#[test]
fn info_matches_decode_for_every_fixture() {
    for (name, bytes) in ALL {
        let i = info(bytes).unwrap_or_else(|e| panic!("{name}: info: {e}"));
        let frames = decode_all(bytes).unwrap_or_else(|e| panic!("{name}: decode_all: {e}"));
        assert_eq!(i.frames as usize, frames.len(), "{name}: frame count");
        let first = &frames[0].image;
        assert_eq!(
            (i.width, i.height),
            (first.width, first.height),
            "{name}: dims"
        );
        assert_eq!(i.format, first.format, "{name}: format");
        assert_eq!(i.color, first.color, "{name}: colour");
        assert_eq!(i.has_icc, first.metadata.icc.is_some(), "{name}: icc");
        assert_eq!(i.has_exif, first.metadata.exif.is_some(), "{name}: exif");
        assert_eq!(i.has_xmp, first.metadata.xmp.is_some(), "{name}: xmp");
        assert_eq!(
            i.is_animated,
            frames.len() > 1 || frames[0].delay.is_some(),
            "{name}"
        );
        // `decode` is the first frame.
        let single = decode(bytes).unwrap();
        assert_eq!(&single, first, "{name}: decode == decode_all[0]");
    }
}

#[test]
fn info_reports_the_native_layout_per_kind() {
    let lossless = info(LOSSLESS_32X32_RGBA).unwrap();
    assert_eq!(lossless.format, PixelFormat::Rgba);
    assert!(lossless.has_alpha && !lossless.is_lossy);
    assert_eq!(lossless.color, ColorInfo::srgb());

    let opaque = info(LOSSLESS_32X32_RGB).unwrap();
    assert!(!opaque.has_alpha);

    let lossy = info(LOSSY_1X1).unwrap();
    assert_eq!(lossy.format, PixelFormat::Yuv420P);
    assert!(lossy.is_lossy && !lossy.has_alpha);
    assert_eq!(lossy.color, ColorInfo::bt601_limited());
    assert_eq!(lossy.color.range, ColorRange::Limited);

    let lossy_a = info(LOSSY_ALPHA).unwrap();
    assert_eq!(lossy_a.format, PixelFormat::Yuva420P);
    assert!(lossy_a.has_alpha);

    let exif = info(EXT_EXIF).unwrap();
    assert!(exif.has_exif && !exif.has_icc && !exif.has_xmp);
    assert!(info(EXT_ICC).unwrap().has_icc);
    assert!(info(EXT_XMP).unwrap().has_xmp);

    let anim = info(ANIM_RGB).unwrap();
    assert!(anim.is_animated);
    assert_eq!(anim.frames, 3);
    assert_eq!((anim.width, anim.height), (64, 64));
    assert_eq!(anim.format, PixelFormat::Rgba);
    assert_eq!(anim.loop_count, Some(0));
    assert_eq!(anim.background_rgba, Some([0xff, 0xff, 0xff, 0xff]));
}

// ────────────────────────────────── decode ───────────────────────────────

#[test]
fn lossless_decodes_to_native_rgba_and_the_known_pixel() {
    let img = decode(LOSSLESS_1X1).unwrap();
    assert_eq!(img.format, PixelFormat::Rgba);
    assert_eq!(img.planes.len(), 1);
    assert_eq!(img.planes[0].stride, 4);
    assert_eq!(img.as_bytes(), Some(&[0xB4u8, 0x3C, 0x5A, 0xFF][..]));
    assert_eq!(img.to_rgb8(), vec![0xB4, 0x3C, 0x5A]);
    assert_eq!(img.to_rgba8(), vec![0xB4, 0x3C, 0x5A, 0xFF]);
    assert!(img.palette.is_none());
    assert_eq!(
        decode_rgb8(LOSSLESS_1X1).unwrap().data,
        vec![0xB4, 0x3C, 0x5A]
    );
    assert_eq!(
        decode_rgba8(LOSSLESS_1X1).unwrap(),
        RgbaImage::new(1, 1, vec![0xB4, 0x3C, 0x5A, 0xFF])
    );
}

#[test]
fn lossy_decodes_to_native_limited_range_yuv420p() {
    // docs fixture `lossy-1x1`: the VP8 key-frame reconstructs to
    // Y'CbCr (101, 122, 177); the reference decoder's non-fancy RGB for
    // it is (0xB1, 0x3D, 0x57) — exactly the limited-range Rec. 601
    // inverse of those samples.
    let img = decode(LOSSY_1X1).unwrap();
    assert_eq!(img.format, PixelFormat::Yuv420P);
    assert_eq!(img.planes.len(), 3);
    assert_eq!(img.planes[0].data, vec![101]);
    assert_eq!(img.planes[1].data, vec![122]);
    assert_eq!(img.planes[2].data, vec![177]);
    assert_eq!(img.color, ColorInfo::bt601_limited());
    assert!(
        img.as_bytes().is_none(),
        "planar layout has no single plane"
    );
    assert_eq!(img.to_rgba8(), vec![0xB1, 0x3D, 0x57, 0xFF]);
    assert_eq!(img.to_rgb8(), vec![0xB1, 0x3D, 0x57]);
    assert_eq!(img.clone().into_raw(), vec![101, 122, 177]);
    assert_eq!(
        decode_rgb8(LOSSY_1X1).unwrap(),
        RgbImage::new(1, 1, vec![0xB1, 0x3D, 0x57])
    );
}

#[test]
fn lossy_with_alph_decodes_to_yuva420p_with_the_alpha_plane() {
    let img = decode(LOSSY_ALPHA).unwrap();
    assert_eq!(img.format, PixelFormat::Yuva420P);
    assert_eq!((img.width, img.height), (128, 128));
    assert_eq!(img.planes.len(), 4);
    assert_eq!(img.planes[0].data.len(), 128 * 128);
    assert_eq!(img.planes[1].data.len(), 64 * 64);
    assert_eq!(img.planes[3].data.len(), 128 * 128);
    // The ALPH plane is the one `decode_alpha_plane` reads.
    let plane = oxideav_webp::decode_alpha_plane(LOSSY_ALPHA)
        .unwrap()
        .unwrap();
    assert_eq!(img.planes[3].data, plane);
    let rgba = img.to_rgba8();
    let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
    assert_eq!(alpha, plane, "to_rgba8 carries the alpha plane through");
    // to_rgb8 drops it.
    assert_eq!(img.to_rgb8().len(), 128 * 128 * 3);
}

#[test]
fn decode_all_composites_animations_and_wraps_stills() {
    let frames = decode_all(ANIM_RGB).unwrap();
    assert_eq!(frames.len(), 3);
    for f in &frames {
        assert_eq!(f.image.format, PixelFormat::Rgba);
        assert_eq!((f.image.width, f.image.height), (64, 64));
        assert_eq!(f.image.as_bytes().unwrap().len(), 64 * 64 * 4);
        assert!(f.delay.is_some());
    }
    let still = decode_all(LOSSY_1X1).unwrap();
    assert_eq!(still.len(), 1);
    assert_eq!(still[0].delay, None);
    assert_eq!(still[0].image.format, PixelFormat::Yuv420P);
}

#[test]
fn decode_from_reads_a_stream() {
    let img = decode_from(Cursor::new(LOSSLESS_32X32_RGBA.to_vec())).unwrap();
    assert_eq!(img, decode(LOSSLESS_32X32_RGBA).unwrap());
    let err = decode_from(Cursor::new(Vec::<u8>::new())).unwrap_err();
    assert!(err.is_invalid_data());
}

// ─────────────────────────────── limits ──────────────────────────────────

#[test]
fn limits_are_enforced_before_decoding() {
    let big = DecodeOptions::default();
    assert!(decode_with(LOSSY_ALPHA, &big).is_ok());
    for opts in [
        DecodeOptions::default().with_max_width(Some(127)),
        DecodeOptions::default().with_max_height(Some(127)),
        DecodeOptions::default().with_max_pixels(Some(128 * 128 - 1)),
        DecodeOptions::default().with_max_bytes(Some(LOSSY_ALPHA.len() as u64 - 1)),
    ] {
        let e = decode_with(LOSSY_ALPHA, &opts).unwrap_err();
        assert!(e.is_limit_exceeded(), "{opts:?}: {e}");
    }
    // Animation canvas limits.
    let e =
        decode_all_with(ANIM_RGB, &DecodeOptions::default().with_max_width(Some(63))).unwrap_err();
    assert!(e.is_limit_exceeded());
    // Lossless too.
    let e = decode_with(
        LOSSLESS_NATURAL,
        &DecodeOptions::default().with_max_pixels(Some(1)),
    )
    .unwrap_err();
    assert!(e.is_limit_exceeded());
    // `None` lifts a limit entirely.
    let unlimited = DecodeOptions::default()
        .with_max_width(None)
        .with_max_height(None)
        .with_max_pixels(None)
        .with_max_bytes(None);
    assert!(decode_with(LOSSY_ALPHA, &unlimited).is_ok());
    let d = DecodeOptions::default();
    assert_eq!(d.max_width, Some(oxideav_webp::MAX_DIMENSION));
    assert_eq!(d.max_bytes, None);
}

#[test]
fn strict_mode_accepts_the_well_formed_corpus() {
    let strict = DecodeOptions::default().with_strict(true);
    for (name, bytes) in ALL {
        decode_all_with(bytes, &strict).unwrap_or_else(|e| panic!("{name}: strict: {e}"));
    }
}

#[test]
fn hostile_inputs_never_panic() {
    let cases: Vec<Vec<u8>> = vec![
        vec![],
        b"RIFF".to_vec(),
        b"RIFF\x04\0\0\0WEBP".to_vec(),
        b"RIFF\xff\xff\xff\xffWEBPVP8L\xff\xff\xff\xff".to_vec(),
        b"RIFF\x14\0\0\0WEBPVP8X\x0a\0\0\0\x10\0\0\0\xff\xff\xff\xff\xff\xff".to_vec(),
        vec![0u8; 1024],
    ];
    for c in &cases {
        let _ = probe(c);
        assert!(info(c).is_err());
        assert!(decode(c).is_err());
        assert!(decode_all(c).is_err());
    }
    // Every fixture truncated at every eighth byte and with one byte
    // flipped must return, never panic.
    for (_, bytes) in ALL {
        for cut in (1..bytes.len()).step_by(bytes.len() / 8 + 1) {
            let t = &bytes[..cut];
            let _ = info(t);
            let _ = decode(t);
            let _ = decode_all(t);
        }
        let mut flipped = bytes.to_vec();
        let i = flipped.len() / 2;
        flipped[i] ^= 0xa5;
        let _ = info(&flipped);
        let _ = decode(&flipped);
        let _ = decode_all(&flipped);
    }
}

// ─────────────────────────────── encode ──────────────────────────────────

fn synthetic_rgba(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&[
                (x.wrapping_mul(37).wrapping_add(y).wrapping_add(seed) & 0xff) as u8,
                (y.wrapping_mul(53).wrapping_add(x).wrapping_mul(7) & 0xff) as u8,
                ((x ^ y).wrapping_mul(101).wrapping_add(seed) & 0xff) as u8,
                (255 - ((x.wrapping_add(y).wrapping_add(seed)) & 0xff)) as u8,
            ]);
        }
    }
    v
}

#[test]
fn lossless_round_trip_is_exact_for_planes_and_metadata() {
    let (w, h) = (23u32, 17u32);
    let rgba = synthetic_rgba(w, h, 9);
    let img = WebpImage::from_rgba8(w, h, rgba.clone())
        .unwrap()
        .with_metadata(
            Metadata::new()
                .with_icc(Some(b"icc".to_vec()))
                .with_exif(Some(b"Exif\0\0II*\0".to_vec()))
                .with_xmp(Some(b"<x:xmpmeta/>".to_vec())),
        );
    let bytes = encode(&img, &EncodeOptions::default()).unwrap();
    assert!(probe(&bytes));
    let back = decode(&bytes).unwrap();
    assert_eq!(back, img, "decode(encode(img)) == img");
    assert_eq!(back.format, PixelFormat::Rgba);
    assert_eq!(back.color, ColorInfo::srgb());

    // RGB8 in → opaque RGBA native.
    let rgb: Vec<u8> = rgba
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    let bytes = encode_rgb8(w, h, &rgb, &EncodeOptions::default()).unwrap();
    let back = decode_rgb8(&bytes).unwrap();
    assert_eq!(back, RgbImage::new(w, h, rgb));
    assert!(!info(&bytes).unwrap().has_alpha);
    // Simple (non-VP8X) layout when there is nothing to declare.
    let c = oxideav_webp::parse_container(&bytes).unwrap();
    assert!(c
        .first_chunk_with_fourcc(oxideav_webp::container::fourcc::VP8X)
        .is_none());

    // The streaming variant writes the same bytes.
    let mut out = Vec::new();
    encode_to(&img, &EncodeOptions::default(), &mut out).unwrap();
    assert_eq!(out, encode(&img, &EncodeOptions::default()).unwrap());
}

#[test]
fn every_lossless_fixture_re_encodes_exactly() {
    for (name, bytes) in ALL {
        let i = info(bytes).unwrap();
        if i.is_lossy || i.is_animated {
            continue;
        }
        let img = decode(bytes).unwrap();
        let re = encode(&img, &EncodeOptions::default()).unwrap();
        let back = decode(&re).unwrap();
        assert_eq!(back.planes, img.planes, "{name}: planes");
        assert_eq!(back.metadata, img.metadata, "{name}: metadata");
    }
}

#[test]
fn encoder_refuses_layouts_webp_cannot_carry() {
    // Lossless Y'CbCr: no such WebP layout.
    let yuv = WebpImage::from_yuv420(2, 2, vec![128; 4], vec![128], vec![128]).unwrap();
    let e = encode(&yuv, &EncodeOptions::default()).unwrap_err();
    assert!(e.is_unsupported(), "{e}");
    // Lossy full-range Y'CbCr: VP8 cannot signal the range.
    let full = yuv
        .clone()
        .with_color(ColorInfo::bt601_limited().with_range(ColorRange::Full));
    let e = encode(&full, &EncodeOptions::default().with_quality(75.0)).unwrap_err();
    assert!(e.is_unsupported(), "{e}");
    // Geometry mismatch is InvalidData at construction, not a panic.
    let short = WebpImage::new(4, 4, PixelFormat::Rgba, vec![Plane::packed(16, vec![0; 8])]);
    assert!(short.unwrap_err().is_invalid_data());
    assert!(encode_rgba8(3, 3, &[0; 4], &EncodeOptions::default()).is_err());
    assert!(encode_rgba8(0, 3, &[], &EncodeOptions::default()).is_err());
}

#[test]
fn lossy_quality_switches_to_vp8_and_round_trips_closely() {
    let (w, h) = (32u32, 24u32);
    let mut rgb = Vec::new();
    for y in 0..h {
        for x in 0..w {
            rgb.extend_from_slice(&[(x * 8) as u8, (y * 10) as u8, ((x + y) * 4) as u8]);
        }
    }
    for q in [100.0f32, 90.0, 60.0] {
        let bytes = encode_rgb8(w, h, &rgb, &EncodeOptions::default().with_quality(q)).unwrap();
        let i = info(&bytes).unwrap();
        assert!(i.is_lossy, "q={q}");
        assert_eq!(i.format, PixelFormat::Yuv420P);
        assert!(!i.has_alpha);
        let back = decode_rgb8(&bytes).unwrap();
        assert_eq!((back.width, back.height), (w, h));
        let mae = back
            .data
            .iter()
            .zip(rgb.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as f64)
            .sum::<f64>()
            / rgb.len() as f64;
        assert!(mae < 16.0, "q={q}: mean abs error {mae}");
    }
    // Native Yuv420P in → VP8 straight through; planes come back intact
    // up to quantisation and the layout is preserved.
    let src = decode(LOSSY_ALPHA).unwrap();
    let bytes = encode(&src, &EncodeOptions::default().with_quality(100.0)).unwrap();
    let back = decode(&bytes).unwrap();
    assert_eq!(back.format, PixelFormat::Yuva420P);
    assert_eq!(back.planes[3], src.planes[3], "ALPH plane is lossless");
}

#[test]
fn lossy_rgba_with_transparency_emits_alph() {
    let (w, h) = (16u32, 16u32);
    let rgba = synthetic_rgba(w, h, 3);
    let bytes = encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_quality(85.0)).unwrap();
    let i = info(&bytes).unwrap();
    assert_eq!(i.format, PixelFormat::Yuva420P);
    assert!(i.has_alpha && i.is_lossy);
    let back = decode(&bytes).unwrap();
    let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
    assert_eq!(back.planes[3].data, alpha);
    // An opaque RGBA image stays plain Yuv420P (no ALPH).
    let opaque: Vec<u8> = rgba
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2], 0xff])
        .collect();
    let bytes = encode_rgba8(w, h, &opaque, &EncodeOptions::default().with_quality(85.0)).unwrap();
    assert_eq!(info(&bytes).unwrap().format, PixelFormat::Yuv420P);
}

#[test]
fn metadata_embed_flags_select_chunks() {
    let img = WebpImage::from_rgba8(2, 2, vec![1; 16])
        .unwrap()
        .with_metadata(
            Metadata::new()
                .with_icc(Some(vec![1]))
                .with_exif(Some(vec![2]))
                .with_xmp(Some(vec![3])),
        );
    for (icc, exif, xmp) in [
        (true, true, true),
        (false, false, false),
        (true, false, true),
        (false, true, false),
    ] {
        let opts = EncodeOptions::default().with_metadata(icc, exif, xmp);
        for lossy in [false, true] {
            let opts = if lossy {
                opts.clone().with_quality(90.0)
            } else {
                opts.clone()
            };
            let bytes = encode(&img, &opts).unwrap();
            let i = info(&bytes).unwrap();
            assert_eq!(
                (i.has_icc, i.has_exif, i.has_xmp),
                (icc, exif, xmp),
                "lossy={lossy}"
            );
            let back = decode(&bytes).unwrap();
            assert_eq!(back.metadata.icc.is_some(), icc);
            assert_eq!(back.metadata.exif.is_some(), exif);
            assert_eq!(back.metadata.xmp.is_some(), xmp);
        }
    }
}

#[test]
fn encode_animation_round_trips_through_decode_all() {
    let mk = |seed: u32| WebpImage::from_rgba8(8, 6, synthetic_rgba(8, 6, seed)).unwrap();
    let frames = vec![
        Frame::new(mk(1), Some(Duration::from_millis(50))),
        Frame::new(mk(2), Some(Duration::from_millis(70))),
        Frame::new(mk(3), None),
    ];
    let opts = EncodeOptions::default()
        .with_loop_count(2)
        .with_background_rgba([1, 2, 3, 4]);
    let bytes = encode_animation(&frames, &opts).unwrap();
    let i = info(&bytes).unwrap();
    assert!(i.is_animated);
    assert_eq!(i.frames, 3);
    assert_eq!(i.loop_count, Some(2));
    assert_eq!(i.background_rgba, Some([1, 2, 3, 4]));
    let back = decode_all(&bytes).unwrap();
    assert_eq!(back.len(), 3);
    for (b, f) in back.iter().zip(frames.iter()) {
        assert_eq!(b.image.planes, f.image.planes);
        assert_eq!(
            b.delay.or(Some(Duration::ZERO)),
            f.delay.or(Some(Duration::ZERO))
        );
    }
    assert!(encode_animation(&[], &opts).unwrap_err().is_invalid_data());
    assert!(encode_animation(&frames, &opts.clone().with_quality(50.0))
        .unwrap_err()
        .is_unsupported());
}
