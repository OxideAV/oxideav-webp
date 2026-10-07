//! Pins of the lossy (`VP8 `) encode path's output bytes.
//!
//! The VP8 bitstream comes from the `oxideav-vp8` encoder; this crate
//! wraps it in the RIFF container, with a `VP8X` header, an `ALPH` plane
//! and metadata chunks when the image needs them. A change to that
//! wrapping meant to leave the output alone must leave every pin as it
//! is.

use oxideav_webp::{decode_all, encode, encode_rgba8, EncodeOptions};

/// 64-bit FNV-1a over `bytes`: a dependency-free fingerprint for pinning
/// encoder output byte for byte.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A deterministic photo-like RGBA image: smooth gradients, a soft bright
/// blob and small per-channel noise. Fully opaque.
fn photo_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut px = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // xorshift64 noise source.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let d = ((fx - 0.5).powi(2) + (fy - 0.4).powi(2)).sqrt();
            let blob = (1.0 - (d * 3.0).min(1.0)) * 60.0;
            let nr = (state & 0xF) as f32 - 8.0;
            let ng = ((state >> 4) & 0xF) as f32 - 8.0;
            let nb = ((state >> 8) & 0xF) as f32 - 8.0;
            let r = (40.0 + 180.0 * fx + blob + nr).clamp(0.0, 255.0) as u8;
            let g = (60.0 + 150.0 * fy + blob + ng).clamp(0.0, 255.0) as u8;
            let b = (200.0 - 120.0 * fx * fy + nb).clamp(0.0, 255.0) as u8;
            px.extend_from_slice(&[r, g, b, 255]);
        }
    }
    px
}

/// `(fixture, length, FNV-1a 64)` of the first frame of every committed
/// fixture, encoded at quality 80 with its metadata.
const FIXTURE_PINS: &[(&str, usize, u64)] = &[
    ("animated-3-frames-rgb.webp", 674, 0x18457de5aca0d009),
    ("animated-with-alpha.webp", 674, 0x18457de5aca0d009),
    ("extended-with-exif.webp", 1284, 0x6036fcdf3caabdc9),
    ("extended-with-icc-profile.webp", 1542, 0x444d02ca2a4fe520),
    ("extended-with-xmp.webp", 1556, 0xb155910ef8db6035),
    ("lossless-128x128-natural.webp", 414, 0xff519b2c84cab2c8),
    ("lossless-1x1.webp", 56, 0x5b4bd1ba9f9a58cc),
    ("lossless-32x32-rgb.webp", 134, 0x0ccedc0c2fe501fa),
    ("lossless-32x32-rgba.webp", 200, 0x86a60121cd80db82),
    ("lossless-color-cache-stress.webp", 3802, 0x2013d9418b6242b3),
    (
        "lossless-color-indexing-paletted.webp",
        650,
        0x504bd6ece19c7aa7,
    ),
    ("lossless-cross-color-active.webp", 206, 0x5ec88d90d5a19a3c),
    ("lossy-1x1.webp", 58, 0x9f844a7d819ec54e),
    ("lossy-near-lossless-q40.webp", 1260, 0x10417e2004277fa4),
    ("lossy-with-alpha-128x128.webp", 2926, 0xa98113323bfcf060),
];

fn assert_pin(what: &str, out: &[u8], len: usize, hash: u64) {
    assert_eq!(
        (out.len(), fnv1a64(out)),
        (len, hash),
        "{what}: lossy output changed"
    );
}

#[test]
fn every_fixture_encodes_to_its_pinned_lossy_bytes() {
    let opts = EncodeOptions::default().with_quality(80.0);
    for &(name, len, hash) in FIXTURE_PINS {
        let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let frames = decode_all(&bytes).expect("fixture decodes");
        let out = encode(&frames[0].image, &opts).expect("lossy encode");
        assert_pin(name, &out, len, hash);
    }
}

#[test]
fn photos_encode_to_their_pinned_lossy_bytes() {
    let opts = EncodeOptions::default().with_quality(80.0);
    for (side, len, hash) in [
        (64u32, 328usize, 0x3480734f0a055024u64),
        (96, 532, 0xa0b46f8ea0b2283f),
    ] {
        let out = encode_rgba8(side, side, &photo_rgba(side, side), &opts).expect("lossy encode");
        assert_pin(&format!("photo {side}"), &out, len, hash);
    }
}

/// The `webp_vp8` framework encoder wraps every packet in the container
/// too.
#[cfg(feature = "registry")]
#[test]
fn the_framework_encoder_packet_keeps_its_pinned_bytes() {
    use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};

    let (w, h) = (48usize, 40usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y: Vec<u8> = (0..w * h)
        .map(|i| ((i % w) * 5 + (i / w) * 3) as u8)
        .collect();
    let u: Vec<u8> = (0..cw * ch).map(|i| (100 + (i % cw) * 2) as u8).collect();
    let v: Vec<u8> = (0..cw * ch).map(|i| (150 - (i / cw) * 2) as u8).collect();
    let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    let mut enc = oxideav_webp::encoder_vp8::make_encoder_with_quality(&params, 80.0)
        .expect("vp8 framework encoder");
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: u,
            },
            VideoPlane {
                stride: cw,
                data: v,
            },
        ],
    });
    enc.send_frame(&frame).expect("send");
    let packet = enc.receive_packet().expect("packet");
    assert_pin("framework packet", &packet.data, 306, 0x607fd4004355b6cd);
}
