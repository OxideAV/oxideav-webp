//! Round-154 published-API smoke tests for the **VP8 lossy encode**
//! surface — the published-0.1.5 `encode_vp8_lossy_*` entry points
//! re-exposed over the `oxideav-vp8` sibling crate's
//! `encode_keyframe` driver.
//!
//! Every test here uses only standalone APIs (no `registry` feature), so
//! the file builds and runs under `--no-default-features`. It exercises:
//!
//! * `encode_vp8_lossy_yuv420p` — packed I420 planes → `.webp`.
//! * `encode_vp8_lossy_yuva420p` — packed I420 + separate alpha plane →
//!   `.webp` with §2.7.1.2 `ALPH` chunk.
//! * `encode_vp8_lossy_rgba` — interleaved RGBA → `.webp` (auto-detects
//!   non-opaque alpha and emits an `ALPH` chunk).
//! * `encode_vp8_lossy_rgb24` — interleaved RGB24 → `.webp` (always
//!   treated as opaque per the published contract).
//! * `WebpMetadata` (borrowed) — the encode-side metadata input.
//! * `CODEC_ID_VP8` / `CODEC_ID_VP8L` — the published registry codec IDs.

use oxideav_webp::{
    container, encode_vp8_lossy_rgb24, encode_vp8_lossy_rgba, encode_vp8_lossy_yuv420p,
    encode_vp8_lossy_yuva420p, vp8_chunk, WebpMetadata, CODEC_ID_VP8, CODEC_ID_VP8L,
};

#[test]
fn codec_ids_are_stable_strings() {
    // The published 0.1.5 codec IDs the registry surface uses.
    assert_eq!(CODEC_ID_VP8L, "webp_vp8l");
    assert_eq!(CODEC_ID_VP8, "webp_vp8");
}

/// Build a 16×16 grey I420 source frame.
fn grey_i420(width: u32, height: u32, luma: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = (width * height) as usize;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let cn = (cw * ch) as usize;
    (vec![luma; n], vec![128; cn], vec![128; cn])
}

#[test]
fn encode_vp8_lossy_yuv420p_emits_riff_vp8_file() {
    const W: u32 = 16;
    const H: u32 = 16;
    let (y, u, v) = grey_i420(W, H, 120);
    let file =
        encode_vp8_lossy_yuv420p(W, H, &y, &u, &v, 75.0, &WebpMetadata::default()).expect("encode");
    // RIFF/WEBP magic at the file head.
    assert_eq!(&file[0..4], b"RIFF");
    assert_eq!(&file[8..12], b"WEBP");
    // Simple layout (empty metadata + no alpha) → no VP8X chunk.
    let c = container::parse(&file).expect("parse");
    assert!(c.first_chunk_with_fourcc(container::fourcc::VP8X).is_none());
    // VP8 chunk reports the source dimensions back through its keyframe
    // header.
    let vp8 = c
        .first_chunk_with_fourcc(container::fourcc::VP8)
        .expect("VP8");
    let lossy = vp8_chunk::WebpLossyChunk::from_chunk(&file, vp8).expect("keyframe");
    assert_eq!(u32::from(lossy.width()), W);
    assert_eq!(u32::from(lossy.height()), H);
}

#[test]
fn encode_vp8_lossy_yuva420p_writes_alph_chunk() {
    const W: u32 = 4;
    const H: u32 = 4;
    let (y, u, v) = grey_i420(W, H, 100);
    // Distinct alpha pattern so the round-trip is detectable.
    let alpha: Vec<u8> = (0..(W * H) as u8).collect();
    let file = encode_vp8_lossy_yuva420p(W, H, &y, &u, &v, &alpha, 75.0, &WebpMetadata::default())
        .expect("encode");
    let c = container::parse(&file).expect("parse");
    let alph = c
        .first_chunk_with_fourcc(container::fourcc::ALPH)
        .expect("ALPH present");
    let payload = alph.payload(&file);
    // §2.7.1.2 info byte is method-0 / no filter / no preprocessing.
    assert_eq!(payload[0], 0x00);
    assert_eq!(&payload[1..], &alpha[..]);
    // The standalone `decode_alpha_plane` helper round-trips the plane.
    let plane = oxideav_webp::decode_alpha_plane(&file)
        .expect("decode_alpha_plane")
        .expect("alpha plane present");
    assert_eq!(plane, alpha);
}

#[test]
fn encode_vp8_lossy_rgba_keeps_simple_layout_when_opaque() {
    const W: u32 = 4;
    const H: u32 = 4;
    let mut rgba = Vec::with_capacity((W * H * 4) as usize);
    for i in 0..W * H {
        rgba.extend_from_slice(&[(i & 0xff) as u8, 0x80, 0x40, 0xff]);
    }
    let file = encode_vp8_lossy_rgba(W, H, &rgba, 75.0, &WebpMetadata::default()).expect("encode");
    let c = container::parse(&file).expect("parse");
    // Opaque RGBA + empty metadata → simple layout, no VP8X.
    assert!(c.first_chunk_with_fourcc(container::fourcc::VP8X).is_none());
    assert!(c.first_chunk_with_fourcc(container::fourcc::ALPH).is_none());
}

#[test]
fn encode_vp8_lossy_rgba_extended_when_alpha_is_present() {
    const W: u32 = 2;
    const H: u32 = 2;
    let rgba = vec![
        10, 20, 30, 0xff, //
        40, 50, 60, 0x80, // non-opaque pixel forces extended layout
        70, 80, 90, 0xff, //
        100, 110, 120, 0xff,
    ];
    let file = encode_vp8_lossy_rgba(W, H, &rgba, 75.0, &WebpMetadata::default()).expect("encode");
    let c = container::parse(&file).expect("parse");
    assert!(c.first_chunk_with_fourcc(container::fourcc::VP8X).is_some());
    assert!(c.first_chunk_with_fourcc(container::fourcc::ALPH).is_some());
}

#[test]
fn encode_vp8_lossy_rgb24_always_simple_alpha_off() {
    const W: u32 = 4;
    const H: u32 = 4;
    let rgb: Vec<u8> = (0..(W * H * 3) as u8).collect();
    let file = encode_vp8_lossy_rgb24(W, H, &rgb, 75.0, &WebpMetadata::default()).expect("encode");
    let c = container::parse(&file).expect("parse");
    // Rgb24 input → opaque, never carries an ALPH chunk.
    assert!(c.first_chunk_with_fourcc(container::fourcc::ALPH).is_none());
}

#[test]
fn encode_vp8_lossy_quality_default_handles_nan() {
    // NaN must be tolerated (clamped to default 75 → qi=32). The file
    // must still encode cleanly.
    const W: u32 = 16;
    const H: u32 = 16;
    let (y, u, v) = grey_i420(W, H, 128);
    let _ = encode_vp8_lossy_yuv420p(W, H, &y, &u, &v, f32::NAN, &WebpMetadata::default())
        .expect("NaN quality tolerated");
}

#[test]
fn encode_vp8_lossy_rejects_zero_dimension() {
    let rgba = vec![0u8; 0];
    assert!(encode_vp8_lossy_rgba(0, 1, &rgba, 75.0, &WebpMetadata::default()).is_err());
    assert!(encode_vp8_lossy_rgba(1, 0, &rgba, 75.0, &WebpMetadata::default()).is_err());
}

#[test]
fn encode_vp8_lossy_rgba_with_metadata_promotes_layout() {
    const W: u32 = 4;
    const H: u32 = 4;
    let rgba = vec![0xffu8; (W * H * 4) as usize]; // opaque white
    let xmp = b"<?xpacket begin?>".to_vec();
    let meta = WebpMetadata {
        icc: None,
        exif: None,
        xmp: Some(&xmp),
    };
    let file = encode_vp8_lossy_rgba(W, H, &rgba, 75.0, &meta).expect("encode");
    let c = container::parse(&file).expect("parse");
    // Metadata pulls the file onto the extended layout, even with no alpha.
    assert!(c.first_chunk_with_fourcc(container::fourcc::VP8X).is_some());
    // XMP chunk landed at the metadata-after-image position.
    assert!(c.first_chunk_with_fourcc(container::fourcc::XMP).is_some());
}
