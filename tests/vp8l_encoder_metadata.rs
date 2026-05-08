//! End-to-end tests for the registry-side VP8L lossless `Vp8lEncoder`
//! adapter's `ICCP` / `EXIF` / `XMP ` chunk passthrough.
//!
//! Mirrors `tests/vp8_lossy_metadata_passthrough.rs` for the lossless
//! side. The standalone [`oxideav_webp::encode_vp8l_argb_with_metadata`]
//! entry point already accepts a `WebpMetadata<'_>` borrow; this file
//! tests the registry-side parity introduced in the streaming-decoder
//! round (the new
//! [`oxideav_webp::encoder::make_encoder_with_metadata`] factory).
//!
//! Coverage:
//!
//! * **`make_encoder_with_metadata` attaches all three chunks** — feed
//!   an opaque RGBA frame, verify the resulting `.webp` carries
//!   `ICCP` / `EXIF` / `XMP ` byte-identical.
//! * **Default `make_encoder` factory unchanged** — opaque RGB24 input
//!   still lands on the simple `RIFF/WEBP/VP8L` layout (no VP8X), so
//!   the metadata field defaults to all-`None` and doesn't accidentally
//!   promote to extended layout.
//! * **Alpha + metadata stacks** — alpha-bearing RGBA + `WebpMetadataOwned`
//!   rides on top of the same `VP8X + VP8L` extended layout; both the
//!   ALPHA flag and the metadata flag bits land in the VP8X header.

#![cfg(feature = "registry")]

use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_webp::encoder::{make_encoder, make_encoder_with_metadata};
use oxideav_webp::{decode_webp, extract_metadata, WebpMetadataOwned, CODEC_ID_VP8L};

const W: u32 = 12;
const H: u32 = 12;

fn rgba_opaque() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let mut buf = Vec::with_capacity(w * h * 4);
    for j in 0..h {
        for i in 0..w {
            let r = (i * 16) as u8;
            let g = (j * 16) as u8;
            let b = ((i + j) * 8) as u8;
            buf.extend_from_slice(&[r, g, b, 0xff]);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w * 4,
            data: buf,
        }],
    }
}

fn rgba_alpha_ramp() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let mut buf = Vec::with_capacity(w * h * 4);
    for j in 0..h {
        for i in 0..w {
            let r = (i * 16) as u8;
            let g = (j * 16) as u8;
            let b = ((i + j) * 8) as u8;
            // Diagonal alpha ramp — a single non-0xff value forces the
            // extended VP8X layout regardless of metadata presence.
            let a = ((i + j) * 8) as u8;
            buf.extend_from_slice(&[r, g, b, a]);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w * 4,
            data: buf,
        }],
    }
}

fn rgb24_opaque() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let mut buf = Vec::with_capacity(w * h * 3);
    for j in 0..h {
        for i in 0..w {
            let r = (i * 16) as u8;
            let g = (j * 16) as u8;
            let b = ((i + j) * 8) as u8;
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w * 3,
            data: buf,
        }],
    }
}

fn params(pix: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8L));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    p
}

fn icc_payload() -> Vec<u8> {
    let mut v = b"VP8L_ICC_TEST".to_vec();
    v.extend_from_slice(&[0xfe, 0xed, 0xfa, 0xce]);
    v
}

fn exif_payload() -> Vec<u8> {
    let mut v = b"II*\0".to_vec();
    v.extend_from_slice(b"vp8l registry encoder EXIF payload");
    v
}

fn xmp_payload() -> Vec<u8> {
    b"<?xml version=\"1.0\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\" id=\"vp8l-test\"/>".to_vec()
}

#[test]
fn make_encoder_with_metadata_attaches_all_three_chunks_rgba_opaque() {
    let icc = icc_payload();
    let exif = exif_payload();
    let xmp = xmp_payload();
    let meta = WebpMetadataOwned {
        icc: Some(icc.clone()),
        exif: Some(exif.clone()),
        xmp: Some(xmp.clone()),
    };
    let p = params(PixelFormat::Rgba);
    let mut enc = make_encoder_with_metadata(&p, meta).expect("build encoder w/ metadata");
    enc.send_frame(&Frame::Video(rgba_opaque())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().expect("receive_packet");
    let bytes = pkt.data;

    let extracted = extract_metadata(&bytes).expect("extract_metadata");
    assert_eq!(
        extracted.icc.as_deref(),
        Some(icc.as_slice()),
        "ICC chunk byte-mismatch through registry-side VP8L encoder"
    );
    assert_eq!(extracted.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(extracted.xmp.as_deref(), Some(xmp.as_slice()));

    // And the same bytes must surface on `WebpImage::metadata` after a
    // full decode — the two paths read the same chunks but via
    // independent code, so this catches drift.
    let img = decode_webp(&bytes).expect("decode_webp");
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.metadata.xmp.as_deref(), Some(xmp.as_slice()));
}

#[test]
fn default_factory_emits_simple_layout_for_opaque_rgb24() {
    // The historical `make_encoder` factory must keep emitting the
    // simple `RIFF/WEBP/VP8L` layout for opaque inputs (no VP8X header,
    // no metadata fields). That's what `extract_metadata` returns
    // `None` for: simple-layout files have no chunks for it to find.
    let p = params(PixelFormat::Rgb24);
    let mut enc = make_encoder(&p).expect("build encoder");
    enc.send_frame(&Frame::Video(rgb24_opaque())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().expect("receive_packet");
    let bytes = pkt.data;

    let extracted = extract_metadata(&bytes).expect("extract_metadata");
    assert!(extracted.icc.is_none());
    assert!(extracted.exif.is_none());
    assert!(extracted.xmp.is_none());
    let img = decode_webp(&bytes).expect("decode_webp");
    assert!(img.metadata.icc.is_none());
    assert!(img.metadata.exif.is_none());
    assert!(img.metadata.xmp.is_none());
}

#[test]
fn alpha_plus_metadata_stacks_in_extended_layout() {
    // Alpha-bearing RGBA already mandates the VP8X + VP8L extended
    // layout (the VP8X ALPHA flag is required). Metadata ride on top
    // — both flags must land, and the chunks must round-trip.
    let icc = icc_payload();
    let exif = exif_payload();
    let meta = WebpMetadataOwned {
        icc: Some(icc.clone()),
        exif: Some(exif.clone()),
        xmp: None,
    };
    let p = params(PixelFormat::Rgba);
    let mut enc = make_encoder_with_metadata(&p, meta).expect("build encoder w/ metadata");
    enc.send_frame(&Frame::Video(rgba_alpha_ramp())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().expect("receive_packet");
    let bytes = pkt.data;

    let extracted = extract_metadata(&bytes).expect("extract_metadata");
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(extracted.exif.as_deref(), Some(exif.as_slice()));
    assert!(extracted.xmp.is_none());

    // The pixel decode must still produce the diagonal alpha ramp the
    // input carried — VP8L is lossless, so any deviation flags a
    // metadata-attach plumbing regression.
    let img = decode_webp(&bytes).expect("decode_webp");
    assert_eq!(img.frames.len(), 1);
    let pixels = &img.frames[0].rgba;
    let stride = (W as usize) * 4;
    for j in 0..H as usize {
        for i in 0..W as usize {
            let idx = j * stride + i * 4;
            let expected_a = ((i + j) * 8) as u8;
            assert_eq!(
                pixels[idx + 3],
                expected_a,
                "alpha mismatch at ({i},{j}) — metadata-attach broke pixel path"
            );
        }
    }
}
