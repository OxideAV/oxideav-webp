//! End-to-end tests for the `Vp8WebpEncoder` adapter's
//! `ICCP` / `EXIF` / `XMP ` chunk passthrough.
//!
//! Coverage:
//!
//! * **Quality factory + metadata** — `make_encoder_with_quality_and_metadata`
//!   accepts a `WebpMetadataOwned`, encodes a YUV420P frame, and the
//!   resulting `.webp` file carries all three metadata chunks
//!   byte-identical with the input + the matching VP8X flag bits set.
//! * **Qindex factory + metadata** — same dance with the explicit-qindex
//!   factory; verifies the same passthrough path.
//! * **Default factory unchanged** — `make_encoder_with_quality` (no
//!   metadata) still emits the simple `RIFF/WEBP/VP8 ` layout for an
//!   opaque YUV420P input, proving the metadata field defaults to
//!   all-`None` and doesn't accidentally promote to extended layout.
//! * **RGBA + metadata** — `WebpMetadataOwned` rides on top of the
//!   `VP8X + ALPH + VP8 ` extended layout an RGBA frame already
//!   produces, with both ALPHA and EXIF flag bits set.

#![cfg(feature = "registry")]

use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_webp::encoder_vp8::{
    make_encoder_with_qindex_and_metadata, make_encoder_with_quality,
    make_encoder_with_quality_and_metadata,
};
use oxideav_webp::{decode_webp, extract_metadata, WebpMetadataOwned, CODEC_ID_VP8};

const W: u32 = 16;
const H: u32 = 16;

fn yuv420_frame() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2 + (w & 1);
    let ch = h / 2 + (h & 1);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // Smooth gradient — VP8 should reconstruct it cleanly.
            y[j * w + i] = ((i + j) * 8) as u8;
        }
    }
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    VideoFrame {
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
    }
}

fn rgba_frame() -> VideoFrame {
    let w = W as usize;
    let h = H as usize;
    let mut buf = Vec::with_capacity(w * h * 4);
    for j in 0..h {
        for i in 0..w {
            let r = (i * 16) as u8;
            let g = (j * 16) as u8;
            let b = ((i + j) * 8) as u8;
            // Diagonal alpha ramp so the encoder takes the VP8X+ALPH
            // layout (any non-0xff alpha forces the extended container).
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

fn params(pix: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    p
}

fn icc_payload() -> Vec<u8> {
    let mut v = b"ICC_VP8_TEST".to_vec();
    v.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04]);
    v
}

fn exif_payload() -> Vec<u8> {
    let mut v = b"II*\0".to_vec();
    v.extend_from_slice(b"oxideav-webp test EXIF payload");
    v
}

fn xmp_payload() -> Vec<u8> {
    b"<?xml version=\"1.0\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec()
}

#[test]
fn quality_factory_with_metadata_attaches_all_three_chunks_yuv420p() {
    let icc = icc_payload();
    let exif = exif_payload();
    let xmp = xmp_payload();
    let meta = WebpMetadataOwned {
        icc: Some(icc.clone()),
        exif: Some(exif.clone()),
        xmp: Some(xmp.clone()),
    };
    let p = params(PixelFormat::Yuv420P);
    let mut enc =
        make_encoder_with_quality_and_metadata(&p, 75.0, meta).expect("build encoder w/ metadata");
    enc.send_frame(&Frame::Video(yuv420_frame())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().expect("receive_packet");
    let bytes = pkt.data;

    // Outer shape: RIFF/WEBP/VP8X (extended layout because metadata
    // was attached, even though the input was opaque YUV).
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WEBP");
    assert_eq!(
        &bytes[12..16],
        b"VP8X",
        "metadata-attached encode must take the extended VP8X layout"
    );
    // VP8X flags: ICCP|EXIF|XMP — no ALPHA (opaque YUV).
    let flags = bytes[20];
    assert_ne!(flags & 0x20, 0, "ICC flag missing");
    assert_ne!(flags & 0x08, 0, "EXIF flag missing");
    assert_ne!(flags & 0x04, 0, "XMP flag missing");
    assert_eq!(flags & 0x10, 0, "ALPHA flag must be clear for opaque YUV");

    // Metadata-only fast-path round-trip.
    let extracted = extract_metadata(&bytes).expect("extract_metadata");
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(extracted.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(extracted.xmp.as_deref(), Some(xmp.as_slice()));

    // Full decode path also returns the metadata on the WebpImage.
    let img = decode_webp(&bytes).expect("decode_webp");
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.metadata.xmp.as_deref(), Some(xmp.as_slice()));
    assert_eq!(img.frames.len(), 1);
}

#[test]
fn qindex_factory_with_metadata_attaches_icc_only() {
    // ICC alone (no EXIF, no XMP) — verifies a partial-metadata case
    // doesn't leak the other two flag bits.
    let icc = icc_payload();
    let meta = WebpMetadataOwned {
        icc: Some(icc.clone()),
        exif: None,
        xmp: None,
    };
    let p = params(PixelFormat::Yuv420P);
    let mut enc = make_encoder_with_qindex_and_metadata(&p, 32, meta).expect("build encoder");
    enc.send_frame(&Frame::Video(yuv420_frame())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().unwrap();
    let bytes = pkt.data;
    // VP8X flags: ICC bit set, EXIF + XMP clear, ALPHA clear.
    assert_eq!(&bytes[12..16], b"VP8X");
    let flags = bytes[20];
    assert_ne!(flags & 0x20, 0, "ICC flag missing (flags={flags:#x})");
    assert_eq!(flags & 0x08, 0, "EXIF flag set unexpectedly");
    assert_eq!(flags & 0x04, 0, "XMP flag set unexpectedly");
    assert_eq!(flags & 0x10, 0, "ALPHA flag set unexpectedly");
    let extracted = extract_metadata(&bytes).unwrap();
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(extracted.exif, None);
    assert_eq!(extracted.xmp, None);
}

#[test]
fn metadata_less_factory_still_emits_simple_layout_for_opaque_yuv() {
    // Sanity: the vanilla `make_encoder_with_quality` factory
    // (no metadata field on the encoder) produces the simple
    // `RIFF/WEBP/VP8 ` layout — no VP8X header. Catches a
    // regression where the metadata field's default was promoted
    // accidentally.
    let p = params(PixelFormat::Yuv420P);
    let mut enc = make_encoder_with_quality(&p, 75.0).expect("build encoder");
    enc.send_frame(&Frame::Video(yuv420_frame())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().unwrap();
    let bytes = pkt.data;
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WEBP");
    assert_eq!(
        &bytes[12..16],
        b"VP8 ",
        "vanilla quality encoder on opaque YUV must take simple layout (got {:?})",
        std::str::from_utf8(&bytes[12..16]).unwrap_or("???")
    );
}

#[test]
fn rgba_with_metadata_combines_alpha_and_exif_flags() {
    let exif = exif_payload();
    let meta = WebpMetadataOwned {
        icc: None,
        exif: Some(exif.clone()),
        xmp: None,
    };
    let p = params(PixelFormat::Rgba);
    let mut enc =
        make_encoder_with_quality_and_metadata(&p, 75.0, meta).expect("build encoder w/ metadata");
    enc.send_frame(&Frame::Video(rgba_frame())).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().unwrap();
    let bytes = pkt.data;
    assert_eq!(&bytes[12..16], b"VP8X", "rgba+metadata must take VP8X");
    let flags = bytes[20];
    // Bit 4 (0x10) ALPHA + bit 3 (0x08) EXIF must both be set.
    assert_ne!(flags & 0x10, 0, "ALPHA flag missing on rgba+metadata");
    assert_ne!(flags & 0x08, 0, "EXIF flag missing on rgba+metadata");
    let extracted = extract_metadata(&bytes).unwrap();
    assert_eq!(extracted.exif.as_deref(), Some(exif.as_slice()));
    // Decode round-trips to a proper RGBA image with the EXIF still
    // attached on the WebpImage.
    let img = decode_webp(&bytes).expect("decode rgba+exif");
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
}

#[test]
fn standalone_yuv420_metadata_borrow_round_trip() {
    // The standalone `encode_vp8_lossy_yuv420p` already accepts a
    // borrowing `WebpMetadata<'_>`; this test just locks down that the
    // path co-exists cleanly with the new owned-metadata adapter.
    use oxideav_webp::{encode_vp8_lossy_yuv420p, WebpMetadata};
    let icc = icc_payload();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: None,
        xmp: None,
    };
    let w = W as usize;
    let h = H as usize;
    let cw = w / 2 + (w & 1);
    let ch = h / 2 + (h & 1);
    let y = vec![100u8; w * h];
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    let bytes = encode_vp8_lossy_yuv420p(W, H, &y, &u, &v, 75.0, &meta).unwrap();
    assert_eq!(&bytes[12..16], b"VP8X");
    let extracted = extract_metadata(&bytes).unwrap();
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
}
