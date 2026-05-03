//! End-to-end tests for the auxiliary `VP8X` metadata chunks
//! (`ICCP` / `EXIF` / `XMP `).
//!
//! Coverage:
//!
//! * **All three chunks in one file** — encode a VP8L bitstream, wrap
//!   it in a VP8X container with all three metadata chunks attached via
//!   [`oxideav_webp::riff::WebpMetadata`], and verify the bytes survive
//!   end-to-end through both [`oxideav_webp::extract_metadata`] (the
//!   metadata-only fast path) and [`oxideav_webp::decode_webp`] (the
//!   full decode that also returns metadata on the resulting
//!   [`oxideav_webp::WebpImage`]).
//! * **Simple-layout files don't fabricate metadata** — a fully-opaque
//!   RGBA encode lands on the simple `RIFF/WEBP/VP8L` layout with no
//!   `VP8X` header, so all three metadata fields must be `None` after
//!   round-trip.
//! * **Pixel decode is unaffected** — verify the decoded pixels still
//!   match the original input bit-for-bit when metadata is attached
//!   (lossless VP8L means byte-identical, so any drift would be a
//!   regression in the metadata-attach plumbing).

use oxideav_webp::riff::{build_vp8l_with_alpha, WebpMetadata};
use oxideav_webp::{decode_webp, encode_vp8l_argb, extract_metadata};

const W: u32 = 16;
const H: u32 = 16;

/// Build a 16×16 RGBA image with a varied colour pattern + a partially-
/// transparent alpha channel (so the encoder takes the extended layout
/// even before we attach metadata).
fn build_test_rgba() -> (Vec<u8>, Vec<u32>) {
    let mut rgba = Vec::with_capacity((W * H * 4) as usize);
    let mut argb = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let r = (x * 16) as u8;
            let g = (y * 16) as u8;
            let b = ((x + y) * 8) as u8;
            // Diagonal alpha ramp — not uniformly opaque so VP8X with
            // ALPHA flag is mandatory.
            let a = ((x + y) * 8) as u8;
            rgba.extend_from_slice(&[r, g, b, a]);
            argb.push(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
        }
    }
    (rgba, argb)
}

/// Synthesise a tiny but distinctive payload for each metadata chunk
/// type. Real-world ICC profiles / EXIF blobs / XMP packets are much
/// bigger but the round-trip property we test here is "bytes in == bytes
/// out", so a 16-byte payload with a recognisable signature is enough.
fn icc_payload() -> Vec<u8> {
    // Real ICC profiles start with a 4-byte size + 4-byte CMM + version;
    // we don't validate the contents, so just use a recognisable header.
    let mut v = b"ICC_TEST".to_vec();
    v.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04]);
    v
}

fn exif_payload() -> Vec<u8> {
    // Real EXIF starts with "II*\0" (little-endian TIFF) — keep it
    // recognisable but don't bother with valid IFD offsets.
    let mut v = b"II*\0".to_vec();
    v.extend_from_slice(b"oxideav-webp test EXIF payload");
    v
}

fn xmp_payload() -> Vec<u8> {
    // Real XMP packets are UTF-8 XML wrapped in <?xpacket begin … ?>
    // boundaries; for round-trip testing the literal bytes don't matter.
    b"<?xml version=\"1.0\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec()
}

#[test]
fn vp8x_with_icc_exif_xmp_round_trips_all_three_chunks() {
    let (rgba_in, argb) = build_test_rgba();
    let bitstream = encode_vp8l_argb(W, H, &argb, true).expect("vp8l encode");

    let icc = icc_payload();
    let exif = exif_payload();
    let xmp = xmp_payload();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: Some(&exif),
        xmp: Some(&xmp),
    };
    let webp_bytes = build_vp8l_with_alpha(&bitstream, W, H, &meta);

    // Sanity: the file has the expected outer shape (RIFF/WEBP/VP8X).
    assert_eq!(&webp_bytes[0..4], b"RIFF", "missing RIFF magic");
    assert_eq!(&webp_bytes[8..12], b"WEBP", "missing WEBP form type");
    assert_eq!(
        &webp_bytes[12..16],
        b"VP8X",
        "metadata-attached file must take the extended VP8X layout"
    );
    // VP8X flags byte at offset 20 — ICCP|EXIF|XMP|ALPHA bits all set
    // (0x20 | 0x08 | 0x04 | 0x10 = 0x3c).
    let flags = webp_bytes[20];
    assert!(
        flags & 0x20 != 0,
        "ICC flag missing in VP8X (flags={flags:#x})"
    );
    assert!(
        flags & 0x08 != 0,
        "EXIF flag missing in VP8X (flags={flags:#x})"
    );
    assert!(
        flags & 0x04 != 0,
        "XMP flag missing in VP8X (flags={flags:#x})"
    );
    assert!(
        flags & 0x10 != 0,
        "ALPHA flag missing in VP8X (flags={flags:#x})"
    );

    // Metadata-only fast path: should return all three chunks byte-
    // identical without touching the pixel decoder.
    let extracted = extract_metadata(&webp_bytes).expect("extract_metadata");
    assert_eq!(
        extracted.icc.as_deref(),
        Some(icc.as_slice()),
        "ICC payload mismatch"
    );
    assert_eq!(
        extracted.exif.as_deref(),
        Some(exif.as_slice()),
        "EXIF payload mismatch"
    );
    assert_eq!(
        extracted.xmp.as_deref(),
        Some(xmp.as_slice()),
        "XMP payload mismatch"
    );
    assert!(
        extracted.any(),
        "any() must return true when chunks are present"
    );

    // Full decode: pixels round-trip lossless AND metadata is exposed
    // on the WebpImage (no need for the caller to make a second parse
    // pass).
    let img = decode_webp(&webp_bytes).expect("decode_webp");
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
    assert_eq!(img.frames.len(), 1);
    assert_eq!(
        img.frames[0].rgba, rgba_in,
        "VP8L is lossless — decoded pixels must match input byte-for-byte"
    );
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.metadata.xmp.as_deref(), Some(xmp.as_slice()));
}

#[test]
fn simple_layout_decode_returns_empty_metadata() {
    // A fully-opaque RGBA frame takes the simple `RIFF/WEBP/VP8L` layout
    // with NO `VP8X` header — by construction it can't carry metadata.
    // The decoder must surface an empty `WebpFileMetadata` (not error,
    // not panic, not fabricate values).
    let mut argb = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let r = x as u32;
            let g = y as u32;
            argb.push(0xff00_0000 | (r << 16) | (g << 8));
        }
    }
    let bitstream = encode_vp8l_argb(W, H, &argb, false).expect("vp8l encode");

    // Wrap in the simple layout via build_webp_file with no metadata
    // and no ALPH — this is exactly what `encoder.rs` does for opaque
    // input.
    let webp_bytes = oxideav_webp::riff::build_webp_file(
        oxideav_webp::riff::ImageKind::Vp8lLossless,
        &bitstream,
        W,
        H,
        None,
        &WebpMetadata::default(),
    );
    assert_eq!(
        &webp_bytes[12..16],
        b"VP8L",
        "opaque metadata-free encode must take the simple VP8L layout"
    );

    let extracted = extract_metadata(&webp_bytes).expect("extract_metadata on simple file");
    assert!(
        !extracted.any(),
        "simple-layout file must not surface any metadata"
    );
    assert!(extracted.icc.is_none());
    assert!(extracted.exif.is_none());
    assert!(extracted.xmp.is_none());

    let img = decode_webp(&webp_bytes).expect("decode_webp on simple file");
    assert!(
        !img.metadata.any(),
        "WebpImage.metadata.any() must be false"
    );
}

#[test]
fn extract_metadata_only_icc_no_exif_no_xmp() {
    // Subset case: only one of the three chunks present. Verifies that
    // the parser doesn't conflate chunks (e.g. accidentally writing the
    // ICC bytes into the EXIF slot).
    let (_, argb) = build_test_rgba();
    let bitstream = encode_vp8l_argb(W, H, &argb, true).expect("vp8l encode");
    let icc = icc_payload();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: None,
        xmp: None,
    };
    let webp_bytes = build_vp8l_with_alpha(&bitstream, W, H, &meta);
    let extracted = extract_metadata(&webp_bytes).expect("extract_metadata");
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
    assert!(extracted.exif.is_none(), "EXIF must stay None");
    assert!(extracted.xmp.is_none(), "XMP must stay None");
}

#[test]
fn extract_metadata_rejects_garbage_input() {
    // Inputs that aren't a valid RIFF/WEBP container must raise an
    // error, not silently return an empty metadata struct (callers may
    // be relying on the error to detect probing failures).
    let garbage = b"not a webp file at all";
    assert!(
        extract_metadata(garbage).is_err(),
        "extract_metadata must reject non-RIFF inputs"
    );

    // RIFF magic but wrong form-type ("AVI " instead of "WEBP").
    let mut not_webp = vec![0u8; 32];
    not_webp[..4].copy_from_slice(b"RIFF");
    not_webp[4..8].copy_from_slice(&24u32.to_le_bytes());
    not_webp[8..12].copy_from_slice(b"AVI ");
    assert!(
        extract_metadata(&not_webp).is_err(),
        "extract_metadata must reject non-WEBP form types"
    );
}
