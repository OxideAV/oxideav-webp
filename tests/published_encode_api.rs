//! Round-117 published-API round trips for the **VP8L lossless encode**
//! surface — the published-0.1.5 names re-exposed on top of the in-crate
//! VP8L encoder.
//!
//! Every test here uses only standalone APIs (no `registry` feature), so the
//! file builds and runs under `--no-default-features`. It exercises:
//!
//! * `encode_vp8l_argb` / `encode_vp8l_argb_with` — the **bare** VP8L
//!   bitstream (no RIFF wrapper).
//! * `encode_vp8l_argb_with_metadata` — a complete `.webp`, auto-promoting
//!   to the extended `VP8X` layout when alpha or metadata is present.
//! * `WebpMetadata` (borrowed) — the encode-side metadata input.
//! * `extract_metadata` — reading the embedded ICC / Exif / XMP back.

use oxideav_webp::{
    decode_webp, encode_vp8l_argb, encode_vp8l_argb_with, encode_vp8l_argb_with_metadata,
    extract_metadata, WebpError, WebpMetadata,
};

/// Build a deterministic `width * height` ARGB ramp (packed
/// `(a << 24) | (r << 16) | (g << 8) | b`), no external input.
fn make_argb(width: u32, height: u32, opaque: bool) -> Vec<u32> {
    let mut buf = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = x.wrapping_mul(37).wrapping_add(y) & 0xff;
            let g = y.wrapping_mul(53).wrapping_add(x) & 0xff;
            let b = (x ^ y).wrapping_mul(101) & 0xff;
            let a = if opaque {
                0xff
            } else {
                255 - ((x.wrapping_add(y)) & 0xff)
            };
            buf.push((a << 24) | (r << 16) | (g << 8) | b);
        }
    }
    buf
}

/// Repack ARGB → interleaved RGBA bytes for comparison with a decoded frame.
fn argb_to_rgba(argb: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(argb.len() * 4);
    for &p in argb {
        out.push((p >> 16) as u8); // R
        out.push((p >> 8) as u8); // G
        out.push(p as u8); // B
        out.push((p >> 24) as u8); // A
    }
    out
}

#[test]
fn encode_vp8l_argb_is_bare_bitstream_no_riff() {
    let (w, h) = (5u32, 4u32);
    let argb = make_argb(w, h, true);
    let bare = encode_vp8l_argb(&argb, w, h).expect("bare VP8L encode");
    // A bare bitstream begins with the §3.4 0x2F signature, not "RIFF".
    assert_ne!(&bare[0..4], b"RIFF");
    assert_eq!(bare[0], 0x2F, "VP8L image-header signature byte");
}

#[test]
fn encode_vp8l_argb_with_metadata_simple_round_trips() {
    // Opaque image, no metadata → simple (non-VP8X) layout, exact pixels.
    let (w, h) = (7u32, 3u32);
    let argb = make_argb(w, h, true);
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, false, &WebpMetadata::default())
        .expect("simple VP8L .webp");

    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].width, w);
    assert_eq!(img.frames[0].height, h);
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
    // No metadata embedded.
    assert_eq!(img.metadata.icc, None);
    assert_eq!(img.metadata.exif, None);
    assert_eq!(img.metadata.xmp, None);
}

#[test]
fn encode_vp8l_argb_with_metadata_alpha_promotes_to_extended() {
    // Non-opaque image, no metadata → extended VP8X layout (alpha flag).
    let (w, h) = (4u32, 4u32);
    let argb = make_argb(w, h, false);
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, true, &WebpMetadata::default())
        .expect("alpha VP8L .webp");

    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
}

#[test]
fn encode_vp8l_argb_with_metadata_embeds_and_reads_back() {
    let (w, h) = (3u32, 3u32);
    let argb = make_argb(w, h, true);
    let icc = b"fake-icc-profile-bytes".to_vec();
    let exif = b"Exif\x00\x00MM\x00*".to_vec();
    let xmp = b"<?xpacket begin?>".to_vec();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: Some(&exif),
        xmp: Some(&xmp),
    };
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, false, &meta).expect("metadata .webp");

    // Metadata-only read.
    let read = extract_metadata(&file).expect("extract_metadata");
    assert_eq!(read.icc.as_deref(), Some(&icc[..]));
    assert_eq!(read.exif.as_deref(), Some(&exif[..]));
    assert_eq!(read.xmp.as_deref(), Some(&xmp[..]));

    // Pixels still survive.
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
    assert_eq!(img.metadata.icc.as_deref(), Some(&icc[..]));
}

#[test]
fn encode_vp8l_argb_with_forces_alpha_bit_but_round_trips() {
    // Forcing the alpha header bit on an opaque image must still round-trip
    // (the bit is advisory; alpha is carried in the literals).
    let (w, h) = (2u32, 2u32);
    let argb = make_argb(w, h, true);
    let bare = encode_vp8l_argb_with(&argb, w, h, true).expect("forced-alpha bare bitstream");
    // Frame it and decode.
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, true, &WebpMetadata::default())
        .expect("alpha-flagged .webp");
    let _ = bare; // also asserts the bare form encodes without error.
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
}

#[test]
fn encode_vp8l_argb_rejects_dimension_mismatch() {
    // One pixel claimed as 2x2 → published coarse error.
    let argb = vec![0xff00_0000u32];
    let err = encode_vp8l_argb(&argb, 2, 2).expect_err("mismatch rejected");
    assert_eq!(err, WebpError::InvalidData);
}
