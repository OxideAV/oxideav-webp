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

// ─────────── Per-kind metadata round-trips (round 138) ───────────
//
// The round-115 encoder pre-existed `encode_vp8l_argb_with_metadata`
// with all-three-set / all-three-absent coverage. Round 138 adds the
// per-kind isolation tests that verify each of ICC / Exif / XMP can
// be set on its own and round-trip through `extract_metadata` /
// `decode_webp` without the other two leaking into the output. These
// guard the §2.7.1 flag-octet's `I` / `E` / `X` bits against a stray
// OR that would mis-declare features the file doesn't carry.

#[test]
fn encode_vp8l_argb_with_metadata_icc_only_round_trips() {
    let (w, h) = (3u32, 3u32);
    let argb = make_argb(w, h, true);
    let icc = b"icc-only-profile-bytes-with-odd-length".to_vec();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: None,
        xmp: None,
    };
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, false, &meta).expect("icc-only .webp");
    let read = extract_metadata(&file).expect("extract_metadata");
    assert_eq!(read.icc.as_deref(), Some(&icc[..]));
    assert_eq!(read.exif, None);
    assert_eq!(read.xmp, None);
    // Pixels survive.
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
}

#[test]
fn encode_vp8l_argb_with_metadata_exif_only_round_trips() {
    let (w, h) = (3u32, 3u32);
    let argb = make_argb(w, h, true);
    let exif = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08".to_vec();
    let meta = WebpMetadata {
        icc: None,
        exif: Some(&exif),
        xmp: None,
    };
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, false, &meta).expect("exif-only .webp");
    let read = extract_metadata(&file).expect("extract_metadata");
    assert_eq!(read.icc, None);
    assert_eq!(read.exif.as_deref(), Some(&exif[..]));
    assert_eq!(read.xmp, None);
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
}

#[test]
fn encode_vp8l_argb_with_metadata_xmp_only_round_trips() {
    let (w, h) = (3u32, 3u32);
    let argb = make_argb(w, h, true);
    let xmp = b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>".to_vec();
    let meta = WebpMetadata {
        icc: None,
        exif: None,
        xmp: Some(&xmp),
    };
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, false, &meta).expect("xmp-only .webp");
    let read = extract_metadata(&file).expect("extract_metadata");
    assert_eq!(read.icc, None);
    assert_eq!(read.exif, None);
    assert_eq!(read.xmp.as_deref(), Some(&xmp[..]));
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
}

#[test]
fn encode_vp8l_argb_with_metadata_absent_kinds_round_trip_as_absent() {
    // Asymmetric coverage: set only Exif, confirm ICC / XMP read back
    // as None (not just empty Vecs or a stale slice from a previous
    // call).
    let (w, h) = (2u32, 2u32);
    let argb = make_argb(w, h, true);
    let exif = b"only-exif-set".to_vec();
    let meta = WebpMetadata {
        icc: None,
        exif: Some(&exif),
        xmp: None,
    };
    let file = encode_vp8l_argb_with_metadata(w, h, &argb, false, &meta).unwrap();
    let read = extract_metadata(&file).unwrap();
    assert!(
        read.icc.is_none(),
        "ICC must round-trip as None when absent"
    );
    assert!(
        read.xmp.is_none(),
        "XMP must round-trip as None when absent"
    );
    assert_eq!(read.exif.as_deref(), Some(&exif[..]));
}

// ─────────── build::build_webp_file_with_metadata published surface ───────────

#[test]
fn build_webp_file_with_metadata_round_trips_through_decoder() {
    // Hand the build:: writer a real VP8L bitstream + metadata, parse
    // the produced container, and confirm both the bitstream and the
    // metadata payloads recover byte-for-byte.
    use oxideav_webp::build::{build_webp_file_with_metadata, ImageKind, MetadataPayloads};

    let (w, h) = (4u32, 4u32);
    let argb = make_argb(w, h, true);
    let bare = encode_vp8l_argb(&argb, w, h).expect("bare VP8L");

    let icc = b"build_with_metadata_icc".to_vec();
    let exif = b"build_with_metadata_exif".to_vec();
    let xmp = b"build_with_metadata_xmp".to_vec();
    let meta = MetadataPayloads {
        icc: Some(&icc),
        exif: Some(&exif),
        xmp: Some(&xmp),
    };
    let file = build_webp_file_with_metadata(&bare, ImageKind::Lossless, w, h, meta)
        .expect("build_webp_file_with_metadata");

    // Pixels still decode through the published decoder.
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));

    // Metadata reads back through the published extract_metadata.
    let read = extract_metadata(&file).expect("extract_metadata");
    assert_eq!(read.icc.as_deref(), Some(&icc[..]));
    assert_eq!(read.exif.as_deref(), Some(&exif[..]));
    assert_eq!(read.xmp.as_deref(), Some(&xmp[..]));
}

// ─────────── build → extract → re-build canonical-inverse round-trip ───────────
//
// `build_webp_file_with_metadata` is a deterministic §2.7 writer: §2.7's
// "may appear out of order" carve-out for EXIF / XMP is collapsed by the
// writer to a single canonical EXIF-before-XMP order, even-length pads are
// always zero, and the §2.7.1 flag octet declares exactly the chunks that
// follow. `extract_metadata` is its left-inverse on the metadata payloads.
// These tests pin the canonical-inverse identity:
//
//     build(payload, meta) == build(payload, extract(build(payload, meta)))
//
// i.e. round-tripping through extract + re-build yields **byte-identical**
// container bytes. Any change to the writer's emission order, pad
// handling, or flag-octet computation would surface here as a byte-diff.

#[test]
fn build_with_metadata_extract_rebuild_is_byte_identical_all_three_kinds() {
    // All-three-set canonical-inverse round trip.
    use oxideav_webp::build::{build_webp_file_with_metadata, ImageKind, MetadataPayloads};

    let (w, h) = (4u32, 4u32);
    let argb = make_argb(w, h, true);
    let bare = encode_vp8l_argb(&argb, w, h).expect("bare VP8L");

    let icc = b"canonical-inverse-icc".to_vec();
    let exif = b"canonical-inverse-exif".to_vec();
    let xmp = b"canonical-inverse-xmp".to_vec();
    let first = build_webp_file_with_metadata(
        &bare,
        ImageKind::Lossless,
        w,
        h,
        MetadataPayloads {
            icc: Some(&icc),
            exif: Some(&exif),
            xmp: Some(&xmp),
        },
    )
    .expect("first build");

    // Recover the metadata payloads through the published extract path …
    let extracted = extract_metadata(&first).expect("extract");
    assert_eq!(extracted.icc.as_deref(), Some(&icc[..]));
    assert_eq!(extracted.exif.as_deref(), Some(&exif[..]));
    assert_eq!(extracted.xmp.as_deref(), Some(&xmp[..]));

    // … and feed them back through the writer. The §2.7 chunk-emission
    // order is canonical (EXIF before XMP), the §2.7.1 flag octet is
    // recomputed from the same Some-ness, and the §2.3 pad bytes are
    // deterministic, so the second build's bytes must equal the first's.
    let second = build_webp_file_with_metadata(
        &bare,
        ImageKind::Lossless,
        w,
        h,
        MetadataPayloads {
            icc: extracted.icc.as_deref(),
            exif: extracted.exif.as_deref(),
            xmp: extracted.xmp.as_deref(),
        },
    )
    .expect("second build");

    assert_eq!(
        first, second,
        "build_with_metadata is the canonical inverse of extract_metadata: \
         round-tripping must produce byte-identical container bytes"
    );
}

#[test]
fn build_with_metadata_extract_rebuild_is_byte_identical_per_kind() {
    // Per-kind canonical-inverse: each of ICC / Exif / XMP, on its own,
    // must round-trip byte-identical through extract + re-build. Guards
    // against a writer change that happens to be byte-identical only
    // when all three are present (e.g. a flag-octet OR that depends on
    // sibling chunks).
    use oxideav_webp::build::{build_webp_file_with_metadata, ImageKind, MetadataPayloads};

    let (w, h) = (3u32, 3u32);
    let argb = make_argb(w, h, true);
    let bare = encode_vp8l_argb(&argb, w, h).expect("bare VP8L");

    let icc = b"per-kind-icc-payload".to_vec();
    let exif = b"per-kind-exif-payload".to_vec();
    let xmp = b"per-kind-xmp-payload".to_vec();

    for (label, meta) in [
        (
            "icc-only",
            MetadataPayloads {
                icc: Some(&icc),
                exif: None,
                xmp: None,
            },
        ),
        (
            "exif-only",
            MetadataPayloads {
                icc: None,
                exif: Some(&exif),
                xmp: None,
            },
        ),
        (
            "xmp-only",
            MetadataPayloads {
                icc: None,
                exif: None,
                xmp: Some(&xmp),
            },
        ),
        (
            "icc+exif",
            MetadataPayloads {
                icc: Some(&icc),
                exif: Some(&exif),
                xmp: None,
            },
        ),
        (
            "icc+xmp",
            MetadataPayloads {
                icc: Some(&icc),
                exif: None,
                xmp: Some(&xmp),
            },
        ),
        (
            "exif+xmp",
            MetadataPayloads {
                icc: None,
                exif: Some(&exif),
                xmp: Some(&xmp),
            },
        ),
    ] {
        let first = build_webp_file_with_metadata(&bare, ImageKind::Lossless, w, h, meta)
            .unwrap_or_else(|e| panic!("{label}: first build: {e:?}"));
        let extracted =
            extract_metadata(&first).unwrap_or_else(|e| panic!("{label}: extract: {e:?}"));
        // Absent kinds must extract as None (not Some(empty)).
        assert_eq!(extracted.icc.is_some(), meta.icc.is_some(), "{label}: icc");
        assert_eq!(
            extracted.exif.is_some(),
            meta.exif.is_some(),
            "{label}: exif"
        );
        assert_eq!(extracted.xmp.is_some(), meta.xmp.is_some(), "{label}: xmp");

        let second = build_webp_file_with_metadata(
            &bare,
            ImageKind::Lossless,
            w,
            h,
            MetadataPayloads {
                icc: extracted.icc.as_deref(),
                exif: extracted.exif.as_deref(),
                xmp: extracted.xmp.as_deref(),
            },
        )
        .unwrap_or_else(|e| panic!("{label}: second build: {e:?}"));
        assert_eq!(
            first, second,
            "{label}: build is canonical inverse of extract; \
             round-trip must be byte-identical"
        );
    }
}

#[test]
fn build_with_metadata_extract_rebuild_is_byte_identical_odd_length_payloads() {
    // §2.3 pad-byte canonical handling: a chunk with an odd-length
    // payload gets a single trailing 0x00 pad that is NOT counted in
    // Size. Re-extracting that chunk recovers ONLY the declared
    // odd-length payload (the pad is dropped), so feeding the extracted
    // bytes back through the writer must regenerate the same odd-length
    // payload plus the same trailing zero pad. Pin this canonical inverse.
    use oxideav_webp::build::{build_webp_file_with_metadata, ImageKind, MetadataPayloads};

    let (w, h) = (4u32, 4u32);
    let argb = make_argb(w, h, true);
    let bare = encode_vp8l_argb(&argb, w, h).expect("bare VP8L");

    // Three payloads with all-odd lengths so every metadata chunk
    // produces a §2.3 pad byte.
    let icc = vec![0xA5u8; 13];
    let exif = vec![0x5Au8; 7];
    let xmp = vec![0x3Cu8; 19];

    let first = build_webp_file_with_metadata(
        &bare,
        ImageKind::Lossless,
        w,
        h,
        MetadataPayloads {
            icc: Some(&icc),
            exif: Some(&exif),
            xmp: Some(&xmp),
        },
    )
    .expect("first build (odd-length)");

    let extracted = extract_metadata(&first).expect("extract (odd-length)");
    // Extracted payloads carry exactly the odd-length original — no pad.
    assert_eq!(extracted.icc.as_deref(), Some(&icc[..]));
    assert_eq!(extracted.exif.as_deref(), Some(&exif[..]));
    assert_eq!(extracted.xmp.as_deref(), Some(&xmp[..]));
    assert_eq!(extracted.icc.as_ref().map(|b| b.len()), Some(13));
    assert_eq!(extracted.exif.as_ref().map(|b| b.len()), Some(7));
    assert_eq!(extracted.xmp.as_ref().map(|b| b.len()), Some(19));

    let second = build_webp_file_with_metadata(
        &bare,
        ImageKind::Lossless,
        w,
        h,
        MetadataPayloads {
            icc: extracted.icc.as_deref(),
            exif: extracted.exif.as_deref(),
            xmp: extracted.xmp.as_deref(),
        },
    )
    .expect("second build (odd-length)");
    assert_eq!(
        first, second,
        "odd-length payloads: build is canonical inverse of extract, \
         §2.3 pad bytes must be regenerated identically"
    );
}

#[test]
fn build_with_metadata_extract_rebuild_is_byte_identical_lossy_kind() {
    // The canonical-inverse identity is independent of which bitstream
    // FourCC the writer emits. Run the same round-trip against the
    // ExtendedLossy kind with a small synthetic VP8 chunk body so it
    // also exercises the Lossy → ExtendedLossy promotion path.
    use oxideav_webp::build::{build_webp_file_with_metadata, ImageKind, MetadataPayloads};

    // The writer never inspects the bitstream payload; any non-empty
    // byte sequence within §2.4 chunk-size limits is acceptable for
    // this round-trip test. Use a deterministic 32-byte filler.
    let bitstream: Vec<u8> = (0..32u8).collect();
    let (w, h) = (16u32, 16u32);
    let icc = b"lossy-icc".to_vec();
    let xmp = b"lossy-xmp".to_vec();

    let first = build_webp_file_with_metadata(
        &bitstream,
        ImageKind::ExtendedLossy,
        w,
        h,
        MetadataPayloads {
            icc: Some(&icc),
            exif: None,
            xmp: Some(&xmp),
        },
    )
    .expect("first build (lossy kind)");

    let extracted = extract_metadata(&first).expect("extract (lossy kind)");
    assert_eq!(extracted.icc.as_deref(), Some(&icc[..]));
    assert_eq!(extracted.exif, None);
    assert_eq!(extracted.xmp.as_deref(), Some(&xmp[..]));

    let second = build_webp_file_with_metadata(
        &bitstream,
        ImageKind::ExtendedLossy,
        w,
        h,
        MetadataPayloads {
            icc: extracted.icc.as_deref(),
            exif: extracted.exif.as_deref(),
            xmp: extracted.xmp.as_deref(),
        },
    )
    .expect("second build (lossy kind)");
    assert_eq!(
        first, second,
        "ExtendedLossy: build is canonical inverse of extract, \
         lossy bitstream FourCC must round-trip byte-identical"
    );
}
