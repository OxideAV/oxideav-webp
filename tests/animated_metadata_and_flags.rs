//! Round-trip tests for the WebP-layer features added in round 39:
//!
//! 1. **Animated WebP file-level metadata pass-through**. The animated
//!    encoder previously had no way to attach `ICCP` / `EXIF` / `XMP `
//!    chunks; this round wires it through [`AnimEncoderOptions::metadata`].
//!    Tests verify the chunks are written in the spec-mandated order
//!    (ICCP after VP8X; EXIF / XMP after the last ANMF) and that both
//!    `extract_metadata` and `decode_webp` surface the bytes verbatim.
//! 2. **Per-frame alpha → VP8X ALPHA flag fidelity**. The previous
//!    encoder unconditionally set `flags = 0x12` (ANIM | ALPHA) on every
//!    animation. We now scan every frame for non-opaque pixels and only
//!    set the ALPHA bit when at least one frame actually carries alpha.
//! 3. **Standalone `encode_vp8l_argb_with_metadata` entry point**. Image-
//!    library consumers that want lossless `.webp` with a colour profile
//!    or EXIF can now do it in one call without going through the
//!    registry trait dance or the lower-level riff helpers.

use oxideav_webp::{
    build_animated_webp_with_options, decode_webp, encode_vp8l_argb_with_metadata,
    extract_metadata, AnimEncoderOptions, AnimFrame, WebpMetadata,
};

const W: u32 = 16;
const H: u32 = 16;

fn solid(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for _ in 0..(w * h) {
        v.extend_from_slice(&rgba);
    }
    v
}

fn icc_payload() -> Vec<u8> {
    let mut v = b"ICC_TEST_ANIM".to_vec();
    v.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    v
}

fn exif_payload() -> Vec<u8> {
    // Real EXIF starts "II*\0"; this just needs to be recognisable for
    // round-trip equality.
    let mut v = b"II*\0".to_vec();
    v.extend_from_slice(b"oxideav-webp anim EXIF");
    v
}

fn xmp_payload() -> Vec<u8> {
    b"<?xml version=\"1.0\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec()
}

#[test]
fn animated_with_metadata_round_trips_all_three_chunks() {
    let f0 = solid(W, H, [0xff, 0, 0, 0xff]);
    let f1 = solid(W, H, [0, 0xff, 0, 0xff]);
    let frames = [
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f0,
        },
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let icc = icc_payload();
    let exif = exif_payload();
    let xmp = xmp_payload();
    let opts = AnimEncoderOptions {
        metadata: WebpMetadata {
            icc: Some(&icc),
            exif: Some(&exif),
            xmp: Some(&xmp),
        },
        ..Default::default()
    };
    let blob =
        build_animated_webp_with_options(W, H, [0u8; 4], 0, &frames, opts).expect("anim encode");

    // Sanity: outer shape is RIFF/WEBP/VP8X.
    assert_eq!(&blob[0..4], b"RIFF");
    assert_eq!(&blob[8..12], b"WEBP");
    assert_eq!(&blob[12..16], b"VP8X");

    // VP8X flags byte (offset 20) must have ANIM, ICCP, EXIF, XMP set.
    let flags = blob[20];
    assert_ne!(flags & 0x02, 0, "ANIM bit missing (flags={flags:#x})");
    assert_ne!(flags & 0x20, 0, "ICCP bit missing (flags={flags:#x})");
    assert_ne!(flags & 0x08, 0, "EXIF bit missing (flags={flags:#x})");
    assert_ne!(flags & 0x04, 0, "XMP bit missing (flags={flags:#x})");
    // No frame in this animation carries alpha → ALPHA bit must be 0.
    assert_eq!(
        flags & 0x10,
        0,
        "ALPHA bit set on a fully-opaque animation (flags={flags:#x})"
    );

    // Round-trip via metadata-only fast path.
    let extracted = extract_metadata(&blob).expect("extract_metadata");
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(extracted.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(extracted.xmp.as_deref(), Some(xmp.as_slice()));

    // Round-trip via full decode (frames decoded too).
    let img = decode_webp(&blob).expect("decode_webp");
    assert_eq!(img.frames.len(), 2);
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.metadata.xmp.as_deref(), Some(xmp.as_slice()));
}

#[test]
fn animated_iccp_chunk_immediately_follows_vp8x() {
    // The container spec requires ICCP to come right after VP8X (it's
    // the only chunk that must be in a fixed slot; everything else can
    // be in any order). Verify we honour that.
    let f = solid(W, H, [0x80, 0x80, 0x80, 0xff]);
    let frames = [AnimFrame {
        width: W,
        height: H,
        x_offset: 0,
        y_offset: 0,
        duration_ms: 50,
        blend: false,
        dispose_to_background: false,
        rgba: &f,
    }];
    let icc = icc_payload();
    let opts = AnimEncoderOptions {
        metadata: WebpMetadata {
            icc: Some(&icc),
            ..Default::default()
        },
        ..Default::default()
    };
    let blob =
        build_animated_webp_with_options(W, H, [0u8; 4], 0, &frames, opts).expect("anim encode");

    // 12 bytes of RIFF envelope, then VP8X (8-byte hdr + 10-byte payload = 18 bytes).
    let vp8x_chunk_size = u32::from_le_bytes([blob[16], blob[17], blob[18], blob[19]]) as usize;
    assert_eq!(vp8x_chunk_size, 10, "VP8X payload should be 10 bytes");
    let after_vp8x = 12 + 8 + vp8x_chunk_size; // even already
    assert_eq!(
        &blob[after_vp8x..after_vp8x + 4],
        b"ICCP",
        "ICCP must immediately follow VP8X (got {:?})",
        std::str::from_utf8(&blob[after_vp8x..after_vp8x + 4]).unwrap_or("???")
    );
}

#[test]
fn animated_alpha_flag_omitted_when_all_frames_opaque() {
    // Pre-round-39 the encoder ALWAYS set ALPHA — wasteful, and some
    // strict readers treat it as malformed when no alpha is found.
    let f = solid(W, H, [0xff, 0x80, 0x40, 0xff]);
    let frames = [AnimFrame {
        width: W,
        height: H,
        x_offset: 0,
        y_offset: 0,
        duration_ms: 50,
        blend: false,
        dispose_to_background: false,
        rgba: &f,
    }];
    let blob =
        build_animated_webp_with_options(W, H, [0u8; 4], 0, &frames, AnimEncoderOptions::default())
            .expect("anim encode");
    let flags = blob[20];
    assert_eq!(
        flags & 0x10,
        0,
        "fully-opaque animation set ALPHA bit (flags={flags:#x})"
    );
    assert_ne!(flags & 0x02, 0, "ANIM bit must still be set");
}

#[test]
fn animated_alpha_flag_set_when_any_frame_carries_alpha() {
    // F0 fully opaque, F1 carries semi-transparent pixels → ALPHA must
    // be set even though only one of the two frames needs it.
    let f0 = solid(W, H, [0xff, 0, 0, 0xff]);
    let f1 = solid(W, H, [0, 0xff, 0, 0x80]);
    let frames = [
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f0,
        },
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let blob =
        build_animated_webp_with_options(W, H, [0u8; 4], 0, &frames, AnimEncoderOptions::default())
            .expect("anim encode");
    let flags = blob[20];
    assert_ne!(
        flags & 0x10,
        0,
        "second-frame alpha didn't trigger canvas ALPHA flag (flags={flags:#x})"
    );
}

#[test]
fn standalone_vp8l_with_metadata_opaque_uses_extended_layout() {
    // Opaque RGBA + ICCP attached: the encoder must promote to the
    // extended layout (VP8X + ICCP + VP8L) WITHOUT setting the ALPHA
    // bit (no alpha sidecar, no per-pixel alpha).
    let mut argb = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            argb.push(0xff00_0000 | (x << 16) | (y << 8));
        }
    }
    let icc = icc_payload();
    let meta = WebpMetadata {
        icc: Some(&icc),
        ..Default::default()
    };
    let blob =
        encode_vp8l_argb_with_metadata(W, H, &argb, false, &meta).expect("encode w/ metadata");

    assert_eq!(&blob[0..4], b"RIFF");
    assert_eq!(&blob[8..12], b"WEBP");
    assert_eq!(
        &blob[12..16],
        b"VP8X",
        "metadata-attached opaque file must take VP8X layout"
    );
    let flags = blob[20];
    assert_ne!(flags & 0x20, 0, "ICCP bit missing (flags={flags:#x})");
    assert_eq!(
        flags & 0x10,
        0,
        "ALPHA bit set on an opaque + ICC-only file (flags={flags:#x})"
    );

    // Pixels must still round-trip lossless and metadata must be
    // surfaced on the decoded WebpImage.
    let img = decode_webp(&blob).expect("decode");
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert!(img.metadata.exif.is_none());
    assert!(img.metadata.xmp.is_none());
    assert_eq!(img.frames.len(), 1);
    let f0 = &img.frames[0];
    assert_eq!(f0.width, W);
    assert_eq!(f0.height, H);
}

#[test]
fn standalone_vp8l_with_metadata_alpha_extended_layout_both_flags() {
    // Alpha + ICCP attached together. Must take extended layout with
    // both ALPHA (0x10) and ICCP (0x20) bits set.
    let mut argb = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            // Alpha gradient — definitely non-opaque on most rows.
            let a = (x + y) * 8;
            argb.push((a << 24) | (x << 16) | (y << 8));
        }
    }
    let icc = icc_payload();
    let meta = WebpMetadata {
        icc: Some(&icc),
        ..Default::default()
    };
    let blob =
        encode_vp8l_argb_with_metadata(W, H, &argb, true, &meta).expect("encode w/ metadata");

    let flags = blob[20];
    assert_ne!(flags & 0x10, 0, "ALPHA bit missing (flags={flags:#x})");
    assert_ne!(flags & 0x20, 0, "ICCP bit missing (flags={flags:#x})");

    // Round-trip metadata.
    let extracted = extract_metadata(&blob).expect("extract_metadata");
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
}

#[test]
fn standalone_vp8l_no_metadata_stays_simple_layout() {
    // No alpha, no metadata → simple `RIFF/WEBP/VP8L` layout. Same as
    // the old `finalize_vp8l_file` path; verify the new entry point
    // doesn't accidentally promote to VP8X for the empty metadata case.
    let argb = vec![0xff80_8080u32; (W * H) as usize];
    let blob = encode_vp8l_argb_with_metadata(W, H, &argb, false, &WebpMetadata::default())
        .expect("encode opaque no meta");
    assert_eq!(&blob[12..16], b"VP8L", "should stay on simple VP8L layout");
}
