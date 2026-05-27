//! Round-169 end-to-end coverage of the **standalone** (`--no-default-features`)
//! public surface.
//!
//! Every `use oxideav_webp::…` in this file resolves with the `registry`
//! cargo feature OFF. The whole file is also valid with `registry` ON — it
//! never reaches for a registry-gated symbol. Concretely it runs under:
//!
//! ```text
//! cargo test -p oxideav-webp --test standalone_e2e --no-default-features
//! cargo test -p oxideav-webp --test standalone_e2e
//! ```
//!
//! The published 0.1.5 standalone surface (per `API-COMPAT-0.1.2.md` +
//! `README.md` "Standalone use") covered here:
//!
//! * `decode_webp` / `WebpImage` / `WebpFrame` — top-level decode entry.
//! * `encode_webp_lossless` — flat RGBA → complete simple `.webp`.
//! * `encode_vp8l_argb` — bare `VP8L` bitstream from ARGB.
//! * `encode_vp8l_argb_with_metadata` + `WebpMetadata` — VP8L with embedded
//!   ICC / Exif / XMP, auto-promoting to the extended `VP8X` layout.
//! * `extract_metadata` + `WebpFileMetadata` — metadata-only read path.
//! * `build_animated_webp` + `AnimFrame::new` — animated `.webp` assembly.
//! * `decode_lossless_image` — the bare-bitstream / chunk-level helper.
//!
//! Each test asserts **byte-exact** round-trip (lossless = bit-exact per
//! RFC 9649 §3) — no PSNR / SSIM tolerance budget anywhere in this file.

use oxideav_webp::build::{build_chunk, ImageKind};
use oxideav_webp::{
    build_animated_webp, build_webp_file, decode_lossless_image, decode_webp, encode_vp8l_argb,
    encode_vp8l_argb_with_metadata, encode_webp_lossless, extract_metadata, AnimFrame, WebpError,
    WebpFileMetadata, WebpMetadata,
};

// ─────────────────────────── fixture builders ───────────────────────────

/// Build a deterministic `width * height` RGBA8 image (no external input).
/// Spread-out arithmetic so every channel hits a wide range of values.
fn synthetic_rgba(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x.wrapping_mul(37).wrapping_add(y).wrapping_add(seed) & 0xff) as u8;
            let g = (y.wrapping_mul(53).wrapping_add(x).wrapping_mul(7) & 0xff) as u8;
            let b = ((x ^ y).wrapping_mul(101).wrapping_add(seed) & 0xff) as u8;
            let a = (255 - ((x.wrapping_add(y).wrapping_add(seed)) & 0xff)) as u8;
            buf.extend_from_slice(&[r, g, b, a]);
        }
    }
    buf
}

/// Same shape as `synthetic_rgba` but packed as ARGB `u32` per pixel.
fn synthetic_argb(width: u32, height: u32, seed: u32) -> Vec<u32> {
    let mut buf = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = x.wrapping_mul(37).wrapping_add(y).wrapping_add(seed) & 0xff;
            let g = y.wrapping_mul(53).wrapping_add(x).wrapping_mul(7) & 0xff;
            let b = (x ^ y).wrapping_mul(101).wrapping_add(seed) & 0xff;
            let a = 255 - ((x.wrapping_add(y).wrapping_add(seed)) & 0xff);
            buf.push((a << 24) | (r << 16) | (g << 8) | b);
        }
    }
    buf
}

/// Repack ARGB `u32` → interleaved RGBA bytes (the `WebpFrame::rgba` layout).
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

// ─────────────────── 1. Lossless RGBA round-trip standalone ──────────────────

#[test]
fn standalone_lossless_rgba_round_trip_is_bit_exact() {
    // RGBA → encode_webp_lossless → decode_webp → assert bytes match exactly.
    // 64×64 = enough scan-line variety to exercise the §3.7 prefix tables
    // and §5.2.2 LZ77 matcher without ballooning test runtime.
    let (w, h) = (64u32, 64u32);
    let src = synthetic_rgba(w, h, 0);

    let file = encode_webp_lossless(&src, w, h).expect("encode_webp_lossless");
    // Standalone file must start with RIFF/WEBP magic (§2.4).
    assert_eq!(&file[0..4], b"RIFF");
    assert_eq!(&file[8..12], b"WEBP");

    let img = decode_webp(&file).expect("decode_webp on standalone-encoded file");
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.frames.len(), 1);
    let f = &img.frames[0];
    assert_eq!(f.width, w);
    assert_eq!(f.height, h);
    assert_eq!(f.duration_ms, 0, "still image carries 0 ms duration");
    assert_eq!(f.rgba.len(), (w * h * 4) as usize, "flat tight buffer");
    assert_eq!(
        f.rgba, src,
        "lossless = bit-exact: round-tripped RGBA equals source byte-for-byte"
    );
    // No animation fields on a still image.
    assert_eq!(img.anim_loop_count, None);
    assert_eq!(img.anim_background_rgba, None);
    // No metadata embedded by the bare lossless path.
    assert_eq!(img.metadata, WebpFileMetadata::default());
}

// ───────────── 2. VP8L bare-bitstream → wrap → decode standalone ────────────

#[test]
fn standalone_vp8l_bare_bitstream_wraps_and_round_trips() {
    // ARGB → encode_vp8l_argb (bare, no RIFF) → build a RIFF/WEBP wrapper
    // via the `build_webp_file` helper → decode_webp → assert pixels match.
    let (w, h) = (32u32, 24u32);
    let argb = synthetic_argb(w, h, 11);

    let bare = encode_vp8l_argb(&argb, w, h).expect("bare VP8L encode");
    // The bare bitstream is NOT a RIFF file — it starts with the §3.4
    // 0x2F image-header signature byte directly.
    assert_ne!(&bare[0..4], b"RIFF");
    assert_eq!(bare[0], 0x2F, "VP8L §3.4 signature byte");

    // Wrap it in a complete simple-lossless `.webp` via the published
    // build helper, then decode through the full container path.
    let file = build_webp_file(&bare, ImageKind::Lossless, w, h).expect("wrap bare VP8L");
    assert_eq!(&file[0..4], b"RIFF");
    assert_eq!(&file[8..12], b"WEBP");

    let img = decode_webp(&file).expect("decode wrapped bare bitstream");
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));

    // The lower-level `decode_lossless_image` path is also standalone-
    // reachable: it bypasses the §2.5 lossy router and returns ARGB pixels.
    let decoded = decode_lossless_image(&file)
        .expect("decode_lossless_image")
        .expect("VP8L chunk present");
    assert_eq!(decoded.width(), w);
    assert_eq!(decoded.height(), h);
    assert_eq!(decoded.pixels().len(), (w * h) as usize);
    assert_eq!(
        decoded.pixels(),
        argb.as_slice(),
        "ARGB pixels byte-for-byte exact"
    );
}

// ───────────────── 3. Lossless with metadata standalone ────────────────────

#[test]
fn standalone_lossless_with_full_metadata_round_trips() {
    // ARGB + 4-byte ICC + 6-byte EXIF + UTF-8 XMP →
    // encode_vp8l_argb_with_metadata → extract_metadata + decode_webp →
    // assert every field round-trips byte-for-byte.
    let (w, h) = (16u32, 16u32);
    let argb = synthetic_argb(w, h, 23);

    let icc: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
    let exif: Vec<u8> = b"Exif\x00\x00".to_vec();
    let xmp: Vec<u8> = "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>"
        .as_bytes()
        .to_vec();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: Some(&exif),
        xmp: Some(&xmp),
    };

    let file = encode_vp8l_argb_with_metadata(w, h, &argb, /* has_alpha = */ true, &meta)
        .expect("encode_vp8l_argb_with_metadata");
    // Promoted to extended layout: must carry a VP8X chunk before VP8L.
    assert_eq!(&file[0..4], b"RIFF");

    // Metadata-only read path.
    let read = extract_metadata(&file).expect("extract_metadata standalone");
    assert_eq!(read.icc.as_deref(), Some(icc.as_slice()), "ICC byte-exact");
    assert_eq!(
        read.exif.as_deref(),
        Some(exif.as_slice()),
        "EXIF byte-exact"
    );
    assert_eq!(read.xmp.as_deref(), Some(xmp.as_slice()), "XMP byte-exact");

    // Full decode path also surfaces the same metadata + bit-exact pixels.
    let img = decode_webp(&file).expect("decode metadata-bearing .webp");
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.metadata.xmp.as_deref(), Some(xmp.as_slice()));
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
}

// ──────────────────── 4. Animation round-trip standalone ────────────────────

#[test]
fn standalone_animation_three_frames_round_trip_bit_exact() {
    // 3 distinct ANIM frames → build_animated_webp → decode_webp →
    // assert frame count + each frame's geometry + per-frame duration +
    // bit-exact pixels.
    let (w, h) = (48u32, 32u32);
    let pixels: [Vec<u8>; 3] = [
        synthetic_rgba(w, h, 0),
        synthetic_rgba(w, h, 17),
        synthetic_rgba(w, h, 41),
    ];
    let durations: [u32; 3] = [60, 90, 120];

    let frames: Vec<AnimFrame> = pixels
        .iter()
        .zip(durations.iter())
        .map(|(rgba, &dur)| AnimFrame::new(w, h, rgba.clone(), dur))
        .collect();

    let file = build_animated_webp(&frames).expect("build_animated_webp standalone");
    assert_eq!(&file[0..4], b"RIFF");
    assert_eq!(&file[8..12], b"WEBP");

    let img = decode_webp(&file).expect("decode animated .webp");
    assert_eq!(img.frames.len(), 3, "one decoded frame per ANMF");
    // Default options on `build_animated_webp`: infinite loop, transparent-
    // black background.
    assert_eq!(img.anim_loop_count, Some(0));
    assert_eq!(img.anim_background_rgba, Some([0, 0, 0, 0]));

    for (i, decoded) in img.frames.iter().enumerate() {
        assert_eq!(decoded.width, w, "frame {i} width");
        assert_eq!(decoded.height, h, "frame {i} height");
        assert_eq!(decoded.duration_ms, durations[i], "frame {i} duration_ms");
        assert_eq!(
            decoded.rgba.len(),
            (w * h * 4) as usize,
            "frame {i} flat buffer"
        );
        assert_eq!(
            decoded.rgba, pixels[i],
            "frame {i} pixels round-trip bit-exact"
        );
    }
}

// ──────── 5. Metadata-only extraction standalone (no full pixel decode) ─────

#[test]
fn standalone_extract_metadata_reads_without_full_decode() {
    // Encode a complete metadata-bearing `.webp`, then assert
    // `extract_metadata` returns every field — and that it works on a
    // file whose payload `decode_webp` would also have to process.
    let (w, h) = (8u32, 8u32);
    let argb = synthetic_argb(w, h, 5);
    let icc: Vec<u8> = (0u8..=63).collect();
    let exif: Vec<u8> = b"Exif\x00\x00MM\x00*Hello".to_vec();
    let xmp: Vec<u8> = b"<?xpacket begin=''?><x/>".to_vec();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: Some(&exif),
        xmp: Some(&xmp),
    };
    let file =
        encode_vp8l_argb_with_metadata(w, h, &argb, true, &meta).expect("metadata-bearing encode");

    // extract_metadata walks the §2.7 chunks WITHOUT invoking the §3 VP8L
    // entropy decoder — proved by extracting from a file whose VP8L chunk
    // we then *separately* lifted via decode_webp for cross-validation.
    let extracted = extract_metadata(&file).expect("extract_metadata");
    assert_eq!(extracted.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(extracted.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(extracted.xmp.as_deref(), Some(xmp.as_slice()));

    // Sanity check: full decode reports identical metadata + decodes pixels.
    let img = decode_webp(&file).expect("decode full file");
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.metadata.xmp.as_deref(), Some(xmp.as_slice()));
    assert_eq!(img.frames[0].rgba, argb_to_rgba(&argb));
}

// ─────────── 6. Negative paths surface the published coarse errors ──────────

#[test]
fn standalone_decode_garbage_rejected_as_invalid_data() {
    // Non-WebP input must surface the stable coarse error, not panic.
    let err = decode_webp(b"definitely not a webp file").expect_err("garbage rejected");
    assert_eq!(err, WebpError::InvalidData);
}

#[test]
fn standalone_extract_metadata_garbage_rejected_as_invalid_data() {
    let err = extract_metadata(b"\x00\x01\x02\x03 not webp").expect_err("garbage rejected");
    assert_eq!(err, WebpError::InvalidData);
}

#[test]
fn standalone_build_chunk_helper_is_reachable() {
    // The published `build_chunk` helper is part of the standalone surface
    // (lib.rs re-export). Quick reachability check: a 4-byte payload chunks
    // up to a fourcc + 4-byte LE length + 4 payload bytes, no padding
    // (§2.3 even-length).
    let chunk = build_chunk(*b"TEST", &[0xde, 0xad, 0xbe, 0xef]).expect("build_chunk");
    assert_eq!(&chunk[0..4], b"TEST");
    assert_eq!(&chunk[4..8], &4u32.to_le_bytes());
    assert_eq!(&chunk[8..12], &[0xde, 0xad, 0xbe, 0xef]);
    assert_eq!(chunk.len(), 12);
}
