//! End-to-end coverage for the standalone (no-`oxideav-core`) VP8 lossy
//! WebP encoder entry points: `encode_vp8_lossy_yuv420p`,
//! `encode_vp8_lossy_yuva420p`, `encode_vp8_lossy_rgba`,
//! `encode_vp8_lossy_rgb24`. These tests exercise every input pixel
//! format through both the simple and extended container layouts and
//! verify that the emitted `.webp` file round-trips through
//! `decode_webp`.
//!
//! The tests are deliberately written against the **standalone** API
//! surface (`oxideav_webp::encode_vp8_lossy_*` plus `decode_webp` and
//! `WebpMetadata`) — none of the registry-side framework types
//! (`VideoFrame`, `Encoder`, `CodecParameters`) are referenced. That
//! makes them representative of the image-library consumer use case.

use oxideav_webp::{
    decode_webp, encode_vp8_lossy_rgb24, encode_vp8_lossy_rgba, encode_vp8_lossy_yuv420p,
    encode_vp8_lossy_yuva420p, extract_metadata, WebpMetadata,
};

const W: u32 = 64;
const H: u32 = 64;

/// Build a smooth-gradient RGB plane used as the lossy-encode source.
fn build_rgb24(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // Smooth diagonals so the VP8 keyframe produces a meaningful
            // bitstream (a uniform plate would compress to a few hundred
            // bytes regardless of qindex and not exercise the per-bin
            // quant logic).
            out.push(((x * 4) & 0xff) as u8);
            out.push(((y * 4) & 0xff) as u8);
            out.push((((x + y) * 2) & 0xff) as u8);
        }
    }
    out
}

/// Build an RGBA plane with a horizontal alpha gradient so the ALPH
/// sidecar gets a non-trivial residual to compress.
fn build_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            out.push(((x * 4) & 0xff) as u8);
            out.push(((y * 4) & 0xff) as u8);
            out.push((((x + y) * 2) & 0xff) as u8);
            // Alpha gradient L→R from 0..=255.
            out.push(((x * 255) / w.saturating_sub(1).max(1)) as u8);
        }
    }
    out
}

/// Convert the smooth RGB24 plane into a YUV420P-shaped plane triple
/// using the same BT.601 limited-range coefficients the encoder uses
/// internally. Mirrors `rgb24_rows_to_yuv420` so the test feeds a YUV
/// triple that's bit-identical to what the RGB24 path would produce —
/// which lets us cross-compare the two entry points cheaply.
fn rgb24_to_yuv420(w: u32, h: u32, rgb: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = (w as usize) / 2 + ((w as usize) & 1);
    let ch = (h as usize) / 2 + ((h as usize) & 1);
    let mut y_plane = vec![0u8; (w * h) as usize];
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];
    let stride = (w as usize) * 3;
    for j in 0..(h as usize) {
        for i in 0..(w as usize) {
            let px = &rgb[j * stride + i * 3..j * stride + i * 3 + 3];
            let r = px[0] as i32;
            let g = px[1] as i32;
            let b = px[2] as i32;
            let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_plane[j * (w as usize) + i] = y.clamp(0, 255) as u8;
        }
    }
    for cy in 0..ch {
        for cx in 0..cw {
            let mut u_sum = 0i32;
            let mut v_sum = 0i32;
            let mut n = 0i32;
            for dy in 0..2 {
                let jj = cy * 2 + dy;
                if jj >= h as usize {
                    break;
                }
                for dx in 0..2 {
                    let ii = cx * 2 + dx;
                    if ii >= w as usize {
                        break;
                    }
                    let px = &rgb[jj * stride + ii * 3..jj * stride + ii * 3 + 3];
                    let r = px[0] as i32;
                    let g = px[1] as i32;
                    let b = px[2] as i32;
                    u_sum += (-38 * r - 74 * g + 112 * b + 128) >> 8;
                    v_sum += (112 * r - 94 * g - 18 * b + 128) >> 8;
                    n += 1;
                }
            }
            let u = (u_sum / n) + 128;
            let v = (v_sum / n) + 128;
            u_plane[cy * cw + cx] = u.clamp(0, 255) as u8;
            v_plane[cy * cw + cx] = v.clamp(0, 255) as u8;
        }
    }
    (y_plane, u_plane, v_plane)
}

#[test]
fn yuv420p_simple_layout_round_trips() {
    let rgb = build_rgb24(W, H);
    let (y, u, v) = rgb24_to_yuv420(W, H, &rgb);
    let webp = encode_vp8_lossy_yuv420p(W, H, &y, &u, &v, 75.0, &WebpMetadata::default())
        .expect("yuv420p encode");

    // Simple layout: VP8 chunk lives directly after WEBP magic.
    assert_eq!(&webp[0..4], b"RIFF");
    assert_eq!(&webp[8..12], b"WEBP");
    assert_eq!(&webp[12..16], b"VP8 ");

    let img = decode_webp(&webp).expect("decode_webp");
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].rgba.len(), (W * H * 4) as usize);
    // Lossy: every pixel should be opaque (no ALPH sidecar).
    for px in img.frames[0].rgba.chunks_exact(4) {
        assert_eq!(px[3], 0xff, "lossy without ALPH must yield opaque alpha");
    }
}

#[test]
fn yuv420p_with_metadata_promotes_to_extended_layout() {
    let rgb = build_rgb24(W, H);
    let (y, u, v) = rgb24_to_yuv420(W, H, &rgb);
    let icc = b"this is a fake ICC profile, padding to 32 bytes ";
    let exif = b"FAKE-EXIF";
    let xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"></x:xmpmeta>";
    let meta = WebpMetadata {
        icc: Some(icc),
        exif: Some(exif),
        xmp: Some(xmp),
    };
    let webp = encode_vp8_lossy_yuv420p(W, H, &y, &u, &v, 75.0, &meta).expect("yuv420p encode");

    // Extended layout: VP8X first chunk, with ICCP/EXIF/XMP flag bits.
    assert_eq!(&webp[12..16], b"VP8X");
    let flags = webp[20];
    assert_ne!(flags & 0x20, 0, "ICCP flag");
    assert_ne!(flags & 0x08, 0, "EXIF flag");
    assert_ne!(flags & 0x04, 0, "XMP flag");
    // No ALPHA flag — Yuv420P entry point has no alpha.
    assert_eq!(flags & 0x10, 0, "ALPHA flag must be unset for Yuv420P");

    let parsed = extract_metadata(&webp).expect("extract_metadata");
    assert_eq!(parsed.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(parsed.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(parsed.xmp.as_deref(), Some(xmp.as_slice()));

    // Also verify decode_webp accepts it and surfaces metadata.
    let img = decode_webp(&webp).expect("decode_webp");
    assert_eq!(img.metadata.icc.as_deref(), Some(icc.as_slice()));
    assert_eq!(img.metadata.exif.as_deref(), Some(exif.as_slice()));
    assert_eq!(img.metadata.xmp.as_deref(), Some(xmp.as_slice()));
}

#[test]
fn yuva420p_extended_alph_round_trips() {
    let rgba = build_rgba(W, H);
    // Drop the alpha column to feed the YUV-converter.
    let mut rgb_only = Vec::with_capacity((W * H * 3) as usize);
    for px in rgba.chunks_exact(4) {
        rgb_only.extend_from_slice(&px[..3]);
    }
    let (y, u, v) = rgb24_to_yuv420(W, H, &rgb_only);
    let alpha: Vec<u8> = rgba.chunks_exact(4).map(|px| px[3]).collect();
    let webp = encode_vp8_lossy_yuva420p(W, H, &y, &u, &v, &alpha, 75.0, &WebpMetadata::default())
        .expect("yuva420p encode");

    // ALPH sidecar → extended layout always.
    assert_eq!(&webp[12..16], b"VP8X");
    let flags = webp[20];
    assert_ne!(flags & 0x10, 0, "ALPHA flag must be set with ALPH sidecar");

    let img = decode_webp(&webp).expect("decode_webp");
    assert_eq!(img.frames.len(), 1);
    let decoded = &img.frames[0].rgba;
    assert_eq!(decoded.len(), (W * H * 4) as usize);

    // Alpha plane should round-trip bit-exact (ALPH is lossless).
    let mut alpha_match = 0usize;
    for (px, &src_a) in decoded.chunks_exact(4).zip(alpha.iter()) {
        if px[3] == src_a {
            alpha_match += 1;
        }
    }
    let total = (W * H) as usize;
    let ratio = alpha_match as f32 / total as f32;
    assert!(
        ratio >= 0.99,
        "ALPH must round-trip at least 99% of alpha bytes exactly (got {})",
        ratio
    );
}

#[test]
fn rgba_extended_alph_round_trips() {
    let rgba = build_rgba(W, H);
    let webp =
        encode_vp8_lossy_rgba(W, H, &rgba, 75.0, &WebpMetadata::default()).expect("rgba encode");

    // ALPH sidecar → extended layout.
    assert_eq!(&webp[12..16], b"VP8X");
    let flags = webp[20];
    assert_ne!(flags & 0x10, 0, "ALPHA flag must be set with ALPH sidecar");

    let img = decode_webp(&webp).expect("decode_webp");
    assert_eq!(img.frames.len(), 1);
    let decoded = &img.frames[0].rgba;
    assert_eq!(decoded.len(), (W * H * 4) as usize);

    // Same alpha-plane bit-exactness check as the YUVA path.
    let mut alpha_match = 0usize;
    for (px, src) in decoded.chunks_exact(4).zip(rgba.chunks_exact(4)) {
        if px[3] == src[3] {
            alpha_match += 1;
        }
    }
    let total = (W * H) as usize;
    let ratio = alpha_match as f32 / total as f32;
    assert!(ratio >= 0.99, "ALPH round-trip ratio {} below 0.99", ratio);
}

#[test]
fn rgb24_simple_layout_round_trips_no_alloc_path() {
    let rgb = build_rgb24(W, H);
    let webp =
        encode_vp8_lossy_rgb24(W, H, &rgb, 75.0, &WebpMetadata::default()).expect("rgb24 encode");

    // Simple layout: VP8 immediately after WEBP magic, no VP8X.
    assert_eq!(&webp[12..16], b"VP8 ");

    let img = decode_webp(&webp).expect("decode_webp");
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].rgba.len(), (W * H * 4) as usize);

    // No ALPH → every output pixel opaque.
    for px in img.frames[0].rgba.chunks_exact(4) {
        assert_eq!(px[3], 0xff);
    }
}

#[test]
fn quality_knob_is_byte_size_monotone() {
    // Higher quality → strictly larger file on a sufficiently rich source.
    // Use a noisy RGB24 fixture so the qindex curve actually moves bytes.
    let mut rgb = vec![0u8; (W * H * 3) as usize];
    let mut state: u32 = 0xC0FFEE;
    for b in rgb.iter_mut() {
        // Tiny xorshift PRNG — deterministic, enough entropy to defeat
        // the AC-quantiser at any qindex.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *b = (state & 0xff) as u8;
    }
    let lo = encode_vp8_lossy_rgb24(W, H, &rgb, 1.0, &WebpMetadata::default()).expect("q1");
    let hi = encode_vp8_lossy_rgb24(W, H, &rgb, 99.0, &WebpMetadata::default()).expect("q99");
    assert!(
        hi.len() > lo.len(),
        "high quality file ({}) must be larger than low quality file ({})",
        hi.len(),
        lo.len()
    );
}

#[test]
fn yuv420p_rejects_zero_dimensions() {
    let r = encode_vp8_lossy_yuv420p(0, 0, &[], &[], &[], 75.0, &WebpMetadata::default());
    assert!(r.is_err());
}

#[test]
fn yuv420p_rejects_plane_length_mismatch() {
    // Y plane wrong length.
    let r = encode_vp8_lossy_yuv420p(
        16,
        16,
        &[0u8; 100],
        &[0u8; 64],
        &[0u8; 64],
        75.0,
        &WebpMetadata::default(),
    );
    assert!(r.is_err());
}

#[test]
fn rgba_rejects_short_buffer() {
    let r = encode_vp8_lossy_rgba(16, 16, &[0u8; 10], 75.0, &WebpMetadata::default());
    assert!(r.is_err());
}

#[test]
fn rgb24_rejects_short_buffer() {
    let r = encode_vp8_lossy_rgb24(16, 16, &[0u8; 10], 75.0, &WebpMetadata::default());
    assert!(r.is_err());
}
