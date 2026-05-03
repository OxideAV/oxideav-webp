//! Integration tests for the VP8-lossy WebP encoder path
//! ([`oxideav_webp::encoder_vp8`]).
//!
//! The encoder takes a `Yuv420P` frame, runs it through the pure-Rust
//! `oxideav-vp8` keyframe encoder, and wraps the resulting bitstream in
//! a RIFF/WEBP container with a single `VP8 ` chunk. We verify the
//! full pipeline by:
//!
//! 1. Building a 128×128 YUV420P test pattern.
//! 2. Feeding it through [`encoder_vp8::make_encoder`] → the
//!    registered `Encoder` trait → a `.webp` packet.
//! 3. Checking the RIFF magic + `VP8 ` FourCC at the expected offsets.
//! 4. Decoding the packet bytes via [`oxideav_webp::decode_webp`]
//!    (the read path already handles RIFF/WEBP with a VP8 chunk).
//! 5. Converting the reconstructed RGBA back to YUV420P and asserting
//!    PSNR > 30 dB on the Y plane.

use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_webp::{decode_webp, encoder_vp8, CODEC_ID_VP8};

const W: u32 = 128;
const H: u32 = 128;

/// Build a deterministic YUV420P test pattern:
///   * Y = smooth diagonal luma gradient.
///   * U = horizontal chroma ramp.
///   * V = vertical chroma ramp.
fn build_test_pattern() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut y = vec![0u8; (W * H) as usize];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..H as usize {
        for col in 0..W as usize {
            // Luma: smooth diagonal gradient in [32..=223].
            let t = ((row + col) * 255) / (W as usize + H as usize - 2);
            y[row * W as usize + col] = (32 + (t * 191) / 255) as u8;
        }
    }
    for row in 0..ch {
        for col in 0..cw {
            u[row * cw + col] = (64 + (col * 127) / cw.max(1)) as u8;
            v[row * cw + col] = (64 + (row * 127) / ch.max(1)) as u8;
        }
    }
    (y, u, v)
}

fn make_yuv420_frame(y: &[u8], u: &[u8], v: &[u8]) -> VideoFrame {
    let cw = (W / 2) as usize;
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: W as usize,
                data: y.to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: u.to_vec(),
            },
            VideoPlane {
                stride: cw,
                data: v.to_vec(),
            },
        ],
    }
}

fn make_encoder_params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p
}

/// BT.601 limited-range RGB → Y conversion (same transform the WebP
/// decoder's YUV→RGB path uses, inverted). Matches the decoder's
/// BT.601 reverse cast closely enough for PSNR purposes.
fn rgb_to_y(r: u8, g: u8, b: u8) -> u8 {
    // Y  = 0.257 R + 0.504 G + 0.098 B + 16 (BT.601 limited range).
    // Use 8-bit fixed-point; rounds to the nearest integer.
    let y = (66 * r as i32 + 129 * g as i32 + 25 * b as i32 + 128) >> 8;
    (y + 16).clamp(0, 255) as u8
}

fn psnr_y(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[test]
fn vp8_lossy_webp_roundtrip_psnr_above_30() {
    let (y, u, v) = build_test_pattern();
    let frame = make_yuv420_frame(&y, &u, &v);

    let params = make_encoder_params();
    let mut enc = encoder_vp8::make_encoder(&params).expect("make vp8 encoder");
    enc.send_frame(&Frame::Video(frame)).expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive_packet");
    let webp_bytes = pkt.data;

    // --- Container sanity: RIFF + WEBP + VP8  marker at the expected offsets.
    assert!(
        webp_bytes.len() >= 20,
        "packet too small: {}",
        webp_bytes.len()
    );
    assert_eq!(&webp_bytes[0..4], b"RIFF", "missing RIFF magic");
    assert_eq!(&webp_bytes[8..12], b"WEBP", "missing WEBP form type");
    assert_eq!(
        &webp_bytes[12..16],
        b"VP8 ",
        "expected VP8 chunk FourCC (with trailing space), got {:?}",
        &webp_bytes[12..16]
    );

    // --- Decode through the full WebP container pipeline.
    let image = decode_webp(&webp_bytes).expect("decode_webp");
    assert_eq!(image.width, W);
    assert_eq!(image.height, H);
    assert_eq!(image.frames.len(), 1);
    let rgba = &image.frames[0].rgba;
    assert_eq!(rgba.len(), (W * H * 4) as usize);

    // Convert decoded RGBA back to Y samples and compute PSNR against the
    // source luma plane. The VP8 encoder at the default qindex (~50) on
    // this smooth test pattern clears ~35 dB comfortably — we assert >30.
    let mut dec_y = vec![0u8; (W * H) as usize];
    for j in 0..H as usize {
        for i in 0..W as usize {
            let p = &rgba[(j * W as usize + i) * 4..(j * W as usize + i) * 4 + 3];
            dec_y[j * W as usize + i] = rgb_to_y(p[0], p[1], p[2]);
        }
    }
    let psnr = psnr_y(&y, &dec_y);
    eprintln!("VP8 lossy WebP Y-plane PSNR: {psnr:.2} dB");
    assert!(
        psnr > 30.0,
        "VP8 lossy WebP PSNR too low: {psnr:.2} dB (expected >30)"
    );
}

#[test]
fn vp8_encoder_accepts_rgba_and_emits_alph_sidecar() {
    // With the ALPH sidecar path in place, the VP8 lossy encoder now
    // accepts RGBA input directly. The resulting file uses the extended
    // layout (VP8X), contains both a VP8 (colour) and an ALPH (alpha)
    // chunk, and round-trips through `decode_webp` with the alpha plane
    // preserved.
    let mut p = make_encoder_params();
    p.pixel_format = Some(PixelFormat::Rgba);
    let mut enc =
        encoder_vp8::make_encoder(&p).expect("rgba params should build a VP8-lossy encoder");
    let rgba = build_rgba_with_alpha_gradient();
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (W as usize) * 4,
            data: rgba.clone(),
        }],
    };
    enc.send_frame(&Frame::Video(frame)).expect("send rgba");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive packet");
    let out = pkt.data;

    // Container shape: RIFF + WEBP + VP8X as the first chunk.
    assert_eq!(&out[0..4], b"RIFF");
    assert_eq!(&out[8..12], b"WEBP");
    assert_eq!(&out[12..16], b"VP8X", "expected VP8X marker at offset 12");
    // ALPHA flag (0x10) must be set in the VP8X flags byte.
    assert!(
        out[20] & 0x10 != 0,
        "VP8X ALPHA flag not set (flags = {:#x})",
        out[20]
    );

    // Scan chunks: the file must contain both a `VP8 ` and an `ALPH`
    // chunk alongside the VP8X header.
    let (has_vp8, has_alph) = find_chunks(&out);
    assert!(has_vp8, "output is missing the VP8 chunk");
    assert!(has_alph, "output is missing the ALPH sidecar chunk");

    // Round-trip through the decoder and confirm the alpha plane is
    // preserved (values are lossless because the ALPH is VP8L-compressed).
    let img = decode_webp(&out).expect("decode");
    assert_eq!(img.frames.len(), 1);
    let decoded = &img.frames[0].rgba;
    assert_eq!(decoded.len(), (W * H * 4) as usize);
    for (i, (a_in, a_out)) in rgba
        .iter()
        .skip(3)
        .step_by(4)
        .zip(decoded.iter().skip(3).step_by(4))
        .enumerate()
    {
        assert_eq!(
            a_in, a_out,
            "alpha byte {i} did not round-trip: in={a_in}, out={a_out}"
        );
    }
}

/// 128×128 RGBA buffer with a known alpha gradient. The colour plane is
/// a smooth diagonal ramp to keep VP8's lossy PSNR reasonable.
fn build_rgba_with_alpha_gradient() -> Vec<u8> {
    let mut rgba = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            rgba[i] = (x * 2) as u8;
            rgba[i + 1] = (y * 2) as u8;
            rgba[i + 2] = (x + y) as u8;
            // Alpha: diagonal ramp 0..=255.
            rgba[i + 3] = ((x + y) * 255 / (W + H - 2)) as u8;
        }
    }
    rgba
}

/// Walk the chunk list of a WebP file and report whether it has a `VP8 `
/// and an `ALPH` chunk. Assumes a well-formed RIFF/WEBP header.
fn find_chunks(buf: &[u8]) -> (bool, bool) {
    let mut has_vp8 = false;
    let mut has_alph = false;
    let mut pos = 12; // skip "RIFF" size "WEBP"
    while pos + 8 <= buf.len() {
        let fourcc = &buf[pos..pos + 4];
        let size =
            u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;
        match fourcc {
            b"VP8 " => has_vp8 = true,
            b"ALPH" => has_alph = true,
            _ => {}
        }
        let padded = size + (size & 1);
        pos += 8 + padded;
    }
    (has_vp8, has_alph)
}

// `vp8_encoder_rejects_rgba_frame_at_send_time` was removed because pixel
// format is no longer carried per-frame — input format is fixed at encoder
// construction via `CodecParameters` and the pipeline upstream is
// responsible for not feeding a mismatching frame.
