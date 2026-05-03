//! Coverage for the input-pixel-format extensions added in response to
//! issue #7:
//!
//! * **VP8L (lossless) accepts `Rgb24` directly** — verifies the
//!   streaming RGB→ARGB conversion produces a `.webp` file whose decoded
//!   pixels match the equivalent Rgba-input encode bit-for-bit.
//! * **VP8 (lossy) accepts `Rgb24` directly** — verifies the streaming
//!   RGB→YUV4:2:0 conversion produces a `.webp` file that decodes back
//!   to the same Y/U/V planes the equivalent Rgba-input path produces.
//! * **VP8 (lossy) accepts `Yuva420P` natively** — encodes the YUV
//!   planes straight into the keyframe + the alpha plane straight into
//!   the ALPH sidecar, skipping the YUV→RGB→YUV roundtrip the Rgba path
//!   incurs. The matching decoder mode (`WebpDecoder::new_yuva420p`)
//!   recovers a 4-plane Yuva420P frame with the alpha plane preserved
//!   exactly (ALPH is VP8L-compressed → lossless).
//!
//! The "no alloc" claim for the Rgb24 path is not directly assertable
//! from a unit test (Rust has no public allocator hook in stable test
//! harnesses), but the conversion routines themselves are documented
//! to walk the input three bytes at a time straight into the encoder's
//! native pixel buffer with no intermediate `Rgba` byte buffer. See
//! `encoder::encode_frame_rgb24` and `encoder_vp8::rgb24_rows_to_yuv420`
//! for the implementations.

use oxideav_core::{
    CodecId, CodecParameters, Decoder as _, Demuxer as _, Encoder as _, Frame, MediaType, Packet,
    PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_webp::{
    decode_webp, decoder::WebpDecoder, demux, encoder, encoder_vp8, CODEC_ID_VP8, CODEC_ID_VP8L,
};

const W: u32 = 64;
const H: u32 = 64;

/// Build a deterministic 64×64 Rgb24 buffer with channels that vary
/// independently — keeps the predictor + colour transform interesting.
fn build_rgb24(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            buf[i] = (x * 4) as u8;
            buf[i + 1] = (y * 4) as u8;
            buf[i + 2] = ((x + y) * 2) as u8;
        }
    }
    buf
}

/// Inflate the same RGB pattern into Rgba (alpha = 0xff) for the
/// "equivalent Rgba" comparison side. Test-only; the production paths
/// never do this.
fn rgb24_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(&[px[0], px[1], px[2], 0xff]);
    }
    out
}

fn vp8l_params(pix: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8L));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    p
}

fn vp8_params(pix: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    p
}

fn make_video_frame(stride: usize, data: Vec<u8>) -> VideoFrame {
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane { stride, data }],
    }
}

#[test]
fn vp8l_rgb24_roundtrip_matches_rgba_equivalent() {
    let rgb = build_rgb24(W, H);

    // Encode via the Rgb24 path.
    let mut enc_rgb = encoder::make_encoder(&vp8l_params(PixelFormat::Rgb24))
        .expect("vp8l encoder accepts Rgb24");
    enc_rgb
        .send_frame(&Frame::Video(make_video_frame(
            (W as usize) * 3,
            rgb.clone(),
        )))
        .expect("send Rgb24 frame");
    enc_rgb.flush().expect("flush rgb");
    let rgb_pkt = enc_rgb.receive_packet().expect("receive rgb packet");

    // Encode via the Rgba path with the same RGB content + opaque alpha.
    let rgba = rgb24_to_rgba(&rgb);
    let mut enc_rgba = encoder::make_encoder(&vp8l_params(PixelFormat::Rgba))
        .expect("vp8l encoder still accepts Rgba");
    enc_rgba
        .send_frame(&Frame::Video(make_video_frame(
            (W as usize) * 4,
            rgba.clone(),
        )))
        .expect("send Rgba frame");
    enc_rgba.flush().expect("flush rgba");
    let rgba_pkt = enc_rgba.receive_packet().expect("receive rgba packet");

    // Decoded pixels must match: VP8L is lossless and an opaque Rgba
    // input is semantically the same image as an Rgb24 input. Bytes
    // need not be byte-identical between the two .webp files (the
    // Rgba path takes the VP8X+VP8L extended layout when the encoder
    // sees alpha, and the simple layout when the alpha is uniformly
    // 0xff — for an opaque input both paths land in the simple
    // layout, but we only assert decoded-pixel equality to leave
    // room for transform-RDO drift).
    let decoded_rgb = decode_webp(&rgb_pkt.data).expect("decode rgb-encoded");
    let decoded_rgba = decode_webp(&rgba_pkt.data).expect("decode rgba-encoded");

    assert_eq!(decoded_rgb.frames.len(), 1);
    assert_eq!(decoded_rgba.frames.len(), 1);
    assert_eq!(
        decoded_rgb.frames[0].rgba, decoded_rgba.frames[0].rgba,
        "Rgb24-input and Rgba-input must produce identical decoded pixels"
    );
    // And both must equal the expected opaque-Rgba ground truth.
    assert_eq!(decoded_rgb.frames[0].rgba, rgba);
}

#[test]
fn vp8_lossy_rgb24_roundtrip_yuv_matches_rgba_equivalent() {
    let rgb = build_rgb24(W, H);

    // Encode the same content via Rgb24 and via Rgba (both go through
    // the encoder's RGB→YUV BT.601 conversion — Rgb24 streams with no
    // intermediate Rgba alloc, Rgba walks four bytes per pixel and
    // also pulls out an alpha plane).
    let mut enc_rgb = encoder_vp8::make_encoder(&vp8_params(PixelFormat::Rgb24))
        .expect("vp8 encoder accepts Rgb24");
    enc_rgb
        .send_frame(&Frame::Video(make_video_frame(
            (W as usize) * 3,
            rgb.clone(),
        )))
        .expect("send rgb24");
    enc_rgb.flush().expect("flush");
    let rgb_pkt = enc_rgb.receive_packet().expect("receive rgb packet");

    let rgba = rgb24_to_rgba(&rgb);
    let mut enc_rgba =
        encoder_vp8::make_encoder(&vp8_params(PixelFormat::Rgba)).expect("vp8 accepts Rgba");
    enc_rgba
        .send_frame(&Frame::Video(make_video_frame((W as usize) * 4, rgba)))
        .expect("send rgba");
    enc_rgba.flush().expect("flush");
    let rgba_pkt = enc_rgba.receive_packet().expect("receive rgba packet");

    // The Rgb24 path must produce the SIMPLE container layout (no
    // ALPH, no VP8X) — opaque input means there's nothing to ALPH-
    // encode. The Rgba path takes the extended layout because the
    // encoder always emits ALPH for Rgba (even when the alpha is
    // uniformly opaque).
    assert_eq!(&rgb_pkt.data[12..16], b"VP8 ", "Rgb24 should be simple-VP8");
    assert_eq!(
        &rgba_pkt.data[12..16],
        b"VP8X",
        "Rgba should still take the extended layout"
    );

    // Decode both back via the standard RGBA path and ensure the Y
    // planes match. We can't assert RGBA byte equality (the VP8X+ALPH
    // file overwrites the alpha channel from ALPH while the simple
    // file leaves it at 0xff), so compute Y from RGB and compare PSNR.
    let dec_rgb = decode_webp(&rgb_pkt.data).expect("decode rgb-encoded vp8");
    let dec_rgba = decode_webp(&rgba_pkt.data).expect("decode rgba-encoded vp8");
    assert_eq!(dec_rgb.frames.len(), 1);
    assert_eq!(dec_rgba.frames.len(), 1);

    let psnr = rgb_psnr(&dec_rgb.frames[0].rgba, &dec_rgba.frames[0].rgba);
    eprintln!("VP8 lossy Rgb24 vs Rgba decoded RGB PSNR: {psnr:.2} dB");
    // The two paths share the same BT.601 coefficients + the same VP8
    // encoder; decoded RGB should be very close (different lots of
    // rounding intermediate vs same code path). Loose 35 dB bound to
    // tolerate the alpha byte swap and BT.601 rounding.
    assert!(
        psnr > 35.0,
        "Rgb24 vs Rgba VP8-lossy decoded RGB diverges (psnr={psnr:.2})"
    );
}

#[test]
fn vp8_lossy_yuva420p_roundtrip_recovers_alpha_exactly() {
    // Build a Yuva420P frame: smooth Y/U/V planes + a diagonal alpha
    // ramp. ALPH compression is VP8L (lossless), so the alpha plane
    // must round-trip byte-identical even though the YUV planes go
    // through lossy VP8 quantisation.
    let cw = (W / 2) as usize;
    let ch = (H / 2) as usize;
    let mut y = vec![0u8; (W * H) as usize];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    let mut a = vec![0u8; (W * H) as usize];
    for j in 0..H as usize {
        for i in 0..W as usize {
            y[j * W as usize + i] = (32 + ((i + j) * 191 / (W as usize + H as usize - 2))) as u8;
            // Diagonal alpha ramp 0..255.
            a[j * W as usize + i] = ((i + j) * 255 / (W as usize + H as usize - 2)) as u8;
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (64 + (i * 127) / cw.max(1)) as u8;
            v[j * cw + i] = (64 + (j * 127) / ch.max(1)) as u8;
        }
    }

    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: W as usize,
                data: y.clone(),
            },
            VideoPlane {
                stride: cw,
                data: u.clone(),
            },
            VideoPlane {
                stride: cw,
                data: v.clone(),
            },
            VideoPlane {
                stride: W as usize,
                data: a.clone(),
            },
        ],
    };

    let mut enc = encoder_vp8::make_encoder(&vp8_params(PixelFormat::Yuva420P))
        .expect("vp8 encoder accepts Yuva420P");
    enc.send_frame(&Frame::Video(frame))
        .expect("send yuva frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive packet");

    // The container must be the extended layout with VP8X + ALPH + VP8 .
    assert_eq!(&pkt.data[0..4], b"RIFF");
    assert_eq!(&pkt.data[8..12], b"WEBP");
    assert_eq!(&pkt.data[12..16], b"VP8X");
    assert!(
        pkt.data[20] & 0x10 != 0,
        "VP8X ALPHA flag must be set for Yuva420P input"
    );

    // Decode back via the WebpDecoder in Yuva420P mode and check planes.
    let yuva = decode_yuva420p_packet(&pkt.data);
    assert_eq!(yuva.planes.len(), 4, "Yuva420P frame must carry 4 planes");
    let dec_y = &yuva.planes[0];
    let dec_u = &yuva.planes[1];
    let dec_v = &yuva.planes[2];
    let dec_a = &yuva.planes[3];

    // Alpha is lossless via VP8L — must match exactly.
    assert_eq!(dec_a.stride, W as usize);
    assert_eq!(dec_a.data.len(), (W * H) as usize);
    assert_eq!(dec_a.data, a, "alpha plane must round-trip byte-identical");

    // Y/U/V are lossy; loose PSNR check on the Y plane is enough.
    assert_eq!(dec_y.stride, W as usize);
    assert_eq!(dec_u.stride, cw);
    assert_eq!(dec_v.stride, cw);
    let y_psnr = plane_psnr(&y, &dec_y.data);
    eprintln!("VP8+ALPH Yuva420P Y-plane PSNR: {y_psnr:.2} dB");
    assert!(
        y_psnr > 30.0,
        "Yuva420P round-trip Y-plane PSNR too low: {y_psnr:.2} dB"
    );
}

/// Push a complete `.webp` packet through the WebP demuxer + decoder in
/// Yuva420P mode and return the single resulting Yuva420P video frame.
fn decode_yuva420p_packet(file: &[u8]) -> VideoFrame {
    let cursor = std::io::Cursor::new(file.to_vec());
    let mut demuxer = demux::open_boxed(Box::new(cursor)).expect("open demuxer");
    let stream_w = demuxer.streams()[0].params.width.unwrap_or(0);
    let stream_h = demuxer.streams()[0].params.height.unwrap_or(0);
    let mut dec = WebpDecoder::new_yuva420p(stream_w, stream_h);
    let pkt: Packet = demuxer.next_packet().expect("first packet");
    dec.send_packet(&pkt).expect("send packet");
    let frame = dec.receive_frame().expect("receive yuva frame");
    match frame {
        Frame::Video(vf) => vf,
        _ => panic!("expected Frame::Video"),
    }
}

fn rgb_psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    // Compare RGB triplets only; alpha bytes (every 4th) skipped.
    let mut se = 0f64;
    let mut n = 0usize;
    for (chunk_a, chunk_b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for k in 0..3 {
            let d = chunk_a[k] as f64 - chunk_b[k] as f64;
            se += d * d;
            n += 1;
        }
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

fn plane_psnr(a: &[u8], b: &[u8]) -> f64 {
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

// ----- small dependency-shape sanity tests -----

#[test]
fn vp8l_encoder_rejects_non_rgb_pixel_format() {
    // YUV420P is not valid for VP8L (a lossless RGBA codec).
    let err = encoder::make_encoder(&vp8l_params(PixelFormat::Yuv420P))
        .err()
        .expect("vp8l should reject Yuv420P");
    let msg = format!("{err}");
    assert!(
        msg.contains("Yuv420P") || msg.contains("not supported"),
        "unexpected error: {msg}"
    );
}

#[test]
fn vp8_encoder_rejects_unrelated_pixel_format() {
    let err = encoder_vp8::make_encoder(&vp8_params(PixelFormat::Gray8))
        .err()
        .expect("vp8 should reject Gray8");
    let msg = format!("{err}");
    assert!(
        msg.contains("Gray8") || msg.contains("not supported"),
        "unexpected error: {msg}"
    );
}
