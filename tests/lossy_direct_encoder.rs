//! `encoder_vp8::Vp8LossyEncoder`: the lossy (`VP8 `) still encoder with a
//! direct constructor. It works without the `registry` feature, reads
//! 4:2:0 planes in place at the caller's strides, and writes the same
//! bytes as the `webp_vp8` framework encoder for the same picture.

use oxideav_webp::encoder_vp8::Vp8LossyEncoder;
use oxideav_webp::{decode, info, PixelFormat};

/// A `w x h` 4:2:0 picture whose rows sit `pad` bytes apart beyond their
/// width: `(y, y_stride, u, v, uv_stride)`.
fn planes(w: usize, h: usize, pad: usize) -> (Vec<u8>, usize, Vec<u8>, Vec<u8>, usize) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (ys, cs) = (w + pad, cw + pad);
    let mut y = vec![0xee; ys * h];
    let mut u = vec![0xee; cs * ch];
    let mut v = vec![0xee; cs * ch];
    for r in 0..h {
        for c in 0..w {
            y[r * ys + c] = (c * 5 + r * 3) as u8;
        }
    }
    for r in 0..ch {
        for c in 0..cw {
            u[r * cs + c] = (100 + c * 2) as u8;
            v[r * cs + c] = (150 - r * 2) as u8;
        }
    }
    (y, ys, u, v, cs)
}

#[test]
fn encodes_a_decodable_simple_lossy_file() {
    let (w, h) = (48u32, 40u32);
    let (y, ys, u, v, cs) = planes(48, 40, 0);
    let file = Vp8LossyEncoder::with_quality(80.0)
        .encode_yuv420(w, h, &y, ys, &u, &v, cs)
        .expect("encode");
    let i = info(&file).expect("info");
    assert_eq!((i.width, i.height), (w, h));
    let image = decode(&file).expect("decode");
    assert_eq!(image.format, PixelFormat::Yuv420P);
}

#[test]
fn padded_rows_encode_like_tight_rows() {
    let enc = Vp8LossyEncoder::with_qindex(30);
    let (y, ys, u, v, cs) = planes(37, 21, 0);
    let tight = enc
        .encode_yuv420(37, 21, &y, ys, &u, &v, cs)
        .expect("tight");
    let (y, ys, u, v, cs) = planes(37, 21, 11);
    let padded = enc
        .encode_yuv420(37, 21, &y, ys, &u, &v, cs)
        .expect("padded");
    assert_eq!(padded, tight);
}

#[test]
fn quality_and_qindex_constructors_agree() {
    assert_eq!(Vp8LossyEncoder::with_quality(80.0).qindex(), 25);
    assert_eq!(
        Vp8LossyEncoder::with_quality(80.0),
        Vp8LossyEncoder::with_qindex(25)
    );
    assert_eq!(Vp8LossyEncoder::with_qindex(200).qindex(), 127);
}

#[test]
fn bad_geometry_is_an_error() {
    let enc = Vp8LossyEncoder::with_quality(80.0);
    let (y, ys, u, v, cs) = planes(16, 16, 0);
    // Zero and oversized dimensions.
    assert!(enc.encode_yuv420(0, 16, &y, ys, &u, &v, cs).is_err());
    assert!(enc.encode_yuv420(16_384, 16, &y, ys, &u, &v, cs).is_err());
    // A stride narrower than the row.
    assert!(enc.encode_yuv420(16, 16, &y, 15, &u, &v, cs).is_err());
    assert!(enc.encode_yuv420(16, 16, &y, ys, &u, &v, 7).is_err());
    // Planes shorter than their rows need.
    assert!(enc
        .encode_yuv420(16, 16, &y[..y.len() - 1], ys, &u, &v, cs)
        .is_err());
    assert!(enc
        .encode_yuv420(16, 16, &y, ys, &u, &v[..v.len() - 1], cs)
        .is_err());
    // A stride whose row offsets overflow.
    assert!(enc
        .encode_yuv420(16, 16, &y, usize::MAX, &u, &v, cs)
        .is_err());
}

/// For the same picture and quality, the direct encoder writes exactly the
/// `webp_vp8` framework encoder's packet.
#[cfg(feature = "registry")]
#[test]
fn matches_the_framework_encoder_byte_for_byte() {
    use oxideav_core::{CodecId, CodecParameters, Frame, VideoFrame, VideoPlane};

    for (w, h, pad, quality) in [
        (48usize, 40usize, 0usize, 80.0f32),
        (37, 21, 5, 55.0),
        (16, 16, 0, 100.0),
    ] {
        let (y, ys, u, v, cs) = planes(w, h, pad);
        let direct = Vp8LossyEncoder::with_quality(quality)
            .encode_yuv420(w as u32, h as u32, &y, ys, &u, &v, cs)
            .expect("direct encode");

        let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8));
        params.width = Some(w as u32);
        params.height = Some(h as u32);
        params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
        let mut enc = oxideav_webp::encoder_vp8::make_encoder_with_quality(&params, quality)
            .expect("framework encoder");
        let frame = Frame::Video(VideoFrame {
            pts: Some(0),
            planes: vec![
                VideoPlane {
                    stride: ys,
                    data: y,
                },
                VideoPlane {
                    stride: cs,
                    data: u,
                },
                VideoPlane {
                    stride: cs,
                    data: v,
                },
            ],
        });
        enc.send_frame(&frame).expect("send");
        let packet = enc.receive_packet().expect("packet");
        assert_eq!(direct, packet.data, "{w}x{h}, pad {pad}, quality {quality}");
    }
}
