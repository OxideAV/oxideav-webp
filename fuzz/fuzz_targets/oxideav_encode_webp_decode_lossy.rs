#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_webp::{encoder_vp8, CODEC_ID_VP8};
use oxideav_webp_fuzz::libwebp;

const MAX_WIDTH: usize = 64;
const MAX_PIXELS: usize = 2048;

fuzz_target!(|data: &[u8]| {
    // Skip silently if libwebp isn't installed on this host.
    if !libwebp::available() {
        return;
    }

    let Some((width, height, qindex, rgba)) = image_from_fuzz_input(data) else {
        return;
    };

    let encoded = encode_webp_lossily(width, height, qindex, rgba);
    let decoded = libwebp::decode_to_rgba(&encoded).expect("libwebp decoding failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.rgba.len(), (width as usize) * (height as usize) * 4);
});

fn image_from_fuzz_input(data: &[u8]) -> Option<(u32, u32, u8, &[u8])> {
    let (&shape, rest) = data.split_first()?;
    let (&qindex, rgba) = rest.split_first()?;

    let pixel_count = (rgba.len() / 4).min(MAX_PIXELS);
    if pixel_count == 0 {
        return None;
    }

    let width = ((shape as usize) % MAX_WIDTH) + 1;
    let width = width.min(pixel_count);
    let height = pixel_count / width;
    let used_len = width * height * 4;
    let rgba = &rgba[..used_len];

    Some((width as u32, height as u32, qindex & 0x7f, rgba))
}

fn encode_webp_lossily(width: u32, height: u32, qindex: u8, rgba: &[u8]) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Rgba);

    let mut encoder =
        encoder_vp8::make_encoder_with_qindex(&params, qindex).expect("make VP8 encoder");
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (width as usize) * 4,
            data: rgba.to_vec(),
        }],
    };

    encoder
        .send_frame(&Frame::Video(frame))
        .expect("oxideav-webp VP8 encoding failed");
    encoder.flush().expect("oxideav-webp encoder flush failed");
    encoder
        .receive_packet()
        .expect("oxideav-webp encoder did not emit a packet")
        .data
}
