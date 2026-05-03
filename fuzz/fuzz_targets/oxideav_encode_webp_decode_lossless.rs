#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_webp::CODEC_ID_VP8L;
use oxideav_webp_fuzz::libwebp;

const MAX_WIDTH: usize = 64;
const MAX_PIXELS: usize = 2048;

fuzz_target!(|data: &[u8]| {
    // Skip silently if libwebp isn't installed on this host.
    if !libwebp::available() {
        return;
    }

    let Some((width, height, rgba)) = image_from_fuzz_input(data) else {
        return;
    };

    let encoded = encode_webp_losslessly(width, height, rgba);
    let decoded = libwebp::decode_to_rgba(&encoded).expect("libwebp decoding failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.rgba.as_slice(), rgba);
});

fn image_from_fuzz_input(data: &[u8]) -> Option<(u32, u32, &[u8])> {
    let (&shape, rgba) = data.split_first()?;

    let pixel_count = (rgba.len() / 4).min(MAX_PIXELS);
    if pixel_count == 0 {
        return None;
    }

    let width = ((shape as usize) % MAX_WIDTH) + 1;
    let width = width.min(pixel_count);
    let height = pixel_count / width;
    let used_len = width * height * 4;
    let rgba = &rgba[..used_len];

    Some((width as u32, height as u32, rgba))
}

fn encode_webp_losslessly(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_VP8L));
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(PixelFormat::Rgba);

    let mut encoder = oxideav_webp::encoder::make_encoder(&params).expect("make VP8L encoder");
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (width as usize) * 4,
            data: rgba.to_vec(),
        }],
    };

    encoder
        .send_frame(&Frame::Video(frame))
        .expect("oxideav-webp VP8L encoding failed");
    encoder.flush().expect("oxideav-webp encoder flush failed");
    encoder
        .receive_packet()
        .expect("oxideav-webp encoder did not emit a packet")
        .data
}
