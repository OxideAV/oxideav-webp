#![no_main]

use libfuzzer_sys::fuzz_target;
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
    // Pack RGBA bytes into u32 ARGB (the encoder's native input).
    let argb: Vec<u32> = rgba
        .chunks_exact(4)
        .map(|p| ((p[3] as u32) << 24) | ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32))
        .collect();
    // Disable strip_transparent_color: the harness asserts decoded RGBA
    // matches the original input byte-for-byte, but the default-on strip
    // zeros RGB on alpha=0 pixels, which would falsely fail the assert.
    let opts = oxideav_webp::EncoderOptions {
        strip_transparent_color: false,
        ..Default::default()
    };
    oxideav_webp::encode_vp8l_argb_with(width, height, &argb, true, opts)
        .expect("oxideav-webp VP8L encoding failed")
}
