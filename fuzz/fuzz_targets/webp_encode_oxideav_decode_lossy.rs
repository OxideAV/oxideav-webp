#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_webp::decode_webp;
use oxideav_webp_fuzz::libwebp;

const MAX_WIDTH: usize = 64;
const MAX_PIXELS: usize = 2048;

fuzz_target!(|data: &[u8]| {
    // Skip silently if libwebp isn't installed on this host.
    if !libwebp::available() {
        return;
    }

    let Some((width, height, quality, rgba)) = image_from_fuzz_input(data) else {
        return;
    };

    let encoded = libwebp::encode_lossy(rgba, width, height, quality)
        .expect("libwebp lossy encoding failed");

    let decoded = decode_webp(&encoded).expect("oxideav-webp decoding failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.frames.len(), 1);
    assert_eq!(
        decoded.frames[0].rgba.len(),
        (width as usize) * (height as usize) * 4
    );
});

fn image_from_fuzz_input(data: &[u8]) -> Option<(u32, u32, f32, &[u8])> {
    let (&shape, rest) = data.split_first()?;
    let (&quality, rgba) = rest.split_first()?;

    let pixel_count = (rgba.len() / 4).min(MAX_PIXELS);
    if pixel_count == 0 {
        return None;
    }

    let width = ((shape as usize) % MAX_WIDTH) + 1;
    let width = width.min(pixel_count);
    let height = pixel_count / width;
    let used_len = width * height * 4;
    let rgba = &rgba[..used_len];
    let quality = (quality as f32) * (100.0 / 255.0);

    Some((width as u32, height as u32, quality, rgba))
}
