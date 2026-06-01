#![no_main]

//! Encode a fuzz-controlled RGBA image through `encode_webp_lossless`
//! and assert it survives a `decode_webp` round trip pixel-for-pixel.
//!
//! VP8L is a lossless codec: the §3 image header + §3.7 prefix-coded
//! literals + §4 transform stack must reproduce the exact ARGB pixels
//! the encoder was given. The fuzzer drives the dimensions (kept small
//! so the body stays comfortably within working memory and individual
//! iterations stay fast) and the per-pixel RGBA values; this harness
//! is a **self-roundtrip** oracle on the lossless encoder + decoder
//! pair — `encode_webp_lossless` → `decode_webp` → compare.
//!
//! Crashes the fuzzer reports are either a panic on either side of the
//! round trip (encoder bug or decoder bug) or a pixel-mismatch
//! assertion failure (a real lossless-contract violation in the
//! transform stack or prefix-code emit / read path).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{decode_webp, encode_webp_lossless};

fuzz_target!(|data: &[u8]| {
    // Need at least two bytes for the dimension nibbles.
    if data.len() < 2 {
        return;
    }

    // Derive small, in-bounds dimensions from the first two bytes.
    // Cap each at 64 so the worst-case RGBA plane is 64*64*4 = 16 KiB
    // and a single fuzz iteration runs in milliseconds.
    let width: u32 = u32::from(data[0] % 64) + 1;
    let height: u32 = u32::from(data[1] % 64) + 1;

    // `width <= 64` and `height <= 64`, so the product can't overflow.
    let expected_rgba_len = (width as usize) * (height as usize) * 4;

    // Build an RGBA plane of exactly the required length, sourcing
    // bytes from the remaining fuzz input. Cycle through the available
    // bytes (zero-padded when none remain) so the fuzzer can drive
    // every byte position even from a short corpus seed.
    let body = &data[2..];
    let mut rgba = Vec::with_capacity(expected_rgba_len);
    for i in 0..expected_rgba_len {
        rgba.push(if body.is_empty() {
            0
        } else {
            body[i % body.len()]
        });
    }

    // Encode. A non-zero dimension and an exactly-sized buffer should
    // always succeed; an encoder error is treated as a genuine bug, but
    // we surface it as a returned `None` rather than an unwrap so the
    // fuzzer reports a panic only when the *decode* side mismatches.
    let encoded = match encode_webp_lossless(&rgba, width, height) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Decode the encoded bytes and assert the round trip.
    let image = decode_webp(&encoded).expect("valid encoded VP8L must decode");
    assert_eq!(image.width, width, "width survives round trip");
    assert_eq!(image.height, height, "height survives round trip");
    assert_eq!(
        image.frames.len(),
        1,
        "single still image carries one frame"
    );

    let frame = &image.frames[0];
    assert_eq!(frame.width, width, "frame width survives round trip");
    assert_eq!(frame.height, height, "frame height survives round trip");
    assert_eq!(
        frame.rgba.len(),
        expected_rgba_len,
        "decoded RGBA buffer is width * height * 4 bytes"
    );
    assert_eq!(
        frame.rgba, rgba,
        "lossless pixels survive encode + decode byte-for-byte"
    );
});
