#![no_main]

//! Lossy effort levels and the encoder thread budget: a fuzz-controlled
//! RGBA image goes through `encode_rgba8` at a fuzz-chosen quality and
//! `EncodeOptions::method`, once on one thread and once on a fuzz-chosen
//! worker budget. The two files must be byte-identical (the VP8
//! macroblock rows run as a wavefront, the `ALPH` plane next to them),
//! and the file must decode to the input's size with its alpha plane
//! exact.
//!
//! The reduced mode searches (methods `0..=2`) are reached by no other
//! harness; `contract_encode` drives the default level on one thread.
//!
//! Byte 0: width - 1 (low 6 bits); byte 1: height - 1 (low 6 bits);
//! byte 2: method (low 3 bits, 7 -> 0), threads (bits 3..5, + 2) and
//! opaque (bit 7: force alpha to 255); byte 3: quality (0..=100, larger
//! values wrap); the rest: pixel bytes, repeated to fill the image.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{EncodeOptions, decode, encode_rgba8};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let width = u32::from(data[0] & 0x3f) + 1;
    let height = u32::from(data[1] & 0x3f) + 1;
    let method = match data[2] & 0x07 {
        7 => 0,
        m => m,
    };
    let threads = usize::from((data[2] >> 3) & 0x03) + 2;
    let opaque = data[2] & 0x80 != 0;
    let quality = f32::from(data[3] % 101);
    let body = &data[4..];
    let len = (width * height * 4) as usize;
    let mut pixels: Vec<u8> = if body.is_empty() {
        vec![0; len]
    } else {
        (0..len).map(|i| body[i % body.len()]).collect()
    };
    if opaque {
        for px in pixels.chunks_exact_mut(4) {
            px[3] = 0xff;
        }
    }

    let opts = EncodeOptions::default()
        .with_quality(quality)
        .with_method(method);
    let serial = encode_rgba8(width, height, &pixels, &opts.clone().with_threads(1))
        .expect("in-bounds lossy encode succeeds");
    let parallel = encode_rgba8(width, height, &pixels, &opts.with_threads(threads))
        .expect("in-bounds lossy encode succeeds");
    assert_eq!(
        serial, parallel,
        "quality {quality}, method {method}: the worker budget changed the output"
    );
    let image = decode(&serial).expect("lossy output decodes");
    assert_eq!((image.width, image.height), (width, height));
    let back = image.to_rgba8();
    assert!(
        back.chunks_exact(4)
            .zip(pixels.chunks_exact(4))
            .all(|(got, want)| got[3] == want[3]),
        "quality {quality}, method {method}: alpha plane is exact"
    );
});
