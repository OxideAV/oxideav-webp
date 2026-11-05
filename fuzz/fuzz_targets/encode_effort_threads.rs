#![no_main]

//! Lossless effort levels and the encoder thread budget: a
//! fuzz-controlled RGBA image goes through `encode_rgba8` at a
//! fuzz-chosen `EncodeOptions::method`, once on one thread and once on a
//! fuzz-chosen worker budget. The two encodes must be byte-identical
//! (every parallel stage folds its results in a fixed order) and the
//! file must decode back to the input exactly.
//!
//! This is the only harness that reaches the screened levels `0..=5`
//! (the screening waves, the one-walk §5.2.3 cache pricing over the
//! parse counts, the §6.2.2 layout stage) at all: the other encode
//! harnesses drive the default level or the forced-parameter entry
//! points. Method 6 (the exhaustive search) is allowed up to 16 x 16
//! pixels, where it stays fast enough for a fuzz iteration.
//!
//! Byte 0: width - 1 (low 5 bits); byte 1: height - 1 (low 5 bits);
//! byte 2: method (low 3 bits, 7 -> 0) and threads (bits 3..5, + 2);
//! the rest: pixel bytes, repeated to fill the image.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{EncodeOptions, decode, encode_rgba8};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let width = u32::from(data[0] & 0x1f) + 1;
    let height = u32::from(data[1] & 0x1f) + 1;
    let mut method = data[2] & 0x07;
    if method == 7 || (method == 6 && width * height > 256) {
        method = 0;
    }
    let threads = usize::from((data[2] >> 3) & 0x03) + 2;
    let body = &data[3..];
    let len = (width * height * 4) as usize;
    let pixels: Vec<u8> = if body.is_empty() {
        vec![0; len]
    } else {
        (0..len).map(|i| body[i % body.len()]).collect()
    };

    let opts = EncodeOptions::default().with_method(method);
    let serial = encode_rgba8(width, height, &pixels, &opts.clone().with_threads(1))
        .expect("in-bounds lossless encode succeeds");
    let parallel = encode_rgba8(width, height, &pixels, &opts.with_threads(threads))
        .expect("in-bounds lossless encode succeeds");
    assert_eq!(
        serial, parallel,
        "method {method}: the worker budget changed the output"
    );
    let back = decode(&serial).expect("lossless output decodes").to_rgba8();
    assert_eq!(back, pixels, "method {method}: lossless round trip");
});
