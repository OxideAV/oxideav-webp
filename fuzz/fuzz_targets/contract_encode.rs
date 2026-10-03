#![no_main]

//! Contract encode round trips: a fuzz-controlled RGB / RGBA image goes
//! through `encode_rgb8` / `encode_rgba8` both **lossless** (the
//! `EncodeOptions::default()` VP8L path — pixel-exact round trip
//! asserted) and **lossy** (`with_quality`, the VP8 key-frame path plus
//! the §2.7.1.2 `ALPH` writer born in round 465 — the alpha plane is
//! asserted byte-exact, the colour planes only have to decode), then back
//! through the contract `decode` / `info` / `decode_all`.
//!
//! The first byte selects the dimensions (≤ 16 per side: the VP8 encoder
//! is the slow half of the iteration), the second the quality and the
//! RGB-vs-RGBA / metadata switches, the rest the pixel bytes.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{
    decode, decode_all, encode_rgb8, encode_rgba8, info, probe, EncodeOptions, PixelFormat,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let width = u32::from(data[0] & 0x0f) + 1;
    let height = u32::from(data[0] >> 4) + 1;
    let quality = f32::from(data[1] & 0x7f) / 1.27;
    let rgba_in = data[1] & 0x80 != 0;
    let bpp = if rgba_in { 4 } else { 3 };
    let len = (width * height) as usize * bpp;
    let body = &data[2..];
    let pixels: Vec<u8> = (0..len)
        .map(|i| if body.is_empty() { 0 } else { body[i % body.len()] })
        .collect();

    // Lossless: exact.
    let lossless = EncodeOptions::default();
    let bytes = if rgba_in {
        encode_rgba8(width, height, &pixels, &lossless)
    } else {
        encode_rgb8(width, height, &pixels, &lossless)
    }
    .expect("in-bounds lossless encode succeeds");
    assert!(probe(&bytes));
    let img = decode(&bytes).expect("lossless output decodes");
    assert_eq!((img.width, img.height), (width, height));
    assert_eq!(img.format, PixelFormat::Rgba);
    let back = img.to_rgba8();
    for (i, px) in pixels.chunks_exact(bpp).enumerate() {
        assert_eq!(&back[i * 4..i * 4 + 3], &px[..3], "lossless pixel {i}");
        assert_eq!(back[i * 4 + 3], if rgba_in { px[3] } else { 0xff });
    }

    // Lossy: decodes, geometry + layout + alpha plane exact.
    let lossy = EncodeOptions::default().with_quality(quality);
    let bytes = if rgba_in {
        encode_rgba8(width, height, &pixels, &lossy)
    } else {
        encode_rgb8(width, height, &pixels, &lossy)
    }
    .expect("in-bounds lossy encode succeeds");
    let i = info(&bytes).expect("lossy output has readable headers");
    assert!(i.is_lossy);
    let img = decode(&bytes).expect("lossy output decodes");
    assert_eq!((img.width, img.height), (width, height));
    let alpha: Vec<u8> = if rgba_in {
        pixels.chunks_exact(4).map(|p| p[3]).collect()
    } else {
        Vec::new()
    };
    if alpha.iter().any(|&a| a != 0xff) {
        assert_eq!(img.format, PixelFormat::Yuva420P);
        assert_eq!(img.planes[3].data, alpha, "ALPH plane is lossless");
    } else {
        assert_eq!(img.format, PixelFormat::Yuv420P);
    }
    let frames = decode_all(&bytes).expect("decode_all agrees");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].image, img);
    let _ = img.to_rgb8();
});
