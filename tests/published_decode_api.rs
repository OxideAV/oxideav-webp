//! Round-116 published-API round trips for the flat, `image`-crate
//! compatible decode surface.
//!
//! These tests build a small RGBA buffer in memory, encode it to a
//! `.webp` with the in-crate VP8L lossless encoder
//! ([`oxideav_webp::encode_webp_lossless`]), then decode it back through
//! the **published** [`oxideav_webp::decode_webp`] entry point and assert
//! the round-tripped [`oxideav_webp::WebpFrame::rgba`] matches the input
//! byte-for-byte — proving the flat `image`-crate buffer shape:
//! `rgba.len() == width * height * 4`, no stride padding, no planar
//! layout.
//!
//! Every test here uses only standalone APIs (no `registry` feature), so
//! the file builds and runs under `--no-default-features`.

use oxideav_webp::{decode_webp, encode_webp_lossless, WebpError, WebpFrame};

/// Build a deterministic `width * height` RGBA8 ramp (no external input).
fn make_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // A spread-out pattern so every channel takes several values.
            buf.push((x.wrapping_mul(37).wrapping_add(y) & 0xff) as u8); // R
            buf.push((y.wrapping_mul(53).wrapping_add(x) & 0xff) as u8); // G
            buf.push(((x ^ y).wrapping_mul(101) & 0xff) as u8); // B
            buf.push((255 - ((x.wrapping_add(y)) & 0xff)) as u8); // A
        }
    }
    buf
}

/// Encode an RGBA buffer to a `.webp`, decode via the published
/// `decode_webp`, and return the single decoded frame.
fn roundtrip(width: u32, height: u32, rgba: &[u8]) -> WebpFrame {
    let file = encode_webp_lossless(rgba, width, height).expect("VP8L lossless encode");
    let image = decode_webp(&file).expect("decode_webp on our own encoded file");
    assert_eq!(
        image.frames.len(),
        1,
        "still image yields exactly one frame"
    );
    // Non-animated: ANIM fields must be absent.
    assert_eq!(image.anim_background_rgba, None);
    assert_eq!(image.anim_loop_count, None);
    image.frames.into_iter().next().unwrap()
}

#[test]
fn decode_webp_roundtrip_is_flat_rgba_buffer() {
    // The headline consumer property: encode → decode round-trips the
    // exact flat RGBA buffer with `len == w * h * 4` and no stride pad.
    let (w, h) = (7u32, 5u32);
    let src = make_rgba(w, h);
    let frame = roundtrip(w, h, &src);

    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert_eq!(frame.duration_ms, 0, "still image has zero frame delay");

    // Flat-buffer invariant: tightly packed, no stride padding.
    assert_eq!(frame.rgba.len(), (w * h * 4) as usize);
    assert_eq!(frame.rgba.len(), src.len());

    // Pixel-exact round trip.
    assert_eq!(frame.rgba, src, "round-tripped RGBA must equal the input");
}

#[test]
fn decode_webp_roundtrip_1x1() {
    // Smallest possible image — single pixel, all four channels distinct.
    let src = vec![0x12, 0x34, 0x56, 0x78];
    let frame = roundtrip(1, 1, &src);
    assert_eq!(frame.width, 1);
    assert_eq!(frame.height, 1);
    assert_eq!(frame.rgba.len(), 4);
    assert_eq!(frame.rgba, src);
}

#[test]
fn decode_webp_roundtrip_larger_buffer_shape() {
    // A 16x16 buffer: confirm the length identity holds at scale and the
    // bytes survive the round trip exactly (the `image` crate wraps this
    // via `ImageBuffer::from_raw(16, 16, rgba)`).
    let (w, h) = (16u32, 16u32);
    let src = make_rgba(w, h);
    let frame = roundtrip(w, h, &src);
    assert_eq!(frame.rgba.len(), (w * h * 4) as usize);
    assert_eq!(frame.rgba, src);
}

#[test]
fn decode_webp_rejects_garbage_as_invalid_data() {
    // A non-WebP buffer must surface the published coarse error, not panic.
    let err = decode_webp(b"not a webp file at all").expect_err("garbage is rejected");
    assert_eq!(err, WebpError::InvalidData);
}
