//! Integration tests that drive [`oxideav_webp::parse_container`]
//! against a small in-crate copy of the WebP fixture corpus.
//!
//! The three `tests/data/*.webp` files are byte-for-byte copies of
//! `docs/image/webp/fixtures/{lossy-1x1, lossless-1x1,
//! extended-with-exif}/input.webp`. They live inside the crate so
//! that CI (which checks out only this repository, not the umbrella
//! workspace) can reach them via `include_bytes!`.
//!
//! Fixtures here are *opaque* inputs to the round-1 walker: we only
//! verify the RIFF/WEBP structural layer (FourCC, sizes, payload
//! bounds). Pixel-level decode is not yet implemented and is
//! explicitly out of scope for this test file.

use oxideav_webp::container::fourcc;
use oxideav_webp::{parse_container, parse_vp8x_header};

const LOSSY_1X1: &[u8] = include_bytes!("data/lossy-1x1.webp");
const LOSSLESS_1X1: &[u8] = include_bytes!("data/lossless-1x1.webp");
const EXTENDED_WITH_EXIF: &[u8] = include_bytes!("data/extended-with-exif.webp");

#[test]
fn fixture_lossy_1x1_has_a_single_vp8_chunk() {
    // RFC 9649 §2.5 — simple lossy: one 'VP8 ' chunk after the
    // 12-byte WebP file header.
    let c = parse_container(LOSSY_1X1).expect("lossy-1x1 fixture parses");
    assert_eq!(c.chunks.len(), 1);
    assert_eq!(c.chunks[0].fourcc, fourcc::VP8);
    assert!(!c.is_extended());
}

#[test]
fn fixture_lossless_1x1_has_a_single_vp8l_chunk() {
    // RFC 9649 §2.6 — simple lossless: one 'VP8L' chunk after the
    // 12-byte WebP file header.
    let c = parse_container(LOSSLESS_1X1).expect("lossless-1x1 fixture parses");
    assert_eq!(c.chunks.len(), 1);
    assert_eq!(c.chunks[0].fourcc, fourcc::VP8L);
    assert!(!c.is_extended());
}

#[test]
fn fixture_extended_with_exif_walks_vp8x_vp8_exif() {
    // RFC 9649 §2.7 — extended layout: 'VP8X' first, then the
    // image data ('VP8 ' here), then auxiliary chunks ('EXIF').
    let c = parse_container(EXTENDED_WITH_EXIF).expect("extended-with-exif fixture parses");
    let order: Vec<_> = c.chunks.iter().map(|c| c.fourcc).collect();
    assert_eq!(&order[0], &fourcc::VP8X);
    assert!(c.is_extended());
    assert!(c.first_chunk_with_fourcc(fourcc::VP8).is_some());
    assert!(c.first_chunk_with_fourcc(fourcc::EXIF).is_some());
}

#[test]
fn fixture_extended_with_exif_vp8x_payload_decodes_to_128x128_exif_only() {
    // Round-2 surface: walker → typed VP8X. Anchors the bit-position
    // decode against a real libwebp-produced VP8X chunk. The fixture's
    // own `trace.txt` reports the same flags / dimensions.
    let c = parse_container(EXTENDED_WITH_EXIF).expect("extended-with-exif fixture parses");
    let vp8x = c
        .first_chunk_with_fourcc(fourcc::VP8X)
        .expect("VP8X chunk present");
    let h = parse_vp8x_header(vp8x.payload(EXTENDED_WITH_EXIF))
        .expect("VP8X payload parses per §2.7.1");
    assert_eq!(h.canvas_width, 128);
    assert_eq!(h.canvas_height, 128);
    assert!(h.has_exif);
    assert!(!h.has_iccp);
    assert!(!h.has_xmp);
    assert!(!h.has_alpha);
    assert!(!h.has_animation);
    // Producer set every §2.7.1 reserved bit to 0 — has_unknown stays
    // clear, matching the trace report of `flags=0x00000008`.
    assert!(!h.has_unknown);
}
