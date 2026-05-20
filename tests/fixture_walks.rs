//! Integration tests that drive [`oxideav_webp::parse_container`]
//! against the fixed WebP corpus in `docs/image/webp/fixtures/`.
//!
//! The fixtures here are *opaque* inputs to the round-1 walker:
//! we only verify the RIFF/WEBP structural layer (FourCC, sizes,
//! payload bounds). Pixel-level decode is not yet implemented and
//! is explicitly out of scope for this test file.

use oxideav_webp::container::fourcc;
use oxideav_webp::parse_container;

const LOSSY_1X1: &[u8] = include_bytes!("../../../docs/image/webp/fixtures/lossy-1x1/input.webp");
const LOSSLESS_1X1: &[u8] =
    include_bytes!("../../../docs/image/webp/fixtures/lossless-1x1/input.webp");
const EXTENDED_WITH_EXIF: &[u8] =
    include_bytes!("../../../docs/image/webp/fixtures/extended-with-exif/input.webp");

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
