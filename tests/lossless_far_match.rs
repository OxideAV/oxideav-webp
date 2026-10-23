//! Regression test: lossless encoding must not panic when the only
//! available backward-reference match sits farther back than the
//! largest distance a VP8L stream can encode.
//!
//! VP8L distance prefix codes span an alphabet of 40 symbols. With
//! the distance-map fallback (`distance + 120`), the largest
//! encodable raw distance is 1_048_456 pixels: past that, the
//! prefix index reaches 40 and indexing the 40-entry distance
//! histogram panicked with
//! `index out of bounds: the len is 40 but the index is 40`.
//!
//! The image below plants a unique 64-pixel pattern at offset 0 and
//! repeats it at offset 1_048_500, so the matcher finds a long match
//! whose distance has no encoding. The encoder must skip it (emit
//! literals instead) and still round-trip losslessly.

use oxideav_webp::{EncodeOptions, decode_rgb8, encode_rgb8};

fn lcg(state: &mut u32) -> u8 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state >> 24) as u8
}

#[test]
fn lossless_encode_with_unencodable_match_distance_does_not_panic() {
    let (w, h) = (1025usize, 1024usize);
    let n = w * h;
    // Smooth gradient background: 4-pixel windows are near-unique,
    // so the planted pattern's hash chain contains only its two
    // occurrences.
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        let (x, y) = (i % w, i / w);
        rgb[i * 3] = (x & 0xFF) as u8;
        rgb[i * 3 + 1] = (y & 0xFF) as u8;
        rgb[i * 3 + 2] = ((x + y) & 0xFF) as u8;
    }
    // Plant a unique random pattern at offset 0 and repeat it at
    // FAR_OFFSET pixels, beyond the largest encodable distance.
    const FAR_OFFSET: usize = 1_048_500;
    let mut state = 0x1234_5678u32;
    let mut pattern = vec![0u8; 64 * 3];
    for b in pattern.iter_mut() {
        *b = lcg(&mut state);
    }
    rgb[..pattern.len()].copy_from_slice(&pattern);
    rgb[FAR_OFFSET * 3..FAR_OFFSET * 3 + pattern.len()].copy_from_slice(&pattern);

    let opts = EncodeOptions::default();
    assert!(
        opts.quality.is_none(),
        "test must exercise the lossless path"
    );
    let encoded = encode_rgb8(w as u32, h as u32, &rgb, &opts).expect("encode must not panic");
    assert_eq!(&encoded[0..4], b"RIFF");
    assert_eq!(&encoded[8..12], b"WEBP");

    let decoded = decode_rgb8(&encoded).expect("decode own output");
    assert_eq!((decoded.width, decoded.height), (w as u32, h as u32));
    assert_eq!(decoded.data, rgb, "lossless round-trip must be exact");
}
