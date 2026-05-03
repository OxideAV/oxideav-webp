//! Regression for issue #8 — VP8L predictor produced wrong B channel at
//! pixel (1, 53) in a 5×78 image encoded by libwebp.
//!
//! The crash was surfaced by the `webp_encode_oxideav_decode_lossless`
//! fuzz harness against `crash-274` (Shnatsel corpus). Pre-fix our
//! decoder returned `[0, 0, 84, 122]` at pixel 266, against the input
//! ground-truth `[0, 0, 0, 122]`.
//!
//! Root cause: the predictor's "top-right" (TR) neighbour at the right-
//! most column was implemented as the LEFT neighbour `out[y*w + x - 1]`
//! instead of the leftmost pixel of the current row `out[y*w]`. The spec
//! (RFC 9649 §4.1, "Addressing the TR-pixel for pixels on the rightmost
//! column is exceptional. … the leftmost pixel on the same row as the
//! current pixel is instead used as the TR-pixel.") makes the bug
//! cascade across rows: every (4, y-1) used by predictor mode 3/5/9/10
//! propagates into subsequent rows' L/T/TL/TR neighbourhoods.
//!
//! Fixture is the libwebp-encoded `.webp` (170 B) plus the original
//! 5×78 RGBA buffer (1560 B) — both produced from `/tmp/crash-274` per
//! the fuzz harness's slicing rules:
//!     shape = data[0]; rgba = data[1..]; pixel_count = (rgba.len()/4).min(2048)
//!     width = (shape%64)+1; height = pixel_count/width
//! For `crash-274`, shape=4 → width=5, height=78, used_len=1560.

use oxideav_webp::decode_webp;

const WEBP_BYTES: &[u8] = include_bytes!("fixtures/issue_8_pixel_266.webp");
const EXPECTED_RGBA: &[u8] = include_bytes!("fixtures/issue_8_pixel_266.expected_rgba");

const WIDTH: u32 = 5;
const HEIGHT: u32 = 78;

#[test]
fn issue_8_predictor_tr_at_rightmost_column() {
    assert_eq!(EXPECTED_RGBA.len(), (WIDTH * HEIGHT * 4) as usize);

    let img = decode_webp(WEBP_BYTES).expect("decode_webp must succeed for issue #8 fixture");
    assert_eq!(img.width, WIDTH);
    assert_eq!(img.height, HEIGHT);
    assert_eq!(img.frames.len(), 1);

    let actual = &img.frames[0].rgba;
    assert_eq!(actual.len(), EXPECTED_RGBA.len(), "RGBA length mismatch");

    // First, sanity-check pixel 266 explicitly so the regression failure
    // mode is clearly labelled if the bug returns.
    let i = 266 * 4;
    assert_eq!(
        &actual[i..i + 4],
        &EXPECTED_RGBA[i..i + 4],
        "pixel 266 (x=1, y=53) regressed — issue #8: predictor TR at \
         rightmost column must use the LEFTMOST pixel of the current \
         row, not the LEFT neighbour"
    );

    // Then assert the whole buffer matches; failures here surface as
    // the harness-style per-pixel diff.
    for (idx, (a, e)) in actual
        .chunks_exact(4)
        .zip(EXPECTED_RGBA.chunks_exact(4))
        .enumerate()
    {
        // Mirror the fuzz harness's leniency for transparent pixels.
        if e[3] == 0 {
            assert_eq!(
                a[3], 0,
                "decoded alpha differs for transparent pixel {idx}"
            );
        } else {
            assert_eq!(a, e, "decoded RGBA differs at pixel {idx}");
        }
    }
}
