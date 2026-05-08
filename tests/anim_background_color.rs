//! End-to-end tests for the ANIM-chunk background-colour fidelity
//! path (RFC 9649 §2.5).
//!
//! Coverage:
//!
//! * **BGRA on disk → RGBA on canvas** — encode an animation with a
//!   non-zero background, decode it, verify the demuxer reads the bytes
//!   verbatim and the decoder converts them to row-major RGBA when
//!   filling dispose-to-background regions and the canvas backdrop
//!   visible through the frame's transparent pixels.
//! * **Loop count round-trip** — non-default loop count survives the
//!   encode → demux → decode cycle.
//! * **Non-animated files don't fabricate a background** — a still
//!   `.webp` (no ANIM chunk) returns `None` for the bg/loop fields on
//!   `WebpImage`.
//! * **Dispose-to-bg on a non-zero background** — when frame N disposes
//!   to background and frame N+1 doesn't cover the same bbox, frame N+1's
//!   canvas must show the background colour where N used to be, not
//!   transparent black.

use oxideav_webp::encoder_anim::{build_animated_webp, AnimFrame};
use oxideav_webp::{decode_webp, encode_vp8l_argb_with_metadata, WebpMetadata};

const W: u32 = 8;
const H: u32 = 8;

fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let n = (width as usize) * (height as usize);
    let mut v = Vec::with_capacity(n * 4);
    for _ in 0..n {
        v.extend_from_slice(&rgba);
    }
    v
}

#[test]
fn anim_background_bgra_to_rgba_conversion_round_trip() {
    // Pick a clearly-asymmetric BGRA so a byte-order mistake stands
    // out: B=0x10, G=0x20, R=0x30, A=0xff. Expected RGBA in the
    // decoder canvas: [0x30, 0x20, 0x10, 0xff].
    let bg_bgra: [u8; 4] = [0x10, 0x20, 0x30, 0xff];
    // Frame is a solid red 4×4 tile that covers the top-left
    // quadrant. After decode, the bottom-right region (never covered
    // by any frame) must show the BG colour in RGBA.
    let red_tile = solid(4, 4, [0xff, 0, 0, 0xff]);
    let frames = [AnimFrame {
        width: 4,
        height: 4,
        x_offset: 0,
        y_offset: 0,
        duration_ms: 50,
        blend: false,
        dispose_to_background: false,
        rgba: &red_tile,
    }];
    let blob = build_animated_webp(W, H, bg_bgra, 0, &frames).expect("encode");
    let img = decode_webp(&blob).expect("decode");
    // Outer metadata round-trip: anim_background_rgba must be the
    // BGRA → RGBA permutation of the encoder input.
    assert_eq!(
        img.anim_background_rgba,
        Some([0x30, 0x20, 0x10, 0xff]),
        "BGRA → RGBA conversion mismatch (decoder must permute the chunk bytes)"
    );
    assert_eq!(img.anim_loop_count, Some(0));
    // Top-left quadrant: red tile.
    let canvas = &img.frames[0].rgba;
    let stride = (W as usize) * 4;
    for y in 0..4 {
        for x in 0..4 {
            let i = y * stride + x * 4;
            assert_eq!(
                &canvas[i..i + 4],
                &[0xff, 0, 0, 0xff],
                "top-left should be red at ({x},{y})"
            );
        }
    }
    // Bottom-right region (uncovered by any frame): the canvas was
    // initialised with the ANIM background colour. Spec says this is
    // a SHOULD/MAY hint, but our decoder honours it for both canvas
    // init + dispose, so the uncovered pixels must be the BG.
    for y in 4..H as usize {
        for x in 4..W as usize {
            let i = y * stride + x * 4;
            assert_eq!(
                &canvas[i..i + 4],
                &[0x30, 0x20, 0x10, 0xff],
                "uncovered pixel ({x},{y}) must show the ANIM BG colour after BGRA→RGBA"
            );
        }
    }
}

#[test]
fn anim_dispose_to_background_uses_bg_color_not_transparent_black() {
    // Two-frame animation:
    //   F0: full-canvas red, dispose=true → after F0, the F0 region
    //       (the whole canvas) is wiped to the BG colour (NOT to
    //       transparent black).
    //   F1: tiny green 2×2 at (0,0), blend=false → overwrites just
    //       the top-left. The rest of the canvas must show the BG.
    let bg_bgra: [u8; 4] = [0x40, 0x50, 0x60, 0xff];
    // Expected RGBA: [R=0x60, G=0x50, B=0x40, A=0xff].
    let bg_rgba_expected = [0x60, 0x50, 0x40, 0xff];

    let f0 = solid(W, H, [0xff, 0, 0, 0xff]);
    let f1 = solid(2, 2, [0, 0xff, 0, 0xff]);
    let frames = [
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 40,
            blend: false,
            dispose_to_background: true,
            rgba: &f0,
        },
        AnimFrame {
            width: 2,
            height: 2,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 40,
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let blob = build_animated_webp(W, H, bg_bgra, 0, &frames).expect("encode");
    let img = decode_webp(&blob).expect("decode");
    assert_eq!(img.frames.len(), 2);
    let canvas = &img.frames[1].rgba;
    let stride = (W as usize) * 4;
    // Top-left 2×2: green.
    for y in 0..2 {
        for x in 0..2 {
            let i = y * stride + x * 4;
            assert_eq!(
                &canvas[i..i + 4],
                &[0, 0xff, 0, 0xff],
                "F1 green tile mismatch at ({x},{y})"
            );
        }
    }
    // Outside the 2×2 tile: BG colour (NOT (0,0,0,0)) — F0's dispose
    // wiped the canvas to BG before F1 rendered.
    for y in 0..H as usize {
        for x in 0..W as usize {
            if x < 2 && y < 2 {
                continue;
            }
            let i = y * stride + x * 4;
            assert_eq!(
                &canvas[i..i + 4],
                &bg_rgba_expected,
                "post-dispose bg-fill mismatch at ({x},{y}); expected RGBA={bg_rgba_expected:?}"
            );
        }
    }
}

#[test]
fn anim_loop_count_round_trips() {
    let red_tile = solid(W, H, [0xff, 0, 0, 0xff]);
    let frames = [AnimFrame {
        width: W,
        height: H,
        x_offset: 0,
        y_offset: 0,
        duration_ms: 100,
        blend: false,
        dispose_to_background: false,
        rgba: &red_tile,
    }];
    // 7 explicit loops (not 0 = infinite, not 1, just a clear value).
    let blob = build_animated_webp(W, H, [0, 0, 0, 0], 7, &frames).expect("encode");
    let img = decode_webp(&blob).expect("decode");
    assert_eq!(img.anim_loop_count, Some(7));
    assert_eq!(img.anim_background_rgba, Some([0, 0, 0, 0]));
}

#[test]
fn still_webp_without_anim_chunk_has_no_background() {
    // Build a non-animated lossless `.webp` and confirm both the
    // anim_background_rgba and anim_loop_count fields are `None`. A
    // simple-layout VP8L file has no VP8X header at all, so no ANIM
    // chunk could possibly be in there.
    let argb = vec![0xff_00_80_40u32; (W * H) as usize];
    let blob = encode_vp8l_argb_with_metadata(W, H, &argb, false, &WebpMetadata::default())
        .expect("vp8l encode");
    let img = decode_webp(&blob).expect("decode");
    assert_eq!(img.anim_background_rgba, None);
    assert_eq!(img.anim_loop_count, None);
}

#[test]
fn alpha_in_bg_color_is_preserved_through_canvas_init() {
    // Spec note: "The background color MAY contain a nonopaque alpha
    // value, even if the Alpha flag in the VP8X Chunk is unset."
    // Encode a still tile that doesn't cover the full canvas, then
    // confirm the uncovered pixels carry the BG's nonopaque alpha
    // verbatim.
    let bg_bgra: [u8; 4] = [0xa0, 0xb0, 0xc0, 0x80];
    let expected_rgba = [0xc0, 0xb0, 0xa0, 0x80];
    let solid_tile = solid(2, 2, [0, 0, 0, 0xff]);
    let frames = [AnimFrame {
        width: 2,
        height: 2,
        x_offset: 0,
        y_offset: 0,
        duration_ms: 50,
        blend: false,
        dispose_to_background: false,
        rgba: &solid_tile,
    }];
    let blob = build_animated_webp(W, H, bg_bgra, 0, &frames).expect("encode");
    let img = decode_webp(&blob).expect("decode");
    let canvas = &img.frames[0].rgba;
    let stride = (W as usize) * 4;
    // Pixel (4, 4) is well outside the tile — must be the BG.
    let i = 4 * stride + 4 * 4;
    assert_eq!(&canvas[i..i + 4], &expected_rgba);
}
