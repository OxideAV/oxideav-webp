//! Round-142 integration tests for the `AnimFrame::with_blend` /
//! `AnimFrame::with_dispose` chainable builders — and their end-to-end
//! round-trip semantics through `decode_webp`'s §2.7.1.1 canvas
//! compositor (background-disposal clear-to-bg, alpha-blend blending).
//!
//! Standalone-only (no `registry` feature), so the file builds under
//! `--no-default-features` as well. The fixtures are tiny offset
//! sub-frames placed on a small canvas with explicit colours so the
//! blend / dispose formulas are easy to verify by hand.

use oxideav_webp::anmf::{BlendingMethod, DisposalMethod};
use oxideav_webp::{
    build_animated_webp, build_animated_webp_with_options, decode_webp, AnimEncoderOptions,
    AnimFrame, DeltaConfig, WebpMetadata,
};

/// Build a `w*h*4` flat RGBA buffer filled with a single colour.
fn solid_rgba(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        v.extend_from_slice(&color);
    }
    v
}

/// Fetch a pixel from a flat RGBA buffer (`canvas_w` pixels wide).
fn pixel(canvas: &[u8], canvas_w: u32, x: u32, y: u32) -> [u8; 4] {
    let off = ((y as usize) * (canvas_w as usize) + (x as usize)) * 4;
    [
        canvas[off],
        canvas[off + 1],
        canvas[off + 2],
        canvas[off + 3],
    ]
}

#[test]
fn with_blend_alpha_blend_round_trips_through_decoder() {
    // Sanity: opaque fixtures encoded with `with_blend(AlphaBlend)` still
    // decode back to exact source pixels (because src.A = 255 makes
    // the alpha-blend formula equivalent to overwrite per §2.7.1.1).
    let (w, h) = (4u32, 4u32);
    let pixels = solid_rgba(w, h, [200, 100, 50, 255]);
    let f = AnimFrame::new(w, h, pixels.clone(), 100).with_blend(BlendingMethod::AlphaBlend);
    let file = build_animated_webp(&[f]).expect("build");
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 1);
    assert_eq!(
        img.frames[0].rgba, pixels,
        "opaque blend round-trips byte-exact"
    );
}

#[test]
fn with_blend_alpha_blend_blends_translucent_sub_frame_onto_previous_canvas() {
    // f0 covers the whole 8×8 canvas with an opaque RED keyframe.
    // f1 is a 2×2 translucent (alpha=128) BLUE sub-rect at (2,2) with
    // BlendingMethod::AlphaBlend. The decoder's §2.7.1.1 compositor
    // must blend f1's blue over f0's red — yielding a *purple-ish*
    // mixture inside the 2×2 rect and untouched red elsewhere.
    let canvas_w = 8u32;
    let canvas_h = 8u32;
    let red = [255u8, 0, 0, 255];
    let translucent_blue = [0u8, 0, 255, 128];

    let f0 = AnimFrame::new(canvas_w, canvas_h, solid_rgba(canvas_w, canvas_h, red), 100);

    let mut f1 = AnimFrame::new(2, 2, solid_rgba(2, 2, translucent_blue), 100);
    f1.x = 2;
    f1.y = 2;
    f1 = f1.with_blend(BlendingMethod::AlphaBlend);

    let file = build_animated_webp(&[f0, f1]).expect("build");
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 2);

    let composited = &img.frames[1].rgba;
    assert_eq!(img.frames[1].width, canvas_w);
    assert_eq!(img.frames[1].height, canvas_h);

    // Pixels OUTSIDE the 2×2 sub-rect are still RED (f0 leftovers).
    for y in 0..canvas_h {
        for x in 0..canvas_w {
            let in_rect = (2..4).contains(&x) && (2..4).contains(&y);
            if !in_rect {
                assert_eq!(
                    pixel(composited, canvas_w, x, y),
                    red,
                    "outside-rect pixel ({x},{y}) untouched"
                );
            }
        }
    }

    // INSIDE the 2×2 sub-rect: §2.7.1.1 blend formula (8-bit integer).
    // src = (0,0,255,128), dst = (255,0,0,255).
    // dst_factor = 255 * (255 - 128) / 255 = 127 (with rounding +127).
    //   (255 * 127 + 127) / 255 = 127.
    // out_a = 128 + 127 = 255.
    // out_r = (0*128 + 255*127 + 255/2) / 255
    //       = (32385 + 127) / 255 = 32512 / 255 = 127.
    // out_g = (0 + 0 + 127) / 255 = 0.
    // out_b = (255*128 + 0 + 127) / 255 = (32640 + 127) / 255 = 128.
    let inside = pixel(composited, canvas_w, 2, 2);
    assert_eq!(inside[3], 255, "blended alpha = 255");
    // R must be near 127 (purple-ish — somewhere between red 255 and
    // blue 0, with src.A = 128/255 weighting).
    assert!(
        inside[0] > 120 && inside[0] < 135,
        "blended R ≈ 127, got {}",
        inside[0]
    );
    assert_eq!(
        inside[1], 0,
        "blended G = 0 (neither src nor dst has green)"
    );
    assert!(
        inside[2] > 120 && inside[2] < 135,
        "blended B ≈ 128, got {}",
        inside[2]
    );
    // Spot-check the precise expected value for the spec formula.
    assert_eq!(inside, [127, 0, 128, 255], "blend formula bit-exact");
}

#[test]
fn with_blend_overwrite_replaces_translucent_sub_frame_byte_for_byte() {
    // Same setup as the alpha-blend test, but Overwrite — the 2×2
    // translucent BLUE pixels REPLACE the previous canvas inside the
    // sub-rect, alpha and all. Outside the rect is still RED.
    let canvas_w = 8u32;
    let canvas_h = 8u32;
    let red = [255u8, 0, 0, 255];
    let translucent_blue = [0u8, 0, 255, 128];

    let f0 = AnimFrame::new(canvas_w, canvas_h, solid_rgba(canvas_w, canvas_h, red), 100);
    let mut f1 = AnimFrame::new(2, 2, solid_rgba(2, 2, translucent_blue), 100);
    f1.x = 2;
    f1.y = 2;
    f1 = f1.with_blend(BlendingMethod::Overwrite);

    let file = build_animated_webp(&[f0, f1]).expect("build");
    let img = decode_webp(&file).expect("decode");
    let composited = &img.frames[1].rgba;

    // Inside the rect: byte-exact translucent blue (src), not a blend.
    for y in 2..4 {
        for x in 2..4 {
            assert_eq!(
                pixel(composited, canvas_w, x, y),
                translucent_blue,
                "overwrite blits src verbatim"
            );
        }
    }
    // Outside still red.
    assert_eq!(pixel(composited, canvas_w, 0, 0), red);
    assert_eq!(pixel(composited, canvas_w, 7, 7), red);
}

#[test]
fn with_dispose_background_clears_sub_rect_to_anim_bg_between_frames() {
    // Three-frame animation: f0 fills canvas with RED, f1 paints a 2×2
    // GREEN sub-rect at (2,2) with dispose=Background, f2 paints a 2×2
    // BLUE sub-rect at (4,4). The decoder's §2.7.1.1 rule "before
    // rendering each frame, the previous frame's Disposal method is
    // applied" must clear f1's 2×2 rect to the ANIM background colour
    // BEFORE drawing f2 — so the f2 snapshot must have:
    //   * The f1 rect at (2,2) cleared back to the ANIM bg.
    //   * The f2 rect at (4,4) holding GREEN over… wait, blue.
    //   * Everywhere else still RED.
    let canvas_w = 8u32;
    let canvas_h = 8u32;
    let red = [255u8, 0, 0, 255];
    let green = [0u8, 255, 0, 255];
    let blue = [0u8, 0, 255, 255];
    let bg = [10u8, 20, 30, 255];

    let f0 = AnimFrame::new(canvas_w, canvas_h, solid_rgba(canvas_w, canvas_h, red), 100);

    let mut f1 = AnimFrame::new(2, 2, solid_rgba(2, 2, green), 100);
    f1.x = 2;
    f1.y = 2;
    f1 = f1.with_dispose(DisposalMethod::Background);

    let mut f2 = AnimFrame::new(2, 2, solid_rgba(2, 2, blue), 100);
    f2.x = 4;
    f2.y = 4;

    let opts = AnimEncoderOptions {
        loop_count: 0,
        background_rgba: bg,
        metadata: WebpMetadata::default(),
        delta: DeltaConfig::default(),
        default_near_lossless_quality: None,
        default_lossy_quality: None,
    };

    let file = build_animated_webp_with_options(&[f0, f1, f2], &opts).expect("build");
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 3);

    // Frame 1 snapshot: GREEN inside the (2..4, 2..4) sub-rect, RED
    // everywhere else (f0's keyframe still there).
    let f1_snap = &img.frames[1].rgba;
    for y in 0..canvas_h {
        for x in 0..canvas_w {
            let in_rect = (2..4).contains(&x) && (2..4).contains(&y);
            let expected = if in_rect { green } else { red };
            assert_eq!(
                pixel(f1_snap, canvas_w, x, y),
                expected,
                "f1 pixel ({x},{y})"
            );
        }
    }

    // Frame 2 snapshot: the (2..4, 2..4) sub-rect from f1 is now BG
    // (dispose=Background cleared it before f2 was drawn). The
    // (4..6, 4..6) sub-rect from f2 is BLUE. Everything else still RED.
    let f2_snap = &img.frames[2].rgba;
    for y in 0..canvas_h {
        for x in 0..canvas_w {
            let in_f1_rect = (2..4).contains(&x) && (2..4).contains(&y);
            let in_f2_rect = (4..6).contains(&x) && (4..6).contains(&y);
            let expected = if in_f2_rect {
                blue
            } else if in_f1_rect {
                bg
            } else {
                red
            };
            assert_eq!(
                pixel(f2_snap, canvas_w, x, y),
                expected,
                "f2 pixel ({x},{y})"
            );
        }
    }
}

#[test]
fn with_dispose_none_leaves_previous_sub_rect_intact_between_frames() {
    // Mirror of the above test, but f1 keeps the default
    // DisposalMethod::None — so f1's GREEN rect remains on the canvas
    // when f2 is drawn.
    let canvas_w = 8u32;
    let canvas_h = 8u32;
    let red = [255u8, 0, 0, 255];
    let green = [0u8, 255, 0, 255];
    let blue = [0u8, 0, 255, 255];
    let bg = [10u8, 20, 30, 255];

    let f0 = AnimFrame::new(canvas_w, canvas_h, solid_rgba(canvas_w, canvas_h, red), 100);

    let mut f1 = AnimFrame::new(2, 2, solid_rgba(2, 2, green), 100);
    f1.x = 2;
    f1.y = 2;
    f1 = f1.with_dispose(DisposalMethod::None);

    let mut f2 = AnimFrame::new(2, 2, solid_rgba(2, 2, blue), 100);
    f2.x = 4;
    f2.y = 4;

    let opts = AnimEncoderOptions {
        loop_count: 0,
        background_rgba: bg,
        metadata: WebpMetadata::default(),
        delta: DeltaConfig::default(),
        default_near_lossless_quality: None,
        default_lossy_quality: None,
    };

    let file = build_animated_webp_with_options(&[f0, f1, f2], &opts).expect("build");
    let img = decode_webp(&file).expect("decode");

    // Frame 2 snapshot: f1's GREEN rect is STILL there (no dispose).
    let f2_snap = &img.frames[2].rgba;
    for y in 0..canvas_h {
        for x in 0..canvas_w {
            let in_f1_rect = (2..4).contains(&x) && (2..4).contains(&y);
            let in_f2_rect = (4..6).contains(&x) && (4..6).contains(&y);
            let expected = if in_f2_rect {
                blue
            } else if in_f1_rect {
                green
            } else {
                red
            };
            assert_eq!(
                pixel(f2_snap, canvas_w, x, y),
                expected,
                "f2 pixel ({x},{y}) — dispose=None preserves f1"
            );
        }
    }
}

#[test]
fn dispose_background_and_alpha_blend_compose_through_encoder() {
    // A more involved fixture: f0 is the keyframe (opaque red), f1 is
    // a translucent green sub-rect with dispose=Background AND
    // blend=AlphaBlend, f2 is a small opaque blue keyframe at a
    // different offset.
    //
    // The blending in f1 must blend over f0's red. After f1, the
    // dispose-to-background then clears f1's rect to the ANIM bg
    // before f2 is drawn. f2's snapshot must therefore show:
    //   * f1 rect = BG (post-dispose).
    //   * f2 rect = BLUE.
    //   * Elsewhere = RED.
    let canvas_w = 8u32;
    let canvas_h = 8u32;
    let red = [255u8, 0, 0, 255];
    let blue = [0u8, 0, 255, 255];
    let bg = [40u8, 50, 60, 255];

    let f0 = AnimFrame::new(canvas_w, canvas_h, solid_rgba(canvas_w, canvas_h, red), 100);

    let mut f1 = AnimFrame::new(2, 2, solid_rgba(2, 2, [0, 255, 0, 128]), 100);
    f1.x = 2;
    f1.y = 2;
    f1 = f1
        .with_blend(BlendingMethod::AlphaBlend)
        .with_dispose(DisposalMethod::Background);

    let mut f2 = AnimFrame::new(2, 2, solid_rgba(2, 2, blue), 100);
    f2.x = 4;
    f2.y = 4;

    let opts = AnimEncoderOptions {
        loop_count: 0,
        background_rgba: bg,
        metadata: WebpMetadata::default(),
        delta: DeltaConfig::default(),
        default_near_lossless_quality: None,
        default_lossy_quality: None,
    };

    let file = build_animated_webp_with_options(&[f0, f1, f2], &opts).expect("build");
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 3);

    // Frame 1 snapshot: f1 rect is the alpha-blend of green/128 over
    // red. Compute the expected pixel via the spec formula:
    //   src = (0, 255, 0, 128), dst = (255, 0, 0, 255).
    //   dst_factor = (255 * 127 + 127) / 255 = 127.
    //   out_a = 128 + 127 = 255.
    //   out_r = (0*128 + 255*127 + 127) / 255 = (32385 + 127) / 255 = 127.
    //   out_g = (255*128 + 0 + 127) / 255 = (32640 + 127) / 255 = 128.
    //   out_b = 0.
    let blended = [127u8, 128, 0, 255];
    let f1_snap = &img.frames[1].rgba;
    for y in 2..4 {
        for x in 2..4 {
            assert_eq!(
                pixel(f1_snap, canvas_w, x, y),
                blended,
                "f1 blended pixel ({x},{y})"
            );
        }
    }

    // Frame 2 snapshot: f1 rect is BG (post-dispose), f2 rect is blue.
    let f2_snap = &img.frames[2].rgba;
    for y in 0..canvas_h {
        for x in 0..canvas_w {
            let in_f1_rect = (2..4).contains(&x) && (2..4).contains(&y);
            let in_f2_rect = (4..6).contains(&x) && (4..6).contains(&y);
            let expected = if in_f2_rect {
                blue
            } else if in_f1_rect {
                bg
            } else {
                red
            };
            assert_eq!(
                pixel(f2_snap, canvas_w, x, y),
                expected,
                "f2 pixel ({x},{y})"
            );
        }
    }
}
