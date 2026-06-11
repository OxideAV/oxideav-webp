#![no_main]

//! Differential oracle on the §2.7.1.1 animation **assembly** path —
//! `build_animated_webp_with_options` → `decode_webp` — with every
//! per-frame carrier field under fuzz control.
//!
//! The existing `roundtrip_animated` harness only drives
//! `AnimFrame::new` defaults: full-canvas frames at `(0, 0)`,
//! `AnimFrameMode::Lossless`, `BlendingMethod::Overwrite`,
//! `DisposalMethod::None`, default `AnimEncoderOptions`. That leaves the
//! whole §2.7.1.1 carrier-semantics surface unfuzzed:
//!
//! * sub-canvas frames at even `(x, y)` offsets (Frame X / Y are stored
//!   as `coord / 2`);
//! * `AnimFrameMode::Delta` / `AnimFrameMode::Auto` — the dirty-rect
//!   sub-frame encoder that diffs each frame against the previous
//!   canvas state and emits only the changed bounding box;
//! * `DisposalMethod::Background` — the "clear the previous frame's
//!   rect to the ANIM background colour before rendering" rule;
//! * `BlendingMethod::AlphaBlend` — the §2.7.1.1 8-bit alpha-blending
//!   compositor formula;
//! * `AnimEncoderOptions` — `loop_count` + `background_rgba` (which
//!   feeds both the initial canvas fill and every Background dispose).
//!
//! The oracle is an independent §2.7.1.1 canvas simulation implemented
//! in this harness: initialise the canvas to the ANIM background, apply
//! the previous frame's disposal method, composite each frame with its
//! blending method, snapshot. Every decoded `WebpFrame` must match its
//! simulated canvas byte-for-byte, and duration / loop count /
//! background colour must carry through. A mismatch is a real
//! §2.7.1.1 contract violation in the assembly or compositing path —
//! e.g. a Delta dirty-rect computed against a reference canvas that
//! disagrees with the decoder-visible state.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::anmf::{BlendingMethod, DisposalMethod};
use oxideav_webp::{
    build_animated_webp_with_options, decode_webp, AnimEncoderOptions, AnimFrame, AnimFrameMode,
};

/// Frame-dimension cap. Offsets are capped at 14 (7 × 2, kept even), so
/// the canvas never exceeds (16 + 14)² = 30 × 30 → a 3.6 KiB RGBA
/// working set per frame and millisecond-scale iterations.
const MAX_DIM: u32 = 16;
/// Frame-count cap (6 frames × 30 × 30 × 4 ≈ 21.6 KiB pixel data max).
const MAX_FRAMES: usize = 6;
/// Fixed header: frame count (1) + background RGBA (4) + loop count (2).
const HEADER_FIXED: usize = 7;
/// Per-frame header: w, h, x, y, duration-lo, duration-hi, flags.
const HEADER_PER_FRAME: usize = 7;

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER_FIXED + HEADER_PER_FRAME {
        return;
    }

    let frame_count = (data[0] as usize % MAX_FRAMES) + 1;
    let background_rgba = [data[1], data[2], data[3], data[4]];
    let loop_count = u16::from_le_bytes([data[5], data[6]]);

    let headers = &data[HEADER_FIXED..];
    if headers.len() < frame_count * HEADER_PER_FRAME {
        return;
    }
    let pool = &headers[frame_count * HEADER_PER_FRAME..];

    // Build the fuzz-described frame list. Pixels are drawn from a
    // shared rolling pool so short pools yield repetitive frames —
    // exactly the equal-region content the Delta dirty-rect diff needs
    // to take its partial-rect and no-diff branches.
    let mut frames = Vec::with_capacity(frame_count);
    let mut cursor = 0usize;
    for i in 0..frame_count {
        let h7 = &headers[i * HEADER_PER_FRAME..(i + 1) * HEADER_PER_FRAME];
        let width = u32::from(h7[0]) % MAX_DIM + 1;
        let height = u32::from(h7[1]) % MAX_DIM + 1;
        // §2.7.1.1 stores Frame X / Y as coord / 2 → even offsets only.
        let x = (u32::from(h7[2]) % 8) * 2;
        let y = (u32::from(h7[3]) % 8) * 2;
        let duration = u32::from(u16::from_le_bytes([h7[4], h7[5]]));
        let flags = h7[6];
        let mode = match flags & 0b11 {
            0 => AnimFrameMode::Auto,
            1 => AnimFrameMode::Delta,
            _ => AnimFrameMode::Lossless,
        };
        let dispose = if flags & 0b100 != 0 {
            DisposalMethod::Background
        } else {
            DisposalMethod::None
        };
        let blend = if flags & 0b1000 != 0 {
            BlendingMethod::AlphaBlend
        } else {
            BlendingMethod::Overwrite
        };

        let len = (width * height * 4) as usize;
        let mut pixels = Vec::with_capacity(len);
        for _ in 0..len {
            pixels.push(if pool.is_empty() {
                0
            } else {
                let b = pool[cursor % pool.len()];
                cursor += 1;
                b
            });
        }

        frames.push(AnimFrame {
            pixels,
            width,
            height,
            x,
            y,
            duration,
            blend,
            dispose,
            mode,
        });
    }

    // §2.7.1.1: the canvas covers every frame rectangle.
    let canvas_w = frames.iter().map(|f| f.x + f.width).max().unwrap();
    let canvas_h = frames.iter().map(|f| f.y + f.height).max().unwrap();

    let opts = AnimEncoderOptions {
        loop_count,
        background_rgba,
        ..AnimEncoderOptions::default()
    };

    // Every frame above is structurally valid (non-zero dims, even
    // offsets, exact pixel length), so the build must succeed.
    let file = build_animated_webp_with_options(&frames, &opts)
        .expect("structurally valid frame list must assemble");
    let img = decode_webp(&file).expect("assembled animation must decode");

    assert_eq!(img.width, canvas_w, "canvas width");
    assert_eq!(img.height, canvas_h, "canvas height");
    assert_eq!(img.frames.len(), frames.len(), "frame count");
    assert_eq!(img.anim_loop_count, Some(loop_count), "ANIM loop count");
    assert_eq!(
        img.anim_background_rgba,
        Some(background_rgba),
        "ANIM background colour"
    );

    // Independent §2.7.1.1 canvas simulation: bg-filled canvas, previous
    // frame's dispose applied before each render, per-frame blend.
    let mut canvas = vec![0u8; (canvas_w * canvas_h * 4) as usize];
    for px in canvas.chunks_exact_mut(4) {
        px.copy_from_slice(&background_rgba);
    }
    let mut prev_dispose: Option<(u32, u32, u32, u32)> = None;

    for (i, (f, decoded)) in frames.iter().zip(img.frames.iter()).enumerate() {
        if let Some((px, py, pw, ph)) = prev_dispose.take() {
            fill_rect(&mut canvas, canvas_w, px, py, pw, ph, background_rgba);
        }
        composite(&mut canvas, canvas_w, f);
        if f.dispose == DisposalMethod::Background {
            prev_dispose = Some((f.x, f.y, f.width, f.height));
        }

        assert_eq!(decoded.width, canvas_w, "frame {i}: canvas-wide snapshot");
        assert_eq!(decoded.height, canvas_h, "frame {i}: canvas-high snapshot");
        assert_eq!(decoded.duration_ms, f.duration, "frame {i}: duration");
        assert_eq!(
            decoded.rgba, canvas,
            "frame {i}: decoded canvas == §2.7.1.1 simulation \
             (mode {:?}, blend {:?}, dispose {:?}, rect {}x{}+{}+{})",
            f.mode, f.blend, f.dispose, f.width, f.height, f.x, f.y
        );
    }
});

/// Fill the rectangle `(x, y, w, h)` of `canvas` with `rgba` — the
/// §2.7.1.1 `DisposalMethod::Background` rule.
fn fill_rect(canvas: &mut [u8], canvas_w: u32, x: u32, y: u32, w: u32, h: u32, rgba: [u8; 4]) {
    let stride = (canvas_w as usize) * 4;
    for row in 0..h as usize {
        let off = (y as usize + row) * stride + (x as usize) * 4;
        for col in 0..w as usize {
            canvas[off + col * 4..off + col * 4 + 4].copy_from_slice(&rgba);
        }
    }
}

/// Composite frame `f` onto `canvas` per its §2.7.1.1 blending method.
fn composite(canvas: &mut [u8], canvas_w: u32, f: &AnimFrame) {
    let stride = (canvas_w as usize) * 4;
    let fw = f.width as usize;
    for row in 0..f.height as usize {
        for col in 0..fw {
            let s = &f.pixels[(row * fw + col) * 4..(row * fw + col) * 4 + 4];
            let off = (f.y as usize + row) * stride + (f.x as usize + col) * 4;
            let d = &mut canvas[off..off + 4];
            match f.blend {
                BlendingMethod::Overwrite => d.copy_from_slice(s),
                BlendingMethod::AlphaBlend => blend_pixel(d, s),
            }
        }
    }
}

/// §2.7.1.1 alpha-blending formula, 8-bit integer approximation:
/// `blend.A = src.A + dst.A * (1 - src.A / 255)` and
/// `blend.RGB = (src.RGB * src.A + dst.RGB * dst.A * (1 - src.A / 255))
///  / blend.A` (RGB := 0 when `blend.A == 0`), with round-to-nearest on
/// both divisions.
fn blend_pixel(dst: &mut [u8], src: &[u8]) {
    let sa = u32::from(src[3]);
    if sa == 255 {
        dst.copy_from_slice(src);
        return;
    }
    if sa == 0 {
        return;
    }
    let da = u32::from(dst[3]);
    let dst_factor = (da * (255 - sa) + 127) / 255;
    let out_a = sa + dst_factor;
    for c in 0..3 {
        let sc = u32::from(src[c]);
        let dc = u32::from(dst[c]);
        let out = (sc * sa + dc * dst_factor + out_a / 2)
            .checked_div(out_a)
            .unwrap_or(0);
        dst[c] = out.min(255) as u8;
    }
    dst[3] = out_a.min(255) as u8;
}
