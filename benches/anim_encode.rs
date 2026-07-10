//! Criterion bench — §2.7.1.1 animation *encode* path
//! (`build_animated_webp`), the encode mirror of `benches/anim_decode.rs`.
//!
//! The animation encoder owns real per-frame work beyond the inner VP8L
//! bitstream: the dirty-rect diff against the previous canvas, the
//! `Auto` mode's keyframe-vs-delta arbitration (two full inner encodes
//! per frame), and the `ANMF`/`ANIM`/`VP8X` container assembly. Three
//! cells drive the same deterministic 4-frame moving-square timeline
//! through each [`AnimFrameMode`]:
//!
//! * `anim_encode_lossless_4f_48` — every frame a full-canvas VP8L
//!   keyframe (no diffing; container + inner-encoder floor).
//! * `anim_encode_delta_4f_48` — dirty-rect sub-frames after the first
//!   frame (diff pass + sub-rect extraction + smaller inner encodes).
//! * `anim_encode_auto_4f_48` — keyframe and delta both encoded per
//!   frame, byte-smaller stream kept (the default mode; upper bound on
//!   per-frame encode cost).
//!
//! The frame list is built once outside `b.iter`; the measured interval
//! is encode-only. Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench anim_encode -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::anim_encode::{build_animated_webp, AnimFrame, AnimFrameMode};

/// Canvas side (48 px keeps one full `Auto` sweep in the tens of
/// milliseconds while still exercising multi-block predictor / LZ77
/// paths in the inner VP8L encoder).
const SIDE: u32 = 48;
/// Timeline length.
const FRAMES: usize = 4;

/// Deterministic 4-frame timeline: a textured background (LCG fill,
/// fixed across frames) with a 12×12 solid square stepping diagonally
/// 8 px per frame — small dirty rects, exactly the shape the `Delta` /
/// `Auto` paths exist for.
fn moving_square_frames(mode: AnimFrameMode) -> Vec<AnimFrame> {
    let n = (SIDE * SIDE) as usize;
    // Background: deterministic LCG, opaque.
    let mut state = 0x1234_5678u32;
    let mut bg = Vec::with_capacity(n * 4);
    for _ in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Coarse texture (few distinct colours) so frames stay
        // compressible and the bench isn't dominated by noise pricing.
        let v = ((state >> 24) & 0xe0) as u8;
        bg.extend_from_slice(&[v, v.wrapping_add(0x20), v.wrapping_add(0x40), 0xff]);
    }
    (0..FRAMES)
        .map(|f| {
            let mut px = bg.clone();
            let off = 8 * f as u32;
            for y in off..(off + 12).min(SIDE) {
                for x in off..(off + 12).min(SIDE) {
                    let i = ((y * SIDE + x) * 4) as usize;
                    px[i..i + 4].copy_from_slice(&[0xff, 0x20, 0x20, 0xff]);
                }
            }
            let mut frame = AnimFrame::new(SIDE, SIDE, px, 40);
            frame.mode = mode;
            frame
        })
        .collect()
}

fn bench_anim_encode(c: &mut Criterion) {
    for (name, mode) in [
        ("anim_encode_lossless_4f_48", AnimFrameMode::Lossless),
        ("anim_encode_delta_4f_48", AnimFrameMode::Delta),
        ("anim_encode_auto_4f_48", AnimFrameMode::Auto),
    ] {
        let frames = moving_square_frames(mode);
        c.bench_function(name, |b| {
            b.iter(|| {
                let out = build_animated_webp(black_box(&frames)).expect("encode animation");
                black_box(out)
            })
        });
    }
}

criterion_group!(benches, bench_anim_encode);
criterion_main!(benches);
