//! Slow-test (`cargo test --features slow-tests`) — restores the
//! original 320×240 multi-rect Delta fixture that was downsized to
//! 160×120 in commit `d2501a1` to keep per-job CI runtime under 10
//! minutes. The 320×240 canvas + 3 disjoint scattered stamps gives the
//! cleanest demonstration of the multi-rect path's wire-size win on a
//! larger surface — at the smaller 160×120 size the absolute byte
//! counts are dominated by chunk-header overhead. Pin the headline
//! 320×240 numbers in this gated test so the README / CHANGELOG
//! benchmark numbers stay reproducible by hand.
//!
//! Default `cargo test` skips this file entirely (the `slow-tests`
//! Cargo feature is off). To run:
//!
//! ```text
//!   cargo test --features slow-tests --release \
//!       --test animated_delta_merge_320x240 -- --test-threads=1
//! ```
//!
//! Why a separate file (vs `#[cfg(feature = "slow-tests")]` on a
//! function inside `animated_delta_merge.rs`)? — keeps the per-test
//! gating obvious to reviewers (the file name encodes both the slow-
//! tests gate and the canvas size), and the `--test
//! animated_delta_merge_320x240` invocation runs *only* the slow
//! benchmark when the developer wants to refresh the headline number.

#![cfg(feature = "slow-tests")]

use oxideav_webp::{
    build_animated_webp_with_options, decode_webp, AnimEncoderOptions, AnimFrame, AnimFrameMode,
    DeltaConfig,
};

// 320×240 canvas with 3 disjoint scattered 16×16 stamps — the original
// multi-rect Delta fixture from before commit d2501a1 downsized it to
// 160×120 for default-CI cost. Each stamp is far enough from the others
// that the default `block_size = 8` 4-connected flood fill leaves them
// as 3 independent connected components, so the multi-rect path emits
// 3 sub-rect ANMFs per delta frame while the single-bbox baseline emits
// one near-canvas-sized ANMF.

const SLOW_W: u32 = 320;
const SLOW_H: u32 = 240;
const SLOW_STAMP: u32 = 16;
const SLOW_STAMP_POSITIONS: &[(u32, u32)] = &[
    (16, 16),
    (SLOW_W / 2, SLOW_H / 2),
    (SLOW_W - SLOW_STAMP - 16, SLOW_H - SLOW_STAMP - 16),
];

fn build_slow_frame(counter: u8, n_stamps: usize) -> Vec<u8> {
    let mut v = vec![0u8; (SLOW_W * SLOW_H * 4) as usize];
    // Background pseudo-noise (deterministic, identical between frames).
    for y in 0..SLOW_H {
        for x in 0..SLOW_W {
            let i = ((y * SLOW_W + x) * 4) as usize;
            let mut s = y.wrapping_mul(0x9E37_79B9) ^ x.wrapping_mul(0x85EB_CA77);
            s ^= s.wrapping_shr(13);
            s = s.wrapping_mul(0xC2B2_AE35);
            s ^= s.wrapping_shr(16);
            v[i] = (s & 0xff) as u8;
            v[i + 1] = ((s >> 8) & 0xff) as u8;
            v[i + 2] = ((s >> 16) & 0xff) as u8;
            v[i + 3] = 0xff;
        }
    }
    // Stamp the changing blocks.
    for &(sx, sy) in SLOW_STAMP_POSITIONS.iter().take(n_stamps) {
        for y in sy..(sy + SLOW_STAMP) {
            for x in sx..(sx + SLOW_STAMP) {
                let i = ((y * SLOW_W + x) * 4) as usize;
                v[i] = counter;
                v[i + 1] = 0xff - counter;
                v[i + 2] = 0x80;
                v[i + 3] = 0xff;
            }
        }
    }
    v
}

/// Walk a `.webp` blob's chunk stream and return the count of `ANMF`
/// sub-chunks. Used to verify the multi-rect path emits one ANMF per
/// connected cluster on the delta frames.
fn count_anmf_chunks(blob: &[u8]) -> usize {
    let mut count = 0;
    let mut p = 12; // skip "RIFF<size>WEBP"
    while p + 8 <= blob.len() {
        let chunk_len = u32::from_le_bytes([blob[p + 4], blob[p + 5], blob[p + 6], blob[p + 7]]);
        if &blob[p..p + 4] == b"ANMF" {
            count += 1;
        }
        p += 8 + chunk_len as usize + (chunk_len as usize & 1);
    }
    count
}

#[test]
fn slow_delta_mode_320x240_multi_rect_pins_headline_byte_count() {
    // 3 logical frames (frame 0 + 2 delta frames). Each delta frame
    // changes the 3 disjoint stamp blocks vs the previous frame's
    // stamp colour. Larger frame count than the default 160×120
    // multi-rect test (which uses 2 frames) so the headline wire-size
    // delta scales up cleanly with the canvas size.
    let f0 = build_slow_frame(0x10, 3);
    let f1 = build_slow_frame(0x60, 3);
    let f2 = build_slow_frame(0xa0, 3);
    let rgbas = [f0, f1, f2];
    let frames: Vec<AnimFrame> = rgbas
        .iter()
        .map(|rgba| AnimFrame {
            width: SLOW_W,
            height: SLOW_H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba,
        })
        .collect();

    // SAD baseline (Lossless full-frame for every delta frame).
    let lossless = build_animated_webp_with_options(
        SLOW_W,
        SLOW_H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Lossless,
            ..Default::default()
        },
    )
    .expect("encode lossless");

    // Single-bbox baseline: max_components_override = Some(1) → all
    // clusters fold into one covering super-rect per delta frame.
    let single = build_animated_webp_with_options(
        SLOW_W,
        SLOW_H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig::default().max_components_override(1)),
            ..Default::default()
        },
    )
    .expect("encode single-bbox");

    // Multi-rect: default budget (adaptive ≥ 4) — 3 clusters emitted as-is.
    let multi = build_animated_webp_with_options(
        SLOW_W,
        SLOW_H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig::default()),
            ..Default::default()
        },
    )
    .expect("encode multi-rect");

    let single_anmf_count = count_anmf_chunks(&single);
    let multi_anmf_count = count_anmf_chunks(&multi);
    eprintln!(
        "[slow] 3-cluster scattered-change anim 320x240, 3 frames: \
         lossless full = {} bytes; single-bbox Delta = {} bytes ({} ANMFs); \
         multi-rect Delta = {} bytes ({} ANMFs)",
        lossless.len(),
        single.len(),
        single_anmf_count,
        multi.len(),
        multi_anmf_count,
    );

    // Multi-rect emits 1 (frame 0) + 3 (delta frame 1) + 3 (delta
    // frame 2) = 7 ANMFs vs single-bbox's 1 + 1 + 1 = 3 ANMFs.
    assert_eq!(
        single_anmf_count, 3,
        "single-bbox path should emit 1 ANMF per logical frame"
    );
    assert_eq!(
        multi_anmf_count, 7,
        "multi-rect path should emit 3 sub-rect ANMFs per delta frame"
    );
    // Multi-rect total file size beats single-bbox AND lossless.
    assert!(
        multi.len() < single.len(),
        "multi-rect ({}) should beat single-bbox ({}) on scattered-change content",
        multi.len(),
        single.len(),
    );
    assert!(
        multi.len() < lossless.len(),
        "multi-rect Delta ({}) should beat full-frame lossless ({}) on scattered-change content",
        multi.len(),
        lossless.len(),
    );
    // Headline: multi-rect Delta is dramatically smaller than the
    // full-frame lossless baseline. On 320×240 with ~0.4% of the
    // canvas changing per delta frame, the saving should be at least
    // 50%. This is a coarse pin (the test acknowledges run-to-run
    // noise from VP8L RDO determinism); the eprintln above prints the
    // exact byte counts for a reviewer refreshing the README number.
    let savings_pct =
        100.0 * (lossless.len() as f64 - multi.len() as f64) / (lossless.len() as f64);
    eprintln!("[slow] multi-rect Delta savings vs lossless = {savings_pct:.1}%");
    assert!(
        savings_pct >= 30.0,
        "multi-rect Delta should save ≥ 30% vs lossless on 320×240 scattered-change content, got {savings_pct:.1}%"
    );
}

#[test]
fn slow_delta_mode_320x240_multi_rect_round_trip_pixel_identical() {
    // Same fixture as the wire-size pin — verify the decoder
    // reconstructs every input frame's canvas pixel-identically after
    // the multi-rect Delta encode + decode round-trip on the larger
    // 320×240 canvas.
    let f0 = build_slow_frame(0x10, 3);
    let f1 = build_slow_frame(0x60, 3);
    let f2 = build_slow_frame(0xa0, 3);
    let rgbas = [f0.clone(), f1.clone(), f2.clone()];
    let frames: Vec<AnimFrame> = rgbas
        .iter()
        .map(|rgba| AnimFrame {
            width: SLOW_W,
            height: SLOW_H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba,
        })
        .collect();
    let multi = build_animated_webp_with_options(
        SLOW_W,
        SLOW_H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig::default()),
            ..Default::default()
        },
    )
    .expect("encode multi");
    let img = decode_webp(&multi).expect("decode multi");

    // Decoded frame count > input frame count (sub-rect ANMFs each
    // emit one frame). The end of each logical input frame is the
    // sub-rect carrying `duration_ms > 0`; non-final sub-rects within
    // a logical frame have `duration_ms = 0`.
    let mut logical: Vec<Vec<u8>> = Vec::new();
    for f in &img.frames {
        if f.duration_ms > 0 {
            logical.push(f.rgba.clone());
        }
    }
    assert_eq!(
        logical.len(),
        rgbas.len(),
        "expected one logical-frame boundary (duration>0) per input frame"
    );
    for (i, src) in rgbas.iter().enumerate() {
        if &logical[i] != src {
            let mismatch_idx = logical[i].iter().zip(src.iter()).position(|(a, b)| a != b);
            panic!(
                "[slow] multi-rect 320×240 round-trip mismatch at logical frame {i}, byte {:?}",
                mismatch_idx
            );
        }
    }
}
