//! Round-141 integration tests for the **animation-wide near-lossless
//! default** — the new [`oxideav_webp::AnimEncoderOptions::default_near_lossless_quality`]
//! field that lets callers set the VP8L near-lossless preprocessing knob
//! once for the whole animation instead of repeating it on every
//! [`oxideav_webp::AnimFrame::near_lossless_quality`].
//!
//! Round-141 contract:
//!
//! * **Per-frame `Some(q)` always wins.** If a frame carries its own
//!   `near_lossless_quality`, the options-level default is ignored for
//!   that frame.
//! * **Per-frame `None` falls back to the default.** If a frame leaves
//!   its `near_lossless_quality` at the default `None`, the encoder
//!   substitutes
//!   [`AnimEncoderOptions::default_near_lossless_quality`].
//! * **Both `None` = baseline.** When neither the frame nor the options
//!   set a value, no quantization is applied; the per-frame VP8L
//!   bitstream is byte-exact-equal to the pre-round-140 encoder output
//!   (equivalent to `Some(100)` in either slot).
//!
//! All tests run under `--no-default-features`, the same shape as the
//! other `published_*` suites.

use oxideav_webp::{
    build_animated_webp, build_animated_webp_with_options, decode_webp, near_lossless,
    AnimEncoderOptions, AnimFrame, AnimFrameMode,
};

/// Deterministic xorshift32-driven noisy RGBA — the same helper the
/// round-140 suite uses; copied here to keep this test file
/// self-contained.
fn make_noisy_rgba(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    let mut s: u32 = 0x9e37_79b9 ^ seed.wrapping_mul(0x85eb_ca6b);
    for _ in 0..(width * height) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        buf.push((s & 0xff) as u8);
        buf.push(((s >> 8) & 0xff) as u8);
        buf.push(((s >> 16) & 0xff) as u8);
        buf.push(0xff);
    }
    buf
}

fn rgba_to_argb(rgba: &[u8]) -> Vec<u32> {
    rgba.chunks_exact(4)
        .map(|px| {
            (px[3] as u32) << 24 | (px[0] as u32) << 16 | (px[1] as u32) << 8 | (px[2] as u32)
        })
        .collect()
}

fn argb_to_rgba(argb: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(argb.len() * 4);
    for &p in argb {
        out.push((p >> 16) as u8);
        out.push((p >> 8) as u8);
        out.push(p as u8);
        out.push((p >> 24) as u8);
    }
    out
}

/// Build the per-frame ANMF chunk-byte-length list from an encoded
/// animation file. The encoder emits one ANMF per frame, in order, so
/// this gives the same per-frame size deltas the round-140 suite would
/// observe.
fn per_frame_anmf_chunk_lengths(file: &[u8]) -> Vec<u32> {
    // RIFF | u32 size | WEBP | chunks…  with each chunk being [fourcc | u32 size | payload | pad].
    let mut out = Vec::new();
    let mut i: usize = 12; // skip RIFF/size/WEBP
    while i + 8 <= file.len() {
        let fourcc = &file[i..i + 4];
        let chunk_size = u32::from_le_bytes(file[i + 4..i + 8].try_into().unwrap());
        let total = (chunk_size as usize) + 8 + (chunk_size as usize & 1);
        if fourcc == b"ANMF" {
            out.push(chunk_size);
        }
        i += total;
    }
    out
}

// ───────── Default fallback shrinks just like the per-frame knob ─────────

#[test]
fn options_default_60_matches_per_frame_60_byte_for_byte() {
    // The contract: a 3-frame animation built with
    // `default_near_lossless_quality = Some(60)` and no per-frame
    // overrides must produce a file byte-for-byte identical to the same
    // 3-frame animation built with `Some(60)` set explicitly on every
    // frame. This is the central "set once instead of per frame" test.
    let (w, h) = (64u32, 64u32);
    let srcs = [
        make_noisy_rgba(w, h, 0),
        make_noisy_rgba(w, h, 1),
        make_noisy_rgba(w, h, 2),
    ];

    // (A) Default-only: every AnimFrame leaves its knob at None.
    let frames_default = vec![
        AnimFrame::new(w, h, srcs[0].clone(), 100),
        AnimFrame::new(w, h, srcs[1].clone(), 110),
        AnimFrame::new(w, h, srcs[2].clone(), 120),
    ];
    let opts_default = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(60));
    let file_default = build_animated_webp_with_options(&frames_default, &opts_default)
        .expect("default Some(60) build");

    // (B) Per-frame Some(60) on every frame, no options-level default.
    let frames_per_frame = vec![
        AnimFrame::new(w, h, srcs[0].clone(), 100).with_near_lossless_quality(Some(60)),
        AnimFrame::new(w, h, srcs[1].clone(), 110).with_near_lossless_quality(Some(60)),
        AnimFrame::new(w, h, srcs[2].clone(), 120).with_near_lossless_quality(Some(60)),
    ];
    let file_per_frame = build_animated_webp(&frames_per_frame).expect("per-frame Some(60) build");

    assert_eq!(
        file_default, file_per_frame,
        "options default Some(60) must produce byte-exact-equal output to per-frame Some(60)"
    );

    // Round-140 measured a ~−24 % drop versus the q=100 baseline; we
    // re-confirm here on the same fixture to keep the "set once" path
    // on the same compression curve.
    let frames_baseline: Vec<AnimFrame> = srcs
        .iter()
        .enumerate()
        .map(|(i, px)| AnimFrame::new(w, h, px.clone(), 100 + (i as u32) * 10))
        .collect();
    let baseline = build_animated_webp(&frames_baseline).expect("baseline");
    assert!(
        file_default.len() < baseline.len(),
        "default Some(60) ({}) must shrink vs baseline ({})",
        file_default.len(),
        baseline.len()
    );
}

#[test]
fn options_default_none_matches_baseline_byte_for_byte() {
    // The "both `None`" leg of the contract: a default-constructed
    // `AnimEncoderOptions` (with `default_near_lossless_quality = None`)
    // produces output byte-exact-equal to the bare `build_animated_webp`
    // convenience entry, which itself is byte-exact-equal to the
    // pre-round-140 baseline (per the round-140 suite).
    let (w, h) = (16u32, 16u32);
    let frames = vec![
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 0), 100),
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 1), 110),
    ];
    let file_convenience = build_animated_webp(&frames).expect("convenience baseline");
    let file_default_opts =
        build_animated_webp_with_options(&frames, &AnimEncoderOptions::default())
            .expect("default opts baseline");
    assert_eq!(
        file_convenience, file_default_opts,
        "default options must be a baseline pass-through"
    );
}

#[test]
fn options_default_above_100_clamps_to_baseline_no_op() {
    // `near_lossless::apply` clamps `quality > 100` down to a no-op, so
    // setting the default to a value above 100 must still produce the
    // baseline bitstream.
    let (w, h) = (16u32, 16u32);
    let frames = vec![
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 0), 100),
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 1), 100),
    ];
    let baseline = build_animated_webp(&frames).expect("baseline");
    let opts_above = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(255));
    let above = build_animated_webp_with_options(&frames, &opts_above).expect("above-100 default");
    assert_eq!(
        baseline, above,
        "Some(255) default must clamp to the baseline no-op"
    );
}

// ───────── Per-frame override beats the default ─────────

#[test]
fn per_frame_some_overrides_options_default() {
    // Mixed sequence: default = Some(60), frame 1 overridden to Some(100).
    // Frames 0 and 2 (None per-frame) inherit the default 60; frame 1's
    // explicit Some(100) wins and produces a baseline-equivalent ANMF.
    let (w, h) = (64u32, 64u32);
    let srcs = [
        make_noisy_rgba(w, h, 0),
        make_noisy_rgba(w, h, 1),
        make_noisy_rgba(w, h, 2),
    ];

    let frames_mixed = vec![
        AnimFrame::new(w, h, srcs[0].clone(), 100),
        AnimFrame::new(w, h, srcs[1].clone(), 110).with_near_lossless_quality(Some(100)),
        AnimFrame::new(w, h, srcs[2].clone(), 120),
    ];
    let opts_mixed = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(60));
    let file_mixed = build_animated_webp_with_options(&frames_mixed, &opts_mixed)
        .expect("mixed default/override build");

    // Reference: the "all Some(60)" file from the same sources — frames
    // 0 and 2 must match those ANMFs byte-for-byte (same effective
    // quality), while frame 1's ANMF must match the baseline (q=100)
    // ANMF byte-for-byte instead.
    let frames_all60: Vec<AnimFrame> = srcs
        .iter()
        .enumerate()
        .map(|(i, px)| {
            AnimFrame::new(w, h, px.clone(), 100 + (i as u32) * 10)
                .with_near_lossless_quality(Some(60))
        })
        .collect();
    let file_all60 = build_animated_webp(&frames_all60).expect("all-60 build");

    let frames_all100: Vec<AnimFrame> = srcs
        .iter()
        .enumerate()
        .map(|(i, px)| AnimFrame::new(w, h, px.clone(), 100 + (i as u32) * 10))
        .collect();
    let file_all100 = build_animated_webp(&frames_all100).expect("all-100 build");

    let mixed_sizes = per_frame_anmf_chunk_lengths(&file_mixed);
    let all60_sizes = per_frame_anmf_chunk_lengths(&file_all60);
    let all100_sizes = per_frame_anmf_chunk_lengths(&file_all100);
    assert_eq!(mixed_sizes.len(), 3);
    assert_eq!(all60_sizes.len(), 3);
    assert_eq!(all100_sizes.len(), 3);

    // Frame 0 (default-applied) matches the all-60 size.
    assert_eq!(
        mixed_sizes[0], all60_sizes[0],
        "frame 0 (None → default 60) must match the all-60 ANMF size"
    );
    // Frame 1 (overridden to 100) matches the all-100 size and differs
    // from the all-60 size (the noisy fixture compresses better at q=60).
    assert_eq!(
        mixed_sizes[1], all100_sizes[1],
        "frame 1 (override Some(100)) must match the all-100 ANMF size"
    );
    assert!(
        mixed_sizes[1] > all60_sizes[1],
        "frame 1's overridden q=100 ANMF ({}) must be larger than its q=60 counterpart ({})",
        mixed_sizes[1],
        all60_sizes[1]
    );
    // Frame 2 (default-applied) matches the all-60 size.
    assert_eq!(
        mixed_sizes[2], all60_sizes[2],
        "frame 2 (None → default 60) must match the all-60 ANMF size"
    );

    // Overall: 2 of 3 frames shrink vs the baseline, the overridden one
    // matches the baseline byte-for-byte.
    assert!(
        mixed_sizes[0] < all100_sizes[0],
        "frame 0 shrinks vs baseline"
    );
    assert_eq!(
        mixed_sizes[1], all100_sizes[1],
        "frame 1 unchanged from baseline"
    );
    assert!(
        mixed_sizes[2] < all100_sizes[2],
        "frame 2 shrinks vs baseline"
    );
}

#[test]
fn per_frame_some_60_beats_options_default_some_100() {
    // Inverse mix: default = Some(100) (baseline), one frame overridden
    // to Some(60). Only that overridden frame shrinks; the other two
    // remain at baseline size.
    let (w, h) = (48u32, 48u32);
    let srcs = [
        make_noisy_rgba(w, h, 10),
        make_noisy_rgba(w, h, 11),
        make_noisy_rgba(w, h, 12),
    ];

    let frames_mixed = vec![
        AnimFrame::new(w, h, srcs[0].clone(), 100),
        AnimFrame::new(w, h, srcs[1].clone(), 100).with_near_lossless_quality(Some(60)),
        AnimFrame::new(w, h, srcs[2].clone(), 100),
    ];
    let opts = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(100));
    let file_mixed = build_animated_webp_with_options(&frames_mixed, &opts).unwrap();

    let frames_all100: Vec<AnimFrame> = srcs
        .iter()
        .map(|px| AnimFrame::new(w, h, px.clone(), 100))
        .collect();
    let file_all100 = build_animated_webp(&frames_all100).unwrap();

    let mixed = per_frame_anmf_chunk_lengths(&file_mixed);
    let all100 = per_frame_anmf_chunk_lengths(&file_all100);
    assert_eq!(mixed[0], all100[0], "frame 0 stays at baseline");
    assert!(
        mixed[1] < all100[1],
        "frame 1 (override Some(60)) must shrink vs baseline"
    );
    assert_eq!(mixed[2], all100[2], "frame 2 stays at baseline");
}

// ───────── Decode round-trip with options default applied ─────────

#[test]
fn options_default_round_trips_through_decoder() {
    // Decoder side: the file produced with the options-level default of
    // Some(60) decodes back to the same quantized pixels the still-image
    // `near_lossless::quantize` would produce on the source — exactly
    // like the round-140 per-frame test, just driven from the options
    // default instead of per-frame.
    let (w, h) = (32u32, 32u32);
    let srcs = [
        make_noisy_rgba(w, h, 100),
        make_noisy_rgba(w, h, 200),
        make_noisy_rgba(w, h, 300),
    ];
    let frames: Vec<AnimFrame> = srcs
        .iter()
        .enumerate()
        .map(|(i, px)| AnimFrame::new(w, h, px.clone(), 100 + (i as u32) * 10))
        .collect();
    let opts = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(60));
    let file = build_animated_webp_with_options(&frames, &opts).expect("build");
    let dec = decode_webp(&file).expect("decode");
    assert_eq!(dec.frames.len(), 3, "one decoded frame per ANMF");
    for (i, fr) in dec.frames.iter().enumerate() {
        let expected_argb = near_lossless::quantize(&rgba_to_argb(&srcs[i]), 60);
        let expected_rgba = argb_to_rgba(&expected_argb);
        assert_eq!(
            fr.rgba, expected_rgba,
            "frame {i} decoded RGBA must equal quantize(src, 60) byte-for-byte"
        );
    }
}

// ───────── Delta/Auto interaction with the options default ─────────

#[test]
fn options_default_applies_to_delta_dirty_rect_path() {
    // The default flows through to the dirty-rect sub-frame the same way
    // the per-frame knob does. Two-frame fixture: frame 0 baseline,
    // frame 1 (Delta mode) inherits the options default = Some(60),
    // shrinking versus the same setup with default = Some(100).
    let (w, h) = (32u32, 32u32);
    let f0 = make_noisy_rgba(w, h, 0);
    let mut f1 = f0.clone();
    for row in 8..24 {
        for col in 8..24 {
            let off = (row * w as usize + col) * 4;
            f1[off] ^= 0xff;
            f1[off + 1] ^= 0xff;
            f1[off + 2] ^= 0xff;
        }
    }

    let mut f1_frame = AnimFrame::new(w, h, f1.clone(), 80);
    f1_frame.mode = AnimFrameMode::Delta;
    let f0_frame = AnimFrame::new(w, h, f0.clone(), 80);

    let opts_baseline = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(100));
    let opts_q60 = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(60));

    let baseline =
        build_animated_webp_with_options(&[f0_frame.clone(), f1_frame.clone()], &opts_baseline)
            .unwrap();
    let q60 = build_animated_webp_with_options(&[f0_frame, f1_frame], &opts_q60).unwrap();
    assert!(
        q60.len() < baseline.len(),
        "delta-mode with default Some(60) ({}) must shrink vs default Some(100) ({})",
        q60.len(),
        baseline.len()
    );
}
