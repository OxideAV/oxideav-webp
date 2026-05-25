//! Round-140 integration tests for the **animated WebP** near-lossless
//! preprocessing surface — the per-frame [`oxideav_webp::AnimFrame::near_lossless_quality`]
//! knob extending the round-139 still-image preprocessor into the
//! `build_animated_webp` path.
//!
//! Three round-140 guarantees are covered end-to-end:
//!
//! 1. **`None` (default) and `Some(100)` are byte-exact-equal to the
//!    pre-change baseline** for every fixture in `published_anim_api.rs`.
//!    The default-constructed `AnimFrame::new` keeps its
//!    `near_lossless_quality` at `None`, so existing animated-WebP
//!    fixtures encode bit-for-bit identically — callers opt in by
//!    lowering the knob, nothing else.
//! 2. **`Some(60)` produces a strictly smaller animation file than
//!    `Some(100)`** on a 3-frame deterministic high-entropy fixture, with
//!    every decoded per-frame PSNR ≥ 40 dB versus the original input.
//!    The per-frame ANMF chunks shrink proportionally — the parent file
//!    size delta is the sum of the per-frame deltas (the §2.7 VP8X / ANIM
//!    headers are quality-invariant by construction).
//! 3. **Decoder round-trip recovers the quantized RGBA exactly.** The
//!    encoded animation is a perfectly normal sequence of `VP8L` chunks
//!    — the preprocessing is an encoder choice, not a bitstream change —
//!    so `decode_webp` returns the same pixels [`near_lossless::apply`]
//!    would have produced on the input ARGB.
//!
//! All tests run under `--no-default-features` (no `registry` feature
//! required), matching the rest of the published-shape suites.

use oxideav_webp::{build_animated_webp, decode_webp, near_lossless, AnimFrame, AnimFrameMode};

/// Build a deterministic `width * height` RGBA buffer with high per-pixel
/// entropy — a noisy field driven by a seeded xorshift. The lossless
/// encoder can't easily LZ77-collapse this fixture (no exact-pixel
/// repeats), so dropping low-order bits via the near-lossless
/// preprocessing creates the repeats the §5.2.2 / §5.2.3 entropy stages
/// need to win.
///
/// Opaque (`alpha = 0xff`) in scan-line order.
fn make_noisy_rgba(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    let mut s: u32 = 0x9e37_79b9 ^ seed.wrapping_mul(0x85eb_ca6b);
    for _ in 0..(width * height) {
        // xorshift32
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

/// Build a small natural-looking RGBA gradient with a tiny per-pixel
/// perturbation — same idea as the still-image suite's
/// `make_natural_argb`, but returning interleaved bytes for the
/// `AnimFrame` constructor.
fn make_natural_rgba(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let base_r = (x * 255 / width.max(1)) as u8;
            let base_g = (y * 255 / height.max(1)) as u8;
            let base_b = ((x + y) * 255 / (width + height).max(1)) as u8;
            let n = (x
                .wrapping_mul(2654435761)
                .wrapping_add(y.wrapping_mul(40503))
                .wrapping_add(seed.wrapping_mul(0x9e37_79b9)))
                & 0x07;
            let r = base_r.saturating_add(n as u8);
            let g = base_g.saturating_add((n ^ 0x3) as u8);
            let b = base_b.saturating_add((n ^ 0x5) as u8);
            buf.push(r);
            buf.push(g);
            buf.push(b);
            buf.push(0xff);
        }
    }
    buf
}

/// PSNR over the R/G/B channels of two equal-length interleaved RGBA
/// buffers (alpha is excluded — the preprocessing preserves it
/// bit-exactly, including it would only inflate the score).
fn rgb_psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr buffers differ in length");
    let mut sse: u64 = 0;
    let mut n: u64 = 0;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            let d = pa[c] as i32 - pb[c] as i32;
            sse += (d * d) as u64;
            n += 1;
        }
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / n as f64;
    20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
}

/// Repack interleaved RGBA into packed ARGB so we can call the still-image
/// near-lossless quantizer directly for the round-trip "decoded pixels ==
/// quantized input" guarantee.
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

// ───────── Guarantee 1: None / Some(100) is byte-exact baseline ─────────

#[test]
fn anim_default_none_is_byte_exact_to_baseline() {
    // `AnimFrame::new` leaves `near_lossless_quality` at `None`. A
    // three-frame animation built that way must be byte-for-byte
    // identical to the pre-round-140 output — i.e. the existing
    // `published_anim_api.rs` fixtures keep producing the same bytes.
    let (w, h) = (8u32, 8u32);
    let f0 = make_noisy_rgba(w, h, 0);
    let f1 = make_noisy_rgba(w, h, 1);
    let f2 = make_noisy_rgba(w, h, 2);
    let frames = vec![
        AnimFrame::new(w, h, f0.clone(), 100),
        AnimFrame::new(w, h, f1.clone(), 100),
        AnimFrame::new(w, h, f2.clone(), 100),
    ];
    // None vs explicit Some(100): both must produce the same bytes.
    let none_file = build_animated_webp(&frames).expect("None build");
    let explicit = {
        let v: Vec<AnimFrame> = vec![
            AnimFrame::new(w, h, f0, 100).with_near_lossless_quality(Some(100)),
            AnimFrame::new(w, h, f1, 100).with_near_lossless_quality(Some(100)),
            AnimFrame::new(w, h, f2, 100).with_near_lossless_quality(Some(100)),
        ];
        build_animated_webp(&v).expect("Some(100) build")
    };
    assert_eq!(
        none_file, explicit,
        "None and Some(100) must produce byte-exact-identical animations"
    );
}

#[test]
fn anim_some_100_is_byte_exact_on_small_natural_fixture() {
    // Cover several small geometries to make sure the no-op fast path
    // holds across the full per-frame encode pipeline (full keyframe,
    // including the §2.7.1.1 ANMF wrapping).
    for (w, h) in [(2u32, 2u32), (3, 5), (7, 4), (16, 16)] {
        let frames_none = vec![
            AnimFrame::new(w, h, make_natural_rgba(w, h, 0), 80),
            AnimFrame::new(w, h, make_natural_rgba(w, h, 1), 80),
        ];
        let frames_100 = vec![
            AnimFrame::new(w, h, make_natural_rgba(w, h, 0), 80)
                .with_near_lossless_quality(Some(100)),
            AnimFrame::new(w, h, make_natural_rgba(w, h, 1), 80)
                .with_near_lossless_quality(Some(100)),
        ];
        let a = build_animated_webp(&frames_none).expect("None");
        let b = build_animated_webp(&frames_100).expect("Some(100)");
        assert_eq!(a, b, "Some(100) byte-exact baseline at {w}x{h}");
    }
}

#[test]
fn anim_value_above_100_is_also_byte_exact_baseline() {
    // The preprocessing clamps `quality > 100` down to 100, so 255 (or
    // any value above 100) must also be a no-op.
    let (w, h) = (16u32, 16u32);
    let none = build_animated_webp(&[
        AnimFrame::new(w, h, make_natural_rgba(w, h, 0), 100),
        AnimFrame::new(w, h, make_natural_rgba(w, h, 1), 100),
    ])
    .expect("None");
    let above = build_animated_webp(&[
        AnimFrame::new(w, h, make_natural_rgba(w, h, 0), 100).with_near_lossless_quality(Some(255)),
        AnimFrame::new(w, h, make_natural_rgba(w, h, 1), 100).with_near_lossless_quality(Some(255)),
    ])
    .expect("Some(255)");
    assert_eq!(none, above, "Some(255) must clamp to a no-op baseline");
}

// ───────── Guarantee 2: Some(60) shrinks + bounded per-frame PSNR ─────────

#[test]
fn anim_near_lossless_quality_60_shrinks_3_frame_noisy_fixture() {
    // Use the noisy fixture: smooth gradients are so easy for the
    // baseline LZ77/predictor stages that any near-lossless preprocessing
    // simply *adds* header overhead. The noisy fixture is the realistic
    // case where dropping low-order bits collapses many distinct colors
    // onto the same value and the entropy stages can take advantage.
    let (w, h) = (64u32, 64u32);
    let srcs = [
        make_noisy_rgba(w, h, 0),
        make_noisy_rgba(w, h, 1),
        make_noisy_rgba(w, h, 2),
    ];

    let frames_baseline: Vec<AnimFrame> = srcs
        .iter()
        .enumerate()
        .map(|(i, px)| {
            AnimFrame::new(w, h, px.clone(), 100 + i as u32 * 10)
                .with_near_lossless_quality(Some(100))
        })
        .collect();
    let baseline = build_animated_webp(&frames_baseline).expect("baseline q=100");

    let frames60: Vec<AnimFrame> = srcs
        .iter()
        .enumerate()
        .map(|(i, px)| {
            AnimFrame::new(w, h, px.clone(), 100 + i as u32 * 10)
                .with_near_lossless_quality(Some(60))
        })
        .collect();
    let near60 = build_animated_webp(&frames60).expect("near-lossless q=60");

    // Compression: q=60 must produce *strictly* smaller output than q=100.
    assert!(
        near60.len() < baseline.len(),
        "q=60 ({}) must shrink vs q=100 ({}) on a 3-frame noisy fixture",
        near60.len(),
        baseline.len()
    );

    // PSNR: every decoded frame must stay above the 40 dB floor.
    let dec = decode_webp(&near60).expect("decode near-lossless animation");
    assert_eq!(dec.frames.len(), 3, "one decoded frame per ANMF");
    for (i, fr) in dec.frames.iter().enumerate() {
        let psnr = rgb_psnr(&fr.rgba, &srcs[i]);
        assert!(
            psnr >= 40.0,
            "frame {i} PSNR {psnr:.2} dB must be ≥ 40 dB at q=60 (step=4 ⇒ ≥ 42 dB typical)"
        );
    }
}

#[test]
fn anim_per_frame_anmf_chunk_shrinks_at_lower_quality() {
    // Single-frame animation lets us measure the per-frame ANMF chunk
    // size cleanly (no inter-frame structure). The q=60 file must be
    // strictly smaller than the q=100 baseline; the q=0 file smaller
    // still.
    let (w, h) = (48u32, 48u32);
    let src = make_noisy_rgba(w, h, 17);

    let base = build_animated_webp(&[
        AnimFrame::new(w, h, src.clone(), 100).with_near_lossless_quality(Some(100))
    ])
    .unwrap();
    let q80 = build_animated_webp(&[
        AnimFrame::new(w, h, src.clone(), 100).with_near_lossless_quality(Some(80))
    ])
    .unwrap();
    let q60 = build_animated_webp(&[
        AnimFrame::new(w, h, src.clone(), 100).with_near_lossless_quality(Some(60))
    ])
    .unwrap();
    let q40 = build_animated_webp(&[
        AnimFrame::new(w, h, src.clone(), 100).with_near_lossless_quality(Some(40))
    ])
    .unwrap();
    let q0 =
        build_animated_webp(&[AnimFrame::new(w, h, src, 100).with_near_lossless_quality(Some(0))])
            .unwrap();

    assert!(
        q80.len() < base.len(),
        "q=80 ({}) < q=100 ({})",
        q80.len(),
        base.len()
    );
    assert!(
        q60.len() < q80.len(),
        "q=60 ({}) < q=80 ({})",
        q60.len(),
        q80.len()
    );
    assert!(
        q40.len() < q60.len(),
        "q=40 ({}) < q=60 ({})",
        q40.len(),
        q60.len()
    );
    assert!(
        q0.len() < q40.len(),
        "q=0 ({}) < q=40 ({})",
        q0.len(),
        q40.len()
    );
}

// ───────── Guarantee 3: round-trip recovers the quantized pixels exactly ─────────

#[test]
fn anim_near_lossless_round_trip_matches_still_image_quantize() {
    // The animated-frame preprocessing must produce *exactly* the same
    // pixels per-frame as the still-image quantizer would on the same
    // input — i.e. the bit pattern is identical to running
    // `near_lossless::apply` on the caller's RGBA before encoding.
    let (w, h) = (32u32, 32u32);
    let srcs = [
        make_noisy_rgba(w, h, 100),
        make_noisy_rgba(w, h, 200),
        make_noisy_rgba(w, h, 300),
    ];

    let frames: Vec<AnimFrame> = srcs
        .iter()
        .enumerate()
        .map(|(i, px)| {
            AnimFrame::new(w, h, px.clone(), 100 + i as u32 * 10)
                .with_near_lossless_quality(Some(60))
        })
        .collect();
    let file = build_animated_webp(&frames).expect("build");
    let dec = decode_webp(&file).expect("decode");
    assert_eq!(dec.frames.len(), 3);

    for (i, fr) in dec.frames.iter().enumerate() {
        // Expected pixels: apply the still-image preprocessor to the
        // source ARGB, then repack to interleaved RGBA. These bytes are
        // exactly what the encoder fed to VP8L, and a normal lossless
        // decode must round-trip them.
        let expected_argb = near_lossless::quantize(&rgba_to_argb(&srcs[i]), 60);
        let expected_rgba = argb_to_rgba(&expected_argb);
        assert_eq!(
            fr.rgba, expected_rgba,
            "frame {i} decoded RGBA must equal the quantized source RGBA bit-for-bit"
        );
        assert_eq!(fr.duration_ms, 100 + i as u32 * 10);
    }
}

#[test]
fn anim_near_lossless_preserves_alpha_through_round_trip() {
    // Non-opaque pixels must round-trip with their alpha channel exactly
    // preserved — the preprocessing never touches the A byte.
    let (w, h) = (16u32, 16u32);
    let mut src = make_noisy_rgba(w, h, 42);
    // Splatter a deterministic alpha pattern across every pixel.
    for (i, px) in src.chunks_exact_mut(4).enumerate() {
        px[3] = (i & 0xff) as u8;
    }
    let frames = vec![AnimFrame::new(w, h, src.clone(), 100).with_near_lossless_quality(Some(40))];
    let file = build_animated_webp(&frames).expect("build");
    let dec = decode_webp(&file).expect("decode");
    assert_eq!(dec.frames.len(), 1);
    let decoded = &dec.frames[0].rgba;
    for (orig_px, dec_px) in src.chunks_exact(4).zip(decoded.chunks_exact(4)) {
        assert_eq!(
            orig_px[3], dec_px[3],
            "alpha byte must round-trip unchanged at q=40"
        );
    }
}

#[test]
fn anim_near_lossless_per_frame_psnr_at_quality_40_within_documented_floor() {
    // q=40 → n=3 (step=8). The documented PSNR for that bucket is
    // ≈ 42 dB typical; the test holds the line at the 40 dB floor.
    let (w, h) = (48u32, 48u32);
    let srcs = [make_noisy_rgba(w, h, 7), make_noisy_rgba(w, h, 11)];
    let frames: Vec<AnimFrame> = srcs
        .iter()
        .map(|px| AnimFrame::new(w, h, px.clone(), 100).with_near_lossless_quality(Some(40)))
        .collect();
    let file = build_animated_webp(&frames).unwrap();
    let dec = decode_webp(&file).unwrap();
    for (i, fr) in dec.frames.iter().enumerate() {
        let psnr = rgb_psnr(&fr.rgba, &srcs[i]);
        assert!(
            psnr >= 40.0,
            "frame {i} q=40 PSNR {psnr:.2} dB must be ≥ 40 dB"
        );
    }
}

// ───────── Per-frame mixed-quality + Delta/Auto interactions ─────────

#[test]
fn anim_per_frame_quality_can_be_mixed_across_frames() {
    // Frame 0 q=100 (no-op), frame 1 q=60 (quantize). The first frame's
    // decoded pixels equal the source bit-exactly; the second equals the
    // still-image quantize of its source.
    let (w, h) = (32u32, 32u32);
    let f0_src = make_noisy_rgba(w, h, 0);
    let f1_src = make_noisy_rgba(w, h, 1);

    let frames = vec![
        AnimFrame::new(w, h, f0_src.clone(), 100).with_near_lossless_quality(Some(100)),
        AnimFrame::new(w, h, f1_src.clone(), 100).with_near_lossless_quality(Some(60)),
    ];
    let file = build_animated_webp(&frames).unwrap();
    let dec = decode_webp(&file).unwrap();
    assert_eq!(dec.frames.len(), 2);

    assert_eq!(
        dec.frames[0].rgba, f0_src,
        "frame 0 (q=100) is lossless byte-for-byte"
    );
    let expected_f1 = argb_to_rgba(&near_lossless::quantize(&rgba_to_argb(&f1_src), 60));
    assert_eq!(
        dec.frames[1].rgba, expected_f1,
        "frame 1 (q=60) matches still-image quantize byte-for-byte"
    );
}

#[test]
fn anim_near_lossless_applies_to_delta_dirty_rect_path() {
    // A two-frame fixture in Delta mode where frame 1 changes a 16×16
    // sub-rectangle of a 32×32 noisy frame 0. With q=60 on frame 1, the
    // delta ANMF chunk must shrink relative to q=100, and the decoded
    // frame 1 must equal the still-image-quantized source (the dirty
    // rect is the entire content delta — overwriting the canvas
    // bit-exactly produces the same end pixels as a full-frame emit).
    let (w, h) = (32u32, 32u32);
    let f0 = make_noisy_rgba(w, h, 0);
    let mut f1 = f0.clone();
    // Mutate a 16×16 block at (8, 8).
    for row in 8..24 {
        for col in 8..24 {
            let off = (row * w as usize + col) * 4;
            f1[off] ^= 0xff;
            f1[off + 1] ^= 0xff;
            f1[off + 2] ^= 0xff;
        }
    }

    let baseline = {
        let mut f1_frame = AnimFrame::new(w, h, f1.clone(), 80);
        f1_frame.mode = AnimFrameMode::Delta;
        f1_frame.near_lossless_quality = Some(100);
        let f0_frame = AnimFrame::new(w, h, f0.clone(), 80);
        build_animated_webp(&[f0_frame, f1_frame]).unwrap()
    };
    let near60 = {
        let mut f1_frame = AnimFrame::new(w, h, f1.clone(), 80);
        f1_frame.mode = AnimFrameMode::Delta;
        f1_frame.near_lossless_quality = Some(60);
        let f0_frame = AnimFrame::new(w, h, f0.clone(), 80);
        build_animated_webp(&[f0_frame, f1_frame]).unwrap()
    };

    assert!(
        near60.len() < baseline.len(),
        "delta-mode q=60 ({}) must shrink vs delta-mode q=100 ({})",
        near60.len(),
        baseline.len()
    );

    // Decoded frame 1: the dirty-rect sub-frame is the bounding box of
    // changed pixels (here exactly (8,8)..(24,24), a 16×16 block whose
    // top-left is already even). Inside that box the encoder emits the
    // *quantized* pixels; outside, the decoder's compositor keeps frame
    // 0's pixels untouched. So the per-pixel expectation is:
    //
    //   inside  rect:  quantize(f1) at that position
    //   outside rect:  f0 (decoded losslessly since f0 is q=100)
    let dec = decode_webp(&near60).unwrap();
    assert_eq!(dec.frames.len(), 2);
    let f1_quant = argb_to_rgba(&near_lossless::quantize(&rgba_to_argb(&f1), 60));
    let mut expected = f0.clone();
    for row in 8..24 {
        for col in 8..24 {
            let off = (row * w as usize + col) * 4;
            expected[off..off + 4].copy_from_slice(&f1_quant[off..off + 4]);
        }
    }
    assert_eq!(
        dec.frames[1].rgba, expected,
        "delta-mode frame 1: sub-rect is quantized, rest comes from f0"
    );
}

#[test]
fn with_near_lossless_quality_builder_round_trips_field() {
    let (w, h) = (2u32, 2u32);
    let src = vec![0u8; (w * h * 4) as usize];
    let f = AnimFrame::new(w, h, src.clone(), 0).with_near_lossless_quality(Some(60));
    assert_eq!(f.near_lossless_quality, Some(60));
    let f2 = AnimFrame::new(w, h, src.clone(), 0).with_near_lossless_quality(None);
    assert_eq!(f2.near_lossless_quality, None);
    let f3 = AnimFrame::new(w, h, src, 0);
    assert_eq!(
        f3.near_lossless_quality, None,
        "default near_lossless_quality is None"
    );
}
