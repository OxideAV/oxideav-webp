//! Round-139 integration tests for the **near-lossless** encoder
//! preprocessing surface:
//!
//! * [`oxideav_webp::encode_vp8l_argb_with_near_lossless`] — the public
//!   entry point that runs the `near_lossless::apply` preprocessing pass
//!   in front of the VP8L encoder.
//! * [`oxideav_webp::near_lossless`] — the standalone preprocessor + the
//!   `quality → bits-to-drop` mapping.
//!
//! Three round-139 guarantees are covered end-to-end:
//!
//! 1. **`quality = 100` is byte-exact-equal to the baseline encoder** for
//!    every fixture we test. This is the no-op safety net: callers can
//!    pass the near-lossless variant unconditionally and only opt into
//!    quantization by lowering the knob.
//! 2. **`quality = 60` produces a smaller bitstream than `quality = 100`**
//!    on a natural-image fixture, with bounded PSNR loss (≥ 40 dB). This
//!    demonstrates the compression win the preprocessing exists to
//!    produce.
//! 3. **Decoder round-trip recovers the quantized ARGB exactly.** The
//!    encoded `.webp` is a perfectly normal lossless chunk — the
//!    preprocessing is an encoder choice, not a bitstream change — so
//!    `decode_webp` returns the same pixels [`near_lossless::apply`]
//!    produced.
//!
//! All tests run under `--no-default-features` (no `registry` feature
//! required), matching the rest of `tests/published_encode_api.rs`.

use oxideav_webp::{
    decode_webp, encode_vp8l_argb_with, encode_vp8l_argb_with_near_lossless, near_lossless,
};

/// Build a `width * height` natural-looking ARGB image: a smooth 2-D
/// gradient with a small per-pixel deterministic perturbation. The
/// perturbation prevents the gradient from collapsing into a handful of
/// identical neighbours (which would let the baseline lossless encoder
/// trivially capture everything via LZ77 + color-cache and leave no
/// headroom for the near-lossless preprocessing to win on).
///
/// The output is opaque ARGB (`alpha = 0xff`) in scan-line order.
fn make_natural_argb(width: u32, height: u32) -> Vec<u32> {
    let mut buf = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            // Smooth gradient base.
            let base_r = (x * 255 / width.max(1)) as u8;
            let base_g = (y * 255 / height.max(1)) as u8;
            let base_b = ((x + y) * 255 / (width + height).max(1)) as u8;
            // Small deterministic perturbation, channel-decorrelated.
            let n = (x
                .wrapping_mul(2654435761)
                .wrapping_add(y.wrapping_mul(40503)))
                & 0x07;
            let r = base_r.saturating_add(n as u8);
            let g = base_g.saturating_add((n ^ 0x3) as u8);
            let b = base_b.saturating_add((n ^ 0x5) as u8);
            buf.push(0xff00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
        }
    }
    buf
}

/// Build a `width * height` ARGB image with high per-pixel entropy — a
/// noisy field driven by a deterministic xorshift sequence — as the
/// natural-image analogue. The baseline lossless encoder cannot easily
/// LZ77-collapse this (no exact-pixel repeats), so dropping low-order
/// bits via near-lossless preprocessing creates the repeats the entropy
/// stages need to win.
///
/// Opaque (`alpha = 0xff`) in scan-line order. The PRNG is seeded
/// per-call so the fixture is reproducible.
fn make_noisy_argb(width: u32, height: u32) -> Vec<u32> {
    let mut buf = Vec::with_capacity((width * height) as usize);
    let mut s: u32 = 0x9e37_79b9;
    for _ in 0..(width * height) {
        // xorshift32
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let r = s & 0xff;
        let g = (s >> 8) & 0xff;
        let b = (s >> 16) & 0xff;
        buf.push(0xff00_0000 | (r << 16) | (g << 8) | b);
    }
    buf
}

/// Repack ARGB → interleaved RGBA bytes for comparison against a decoded
/// frame's pixel buffer.
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

/// PSNR (peak signal-to-noise ratio) in dB between two equal-length ARGB
/// pixel buffers, computed over the R/G/B channels only (alpha excluded —
/// the preprocessing preserves it bit-exactly, so it would just inflate
/// the score with zero error).
///
/// Returns `f64::INFINITY` when the two buffers are identical (MSE = 0).
fn rgb_psnr(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr buffers differ in length");
    let mut sse: u64 = 0;
    let mut n: u64 = 0;
    for (&pa, &pb) in a.iter().zip(b.iter()) {
        for shift in [0u32, 8, 16] {
            let ca = ((pa >> shift) & 0xff) as i32;
            let cb = ((pb >> shift) & 0xff) as i32;
            let d = ca - cb;
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

// ───────── Guarantee 1: quality=100 is byte-exact-equal to baseline ─────────

#[test]
fn near_lossless_quality_100_is_byte_exact_baseline_on_natural_fixture() {
    let (w, h) = (64u32, 64u32);
    let argb = make_natural_argb(w, h);

    let baseline = encode_vp8l_argb_with(&argb, w, h, false).expect("baseline encode");
    let near = encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 100)
        .expect("near-lossless q=100 encode");

    assert_eq!(
        baseline, near,
        "quality=100 must produce byte-exact-identical output to baseline encoder"
    );
}

#[test]
fn near_lossless_quality_100_is_byte_exact_baseline_on_small_fixtures() {
    // Cover several geometries to make sure the no-op fast path holds
    // across the encoder's whole input shape range: degenerate (1x1),
    // non-power-of-two, and a small square that exercises every encode
    // path candidate.
    for (w, h) in [(1u32, 1u32), (3, 5), (7, 4), (16, 16)] {
        let argb = make_natural_argb(w, h);
        let baseline = encode_vp8l_argb_with(&argb, w, h, false).expect("baseline");
        let near = encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 100)
            .expect("near-lossless q=100");
        assert_eq!(baseline, near, "byte-exact mismatch for {w}x{h}");
    }
}

#[test]
fn near_lossless_quality_above_100_is_also_byte_exact_baseline() {
    let (w, h) = (16u32, 16u32);
    let argb = make_natural_argb(w, h);
    let baseline = encode_vp8l_argb_with(&argb, w, h, false).expect("baseline");
    let near = encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 255)
        .expect("near-lossless q=255 (clamped to 100)");
    assert_eq!(baseline, near, "values above 100 must clamp to no-op");
}

// ───────── Guarantee 2: quality=60 shrinks + bounded PSNR ─────────

#[test]
fn near_lossless_quality_60_shrinks_noisy_image_with_bounded_psnr() {
    // Use the noisy-natural fixture: smooth-gradient input is so easy for
    // the baseline LZ77 + predictor stages that any near-lossless
    // preprocessing simply *adds* header overhead without unlocking new
    // repeats. The noisy fixture is the realistic case where dropping
    // low-order bits collapses many distinct colors onto the same value
    // and the entropy stages can take advantage.
    let (w, h) = (96u32, 96u32);
    let argb = make_noisy_argb(w, h);

    let baseline =
        encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 100).expect("baseline (q=100)");
    let near60 =
        encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 60).expect("near-lossless q=60");

    // Compression: q=60 must produce *strictly* smaller output than q=100
    // on a noisy natural-image fixture. The preprocessing exists
    // precisely to make this true.
    assert!(
        near60.len() < baseline.len(),
        "q=60 output ({}) must be smaller than q=100 ({}) on noisy image",
        near60.len(),
        baseline.len()
    );

    // PSNR: the quantized pixels (what a decoder actually recovers) must
    // stay above 40 dB vs the original input. q=60 is n=2 (step=4); the
    // worst-case ±2 per-channel error puts the floor at ~42 dB by the
    // formula 20·log10(255 / (step/2)). Our deterministic fixture lands
    // well above that.
    let quantized = near_lossless::quantize(&argb, 60);
    let psnr = rgb_psnr(&argb, &quantized);
    assert!(
        psnr >= 40.0,
        "PSNR floor violated at q=60: {psnr:.2} dB (need ≥ 40 dB)"
    );
}

#[test]
fn near_lossless_quality_lowers_psnr_monotonically_in_expectation() {
    // The PSNR vs the original should never *improve* as quality drops
    // (more bits get dropped at lower quality). Equality is allowed for
    // adjacent buckets that map to the same n value.
    let (w, h) = (32u32, 32u32);
    let argb = make_natural_argb(w, h);

    // Sample one representative quality per documented bucket boundary.
    let qualities = [100u8, 99, 80, 79, 60, 59, 40, 39, 20, 19, 0];
    let mut last_psnr = f64::INFINITY;
    for &q in &qualities {
        let quantized = near_lossless::quantize(&argb, q);
        let psnr = rgb_psnr(&argb, &quantized);
        assert!(
            psnr <= last_psnr + 1e-9,
            "PSNR went up as quality fell: q={q} psnr={psnr:.4} prev={last_psnr:.4}"
        );
        last_psnr = psnr;
    }
}

// ───────── Guarantee 3: decoder roundtrip recovers quantized ARGB ─────────

#[test]
fn near_lossless_quality_60_roundtrip_recovers_quantized_pixels() {
    // The encoded chunk is a perfectly normal VP8L bitstream — the
    // preprocessing is encoder-side only — so the decoder must recover
    // the *quantized* ARGB (not the original) bit-exactly.
    let (w, h) = (32u32, 32u32);
    let argb = make_natural_argb(w, h);

    // Build a complete .webp around the bare bitstream so decode_webp
    // (which walks RIFF) can read it. Wrap manually so this test stays
    // a pure exercise of the near-lossless path without depending on
    // encode_vp8l_argb_with_metadata's specifics.
    use oxideav_webp::build::{build_webp_file, ImageKind};
    let bare = encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 60)
        .expect("near-lossless q=60 bare bitstream");
    let file = build_webp_file(&bare, ImageKind::Lossless, w, h).expect("RIFF wrap");

    let img = decode_webp(&file).expect("decode .webp");
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].width, w);
    assert_eq!(img.frames[0].height, h);

    // The decoder must reproduce the quantized pixels, *not* the
    // original. Compare against `near_lossless::quantize(&argb, 60)`.
    let quantized = near_lossless::quantize(&argb, 60);
    let expected_rgba = argb_to_rgba(&quantized);
    assert_eq!(
        img.frames[0].rgba, expected_rgba,
        "decoder round trip must recover quantized ARGB exactly"
    );
}

#[test]
fn near_lossless_quality_0_max_quantization_still_round_trips() {
    // Even at the maximum quantization (n=5, step=32) the bitstream is
    // still a perfectly normal VP8L chunk and round-trips bit-exact.
    let (w, h) = (16u32, 16u32);
    let argb = make_natural_argb(w, h);

    use oxideav_webp::build::{build_webp_file, ImageKind};
    let bare = encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 0)
        .expect("near-lossless q=0 bare bitstream");
    let file = build_webp_file(&bare, ImageKind::Lossless, w, h).expect("RIFF wrap");

    let img = decode_webp(&file).expect("decode");
    let quantized = near_lossless::quantize(&argb, 0);
    let expected_rgba = argb_to_rgba(&quantized);
    assert_eq!(img.frames[0].rgba, expected_rgba);
}

#[test]
fn near_lossless_preserves_alpha_through_encode_decode() {
    // Build a non-opaque image; alpha must round-trip the *full* encode +
    // decode cycle bit-exactly because near-lossless does not quantize
    // alpha.
    let (w, h) = (8u32, 8u32);
    let mut argb = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let a = (x * 32 + y * 4) as u8;
            let r = (x * 30) as u8;
            let g = (y * 30) as u8;
            let b = ((x + y) * 15) as u8;
            argb.push(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
        }
    }

    use oxideav_webp::build::{build_webp_file, ImageKind};
    let bare =
        encode_vp8l_argb_with_near_lossless(&argb, w, h, true, 40).expect("alpha + near-lossless");
    let file = build_webp_file(&bare, ImageKind::Lossless, w, h).expect("RIFF wrap");
    let img = decode_webp(&file).expect("decode");

    // Each decoded pixel's alpha byte (index 3 of every RGBA quad) must
    // equal the input alpha byte — the quantization touches only R/G/B.
    for (i, px) in argb.iter().enumerate() {
        let want_a = (px >> 24) as u8;
        let got_a = img.frames[0].rgba[i * 4 + 3];
        assert_eq!(got_a, want_a, "alpha changed at pixel {i}");
    }
}

#[test]
fn near_lossless_dimension_mismatch_is_published_error() {
    // The wrapper validates dimensions the same way the underlying encoder
    // does — a 1-pixel buffer claimed as 2x2 is rejected, *before* the
    // quantization pass runs.
    let argb = vec![0xff00_0000u32];
    let err =
        encode_vp8l_argb_with_near_lossless(&argb, 2, 2, false, 60).expect_err("dim mismatch");
    assert_eq!(err, oxideav_webp::WebpError::InvalidData);
}

// ───────── Measurement helper (prints sizes/PSNR under --nocapture) ─────────

#[test]
fn near_lossless_measurement_table() {
    // Not a strict assert beyond "every entry encodes successfully" — exists
    // so the round can capture a numeric table by running
    // `cargo test --test published_near_lossless_api -- --nocapture
    //  near_lossless_measurement_table`. The size/PSNR numbers in the
    // round-139 README + CHANGELOG come from this.
    let (w, h) = (96u32, 96u32);
    let argb = make_noisy_argb(w, h);
    println!("\nnoisy {w}x{h} ARGB measurement table:");
    println!("  quality  bytes    delta   PSNR(dB)");
    let baseline = encode_vp8l_argb_with_near_lossless(&argb, w, h, false, 100)
        .expect("baseline")
        .len();
    for q in [100u8, 95, 80, 60, 40, 20, 0] {
        let bytes = encode_vp8l_argb_with_near_lossless(&argb, w, h, false, q)
            .expect("encode")
            .len();
        let delta = (bytes as i64) - (baseline as i64);
        let quantized = near_lossless::quantize(&argb, q);
        let psnr = rgb_psnr(&argb, &quantized);
        let psnr_s = if psnr.is_finite() {
            format!("{:7.2}", psnr)
        } else {
            "    inf".to_string()
        };
        println!("  {q:3}      {bytes:6}  {delta:+6}   {psnr_s}");
    }
}
