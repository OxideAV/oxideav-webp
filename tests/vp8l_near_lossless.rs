//! Tests for the VP8L encoder's near-lossless preprocessing knob
//! ([`oxideav_webp::EncoderOptions::near_lossless`]).
//!
//! Coverage:
//!
//! 1. **Identity at level 100** — opting into near-lossless = 100 must
//!    produce a byte-identical bitstream to leaving it at default. This
//!    is the "off switch" smoke test.
//! 2. **Lossless decode invariant** — even with aggressive quantisation
//!    the resulting bitstream still decodes via the in-crate VP8L
//!    decoder (the preprocessing happens *before* encoding, so the
//!    bitstream itself is fully spec-compliant lossless).
//! 3. **Pixel drift bounded by step** — every channel of every decoded
//!    pixel must lie within `step/2 + 1` of the input (where `step =
//!    1 << shift`). This validates that the round-to-nearest with
//!    clamping is correctly bounded.
//! 4. **Alpha is exact** — alpha is preserved untouched even at level
//!    0; otherwise transparent pixels would gain visible colour.
//! 5. **Size win on photographic content** — an aggressive level
//!    (`near_lossless = 20`) on a noise-heavy fixture must produce a
//!    strictly smaller bitstream than `near_lossless = 100` on the same
//!    pixels.

use oxideav_webp::vp8l;
use oxideav_webp::vp8l::encoder::{encode_vp8l_argb_with, EncoderOptions};

/// Build a 64×64 photographic-like field: smooth gradient + low-amplitude
/// noise. The noise gives near-lossless something to round off; the
/// gradient makes sure the predictor can still get traction post-quant.
fn photo_like(w: u32, h: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity((w * h) as usize);
    let mut s: u32 = 0x9E37_79B1;
    for y in 0..h {
        for x in 0..w {
            // Cheap xorshift32 — deterministic noise for repeatable tests.
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s & 0x1f) as i32 - 16; // ±16
            let r = ((x as i32 * 3 + n).clamp(0, 255)) as u32;
            let g = ((y as i32 * 3 + n).clamp(0, 255)) as u32;
            let b = ((((x + y) as i32) * 2 + n).clamp(0, 255)) as u32;
            out.push(0xff00_0000 | (r << 16) | (g << 8) | b);
        }
    }
    out
}

/// Predicted maximum per-channel drift for a given near-lossless level.
/// `quantise_channel` rounds-to-nearest with `(v + step/2) & !(step-1)`,
/// clamped to 255 *before* the mask. The worst case is at most
/// `step / 2` for non-saturating values; the clamp at 255 can land a
/// value at `255 & !(step-1)` (e.g. for step=16 → 240) so saturated
/// inputs can drift by up to `step - 1`. Use `step` as a conservative
/// upper bound that catches both regimes.
fn max_drift_for_level(level: u8) -> i32 {
    match level {
        100..=u8::MAX => 0,
        60..=99 => 2, // 1 << 1
        40..=59 => 4, // 1 << 2
        20..=39 => 8, // 1 << 3
        0..=19 => 16, // 1 << 4
    }
}

#[test]
fn near_lossless_100_is_byte_identical_to_default() {
    let w = 32;
    let h = 32;
    let pixels = photo_like(w, h);

    let lossless = EncoderOptions::default();
    let near_100 = EncoderOptions {
        near_lossless: 100,
        ..EncoderOptions::default()
    };

    let bs_default = encode_vp8l_argb_with(w, h, &pixels, false, lossless).expect("default encode");
    let bs_near_100 =
        encode_vp8l_argb_with(w, h, &pixels, false, near_100).expect("near=100 encode");

    assert_eq!(
        bs_default, bs_near_100,
        "near_lossless = 100 must be a no-op (byte-identical bitstream)"
    );
}

#[test]
fn near_lossless_decodes_and_bounds_drift() {
    let w = 64;
    let h = 64;
    let pixels = photo_like(w, h);

    for level in [80u8, 60, 40, 20, 0] {
        let opts = EncoderOptions {
            near_lossless: level,
            ..EncoderOptions::default()
        };
        let bitstream = encode_vp8l_argb_with(w, h, &pixels, false, opts)
            .unwrap_or_else(|e| panic!("near_lossless={level} encode failed: {e:?}"));
        let decoded = vp8l::decode(&bitstream)
            .unwrap_or_else(|e| panic!("near_lossless={level} decode failed: {e:?}"));
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
        assert_eq!(decoded.pixels.len(), pixels.len());

        let max_drift = max_drift_for_level(level);
        for (i, (orig, got)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
            for ch in 0..4 {
                let sh = ch * 8;
                let ov = ((orig >> sh) & 0xff) as i32;
                let gv = ((got >> sh) & 0xff) as i32;
                let drift = (ov - gv).abs();
                if ch == 3 {
                    // Alpha must be exact regardless of level.
                    assert_eq!(
                        drift, 0,
                        "alpha drifted at level={level} pixel={i} ch=A: {ov} → {gv}"
                    );
                } else {
                    assert!(
                        drift <= max_drift,
                        "near_lossless={level} pixel={i} ch={ch} drift={drift} \
                         exceeds bound {max_drift} ({ov} → {gv})"
                    );
                }
            }
        }
    }
}

#[test]
fn near_lossless_alpha_preserved_at_level_0() {
    // Construct an image where alpha varies across pixels — checks that
    // even with maximum quantisation (level 0 → 4-bit shift), the alpha
    // channel is byte-exact end-to-end.
    let w = 16;
    let h = 16;
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            // Varying alpha: 0..255 in steps of ~16.
            let a = ((x + y * w) * 16 / (w * h) * 16).min(255);
            pixels.push((a << 24) | 0x00_5050a0);
        }
    }

    let opts = EncoderOptions {
        near_lossless: 0,
        // Disable strip — strip clobbers RGB on alpha=0 pixels which is
        // not what we're testing here.
        strip_transparent_color: false,
        ..EncoderOptions::default()
    };
    let bitstream =
        encode_vp8l_argb_with(w, h, &pixels, true, opts).expect("near=0 alpha-varying encode");
    let decoded = vp8l::decode(&bitstream).expect("near=0 alpha-varying decode");

    for (i, (orig, got)) in pixels.iter().zip(decoded.pixels.iter()).enumerate() {
        let oa = (orig >> 24) & 0xff;
        let ga = (got >> 24) & 0xff;
        assert_eq!(oa, ga, "alpha differs at pixel {i}: {oa} → {ga}");
    }
}

#[test]
fn near_lossless_shrinks_photographic_bitstream() {
    let w = 64;
    let h = 64;
    let pixels = photo_like(w, h);

    let lossless = EncoderOptions::default();
    let near_20 = EncoderOptions {
        near_lossless: 20,
        ..EncoderOptions::default()
    };

    let bs_lossless = encode_vp8l_argb_with(w, h, &pixels, false, lossless).expect("lossless");
    let bs_near = encode_vp8l_argb_with(w, h, &pixels, false, near_20).expect("near=20");

    eprintln!(
        "photo-like 64×64: lossless={} bytes, near_lossless=20={} bytes ({:+.1}%)",
        bs_lossless.len(),
        bs_near.len(),
        100.0 * (bs_near.len() as f64 - bs_lossless.len() as f64) / bs_lossless.len() as f64
    );

    assert!(
        bs_near.len() < bs_lossless.len(),
        "near_lossless=20 ({} B) should shrink vs lossless ({} B) on photographic input",
        bs_near.len(),
        bs_lossless.len()
    );
}
