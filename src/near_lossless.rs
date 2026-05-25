//! VP8L (WebP-Lossless) **near-lossless** encoder preprocessing.
//!
//! Near-lossless is an *encoder-side* preprocessing step: the input ARGB
//! pixels are quantized to a coarser color precision **before** they are
//! handed to the existing VP8L pipeline ([`crate::vp8l_encode`]). The
//! bitstream itself is unchanged — a near-lossless `.webp` decodes through
//! the exact same §3.4 + §3.8 lossless path, producing the (quantized)
//! ARGB values bit-exactly. The compression win comes from the §5.2.2
//! LZ77 / §5.2.3 color-cache stages, which see fewer distinct colors and
//! longer repeats once low-order bits have been zeroed.
//!
//! ## Why this lives in the encoder, not the spec
//!
//! RFC 9649 standardises only the *bitstream*; the only mention of
//! near-lossless in the WebP container/lossless specifications is the file
//! it produces being a perfectly normal `VP8L` chunk. The specific
//! quantization formula — how many low bits to drop per channel for a
//! given quality level, the rounding direction, whether neighbouring
//! pixels influence the decision — is left entirely to the encoder. This
//! module therefore documents the chosen formula explicitly so that
//! reproducibility is a property of *our* encoder, not a portable
//! cross-encoder guarantee.
//!
//! ## Quality knob and quantization step
//!
//! The convention matches the cwebp `-near_lossless` flag: a `u8` in
//! `[0..=100]` where **100 = lossless (no-op)** and **0 = maximum
//! quantization**.
//!
//! The encoder maps `quality` to a *bits-to-drop* count
//! `n ∈ [0..=5]` per the table below; that becomes the precision of each
//! RGB channel (8 − n bits). Alpha is **never** quantized — its bit
//! pattern is preserved exactly.
//!
//! | quality range | `n` (bits dropped) | retained precision | step |
//! |---------------|--------------------|--------------------|------|
//! | 100           | 0                  | 8 bits             |  1   |
//! | 80..=99       | 1                  | 7 bits             |  2   |
//! | 60..=79       | 2                  | 6 bits             |  4   |
//! | 40..=59       | 3                  | 5 bits             |  8   |
//! | 20..=39       | 4                  | 4 bits             | 16   |
//! | 0..=19        | 5                  | 3 bits             | 32   |
//!
//! The mapping is `n = clamp((100 - q + 19) / 20, 0, 5)` — a 20-point
//! window per step, biased so `q = 100` cleanly produces `n = 0` (the
//! no-op identity, byte-exact-equal to the baseline encoder).
//!
//! ## Per-channel rounding
//!
//! Each of R, G, B is quantized with **round-half-up to the nearest step**:
//!
//! ```text
//! step  = 1 << n
//! half  = step >> 1
//! c'    = ((c + half) >> n) << n   // unclamped sum + then clamped to 255
//! ```
//!
//! The clamp prevents `c = 0xff` from rolling over to `0x100` (which would
//! truncate back to `0x00`); concretely, when `c + half >= 256`, the
//! quantized value is forced to the largest multiple of `step` ≤ 255
//! (`255 & !(step - 1)`).
//!
//! Per-channel error is bounded by **`step / 2`** for inputs in
//! `[0, 256 − step/2)` (the round-half-up region) and by **`step − 1`**
//! for inputs in the upper clamp window `[256 − step/2, 255]` (where the
//! rounded-up value would overflow 255 and is forced down to the largest
//! multiple of `step` ≤ 255 instead). Concretely: worst-case error is up
//! to 31 at `n = 5` (q = 0) and 1 at `n = 1` (q ≥ 80). PSNR is dominated
//! by the typical `step/2` case — `20·log10(255 / (step/2))` gives
//! ≈ 60 dB at `n = 1`, ≈ 48 dB at `n = 2`, ≈ 42 dB at `n = 3`,
//! ≈ 36 dB at `n = 4`, ≈ 30 dB at `n = 5` — with the clamp window
//! affecting at most `step/2 − 1` of the 256 byte values per channel.
//!
//! ## Bit-exact decode contract
//!
//! Because [`apply`] mutates the pixels *before* the standard lossless
//! pipeline, the result is a perfectly normal `VP8L` chunk that decodes
//! back to the quantized values bit-exactly through
//! [`crate::vp8l_transform::decode_lossless`]. The encoder offers no
//! cross-channel filtering or context-dependent adjustment, so each pixel
//! is quantized independently — easy to reason about, and trivially
//! parallelisable if a future round wants to chunk the work.

/// Quality value at which [`apply`] is a no-op — every pixel is left
/// untouched, and the subsequent encode is byte-exact-identical to the
/// baseline `encode_vp8l_argb` output.
pub const QUALITY_LOSSLESS: u8 = 100;

/// Default near-lossless quality used by [`crate::encode_vp8l_argb_with_near_lossless`]
/// when callers want a "moderate" preset. Picked to fall in the
/// `60..=79` bucket (`n = 2`, step = 4), trading ≤ ±2 per-channel error
/// for a noticeable reduction in distinct colors on natural images.
pub const DEFAULT_QUALITY: u8 = 60;

/// Largest number of low-order bits the encoder ever drops from a color
/// channel. Reached at `quality = 0` (`n = 5`, step = 32).
pub const MAX_BITS_TO_DROP: u8 = 5;

/// Translate a `[0..=100]` quality knob into the number of low-order bits
/// to drop from each color channel.
///
/// The mapping is the documented `n = clamp((100 - q + 19) / 20, 0, 5)`
/// step table — `q = 100` → 0, `q ∈ 80..=99` → 1, `q ∈ 60..=79` → 2,
/// `q ∈ 40..=59` → 3, `q ∈ 20..=39` → 4, `q ∈ 0..=19` → 5. Values above
/// 100 are clamped down to 100 (still a no-op).
pub fn bits_to_drop_for_quality(quality: u8) -> u8 {
    let q = quality.min(QUALITY_LOSSLESS) as i32;
    let raw = (QUALITY_LOSSLESS as i32 - q + 19) / 20;
    raw.clamp(0, MAX_BITS_TO_DROP as i32) as u8
}

/// Round one 8-bit channel to the nearest multiple of `1 << n`, clamping
/// the rounded-up boundary so 255 cannot roll over to 256.
///
/// `n = 0` is the identity. The result is always a multiple of `1 << n`.
#[inline]
pub fn quantize_channel(c: u8, n: u8) -> u8 {
    if n == 0 {
        return c;
    }
    let step = 1u32 << n;
    let half = step >> 1;
    let sum = c as u32 + half;
    // Force the rolled-over case down to the largest multiple of step ≤ 255.
    let q = if sum >= 256 {
        255 & !(step - 1)
    } else {
        (sum >> n) << n
    };
    q as u8
}

/// Quantize every pixel of `argb` in place per the near-lossless
/// preprocessing described in this module's documentation.
///
/// Each pixel is interpreted as `(a << 24) | (r << 16) | (g << 8) | b`.
/// R, G, B are independently rounded to the nearest multiple of
/// `1 << bits_to_drop_for_quality(quality)`; alpha is left untouched.
/// `quality = 100` (or any value ≥ 100) is a no-op — every pixel is
/// returned unchanged, and a subsequent encode is byte-exact-identical
/// to the baseline `encode_vp8l_argb`.
pub fn apply(argb: &mut [u32], quality: u8) {
    let n = bits_to_drop_for_quality(quality);
    if n == 0 {
        return;
    }
    for px in argb.iter_mut() {
        let a = (*px >> 24) & 0xff;
        let r = quantize_channel(((*px >> 16) & 0xff) as u8, n) as u32;
        let g = quantize_channel(((*px >> 8) & 0xff) as u8, n) as u32;
        let b = quantize_channel((*px & 0xff) as u8, n) as u32;
        *px = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// Out-of-place sibling of [`apply`]: returns a fresh `Vec<u32>` of the
/// quantized pixels, leaving the caller's input untouched. Used when a
/// caller wants to compare the original and quantized buffers (e.g. for
/// PSNR measurement) or when the input is borrowed and cannot be mutated.
pub fn quantize(argb: &[u32], quality: u8) -> Vec<u32> {
    let mut out = argb.to_vec();
    apply(&mut out, quality);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- bits_to_drop mapping ----

    #[test]
    fn quality_100_is_identity_n_zero() {
        assert_eq!(bits_to_drop_for_quality(100), 0);
        assert_eq!(bits_to_drop_for_quality(255), 0); // clamps to 100
    }

    #[test]
    fn quality_buckets_match_table() {
        // 100 → 0
        assert_eq!(bits_to_drop_for_quality(100), 0);
        // 99..=80 → 1
        assert_eq!(bits_to_drop_for_quality(99), 1);
        assert_eq!(bits_to_drop_for_quality(81), 1);
        assert_eq!(bits_to_drop_for_quality(80), 1);
        // 79..=60 → 2
        assert_eq!(bits_to_drop_for_quality(79), 2);
        assert_eq!(bits_to_drop_for_quality(60), 2);
        // 59..=40 → 3
        assert_eq!(bits_to_drop_for_quality(59), 3);
        assert_eq!(bits_to_drop_for_quality(40), 3);
        // 39..=20 → 4
        assert_eq!(bits_to_drop_for_quality(39), 4);
        assert_eq!(bits_to_drop_for_quality(20), 4);
        // 19..=0 → 5
        assert_eq!(bits_to_drop_for_quality(19), 5);
        assert_eq!(bits_to_drop_for_quality(0), 5);
    }

    #[test]
    fn bits_to_drop_is_monotone_nondecreasing_as_quality_drops() {
        let mut prev = 0u8;
        for q in (0..=100).rev() {
            let n = bits_to_drop_for_quality(q);
            assert!(n >= prev, "q={q} n={n} prev={prev}");
            prev = n;
        }
    }

    // ---- quantize_channel ----

    #[test]
    fn quantize_channel_identity_when_n_zero() {
        for c in 0u8..=255 {
            assert_eq!(quantize_channel(c, 0), c);
        }
    }

    #[test]
    fn quantize_channel_rounds_to_nearest_step() {
        // n=1 (step=2): even values stay, odd values round up.
        assert_eq!(quantize_channel(0, 1), 0);
        assert_eq!(quantize_channel(1, 1), 2);
        assert_eq!(quantize_channel(2, 1), 2);
        assert_eq!(quantize_channel(127, 1), 128);
        assert_eq!(quantize_channel(128, 1), 128);
        // n=2 (step=4): 0..3 round to 0/4, etc.
        assert_eq!(quantize_channel(0, 2), 0);
        assert_eq!(quantize_channel(1, 2), 0);
        assert_eq!(quantize_channel(2, 2), 4);
        assert_eq!(quantize_channel(3, 2), 4);
        assert_eq!(quantize_channel(4, 2), 4);
    }

    #[test]
    fn quantize_channel_clamps_at_255() {
        // 255 + half would overflow the byte boundary; we clamp down to
        // the largest multiple of step ≤ 255 instead of rolling to 0.
        // n=1: largest multiple of 2 ≤ 255 is 254.
        assert_eq!(quantize_channel(255, 1), 254);
        // n=2: largest multiple of 4 ≤ 255 is 252.
        assert_eq!(quantize_channel(255, 2), 252);
        assert_eq!(quantize_channel(254, 2), 252);
        // n=5: largest multiple of 32 ≤ 255 is 224.
        assert_eq!(quantize_channel(255, 5), 224);
    }

    #[test]
    fn quantize_channel_max_error_is_bounded_by_step_minus_one() {
        // For every n in 1..=5 and every byte value, the per-channel error
        // is at most `step - 1`. Inside `[0, 255 - step/2 + 1]` the bound
        // is the round-half-up `step/2`; in the upper clamp window
        // `[256 - step/2, 255]` we floor down to the largest multiple of
        // `step` ≤ 255, which can push the error up to `step - 1`
        // (e.g. n=5, c=255, step=32 → q=224, err=31).
        for n in 1..=MAX_BITS_TO_DROP {
            let step = 1u16 << n;
            let max_err = (step - 1) as i32;
            for c in 0u8..=255 {
                let q = quantize_channel(c, n) as i32;
                let err = (q - c as i32).abs();
                assert!(err <= max_err, "n={n} c={c} q={q} err={err}");
            }
        }
    }

    #[test]
    fn quantize_channel_typical_error_is_bounded_by_step_half() {
        // Outside the upper clamp window `(255 - step/2, 255]` every
        // channel value satisfies the tighter round-half-up bound
        // `|q - c| <= step/2`. This is the bound the PSNR floor in the
        // module-level docs relies on for the "typical" case.
        for n in 1..=MAX_BITS_TO_DROP {
            let step = 1u16 << n;
            let half = (step / 2) as i32;
            let upper_clamp_start = 256i32 - half;
            for c in 0u8..=255 {
                let ci = c as i32;
                if ci >= upper_clamp_start {
                    continue;
                }
                let q = quantize_channel(c, n) as i32;
                let err = (q - ci).abs();
                assert!(err <= half, "n={n} c={c} q={q} err={err}");
            }
        }
    }

    #[test]
    fn quantize_channel_result_is_multiple_of_step() {
        for n in 0..=MAX_BITS_TO_DROP {
            let step = 1u32 << n;
            for c in 0u8..=255 {
                let q = quantize_channel(c, n) as u32;
                assert_eq!(q % step, 0, "n={n} c={c} q={q} step={step}");
            }
        }
    }

    // ---- apply / quantize on ARGB pixels ----

    #[test]
    fn apply_quality_100_is_byte_for_byte_identity() {
        let mut pixels = vec![
            0xff_00_00_00, // black opaque
            0x80_aa_bb_cc, // alpha=0x80, RGB=AA BB CC
            0x00_ff_ff_ff, // transparent white
            0x40_01_02_03, // small values
        ];
        let original = pixels.clone();
        apply(&mut pixels, 100);
        assert_eq!(pixels, original);
        // Also out-of-place form.
        let q = quantize(&original, 100);
        assert_eq!(q, original);
    }

    #[test]
    fn apply_quality_above_100_is_also_identity() {
        let mut pixels = vec![0x80_aa_bb_cc, 0xff_00_ff_00];
        let original = pixels.clone();
        apply(&mut pixels, 255);
        assert_eq!(pixels, original);
    }

    #[test]
    fn apply_preserves_alpha_exactly() {
        // Every alpha value must round-trip the apply() call unchanged at
        // every quality level — alpha is intentionally not quantized.
        let mut pixels: Vec<u32> = (0u32..=255).map(|a| (a << 24) | 0x00_55_aa_ff).collect();
        let alphas_before: Vec<u8> = pixels.iter().map(|p| (*p >> 24) as u8).collect();
        for q in 0u8..=100 {
            let mut p = pixels.clone();
            apply(&mut p, q);
            let alphas_after: Vec<u8> = p.iter().map(|x| (*x >> 24) as u8).collect();
            assert_eq!(alphas_after, alphas_before, "quality={q} alpha changed");
        }
        // Tiny touch so clippy/rustc doesn't complain about unused mut.
        apply(&mut pixels, 100);
    }

    #[test]
    fn apply_quantizes_rgb_channels_per_step() {
        // q=60 → n=2 → step=4. Every R,G,B byte of the result is a
        // multiple of 4 (or 0).
        let mut pixels = vec![0xff_aa_bb_cc, 0x80_01_02_03, 0x10_ff_ff_ff];
        apply(&mut pixels, 60);
        for px in &pixels {
            let r = ((*px >> 16) & 0xff) as u8;
            let g = ((*px >> 8) & 0xff) as u8;
            let b = (*px & 0xff) as u8;
            assert_eq!(r % 4, 0, "r={r}");
            assert_eq!(g % 4, 0, "g={g}");
            assert_eq!(b % 4, 0, "b={b}");
        }
    }

    #[test]
    fn apply_max_per_channel_error_bounded_for_every_quality() {
        // Build a small image carrying every byte value across the RGB
        // channels and verify the per-channel error never exceeds the
        // documented `step - 1` worst case (the clamp window absorbs the
        // upper end; outside it the tighter `step / 2` bound applies, as
        // shown by `quantize_channel_typical_error_is_bounded_by_step_half`).
        let mut pixels: Vec<u32> = (0u32..=255)
            .map(|v| 0xff_00_00_00 | (v << 16) | (v << 8) | v)
            .collect();
        let original = pixels.clone();
        for q in 0u8..=100 {
            let mut p = pixels.clone();
            apply(&mut p, q);
            let n = bits_to_drop_for_quality(q);
            let step = 1u32 << n;
            let max_err = if step == 0 { 0 } else { (step - 1) as i32 };
            for (a, b) in original.iter().zip(p.iter()) {
                for shift in [0u32, 8, 16] {
                    let oc = ((*a >> shift) & 0xff) as i32;
                    let qc = ((*b >> shift) & 0xff) as i32;
                    let err = (oc - qc).abs();
                    assert!(err <= max_err, "q={q} n={n} shift={shift} err={err}");
                }
            }
        }
        // Touch pixels to keep the binding live.
        apply(&mut pixels, 100);
    }

    #[test]
    fn quantize_is_apply_out_of_place() {
        // The free `quantize` function must produce the same pixels as a
        // clone-then-apply, for every quality level.
        let original: Vec<u32> = (0u32..256)
            .map(|i| {
                let r = (i * 7) as u8 as u32;
                let g = (i * 31) as u8 as u32;
                let b = (i * 53) as u8 as u32;
                (0xff << 24) | (r << 16) | (g << 8) | b
            })
            .collect();
        for q in (0u8..=100).step_by(5) {
            let mut a = original.clone();
            apply(&mut a, q);
            let b = quantize(&original, q);
            assert_eq!(a, b, "quality={q}");
        }
    }
}
