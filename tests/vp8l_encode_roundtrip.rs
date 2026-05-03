//! Encode → decode roundtrip tests for the pure-Rust VP8L encoder.
//!
//! The encoder emits a bare VP8L bitstream (no RIFF wrapper). The
//! existing crate-internal decoder (`vp8l::decode`) takes the same bare
//! bitstream, so we keep the tests self-contained — no `cwebp`/`libwebp`
//! fallbacks required. Each test asserts byte-identical RGBA output
//! after the round trip, since VP8L is lossless.

use oxideav_webp::encode_vp8l_argb;
use oxideav_webp::vp8l;
use oxideav_webp::vp8l::encoder::{encode_vp8l_argb_with, EncoderOptions};

/// Pack an R,G,B,A byte slice into ARGB u32 pixels (the layout the VP8L
/// encoder expects).
fn rgba_bytes_to_argb_pixels(rgba: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(rgba.len() / 4);
    for p in rgba.chunks_exact(4) {
        let r = p[0] as u32;
        let g = p[1] as u32;
        let b = p[2] as u32;
        let a = p[3] as u32;
        out.push((a << 24) | (r << 16) | (g << 8) | b);
    }
    out
}

/// Encode an RGBA buffer, round-trip it through the VP8L decoder, and
/// assert the emitted RGBA bytes match the input exactly.
fn roundtrip(width: u32, height: u32, rgba: &[u8], has_alpha: bool) {
    assert_eq!(
        rgba.len(),
        (width as usize) * (height as usize) * 4,
        "test rgba buffer has wrong length"
    );
    let pixels = rgba_bytes_to_argb_pixels(rgba);
    let bitstream = encode_vp8l_argb(width, height, &pixels, has_alpha).expect("encode succeeds");
    let decoded = vp8l::decode(&bitstream).expect("decode succeeds");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    let out_rgba = decoded.to_rgba();
    assert_eq!(out_rgba.len(), rgba.len(), "decoded RGBA length mismatch");
    if out_rgba != rgba {
        // Give a more useful message than "slices differ" on failure.
        let mut diff_count = 0usize;
        let mut first_diff = None;
        for (i, (a, b)) in rgba.iter().zip(out_rgba.iter()).enumerate() {
            if a != b {
                diff_count += 1;
                if first_diff.is_none() {
                    first_diff = Some((i, *a, *b));
                }
            }
        }
        panic!("VP8L roundtrip differs at {diff_count} byte(s); first diff @ {first_diff:?}");
    }
}

#[test]
fn vp8l_encode_gradient_128x128() {
    // Build a 128×128 RGBA gradient. Use non-aligned coefficients so the
    // encoder can't accidentally collapse everything to a single colour.
    let w = 128u32;
    let h = 128u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = (x * 2) as u8; // R: horizontal gradient
            rgba[i + 1] = (y * 2) as u8; // G: vertical gradient
            rgba[i + 2] = ((x + y) * 2) as u8; // B: diagonal
            rgba[i + 3] = 0xff; // A: opaque
        }
    }
    roundtrip(w, h, &rgba, false);
}

#[test]
fn vp8l_encode_solid_colour() {
    // 32×32 solid pink — exercises the encoder's degenerate-histogram
    // paths (only one green/red/blue/alpha value ever shows up).
    let w = 32u32;
    let h = 32u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for i in (0..rgba.len()).step_by(4) {
        rgba[i] = 0xFF;
        rgba[i + 1] = 0x66;
        rgba[i + 2] = 0x99;
        rgba[i + 3] = 0xFF;
    }
    roundtrip(w, h, &rgba, false);
}

#[test]
fn vp8l_encode_random_64x64() {
    // Deterministic pseudo-random RGBA — avoids `rand` dep. Uses a
    // tiny xorshift32 PRNG seeded from a constant.
    let w = 64u32;
    let h = 64u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut s: u32 = 0xC0DE_F00D;
    for b in rgba.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *b = (s & 0xff) as u8;
    }
    // The default encoder strips RGB from alpha-zero pixels (matches
    // libwebp's `exact = false` default), which would break the
    // byte-identical round-trip for any alpha-zero pixel here. Force
    // every alpha byte to be non-zero so this stays a pure-roundtrip
    // test of Huffman / LZ77 / transforms — see
    // `vp8l_encode_strip_transparent_color_shrinks_output` for the
    // dedicated coverage of the alpha-zero strip path.
    for chunk in rgba.chunks_exact_mut(4) {
        if chunk[3] == 0 {
            chunk[3] = 1;
        }
    }
    // Round-trip; `has_alpha=true` because random bytes will include
    // alpha != 0xff — we want the header flag to reflect reality.
    roundtrip(w, h, &rgba, true);
}

#[test]
fn vp8l_encode_tiny_1x1() {
    // Smallest valid image: 1×1 single pixel. Edge case for the header
    // width/height encoding + the single-symbol Huffman path.
    let rgba = [0x12u8, 0x34, 0x56, 0x78];
    roundtrip(1, 1, &rgba, true);
}

#[test]
fn vp8l_encode_two_pixel_wide() {
    // 2×1 image — LZ77 min-match is 3 pixels, so this is pure literals.
    let rgba = [0x01u8, 0x02, 0x03, 0xff, 0x04, 0x05, 0x06, 0xff];
    roundtrip(2, 1, &rgba, false);
}

#[test]
fn vp8l_encode_color_transform_beats_subtract_green_on_bgr_gradient() {
    // Build a 128×128 BGR-gradient image: R tracks B with a known
    // offset, G varies independently. This is the correlation the
    // colour transform is designed to exploit — once subtract-green has
    // taken care of the green axis, the `g→r` / `g→b` coefficients
    // should pull out the remaining B ↔ R correlation and shrink the
    // encoded size vs a subtract-green-only pass.
    let w = 128u32;
    let h = 128u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            // B: diagonal ramp.
            let b = ((x + y) & 0xff) as u8;
            // R: shifted version of B (strong R↔B correlation, weak
            // correlation with G).
            let r = b.wrapping_add(37);
            // G: independent horizontal ramp.
            let g = ((x * 2) & 0xff) as u8;
            rgba[i] = r;
            rgba[i + 1] = g;
            rgba[i + 2] = b;
            rgba[i + 3] = 0xff;
        }
    }
    let pixels = rgba_bytes_to_argb_pixels(&rgba);

    let sg_only =
        encode_vp8l_argb_with(w, h, &pixels, false, EncoderOptions::subtract_green_only())
            .expect("subtract-green-only encode");
    let mut sg_plus_color = EncoderOptions::subtract_green_only();
    sg_plus_color.use_color_transform = true;
    let with_color = encode_vp8l_argb_with(w, h, &pixels, false, sg_plus_color)
        .expect("subtract-green + colour-transform encode");

    eprintln!(
        "BGR gradient 128×128: sg_only={} bytes, sg+colour_transform={} bytes ({:+.1}%)",
        sg_only.len(),
        with_color.len(),
        100.0 * (with_color.len() as f64 - sg_only.len() as f64) / sg_only.len() as f64,
    );
    assert!(
        with_color.len() < sg_only.len(),
        "colour transform failed to shrink output on a correlated-BGR gradient: \
         sg_only={} bytes, with_colour={} bytes",
        sg_only.len(),
        with_color.len(),
    );

    // And the decoded pixels must still match bit-for-bit (lossless).
    let decoded = vp8l::decode(&with_color).expect("decode sg+colour_transform");
    assert_eq!(
        decoded.to_rgba(),
        rgba,
        "colour transform round-trip lost data"
    );
}

#[test]
fn vp8l_encode_strip_transparent_color_shrinks_output() {
    // Build a 64×64 RGBA buffer where exactly half the pixels are
    // fully transparent (alpha=0) and carry pseudo-random RGB. The
    // other half are opaque with the same pseudo-random RGB (so the
    // visible-pixel entropy is the same in both encodes — the only
    // delta comes from how the encoder treats the alpha-zero pixels).
    //
    // With `strip_transparent_color = true` (the default) the encoder
    // collapses the transparent half to RGB=0, which the predictor +
    // colour cache + LZ77 then compress aggressively. With the option
    // off, the transparent half carries full-entropy RGB, which
    // bloats the bitstream.
    let w = 64u32;
    let h = 64u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut s: u32 = 0xC0DE_F00D;
    for y in 0..h {
        for x in 0..w {
            // Cheap xorshift32 for deterministic "random" RGB.
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let r = (s & 0xff) as u8;
            let g = ((s >> 8) & 0xff) as u8;
            let b = ((s >> 16) & 0xff) as u8;
            // Half the pixels (a checkerboard tile pattern at 8-pixel
            // granularity) are fully transparent; the rest are opaque.
            let transparent = ((x / 8) + (y / 8)) % 2 == 0;
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = r;
            rgba[i + 1] = g;
            rgba[i + 2] = b;
            rgba[i + 3] = if transparent { 0 } else { 0xff };
        }
    }
    let pixels = rgba_bytes_to_argb_pixels(&rgba);

    // RDO is hard-wired strip-on; sidestep the RDO loop and pin both
    // trials to the same transform configuration so the only knob that
    // moves is `strip_transparent_color`.
    let on = EncoderOptions {
        strip_transparent_color: true,
        ..EncoderOptions::default()
    };
    let off = EncoderOptions {
        strip_transparent_color: false,
        ..EncoderOptions::default()
    };

    let with_strip = encode_vp8l_argb_with(w, h, &pixels, true, on).expect("strip-on encode");
    let without_strip = encode_vp8l_argb_with(w, h, &pixels, true, off).expect("strip-off encode");

    eprintln!(
        "alpha-zero half random RGB 64×64: strip_on={} bytes, strip_off={} bytes ({:+.1}%)",
        with_strip.len(),
        without_strip.len(),
        100.0 * (with_strip.len() as f64 - without_strip.len() as f64) / without_strip.len() as f64,
    );

    assert!(
        with_strip.len() < without_strip.len(),
        "strip_transparent_color = true did not shrink output: \
         strip_on={} bytes, strip_off={} bytes",
        with_strip.len(),
        without_strip.len(),
    );

    // Both bitstreams must round-trip — strip-on changes the RGB of
    // alpha-zero pixels but those are visually identical (alpha hides
    // them); strip-off must be byte-identical to the input.
    let decoded_off = vp8l::decode(&without_strip).expect("strip-off decode");
    assert_eq!(
        decoded_off.to_rgba(),
        rgba,
        "strip-off encode must preserve every input byte"
    );
    // Strip-on side: alpha plane preserved exactly, RGB preserved on
    // visible pixels, alpha-zero pixels' RGB collapsed to 0.
    let decoded_on = vp8l::decode(&with_strip).expect("strip-on decode");
    let out_on = decoded_on.to_rgba();
    for (i, chunk) in rgba.chunks_exact(4).enumerate() {
        let oi = i * 4;
        assert_eq!(out_on[oi + 3], chunk[3], "alpha must round-trip exactly");
        if chunk[3] == 0 {
            assert_eq!(
                &out_on[oi..oi + 3],
                &[0, 0, 0],
                "strip-on must zero RGB of alpha-zero pixel #{i}"
            );
        } else {
            assert_eq!(
                &out_on[oi..oi + 3],
                &chunk[..3],
                "visible pixel #{i} RGB must round-trip exactly"
            );
        }
    }
}

#[test]
fn vp8l_encode_transforms_shrink_non_trivial_image() {
    // Build a 64×64 RGBA gradient that has real spatial correlation in
    // all three colour channels + a constant alpha plane. Predictor,
    // subtract-green, and colour-cache should all pay off here; the
    // "bare" (no-transform) encoder is known to be strictly larger.
    let w = 64u32;
    let h = 64u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = (x * 3) as u8;
            rgba[i + 1] = (y * 3) as u8;
            rgba[i + 2] = ((x + y) * 3) as u8;
            rgba[i + 3] = 0xff;
        }
    }
    let pixels = rgba_bytes_to_argb_pixels(&rgba);
    let bare =
        encode_vp8l_argb_with(w, h, &pixels, false, EncoderOptions::bare()).expect("bare encode");
    let full = encode_vp8l_argb(w, h, &pixels, false).expect("full encode");
    assert!(
        full.len() < bare.len(),
        "transforms did not shrink output: bare={} bytes, with-transforms={} bytes",
        bare.len(),
        full.len()
    );
    // Round-trip the full version so the test also covers the decode
    // side of every enabled transform.
    let decoded = vp8l::decode(&full).expect("full decode");
    assert_eq!(
        decoded.to_rgba(),
        rgba,
        "full-transform round-trip lost data"
    );
}

#[test]
fn vp8l_encode_widened_predictor_pool_shrinks_diagonal() {
    // Diagonal stripes that lean north-east → south-west: the strongest
    // single-neighbour correlation is with the top-right neighbour
    // (predictor mode 3). With the old [0, 1, 2, 11] pool the encoder
    // had to settle for mode 11 ("select"), which on this image shape
    // pays for itself worse than a direct TR copy. The widened 14-mode
    // pool should pick mode 3 on at least one tile and shrink the output.
    //
    // We build the test input post-subtract-green-decorrelation in
    // mind: each diagonal stripe carries the same green value, so a
    // predictor that copies the top-right neighbour is exact and the
    // residual collapses to all zeros on most of the image.
    let w = 96u32;
    let h = 96u32;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            // Stripe index = x + y truncated to a small modulus. Each
            // stripe lasts exactly one pixel diagonal because the TR
            // neighbour at (x+1, y-1) has the SAME stripe index as
            // (x, y) — perfect for predictor mode 3.
            let stripe = ((x + y) % 8) as u8;
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = stripe.wrapping_mul(31);
            rgba[i + 1] = stripe.wrapping_mul(17);
            rgba[i + 2] = stripe.wrapping_mul(53);
            rgba[i + 3] = 0xff;
        }
    }
    let pixels = rgba_bytes_to_argb_pixels(&rgba);

    // Default options run the full 14-mode pool through `encode_vp8l_argb_with`.
    // The RDO sweep is irrelevant here — we want a like-for-like predictor
    // comparison so we hold the rest of the configuration fixed and let
    // the per-tile mode search make the only choice.
    let opts = EncoderOptions::default();
    let widened =
        encode_vp8l_argb_with(w, h, &pixels, false, opts).expect("widened-predictor-pool encode");

    // Self-consistency: bit-for-bit lossless round-trip.
    let decoded = vp8l::decode(&widened).expect("widened-pool decode");
    assert_eq!(
        decoded.to_rgba(),
        rgba,
        "widened-pool encode lost data on diagonal stripes"
    );

    // Also assert the widened encoder beats the bare baseline by a
    // wide margin on this fixture (the diagonal correlation is exact
    // in mode 3, so residuals should be all zero on most tiles).
    let bare =
        encode_vp8l_argb_with(w, h, &pixels, false, EncoderOptions::bare()).expect("bare encode");
    eprintln!(
        "diagonal stripes 96×96: bare={} bytes, with-predictor-pool={} bytes ({:+.1}%)",
        bare.len(),
        widened.len(),
        100.0 * (widened.len() as f64 - bare.len() as f64) / bare.len() as f64,
    );
    assert!(
        widened.len() * 2 < bare.len(),
        "widened-pool encode should beat bare encode by 2x+ on diagonal stripes: \
         bare={} bytes, widened={} bytes",
        bare.len(),
        widened.len(),
    );
}
