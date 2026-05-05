//! Regression for the `vp8l_lossless_roundtrip` fuzz crash on input
//! `[0, 0, 0, 0, 255]` (5 bytes), originally surfaced by `cargo fuzz`
//! against the pre-fix `try_emit_simple_huffman`.
//!
//! Sliced through the fuzz harness's `image_from_fuzz_input` (`shape =
//! data[0]`, `rgba = data[1..]`, `width = (shape % 64) + 1`,
//! `pixel_count = (rgba.len() / 4).min(2048)`, `height = pixel_count /
//! width`), the artifact yields a **1×1 RGBA black-opaque image**:
//! `RGBA = (0, 0, 0, 255)` → ARGB = `0xff_00_00_00`.
//!
//! Pre-fix the encoder produced a VP8L bitstream that our own decoder
//! rejected with `"VP8L: canonical Huffman length table self-collides"`.
//!
//! ## Root cause
//!
//! [`vp8l::encoder::try_emit_simple_huffman`] handles a 1- or 2-active-
//! symbol alphabet by emitting a 4-or-12-bit "simple-Huffman" header
//! (`simple=1, num_symbols-1, is_first_8bits, sym0[, sym1]`). The
//! simple wire format only supports symbols < 256: anything wider has
//! to fall back to the normal-tree path. The pre-fix function wrote
//! the leading `simple=1, num_symbols-1=0` bits **before** branching
//! on the symbol-index range, so for any single-active-symbol alphabet
//! whose only nonzero index is ≥ 256 the function leaked 2 stale
//! header bits into `bw` and then returned `None`. The caller then
//! fell through to `emit_huffman_tree` which wrote its own normal-tree
//! preamble — the resulting bitstream had a bogus `simple=1` prefix
//! followed by normal-tree contents, and every subsequent tree's
//! length-table read by the decoder was bit-misaligned.
//!
//! Predictor + colour-cache on a 1×1 black-opaque image is the minimum
//! repro because the residual collapses to `0x00000000` which lands at
//! cache index 0 (matching the cache's zero-initialised state), so the
//! single emitted symbol is `CacheRef{index: 0}`. That sets
//! `green_freq[GREEN_BASE_CODES + 0] = green_freq[280] = 1`, all other
//! alphabets stay empty. Index 280 is unrepresentable in the simple-
//! Huffman 8-bit field, triggering the leak.
//!
//! ## Fix
//!
//! Move the `s >= 256` (1-symbol) and `a >= 256 || b >= 256` (2-symbol)
//! eligibility checks **above** any `bw.write` call, so the function
//! either commits its full header in one shot or returns `None`
//! without touching the writer. See
//! `crates/oxideav-webp/src/vp8l/encoder.rs::try_emit_simple_huffman`.

use oxideav_webp::riff::{build_vp8l_with_alpha, WebpMetadata};
use oxideav_webp::{decode_webp, encode_vp8l_argb_with, EncoderOptions};

/// The minimum direct trigger: 1×1 ARGB = `0xff_00_00_00` (black, opaque)
/// encoded with predictor + colour-cache → single `CacheRef{index: 0}`
/// → green-alphabet single-active-symbol at index 280 → simple-Huffman
/// fallback to normal-tree → bitstream desync (pre-fix).
#[test]
fn black_opaque_1x1_predictor_plus_cache_roundtrips() {
    let opts = EncoderOptions {
        // The exact triggering combination: predictor *and* colour
        // cache *both* on. Either alone produces a self-roundtrippable
        // stream.
        use_subtract_green: false,
        use_color_transform: false,
        use_predictor: true,
        use_color_cache: true,
        cache_bits: 8,
        strip_transparent_color: false,
        use_color_index: false,
        near_lossless: 100,
        predictor_tile_bits: 4,
    };
    let argb = vec![0xff_00_00_00u32];
    let bitstream = encode_vp8l_argb_with(1, 1, &argb, true, opts).expect("encode must succeed");
    let wrapped = build_vp8l_with_alpha(&bitstream, 1, 1, &WebpMetadata::default());
    let img = decode_webp(&wrapped).expect("self-decode must succeed (was the fuzz crash)");
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].rgba.as_slice(), &[0u8, 0, 0, 255]);
}

/// Default-options path (the production `encode_vp8l_argb_with` shape
/// with everything turned on plus `strip_transparent_color = false`,
/// matching the fuzz harness's exact configuration). The triggering
/// pixel content is the same.
#[test]
fn black_opaque_1x1_default_options_roundtrips() {
    let opts = EncoderOptions {
        strip_transparent_color: false,
        ..Default::default()
    };
    let argb = vec![0xff_00_00_00u32];
    let bitstream = encode_vp8l_argb_with(1, 1, &argb, true, opts).expect("encode must succeed");
    let wrapped = build_vp8l_with_alpha(&bitstream, 1, 1, &WebpMetadata::default());
    let img = decode_webp(&wrapped).expect("self-decode must succeed");
    assert_eq!(img.frames[0].rgba.as_slice(), &[0u8, 0, 0, 255]);
}

/// Cover the broader class: 1×1 with *any* ARGB colour and the
/// triggering predictor+cache combo. Sweeps a handful of values that
/// hit different cache-hash slots to make sure the fix isn't accidentally
/// specific to index 0. (The cache hash is
/// `0x1e35_a7bd * argb >> (32 - cache_bits)` so each value lands at a
/// distinct slot.)
#[test]
fn one_pixel_predictor_plus_cache_sweep_roundtrips() {
    let opts = EncoderOptions {
        use_subtract_green: false,
        use_color_transform: false,
        use_predictor: true,
        use_color_cache: true,
        cache_bits: 8,
        strip_transparent_color: false,
        use_color_index: false,
        near_lossless: 100,
        predictor_tile_bits: 4,
    };
    for &argb in &[
        0xff_00_00_00u32,
        0xff_ff_ff_ffu32,
        0xff_12_34_56u32,
        0xff_ab_cd_efu32,
        0x80_80_80_80u32,
        0x00_00_00_00u32,
        0xff_de_ad_beu32,
    ] {
        let pixels = vec![argb];
        let bitstream =
            encode_vp8l_argb_with(1, 1, &pixels, true, opts).expect("encode must succeed");
        let wrapped = build_vp8l_with_alpha(&bitstream, 1, 1, &WebpMetadata::default());
        let img = decode_webp(&wrapped).unwrap_or_else(|e| {
            panic!(
                "1×1 ARGB={:#010x} self-decode failed: {:?} (bitstream={:02x?})",
                argb, e, bitstream
            )
        });
        let dec = u32::from_be_bytes([
            img.frames[0].rgba[3],
            img.frames[0].rgba[0],
            img.frames[0].rgba[1],
            img.frames[0].rgba[2],
        ]);
        assert_eq!(dec, argb, "1×1 ARGB roundtrip mismatch for {:#010x}", argb);
    }
}
