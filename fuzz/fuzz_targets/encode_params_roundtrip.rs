#![no_main]

//! Lossless round-trip oracle over the encoder's **parameterised**
//! entry points — the path combinations the public one-shot façades
//! (`encode_webp_lossless`, `encode_vp8l_argb`) choose *for* the caller
//! and the existing roundtrip harnesses therefore only ever exercise at
//! whatever parameters the internal chooser elected.
//!
//! ## Why this harness
//!
//! `roundtrip_lossless` drives `encode_webp_lossless` → `decode_webp`,
//! so every iteration rides the encoder's own size-driven chooser: the
//! 2×2 `(no-tx | subtract-green) × (no-cache | cache)` sweep picks one
//! winner and only that winner's bits ever meet the decoder. The
//! *forced-parameter* encoder entry points — subtract-green forced on,
//! a caller-fixed §5.2.3 `cache_code_bits ∈ [1, 11]`, the literal-only
//! baseline, the width-threaded §5.2.2 distance-map form, and the
//! caller-fixed §3.4 `alpha_is_used` bit — stay cold on their losing
//! branches. This harness makes the *parameters themselves* fuzz input:
//!
//! * `encode_vp8l_argb_with(pixels, w, h, has_alpha)` — the §3.4 header
//!   bit fixed by the fuzzer rather than scanned from the pixels; the
//!   emitted 5-byte header is re-derived byte-for-byte (signature,
//!   dimensions, bit 28 = `has_alpha`, version 0) and the stream is
//!   decoded back through the §4 driver at the header's dimensions.
//! * `encode_argb_literals_with_width(pixels, w)` — the production
//!   width-threaded entry, decoded headerless at `(w, h)`.
//! * `encode_argb_literals` / `encode_argb_literals_only` /
//!   `encode_argb_literals_subtract_green` /
//!   `encode_argb_literals_color_cache(pixels, bits)` — the width-less
//!   entries, whose §5.2.2 distance chooser runs at `image_width = 1`;
//!   each stream is decoded headerless at the self-consistent `(1, N)`
//!   framing (one pixel per scan line), which is the only geometry the
//!   width-1 distance codes are defined against.
//!
//! Every decode must reproduce the input ARGB pixels **exactly** — VP8L
//! is lossless at every parameter point, not just at the chooser's
//! winner. A crash is an encoder/decoder panic; a mismatch is a real
//! lossless-contract violation on a forced parameter combination.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_encode::{
    encode_argb_literals, encode_argb_literals_color_cache, encode_argb_literals_only,
    encode_argb_literals_subtract_green, encode_argb_literals_with_width,
};
use oxideav_webp::{encode_vp8l_argb_with, vp8l_transform};

/// Per-side dimension ceiling: 32×32 = at most 1024 pixels per exec.
const MAX_DIM: u32 = 32;

/// Decode a headerless VP8L image-stream at `(width, height)` and
/// assert it reproduces `pixels` exactly.
fn assert_headerless_roundtrip(stream: &[u8], width: u32, height: u32, pixels: &[u32], tag: &str) {
    let decoded = vp8l_transform::decode_lossless_headerless(stream, width, height)
        .unwrap_or_else(|e| panic!("{tag}: encoder output must decode, got {e:?}"));
    assert_eq!(
        decoded.pixels(),
        pixels,
        "{tag}: VP8L is lossless — decoded ARGB must equal the encoder input",
    );
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let width = u32::from(data[0]) % MAX_DIM + 1;
    let height = u32::from(data[1]) % MAX_DIM + 1;
    let control = data[2];
    // §5.2.3 cache_code_bits forced into the wire-legal [1, 11] window.
    let cache_bits = u32::from(data[3]) % 11 + 1;
    let has_alpha = control & 1 != 0;

    // Fuzz-controlled ARGB pixels (alpha included — the encoder must be
    // lossless on translucent and on garbage alpha alike).
    let body = &data[4..];
    let n = (width * height) as usize;
    let mut pixels = Vec::with_capacity(n);
    for i in 0..n {
        let at = |k: usize| -> u32 {
            if body.is_empty() {
                0
            } else {
                u32::from(body[(i * 4 + k) % body.len()])
            }
        };
        pixels.push((at(0) << 24) | (at(1) << 16) | (at(2) << 8) | at(3));
    }

    // ── Full-header form, caller-fixed §3.4 alpha bit ────────────────
    let with_header = encode_vp8l_argb_with(&pixels, width, height, has_alpha)
        .expect("encode_vp8l_argb_with must accept an exactly-sized in-bounds image");
    // Re-derive the emitted §3.4 / §7.1 image header byte-for-byte.
    assert!(
        with_header.len() >= 5 && with_header[0] == 0x2F,
        "the emitted stream must open with the 5-byte §3.4 header behind 0x2F",
    );
    let packed = u32::from(with_header[1])
        | (u32::from(with_header[2]) << 8)
        | (u32::from(with_header[3]) << 16)
        | (u32::from(with_header[4]) << 24);
    assert_eq!(
        ((packed & 0x3FFF) + 1, ((packed >> 14) & 0x3FFF) + 1),
        (width, height),
        "the §3.4 header must encode the caller dimensions as the 14-bit minus-one fields",
    );
    assert_eq!(
        (packed >> 28) & 1 == 1,
        has_alpha,
        "the §3.4 alpha_is_used bit must echo the caller's has_alpha verbatim",
    );
    assert_eq!(
        (packed >> 29) & 0x7,
        0,
        "the §3.4 version field MUST be emitted as 0",
    );
    let decoded = vp8l_transform::decode_lossless(&with_header, width, height)
        .expect("encode_vp8l_argb_with output must decode");
    assert_eq!(
        decoded.pixels(),
        &pixels,
        "encode_vp8l_argb_with: decoded ARGB must equal the encoder input",
    );

    // ── Width-threaded literals form at the caller geometry ──────────
    let ww = encode_argb_literals_with_width(&pixels, width);
    assert_headerless_roundtrip(
        &ww,
        width,
        height,
        &pixels,
        "encode_argb_literals_with_width",
    );

    // ── Width-less literal forms at the (1, N) framing ───────────────
    let n32 = n as u32;
    let chosen = encode_argb_literals(&pixels);
    assert_headerless_roundtrip(&chosen, 1, n32, &pixels, "encode_argb_literals");

    let lit_only = encode_argb_literals_only(&pixels);
    assert_headerless_roundtrip(&lit_only, 1, n32, &pixels, "encode_argb_literals_only");

    let sg = encode_argb_literals_subtract_green(&pixels);
    assert_headerless_roundtrip(&sg, 1, n32, &pixels, "encode_argb_literals_subtract_green");

    let cached = encode_argb_literals_color_cache(&pixels, cache_bits);
    assert_headerless_roundtrip(&cached, 1, n32, &pixels, "encode_argb_literals_color_cache");

    // The parameterised encoders are pure: replaying the cache-forced
    // form (the one carrying a fuzz-chosen parameter) must reproduce
    // byte-identical output.
    assert_eq!(
        encode_argb_literals_color_cache(&pixels, cache_bits),
        cached,
        "encode_argb_literals_color_cache must be deterministic over the same input",
    );
});
