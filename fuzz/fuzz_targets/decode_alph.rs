#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the §2.7.1.2 ALPH
//! standalone entry point `oxideav_webp::alph::decode_alpha`.
//!
//! The §2 RIFF walk in `decode_webp` only reaches `alph::decode_alpha`
//! with `(width, height)` taken from a *validated* §2.7.1 VP8X / §2.5
//! VP8 keyframe header — so the canvas-dimension cross-product the
//! decoder-side fuzz harness can drive through `decode.rs` is bounded
//! by what an earlier chunk-header validator accepted. The
//! `decode_alpha` function is also a **public standalone surface**
//! (`pub fn decode_alpha(payload: &[u8], width: u32, height: u32) ->
//! Result<Vec<u8>, AlphError>`) that downstream callers can invoke
//! directly with arbitrary dimensions — for example, to decode a
//! per-frame §2.7.1.1 ANMF alpha plane against the ANMF frame
//! dimensions. This harness widens fuzz coverage onto that direct
//! entry point.
//!
//! The contract under test, per RFC 9649 §2.7.1.2:
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow, no out-of-bounds index, no allocation sized by
//!   the §2.7.1.2 width × height field before the call's own arithmetic
//!   overflow check rejects it.
//! * If the call returns `Ok(plane)`, the plane is exactly
//!   `width * height` bytes long, per the §2.7.1.2 contract
//!   ("a byte sequence of length = width * height"). The two
//!   compression methods (raw + §3 VP8L lossless headerless image-
//!   stream) and the four §2.7.1.2 filter methods (none / horizontal /
//!   vertical / gradient) all flow through this same length guarantee.
//!
//! Both branches of the contract are asserted: the call returning is
//! observable to libFuzzer (a panic short-circuits), and the
//! length-equality assertion is a real §2.7.1.2 carrier violation if
//! it ever fires.
//!
//! ## Iteration cost bound
//!
//! Method 1 (lossless) decodes the alpha bitstream through the full
//! §3 VP8L decoder, including §4 transforms and §3.7 prefix-coded
//! entropy. That can be expensive on a large canvas. To keep a single
//! fuzz iteration within milliseconds the canvas is capped at 64 × 64
//! = 4 096 pixels (worst-case plane = 4 KiB, worst-case method-1
//! intermediate ARGB = 16 KiB).
//!
//! ## Input layout
//!
//! * Byte 0 — width nibble. Width is `(byte_0 % 64) + 1` ∈ [1, 64].
//! * Byte 1 — height nibble. Height is `(byte_1 % 64) + 1` ∈ [1, 64].
//! * Bytes 2.. — the §2.7.1.2 ALPH chunk payload. The first byte is
//!   the §2.7.1.2 info byte (`Rsv|P|F|C`); subsequent bytes are the
//!   alpha bitstream. If only two header bytes are provided, the
//!   payload is empty — `decode_alpha` will surface `AlphError::
//!   EmptyPayload` and return.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::alph::decode_alpha;

// Per-fuzz-iter upper bounds. 64 × 64 keeps the method-1 §3 VP8L
// sub-decode bounded and the per-iteration RGBA scratch under 16 KiB.
const MAX_W: u8 = 64;
const MAX_H: u8 = 64;

fuzz_target!(|data: &[u8]| {
    // Need the two dimension bytes plus at least one info byte so the
    // call exercises *something* — otherwise `decode_alpha` rejects on
    // the §2.7.1.2 empty-payload check before any interesting branch
    // runs, which the existing `decode.rs` harness already covers via
    // RIFF-level walks. (We do still allow short / one-byte payloads
    // below; they hit the EmptyPayload + RawLengthMismatch branches
    // that the existing harnesses don't reach by construction.)
    if data.len() < 2 {
        return;
    }

    // Derive small, in-bounds dimensions from the first two bytes. The
    // `% MAX_W` arithmetic stays well within `u8`; the `+ 1` keeps the
    // dimensions in [1, MAX_W] / [1, MAX_H] so a successful call always
    // has a non-zero plane to filter.
    let width: u32 = u32::from(data[0] % MAX_W) + 1;
    let height: u32 = u32::from(data[1] % MAX_H) + 1;

    // The remaining bytes are the ALPH chunk payload. The first byte
    // is the §2.7.1.2 info byte; the rest is the alpha bitstream. A
    // zero-length tail leaves `decode_alpha` to surface
    // `EmptyPayload` from the info-byte extraction.
    let payload = &data[2..];

    // Cap on width × height ensures the §2.7.1.2 plane allocation
    // stays bounded regardless of the fuzz-controlled `(width,
    // height)` pair: 64 × 64 × 1 = 4 096 bytes worst case.
    let expected_plane_len = (width as usize) * (height as usize);

    // The §2.7.1.2 contract: a successful decode produces exactly
    // `width * height` bytes. Any other length is a §2.7.1.2 carrier
    // violation — the assertion below treats it as a fuzz failure.
    if let Ok(plane) = decode_alpha(payload, width, height) {
        assert_eq!(
            plane.len(),
            expected_plane_len,
            "§2.7.1.2 ALPH plane length must equal width * height",
        );
    }
});
