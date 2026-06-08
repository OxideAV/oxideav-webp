#![no_main]

//! Resolve arbitrary fuzz-supplied `(distance_code, image_width)` pairs
//! through the §5.2.2 distance-code-to-pixel-distance pure-function
//! lookup `oxideav_webp::vp8l_decode::distance_code_to_pixel_distance`.
//!
//! Every backward reference in a VP8L §5.2 LZ77 stream resolves through
//! exactly this function. The LZ77 length / distance prefix-code pair
//! decodes to a `(length, distance_code)` pair (§5.2.2 Table 4 plus the
//! distance prefix code group); the `distance_code` is then mapped to
//! the actual scan-line pixel distance `D` either through the §5.2.2
//! distance map (codes `1..=120` — a 120-entry `(xi, yi)` neighborhood
//! lookup table evaluated as `xi + yi * image_width`) or by subtracting
//! the §5.2.2 reservation offset (codes `> 120` denote a raw
//! scan-line distance of `code - 120`). A clamp of `D = max(D, 1)`
//! prevents the §5.2.2 negative-offset neighbors (the "left side" of
//! the neighborhood — `(-1, 1)`, `(-2, 1)`, etc.) from yielding a
//! zero or negative distance on the leftmost column of a 1-pixel-wide
//! row. Every byte feeding the function is attacker-controlled: the
//! `distance_code` comes off the entropy stream directly (only the
//! prefix-code envelope is symbol-table-clamped, not the payload value),
//! and the `image_width` comes from the §3.4 14-bit `width-1` field at
//! the start of the §2.6 VP8L bitstream.
//!
//! Sibling harnesses cover every layer **above** this primitive —
//! `parse_meta_prefix` (§5.2.3 + §6.2.2 + §6.2 preamble that produces
//! the 5 prefix codes the LZ77 stream consumes), `parse_transform_list`
//! (§4 transform-list reader that runs before the §5 entropy body),
//! `parse_container` (§2.3 / §2.4 RIFF walker that locates the §2.6
//! VP8L chunk), `decode` (full §2 RIFF + §3..§5 entry point that wraps
//! every primitive), `roundtrip_lossless` (encode→decode equality
//! oracle on the full §3 lossless contract) — but **none** of them
//! reaches `distance_code_to_pixel_distance` directly: they reach it
//! through whichever §5.2 LZ77 length/distance pair the upstream prefix
//! code produces, which means the actual `distance_code` values
//! visited per iteration are bounded by the entropy stream the upstream
//! reader produces. This thirteenth harness drives the §5.2.2
//! pure-function distance lookup directly across the full attacker-
//! reachable `distance_code ∈ [1, u32::MAX]` × `image_width ∈ [1,
//! 0x3FFF]` cross-product (the §3.4 14-bit image-width ceiling is
//! 16383), with every result cross-checked against the §5.2.2 spec
//! formula and the documented §5.2.2 clamp.
//!
//! The contract under test, per RFC 9649 §5.2.2:
//!
//! * The call must always return a `usize` — no panic, no debug-build
//!   integer overflow in `yi * image_width`, no out-of-bounds index
//!   into the 120-entry `DISTANCE_MAP`.
//! * For `distance_code` in `1..=120`:
//!     * The result equals `max(1, DISTANCE_MAP[distance_code - 1].0 +
//!       DISTANCE_MAP[distance_code - 1].1 * image_width)` (the §5.2.2
//!       neighborhood-lookup formula plus the clamp).
//!     * The result is `>= 1` (the §5.2.2 clamp guarantee).
//! * For `distance_code > 120`:
//!     * The result equals `distance_code - 120` (the §5.2.2 raw-
//!       scan-line-distance reservation).
//!     * The result is `>= 1` (the smallest reachable value here is
//!       `121 - 120 = 1`).
//! * For any `(distance_code, image_width)` pair, the result is
//!   deterministic — calling twice must produce the same value
//!   (the function is pure).
//!
//! Every assertion below is a real §5.2.2 carrier violation if it ever
//! fires; a panic short-circuits to libFuzzer.
//!
//! ## Iteration cost bound
//!
//! Each pair processed is a constant-time lookup — either a single
//! 120-entry table indexing followed by two arithmetic ops and a
//! comparison, or a single subtraction. The harness slices the fuzz
//! buffer into `(image_width, distance_code)` tuples (4 + 4 bytes per
//! tuple) and processes at most `data.len() / 8` of them per iteration,
//! capping the iteration cost at the libFuzzer 4 KiB default
//! (~512 tuples) and 64 KiB cap (~8192 tuples).
//!
//! ## Input layout
//!
//! * Bytes `[0..4]` — `image_width_raw` (little-endian u32). Masked to
//!   `0x3FFF` to model the §3.4 14-bit image-width ceiling; the floor
//!   is then bumped to `1` so the call is never made with width zero
//!   (the §3.4 wire encoding always represents at least a 1-pixel
//!   image).
//! * Bytes `[4..]` — repeated little-endian u32 `distance_code` words.
//!   Each is bumped to a minimum of `1` (the §5.2.2 wire-encoded
//!   distance code is always positive; `0` is the only value the
//!   pure-function lookup is not specified for — it would index the
//!   `DISTANCE_MAP` at `u32::MAX as usize` after the unchecked
//!   `(distance_code - 1)` subtraction).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_decode::{
    distance_code_to_pixel_distance, DISTANCE_MAP, NUM_DISTANCE_MAP_CODES,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // §3.4 14-bit image-width ceiling; the §3.4 wire encoding bumps the
    // stored value by 1, so the smallest reachable image_width is 1
    // (stored as 0). The largest is 2^14 = 16384 (stored as 16383). We
    // mask the fuzz bytes to the 14-bit window then bump to a minimum
    // of 1 to model the §3.4 reachable range exactly.
    let raw_w = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let image_width = (raw_w & 0x3FFF).max(1);

    // Every subsequent 4-byte word is a fuzz-controlled `distance_code`.
    // Floor at 1: `distance_code == 0` is undefined per §5.2.2 (the
    // wire encoding never produces it; the pure function would index
    // DISTANCE_MAP at `u32::MAX as usize` after the unchecked
    // subtraction).
    for chunk in data[4..].chunks_exact(4) {
        let raw_code = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let distance_code = raw_code.max(1);
        check_one(distance_code, image_width);
    }
});

/// Cross-check a single `(distance_code, image_width)` resolution
/// against the §5.2.2 spec formula and the §5.2.2 clamp.
fn check_one(distance_code: u32, image_width: u32) {
    let dist = distance_code_to_pixel_distance(distance_code, image_width);

    // §5.2.2 carrier rule: the result is always >= 1 (either by the
    // clamp on the neighborhood-lookup branch or by the smallest
    // reachable raw scan-line distance of 121 - 120 = 1).
    assert!(
        dist >= 1,
        "§5.2.2 distance_code_to_pixel_distance must always return >= 1; got {dist} for distance_code={distance_code}, image_width={image_width}",
    );

    if distance_code as usize > NUM_DISTANCE_MAP_CODES {
        // §5.2.2 raw-scan-line branch: `dist == distance_code - 120`.
        let expected = (distance_code - NUM_DISTANCE_MAP_CODES as u32) as usize;
        assert_eq!(
            dist, expected,
            "§5.2.2 raw-scan-line distance must equal distance_code {distance_code} - 120 = {expected}; got {dist}",
        );
        // Smallest reachable here: distance_code = 121 → dist = 1.
        // Largest reachable: distance_code = u32::MAX → dist = u32::MAX - 120.
        assert!(
            dist >= 1,
            "§5.2.2 raw-scan-line distance >= 1 (smallest case is 121 - 120)",
        );
    } else {
        // §5.2.2 neighborhood-lookup branch: `dist == max(1, xi + yi *
        // image_width)`. distance_code is in 1..=120 here.
        let idx = (distance_code - 1) as usize;
        assert!(
            idx < DISTANCE_MAP.len(),
            "§5.2.2 DISTANCE_MAP index {idx} must be in [0, 120)",
        );
        let (xi, yi) = DISTANCE_MAP[idx];

        // §5.2.2 spec arithmetic: `xi + yi * image_width`. With
        // image_width capped at 16383 and yi <= 8, the product fits in
        // i32 with plenty of headroom (max ~131 K).
        let signed_dist = xi + yi * image_width as i32;
        let expected = if signed_dist < 1 {
            1usize
        } else {
            signed_dist as usize
        };
        assert_eq!(
            dist, expected,
            "§5.2.2 neighborhood-lookup distance must equal max(1, xi + yi * image_width); got {dist}, expected {expected} for distance_code={distance_code} (xi={xi}, yi={yi}), image_width={image_width}",
        );

        // §5.2.2 clamp guarantee: a `(xi, yi)` whose evaluation gives
        // a non-positive value (e.g. distance_code 4 maps to (-1, 1)
        // and image_width 1 gives -1 + 1 = 0) must clamp to 1.
        if signed_dist < 1 {
            assert_eq!(
                dist, 1,
                "§5.2.2 clamp: non-positive xi + yi * image_width = {signed_dist} must clamp to 1; got {dist}",
            );
        }
    }

    // Determinism: calling twice produces the same value (the function
    // is pure; this catches any latent hidden state).
    let dist2 = distance_code_to_pixel_distance(distance_code, image_width);
    assert_eq!(
        dist, dist2,
        "§5.2.2 distance_code_to_pixel_distance must be deterministic; got {dist} then {dist2} for distance_code={distance_code}, image_width={image_width}",
    );
}
