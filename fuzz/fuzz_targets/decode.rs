#![no_main]

//! Decode arbitrary fuzz-supplied bytes through `decode_webp`. The
//! decoder must always return a `Result` and never panic / abort / OOM,
//! regardless of how malformed the input is.
//!
//! The contract under test is purely that the call *returns*. A
//! malformed input must yield `Err(WebpError::…)`, a well-formed one
//! yields `Ok(WebpImage)`, and neither path may panic,
//! integer-overflow in a debug build, index out of bounds, or try to
//! allocate an attacker-controlled pixel buffer the size of the
//! declared width × height before validating the §2.7.1 / §3 width and
//! height fields. The return value is intentionally discarded.
//!
//! Round 432: iterations whose *declared* pixel load exceeds the shared
//! [`oxideav_webp_fuzz::MAX_DECLARED_PIXELS`] budget are skipped — a
//! spec-legal §5.2.2 backward-reference stream can expand a ~40-byte
//! chunk into ~10^8 pixels (~16 s, ~2.4 GiB under the address-sanitized
//! build), which is a decompression-ratio property of the format, not a
//! decoder defect, and would otherwise surface as a false-positive OOM
//! / slow-unit in the scheduled run. Pre-validation allocation bugs
//! remain in scope: the gate only skips files whose headers *parse* and
//! declare a large product, exactly the files the decoder is entitled
//! to decode at full size.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::decode_webp;
use oxideav_webp_fuzz::over_declared_pixel_budget;

fuzz_target!(|data: &[u8]| {
    if over_declared_pixel_budget(data) {
        return;
    }
    let _ = decode_webp(data);
});
