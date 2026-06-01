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

use libfuzzer_sys::fuzz_target;
use oxideav_webp::decode_webp;

fuzz_target!(|data: &[u8]| {
    let _ = decode_webp(data);
});
