#![no_main]

//! Read embedded metadata out of arbitrary fuzz-supplied bytes through
//! `extract_metadata`. The metadata-only entry point walks the §2 RIFF
//! container looking for the optional `ICCP`, `EXIF`, and `XMP ` chunks
//! without doing any pixel decode; it must always return a `Result`
//! rather than panic / abort / OOM regardless of how malformed the
//! input is.
//!
//! The contract is the same shape as `decode.rs`: every byte of the
//! input is attacker-controlled, the call must always return, and a
//! claimed chunk size that overflows the parent RIFF chunk must surface
//! as `Err(WebpError::…)` instead of an index-out-of-bounds panic or a
//! sized-by-header-field allocation. Return value is intentionally
//! discarded.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::extract_metadata;

fuzz_target!(|data: &[u8]| {
    let _ = extract_metadata(data);
});
