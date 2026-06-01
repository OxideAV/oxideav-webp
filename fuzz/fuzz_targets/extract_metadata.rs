#![no_main]

//! Extract metadata from arbitrary fuzz-supplied bytes through
//! `extract_metadata`.
//!
//! The reader must always return a `Result` and never panic / abort
//! / OOM, regardless of how malformed the input is. The return value
//! is intentionally discarded — the contract under test is *the call
//! returns*.
//!
//! Specifically targeted: the RIFF + VP8X + `ICCP` / `EXIF` / `XMP `
//! chunk readers without paying for the pixel decode. A hostile
//! chunk-size field on any of the three metadata chunks must not
//! allow the reader to allocate beyond the declared input length.
//!
//! The metadata path is a strict subset of the full `decode_webp`
//! path — it walks the same RIFF skeleton, picks the same VP8X
//! chunk for the canvas declaration, and stops at the three
//! metadata chunks instead of also decoding pixels. Splitting it
//! into its own fuzz target keeps the corpus minimal (bytes that
//! exercise only the metadata-walk path don't have to also satisfy
//! the bitstream-decoder's wider validation surface) and makes any
//! ICCP / EXIF / XMP-specific panic regression easy to localise.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::extract_metadata;

fuzz_target!(|data: &[u8]| {
    let _ = extract_metadata(data);
});
