#![no_main]

//! End-to-end `decode_webp` harness — feeds attacker-controlled bytes
//! through the top-level container walker that internally exercises
//! every chunk type RFC 9649 §2 defines (`RIFF`, `WEBP`, `VP8 `,
//! `VP8L`, `VP8X`, `ALPH`, `ANIM`, `ANMF`, `ICCP`, `EXIF`, `XMP `).
//!
//! Contract: the call must always return a [`Result`]. A malformed
//! stream yields [`Err(WebpError::…)`]; a well-formed one yields
//! [`Ok(WebpImage)`]. Neither path may panic, integer-overflow in a
//! debug build, index out of bounds, or attempt to allocate an
//! attacker-controlled pixel buffer the size of a forged canvas.
//!
//! This is the highest-coverage target — it reaches every container
//! variant — but it spends most of its budget on the lossless
//! Huffman bit-stream because that path has the most state. The
//! three sibling targets exercise the header parsers in isolation
//! so the corpus minimiser is not biased by the LZ77 search-space
//! depth.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::decode_webp;

fuzz_target!(|data: &[u8]| {
    let _ = decode_webp(data);
});
