#![no_main]

//! `parse_anim_header` + `parse_anmf_header` harness — RFC 9649
//! §2.7.1.1 animation chunks.
//!
//! * `ANIM` is 6 bytes: BGRA background colour (4 bytes) + le16
//!   loop count.
//! * `ANMF` is a 16-byte fixed header (frame X, Y, width-1,
//!   height-1 each as 24-bit le, then le24 duration in ms, then a
//!   reserved/blend/dispose flags byte) followed by per-frame
//!   sub-RIFF chunks.
//!
//! Both parsers must always return a [`Result`] — never panic on
//! short or attacker-padded input, never accept an offset/size pair
//! that would overflow a 24-bit canvas position, never read past
//! `payload.len()`.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{parse_anim_header, parse_anmf_header};

fuzz_target!(|data: &[u8]| {
    // Split the input between the two parsers so a single fuzz
    // iteration exercises both at once. The split point is itself
    // attacker-derived (so libfuzzer can drive coverage on either
    // side) but bounded so we never overflow the slice.
    let pivot = data.first().copied().unwrap_or(0) as usize % data.len().max(1);
    let (anim_bytes, anmf_bytes) = if data.is_empty() {
        (&[][..], &[][..])
    } else {
        // Drop the pivot byte itself and split the rest.
        let tail = &data[1..];
        let pivot = pivot.min(tail.len());
        tail.split_at(pivot)
    };
    let _ = parse_anim_header(anim_bytes);
    let _ = parse_anmf_header(anmf_bytes);
});
