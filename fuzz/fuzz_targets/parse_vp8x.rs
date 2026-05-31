#![no_main]

//! `parse_vp8x_header` harness — RFC 9649 §2.7.1 VP8X extended-image
//! header. The payload is a fixed 10 bytes: 1 reserved/flags byte,
//! 3 reserved bytes, then two 24-bit little-endian `canvas_width-1`
//! / `canvas_height-1` fields. The parser must reject:
//!
//! * payload length ≠ 10 bytes
//! * the 7 reserved bits inside the flags byte being non-zero
//! * the dimensions decoding to a canvas larger than the
//!   §2.7.1 16384 × 16384 maximum
//! * any reserved-zero region carrying a non-zero bit
//!
//! and may NEVER panic / abort / OOM on attacker bytes.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::parse_vp8x_header;

fuzz_target!(|data: &[u8]| {
    let _ = parse_vp8x_header(data);
});
