#![no_main]

//! Parse arbitrary fuzz-supplied bytes through the §2.5 simple-lossy
//! `VP8 ` chunk handle entry point
//! `oxideav_webp::vp8_chunk::WebpLossyChunk::from_payload`.
//!
//! The §2 RIFF walker in `decode_webp` reaches the lossy keyframe-header
//! peek only with a payload slice the container layer has already
//! validated as a `VP8 ` chunk body, so the cross-product reachable
//! through the existing `decode.rs` / `decode_still_paths.rs` harnesses
//! exercises this peek only along the "well-formed RIFF" code path.
//! `WebpLossyChunk::from_payload` is also a public standalone surface —
//! `pub fn from_payload(payload: &[u8]) -> Result<Self, WebpLossyError>`
//! exported through `pub mod vp8_chunk` — that a container-layer caller
//! invokes against the bytes that follow the 8-byte §2.3 chunk header,
//! which are attacker-controlled and of arbitrary length. This harness
//! widens fuzz coverage onto that direct entry point: every byte of the
//! fuzz buffer is forwarded verbatim as the §2.5 `VP8 ` payload
//! candidate.
//!
//! The contract under test, per RFC 9649 §2.5 ("Simple File Format
//! (Lossy)") wrapping the RFC 6386 §9.1 key-frame header:
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow when assembling the 19-bit first-partition-size
//!   from the three frame-tag bytes, and no out-of-bounds index when the
//!   payload is shorter or longer than the 10 bytes §9.1 fixes for the
//!   key-frame header (3-byte frame tag + 3-byte start code + 4 bytes of
//!   scale/dimension words).
//! * If the call returns `Ok(hdr)`, every decoded field is internally
//!   consistent with the RFC 6386 §9.1 byte layout re-derived from the
//!   input bytes:
//!     * `payload.len() >= 10` — any successful parse implies the input
//!       carried at least the §9.1 key-frame header.
//!     * The §9.1 frame tag is a little-endian assembly of bytes 0..3.
//!       A successful parse implies frame-tag bit 0 (frame type) is 0
//!       (key frame); bits 1..3 are the `version`; bit 4 is `show_frame`;
//!       bits 5..23 are the 19-bit `first_partition_size`.
//!     * `bitstream()` is the verbatim payload slice — the §9.1 header is
//!       included at offset 0 and the entropy-coded remainder follows
//!       byte-for-byte, so `bitstream()` equals the input slice exactly.
//!     * Bytes 3..6 are the §9.1 start code `0x9D 0x01 0x2A`; a
//!       successful parse implies those three bytes matched.
//!     * The 16-bit width word (bytes 6..8, little-endian) splits into a
//!       14-bit `width` in bits 0..14 and a 2-bit `horizontal_scale` in
//!       bits 14..16; the 16-bit height word (bytes 8..10) splits the
//!       same way into `height` and `vertical_scale`.
//! * If the call returns `Err(_)`, the variant is cross-checked against
//!   its RFC 6386 §9.1 / RFC 9649 §2.5 refusal trigger:
//!     * `PayloadTooShortForKeyframe { got }` ⇒ `got` equals the slice
//!       length and that length is below 10.
//!     * `NotAKeyframe` ⇒ the payload was at least 10 bytes and
//!       frame-tag bit 0 (byte 0 bit 0) was 1 (interframe), which §2.5
//!       forbids inside a `VP8 ` WebP chunk.
//!     * `BadStartCode { got }` ⇒ the payload was at least 10 bytes, the
//!       frame type was a key frame, `got` equals bytes 3..6 verbatim,
//!       and those three bytes were not the §9.1 start code.
//!
//!   The `NotVp8Chunk` variant is only reachable through `from_chunk`'s
//!   FourCC gate, which `from_payload` does not exercise, so it never
//!   appears on this path.
//!
//! Every branch of the contract is observable to libFuzzer: a panic
//! short-circuits, and every assertion below is a real RFC 6386 §9.1 /
//! RFC 9649 §2.5 carrier violation if it ever fires.
//!
//! ## Iteration cost bound
//!
//! `from_payload` is a fixed-cost branch chain (one length test, three
//! frame-tag byte reads, one start-code comparison, two 2-byte
//! little-endian word reloads) — there is no allocation, no loop sized
//! by the input, and no recursion. A single fuzz iteration is
//! microseconds regardless of input length, so the harness can attempt
//! arbitrary payload sizes without iteration-cost concerns.
//!
//! ## Input layout
//!
//! The entire fuzz buffer is forwarded verbatim as the §2.5 `VP8 `
//! payload candidate. Inputs shorter than 10 bytes hit the
//! `PayloadTooShortForKeyframe` refusal path; 10-or-more-byte inputs
//! cover the full §9.1 frame-tag × start-code × dimension cross-product
//! (frame type, version, show_frame, 19-bit partition size, 14-bit
//! width/height, 2-bit horizontal/vertical scale).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8_chunk::{WebpLossyChunk, WebpLossyError};

/// The RFC 6386 §9.1 key-frame header length: 3-byte frame tag +
/// 3-byte start code + 4 bytes of scale/dimension words. Redefined
/// locally so the harness does not depend on a constant the crate's
/// surface might rename.
const VP8_KEYFRAME_HEADER_LEN: usize = 10;

/// The RFC 6386 §9.1 start code bytes that follow the 3-byte frame tag
/// in a key frame. Redefined locally for the same reason.
const VP8_START_CODE: [u8; 3] = [0x9D, 0x01, 0x2A];

fuzz_target!(|data: &[u8]| {
    match WebpLossyChunk::from_payload(data) {
        Ok(hdr) => {
            // RFC 6386 §9.1: success implies the payload carried at
            // least the 10-byte key-frame header — every shorter input
            // lands on PayloadTooShortForKeyframe.
            assert!(
                data.len() >= VP8_KEYFRAME_HEADER_LEN,
                "§9.1 from_payload returned Ok on a payload shorter than the key-frame header",
            );

            // RFC 6386 §9.1: the frame tag is bytes 0..3, little-endian.
            let tag = (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16);

            // §9.1 bit 0 — frame type. A successful parse means key
            // frame (bit clear); §2.5 forbids interframes in a WebP
            // `VP8 ` chunk.
            assert_eq!(
                tag & 0x1,
                0,
                "§9.1 from_payload returned Ok with frame-tag bit 0 set (interframe)",
            );

            // §9.1 bits 1..3 — version.
            assert_eq!(
                hdr.version(),
                ((tag >> 1) & 0x7) as u8,
                "§9.1 version must equal frame-tag bits 1..3",
            );

            // §9.1 bit 4 — show_frame.
            assert_eq!(
                hdr.show_frame(),
                ((tag >> 4) & 0x1) == 1,
                "§9.1 show_frame must equal frame-tag bit 4",
            );

            // §9.1 bits 5..23 — 19-bit first-partition-size.
            assert_eq!(
                hdr.first_partition_size(),
                (tag >> 5) & 0x7_FFFF,
                "§9.1 first_partition_size must equal frame-tag bits 5..23",
            );

            // §9.1 bytes 3..6 — start code. A successful parse implies a
            // match.
            assert_eq!(
                [data[3], data[4], data[5]],
                VP8_START_CODE,
                "§9.1 from_payload returned Ok with a non-matching start code",
            );

            // §9.1 bytes 6..8 — 16-bit width word, little-endian:
            // 14-bit width in bits 0..14, 2-bit horizontal scale in
            // bits 14..16.
            let w_word = (data[6] as u16) | ((data[7] as u16) << 8);
            assert_eq!(
                hdr.width(),
                w_word & 0x3FFF,
                "§9.1 width must equal the low 14 bits of the width word",
            );
            assert_eq!(
                hdr.horizontal_scale(),
                ((w_word >> 14) & 0x3) as u8,
                "§9.1 horizontal_scale must equal the high 2 bits of the width word",
            );

            // §9.1 bytes 8..10 — 16-bit height word, same split.
            let h_word = (data[8] as u16) | ((data[9] as u16) << 8);
            assert_eq!(
                hdr.height(),
                h_word & 0x3FFF,
                "§9.1 height must equal the low 14 bits of the height word",
            );
            assert_eq!(
                hdr.vertical_scale(),
                ((h_word >> 14) & 0x3) as u8,
                "§9.1 vertical_scale must equal the high 2 bits of the height word",
            );

            // RFC 9649 §2.5: bitstream() borrows the payload verbatim —
            // header at offset 0, entropy-coded remainder following
            // byte-for-byte.
            assert_eq!(
                hdr.bitstream(),
                data,
                "§2.5 bitstream() must equal the input payload byte-for-byte",
            );
        }
        Err(WebpLossyError::PayloadTooShortForKeyframe { got }) => {
            // RFC 6386 §9.1: too-short refusal is raised iff the input
            // is below the 10-byte key-frame header, and `got` reports
            // the actual length the parser observed.
            assert_eq!(
                got,
                data.len(),
                "§9.1 PayloadTooShortForKeyframe.got must equal the input slice length",
            );
            assert!(
                got < VP8_KEYFRAME_HEADER_LEN,
                "§9.1 PayloadTooShortForKeyframe must not be raised on a >=10-byte payload",
            );
        }
        Err(WebpLossyError::NotAKeyframe) => {
            // RFC 9649 §2.5 / RFC 6386 §9.1: interframe refusal implies
            // the payload was long enough to read the frame tag and that
            // frame-tag bit 0 was set.
            assert!(
                data.len() >= VP8_KEYFRAME_HEADER_LEN,
                "§9.1 NotAKeyframe requires a payload at least as long as the key-frame header",
            );
            assert_eq!(
                data[0] & 0x1,
                1,
                "§9.1 NotAKeyframe requires frame-tag bit 0 (interframe) set",
            );
        }
        Err(WebpLossyError::BadStartCode { got }) => {
            // RFC 6386 §9.1: start-code refusal implies the payload was
            // long enough, the frame type was a key frame, `got` echoes
            // bytes 3..6 verbatim, and those bytes were not the §9.1
            // start code.
            assert!(
                data.len() >= VP8_KEYFRAME_HEADER_LEN,
                "§9.1 BadStartCode requires a payload at least as long as the key-frame header",
            );
            assert_eq!(
                data[0] & 0x1,
                0,
                "§9.1 BadStartCode is only reachable past the key-frame frame-type gate",
            );
            assert_eq!(
                got,
                [data[3], data[4], data[5]],
                "§9.1 BadStartCode.got must echo payload bytes 3..6 verbatim",
            );
            assert_ne!(
                got, VP8_START_CODE,
                "§9.1 BadStartCode must not be raised when bytes 3..6 are the start code",
            );
        }
        Err(WebpLossyError::NotVp8Chunk { .. }) => {
            // NotVp8Chunk is only reachable through from_chunk's FourCC
            // gate; from_payload never produces it.
            unreachable!(
                "§2.5 from_payload cannot raise NotVp8Chunk (FourCC gate is in from_chunk)"
            );
        }
    }
});
