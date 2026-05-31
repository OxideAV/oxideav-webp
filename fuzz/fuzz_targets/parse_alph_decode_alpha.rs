#![no_main]

//! `parse_alph_header` + `decode_alpha_plane` harness — RFC 9649
//! §2.7.1.2 alpha chunk (ALPH). The first byte carries:
//!
//! * bits 0..1 — filtering method (none / horizontal / vertical /
//!   gradient);
//! * bits 2..3 — compression method (raw / VP8L-lossless);
//! * bits 4..5 — pre-processing flag;
//! * bits 6..7 — must be zero (reserved).
//!
//! The rest of the payload is either a raw width × height byte plane
//! (compression method 0) or a VP8L-lossless ARGB sub-stream whose
//! green channel is interpreted as the alpha plane (method 1).
//!
//! The target splits the input bytes into a 12-byte VP8X header (so
//! `decode_alpha_plane` finds a canvas size to allocate against) and
//! the remainder which we pack into a synthetic `RIFF/WEBP` carrying
//! a VP8X + ALPH chunk pair. `decode_alpha_plane` must always
//! return a [`Result`] — never panic, OOM on a forged canvas size,
//! or run away on a degenerate LZ77 stream.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{decode_alpha_plane, parse_alph_header};

// ---------------------------------------------------------------------------
// Helpers for building a minimal RIFF/WEBP wrapper.
//
// The wrapper layout — 12-byte RIFF header (`RIFF` + le32 size + `WEBP`),
// 8-byte chunk header (FourCC + le32 payload size), payload + 1-byte
// pad if payload size is odd — is fully described in RFC 9649 §2.3, and
// is replicated here from the spec rather than from any reference
// implementation.
// ---------------------------------------------------------------------------

fn push_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

fuzz_target!(|data: &[u8]| {
    // Always exercise the header parser by itself — it's a 1-byte
    // header so even an empty input hits the "expected ≥ 1 byte"
    // path. Drop the first byte and feed the rest as the chunk
    // payload (this is the actual call shape from `lib.rs`).
    let _ = parse_alph_header(data);

    // The combined decode path needs a 10-byte VP8X payload (so we
    // can hand the rest to ALPH). Skip if too short.
    if data.len() < 10 {
        return;
    }
    let (vp8x_payload, alph_payload) = data.split_at(10);

    let mut riff_body = Vec::with_capacity(20 + alph_payload.len());
    riff_body.extend_from_slice(b"WEBP");
    push_chunk(&mut riff_body, b"VP8X", vp8x_payload);
    push_chunk(&mut riff_body, b"ALPH", alph_payload);

    let mut riff = Vec::with_capacity(8 + riff_body.len());
    riff.extend_from_slice(b"RIFF");
    riff.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    riff.extend_from_slice(&riff_body);

    let _ = decode_alpha_plane(&riff);
});
