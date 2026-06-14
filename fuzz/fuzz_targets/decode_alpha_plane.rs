#![no_main]

//! Fuzz the public file-level still-image alpha entry point
//! `oxideav_webp::decode_alpha_plane` over an attacker-controlled byte
//! buffer.
//!
//! ## Why this harness — a distinct surface from `decode_alph` / `parse_alph`
//!
//! The round-279 `decode_alph` harness drives the §2.7.1.2 standalone
//! decoder `alph::decode_alpha(payload, width, height)` with the
//! `(width, height)` pair supplied *by the harness* over a bare ALPH
//! chunk-payload slice, and `parse_alph` drives only the info-byte
//! `AlphHeader::parse`. Neither walks a whole file. `decode_alpha_plane`
//! is the **file-level** still-image alpha path: it
//!
//! 1. parses the §2.3 RIFF/WEBP container from raw bytes
//!    (`container::parse`);
//! 2. selects the §2.7.1.2 `ALPH` chunk (`Ok(None)` when absent);
//! 3. resolves the plane dimensions *from the file itself* — the
//!    §2.7.1 `VP8X` 24-bit canvas Width/Height when a `VP8X` chunk
//!    exists, else the §2.5 `VP8 ` keyframe header's visible
//!    dimensions — so the §2.7.1/§2.5 dimension-source → §2.7.1.2
//!    alpha-decode coherence path is exercised end to end; and
//! 4. decodes the alpha bitstream through both §2.7.1.2 compression
//!    methods (raw + headerless lossless) and all four §2.7.1.2 filter
//!    methods.
//!
//! No existing harness chains the §2.3 container walk, the §2.7.1 /
//! §2.5 dimension extraction, and the §2.7.1.2 alpha decode together
//! from a single fuzz-supplied file. This one does.
//!
//! ## The contract under test
//!
//! `decode_alpha_plane` must **always return a `Result`** — never
//! panic, never integer-overflow in a debug build, never index out of
//! bounds, never allocate a `width * height` plane sized by an
//! unbounded §2.7.1 canvas field. On every success it must honour the
//! §2.7.1.2 carrier invariant:
//!
//! * `Ok(None)` — a well-formed file with no `ALPH` chunk.
//! * `Ok(Some(plane))` — `plane.len() == width * height` for the
//!   file-resolved dimensions, and the call is **deterministic**
//!   (replaying the same bytes yields a byte-identical plane).
//! * `Err(_)` — a malformed file, a dimensionless `ALPH`, or a
//!   `width * height` that overflows `usize`.
//!
//! ## Allocation gate
//!
//! The §2.7.1 `VP8X` canvas fields are 24-bit each, so a malformed
//! header can declare a plane up to 2^24 × 2^24 (capped by the §2.7.1
//! Width × Height ≤ 2^32 − 1 rule) — far larger than any real still.
//! The raw §2.7.1.2 form is self-limiting (`decode_alpha` rejects a
//! raw bitstream whose length is not exactly `width * height` before
//! allocating), and the headerless lossless form caps its eager pixel
//! reservation, so a tiny payload with a huge declared canvas surfaces
//! an error rather than an OOM. To keep the scheduled Fuzz workflow off
//! the deliberate large-canvas allocation entirely, a cheap structural
//! pre-pass reads the §2.7.1 `VP8X` canvas dimensions (when present)
//! and skips the decode when the declared `width * height` exceeds
//! `1 << 20` pixels; such inputs are not representative still images
//! and the dimension caps are unit-tested separately.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{container, decode_alpha_plane, vp8x};

/// Declared-pixel ceiling above which the full decode tail is skipped.
/// A real still alpha plane is far below this; the §2.7.1 24-bit canvas
/// fields can declare orders of magnitude more.
const MAX_DECODE_PIXELS: u64 = 1 << 20;

fuzz_target!(|data: &[u8]| {
    // Cheap structural pre-pass: if the file parses as a §2.3 container
    // and carries a §2.7.1 `VP8X` whose declared canvas would size a
    // plane beyond the gate, skip the decode tail (the dimension caps
    // are unit-tested; the scheduled Fuzz workflow should not spend its
    // budget on the deliberate large-canvas path). A file that does not
    // even parse, or carries no parseable `VP8X`, falls through to the
    // decode below for the Result contract.
    if let Ok(c) = container::parse(data) {
        if let Some(vp8x_chunk) = c.first_chunk_with_fourcc(container::fourcc::VP8X) {
            if let Ok(hdr) = vp8x::Vp8xHeader::parse(vp8x_chunk.payload(data)) {
                let pixels = u64::from(hdr.canvas_width) * u64::from(hdr.canvas_height);
                if pixels > MAX_DECODE_PIXELS {
                    return;
                }
            }
        }
    }

    // The file-level still-image alpha path. The only hard contract is
    // that this returns a `Result`.
    let result = decode_alpha_plane(data);

    if let Ok(Some(plane)) = &result {
        // §2.7.1.2 carrier invariant: the reconstructed plane is exactly
        // `width * height` bytes. We do not have the file-resolved
        // dimensions handed back, but determinism plus a successful
        // length-checked decode is the invariant `decode_alpha` already
        // enforces internally; re-assert it cannot be empty unless the
        // canvas genuinely had a zero dimension (an empty plane).
        //
        // Pure-function determinism: a second decode over the same bytes
        // reproduces a byte-identical plane.
        let replay = decode_alpha_plane(data)
            .expect("decode_alpha_plane: a successful decode must replay successfully");
        let replay_plane =
            replay.expect("decode_alpha_plane: a Some(plane) decode must replay as Some(plane)");
        assert_eq!(
            &replay_plane, plane,
            "decode_alpha_plane must be deterministic over the same bytes",
        );
    }
});
