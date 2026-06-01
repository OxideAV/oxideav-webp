#![no_main]

//! Decode arbitrary fuzz-supplied bytes through `decode_webp`.
//!
//! The decoder must always return a `Result` and never panic / abort
//! / OOM, regardless of how malformed the input is. The return value
//! is intentionally discarded — the contract under test is *the call
//! returns*, not what it returns.
//!
//! Classic WebP danger spots this target drives:
//!
//! * **RIFF preamble walk** — a hostile chunk-size word (the 4-byte
//!   `u32` at offset 4) must not allow reading past the slice when
//!   the declared size exceeds the actual payload length. Commonly
//!   seen in truncated downloads where the header survives but the
//!   payload is cut short.
//! * **VP8L canvas declaration** — §5.1 stores width-1 / height-1
//!   each in 14 bits, so the legal canvas is 1..=16384 in each
//!   dimension. A hostile combination must cross-check against the
//!   bitstream budget before allocating `width * height * 4` bytes.
//! * **VP8L §5.2.2 LZ77 backref length / distance** — the prefix
//!   codes admit any length/distance pair on the wire; a hostile
//!   distance > current decode position or length that walks past
//!   the canvas must surface as `Err(…)` rather than indexing into
//!   uninitialised memory.
//! * **VP8L §3.5 meta-prefix code** — the §3.5.6 "simple length
//!   code" vs "normal" distinction and the §3.5.7 normal-form
//!   reader-of-readers (the code-length code is itself prefix-coded)
//!   are a classic infinite-loop trap. Every combination of header
//!   bits must terminate.
//! * **VP8L §4 transform chain** — up to four transforms may appear
//!   in the bitstream, applied in reverse on decode. A hostile chain
//!   that repeats the same transform type, or declares a sub-image
//!   `size_bits` that overflows the sub-image dimensions, must
//!   refuse rather than re-apply.
//! * **VP8X extended layout** — the 10-byte VP8X chunk carries a
//!   bitmask telling the decoder which optional chunks to expect
//!   (`ALPH` / `ANIM` / `ICCP` / `EXIF` / `XMP`). A hostile mask
//!   pointing at chunks that don't exist (or omitting bits for
//!   chunks that do) must classify without panic.
//! * **ANMF rectangle bounds** — per-frame tile offset + width +
//!   height must stay inside the canvas; a hostile `(x_offset,
//!   frame_width)` that overflows `u32` arithmetic must surface as
//!   `Err(…)`.
//! * **`ALPH` filter / preproc bits** — four filter methods + two
//!   preproc modes + an optional compressed alpha lossless
//!   sub-bitstream — every combination must classify without
//!   panicking.
//!
//! No external library is consulted as a cross-decode oracle: the
//! clean-room wall bars libwebp / Pillow / the `image` crate, and
//! the format is well-specified enough that panic-freedom is the
//! contract worth driving. The harness uses the standalone API
//! (`default-features = false`) so no `oxideav-core` runtime is
//! built.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::decode_webp;

fuzz_target!(|data: &[u8]| {
    let _ = decode_webp(data);
});
