#![no_main]

//! Assemble arbitrary fuzz-supplied §5.2.2 backward-reference runs into a
//! decoded pixel buffer through the pure-function entry point
//! `oxideav_webp::vp8l_decode::apply_backward_reference`.
//!
//! Every LZ77 backward reference in a VP8L §5.2 entropy stream is
//! realised by exactly this assembler. After the §5.2.2 length prefix
//! code yields a length `L` and the §5.2.2 distance prefix code yields a
//! `distance_code` (resolved to a scan-line pixel distance `D` by the
//! sibling §5.2.2 `distance_code_to_pixel_distance` lookup), the decoder
//! copies `L` pixels from `position - D` into the growing pixel buffer.
//! Both the length and the distance originate on the attacker-controlled
//! entropy stream, so the assembler must defend its two §5.2.2 carrier
//! invariants on every call:
//!
//! * **Underflow** (`D > position`): the source `position - D` would
//!   reach back before the first decoded pixel. Refused with
//!   `DecodeError::BackwardReferenceUnderflow`, buffer untouched.
//! * **Overflow** (`position + L > total_pixels`): the run would write
//!   more pixels than the image holds. Refused with
//!   `DecodeError::BackwardReferenceOverflow`, buffer untouched.
//!
//! On success exactly `L` pixels are appended; the copy is the standard
//! byte-for-byte LZ77 walk, so an overlapping run (`D < L`) repeats the
//! pixels it is itself emitting — each source index is read *after* the
//! preceding appends, never from a pre-snapshot.
//!
//! Sibling harnesses cover the layers around this primitive —
//! `distance_code` (§5.2.2 `distance_code` → `D` lookup feeding this
//! assembler's `dist`), `parse_meta_prefix` / `parse_transform_list`
//! (§4 / §5.2.3 / §6.2 preamble before the §5.2 entropy body), `decode`
//! (full §2 RIFF + §3..§5 entry point), `roundtrip_lossless` (encode →
//! decode equality oracle) — but **none** of them drives the §5.2.2 copy
//! assembler directly across the full attacker-reachable `(length,
//! dist, total_pixels, prefill)` cross-product. This seventeenth harness
//! does, cross-checking every outcome against the §5.2.2 spec rules with
//! a parallel reference model.
//!
//! The contract under test, per RFC 9649 §5.2.2:
//!
//! * The call always returns a `Result` — no panic, no debug-build
//!   integer overflow in `position + length`, no out-of-bounds index
//!   into the source pixels.
//! * `D > position` ⇒ `BackwardReferenceUnderflow { position, distance }`
//!   with `position == pixels.len()` at call time and `distance == D`,
//!   and the buffer is byte-identical to its pre-call contents.
//! * `position + L > total_pixels` ⇒ `BackwardReferenceOverflow
//!   { position, length, total_pixels }` with the three fields equal to
//!   the call inputs, and the buffer is byte-identical to its pre-call
//!   contents.
//! * Otherwise the run succeeds, appends exactly `L` pixels, returns the
//!   range `position..position + L`, and every appended pixel equals the
//!   parallel reference model's LZ77 walk
//!   (`out[position + i] = out[position + i - D]`).
//! * The function is deterministic — the same inputs replay identically.
//!
//! Every assertion below is a real §5.2.2 carrier violation if it ever
//! fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Bytes `[0..2]` — `prefill_len` (little-endian u16) masked to
//!   `[0, 4096]`: how many fuzz-derived ARGB pixels seed the buffer
//!   before the run, modelling the §5.2 decode position the assembler
//!   is called at.
//! * Bytes `[2..4]` — `length` (little-endian u16): the §5.2.2 copy
//!   length `L`.
//! * Bytes `[4..6]` — `dist` (little-endian u16): the §5.2.2 scan-line
//!   pixel distance `D`.
//! * Bytes `[6..8]` — `headroom` (little-endian u16) masked to
//!   `[0, 8192]`: added to `prefill_len + length` to derive
//!   `total_pixels`, so both the exact-fit boundary and the overflow
//!   refusal are reachable depending on the fuzz draw (a separate draw
//!   path forces overflow by shrinking `total_pixels` below
//!   `prefill_len`).
//! * Bytes `[8..]` — repeated little-endian u32 words seeding the
//!   pre-fill ARGB pixels (cycled if fewer than `prefill_len` words are
//!   supplied).
//!
//! ## Iteration cost bound
//!
//! `prefill_len` is capped at 4096 and `length` at 65535; the copy is a
//! single linear walk, so the per-iteration cost is bounded at
//! ~70 K constant-time pushes — well inside the libFuzzer budget.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_decode::{apply_backward_reference, DecodeError};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let prefill_len = (u16::from_le_bytes([data[0], data[1]]) as usize).min(4096);
    let length = u16::from_le_bytes([data[2], data[3]]) as usize;
    // §5.2.2 carrier precondition: the scan-line pixel distance `D` is
    // always >= 1 — `distance_code_to_pixel_distance` clamps it (the
    // §5.2.2 `if (dist < 1) dist = 1` rule), so the assembler is never
    // called with `dist == 0` (a zero source offset would read the
    // not-yet-written pixel at `position`). Floor at 1 to model exactly
    // the values the real §5.2 decode path can reach.
    let dist = (u16::from_le_bytes([data[4], data[5]]) as usize).max(1);
    let headroom = (u16::from_le_bytes([data[6], data[7]]) as usize).min(8192);

    // Seed the pre-fill pixels from the remaining fuzz bytes, cycling if
    // there are not enough words. A run with no source words still gets
    // a deterministic fill (all zero) so the underflow / overflow guards
    // are exercised independently of the source content.
    let words: Vec<u32> = data[8..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut pixels: Vec<u32> = Vec::with_capacity(prefill_len + length);
    for i in 0..prefill_len {
        let w = if words.is_empty() {
            0
        } else {
            words[i % words.len()]
        };
        pixels.push(w);
    }

    // Two total_pixels regimes, alternated by the low bit of `headroom`
    // so both the success/exact-fit path and the overflow-refusal path
    // are routinely reached:
    //   * even headroom → total_pixels = prefill_len + length + headroom
    //     (success or exact-fit when no underflow).
    //   * odd headroom  → total_pixels = prefill_len.saturating_sub(...)
    //     (forces position + length > total_pixels → overflow refusal),
    //     unless length is 0.
    let total_pixels = if headroom & 1 == 0 {
        prefill_len + length + headroom
    } else {
        // Shrink below prefill_len + length to provoke overflow; never
        // below prefill_len itself (the buffer already holds that many,
        // and the assembler is never called with a buffer longer than
        // total_pixels in real decode).
        prefill_len + (headroom >> 1).min(length.saturating_sub(1))
    };

    check_one(&mut pixels, length, dist, total_pixels);
});

/// Cross-check a single §5.2.2 backward-reference assembly against the
/// spec rules with a parallel reference model.
fn check_one(pixels: &mut Vec<u32>, length: usize, dist: usize, total_pixels: usize) {
    let position = pixels.len();
    let snapshot = pixels.clone();

    let result = apply_backward_reference(pixels, length, dist, total_pixels);

    match result {
        Ok(range) => {
            // §5.2.2 success: the guards both passed.
            assert!(
                dist <= position,
                "§5.2.2 success implies dist {dist} <= position {position}",
            );
            assert!(
                position + length <= total_pixels,
                "§5.2.2 success implies position {position} + length {length} <= total_pixels {total_pixels}",
            );
            // The returned range names the freshly-appended pixels.
            assert_eq!(
                range,
                position..position + length,
                "§5.2.2 emitted range must be position..position+length",
            );
            // Exactly `length` pixels were appended; the pre-fill prefix
            // is unchanged.
            assert_eq!(
                pixels.len(),
                position + length,
                "§5.2.2 success must append exactly length {length} pixels",
            );
            assert_eq!(
                &pixels[..position],
                &snapshot[..],
                "§5.2.2 copy must not disturb the already-decoded prefix",
            );

            // Parallel reference LZ77 walk: out[position + i] must equal
            // out[position + i - dist] read *after* the preceding writes.
            let mut model = snapshot.clone();
            for i in 0..length {
                let v = model[position + i - dist];
                model.push(v);
            }
            assert_eq!(
                pixels, &model,
                "§5.2.2 overlapping LZ77 copy must match the byte-for-byte reference walk",
            );

            // Determinism: replaying the same inputs from the same
            // pre-fill produces the identical buffer.
            let mut replay = snapshot.clone();
            let r2 = apply_backward_reference(&mut replay, length, dist, total_pixels)
                .expect("§5.2.2 replay of a successful run must also succeed");
            assert_eq!(r2, range, "§5.2.2 replay range must match");
            assert_eq!(&replay, pixels, "§5.2.2 replay buffer must match");
        }
        Err(DecodeError::BackwardReferenceUnderflow { position: p, distance }) => {
            // §5.2.2 underflow trigger: dist strictly exceeds position.
            assert_eq!(p, position, "§5.2.2 underflow reports the call position");
            assert_eq!(distance, dist, "§5.2.2 underflow reports the call distance");
            assert!(
                dist > position,
                "§5.2.2 underflow must imply dist {dist} > position {position}",
            );
            // No partial write.
            assert_eq!(
                pixels, &snapshot,
                "§5.2.2 refused (underflow) run must leave the buffer untouched",
            );
        }
        Err(DecodeError::BackwardReferenceOverflow {
            position: p,
            length: l,
            total_pixels: t,
        }) => {
            // §5.2.2 overflow trigger: the run would exceed the image.
            assert_eq!(p, position, "§5.2.2 overflow reports the call position");
            assert_eq!(l, length, "§5.2.2 overflow reports the call length");
            assert_eq!(t, total_pixels, "§5.2.2 overflow reports total_pixels");
            assert!(
                position + length > total_pixels,
                "§5.2.2 overflow must imply position {position} + length {length} > total_pixels {total_pixels}",
            );
            // Underflow takes precedence in the assembler; if overflow
            // was raised, the underflow guard must have passed.
            assert!(
                dist <= position,
                "§5.2.2 overflow implies the underflow guard passed (dist {dist} <= position {position})",
            );
            assert_eq!(
                pixels, &snapshot,
                "§5.2.2 refused (overflow) run must leave the buffer untouched",
            );
        }
        Err(other) => {
            panic!("§5.2.2 apply_backward_reference returned an unexpected error: {other:?}");
        }
    }
}
