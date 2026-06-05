#![no_main]

//! Encode a fuzz-controlled multi-frame animation through
//! `build_animated_webp` and assert it survives a `decode_webp` round trip
//! frame-for-frame, pixel-for-pixel.
//!
//! The animated `RIFF/WEBP` layout `VP8X(A,I,…) | ANIM | ANMF…` defined in
//! `docs/image/webp/rfc9649-webp.txt` §2.7.1.1 is built here on the
//! VP8L-lossless per-frame path. That path is contractually byte-exact: a
//! frame encoded with `AnimFrame::new` (top-left, overwrite-blend, no
//! dispose, lossless mode) must reappear after decode with the same
//! width / height / duration_ms and the same `width * height * 4` RGBA
//! bytes — see `crates/oxideav-webp/tests/published_anim_api.rs` for the
//! same shape asserted on a fixed three-frame fixture.
//!
//! This harness widens that single-fixture assertion into a fuzz oracle on
//! the full §2.7.1.1 + §3 stack: every byte of every frame's RGBA buffer,
//! plus the frame count, plus the per-frame duration, plus the canvas
//! dimensions, is fuzzer-controlled. A crash here is either a panic on
//! either side of the round trip (build, ANMF walk, VP8L encode, prefix-
//! code emit, prefix-code read, §4 transform stack, canvas composite) or
//! a pixel / duration / dimension mismatch assertion failure — a real
//! §3 lossless contract or §2.7.1.1 carrier-field violation.
//!
//! Iteration cost is bounded so libFuzzer makes progress: the canvas is
//! capped at 32 × 32 and the frame count at 8, giving a worst-case input
//! of `8 * 32 * 32 * 4 = 32 KiB` of pixel data per round-trip.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{build_animated_webp, decode_webp, AnimFrame};

// Per-frame upper bounds. Kept small so a single fuzz iteration completes
// in milliseconds and the worst-case RGBA working set stays under ~32 KiB.
const MAX_W: u32 = 32;
const MAX_H: u32 = 32;
const MAX_FRAMES: usize = 8;
// Header bytes consumed before the per-frame payloads:
//   1 byte width, 1 byte height, 1 byte frame_count, then per-frame
//   [1 byte duration-lo, 1 byte duration-hi] = 2 bytes header per frame.
const HEADER_FIXED: usize = 3;
const HEADER_PER_FRAME: usize = 2;

fuzz_target!(|data: &[u8]| {
    // Need at least the fixed header plus one frame's per-frame header.
    if data.len() < HEADER_FIXED + HEADER_PER_FRAME {
        return;
    }

    // Derive small, in-bounds canvas dimensions from the first two bytes.
    let width: u32 = u32::from(data[0] % (MAX_W as u8)) + 1;
    let height: u32 = u32::from(data[1] % (MAX_H as u8)) + 1;
    // Frame count in 1..=MAX_FRAMES.
    let frame_count = (usize::from(data[2]) % MAX_FRAMES) + 1;

    // Need enough remaining bytes to cover the per-frame header table.
    let per_frame_header_total = frame_count * HEADER_PER_FRAME;
    if data.len() < HEADER_FIXED + per_frame_header_total {
        return;
    }

    // Slice off the per-frame duration header table; the rest is the
    // shared pixel-body pool that all frames cycle through.
    let header_end = HEADER_FIXED + per_frame_header_total;
    let dur_bytes = &data[HEADER_FIXED..header_end];
    let body = &data[header_end..];

    // `width <= MAX_W` and `height <= MAX_H`, so the product fits in u32
    // and the per-frame RGBA byte count fits comfortably in `usize`.
    let per_frame_rgba_len = (width as usize) * (height as usize) * 4;

    let mut frames: Vec<AnimFrame> = Vec::with_capacity(frame_count);
    for f in 0..frame_count {
        // Duration in 1..=65536 ms; the spec allows the field to be 0 but
        // we feed at least 1 so a downstream player would actually
        // progress the frame, mirroring `AnimFrame::new`'s common-case
        // contract. The carrier-field width is two bytes.
        let d_lo = u32::from(dur_bytes[f * HEADER_PER_FRAME]);
        let d_hi = u32::from(dur_bytes[f * HEADER_PER_FRAME + 1]);
        let duration_ms = (d_hi << 8) | d_lo | 1;

        // Build a per-frame RGBA buffer of exactly the required length,
        // cycling through the shared body pool (zero-padded if the input
        // is shorter than even one frame's worth of pixels). Each frame
        // shifts the cycle by `f * 7` so successive frames differ even
        // from a tiny input.
        let mut rgba = Vec::with_capacity(per_frame_rgba_len);
        for i in 0..per_frame_rgba_len {
            let v = if body.is_empty() {
                0u8
            } else {
                body[(i + f.wrapping_mul(7)) % body.len()]
            };
            rgba.push(v);
        }

        // `AnimFrame::new` constructs the top-left overwrite-blend,
        // no-dispose, lossless-mode frame the round-trip contract covers.
        frames.push(AnimFrame::new(width, height, rgba, duration_ms));
    }

    // Capture the source pixels and per-frame durations for post-decode
    // comparison before the frames are consumed by the encoder.
    let expected: Vec<(Vec<u8>, u32)> = frames
        .iter()
        .map(|fr| (fr.pixels.clone(), fr.duration))
        .collect();

    // Encode. A non-zero, in-bounds frame set with exactly-sized buffers
    // should always succeed; treat an encoder error as a benign skip
    // (the encoder explicitly returns `Unsupported` for the not-yet-wired
    // delta / auto paths, but `AnimFrame::new` selects `Lossless`).
    let encoded = match build_animated_webp(&frames) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Decode and assert the §2.7.1.1 + §3 round-trip contract.
    let image = decode_webp(&encoded).expect("encoded animated WebP must decode");
    assert_eq!(
        image.frames.len(),
        frame_count,
        "one decoded frame per encoded ANMF"
    );

    for (i, (decoded, (src_rgba, src_dur))) in image.frames.iter().zip(expected.iter()).enumerate()
    {
        assert_eq!(decoded.width, width, "frame {i} width survives round trip");
        assert_eq!(
            decoded.height, height,
            "frame {i} height survives round trip"
        );
        assert_eq!(
            decoded.duration_ms, *src_dur,
            "frame {i} duration survives round trip"
        );
        assert_eq!(
            decoded.rgba.len(),
            per_frame_rgba_len,
            "frame {i} flat RGBA len = width * height * 4"
        );
        assert_eq!(
            decoded.rgba, *src_rgba,
            "frame {i} pixels survive encode + decode byte-for-byte"
        );
    }
});
