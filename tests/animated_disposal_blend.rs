//! Cross-frame disposal + blending coverage for the animated WebP path.
//!
//! The WebP container spec encodes per-frame composite behaviour as two
//! independent flag bits (§ANMF flags byte):
//!
//! * **blending_method** (bit 0): 0 = alpha-blend with canvas, 1 = overwrite.
//! * **disposal_method** (bit 1): 0 = leave as-is, 1 = clear to background.
//!
//! That's a 2×2 matrix of behaviours. The existing
//! `tests/animated_encode.rs` covers the (overwrite, no-dispose) corner
//! plus a couple of round-trip cases; this file fills in the other
//! three corners with deterministic small-canvas tests, asserting:
//!
//! 1. The encoder writes the right `flags` byte for each combination —
//!    we re-parse the ANMF chunk header out of the emitted bytes.
//! 2. The decoder honours the flags on a multi-frame file: the canvas
//!    state at each frame matches what the spec says it should given
//!    the previous frame's disposal + this frame's blending.
//!
//! Tests are pure round-trip — no external binaries — so they run on
//! every host without `cwebp` / `dwebp` / `webpmux` installed.

use oxideav_webp::{build_animated_webp, decode_webp, AnimFrame};

const W: u32 = 8;
const H: u32 = 8;

/// Solid-colour `width × height` RGBA tile.
fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        v.extend_from_slice(&rgba);
    }
    v
}

/// Walk the RIFF body and return the first ANMF chunk's `flags` byte
/// (the 16th byte of the ANMF payload, immediately after the 5×3-byte
/// X/Y/W/H/duration fields). Used to assert encoder side of the
/// disposal/blend flag bookkeeping without having to round-trip
/// through a third-party binary.
fn first_anmf_flags(blob: &[u8]) -> u8 {
    // RIFF/WEBP envelope: skip 12 bytes (RIFF + size + WEBP).
    let body = &blob[12..];
    let mut pos = 0usize;
    while pos + 8 <= body.len() {
        let id = &body[pos..pos + 4];
        let size = u32::from_le_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]])
            as usize;
        let payload_start = pos + 8;
        let payload_end = payload_start + size;
        if id == b"ANMF" {
            // ANMF flags = byte 15 of the chunk payload (3+3+3+3+3 = 15
            // bytes of x/y/w-1/h-1/duration before the flags byte).
            return body[payload_start + 15];
        }
        pos = payload_end + (size & 1);
    }
    panic!("no ANMF chunk found in encoded blob");
}

/// Encode a single-frame animation with the given (blend, dispose)
/// combination + return the emitted bytes for inspection.
fn encode_one(blend: bool, dispose: bool) -> Vec<u8> {
    let f = solid(W, H, [0xff, 0x80, 0x40, 0xff]);
    let frames = [AnimFrame {
        width: W,
        height: H,
        x_offset: 0,
        y_offset: 0,
        duration_ms: 50,
        blend,
        dispose_to_background: dispose,
        rgba: &f,
    }];
    build_animated_webp(W, H, [0u8; 4], 0, &frames).expect("build_animated_webp")
}

#[test]
fn encoder_writes_flags_byte_for_all_four_blend_dispose_combos() {
    // Spec mapping:
    //   bit 0 (blending) — 0 = blend, 1 = overwrite.
    //   bit 1 (disposal) — 0 = none,  1 = dispose-to-background.
    // So our encoder's `blend=true, dispose=false` → flags = 0x00,
    // `blend=false, dispose=false` → 0x01, etc.
    let cases = [
        (true, false, 0x00u8),
        (false, false, 0x01),
        (true, true, 0x02),
        (false, true, 0x03),
    ];
    for (blend, dispose, want) in cases {
        let blob = encode_one(blend, dispose);
        let got = first_anmf_flags(&blob);
        assert_eq!(
            got, want,
            "blend={blend} dispose={dispose} → expected ANMF flags {want:#04x}, got {got:#04x}"
        );
    }
}

#[test]
fn decoder_honours_dispose_to_background_between_frames() {
    // Two-frame animation:
    //   F0: full-canvas red, dispose=true → after F0 renders, the
    //       canvas's F0 region (the whole canvas) is wiped to (0,0,0,0).
    //   F1: small green tile at (0,0) sized 4×4, blend=false →
    //       overwrites just the top-left quadrant. Pixels outside that
    //       quadrant must be (0,0,0,0) thanks to F0's disposal — if the
    //       decoder ignored disposal, they'd still be red.
    let f0 = solid(W, H, [0xff, 0, 0, 0xff]); // red
    let f1 = solid(4, 4, [0, 0xff, 0, 0xff]); // green 4x4
    let frames = [
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: true,
            rgba: &f0,
        },
        AnimFrame {
            width: 4,
            height: 4,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let blob = build_animated_webp(W, H, [0u8; 4], 0, &frames).expect("encode");
    let img = decode_webp(&blob).expect("decode");
    assert_eq!(img.frames.len(), 2);
    let f1_canvas = &img.frames[1].rgba;
    // Top-left 4x4: green from F1.
    for y in 0..4 {
        for x in 0..4 {
            let i = ((y * W as usize + x) * 4) as usize;
            assert_eq!(
                &f1_canvas[i..i + 4],
                &[0, 0xff, 0, 0xff],
                "top-left quadrant should be green at ({x},{y})"
            );
        }
    }
    // Outside the 4x4 tile: must be (0,0,0,0) due to F0's dispose-to-bg.
    for y in 4..H as usize {
        for x in 0..W as usize {
            let i = ((y * W as usize + x) * 4) as usize;
            assert_eq!(
                &f1_canvas[i..i + 4],
                &[0, 0, 0, 0],
                "row past tile bottom should be (0,0,0,0) after F0 dispose at ({x},{y}); \
                 decoder probably skipped disposal"
            );
        }
    }
}

#[test]
fn decoder_blends_alpha_when_blend_flag_set() {
    // Two-frame animation:
    //   F0: full-canvas opaque red, no dispose.
    //   F1: full-canvas semi-transparent green (alpha=0x80), blend=true.
    //
    // Expected canvas after F1: alpha-blended mix of red + green per
    // standard "src over dst" formula. With sa=0x80 (=128/255) and
    // ia=0x7f (=127/255) the per-channel result is roughly
    // (red*127 + green*128)/255 — a mid-mix. The exact integer math the
    // decoder uses is `(s*sa + d*ia + 127) / 255`; we verify the green
    // channel is non-zero (proving blending happened) and the red
    // channel is below the original 0xff (proving the blend pulled it
    // down, not just overwriting).
    let f0 = solid(W, H, [0xff, 0, 0, 0xff]);
    let f1 = solid(W, H, [0, 0xff, 0, 0x80]);
    let frames = [
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f0,
        },
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: true,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let blob = build_animated_webp(W, H, [0u8; 4], 0, &frames).expect("encode");
    let img = decode_webp(&blob).expect("decode");
    assert_eq!(img.frames.len(), 2);
    let canvas = &img.frames[1].rgba;
    let r = canvas[0];
    let g = canvas[1];
    let b = canvas[2];
    let a = canvas[3];
    // Green channel was 0 in F0 and 0xff in F1; blend must produce a
    // non-zero (and non-0xff) result.
    assert!(
        g > 0x40 && g < 0xff,
        "green channel {g:#x} suggests blending didn't happen \
         (expected mid-range, got bounds-of-overwrite-or-hold)"
    );
    // Red channel was 0xff in F0 and 0 in F1; blend must drag it down.
    assert!(
        r < 0xff && r > 0x00,
        "red channel {r:#x} suggests blending didn't happen"
    );
    // Blue stays at 0 either way.
    assert_eq!(b, 0, "blue channel should stay 0");
    // Alpha after blending: sa=0x80 + (da*ia/255) where da=0xff, ia=0x7f.
    // = 128 + (255*127)/255 = 128 + 127 = 255 → 0xff.
    assert_eq!(a, 0xff, "post-blend alpha should still be opaque");
}

#[test]
fn decoder_overwrites_when_blend_flag_clear() {
    // Two-frame animation with blend=false on F1 — F1 must overwrite
    // F0's pixels exactly, ignoring source alpha.
    let f0 = solid(W, H, [0xff, 0, 0, 0xff]);
    let f1 = solid(W, H, [0x10, 0x20, 0x30, 0x40]);
    let frames = [
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f0,
        },
        AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false, // overwrite
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let blob = build_animated_webp(W, H, [0u8; 4], 0, &frames).expect("encode");
    let img = decode_webp(&blob).expect("decode");
    assert_eq!(img.frames.len(), 2);
    let canvas = &img.frames[1].rgba;
    // Every pixel of F1's canvas must be the literal F1 source pixel —
    // no blending into F0's red.
    for px in canvas.chunks_exact(4) {
        assert_eq!(
            px,
            &[0x10, 0x20, 0x30, 0x40],
            "blend=false should produce a verbatim overwrite, got {px:?}"
        );
    }
}
