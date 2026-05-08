//! Integration tests for AVIF-style delta-merge animated encoding
//! (`AnimFrameMode::Delta`).
//!
//! The Delta mode flow:
//!  1. The first frame is always emitted as a full-canvas ANMF.
//!  2. For each subsequent frame, the encoder walks the canvas in
//!     `block_size`-sized blocks, computes a luminance-biased SAD per
//!     block, takes the bounding box of all blocks above the threshold,
//!     and (when the bbox is smaller than `max_bbox_fraction` of the
//!     canvas) emits an ANMF that places the sub-rectangle at the bbox
//!     location with `blending_method = 1` (DoNotBlend / overwrite).
//!
//! The wire-size win comes from frames where only a small part of the
//! canvas changes between frame N-1 and frame N — typical of animation
//! "rolling counter" / "small UI element changing on a static
//! background" use-cases.
//!
//! Tests (canvas + corner-block size are tuned for fast RDO encoding;
//! see the constant block at the top of the file):
//!  * **`delta_mode_shrinks_anmf_for_corner_change_frames`** —
//!    160×120 canvas, 32×32 corner block (~5% changing region); frame
//!    0 is fixed pseudo-noise; frames 1..N change only the corner. The
//!    test asserts the Delta-mode per-frame ANMF payload for frames
//!    1..N is at most 20% of the equivalent full-frame `Lossless`
//!    baseline (i.e. ≥ 80% smaller).
//!  * **`delta_mode_round_trip_decodes_to_input_frames`** — verify that
//!    after delta encoding, the in-crate decoder reconstructs each
//!    frame's canvas pixel-identically with the source frames (Delta
//!    mode emits VP8L sub-rect tiles + DoNotBlend overwrite, so the
//!    composite is bit-exact).
//!  * **`delta_mode_first_frame_full_subsequent_subrect`** — inspect
//!    raw bytes: confirm frame 0's ANMF dimensions match the canvas,
//!    while frame 1's are the (small) bbox at the top-right corner
//!    with `blending_method = 1` (DoNotBlend) set.
//!  * **`delta_mode_rejects_oversized_bbox_frame_via_full_fallback`** —
//!    when a frame changes too much, Delta mode falls back to a full-
//!    canvas encode rather than emit a near-canvas-sized sub-rect.
//!  * **`delta_mode_identical_frame_emits_minimal_sub_rect`** — when
//!    frame N == frame N-1, the cost-model bbox is empty; the encoder
//!    emits a 1×1 overwrite tile so the duration counter still ticks
//!    without any visible canvas change.

use oxideav_webp::{
    build_animated_webp_with_options, decode_webp, AnimEncoderOptions, AnimFrame, AnimFrameMode,
    DeltaConfig,
};

// Canvas size + corner-block size are tuned so:
//   * the full RDO VP8L encoder runs in seconds, not minutes
//     (320×240 takes ~30s/frame on debug, ~5s/frame on release —
//     too slow even for release tests; 160×120 keeps it ≤ 1s/frame),
//   * the changing corner is ~10% of the canvas (the dispatch's
//     "rolling counter in the corner" example),
//   * the absolute byte counts are large enough that the wire-size
//     ratio assertion isn't dominated by chunk-header overhead.
//
// 160×120 with a 32×32 corner = 1024/19200 ≈ 5.3% changing region.
const W: u32 = 160;
const H: u32 = 120;
const CORNER: u32 = 32;
const CORNER_X0: u32 = W - CORNER; // top-right corner

/// Build a `W × H` RGBA frame: the canvas is fixed pseudo-noise, with
/// an optional `counter`-coloured `CORNER × CORNER` block in the
/// top-right corner. Frames with different `counter` values share ≥ 94%
/// of their pixels — the perfect case for delta-merge.
///
/// Pseudo-noise (xorshift hash per pixel) is used instead of a smooth
/// gradient so VP8L can't compress the background to a handful of
/// predictor literals — the baseline (full-frame) ANMF stays
/// representative of a real photographic frame's encoded size, making
/// the wire-size win from delta-merge visible.
fn build_frame_with_corner(counter: u8) -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            // Background pseudo-noise — pixel-identical between frames
            // (deterministic hash of (x, y)).
            let mut s = y.wrapping_mul(0x9E37_79B9) ^ x.wrapping_mul(0x85EB_CA77);
            s ^= s.wrapping_shr(13);
            s = s.wrapping_mul(0xC2B2_AE35);
            s ^= s.wrapping_shr(16);
            v[i] = (s & 0xff) as u8;
            v[i + 1] = ((s >> 8) & 0xff) as u8;
            v[i + 2] = ((s >> 16) & 0xff) as u8;
            v[i + 3] = 0xff;
        }
    }
    // Stamp the corner block — the only changing region across frames.
    for y in 0..CORNER {
        for x in CORNER_X0..(CORNER_X0 + CORNER) {
            let i = ((y * W + x) * 4) as usize;
            v[i] = counter; // R varies with frame
            v[i + 1] = 0xff - counter; // G varies inversely
            v[i + 2] = 0x80;
            v[i + 3] = 0xff;
        }
    }
    v
}

/// Sum of every ANMF chunk's payload size across the file (i.e. the
/// bytes that would shrink with a smaller per-frame sub-rectangle).
/// Returns one u64 per ANMF chunk in file order.
fn anmf_payload_sizes(blob: &[u8]) -> Vec<u32> {
    let body = &blob[12..]; // skip RIFF + size + WEBP
    let mut sizes = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= body.len() {
        let id = &body[pos..pos + 4];
        let size = u32::from_le_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]]);
        if id == b"ANMF" {
            sizes.push(size);
        }
        pos += 8 + size as usize + (size as usize & 1);
    }
    sizes
}

/// Read the (x, y, w, h, flags) header of the Nth ANMF chunk in the file.
fn anmf_header_n(blob: &[u8], n: usize) -> (u32, u32, u32, u32, u8) {
    let body = &blob[12..];
    let mut pos = 0usize;
    let mut idx = 0usize;
    while pos + 8 <= body.len() {
        let id = &body[pos..pos + 4];
        let size = u32::from_le_bytes([body[pos + 4], body[pos + 5], body[pos + 6], body[pos + 7]])
            as usize;
        let payload_start = pos + 8;
        if id == b"ANMF" {
            if idx == n {
                let p = &body[payload_start..];
                let x = u32::from_le_bytes([p[0], p[1], p[2], 0]) * 2;
                let y = u32::from_le_bytes([p[3], p[4], p[5], 0]) * 2;
                let w = u32::from_le_bytes([p[6], p[7], p[8], 0]) + 1;
                let h = u32::from_le_bytes([p[9], p[10], p[11], 0]) + 1;
                let flags = p[15];
                return (x, y, w, h, flags);
            }
            idx += 1;
        }
        pos = payload_start + size + (size & 1);
    }
    panic!("ANMF #{n} not found");
}

/// Build the 4-frame fixture (frame 0 + 3 delta frames) shared by the
/// wire-size + round-trip tests.
fn build_4frame_fixture() -> Vec<Vec<u8>> {
    vec![
        build_frame_with_corner(0x10),
        build_frame_with_corner(0x40),
        build_frame_with_corner(0x80),
        build_frame_with_corner(0xc0),
    ]
}

fn frames_from(rgbas: &[Vec<u8>]) -> Vec<AnimFrame<'_>> {
    rgbas
        .iter()
        .map(|rgba| AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba,
        })
        .collect()
}

#[test]
fn delta_mode_shrinks_anmf_for_corner_change_frames() {
    let rgbas = build_4frame_fixture();
    let frames = frames_from(&rgbas);

    // Lossless baseline — every frame full-canvas.
    let baseline = build_animated_webp_with_options(
        W,
        H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Lossless,
            ..Default::default()
        },
    )
    .expect("encode baseline");

    // Delta mode — first frame full, subsequent frames bbox sub-rect.
    let delta = build_animated_webp_with_options(
        W,
        H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig::default()),
            ..Default::default()
        },
    )
    .expect("encode delta");

    let baseline_anmfs = anmf_payload_sizes(&baseline);
    let delta_anmfs = anmf_payload_sizes(&delta);
    assert_eq!(baseline_anmfs.len(), 4, "baseline ANMF count");
    assert_eq!(delta_anmfs.len(), 4, "delta ANMF count");

    // Frame 0 ≈ same size in both (full-canvas in both encodings).
    // Frames 1..N: delta should be ≥80% smaller (the corner block is
    // ~32×32 = 1024 px ≈ 1.3% of the canvas; the encoded VP8L bitstream
    // for that tile is dominated by the chunk overhead but stays well
    // under 20% of the full-canvas VP8L payload).
    eprintln!(
        "anmf sizes: baseline={baseline_anmfs:?} delta={delta_anmfs:?}\n  total: baseline={} delta={}",
        baseline.len(),
        delta.len(),
    );
    for i in 1..4 {
        let ratio = (delta_anmfs[i] as f64) / (baseline_anmfs[i] as f64);
        assert!(
            ratio < 0.20,
            "frame {i}: delta ANMF {} is not ≤20% of baseline {} (ratio {:.3})",
            delta_anmfs[i],
            baseline_anmfs[i],
            ratio,
        );
    }
}

#[test]
fn delta_mode_round_trip_decodes_to_input_frames() {
    let rgbas = build_4frame_fixture();
    let frames = frames_from(&rgbas);
    let delta = build_animated_webp_with_options(
        W,
        H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig::default()),
            ..Default::default()
        },
    )
    .expect("encode delta");

    let img = decode_webp(&delta).expect("decode delta");
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
    assert_eq!(img.frames.len(), 4);

    // Each decoded canvas must match the source frame pixel-identically:
    // we use lossless under the hood for Delta sub-rect tiles in
    // practice (Auto mode picks the smaller payload, which for 32×32
    // synthetic blocks is VP8L). Decoded pixels live in the full canvas
    // — the decoder composites the sub-rect onto the prior canvas.
    for (i, decoded) in img.frames.iter().enumerate() {
        assert_eq!(
            decoded.rgba.len(),
            (W * H * 4) as usize,
            "frame {i} canvas length"
        );
        // Find first mismatch for diagnostic.
        if decoded.rgba != rgbas[i] {
            let mismatch_idx = decoded
                .rgba
                .iter()
                .zip(rgbas[i].iter())
                .position(|(a, b)| a != b);
            panic!(
                "frame {i} pixel mismatch at byte {:?}: decoded[..16]={:?} src[..16]={:?}",
                mismatch_idx,
                &decoded.rgba[..16],
                &rgbas[i][..16],
            );
        }
    }
}

#[test]
fn delta_mode_first_frame_full_subsequent_subrect() {
    let rgbas = build_4frame_fixture();
    let frames = frames_from(&rgbas);
    let delta = build_animated_webp_with_options(
        W,
        H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig::default()),
            ..Default::default()
        },
    )
    .expect("encode delta");

    // Frame 0 covers the full canvas at (0, 0).
    let (x0, y0, w0, h0, flags0) = anmf_header_n(&delta, 0);
    assert_eq!(
        (x0, y0, w0, h0),
        (0, 0, W, H),
        "frame 0 must be full canvas"
    );
    let _ = flags0; // first frame inherits caller's blend flag

    // Frames 1..3: bbox-sized sub-rectangle in the top-right corner,
    // forced overwrite (blending_method bit set, so flags & 0x01 != 0).
    for i in 1..4 {
        let (x, y, w, h, flags) = anmf_header_n(&delta, i);
        assert!(
            w < W && h < H,
            "frame {i}: bbox should be smaller than canvas, got {w}x{h}"
        );
        // Bbox must be inside the canvas (offset + size ≤ canvas).
        assert!(x + w <= W && y + h <= H, "frame {i}: bbox out of bounds");
        // The CORNER×CORNER corner block at (CORNER_X0, 0) → bbox
        // should land in the top-right area: x ≥ CORNER_X0 - block_size
        // (block-aligned), y ≤ CORNER + block_size.
        assert!(
            x + 8 >= CORNER_X0 && y <= CORNER + 8,
            "frame {i}: bbox not in expected top-right region: ({x},{y},{w},{h})"
        );
        // Delta mode forces blending_method = 1 (overwrite) on
        // sub-rect frames so the decoder paints the tile fresh over
        // the prior canvas.
        assert_ne!(
            flags & 0x01,
            0,
            "frame {i}: delta mode must set blending_method=1 (DoNotBlend), flags={flags:#04x}"
        );
    }
}

#[test]
fn delta_mode_rejects_oversized_bbox_frame_via_full_fallback() {
    // When a frame changes too much (more than `max_bbox_fraction` of
    // the canvas), Delta mode should fall back to a full-canvas encode
    // for that frame rather than emit a useless near-canvas-sized sub-
    // rectangle. This is the safety valve for animations where the
    // delta-merge heuristic can't help.
    let f0 = build_frame_with_corner(0x10);
    // Build a frame whose every pixel differs from f0 — should force
    // the full-fallback path on frame 1.
    let mut f1 = vec![0u8; (W * H * 4) as usize];
    for px in f1.chunks_exact_mut(4) {
        px[0] = 0xff;
        px[1] = 0xff;
        px[2] = 0xff;
        px[3] = 0xff;
    }
    let frames = vec![
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
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let delta = build_animated_webp_with_options(
        W,
        H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig {
                max_bbox_fraction: 0.5, // tighter than default
                ..DeltaConfig::default()
            }),
            ..Default::default()
        },
    )
    .expect("encode delta");
    // Frame 1's ANMF header should report a full-canvas tile, not a
    // bbox sub-rect, because the change covers 100% of the canvas.
    let (x, y, w, h, _flags) = anmf_header_n(&delta, 1);
    assert_eq!(
        (x, y, w, h),
        (0, 0, W, H),
        "frame 1 fully changed: should fall back to full-canvas"
    );
}

#[test]
fn delta_mode_identical_frame_emits_minimal_sub_rect() {
    // When two frames are bit-identical, the cost-model bbox is empty
    // (no blocks above threshold). The encoder emits a minimal 1×1
    // overwrite at the origin so the frame still ticks the duration
    // counter without any visible change to the canvas.
    let f0 = build_frame_with_corner(0x80);
    let f1 = f0.clone();
    let frames = vec![
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
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let delta = build_animated_webp_with_options(
        W,
        H,
        [0u8; 4],
        0,
        &frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Delta(DeltaConfig::default()),
            ..Default::default()
        },
    )
    .expect("encode delta");
    let (x, y, w, h, flags) = anmf_header_n(&delta, 1);
    assert_eq!(
        (x, y, w, h),
        (0, 0, 1, 1),
        "identical frame should collapse to 1x1 minimal tile"
    );
    assert_ne!(flags & 0x01, 0, "minimal tile must be overwrite");

    // Round-trip — the canvas should stay pixel-identical to f0.
    let img = decode_webp(&delta).expect("decode");
    assert_eq!(
        img.frames[1].rgba, f0,
        "identical-frame round-trip must produce f0 unchanged"
    );
}
