//! Round-118 published-API round trips for the **animation encode** surface —
//! the published-0.1.5 `build_animated_webp` names re-exposed on top of the
//! in-crate VP8L encoder + the §2.7.1.1 ANIM / ANMF container framing.
//!
//! Every test here uses only standalone APIs (no `registry` feature), so the
//! file builds and runs under `--no-default-features`. It exercises:
//!
//! * `build_animated_webp` / `build_animated_webp_with_options` — assembling a
//!   multi-frame `.webp` from `AnimFrame`s (VP8L-lossless path).
//! * `AnimFrame` / `AnimFrameMode` / `AnimEncoderOptions` — the encode inputs.
//! * `decode_webp` — round-tripping the animation back to N `WebpFrame`s, each
//!   a byte-exact flat RGBA buffer, plus the ANIM background / loop count.
//! * `DeltaConfig` / `DownsampleKernel` — the (blocked) delta-path knobs'
//!   builder shape.

use oxideav_webp::anmf::{BlendingMethod, DisposalMethod};
use oxideav_webp::{
    build_animated_webp, build_animated_webp_with_options, decode_webp, AnimEncoderOptions,
    AnimFrame, AnimFrameMode, DeltaConfig, DownsampleKernel, WebpError, WebpMetadata,
};

/// Build a deterministic `width * height` RGBA ramp seeded by `seed` so each
/// frame differs (no external input).
fn make_rgba(width: u32, height: u32, seed: u32, opaque: bool) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            buf.push((x.wrapping_mul(37).wrapping_add(y).wrapping_add(seed) & 0xff) as u8);
            buf.push((y.wrapping_mul(53).wrapping_add(x).wrapping_mul(seed.max(1)) & 0xff) as u8);
            buf.push(((x ^ y).wrapping_mul(101).wrapping_add(seed) & 0xff) as u8);
            let a = if opaque {
                0xff
            } else {
                (255 - ((x.wrapping_add(y).wrapping_add(seed)) & 0xff)) as u8
            };
            buf.push(a);
        }
    }
    buf
}

#[test]
fn build_animated_webp_round_trips_three_frames() {
    let (w, h) = (6u32, 5u32);
    let f0 = make_rgba(w, h, 0, true);
    let f1 = make_rgba(w, h, 7, true);
    let f2 = make_rgba(w, h, 19, true);
    let frames = vec![
        AnimFrame::new(w, h, f0.clone(), 100),
        AnimFrame::new(w, h, f1.clone(), 150),
        AnimFrame::new(w, h, f2.clone(), 200),
    ];

    let file = build_animated_webp(&frames).expect("build animated webp");
    // RIFF/WEBP magic.
    assert_eq!(&file[0..4], b"RIFF");
    assert_eq!(&file[8..12], b"WEBP");

    let img = decode_webp(&file).expect("decode animation");
    assert_eq!(img.frames.len(), 3, "one WebpFrame per ANMF");

    // Per-frame pixels survive byte-for-byte, with durations preserved.
    for (i, (decoded, (src, dur))) in img
        .frames
        .iter()
        .zip([(f0, 100u32), (f1, 150), (f2, 200)])
        .enumerate()
    {
        assert_eq!(decoded.width, w, "frame {i} width");
        assert_eq!(decoded.height, h, "frame {i} height");
        assert_eq!(decoded.duration_ms, dur, "frame {i} duration");
        assert_eq!(
            decoded.rgba.len(),
            (w * h * 4) as usize,
            "frame {i} flat len"
        );
        assert_eq!(decoded.rgba, src, "frame {i} pixels round-trip exactly");
    }

    // Default options: infinite loop, transparent-black background.
    assert_eq!(img.anim_loop_count, Some(0));
    assert_eq!(img.anim_background_rgba, Some([0, 0, 0, 0]));
}

#[test]
fn build_animated_webp_with_options_carries_loop_bg_and_metadata() {
    let (w, h) = (4u32, 4u32);
    let frames = vec![
        AnimFrame::new(w, h, make_rgba(w, h, 1, false), 80),
        AnimFrame::new(w, h, make_rgba(w, h, 2, false), 80),
    ];
    let icc = b"icc-bytes".to_vec();
    let exif = b"Exif\x00\x00MM".to_vec();
    let xmp = b"<x:xmpmeta/>".to_vec();
    let opts = AnimEncoderOptions {
        loop_count: 3,
        background_rgba: [10, 20, 30, 255],
        metadata: WebpMetadata {
            icc: Some(&icc),
            exif: Some(&exif),
            xmp: Some(&xmp),
        },
        delta: DeltaConfig::default(),
        default_near_lossless_quality: None,
        default_lossy_quality: None,
    };

    let file = build_animated_webp_with_options(&frames, &opts).expect("build with options");
    let img = decode_webp(&file).expect("decode");

    assert_eq!(img.frames.len(), 2);
    assert_eq!(img.anim_loop_count, Some(3));
    assert_eq!(img.anim_background_rgba, Some([10, 20, 30, 255]));
    // Metadata reads back from the file-level chunks.
    assert_eq!(img.metadata.icc.as_deref(), Some(&icc[..]));
    assert_eq!(img.metadata.exif.as_deref(), Some(&exif[..]));
    assert_eq!(img.metadata.xmp.as_deref(), Some(&xmp[..]));
    // Alpha frames round-trip exactly.
    assert_eq!(img.frames[0].rgba, make_rgba(w, h, 1, false));
    assert_eq!(img.frames[1].rgba, make_rgba(w, h, 2, false));
}

#[test]
fn frame_blend_dispose_and_offset_fields_are_carried() {
    // A frame placed at an even offset with explicit blend/dispose must
    // produce a parseable file whose decoded duration matches, and the
    // canvas-compositing decoder (round 127) returns a full-canvas frame
    // sized to cover the rect (`max(x+w, y+h)`) with the sub-rect filled
    // by the input pixels and the rest left as the ANIM background
    // (transparent black by default).
    let (w, h) = (2u32, 2u32);
    let frame = AnimFrame {
        pixels: make_rgba(w, h, 5, true),
        width: w,
        height: h,
        x: 2,
        y: 4,
        duration: 42,
        blend: BlendingMethod::Overwrite,
        dispose: DisposalMethod::Background,
        mode: AnimFrameMode::Lossless,
        near_lossless_quality: None,
    };
    let file = build_animated_webp(&[frame]).expect("build offset frame");
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 1);
    let f = &img.frames[0];
    assert_eq!(f.duration_ms, 42);
    // Canvas dims = (x + w, y + h) = (4, 6).
    assert_eq!(f.width, 4);
    assert_eq!(f.height, 6);
    assert_eq!(f.rgba.len(), (4 * 6 * 4) as usize);
    // Sub-rect at (2,4) holds the input pixels exactly (Overwrite blend).
    let src = make_rgba(w, h, 5, true);
    for row in 0..h {
        for col in 0..w {
            let src_off = ((row * w + col) * 4) as usize;
            let dst_off = (((4 + row) * 4 + (2 + col)) * 4) as usize;
            assert_eq!(
                f.rgba[dst_off..dst_off + 4],
                src[src_off..src_off + 4],
                "sub-rect pixel ({col},{row}) round-trips byte-exact"
            );
        }
    }
    // Pixels outside the sub-rect are the ANIM bg (transparent black default).
    for row in 0..6u32 {
        for col in 0..4u32 {
            let in_sub = (2..4).contains(&col) && (4..6).contains(&row);
            if !in_sub {
                let off = ((row * 4 + col) * 4) as usize;
                assert_eq!(
                    &f.rgba[off..off + 4],
                    &[0, 0, 0, 0],
                    "outside-sub-rect pixel ({col},{row}) = ANIM bg"
                );
            }
        }
    }
}

#[test]
fn auto_and_delta_modes_round_trip_byte_exact() {
    // Round 127: Auto and Delta are no longer `Unsupported`. Both modes
    // encode the caller's frames against the previous canvas and the
    // file round-trips byte-for-byte through `decode_webp`'s canvas
    // compositor — same observable behaviour as Lossless, just with a
    // (potentially) smaller bitstream for slow-motion / cartoon content
    // where most pixels are unchanged frame-to-frame.
    let (w, h) = (8u32, 8u32);
    let base = make_rgba(w, h, 0, true);
    let mut moved = base.clone();
    // Twiddle a single pixel at (3, 4) so frame 1 differs from frame 0.
    moved[(4 * 8 + 3) * 4] ^= 0xff;

    for mode in [AnimFrameMode::Auto, AnimFrameMode::Delta] {
        let f0 = AnimFrame::new(w, h, base.clone(), 100);
        let mut f1 = AnimFrame::new(w, h, moved.clone(), 150);
        f1.mode = mode;
        let file = build_animated_webp(&[f0, f1]).expect("build animated webp");
        let img = decode_webp(&file).expect("decode animation");
        assert_eq!(img.frames.len(), 2, "{mode:?}: one WebpFrame per ANMF");
        assert_eq!(
            img.frames[0].rgba, base,
            "{mode:?}: frame 0 round-trips byte-exact"
        );
        assert_eq!(
            img.frames[1].rgba, moved,
            "{mode:?}: frame 1 round-trips byte-exact"
        );
        assert_eq!(img.frames[1].duration_ms, 150);
    }
}

#[test]
fn delta_mode_three_frames_round_trip_byte_exact() {
    // Three consecutive frames where only small per-frame regions differ
    // — the realistic Delta use case. Every frame round-trips byte-for-
    // byte and the encoded file is materially smaller than the same
    // sequence emitted as full Lossless keyframes.
    let (w, h) = (24u32, 24u32);
    let f0_px = make_rgba(w, h, 1, true);
    let mut f1_px = f0_px.clone();
    // Frame 1: change a 4×4 block at (10,10).
    for row in 10..14 {
        for col in 10..14 {
            let off = (row * w as usize + col) * 4;
            f1_px[off] ^= 0xff;
            f1_px[off + 1] ^= 0xff;
        }
    }
    // Frame 2: same as frame 1, but with an extra 4×4 change at (4,4).
    let mut f2_px = f1_px.clone();
    for row in 4..8 {
        for col in 4..8 {
            let off = (row * w as usize + col) * 4;
            f2_px[off + 2] ^= 0xff;
        }
    }

    let f0 = AnimFrame::new(w, h, f0_px.clone(), 80);
    let mut f1 = AnimFrame::new(w, h, f1_px.clone(), 80);
    f1.mode = AnimFrameMode::Delta;
    let mut f2 = AnimFrame::new(w, h, f2_px.clone(), 80);
    f2.mode = AnimFrameMode::Delta;

    let file = build_animated_webp(&[f0.clone(), f1.clone(), f2.clone()]).expect("build delta");
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 3);
    assert_eq!(img.frames[0].rgba, f0_px, "frame 0 round-trips");
    assert_eq!(img.frames[1].rgba, f1_px, "frame 1 round-trips");
    assert_eq!(img.frames[2].rgba, f2_px, "frame 2 round-trips");

    // Compare against an all-Lossless emit of the same content.
    let f1_l = {
        let mut x = AnimFrame::new(w, h, f1_px, 80);
        x.mode = AnimFrameMode::Lossless;
        x
    };
    let f2_l = {
        let mut x = AnimFrame::new(w, h, f2_px, 80);
        x.mode = AnimFrameMode::Lossless;
        x
    };
    let file_lossless = build_animated_webp(&[f0, f1_l, f2_l]).expect("build lossless");
    assert!(
        file.len() < file_lossless.len(),
        "3-frame delta ({} B) beats 3-frame lossless ({} B)",
        file.len(),
        file_lossless.len(),
    );
}

#[test]
fn auto_mode_picks_dirty_rect_on_small_localised_change() {
    // A 32×32 frame pair with a 2-pixel change in the middle. Auto
    // should produce a file noticeably smaller than the equivalent
    // Lossless re-encode of frame 1.
    let (w, h) = (32u32, 32u32);
    let base = make_rgba(w, h, 0, true);
    let mut moved = base.clone();
    let off = (16 * 32 + 16) * 4;
    moved[off] ^= 0xff;
    moved[off + 1] ^= 0xff;

    let f0_lossless = AnimFrame::new(w, h, base.clone(), 80);
    let mut f1_lossless = AnimFrame::new(w, h, moved.clone(), 80);
    f1_lossless.mode = AnimFrameMode::Lossless;
    let mut f1_auto = AnimFrame::new(w, h, moved, 80);
    f1_auto.mode = AnimFrameMode::Auto;

    let file_lossless =
        build_animated_webp(&[f0_lossless.clone(), f1_lossless]).expect("lossless build");
    let file_auto = build_animated_webp(&[f0_lossless, f1_auto]).expect("auto build");

    assert!(
        file_auto.len() < file_lossless.len(),
        "auto-mode ({} B) must beat lossless ({} B) on a 2-pixel change",
        file_auto.len(),
        file_lossless.len(),
    );
    // Avoid an unused import warning if WebpError stops being referenced.
    let _ = WebpError::InvalidData;
}

#[test]
fn empty_frame_list_is_invalid_data() {
    assert_eq!(build_animated_webp(&[]), Err(WebpError::InvalidData));
}

#[test]
fn delta_config_builder_methods_are_exposed() {
    let cfg = DeltaConfig::default()
        .max_components_override(4)
        .auto_inner_threshold_bytes(Some(256))
        .msssim_downsample_kernel(DownsampleKernel::Gaussian);
    assert_eq!(cfg.max_components, 4);
    assert_eq!(cfg.auto_inner_threshold_bytes, Some(256));
    assert_eq!(cfg.msssim_downsample_kernel, DownsampleKernel::Gaussian);
}
