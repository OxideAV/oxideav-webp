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
    // produce a parseable file whose decoded duration matches.
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
    };
    let file = build_animated_webp(&[frame]).expect("build offset frame");
    let img = decode_webp(&file).expect("decode");
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].duration_ms, 42);
    assert_eq!(img.frames[0].rgba, make_rgba(w, h, 5, true));
}

#[test]
fn auto_and_delta_modes_report_unsupported() {
    let (w, h) = (2u32, 2u32);
    let mut frame = AnimFrame::new(w, h, make_rgba(w, h, 0, true), 100);
    frame.mode = AnimFrameMode::Auto;
    assert_eq!(
        build_animated_webp(&[frame.clone()]),
        Err(WebpError::Unsupported)
    );
    frame.mode = AnimFrameMode::Delta;
    assert_eq!(build_animated_webp(&[frame]), Err(WebpError::Unsupported));
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
