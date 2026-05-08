//! Integration tests for the streaming `WebpAnimDecoder` API.
//!
//! Inline unit tests in `src/anim_decoder.rs` cover the basic ordering /
//! reset / dispose / info paths against in-test-built blobs. This file
//! adds end-to-end tests that exercise the public API surface re-exported
//! from the crate root, the way an external consumer would call it.
//!
//! Coverage:
//!
//! * **Public re-exports** — `WebpAnimDecoder::new(bytes)` works through
//!   the crate root; `WebpAnimFrame` / `WebpAnimInfo` are constructible
//!   without going through `anim_decoder::*` paths.
//! * **Streaming consumer scenario** — pull only the first 2 of 5
//!   frames; verify subsequent frames remain undecoded (`frame_count`
//!   knows the total, `next_frame_index` lags behind).
//! * **PTS arithmetic vs. eager decoder** — every streamed frame's
//!   pixel buffer matches the eager `decode_webp` path's same-index
//!   frame byte-for-byte; cumulative durations match the spec's PTS
//!   sum (with the `1` ms floor for zero-duration frames).
//! * **Metadata + bg colour parity** — `info()` exposes the same
//!   `metadata` / `loop_count` / `background_rgba` fields that
//!   `decode_webp` returns on `WebpImage`.

use oxideav_webp::encoder_anim::{
    build_animated_webp, build_animated_webp_with_options, AnimEncoderOptions, AnimFrame,
};
use oxideav_webp::{decode_webp, WebpAnimDecoder, WebpMetadata};

const W: u32 = 8;
const H: u32 = 8;

fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let n = (width as usize) * (height as usize);
    let mut v = Vec::with_capacity(n * 4);
    for _ in 0..n {
        v.extend_from_slice(&rgba);
    }
    v
}

fn five_frame_anim() -> Vec<u8> {
    // Five solid-colour full-canvas frames with varied durations.
    let frames_data: Vec<(Vec<u8>, u32)> = vec![
        (solid(W, H, [0xff, 0, 0, 0xff]), 30),
        (solid(W, H, [0, 0xff, 0, 0xff]), 40),
        (solid(W, H, [0, 0, 0xff, 0xff]), 50),
        (solid(W, H, [0xff, 0xff, 0, 0xff]), 0), // duration 0 → PTS floor
        (solid(W, H, [0, 0xff, 0xff, 0xff]), 60),
    ];
    let frames: Vec<AnimFrame<'_>> = frames_data
        .iter()
        .map(|(data, dur)| AnimFrame {
            width: W,
            height: H,
            x_offset: 0,
            y_offset: 0,
            duration_ms: *dur,
            blend: false,
            dispose_to_background: false,
            rgba: data,
        })
        .collect();
    build_animated_webp(W, H, [0, 0, 0, 0], 0, &frames).expect("encode")
}

#[test]
fn streaming_decoder_visible_through_crate_root() {
    // Sanity test: the public `WebpAnimDecoder` re-export at the crate
    // root is what an external consumer would import. Should construct
    // without going through `oxideav_webp::anim_decoder::*`.
    let blob = five_frame_anim();
    let dec = WebpAnimDecoder::new(&blob).expect("new");
    assert_eq!(dec.info().frame_count, 5);
}

#[test]
fn early_stop_after_two_frames_keeps_remainder_undecoded() {
    let blob = five_frame_anim();
    let mut dec = WebpAnimDecoder::new(&blob).expect("new");
    let _f0 = dec.next_frame().expect("ok").expect("Some");
    let _f1 = dec.next_frame().expect("ok").expect("Some");
    // The streaming guarantee: we *can* stop here without paying for
    // frames 2..5. We can't observe the absence of work, but the
    // decoder's internal cursor does:
    assert_eq!(dec.next_frame_index(), 2);
    assert_eq!(dec.info().frame_count, 5);
    assert!(!dec.done());
}

#[test]
fn streamed_pixel_buffers_match_eager_decoder_byte_for_byte() {
    let blob = five_frame_anim();
    let eager = decode_webp(&blob).expect("eager decode");
    let mut dec = WebpAnimDecoder::new(&blob).expect("streaming new");
    assert_eq!(dec.info().frame_count, eager.frames.len());
    for i in 0..eager.frames.len() {
        let stream_f = dec.next_frame().expect("ok").expect("Some");
        let eager_f = &eager.frames[i];
        assert_eq!(
            stream_f.rgba, eager_f.rgba,
            "frame {i} canvas bytes diverge between streaming + eager paths"
        );
        assert_eq!(stream_f.duration_ms, eager_f.duration_ms);
    }
    assert!(dec.next_frame().expect("ok").is_none());
}

#[test]
fn pts_advances_with_one_ms_floor_for_zero_duration_frames() {
    let blob = five_frame_anim();
    let mut dec = WebpAnimDecoder::new(&blob).expect("new");
    // PTS sequence: 0, 30, 70, 120 (50 + 70), 121 (max(0,1)+120),
    //               181 (60 + 121).
    let f0 = dec.next_frame().expect("ok").expect("Some");
    assert_eq!(f0.pts_ms, 0);
    let f1 = dec.next_frame().expect("ok").expect("Some");
    assert_eq!(f1.pts_ms, 30);
    let f2 = dec.next_frame().expect("ok").expect("Some");
    assert_eq!(f2.pts_ms, 70);
    let f3 = dec.next_frame().expect("ok").expect("Some");
    assert_eq!(f3.pts_ms, 120);
    let f4 = dec.next_frame().expect("ok").expect("Some");
    // f3.duration_ms was 0 → pts_ms gains 1 (the spec-floor) → f4 = 121.
    assert_eq!(f4.pts_ms, 121);
}

#[test]
fn info_metadata_matches_eager_webpimage_fields() {
    // Build with all three metadata chunks + a non-zero BG + non-default
    // loop count, then verify `WebpAnimDecoder::info` and
    // `WebpImage` fields agree.
    let icc = b"VP8L_streaming_test_ICC".to_vec();
    let exif = b"II*\0streaming-test-EXIF".to_vec();
    let xmp = b"<?xml version=\"1.0\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec();
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: Some(&exif),
        xmp: Some(&xmp),
    };
    let bg_bgra: [u8; 4] = [0x10, 0x20, 0x30, 0xff];
    let red_tile = solid(4, 4, [0xff, 0, 0, 0xff]);
    let frames = [AnimFrame {
        width: 4,
        height: 4,
        x_offset: 0,
        y_offset: 0,
        duration_ms: 50,
        blend: false,
        dispose_to_background: false,
        rgba: &red_tile,
    }];
    let opts = AnimEncoderOptions {
        metadata: meta,
        ..AnimEncoderOptions::default()
    };
    let blob = build_animated_webp_with_options(W, H, bg_bgra, 13, &frames, opts).expect("encode");

    let img = decode_webp(&blob).expect("decode");
    let dec = WebpAnimDecoder::new(&blob).expect("streaming new");
    let info = dec.info();
    assert_eq!(info.canvas_width, img.width);
    assert_eq!(info.canvas_height, img.height);
    assert_eq!(info.frame_count, img.frames.len());
    assert_eq!(info.loop_count, img.anim_loop_count);
    assert_eq!(info.background_rgba, img.anim_background_rgba);
    assert_eq!(info.metadata.icc, img.metadata.icc);
    assert_eq!(info.metadata.exif, img.metadata.exif);
    assert_eq!(info.metadata.xmp, img.metadata.xmp);
}
