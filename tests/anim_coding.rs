//! Animation encode: lossy (`VP8 ` + `ALPH`) frames under a quality, and
//! frames encoded concurrently on the thread budget — the bytes must not
//! depend on it, for lossless and lossy frames and every frame mode.

mod common;

use std::time::Duration;

use oxideav_webp::{
    AnimFrame, AnimFrameMode, BlendingMethod, DisposalMethod, EncodeOptions, Frame, Metadata,
    WebpImage, decode_all, encode_animation, encode_animation_frames, info,
};

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

/// A photo-like canvas with a bright square at `(x0, y0)`: successive
/// frames differ in a small rectangle only.
fn scene(w: u32, h: u32, x0: u32, y0: u32) -> Vec<u8> {
    let mut rgba = common::photo_rgba(w, h);
    for y in y0..(y0 + 24).min(h) {
        for x in x0..(x0 + 24).min(w) {
            let i = ((y * w + x) * 4) as usize;
            rgba[i..i + 3].copy_from_slice(&[250, 240, 40]);
        }
    }
    rgba
}

fn scene_frames(w: u32, h: u32, n: u32) -> Vec<Frame> {
    (0..n)
        .map(|i| {
            let image =
                WebpImage::from_rgba8(w, h, scene(w, h, 8 + 14 * i, 10 + 6 * i)).expect("image");
            Frame::new(image, Some(Duration::from_millis(40 + 10 * u64::from(i))))
        })
        .collect()
}

#[test]
fn lossy_animation_round_trips_approximately() {
    let (w, h) = (128u32, 96u32);
    let frames = scene_frames(w, h, 5);
    for mode in [
        AnimFrameMode::Auto,
        AnimFrameMode::Delta,
        AnimFrameMode::Lossless,
    ] {
        let opts = EncodeOptions::default()
            .with_quality(85.0)
            .with_frame_mode(mode)
            .with_loop_count(2);
        let file = encode_animation(&frames, &opts).expect("lossy animation");
        let header = info(&file).expect("info");
        assert!(header.is_animated && header.is_lossy, "{mode:?}");
        assert_eq!((header.width, header.height), (w, h), "{mode:?}");
        assert_eq!(header.frames, 5, "{mode:?}");
        assert_eq!(header.loop_count, Some(2), "{mode:?}");
        assert!(
            !header.has_alpha,
            "{mode:?}: opaque frames need no alpha flag"
        );
        let back = decode_all(&file).expect("decode_all");
        assert_eq!(back.len(), frames.len(), "{mode:?}");
        for (i, (got, want)) in back.iter().zip(&frames).enumerate() {
            assert_eq!(got.delay, want.delay, "{mode:?} frame {i}");
            let db = psnr(&got.image.to_rgba8(), &want.image.to_rgba8());
            assert!(db > 32.0, "{mode:?} frame {i}: PSNR {db:.2} dB");
        }
    }
    // The dirty-rectangle modes code only the moving square after the
    // first frame; whole frames cost several times more.
    let size = |mode| {
        let opts = EncodeOptions::default()
            .with_quality(85.0)
            .with_frame_mode(mode);
        encode_animation(&frames, &opts).expect("encode").len()
    };
    let (auto, delta, whole) = (
        size(AnimFrameMode::Auto),
        size(AnimFrameMode::Delta),
        size(AnimFrameMode::Lossless),
    );
    // (Auto takes a small dirty rectangle without coding the whole
    // frame for comparison: here it is the delta file.)
    assert_eq!(auto, delta, "auto {auto} vs delta {delta}");
    assert!(2 * delta < whole, "delta {delta} vs whole frames {whole}");
    // And a lossy animation is much smaller than the lossless one.
    let lossless = encode_animation(&frames, &EncodeOptions::default())
        .expect("lossless")
        .len();
    assert!(2 * auto < lossless, "lossy {auto} vs lossless {lossless}");
}

#[test]
fn lossy_animation_keeps_alpha_exact() {
    let (w, h) = (64u32, 48u32);
    let frames: Vec<Frame> = (0..3u32)
        .map(|i| {
            let mut rgba = scene(w, h, 4 + 10 * i, 6);
            for (p, px) in rgba.chunks_mut(4).enumerate() {
                let (x, y) = (p as u32 % w, p as u32 / w);
                px[3] = if (x / 8 + y / 8 + i) % 3 == 0 {
                    0
                } else {
                    255 - (x * 2) as u8
                };
            }
            Frame::new(
                WebpImage::from_rgba8(w, h, rgba).expect("image"),
                Some(Duration::from_millis(50)),
            )
        })
        .collect();
    for mode in [AnimFrameMode::Auto, AnimFrameMode::Lossless] {
        let opts = EncodeOptions::default()
            .with_quality(80.0)
            .with_frame_mode(mode);
        let file = encode_animation(&frames, &opts).expect("lossy animation");
        let header = info(&file).expect("info");
        assert!(header.is_lossy && header.has_alpha, "{mode:?}");
        let back = decode_all(&file).expect("decode_all");
        for (i, (got, want)) in back.iter().zip(&frames).enumerate() {
            let (got, want) = (got.image.to_rgba8(), want.image.to_rgba8());
            assert!(
                got.chunks(4).zip(want.chunks(4)).all(|(g, w)| g[3] == w[3]),
                "{mode:?} frame {i}: alpha is carried losslessly"
            );
        }
    }
}

#[test]
fn positioned_lossy_frames_and_size_limit() {
    // A lossy sub-frame at an offset, alpha-blended over the first frame.
    let (w, h) = (80u32, 64u32);
    let base = AnimFrame::new(w, h, scene(w, h, 0, 0), 100);
    let mut patch_pixels = vec![0u8; 32 * 32 * 4];
    for (p, px) in patch_pixels.chunks_mut(4).enumerate() {
        let (x, y) = (p as u32 % 32, p as u32 / 32);
        px.copy_from_slice(&[200, 30, 60, if (x + y) % 2 == 0 { 255 } else { 0 }]);
    }
    let patch = AnimFrame::new(32, 32, patch_pixels, 100)
        .with_position(20, 16)
        .with_blend(BlendingMethod::AlphaBlend)
        .with_dispose(DisposalMethod::None)
        .with_mode(AnimFrameMode::Lossless);
    let opts = EncodeOptions::default().with_quality(90.0);
    let file =
        encode_animation_frames(&[base, patch], &Metadata::default(), &opts).expect("encode");
    let back = decode_all(&file).expect("decode_all");
    assert_eq!(back.len(), 2);
    let second = back[1].image.to_rgba8();
    // Inside the patch, opaque patch pixels are reddish; outside it the
    // canvas is what frame one left.
    let at = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        [second[i], second[i + 1], second[i + 2], second[i + 3]]
    };
    let first = back[0].image.to_rgba8();
    assert_eq!(&second[..4], &first[..4], "outside the patch: untouched");
    let inside = at(20, 16);
    assert!(inside[0] > 150 && inside[1] < 100, "patch pixel {inside:?}");

    // A VP8 frame is at most 16383 pixels per side.
    let wide = AnimFrame::new(16_384, 1, vec![0xff; 16_384 * 4], 10);
    assert!(
        encode_animation_frames(std::slice::from_ref(&wide), &Metadata::default(), &opts).is_err(),
        "a 16384-wide lossy frame is refused"
    );
    assert!(
        encode_animation_frames(&[wide], &Metadata::default(), &EncodeOptions::default()).is_ok(),
        "the same frame is fine lossless"
    );
}

/// Frames are encoded several at a time on the thread budget; the file
/// is the same for every budget, lossless and lossy, in every mode, with
/// a background-disposed frame in the sequence.
#[test]
fn animation_bytes_do_not_depend_on_the_thread_budget() {
    let (w, h) = (96u32, 72u32);
    for mode in [
        AnimFrameMode::Auto,
        AnimFrameMode::Delta,
        AnimFrameMode::Lossless,
    ] {
        let frames: Vec<AnimFrame> = (0..7u32)
            .map(|i| {
                let frame = AnimFrame::new(w, h, scene(w, h, 6 * i, 4 * i), 30 + i).with_mode(mode);
                if i == 3 {
                    frame.with_dispose(DisposalMethod::Background)
                } else {
                    frame
                }
            })
            .collect();
        for quality in [None, Some(75.0f32)] {
            let base = match quality {
                None => EncodeOptions::default().with_method(2),
                Some(q) => EncodeOptions::default().with_quality(q).with_method(1),
            };
            let serial = encode_animation_frames(
                &frames,
                &Metadata::default(),
                &base.clone().with_threads(1),
            )
            .expect("serial encode");
            for threads in [2usize, 3, 16, 0] {
                let file = encode_animation_frames(
                    &frames,
                    &Metadata::default(),
                    &base.clone().with_threads(threads),
                )
                .expect("threaded encode");
                assert!(
                    file == serial,
                    "{mode:?}, quality {quality:?}: {threads} threads changed the bytes"
                );
            }
            if quality.is_none() {
                let back = decode_all(&serial).expect("decode_all");
                assert_eq!(back.len(), frames.len(), "{mode:?}");
                // The last frame has no disposal after it: the canvas is
                // exactly the last frame's pixels.
                assert_eq!(
                    back[6].image.to_rgba8(),
                    frames[6].pixels,
                    "{mode:?}: lossless frames are exact"
                );
            }
        }
    }
}

/// A black-box container tool lists the frames of a lossy animation with
/// alpha as written (skipped when the tool is not installed).
#[test]
fn black_box_tool_reads_a_lossy_animation() {
    let found = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join("webpmux"))
            .find(|candidate| candidate.is_file())
    });
    let Some(tool) = found else {
        eprintln!("skip: no black-box WebP container tool on PATH");
        return;
    };
    let (w, h) = (64u32, 48u32);
    let frames: Vec<AnimFrame> = (0..3u32)
        .map(|i| {
            let mut rgba = scene(w, h, 8 * i, 4);
            for px in rgba.chunks_mut(4).take(200) {
                px[3] = 40;
            }
            AnimFrame::new(w, h, rgba, 70 + 10 * i).with_mode(AnimFrameMode::Lossless)
        })
        .collect();
    let file = encode_animation_frames(
        &frames,
        &Metadata::default(),
        &EncodeOptions::default().with_quality(75.0),
    )
    .expect("encode");
    let dir = std::env::temp_dir().join(format!("oxideav-webp-anim-coding-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("lossy-anim.webp");
    std::fs::write(&path, &file).expect("write");
    let output = std::process::Command::new(tool)
        .arg("-info")
        .arg(&path)
        .output()
        .expect("run the container tool");
    let _ = std::fs::remove_dir_all(&dir);
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "container tool refused the file: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains("Number of frames: 3"),
        "container tool output:\n{text}"
    );
    // One row per frame at the written duration; where the tool prints a
    // compression column, it says lossy for each of them.
    let rows: Vec<Vec<&str>> = text
        .lines()
        .map(|line| line.split_whitespace().collect())
        .collect();
    for duration in ["70", "80", "90"] {
        let row = rows
            .iter()
            .find(|fields| {
                fields
                    .first()
                    .is_some_and(|f| f.trim_end_matches(':').parse::<u32>().is_ok())
                    && fields.contains(&duration)
            })
            .unwrap_or_else(|| panic!("no frame row with duration {duration}:\n{text}"));
        assert!(
            !row.contains(&"lossless"),
            "frame with duration {duration} listed as lossless:\n{text}"
        );
    }
}
