//! Integration tests for the animated WebP encoder
//! ([`oxideav_webp::build_animated_webp`]).
//!
//! Two layers of validation are exercised:
//!
//! 1. **In-crate round-trip** — encode a 4-frame 32×32 RGBA animation,
//!    decode it back through this crate's own demuxer ([`decode_webp`]),
//!    and assert frame count + per-frame durations + bit-exact pixel
//!    contents.
//! 2. **External cross-validation** — when `webpinfo` and/or `dwebp` are
//!    available on the host, also run the encoded blob through them.
//!    `webpinfo` is asked to validate the file structure and the test
//!    asserts the chunk header is present in its output. The check
//!    degrades to a printed `skip:` line when the binaries aren't found.

use std::path::Path;
use std::process::Command;

use oxideav_webp::{build_animated_webp, decode_webp, AnimFrame};

const HOMEBREW_WEBPINFO: &str = "/opt/homebrew/bin/webpinfo";
const HOMEBREW_DWEBP: &str = "/opt/homebrew/bin/dwebp";
const HOMEBREW_WEBPMUX: &str = "/opt/homebrew/bin/webpmux";
const SYSTEM_WEBPINFO: &str = "/usr/bin/webpinfo";
const SYSTEM_DWEBP: &str = "/usr/bin/dwebp";
const SYSTEM_WEBPMUX: &str = "/usr/bin/webpmux";

fn webpinfo_path() -> Option<&'static str> {
    if Path::new(HOMEBREW_WEBPINFO).exists() {
        Some(HOMEBREW_WEBPINFO)
    } else if Path::new(SYSTEM_WEBPINFO).exists() {
        Some(SYSTEM_WEBPINFO)
    } else {
        None
    }
}

fn dwebp_path() -> Option<&'static str> {
    if Path::new(HOMEBREW_DWEBP).exists() {
        Some(HOMEBREW_DWEBP)
    } else if Path::new(SYSTEM_DWEBP).exists() {
        Some(SYSTEM_DWEBP)
    } else {
        None
    }
}

fn webpmux_path() -> Option<&'static str> {
    if Path::new(HOMEBREW_WEBPMUX).exists() {
        Some(HOMEBREW_WEBPMUX)
    } else if Path::new(SYSTEM_WEBPMUX).exists() {
        Some(SYSTEM_WEBPMUX)
    } else {
        None
    }
}

/// Build a `width × height` solid-colour RGBA frame.
fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        v.extend_from_slice(&rgba);
    }
    v
}

#[test]
fn anim_4frame_32x32_round_trips_through_in_crate_decoder() {
    let w = 32u32;
    let h = 32u32;
    let f0 = solid(w, h, [0xff, 0x00, 0x00, 0xff]);
    let f1 = solid(w, h, [0x00, 0xff, 0x00, 0xff]);
    let f2 = solid(w, h, [0x00, 0x00, 0xff, 0xff]);
    let f3 = solid(w, h, [0xff, 0xff, 0xff, 0xff]);
    let durations = [100u32, 100, 200, 100];
    let frames = [
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: durations[0],
            blend: false,
            dispose_to_background: false,
            rgba: &f0,
        },
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: durations[1],
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: durations[2],
            blend: false,
            dispose_to_background: false,
            rgba: &f2,
        },
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: durations[3],
            blend: false,
            dispose_to_background: false,
            rgba: &f3,
        },
    ];

    let blob = build_animated_webp(w, h, [0u8; 4], 0, &frames).expect("encode anim");
    // Sanity: file header looks right.
    assert_eq!(&blob[0..4], b"RIFF");
    assert_eq!(&blob[8..12], b"WEBP");
    assert_eq!(&blob[12..16], b"VP8X");

    // Decode back through the in-crate demuxer.
    let img = decode_webp(&blob).expect("decode anim roundtrip");
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.frames.len(), 4, "expected 4 ANMF frames");
    let originals: [&[u8]; 4] = [&f0, &f1, &f2, &f3];
    for (i, frame) in img.frames.iter().enumerate() {
        assert_eq!(
            frame.duration_ms, durations[i],
            "frame {i} duration mismatch: got {}, want {}",
            frame.duration_ms, durations[i]
        );
        assert_eq!(
            frame.rgba.len(),
            (w * h * 4) as usize,
            "frame {i} rgba length wrong"
        );
        // Lossless frame, blend=false → frame fills the canvas exactly
        // with the source pixels.
        assert_eq!(
            frame.rgba, originals[i],
            "frame {i} pixels don't match source"
        );
    }
}

#[test]
fn anim_validates_with_webpinfo_when_available() {
    let Some(webpinfo) = webpinfo_path() else {
        eprintln!("skip: webpinfo not on PATH");
        return;
    };

    let w = 16u32;
    let h = 16u32;
    let f0 = solid(w, h, [0xff, 0, 0, 0xff]);
    let f1 = solid(w, h, [0, 0xff, 0, 0xff]);
    let f2 = solid(w, h, [0, 0, 0xff, 0xff]);
    let f3 = solid(w, h, [0xff, 0xff, 0xff, 0xff]);
    let frames = [
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 100,
            blend: false,
            dispose_to_background: false,
            rgba: &f0,
        },
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 100,
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 200,
            blend: false,
            dispose_to_background: false,
            rgba: &f2,
        },
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 100,
            blend: false,
            dispose_to_background: false,
            rgba: &f3,
        },
    ];

    let blob = build_animated_webp(w, h, [0u8; 4], 0, &frames).expect("encode");
    let path = "/tmp/oxideav-webp-anim-test.webp";
    std::fs::write(path, &blob).expect("write blob");

    let out = Command::new(webpinfo)
        .arg(path)
        .output()
        .expect("invoke webpinfo");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "webpinfo failed: status {:?}\nstdout: {stdout}\nstderr: {stderr}",
        out.status
    );
    // Spot-check: webpinfo prints one "Chunk ANMF" line per ANMF chunk.
    let anmf_lines = stdout.matches("ANMF").count();
    assert_eq!(
        anmf_lines, 4,
        "expected 4 ANMF chunks reported by webpinfo, got {anmf_lines}\nfull output:\n{stdout}"
    );
    // Sanity: the validator should also report the VP8X + ANIM headers.
    assert!(
        stdout.contains("VP8X"),
        "webpinfo did not mention VP8X chunk\noutput:\n{stdout}"
    );
    assert!(
        stdout.contains("ANIM"),
        "webpinfo did not mention ANIM chunk\noutput:\n{stdout}"
    );
}

#[test]
fn anim_webpmux_extracts_each_frame_as_decodable_webp() {
    // libwebp's `dwebp` refuses animated input outright (exit code 4 /
    // UNSUPPORTED_FEATURE) and tells you to use webpmux to extract
    // individual frames. So the cross-validator is a two-stage pipeline:
    //   1. webpmux -get frame N → per-frame static .webp
    //   2. dwebp on each extracted file → solid-colour PPM
    let Some(webpmux) = webpmux_path() else {
        eprintln!("skip: webpmux not on PATH");
        return;
    };
    let Some(dwebp) = dwebp_path() else {
        eprintln!("skip: dwebp not on PATH");
        return;
    };

    let w = 16u32;
    let h = 16u32;
    let f0 = solid(w, h, [0xff, 0x80, 0x40, 0xff]);
    let f1 = solid(w, h, [0x10, 0x20, 0x30, 0xff]);
    let frames = [
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f0,
        },
        AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &f1,
        },
    ];
    let blob = build_animated_webp(w, h, [0u8; 4], 0, &frames).expect("encode");
    let path = "/tmp/oxideav-webp-anim-webpmux.webp";
    std::fs::write(path, &blob).expect("write");

    for frame_idx in 1..=2u32 {
        let extracted = format!("/tmp/oxideav-webp-anim-webpmux-frame{frame_idx}.webp");
        let out = Command::new(webpmux)
            .args([
                "-get",
                "frame",
                &frame_idx.to_string(),
                path,
                "-o",
                &extracted,
            ])
            .output()
            .expect("invoke webpmux");
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "webpmux frame {frame_idx} failed: {:?}\nstdout: {stdout}\nstderr: {stderr}",
            out.status
        );
        let extracted_size = std::fs::metadata(&extracted)
            .expect("extracted webp metadata")
            .len();
        assert!(
            extracted_size > 16,
            "webpmux extracted an unexpectedly tiny frame {frame_idx} ({extracted_size} bytes)"
        );

        // Now decode the extracted single-frame .webp with dwebp.
        let ppm = format!("/tmp/oxideav-webp-anim-webpmux-frame{frame_idx}.ppm");
        let out = Command::new(dwebp)
            .args([&extracted, "-quiet", "-o", &ppm])
            .output()
            .expect("invoke dwebp on extracted frame");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "dwebp on extracted frame {frame_idx} failed: {:?}\nstderr: {stderr}",
            out.status
        );
        let ppm_bytes = std::fs::read(&ppm).expect("read ppm");
        assert!(
            ppm_bytes.len() > 16,
            "dwebp produced an unexpectedly tiny PPM for frame {frame_idx} ({} bytes)",
            ppm_bytes.len()
        );
    }
}
