//! Round-169 end-to-end cross-validation against external WebP tools.
//!
//! Each test spawns a third-party command-line decoder / muxing tool as
//! an **opaque** byte-in / byte-out process (no source code consulted)
//! and asserts the bytes line up against our own encoder / decoder.
//! This proves our `.webp` output is universally readable and that our
//! decoder agrees with a reference implementation on fixtures we did not
//! produce ourselves.
//!
//! ## Oracle directions
//!
//! * **A — our lossless encode → reference decoder → compare.**
//!   Synthetic RGBA → `encode_webp_lossless` → temp `.webp` → spawn the
//!   reference WebP decoder (`-pam` raw RGBA output) → strip the PAM
//!   ASCII header → assert byte-for-byte equal to the source RGBA.
//!   Lossless = bit-exact, so the budget is zero tolerance.
//! * **B — our animation encode → reference muxing tool → frame count
//!   match.** 3-frame animation → `build_animated_webp` → temp `.webp`
//!   → spawn the reference muxing tool's `-info` command → parse the
//!   "Number of frames" line and the per-frame width/height/duration
//!   table → assert all three frames present with our chosen geometry
//!   and duration.
//! * **C — reference-encoded fixture → our decode → ffmpeg decode →
//!   compare.** The `tests/data/lossless-32x32-rgba.webp` fixture (a
//!   byte-for-byte copy of a reference-tool encode) → `decode_webp` →
//!   spawn `ffmpeg -f rawvideo -pix_fmt rgba` decode of the same file →
//!   assert the two RGBA buffers are byte-identical.
//!
//! ## Skip semantics
//!
//! Each direction gracefully skips when its oracle binary is not on the
//! `PATH`: `eprintln!("skip: …")` then `return` — never `#[ignore]`.
//! On a host with all four binaries installed (the WebP reference tools
//! plus ffmpeg), all three directions run.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use oxideav_webp::{build_animated_webp, decode_webp, encode_webp_lossless, AnimFrame};

// ──────────────────────────── shared utilities ──────────────────────────

/// Look up an executable on `PATH`. Returns `None` when the binary is not
/// installed (the test then skips cleanly).
fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            // Best-effort executability check — on Unix the OS will refuse
            // to spawn a non-executable file anyway, so `is_file` is enough.
            return Some(candidate);
        }
    }
    None
}

/// Build a deterministic `width * height` RGBA8 image. Spread-out arithmetic
/// so every channel hits a wide range of values, matching the standalone-API
/// fixture builder.
fn synthetic_rgba(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x.wrapping_mul(37).wrapping_add(y).wrapping_add(seed) & 0xff) as u8;
            let g = (y.wrapping_mul(53).wrapping_add(x).wrapping_mul(7) & 0xff) as u8;
            let b = ((x ^ y).wrapping_mul(101).wrapping_add(seed) & 0xff) as u8;
            let a = (255 - ((x.wrapping_add(y).wrapping_add(seed)) & 0xff)) as u8;
            buf.extend_from_slice(&[r, g, b, a]);
        }
    }
    buf
}

/// Pick a fresh per-test directory under `target/tmp/` (or `$TMPDIR`), so
/// parallel test invocations don't collide on filenames. The directory is
/// not removed at the end — it's tiny and `cargo clean` clears it.
fn tmp_dir(tag: &str) -> PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let dir = base.join(format!("oxideav-webp-oracle-{tag}-{pid}-{nano}"));
    std::fs::create_dir_all(&dir).expect("create tmp dir");
    dir
}

/// Parse a `P7` (PAM) image. Returns the raw RGBA bytes (header stripped),
/// plus the declared width / height / depth. Only the layout the reference
/// decoder's `-pam` mode emits is supported: `P7` magic, ASCII key-value
/// header lines, `ENDHDR\n` terminator, then `width * height * depth` raw
/// bytes (depth=4 for RGBA, depth=3 for RGB).
fn parse_pam(bytes: &[u8]) -> (Vec<u8>, u32, u32, u32) {
    assert_eq!(
        &bytes[0..3],
        b"P7\n",
        "PAM file must start with the `P7` magic"
    );
    let endhdr = b"ENDHDR\n";
    let end = bytes
        .windows(endhdr.len())
        .position(|w| w == endhdr)
        .expect("ENDHDR\\n terminator");
    let header = std::str::from_utf8(&bytes[0..end]).expect("header is ASCII");
    let mut width = 0u32;
    let mut height = 0u32;
    let mut depth = 0u32;
    for line in header.lines() {
        if let Some(v) = line.strip_prefix("WIDTH ") {
            width = v.trim().parse().expect("width");
        } else if let Some(v) = line.strip_prefix("HEIGHT ") {
            height = v.trim().parse().expect("height");
        } else if let Some(v) = line.strip_prefix("DEPTH ") {
            depth = v.trim().parse().expect("depth");
        }
    }
    assert!(width > 0 && height > 0 && depth > 0, "PAM header populated");
    let data = bytes[end + endhdr.len()..].to_vec();
    assert_eq!(
        data.len(),
        (width * height * depth) as usize,
        "PAM payload length matches header"
    );
    (data, width, height, depth)
}

/// Run a command and panic with a descriptive message if it doesn't exit 0.
/// Returns the captured stdout bytes.
fn run_or_fail(cmd: &mut Command) -> Vec<u8> {
    let label = format!("{cmd:?}");
    let out = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("spawn {label}: {e}"));
    if !out.status.success() {
        panic!(
            "{label} failed: status={:?}\nstdout: {}\nstderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
    out.stdout
}

// ─────────────── Direction A: our lossless encode → reference decode ────────

#[test]
fn direction_a_our_lossless_encode_is_readable_by_reference_decoder() {
    // Skip cleanly if the reference WebP decoder isn't installed.
    let Some(decoder_bin) = which("dwebp") else {
        eprintln!("skip: `dwebp` not on PATH — install the WebP reference tools to exercise");
        return;
    };

    // Synthesise a 96×48 lossless RGBA image and encode it.
    let (w, h) = (96u32, 48u32);
    let src = synthetic_rgba(w, h, 0);
    let file = encode_webp_lossless(&src, w, h).expect("encode_webp_lossless");

    // Drop the encoded `.webp` into a temp file the reference decoder can read.
    let dir = tmp_dir("direction-a");
    let in_path = dir.join("our.webp");
    let out_path = dir.join("ref.pam");
    std::fs::write(&in_path, &file).expect("write our.webp");

    // Run the reference decoder in `-pam` raw-RGBA mode.
    let mut cmd = Command::new(&decoder_bin);
    cmd.arg(&in_path)
        .arg("-pam")
        .arg("-o")
        .arg(&out_path)
        .arg("-quiet");
    let _ = run_or_fail(&mut cmd);

    let pam_bytes = std::fs::read(&out_path).expect("read decoded PAM");
    let (rgba, ref_w, ref_h, depth) = parse_pam(&pam_bytes);
    assert_eq!(ref_w, w, "reference decoder agrees on width");
    assert_eq!(ref_h, h, "reference decoder agrees on height");
    assert_eq!(depth, 4, "PAM depth must be 4 for our RGBA encode");
    assert_eq!(
        rgba.len(),
        src.len(),
        "reference decoder produced the right number of bytes"
    );
    assert_eq!(
        rgba, src,
        "our lossless encode round-trips bit-exact through the reference decoder"
    );
}

/// Round 383 — direction A over the content regimes that trigger the
/// round-383 encoder machinery: two-regime content (smooth + noisy →
/// the §6.2.2 multi-group entropy image from the entropy-merge
/// partition), palette content (§4.4 color indexing across the
/// palette-ordering sweep), unit-slope channel-correlated content
/// (§4.3 subtract-green → §4.1 predictor stack), and run/noise-mixed
/// content (the cost-priced DP token planner). Each encode must decode
/// bit-exactly through the reference decoder, proving the new wire
/// shapes are universally readable.
#[test]
fn direction_a_round_383_encoder_paths_are_readable_by_reference_decoder() {
    let Some(decoder_bin) = which("dwebp") else {
        eprintln!("skip: `dwebp` not on PATH — install the WebP reference tools to exercise");
        return;
    };

    let (w, h) = (96u32, 80u32);
    let mut regimes: Vec<(&str, Vec<u8>)> = Vec::new();

    // (1) Two-regime: smooth gradient top half, noise bottom half.
    let mut two_regime = Vec::with_capacity((w * h * 4) as usize);
    let mut state = 0x1357_9bdfu32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if y < h / 2 {
                let g = ((x + 2 * y) % 256) as u8;
                two_regime.extend_from_slice(&[g, g, g, 0xff]);
            } else {
                two_regime.extend_from_slice(&[
                    (state >> 8) as u8,
                    (state >> 16) as u8,
                    state as u8,
                    0xff,
                ]);
            }
        }
    }
    regimes.push(("two-regime", two_regime));

    // (2) Palette: 12 colors in flat regions + dithered stripes.
    let palette: [[u8; 4]; 12] = [
        [0x10, 0x20, 0x30, 0xff],
        [0x40, 0x50, 0x60, 0xff],
        [0x70, 0x80, 0x90, 0xff],
        [0xa0, 0xb0, 0xc0, 0xff],
        [0xd0, 0xe0, 0xf0, 0xff],
        [0x01, 0x02, 0x03, 0xff],
        [0xff, 0x00, 0x00, 0xff],
        [0x00, 0xff, 0x00, 0xff],
        [0x00, 0x00, 0xff, 0xff],
        [0xff, 0xff, 0x00, 0xff],
        [0x00, 0xff, 0xff, 0xff],
        [0xff, 0x00, 0xff, 0xff],
    ];
    let mut paletted = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let idx = if x < w / 2 {
                (y / 10) as usize % 6
            } else {
                6 + (((x ^ y) & 1) as usize) + 2 * ((y / 20) as usize % 3)
            };
            paletted.extend_from_slice(&palette[idx]);
        }
    }
    regimes.push(("paletted", paletted));

    // (3) Unit-slope channel correlation with green-carried noise.
    let mut unit_slope = Vec::with_capacity((w * h * 4) as usize);
    let mut state2 = 0x0f1e_2d3cu32;
    for y in 0..h {
        for x in 0..w {
            state2 = state2.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let n = (state2 >> 26) as i32 - 32;
            let g = ((x * 2 + y) as i32 + n).rem_euclid(256) as u8;
            unit_slope.extend_from_slice(&[g.wrapping_add(40), g, g.wrapping_add(231), 0xff]);
        }
    }
    regimes.push(("unit-slope", unit_slope));

    // (4) Alternating exact-repeat rows and noise rows (DP planner bait).
    let mut runs_noise = Vec::with_capacity((w * h * 4) as usize);
    let mut state3 = 0xfeed_beefu32;
    for y in 0..h {
        for x in 0..w {
            if y % 2 == 0 {
                let g = ((x * 3) % 256) as u8;
                runs_noise.extend_from_slice(&[g, g, g, 0xff]);
            } else {
                state3 = state3.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                runs_noise.extend_from_slice(&[
                    (state3 >> 8) as u8,
                    (state3 >> 16) as u8,
                    state3 as u8,
                    0xff,
                ]);
            }
        }
    }
    regimes.push(("runs-noise", runs_noise));

    let dir = tmp_dir("direction-a-r383");
    for (tag, src) in regimes {
        let file = encode_webp_lossless(&src, w, h).expect("encode_webp_lossless");
        let in_path = dir.join(format!("{tag}.webp"));
        let out_path = dir.join(format!("{tag}.pam"));
        std::fs::write(&in_path, &file).expect("write our.webp");
        let mut cmd = Command::new(&decoder_bin);
        cmd.arg(&in_path)
            .arg("-pam")
            .arg("-o")
            .arg(&out_path)
            .arg("-quiet");
        let _ = run_or_fail(&mut cmd);
        let pam_bytes = std::fs::read(&out_path).expect("read decoded PAM");
        let (rgba, ref_w, ref_h, depth) = parse_pam(&pam_bytes);
        assert_eq!((ref_w, ref_h, depth), (w, h, 4), "{tag}: geometry");
        assert_eq!(
            rgba, src,
            "{tag}: our encode must round-trip bit-exact through the reference decoder"
        );
    }
}

// ───────────── Direction B: our animation encode → reference mux info ───────

/// Parse the reference muxing tool's `-info` output. We only assert on the
/// structured fields (frame count + per-frame width/height/duration) — the
/// surrounding ASCII layout has been stable for years across reference-tool
/// releases.
fn parse_mux_info(stdout: &str) -> (u32, Vec<(u32, u32, u32)>) {
    let mut count = 0u32;
    let mut frames: Vec<(u32, u32, u32)> = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("Number of frames: ") {
            count = rest.trim().parse().expect("frame count");
        } else {
            // Per-frame rows look like:
            //   "  1:    64    64    no        0        0      100  ..."
            // i.e. `<idx>:` then whitespace-separated numbers; we read
            // the first three after the colon as width / height / x_offset
            // and the sixth as duration.
            if let Some((idx, rest)) = trimmed.split_once(':') {
                if idx.trim().parse::<u32>().is_ok() {
                    let cols: Vec<&str> = rest.split_whitespace().collect();
                    if cols.len() >= 6 {
                        // Defensive parse: skip non-numeric "alpha"/"yes"/"no"
                        // column at index 2 — we only need cols 0/1 (w/h)
                        // and 5 (duration).
                        if let (Ok(w), Ok(h), Ok(dur)) = (
                            cols[0].parse::<u32>(),
                            cols[1].parse::<u32>(),
                            cols[5].parse::<u32>(),
                        ) {
                            frames.push((w, h, dur));
                        }
                    }
                }
            }
        }
    }
    (count, frames)
}

#[test]
fn direction_b_our_animation_is_readable_by_reference_mux_tool() {
    let Some(mux_bin) = which("webpmux") else {
        eprintln!("skip: `webpmux` not on PATH — install the WebP reference tools to exercise");
        return;
    };

    let (w, h) = (40u32, 40u32);
    let frames = vec![
        AnimFrame::new(w, h, synthetic_rgba(w, h, 0), 80),
        AnimFrame::new(w, h, synthetic_rgba(w, h, 13), 120),
        AnimFrame::new(w, h, synthetic_rgba(w, h, 42), 160),
    ];
    let file = build_animated_webp(&frames).expect("build_animated_webp");

    let dir = tmp_dir("direction-b");
    let in_path = dir.join("anim.webp");
    std::fs::write(&in_path, &file).expect("write anim.webp");

    let mut cmd = Command::new(&mux_bin);
    cmd.arg("-info").arg(&in_path);
    let stdout = run_or_fail(&mut cmd);
    let stdout_str = String::from_utf8_lossy(&stdout);

    let (count, parsed_frames) = parse_mux_info(&stdout_str);
    assert_eq!(
        count, 3,
        "reference mux tool reports 3 frames; full output:\n{stdout_str}",
    );
    assert!(
        parsed_frames.len() >= 3,
        "parsed >=3 frame rows; full output:\n{stdout_str}",
    );
    for (i, (got_w, got_h, got_dur)) in parsed_frames.iter().take(3).enumerate() {
        assert_eq!(*got_w, w, "frame {i} width via mux tool");
        assert_eq!(*got_h, h, "frame {i} height via mux tool");
        let expected_dur = [80u32, 120, 160][i];
        assert_eq!(*got_dur, expected_dur, "frame {i} duration via mux tool",);
    }
}

// ── Direction C: reference-encoded fixture → our decode vs. ffmpeg decode ──

/// The pre-encoded 32×32 RGBA lossless fixture (a byte-for-byte copy of
/// `docs/image/webp/fixtures/lossless-32x32-rgba/input.webp` — not produced
/// by our encoder, so it's a genuine cross-codec check).
const FIXTURE_LOSSLESS_32X32: &[u8] = include_bytes!("data/lossless-32x32-rgba.webp");

#[test]
fn direction_c_reference_fixture_matches_ffmpeg_decode_byte_for_byte() {
    let Some(ffmpeg_bin) = which("ffmpeg") else {
        eprintln!("skip: `ffmpeg` not on PATH — install ffmpeg to exercise this oracle");
        return;
    };

    // Step 1: decode the fixture with our crate.
    let img = decode_webp(FIXTURE_LOSSLESS_32X32).expect("our decode of fixture");
    assert_eq!(img.frames.len(), 1, "still image: one frame");
    let our_rgba = &img.frames[0].rgba;
    let w = img.frames[0].width;
    let h = img.frames[0].height;
    assert_eq!(
        our_rgba.len(),
        (w * h * 4) as usize,
        "flat tight RGBA buffer",
    );

    // Step 2: drop the fixture to a temp file and decode it with ffmpeg as
    // raw RGBA (pixel format `rgba` is the same channel order our crate
    // produces: R G B A interleaved).
    let dir = tmp_dir("direction-c");
    let in_path = dir.join("fixture.webp");
    let out_path = dir.join("ffmpeg.rgba");
    std::fs::write(&in_path, FIXTURE_LOSSLESS_32X32).expect("write fixture");

    let mut cmd = Command::new(&ffmpeg_bin);
    cmd.arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(&in_path)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgba")
        .arg(&out_path);
    let _ = run_or_fail(&mut cmd);

    let ffmpeg_rgba = std::fs::read(&out_path).expect("read ffmpeg output");
    assert_eq!(
        ffmpeg_rgba.len(),
        our_rgba.len(),
        "ffmpeg produced the same number of RGBA bytes as our decoder",
    );
    assert_eq!(
        ffmpeg_rgba, *our_rgba,
        "our decoder and ffmpeg agree on every RGBA byte of the reference fixture",
    );

    // Extra: any installed reference decoder must also agree.
    if let Some(decoder_bin) = which("dwebp") {
        let pam_path = dir.join("ref.pam");
        let mut cmd = Command::new(&decoder_bin);
        cmd.arg(&in_path)
            .arg("-pam")
            .arg("-o")
            .arg(&pam_path)
            .arg("-quiet");
        let _ = run_or_fail(&mut cmd);
        let pam = std::fs::read(&pam_path).expect("read pam");
        let (ref_rgba, ref_w, ref_h, depth) = parse_pam(&pam);
        assert_eq!(ref_w, w);
        assert_eq!(ref_h, h);
        assert_eq!(depth, 4);
        assert_eq!(
            ref_rgba, *our_rgba,
            "our decoder and the reference decoder agree on every byte",
        );
    }
}

// Used only when an oracle is missing — the `Path` import would otherwise
// be unreferenced under the skip-only-path. Keeping a touch on `Path` lets
// the file compile cleanly regardless of which oracles are installed.
#[allow(dead_code)]
fn _touch_path_import(p: &Path) -> &Path {
    p
}
