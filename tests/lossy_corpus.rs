//! Integration test: decode the docs/image/webp/fixtures/lossy-* corpus
//! and compare against the libwebp-encoded `expected.png` ground truth.
//!
//! Each fixture directory under `tests/fixtures/lossy_corpus/<name>/`
//! mirrors a directory in `docs/image/webp/fixtures/<name>/` (the
//! workspace-level corpus produced by an instrumented libwebp build).
//! For this test we only need `input.webp` (our decoder's input) and
//! `expected.png` (the libwebp ground truth, decoded via `oxideav-png`).
//! The matching `trace.txt` and `notes.md` documents in the workspace
//! corpus capture the *expected* per-step decoder events from the
//! instrumented libwebp run — a divergence here is cross-checked against
//! that trace to localise the bug.
//!
//! Tiers:
//!   * `BitExact`   — every channel of every pixel must match.
//!   * `ReportOnly` — divergences are recorded and printed but do NOT
//!                    fail the test. Used for fixtures we know exercise
//!                    paths still under construction (loop filter,
//!                    quantizer extremes, ALPH overlay, …).
//!   * `Ignored`    — fixture is loaded and reported but no comparison
//!                    is performed (e.g. our decoder errors out and we
//!                    just want a categorised log line).
//!
//! The test deliberately does NOT modify the decoder. Bugs surfaced here
//! become separate follow-up tasks.
//!
//! The `expected.png` files in this corpus are 8-bit Rgb24 for the five
//! opaque fixtures and 8-bit Rgba for `lossy-with-alpha-128x128`. Our
//! `decode_webp` always returns RGBA, so the compare path expands an
//! Rgb24 reference into RGBA (alpha = 255) before per-channel diffing.

use oxideav_png::decode_png_to_frame;
use oxideav_webp::decode_webp;

/// How strictly we expect a fixture to decode. Start every fixture at
/// `ReportOnly`; promote to `BitExact` once we've verified bit-for-bit
/// agreement on CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    BitExact,
    ReportOnly,
    #[allow(dead_code)]
    Ignored,
}

struct Fixture {
    name: &'static str,
    webp: &'static [u8],
    png: &'static [u8],
    tier: Tier,
    /// Path of the matching workspace trace file (relative to the
    /// repository root). Printed alongside divergences as a debugging
    /// breadcrumb.
    trace_doc: &'static str,
}

const LOSSY_1X1: Fixture = Fixture {
    name: "lossy-1x1",
    webp: include_bytes!("fixtures/lossy_corpus/lossy-1x1/input.webp"),
    png: include_bytes!("fixtures/lossy_corpus/lossy-1x1/expected.png"),
    tier: Tier::ReportOnly,
    trace_doc: "docs/image/webp/fixtures/lossy-1x1/trace.txt",
};

const LOSSY_128_Q1: Fixture = Fixture {
    name: "lossy-128x128-q1",
    webp: include_bytes!("fixtures/lossy_corpus/lossy-128x128-q1/input.webp"),
    png: include_bytes!("fixtures/lossy_corpus/lossy-128x128-q1/expected.png"),
    tier: Tier::ReportOnly,
    trace_doc: "docs/image/webp/fixtures/lossy-128x128-q1/trace.txt",
};

const LOSSY_128_Q75: Fixture = Fixture {
    name: "lossy-128x128-q75",
    webp: include_bytes!("fixtures/lossy_corpus/lossy-128x128-q75/input.webp"),
    png: include_bytes!("fixtures/lossy_corpus/lossy-128x128-q75/expected.png"),
    tier: Tier::ReportOnly,
    trace_doc: "docs/image/webp/fixtures/lossy-128x128-q75/trace.txt",
};

const LOSSY_128_Q100: Fixture = Fixture {
    name: "lossy-128x128-q100",
    webp: include_bytes!("fixtures/lossy_corpus/lossy-128x128-q100/input.webp"),
    png: include_bytes!("fixtures/lossy_corpus/lossy-128x128-q100/expected.png"),
    tier: Tier::ReportOnly,
    trace_doc: "docs/image/webp/fixtures/lossy-128x128-q100/trace.txt",
};

const LOSSY_NEAR_LOSSLESS_Q40: Fixture = Fixture {
    name: "lossy-near-lossless-q40",
    webp: include_bytes!("fixtures/lossy_corpus/lossy-near-lossless-q40/input.webp"),
    png: include_bytes!("fixtures/lossy_corpus/lossy-near-lossless-q40/expected.png"),
    tier: Tier::ReportOnly,
    trace_doc: "docs/image/webp/fixtures/lossy-near-lossless-q40/trace.txt",
};

const LOSSY_WITH_ALPHA: Fixture = Fixture {
    name: "lossy-with-alpha-128x128",
    webp: include_bytes!("fixtures/lossy_corpus/lossy-with-alpha-128x128/input.webp"),
    png: include_bytes!("fixtures/lossy_corpus/lossy-with-alpha-128x128/expected.png"),
    tier: Tier::ReportOnly,
    trace_doc: "docs/image/webp/fixtures/lossy-with-alpha-128x128/trace.txt",
};

const FIXTURES: &[Fixture] = &[
    LOSSY_1X1,
    LOSSY_128_Q1,
    LOSSY_128_Q75,
    LOSSY_128_Q100,
    LOSSY_NEAR_LOSSLESS_Q40,
    LOSSY_WITH_ALPHA,
];

/// Per-channel comparison stats — counted independently so we can spot
/// e.g. "RGB matches but A doesn't" or vice-versa.
#[derive(Debug, Default, Clone)]
struct Stats {
    pixels: usize,
    r_match: usize,
    g_match: usize,
    b_match: usize,
    a_match: usize,
    /// Sum of squared per-channel differences, used to compute PSNR
    /// across all 4 channels jointly (standard 8-bit PSNR formula).
    sse: u64,
}

impl Stats {
    fn psnr(&self) -> f64 {
        let n_samples = (self.pixels * 4) as f64;
        if n_samples == 0.0 || self.sse == 0 {
            return f64::INFINITY;
        }
        let mse = self.sse as f64 / n_samples;
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }

    fn channel_match_pct(&self, n: usize) -> f64 {
        if self.pixels == 0 {
            return 0.0;
        }
        100.0 * n as f64 / self.pixels as f64
    }

    fn all_pixels_match(&self) -> bool {
        self.pixels > 0
            && self.r_match == self.pixels
            && self.g_match == self.pixels
            && self.b_match == self.pixels
            && self.a_match == self.pixels
    }
}

/// Pull the reference PNG into a (w, h, RGBA) triple. Handles both 8-bit
/// Rgb24 (most fixtures) and 8-bit Rgba (the ALPH fixture).
fn decode_reference_png(png: &[u8]) -> (u32, u32, Vec<u8>) {
    let vf = decode_png_to_frame(png, None).expect("expected.png must decode");
    assert_eq!(
        vf.planes.len(),
        1,
        "PNG decoder must return a single packed plane"
    );
    let plane = &vf.planes[0];
    // Re-parse IHDR for authoritative width/height/bpp — the public
    // PNG API only hands us a `VideoFrame`, so we go to the file
    // header directly to disambiguate Rgb24 vs Rgba (both are common
    // and the plane stride alone is ambiguous for some image sizes).
    let (w, h, bpp) = parse_png_ihdr(png);
    let stride = (w as usize) * bpp;
    assert_eq!(
        plane.stride, stride,
        "PNG plane stride {} disagrees with IHDR {}×{}×{}bpp",
        plane.stride, w, h, bpp
    );
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    match bpp {
        3 => {
            for px in plane.data.chunks_exact(3) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        4 => {
            rgba.extend_from_slice(&plane.data);
        }
        other => panic!("PNG: unexpected bpp {other} in lossy fixture corpus"),
    }
    assert_eq!(rgba.len(), (w * h * 4) as usize);
    (w, h, rgba)
}

/// Hand-rolled minimal PNG IHDR parser. We use this rather than relying
/// on `oxideav_png` to expose the raw header because the public API
/// (`decode_png_to_frame`) only returns a `VideoFrame`. The IHDR layout
/// is fixed (PNG §11.2.2): 8-byte signature + length(4) + "IHDR"(4) +
/// width(4 BE) + height(4 BE) + bit_depth(1) + colour_type(1) + ... .
fn parse_png_ihdr(png: &[u8]) -> (u32, u32, usize) {
    assert!(png.len() >= 24, "PNG too short for IHDR");
    assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n", "PNG signature mismatch");
    assert_eq!(&png[12..16], b"IHDR", "first chunk must be IHDR");
    let w = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let h = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    let bit_depth = png[24];
    let colour_type = png[25];
    assert_eq!(bit_depth, 8, "lossy-corpus PNGs are all 8-bit");
    let bpp = match colour_type {
        2 => 3, // RGB
        6 => 4, // RGBA
        other => panic!("unexpected PNG colour_type {other} in lossy fixture corpus"),
    };
    (w, h, bpp)
}

fn compare(actual_rgba: &[u8], expected_rgba: &[u8]) -> Stats {
    assert_eq!(actual_rgba.len(), expected_rgba.len());
    let mut s = Stats {
        pixels: actual_rgba.len() / 4,
        ..Default::default()
    };
    for (a, e) in actual_rgba.chunks_exact(4).zip(expected_rgba.chunks_exact(4)) {
        if a[0] == e[0] {
            s.r_match += 1;
        }
        if a[1] == e[1] {
            s.g_match += 1;
        }
        if a[2] == e[2] {
            s.b_match += 1;
        }
        if a[3] == e[3] {
            s.a_match += 1;
        }
        for c in 0..4 {
            let d = (a[c] as i32 - e[c] as i32).unsigned_abs() as u64;
            s.sse += d * d;
        }
    }
    s
}

/// Drive a single fixture and return its result line for the summary
/// table. Side-effect: prints per-fixture diagnostics.
fn run_one(fix: &Fixture) -> String {
    eprintln!("--- fixture: {} (tier={:?}) ---", fix.name, fix.tier);
    eprintln!("    trace doc: {}", fix.trace_doc);
    eprintln!(
        "    input.webp = {} B   expected.png = {} B",
        fix.webp.len(),
        fix.png.len()
    );
    let (w, h, expected) = decode_reference_png(fix.png);
    eprintln!("    expected: {}×{} ({} RGBA bytes)", w, h, expected.len());

    let img = match decode_webp(fix.webp) {
        Ok(img) => img,
        Err(e) => {
            let line = format!("{:30}  decode-error: {e}", fix.name);
            eprintln!("    DECODE ERROR: {e}");
            if fix.tier == Tier::BitExact {
                panic!(
                    "{}: BitExact tier but decode_webp failed: {e}\n  see {} for expected events",
                    fix.name, fix.trace_doc
                );
            }
            return line;
        }
    };

    if img.frames.is_empty() {
        let line = format!("{:30}  decoded 0 frames", fix.name);
        eprintln!("    NO FRAMES decoded");
        return line;
    }
    if img.width != w || img.height != h {
        let line = format!(
            "{:30}  size-mismatch: webp={}×{} png={}×{}",
            fix.name, img.width, img.height, w, h
        );
        eprintln!("    SIZE MISMATCH: webp={}×{} png={}×{}", img.width, img.height, w, h);
        if fix.tier == Tier::BitExact {
            panic!("{}: BitExact tier but size mismatched", fix.name);
        }
        return line;
    }

    let actual = &img.frames[0].rgba;
    if actual.len() != expected.len() {
        let line = format!(
            "{:30}  buf-mismatch: actual={}B expected={}B",
            fix.name,
            actual.len(),
            expected.len()
        );
        eprintln!(
            "    BUFFER LEN MISMATCH: actual={} expected={}",
            actual.len(),
            expected.len()
        );
        if fix.tier == Tier::BitExact {
            panic!("{}: BitExact tier but RGBA buffer length differs", fix.name);
        }
        return line;
    }

    let s = compare(actual, &expected);
    let psnr = s.psnr();
    eprintln!(
        "    per-channel match: R={:.2}%  G={:.2}%  B={:.2}%  A={:.2}%   PSNR={:.2} dB",
        s.channel_match_pct(s.r_match),
        s.channel_match_pct(s.g_match),
        s.channel_match_pct(s.b_match),
        s.channel_match_pct(s.a_match),
        psnr,
    );
    if !s.all_pixels_match() {
        let mismatched = first_mismatch(actual, &expected, w as usize);
        if let Some((idx, x, y, a, e)) = mismatched {
            eprintln!(
                "    first divergence: pixel #{idx} at ({x},{y})  actual={:?}  expected={:?}",
                a, e
            );
        }
    }

    let line = format!(
        "{:30}  R={:6.2}% G={:6.2}% B={:6.2}% A={:6.2}%  PSNR={:6.2}dB  tier={:?}",
        fix.name,
        s.channel_match_pct(s.r_match),
        s.channel_match_pct(s.g_match),
        s.channel_match_pct(s.b_match),
        s.channel_match_pct(s.a_match),
        psnr,
        fix.tier,
    );

    if fix.tier == Tier::BitExact {
        assert!(
            s.all_pixels_match(),
            "{}: BitExact tier but pixels diverged. PSNR={:.2}dB. \
             Cross-check expected libwebp events in {}",
            fix.name,
            psnr,
            fix.trace_doc
        );
    }

    line
}

fn first_mismatch(
    actual: &[u8],
    expected: &[u8],
    w: usize,
) -> Option<(usize, usize, usize, [u8; 4], [u8; 4])> {
    for (idx, (a, e)) in actual.chunks_exact(4).zip(expected.chunks_exact(4)).enumerate() {
        if a != e {
            let x = idx % w;
            let y = idx / w;
            let mut aa = [0u8; 4];
            let mut ee = [0u8; 4];
            aa.copy_from_slice(a);
            ee.copy_from_slice(e);
            return Some((idx, x, y, aa, ee));
        }
    }
    None
}

#[test]
fn lossy_corpus_pixel_correctness() {
    let mut summary = Vec::with_capacity(FIXTURES.len());
    for fix in FIXTURES {
        let line = run_one(fix);
        summary.push(line);
    }

    // One-shot summary at the bottom of the test output so reviewers can
    // see all fixtures at a glance without scrolling per-fixture noise.
    eprintln!("\n========================================================================");
    eprintln!("lossy_corpus_pixel_correctness summary");
    eprintln!("========================================================================");
    for line in &summary {
        eprintln!("{line}");
    }
    eprintln!("========================================================================");
}
