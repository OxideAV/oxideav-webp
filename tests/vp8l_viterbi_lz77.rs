//! Tests for the VP8L encoder's **Viterbi-style optimal LZ77** pass —
//! pass 3 on top of the existing two-pass cost-modelled LZ77 pipeline
//! ([`oxideav_webp::vp8l::encoder::build_cost_modelled_stream`]).
//!
//! Coverage:
//!
//! 1. **Round-trip on natural ≥ 256×256 fixtures.** Three synthesised
//!    photo-like 256×256 RGBA fixtures (sky/foliage/foreground landscape;
//!    portrait-on-textured-background; brick + plaster wall mosaic) round-
//!    trip bit-exact through the in-crate VP8L decoder. Uses a single
//!    `EncoderOptions::default()` configuration (not the full RDO sweep)
//!    so the test runs in seconds in debug mode while still exercising
//!    the Viterbi pass — pass 3 is gated on ≥ 65 536 px and 256×256
//!    sits exactly on that threshold.
//! 2. **External `dwebp` cross-decode.** One fixture (the landscape) is
//!    cross-decoded through libwebp's `dwebp` binary too (when
//!    installed), confirming the Viterbi-picked tokens are spec-conformant
//!    on natural-image content. Silently skipped when `dwebp` isn't on
//!    `PATH` so the test still passes on CI hosts without libwebp.
//! 3. **cwebp parity tracking.** The landscape fixture's full-RDO encode
//!    is compared to `cwebp -lossless -m 6 -z 9` for the same input;
//!    the test prints the ratio for visibility but only asserts a
//!    generous ceiling (≤ 1.05x) so the test stays green if cwebp ever
//!    tightens its coder. The ratio history is the per-round size-
//!    tracking signal. Only one fixture goes through the full RDO sweep
//!    here because RDO costs ~10 trials × 1 second of Viterbi each on a
//!    256×256 image — running three RDO encodes in CI's debug mode
//!    would push the test runtime past 10 minutes.
//!
//! All fixtures are built deterministically from xorshift PRNGs so
//! every CI run sees the same numbers.

use std::path::Path;
use std::process::Command;

use oxideav_webp::encode_vp8l_argb;
use oxideav_webp::vp8l;
use oxideav_webp::vp8l::encoder::{encode_vp8l_argb_with, EncoderOptions};

const HOMEBREW_CWEBP: &str = "/opt/homebrew/bin/cwebp";
const HOMEBREW_DWEBP: &str = "/opt/homebrew/bin/dwebp";
const SYSTEM_CWEBP: &str = "/usr/bin/cwebp";
const SYSTEM_DWEBP: &str = "/usr/bin/dwebp";

fn cwebp_path() -> Option<&'static str> {
    if Path::new(HOMEBREW_CWEBP).exists() {
        Some(HOMEBREW_CWEBP)
    } else if Path::new(SYSTEM_CWEBP).exists() {
        Some(SYSTEM_CWEBP)
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

fn rgba_to_argb(rgba: &[u8]) -> Vec<u32> {
    rgba.chunks_exact(4)
        .map(|px| {
            ((px[3] as u32) << 24) | ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32
        })
        .collect()
}

fn write_pam_rgba(path: &str, w: u32, h: u32, rgba: &[u8]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("create pam");
    write!(
        f,
        "P7\nWIDTH {w}\nHEIGHT {h}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
    )
    .expect("pam header");
    f.write_all(rgba).expect("pam pixels");
}

fn read_pam_rgba(path: &str) -> (u32, u32, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read pam");
    let end_marker = b"ENDHDR\n";
    let end_pos = bytes
        .windows(end_marker.len())
        .position(|w| w == end_marker)
        .expect("PAM ENDHDR marker not found");
    let header = std::str::from_utf8(&bytes[..end_pos]).expect("PAM header utf8");
    let mut w = 0u32;
    let mut h = 0u32;
    let mut depth = 0u32;
    for line in header.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("WIDTH ") {
            w = rest.parse().expect("WIDTH parse");
        } else if let Some(rest) = line.strip_prefix("HEIGHT ") {
            h = rest.parse().expect("HEIGHT parse");
        } else if let Some(rest) = line.strip_prefix("DEPTH ") {
            depth = rest.parse().expect("DEPTH parse");
        }
    }
    assert_eq!(depth, 4, "expected RGBA PAM (DEPTH 4), got {depth}");
    let pixel = &bytes[end_pos + end_marker.len()..];
    assert_eq!(pixel.len(), (w * h * 4) as usize, "PAM body size mismatch");
    (w, h, pixel.to_vec())
}

/// Wrap a bare VP8L bitstream in a minimal RIFF/WEBP/VP8L file. Same as
/// the helper in `vp8l_rdo.rs`, kept local to avoid cross-test sharing.
fn wrap_riff_vp8l(bare: &[u8]) -> Vec<u8> {
    let mut wrapped: Vec<u8> = Vec::with_capacity(20 + bare.len() + 1);
    wrapped.extend_from_slice(b"RIFF");
    let chunk_len = bare.len() as u32;
    let pad = chunk_len & 1;
    let riff_size = 4 + 8 + chunk_len + pad;
    wrapped.extend_from_slice(&riff_size.to_le_bytes());
    wrapped.extend_from_slice(b"WEBP");
    wrapped.extend_from_slice(b"VP8L");
    wrapped.extend_from_slice(&chunk_len.to_le_bytes());
    wrapped.extend_from_slice(bare);
    if pad == 1 {
        wrapped.push(0);
    }
    wrapped
}

/// Synthesise a 256×256 photo-like landscape RGBA fixture. Three
/// vertically-stacked regions (sky, foliage, foreground) with different
/// statistics so the LZ77 backref graph has heterogeneous per-region
/// optima — the kind of input where Viterbi vs lazy-greedy diverges.
fn landscape_256(w: u32, h: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut s: u32 = 0xC0DE_F00D;
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if y < h / 3 {
                // Sky: smooth blue gradient + tiny noise.
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s & 0x07) as u8;
                rgba[i] = (60 + (y * 40 / (h / 3)) as u8).saturating_add(n);
                rgba[i + 1] = (100 + (y * 60 / (h / 3)) as u8).saturating_add(n);
                rgba[i + 2] = (180 + (y * 30 / (h / 3)) as u8).saturating_add(n);
            } else if y < 2 * h / 3 {
                // Foliage: mid-saturation green with broader noise.
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s & 0x1f) as u8;
                rgba[i] = 40u8.saturating_add(n);
                rgba[i + 1] = 120u8.saturating_add(n);
                rgba[i + 2] = 50u8.saturating_add(n / 2);
            } else {
                // Foreground: warm earth tones + structured pattern.
                let stripe = ((x / 4) ^ (y / 4)) & 1 == 0;
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s & 0x0f) as u8;
                rgba[i] = (if stripe { 140u8 } else { 110u8 }).saturating_add(n);
                rgba[i + 1] = (if stripe { 90u8 } else { 70u8 }).saturating_add(n);
                rgba[i + 2] = (if stripe { 50u8 } else { 30u8 }).saturating_add(n);
            }
            rgba[i + 3] = 0xff;
        }
    }
    rgba
}

/// Synthesise a 256×256 portrait-on-textured-background fixture. Centre
/// disc with smooth radial gradient, surrounding ring with stippled noise,
/// outer band with cross-hatched lines. Multiple-orientation features
/// stress predictor + LZ77 distance bins differently from the landscape.
fn portrait_textured_256(w: u32, h: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let cx = (w / 2) as i32;
    let cy = (h / 2) as i32;
    let inner_r2 = (w as i32 / 4).pow(2);
    let outer_r2 = (w as i32 / 2 - 8).pow(2);
    let mut s: u32 = 0xBADC_0DEF;
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let dx = x as i32 - cx;
            let dy = y as i32 - cy;
            let d2 = dx * dx + dy * dy;
            if d2 < inner_r2 {
                // Centre disc — smooth radial gradient, skin-tone-ish.
                let d = (d2 as f32).sqrt() as i32;
                let t = (d * 255 / (w as i32 / 4)).min(255) as u8;
                rgba[i] = 200u8.saturating_sub(t / 4);
                rgba[i + 1] = 160u8.saturating_sub(t / 4);
                rgba[i + 2] = 140u8.saturating_sub(t / 6);
            } else if d2 < outer_r2 {
                // Stippled ring.
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s & 0x3f) as u8;
                let on = (s >> 8) & 1 == 0;
                rgba[i] = if on { 60u8 } else { 90u8 }.saturating_add(n);
                rgba[i + 1] = if on { 80u8 } else { 110u8 }.saturating_add(n);
                rgba[i + 2] = if on { 100u8 } else { 130u8 }.saturating_add(n);
            } else {
                // Cross-hatched outer band. Both diagonal lines computed
                // with i32 arithmetic so the wrap-around at y > x + 64
                // doesn't underflow u32.
                let h1 = ((x + y) % 8) < 2;
                let h2 = ((x as i32 + 64 - y as i32).rem_euclid(8)) < 2;
                let v = if h1 || h2 { 30u8 } else { 200u8 };
                rgba[i] = v;
                rgba[i + 1] = v;
                rgba[i + 2] = v.saturating_add(10);
            }
            rgba[i + 3] = 0xff;
        }
    }
    rgba
}

/// Synthesise a 256×256 brick-and-plaster wall mosaic fixture. Repeating
/// horizontal courses with vertically-offset bricks plus mortar gaps; the
/// repeating block structure is exactly what LZ77 backreferences ride on,
/// so the Viterbi DP has plenty of "shorter-cheaper-here vs longer-here"
/// candidate pairs to pick between.
fn brick_wall_256(w: u32, h: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let mut s: u32 = 0xFEED_FACE;
    let course_h = 14u32;
    let brick_w = 32u32;
    let mortar = 2u32;
    for y in 0..h {
        let course = y / course_h;
        let row_in_course = y % course_h;
        let in_mortar_row = row_in_course >= course_h - mortar;
        let xshift = if course % 2 == 0 { 0u32 } else { brick_w / 2 };
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let xs = (x + xshift) % brick_w;
            let in_mortar_col = xs >= brick_w - mortar;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if in_mortar_row || in_mortar_col {
                // Mortar — light cream with subtle noise.
                let n = (s & 0x07) as u8;
                rgba[i] = 220u8.saturating_sub(n);
                rgba[i + 1] = 215u8.saturating_sub(n);
                rgba[i + 2] = 200u8.saturating_sub(n);
            } else {
                // Brick — warm red-brown with grain.
                let n = (s & 0x1f) as u8;
                rgba[i] = 160u8.saturating_add(n);
                rgba[i + 1] = 80u8.saturating_add(n / 2);
                rgba[i + 2] = 50u8.saturating_add(n / 3);
            }
            rgba[i + 3] = 0xff;
        }
    }
    rgba
}

struct Fixture {
    name: &'static str,
    builder: fn(u32, u32) -> Vec<u8>,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "landscape-256",
        builder: landscape_256,
    },
    Fixture {
        name: "portrait-textured-256",
        builder: portrait_textured_256,
    },
    Fixture {
        name: "brick-wall-256",
        builder: brick_wall_256,
    },
];

/// Default-options config that triggers the Viterbi pass on ≥ 256×256
/// inputs. Avoids the full 32-trial RDO sweep so the test stays cheap
/// in debug mode (single config × Viterbi inside `encode_image_stream`
/// instead of 32 configs × Viterbi each).
fn default_opts() -> EncoderOptions {
    EncoderOptions::default()
}

#[test]
fn viterbi_outputs_round_trip_through_in_crate_decoder() {
    let w = 256u32;
    let h = 256u32;
    for fx in FIXTURES {
        let rgba = (fx.builder)(w, h);
        let pixels = rgba_to_argb(&rgba);
        let bare = encode_vp8l_argb_with(w, h, &pixels, false, default_opts())
            .expect("default-config encode");
        let decoded = vp8l::decode(&bare).expect("decode our default-config output");
        assert_eq!(
            decoded.to_rgba(),
            rgba,
            "{}: in-crate round-trip pixel mismatch (default config / Viterbi pass enabled)",
            fx.name,
        );
    }
}

#[test]
fn viterbi_output_decodes_through_external_dwebp_on_landscape_256() {
    let Some(dwebp) = dwebp_path() else {
        eprintln!("skip: dwebp not on PATH");
        return;
    };
    let w = 256u32;
    let h = 256u32;
    let rgba = landscape_256(w, h);
    let pixels = rgba_to_argb(&rgba);
    let bare = encode_vp8l_argb_with(w, h, &pixels, false, default_opts()).expect("encode");
    let wrapped = wrap_riff_vp8l(&bare);

    let webp_path = "/tmp/oxideav-webp-viterbi-landscape.webp";
    let pam_path = "/tmp/oxideav-webp-viterbi-landscape.pam";
    std::fs::write(webp_path, &wrapped).expect("write webp");
    let status = Command::new(dwebp)
        .args([webp_path, "-quiet", "-pam", "-o", pam_path])
        .status()
        .expect("invoke dwebp");
    assert!(status.success(), "external dwebp failed on Viterbi output");
    let (out_w, out_h, out_rgba) = read_pam_rgba(pam_path);
    assert_eq!(out_w, w);
    assert_eq!(out_h, h);
    assert_eq!(
        out_rgba, rgba,
        "external dwebp decode mismatch — Viterbi picked spec-non-conformant tokens",
    );
}

#[test]
fn viterbi_within_5pct_of_cwebp_lossless_on_landscape_256() {
    let Some(cwebp) = cwebp_path() else {
        eprintln!("skip: cwebp not on PATH");
        return;
    };
    let w = 256u32;
    let h = 256u32;
    let rgba = landscape_256(w, h);
    let pam = "/tmp/oxideav-webp-viterbi-cmp-landscape.pam";
    let cwebp_out = "/tmp/oxideav-webp-viterbi-cmp-landscape-cwebp.webp";
    write_pam_rgba(pam, w, h, &rgba);
    let status = Command::new(cwebp)
        .args([
            "-lossless", "-m", "6", "-z", "9", "-quiet", pam, "-o", cwebp_out,
        ])
        .status()
        .expect("invoke cwebp");
    assert!(status.success(), "cwebp failed");
    let cwebp_size = std::fs::metadata(cwebp_out)
        .expect("cwebp out metadata")
        .len() as usize;

    // Use the full RDO entry point here (the only test that does so) so
    // the cwebp comparison is apples-to-apples — both encoders are
    // running their highest compression preset.
    let pixels = rgba_to_argb(&rgba);
    let bare = encode_vp8l_argb(w, h, &pixels, false).expect("RDO encode");
    let our_size = bare.len() + 20 + (bare.len() & 1);
    let ratio = our_size as f64 / cwebp_size as f64;
    eprintln!(
        "[viterbi/cwebp] landscape-256: ours={} cwebp={} ratio={:.4}",
        our_size, cwebp_size, ratio
    );
    // Generous ceiling — we track parity but the assertion is "must not
    // regress past 5 % over cwebp". Tighter parity is reported per
    // round in the README and CHANGELOG and tracked across releases.
    assert!(
        ratio <= 1.05,
        "VP8L RDO output ({} bytes) exceeds 1.05x cwebp's lossless output \
         ({} bytes); ratio = {:.4}",
        our_size,
        cwebp_size,
        ratio,
    );
}
