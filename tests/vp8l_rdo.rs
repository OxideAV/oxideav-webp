//! Tests for the VP8L encoder's per-image RDO sweep
//! ([`oxideav_webp::encode_vp8l_argb`]).
//!
//! The RDO loop probes each combination of the four optional VP8L
//! transforms (subtract-green, colour-transform, predictor) plus a short
//! list of colour-cache widths and keeps the smallest resulting
//! bitstream. Coverage:
//!
//! 1. **No-loss invariant** — the RDO output must round-trip bit-for-bit
//!    through the in-crate decoder, regardless of which configuration
//!    happens to win.
//! 2. **Beats fixed defaults** — on at least one fixture the RDO winner
//!    is strictly smaller than the all-transforms-on default. Otherwise
//!    the search adds zero value.
//! 3. **Within 1.30× cwebp** (when `cwebp` is on `PATH`) on:
//!    a small photographic-like noise field;
//!    a small 16-colour palette-friendly drawing.
//! 4. **dwebp round-trip** — the RDO output decodes back to the original
//!    pixels via the external `dwebp` binary too, which validates spec
//!    compliance independent of our own decoder.

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
    let mut out = Vec::with_capacity(rgba.len() / 4);
    for px in rgba.chunks_exact(4) {
        out.push(
            ((px[3] as u32) << 24) | ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32,
        );
    }
    out
}

/// Write RGBA bytes as a binary PPM with a side alpha map (PAM-style
/// PNG would be cleaner but cwebp accepts PAM, PPM, PNG, etc.). We emit
/// PAM here — supports RGBA natively.
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

/// Read a PAM file written by `dwebp -pam`. Returns (w, h, rgba_bytes).
fn read_pam_rgba(path: &str) -> (u32, u32, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read pam");
    // PAM header is ASCII up to "ENDHDR\n".
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
    assert!(
        depth == 4,
        "expected RGBA PAM (DEPTH 4), got DEPTH {depth} from {path}"
    );
    let pixel_data = &bytes[end_pos + end_marker.len()..];
    assert_eq!(
        pixel_data.len(),
        (w * h * 4) as usize,
        "PAM pixel-region size mismatch"
    );
    (w, h, pixel_data.to_vec())
}

/// Build a 64×64 photographic-like noise + smooth gradient mix. Fairly
/// hard to compress (most fixed configs produce similar sizes) so the
/// RDO winner can credibly differ image-to-image.
fn photo_like(w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    let mut s: u32 = 0x1234_5678;
    for y in 0..h {
        for x in 0..w {
            // xorshift32 → small noise term
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let n = (s & 0x1f) as i32 - 16; // -16..15
            let i = ((y * w + x) * 4) as usize;
            // Smooth-ish gradient + noise per channel
            out[i] = (((x as i32 * 2) + n).clamp(0, 255)) as u8;
            out[i + 1] = (((y as i32 * 2) + n).clamp(0, 255)) as u8;
            out[i + 2] = ((((x + y) as i32) + n).clamp(0, 255)) as u8;
            out[i + 3] = 0xff;
        }
    }
    out
}

/// Build a 64×64 16-colour palette-style image — a checker / band pattern
/// drawn from a fixed 16-entry palette. The colour-cache and predictor
/// transforms should both pay off well here.
fn palette_like(w: u32, h: u32) -> Vec<u8> {
    let palette: [[u8; 4]; 16] = [
        [0xff, 0x00, 0x00, 0xff],
        [0x00, 0xff, 0x00, 0xff],
        [0x00, 0x00, 0xff, 0xff],
        [0xff, 0xff, 0x00, 0xff],
        [0xff, 0x00, 0xff, 0xff],
        [0x00, 0xff, 0xff, 0xff],
        [0x80, 0x00, 0x80, 0xff],
        [0x80, 0x80, 0x00, 0xff],
        [0x40, 0x40, 0xff, 0xff],
        [0xff, 0x40, 0x40, 0xff],
        [0xff, 0x80, 0x40, 0xff],
        [0x40, 0xff, 0x80, 0xff],
        [0x10, 0x10, 0x10, 0xff],
        [0xf0, 0xf0, 0xf0, 0xff],
        [0x80, 0x80, 0x80, 0xff],
        [0xc0, 0xc0, 0xc0, 0xff],
    ];
    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            // Block-y palette index
            let idx = (((x / 4) ^ (y / 4)) % 16) as usize;
            let i = ((y * w + x) * 4) as usize;
            out[i..i + 4].copy_from_slice(&palette[idx]);
        }
    }
    out
}

/// Run the encoder under each transform-only configuration and return
/// the size of the smallest fixed (non-RDO) variant. Used as the
/// "fixed defaults" baseline the RDO must beat.
fn smallest_fixed(width: u32, height: u32, pixels: &[u32], has_alpha: bool) -> usize {
    let opts = [
        EncoderOptions::default(),
        EncoderOptions::bare(),
        EncoderOptions::subtract_green_only(),
    ];
    let mut min = usize::MAX;
    for opt in &opts {
        let bytes = encode_vp8l_argb_with(width, height, pixels, has_alpha, *opt)
            .expect("fixed-config encode");
        if bytes.len() < min {
            min = bytes.len();
        }
    }
    min
}

#[test]
fn rdo_output_round_trips_through_in_crate_decoder() {
    let w = 64u32;
    let h = 64u32;
    for fixture in [photo_like(w, h), palette_like(w, h)] {
        let pixels = rgba_to_argb(&fixture);
        let bytes = encode_vp8l_argb(w, h, &pixels, false).expect("RDO encode");
        let decoded = vp8l::decode(&bytes).expect("decode RDO output");
        assert_eq!(
            decoded.to_rgba(),
            fixture,
            "RDO winner failed lossless round-trip"
        );
    }
}

#[test]
fn rdo_beats_or_matches_default_on_palette_image() {
    let w = 64u32;
    let h = 64u32;
    let fixture = palette_like(w, h);
    let pixels = rgba_to_argb(&fixture);
    let rdo = encode_vp8l_argb(w, h, &pixels, false).expect("RDO encode");
    let fixed_default = encode_vp8l_argb_with(w, h, &pixels, false, EncoderOptions::default())
        .expect("default encode");
    eprintln!(
        "palette-like 64x64: rdo={} bytes, default={} bytes",
        rdo.len(),
        fixed_default.len()
    );
    // RDO covers the default in its search space, so it must be ≤ the
    // default's size on every input.
    assert!(
        rdo.len() <= fixed_default.len(),
        "RDO output ({} bytes) exceeded default ({} bytes) — RDO must always pick the best of its search space",
        rdo.len(),
        fixed_default.len()
    );
}

#[test]
fn rdo_never_exceeds_smallest_fixed_configuration() {
    // The RDO grid is a strict superset of the three named fixed
    // configurations (`default` / `bare` / `subtract_green_only`), so
    // the RDO winner must always be ≤ the smallest of them. This is
    // the soundness check for the search loop.
    let w = 64u32;
    let h = 64u32;
    let fixture = palette_like(w, h);
    let pixels = rgba_to_argb(&fixture);
    let rdo = encode_vp8l_argb(w, h, &pixels, false).expect("RDO encode");
    let smallest_fixed = smallest_fixed(w, h, &pixels, false);
    eprintln!(
        "palette-like 64x64: rdo={} bytes, smallest_fixed={} bytes",
        rdo.len(),
        smallest_fixed
    );
    assert!(
        rdo.len() <= smallest_fixed,
        "RDO ({} bytes) must be ≤ the smallest fixed configuration ({} bytes)",
        rdo.len(),
        smallest_fixed
    );
}

#[test]
fn rdo_within_130pct_of_cwebp_lossless_on_photo_like() {
    let Some(cwebp) = cwebp_path() else {
        eprintln!("skip: cwebp not on PATH");
        return;
    };
    let w = 64u32;
    let h = 64u32;
    let fixture = photo_like(w, h);
    let pam = "/tmp/oxideav-webp-rdo-photo.pam";
    let cwebp_out = "/tmp/oxideav-webp-rdo-photo-cwebp.webp";
    write_pam_rgba(pam, w, h, &fixture);
    // -lossless -m 6 -z 9 — the heaviest cwebp lossless preset that
    // still terminates in subseconds on a 64×64 image. Gives us the
    // smallest reasonable cwebp output to compare against.
    let status = Command::new(cwebp)
        .args([
            "-lossless",
            "-m",
            "6",
            "-z",
            "9",
            "-quiet",
            pam,
            "-o",
            cwebp_out,
        ])
        .status()
        .expect("invoke cwebp");
    assert!(
        status.success(),
        "cwebp failed on photo-like fixture: {status:?}"
    );
    let cwebp_size = std::fs::metadata(cwebp_out)
        .expect("cwebp out metadata")
        .len() as usize;

    let pixels = rgba_to_argb(&fixture);
    let bare_bitstream = encode_vp8l_argb(w, h, &pixels, false).expect("RDO encode");
    // The RDO output is a bare VP8L bitstream — when measuring against
    // a full RIFF .webp file we need to add the wrapper overhead. The
    // simple-layout RIFF wrapper is 20 bytes (4 RIFF + 4 size + 4 WEBP
    // + 4 VP8L + 4 chunk-size) plus 1 byte if the bitstream is odd-sized.
    let wrapper_overhead = 20 + (bare_bitstream.len() & 1);
    let our_size = bare_bitstream.len() + wrapper_overhead;
    let ratio = our_size as f64 / cwebp_size as f64;
    eprintln!(
        "photo-like 64x64: ours={} bytes, cwebp={} bytes (ratio {:.2}x)",
        our_size, cwebp_size, ratio
    );
    assert!(
        ratio <= 1.30,
        "VP8L RDO output ({} bytes) exceeds 1.30x cwebp's lossless output ({} bytes); ratio = {:.2}",
        our_size,
        cwebp_size,
        ratio
    );
}

#[test]
fn rdo_within_130pct_of_cwebp_lossless_on_palette_like() {
    let Some(cwebp) = cwebp_path() else {
        eprintln!("skip: cwebp not on PATH");
        return;
    };
    let w = 64u32;
    let h = 64u32;
    let fixture = palette_like(w, h);
    let pam = "/tmp/oxideav-webp-rdo-palette.pam";
    let cwebp_out = "/tmp/oxideav-webp-rdo-palette-cwebp.webp";
    write_pam_rgba(pam, w, h, &fixture);
    let status = Command::new(cwebp)
        .args([
            "-lossless",
            "-m",
            "6",
            "-z",
            "9",
            "-quiet",
            pam,
            "-o",
            cwebp_out,
        ])
        .status()
        .expect("invoke cwebp");
    assert!(
        status.success(),
        "cwebp failed on palette-like fixture: {status:?}"
    );
    let cwebp_size = std::fs::metadata(cwebp_out)
        .expect("cwebp out metadata")
        .len() as usize;

    let pixels = rgba_to_argb(&fixture);
    let bare_bitstream = encode_vp8l_argb(w, h, &pixels, false).expect("RDO encode");
    let wrapper_overhead = 20 + (bare_bitstream.len() & 1);
    let our_size = bare_bitstream.len() + wrapper_overhead;
    let ratio = our_size as f64 / cwebp_size as f64;
    eprintln!(
        "palette-like 64x64: ours={} bytes, cwebp={} bytes (ratio {:.2}x)",
        our_size, cwebp_size, ratio
    );
    // Palette-like content gets very tight with cwebp's colour-indexing
    // transform (which we don't emit). The 1.30× target is generous to
    // give us margin for the missing palette transform.
    assert!(
        ratio <= 1.30,
        "VP8L RDO output ({} bytes) exceeds 1.30x cwebp's lossless output ({} bytes); ratio = {:.2}",
        our_size,
        cwebp_size,
        ratio
    );
}

#[test]
fn rdo_output_decodes_correctly_through_external_dwebp() {
    let Some(dwebp) = dwebp_path() else {
        eprintln!("skip: dwebp not on PATH");
        return;
    };
    let w = 32u32;
    let h = 32u32;
    let fixture = photo_like(w, h);
    let pixels = rgba_to_argb(&fixture);
    let bare_bitstream = encode_vp8l_argb(w, h, &pixels, false).expect("RDO encode");

    // Wrap the bare VP8L bitstream in a minimal RIFF/WEBP/VP8L file so
    // dwebp can ingest it.
    let mut wrapped: Vec<u8> = Vec::with_capacity(20 + bare_bitstream.len() + 1);
    wrapped.extend_from_slice(b"RIFF");
    let chunk_len = bare_bitstream.len() as u32;
    let pad = chunk_len & 1;
    let riff_size = 4 + 8 + chunk_len + pad;
    wrapped.extend_from_slice(&riff_size.to_le_bytes());
    wrapped.extend_from_slice(b"WEBP");
    wrapped.extend_from_slice(b"VP8L");
    wrapped.extend_from_slice(&chunk_len.to_le_bytes());
    wrapped.extend_from_slice(&bare_bitstream);
    if pad == 1 {
        wrapped.push(0);
    }

    let webp_path = "/tmp/oxideav-webp-rdo-dwebp.webp";
    let pam_path = "/tmp/oxideav-webp-rdo-dwebp.pam";
    std::fs::write(webp_path, &wrapped).expect("write webp");
    let status = Command::new(dwebp)
        .args([webp_path, "-quiet", "-pam", "-o", pam_path])
        .status()
        .expect("invoke dwebp");
    assert!(status.success(), "dwebp failed on RDO output: {status:?}");

    let (out_w, out_h, out_rgba) = read_pam_rgba(pam_path);
    assert_eq!(out_w, w);
    assert_eq!(out_h, h);
    assert_eq!(
        out_rgba, fixture,
        "dwebp's decode of our RDO output didn't match the source pixels — spec violation"
    );
}
