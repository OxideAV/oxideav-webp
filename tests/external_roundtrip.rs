//! End-to-end external roundtrip test: oxideav-encode →
//! libwebp-decode → libwebp-encode → oxideav-decode → assert
//! byte-exact (lossless VP8L).
//!
//! This is the standard "agree-with-the-reference" cross-check the
//! workspace runs against every codec that has both an encoder and a
//! decoder. The two halves already exist as fuzz harnesses in
//! `fuzz/fuzz_targets/{oxideav_encode_webp_decode_lossless,
//! webp_encode_oxideav_decode_lossless}.rs`; this test stitches them
//! into a single round trip and runs it once on a deterministic
//! 640×480 RGBA image.
//!
//! libwebp is loaded with `libloading` at runtime — no `*-sys` crate,
//! no libwebp source pulled into the workspace dep tree (workspace
//! policy bars external library code as a dependency). The test
//! `eprintln`-skips silently when libwebp isn't installed, so CI
//! hosts without the shared library still get a green run.
//!
//! Install libwebp with `brew install webp` (macOS) or
//! `apt install libwebp7` (Debian/Ubuntu). The loader probes the
//! conventional shared-object names for both platforms.
//!
//! `strip_transparent_color` is forced **off** so RGB on alpha=0
//! pixels is preserved — but libwebp's encoder itself zeros RGB on
//! transparent pixels by default (`exact = false`), so the final
//! assertion uses `assert_rgba_allow_transparent_rgb_differences`
//! (alpha must match exactly, RGB only enforced on non-transparent
//! pixels). Same quirk handling as the fuzz harness on the libwebp
//! → oxideav direction.

#![allow(unsafe_code)]

use oxideav_webp::decode_webp;

/// Runtime libwebp shim — mirrors `oxideav-webp-fuzz`'s `libwebp`
/// module shape but kept private to this test file to avoid having
/// the main crate dev-depend on the fuzz crate.
mod libwebp {
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    /// Conventional libwebp shared-object names the loader will try
    /// in order. Covers macOS (`.dylib`), Linux (versioned + plain
    /// `.so`), and Windows (`.dll`).
    const CANDIDATES: &[&str] = &[
        "libwebp.dylib",
        "libwebp.7.dylib",
        "libwebp.so.7",
        "libwebp.so",
        "libwebp.dll",
    ];

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for test tooling — libwebp is a
                // well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libwebp shared library was successfully loaded.
    /// The integration test early-returns when this is false so the
    /// run still passes on hosts without libwebp installed.
    pub fn available() -> bool {
        lib().is_some()
    }

    /// Encode an RGBA image losslessly via `WebPEncodeLosslessRGBA`.
    /// Returns `None` if libwebp isn't available or the encoder
    /// returned a null pointer.
    pub fn encode_lossless(rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        type EncFn = unsafe extern "C" fn(*const u8, i32, i32, i32, *mut *mut u8) -> usize;
        type FreeFn = unsafe extern "C" fn(*mut u8);
        let l = lib()?;
        unsafe {
            let enc: Symbol<EncFn> = l.get(b"WebPEncodeLosslessRGBA").ok()?;
            let free: Symbol<FreeFn> = l.get(b"WebPFree").ok()?;
            let mut out: *mut u8 = std::ptr::null_mut();
            let stride = (width as i32).checked_mul(4)?;
            let n = enc(rgba.as_ptr(), width as i32, height as i32, stride, &mut out);
            if out.is_null() || n == 0 {
                return None;
            }
            let v = std::slice::from_raw_parts(out, n).to_vec();
            free(out);
            Some(v)
        }
    }

    /// A WebP frame as decoded by libwebp, normalised to RGBA.
    pub struct DecodedRgba {
        pub width: u32,
        pub height: u32,
        /// Tightly packed RGBA, length `width * height * 4`.
        pub rgba: Vec<u8>,
    }

    /// Decode a WebP byte string to RGBA via `WebPGetInfo` +
    /// `WebPDecodeRGBAInto`. Returns `None` on libwebp unavailable,
    /// header parse failure, allocation overflow, or decode failure.
    pub fn decode_to_rgba(data: &[u8]) -> Option<DecodedRgba> {
        type GetInfoFn = unsafe extern "C" fn(*const u8, usize, *mut i32, *mut i32) -> i32;
        type DecodeIntoFn = unsafe extern "C" fn(*const u8, usize, *mut u8, usize, i32) -> *mut u8;
        let l = lib()?;
        unsafe {
            let info: Symbol<GetInfoFn> = l.get(b"WebPGetInfo").ok()?;
            let decode: Symbol<DecodeIntoFn> = l.get(b"WebPDecodeRGBAInto").ok()?;
            let mut w: i32 = 0;
            let mut h: i32 = 0;
            if info(data.as_ptr(), data.len(), &mut w, &mut h) == 0 || w <= 0 || h <= 0 {
                return None;
            }
            let stride = (w as usize).checked_mul(4)?;
            let size = stride.checked_mul(h as usize)?;
            let mut buf = vec![0u8; size];
            let result = decode(
                data.as_ptr(),
                data.len(),
                buf.as_mut_ptr(),
                size,
                stride as i32,
            );
            if result.is_null() {
                return None;
            }
            Some(DecodedRgba {
                width: w as u32,
                height: h as u32,
                rgba: buf,
            })
        }
    }
}

/// Build a deterministic pseudo-random 640×480 RGBA buffer. Uses a
/// tiny xorshift32 PRNG seeded from a constant — keeps the test
/// `rand`-dep-free and reproducible across hosts.
fn deterministic_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
    let mut s: u32 = 0xC0DE_F00D;
    for b in rgba.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *b = (s & 0xff) as u8;
    }
    rgba
}

/// Pack an R,G,B,A byte slice into ARGB u32 pixels (the layout the
/// VP8L encoder expects).
fn rgba_bytes_to_argb_pixels(rgba: &[u8]) -> Vec<u32> {
    rgba.chunks_exact(4)
        .map(|p| {
            ((p[3] as u32) << 24) | ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32)
        })
        .collect()
}

/// Final assertion: alpha must match exactly. RGB is only enforced
/// for non-transparent pixels — libwebp's encoder zeros RGB on
/// alpha=0 pixels by default (`exact = false`), so a strict
/// byte-equality check would falsely fail on any transparent pixel.
/// Mirrors the helper in
/// `fuzz/fuzz_targets/webp_encode_oxideav_decode_lossless.rs`.
fn assert_rgba_allow_transparent_rgb_differences(expected: &[u8], actual: &[u8]) {
    assert_eq!(actual.len(), expected.len(), "decoded RGBA length mismatch");

    for (pixel_index, (expected, actual)) in expected
        .chunks_exact(4)
        .zip(actual.chunks_exact(4))
        .enumerate()
    {
        if expected[3] == 0 {
            assert_eq!(
                actual[3], 0,
                "decoded alpha differs for transparent pixel {pixel_index}"
            );
        } else {
            assert_eq!(
                actual, expected,
                "decoded RGBA differs at pixel {pixel_index}"
            );
        }
    }
}

#[test]
fn vp8l_external_roundtrip_640x480() {
    if !libwebp::available() {
        eprintln!(
            "[external_roundtrip] SKIP: libwebp not installed — \
             install with `brew install webp` or `apt install libwebp7`"
        );
        return;
    }

    let width = 640u32;
    let height = 480u32;
    let original_rgba = deterministic_rgba(width, height);

    // Step 1: oxideav VP8L encode → bare bitstream → wrap in RIFF.
    // `strip_transparent_color: false` preserves RGB on alpha=0
    // pixels (matches the fuzz pattern). `has_alpha = true` because
    // the random bytes will include alpha != 0xff — flag the file
    // accordingly so the extended (VP8X) layout's ALPHA bit is set,
    // which is what `build_vp8l_with_alpha` produces.
    let argb = rgba_bytes_to_argb_pixels(&original_rgba);
    let opts = oxideav_webp::EncoderOptions {
        strip_transparent_color: false,
        ..Default::default()
    };
    let bitstream = oxideav_webp::encode_vp8l_argb_with(width, height, &argb, true, opts)
        .expect("oxideav-webp VP8L encode failed");
    let webp_from_oxideav = oxideav_webp::riff::build_vp8l_with_alpha(
        &bitstream,
        width,
        height,
        &oxideav_webp::riff::WebpMetadata::default(),
    );

    // Step 2: libwebp decode → RGBA. This validates that what
    // oxideav emits is parseable by the reference decoder — the
    // first half of the cross-check.
    let decoded_by_libwebp = libwebp::decode_to_rgba(&webp_from_oxideav)
        .expect("libwebp failed to decode oxideav-encoded WebP");
    assert_eq!(decoded_by_libwebp.width, width);
    assert_eq!(decoded_by_libwebp.height, height);

    // Step 3: libwebp re-encode losslessly. Round-trips into the
    // reference encoder so the bytes we hand to the oxideav decoder
    // in step 4 weren't produced by the same code that's about to
    // decode them.
    let webp_from_libwebp = libwebp::encode_lossless(&decoded_by_libwebp.rgba, width, height)
        .expect("libwebp lossless re-encode failed");

    // Step 4: oxideav decode → final frame. This validates that
    // oxideav can parse a libwebp-emitted file — the second half of
    // the cross-check.
    let final_image =
        decode_webp(&webp_from_libwebp).expect("oxideav-webp failed to decode libwebp output");
    assert_eq!(final_image.width, width);
    assert_eq!(final_image.height, height);
    assert_eq!(
        final_image.frames.len(),
        1,
        "lossless still image should produce exactly one frame"
    );

    // Final assertion: byte-exact RGBA, alpha must always match,
    // RGB only enforced on non-transparent pixels (libwebp's
    // re-encode in step 3 zeros transparent-pixel RGB).
    assert_rgba_allow_transparent_rgb_differences(&original_rgba, &final_image.frames[0].rgba);
}
