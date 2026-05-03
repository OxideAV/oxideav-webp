//! Runtime libwebp interop for the cross-decode fuzz harnesses.
//!
//! libwebp is loaded via `dlopen` at first call — there is no
//! `webp-sys`-style build-script dep that would pull libwebp source
//! into the workspace's cargo dep tree. Each harness checks
//! [`libwebp::available`] up front and `return`s early when the
//! shared library isn't installed, so fuzz binaries built on a host
//! without libwebp simply do nothing instead of panicking.
//!
//! Install libwebp with `brew install webp` (macOS) or
//! `apt install libwebp7` (Debian/Ubuntu). The loader probes the
//! conventional shared-object names for both platforms.

#![allow(unsafe_code)]

pub mod libwebp {
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
                // accept that risk for fuzz tooling — libwebp is a
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
    /// Cross-decode fuzz harnesses early-return when this is false so
    /// the binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    /// Encode an RGBA image losslessly via `WebPEncodeLosslessRGBA`.
    /// Returns `None` if libwebp isn't available or the encoder
    /// returned a null pointer.
    pub fn encode_lossless(rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        type EncFn =
            unsafe extern "C" fn(*const u8, i32, i32, i32, *mut *mut u8) -> usize;
        type FreeFn = unsafe extern "C" fn(*mut u8);
        let l = lib()?;
        unsafe {
            let enc: Symbol<EncFn> = l.get(b"WebPEncodeLosslessRGBA").ok()?;
            let free: Symbol<FreeFn> = l.get(b"WebPFree").ok()?;
            let mut out: *mut u8 = std::ptr::null_mut();
            let stride = (width as i32).checked_mul(4)?;
            let n = enc(
                rgba.as_ptr(),
                width as i32,
                height as i32,
                stride,
                &mut out,
            );
            if out.is_null() || n == 0 {
                return None;
            }
            let v = std::slice::from_raw_parts(out, n).to_vec();
            free(out);
            Some(v)
        }
    }

    /// Encode an RGBA image lossily via `WebPEncodeRGBA`. `quality` is
    /// libwebp's 0.0..=100.0 scale.
    pub fn encode_lossy(rgba: &[u8], width: u32, height: u32, quality: f32) -> Option<Vec<u8>> {
        type EncFn =
            unsafe extern "C" fn(*const u8, i32, i32, i32, f32, *mut *mut u8) -> usize;
        type FreeFn = unsafe extern "C" fn(*mut u8);
        let l = lib()?;
        unsafe {
            let enc: Symbol<EncFn> = l.get(b"WebPEncodeRGBA").ok()?;
            let free: Symbol<FreeFn> = l.get(b"WebPFree").ok()?;
            let mut out: *mut u8 = std::ptr::null_mut();
            let stride = (width as i32).checked_mul(4)?;
            let n = enc(
                rgba.as_ptr(),
                width as i32,
                height as i32,
                stride,
                quality,
                &mut out,
            );
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
        type DecodeIntoFn =
            unsafe extern "C" fn(*const u8, usize, *mut u8, usize, i32) -> *mut u8;
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
