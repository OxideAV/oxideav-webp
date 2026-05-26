//! Published-shape §2.5 `VP8 ` lossy encode entry points — the
//! `encode_vp8_lossy_yuv420p` / `_yuva420p` / `_rgba` / `_rgb24` surface
//! described in `API-COMPAT.md`, routed through the `oxideav-vp8` sibling
//! crate's [`oxideav_vp8::encode_keyframe`] driver.
//!
//! These functions wrap a single-keyframe VP8 bitstream into a complete
//! `.webp` file. Three §2.7 layouts are emitted:
//!
//! * **Simple lossy** (`RIFF` + `VP8 `) — when no alpha plane is supplied
//!   and the [`WebpMetadata`] is empty.
//! * **Extended lossy** (`RIFF` + `VP8X` + `ICCP?` + `VP8 ` + `EXIF?` +
//!   `XMP ?`) — when ICC / Exif / XMP metadata is present (still no
//!   alpha).
//! * **Extended lossy + alpha** (`RIFF` + `VP8X` + `ICCP?` + `ALPH` +
//!   `VP8 ` + `EXIF?` + `XMP ?`) — when the input carries a per-pixel
//!   alpha plane. The `VP8X` flag byte declares the `L` (alpha) bit; the
//!   `ALPH` chunk is emitted in §2.7.1.2 *method 0* (raw, uncompressed)
//!   form with the *None* filter — one byte of info (`0x00`) followed by
//!   `width * height` raw alpha bytes.
//!
//! ### Quality → quantiser mapping
//!
//! All four entry points take a published-shape `quality: f32` in
//! `0.0..=100.0` (default 75.0) — higher is better. RFC 6386 §9.6's
//! `y_ac_qi` runs the other way (`0..=127`, lower is better), so the
//! mapping is the linear inversion
//!
//! ```text
//!   q   = clamp(quality, 0, 100)
//!   qi  = round(127 * (1 - q / 100))
//! ```
//!
//! Special-cased so `quality = 100` lands at the minimum-quantiser
//! `y_ac_qi = 0` and `quality = 0` lands at the maximum `y_ac_qi = 127`,
//! with `quality = 75` ≈ `y_ac_qi = 32` (the [`KeyframeParams`] default).
//!
//! ### Colourspace conversion (RGB / RGBA entry points)
//!
//! The RGB / RGBA paths convert the source bytes to a tightly-packed
//! I420 plane in scan-line order via the **full-range ITU-R BT.601**
//! forward matrix (the inverse of the matrix the §2.5 decode path uses
//! to convert YCbCr back to RGB — see [`crate::vp8_decode`]). RFC 6386
//! §9.2 / RFC 9649 §10 specify BT.601 as VP8's colourspace; the full-
//! range (no 16..235 luma head-room) form matches the decode-side
//! conversion so a round-trip is consistent.
//!
//! Chroma subsampling is 4:2:0 with **2×2 box averaging** at the chroma-
//! plane boundaries: each chroma sample averages the four (or fewer, on
//! a partial right / bottom row) luma-resolution Cb / Cr values that
//! cover its 2×2 footprint. This is the encode-side mirror of the
//! decode-side §9.2 nearest-neighbour upsample (the encoder picks a
//! reasonable downsample; the decoder's upsample is fixed by RFC 6386).
//!
//! Spec sources consulted: RFC 9649 (WebP container) §2.5, §2.7,
//! §2.7.1.2; RFC 6386 (VP8) §9.1, §9.2, §9.6; ITU-R BT.601.

use crate::{build, container, Error, WebpError, WebpMetadata};

// `Error` is used inside `frame_lossy` to convert `build::BuildError` into
// the published `WebpError`.
use oxideav_vp8::{encode_keyframe, EncodeError, I420Frame, KeyframeParams};

/// Default published-shape quality knob the entry points fall back to when
/// the caller passes a sentinel out-of-range value. Matches the
/// published-0.1.5 default of `75.0`.
pub const DEFAULT_QUALITY: f32 = 75.0;

/// Map a published-shape `quality: f32` in `0..=100` onto RFC 6386 §9.6's
/// `y_ac_qi: u8` in `0..=127` (lower = better).
///
/// `NaN` or out-of-range inputs fall back to [`DEFAULT_QUALITY`].
fn quality_to_qindex(quality: f32) -> u8 {
    let q = if quality.is_nan() {
        DEFAULT_QUALITY
    } else {
        quality.clamp(0.0, 100.0)
    };
    // 0..=100 (good) → 127..=0 (bad). The rounding is round-half-to-even
    // via `(x + 0.5).floor()`; `0` → 127, `100` → 0, `75` → 32.
    let qi = (127.0 * (1.0 - q / 100.0)) + 0.5;
    qi.clamp(0.0, 127.0) as u8
}

/// Map an `oxideav-vp8` encode failure onto the coarse [`WebpError`].
///
/// Every encoder-side failure (a malformed input dimension, an out-of-
/// range quantiser, a token-emit refusal …) collapses to
/// [`WebpError::InvalidData`] — the caller-facing shape only distinguishes
/// "the input was rejected" from "this build does not support the feature".
impl From<EncodeError> for WebpError {
    fn from(_: EncodeError) -> Self {
        WebpError::InvalidData
    }
}

/// Encode a complete `.webp` file from a packed I420 source.
///
/// `y` is `width * height` luma bytes, scan-line order. `u` and `v` are
/// each `((width + 1) / 2) * ((height + 1) / 2)` chroma bytes (4:2:0).
/// `quality` is the published-shape `0..=100` knob; values outside the
/// range are clamped, `NaN` falls back to [`DEFAULT_QUALITY`].
///
/// The output layout depends on whether `meta` is empty:
///
/// * Empty `meta` → simple `RIFF` + `VP8 ` (no `VP8X` chunk).
/// * Non-empty `meta` → extended `RIFF` + `VP8X` + `ICCP?` + `VP8 ` +
///   `EXIF?` + `XMP ?`. The `VP8X` canvas dimensions match `width`/
///   `height`; the bitstream's own §9.1 dimensions are also `width`/
///   `height`.
pub fn encode_vp8_lossy_yuv420p(
    width: u32,
    height: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    quality: f32,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let bitstream = encode_vp8_keyframe_packed(width, height, y, u, v, quality)?;
    frame_lossy(width, height, &bitstream, None, meta)
}

/// Encode a complete `.webp` file from a packed I420 source plus a
/// separate per-pixel alpha plane.
///
/// `alpha` is `width * height` bytes in scan-line order — the same shape
/// the §2.7.1.2 `ALPH` decode path produces. The output is always the
/// extended `VP8X` layout with the `L` (alpha) flag set; the alpha plane
/// is emitted as a raw (method-0, no-filter) §2.7.1.2 `ALPH` chunk
/// preceding the `VP8 ` chunk.
///
/// VP8 itself carries no alpha; the alpha plane lives in a separate chunk
/// per RFC 9649 §2.7.1.2 — see [`crate::alph::decode_alpha`] for the
/// inverse path.
///
/// The argument count matches the published 0.1.5 shape (eight: width,
/// height, three planes, alpha, quality, metadata); the
/// `too_many_arguments` lint is allowed at the published surface boundary.
#[allow(clippy::too_many_arguments)]
pub fn encode_vp8_lossy_yuva420p(
    width: u32,
    height: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    alpha: &[u8],
    quality: f32,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let n = (width as usize)
        .checked_mul(height as usize)
        .ok_or(WebpError::InvalidData)?;
    if alpha.len() != n {
        return Err(WebpError::InvalidData);
    }
    let bitstream = encode_vp8_keyframe_packed(width, height, y, u, v, quality)?;
    frame_lossy(width, height, &bitstream, Some(alpha), meta)
}

/// Encode a complete `.webp` file from interleaved RGBA bytes.
///
/// `rgba` is `width * height * 4` bytes in scan-line order — the
/// `oxideav_core::PixelFormat::Rgba` layout the workspace's image crates
/// share, and the exact buffer
/// `image::ImageBuffer<Rgba<u8>, Vec<u8>>` exposes via
/// [`image::ImageBuffer::into_raw`]. The conversion is
/// full-range BT.601 (R/G/B → Y/Cb/Cr) with 2×2 box-averaged chroma.
///
/// If any pixel's alpha differs from `0xff`, the alpha plane is emitted
/// alongside the VP8 bitstream in a §2.7.1.2 `ALPH` chunk (method 0, no
/// filter) — see [`encode_vp8_lossy_yuva420p`]. An all-opaque image
/// emits the same simple / extended layout
/// [`encode_vp8_lossy_yuv420p`] would.
///
/// [`image::ImageBuffer<Rgba<u8>, Vec<u8>>`]: https://docs.rs/image/
/// [`image::ImageBuffer::into_raw`]: https://docs.rs/image/
pub fn encode_vp8_lossy_rgba(
    width: u32,
    height: u32,
    rgba: &[u8],
    quality: f32,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let (n, _stride4) = check_rgba_dimensions(width, height, rgba.len(), 4)?;
    let (y, u, v) = rgba_planes_to_i420(width, height, rgba, 4);
    // Scan the alpha channel for non-opaque pixels; if any exist, attach
    // an ALPH chunk. Most RGB-sourced photographic content is fully
    // opaque so the common case stays on the simple / extended-no-alpha
    // shape.
    let alpha = (0..n).map(|i| rgba[i * 4 + 3]).collect::<Vec<u8>>();
    let has_alpha = alpha.iter().any(|&a| a != 0xff);

    let bitstream = encode_vp8_keyframe_packed(width, height, &y, &u, &v, quality)?;
    let alpha_slice = if has_alpha {
        Some(alpha.as_slice())
    } else {
        None
    };
    frame_lossy(width, height, &bitstream, alpha_slice, meta)
}

/// Encode a complete `.webp` file from interleaved RGB24 bytes
/// (3 bytes per pixel — no alpha channel).
///
/// `rgb` is `width * height * 3` bytes in scan-line order — the
/// `image::ImageBuffer<Rgb<u8>, Vec<u8>>` backing buffer. The output is
/// always treated as opaque (no `ALPH` chunk, no `VP8X` alpha flag), per
/// the published 0.1.5 contract that `Rgb24` inputs map to the simple
/// layout. The simple / extended choice still keys on `meta` so a caller
/// that wants ICC / Exif / XMP metadata still pays a `VP8X` chunk.
pub fn encode_vp8_lossy_rgb24(
    width: u32,
    height: u32,
    rgb: &[u8],
    quality: f32,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let (_n, _stride3) = check_rgba_dimensions(width, height, rgb.len(), 3)?;
    let (y, u, v) = rgba_planes_to_i420(width, height, rgb, 3);
    let bitstream = encode_vp8_keyframe_packed(width, height, &y, &u, &v, quality)?;
    frame_lossy(width, height, &bitstream, None, meta)
}

// ─────────────────────── internals ───────────────────────

/// Validate the buffer length matches `width * height * channels` and
/// return `(pixel_count, row_stride_bytes)`.
fn check_rgba_dimensions(
    width: u32,
    height: u32,
    buf_len: usize,
    channels: usize,
) -> Result<(usize, usize), WebpError> {
    if width == 0 || height == 0 {
        return Err(WebpError::InvalidData);
    }
    let n = (width as usize)
        .checked_mul(height as usize)
        .ok_or(WebpError::InvalidData)?;
    let expected = n.checked_mul(channels).ok_or(WebpError::InvalidData)?;
    if buf_len != expected {
        return Err(WebpError::InvalidData);
    }
    let stride = (width as usize)
        .checked_mul(channels)
        .ok_or(WebpError::InvalidData)?;
    Ok((n, stride))
}

/// Convert an interleaved RGB(A) buffer to a tightly-packed I420 set of
/// planes via the full-range ITU-R BT.601 forward matrix.
///
/// `channels` is 3 (RGB24) or 4 (RGBA). Alpha bytes (if any) are dropped
/// by the caller via a separate scan.
fn rgba_planes_to_i420(
    width: u32,
    height: u32,
    pixels: &[u8],
    channels: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    let mut y_plane = vec![0u8; w * h];
    // Accumulate 2×2-block Cb / Cr sums in i32 then average; the count
    // covers partial-row boundaries (1 / 2 / 4 samples per block).
    let mut u_sum = vec![0i32; cw * ch];
    let mut v_sum = vec![0i32; cw * ch];
    let mut count = vec![0u32; cw * ch];

    for py in 0..h {
        for px in 0..w {
            let idx = py * w + px;
            let off = idx * channels;
            let r = pixels[off] as i32;
            let g = pixels[off + 1] as i32;
            let b = pixels[off + 2] as i32;

            // BT.601 full-range forward (Q16):
            //   Y  =  0.299    R + 0.587    G + 0.114    B
            //   Cb = -0.168736 R - 0.331264 G + 0.5      B + 128
            //   Cr =  0.5      R - 0.418688 G - 0.081312 B + 128
            //
            // Q16 coefficients:
            //   0.299    -> 19595
            //   0.587    -> 38470
            //   0.114    -> 7471      (sum = 65536 = 1.0)
            //   0.168736 -> 11059
            //   0.331264 -> 21709
            //   0.5      -> 32768     (sum = 65536 = 1.0)
            //   0.418688 -> 27439
            //   0.081312 -> 5329      (sum = 65536 = 1.0)
            const HALF: i32 = 1 << 15;
            let yi = (19_595 * r + 38_470 * g + 7_471 * b + HALF) >> 16;
            let cb = (-11_059 * r - 21_709 * g + 32_768 * b + HALF) >> 16;
            let cr = (32_768 * r - 27_439 * g - 5_329 * b + HALF) >> 16;

            y_plane[idx] = yi.clamp(0, 255) as u8;
            let cbi = (cb + 128).clamp(0, 255);
            let cri = (cr + 128).clamp(0, 255);

            let cx = px / 2;
            let cy = py / 2;
            let cidx = cy * cw + cx;
            u_sum[cidx] += cbi;
            v_sum[cidx] += cri;
            count[cidx] += 1;
        }
    }

    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];
    for i in 0..u_plane.len() {
        let n = count[i].max(1);
        u_plane[i] = ((u_sum[i] + (n as i32) / 2) / (n as i32)).clamp(0, 255) as u8;
        v_plane[i] = ((v_sum[i] + (n as i32) / 2) / (n as i32)).clamp(0, 255) as u8;
    }

    (y_plane, u_plane, v_plane)
}

/// Drive `oxideav_vp8::encode_keyframe` with a packed I420 source and the
/// quality-derived [`KeyframeParams`].
fn encode_vp8_keyframe_packed(
    width: u32,
    height: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    quality: f32,
) -> Result<Vec<u8>, WebpError> {
    if width == 0 || height == 0 {
        return Err(WebpError::InvalidData);
    }
    let exp_y = (width as usize)
        .checked_mul(height as usize)
        .ok_or(WebpError::InvalidData)?;
    let cw = width.div_ceil(2) as usize;
    let ch = height.div_ceil(2) as usize;
    let exp_uv = cw.checked_mul(ch).ok_or(WebpError::InvalidData)?;
    if y.len() != exp_y || u.len() != exp_uv || v.len() != exp_uv {
        return Err(WebpError::InvalidData);
    }

    let frame = I420Frame::packed(width, height, y, u, v);
    let params = KeyframeParams {
        y_ac_qi: quality_to_qindex(quality),
        ..KeyframeParams::default()
    };
    encode_keyframe(&frame, &params).map_err(WebpError::from)
}

/// Build a `.webp` around a VP8 bitstream, with optional alpha plane and
/// optional file-level metadata.
fn frame_lossy(
    width: u32,
    height: u32,
    vp8_bitstream: &[u8],
    alpha: Option<&[u8]>,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let has_alpha = alpha.is_some();

    // Simple layout: no alpha, no metadata. RIFF + VP8 only.
    if !has_alpha && meta.is_empty() {
        return build::build_webp_file(vp8_bitstream, build::ImageKind::Lossy, width, height)
            .map_err(Error::from)
            .map_err(WebpError::from);
    }

    // Extended layout: VP8X declares which features are present. §2.7
    // chunk order: VP8X, ICCP?, ALPH?, VP8/VP8L, EXIF?, XMP?.
    let flags = build::Vp8xFlags {
        has_iccp: meta.icc.is_some(),
        has_alpha,
        has_exif: meta.exif.is_some(),
        has_xmp: meta.xmp.is_some(),
        has_animation: false,
    };
    let vp8x_payload = build::build_vp8x_chunk(width, height, flags)
        .map_err(Error::from)
        .map_err(WebpError::from)?;

    let mut body = Vec::new();
    let mut push_chunk = |fourcc, payload: &[u8]| -> Result<(), WebpError> {
        let chunk = build::build_chunk(fourcc, payload)
            .map_err(Error::from)
            .map_err(WebpError::from)?;
        body.extend_from_slice(&chunk);
        Ok(())
    };

    push_chunk(container::fourcc::VP8X, &vp8x_payload)?;
    if let Some(icc) = meta.icc {
        push_chunk(container::fourcc::ICCP, icc)?;
    }
    if let Some(alpha_plane) = alpha {
        // §2.7.1.2 method 0 (raw uncompressed), no filter, no
        // preprocessing: a single `0x00` info byte followed by
        // `width * height` raw alpha bytes.
        let mut alph_payload = Vec::with_capacity(1 + alpha_plane.len());
        alph_payload.push(0x00);
        alph_payload.extend_from_slice(alpha_plane);
        push_chunk(container::fourcc::ALPH, &alph_payload)?;
    }
    push_chunk(container::fourcc::VP8, vp8_bitstream)?;
    if let Some(exif) = meta.exif {
        push_chunk(container::fourcc::EXIF, exif)?;
    }
    if let Some(xmp) = meta.xmp {
        push_chunk(container::fourcc::XMP, xmp)?;
    }

    // §2.4 file framing.
    let file_size = (body.len() as u64) + 4;
    if file_size > u64::from(u32::MAX) {
        return Err(WebpError::InvalidData);
    }
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&container::fourcc::RIFF);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&container::fourcc::WEBP);
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_maps_to_qindex_bounds_and_default() {
        assert_eq!(quality_to_qindex(100.0), 0);
        assert_eq!(quality_to_qindex(0.0), 127);
        // 75 → ~32 (the KeyframeParams default).
        assert_eq!(quality_to_qindex(75.0), 32);
        // Out-of-range inputs clamp into the table.
        assert_eq!(quality_to_qindex(200.0), 0);
        assert_eq!(quality_to_qindex(-50.0), 127);
        // NaN falls back to default 75 → 32.
        assert_eq!(quality_to_qindex(f32::NAN), 32);
    }

    #[test]
    fn rgb_to_i420_neutral_grey_is_neutral_chroma() {
        // A flat grey 4x4 image: every chroma sample should land near 128
        // (neutral) and every luma sample near the grey value.
        const W: u32 = 4;
        const H: u32 = 4;
        let mut rgb = Vec::with_capacity((W * H * 3) as usize);
        for _ in 0..W * H {
            rgb.extend_from_slice(&[100, 100, 100]);
        }
        let (y, u, v) = rgba_planes_to_i420(W, H, &rgb, 3);
        for &yi in &y {
            assert_eq!(yi, 100, "grey luma should equal source grey");
        }
        for (&ui, &vi) in u.iter().zip(v.iter()) {
            assert!((127..=129).contains(&ui), "neutral Cb ≈ 128, got {ui}");
            assert!((127..=129).contains(&vi), "neutral Cr ≈ 128, got {vi}");
        }
    }

    #[test]
    fn rgba_alpha_scan_promotes_to_extended() {
        // A 2x2 RGBA with one non-opaque pixel must produce a file whose
        // §2.7.1.1 VP8X chunk has the `L` (alpha) flag set, and an ALPH
        // chunk following the VP8X.
        const W: u32 = 2;
        const H: u32 = 2;
        let rgba = vec![
            255, 0, 0, 255, // opaque red
            0, 255, 0, 128, // half-transparent green
            0, 0, 255, 255, // opaque blue
            255, 255, 0, 255, // opaque yellow
        ];
        let file =
            encode_vp8_lossy_rgba(W, H, &rgba, 75.0, &WebpMetadata::default()).expect("encode");

        // Walk the container and confirm VP8X exists with alpha bit set.
        let c = container::parse(&file).expect("parse RIFF");
        let vp8x_chunk = c
            .first_chunk_with_fourcc(container::fourcc::VP8X)
            .expect("VP8X present");
        let hdr = crate::vp8x::Vp8xHeader::parse(vp8x_chunk.payload(&file)).expect("VP8X parse");
        assert!(hdr.has_alpha, "alpha bit set in VP8X");
        // ALPH chunk present with method-0 info byte.
        let alph = c
            .first_chunk_with_fourcc(container::fourcc::ALPH)
            .expect("ALPH present");
        assert_eq!(
            alph.payload(&file)[0],
            0x00,
            "ALPH info byte is method-0, no-filter"
        );
        // VP8 chunk present.
        assert!(c.first_chunk_with_fourcc(container::fourcc::VP8).is_some());
    }

    #[test]
    fn rgba_all_opaque_is_simple_layout() {
        // Fully-opaque RGBA → simple (no VP8X) layout per the published
        // contract: empty metadata, no alpha-flagged extended layout.
        const W: u32 = 2;
        const H: u32 = 2;
        let rgba = vec![
            10, 20, 30, 255, //
            40, 50, 60, 255, //
            70, 80, 90, 255, //
            100, 110, 120, 255, //
        ];
        let file =
            encode_vp8_lossy_rgba(W, H, &rgba, 75.0, &WebpMetadata::default()).expect("encode");
        let c = container::parse(&file).expect("parse RIFF");
        assert!(c.first_chunk_with_fourcc(container::fourcc::VP8X).is_none());
        assert!(c.first_chunk_with_fourcc(container::fourcc::ALPH).is_none());
        assert!(c.first_chunk_with_fourcc(container::fourcc::VP8).is_some());
    }

    #[test]
    fn rgb24_metadata_promotes_to_extended_but_no_alpha() {
        // Rgb24 input + metadata → VP8X declares I/E/X but NOT L.
        const W: u32 = 2;
        const H: u32 = 2;
        let rgb = vec![
            255, 0, 0, //
            0, 255, 0, //
            0, 0, 255, //
            128, 128, 128, //
        ];
        let icc = b"fake-icc".to_vec();
        let meta = WebpMetadata {
            icc: Some(&icc),
            exif: None,
            xmp: None,
        };
        let file = encode_vp8_lossy_rgb24(W, H, &rgb, 75.0, &meta).expect("encode");
        let c = container::parse(&file).expect("parse");
        let vp8x_chunk = c
            .first_chunk_with_fourcc(container::fourcc::VP8X)
            .expect("VP8X");
        let hdr = crate::vp8x::Vp8xHeader::parse(vp8x_chunk.payload(&file)).expect("VP8X parse");
        assert!(hdr.has_iccp);
        assert!(!hdr.has_alpha, "rgb24 input must not flag alpha");
        // No ALPH chunk for opaque rgb24 inputs.
        assert!(c.first_chunk_with_fourcc(container::fourcc::ALPH).is_none());
    }

    #[test]
    fn yuv420p_simple_layout_round_trips_dimensions() {
        // A neutral grey I420 source (128 luma, 128 chroma) must encode
        // cleanly; the encoded file's parsed VP8 keyframe header reports
        // the source dimensions.
        const W: u32 = 16;
        const H: u32 = 16;
        let y = vec![128u8; (W * H) as usize];
        let u = vec![128u8; ((W / 2) * (H / 2)) as usize];
        let v = vec![128u8; ((W / 2) * (H / 2)) as usize];
        let file = encode_vp8_lossy_yuv420p(W, H, &y, &u, &v, 75.0, &WebpMetadata::default())
            .expect("encode");
        // Round-trip the container.
        let c = container::parse(&file).expect("parse");
        let vp8 = c
            .first_chunk_with_fourcc(container::fourcc::VP8)
            .expect("VP8 present");
        let lossy =
            crate::vp8_chunk::WebpLossyChunk::from_chunk(&file, vp8).expect("VP8 keyframe header");
        assert_eq!(u32::from(lossy.width()), W);
        assert_eq!(u32::from(lossy.height()), H);
    }

    #[test]
    fn yuva420p_emits_alph_chunk_with_alpha_plane() {
        const W: u32 = 4;
        const H: u32 = 4;
        let y = vec![100u8; (W * H) as usize];
        let u = vec![128u8; ((W / 2) * (H / 2)) as usize];
        let v = vec![128u8; ((W / 2) * (H / 2)) as usize];
        // Mix of opaque + transparent alpha; the ALPH chunk carries them
        // verbatim under method 0 + no filter.
        let alpha: Vec<u8> = (0..(W * H)).map(|i| (i * 17) as u8).collect();

        let file =
            encode_vp8_lossy_yuva420p(W, H, &y, &u, &v, &alpha, 75.0, &WebpMetadata::default())
                .expect("encode");
        let c = container::parse(&file).expect("parse");
        let alph = c
            .first_chunk_with_fourcc(container::fourcc::ALPH)
            .expect("ALPH present");
        let payload = alph.payload(&file);
        assert_eq!(payload[0], 0x00, "method-0 info byte");
        assert_eq!(&payload[1..], &alpha[..], "alpha plane raw round-trips");
        // ALPH decode helper also reproduces the alpha plane.
        let decoded = crate::decode_alpha_plane(&file)
            .expect("decode alpha")
            .expect("alpha present");
        assert_eq!(decoded, alpha);
    }

    #[test]
    fn yuv420p_rejects_short_plane() {
        const W: u32 = 4;
        const H: u32 = 4;
        let y = vec![128u8; (W * H) as usize];
        // Chroma plane one byte short of the (2×2) expectation.
        let u = vec![128u8; ((W / 2) * (H / 2)) as usize - 1];
        let v = vec![128u8; ((W / 2) * (H / 2)) as usize];
        let err = encode_vp8_lossy_yuv420p(W, H, &y, &u, &v, 75.0, &WebpMetadata::default())
            .expect_err("short plane rejected");
        assert_eq!(err, WebpError::InvalidData);
    }

    #[test]
    fn rgba_rejects_short_buffer() {
        // 2x2 RGBA needs 16 bytes; pass 12.
        let rgba = vec![0u8; 12];
        let err = encode_vp8_lossy_rgba(2, 2, &rgba, 75.0, &WebpMetadata::default())
            .expect_err("short buffer rejected");
        assert_eq!(err, WebpError::InvalidData);
    }

    #[test]
    fn rgb24_rejects_short_buffer() {
        // 2x2 RGB24 needs 12 bytes; pass 9.
        let rgb = vec![0u8; 9];
        let err = encode_vp8_lossy_rgb24(2, 2, &rgb, 75.0, &WebpMetadata::default())
            .expect_err("short rgb24 rejected");
        assert_eq!(err, WebpError::InvalidData);
    }
}
