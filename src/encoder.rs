//! `oxideav_core::Encoder` adapter around the bare VP8L bitstream encoder
//! in [`crate::vp8l::encoder`].
//!
//! The encoder accepts a single `Rgba` or `Rgb24` video frame per
//! `send_frame` and emits a RIFF-wrapped `.webp` file on
//! `receive_packet`. Frames whose alpha is uniformly opaque go into a
//! simple `RIFF/WEBP/VP8L` layout; frames that carry alpha information
//! get an extended `RIFF/WEBP/VP8X + VP8L` layout so the VP8X header can
//! advertise the alpha flag and the canvas size — required for any WebP
//! reader that honours the extended format spec.
//!
//! `Rgb24` input is converted on the fly into the encoder's internal
//! `u32` ARGB buffer (one ARGB word per output pixel, alpha forced to
//! `0xff`), without ever materialising an intermediate `Rgba` byte
//! buffer. That matters for callers that decoded a JPEG/PNG-without-alpha
//! into an RGB image and want to re-encode as WebP without paying for
//! the RGB→RGBA expansion (issue #7).
//!
//! Callers that want to stay on the bare-bitstream path (decodable
//! directly by [`crate::vp8l::decode`]) should call
//! [`crate::vp8l::encode_vp8l_argb`] themselves — that entry point
//! remains unchanged.

#[cfg(feature = "registry")]
use std::collections::VecDeque;

#[cfg(feature = "registry")]
use oxideav_core::Encoder;
#[cfg(feature = "registry")]
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, TimeBase, VideoFrame,
};

use crate::error::Result;
#[cfg(feature = "registry")]
use crate::error::WebpError as Error;
#[cfg(feature = "registry")]
use crate::riff::WebpMetadataOwned;
use crate::riff::{build_webp_file, ImageKind, WebpMetadata};
use crate::vp8l::encode_vp8l_argb;
#[cfg(feature = "registry")]
use crate::CODEC_ID_VP8L;

#[cfg(feature = "registry")]
pub fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    make_encoder_with_metadata(params, WebpMetadataOwned::default())
}

/// Build a registry-side VP8L lossless WebP encoder with caller-
/// supplied auxiliary metadata chunks (`ICCP` / `EXIF` / `XMP `).
/// Mirrors the standalone [`encode_vp8l_argb_with_metadata`] entry
/// point — registry-side callers go through this factory so the
/// metadata buffers can outlive the `send_frame` call (the encoder
/// takes ownership).
///
/// The lossy [`crate::encoder_vp8::make_encoder_with_qindex_and_metadata`]
/// / [`crate::encoder_vp8::make_encoder_with_quality_and_metadata`]
/// factories are the matching VP8 lossy entry points; this is the
/// VP8L (lossless) parity.
///
/// Layout notes — same as [`encode_vp8l_argb_with_metadata`]:
/// * Opaque + no metadata → simple `RIFF/WEBP/VP8L` layout.
/// * Opaque + metadata    → extended `RIFF/WEBP/VP8X + VP8L` layout
///   (no `ALPH` chunk, but the VP8X header gets the matching metadata
///   flag bits and the chunks are written in the spec-mandated order).
/// * Alpha (regardless of metadata) → extended `RIFF/WEBP/VP8X + …` —
///   VP8X is mandatory for any frame carrying alpha.
#[cfg(feature = "registry")]
pub fn make_encoder_with_metadata(
    params: &CodecParameters,
    metadata: WebpMetadataOwned,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| oxideav_core::Error::invalid("VP8L encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| oxideav_core::Error::invalid("VP8L encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Rgba);
    if pix != PixelFormat::Rgba && pix != PixelFormat::Rgb24 {
        return Err(oxideav_core::Error::unsupported(format!(
            "VP8L encoder: pixel format {pix:?} not supported — feed Rgba or Rgb24"
        )));
    }

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(CODEC_ID_VP8L);
    output_params.pixel_format = Some(pix);
    output_params.width = Some(width);
    output_params.height = Some(height);

    let time_base = TimeBase::new(1, 1000);

    Ok(Box::new(Vp8lEncoder {
        output_params,
        width,
        height,
        input_format: pix,
        time_base,
        pending: VecDeque::new(),
        eof: false,
        metadata,
    }))
}

#[cfg(feature = "registry")]
struct Vp8lEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    input_format: PixelFormat,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
    /// Caller-supplied auxiliary metadata (`ICCP` / `EXIF` / `XMP `)
    /// to attach to every emitted `.webp` file. Defaults to all-`None`
    /// for the historical [`make_encoder`] factory; the
    /// [`make_encoder_with_metadata`] factory threads through caller-
    /// supplied bytes.
    metadata: WebpMetadataOwned,
}

#[cfg(feature = "registry")]
impl Encoder for Vp8lEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> oxideav_core::Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => {
                return Err(oxideav_core::Error::invalid(
                    "VP8L encoder: video frames only",
                ))
            }
        };
        // Frame dimensions and pixel format are now stream-level — the
        // pipeline upstream is responsible for matching `output_params`.
        let meta_borrow = self.metadata.as_borrow();
        let bytes = match self.input_format {
            PixelFormat::Rgba => encode_frame_rgba(v, self.width, self.height, &meta_borrow)?,
            PixelFormat::Rgb24 => encode_frame_rgb24(v, self.width, self.height, &meta_borrow)?,
            other => {
                return Err(oxideav_core::Error::unsupported(format!(
                    "VP8L encoder: pixel format {other:?} not supported"
                )))
            }
        };
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = pkt.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> oxideav_core::Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
            return Err(oxideav_core::Error::Eof);
        }
        Err(oxideav_core::Error::NeedMore)
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Pack an Rgba `VideoFrame` into ARGB u32 pixels and run the VP8L encoder.
/// Returns a full `.webp` file — simple-layout when the frame is fully
/// opaque (and metadata is empty), extended (VP8X + VP8L, optionally with
/// the caller-supplied metadata chunks) when alpha is present or
/// metadata is non-empty.
#[cfg(feature = "registry")]
fn encode_frame_rgba(
    v: &VideoFrame,
    width: u32,
    height: u32,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if v.planes.is_empty() {
        return Err(Error::invalid("VP8L encoder: frame has no planes"));
    }
    let plane = &v.planes[0];
    if plane.stride < w * 4 {
        return Err(Error::invalid("VP8L encoder: RGBA stride too small"));
    }
    let mut pixels = Vec::with_capacity(w * h);
    let mut has_alpha = false;
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + w * 4];
        for x in 0..w {
            let r = row[x * 4] as u32;
            let g = row[x * 4 + 1] as u32;
            let b = row[x * 4 + 2] as u32;
            let a = row[x * 4 + 3] as u32;
            if a != 0xff {
                has_alpha = true;
            }
            pixels.push((a << 24) | (r << 16) | (g << 8) | b);
        }
    }
    finalize_vp8l_file_with_meta(width, height, &pixels, has_alpha, meta)
}

/// Pack an Rgb24 `VideoFrame` directly into ARGB u32 pixels and run the
/// VP8L encoder. The conversion **streams** through the input — we walk
/// the RGB rows three bytes at a time and push the resulting ARGB word
/// (alpha forced to `0xff`) straight into the encoder's `Vec<u32>`
/// buffer. There is no intermediate `Rgba` byte buffer, which is the
/// whole point of issue #7: a caller that already holds an RGB image
/// (e.g. JPEG, PNG-without-alpha decoded via the `image` crate) avoids
/// both the alloc and the byte-shuffle of building a 4-byte expansion.
///
/// Returns a full `.webp` file in the simple `RIFF/WEBP/VP8L` layout —
/// Rgb24 input is implicitly opaque, so we never need the VP8X+ALPHA
/// extension.
#[cfg(feature = "registry")]
fn encode_frame_rgb24(
    v: &VideoFrame,
    width: u32,
    height: u32,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if v.planes.is_empty() {
        return Err(Error::invalid("VP8L encoder: frame has no planes"));
    }
    let plane = &v.planes[0];
    if plane.stride < w * 3 {
        return Err(Error::invalid("VP8L encoder: RGB24 stride too small"));
    }
    // Streaming RGB → ARGB: one pass over the input, no Rgba alloc.
    let mut pixels = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + w * 3];
        for x in 0..w {
            let r = row[x * 3] as u32;
            let g = row[x * 3 + 1] as u32;
            let b = row[x * 3 + 2] as u32;
            pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
        }
    }
    // Rgb24 has no alpha plane, so the file is always fully opaque.
    finalize_vp8l_file_with_meta(width, height, &pixels, false, meta)
}

/// One-shot VP8L → `.webp` file builder with caller-supplied auxiliary
/// metadata (`ICCP` / `EXIF` / `XMP `). Encodes the ARGB pixel buffer
/// through [`encode_vp8l_argb`] and wraps the resulting bitstream in
/// the RIFF/WEBP container, picking the simple layout when neither
/// alpha nor metadata is present, and the extended `VP8X` layout
/// otherwise (with the matching VP8X flag bits set per chunk).
///
/// This is the standalone public entry point for image-library
/// consumers that want lossless `.webp` output **with** a colour
/// profile / EXIF / XMP attached, without going through the registry
/// trait dance. The lower-level [`crate::riff::build_webp_file`] /
/// [`crate::riff::build_vp8l_with_alpha`] helpers stay available for
/// callers that already hold a VP8L bitstream.
pub fn encode_vp8l_argb_with_metadata(
    width: u32,
    height: u32,
    pixels: &[u32],
    has_alpha: bool,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>> {
    finalize_vp8l_file_with_meta(width, height, pixels, has_alpha, meta)
}

fn finalize_vp8l_file_with_meta(
    width: u32,
    height: u32,
    pixels: &[u32],
    has_alpha: bool,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>> {
    let bitstream = encode_vp8l_argb(width, height, pixels, has_alpha)?;
    if has_alpha {
        // When the frame carries alpha we *must* emit the extended layout
        // (VP8X) per the RIFF container spec — readers that parse only
        // the simple form would otherwise miss the alpha flag. The same
        // VP8X header carries any caller-supplied metadata flags.
        Ok(crate::riff::build_vp8l_with_alpha(
            &bitstream, width, height, meta,
        ))
    } else if meta.any() {
        // Opaque + metadata → extended layout WITHOUT the ALPHA flag.
        // `build_webp_file` honours the metadata fields and only forces
        // ALPHA when an `ALPH` sidecar is supplied — perfect for this
        // path.
        Ok(build_webp_file(
            ImageKind::Vp8lLossless,
            &bitstream,
            width,
            height,
            None,
            meta,
        ))
    } else {
        Ok(build_webp_file(
            ImageKind::Vp8lLossless,
            &bitstream,
            width,
            height,
            None,
            meta,
        ))
    }
}
