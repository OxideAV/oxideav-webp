//! # oxideav-webp
//!
//! Pure-Rust WebP (RFC 9649) image codec: lossless `VP8L` decode + encode,
//! lossy `VP8 ` decode + encode (through the `oxideav-vp8` sibling
//! crate), `ALPH` alpha planes, `ANIM` / `ANMF` animation decode and
//! encode, and the `ICCP` / `EXIF` / `XMP ` metadata chunks — all
//! clean-room against the staged specification text.
//!
//! The crate follows the OxideAV **image-crate API contract**: a small
//! standalone vocabulary at the root, usable with
//! `default-features = false` and no `oxideav-core`, returning raw
//! `Vec<u8>` pixels.
//!
//! ```no_run
//! # fn main() -> Result<(), oxideav_webp::Error> {
//! let bytes = std::fs::read("in.webp").map_err(oxideav_webp::Error::Io)?;
//! if oxideav_webp::probe(&bytes) {
//!     let info = oxideav_webp::info(&bytes)?;          // header only
//!     let img = oxideav_webp::decode(&bytes)?;         // WebpImage, native layout
//!     let rgba: Vec<u8> = img.to_rgba8();              // packed RGBA, 4 × width per row
//!     let (w, h) = (img.width(), img.height());
//!     let _ = info;
//!
//!     let opts = oxideav_webp::EncodeOptions::default(); // lossless
//!     let out = oxideav_webp::encode_rgba8(w, h, &rgba, &opts)?;
//!     std::fs::write("out.webp", out).map_err(oxideav_webp::Error::Io)?;
//! }
//! # Ok(()) }
//! ```
//!
//! * [`probe`] / [`info`] / [`decode`] / [`decode_with`] / [`decode_rgb8`] /
//!   [`decode_rgba8`] / [`decode_all`] / [`decode_from`] — decode side.
//!   [`decode`] returns the **native** layout: [`PixelFormat::Rgba`] for a
//!   lossless image, [`PixelFormat::Yuv420P`] / [`PixelFormat::Yuva420P`]
//!   (limited-range BT.601 planes, plus the `ALPH` plane) for a lossy one.
//!   [`WebpImage::to_rgb8`] / [`WebpImage::to_rgba8`] convert exactly.
//! * [`encode`] / [`encode_rgb8`] / [`encode_rgba8`] / [`encode_to`] /
//!   [`encode_animation`] — encode side. [`EncodeOptions::default`] is
//!   lossless; [`EncodeOptions::with_quality`] selects the lossy `VP8 `
//!   path.
//! * [`WebpImage`], [`RgbImage`], [`RgbaImage`], [`Plane`], [`ColorInfo`],
//!   [`Metadata`], [`ImageInfo`], [`Frame`], [`EncodeOptions`],
//!   [`DecodeOptions`], [`PixelFormat`], [`WebpError`] / [`Error`] — the
//!   contract types.
//!
//! With the default `registry` feature, [`register`] installs the
//! framework decoder / encoders into an `oxideav_core::RuntimeContext`;
//! the framework path is a thin adapter over the functions above.
//!
//! The parser / builder modules (`container`, `vp8x`, `alph`, `anim`,
//! `anmf`, `build`, `vp8l_*`, …) are the clean-room implementation and
//! stay public for the test / fuzz / bench harnesses; they are
//! `#[doc(hidden)]` and not part of the stable surface.

#![warn(missing_debug_implementations)]
// Opt-in `std::simd` acceleration of the hottest pixel-repack /
// inverse-transform loops. Nightly-only because `portable_simd` is
// still an unstable feature; every SIMD path has a stable scalar
// fallback that produces byte-identical output. See `BENCHMARKS.md`
// and the `simd` cargo feature in `Cargo.toml`.
#![cfg_attr(feature = "simd", feature(portable_simd))]

// ───────────────────────────── contract surface ──────────────────────────

mod api;
mod image;
mod yuv;

pub use api::{
    animation_params, decode, decode_all, decode_all_with, decode_from, decode_rgb8, decode_rgba8,
    decode_with, encode, encode_animation, encode_animation_frames, encode_rgb8, encode_rgba8,
    encode_to, info, probe, read_metadata, AnimFrame, AnimFrameMode, DecodeOptions, DeltaConfig,
    DownsampleKernel, EncodeOptions, MAX_DIMENSION,
};
pub use image::{
    ColorInfo, ColorRange, Frame, ImageInfo, Metadata, Palette, PixelFormat, Plane, RgbImage,
    RgbaImage, WebpImage, WebpPixelFormat,
};

pub mod error;
pub use error::WebpError;

/// Contract alias: every image crate exposes `Error` at its root.
pub type Error = WebpError;

// ───────────────────────── clean-room implementation ─────────────────────

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod alph;
#[doc(hidden)]
pub mod anim;
#[doc(hidden)]
pub mod anim_encode;
#[doc(hidden)]
pub mod anmf;
#[doc(hidden)]
pub mod build;
#[doc(hidden)]
pub mod container;
#[doc(hidden)]
pub mod decoder;
#[doc(hidden)]
pub mod demux;
#[doc(hidden)]
pub mod encoder;
#[doc(hidden)]
pub mod encoder_anim;
pub mod encoder_vp8;
#[doc(hidden)]
pub mod meta_prefix;
#[cfg(feature = "registry")]
pub mod registry;
#[doc(hidden)]
pub mod riff;
#[doc(hidden)]
pub mod vp8_chunk;
#[doc(hidden)]
pub mod vp8_decode;
#[doc(hidden)]
pub mod vp8l;
#[doc(hidden)]
pub mod vp8l_chunk;
#[doc(hidden)]
pub mod vp8l_decode;
#[doc(hidden)]
pub mod vp8l_encode;
#[doc(hidden)]
pub mod vp8l_prefix;
#[doc(hidden)]
pub mod vp8l_stream;
#[doc(hidden)]
pub mod vp8l_transform;
#[doc(hidden)]
pub mod vp8x;

#[cfg(feature = "registry")]
use oxideav_core::RuntimeContext;

#[cfg(feature = "registry")]
pub use registry::{make_decoder, make_encoder, WebpDecoder};

/// Stable codec identifier the VP8L lossless encoder registers under in the
/// codec registry — the published `"webp_vp8l"` name.
pub const CODEC_ID_VP8L: &str = "webp_vp8l";

/// Stable codec identifier the VP8 lossy encoder registers under in the
/// codec registry — the published `"webp_vp8"` name.
pub const CODEC_ID_VP8: &str = "webp_vp8";

// ─────────────────────── hidden container-level helpers ──────────────────
//
// Thin wrappers over the parser / builder modules returning the crate
// error, kept for the test / fuzz / bench harnesses.

/// Walk a `RIFF/WEBP` container per RFC 9649 §2.3–§2.7 and return the
/// structural chunk list; decodes no payload.
#[doc(hidden)]
pub fn parse_container(bytes: &[u8]) -> Result<container::WebpContainer, WebpError> {
    container::parse(bytes).map_err(Into::into)
}

/// Decode a §2.7.1 `VP8X` chunk payload to a typed [`vp8x::Vp8xHeader`].
#[doc(hidden)]
pub fn parse_vp8x_header(payload: &[u8]) -> Result<vp8x::Vp8xHeader, WebpError> {
    vp8x::Vp8xHeader::parse(payload).map_err(Into::into)
}

/// Decode the §2.7.1.2 `ALPH` info byte to a typed [`alph::AlphHeader`].
#[doc(hidden)]
pub fn parse_alph_header(payload: &[u8]) -> Result<alph::AlphHeader, WebpError> {
    alph::AlphHeader::parse(payload).map_err(Into::into)
}

/// Decode the §2.7.1.2 `ALPH` chunk of a still file to a `width × height`
/// alpha plane; `Ok(None)` when the file has no `ALPH` chunk. Dimensions
/// come from `VP8X`, else from the `VP8 ` key-frame header.
#[doc(hidden)]
pub fn decode_alpha_plane(bytes: &[u8]) -> Result<Option<Vec<u8>>, WebpError> {
    let c = container::parse(bytes)?;
    let alph_chunk = match c.first_chunk_with_fourcc(container::fourcc::ALPH) {
        Some(chunk) => chunk,
        None => return Ok(None),
    };
    let (width, height) = if let Some(vp8x) = c.first_chunk_with_fourcc(container::fourcc::VP8X) {
        let hdr = vp8x::Vp8xHeader::parse(vp8x.payload(bytes))?;
        (hdr.canvas_width, hdr.canvas_height)
    } else if let Some(vp8) = c.first_chunk_with_fourcc(container::fourcc::VP8) {
        let lossy = vp8_chunk::WebpLossyChunk::from_chunk(bytes, vp8)?;
        (u32::from(lossy.width()), u32::from(lossy.height()))
    } else {
        return Err(WebpError::invalid(
            "ALPH chunk without a VP8X or VP8 dimension source",
        ));
    };
    let plane = alph::decode_alpha(alph_chunk.payload(bytes), width, height)?;
    Ok(Some(plane))
}

/// Decode the §2.7.1.1 `ANIM` chunk payload to a typed [`anim::AnimHeader`].
#[doc(hidden)]
pub fn parse_anim_header(payload: &[u8]) -> Result<anim::AnimHeader, WebpError> {
    anim::AnimHeader::parse(payload).map_err(Into::into)
}

/// Decode a §2.7.1.1 `ANMF` chunk payload's 16-byte header to a typed
/// [`anmf::AnmfHeader`].
#[doc(hidden)]
pub fn parse_anmf_header(payload: &[u8]) -> Result<anmf::AnmfHeader, WebpError> {
    anmf::AnmfHeader::parse(payload).map_err(Into::into)
}

/// Wrap a pre-computed `VP8 ` / `VP8L` payload in a complete `RIFF/WEBP`
/// file (RFC 9649 §2.4 + §2.5 / §2.6 / §2.7).
#[doc(hidden)]
pub fn build_webp_file(
    payload: &[u8],
    image_kind: build::ImageKind,
    canvas_width: u32,
    canvas_height: u32,
) -> Result<Vec<u8>, WebpError> {
    build::build_webp_file(payload, image_kind, canvas_width, canvas_height).map_err(Into::into)
}

/// Build the 10-byte §2.7.1 `VP8X` chunk payload.
#[doc(hidden)]
pub fn build_vp8x_chunk(
    canvas_width: u32,
    canvas_height: u32,
    flags: build::Vp8xFlags,
) -> Result<Vec<u8>, WebpError> {
    build::build_vp8x_chunk(canvas_width, canvas_height, flags).map_err(Into::into)
}

/// Return the typed §2.5 `VP8 ` chunk handle of a lossy file, or
/// `Ok(None)` for a file without one.
#[doc(hidden)]
pub fn extract_lossy_chunk(
    bytes: &[u8],
) -> Result<Option<vp8_chunk::WebpLossyChunk<'_>>, WebpError> {
    let c = container::parse(bytes)?;
    vp8_chunk::extract_lossy(bytes, &c).map_err(Into::into)
}

/// Return the typed §2.6 `VP8L` chunk handle of a lossless file, or
/// `Ok(None)` for a file without one.
#[doc(hidden)]
pub fn extract_lossless_chunk(
    bytes: &[u8],
) -> Result<Option<vp8l_chunk::WebpLosslessChunk<'_>>, WebpError> {
    let c = container::parse(bytes)?;
    vp8l_chunk::extract_lossless(bytes, &c).map_err(Into::into)
}

/// Read the §4 transform list of a lossless file's `VP8L` bitstream, or
/// `Ok(None)` for a file without a `VP8L` chunk.
#[doc(hidden)]
pub fn read_vp8l_transform_list(
    bytes: &[u8],
) -> Result<Option<vp8l_stream::TransformList>, WebpError> {
    let c = container::parse(bytes)?;
    let chunk = match vp8l_chunk::extract_lossless(bytes, &c)? {
        Some(chunk) => chunk,
        None => return Ok(None),
    };
    let mut reader = vp8l_stream::BitReader::new_after_image_header(chunk.bitstream());
    let list = vp8l_stream::TransformList::read(&mut reader)?;
    Ok(Some(list))
}

/// Decode a lossless file's `VP8L` bitstream to the ARGB
/// [`vp8l_decode::DecodedImage`] (before the RGBA repack), or `Ok(None)`
/// for a file without a `VP8L` chunk.
#[doc(hidden)]
pub fn decode_lossless_image(bytes: &[u8]) -> Result<Option<vp8l_decode::DecodedImage>, WebpError> {
    let c = container::parse(bytes)?;
    let chunk = match vp8l_chunk::extract_lossless(bytes, &c)? {
        Some(chunk) => chunk,
        None => return Ok(None),
    };
    let image = vp8l_transform::decode_lossless(chunk.bitstream(), chunk.width(), chunk.height())?;
    Ok(Some(image))
}

// ────────────────────────────── metadata types ───────────────────────────

/// Borrowed file-level metadata for the low-level encode helpers — the
/// §2.7.1.4 `ICCP`, §2.7.1.5 `EXIF`, and §2.7.1.5 `XMP ` payloads to embed,
/// each `None` to omit the chunk. The contract-level counterpart is the
/// owned [`Metadata`] on [`WebpImage`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebpMetadata<'a> {
    /// §2.7.1.4 `ICCP` payload to embed, if any.
    pub icc: Option<&'a [u8]>,
    /// §2.7.1.5 `EXIF` payload to embed, if any.
    pub exif: Option<&'a [u8]>,
    /// §2.7.1.5 `XMP ` payload to embed, if any.
    pub xmp: Option<&'a [u8]>,
}

impl WebpMetadata<'_> {
    /// `true` if every field is `None`.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none()
    }
}

impl<'a> From<&'a Metadata> for WebpMetadata<'a> {
    fn from(m: &'a Metadata) -> Self {
        Self {
            icc: m.icc.as_deref(),
            exif: m.exif.as_deref(),
            xmp: m.xmp.as_deref(),
        }
    }
}

/// Owned file-level metadata — the registry-side counterpart of the
/// borrowed [`WebpMetadata`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct WebpMetadataOwned {
    /// §2.7.1.4 `ICCP` payload to embed, if any.
    pub icc: Option<Vec<u8>>,
    /// §2.7.1.5 `EXIF` payload to embed, if any.
    pub exif: Option<Vec<u8>>,
    /// §2.7.1.5 `XMP ` payload to embed, if any.
    pub xmp: Option<Vec<u8>>,
}

impl WebpMetadataOwned {
    /// Borrow as a [`WebpMetadata`] for an encode call.
    pub fn as_borrowed(&self) -> WebpMetadata<'_> {
        WebpMetadata {
            icc: self.icc.as_deref(),
            exif: self.exif.as_deref(),
            xmp: self.xmp.as_deref(),
        }
    }

    /// `true` if every field is `None`.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none()
    }
}

impl From<WebpMetadataOwned> for Metadata {
    fn from(m: WebpMetadataOwned) -> Self {
        Metadata {
            icc: m.icc,
            exif: m.exif,
            xmp: m.xmp,
            gamma: None,
        }
    }
}

impl From<Metadata> for WebpMetadataOwned {
    fn from(m: Metadata) -> Self {
        Self {
            icc: m.icc,
            exif: m.exif,
            xmp: m.xmp,
        }
    }
}

/// The pre-contract name of [`Metadata`].
#[deprecated(since = "0.3.0", note = "use `oxideav_webp::Metadata`")]
pub type WebpFileMetadata = Metadata;

// ───────────────────── bare-VP8L (format-specific) encode ────────────────

/// Encode an ARGB image to a **bare** §2.6 / §3.4 `VP8L` bitstream — the
/// chunk payload (image header + image stream) with **no** RIFF wrapper.
///
/// `argb` is `width × height` packed `(a << 24) | (r << 16) | (g << 8) | b`
/// values in scan-line order. The §3.4 `alpha_is_used` bit is set iff any
/// pixel's alpha is not `0xff`. Wrapping the result with
/// [`build_webp_file`]`(.., ImageKind::Lossless, ..)` yields a complete
/// `.webp`.
pub fn encode_vp8l_argb(argb: &[u32], width: u32, height: u32) -> Result<Vec<u8>, WebpError> {
    vp8l_encode::encode_vp8l_argb(argb, width, height).map_err(Into::into)
}

/// [`encode_vp8l_argb`] with the §3.4 `alpha_is_used` bit set explicitly.
#[doc(hidden)]
pub fn encode_vp8l_argb_with(
    argb: &[u32],
    width: u32,
    height: u32,
    has_alpha: bool,
) -> Result<Vec<u8>, WebpError> {
    vp8l_encode::encode_vp8l_argb_with(argb, width, height, has_alpha).map_err(Into::into)
}

/// Encode an ARGB image to a complete lossless `.webp`, embedding the
/// supplied metadata: the simple `VP8L` layout when `has_alpha` is false
/// and `meta` is empty, else the §2.7 `VP8X` layout (`VP8X`, `ICCP`,
/// `VP8L`, `EXIF`, `XMP `).
#[doc(hidden)]
pub fn encode_vp8l_argb_with_metadata(
    width: u32,
    height: u32,
    argb: &[u32],
    has_alpha: bool,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let payload = encode_vp8l_argb_with(argb, width, height, has_alpha)?;
    if !has_alpha && meta.is_empty() {
        return build::build_webp_file(&payload, build::ImageKind::Lossless, width, height)
            .map_err(Into::into);
    }
    let flags = build::Vp8xFlags {
        has_iccp: meta.icc.is_some(),
        has_alpha,
        has_exif: meta.exif.is_some(),
        has_xmp: meta.xmp.is_some(),
        has_animation: false,
    };
    let vp8x_payload = build::build_vp8x_chunk(width, height, flags)?;
    let mut body = Vec::new();
    let mut push_chunk = |fourcc, payload: &[u8]| -> Result<(), WebpError> {
        body.extend_from_slice(&build::build_chunk(fourcc, payload)?);
        Ok(())
    };
    push_chunk(container::fourcc::VP8X, &vp8x_payload)?;
    if let Some(icc) = meta.icc {
        push_chunk(container::fourcc::ICCP, icc)?;
    }
    push_chunk(container::fourcc::VP8L, &payload)?;
    if let Some(exif) = meta.exif {
        push_chunk(container::fourcc::EXIF, exif)?;
    }
    if let Some(xmp) = meta.xmp {
        push_chunk(container::fourcc::XMP, xmp)?;
    }
    api::frame_riff(body)
}

// ───────────────────────── deprecated pre-contract API ───────────────────
//
// Thin wrappers over the contract functions, kept for one release so
// in-workspace consumers migrate without a cascade.

/// Pre-contract still-image result: packed RGBA plus dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
#[deprecated(
    since = "0.3.0",
    note = "use `oxideav_webp::decode_rgba8` → `RgbaImage`"
)]
pub struct DecodedWebp {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height × 4` packed `[R, G, B, A]` bytes.
    pub rgba: Vec<u8>,
}

/// Pre-contract decode: the primary image as packed RGBA.
#[deprecated(
    since = "0.3.0",
    note = "use `oxideav_webp::decode_rgba8` (or `decode` for the native layout)"
)]
#[allow(deprecated)]
pub fn decode_webp_image(bytes: &[u8]) -> Result<DecodedWebp, WebpError> {
    let img = decode_rgba8(bytes)?;
    Ok(DecodedWebp {
        width: img.width,
        height: img.height,
        rgba: img.data,
    })
}

/// Pre-contract frame: packed RGBA plus its duration.
#[derive(Debug, Clone, PartialEq, Eq)]
#[deprecated(since = "0.3.0", note = "use `oxideav_webp::Frame` from `decode_all`")]
pub struct WebpFrame {
    /// `width × height × 4` packed `[R, G, B, A]` bytes.
    pub rgba: Vec<u8>,
    /// Frame (canvas) width in pixels.
    pub width: u32,
    /// Frame (canvas) height in pixels.
    pub height: u32,
    /// Display duration in milliseconds; `0` for a still.
    pub duration_ms: u32,
}

/// Pre-contract whole-file result: every frame as packed RGBA, the
/// metadata and the animation parameters.
#[derive(Debug, Clone, PartialEq)]
#[deprecated(since = "0.3.0", note = "use `oxideav_webp::decode_all` + `info`")]
#[allow(deprecated)]
pub struct DecodedWebpFile {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// One frame for a still, one per `ANMF` for an animation.
    pub frames: Vec<WebpFrame>,
    /// File-level metadata.
    pub metadata: Metadata,
    /// `ANIM` background colour `[R, G, B, A]`, `None` for a still.
    pub anim_background_rgba: Option<[u8; 4]>,
    /// `ANIM` loop count, `None` for a still.
    pub anim_loop_count: Option<u16>,
}

/// Pre-contract whole-file decode (stills and animations, packed RGBA).
#[deprecated(
    since = "0.3.0",
    note = "use `oxideav_webp::decode_all` (frames) + `info` (animation params)"
)]
#[allow(deprecated)]
pub fn decode_webp(bytes: &[u8]) -> Result<DecodedWebpFile, WebpError> {
    let frames = decode_all(bytes)?;
    let (width, height, metadata) = frames
        .first()
        .map(|f| (f.image.width, f.image.height, f.image.metadata.clone()))
        .ok_or_else(|| WebpError::invalid("no frames"))?;
    let anim = animation_params(bytes)?;
    Ok(DecodedWebpFile {
        width,
        height,
        frames: frames
            .into_iter()
            .map(|f| WebpFrame {
                width: f.image.width,
                height: f.image.height,
                rgba: f.image.to_rgba8(),
                duration_ms: f
                    .delay
                    .map(|d| u32::try_from(d.as_millis()).unwrap_or(u32::MAX))
                    .unwrap_or(0),
            })
            .collect(),
        metadata,
        anim_background_rgba: anim.map(|(_, bg)| bg),
        anim_loop_count: anim.map(|(n, _)| n),
    })
}

/// Pre-contract metadata read (no pixel decode).
#[deprecated(
    since = "0.3.0",
    note = "use `oxideav_webp::info` for presence or `decode(..).metadata` for payloads"
)]
pub fn extract_metadata(bytes: &[u8]) -> Result<Metadata, WebpError> {
    let c = container::parse(bytes)?;
    Ok(api::metadata_from_container(bytes, &c))
}

/// Pre-contract lossless encode of packed RGBA.
#[deprecated(
    since = "0.3.0",
    note = "use `oxideav_webp::encode_rgba8(width, height, rgba, &EncodeOptions::default())`"
)]
pub fn encode_webp_lossless(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, WebpError> {
    encode_rgba8(width, height, rgba, &EncodeOptions::default())
}

#[doc(inline)]
#[allow(deprecated)]
pub use anim_encode::{build_animated_webp, build_animated_webp_with_options, AnimEncoderOptions};

// ─────────────────────────────── registration ────────────────────────────

/// Install the WebP decoder / encoder factories and the `.webp` extension
/// hint into `ctx`. The registered codecs are thin adapters over
/// [`decode_with`] / [`encode`].
#[cfg(feature = "registry")]
pub fn register(ctx: &mut RuntimeContext) {
    registry::register(ctx);
}

/// Install only the WebP **codec** factories into a
/// [`oxideav_core::CodecRegistry`] — the fleet-wide signature.
#[cfg(feature = "registry")]
pub fn register_codecs(reg: &mut oxideav_core::CodecRegistry) {
    registry::register_codecs(reg);
}

/// Install only the WebP **container** hooks (the `.webp` extension
/// mapping) into a [`oxideav_core::ContainerRegistry`].
#[cfg(feature = "registry")]
pub fn register_containers(reg: &mut oxideav_core::ContainerRegistry) {
    registry::register_containers(reg);
}

#[cfg(feature = "registry")]
oxideav_core::register!("webp", register);
