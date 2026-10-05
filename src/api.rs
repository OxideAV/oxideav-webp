//! The contract root functions — `probe` / `info` / `decode*` / `encode*`
//! — and the options records that drive them.
//!
//! Everything here is standalone (no `oxideav-core`); the `registry`
//! adapter in [`crate::registry`] is a thin wrapper over these functions.

use core::time::Duration;
use std::io::{Read, Write};

use crate::alph;
use crate::anim;
use crate::anmf;
use crate::build::{self, Vp8xFlags};
use crate::container::{self, fourcc, WebpContainer};
use crate::image::{Frame, ImageInfo, Metadata, Plane, RgbImage, RgbaImage, WebpImage};
use crate::vp8_chunk;
use crate::vp8l_chunk;
use crate::vp8l_encode;
use crate::vp8l_transform;
use crate::vp8x;
use crate::yuv;
use crate::{WebpError, WebpPixelFormat};

/// §3.4 / RFC 6386 §9.1 per-side dimension ceiling: both the `VP8L`
/// image header (14-bit `width − 1`) and the `VP8 ` key-frame header
/// (14-bit `width`) top out at 16384 / 16383 pixels per side, so no
/// spec-valid bitstream — and no `ANMF` sub-frame — can exceed it. The
/// §2.7.1 `VP8X` canvas nominally allows 2^24 per side; a canvas wider
/// than any frame could fill is rejected before allocation.
pub const MAX_DIMENSION: u32 = 1 << 14;

// ───────────────────────────────── options ───────────────────────────────

/// Decode limits and strictness. `decode` uses `DecodeOptions::default()`.
///
/// Every limit is an `Option`; `None` means unlimited. The defaults are
/// the format's own ceilings — 16384 per side (the VP8L / VP8 header
/// range), 16384² pixels (1 GiB of RGBA) — and no byte limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Reject an image (or animation canvas) wider than this.
    pub max_width: Option<u32>,
    /// Reject an image (or animation canvas) taller than this.
    pub max_height: Option<u32>,
    /// Reject an image whose `width × height` exceeds this.
    pub max_pixels: Option<u64>,
    /// Reject an input longer than this many bytes.
    pub max_bytes: Option<u64>,
    /// Refuse spec-discouraged or inconsistent files instead of decoding
    /// them leniently: a `VP8X` canvas that disagrees with the bitstream
    /// dimensions, `VP8X` reserved bits set, an `ALPH` chunk next to a
    /// `VP8L` image, or an `ANMF` frame whose declared rectangle differs
    /// from its bitstream.
    pub strict: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_width: Some(MAX_DIMENSION),
            max_height: Some(MAX_DIMENSION),
            max_pixels: Some((MAX_DIMENSION as u64) * (MAX_DIMENSION as u64)),
            max_bytes: None,
            strict: false,
        }
    }
}

impl DecodeOptions {
    /// The defaults: the format's own ceilings, lenient.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: maximum width (`None` = unlimited).
    pub fn with_max_width(mut self, v: Option<u32>) -> Self {
        self.max_width = v;
        self
    }

    /// Builder: maximum height (`None` = unlimited).
    pub fn with_max_height(mut self, v: Option<u32>) -> Self {
        self.max_height = v;
        self
    }

    /// Builder: maximum pixel count (`None` = unlimited).
    pub fn with_max_pixels(mut self, v: Option<u64>) -> Self {
        self.max_pixels = v;
        self
    }

    /// Builder: maximum input length in bytes (`None` = unlimited).
    pub fn with_max_bytes(mut self, v: Option<u64>) -> Self {
        self.max_bytes = v;
        self
    }

    /// Builder: strict mode.
    pub fn with_strict(mut self, v: bool) -> Self {
        self.strict = v;
        self
    }

    /// The pixel cap handed to the VP8 decoder (`u64::MAX` when unlimited).
    pub(crate) fn pixel_cap(&self) -> u64 {
        self.max_pixels.unwrap_or(u64::MAX)
    }

    /// Check `width × height` against the limits.
    pub(crate) fn check_dimensions(&self, width: u32, height: u32) -> Result<(), WebpError> {
        if width == 0 || height == 0 {
            return Err(WebpError::invalid(format!(
                "zero image dimension {width}x{height}"
            )));
        }
        if self.max_width.is_some_and(|m| width > m) || self.max_height.is_some_and(|m| height > m)
        {
            return Err(WebpError::limit(format!(
                "{width}x{height} exceeds max {:?}x{:?}",
                self.max_width, self.max_height
            )));
        }
        let pixels = (width as u64) * (height as u64);
        if self.max_pixels.is_some_and(|m| pixels > m) {
            return Err(WebpError::limit(format!(
                "{width}x{height} = {pixels} pixels exceeds max {:?}",
                self.max_pixels
            )));
        }
        Ok(())
    }

    fn check_bytes(&self, len: usize) -> Result<(), WebpError> {
        if self.max_bytes.is_some_and(|m| len as u64 > m) {
            return Err(WebpError::limit(format!(
                "{len} input bytes exceeds max {:?}",
                self.max_bytes
            )));
        }
        Ok(())
    }
}

/// A positioned animation frame for [`encode_animation_frames`].
pub use crate::anim_encode::AnimFrame;
/// How each animation frame's pixels are compressed into its `ANMF`
/// sub-frame.
pub use crate::anim_encode::AnimFrameMode;
/// Tuning knobs for the animation delta path.
pub use crate::anim_encode::{DeltaConfig, DownsampleKernel};

/// Encoder options. One struct for stills and animations; behaviour
/// variants are fields.
///
/// The default is **lossless** (`VP8L`): `decode(encode(img)) == img`.
/// Setting a quality with [`with_quality`](Self::with_quality) switches to
/// the **lossy** `VP8 ` path (the `oxideav-vp8` keyframe encoder), with
/// alpha carried as a §2.7.1.2 `ALPH` chunk.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// `None` → lossless `VP8L`. `Some(q)`, `q` in `0.0..=100.0` → lossy
    /// `VP8 ` at that quality (`100` = best; mapped onto the VP8 qindex by
    /// `oxideav_vp8::encoder::quality_to_qindex`).
    pub quality: Option<f32>,
    /// Embed the image's ICC profile when present.
    pub embed_icc: bool,
    /// Embed the image's Exif payload when present.
    pub embed_exif: bool,
    /// Embed the image's XMP payload when present.
    pub embed_xmp: bool,
    /// Animation only — §2.7.1.1 `ANIM` loop count (`0` = forever).
    pub loop_count: u16,
    /// Animation only — §2.7.1.1 `ANIM` background colour `[R, G, B, A]`.
    pub background_rgba: [u8; 4],
    /// Animation only — per-frame compression strategy.
    pub frame_mode: AnimFrameMode,
    /// Animation only — delta-path tuning.
    pub delta: DeltaConfig,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quality: None,
            embed_icc: true,
            embed_exif: true,
            embed_xmp: true,
            loop_count: 0,
            background_rgba: [0, 0, 0, 0],
            frame_mode: AnimFrameMode::Auto,
            delta: DeltaConfig::default(),
        }
    }
}

impl EncodeOptions {
    /// The defaults: lossless, all metadata embedded, infinite loop.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: lossy at `quality` (`0.0..=100.0`, clamped; `NaN` → worst).
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Builder: back to lossless.
    pub fn with_lossless(mut self) -> Self {
        self.quality = None;
        self
    }

    /// Builder: which metadata payloads to embed.
    pub fn with_metadata(mut self, icc: bool, exif: bool, xmp: bool) -> Self {
        self.embed_icc = icc;
        self.embed_exif = exif;
        self.embed_xmp = xmp;
        self
    }

    /// Builder: animation loop count.
    pub fn with_loop_count(mut self, n: u16) -> Self {
        self.loop_count = n;
        self
    }

    /// Builder: animation background colour.
    pub fn with_background_rgba(mut self, rgba: [u8; 4]) -> Self {
        self.background_rgba = rgba;
        self
    }

    /// Builder: animation frame compression mode.
    pub fn with_frame_mode(mut self, mode: AnimFrameMode) -> Self {
        self.frame_mode = mode;
        self
    }

    /// Builder: animation delta tuning.
    pub fn with_delta(mut self, delta: DeltaConfig) -> Self {
        self.delta = delta;
        self
    }

    /// `true` when the lossy `VP8 ` path is selected.
    pub fn is_lossy(&self) -> bool {
        self.quality.is_some()
    }

    /// The metadata of `image` filtered by the embed flags.
    fn filtered_metadata<'a>(&self, m: &'a Metadata) -> crate::WebpMetadata<'a> {
        crate::WebpMetadata {
            icc: if self.embed_icc {
                m.icc.as_deref()
            } else {
                None
            },
            exif: if self.embed_exif {
                m.exif.as_deref()
            } else {
                None
            },
            xmp: if self.embed_xmp {
                m.xmp.as_deref()
            } else {
                None
            },
        }
    }
}

// ────────────────────────────────── probe ────────────────────────────────

/// `true` when `bytes` starts with the §2.4 `RIFF ???? WEBP` file header.
/// No allocation, never panics, `false` on short input.
pub fn probe(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

// ────────────────────────────────── info ─────────────────────────────────

/// What kind of still-image bitstream a container (or `ANMF` frame data)
/// carries.
#[derive(Clone, Copy)]
enum Kind<'a> {
    Lossless(vp8l_chunk::WebpLosslessChunk<'a>),
    Lossy(vp8_chunk::WebpLossyChunk<'a>),
}

impl Kind<'_> {
    fn dims(&self) -> (u32, u32) {
        match self {
            Kind::Lossless(c) => (c.width(), c.height()),
            Kind::Lossy(c) => (u32::from(c.width()), u32::from(c.height())),
        }
    }
}

/// Locate the `VP8L` / `VP8 ` image-data chunk of a still file.
fn still_kind<'a>(bytes: &'a [u8], c: &WebpContainer) -> Result<Option<Kind<'a>>, WebpError> {
    if let Some(chunk) = c.first_chunk_with_fourcc(fourcc::VP8L) {
        return Ok(Some(Kind::Lossless(
            vp8l_chunk::WebpLosslessChunk::from_chunk(bytes, chunk)?,
        )));
    }
    if let Some(chunk) = c.first_chunk_with_fourcc(fourcc::VP8) {
        return Ok(Some(Kind::Lossy(vp8_chunk::WebpLossyChunk::from_chunk(
            bytes, chunk,
        )?)));
    }
    Ok(None)
}

/// Header-only description: dimensions, native layout, frame count,
/// alpha and metadata presence. Parses the container, the `VP8X`
/// header and the image header of the first bitstream; decodes nothing.
pub fn info(bytes: &[u8]) -> Result<ImageInfo, WebpError> {
    let c = container::parse(bytes)?;
    let vp8x = match c.first_chunk_with_fourcc(fourcc::VP8X) {
        Some(ch) => Some(vp8x::Vp8xHeader::parse(ch.payload(bytes))?),
        None => None,
    };
    let has_icc = c.first_chunk_with_fourcc(fourcc::ICCP).is_some();
    let has_exif = c.first_chunk_with_fourcc(fourcc::EXIF).is_some();
    let has_xmp = c.first_chunk_with_fourcc(fourcc::XMP).is_some();

    let anim_chunk = c.first_chunk_with_fourcc(fourcc::ANIM);
    let is_animated = vp8x.as_ref().is_some_and(|h| h.has_animation) || anim_chunk.is_some();

    if is_animated {
        let hdr = vp8x.ok_or_else(|| WebpError::invalid("animated file without VP8X header"))?;
        let anim = match anim_chunk {
            Some(ch) => Some(anim::AnimHeader::parse(ch.payload(bytes))?),
            None => None,
        };
        let frames = c.chunks_with_fourcc(fourcc::ANMF).count() as u32;
        let first_lossy = c
            .chunks_with_fourcc(fourcc::ANMF)
            .next()
            .and_then(|ch| {
                let payload = ch.payload(bytes);
                let h = anmf::AnmfHeader::parse(payload).ok()?;
                let data = payload.get(h.frame_data_offset()..)?;
                Some(find_subchunk(data, fourcc::VP8).is_some())
            })
            .unwrap_or(false);
        let mut out = ImageInfo::new(hdr.canvas_width, hdr.canvas_height, WebpPixelFormat::Rgba);
        out.frames = frames;
        out.has_alpha = hdr.has_alpha;
        out.has_icc = has_icc || hdr.has_iccp;
        out.has_exif = has_exif || hdr.has_exif;
        out.has_xmp = has_xmp || hdr.has_xmp;
        out.is_animated = true;
        out.loop_count = anim.as_ref().map(|a| a.loop_count);
        out.background_rgba = anim.as_ref().map(|a| {
            [
                a.background_color.red,
                a.background_color.green,
                a.background_color.blue,
                a.background_color.alpha,
            ]
        });
        out.is_lossy = first_lossy;
        return Ok(out);
    }

    let kind = still_kind(bytes, &c)?
        .ok_or_else(|| WebpError::unsupported("no VP8L/VP8 image-data chunk"))?;
    let has_alph = c.first_chunk_with_fourcc(fourcc::ALPH).is_some();
    let (w, h) = match &vp8x {
        Some(x) => (x.canvas_width, x.canvas_height),
        None => kind.dims(),
    };
    let (format, has_alpha) = match &kind {
        Kind::Lossless(ch) => (
            WebpPixelFormat::Rgba,
            ch.alpha_is_used() || has_alph || vp8x.as_ref().is_some_and(|x| x.has_alpha),
        ),
        Kind::Lossy(_) => {
            if has_alph {
                (WebpPixelFormat::Yuva420P, true)
            } else {
                (WebpPixelFormat::Yuv420P, false)
            }
        }
    };
    let mut out = ImageInfo::new(w, h, format);
    out.has_alpha = has_alpha;
    out.has_icc = has_icc;
    out.has_exif = has_exif;
    out.has_xmp = has_xmp;
    out.is_lossy = matches!(kind, Kind::Lossy(_));
    Ok(out)
}

// ───────────────────────────────── decode ────────────────────────────────

/// Decode the primary image in its native layout: [`Rgba`](WebpPixelFormat::Rgba)
/// for lossless `VP8L`, [`Yuv420P`](WebpPixelFormat::Yuv420P) /
/// [`Yuva420P`](WebpPixelFormat::Yuva420P) for lossy `VP8 ` (with / without
/// an `ALPH` chunk). An animation decodes to its first composited frame
/// (`Rgba`). Uses `DecodeOptions::default()`.
pub fn decode(bytes: &[u8]) -> Result<WebpImage, WebpError> {
    decode_with(bytes, &DecodeOptions::default())
}

/// [`decode`] with explicit limits / strictness.
pub fn decode_with(bytes: &[u8], opts: &DecodeOptions) -> Result<WebpImage, WebpError> {
    opts.check_bytes(bytes.len())?;
    let c = container::parse(bytes)?;
    if is_animated(bytes, &c) {
        let mut frames = decode_animation(bytes, &c, opts, true)?;
        return frames
            .pop()
            .map(|f| f.image)
            .ok_or_else(|| WebpError::invalid("animation has no frames"));
    }
    decode_still(bytes, &c, opts)
}

/// Decode to tightly packed 8-bit RGB (alpha dropped).
pub fn decode_rgb8(bytes: &[u8]) -> Result<RgbImage, WebpError> {
    let img = decode(bytes)?;
    Ok(RgbImage::new(img.width, img.height, img.to_rgb8()))
}

/// Decode to tightly packed 8-bit RGBA (alpha `255` when absent).
pub fn decode_rgba8(bytes: &[u8]) -> Result<RgbaImage, WebpError> {
    let img = decode(bytes)?;
    Ok(RgbaImage::new(img.width, img.height, img.to_rgba8()))
}

/// Decode every image: one [`Frame`] per `ANMF` chunk of an animation,
/// each the full canvas after compositing per §2.7.1.1 (disposal of the
/// previous frame, then this frame's blend), with its display `delay`; a
/// still file yields one frame with `delay == None`.
pub fn decode_all(bytes: &[u8]) -> Result<Vec<Frame>, WebpError> {
    decode_all_with(bytes, &DecodeOptions::default())
}

/// [`decode_all`] with explicit limits / strictness.
pub fn decode_all_with(bytes: &[u8], opts: &DecodeOptions) -> Result<Vec<Frame>, WebpError> {
    opts.check_bytes(bytes.len())?;
    let c = container::parse(bytes)?;
    if is_animated(bytes, &c) {
        return decode_animation(bytes, &c, opts, false);
    }
    Ok(vec![Frame::new(decode_still(bytes, &c, opts)?, None)])
}

/// Read `r` to end and [`decode`] it.
pub fn decode_from<R: Read>(mut r: R) -> Result<WebpImage, WebpError> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    decode(&buf)
}

pub(crate) fn is_animated(bytes: &[u8], c: &WebpContainer) -> bool {
    if c.first_chunk_with_fourcc(fourcc::ANIM).is_some() {
        return true;
    }
    c.first_chunk_with_fourcc(fourcc::VP8X)
        .and_then(|ch| vp8x::Vp8xHeader::parse(ch.payload(bytes)).ok())
        .is_some_and(|h| h.has_animation)
}

/// Lift the `ICCP` / `EXIF` / `XMP ` payloads out of a parsed container.
pub(crate) fn metadata_from_container(bytes: &[u8], c: &WebpContainer) -> Metadata {
    let payload_of = |fourcc| {
        c.first_chunk_with_fourcc(fourcc)
            .map(|chunk| chunk.payload(bytes).to_vec())
    };
    Metadata {
        icc: payload_of(fourcc::ICCP),
        exif: payload_of(fourcc::EXIF),
        xmp: payload_of(fourcc::XMP),
        gamma: None,
    }
}

/// Decode a still (`VP8L` or `VP8 `, optionally `VP8X`-extended with
/// `ALPH`) to its native layout.
fn decode_still(
    bytes: &[u8],
    c: &WebpContainer,
    opts: &DecodeOptions,
) -> Result<WebpImage, WebpError> {
    let vp8x = match c.first_chunk_with_fourcc(fourcc::VP8X) {
        Some(ch) => Some(vp8x::Vp8xHeader::parse(ch.payload(bytes))?),
        None => None,
    };
    if opts.strict {
        if let Some(x) = &vp8x {
            if x.has_unknown {
                return Err(WebpError::invalid("VP8X reserved bits set"));
            }
        }
    }
    let kind = still_kind(bytes, c)?
        .ok_or_else(|| WebpError::unsupported("no VP8L/VP8 image-data chunk"))?;
    let (w, h) = kind.dims();
    if let Some(x) = &vp8x {
        if opts.strict && (x.canvas_width, x.canvas_height) != (w, h) {
            return Err(WebpError::invalid(format!(
                "VP8X canvas {}x{} disagrees with bitstream {w}x{h}",
                x.canvas_width, x.canvas_height
            )));
        }
    }
    opts.check_dimensions(w, h)?;
    let alph = c
        .first_chunk_with_fourcc(fourcc::ALPH)
        .map(|ch| ch.payload(bytes));
    let mut img = decode_bitstream(kind, alph, opts)?;
    img.metadata = metadata_from_container(bytes, c);
    Ok(img)
}

/// Decode one bitstream (still or `ANMF` sub-frame) plus its optional
/// `ALPH` payload to the native layout. Dimensions are already checked.
fn decode_bitstream(
    kind: Kind<'_>,
    alph: Option<&[u8]>,
    opts: &DecodeOptions,
) -> Result<WebpImage, WebpError> {
    let (w, h) = kind.dims();
    match kind {
        Kind::Lossless(chunk) => {
            if opts.strict && alph.is_some() {
                return Err(WebpError::invalid(
                    "ALPH chunk alongside a VP8L image (RFC 9649 §2.7.1.2)",
                ));
            }
            let mut image = vp8l_transform::decode_lossless(chunk.bitstream(), w, h)?;
            // §2.7.1.2: an ALPH chunk alongside VP8L is discouraged but
            // not forbidden; when present its plane overrides the
            // per-pixel alpha.
            if let Some(payload) = alph {
                let plane = alph::decode_alpha(payload, w, h)?;
                let pixels = image.pixels_mut();
                if plane.len() == pixels.len() {
                    for (px, &a) in pixels.iter_mut().zip(plane.iter()) {
                        *px = (*px & 0x00ff_ffff) | (u32::from(a) << 24);
                    }
                }
            }
            WebpImage::from_rgba8(w, h, argb_to_rgba(image.pixels()))
        }
        Kind::Lossy(chunk) => {
            let frame =
                oxideav_vp8::decode_vp8_with_max_pixels(chunk.bitstream(), opts.pixel_cap())?;
            let (fw, fh) = (frame.width, frame.height);
            if (fw, fh) != (w, h) {
                return Err(WebpError::invalid(format!(
                    "VP8 chunk header {w}x{h} disagrees with decoded frame {fw}x{fh}"
                )));
            }
            let cw = (fw as usize).div_ceil(2);
            let mut img = WebpImage::new_unchecked(
                fw,
                fh,
                WebpPixelFormat::Yuv420P,
                vec![
                    Plane::packed(fw as usize, frame.y),
                    Plane::packed(cw, frame.u),
                    Plane::packed(cw, frame.v),
                ],
            );
            if let Some(payload) = alph {
                let plane = alph::decode_alpha(payload, fw, fh)?;
                if plane.len() == (fw as usize) * (fh as usize) {
                    img.format = WebpPixelFormat::Yuva420P;
                    img.planes.push(Plane::packed(fw as usize, plane));
                }
            }
            Ok(img)
        }
    }
}

/// Decode an animation (`ANIM` + `ANMF…`) to composited full-canvas RGBA
/// frames per RFC 9649 §2.7.1.1. With `first_only`, stop after the first
/// frame.
///
/// The canvas is sized from `VP8X` and initialised to the `ANIM`
/// background colour. Before each frame the *previous* frame's disposal
/// is applied to its rectangle (`Background` fills it with the background
/// colour); the frame is then drawn with its blending method —
/// `Overwrite` replaces the rectangle, `AlphaBlend` composites with the
/// §2.7.1.1 "Alpha-blending" formula in 8-bit integer arithmetic (no
/// gamma linearisation).
fn decode_animation(
    bytes: &[u8],
    c: &WebpContainer,
    opts: &DecodeOptions,
    first_only: bool,
) -> Result<Vec<Frame>, WebpError> {
    let (hdr, anim) = animation_headers(bytes, c, opts)?;
    let mut compositor = AnimCompositor::new(&hdr, &anim, opts)?;
    let metadata = metadata_from_container(bytes, c);

    let mut frames = Vec::new();
    for anmf_chunk in c.chunks_with_fourcc(fourcc::ANMF) {
        let (canvas, duration_ms) = compositor.step(anmf_chunk.payload(bytes), opts)?;
        let image = WebpImage::from_rgba8(hdr.canvas_width, hdr.canvas_height, canvas)?
            .with_metadata(metadata.clone());
        frames.push(Frame::new(
            image,
            Some(Duration::from_millis(u64::from(duration_ms))),
        ));
        if first_only {
            break;
        }
    }
    if frames.is_empty() {
        return Err(WebpError::invalid("animation has no ANMF frames"));
    }
    Ok(frames)
}

/// The `VP8X` and `ANIM` headers an animated container must carry.
pub(crate) fn animation_headers(
    bytes: &[u8],
    c: &WebpContainer,
    opts: &DecodeOptions,
) -> Result<(vp8x::Vp8xHeader, anim::AnimHeader), WebpError> {
    let anim_chunk = c
        .first_chunk_with_fourcc(fourcc::ANIM)
        .ok_or_else(|| WebpError::invalid("animated VP8X without ANIM chunk"))?;
    let anim = anim::AnimHeader::parse(anim_chunk.payload(bytes))?;
    let vp8x_chunk = c
        .first_chunk_with_fourcc(fourcc::VP8X)
        .ok_or_else(|| WebpError::invalid("animation without VP8X header"))?;
    let hdr = vp8x::Vp8xHeader::parse(vp8x_chunk.payload(bytes))?;
    if opts.strict && hdr.has_unknown {
        return Err(WebpError::invalid("VP8X reserved bits set"));
    }
    Ok((hdr, anim))
}

/// The §2.7.1.1 animation canvas as a persistent object: one `ANMF`
/// chunk at a time through [`AnimCompositor::step`].
///
/// [`decode_all`] drives it over a whole file; the framework decoder
/// drives it across packets when the demuxer hands it one `ANMF` frame
/// per packet, so both paths produce byte-identical canvases.
pub(crate) struct AnimCompositor {
    canvas_w: u32,
    canvas_h: u32,
    bg_rgba: [u8; 4],
    canvas: Vec<u8>,
    prev_rect: Option<(u32, u32, u32, u32, anmf::DisposalMethod)>,
}

impl AnimCompositor {
    /// A canvas of the `VP8X` size cleared to the `ANIM` background
    /// colour. The dimensions are checked against `opts` and the format
    /// ceiling before the buffer is allocated.
    pub(crate) fn new(
        hdr: &vp8x::Vp8xHeader,
        anim: &anim::AnimHeader,
        opts: &DecodeOptions,
    ) -> Result<Self, WebpError> {
        let bg = anim.background_color;
        let bg_rgba = [bg.red, bg.green, bg.blue, bg.alpha];
        let canvas_w = hdr.canvas_width;
        let canvas_h = hdr.canvas_height;
        // A canvas wider than any spec-valid frame could cover is rejected
        // before the full-canvas buffer is allocated (see MAX_DIMENSION).
        if canvas_w > MAX_DIMENSION || canvas_h > MAX_DIMENSION {
            return Err(WebpError::invalid(format!(
                "animation canvas {canvas_w}x{canvas_h} exceeds the {MAX_DIMENSION} frame ceiling"
            )));
        }
        opts.check_dimensions(canvas_w, canvas_h)?;

        let canvas_bytes = (canvas_w as usize) * (canvas_h as usize) * 4;
        let mut canvas: Vec<u8> = Vec::with_capacity(canvas_bytes);
        for _ in 0..(canvas_bytes / 4) {
            canvas.extend_from_slice(&bg_rgba);
        }
        Ok(Self {
            canvas_w,
            canvas_h,
            bg_rgba,
            canvas,
            prev_rect: None,
        })
    }

    /// Canvas width in pixels.
    #[cfg_attr(not(feature = "registry"), allow(dead_code))]
    pub(crate) fn canvas_width(&self) -> u32 {
        self.canvas_w
    }

    /// Canvas height in pixels.
    #[cfg_attr(not(feature = "registry"), allow(dead_code))]
    pub(crate) fn canvas_height(&self) -> u32 {
        self.canvas_h
    }

    /// Dispose of the previous frame's rectangle, decode and draw one
    /// `ANMF` chunk payload (header + frame data) with its blending
    /// method, and return the composited canvas (packed `Rgba`) plus
    /// the frame's duration in milliseconds.
    pub(crate) fn step(
        &mut self,
        payload: &[u8],
        opts: &DecodeOptions,
    ) -> Result<(Vec<u8>, u32), WebpError> {
        let (canvas_w, canvas_h, bg_rgba) = (self.canvas_w, self.canvas_h, self.bg_rgba);
        let header = anmf::AnmfHeader::parse(payload)?;
        let frame_data = payload
            .get(header.frame_data_offset()..)
            .ok_or_else(|| WebpError::invalid("ANMF frame data truncated"))?;

        // §2.7.1.1 / §2.7.2 Figure 14: the frame data is a flat chunk
        // list — `VP8L` or `VP8 ` plus an optional `ALPH`. The pixel
        // dimensions come from the bitstream's own header; the ANMF
        // width / height govern only the placement rectangle.
        let kind = if let Some(p) = find_subchunk(frame_data, fourcc::VP8L) {
            Kind::Lossless(vp8l_chunk::WebpLosslessChunk::from_payload(p)?)
        } else if let Some(p) = find_subchunk(frame_data, fourcc::VP8) {
            Kind::Lossy(vp8_chunk::WebpLossyChunk::from_payload(p)?)
        } else {
            return Err(WebpError::invalid("ANMF frame without VP8L/VP8 sub-chunk"));
        };
        let (sub_w, sub_h) = kind.dims();
        if opts.strict && (header.width, header.height) != (sub_w, sub_h) {
            return Err(WebpError::invalid(format!(
                "ANMF rectangle {}x{} disagrees with frame bitstream {sub_w}x{sub_h}",
                header.width, header.height
            )));
        }
        opts.check_dimensions(sub_w, sub_h)?;
        // §2.7.1.1: the frame must fit inside the canvas.
        let right = header
            .x
            .checked_add(sub_w)
            .ok_or_else(|| WebpError::invalid("ANMF x + width overflows"))?;
        let bottom = header
            .y
            .checked_add(sub_h)
            .ok_or_else(|| WebpError::invalid("ANMF y + height overflows"))?;
        if right > canvas_w || bottom > canvas_h {
            return Err(WebpError::invalid(format!(
                "ANMF frame {sub_w}x{sub_h} at ({}, {}) overflows the {canvas_w}x{canvas_h} canvas",
                header.x, header.y
            )));
        }

        let alph = find_subchunk(frame_data, fourcc::ALPH);
        // An unreadable ALPH sub-chunk is tolerated leniently (the frame
        // stays opaque), refused in strict mode.
        let sub = match decode_bitstream(kind, alph, opts) {
            Ok(img) => img,
            Err(e) if alph.is_some() && !opts.strict => {
                let _ = e;
                decode_bitstream(kind, None, opts)?
            }
            Err(e) => return Err(e),
        };
        let sub_rgba = sub.to_rgba8();

        let canvas = &mut self.canvas;
        if let Some((px, py, pw, ph, anmf::DisposalMethod::Background)) = self.prev_rect {
            fill_canvas_rect(canvas, canvas_w, px, py, pw, ph, bg_rgba);
        }
        match header.blend {
            anmf::BlendingMethod::Overwrite => blit_rect_overwrite(
                canvas, canvas_w, header.x, header.y, sub_w, sub_h, &sub_rgba,
            ),
            anmf::BlendingMethod::AlphaBlend => blit_rect_alpha_blend(
                canvas, canvas_w, header.x, header.y, sub_w, sub_h, &sub_rgba,
            ),
        }
        self.prev_rect = Some((header.x, header.y, sub_w, sub_h, header.dispose));
        Ok((canvas.clone(), header.duration_ms))
    }
}

/// Fill an axis-aligned rectangle of `canvas` with `rgba`. Bounds are
/// pre-validated by the caller.
fn fill_canvas_rect(
    canvas: &mut [u8],
    canvas_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    rgba: [u8; 4],
) {
    let cw_bytes = canvas_w as usize * 4;
    let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
    for row in 0..h {
        let off = (y + row) * cw_bytes + x * 4;
        for px in canvas[off..off + w * 4].chunks_exact_mut(4) {
            px.copy_from_slice(&rgba);
        }
    }
}

/// Copy `src` (flat `w*h*4` RGBA) into `canvas` at `(x, y)` byte-for-byte
/// (§2.7.1.1 blending method `1`).
fn blit_rect_overwrite(
    canvas: &mut [u8],
    canvas_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    src: &[u8],
) {
    let cw_bytes = canvas_w as usize * 4;
    let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
    let sw_bytes = w * 4;
    for row in 0..h {
        let src_off = row * sw_bytes;
        let dst_off = (y + row) * cw_bytes + x * 4;
        canvas[dst_off..dst_off + sw_bytes].copy_from_slice(&src[src_off..src_off + sw_bytes]);
    }
}

/// Composite `src` over `canvas` at `(x, y)` per the §2.7.1.1
/// "Alpha-blending" formula (8-bit integer approximation, no gamma
/// linearisation).
fn blit_rect_alpha_blend(
    canvas: &mut [u8],
    canvas_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    src: &[u8],
) {
    let cw_bytes = canvas_w as usize * 4;
    let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
    for row in 0..h {
        for col in 0..w {
            let s = &src[(row * w + col) * 4..(row * w + col) * 4 + 4];
            let d_off = (y + row) * cw_bytes + (x + col) * 4;
            let (sr, sg, sb, sa) = (s[0] as u32, s[1] as u32, s[2] as u32, s[3] as u32);
            // Fully-opaque source: "assume the alpha value is 255,
            // effectively replacing the rectangle".
            if sa == 255 {
                canvas[d_off..d_off + 4].copy_from_slice(s);
                continue;
            }
            if sa == 0 {
                continue;
            }
            let (dr, dg, db, da) = (
                canvas[d_off] as u32,
                canvas[d_off + 1] as u32,
                canvas[d_off + 2] as u32,
                canvas[d_off + 3] as u32,
            );
            // blend.A = src.A + dst.A * (1 - src.A / 255)
            let dst_factor = (da * (255 - sa) + 127) / 255;
            let out_a = sa + dst_factor;
            // blend.RGB = (src.RGB * src.A + dst.RGB * dst.A * (1 - src.A / 255)) / blend.A;
            // both transparent → blend.RGB := 0.
            let mix = |s: u32, d: u32| {
                (s * sa + d * dst_factor + out_a / 2)
                    .checked_div(out_a)
                    .unwrap_or(0)
                    .min(255) as u8
            };
            canvas[d_off] = mix(sr, dr);
            canvas[d_off + 1] = mix(sg, dg);
            canvas[d_off + 2] = mix(sb, db);
            canvas[d_off + 3] = out_a.min(255) as u8;
        }
    }
}

/// Walk a flat §2.3 chunk list (the `ANMF` frame data — no outer
/// `RIFF`/`WEBP` header) and return the payload of the first chunk with
/// `target` FourCC. `None` on a truncated header or no match.
pub(crate) fn find_subchunk(mut data: &[u8], target: container::FourCc) -> Option<&[u8]> {
    while data.len() >= 8 {
        let fourcc: container::FourCc = data[0..4].try_into().ok()?;
        let size = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let payload_end = 8usize.checked_add(size)?;
        if payload_end > data.len() {
            return None;
        }
        if fourcc == target {
            return Some(&data[8..payload_end]);
        }
        // §2.3: odd Size is followed by one pad byte not counted in Size.
        let advance = payload_end + (size & 1);
        if advance > data.len() {
            return None;
        }
        data = &data[advance..];
    }
    None
}

/// Repack scan-line ARGB (`(a<<24)|(r<<16)|(g<<8)|b`) into packed
/// `[R, G, B, A]` bytes.
pub(crate) fn argb_to_rgba(pixels: &[u32]) -> Vec<u8> {
    let mut out = vec![0u8; pixels.len() * 4];
    for (chunk, &argb) in out.chunks_exact_mut(4).zip(pixels.iter()) {
        chunk[0] = (argb >> 16) as u8;
        chunk[1] = (argb >> 8) as u8;
        chunk[2] = argb as u8;
        chunk[3] = (argb >> 24) as u8;
    }
    out
}

/// Repack packed RGB(A) rows (`bpp` 3 or 4) into scan-line ARGB; the
/// returned flag is `true` when any alpha byte is not `255`.
pub(crate) fn packed_to_argb(
    width: usize,
    height: usize,
    data: &[u8],
    stride: usize,
    bpp: usize,
) -> (Vec<u32>, bool) {
    let mut pixels = Vec::with_capacity(width * height);
    let mut alpha_is_used = false;
    for y in 0..height {
        let row = &data[y * stride..y * stride + width * bpp];
        for p in row.chunks_exact(bpp) {
            let a = if bpp == 4 { p[3] as u32 } else { 0xff };
            if a != 0xff {
                alpha_is_used = true;
            }
            pixels.push((a << 24) | ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32);
        }
    }
    (pixels, alpha_is_used)
}

// ───────────────────────────────── encode ────────────────────────────────

/// Encode `image` as a still `.webp`.
///
/// * `EncodeOptions::default()` (lossless) accepts [`Rgb24`](WebpPixelFormat::Rgb24)
///   and [`Rgba`](WebpPixelFormat::Rgba) and writes a `VP8L` bitstream; a
///   Y′CbCr image is [`WebpError::Unsupported`] (WebP has no lossless
///   Y′CbCr layout — convert first, nothing is converted silently).
/// * With a quality, every layout is accepted: RGB(A) is converted to
///   limited-range BT.601 4:2:0 and a `VP8 ` keyframe is written; alpha
///   (an `Rgba` image with any non-opaque pixel, or a `Yuva420P` plane)
///   becomes a §2.7.1.2 `ALPH` chunk. Y′CbCr planes tagged
///   [`ColorRange::Full`](crate::ColorRange::Full) are refused: the VP8
///   bitstream cannot signal a range.
///
/// Metadata present on the image is embedded (`ICCP`, `EXIF`, `XMP `)
/// subject to the `embed_*` flags; the extended `VP8X` layout is chosen
/// automatically when alpha or metadata needs declaring.
pub fn encode(image: &WebpImage, opts: &EncodeOptions) -> Result<Vec<u8>, WebpError> {
    image.check_geometry()?;
    if image.width == 0 || image.height == 0 {
        return Err(WebpError::invalid("zero image dimension"));
    }
    if image.width > MAX_DIMENSION || image.height > MAX_DIMENSION {
        return Err(WebpError::invalid(format!(
            "{}x{} exceeds the WebP {MAX_DIMENSION} per-side ceiling",
            image.width, image.height
        )));
    }
    let meta = opts.filtered_metadata(&image.metadata);
    match opts.quality {
        None => encode_lossless(image, &meta),
        Some(q) => encode_lossy(image, q, &meta),
    }
}

/// Encode packed 8-bit RGB (3 bytes per pixel, `width × height × 3`).
pub fn encode_rgb8(
    width: u32,
    height: u32,
    rgb: &[u8],
    opts: &EncodeOptions,
) -> Result<Vec<u8>, WebpError> {
    encode(&WebpImage::from_rgb8(width, height, rgb.to_vec())?, opts)
}

/// Encode packed 8-bit RGBA (4 bytes per pixel, `width × height × 4`).
pub fn encode_rgba8(
    width: u32,
    height: u32,
    rgba: &[u8],
    opts: &EncodeOptions,
) -> Result<Vec<u8>, WebpError> {
    encode(&WebpImage::from_rgba8(width, height, rgba.to_vec())?, opts)
}

/// [`encode`] and write the bytes to `w`.
pub fn encode_to<W: Write>(
    image: &WebpImage,
    opts: &EncodeOptions,
    mut w: W,
) -> Result<(), WebpError> {
    let bytes = encode(image, opts)?;
    w.write_all(&bytes)?;
    Ok(())
}

fn encode_lossless(
    image: &WebpImage,
    meta: &crate::WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let bpp = match image.format {
        WebpPixelFormat::Rgb24 => 3,
        WebpPixelFormat::Rgba => 4,
        other => {
            return Err(WebpError::unsupported(format!(
                "lossless WebP carries RGB(A) only, not {other:?}; convert with to_rgba8 or set a quality"
            )))
        }
    };
    let plane = &image.planes[0];
    let (argb, has_alpha) = packed_to_argb(
        image.width as usize,
        image.height as usize,
        &plane.data,
        plane.stride,
        bpp,
    );
    // RFC 9649 §2.6: the simple lossless layout carries alpha inside the
    // VP8L bitstream itself, so the extended VP8X header is only needed
    // when there is metadata to declare (the reference lossless-RGBA
    // fixtures in docs/ use the simple layout too).
    if meta.is_empty() {
        let payload =
            vp8l_encode::encode_vp8l_argb_with(&argb, image.width, image.height, has_alpha)?;
        return Ok(build::build_webp_file(
            &payload,
            build::ImageKind::Lossless,
            image.width,
            image.height,
        )?);
    }
    crate::encode_vp8l_argb_with_metadata(image.width, image.height, &argb, has_alpha, meta)
}

fn encode_lossy(
    image: &WebpImage,
    quality: f32,
    meta: &crate::WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    let w = image.width as usize;
    let h = image.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    // RFC 6386 §9.1: the key-frame width / height fields are 14 bits.
    if image.width >= MAX_DIMENSION || image.height >= MAX_DIMENSION {
        return Err(WebpError::invalid(format!(
            "{}x{} exceeds the VP8 key-frame {} per-side ceiling",
            image.width,
            image.height,
            MAX_DIMENSION - 1
        )));
    }

    // Planes to feed the VP8 encoder, plus the alpha plane when any.
    let (y, u, v, alpha): (Vec<u8>, Vec<u8>, Vec<u8>, Option<Vec<u8>>) = match image.format {
        WebpPixelFormat::Rgb24 | WebpPixelFormat::Rgba => {
            let bpp = image.format.packed_bytes_per_pixel().unwrap_or(4);
            let plane = &image.planes[0];
            let (y, u, v) = yuv::rgb_to_yuv420(w, h, &plane.data, plane.stride, bpp);
            let alpha = if bpp == 4 {
                let mut a = Vec::with_capacity(w * h);
                let mut used = false;
                for row in 0..h {
                    for px in
                        plane.data[row * plane.stride..row * plane.stride + w * 4].chunks_exact(4)
                    {
                        used |= px[3] != 0xff;
                        a.push(px[3]);
                    }
                }
                used.then_some(a)
            } else {
                None
            };
            (y, u, v, alpha)
        }
        WebpPixelFormat::Yuv420P | WebpPixelFormat::Yuva420P => {
            if image.color.range == crate::ColorRange::Full {
                return Err(WebpError::unsupported(
                    "VP8 cannot signal full-range Y'CbCr; supply limited-range planes or RGB",
                ));
            }
            let tight = |p: &Plane, row_bytes: usize, rows: usize| -> Vec<u8> {
                let mut out = Vec::with_capacity(row_bytes * rows);
                for r in 0..rows {
                    out.extend_from_slice(&p.data[r * p.stride..r * p.stride + row_bytes]);
                }
                out
            };
            let y = tight(&image.planes[0], w, h);
            let u = tight(&image.planes[1], cw, ch);
            let v = tight(&image.planes[2], cw, ch);
            let alpha = if image.format == WebpPixelFormat::Yuva420P {
                Some(tight(&image.planes[3], w, h))
            } else {
                None
            };
            (y, u, v, alpha)
        }
    };

    let frame = oxideav_vp8::I420Frame::packed(image.width, image.height, &y, &u, &v);
    let params = oxideav_vp8::KeyframeParams {
        y_ac_qi: oxideav_vp8::encoder::quality_to_qindex(quality),
        trellis_strength: oxideav_vp8::encoder::quality_to_trellis_strength(quality),
        ..oxideav_vp8::KeyframeParams::default()
    };
    let vp8 = oxideav_vp8::encode_keyframe(&frame, &params)?;

    // Simple-lossy layout when there is nothing to declare in a VP8X.
    if alpha.is_none() && meta.is_empty() {
        return Ok(build::build_webp_file(
            &vp8,
            build::ImageKind::Lossy,
            image.width,
            image.height,
        )?);
    }

    // §2.7 extended layout: VP8X, ICCP, ALPH, VP8, EXIF, XMP.
    let flags = Vp8xFlags {
        has_iccp: meta.icc.is_some(),
        has_alpha: alpha.is_some(),
        has_exif: meta.exif.is_some(),
        has_xmp: meta.xmp.is_some(),
        has_animation: false,
    };
    let vp8x_payload = build::build_vp8x_chunk(image.width, image.height, flags)?;
    let mut body = Vec::new();
    let mut push = |fourcc, payload: &[u8]| -> Result<(), WebpError> {
        body.extend_from_slice(&build::build_chunk(fourcc, payload)?);
        Ok(())
    };
    push(fourcc::VP8X, &vp8x_payload)?;
    if let Some(icc) = meta.icc {
        push(fourcc::ICCP, icc)?;
    }
    if let Some(a) = &alpha {
        push(
            fourcc::ALPH,
            &build_alph_payload(a, image.width, image.height),
        )?;
    }
    push(fourcc::VP8, &vp8)?;
    if let Some(exif) = meta.exif {
        push(fourcc::EXIF, exif)?;
    }
    if let Some(xmp) = meta.xmp {
        push(fourcc::XMP, xmp)?;
    }
    frame_riff(body)
}

/// Wrap an assembled chunk body in the §2.4 `RIFF` / `WEBP` file header.
pub(crate) fn frame_riff(body: Vec<u8>) -> Result<Vec<u8>, WebpError> {
    let file_size = (body.len() as u64) + 4;
    if file_size > u64::from(u32::MAX) {
        return Err(WebpError::invalid("RIFF file size exceeds u32"));
    }
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&fourcc::RIFF);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&fourcc::WEBP);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Build a §2.7.1.2 `ALPH` chunk payload (info byte + bitstream) for a
/// `width × height` alpha plane.
///
/// Two candidates are produced and the smaller kept: compression method
/// `1` — the plane coded as a headerless §3 VP8L image-stream with the
/// alpha in the GREEN channel (the §3.4 image header is exactly five
/// bytes, so the image-stream is the `VP8L` payload from byte 5 on) —
/// and method `0`, the raw plane. Filtering `F = 0`, preprocessing
/// `P = 0`.
pub(crate) fn build_alph_payload(alpha: &[u8], width: u32, height: u32) -> Vec<u8> {
    let argb: Vec<u32> = alpha
        .iter()
        .map(|&a| 0xff00_0000 | (u32::from(a) << 8))
        .collect();
    let compressed =
        vp8l_encode::encode_vp8l_argb_with(&argb, width, height, false).unwrap_or_default();
    let stream = compressed
        .get(vp8l_chunk::VP8L_IMAGE_HEADER_LEN..)
        .unwrap_or(&[]);
    if !stream.is_empty() && stream.len() < alpha.len() {
        let mut out = Vec::with_capacity(1 + stream.len());
        out.push(0x01); // Rsv=0 P=0 F=0 C=1
        out.extend_from_slice(stream);
        out
    } else {
        let mut out = Vec::with_capacity(1 + alpha.len());
        out.push(0x00); // Rsv=0 P=0 F=0 C=0
        out.extend_from_slice(alpha);
        out
    }
}

// ─────────────────────────────── animation ───────────────────────────────

/// Encode `frames` as one file — the mirror of [`decode_all`]. A single
/// frame with no delay is written as a still ([`encode`]); anything else
/// as an animation via [`encode_animation`] (lossless `VP8L` sub-frames,
/// so a `quality` option is [`WebpError::Unsupported`]). Every frame is
/// composited full-canvas as `Rgba` — the layout [`decode_all`] returns
/// — so `decode_all(encode_all(frames)) == frames` holds for `Rgba`
/// frames whose metadata matches the first frame's (the file carries one
/// metadata set) and whose delays are whole milliseconds.
pub fn encode_all(frames: &[Frame], opts: &EncodeOptions) -> Result<Vec<u8>, WebpError> {
    match frames {
        [] => Err(WebpError::invalid("encode_all needs at least one frame")),
        [single] if single.delay.is_none() => encode(&single.image, opts),
        _ => encode_animation(frames, opts),
    }
}

/// Encode `frames` as an animated `.webp` (RFC 9649 §2.7.1.1) — the
/// WebP-specific depth name under [`encode_all`].
///
/// Each frame's image is composited as a full-canvas RGBA picture (any
/// layout is accepted — [`WebpImage::to_rgba8`] runs first) placed at the
/// canvas origin; the canvas is the largest frame. Frames are coded
/// losslessly as `VP8L` sub-frames per `opts.frame_mode` (`Auto` picks the
/// smaller of a full keyframe and a dirty-rectangle delta). `delay` is the
/// §2.7.1.1 frame duration (`None` → 0 ms). `loop_count`,
/// `background_rgba` and the metadata of the first frame's image feed the
/// `ANIM` / `ICCP` / `EXIF` / `XMP ` chunks. A lossy (`quality`) option is
/// [`WebpError::Unsupported`] for animations in this release.
pub fn encode_animation(frames: &[Frame], opts: &EncodeOptions) -> Result<Vec<u8>, WebpError> {
    if frames.is_empty() {
        return Err(WebpError::invalid("animation needs at least one frame"));
    }
    if opts.quality.is_some() {
        return Err(WebpError::unsupported(
            "lossy animation frames are not implemented; use the lossless default",
        ));
    }
    let anim_frames: Vec<crate::anim_encode::AnimFrame> = frames
        .iter()
        .map(|f| {
            f.image.check_geometry()?;
            let ms = f
                .delay
                .map(|d| u32::try_from(d.as_millis()).unwrap_or(u32::MAX))
                .unwrap_or(0);
            let mut af = crate::anim_encode::AnimFrame::new(
                f.image.width,
                f.image.height,
                f.image.to_rgba8(),
                ms,
            );
            af.mode = opts.frame_mode;
            Ok(af)
        })
        .collect::<Result<_, WebpError>>()?;
    let meta = opts.filtered_metadata(&frames[0].image.metadata);
    crate::anim_encode::build_animation(
        &anim_frames,
        opts.loop_count,
        opts.background_rgba,
        &meta,
        &opts.delta,
    )
}

/// Encode positioned animation frames — the WebP-specific depth under
/// [`encode_animation`]: each [`AnimFrame`] carries its own canvas offset
/// (even `x` / `y`), blend / dispose flags and compression mode. The
/// canvas is sized to cover every frame; `opts` supplies `loop_count`,
/// `background_rgba`, `delta` and the metadata embed flags (`metadata`
/// itself travels on the frames' images in [`encode_animation`]; here
/// pass it through [`EncodeOptions`]-filtered `metadata`).
pub fn encode_animation_frames(
    frames: &[crate::anim_encode::AnimFrame],
    metadata: &Metadata,
    opts: &EncodeOptions,
) -> Result<Vec<u8>, WebpError> {
    if opts.quality.is_some() {
        return Err(WebpError::unsupported(
            "lossy animation frames are not implemented; use the lossless default",
        ));
    }
    let meta = opts.filtered_metadata(metadata);
    crate::anim_encode::build_animation(
        frames,
        opts.loop_count,
        opts.background_rgba,
        &meta,
        &opts.delta,
    )
}

/// Read the `ICCP` / `EXIF` / `XMP ` payloads without decoding pixels.
/// [`info`] reports their presence; [`decode`] fills
/// [`WebpImage::metadata`] with the same bytes.
pub fn read_metadata(bytes: &[u8]) -> Result<Metadata, WebpError> {
    let c = container::parse(bytes)?;
    Ok(metadata_from_container(bytes, &c))
}

/// Decode an animated file's §2.7.1.1 `ANIM` parameters without
/// decoding pixels: `(loop_count, background_rgba)`; `None` for a still.
pub fn animation_params(bytes: &[u8]) -> Result<Option<(u16, [u8; 4])>, WebpError> {
    let c = container::parse(bytes)?;
    let Some(ch) = c.first_chunk_with_fourcc(fourcc::ANIM) else {
        return Ok(None);
    };
    let a = anim::AnimHeader::parse(ch.payload(bytes))?;
    let bg = a.background_color;
    Ok(Some((a.loop_count, [bg.red, bg.green, bg.blue, bg.alpha])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ColorInfo;

    const LOSSY_1X1: &[u8] = include_bytes!("../tests/data/lossy-1x1.webp");
    const LOSSLESS_1X1: &[u8] = include_bytes!("../tests/data/lossless-1x1.webp");
    const LOSSY_ALPHA: &[u8] = include_bytes!("../tests/data/lossy-with-alpha-128x128.webp");
    const ANIM: &[u8] = include_bytes!("../tests/data/animated-3-frames-rgb.webp");

    #[test]
    fn probe_sniffs_the_riff_webp_header() {
        assert!(probe(LOSSY_1X1));
        assert!(probe(LOSSLESS_1X1));
        assert!(!probe(b"RIFF"));
        assert!(!probe(&[]));
        assert!(!probe(b"RIFF\0\0\0\0WAVE"));
    }

    #[test]
    fn info_reports_native_layouts() {
        let i = info(LOSSY_1X1).unwrap();
        assert_eq!((i.width, i.height), (1, 1));
        assert_eq!(i.format, WebpPixelFormat::Yuv420P);
        assert!(!i.has_alpha && i.is_lossy && !i.is_animated);
        assert_eq!(i.color, ColorInfo::bt601_limited());

        let i = info(LOSSLESS_1X1).unwrap();
        assert_eq!(i.format, WebpPixelFormat::Rgba);
        assert!(!i.is_lossy);

        let i = info(LOSSY_ALPHA).unwrap();
        assert_eq!(i.format, WebpPixelFormat::Yuva420P);
        assert!(i.has_alpha);

        let i = info(ANIM).unwrap();
        assert!(i.is_animated);
        assert_eq!(i.frames, 3);
        assert_eq!(i.format, WebpPixelFormat::Rgba);
        assert_eq!(i.loop_count, Some(0));
    }

    #[test]
    fn decode_lossy_is_native_yuv_and_converts_to_the_reference_pixel() {
        let img = decode(LOSSY_1X1).unwrap();
        assert_eq!(img.format, WebpPixelFormat::Yuv420P);
        assert_eq!(img.planes.len(), 3);
        // The reference decoder's black-box non-fancy output for the
        // docs fixture: 0xB1 0x3D 0x57 from Y'CbCr (101, 122, 177).
        assert_eq!(img.planes[0].data, vec![101]);
        assert_eq!(img.planes[1].data, vec![122]);
        assert_eq!(img.planes[2].data, vec![177]);
        assert_eq!(img.to_rgba8(), vec![0xB1, 0x3D, 0x57, 0xFF]);
        assert_eq!(decode_rgb8(LOSSY_1X1).unwrap().data, vec![0xB1, 0x3D, 0x57]);
    }

    #[test]
    fn decode_lossy_with_alpha_is_yuva() {
        let img = decode(LOSSY_ALPHA).unwrap();
        assert_eq!(img.format, WebpPixelFormat::Yuva420P);
        assert_eq!(img.planes.len(), 4);
        assert!(img.planes[3].data.iter().any(|&a| a != 0xff));
        let rgba = img.to_rgba8();
        assert_eq!(rgba.len(), 128 * 128 * 4);
    }

    #[test]
    fn limits_fire_before_decode() {
        let opts = DecodeOptions::default().with_max_width(Some(64));
        let e = decode_with(LOSSY_ALPHA, &opts).unwrap_err();
        assert!(e.is_limit_exceeded(), "{e}");
        let e = decode_with(
            LOSSY_1X1,
            &DecodeOptions::default().with_max_bytes(Some(10)),
        )
        .unwrap_err();
        assert!(e.is_limit_exceeded());
        let e =
            decode_all_with(ANIM, &DecodeOptions::default().with_max_pixels(Some(10))).unwrap_err();
        assert!(e.is_limit_exceeded());
    }

    #[test]
    fn decode_all_composites_every_animation_frame() {
        let frames = decode_all(ANIM).unwrap();
        assert_eq!(frames.len(), 3);
        for f in &frames {
            assert_eq!(f.image.format, WebpPixelFormat::Rgba);
            assert_eq!((f.image.width, f.image.height), (64, 64));
            assert!(f.delay.is_some());
        }
        let first = decode(ANIM).unwrap();
        assert_eq!(first, frames[0].image);
        let still = decode_all(LOSSLESS_1X1).unwrap();
        assert_eq!(still.len(), 1);
        assert!(still[0].delay.is_none());
    }

    #[test]
    fn lossless_round_trip_is_exact_for_rgb_and_rgba() {
        let rgba: Vec<u8> = (0..16 * 8 * 4).map(|i| (i * 7 % 251) as u8).collect();
        let bytes = encode_rgba8(16, 8, &rgba, &EncodeOptions::default()).unwrap();
        assert!(probe(&bytes));
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, WebpPixelFormat::Rgba);
        assert_eq!(back.as_bytes().unwrap(), &rgba[..]);
        assert_eq!(decode_rgba8(&bytes).unwrap().data, rgba);

        let rgb: Vec<u8> = (0..5 * 3 * 3).map(|i| (i * 13 % 256) as u8).collect();
        let bytes = encode_rgb8(5, 3, &rgb, &EncodeOptions::default()).unwrap();
        assert_eq!(decode_rgb8(&bytes).unwrap().data, rgb);
        assert!(!info(&bytes).unwrap().has_alpha);
    }

    #[test]
    fn lossless_round_trip_keeps_metadata() {
        let img = WebpImage::from_rgba8(2, 2, vec![9; 16])
            .unwrap()
            .with_metadata(
                Metadata::new()
                    .with_icc(Some(vec![1, 2, 3]))
                    .with_exif(Some(vec![4, 5]))
                    .with_xmp(Some(vec![6])),
            );
        let bytes = encode(&img, &EncodeOptions::default()).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back, img);
        let i = info(&bytes).unwrap();
        assert!(i.has_icc && i.has_exif && i.has_xmp);
        let filtered = encode(
            &img,
            &EncodeOptions::default().with_metadata(false, true, false),
        )
        .unwrap();
        let i = info(&filtered).unwrap();
        assert!(!i.has_icc && i.has_exif && !i.has_xmp);
    }

    #[test]
    fn lossless_refuses_yuv_without_conversion() {
        let img = WebpImage::from_yuv420(2, 2, vec![128; 4], vec![128], vec![128]).unwrap();
        let e = encode(&img, &EncodeOptions::default()).unwrap_err();
        assert!(e.is_unsupported(), "{e}");
    }

    #[test]
    fn lossy_encode_round_trips_close_and_reports_yuv() {
        let (w, h) = (16u32, 16u32);
        let mut rgb = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[(x * 16) as u8, (y * 16) as u8, 128]);
            }
        }
        let bytes = encode_rgb8(w, h, &rgb, &EncodeOptions::default().with_quality(95.0)).unwrap();
        let i = info(&bytes).unwrap();
        assert!(i.is_lossy);
        assert_eq!(i.format, WebpPixelFormat::Yuv420P);
        let back = decode_rgb8(&bytes).unwrap();
        let mae: f64 = back
            .data
            .iter()
            .zip(rgb.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
            .sum::<f64>()
            / rgb.len() as f64;
        assert!(mae < 12.0, "mean abs error {mae}");
    }

    #[test]
    fn lossy_encode_carries_alpha_as_alph() {
        let (w, h) = (8u32, 6u32);
        let mut rgba = Vec::new();
        for i in 0..(w * h) {
            rgba.extend_from_slice(&[200, 100, 50, (i * 5) as u8]);
        }
        let bytes =
            encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_quality(80.0)).unwrap();
        let i = info(&bytes).unwrap();
        assert_eq!(i.format, WebpPixelFormat::Yuva420P);
        assert!(i.has_alpha);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, WebpPixelFormat::Yuva420P);
        let alpha: Vec<u8> = rgba.chunks_exact(4).map(|p| p[3]).collect();
        assert_eq!(back.planes[3].data, alpha, "ALPH is lossless");
        // A Yuva420P image encodes straight through.
        let again = encode(&back, &EncodeOptions::default().with_quality(80.0)).unwrap();
        assert_eq!(decode(&again).unwrap().planes[3].data, alpha);
    }

    #[test]
    fn lossy_refuses_full_range_yuv() {
        let img = WebpImage::from_yuv420(2, 2, vec![128; 4], vec![128], vec![128])
            .unwrap()
            .with_color(ColorInfo::bt601_limited().with_range(crate::ColorRange::Full));
        let e = encode(&img, &EncodeOptions::default().with_quality(50.0)).unwrap_err();
        assert!(e.is_unsupported());
    }

    #[test]
    fn alph_payload_round_trips_through_decode_alpha() {
        for (w, h) in [(1u32, 1u32), (3, 2), (17, 9), (64, 64)] {
            let alpha: Vec<u8> = (0..w * h).map(|i| (i % 7 * 36) as u8).collect();
            let payload = build_alph_payload(&alpha, w, h);
            let back = alph::decode_alpha(&payload, w, h).unwrap();
            assert_eq!(back, alpha, "{w}x{h}");
        }
    }

    #[test]
    fn encode_animation_round_trips_frames() {
        let mk = |v: u8| WebpImage::from_rgba8(4, 4, vec![v; 64]).unwrap();
        let frames = vec![
            Frame::new(mk(10), Some(Duration::from_millis(40))),
            Frame::new(mk(20), Some(Duration::from_millis(80))),
        ];
        let bytes =
            encode_animation(&frames, &EncodeOptions::default().with_loop_count(3)).unwrap();
        let i = info(&bytes).unwrap();
        assert!(i.is_animated);
        assert_eq!(i.frames, 2);
        assert_eq!(i.loop_count, Some(3));
        let back = decode_all(&bytes).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].image.as_bytes().unwrap(), &[10u8; 64][..]);
        assert_eq!(back[1].image.as_bytes().unwrap(), &[20u8; 64][..]);
        assert_eq!(back[1].delay, Some(Duration::from_millis(80)));
        assert_eq!(animation_params(&bytes).unwrap(), Some((3, [0, 0, 0, 0])));
    }

    #[test]
    fn encode_all_mirrors_decode_all() {
        // One delay-less frame: a still, byte-identical to `encode`.
        let still = WebpImage::from_rgba8(2, 2, vec![7; 16]).unwrap();
        let bytes = encode_all(
            &[Frame::new(still.clone(), None)],
            &EncodeOptions::default(),
        )
        .unwrap();
        assert_eq!(bytes, encode(&still, &EncodeOptions::default()).unwrap());
        assert_eq!(decode_all(&bytes).unwrap(), vec![Frame::new(still, None)]);

        // Several frames: lossless animation; frames + delays read back equal.
        let meta = Metadata::new().with_xmp(Some(b"<x/>".to_vec()));
        let mk = |v: u8| {
            WebpImage::from_rgba8(4, 4, (0..64).map(|i| (i as u8).wrapping_mul(v)).collect())
                .unwrap()
                .with_metadata(meta.clone())
        };
        let frames = vec![
            Frame::new(mk(3), Some(Duration::from_millis(40))),
            Frame::new(mk(5), Some(Duration::from_millis(1500))),
            Frame::new(mk(7), Some(Duration::from_millis(0))),
        ];
        let opts = EncodeOptions::default().with_loop_count(2);
        let bytes = encode_all(&frames, &opts).unwrap();
        assert_eq!(info(&bytes).unwrap().frames, 3);
        assert_eq!(decode_all(&bytes).unwrap(), frames);
        assert_eq!(bytes, encode_animation(&frames, &opts).unwrap());

        // A lone frame WITH a delay is a one-frame animation.
        let one = encode_all(&frames[..1], &opts).unwrap();
        assert_eq!(decode_all(&one).unwrap(), frames[..1]);

        assert!(matches!(
            encode_all(&[], &EncodeOptions::default()),
            Err(WebpError::InvalidData(_))
        ));
        assert!(matches!(
            encode_all(&frames, &EncodeOptions::default().with_quality(50.0)),
            Err(WebpError::Unsupported(_))
        ));
    }

    #[test]
    fn decode_from_and_encode_to_stream() {
        let img = decode_from(std::io::Cursor::new(LOSSLESS_1X1)).unwrap();
        let mut out = Vec::new();
        encode_to(&img, &EncodeOptions::default(), &mut out).unwrap();
        assert_eq!(decode(&out).unwrap(), img);
    }

    #[test]
    fn hostile_inputs_return_errors() {
        for bad in [&b""[..], b"RIFF", b"RIFF\x04\0\0\0WEBP", &[0u8; 64][..]] {
            assert!(decode(bad).is_err());
            assert!(info(bad).is_err());
            assert!(decode_all(bad).is_err());
        }
        let mut t = LOSSY_ALPHA.to_vec();
        t.truncate(t.len() / 2);
        assert!(decode(&t).is_err());
    }
}
