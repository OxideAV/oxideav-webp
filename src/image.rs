//! The contract image types: [`WebpImage`] (native layout), the packed
//! [`RgbImage`] / [`RgbaImage`] conveniences, and the small records they
//! carry ([`Plane`], [`ColorInfo`], [`Metadata`], [`Palette`],
//! [`ImageInfo`], [`Frame`]).
//!
//! These shapes are identical across every OxideAV image crate (a copy per
//! crate, not a shared dependency) so a consumer that reads PNG, JPEG and
//! WebP through the standalone layer sees one vocabulary.

use core::time::Duration;

use crate::yuv;

// ───────────────────────────── pixel formats ─────────────────────────────

/// Pixel layouts this crate can decode to or encode from.
///
/// Variant names mirror `oxideav_core::PixelFormat` exactly, so the
/// framework adapter maps them 1:1 by name. Only the layouts WebP can
/// produce or accept are present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WebpPixelFormat {
    /// Packed 8-bit RGB, 3 bytes per pixel. Accepted by the encoder (an
    /// opaque lossless or lossy image); never produced by the decoder.
    Rgb24,
    /// Packed 8-bit RGBA, 4 bytes per pixel. The native layout of a
    /// lossless (`VP8L`) image and of every composited animation frame.
    Rgba,
    /// Planar 8-bit Y′CbCr 4:2:0 (Y, Cb, Cr). The native layout of a lossy
    /// (`VP8 `) image without an `ALPH` chunk; limited (studio) range per
    /// Rec. ITU-R BT.601.
    Yuv420P,
    /// [`Yuv420P`](Self::Yuv420P) plus a fourth full-resolution alpha
    /// plane. The native layout of a lossy image with an `ALPH` chunk.
    Yuva420P,
}

/// Contract alias: every image crate exposes `PixelFormat` at its root.
pub type PixelFormat = WebpPixelFormat;

impl WebpPixelFormat {
    /// Number of planes the layout carries.
    pub fn plane_count(self) -> usize {
        match self {
            Self::Rgb24 | Self::Rgba => 1,
            Self::Yuv420P => 3,
            Self::Yuva420P => 4,
        }
    }

    /// Bytes per pixel of the single plane of a packed layout, `None` for
    /// planar layouts.
    pub fn packed_bytes_per_pixel(self) -> Option<usize> {
        match self {
            Self::Rgb24 => Some(3),
            Self::Rgba => Some(4),
            Self::Yuv420P | Self::Yuva420P => None,
        }
    }

    /// `true` when the layout carries an alpha channel or plane.
    pub fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba | Self::Yuva420P)
    }

    /// `true` for the planar Y′CbCr layouts.
    pub fn is_yuv(self) -> bool {
        matches!(self, Self::Yuv420P | Self::Yuva420P)
    }

    /// Expected plane geometry — `(stride, rows)` per plane — for a
    /// `width × height` image in this layout, in plane order.
    pub fn plane_geometry(self, width: u32, height: u32) -> Vec<(usize, usize)> {
        let w = width as usize;
        let h = height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        match self {
            Self::Rgb24 => vec![(w * 3, h)],
            Self::Rgba => vec![(w * 4, h)],
            Self::Yuv420P => vec![(w, h), (cw, ch), (cw, ch)],
            Self::Yuva420P => vec![(w, h), (cw, ch), (cw, ch), (w, h)],
        }
    }
}

// ─────────────────────────────── records ─────────────────────────────────

/// One image plane: `stride` bytes per row, `data` holding at least
/// `stride × rows` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Plane {
    /// Bytes per row.
    pub stride: usize,
    /// Row-major sample bytes.
    pub data: Vec<u8>,
}

impl Plane {
    /// Build a plane from its stride and bytes.
    pub fn new(stride: usize, data: Vec<u8>) -> Self {
        Self { stride, data }
    }

    /// A tightly packed plane: `stride == row_bytes`.
    pub fn packed(row_bytes: usize, data: Vec<u8>) -> Self {
        Self::new(row_bytes, data)
    }

    /// Row `y` of the plane (`row_bytes` wide), or `None` when the data
    /// is shorter than the stride geometry claims.
    pub fn row(&self, y: usize, row_bytes: usize) -> Option<&[u8]> {
        let start = y.checked_mul(self.stride)?;
        let end = start.checked_add(row_bytes)?;
        self.data.get(start..end)
    }
}

/// Nominal sample range of the stored samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorRange {
    /// No range signalled.
    #[default]
    Unspecified,
    /// Limited (video / studio) range — Y′ in `16..=235`, Cb/Cr in
    /// `16..=240` for 8-bit samples.
    Limited,
    /// Full (PC / JPEG) range — the whole `0..=255` code space.
    Full,
}

/// Colour description of the stored samples: the sample range plus the
/// Rec. ITU-T H.273 `ColourPrimaries` / `TransferCharacteristics` /
/// `MatrixCoefficients` code points (`2` = unspecified).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ColorInfo {
    /// Sample range.
    pub range: ColorRange,
    /// H.273 `ColourPrimaries` code point.
    pub primaries: u8,
    /// H.273 `TransferCharacteristics` code point.
    pub transfer: u8,
    /// H.273 `MatrixCoefficients` code point.
    pub matrix: u8,
}

impl ColorInfo {
    /// H.273 "unspecified" code point.
    pub const UNSPECIFIED_CODE_POINT: u8 = 2;

    /// Build a description from its four parts.
    pub const fn new(range: ColorRange, primaries: u8, transfer: u8, matrix: u8) -> Self {
        Self {
            range,
            primaries,
            transfer,
            matrix,
        }
    }

    /// Every field unspecified.
    pub const fn unspecified() -> Self {
        Self::new(
            ColorRange::Unspecified,
            Self::UNSPECIFIED_CODE_POINT,
            Self::UNSPECIFIED_CODE_POINT,
            Self::UNSPECIFIED_CODE_POINT,
        )
    }

    /// sRGB (IEC 61966-2-1): BT.709 primaries (`1`), sRGB transfer (`13`),
    /// identity matrix (`0`), full range. RFC 9649 §2.7.1.4: "If this
    /// chunk is not present, sRGB SHOULD be assumed" — the default of a
    /// lossless image and of every RGB(A) output.
    pub const fn srgb() -> Self {
        Self::new(ColorRange::Full, 1, 13, 0)
    }

    /// The colour of a lossy (`VP8 `) image's Y′CbCr samples: Rec. ITU-R
    /// BT.601 matrix (H.273 code point `6`, `KR = 0.299`, `KB = 0.114`),
    /// limited range, over the sRGB primaries / transfer the RGB side is
    /// assumed to carry (RFC 6386 §9.2 + RFC 9649 §2.5 / §2.7.1.4).
    pub const fn bt601_limited() -> Self {
        Self::new(ColorRange::Limited, 1, 13, 6)
    }

    /// Builder: replace the range.
    pub const fn with_range(mut self, range: ColorRange) -> Self {
        self.range = range;
        self
    }
}

impl Default for ColorInfo {
    fn default() -> Self {
        Self::unspecified()
    }
}

/// File-level metadata payloads, each raw and uninterpreted.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Metadata {
    /// §2.7.1.4 `ICCP` ICC colour-profile payload.
    pub icc: Option<Vec<u8>>,
    /// §2.7.1.5 `EXIF` payload.
    pub exif: Option<Vec<u8>>,
    /// §2.7.1.5 `XMP ` payload.
    pub xmp: Option<Vec<u8>>,
    /// Display gamma, when the format signals one. WebP has no such
    /// field; always `None` on decode and ignored on encode.
    pub gamma: Option<f32>,
}

impl Metadata {
    /// No metadata at all.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the ICC profile.
    pub fn with_icc(mut self, icc: Option<Vec<u8>>) -> Self {
        self.icc = icc;
        self
    }

    /// Builder: set the Exif payload.
    pub fn with_exif(mut self, exif: Option<Vec<u8>>) -> Self {
        self.exif = exif;
        self
    }

    /// Builder: set the XMP payload.
    pub fn with_xmp(mut self, xmp: Option<Vec<u8>>) -> Self {
        self.xmp = xmp;
        self
    }

    /// `true` when every payload is absent.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none()
    }
}

/// Colour table of an indexed image. WebP has no indexed layout at the
/// API level (VP8L's colour-indexing transform is undone by the decoder),
/// so [`WebpImage::palette`] is always `None`; the type exists so the
/// image shape matches the other crates.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct Palette {
    /// Entries as packed `[R, G, B, A]`.
    pub entries: Vec<[u8; 4]>,
}

impl Palette {
    /// Build a palette from its entries.
    pub fn new(entries: Vec<[u8; 4]>) -> Self {
        Self { entries }
    }
}

// ────────────────────────────── WebpImage ────────────────────────────────

/// A decoded (or to-be-encoded) WebP picture in its native layout.
///
/// `planes` holds exactly one tightly packed plane for the packed layouts
/// ([`Rgb24`](WebpPixelFormat::Rgb24), [`Rgba`](WebpPixelFormat::Rgba)) and
/// three or four planes for the Y′CbCr layouts. Use
/// [`to_rgb8`](Self::to_rgb8) / [`to_rgba8`](Self::to_rgba8) for packed
/// 8-bit output whatever the layout; 16-bit, float and other re-layouts
/// belong to `oxideav-image` / `oxideav-pixfmt`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WebpImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Native layout of `planes`.
    pub format: WebpPixelFormat,
    /// Sample planes in the layout's plane order.
    pub planes: Vec<Plane>,
    /// Colour description of the samples.
    pub color: ColorInfo,
    /// File-level metadata.
    pub metadata: Metadata,
    /// Always `None` for WebP (see [`Palette`]).
    pub palette: Option<Palette>,
}

impl WebpImage {
    /// Build an image from its geometry and planes. The colour defaults to
    /// the layout's WebP default ([`ColorInfo::srgb`] for RGB(A),
    /// [`ColorInfo::bt601_limited`] for Y′CbCr); metadata is empty.
    ///
    /// Rejects with [`WebpError::InvalidData`](crate::WebpError) a zero
    /// dimension or a plane set that does not fit the layout (plane
    /// count, a stride shorter than the row, a buffer shorter than the
    /// rows it must hold — [`Self::check_geometry`]), so an image that
    /// exists is always consistent and [`Self::to_rgb8`] /
    /// [`Self::to_rgba8`] never need to fail.
    pub fn new(
        width: u32,
        height: u32,
        format: WebpPixelFormat,
        planes: Vec<Plane>,
    ) -> Result<Self, crate::WebpError> {
        check_nonzero(width, height)?;
        let img = Self::new_unchecked(width, height, format, planes);
        img.check_geometry()?;
        Ok(img)
    }

    /// The checks [`Self::new`] runs for a tightly packed one-plane image
    /// of `format` (`Rgb24` or `Rgba`) held in `len` bytes, without taking
    /// the bytes. Lets the lossless encoder read a caller's buffer in
    /// place instead of copying it into an image first.
    pub(crate) fn check_packed(
        width: u32,
        height: u32,
        format: WebpPixelFormat,
        len: usize,
    ) -> Result<(), crate::WebpError> {
        check_nonzero(width, height)?;
        let bpp = format.packed_bytes_per_pixel().unwrap_or(4);
        let stride = (width as usize) * bpp;
        check_plane_geometry(format, width, height, std::iter::once((stride, len)))
    }

    /// [`Self::new`] without the geometry check, for images the crate
    /// assembles itself from already-validated buffers.
    pub(crate) fn new_unchecked(
        width: u32,
        height: u32,
        format: WebpPixelFormat,
        planes: Vec<Plane>,
    ) -> Self {
        let color = if format.is_yuv() {
            ColorInfo::bt601_limited()
        } else {
            ColorInfo::srgb()
        };
        Self {
            width,
            height,
            format,
            planes,
            color,
            metadata: Metadata::default(),
            palette: None,
        }
    }

    /// A packed [`Rgb24`](WebpPixelFormat::Rgb24) image, stride `3 × width`;
    /// `InvalidData` when `data` is shorter than `3 × width × height`.
    pub fn from_rgb8(width: u32, height: u32, data: Vec<u8>) -> Result<Self, crate::WebpError> {
        let stride = (width as usize) * 3;
        Self::new(
            width,
            height,
            WebpPixelFormat::Rgb24,
            vec![Plane::packed(stride, data)],
        )
    }

    /// A packed [`Rgba`](WebpPixelFormat::Rgba) image, stride `4 × width`;
    /// `InvalidData` when `data` is shorter than `4 × width × height`.
    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Result<Self, crate::WebpError> {
        let stride = (width as usize) * 4;
        Self::new(
            width,
            height,
            WebpPixelFormat::Rgba,
            vec![Plane::packed(stride, data)],
        )
    }

    /// A planar [`Yuv420P`](WebpPixelFormat::Yuv420P) image from tightly
    /// packed Y (`width × height`), Cb and Cr (`⌈width/2⌉ × ⌈height/2⌉`)
    /// planes, limited-range BT.601; `InvalidData` when a plane is short.
    pub fn from_yuv420(
        width: u32,
        height: u32,
        y: Vec<u8>,
        cb: Vec<u8>,
        cr: Vec<u8>,
    ) -> Result<Self, crate::WebpError> {
        let w = width as usize;
        let cw = w.div_ceil(2);
        Self::new(
            width,
            height,
            WebpPixelFormat::Yuv420P,
            vec![
                Plane::packed(w, y),
                Plane::packed(cw, cb),
                Plane::packed(cw, cr),
            ],
        )
    }

    /// Builder: replace the colour description.
    pub fn with_color(mut self, color: ColorInfo) -> Self {
        self.color = color;
        self
    }

    /// Builder: replace the metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Builder: replace the palette (kept for shape parity; WebP ignores it).
    pub fn with_palette(mut self, palette: Option<Palette>) -> Self {
        self.palette = palette;
        self
    }

    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Native layout.
    pub fn format(&self) -> WebpPixelFormat {
        self.format
    }

    /// `true` when the layout carries alpha.
    pub fn has_alpha(&self) -> bool {
        self.format.has_alpha()
    }

    /// The single plane's bytes for a packed layout; `None` for planar
    /// layouts (use [`into_raw`](Self::into_raw)).
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if self.format.packed_bytes_per_pixel().is_some() {
            self.planes.first().map(|p| p.data.as_slice())
        } else {
            None
        }
    }

    /// Consume the image: the plane bytes for a packed layout, or every
    /// plane concatenated in order (strides as reported) for a planar one.
    pub fn into_raw(self) -> Vec<u8> {
        let mut planes = self.planes.into_iter();
        let Some(first) = planes.next() else {
            return Vec::new();
        };
        let mut out = first.data;
        for p in planes {
            out.extend_from_slice(&p.data);
        }
        out
    }

    /// Validate that `planes` match the layout's geometry: the right plane
    /// count, every stride at least the row width, every plane at least
    /// `stride × rows` bytes.
    pub fn check_geometry(&self) -> Result<(), crate::WebpError> {
        check_plane_geometry(
            self.format,
            self.width,
            self.height,
            self.planes.iter().map(|p| (p.stride, p.data.len())),
        )
    }

    /// Packed 8-bit RGB, `3 × width` bytes per row, alpha dropped.
    ///
    /// Y′CbCr sources go through the exact integer BT.601 kernel in
    /// the `yuv` module (limited range unless `color.range` says `Full`); chroma is
    /// upsampled nearest-neighbour (RFC 6386 leaves the kernel to the
    /// decoder). An image whose planes are shorter than its geometry
    /// yields black for the missing samples rather than panicking.
    pub fn to_rgb8(&self) -> Vec<u8> {
        self.to_packed(3)
    }

    /// Packed 8-bit RGBA, `4 × width` bytes per row, alpha `255` when the
    /// source has none.
    pub fn to_rgba8(&self) -> Vec<u8> {
        self.to_packed(4)
    }

    fn to_packed(&self, out_bpp: usize) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut out = vec![0u8; w * h * out_bpp];
        if out_bpp == 4 {
            for px in out.chunks_exact_mut(4) {
                px[3] = 0xff;
            }
        }
        match self.format {
            WebpPixelFormat::Rgb24 | WebpPixelFormat::Rgba => {
                let in_bpp = self.format.packed_bytes_per_pixel().unwrap_or(4);
                let Some(plane) = self.planes.first() else {
                    return out;
                };
                let n = in_bpp.min(out_bpp);
                for y in 0..h {
                    let Some(row) = plane.row(y, w * in_bpp) else {
                        break;
                    };
                    let orow = &mut out[y * w * out_bpp..(y + 1) * w * out_bpp];
                    for (src, dst) in row.chunks_exact(in_bpp).zip(orow.chunks_exact_mut(out_bpp)) {
                        dst[..n].copy_from_slice(&src[..n]);
                    }
                }
            }
            WebpPixelFormat::Yuv420P | WebpPixelFormat::Yuva420P => {
                let full = self.color.range == ColorRange::Full;
                let cw = w.div_ceil(2);
                let (Some(yp), Some(up), Some(vp)) =
                    (self.planes.first(), self.planes.get(1), self.planes.get(2))
                else {
                    return out;
                };
                let ap = if self.format == WebpPixelFormat::Yuva420P {
                    self.planes.get(3)
                } else {
                    None
                };
                for y in 0..h {
                    let Some(y_row) = yp.row(y, w) else { break };
                    let (Some(u_row), Some(v_row)) = (up.row(y / 2, cw), vp.row(y / 2, cw)) else {
                        break;
                    };
                    let a_row = ap.and_then(|p| p.row(y, w));
                    let orow = &mut out[y * w * out_bpp..(y + 1) * w * out_bpp];
                    yuv::convert_row(y_row, u_row, v_row, a_row, orow, out_bpp, full);
                }
            }
        }
        out
    }
}

// ───────────────────────── packed convenience images ─────────────────────

/// Tightly packed 8-bit RGB, 3 bytes per pixel, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height × 3` bytes.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Build from parts.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel bytes.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

/// Tightly packed 8-bit RGBA, 4 bytes per pixel, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height × 4` bytes.
    pub data: Vec<u8>,
}

impl RgbaImage {
    /// Build from parts.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel bytes.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

// ─────────────────────────────── ImageInfo ───────────────────────────────

/// Header-only description of a WebP file — what [`info`](crate::info)
/// returns without decoding pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageInfo {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Layout [`decode`](crate::decode) would return.
    pub format: WebpPixelFormat,
    /// Number of images: `1` for a still, the `ANMF` count for an
    /// animation.
    pub frames: u32,
    /// `true` when the file signals alpha (`VP8X` `L` flag, a `VP8L`
    /// `alpha_is_used` bit, or an `ALPH` chunk).
    pub has_alpha: bool,
    /// Colour description of the native samples.
    pub color: ColorInfo,
    /// An `ICCP` chunk is present.
    pub has_icc: bool,
    /// An `EXIF` chunk is present.
    pub has_exif: bool,
    /// An `XMP ` chunk is present.
    pub has_xmp: bool,
    /// `true` for an animated file (`VP8X` `A` flag / `ANIM` chunk).
    pub is_animated: bool,
    /// §2.7.1.1 `ANIM` loop count (`0` = forever) for an animation.
    pub loop_count: Option<u16>,
    /// §2.7.1.1 `ANIM` background colour as `[R, G, B, A]` for an
    /// animation.
    pub background_rgba: Option<[u8; 4]>,
    /// `true` when the still image is a lossy `VP8 ` bitstream, `false`
    /// for lossless `VP8L` (an animation reports its first frame).
    pub is_lossy: bool,
}

impl ImageInfo {
    /// Build a still-image description with every flag cleared.
    pub fn new(width: u32, height: u32, format: WebpPixelFormat) -> Self {
        let color = if format.is_yuv() {
            ColorInfo::bt601_limited()
        } else {
            ColorInfo::srgb()
        };
        Self {
            width,
            height,
            format,
            frames: 1,
            has_alpha: format.has_alpha(),
            color,
            has_icc: false,
            has_exif: false,
            has_xmp: false,
            is_animated: false,
            loop_count: None,
            background_rgba: None,
            is_lossy: format.is_yuv(),
        }
    }
}

// ───────────────────────────────── Frame ─────────────────────────────────

/// One image of a multi-image file — an animation frame composited onto
/// the canvas, or the single still.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Frame {
    /// The composited canvas after this frame is rendered (always the
    /// canvas size, [`Rgba`](WebpPixelFormat::Rgba) for an animation; the
    /// native layout for a still).
    pub image: WebpImage,
    /// Display duration; `None` for a still image.
    pub delay: Option<Duration>,
}

impl Frame {
    /// Build a frame.
    pub fn new(image: WebpImage, delay: Option<Duration>) -> Self {
        Self { image, delay }
    }
}

/// The zero-dimension check of [`WebpImage::new`].
fn check_nonzero(width: u32, height: u32) -> Result<(), crate::WebpError> {
    if width == 0 || height == 0 {
        return Err(crate::WebpError::invalid(format!(
            "image has a zero dimension ({width}x{height})"
        )));
    }
    Ok(())
}

/// [`WebpImage::check_geometry`] over `(stride, byte length)` per plane.
fn check_plane_geometry(
    format: WebpPixelFormat,
    width: u32,
    height: u32,
    planes: impl ExactSizeIterator<Item = (usize, usize)>,
) -> Result<(), crate::WebpError> {
    let geom = format.plane_geometry(width, height);
    if planes.len() != geom.len() {
        return Err(crate::WebpError::invalid(format!(
            "image has {} plane(s), {:?} needs {}",
            planes.len(),
            format,
            geom.len()
        )));
    }
    for (i, ((stride, len), (row_bytes, rows))) in planes.zip(geom).enumerate() {
        if stride < row_bytes {
            return Err(crate::WebpError::invalid(format!(
                "plane {i}: stride {stride} < row width {row_bytes}"
            )));
        }
        let need = if rows == 0 {
            0
        } else {
            stride
                .checked_mul(rows - 1)
                .and_then(|v| v.checked_add(row_bytes))
                .ok_or_else(|| crate::WebpError::invalid("plane geometry overflows"))?
        };
        if len < need {
            return Err(crate::WebpError::invalid(format!(
                "plane {i}: {len} bytes, geometry needs {need}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_to_rgb8_drops_alpha_and_rgb_to_rgba8_sets_opaque() {
        let img = WebpImage::from_rgba8(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!(img.to_rgb8(), vec![1, 2, 3, 5, 6, 7]);
        assert_eq!(img.as_bytes(), Some(&[1u8, 2, 3, 4, 5, 6, 7, 8][..]));
        let rgb = WebpImage::from_rgb8(2, 1, vec![1, 2, 3, 5, 6, 7]).unwrap();
        assert_eq!(rgb.to_rgba8(), vec![1, 2, 3, 255, 5, 6, 7, 255]);
        assert_eq!(rgb.to_rgb8(), vec![1, 2, 3, 5, 6, 7]);
    }

    #[test]
    fn yuv_neutral_grey_round_trips_limited_range() {
        // Y′ = 16 → black, Y′ = 235 → white, Cb = Cr = 128 neutral.
        let img =
            WebpImage::from_yuv420(2, 2, vec![16, 235, 126, 16], vec![128], vec![128]).unwrap();
        assert!(img.as_bytes().is_none());
        let rgba = img.to_rgba8();
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[255, 255, 255, 255]);
        assert_eq!(&rgba[8..12], &[128, 128, 128, 255]);
        let rgb = img.to_rgb8();
        assert_eq!(&rgb[0..3], &[0, 0, 0]);
        assert_eq!(&rgb[3..6], &[255, 255, 255]);
    }

    #[test]
    fn yuva_carries_the_alpha_plane() {
        let mut img = WebpImage::from_yuv420(1, 1, vec![128], vec![128], vec![128]).unwrap();
        img.format = WebpPixelFormat::Yuva420P;
        img.planes.push(Plane::packed(1, vec![7]));
        assert_eq!(img.to_rgba8(), vec![130, 130, 130, 7]);
        assert!(img.check_geometry().is_ok());
    }

    #[test]
    fn into_raw_concatenates_planes() {
        let img = WebpImage::from_yuv420(2, 2, vec![1, 2, 3, 4], vec![5], vec![6]).unwrap();
        assert_eq!(img.into_raw(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn constructors_reject_bad_geometry() {
        let bad = |r: Result<WebpImage, crate::WebpError>| {
            assert!(matches!(r, Err(crate::WebpError::InvalidData(_))), "{r:?}")
        };
        bad(WebpImage::from_rgb8(2, 1, vec![0; 5]));
        bad(WebpImage::from_rgba8(1, 2, vec![0; 7]));
        bad(WebpImage::from_rgba8(0, 2, vec![]));
        bad(WebpImage::from_yuv420(2, 2, vec![0; 4], vec![0], vec![]));
        bad(WebpImage::new(
            4,
            4,
            WebpPixelFormat::Rgba,
            vec![Plane::packed(16, vec![9; 20])],
        ));
        bad(WebpImage::new(
            4,
            4,
            WebpPixelFormat::Yuv420P,
            vec![Plane::packed(4, vec![0; 16])],
        ));
        bad(WebpImage::new(
            2,
            1,
            WebpPixelFormat::Rgb24,
            vec![Plane::new(5, vec![0; 6])],
        ));
        // The last row may be unpadded.
        assert!(WebpImage::new(
            1,
            2,
            WebpPixelFormat::Rgba,
            vec![Plane::new(8, vec![0; 12])]
        )
        .is_ok());
        assert!(WebpImage::from_rgba8(2, 2, vec![0; 16]).is_ok());
    }

    #[test]
    fn short_planes_never_panic() {
        // `new` refuses this; the kernels stay defensive for images the
        // crate assembles itself.
        let img = WebpImage::new_unchecked(
            4,
            4,
            WebpPixelFormat::Rgba,
            vec![Plane::packed(16, vec![9; 20])],
        );
        assert!(img.check_geometry().is_err());
        let rgba = img.to_rgba8();
        assert_eq!(rgba.len(), 64);
        assert_eq!(&rgba[0..4], &[9, 9, 9, 9]);
        assert_eq!(&rgba[60..64], &[0, 0, 0, 255]);
    }
}
