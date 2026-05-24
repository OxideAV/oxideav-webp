//! Animation *encoder* — the published-0.1.5 `build_animated_webp` surface,
//! rebuilt clean-room on top of the in-crate §3.7 VP8L lossless encoder and
//! the §2.7.1.1 `ANIM` / `ANMF` container framing.
//!
//! Where [`crate::anim`] / [`crate::anmf`] *parse* the global and per-frame
//! animation headers, and [`crate::build`] frames a single still image, this
//! module assembles a **multi-frame** animated `.webp` from a list of
//! caller-supplied frames:
//!
//! ```text
//! RIFF | WEBP | VP8X(A flag) | [ICCP] | ANIM | ANMF… ANMF | [EXIF] | [XMP ]
//! ```
//!
//! Each `ANMF` carries the §2.7.1.1 Figure 9 16-byte header (frame X / Y /
//! width / height / duration plus the `Reserved|B|D` info byte) followed by
//! its "Frame Data" — a padded §2.3 sub-RIFF holding a single §2.6 `VP8L`
//! chunk for the [`AnimFrameMode::Lossless`] path. The bitstream itself is
//! produced by [`crate::vp8l_encode::encode_vp8l_argb_with`], so the encoded
//! file decodes back through [`crate::decode_webp`] (animation path) to the
//! exact input pixels.
//!
//! ## What this module does NOT do
//!
//! The [`AnimFrameMode::Auto`] and [`AnimFrameMode::Delta`] modes need the
//! VP8 *lossy* bitstream encoder and inter-frame delta detection, which are
//! blocked on the `oxideav-vp8` dependency (workspace task #1041). Selecting
//! them returns [`WebpError::Unsupported`] rather than silently falling back
//! to a lossless encode. The [`DeltaConfig`] / [`DownsampleKernel`] knobs are
//! re-exposed for API-shape compatibility but only feed the (still blocked)
//! delta path.

use crate::anmf::{BlendingMethod, DisposalMethod};
use crate::build::{self, Vp8xFlags};
use crate::container::fourcc;
use crate::vp8l_encode;
use crate::{Error, WebpError, WebpMetadata};

/// §2.7.1.1 Figure 9 fixed `ANMF` header length (5 × uint24 + 1 info byte).
const ANMF_HEADER_LEN: usize = 16;

/// §2.7.1.1 Figure 8 fixed `ANIM` payload length (uint32 bg + uint16 loop).
const ANIM_PAYLOAD_LEN: usize = 6;

/// How a single animation frame's pixels are compressed into its `ANMF`
/// "Frame Data" bitstream subchunk.
///
/// Reproduces the published-0.1.5 variant set. Only [`Self::Lossless`] is
/// wired up in this build; the lossy / delta modes are blocked on the
/// `oxideav-vp8` bitstream encoder (workspace task #1041) and surface a
/// [`WebpError::Unsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimFrameMode {
    /// Pick the smaller of a lossy keyframe and an inter-frame delta per
    /// frame. **Blocked** — needs the VP8 lossy encoder; returns
    /// [`WebpError::Unsupported`].
    #[default]
    Auto,
    /// Encode the frame as an inter-frame delta against the previous canvas.
    /// **Blocked** — needs the VP8 lossy encoder; returns
    /// [`WebpError::Unsupported`].
    Delta,
    /// Encode the frame as a standalone §2.6 `VP8L` lossless keyframe. Fully
    /// supported.
    Lossless,
}

/// A single animation frame to encode.
///
/// `pixels` is `width * height * 4` interleaved 8-bit `[R, G, B, A]` bytes in
/// scan-line order — the same flat layout [`crate::WebpFrame::rgba`] decodes
/// to. `x` / `y` place the frame's upper-left corner on the canvas (must be
/// even per §2.7.1.1, since the on-disk field is the coordinate / 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimFrame {
    /// `width * height * 4` interleaved `[R, G, B, A]` bytes, scan order.
    pub pixels: Vec<u8>,
    /// Frame width in pixels (≥ 1).
    pub width: u32,
    /// Frame height in pixels (≥ 1).
    pub height: u32,
    /// X coordinate of the frame's upper-left corner on the canvas. Must be
    /// even (§2.7.1.1 stores `x / 2`).
    pub x: u32,
    /// Y coordinate of the frame's upper-left corner on the canvas. Must be
    /// even (§2.7.1.1 stores `y / 2`).
    pub y: u32,
    /// Display duration in 1-millisecond units (the §2.7.1.1 `Frame
    /// Duration` field).
    pub duration: u32,
    /// §2.7.1.1 blending method (`B` bit).
    pub blend: BlendingMethod,
    /// §2.7.1.1 disposal method (`D` bit).
    pub dispose: DisposalMethod,
    /// Per-frame compression mode.
    pub mode: AnimFrameMode,
}

impl AnimFrame {
    /// Construct an opaque, top-left, alpha-blended, non-disposed lossless
    /// frame from a flat RGBA buffer — the common case.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>, duration: u32) -> Self {
        Self {
            pixels,
            width,
            height,
            x: 0,
            y: 0,
            duration,
            blend: BlendingMethod::AlphaBlend,
            dispose: DisposalMethod::None,
            mode: AnimFrameMode::Lossless,
        }
    }
}

/// Multi-scale SSIM downsample kernel selector for the (blocked) delta path.
///
/// Re-exposed for published-API shape compatibility. Has no effect on the
/// lossless path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DownsampleKernel {
    /// Plain box (average) downsample.
    #[default]
    Box,
    /// Gaussian-weighted downsample.
    Gaussian,
}

/// Tuning knobs for the inter-frame delta path.
///
/// Re-exposed for published-API shape compatibility. The fields feed the
/// (still blocked) [`AnimFrameMode::Delta`] / [`AnimFrameMode::Auto`] paths;
/// they have no effect on the lossless path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaConfig {
    /// Maximum number of disjoint dirty-rectangle components to keep when
    /// diffing two frames before falling back to a full keyframe.
    pub max_components: usize,
    /// Optional byte threshold below which an inner sub-rectangle delta is
    /// preferred. `None` disables the heuristic.
    pub auto_inner_threshold_bytes: Option<usize>,
    /// Which kernel to use when downsampling for the MS-SSIM quality gate.
    pub msssim_downsample_kernel: DownsampleKernel,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            max_components: 8,
            auto_inner_threshold_bytes: None,
            msssim_downsample_kernel: DownsampleKernel::Box,
        }
    }
}

impl DeltaConfig {
    /// Override the maximum dirty-rectangle component count (builder form).
    pub fn max_components_override(mut self, n: usize) -> Self {
        self.max_components = n;
        self
    }

    /// Set the inner-rectangle byte threshold (builder form).
    pub fn auto_inner_threshold_bytes(mut self, bytes: Option<usize>) -> Self {
        self.auto_inner_threshold_bytes = bytes;
        self
    }

    /// Select the MS-SSIM downsample kernel (builder form).
    pub fn msssim_downsample_kernel(mut self, kernel: DownsampleKernel) -> Self {
        self.msssim_downsample_kernel = kernel;
        self
    }
}

/// Options for [`build_animated_webp_with_options`].
///
/// `loop_count` is the §2.7.1.1 `ANIM` loop count (`0` = loop forever).
/// `background_rgba` is the `ANIM` background colour as `[R, G, B, A]`. The
/// borrowed [`WebpMetadata`] carries optional ICC / Exif / XMP payloads to
/// embed in the §2.7 chunk order. `delta` tunes the (blocked) delta path.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnimEncoderOptions<'a> {
    /// §2.7.1.1 `ANIM` loop count. `0` means "loop infinitely".
    pub loop_count: u16,
    /// §2.7.1.1 `ANIM` background colour as `[R, G, B, A]`.
    pub background_rgba: [u8; 4],
    /// File-level metadata (ICC / Exif / XMP) to embed, borrowed.
    pub metadata: WebpMetadata<'a>,
    /// Tuning for the (blocked) inter-frame delta path.
    pub delta: DeltaConfig,
}

/// Build an animated `.webp` from `frames`, defaulting all encoder options
/// (infinite loop, transparent-black background, no metadata).
///
/// Convenience wrapper over [`build_animated_webp_with_options`]; see that
/// function for the full semantics.
pub fn build_animated_webp(frames: &[AnimFrame]) -> Result<Vec<u8>, WebpError> {
    build_animated_webp_with_options(frames, &AnimEncoderOptions::default())
}

/// Assemble a complete animated `RIFF/WEBP` file from `frames` per
/// RFC 9649 §2.7.1.1.
///
/// Output layout:
///
/// ```text
/// RIFF | WEBP | VP8X(A[,L][,I][,E][,X]) | [ICCP] | ANIM | ANMF… | [EXIF] | [XMP ]
/// ```
///
/// The §2.7.1 `VP8X` canvas is sized to cover every frame
/// (`max(frame.x + frame.width)` × `max(frame.y + frame.height)`). The `A`
/// (animation) flag is always set; `L` (alpha) is set when any frame carries
/// a non-opaque pixel; `I` / `E` / `X` follow the supplied metadata. Each
/// frame's [`AnimFrameMode::Lossless`] pixels are encoded to a §2.6 `VP8L`
/// chunk via [`crate::vp8l_encode::encode_vp8l_argb_with`] and wrapped in the
/// `ANMF` "Frame Data" sub-RIFF.
///
/// [`AnimFrameMode::Auto`] / [`AnimFrameMode::Delta`] frames return
/// [`WebpError::Unsupported`] (the VP8 lossy + delta paths are blocked on
/// `oxideav-vp8`). An empty `frames` slice, a frame whose `pixels` length
/// disagrees with `width * height * 4`, or an odd `x` / `y` offset is
/// [`WebpError::InvalidData`].
pub fn build_animated_webp_with_options(
    frames: &[AnimFrame],
    opts: &AnimEncoderOptions<'_>,
) -> Result<Vec<u8>, WebpError> {
    if frames.is_empty() {
        return Err(WebpError::InvalidData);
    }

    // §2.7.1.1: canvas must cover every frame rectangle.
    let mut canvas_width = 0u32;
    let mut canvas_height = 0u32;
    let mut any_alpha = false;

    for f in frames {
        match f.mode {
            AnimFrameMode::Lossless => {}
            // Lossy keyframe / inter-frame delta need the VP8 lossy encoder.
            AnimFrameMode::Auto | AnimFrameMode::Delta => return Err(WebpError::Unsupported),
        }
        if f.width == 0 || f.height == 0 {
            return Err(WebpError::InvalidData);
        }
        // §2.7.1.1 stores Frame X / Frame Y as coord/2, so only even
        // offsets are representable.
        if f.x & 1 != 0 || f.y & 1 != 0 {
            return Err(WebpError::InvalidData);
        }
        let expected = (f.width as usize)
            .checked_mul(f.height as usize)
            .and_then(|n| n.checked_mul(4));
        if expected != Some(f.pixels.len()) {
            return Err(WebpError::InvalidData);
        }
        let right = f.x.checked_add(f.width).ok_or(WebpError::InvalidData)?;
        let bottom = f.y.checked_add(f.height).ok_or(WebpError::InvalidData)?;
        canvas_width = canvas_width.max(right);
        canvas_height = canvas_height.max(bottom);
        if f.pixels.chunks_exact(4).any(|px| px[3] != 0xff) {
            any_alpha = true;
        }
    }

    let meta = &opts.metadata;

    // §2.7.1 VP8X flag octet — animation always; alpha/metadata as present.
    let flags = Vp8xFlags {
        has_iccp: meta.icc.is_some(),
        has_alpha: any_alpha,
        has_exif: meta.exif.is_some(),
        has_xmp: meta.xmp.is_some(),
        has_animation: true,
    };
    let vp8x_payload = build::build_vp8x_chunk(canvas_width, canvas_height, flags).map_err(to_w)?;

    let mut body = Vec::new();
    let mut push = |fourcc, payload: &[u8]| -> Result<(), WebpError> {
        let chunk = build::build_chunk(fourcc, payload).map_err(to_w)?;
        body.extend_from_slice(&chunk);
        Ok(())
    };

    // §2.7 chunk order: VP8X, ICCP, ANIM, ANMF…, EXIF, XMP.
    push(fourcc::VP8X, &vp8x_payload)?;
    if let Some(icc) = meta.icc {
        push(fourcc::ICCP, icc)?;
    }
    push(fourcc::ANIM, &build_anim_payload(opts))?;
    for f in frames {
        let anmf_payload = build_anmf_payload(f)?;
        push(fourcc::ANMF, &anmf_payload)?;
    }
    if let Some(exif) = meta.exif {
        push(fourcc::EXIF, exif)?;
    }
    if let Some(xmp) = meta.xmp {
        push(fourcc::XMP, xmp)?;
    }

    // §2.4 file framing: "RIFF" | File Size (= body + 4 for "WEBP") | "WEBP".
    let file_size = (body.len() as u64) + 4;
    if file_size > u64::from(u32::MAX) {
        return Err(WebpError::InvalidData);
    }
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&fourcc::RIFF);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&fourcc::WEBP);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Build the 6-byte §2.7.1.1 Figure 8 `ANIM` payload: BGRA background colour
/// (the `[R,G,B,A]` option re-ordered to on-disk `[B,G,R,A]`) + LE u16 loop
/// count.
fn build_anim_payload(opts: &AnimEncoderOptions<'_>) -> Vec<u8> {
    let [r, g, b, a] = opts.background_rgba;
    let mut p = Vec::with_capacity(ANIM_PAYLOAD_LEN);
    // §2.7.1.1: on-disk byte order is [Blue, Green, Red, Alpha].
    p.push(b);
    p.push(g);
    p.push(r);
    p.push(a);
    p.extend_from_slice(&opts.loop_count.to_le_bytes());
    p
}

/// Build a single `ANMF` chunk payload: the 16-byte §2.7.1.1 Figure 9 header
/// followed by the per-frame "Frame Data" sub-RIFF (a §2.6 `VP8L` chunk).
fn build_anmf_payload(f: &AnimFrame) -> Result<Vec<u8>, WebpError> {
    // Encode the frame's pixels to a bare VP8L bitstream (image-header +
    // image stream), then wrap as a VP8L chunk for the Frame Data sub-RIFF.
    let argb = rgba_to_argb(&f.pixels);
    let has_alpha = f.pixels.chunks_exact(4).any(|px| px[3] != 0xff);
    let bitstream = vp8l_encode::encode_vp8l_argb_with(&argb, f.width, f.height, has_alpha)
        .map_err(Error::from)
        .map_err(WebpError::from)?;
    let frame_data = build::build_chunk(fourcc::VP8L, &bitstream).map_err(to_w)?;

    let mut payload = Vec::with_capacity(ANMF_HEADER_LEN + frame_data.len());
    // §2.7.1.1 Figure 9: Frame X / Y stored as coord/2 (uint24 LE).
    push_u24_le(&mut payload, f.x / 2);
    push_u24_le(&mut payload, f.y / 2);
    // Frame Width / Height Minus One (uint24 LE).
    push_u24_le(&mut payload, f.width - 1);
    push_u24_le(&mut payload, f.height - 1);
    // Frame Duration (uint24 LE, ms).
    push_u24_le(&mut payload, f.duration & 0x00FF_FFFF);
    // Reserved(6) | B | D info byte.
    let b_bit = match f.blend {
        BlendingMethod::AlphaBlend => 0,
        BlendingMethod::Overwrite => 1,
    };
    let d_bit = match f.dispose {
        DisposalMethod::None => 0,
        DisposalMethod::Background => 1,
    };
    payload.push((b_bit << 1) | d_bit);

    payload.extend_from_slice(&frame_data);
    Ok(payload)
}

/// Push the low 24 bits of `v` as three little-endian bytes.
fn push_u24_le(out: &mut Vec<u8>, v: u32) {
    out.push((v & 0xFF) as u8);
    out.push(((v >> 8) & 0xFF) as u8);
    out.push(((v >> 16) & 0xFF) as u8);
}

/// Repack interleaved `[R, G, B, A]` bytes into packed ARGB
/// (`(a<<24)|(r<<16)|(g<<8)|b`) — the layout the VP8L encoder consumes.
fn rgba_to_argb(rgba: &[u8]) -> Vec<u32> {
    rgba.chunks_exact(4)
        .map(|px| {
            let (r, g, b, a) = (px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32);
            (a << 24) | (r << 16) | (g << 8) | b
        })
        .collect()
}

/// Collapse a [`crate::build::BuildError`] into the published coarse
/// [`WebpError::InvalidData`].
fn to_w(_e: build::BuildError) -> WebpError {
    WebpError::InvalidData
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&color);
        }
        v
    }

    #[test]
    fn empty_frames_is_invalid_data() {
        assert_eq!(build_animated_webp(&[]), Err(WebpError::InvalidData));
    }

    #[test]
    fn auto_and_delta_modes_are_unsupported() {
        let mut f = AnimFrame::new(2, 2, solid_rgba(2, 2, [1, 2, 3, 255]), 100);
        f.mode = AnimFrameMode::Auto;
        assert_eq!(
            build_animated_webp(&[f.clone()]),
            Err(WebpError::Unsupported)
        );
        f.mode = AnimFrameMode::Delta;
        assert_eq!(build_animated_webp(&[f]), Err(WebpError::Unsupported));
    }

    #[test]
    fn pixel_length_mismatch_is_invalid_data() {
        let mut f = AnimFrame::new(2, 2, solid_rgba(2, 2, [0, 0, 0, 255]), 0);
        f.pixels.truncate(4);
        assert_eq!(build_animated_webp(&[f]), Err(WebpError::InvalidData));
    }

    #[test]
    fn odd_offset_is_invalid_data() {
        let mut f = AnimFrame::new(2, 2, solid_rgba(2, 2, [0, 0, 0, 255]), 0);
        f.x = 1;
        assert_eq!(build_animated_webp(&[f]), Err(WebpError::InvalidData));
    }

    #[test]
    fn output_begins_with_riff_webp_and_is_parseable() {
        let f = AnimFrame::new(4, 4, solid_rgba(4, 4, [10, 20, 30, 255]), 100);
        let file = build_animated_webp(&[f]).expect("build animated webp");
        assert_eq!(&file[0..4], b"RIFF");
        assert_eq!(&file[8..12], b"WEBP");
        // The container walker must accept it.
        let c = crate::container::parse(&file).expect("parseable container");
        // VP8X then ANIM then ANMF must all be present.
        assert!(c.first_chunk_with_fourcc(fourcc::VP8X).is_some());
        assert!(c.first_chunk_with_fourcc(fourcc::ANIM).is_some());
        assert!(c.first_chunk_with_fourcc(fourcc::ANMF).is_some());
    }

    #[test]
    fn delta_config_builders_chain() {
        let cfg = DeltaConfig::default()
            .max_components_override(3)
            .auto_inner_threshold_bytes(Some(512))
            .msssim_downsample_kernel(DownsampleKernel::Gaussian);
        assert_eq!(cfg.max_components, 3);
        assert_eq!(cfg.auto_inner_threshold_bytes, Some(512));
        assert_eq!(cfg.msssim_downsample_kernel, DownsampleKernel::Gaussian);
    }
}
