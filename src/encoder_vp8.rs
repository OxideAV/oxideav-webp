//! Published-API `oxideav_webp::encoder_vp8` module — VP8 lossy
//! factory façade, plus the §6 / §9 quality-knob types reproduced on
//! top of the workspace `oxideav-vp8` sibling crate's public surface.
//!
//! Per the published 0.1.2 surface, this module exposes:
//!
//! * [`make_encoder`], [`make_encoder_with_quality`],
//!   [`make_encoder_with_qindex`],
//!   [`make_encoder_with_qindex_and_freq_deltas`],
//!   [`make_encoder_with_quality_and_freq_deltas`] — `Box<dyn Encoder>`
//!   factories. Gated behind the default-on `registry` feature because
//!   the framework `Encoder` trait lives in `oxideav-core`.
//! * [`Vp8FreqDeltas`] — the 5-field i8 per-band quantiser-delta
//!   record (`y_dc_delta`, `y2_dc_delta`, `y2_ac_delta`, `uv_dc_delta`,
//!   `uv_ac_delta`). Each on-disk field is a 5-bit signed-magnitude
//!   number clamped to [-15, 15] per RFC 6386 §9.1.
//! * [`quality_to_qindex`] — the §9 quality → qindex projection
//!   `round((100 - quality) * 1.27)`, clamped to `0..=127`; NaN
//!   collapses to `127`. Standalone (no `oxideav-core` dependency).
//!
//! ## Round-168 wiring
//!
//! As of `oxideav-vp8 0.2.1` the VP8 encoder factories are wired up: the
//! `make_encoder*` family below delegates to
//! [`oxideav_vp8::encoder::make_encoder_with_qindex`] /
//! [`oxideav_vp8::encoder::make_encoder_with_quality`] and wraps the
//! emitted raw VP8 keyframe bitstream in the §2.5 `RIFF/WEBP` container
//! framing so the output decodes back through [`crate::decode_all`].
//! The `_freq_deltas` variants pass through to the matching no-deltas
//! factory in this round — the `Vp8FreqDeltas` argument is forwarded as
//! a hint (the surface stays unchanged) and the per-band quantiser-delta
//! plumbing into the underlying `oxideav-vp8` encoder is deferred.

use crate::WebpError;

/// Per-band quantiser deltas for fine-grained VP8 quality tuning,
/// reproduced from the RFC 6386 §9.1 frame-header `y_dc_delta_q`,
/// `y2_dc_delta_q`, `y2_ac_delta_q`, `uv_dc_delta_q`, `uv_ac_delta_q`
/// fields.
///
/// Each on-disk field is a 5-bit signed-magnitude number — 4 magnitude
/// bits plus a sign bit — clamped to `[-15, 15]` by the bitstream
/// format. The Rust struct keeps the values as plain `i8` for ease of
/// use; consumers that bypass [`make_encoder_with_qindex_and_freq_deltas`]
/// and emit the on-disk bits themselves are responsible for the clamp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vp8FreqDeltas {
    /// `y_dc_delta_q` — luma DC quantiser delta.
    pub y_dc_delta: i8,
    /// `y2_dc_delta_q` — WHT (Y2) DC quantiser delta.
    pub y2_dc_delta: i8,
    /// `y2_ac_delta_q` — WHT (Y2) AC quantiser delta.
    pub y2_ac_delta: i8,
    /// `uv_dc_delta_q` — chroma DC quantiser delta.
    pub uv_dc_delta: i8,
    /// `uv_ac_delta_q` — chroma AC quantiser delta.
    pub uv_ac_delta: i8,
}

/// Map a WebP-canonical `0.0..=100.0` quality scale to a VP8 qindex
/// (`0..=127`, lower = better).
///
/// The mapping is `round((100 - quality) * 1.27)`, clamped to the
/// representable qindex range:
///
/// * `quality = 100.0` → `qindex = 0`  (best quality).
/// * `quality =   0.0` → `qindex = 127` (worst quality).
/// * `quality.is_nan()` → `qindex = 127`.
///
/// Out-of-range quality values are clamped to `[0.0, 100.0]` before
/// projection so the returned qindex is always in `0..=127`.
pub fn quality_to_qindex(quality: f32) -> u8 {
    if quality.is_nan() {
        return 127;
    }
    let q = quality.clamp(0.0, 100.0);
    let qi = ((100.0 - q) * 1.27).round();
    qi.clamp(0.0, 127.0) as u8
}

/// Largest VP8 frame side: the RFC 6386 §9.1 key-frame header stores
/// 14-bit sizes.
const MAX_VP8_DIMENSION: u32 = 0x3FFF;

/// Lossy (`VP8 `) still encoder with a direct constructor: the keyframe
/// encoder the [`make_encoder_with_qindex`] family wraps, built without
/// the `registry` feature or a framework `Encoder`.
///
/// [`encode_yuv420`](Self::encode_yuv420) reads a 4:2:0 picture's planes
/// in place at the caller's strides (the framework encoder copies each
/// plane into a tightly packed buffer first) and writes a simple-lossy
/// `.webp`. For the same picture and quantiser the bytes are those of the
/// `webp_vp8` framework encoder's packet.
///
/// ```
/// use oxideav_webp::encoder_vp8::Vp8LossyEncoder;
///
/// let (w, h) = (32u32, 16u32);
/// let y = vec![128u8; 32 * 16];
/// let (u, v) = (vec![128u8; 16 * 8], vec![128u8; 16 * 8]);
/// let webp = Vp8LossyEncoder::with_quality(80.0)
///     .encode_yuv420(w, h, &y, 32, &u, &v, 16)
///     .expect("encode");
/// assert_eq!(&webp[..4], b"RIFF");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8LossyEncoder {
    qindex: u8,
}

impl Vp8LossyEncoder {
    /// An encoder at WebP `quality` (`0.0..=100.0`), projected onto the VP8
    /// quantiser index with [`quality_to_qindex`], as
    /// [`make_encoder_with_quality`] does.
    pub fn with_quality(quality: f32) -> Self {
        Self::with_qindex(quality_to_qindex(quality))
    }

    /// An encoder at VP8 quantiser index `qindex` (`0..=127`, lower is
    /// better quality); larger values are treated as `127`.
    pub fn with_qindex(qindex: u8) -> Self {
        Self {
            qindex: qindex.min(127),
        }
    }

    /// The VP8 quantiser index this encoder uses.
    pub fn qindex(&self) -> u8 {
        self.qindex
    }

    /// Encode one `width x height` 4:2:0 picture to a simple-lossy
    /// `.webp`. `y` holds `height` rows `y_stride` bytes apart; `u` and
    /// `v` hold `ceil(height / 2)` rows of `ceil(width / 2)` samples,
    /// `uv_stride` bytes apart. Samples are limited-range BT.601, as
    /// VP8 expects.
    ///
    /// A zero or over-16383 dimension, a stride narrower than its row, or
    /// a plane shorter than its rows need is
    /// [`WebpError::InvalidData`](crate::WebpError).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_yuv420(
        &self,
        width: u32,
        height: u32,
        y: &[u8],
        y_stride: usize,
        u: &[u8],
        v: &[u8],
        uv_stride: usize,
    ) -> Result<Vec<u8>, WebpError> {
        if width == 0 || height == 0 || width > MAX_VP8_DIMENSION || height > MAX_VP8_DIMENSION {
            return Err(WebpError::invalid(format!(
                "VP8 picture {width}x{height} is outside 1..={MAX_VP8_DIMENSION} per side"
            )));
        }
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        check_plane("Y", y, y_stride, w, h)?;
        check_plane("U", u, uv_stride, cw, ch)?;
        check_plane("V", v, uv_stride, cw, ch)?;
        let frame = oxideav_vp8::I420Frame {
            width,
            height,
            y,
            u,
            v,
            y_stride,
            uv_stride,
        };
        let params = oxideav_vp8::KeyframeParams {
            y_ac_qi: self.qindex,
            ..oxideav_vp8::KeyframeParams::default()
        };
        let vp8 = oxideav_vp8::encode_keyframe(&frame, &params)?;
        Ok(crate::build::build_webp_file(
            &vp8,
            crate::build::ImageKind::Lossy,
            width,
            height,
        )?)
    }
}

/// Check that `plane` holds `rows` rows of `row_width` bytes, `stride`
/// bytes apart.
fn check_plane(
    name: &str,
    plane: &[u8],
    stride: usize,
    row_width: usize,
    rows: usize,
) -> Result<(), WebpError> {
    if stride < row_width {
        return Err(WebpError::invalid(format!(
            "VP8 {name} plane stride {stride} is narrower than its {row_width}-byte rows"
        )));
    }
    let need = stride
        .checked_mul(rows - 1)
        .and_then(|n| n.checked_add(row_width))
        .ok_or_else(|| WebpError::invalid(format!("VP8 {name} plane geometry overflows")))?;
    if plane.len() < need {
        return Err(WebpError::invalid(format!(
            "VP8 {name} plane holds {} bytes, its rows need {need}",
            plane.len()
        )));
    }
    Ok(())
}

// ───────────────────────── framework-side factories ─────────────────────────

#[cfg(feature = "registry")]
use oxideav_core::{
    time::TimeBase, CodecId, CodecParameters, Encoder, Error as CoreError, Frame, MediaType,
    Packet, Result as CoreResult,
};
#[cfg(feature = "registry")]
use std::collections::VecDeque;

/// Build a `Box<dyn Encoder>` for the published `"webp_vp8"` codec id.
///
/// Routes to the `oxideav-vp8 0.2.1` framework factory
/// [`oxideav_vp8::encoder::make_encoder`] (default `y_ac_qi = 32`) and
/// wraps every emitted raw VP8 keyframe in a §2.5 simple-lossy
/// `RIFF/WEBP` container so the output decodes through
/// [`crate::decode_all`].
#[cfg(feature = "registry")]
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, 32)
}

/// `make_encoder` plus a WebP-canonical `0.0..=100.0` quality knob.
///
/// `quality` is projected to a VP8 qindex (`0..=127`) via
/// [`quality_to_qindex`] and forwarded to the underlying
/// `oxideav-vp8 0.2.1` factory.
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality(
    params: &CodecParameters,
    quality: f32,
) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, quality_to_qindex(quality))
}

/// `make_encoder` plus an explicit `qindex` (`0..=127`, lower = better).
///
/// Builds a [`WebpVp8LossyEncoder`] which delegates to the underlying
/// `oxideav-vp8 0.2.1` framework encoder for the actual VP8 keyframe
/// bitstream, then wraps each emitted packet in a §2.5 simple-lossy
/// `RIFF/WEBP` container.
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex(
    params: &CodecParameters,
    qindex: u8,
) -> CoreResult<Box<dyn Encoder>> {
    let inner = oxideav_vp8::encoder::make_encoder_with_qindex(params, qindex)?;
    let width = params
        .width
        .ok_or_else(|| CoreError::invalid("webp_vp8 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| CoreError::invalid("webp_vp8 encoder: missing height"))?;
    if width == 0 || height == 0 {
        return Err(CoreError::invalid(
            "webp_vp8 encoder: width and height must be positive",
        ));
    }
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(crate::CODEC_ID_VP8);
    output_params.width = Some(width);
    output_params.height = Some(height);
    let time_base = params
        .frame_rate
        .map_or(TimeBase::new(1, 1_000), |r| TimeBase::new(r.den, r.num));
    Ok(Box::new(WebpVp8LossyEncoder {
        inner,
        output_params,
        time_base,
        pending: VecDeque::new(),
        flushed: false,
    }))
}

/// `make_encoder` plus an explicit `qindex` and a [`Vp8FreqDeltas`]
/// record of per-band quantiser deltas.
///
/// In this round the `deltas` argument is forwarded as a hint only —
/// the surface is preserved per the published 0.1.2 surface but plumbing the
/// per-band deltas into the underlying `oxideav-vp8` encoder's
/// `KeyframeParams` is deferred to a follow-up. The qindex IS honoured.
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex_and_freq_deltas(
    params: &CodecParameters,
    qindex: u8,
    _deltas: Vp8FreqDeltas,
) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, qindex)
}

/// `make_encoder` plus a WebP-canonical `0.0..=100.0` quality knob
/// and a [`Vp8FreqDeltas`] record of per-band quantiser deltas.
///
/// In this round the `deltas` argument is forwarded as a hint only —
/// see [`make_encoder_with_qindex_and_freq_deltas`]. The quality IS
/// honoured (projected via [`quality_to_qindex`]).
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality_and_freq_deltas(
    params: &CodecParameters,
    quality: f32,
    _deltas: Vp8FreqDeltas,
) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_quality(params, quality)
}

/// `Encoder` adapter that wraps an inner `oxideav-vp8` encoder and
/// frames every emitted raw VP8 keyframe in a §2.5 simple-lossy
/// `RIFF/WEBP` container. One source frame → one `.webp` packet.
#[cfg(feature = "registry")]
pub(crate) struct WebpVp8LossyEncoder {
    inner: Box<dyn Encoder>,
    output_params: CodecParameters,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    flushed: bool,
}

#[cfg(feature = "registry")]
impl std::fmt::Debug for WebpVp8LossyEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebpVp8LossyEncoder")
            .field("width", &self.output_params.width)
            .field("height", &self.output_params.height)
            .field("pending", &self.pending.len())
            .field("flushed", &self.flushed)
            .finish()
    }
}

#[cfg(feature = "registry")]
impl Encoder for WebpVp8LossyEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> CoreResult<()> {
        // Pass the source frame through to the inner VP8 encoder.
        self.inner.send_frame(frame)?;
        // Drain one keyframe packet, wrap in RIFF/WEBP, queue it.
        let vp8_pkt = self.inner.receive_packet()?;
        let width = self.output_params.width.unwrap_or(0);
        let height = self.output_params.height.unwrap_or(0);
        let webp_bytes = crate::build::build_webp_file(
            &vp8_pkt.data,
            crate::build::ImageKind::Lossy,
            width,
            height,
        )
        .map_err(|e| CoreError::invalid(format!("webp_vp8 encoder: container framing: {e}")))?;
        let mut pkt = Packet::new(0, self.time_base, webp_bytes);
        pkt.pts = vp8_pkt.pts;
        pkt.dts = vp8_pkt.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.flushed {
            Err(CoreError::Eof)
        } else {
            Err(CoreError::NeedMore)
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.flushed = true;
        // Propagate flush to the inner encoder. Ignore errors so we
        // always honour our own flush state.
        let _ = self.inner.flush();
        Ok(())
    }
}

/// Crate-local convenience: surface a `WebpError::Unsupported` over the
/// same message used by the framework-trait factories. Used by the
/// standalone-side tests so they can assert the same intent without
/// pulling `oxideav-core`.
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub fn unsupported_for_standalone() -> WebpError {
    WebpError::unsupported("webp_vp8 framework encoder needs the `registry` feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_to_qindex_endpoints() {
        // Headline endpoint values from the RFC 6386 §9 projection.
        assert_eq!(quality_to_qindex(100.0), 0);
        assert_eq!(quality_to_qindex(0.0), 127);
        // NaN collapses to worst-quality qindex.
        assert_eq!(quality_to_qindex(f32::NAN), 127);
        // Out-of-range values clamp.
        assert_eq!(quality_to_qindex(-1.0), 127);
        assert_eq!(quality_to_qindex(1000.0), 0);
    }

    #[test]
    fn quality_to_qindex_midpoints() {
        // round((100 - 50) * 1.27) = round(63.5) = 64 (banker's? .5 rounds away from 0 for `round`).
        let mid = quality_to_qindex(50.0);
        assert!((63..=64).contains(&mid), "midpoint qindex = {mid}");
        // round((100 - 75) * 1.27) = round(31.75) = 32.
        assert_eq!(quality_to_qindex(75.0), 32);
    }

    #[test]
    fn vp8_freq_deltas_default_is_zero() {
        let d = Vp8FreqDeltas::default();
        assert_eq!(d.y_dc_delta, 0);
        assert_eq!(d.y2_dc_delta, 0);
        assert_eq!(d.y2_ac_delta, 0);
        assert_eq!(d.uv_dc_delta, 0);
        assert_eq!(d.uv_ac_delta, 0);
    }
}
