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
//! * [`WebpVp8EncoderOptions`], [`make_encoder_with_options`],
//!   [`make_encoder_from_param_options`] — the typed options of the
//!   registered `webp_vp8` encoder (`effort`, `threads`, `qindex`,
//!   `quality`) and the factories that take them.
//! * [`Vp8LossyEncoder`] — the same encoder with a direct constructor,
//!   no framework types.
//!
//! Every door ends in one implementation: the key frame is encoded by
//! `oxideav-vp8` from the caller's 4:2:0 planes, read in place, and
//! wrapped in the §2.5 simple-lossy `RIFF` / `WEBP` layout so the output
//! decodes back through [`crate::decode_all`]. The `_freq_deltas`
//! variants accept a [`Vp8FreqDeltas`] record for signature
//! compatibility and encode as the matching no-deltas factory does.

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

/// Map a `0.0..=100.0` quality (higher = better) to a VP8 qindex
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
/// in place at the caller's strides and writes a simple-lossy `.webp`.
/// For the same picture, quantiser and effort the bytes are those of the
/// `webp_vp8` framework encoder's packet.
///
/// ```
/// use oxideav_webp::encoder_vp8::Vp8LossyEncoder;
///
/// let (w, h) = (32u32, 16u32);
/// let y = vec![128u8; 32 * 16];
/// let (u, v) = (vec![128u8; 16 * 8], vec![128u8; 16 * 8]);
/// let webp = Vp8LossyEncoder::with_quality(80.0)
///     .with_effort(1) // a reduced mode search: faster, slightly larger
///     .with_threads(4) // same bytes for every thread count
///     .encode_yuv420(w, h, &y, 32, &u, &v, 16)
///     .expect("encode");
/// assert_eq!(&webp[..4], b"RIFF");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Vp8LossyEncoder {
    qindex: u8,
    effort: u8,
    threads: usize,
}

impl Vp8LossyEncoder {
    /// An encoder at WebP `quality` (`0.0..=100.0`), projected onto the VP8
    /// quantiser index with [`quality_to_qindex`], as
    /// [`make_encoder_with_quality`] does.
    pub fn with_quality(quality: f32) -> Self {
        Self::with_qindex(quality_to_qindex(quality))
    }

    /// An encoder at VP8 quantiser index `qindex` (`0..=127`, lower is
    /// better quality); larger values are treated as `127`. Effort
    /// [`DEFAULT_EFFORT`], one thread.
    pub fn with_qindex(qindex: u8) -> Self {
        Self {
            qindex: qindex.min(127),
            effort: DEFAULT_EFFORT,
            threads: 1,
        }
    }

    /// Builder: encoder effort `0` (fastest) to `6` (slowest, smallest);
    /// larger values are treated as `6`. The levels are the lossy ones of
    /// [`EncodeOptions::with_method`](crate::EncodeOptions::with_method):
    /// `0`, `1`, `2` are reduced mode searches, `3..=4` the exhaustive
    /// search (the default, and the bytes of earlier versions), `5` adds
    /// fitted token probabilities, `6` a loop-filter search.
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort.min(MAX_EFFORT);
        self
    }

    /// Builder: encode on up to `threads` threads (`0` and `1` are both
    /// serial, the default). The bytes are the same for every value.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// The VP8 quantiser index this encoder uses.
    pub fn qindex(&self) -> u8 {
        self.qindex
    }

    /// The effort level this encoder uses (`0..=6`).
    pub fn effort(&self) -> u8 {
        self.effort
    }

    /// The thread budget this encoder uses (at least 1).
    pub fn threads(&self) -> usize {
        self.threads
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
        encode_yuv420_planes(
            width,
            height,
            (y, y_stride),
            (u, uv_stride),
            (v, uv_stride),
            crate::api::LossyTuning::for_qindex(self.qindex, self.effort),
            self.threads,
        )
    }
}

/// Effort of the lossy encoders when none is asked for: the exhaustive
/// mode search, one pass.
pub const DEFAULT_EFFORT: u8 = 4;

/// Highest effort level.
pub const MAX_EFFORT: u8 = 6;

/// Encode three 4:2:0 planes, each `(samples, stride)`, to a simple-lossy
/// `.webp` under `tuning` on up to `threads` threads. The planes are read
/// in place; only a chroma pair with two different strides is repacked.
pub(crate) fn encode_yuv420_planes(
    width: u32,
    height: u32,
    (y, y_stride): (&[u8], usize),
    (u, u_stride): (&[u8], usize),
    (v, v_stride): (&[u8], usize),
    tuning: crate::api::LossyTuning,
    threads: usize,
) -> Result<Vec<u8>, WebpError> {
    if width == 0 || height == 0 || width > MAX_VP8_DIMENSION || height > MAX_VP8_DIMENSION {
        return Err(WebpError::invalid(format!(
            "VP8 picture {width}x{height} is outside 1..={MAX_VP8_DIMENSION} per side"
        )));
    }
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    check_plane("Y", y, y_stride, w, h)?;
    check_plane("U", u, u_stride, cw, ch)?;
    check_plane("V", v, v_stride, cw, ch)?;
    let repacked;
    let (u, v, uv_stride) = if u_stride == v_stride {
        (u, v, u_stride)
    } else {
        let tight = |plane: &[u8], stride: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(cw * ch);
            for row in 0..ch {
                out.extend_from_slice(&plane[row * stride..row * stride + cw]);
            }
            out
        };
        repacked = (tight(u, u_stride), tight(v, v_stride));
        (&repacked.0[..], &repacked.1[..], cw)
    };
    let frame = oxideav_vp8::I420Frame {
        width,
        height,
        y,
        u,
        v,
        y_stride,
        uv_stride,
    };
    crate::vp8l_encode::with_thread_budget(threads, || {
        crate::api::encode_lossy_planes(&frame, None, &crate::WebpMetadata::default(), tuning)
    })
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
    CodecId, CodecParameters, Encoder, Error as CoreError, Frame, MediaType, Packet, PixelFormat,
    Result as CoreResult, time::TimeBase,
};
#[cfg(feature = "registry")]
use std::collections::VecDeque;

/// Typed options of the registered `webp_vp8` encoder — the struct behind
/// the `CodecParameters::options` bag [`make_encoder_from_param_options`]
/// parses, and the argument of [`make_encoder_with_options`].
///
/// | key | kind | default | meaning |
/// |---|---|---|---|
/// | `effort` | u32 | `4` | encoder effort `0` (fastest) ..= `6` (smallest); see [`EncodeOptions::with_method`](crate::EncodeOptions::with_method) |
/// | `threads` | u32 | `0` | thread budget; `0` follows the `ExecutionContext` (serial when none is announced) |
/// | `qindex` | u32 | `32` | RFC 6386 §9.6 quantiser index `0..=127`, lower is better |
/// | `quality` | f32 | `-1` | `0..=100`, higher is better; when ≥ 0 it replaces `qindex` and also sets the trellis strength, giving the bytes of [`EncodeOptions::with_quality`](crate::EncodeOptions::with_quality) on the same planes |
///
/// The output bytes never depend on `threads`.
#[cfg(feature = "registry")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct WebpVp8EncoderOptions {
    /// Encoder effort, `0..=6`.
    pub effort: u8,
    /// Thread budget; `0` follows the announced `ExecutionContext`.
    pub threads: usize,
    /// Quantiser index `0..=127` (ignored when `quality` is set).
    pub qindex: u8,
    /// `0..=100` quality; `None` uses `qindex`.
    pub quality: Option<f32>,
}

#[cfg(feature = "registry")]
impl Default for WebpVp8EncoderOptions {
    fn default() -> Self {
        Self {
            effort: DEFAULT_EFFORT,
            threads: 0,
            qindex: DEFAULT_QINDEX,
            quality: None,
        }
    }
}

/// Quantiser index of [`make_encoder`] and of the registered encoder when
/// neither `qindex` nor `quality` is given.
#[cfg(feature = "registry")]
const DEFAULT_QINDEX: u8 = 32;

#[cfg(feature = "registry")]
impl WebpVp8EncoderOptions {
    /// The defaults (see the table on the type).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `effort` (values above `6` are treated as `6`).
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort.min(MAX_EFFORT);
        self
    }

    /// Set `threads`.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Set `qindex`.
    pub fn with_qindex(mut self, qindex: u8) -> Self {
        self.qindex = qindex;
        self
    }

    /// Set `quality`.
    pub fn with_quality(mut self, quality: Option<f32>) -> Self {
        self.quality = quality;
        self
    }
}

#[cfg(feature = "registry")]
impl oxideav_core::CodecOptionsStruct for WebpVp8EncoderOptions {
    const SCHEMA: &'static [oxideav_core::OptionField] = &[
        oxideav_core::OptionField {
            name: "effort",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(DEFAULT_EFFORT as u32),
            help: "encoder effort: 0 fastest .. 6 smallest (0-2 reduced mode search, 3-4 exhaustive, 5 fitted token probabilities, 6 loop-filter search)",
        },
        oxideav_core::OptionField {
            name: "threads",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(0),
            help: "thread budget (0 = follow the execution context)",
        },
        oxideav_core::OptionField {
            name: "qindex",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(DEFAULT_QINDEX as u32),
            help: "quantiser index 0..=127 (lower = better)",
        },
        oxideav_core::OptionField {
            name: "quality",
            kind: oxideav_core::OptionKind::F32,
            default: oxideav_core::OptionValue::F32(-1.0),
            help: "quality 0..=100 (overrides qindex; negative = unset)",
        },
        oxideav_core::OptionField {
            name: "loop_count",
            kind: oxideav_core::OptionKind::String,
            default: oxideav_core::OptionValue::String(String::new()),
            help: "read by the webp muxer from the same stream parameters; the encoder ignores it",
        },
        oxideav_core::OptionField {
            name: "background_color",
            kind: oxideav_core::OptionKind::String,
            default: oxideav_core::OptionValue::String(String::new()),
            help: "read by the webp muxer from the same stream parameters; the encoder ignores it",
        },
    ];

    fn apply(&mut self, key: &str, value: &oxideav_core::OptionValue) -> CoreResult<()> {
        let bounded = |v: u32, max: u32| {
            if v > max {
                Err(CoreError::invalid(format!(
                    "webp_vp8 encoder: option '{key}' out of range (0..={max}), got {v}"
                )))
            } else {
                Ok(v as u8)
            }
        };
        match key {
            "effort" => self.effort = bounded(value.as_u32()?, u32::from(MAX_EFFORT))?,
            "threads" => self.threads = value.as_u32()? as usize,
            "qindex" => self.qindex = bounded(value.as_u32()?, 127)?,
            "quality" => {
                let q = value.as_f32()?;
                self.quality = (q >= 0.0).then_some(q);
            }
            // The muxer's keys ride in the same options bag.
            "loop_count" | "background_color" => {}
            other => {
                return Err(CoreError::invalid(format!(
                    "webp_vp8 encoder: unknown option '{other}'"
                )));
            }
        }
        Ok(())
    }
}

/// Build a `Box<dyn Encoder>` for the published `"webp_vp8"` codec id at
/// the default quantiser index (`32`): `Yuv420P` frames in, one
/// simple-lossy `.webp` packet per frame out.
#[cfg(feature = "registry")]
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, DEFAULT_QINDEX)
}

/// `make_encoder` plus a `0.0..=100.0` quality knob (higher is better),
/// projected onto the quantiser index with [`quality_to_qindex`]; the
/// trellis keeps its default strength, as in earlier versions. (The
/// `quality` *option* of [`WebpVp8EncoderOptions`] also tunes the
/// trellis, like [`EncodeOptions::with_quality`](crate::EncodeOptions::with_quality).)
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality(
    params: &CodecParameters,
    quality: f32,
) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, quality_to_qindex(quality))
}

/// `make_encoder` plus an explicit `qindex` (`0..=127`, lower = better).
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex(
    params: &CodecParameters,
    qindex: u8,
) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_options(
        params,
        &WebpVp8EncoderOptions::default().with_qindex(qindex),
    )
}

/// The registry's `webp_vp8` encoder factory: parse `params.options`
/// (strictly — an unknown key or a malformed value is an error) into
/// [`WebpVp8EncoderOptions`] and build the encoder from it. An empty bag
/// gives the same encoder as [`make_encoder`].
#[cfg(feature = "registry")]
pub fn make_encoder_from_param_options(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    let options = oxideav_core::parse_options::<WebpVp8EncoderOptions>(&params.options)?;
    make_encoder_with_options(params, &options)
}

/// Build the `webp_vp8` encoder from typed [`WebpVp8EncoderOptions`] —
/// the direct twin of the registered factory.
#[cfg(feature = "registry")]
pub fn make_encoder_with_options(
    params: &CodecParameters,
    options: &WebpVp8EncoderOptions,
) -> CoreResult<Box<dyn Encoder>> {
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
    if width > MAX_VP8_DIMENSION || height > MAX_VP8_DIMENSION {
        return Err(CoreError::invalid(format!(
            "webp_vp8 encoder: width/height exceed the VP8 14-bit size fields (max {MAX_VP8_DIMENSION})"
        )));
    }
    if params
        .pixel_format
        .is_some_and(|format| format != PixelFormat::Yuv420P)
    {
        return Err(CoreError::unsupported(
            "webp_vp8 encoder: only PixelFormat::Yuv420P is supported",
        ));
    }
    if options.qindex > 127 {
        return Err(CoreError::invalid(
            "webp_vp8 encoder: qindex out of range (must be 0..=127)",
        ));
    }
    let effort = options.effort.min(MAX_EFFORT);
    let tuning = match options.quality {
        Some(quality) => crate::api::LossyTuning::for_quality(quality, effort),
        None => crate::api::LossyTuning::for_qindex(options.qindex, effort),
    };
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(crate::CODEC_ID_VP8);
    output_params.width = Some(width);
    output_params.height = Some(height);
    let time_base = params
        .frame_rate
        .map_or(TimeBase::new(1, 1_000), |r| TimeBase::new(r.den, r.num));
    Ok(Box::new(WebpVp8LossyEncoder {
        width,
        height,
        tuning,
        option_threads: options.threads,
        context_threads: 1,
        output_params,
        time_base,
        pending: VecDeque::new(),
        flushed: false,
    }))
}

/// `make_encoder` plus an explicit `qindex` and a [`Vp8FreqDeltas`]
/// record of per-band quantiser deltas.
///
/// The `deltas` argument is accepted for signature compatibility and not
/// applied; the qindex is honoured.
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex_and_freq_deltas(
    params: &CodecParameters,
    qindex: u8,
    _deltas: Vp8FreqDeltas,
) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, qindex)
}

/// `make_encoder` plus a `0.0..=100.0` quality knob and a
/// [`Vp8FreqDeltas`] record of per-band quantiser deltas.
///
/// The `deltas` argument is accepted for signature compatibility and not
/// applied — see [`make_encoder_with_qindex_and_freq_deltas`]. The
/// quality is honoured.
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality_and_freq_deltas(
    params: &CodecParameters,
    quality: f32,
    _deltas: Vp8FreqDeltas,
) -> CoreResult<Box<dyn Encoder>> {
    make_encoder_with_quality(params, quality)
}

/// The `webp_vp8` framework encoder: each `Yuv420P` frame becomes one
/// simple-lossy `.webp` packet, encoded from the frame's planes in place.
#[cfg(feature = "registry")]
pub(crate) struct WebpVp8LossyEncoder {
    width: u32,
    height: u32,
    tuning: crate::api::LossyTuning,
    /// Explicit thread budget from the options (`0` = none given).
    option_threads: usize,
    /// Budget announced through `set_execution_context` (`1` until one
    /// is).
    context_threads: usize,
    output_params: CodecParameters,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    flushed: bool,
}

#[cfg(feature = "registry")]
impl std::fmt::Debug for WebpVp8LossyEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebpVp8LossyEncoder")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("tuning", &self.tuning)
            .field("pending", &self.pending.len())
            .field("flushed", &self.flushed)
            .finish()
    }
}

#[cfg(feature = "registry")]
impl WebpVp8LossyEncoder {
    /// Follow a [`Frame::StreamParams`] event. Every packet is a key
    /// frame restating its size, so a new size simply applies from the
    /// next frame; a description that is not `Yuv420P` video inside the
    /// RFC 6386 §9.1 size fields cannot be followed.
    fn reconfigure(&mut self, params: &CodecParameters) -> CoreResult<()> {
        if params.media_type != MediaType::Video {
            return Err(CoreError::unsupported(
                "webp_vp8 encoder: stream-params event does not describe video",
            ));
        }
        if params
            .pixel_format
            .is_some_and(|format| format != PixelFormat::Yuv420P)
        {
            return Err(CoreError::unsupported(
                "webp_vp8 encoder: cannot switch away from PixelFormat::Yuv420P mid-stream",
            ));
        }
        let width = params.width.unwrap_or(self.width);
        let height = params.height.unwrap_or(self.height);
        if width == 0 || height == 0 || width > MAX_VP8_DIMENSION || height > MAX_VP8_DIMENSION {
            return Err(CoreError::unsupported(format!(
                "webp_vp8 encoder: stream-params size outside 1..={MAX_VP8_DIMENSION} per side"
            )));
        }
        (self.width, self.height) = (width, height);
        self.output_params.width = Some(width);
        self.output_params.height = Some(height);
        Ok(())
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
        let video = match frame {
            Frame::Video(video) => video,
            Frame::StreamParams(params) => return self.reconfigure(params),
            _ => return Err(CoreError::invalid("webp_vp8 encoder: video frames only")),
        };
        let [y, u, v, ..] = video.image_planes() else {
            return Err(CoreError::invalid(
                "webp_vp8 encoder: a frame must carry 3 planes (Y/U/V)",
            ));
        };
        let threads = match self.option_threads {
            0 => self.context_threads,
            n => n,
        };
        let webp_bytes = encode_yuv420_planes(
            self.width,
            self.height,
            (&y.data, y.stride),
            (&u.data, u.stride),
            (&v.data, v.stride),
            self.tuning,
            threads,
        )
        .map_err(|e| CoreError::invalid(format!("webp_vp8 encoder: {e}")))?;
        let mut pkt = Packet::new(0, self.time_base, webp_bytes);
        pkt.pts = video.pts;
        pkt.dts = video.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    /// Record the announced thread budget; it applies unless the
    /// `threads` option named one. The bytes do not depend on it.
    fn set_execution_context(&mut self, ctx: &oxideav_core::ExecutionContext) {
        self.context_threads = ctx.threads.max(1);
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
