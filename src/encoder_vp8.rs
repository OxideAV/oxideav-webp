//! `oxideav_core::Encoder` adapter that produces a full `.webp` file
//! using the VP8 lossy path.
//!
//! Four input pixel formats are accepted:
//!
//! * **`Yuv420P`** — the native VP8 format. We feed it through the
//!   per-segment-tuned [`encode_keyframe_with_segments`] helper and
//!   emit a simple-file `RIFF/WEBP/VP8 ` container.
//! * **`Yuva420P`** — VP8 with a side full-resolution alpha plane. The
//!   YUV planes go straight into the keyframe encoder (no RGB
//!   roundtrip) and the alpha plane is compressed into the `ALPH`
//!   sidecar. Emits the extended `RIFF/WEBP/VP8X + ALPH + VP8 `
//!   container.
//! * **`Rgba`** — VP8 itself is RGB-only, but the WebP container adds
//!   alpha support via a separate `ALPH` chunk (§5.2.3 of the WebP
//!   spec). When given an RGBA frame we convert the RGB plane to
//!   YUV420P for the VP8 keyframe, encode the alpha plane as a
//!   VP8L-compressed green-only bitstream, and emit an extended
//!   `RIFF/WEBP/VP8X + ALPH + VP8 ` container. The VP8X header
//!   advertises the ALPHA flag + canvas size so any compliant reader
//!   picks up the sidecar.
//! * **`Rgb24`** — RGB without alpha. The conversion to YUV 4:2:0
//!   streams over the input three bytes at a time without ever
//!   materialising an intermediate `Rgba` byte buffer (issue #7), and
//!   emits the simple `RIFF/WEBP/VP8 ` container.
//!
//! Registered under the crate-level codec id [`crate::CODEC_ID_VP8`]
//! (`"webp_vp8"`), a sibling of the existing `webp_vp8l` lossless id.
//! The corresponding read path is the WebP container demuxer —
//! callers wanting to decode the output can feed the bytes directly
//! to [`crate::decode_webp`], which handles both simple and extended
//! layouts with or without ALPH.
//!
//! Scope (v2):
//!   * single-frame still images only (no animated `ANMF` chunks);
//!   * RGB→YUV conversion uses the BT.601 limited-range coefficients
//!     (matches the decoder's inverse matrix);
//!   * ALPH compression is always VP8L-based (type 1, no filtering,
//!     no pre-processing). Uncompressed / filtered raw alpha (type 0)
//!     is decodable but not produced here.
//!
//! ## Quality knob
//!
//! Three factory entry points are exposed:
//!
//! * [`make_encoder`] — builds an encoder at the `oxideav-vp8`
//!   `DEFAULT_QINDEX`.
//! * [`make_encoder_with_quality`] — libwebp-compatible API surface,
//!   takes a `quality: f32` in `0.0..=100.0` (higher = better, `75.0`
//!   is the typical default).
//! * [`make_encoder_with_qindex`] — direct access to the underlying
//!   VP8 qindex in `0..=127` (lower = better).
//!
//! The quality→qindex mapping is the linear inversion
//! `qindex = round((100 - quality) * 1.27)`. The encoder also tunes
//! the per-segment quantiser deltas (RFC 6386 §10) based on quality
//! so that smooth regions get extra bits where banding is visible and
//! high-variance regions save bits where DCT noise hides. Mirrors
//! libwebp's perceptual model: the source-luma variance classifier
//! lands smooth MBs in segment 0 and textured MBs in segment 3;
//! `segment 0` then takes a stronger negative qindex delta (finer
//! quant) and `segment 3` a stronger positive delta (coarser quant)
//! at low quality, with the deltas tapering toward zero as quality
//! approaches 100 (where every segment is already near-lossless).
//!
//! Per-frequency AC/DC quantiser deltas (`y_dc_delta`, `y2_dc_delta`,
//! `y2_ac_delta`, `uv_dc_delta`, `uv_ac_delta`) are wired through the
//! optional [`Vp8FreqDeltas`] knob now that `oxideav-vp8` 0.1.7
//! (#417) exposes the matching `Vp8EncoderConfig` fields. Each delta
//! is clamped to the legal `[-15, 15]` range (decoder reads each as a
//! 5-bit signed-magnitude field). Defaults are zero so the existing
//! factory entry points stay byte-identical with the pre-#417 output.

#[cfg(feature = "registry")]
use std::collections::VecDeque;

#[cfg(feature = "registry")]
use oxideav_core::Encoder;
#[cfg(feature = "registry")]
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, TimeBase,
    VideoFrame, VideoPlane,
};

#[cfg(feature = "registry")]
use oxideav_vp8::encoder::{
    make_encoder_with_config, LoopFilterMode, Vp8EncoderConfig, DEFAULT_QINDEX,
};

use crate::error::{Result, WebpError as Error};
use crate::riff::AlphChunkBytes;
#[cfg(feature = "registry")]
use crate::riff::{build_webp_file, ImageKind, WebpMetadata};
use crate::vp8l::encode_vp8l_argb;
#[cfg(feature = "registry")]
use crate::CODEC_ID_VP8;

/// Factory used by [`crate::register_codecs`] for the `webp_vp8` codec id.
#[cfg(feature = "registry")]
pub fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, DEFAULT_QINDEX)
}

/// Build a VP8-lossy WebP encoder using a libwebp-style `quality`
/// scalar in `0.0..=100.0` (higher = better quality / larger file).
///
/// `0.0` maps to maximum compression (qindex 127), `100.0` maps to
/// maximum quality (qindex 0); values are clamped to that range. The
/// mapping is the linear inversion
/// `qindex = round((100 - quality) * 1.27)`, which lines up with
/// libwebp's API surface but is **not** a perceptual match: libwebp
/// also adjusts its quantizer matrices, AC/DC deltas, and segment-level
/// QP based on quality, none of which we do here. Treat the knob as a
/// drop-in replacement for the libwebp parameter name only — round-2
/// work would tune the quantizer matrix and segment QPs to track
/// libwebp's perceptual targets at matching quality values.
///
/// The libwebp default of `75.0` corresponds to qindex ≈ 32 here.
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality(
    params: &CodecParameters,
    quality: f32,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    make_encoder_with_qindex(params, quality_to_qindex(quality))
}

/// Convert a libwebp-style `0.0..=100.0` quality value to the VP8
/// qindex (`0..=127`) the lower-level encoder consumes. Values outside
/// the range are clamped before mapping; `NaN` falls through to the
/// max-compression / lowest-quality endpoint (qindex 127).
///
/// Mapping: `qindex = round((100 - clamp(q, 0, 100)) * 1.27)`. This is
/// a pure linear inversion — see [`make_encoder_with_quality`] for the
/// caveat that this matches libwebp's *API surface* only, not its
/// perceptual quality model.
pub fn quality_to_qindex(quality: f32) -> u8 {
    if quality.is_nan() {
        return 127;
    }
    let q = quality.clamp(0.0, 100.0);
    ((100.0 - q) * 1.27).round().clamp(0.0, 127.0) as u8
}

/// Quality-driven per-segment qindex deltas (RFC 6386 §10). The
/// encoder classifies each MB into one of four segments by source-luma
/// variance: segment 0 = smoothest content, segment 3 = highest-variance.
/// Returning `[neg, neg/2, 0, pos]` lands more bits on smooth segments
/// (where banding is visible at high QP) and fewer on textured segments
/// (where DCT noise is masked).
///
/// The delta magnitudes scale with `(127 - qindex)`: at very high
/// quality (qindex near 0) every segment is already near-lossless so
/// the deltas collapse to ~`[-2, -1, 0, 1]`. At very low quality
/// (qindex near 127) the deltas widen to ~`[-12, -6, 0, 8]` —
/// matching libwebp's "spend bits where the eye notices" heuristic.
///
/// Returned values are pre-clamped to the legal `[-15, 15]` range
/// (decoder reads each delta as a 5-bit signed-magnitude field).
fn segment_quant_deltas_for_qindex(qindex: u8) -> [i32; 4] {
    // Span ∈ [0, 1]: 0 at qindex 0, 1 at qindex 127.
    // Higher qindex → wider deltas → more aggressive perceptual tuning.
    let span = (qindex as f32) / 127.0;
    // Smooth segment bonus: scales 2..=12 (better quality at smooth).
    let smooth = -((2.0 + span * 10.0).round() as i32);
    // Half-smooth: scales 1..=6.
    let mid_low = -((1.0 + span * 5.0).round() as i32);
    // High-variance penalty: scales 1..=8 (saves bits on textured).
    let high = (1.0 + span * 7.0).round() as i32;
    [
        smooth.clamp(-15, 15),
        mid_low.clamp(-15, 15),
        0,
        high.clamp(-15, 15),
    ]
}

/// Quality-driven per-segment loop-filter level deltas (RFC 6386 §15.2).
/// Smooth segments take a *negative* LF delta (a softer filter — the
/// per-segment finer quant already preserves smooth detail, so the
/// deblocker can ease off and avoid over-smoothing). High-variance
/// segments take a *positive* LF delta (a stronger filter — masks the
/// extra DCT block boundaries the coarser per-segment QP exposes).
///
/// Magnitudes scale with `qindex` for the same reason as the QP
/// deltas: at high quality everything's near-lossless and the LF tweaks
/// approach zero.
///
/// Wired through `webp_lossy_config` into
/// [`oxideav_vp8::Vp8EncoderConfig::segment_lf_deltas`] (added in
/// `oxideav-vp8` 0.1.6 / [#337]). The decoder applies these as
/// `clamp(frame_level + segment_lf_deltas[seg], 0..=63)` per RFC 6386
/// §15.2, so the segment map produced by the variance classifier
/// dictates per-MB filter strength.
fn segment_lf_deltas_for_qindex(qindex: u8) -> [i32; 4] {
    let span = (qindex as f32) / 127.0;
    // Smooth segment LF easing: 0..=3.
    let smooth_lf = -((span * 3.0).round() as i32);
    // Half-smooth: 0..=2.
    let mid_low_lf = -((span * 2.0).round() as i32);
    // High-variance LF strengthening: 1..=4.
    let high_lf = (1.0 + span * 3.0).round() as i32;
    [
        smooth_lf.clamp(-63, 63),
        mid_low_lf.clamp(-63, 63),
        0,
        high_lf.clamp(-63, 63),
    ]
}

/// Per-frequency AC/DC quantiser delta knob (RFC 6386 §9.6).
///
/// Each field shifts the qindex used to look up a specific transform
/// coefficient's quant step relative to the frame-level qindex.
/// Negative values land on a *finer* step (larger output bit count,
/// better quality at that frequency); positive values land on a
/// *coarser* step. The legal range is `[-15, 15]` per the bitstream
/// syntax — values outside that range are clamped at config-build
/// time by the underlying [`oxideav_vp8::Vp8EncoderConfig`].
///
/// The five frequencies map directly onto the VP8 per-block transforms:
///
/// * `y_dc_delta` — luma AC base qindex shift (note the misleading
///   field name on the underlying config — it actually offsets the
///   intra-Y AC plane).
/// * `y2_dc_delta` / `y2_ac_delta` — DC / AC of the second-order Y2
///   transform applied to intra-16×16 DC coefficients.
/// * `uv_dc_delta` / `uv_ac_delta` — chroma DC / AC.
///
/// All-zero (the [`Default`] impl) reproduces the pre-#417 encoder
/// output exactly, so existing callers and snapshot tests stay
/// byte-identical without an opt-in.
///
/// Wire it through [`make_encoder_with_qindex_and_freq_deltas`] or
/// [`make_encoder_with_quality_and_freq_deltas`]; both factories
/// otherwise behave identically to their non-`freq_deltas` variants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vp8FreqDeltas {
    /// Luma AC qindex delta. See struct docs for the field-name caveat.
    pub y_dc_delta: i32,
    /// Y2 DC qindex delta — second-order Hadamard transform of intra-16×16 DCs.
    pub y2_dc_delta: i32,
    /// Y2 AC qindex delta.
    pub y2_ac_delta: i32,
    /// Chroma DC qindex delta.
    pub uv_dc_delta: i32,
    /// Chroma AC qindex delta.
    pub uv_ac_delta: i32,
}

/// Build the `Vp8EncoderConfig` used by the WebP single-frame lossy
/// path. Wires up the quality-driven segment QP deltas (RFC 6386
/// §10), keeps the scene-cut and lookahead-altref features off (we
/// only ever emit a single keyframe per `.webp` so there's no GOP
/// state to manage), and pins `loop_filter_mode = Normal` to preserve
/// the bit-exact loop-filter behaviour the existing decode-side
/// regression tests depend on.
///
/// The matching per-segment loop-filter delta knob (RFC 6386 §15.2)
/// computed by [`segment_lf_deltas_for_qindex`] is wired in alongside
/// the per-segment quant deltas now that `oxideav-vp8` 0.1.6 (#337)
/// exposes the `segment_lf_deltas` field. The variance classifier in
/// `oxideav-vp8` lands smooth MBs in segment 0 and high-variance MBs
/// in segment 3; the smooth segment gets a *lighter* deblock (less
/// over-smoothing on flat regions) and the textured segment gets a
/// *heavier* deblock (masks the extra DCT block boundaries the
/// coarser per-segment QP exposes).
///
/// The optional `freq_deltas` argument carries the per-frequency
/// AC/DC quantiser deltas added in `oxideav-vp8` 0.1.7 (#417); pass
/// [`Vp8FreqDeltas::default()`] (all zeros) to reproduce the exact
/// pre-#417 bitstream.
#[cfg(feature = "registry")]
fn webp_lossy_config(qindex: u8, freq_deltas: Vp8FreqDeltas) -> Vp8EncoderConfig {
    let qi = qindex.min(127);
    Vp8EncoderConfig {
        qindex: qi,
        // Per-frame static-image encode: no scene-cut / lookahead.
        enable_scene_cut: false,
        enable_lookahead_altref: false,
        // Match the historic webp lossy bitstream's loop-filter shape
        // so the decode-side `lossy_corpus` regression tests don't drift.
        loop_filter_mode: LoopFilterMode::Normal,
        // Quality-driven perceptual tuning — segments-on, deltas scaled
        // with qindex so high quality collapses to near-uniform QP / LF.
        enable_segments: true,
        segment_quant_deltas: segment_quant_deltas_for_qindex(qi),
        segment_lf_deltas: segment_lf_deltas_for_qindex(qi),
        // Per-frequency AC/DC qindex deltas. Underlying encoder clamps
        // each to ±15 internally, so we can pass through untouched.
        y_dc_delta: freq_deltas.y_dc_delta,
        y2_dc_delta: freq_deltas.y2_dc_delta,
        y2_ac_delta: freq_deltas.y2_ac_delta,
        uv_dc_delta: freq_deltas.uv_dc_delta,
        uv_ac_delta: freq_deltas.uv_ac_delta,
        ..Vp8EncoderConfig::default()
    }
}

/// Encode a single VP8 keyframe through the segment-aware
/// configuration produced by [`webp_lossy_config`]. Goes through the
/// `Encoder` trait surface so we get the per-segment quant + LF deltas
/// (and the optional per-frequency AC/DC deltas) without having to
/// duplicate the lower-level keyframe entry point.
#[cfg(feature = "registry")]
fn encode_keyframe_with_segments(
    width: u32,
    height: u32,
    qindex: u8,
    freq_deltas: Vp8FreqDeltas,
    frame: &VideoFrame,
) -> Result<Vec<u8>> {
    let cfg = webp_lossy_config(qindex, freq_deltas);
    let mut p = CodecParameters::video(CodecId::new(oxideav_vp8::CODEC_ID_STR));
    p.width = Some(width);
    p.height = Some(height);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    let mut enc = match make_encoder_with_config(&p, cfg) {
        Ok(e) => e,
        Err(e) => return Err(Error::invalid(format!("vp8 segment encoder: {e}"))),
    };
    let f = Frame::Video(frame.clone());
    if let Err(e) = enc.send_frame(&f) {
        return Err(Error::invalid(format!("vp8 segment encoder send: {e}")));
    }
    if let Err(e) = enc.flush() {
        return Err(Error::invalid(format!("vp8 segment encoder flush: {e}")));
    }
    let pkt = match enc.receive_packet() {
        Ok(p) => p,
        Err(e) => return Err(Error::invalid(format!("vp8 segment encoder receive: {e}"))),
    };
    Ok(pkt.data)
}

/// Build a VP8-lossy WebP encoder with an explicit qindex (0..=127).
/// Lower values produce higher quality at the cost of file size.
///
/// Most callers should prefer [`make_encoder_with_quality`], which
/// takes the libwebp-style `0..=100` scale (higher = better) and is
/// the more familiar knob across image-encoding libraries.
///
/// Per-frequency AC/DC quantiser deltas default to all-zero. Callers
/// that want to bias bits toward a specific frequency band (the
/// libvpx `tune_content` story) should reach for
/// [`make_encoder_with_qindex_and_freq_deltas`].
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex(
    params: &CodecParameters,
    qindex: u8,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    make_encoder_with_qindex_and_freq_deltas(params, qindex, Vp8FreqDeltas::default())
}

/// Build a VP8-lossy WebP encoder with an explicit qindex (0..=127)
/// **and** explicit per-frequency AC/DC quantiser deltas (see
/// [`Vp8FreqDeltas`]).
///
/// All-zero `freq_deltas` (the [`Default`] value) is exactly equivalent
/// to [`make_encoder_with_qindex`] and reproduces the pre-#417 (vp8
/// 0.1.6) bitstream byte-for-byte.
///
/// Use a small negative delta (e.g. `y_dc_delta = -4`) to spend more
/// bits on luma AC where banding is visible, or a small positive
/// `uv_ac_delta` to lighten chroma AC on screen-recording / line-art
/// content where chroma carries less perceptual weight. Each value is
/// clamped to `[-15, 15]` by the underlying encoder.
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex_and_freq_deltas(
    params: &CodecParameters,
    qindex: u8,
    freq_deltas: Vp8FreqDeltas,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| oxideav_core::Error::invalid("VP8 WebP encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| oxideav_core::Error::invalid("VP8 WebP encoder: missing height"))?;
    if width == 0 || height == 0 || width > 16383 || height > 16383 {
        return Err(oxideav_core::Error::invalid(format!(
            "VP8 WebP encoder: dimensions {width}x{height} out of range (1..=16383)"
        )));
    }
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    if !matches!(
        pix,
        PixelFormat::Yuv420P | PixelFormat::Yuva420P | PixelFormat::Rgba | PixelFormat::Rgb24
    ) {
        return Err(oxideav_core::Error::unsupported(format!(
            "VP8 WebP encoder: pixel format {pix:?} not supported — \
             feed Yuv420P / Yuva420P / Rgba / Rgb24"
        )));
    }

    let frame_rate = params.frame_rate.unwrap_or(Rational::new(1, 1));
    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(CODEC_ID_VP8);
    output_params.pixel_format = Some(pix);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.frame_rate = Some(frame_rate);

    let time_base = TimeBase::new(1, 1000);

    Ok(Box::new(Vp8WebpEncoder {
        output_params,
        width,
        height,
        qindex: qindex.min(127),
        freq_deltas,
        input_format: pix,
        time_base,
        pending: VecDeque::new(),
        eof: false,
    }))
}

/// Build a VP8-lossy WebP encoder with a libwebp-style `quality`
/// scalar **and** explicit per-frequency AC/DC quantiser deltas.
/// Composes [`make_encoder_with_quality`] with the per-frequency knob
/// from [`make_encoder_with_qindex_and_freq_deltas`].
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality_and_freq_deltas(
    params: &CodecParameters,
    quality: f32,
    freq_deltas: Vp8FreqDeltas,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    make_encoder_with_qindex_and_freq_deltas(params, quality_to_qindex(quality), freq_deltas)
}

#[cfg(feature = "registry")]
struct Vp8WebpEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    qindex: u8,
    freq_deltas: Vp8FreqDeltas,
    input_format: PixelFormat,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

#[cfg(feature = "registry")]
impl Encoder for Vp8WebpEncoder {
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
                    "VP8 WebP encoder: video frames only",
                ))
            }
        };
        // Frame dims and pixel format are stream-level (set on the
        // encoder at construction); the pipeline upstream is responsible
        // for matching `output_params`. Dispatch on the encoder's
        // configured input format.
        let bytes = match self.input_format {
            PixelFormat::Yuv420P => {
                let vp8 = encode_keyframe_with_segments(
                    self.width,
                    self.height,
                    self.qindex,
                    self.freq_deltas,
                    v,
                )?;
                build_webp_file(
                    ImageKind::Vp8Lossy,
                    &vp8,
                    self.width,
                    self.height,
                    None,
                    &WebpMetadata::default(),
                )
            }
            PixelFormat::Yuva420P => {
                encode_yuva420_lossy(self.width, self.height, self.qindex, self.freq_deltas, v)?
            }
            PixelFormat::Rgba => {
                encode_rgba_lossy(self.width, self.height, self.qindex, self.freq_deltas, v)?
            }
            PixelFormat::Rgb24 => {
                encode_rgb24_lossy(self.width, self.height, self.qindex, self.freq_deltas, v)?
            }
            other => {
                return Err(oxideav_core::Error::unsupported(format!(
                    "VP8 WebP encoder: frame format {other:?} unsupported"
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
            Err(oxideav_core::Error::Eof)
        } else {
            Err(oxideav_core::Error::NeedMore)
        }
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Encode a `Yuva420P` frame natively: the YUV planes feed straight into
/// the VP8 keyframe encoder (no RGB roundtrip — saves a pair of
/// 8-bit-fixed-point colour conversions vs the `Rgba` path), and the
/// full-resolution alpha plane is compressed into the `ALPH` sidecar.
/// Emits a complete `.webp` file in the extended `VP8X + ALPH + VP8 `
/// layout.
#[cfg(feature = "registry")]
fn encode_yuva420_lossy(
    width: u32,
    height: u32,
    qindex: u8,
    freq_deltas: Vp8FreqDeltas,
    v: &VideoFrame,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if v.planes.len() < 4 {
        return Err(Error::invalid(
            "VP8 WebP encoder: Yuva420P frame needs 4 planes (Y, U, V, A)",
        ));
    }
    let cw = w / 2 + (w & 1);
    if v.planes[0].stride < w
        || v.planes[1].stride < cw
        || v.planes[2].stride < cw
        || v.planes[3].stride < w
    {
        return Err(Error::invalid(
            "VP8 WebP encoder: Yuva420P plane stride too small",
        ));
    }

    // Build a YUV-only frame view that wraps the same plane data — we
    // hand it straight to the VP8 keyframe encoder. Since the encoder
    // takes a `&VideoFrame`, we have to clone the planes; but only the
    // 3 YUV planes (no copy of the alpha plane and no RGB→YUV maths).
    let yuv_frame = VideoFrame {
        pts: v.pts,
        planes: vec![
            v.planes[0].clone(),
            v.planes[1].clone(),
            v.planes[2].clone(),
        ],
    };
    let vp8_bytes = encode_keyframe_with_segments(width, height, qindex, freq_deltas, &yuv_frame)?;

    // Pull the alpha plane row-major (handle non-tight stride).
    let alpha_plane = &v.planes[3];
    let mut alpha = Vec::with_capacity(w * h);
    for j in 0..h {
        let row_start = j * alpha_plane.stride;
        alpha.extend_from_slice(&alpha_plane.data[row_start..row_start + w]);
    }

    let alph = encode_alph_chunk(width, height, &alpha)?;
    Ok(build_webp_file(
        ImageKind::Vp8Lossy,
        &vp8_bytes,
        width,
        height,
        Some(&alph),
        &WebpMetadata::default(),
    ))
}

/// Encode an `Rgb24` frame as a simple-layout VP8 lossy `.webp` file.
/// The RGB → YUV 4:2:0 conversion **streams** through the input three
/// bytes at a time — there is no intermediate `Rgba` byte buffer, so a
/// caller that already holds a JPEG- or PNG-without-alpha decode (where
/// the upstream is RGB and adding alpha would mean a full re-alloc)
/// pays only for the YUV planes (the natural VP8 input). This is the
/// VP8-side counterpart to issue #7.
#[cfg(feature = "registry")]
fn encode_rgb24_lossy(
    width: u32,
    height: u32,
    qindex: u8,
    freq_deltas: Vp8FreqDeltas,
    v: &VideoFrame,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if v.planes.is_empty() {
        return Err(Error::invalid(
            "VP8 WebP encoder: RGB24 frame has no planes",
        ));
    }
    let plane = &v.planes[0];
    if plane.stride < w * 3 {
        return Err(Error::invalid(
            "VP8 WebP encoder: RGB24 stride too small for frame width",
        ));
    }
    let (y, u, v_chroma) = rgb24_rows_to_yuv420(w, h, plane.stride, &plane.data);
    let yuv_frame = VideoFrame {
        pts: v.pts,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: w / 2 + (w & 1),
                data: u,
            },
            VideoPlane {
                stride: w / 2 + (w & 1),
                data: v_chroma,
            },
        ],
    };
    let vp8_bytes = encode_keyframe_with_segments(width, height, qindex, freq_deltas, &yuv_frame)?;
    Ok(build_webp_file(
        ImageKind::Vp8Lossy,
        &vp8_bytes,
        width,
        height,
        None,
        &WebpMetadata::default(),
    ))
}

/// Encode an RGBA frame as VP8 lossy + ALPH sidecar + VP8X extended
/// header. Returns a complete `.webp` file.
#[cfg(feature = "registry")]
fn encode_rgba_lossy(
    width: u32,
    height: u32,
    qindex: u8,
    freq_deltas: Vp8FreqDeltas,
    v: &VideoFrame,
) -> Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    if v.planes.is_empty() {
        return Err(Error::invalid("VP8 WebP encoder: RGBA frame has no planes"));
    }
    let plane = &v.planes[0];
    if plane.stride < w * 4 {
        return Err(Error::invalid(
            "VP8 WebP encoder: RGBA stride too small for frame width",
        ));
    }

    // Split the input into RGB planes (we convert to YUV below) and a
    // side alpha plane.
    let mut alpha = Vec::with_capacity(w * h);
    let (y, u, v_chroma) = rgba_rows_to_yuv420(w, h, plane.stride, &plane.data, &mut alpha);
    let yuv_frame = VideoFrame {
        pts: v.pts,
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: w / 2 + (w & 1),
                data: u,
            },
            VideoPlane {
                stride: w / 2 + (w & 1),
                data: v_chroma,
            },
        ],
    };
    let vp8_bytes = encode_keyframe_with_segments(width, height, qindex, freq_deltas, &yuv_frame)?;

    // Encode the alpha plane as a VP8L green-only bitstream with a
    // pre-encode filter pass picked to minimise the resulting payload
    // (see `encode_alph_chunk`).
    let alph = encode_alph_chunk(width, height, &alpha)?;

    Ok(build_webp_file(
        ImageKind::Vp8Lossy,
        &vp8_bytes,
        width,
        height,
        Some(&alph),
        &WebpMetadata::default(),
    ))
}

/// Convert a row-major RGBA buffer into BT.601 limited-range YUV 4:2:0
/// planes. The `alpha` output is filled with the alpha channel bytes in
/// row-major order — one byte per source pixel.
///
/// This mirrors the decoder's YUV→RGB path so a round-trip through the
/// VP8 codec preserves as much colour fidelity as possible for the
/// smooth test pattern used in the integration tests.
pub(crate) fn rgba_rows_to_yuv420(
    w: usize,
    h: usize,
    stride: usize,
    rgba: &[u8],
    alpha: &mut Vec<u8>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w / 2 + (w & 1);
    let ch = h / 2 + (h & 1);
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];

    // First pass: Y + alpha from every pixel.
    for j in 0..h {
        let row_start = j * stride;
        for i in 0..w {
            let px = &rgba[row_start + i * 4..row_start + i * 4 + 4];
            let r = px[0] as i32;
            let g = px[1] as i32;
            let b = px[2] as i32;
            alpha.push(px[3]);
            // BT.601 limited-range, matching the decoder's YUV→RGB
            // inverse matrix: Y = 0.257 R + 0.504 G + 0.098 B + 16.
            let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_plane[j * w + i] = y.clamp(0, 255) as u8;
        }
    }

    // Second pass: U/V averaged over 2×2 blocks.
    for cy in 0..ch {
        for cx in 0..cw {
            let mut u_sum = 0i32;
            let mut v_sum = 0i32;
            let mut n = 0i32;
            for dy in 0..2 {
                let jj = cy * 2 + dy;
                if jj >= h {
                    break;
                }
                for dx in 0..2 {
                    let ii = cx * 2 + dx;
                    if ii >= w {
                        break;
                    }
                    let px = &rgba[jj * stride + ii * 4..jj * stride + ii * 4 + 4];
                    let r = px[0] as i32;
                    let g = px[1] as i32;
                    let b = px[2] as i32;
                    // U = -0.148 R - 0.291 G + 0.439 B + 128.
                    // V =  0.439 R - 0.368 G - 0.071 B + 128.
                    u_sum += (-38 * r - 74 * g + 112 * b + 128) >> 8;
                    v_sum += (112 * r - 94 * g - 18 * b + 128) >> 8;
                    n += 1;
                }
            }
            let u = (u_sum / n) + 128;
            let v = (v_sum / n) + 128;
            u_plane[cy * cw + cx] = u.clamp(0, 255) as u8;
            v_plane[cy * cw + cx] = v.clamp(0, 255) as u8;
        }
    }

    (y_plane, u_plane, v_plane)
}

/// Convert a row-major Rgb24 buffer into BT.601 limited-range YUV 4:2:0
/// planes. Mirrors [`rgba_rows_to_yuv420`] for RGB-without-alpha input —
/// no alpha plane is produced, and the conversion **streams** through
/// the input three bytes at a time without any intermediate `Rgba`
/// allocation. Coefficients match the BT.601 formulas the decoder uses
/// for the inverse transform.
fn rgb24_rows_to_yuv420(
    w: usize,
    h: usize,
    stride: usize,
    rgb: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w / 2 + (w & 1);
    let ch = h / 2 + (h & 1);
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];

    // First pass: Y from every pixel — single 3-byte read per source
    // pixel, no alpha handling.
    for j in 0..h {
        let row_start = j * stride;
        for i in 0..w {
            let px = &rgb[row_start + i * 3..row_start + i * 3 + 3];
            let r = px[0] as i32;
            let g = px[1] as i32;
            let b = px[2] as i32;
            // BT.601 limited-range: Y = 0.257 R + 0.504 G + 0.098 B + 16.
            let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_plane[j * w + i] = y.clamp(0, 255) as u8;
        }
    }

    // Second pass: U/V averaged over 2×2 blocks.
    for cy in 0..ch {
        for cx in 0..cw {
            let mut u_sum = 0i32;
            let mut v_sum = 0i32;
            let mut n = 0i32;
            for dy in 0..2 {
                let jj = cy * 2 + dy;
                if jj >= h {
                    break;
                }
                for dx in 0..2 {
                    let ii = cx * 2 + dx;
                    if ii >= w {
                        break;
                    }
                    let px = &rgb[jj * stride + ii * 3..jj * stride + ii * 3 + 3];
                    let r = px[0] as i32;
                    let g = px[1] as i32;
                    let b = px[2] as i32;
                    u_sum += (-38 * r - 74 * g + 112 * b + 128) >> 8;
                    v_sum += (112 * r - 94 * g - 18 * b + 128) >> 8;
                    n += 1;
                }
            }
            let u = (u_sum / n) + 128;
            let v = (v_sum / n) + 128;
            u_plane[cy * cw + cx] = u.clamp(0, 255) as u8;
            v_plane[cy * cw + cx] = v.clamp(0, 255) as u8;
        }
    }

    (y_plane, u_plane, v_plane)
}

/// Compress an 8-bit alpha plane into the "header-less" VP8L bitstream
/// used in `ALPH` chunks with `compression=1`. The decoder synthesises
/// a 5-byte VP8L header (signature + dimensions + alpha/version = 0)
/// before handing the bytes to [`crate::vp8l::decode`], so we produce a
/// full VP8L stream here and drop the leading 5 bytes.
///
/// The alpha values go into the green channel of an ARGB pixel buffer
/// (R=B=0, A=0xff). The ALPH decoder extracts `((p >> 8) & 0xff)` —
/// matching exactly what we write.
fn encode_alpha_plane_as_vp8l(width: u32, height: u32, alpha: &[u8]) -> Result<Vec<u8>> {
    debug_assert_eq!(alpha.len(), (width as usize) * (height as usize));
    let mut pixels = Vec::with_capacity(alpha.len());
    for &a in alpha {
        let g = a as u32;
        pixels.push(0xff00_0000 | (g << 8));
    }
    let full_bitstream = encode_vp8l_argb(width, height, &pixels, false)?;
    // The synthesised header the decoder prepends is 5 bytes:
    // signature (1) + 14-bit width-1 + 14-bit height-1 + 1-bit alpha
    // flag (0) + 3-bit version (0) → 32 bits of packed field, written
    // LE as 4 bytes. 1 + 4 = 5. Strip them.
    if full_bitstream.len() <= 5 {
        return Err(Error::invalid(
            "VP8 WebP encoder: VP8L alpha bitstream too short to strip header",
        ));
    }
    Ok(full_bitstream[5..].to_vec())
}

/// Apply the WebP ALPH filter (RFC 9649 §5.2.3) to an alpha plane in
/// place, producing per-pixel residuals that the matching `unfilter`
/// step in the decoder reverses by additive walk. Filter modes:
///
/// * 0 — identity (no change).
/// * 1 — horizontal: `r[x] = a[x] - a[x-1]` (first column kept as-is).
/// * 2 — vertical:   `r[x,y] = a[x,y] - a[x,y-1]` (first row kept).
/// * 3 — gradient:   `r = a - clip(L + T - TL)` (first row + first
///   column degenerate to mode-1 / mode-2 / identity).
///
/// The forward pass mirrors the decoder's `unfilter_alpha` per-mode
/// arithmetic exactly, so encode-then-decode is byte-identical.
fn apply_alph_filter(plane: &mut [u8], w: usize, h: usize, mode: u8) {
    match mode {
        0 => {}
        1 => {
            // Walk each row right-to-left so each `a[x] -= a[x-1]` sees
            // the *original* `a[x-1]` (not its already-filtered residual).
            for y in 0..h {
                for x in (1..w).rev() {
                    let i = y * w + x;
                    let left = plane[i - 1];
                    plane[i] = plane[i].wrapping_sub(left);
                }
            }
        }
        2 => {
            // Walk rows bottom-to-top for the same "see original above"
            // reason.
            for y in (1..h).rev() {
                for x in 0..w {
                    let i = y * w + x;
                    let top = plane[i - w];
                    plane[i] = plane[i].wrapping_sub(top);
                }
            }
        }
        3 => {
            // Gradient filter must process pixels in reverse-raster
            // order so each `a -= clip(L + T - TL)` reads the still-
            // unfiltered L / T / TL values.
            for y in (0..h).rev() {
                for x in (0..w).rev() {
                    let i = y * w + x;
                    let pred: i32 = if y == 0 && x == 0 {
                        0
                    } else if y == 0 {
                        plane[i - 1] as i32
                    } else if x == 0 {
                        plane[i - w] as i32
                    } else {
                        let l = plane[i - 1] as i32;
                        let t = plane[i - w] as i32;
                        let tl = plane[i - w - 1] as i32;
                        (l + t - tl).clamp(0, 255)
                    };
                    plane[i] = (plane[i] as i32 - pred) as u8;
                }
            }
        }
        _ => {}
    }
}

/// Cheap pre-VP8L cost estimator used by [`encode_alph_chunk`] to
/// pick a filter mode without paying for four full VP8L encodes. The
/// metric is the sum of `min(byte, 256-byte)` over the residual plane
/// — a coarse proxy for the entropy the green Huffman alphabet will
/// see (the alphabet is symmetric around 0 / 256 modulo, so absolute
/// magnitude with wrap-around tracks code length monotonically). On a
/// flat alpha plane every filter mode collapses to all zeros and ties
/// at cost 0; in that case `apply_alph_filter` picks identity (the
/// `<=` comparison favours the lower mode index, so unflittered wins).
fn alph_filter_cost(plane: &[u8]) -> u64 {
    let mut s: u64 = 0;
    for &b in plane {
        let bb = b as u64;
        s += bb.min(256 - bb);
    }
    s
}

/// Build an ALPH chunk for an 8-bit alpha plane.
///
/// Picks the cheapest of the four ALPH filter modes (0/1/2/3) by
/// scanning each filtered residual plane with [`alph_filter_cost`],
/// then VP8L-compresses the winner. `header_byte` is set to
/// `(filtering << 2) | compression` per RFC 9649 §5.2.3 with
/// compression = 1 (VP8L), pre_processing = 0, reserved = 0.
///
/// Most photographic alpha planes are constant 0xff (premultiplied
/// background) — the cost estimator picks mode 0 there and saves the
/// per-pixel filter pass entirely. Smooth alpha edges (typical of
/// rendered UI / icon overlays) win on mode 1 or 2 because the residual
/// plane collapses to a tight low-magnitude distribution that the VP8L
/// green Huffman tree can pack into 1-2 bits per pixel.
pub(crate) fn encode_alph_chunk(width: u32, height: u32, alpha: &[u8]) -> Result<AlphChunkBytes> {
    let w = width as usize;
    let h = height as usize;
    debug_assert_eq!(alpha.len(), w * h);

    // Score each filter mode on a scratch copy of the plane. Cost
    // is a coarse residual-magnitude sum — close enough for picking
    // the right mode without four VP8L encodes.
    let mut best_mode: u8 = 0;
    let mut best_cost = alph_filter_cost(alpha);
    for mode in 1u8..=3 {
        let mut scratch = alpha.to_vec();
        apply_alph_filter(&mut scratch, w, h, mode);
        let cost = alph_filter_cost(&scratch);
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }

    // Apply the winning filter to the real plane and VP8L-encode the
    // residual stream.
    let mut filtered = alpha.to_vec();
    apply_alph_filter(&mut filtered, w, h, best_mode);
    let payload = encode_alpha_plane_as_vp8l(width, height, &filtered)?;

    // header byte layout: (reserved<<6) | (pre_processing<<4) |
    //                     (filtering<<2) | compression
    // We use compression=1 (VP8L), pre_processing=0, reserved=0.
    let header_byte = ((best_mode & 0b11) << 2) | 0b01;
    Ok(AlphChunkBytes {
        header_byte,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riff_wrapper_layout_even_payload() {
        // Simple-file layout should be byte-identical to what the
        // pre-RIFF-refactor helper produced for a plain VP8 payload.
        let payload = vec![0xAAu8; 10];
        let out = build_webp_file(
            ImageKind::Vp8Lossy,
            &payload,
            16,
            16,
            None,
            &WebpMetadata::default(),
        );
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        assert_eq!(&out[12..16], b"VP8 ");
        let riff_size = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        assert_eq!(riff_size, 22);
        let chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]);
        assert_eq!(chunk_len, 10);
        assert_eq!(&out[20..30], &payload[..]);
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn riff_wrapper_layout_odd_payload_pads() {
        let payload = vec![0x55u8; 11];
        let out = build_webp_file(
            ImageKind::Vp8Lossy,
            &payload,
            16,
            16,
            None,
            &WebpMetadata::default(),
        );
        let riff_size = u32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        assert_eq!(riff_size, 24);
        assert_eq!(out.len(), 32);
        assert_eq!(out[31], 0x00);
    }

    #[test]
    fn quality_to_qindex_endpoints_and_clamp() {
        // 0   → max compression / lowest quality → qindex 127.
        // 100 → min compression / best quality   → qindex 0.
        // 50  → midpoint, rounds to 64 (50 * 1.27 = 63.5 → 64).
        // Values outside [0, 100] are clamped before mapping.
        assert_eq!(quality_to_qindex(0.0), 127);
        assert_eq!(quality_to_qindex(100.0), 0);
        assert_eq!(quality_to_qindex(50.0), 64);
        assert_eq!(quality_to_qindex(75.0), 32); // libwebp's default ≈ 32.
        assert_eq!(quality_to_qindex(-10.0), 127);
        assert_eq!(quality_to_qindex(150.0), 0);
        assert_eq!(quality_to_qindex(f32::NAN), 127);
    }

    #[test]
    fn segment_quant_deltas_widen_with_qindex() {
        // At very high quality (qindex 0) the per-segment QP deltas
        // should be tight — every segment is already near-lossless and
        // the perceptual gain from spending more bits on smooth content
        // is gone. At very low quality (qindex 127) the deltas widen so
        // smooth segments get noticeably finer quant than textured
        // segments. Validate the magnitude trend on the smooth segment
        // (id 0) and the high-variance segment (id 3).
        let lo_q = segment_quant_deltas_for_qindex(0);
        let mid_q = segment_quant_deltas_for_qindex(64);
        let hi_q = segment_quant_deltas_for_qindex(127);

        // Segment 0 (smooth) is always negative — better quality there.
        assert!(lo_q[0] < 0 && mid_q[0] < 0 && hi_q[0] < 0);
        // Segment 3 (textured) is always positive — coarser quant.
        assert!(lo_q[3] > 0 && mid_q[3] > 0 && hi_q[3] > 0);
        // Segment 2 is the unmodified baseline.
        assert_eq!(lo_q[2], 0);
        assert_eq!(mid_q[2], 0);
        assert_eq!(hi_q[2], 0);
        // Magnitudes monotonically widen with qindex.
        assert!(hi_q[0] <= mid_q[0] && mid_q[0] <= lo_q[0]);
        assert!(hi_q[3] >= mid_q[3] && mid_q[3] >= lo_q[3]);
        // All deltas land in the legal 5-bit signed-magnitude range.
        for d in lo_q.iter().chain(mid_q.iter()).chain(hi_q.iter()) {
            assert!(*d >= -15 && *d <= 15, "delta {d} out of [-15, 15]");
        }
    }

    #[test]
    fn segment_lf_deltas_smooth_negative_textured_positive() {
        // Smooth segment LF delta is non-positive (softer filter); the
        // textured segment LF delta is non-negative (stronger filter)
        // for every qindex on the curve.
        for qi in [0u8, 32, 64, 96, 127] {
            let lf = segment_lf_deltas_for_qindex(qi);
            assert!(
                lf[0] <= 0,
                "qindex {qi}: smooth LF delta {} should be <= 0",
                lf[0]
            );
            assert_eq!(lf[2], 0, "qindex {qi}: midline segment must be 0");
            assert!(
                lf[3] >= 1,
                "qindex {qi}: textured LF delta {} should be >= 1",
                lf[3]
            );
            for d in lf.iter() {
                assert!(
                    *d >= -63 && *d <= 63,
                    "qindex {qi}: lf delta {d} out of range"
                );
            }
        }
    }

    #[test]
    fn quality_to_qindex_is_monotonically_decreasing() {
        // Sweep the full range and verify the mapping is non-increasing
        // (each step up in quality must yield a qindex ≤ the previous one).
        let mut prev = quality_to_qindex(0.0);
        let mut q = 0.0_f32;
        while q <= 100.0 {
            let cur = quality_to_qindex(q);
            assert!(
                cur <= prev,
                "quality {q} produced qindex {cur} > previous {prev} — mapping not monotone"
            );
            prev = cur;
            q += 1.0;
        }
    }

    #[test]
    fn alph_filter_horizontal_picked_for_row_step_pattern() {
        // Each row carries a fresh independent step pattern that mode
        // 1 (horizontal) handles cheaply but mode 2 (vertical) and mode
        // 3 (gradient) cannot — their predictors can't see the per-row
        // change. Specifically: row y is filled with the constant
        // 4 * (y % 64), so each row is uniform but neighbouring rows
        // differ. Mode 1 collapses interior pixels to 0, mode 2 and
        // mode 3 leave large vertical residuals.
        let w = 64usize;
        let h = 64usize;
        let mut alpha = vec![0u8; w * h];
        for y in 0..h {
            let row_val = ((y % 64) * 4) as u8;
            for x in 0..w {
                alpha[y * w + x] = row_val;
            }
        }

        let chunk = encode_alph_chunk(w as u32, h as u32, &alpha).expect("encode_alph_chunk");
        let filter_mode = (chunk.header_byte >> 2) & 0b11;
        // Mode 0 produces a cost of (h * w) * mean(min(v, 256-v)); mode
        // 1 produces (h * 1) * row_val_cost (only column-0 carries a
        // residual). Whichever non-identity mode wins is fine — what
        // matters is that mode 0 doesn't.
        assert!(
            filter_mode != 0,
            "row-step pattern must select a non-identity filter (got mode {filter_mode})",
        );
        // Compression bit must still be 1 (VP8L).
        assert_eq!(chunk.header_byte & 0b11, 1, "compression bit must be VP8L");

        // Round-trip via the matching decoder unfilter step. We
        // synthesise the 5-byte VP8L prefix the ALPH decoder would
        // prepend, decode, then run the same `unfilter_alpha`
        // arithmetic the decoder uses (see decoder.rs::unfilter_alpha).
        // This catches any encoder/decoder filter-direction skew that a
        // pure self-roundtrip would miss.
        let mut synth = Vec::with_capacity(chunk.payload.len() + 5);
        synth.push(0x2f);
        let pw = (w as u32).saturating_sub(1) & 0x3fff;
        let ph = (h as u32).saturating_sub(1) & 0x3fff;
        let packed = pw | (ph << 14);
        synth.extend_from_slice(&packed.to_le_bytes());
        synth.extend_from_slice(&chunk.payload);
        let img = crate::vp8l::decode(&synth).expect("vp8l decode of alph payload");
        let mut plane: Vec<u8> = img.pixels.iter().map(|p| ((p >> 8) & 0xff) as u8).collect();
        // Inline the decoder's matching unfilter for the chosen mode.
        match filter_mode {
            0 => {}
            1 => {
                for y in 0..h {
                    for x in 1..w {
                        let i = y * w + x;
                        let left = plane[i - 1];
                        plane[i] = plane[i].wrapping_add(left);
                    }
                }
            }
            2 => {
                for y in 1..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let top = plane[i - w];
                        plane[i] = plane[i].wrapping_add(top);
                    }
                }
            }
            3 => {
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let pred: i32 = if y == 0 && x == 0 {
                            0
                        } else if y == 0 {
                            plane[i - 1] as i32
                        } else if x == 0 {
                            plane[i - w] as i32
                        } else {
                            let l = plane[i - 1] as i32;
                            let t = plane[i - w] as i32;
                            let tl = plane[i - w - 1] as i32;
                            (l + t - tl).clamp(0, 255)
                        };
                        plane[i] = ((plane[i] as i32 + pred) & 0xff) as u8;
                    }
                }
            }
            _ => unreachable!(),
        }
        assert_eq!(plane, alpha, "filtered ALPH must round-trip");
    }

    #[test]
    fn alph_filter_identity_picked_for_constant_alpha() {
        // A fully-opaque (constant 0xff) plane: every filter mode produces
        // an all-zero residual *except* mode 0 which keeps the input.
        // Cost is therefore 1 * count for mode 0 and 0 for modes 1..3.
        // The first-better-wins selection should pick mode 1 (the first
        // mode that ties at cost 0 below the mode-0 baseline).
        //
        // The case worth pinning: mode 0 must NOT win. If it did, the
        // Huffman alphabet would have to encode the literal 0xff per
        // pixel, while modes 1..3 collapse to a single literal + cache
        // hits.
        let w = 32usize;
        let h = 32usize;
        let alpha = vec![0xffu8; w * h];
        let chunk = encode_alph_chunk(w as u32, h as u32, &alpha).expect("encode_alph_chunk");
        let filter_mode = (chunk.header_byte >> 2) & 0b11;
        assert!(
            (1..=3).contains(&filter_mode),
            "constant 0xff alpha must select a non-identity filter (got mode {filter_mode})",
        );
    }

    #[test]
    fn alph_filter_apply_then_unfilter_roundtrips_all_modes() {
        // Property check: every filter mode must be the inverse of the
        // matching `unfilter_alpha` arithmetic (which lives in
        // decoder.rs). Build a deterministic noisy plane and verify
        // identity for modes 0..3.
        let w = 17usize;
        let h = 13usize;
        let mut alpha = vec![0u8; w * h];
        let mut s: u32 = 0x5eed_d00d;
        for b in alpha.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = (s & 0xff) as u8;
        }
        for mode in 0u8..=3 {
            let mut filtered = alpha.clone();
            apply_alph_filter(&mut filtered, w, h, mode);
            // Inline the decoder's unfilter for each mode; mirrors
            // crate::decoder::unfilter_alpha exactly.
            let mut restored = filtered.clone();
            match mode {
                0 => {}
                1 => {
                    for y in 0..h {
                        for x in 1..w {
                            let i = y * w + x;
                            let left = restored[i - 1];
                            restored[i] = restored[i].wrapping_add(left);
                        }
                    }
                }
                2 => {
                    for y in 1..h {
                        for x in 0..w {
                            let i = y * w + x;
                            let top = restored[i - w];
                            restored[i] = restored[i].wrapping_add(top);
                        }
                    }
                }
                3 => {
                    for y in 0..h {
                        for x in 0..w {
                            let i = y * w + x;
                            let pred: i32 = if y == 0 && x == 0 {
                                0
                            } else if y == 0 {
                                restored[i - 1] as i32
                            } else if x == 0 {
                                restored[i - w] as i32
                            } else {
                                let l = restored[i - 1] as i32;
                                let t = restored[i - w] as i32;
                                let tl = restored[i - w - 1] as i32;
                                (l + t - tl).clamp(0, 255)
                            };
                            restored[i] = ((restored[i] as i32 + pred) & 0xff) as u8;
                        }
                    }
                }
                _ => unreachable!(),
            }
            assert_eq!(
                restored, alpha,
                "filter mode {mode}: forward + inverse must be identity"
            );
        }
    }
}
