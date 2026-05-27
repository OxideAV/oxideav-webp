//! Published-API `oxideav_webp::encoder_vp8` module — VP8 lossy
//! factory façade, plus the §6 / §9 quality-knob types reproduced on
//! top of the workspace `oxideav-vp8` sibling crate's public surface.
//!
//! Per `API-COMPAT-0.1.2.md` this module exposes:
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
//! The factories themselves are blocked on `oxideav-vp8` growing a
//! `CodecParameters`-typed factory; until then they surface
//! [`crate::WebpError::Unsupported`] with a clear "VP8 lossy encoder not
//! yet rebuilt" message rather than silently misbehaving.

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

/// Map a libwebp-style quality scale (`0.0..=100.0`) to a VP8 qindex
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

// ───────────────────────── framework-side factories ─────────────────────────

#[cfg(feature = "registry")]
use oxideav_core::{CodecParameters, Encoder};

/// Build a `Box<dyn Encoder>` for the published `"webp_vp8"` codec id.
///
/// The VP8 lossy encoder is not yet rebuilt clean-room (workspace
/// task #1041 — the `oxideav-vp8` sibling has only Phase-1 silent-keyframe
/// support published today). Until the lossy encoder lands, the factory
/// surfaces [`WebpError::Unsupported`] so the registry path fails loudly
/// instead of silently emitting a degenerate file.
#[cfg(feature = "registry")]
pub fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>, oxideav_core::Error> {
    Err(oxideav_core::Error::Unsupported(
        "oxideav-webp encoder_vp8: VP8 lossy encoder not yet rebuilt".to_string(),
    ))
}

/// `make_encoder` plus a libwebp-style quality knob (`0.0..=100.0`).
///
/// See [`quality_to_qindex`] for the projection. Currently routes to
/// [`make_encoder`] (i.e. surfaces [`WebpError::Unsupported`]); the
/// quality parameter is forwarded once the lossy encoder lands.
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality(
    params: &CodecParameters,
    _quality: f32,
) -> Result<Box<dyn Encoder>, oxideav_core::Error> {
    make_encoder(params)
}

/// `make_encoder` plus an explicit `qindex` (`0..=127`, lower = better).
///
/// Currently surfaces [`WebpError::Unsupported`] until the VP8 lossy
/// encoder lands.
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex(
    params: &CodecParameters,
    _qindex: u8,
) -> Result<Box<dyn Encoder>, oxideav_core::Error> {
    make_encoder(params)
}

/// `make_encoder` plus an explicit `qindex` and a [`Vp8FreqDeltas`]
/// record of per-band quantiser deltas.
///
/// Currently surfaces [`WebpError::Unsupported`] until the VP8 lossy
/// encoder lands.
#[cfg(feature = "registry")]
pub fn make_encoder_with_qindex_and_freq_deltas(
    params: &CodecParameters,
    _qindex: u8,
    _deltas: Vp8FreqDeltas,
) -> Result<Box<dyn Encoder>, oxideav_core::Error> {
    make_encoder(params)
}

/// `make_encoder` plus a libwebp-style quality knob and a
/// [`Vp8FreqDeltas`] record of per-band quantiser deltas.
///
/// Currently surfaces [`WebpError::Unsupported`] until the VP8 lossy
/// encoder lands.
#[cfg(feature = "registry")]
pub fn make_encoder_with_quality_and_freq_deltas(
    params: &CodecParameters,
    _quality: f32,
    _deltas: Vp8FreqDeltas,
) -> Result<Box<dyn Encoder>, oxideav_core::Error> {
    make_encoder(params)
}

/// Crate-local convenience: surface a `WebpError::Unsupported` over the
/// same message used by the framework-trait factories. Used by the
/// standalone-side tests so they can assert the same intent without
/// pulling `oxideav-core`.
pub fn unsupported_for_standalone() -> WebpError {
    WebpError::Unsupported
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
