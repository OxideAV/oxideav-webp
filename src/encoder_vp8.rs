//! Published §2.5 `VP8 ` (lossy) **encoder** surface — API-shape stubs.
//!
//! ## Status: API-only stub, no pixel encode (round 131)
//!
//! The published-0.1.5 `oxideav-webp` crate exposed a family of VP8
//! lossy encode entry points (`encode_vp8_lossy_rgba`,
//! `encode_vp8_lossy_yuv420p`, `encoder_vp8::make_encoder_with_quality`,
//! …) — see `API-COMPAT.md` for the full list. The current sibling
//! crate `oxideav-vp8 = "0.2"` ships only **Phase 1** of the encoder:
//!
//! * a working §7.3 boolean-range entropy *encoder* primitive;
//! * the §9.1 / §9.3 / §9.4 / §9.5 / §9.6 / §9.9 / §9.10 / §9.11
//!   frame-header writer subroutines;
//! * [`oxideav_vp8::encode_silent_keyframe`] — a top-level driver
//!   that emits a *structurally valid VP8 key frame* in which every
//!   macroblock carries `mb_skip_coeff = 1` with `DC_PRED` luma +
//!   chroma. The encoder **ignores its pixel input entirely** — the
//!   produced frame is a constant-grey picture regardless of what
//!   `rgba` the caller passes; and
//! * the §13 [`oxideav_vp8::TokenEncoder`] primitive (Phase 2,
//!   per-block), which can encode pre-computed `[i16; 16]`
//!   coefficient blocks but is **not** driven from pixels yet.
//!
//! Specifically, `oxideav-vp8 = "0.2"` does **not** expose:
//!
//! 1. A forward DCT (§14.4) / forward WHT (§14.3) on a 4×4 / Y2 block.
//! 2. A pixel-driven §12 intra-prediction search or per-MB residual
//!    computation.
//! 3. §14.1 quantization (the encoder side, with its `ac_qlookup` /
//!    `dc_qlookup` rounding rules).
//! 4. A per-MB encode driver that consumes a 16×16 luma + 8×8 chroma
//!    block and emits the §11 mode records plus §13 token stream that
//!    [`TokenEncoder`] would consume.
//! 5. A top-level "encode RGBA → VP8 keyframe" function.
//!
//! Without (1)–(5), threading a real RGBA picture through
//! `encode_silent_keyframe` would produce a constant-grey VP8 stream
//! that the WebP RIFF wrapper would dutifully package — i.e. **garbage
//! bytes**, the exact failure mode the round-131 directive explicitly
//! forbids ("DO NOT pretend to wire it … what's NOT valid is producing
//! a 'VP8 lossy' function that returns garbage bytes").
//!
//! This module therefore lands the **API surface** required by
//! `API-COMPAT.md` — the function names, types, constants, and registry
//! entry — so downstream consumers compile and so a future round can
//! drop a working encoder in without an API churn. Every entry point
//! returns a clean
//! [`crate::WebpError::Unsupported`] / `oxideav_core::Error::Unsupported`
//! at call time; no encoded bytes are produced unless and until the
//! upstream §13 / §14 encode round on `oxideav-vp8` lands the pixel
//! path.
//!
//! ## Gap report (for the parent's round-131 ledger)
//!
//! The work that must land on `oxideav-vp8` to unblock real wiring
//! here:
//!
//! * **§14.3 forward WHT** + **§14.4 forward 4×4 DCT** — inverse
//!   primitives already exist on the decode side; the forward
//!   primitives are pure numeric kernels.
//! * **§14.1 forward quantization** — multiply / round-to-nearest
//!   against `ac_qlookup` / `dc_qlookup`, mirroring
//!   [`oxideav_vp8::MbDequantFactors`].
//! * **§12 pixel-driven intra-mode search** — pick `DC_PRED` /
//!   `V_PRED` / `H_PRED` / `TM_PRED` (or `B_PRED`) per MB; for a v0
//!   encoder, a fixed `DC_PRED` choice that *honours the predicted
//!   pixels* is sufficient.
//! * **Per-MB encode driver** consuming a 16×16 luma + two 8×8 chroma
//!   block, emitting (a) the §11 mode record, (b) the dequantized
//!   reference reconstruction (for the §15 loop filter's neighbour
//!   reads), and (c) the §13 token-stream bytes [`TokenEncoder`]
//!   already wraps.
//! * **Top-level "encode I420 → VP8 keyframe"** driver — this is the
//!   one this module would call. Inputs: planar I420 + dimensions +
//!   `qindex`; output: a complete VP8 keyframe bitstream the existing
//!   [`oxideav_vp8::decode_vp8`] decodes back.
//!
//! Once that lands, the bodies of
//! [`crate::encode_vp8_lossy_yuv420p`] / [`crate::encode_vp8_lossy_rgba`]
//! / [`make_encoder_with_quality`] become a thin call into the new
//! `oxideav_vp8` entry plus the existing
//! [`crate::build::build_webp_file`] RIFF wrapper.

use crate::WebpError;
#[cfg(feature = "registry")]
use crate::WebpMetadataOwned;

/// `qindex` band for [`make_encoder_with_qindex`] — RFC 6386 §9.6
/// `y_ac_qi` baseline (0..=127, lower = better quality, higher = more
/// compression). Mirrors `oxideav-vp8`'s
/// [`oxideav_vp8::SilentKeyframeParams::y_ac_qi`] field.
pub const QINDEX_MIN: u8 = 0;
/// Upper bound on the §9.6 `y_ac_qi` baseline (RFC 6386 — 7-bit
/// quantiser index).
pub const QINDEX_MAX: u8 = 127;

/// libwebp-style quality scale used by [`make_encoder_with_quality`].
///
/// The mapping is `qindex = round((100 - quality) * 127 / 100)`,
/// i.e. `quality = 100` → `qindex = 0` (best quality, biggest file),
/// `quality = 0` → `qindex = 127` (worst quality, smallest file).
pub const QUALITY_MIN: f32 = 0.0;
/// Upper bound on the libwebp-style quality scale.
pub const QUALITY_MAX: f32 = 100.0;
/// Default quality libwebp's `cwebp` uses when no `-q` flag is supplied —
/// preserved as the default for [`make_encoder_with_quality`] omissions.
pub const DEFAULT_QUALITY: f32 = 75.0;

/// Map a libwebp-style 0..=100 quality value to the RFC 6386 §9.6
/// 0..=127 `y_ac_qi` baseline.
///
/// Out-of-range values clamp; NaN folds to [`DEFAULT_QUALITY`]'s mapping.
/// This is a public utility so callers can mix-and-match the two scales
/// freely.
pub fn quality_to_qindex(quality: f32) -> u8 {
    let q = if quality.is_nan() {
        DEFAULT_QUALITY
    } else {
        quality.clamp(QUALITY_MIN, QUALITY_MAX)
    };
    let qi = ((QUALITY_MAX - q) * (QINDEX_MAX as f32) / QUALITY_MAX).round();
    qi.clamp(QINDEX_MIN as f32, QINDEX_MAX as f32) as u8
}

/// Frequency-band delta knobs the published-0.1.5 API surfaced for the
/// VP8 lossy encoder's perceptual-RDO ladder.
///
/// **API-shape stub.** The published crate carried these so a caller
/// could fine-tune the quantiser deltas applied per AC-frequency band
/// (low / mid / high luma + chroma) without dropping out to a custom
/// `qindex`. The current rebuild stores the fields verbatim so the type
/// round-trips through `params.extradata`, but they have **no effect**
/// on the encoder output yet — the encoder itself is not pixel-driven
/// (see the module-level note).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vp8FreqDeltas {
    /// Quantiser delta applied to the luma DC coefficient (low band).
    pub luma_dc: i8,
    /// Quantiser delta applied to the luma low-frequency AC band.
    pub luma_low_ac: i8,
    /// Quantiser delta applied to the luma high-frequency AC band.
    pub luma_high_ac: i8,
    /// Quantiser delta applied to the chroma DC coefficient.
    pub chroma_dc: i8,
    /// Quantiser delta applied to the chroma AC coefficients.
    pub chroma_ac: i8,
}

impl Vp8FreqDeltas {
    /// All-zero deltas — the baseline a stub encoder would emit even if
    /// it were pixel-driven.
    pub const fn zero() -> Self {
        Self {
            luma_dc: 0,
            luma_low_ac: 0,
            luma_high_ac: 0,
            chroma_dc: 0,
            chroma_ac: 0,
        }
    }
}

/// Per-image psy-visual statistics the published-0.1.5
/// [`compute_psy_stats`] surface returned.
///
/// **API-shape stub.** A future round will populate this from the
/// caller's RGBA / YUV input (band energy, edge density, …). Today,
/// [`compute_psy_stats`] returns the all-zero default so downstream
/// consumers that pipe it back into
/// [`make_encoder_with_quality_and_freq_deltas`] / similar
/// see the same value they'd see from a no-op.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vp8PsyStats {
    /// Mean per-pixel luma activity. Placeholder; not yet computed.
    pub luma_activity: f32,
    /// Mean per-pixel chroma activity. Placeholder; not yet computed.
    pub chroma_activity: f32,
}

/// Compute the [`Vp8PsyStats`] for an RGBA input — published-API
/// shape; **returns all-zero** until the VP8 lossy encoder is
/// pixel-driven.
///
/// Length validation is still enforced (`rgba.len() ==
/// width * height * 4`) so callers see the same error they would
/// from a real implementation.
pub fn compute_psy_stats(width: u32, height: u32, rgba: &[u8]) -> Result<Vp8PsyStats, WebpError> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(WebpError::InvalidData)?;
    if rgba.len() != expected {
        return Err(WebpError::InvalidData);
    }
    Ok(Vp8PsyStats::default())
}

/// Derive the per-band `Vp8FreqDeltas` an encoder would apply for the
/// given baseline `qindex` — published-API shape; **returns
/// `Vp8FreqDeltas::zero()`** until the VP8 lossy encoder is
/// pixel-driven.
pub fn freq_deltas_for_qindex(qindex: u8) -> Vp8FreqDeltas {
    debug_assert!(qindex <= QINDEX_MAX);
    Vp8FreqDeltas::zero()
}

// ───────────────────────── Encoder factories (registry-side) ─────────────────────────
//
// API-COMPAT.md spells out the published direct-factory family
// (`make_encoder_with_quality`, `make_encoder_with_qindex`,
// `make_encoder_with_target_size`, and the
// `_and_metadata` / `_and_freq_deltas` variants). The registry-feature
// gate matches `oxideav-core`'s — when `default-features = false` is
// set on this crate the framework dep drops, so the trait-object
// factories below drop with it.

#[cfg(feature = "registry")]
mod factories {
    use super::*;
    use crate::registry::{Vp8LossyEncoderConfig, WebpVp8LossyEncoder};
    use oxideav_core::{CodecId, CodecParameters, Encoder, Error as CoreError, MediaType};

    /// Build a configured VP8 lossy encoder from a baseline
    /// libwebp-style `quality` value (0..=100, default 75).
    ///
    /// **Stub.** The encoder builds and registers fine; the first
    /// `send_frame` will fail with [`oxideav_core::Error::Unsupported`]
    /// per the module-level gap note. The published surface keeps the
    /// factory signature so callers compile; once `oxideav-vp8` lands
    /// the §13/§14 encode round the same construction starts emitting
    /// real bytes.
    pub fn make_encoder_with_quality(
        params: &CodecParameters,
        quality: f32,
    ) -> oxideav_core::Result<Box<dyn Encoder>> {
        make_encoder_with_qindex(params, quality_to_qindex(quality))
    }

    /// Build a configured VP8 lossy encoder from a baseline `qindex`
    /// (RFC 6386 §9.6 `y_ac_qi`, 0..=127 — lower = better quality).
    ///
    /// **Stub.** See [`make_encoder_with_quality`].
    pub fn make_encoder_with_qindex(
        params: &CodecParameters,
        qindex: u8,
    ) -> oxideav_core::Result<Box<dyn Encoder>> {
        make_encoder_with_qindex_and_metadata(params, qindex, WebpMetadataOwned::default())
    }

    /// Build a configured VP8 lossy encoder driven by a target output
    /// size in bytes — the published "size-pass" mode.
    ///
    /// **Stub.** Currently maps to a deterministic
    /// `qindex` chosen by [`quality_to_qindex`] for the libwebp default
    /// quality; the real size search is part of the deferred
    /// pixel-encode round.
    pub fn make_encoder_with_target_size(
        params: &CodecParameters,
        _target_bytes: usize,
    ) -> oxideav_core::Result<Box<dyn Encoder>> {
        make_encoder_with_qindex(params, quality_to_qindex(DEFAULT_QUALITY))
    }

    /// Build a configured VP8 lossy encoder from a baseline `qindex`
    /// and embed the supplied file-level metadata into every encoded
    /// `.webp`.
    pub fn make_encoder_with_qindex_and_metadata(
        params: &CodecParameters,
        qindex: u8,
        metadata: WebpMetadataOwned,
    ) -> oxideav_core::Result<Box<dyn Encoder>> {
        make_encoder_with_qindex_and_freq_deltas(params, qindex, Vp8FreqDeltas::zero(), metadata)
    }

    /// Build a configured VP8 lossy encoder from a baseline `quality`
    /// and embed the supplied file-level metadata.
    pub fn make_encoder_with_quality_and_metadata(
        params: &CodecParameters,
        quality: f32,
        metadata: WebpMetadataOwned,
    ) -> oxideav_core::Result<Box<dyn Encoder>> {
        make_encoder_with_qindex_and_metadata(params, quality_to_qindex(quality), metadata)
    }

    /// Build a configured VP8 lossy encoder with explicit per-band
    /// quantiser deltas on top of a baseline `qindex`.
    pub fn make_encoder_with_qindex_and_freq_deltas(
        params: &CodecParameters,
        qindex: u8,
        freq_deltas: Vp8FreqDeltas,
        metadata: WebpMetadataOwned,
    ) -> oxideav_core::Result<Box<dyn Encoder>> {
        if qindex > QINDEX_MAX {
            return Err(CoreError::invalid(format!(
                "webp_vp8 encoder: qindex {qindex} exceeds the §9.6 0..=127 range"
            )));
        }
        let width = params
            .width
            .ok_or_else(|| CoreError::invalid("webp_vp8 encoder: missing width"))?;
        let height = params
            .height
            .ok_or_else(|| CoreError::invalid("webp_vp8 encoder: missing height"))?;

        let mut output_params = params.clone();
        output_params.media_type = MediaType::Video;
        output_params.codec_id = CodecId::new(crate::CODEC_ID_VP8);

        Ok(Box::new(WebpVp8LossyEncoder::new(
            output_params,
            Vp8LossyEncoderConfig {
                width,
                height,
                pix: params.pixel_format,
                qindex,
                freq_deltas,
                metadata,
            },
        )))
    }

    /// Build a configured VP8 lossy encoder with explicit per-band
    /// quantiser deltas on top of a baseline `quality`.
    pub fn make_encoder_with_quality_and_freq_deltas(
        params: &CodecParameters,
        quality: f32,
        freq_deltas: Vp8FreqDeltas,
        metadata: WebpMetadataOwned,
    ) -> oxideav_core::Result<Box<dyn Encoder>> {
        make_encoder_with_qindex_and_freq_deltas(
            params,
            quality_to_qindex(quality),
            freq_deltas,
            metadata,
        )
    }
}

#[cfg(feature = "registry")]
pub use factories::{
    make_encoder_with_qindex, make_encoder_with_qindex_and_freq_deltas,
    make_encoder_with_qindex_and_metadata, make_encoder_with_quality,
    make_encoder_with_quality_and_freq_deltas, make_encoder_with_quality_and_metadata,
    make_encoder_with_target_size,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_to_qindex_endpoints_round_correctly() {
        // libwebp convention: quality=100 → qindex=0; quality=0 → qindex=127.
        assert_eq!(quality_to_qindex(100.0), 0);
        assert_eq!(quality_to_qindex(0.0), 127);
        // Default sits roughly in the middle of the band.
        let qi = quality_to_qindex(DEFAULT_QUALITY);
        assert!(qi > 0 && qi < QINDEX_MAX, "qindex {qi} should be mid-band");
    }

    #[test]
    fn quality_to_qindex_clamps_out_of_range() {
        assert_eq!(quality_to_qindex(-10.0), QINDEX_MAX);
        assert_eq!(quality_to_qindex(200.0), QINDEX_MIN);
        // NaN folds to the default quality's qindex.
        assert_eq!(
            quality_to_qindex(f32::NAN),
            quality_to_qindex(DEFAULT_QUALITY)
        );
    }

    #[test]
    fn freq_deltas_default_is_zero() {
        let d = Vp8FreqDeltas::default();
        assert_eq!(d, Vp8FreqDeltas::zero());
    }

    #[test]
    fn freq_deltas_for_qindex_is_zero_stub() {
        // Stub: every qindex returns zero deltas today.
        for qi in 0u8..=QINDEX_MAX {
            assert_eq!(freq_deltas_for_qindex(qi), Vp8FreqDeltas::zero());
        }
    }

    #[test]
    fn compute_psy_stats_validates_buffer_length() {
        // 2×2 image → 16 RGBA bytes.
        assert!(compute_psy_stats(2, 2, &[0u8; 16]).is_ok());
        // Wrong length — caller bug, not a pixel-encoder bug.
        assert_eq!(
            compute_psy_stats(2, 2, &[0u8; 15]),
            Err(WebpError::InvalidData)
        );
        assert_eq!(
            compute_psy_stats(2, 2, &[0u8; 17]),
            Err(WebpError::InvalidData)
        );
    }

    #[test]
    fn compute_psy_stats_returns_default_stub() {
        let s = compute_psy_stats(1, 1, &[0u8; 4]).unwrap();
        assert_eq!(s, Vp8PsyStats::default());
    }
}
