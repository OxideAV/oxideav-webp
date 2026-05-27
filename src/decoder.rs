//! Published-API `oxideav_webp::decoder` module — the published decode
//! surface grouped under its qualified path.
//!
//! Per `API-COMPAT-0.1.2.md` consumers may reach the decode entry
//! points either at the crate root (`oxideav_webp::decode_webp`) or via
//! this module (`oxideav_webp::decoder::decode_webp`). Both paths return
//! the same [`WebpImage`] / [`WebpFrame`] values via the same
//! [`WebpError`].
//!
//! The streaming framework-side [`WebpDecoder`] handle lives in
//! [`crate::registry`] and is only available with the default
//! `registry` Cargo feature on; it is re-exported here under the same
//! cfg gate.

pub use crate::{decode_webp, WebpFrame, WebpImage};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;

#[cfg(feature = "registry")]
pub use crate::registry::{make_decoder, WebpDecoder};

/// Direct factory for a streaming VP8L lossless [`WebpDecoder`] —
/// mirrors the dual-API convention `<crate>::decoder::make_*_decoder`.
///
/// The framework-side [`crate::registry::make_decoder`] is the
/// `CodecParameters`-typed factory installed in the codec registry; this
/// is the bare-parameters convenience form that constructs the decoder
/// straight from canvas dimensions, for callers that already know the
/// WebP `.webp` is a single-frame VP8L image.
#[cfg(feature = "registry")]
pub fn make_vp8l_decoder(width: u32, height: u32) -> WebpDecoder {
    use oxideav_core::{CodecId, CodecParameters, MediaType, PixelFormat};
    let mut params = CodecParameters::video(CodecId::new(crate::registry::CODEC_ID_STR));
    params.media_type = MediaType::Video;
    params.pixel_format = Some(PixelFormat::Rgba);
    params.width = Some(width);
    params.height = Some(height);
    WebpDecoder::new(params)
}
