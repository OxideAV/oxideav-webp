//! Pre-contract `oxideav_webp::decoder` module path — re-exports of the
//! decode surface under its historical qualified name.
//!
//! New code uses the crate root ([`crate::decode`], [`crate::decode_all`],
//! [`crate::WebpImage`]); this module only keeps the old paths resolving.

pub use crate::{decode, decode_all, decode_rgb8, decode_rgba8, decode_with, Frame, WebpImage};

#[allow(deprecated)]
pub use crate::{decode_webp, DecodedWebpFile, WebpFrame};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;

#[cfg(feature = "registry")]
pub use crate::registry::{make_decoder, WebpDecoder};

/// Direct factory for a framework [`WebpDecoder`] whose output parameters
/// start from the given canvas dimensions — the dual-API convenience
/// counterpart of the `CodecParameters`-typed [`make_decoder`].
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
