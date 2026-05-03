//! Local error type used by `oxideav-webp`'s standalone (no
//! `oxideav-core`) public API.
//!
//! When the `registry` feature is enabled, [`WebpError`] gains a
//! `From<WebpError> for oxideav_core::Error` impl (defined in
//! [`crate::registry`]) so the trait-side surface (`Decoder` /
//! `Encoder` / `Demuxer`) can keep returning `oxideav_core::Result<T>`
//! while the underlying decode/encode functions stay framework-free.

use std::fmt;

/// `Result` alias scoped to `oxideav-webp`. Standalone (no
/// `oxideav-core`) callers see this; framework callers convert via
/// the gated `From<WebpError> for oxideav_core::Error` impl.
pub type Result<T> = std::result::Result<T, WebpError>;

/// Error variants returned by `oxideav-webp`'s standalone API.
///
/// The variants mirror the subset of `oxideav_core::Error` the codec
/// can hit. The crate intentionally avoids surfacing transport (`Io`)
/// or framework-specific (`FormatNotFound`, `CodecNotFound`) errors —
/// those originate in callers that are already linking `oxideav-core`.
#[derive(Debug)]
pub enum WebpError {
    /// The input bitstream / container is malformed (bad RIFF / VP8 /
    /// VP8L magic, truncated chunk, contradictory header, etc.).
    InvalidData(String),
    /// The bitstream uses a feature this decoder doesn't implement,
    /// or the encoder was asked to emit a frame format it doesn't
    /// support.
    Unsupported(String),
    /// End of stream — no more packets / frames forthcoming.
    Eof,
    /// More input is required before another frame can be produced
    /// (decoder) or another packet can be flushed (encoder).
    NeedMore,
}

impl WebpError {
    /// Construct a [`WebpError::InvalidData`] from a stringy message.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Construct a [`WebpError::Unsupported`] from a stringy message.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
}

impl fmt::Display for WebpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Eof => write!(f, "end of stream"),
            Self::NeedMore => write!(f, "need more data"),
        }
    }
}

impl std::error::Error for WebpError {}

/// Bubble VP8 decode failures up through the WebP decode/encode
/// pipeline. The variants map 1:1 — `Vp8Error::InvalidData` becomes
/// `WebpError::InvalidData`, etc.
impl From<oxideav_vp8::Vp8Error> for WebpError {
    fn from(e: oxideav_vp8::Vp8Error) -> Self {
        match e {
            oxideav_vp8::Vp8Error::InvalidData(s) => WebpError::InvalidData(s),
            oxideav_vp8::Vp8Error::Unsupported(s) => WebpError::Unsupported(s),
            oxideav_vp8::Vp8Error::Eof => WebpError::Eof,
            oxideav_vp8::Vp8Error::NeedMore => WebpError::NeedMore,
        }
    }
}
