//! The one crate error type — [`WebpError`] (aliased as
//! [`Error`](crate::Error)) — plus the [`Result`] alias.
//!
//! Every root entry point (`probe` aside, which cannot fail) returns
//! `WebpError`. The per-module parser error enums (`container::ContainerError`,
//! `vp8x::Vp8xError`, `alph::AlphError`, …) are implementation detail: each
//! converts into a `WebpError` variant through `From`, so `?` works across
//! the whole crate and a consumer never sees more than the four contract
//! variants plus the two streaming states.

use core::fmt;

/// Error type for every fallible `oxideav_webp` entry point.
///
/// The variants follow the workspace image-crate contract:
///
/// * [`InvalidData`](Self::InvalidData) — the bytes are not a well-formed
///   WebP file, or a sub-bitstream (VP8L, VP8, ALPH, …) is corrupt. The
///   payload is a human-readable diagnostic naming the layer that refused.
/// * [`Unsupported`](Self::Unsupported) — the file is well-formed but asks
///   for something this crate does not implement (an inter-frame VP8
///   bitstream, an encode of a layout WebP cannot carry, …).
/// * [`LimitExceeded`](Self::LimitExceeded) — a [`DecodeOptions`](crate::DecodeOptions)
///   limit (dimensions, pixel count, byte count) was hit before allocation.
/// * [`Io`](Self::Io) — the `Read` / `Write` adapter failed.
/// * [`Eof`](Self::Eof) / [`NeedMore`](Self::NeedMore) — the streaming
///   states the historical `oxideav-webp` surface exposed; kept so
///   framework adapters can map them 1:1.
#[derive(Debug)]
#[non_exhaustive]
pub enum WebpError {
    /// Malformed container or bitstream.
    InvalidData(String),
    /// Well-formed input the crate does not handle.
    Unsupported(String),
    /// A decode limit was exceeded.
    LimitExceeded(String),
    /// An I/O error from the `Read` / `Write` adapters.
    Io(std::io::Error),
    /// The input ended before a complete image could be read.
    Eof,
    /// More input is required to complete the decode (streaming callers).
    NeedMore,
}

impl WebpError {
    /// Build an [`InvalidData`](Self::InvalidData) error.
    pub fn invalid<S: Into<String>>(msg: S) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Build an [`Unsupported`](Self::Unsupported) error.
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Build a [`LimitExceeded`](Self::LimitExceeded) error.
    pub fn limit<S: Into<String>>(msg: S) -> Self {
        Self::LimitExceeded(msg.into())
    }

    /// `true` for [`InvalidData`](Self::InvalidData).
    pub fn is_invalid_data(&self) -> bool {
        matches!(self, Self::InvalidData(_))
    }

    /// `true` for [`Unsupported`](Self::Unsupported).
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }

    /// `true` for [`LimitExceeded`](Self::LimitExceeded).
    pub fn is_limit_exceeded(&self) -> bool {
        matches!(self, Self::LimitExceeded(_))
    }
}

impl fmt::Display for WebpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(m) => write!(f, "oxideav-webp: invalid WebP data: {m}"),
            Self::Unsupported(m) => write!(f, "oxideav-webp: unsupported: {m}"),
            Self::LimitExceeded(m) => write!(f, "oxideav-webp: limit exceeded: {m}"),
            Self::Io(e) => write!(f, "oxideav-webp: i/o error: {e}"),
            Self::Eof => f.write_str("oxideav-webp: unexpected end of input"),
            Self::NeedMore => f.write_str("oxideav-webp: more input required"),
        }
    }
}

impl std::error::Error for WebpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Two errors compare equal when they are the same variant with the same
/// message; two [`Io`](WebpError::Io) errors compare by `kind()` and
/// rendered text (an `io::Error` has no structural equality of its own).
impl PartialEq for WebpError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::InvalidData(a), Self::InvalidData(b)) => a == b,
            (Self::Unsupported(a), Self::Unsupported(b)) => a == b,
            (Self::LimitExceeded(a), Self::LimitExceeded(b)) => a == b,
            (Self::Io(a), Self::Io(b)) => a.kind() == b.kind() && a.to_string() == b.to_string(),
            (Self::Eof, Self::Eof) | (Self::NeedMore, Self::NeedMore) => true,
            _ => false,
        }
    }
}

impl Eq for WebpError {}

impl Clone for WebpError {
    fn clone(&self) -> Self {
        match self {
            Self::InvalidData(m) => Self::InvalidData(m.clone()),
            Self::Unsupported(m) => Self::Unsupported(m.clone()),
            Self::LimitExceeded(m) => Self::LimitExceeded(m.clone()),
            Self::Io(e) => Self::Io(std::io::Error::new(e.kind(), e.to_string())),
            Self::Eof => Self::Eof,
            Self::NeedMore => Self::NeedMore,
        }
    }
}

impl From<std::io::Error> for WebpError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Every per-module parser error is a malformed-input report at the layer
/// named by its `Display` text.
macro_rules! invalid_data_from {
    ($($ty:ty => $layer:literal),* $(,)?) => {
        $(
            impl From<$ty> for WebpError {
                fn from(e: $ty) -> Self {
                    Self::InvalidData(format!(concat!($layer, ": {}"), e))
                }
            }
        )*
    };
}

invalid_data_from! {
    crate::container::ContainerError => "container",
    crate::vp8x::Vp8xError => "vp8x",
    crate::alph::AlphError => "alph",
    crate::anim::AnimError => "anim",
    crate::anmf::AnmfError => "anmf",
    crate::build::BuildError => "build",
    crate::vp8_chunk::WebpLossyError => "vp8 chunk",
    crate::vp8l_chunk::WebpLosslessError => "vp8l chunk",
    crate::vp8l_stream::TransformListError => "vp8l transform list",
    crate::vp8l_prefix::PrefixError => "vp8l prefix code",
    crate::meta_prefix::MetaPrefixError => "vp8l meta prefix",
    crate::vp8l_decode::DecodeError => "vp8l decode",
    crate::vp8l_encode::EncodeError => "vp8l encode",
}

/// The `oxideav-vp8` keyframe decoder refuses an inter-frame
/// ([`oxideav_vp8::DecodeError::Unsupported`]) — a recognised-but-unsupported
/// feature, not a corrupt bitstream. A frame declaring more pixels than the
/// caller's cap is a limit, and everything else is a bitstream problem.
impl From<oxideav_vp8::DecodeError> for WebpError {
    fn from(e: oxideav_vp8::DecodeError) -> Self {
        match e {
            oxideav_vp8::DecodeError::Unsupported(_) => Self::Unsupported(format!("vp8: {e}")),
            oxideav_vp8::DecodeError::FrameTooLarge { .. } => {
                Self::LimitExceeded(format!("vp8: {e}"))
            }
            _ => Self::InvalidData(format!("vp8: {e}")),
        }
    }
}

/// The `oxideav-vp8` umbrella error maps 1:1 onto the four shared variants.
impl From<oxideav_vp8::Vp8Error> for WebpError {
    fn from(e: oxideav_vp8::Vp8Error) -> Self {
        match e {
            oxideav_vp8::Vp8Error::InvalidData(m) => Self::InvalidData(format!("vp8: {m}")),
            oxideav_vp8::Vp8Error::Unsupported(m) => Self::Unsupported(format!("vp8: {m}")),
            oxideav_vp8::Vp8Error::Eof => Self::Eof,
            oxideav_vp8::Vp8Error::NeedMore => Self::NeedMore,
        }
    }
}

/// The `oxideav-vp8` keyframe encoder's failures are all "cannot encode
/// this input" reports.
impl From<oxideav_vp8::encoder::EncodeError> for WebpError {
    fn from(e: oxideav_vp8::encoder::EncodeError) -> Self {
        Self::InvalidData(format!("vp8 encode: {e}"))
    }
}

/// Result alias for the crate's entry points.
///
/// Equivalent to `core::result::Result<T, oxideav_webp::WebpError>`. Not
/// re-exported at the crate root (the crate's own source uses the std
/// two-parameter `Result` throughout); reach it as
/// `oxideav_webp::error::Result`.
pub type Result<T> = core::result::Result<T, WebpError>;
