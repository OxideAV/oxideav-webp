//! # oxideav-webp
//!
//! Pure-Rust WebP image codec — clean-room round-1 scaffold built
//! against RFC 9649 (WebP Image Format).
//!
//! Round 1 lands only the **structural** RIFF/WEBP container walker
//! ([`container::parse`]); the `VP8 ` / `VP8L` bitstreams and the
//! extended-format `VP8X` field decoding remain stubs that return
//! [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

pub mod container;

#[cfg(feature = "registry")]
use oxideav_core::RuntimeContext;

/// Crate-local error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A code path that has not been wired up yet in this round.
    NotImplemented,
    /// The RIFF/WEBP container walker rejected the input.
    Container(container::ContainerError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-webp: pixel decode not implemented yet"),
            Self::Container(e) => write!(f, "oxideav-webp container: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<container::ContainerError> for Error {
    fn from(e: container::ContainerError) -> Self {
        Self::Container(e)
    }
}

/// Walk a `RIFF/WEBP` container per RFC 9649 §2.3–§2.7 and return
/// the structural chunk list. This is the round-1 surface: it does
/// not decode any payload.
pub fn parse_container(bytes: &[u8]) -> Result<container::WebpContainer, Error> {
    container::parse(bytes).map_err(Into::into)
}

/// Decode a WebP file to pixels.
///
/// Returns [`Error::NotImplemented`] — round 1 only ships the
/// structural walker; pixel decode (`VP8 ` / `VP8L` / `ALPH`) is
/// scheduled for a later round.
pub fn decode_webp(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the round-1 scaffold has no decoder
/// to register into the runtime context.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("webp", register);
