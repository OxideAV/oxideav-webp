//! # oxideav-webp
//!
//! Pure-Rust WebP image codec — clean-room scaffold built against
//! RFC 9649 (WebP Image Format).
//!
//! Round 1 landed the **structural** RIFF/WEBP container walker
//! ([`container::parse`]). Round 2 added typed field decoding for the
//! `VP8X` extended-format header ([`vp8x::Vp8xHeader::parse`]). Round 3
//! adds typed field decoding for the two §2.7.1.1 / §2.7.1.2 metadata
//! chunks that travel alongside VP8X:
//!
//! * [`alph::AlphHeader::parse`] — the `ALPH` info byte
//!   (`Rsv|P|F|C`).
//! * [`anim::AnimHeader::parse`] — the `ANIM` 6-byte payload
//!   (BGRA background colour + u16 loop count).
//!
//! `VP8 ` / `VP8L` bitstream decode and the actual ALPH alpha
//! bitstream remain stubs returning [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

pub mod alph;
pub mod anim;
pub mod container;
pub mod vp8x;

#[cfg(feature = "registry")]
use oxideav_core::RuntimeContext;

/// Crate-local error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A code path that has not been wired up yet in this round.
    NotImplemented,
    /// The RIFF/WEBP container walker rejected the input.
    Container(container::ContainerError),
    /// The §2.7.1 VP8X chunk parser rejected the input.
    Vp8x(vp8x::Vp8xError),
    /// The §2.7.1.2 ALPH info-byte parser rejected the input.
    Alph(alph::AlphError),
    /// The §2.7.1.1 ANIM payload parser rejected the input.
    Anim(anim::AnimError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-webp: pixel decode not implemented yet"),
            Self::Container(e) => write!(f, "oxideav-webp container: {e}"),
            Self::Vp8x(e) => write!(f, "oxideav-webp vp8x: {e}"),
            Self::Alph(e) => write!(f, "oxideav-webp alph: {e}"),
            Self::Anim(e) => write!(f, "oxideav-webp anim: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<container::ContainerError> for Error {
    fn from(e: container::ContainerError) -> Self {
        Self::Container(e)
    }
}

impl From<vp8x::Vp8xError> for Error {
    fn from(e: vp8x::Vp8xError) -> Self {
        Self::Vp8x(e)
    }
}

impl From<alph::AlphError> for Error {
    fn from(e: alph::AlphError) -> Self {
        Self::Alph(e)
    }
}

impl From<anim::AnimError> for Error {
    fn from(e: anim::AnimError) -> Self {
        Self::Anim(e)
    }
}

/// Walk a `RIFF/WEBP` container per RFC 9649 §2.3–§2.7 and return
/// the structural chunk list. This is the round-1 surface: it does
/// not decode any payload.
pub fn parse_container(bytes: &[u8]) -> Result<container::WebpContainer, Error> {
    container::parse(bytes).map_err(Into::into)
}

/// Decode the §2.7.1 `VP8X` chunk payload to a typed
/// [`vp8x::Vp8xHeader`].
///
/// The argument is the **payload** of a `VP8X` chunk — exactly the
/// 10 bytes following the 8-byte chunk header. The recommended call
/// pattern is to walk the container first, locate the chunk whose
/// FourCC is [`container::fourcc::VP8X`], borrow its payload via
/// [`container::WebpChunk::payload`], and hand that slice to this
/// function.
pub fn parse_vp8x_header(payload: &[u8]) -> Result<vp8x::Vp8xHeader, Error> {
    vp8x::Vp8xHeader::parse(payload).map_err(Into::into)
}

/// Decode the §2.7.1.2 `ALPH` chunk info byte to a typed
/// [`alph::AlphHeader`].
///
/// The argument is the **payload** of an `ALPH` chunk — i.e. the
/// slice returned by [`container::WebpChunk::payload`] for a chunk
/// whose FourCC is [`container::fourcc::ALPH`]. Only the first byte
/// is consumed by this layer; the rest of the payload is the alpha
/// bitstream proper, which is not decoded here.
pub fn parse_alph_header(payload: &[u8]) -> Result<alph::AlphHeader, Error> {
    alph::AlphHeader::parse(payload).map_err(Into::into)
}

/// Decode the §2.7.1.1 `ANIM` chunk payload to a typed
/// [`anim::AnimHeader`].
///
/// The argument is the 6-byte chunk payload — the BGRA background
/// colour followed by the little-endian u16 loop count.
pub fn parse_anim_header(payload: &[u8]) -> Result<anim::AnimHeader, Error> {
    anim::AnimHeader::parse(payload).map_err(Into::into)
}

/// Decode a WebP file to pixels.
///
/// Returns [`Error::NotImplemented`] — rounds 1 through 3 only ship
/// the structural plus header-field parsers (`container`, `vp8x`,
/// `alph`, `anim`). Pixel decode (`VP8 ` / `VP8L` plus the actual ALPH
/// alpha bitstream) is scheduled for later rounds.
pub fn decode_webp(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the round-1 scaffold has no decoder
/// to register into the runtime context.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("webp", register);
