//! # oxideav-webp
//!
//! Pure-Rust WebP image codec — clean-room scaffold built against
//! RFC 9649 (WebP Image Format).
//!
//! Round 1 landed the **structural** RIFF/WEBP container walker
//! ([`container::parse`]). Round 2 added typed field decoding for the
//! `VP8X` extended-format header ([`vp8x::Vp8xHeader::parse`]). Round 3
//! added typed field decoding for the §2.7.1.1 `ANIM` / §2.7.1.2 `ALPH`
//! metadata chunks. Round 4 added typed field decoding for the
//! per-frame §2.7.1.1 `ANMF` header. Round 5 adds the **builder**
//! side of the RIFF/WEBP container — the inverse of the walker — so
//! external encoders can wrap a `VP8 ` / `VP8L` payload in a
//! well-formed file:
//!
//! * [`alph::AlphHeader::parse`] — the `ALPH` info byte
//!   (`Rsv|P|F|C`).
//! * [`anim::AnimHeader::parse`] — the `ANIM` 6-byte payload
//!   (BGRA background colour + u16 loop count).
//! * [`anmf::AnmfHeader::parse`] — the `ANMF` 16-byte per-frame
//!   header (frame X / Y / width / height / duration plus
//!   `Reserved|B|D` info byte).
//! * [`build::build_chunk`] — generic §2.3 chunk writer (FourCC +
//!   Size + payload + odd-size pad).
//! * [`build::build_vp8x_chunk`] — §2.7.1 Figure 7 typed VP8X
//!   payload writer.
//! * [`build::build_webp_file`] — §2.4 file writer for simple
//!   (`VP8 ` / `VP8L`) and extended (`VP8X` + `VP8 ` / `VP8L`)
//!   layouts.
//!
//! `VP8 ` / `VP8L` bitstream decode and the actual ALPH alpha
//! bitstream remain stubs returning [`Error::NotImplemented`]; the
//! builders are deliberately framing-only so an external encoder can
//! pre-compute the codec payload bytes.

#![warn(missing_debug_implementations)]

pub mod alph;
pub mod anim;
pub mod anmf;
pub mod build;
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
    /// The §2.7.1.1 ANMF per-frame header parser rejected the input.
    Anmf(anmf::AnmfError),
    /// The §2.3 / §2.4 / §2.7.1 RIFF/WEBP builders rejected the input.
    Build(build::BuildError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-webp: pixel decode not implemented yet"),
            Self::Container(e) => write!(f, "oxideav-webp container: {e}"),
            Self::Vp8x(e) => write!(f, "oxideav-webp vp8x: {e}"),
            Self::Alph(e) => write!(f, "oxideav-webp alph: {e}"),
            Self::Anim(e) => write!(f, "oxideav-webp anim: {e}"),
            Self::Anmf(e) => write!(f, "oxideav-webp anmf: {e}"),
            Self::Build(e) => write!(f, "oxideav-webp build: {e}"),
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

impl From<anmf::AnmfError> for Error {
    fn from(e: anmf::AnmfError) -> Self {
        Self::Anmf(e)
    }
}

impl From<build::BuildError> for Error {
    fn from(e: build::BuildError) -> Self {
        Self::Build(e)
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

/// Decode the §2.7.1.1 `ANMF` per-frame header to a typed
/// [`anmf::AnmfHeader`].
///
/// The argument is the **payload** of an `ANMF` chunk — the slice
/// returned by [`container::WebpChunk::payload`] for a chunk whose
/// FourCC is [`container::fourcc::ANMF`]. Only the first 16 bytes
/// are consumed; the remainder is the per-frame `Frame Data`
/// sub-RIFF, which is not decoded here.
pub fn parse_anmf_header(payload: &[u8]) -> Result<anmf::AnmfHeader, Error> {
    anmf::AnmfHeader::parse(payload).map_err(Into::into)
}

/// Assemble a `RIFF/WEBP` file around a single bitstream payload per
/// RFC 9649 §2.4 + §2.5 / §2.6 / §2.7. Convenience wrapper over
/// [`build::build_webp_file`] returning the crate-wide [`Error`].
pub fn build_webp_file(
    payload: &[u8],
    image_kind: build::ImageKind,
    canvas_width: u32,
    canvas_height: u32,
) -> Result<Vec<u8>, Error> {
    build::build_webp_file(payload, image_kind, canvas_width, canvas_height).map_err(Into::into)
}

/// Build the 10-byte §2.7.1 `VP8X` chunk payload (flags + reserved +
/// canvas dims). Convenience wrapper over [`build::build_vp8x_chunk`]
/// returning the crate-wide [`Error`].
pub fn build_vp8x_chunk(
    canvas_width: u32,
    canvas_height: u32,
    flags: build::Vp8xFlags,
) -> Result<Vec<u8>, Error> {
    build::build_vp8x_chunk(canvas_width, canvas_height, flags).map_err(Into::into)
}

/// Decode a WebP file to pixels.
///
/// Returns [`Error::NotImplemented`] — rounds 1 through 4 only ship
/// the structural plus header-field parsers (`container`, `vp8x`,
/// `alph`, `anim`, `anmf`). Pixel decode (`VP8 ` / `VP8L` plus the
/// actual ALPH alpha bitstream) is scheduled for later rounds.
pub fn decode_webp(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the round-1 scaffold has no decoder
/// to register into the runtime context.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("webp", register);
