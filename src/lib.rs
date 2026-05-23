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
//! per-frame §2.7.1.1 `ANMF` header. Round 5 added the **builder**
//! side of the RIFF/WEBP container — the inverse of the walker — so
//! external encoders can wrap a `VP8 ` / `VP8L` payload in a
//! well-formed file. Round 6 adds a typed §2.5 `VP8 ` chunk handle
//! ([`vp8_chunk::WebpLossyChunk`]) that lets container-layer callers
//! route the VP8 payload to a downstream VP8 decoder **without**
//! `oxideav-webp` taking a runtime dependency on `oxideav-vp8`.
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
//! * [`vp8_chunk::WebpLossyChunk`] — typed §2.5 `VP8 ` chunk
//!   handle. Peeks the RFC 6386 §9.1 keyframe header (width /
//!   height / version / first_partition_size / scale fields) and
//!   exposes the chunk payload via [`vp8_chunk::WebpLossyChunk::bitstream`]
//!   for routing to an external VP8 decoder.
//! * [`vp8l_chunk::WebpLosslessChunk`] — typed §2.6 `VP8L` chunk
//!   handle. Peeks the §3.4 / §7.1 5-byte VP8L image-header
//!   (`0x2F` signature + 14-bit `width-1` + 14-bit `height-1` +
//!   `alpha_is_used` bit + 3-bit `version`) and exposes the chunk
//!   payload via [`vp8l_chunk::WebpLosslessChunk::bitstream`] for
//!   routing to an external VP8L decoder.
//! * [`vp8l_stream::TransformList`] — the §4 transform-presence loop
//!   (round 99): each present transform's leading fixed fields, stopping
//!   at the first §5 entropy-coded body.
//! * [`vp8l_prefix::PrefixCode`] — the §6.2.1 prefix-code reader
//!   (round 104): reads a single canonical prefix code's lengths off
//!   the wire (simple or normal code length code) and decodes symbols
//!   one at a time. This is the first piece of the §5 / §6 entropy
//!   machinery the §4 transform bodies and the main image stream both
//!   consume.
//! * [`meta_prefix::MetaPrefixHeader`] — the §5.2.3 color-cache info,
//!   §6.2.2 meta-prefix dispatch, and §6.2 5-prefix-code-group reader
//!   (round 106). Surfaces either a fully-built single
//!   [`meta_prefix::PrefixCodeGroup`] (the common case: single
//!   meta-Huffman group, or any non-ARGB role) or, when an ARGB image
//!   selects an entropy image, the entropy-image dimensions plus the
//!   bit position at which the §5.2-encoded entropy image starts (for
//!   the next round to resume from once §5.2 LZ77 + color-cache decode
//!   lands).
//! * [`vp8l_decode::decode_image`] — the §5.2 LZ77 backward-reference +
//!   §5.2.3 color-cache per-pixel ARGB decode loop (round 107). Runs
//!   the §6.2.3 GREEN symbol dispatch (literal / LZ77 length+distance /
//!   color-cache code) over a single [`meta_prefix::PrefixCodeGroup`]
//!   and produces a [`vp8l_decode::DecodedImage`] of ARGB pixels in
//!   scan-line order (before any §4 inverse transform). Includes the
//!   §5.2.2 prefix→value transform, the 120-element distance map, and
//!   the §5.2.3 `0x1e35a7bd` color cache.
//!
//! `VP8 ` / `VP8L` bitstream decode and the actual ALPH alpha
//! bitstream remain stubs returning [`Error::NotImplemented`]; the
//! builders are deliberately framing-only so an external encoder can
//! pre-compute the codec payload bytes. The round-6 lossy handle and
//! the round-7 lossless handle are likewise framing-only — they
//! surface canvas dims and the routing slice but perform no VP8 /
//! VP8L bitstream decode.

#![warn(missing_debug_implementations)]

pub mod alph;
pub mod anim;
pub mod anmf;
pub mod build;
pub mod container;
pub mod meta_prefix;
pub mod vp8_chunk;
pub mod vp8l_chunk;
pub mod vp8l_decode;
pub mod vp8l_prefix;
pub mod vp8l_stream;
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
    /// The §2.5 typed `VP8 ` chunk handle rejected the chunk payload.
    Lossy(vp8_chunk::WebpLossyError),
    /// The §2.6 typed `VP8L` chunk handle rejected the chunk payload.
    Lossless(vp8l_chunk::WebpLosslessError),
    /// The §4 VP8L transform-list reader rejected the bitstream.
    Vp8lTransform(vp8l_stream::TransformListError),
    /// The §6.2.1 VP8L prefix-code reader rejected the bitstream.
    Vp8lPrefix(vp8l_prefix::PrefixError),
    /// The §5.2.3 / §6.2.2 VP8L meta-prefix header reader rejected the
    /// bitstream.
    Vp8lMetaPrefix(meta_prefix::MetaPrefixError),
    /// The §5.2 VP8L per-pixel ARGB decode loop rejected the bitstream.
    Vp8lDecode(vp8l_decode::DecodeError),
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
            Self::Lossy(e) => write!(f, "oxideav-webp lossy: {e}"),
            Self::Lossless(e) => write!(f, "oxideav-webp lossless: {e}"),
            Self::Vp8lTransform(e) => write!(f, "oxideav-webp vp8l-transform: {e}"),
            Self::Vp8lPrefix(e) => write!(f, "oxideav-webp vp8l-prefix: {e}"),
            Self::Vp8lMetaPrefix(e) => write!(f, "oxideav-webp vp8l-meta-prefix: {e}"),
            Self::Vp8lDecode(e) => write!(f, "oxideav-webp vp8l-decode: {e}"),
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

impl From<vp8_chunk::WebpLossyError> for Error {
    fn from(e: vp8_chunk::WebpLossyError) -> Self {
        Self::Lossy(e)
    }
}

impl From<vp8l_chunk::WebpLosslessError> for Error {
    fn from(e: vp8l_chunk::WebpLosslessError) -> Self {
        Self::Lossless(e)
    }
}

impl From<vp8l_stream::TransformListError> for Error {
    fn from(e: vp8l_stream::TransformListError) -> Self {
        Self::Vp8lTransform(e)
    }
}

impl From<vp8l_prefix::PrefixError> for Error {
    fn from(e: vp8l_prefix::PrefixError) -> Self {
        Self::Vp8lPrefix(e)
    }
}

impl From<meta_prefix::MetaPrefixError> for Error {
    fn from(e: meta_prefix::MetaPrefixError) -> Self {
        Self::Vp8lMetaPrefix(e)
    }
}

impl From<vp8l_decode::DecodeError> for Error {
    fn from(e: vp8l_decode::DecodeError) -> Self {
        Self::Vp8lDecode(e)
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

/// Walk a `RIFF/WEBP` buffer and, if it carries a §2.5 simple-lossy
/// `VP8 ` chunk (or a §2.7 extended-lossy file with a `VP8 ` chunk
/// alongside `VP8X`), return a typed [`vp8_chunk::WebpLossyChunk`]
/// handle whose [`bitstream`](vp8_chunk::WebpLossyChunk::bitstream)
/// slice can be routed to an external VP8 decoder.
///
/// Returns `Ok(None)` if the file is well-formed but carries no
/// `VP8 ` chunk (e.g. a `VP8L`-only simple-lossless file).
///
/// The returned handle borrows out of `bytes`, so the slice must
/// outlive the handle.
///
/// This is the round-6 routing API — `oxideav-webp` deliberately
/// does **not** take a runtime dependency on `oxideav-vp8`; the
/// caller picks which VP8 decoder consumes the borrowed payload.
pub fn extract_lossy_chunk(bytes: &[u8]) -> Result<Option<vp8_chunk::WebpLossyChunk<'_>>, Error> {
    let c = container::parse(bytes)?;
    vp8_chunk::extract_lossy(bytes, &c).map_err(Into::into)
}

/// Walk a `RIFF/WEBP` buffer and, if it carries a §2.6 simple-lossless
/// `VP8L` chunk (or a §2.7 extended-lossless file with a `VP8L` chunk
/// alongside `VP8X`), return a typed [`vp8l_chunk::WebpLosslessChunk`]
/// handle whose [`bitstream`](vp8l_chunk::WebpLosslessChunk::bitstream)
/// slice can be routed to an external VP8L decoder.
///
/// Returns `Ok(None)` if the file is well-formed but carries no
/// `VP8L` chunk (e.g. a `VP8 `-only simple-lossy file).
///
/// The returned handle borrows out of `bytes`, so the slice must
/// outlive the handle.
///
/// This is the round-7 routing API — `oxideav-webp` deliberately
/// does **not** take a runtime dependency on a VP8L decoder; the
/// caller picks which lossless-WebP decoder consumes the borrowed
/// payload.
pub fn extract_lossless_chunk(
    bytes: &[u8],
) -> Result<Option<vp8l_chunk::WebpLosslessChunk<'_>>, Error> {
    let c = container::parse(bytes)?;
    vp8l_chunk::extract_lossless(bytes, &c).map_err(Into::into)
}

/// Walk a `RIFF/WEBP` buffer, extract its §2.6 / §3.4 `VP8L` chunk,
/// and read the §4 transform-presence list that follows the 5-byte
/// VP8L image-header.
///
/// Returns `Ok(None)` if the file carries no `VP8L` chunk. Otherwise
/// returns the parsed [`vp8l_stream::TransformList`] — the transforms
/// in read order plus the bit position where the §5 entropy-coded
/// image data (or the first transform's §5 body) begins.
///
/// This is the round-99 surface: it reads each transform's leading
/// fixed-size fields (predictor / color `size_bits`, color-indexing
/// `color_table_size`) but does **not** decode the §5 entropy-coded
/// transform bodies or image data — those are returned-to boundaries
/// for the next layer.
pub fn read_vp8l_transform_list(bytes: &[u8]) -> Result<Option<vp8l_stream::TransformList>, Error> {
    let c = container::parse(bytes)?;
    let chunk = match vp8l_chunk::extract_lossless(bytes, &c)? {
        Some(chunk) => chunk,
        None => return Ok(None),
    };
    let mut reader = vp8l_stream::BitReader::new_after_image_header(chunk.bitstream());
    let list = vp8l_stream::TransformList::read(&mut reader)?;
    Ok(Some(list))
}

/// Decode a WebP file to pixels.
///
/// Returns [`Error::NotImplemented`] — rounds 1 through 7 only ship
/// the structural plus header-field parsers (`container`, `vp8x`,
/// `alph`, `anim`, `anmf`, `vp8_chunk`, `vp8l_chunk`) plus the
/// round-5 builder. Pixel decode (`VP8 ` / `VP8L` plus the actual
/// ALPH alpha bitstream) is the responsibility of downstream
/// decoder crates; callers use [`extract_lossy_chunk`] /
/// [`extract_lossless_chunk`] to route the bitstream bytes onward.
pub fn decode_webp(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the round-1 scaffold has no decoder
/// to register into the runtime context.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("webp", register);
