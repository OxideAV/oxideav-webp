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
//! * [`alph::decode_alpha`] — the §2.7.1.2 alpha-bitstream decode
//!   (round 110): both compression methods (raw + headerless VP8L,
//!   the latter lifting alpha from the GREEN channel) and the four
//!   inverse filters (none / horizontal / vertical / gradient) with
//!   the documented left-most / top-most edge cases, producing the
//!   full-resolution alpha plane. [`decode_alpha_plane`] is the
//!   container-level entry point: walk the file, take dimensions from
//!   `VP8X` (or the `VP8 ` keyframe), find the `ALPH` chunk, decode.
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
//! * [`vp8l_decode::decode_argb`] — the §6.2.2 multi-group ARGB decode
//!   (round 108). Reads the round-106 [`meta_prefix::MetaPrefixHeader`]
//!   for the ARGB role and, when the meta-prefix bit selects multiple
//!   groups, decodes the §6.2.2 *entropy image*
//!   ([`vp8l_decode::decode_entropy_image`] →
//!   [`vp8l_decode::MetaPrefixIndex`]), derives
//!   `num_prefix_groups = max(entropy image) + 1`, reads that many
//!   prefix-code groups, and runs the §6.2.3 loop selecting a group per
//!   pixel block via
//!   `meta_index[(y >> prefix_bits) * block_width + (x >> prefix_bits)]`.
//!   Single-group images degrade to the round-107 path. Per §6.2.2 each
//!   block's meta-prefix code is the red+green channels of its
//!   entropy-image pixel (`(argb >> 8) & 0xffff`).
//! * [`vp8l_transform::decode_lossless`] — the §4 inverse-transform
//!   passes (round 109). Reads the §4 transform list (each transform's
//!   fixed fields **and** its §5-encoded body), decodes the main ARGB
//!   image at the (color-indexing-subsampled) width, then applies the
//!   four inverse transforms in reverse read order: §4.1 predictor (14
//!   prediction modes + border rules over the block grid), §4.2 color
//!   (per-block `ColorTransformElement` add-back), §4.3 subtract-green
//!   (add green into red/blue), and §4.4 color-indexing (palette lookup
//!   plus ≤16-color pixel un-bundling). The container-level entry point,
//!   [`decode_lossless_image`], walks the file, extracts the `VP8L`
//!   chunk, and decodes to a [`vp8l_decode::DecodedImage`]. Bit-exact
//!   against the `lossless-1x1`, `lossless-color-indexing-paletted`, and
//!   `lossless-32x32-rgba` (SUBTRACT_GREEN + PREDICTOR + CROSS_COLOR +
//!   color cache) fixture PNGs.
//!
//! * [`decode_webp_image`] / [`decode_webp`] — the top-level still-image
//!   entry points (round 111). They walk the container, decode a §2.6 /
//!   §3.4 `VP8L` lossless image (simple or `VP8X`-extended) through the
//!   full §4–§6 chain, optionally override its alpha from a §2.7.1.2
//!   `ALPH` chunk, and return interleaved 8-bit `[R, G, B, A]` pixels
//!   ([`DecodedWebp`]) — the `oxideav_core::PixelFormat::Rgba` layout
//!   the workspace's image crates share. A §2.5 `VP8 ` lossy file is a
//!   clean [`Error::Unsupported`]`(`[`UnsupportedKind::LossyVp8`]`)` —
//!   route it onward with [`extract_lossy_chunk`].
//!
//! `VP8 ` lossy bitstream decode is not performed in this crate; the
//! builders are deliberately framing-only so an external encoder can
//! pre-compute the codec payload bytes, and the round-6 lossy handle is
//! likewise framing-only — it surfaces canvas dims and the routing slice
//! but performs no VP8 bitstream decode. The §2.7.1.2 ALPH alpha
//! bitstream **is** decoded end-to-end ([`alph::decode_alpha`] /
//! [`decode_alpha_plane`]).

#![warn(missing_debug_implementations)]

pub mod alph;
pub mod anim;
pub mod anim_encode;
pub mod anmf;
pub mod build;
pub mod container;
pub mod meta_prefix;
#[cfg(feature = "registry")]
pub mod registry;
pub mod vp8_chunk;
pub mod vp8l_chunk;
pub mod vp8l_decode;
pub mod vp8l_encode;
pub mod vp8l_prefix;
pub mod vp8l_stream;
pub mod vp8l_transform;
pub mod vp8x;

#[cfg(feature = "registry")]
use oxideav_core::RuntimeContext;

/// Crate-local error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A code path that has not been wired up yet in this round.
    NotImplemented,
    /// The file is well-formed but carries an image kind this crate does
    /// not decode yet. Currently this is the §2.5 `VP8 ` lossy
    /// bitstream — routed out via [`extract_lossy_chunk`] to a downstream
    /// VP8 decoder rather than decoded here.
    Unsupported(UnsupportedKind),
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
    /// The §3.7 / §3.8 VP8L lossless encoder rejected the input.
    Vp8lEncode(vp8l_encode::EncodeError),
}

/// Which image kind [`decode_webp`] declined to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedKind {
    /// The file's image data is a §2.5 `VP8 ` lossy bitstream. This
    /// crate decodes only the §2.6 `VP8L` lossless bitstream so far; the
    /// lossy payload is meant to be routed to a downstream VP8 decoder
    /// via [`extract_lossy_chunk`].
    LossyVp8,
    /// The file carries neither a `VP8L` nor a `VP8 ` image-data chunk
    /// (e.g. an animation: the pixels live inside per-frame `ANMF`
    /// sub-RIFFs, which this still-image entry point does not assemble).
    NoImageData,
}

impl core::fmt::Display for UnsupportedKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LossyVp8 => f.write_str("VP8 lossy bitstream (route to a VP8 decoder)"),
            Self::NoImageData => {
                f.write_str("no VP8L/VP8 image-data chunk (animation or header-only)")
            }
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("oxideav-webp: pixel decode not implemented yet"),
            Self::Unsupported(k) => write!(f, "oxideav-webp: unsupported image kind: {k}"),
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
            Self::Vp8lEncode(e) => write!(f, "oxideav-webp vp8l-encode: {e}"),
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

impl From<vp8l_encode::EncodeError> for Error {
    fn from(e: vp8l_encode::EncodeError) -> Self {
        Self::Vp8lEncode(e)
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
/// bitstream proper, which is decoded by [`alph::decode_alpha`].
pub fn parse_alph_header(payload: &[u8]) -> Result<alph::AlphHeader, Error> {
    alph::AlphHeader::parse(payload).map_err(Into::into)
}

/// Walk a `RIFF/WEBP` buffer and, if it carries a §2.7.1.2 `ALPH`
/// chunk, fully decode the alpha bitstream to a `width * height` plane
/// of 8-bit alpha values in scan order.
///
/// The alpha-plane dimensions are taken from the file in this priority
/// order, matching how a still image carries its canvas size:
///
/// 1. the §2.7.1 `VP8X` canvas dimensions, if a `VP8X` chunk exists;
/// 2. otherwise the §2.5 `VP8 ` keyframe dimensions (a simple-lossy
///    file with an `ALPH` chunk but no `VP8X`).
///
/// Returns `Ok(None)` if the file is well-formed but carries no `ALPH`
/// chunk. The decode covers both §2.7.1.2 compression methods
/// (raw + VP8L-lossless) and all four filtering methods — see
/// [`alph::decode_alpha`].
///
/// This handles the **still-image** alpha path. Per-frame (`ANMF`)
/// alpha planes are addressed by walking the `ANMF` frame data with
/// [`alph::decode_alpha`] directly, using the frame dimensions.
pub fn decode_alpha_plane(bytes: &[u8]) -> Result<Option<Vec<u8>>, Error> {
    let c = container::parse(bytes)?;
    let alph_chunk = match c.first_chunk_with_fourcc(container::fourcc::ALPH) {
        Some(chunk) => chunk,
        None => return Ok(None),
    };

    // Dimensions: VP8X canvas first, else the VP8 keyframe header.
    let (width, height) = if let Some(vp8x) = c.first_chunk_with_fourcc(container::fourcc::VP8X) {
        let hdr = vp8x::Vp8xHeader::parse(vp8x.payload(bytes))?;
        (hdr.canvas_width, hdr.canvas_height)
    } else if let Some(vp8) = c.first_chunk_with_fourcc(container::fourcc::VP8) {
        let lossy = vp8_chunk::WebpLossyChunk::from_chunk(bytes, vp8)?;
        (u32::from(lossy.width()), u32::from(lossy.height()))
    } else {
        // No dimension source — an ALPH with neither VP8X nor VP8 is
        // not a shape RFC 9649 §2.5/§2.7 describes for still images.
        return Err(Error::Alph(alph::AlphError::EmptyPayload));
    };

    let plane = alph::decode_alpha(alph_chunk.payload(bytes), width, height)?;
    Ok(Some(plane))
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

/// Walk a `RIFF/WEBP` buffer, extract its §2.6 / §3.4 `VP8L` chunk, and
/// fully decode it to ARGB pixels.
///
/// This runs the round-108 §5/§6 entropy decode of the main ARGB image
/// then applies the round-109 §4 inverse-transform chain
/// ([`vp8l_transform::decode_lossless`]): predictor, color, subtract-green,
/// and color-indexing, applied in reverse of the order the transforms
/// were read.
///
/// Returns `Ok(None)` if the file carries no `VP8L` chunk. Otherwise the
/// returned [`vp8l_decode::DecodedImage`] holds `width * height` ARGB
/// pixels in scan-line order, each `(alpha << 24) | (red << 16) |
/// (green << 8) | blue`.
pub fn decode_lossless_image(bytes: &[u8]) -> Result<Option<vp8l_decode::DecodedImage>, Error> {
    let c = container::parse(bytes)?;
    let chunk = match vp8l_chunk::extract_lossless(bytes, &c)? {
        Some(chunk) => chunk,
        None => return Ok(None),
    };
    let width = chunk.width();
    let height = chunk.height();
    let image = vp8l_transform::decode_lossless(chunk.bitstream(), width, height)?;
    Ok(Some(image))
}

/// A fully decoded still WebP image: 8-bit RGBA pixels plus dimensions.
///
/// `rgba` is `width * height * 4` bytes in scan-line (top-to-bottom,
/// left-to-right) order, each pixel laid out `[R, G, B, A]`. This is the
/// canonical interleaved-RGBA surface
/// (`oxideav_core::PixelFormat::Rgba`) the workspace's image crates
/// emit, so a `VideoFrame` wrapper is a single 1-plane copy away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedWebp {
    /// Image width in pixels (the §2.7.1 `VP8X` canvas width, or the
    /// §3.4 `VP8L` image width for a simple-lossless file).
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `width * height * 4` interleaved `[R, G, B, A]` bytes, scan order.
    pub rgba: Vec<u8>,
}

/// Decode a still WebP file to a typed [`DecodedWebp`] (RGBA + dims).
///
/// Handles the two cases this crate can fully decode today:
///
/// 1. **Simple lossless** — a §2.6 `VP8L` chunk (optionally fronted by a
///    §2.7.1 `VP8X` header): decoded to ARGB via
///    [`vp8l_transform::decode_lossless`], with alpha carried inside the
///    `VP8L` bitstream itself.
/// 2. **Extended lossless** — a §2.7 `VP8X` file whose image data is a
///    `VP8L` chunk. If the (spec-discouraged, per RFC 9649 §2.7.1.2) case
///    of an accompanying §2.7.1.2 `ALPH` chunk is present, its decoded
///    alpha plane overrides the per-pixel alpha channel.
///
/// A §2.5 `VP8 ` lossy bitstream is **not** decoded here — it returns
/// [`Error::Unsupported`]`(`[`UnsupportedKind::LossyVp8`]`)`. Route it to
/// a downstream VP8 decoder with [`extract_lossy_chunk`] instead.
///
/// Animations and header-only files (no `VP8L`/`VP8 ` chunk) return
/// [`Error::Unsupported`]`(`[`UnsupportedKind::NoImageData`]`)`.
pub fn decode_webp_image(bytes: &[u8]) -> Result<DecodedWebp, Error> {
    let c = container::parse(bytes)?;

    // §2.6 / §3.4: the VP8L lossless image is the only pixel source this
    // crate decodes. Decode it (alpha is carried inside the VP8L stream).
    let vp8l = vp8l_chunk::extract_lossless(bytes, &c)?;
    let Some(chunk) = vp8l else {
        // No VP8L. A VP8 lossy chunk is recognized-but-unsupported here;
        // anything else has no still-image pixel data.
        if c.first_chunk_with_fourcc(container::fourcc::VP8).is_some() {
            return Err(Error::Unsupported(UnsupportedKind::LossyVp8));
        }
        return Err(Error::Unsupported(UnsupportedKind::NoImageData));
    };

    let width = chunk.width();
    let height = chunk.height();
    let mut image = vp8l_transform::decode_lossless(chunk.bitstream(), width, height)?;

    // §2.7.1.2: an ALPH chunk alongside a VP8L image is discouraged by
    // the spec ("A frame containing a 'VP8L' Chunk SHOULD NOT contain
    // this chunk"), but is not forbidden. When present, its decoded alpha
    // plane overrides the VP8L per-pixel alpha. The plane dimensions come
    // from the VP8X canvas, which for a well-formed file equals the VP8L
    // image dimensions.
    if let Some(alph) = c.first_chunk_with_fourcc(container::fourcc::ALPH) {
        let plane = alph::decode_alpha(alph.payload(bytes), width, height)?;
        let pixels = image.pixels_mut();
        if plane.len() == pixels.len() {
            for (px, &a) in pixels.iter_mut().zip(plane.iter()) {
                *px = (*px & 0x00ff_ffff) | (u32::from(a) << 24);
            }
        }
    }

    Ok(DecodedWebp {
        width,
        height,
        rgba: argb_to_rgba(image.pixels()),
    })
}

// ─────────────────────── Published-shape decode API ───────────────────────
//
// The free `decode_webp` path the published crates.io releases exposed —
// the flat, `image`-crate-compatible RGBA surface downstream consumers
// depend on. See `API-COMPAT.md`. The `WebpImage` / `WebpFrame` /
// `WebpFileMetadata` / `WebpError` shapes here are the published shapes;
// the round-115 `DecodedWebp` / `decode_webp_image` / `decode_lossless_image`
// helpers above are the rebuild's own low-level surface and stay as
// additional API.

/// A fully decoded WebP file: one frame for a still image, N frames for an
/// animation, plus the file-level metadata and animation parameters.
///
/// This is the published-API decode result. The single most important
/// consumer property is the flat-buffer shape of each [`WebpFrame::rgba`]:
/// `width * height * 4` tightly packed `[R, G, B, A]` bytes, no per-row
/// stride padding, so it drops straight into
/// `image::ImageBuffer::from_raw(width, height, rgba)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebpImage {
    /// Decoded frames. A still image yields exactly one frame; an
    /// animation yields one per `ANMF` chunk (animation decode is not
    /// rebuilt yet — see [`decode_webp`]).
    pub frames: Vec<WebpFrame>,
    /// File-level metadata (ICC / Exif / XMP), each `None` when absent.
    pub metadata: WebpFileMetadata,
    /// The §2.7.1.1 `ANIM` background colour as `[R, G, B, A]`, or `None`
    /// for a non-animated file.
    pub anim_background_rgba: Option<[u8; 4]>,
    /// The §2.7.1.1 `ANIM` loop count (`0` = loop forever), or `None` for
    /// a non-animated file.
    pub anim_loop_count: Option<u16>,
}

/// A single decoded WebP frame: a flat RGBA pixel buffer plus its size
/// and (for animations) its display duration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebpFrame {
    /// `width * height * 4` interleaved `[R, G, B, A]` bytes in scan-line
    /// (top-to-bottom, left-to-right) order, no stride padding.
    pub rgba: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Per-frame display duration in milliseconds (the §2.7.1.1 `ANMF`
    /// frame delay). `0` for a still image.
    pub duration_ms: u32,
}

/// File-level metadata chunks, each carrying the raw chunk payload bytes
/// when present.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WebpFileMetadata {
    /// §2.7.1.4 `ICCP` ICC color-profile payload, if present.
    pub icc: Option<Vec<u8>>,
    /// §2.7.1.5 `EXIF` Exif payload, if present.
    pub exif: Option<Vec<u8>>,
    /// §2.7.1.5 `XMP ` XMP payload, if present.
    pub xmp: Option<Vec<u8>>,
}

/// Borrowed file-level metadata for the encode side — the §2.7.1.4 `ICCP`,
/// §2.7.1.5 `EXIF`, and §2.7.1.5 `XMP ` payloads to embed, each `None` to
/// omit the corresponding chunk.
///
/// This is the borrowed form: the slices are not copied until the encoder
/// frames them. The owned counterpart is [`WebpMetadataOwned`]. The default
/// is all-`None` — embed no metadata — so a `VP8L` encode with
/// `WebpMetadata::default()` emits the simple (non-`VP8X`) layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebpMetadata<'a> {
    /// §2.7.1.4 `ICCP` ICC color-profile payload to embed, if any.
    pub icc: Option<&'a [u8]>,
    /// §2.7.1.5 `EXIF` Exif payload to embed, if any.
    pub exif: Option<&'a [u8]>,
    /// §2.7.1.5 `XMP ` XMP payload to embed, if any.
    pub xmp: Option<&'a [u8]>,
}

impl<'a> WebpMetadata<'a> {
    /// True if every field is `None` — encoding can stay on the simple
    /// (non-`VP8X`) layout when no alpha is present either.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none()
    }
}

/// Owned file-level metadata — the registry-side counterpart of the borrowed
/// [`WebpMetadata`].
///
/// Carries owned `Vec<u8>` payloads so it can be stored on an encoder /
/// codec-parameters struct without borrowing the caller's buffers. Convert
/// to the borrowed form for an encode call with [`Self::as_borrowed`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WebpMetadataOwned {
    /// §2.7.1.4 `ICCP` ICC color-profile payload to embed, if any.
    pub icc: Option<Vec<u8>>,
    /// §2.7.1.5 `EXIF` Exif payload to embed, if any.
    pub exif: Option<Vec<u8>>,
    /// §2.7.1.5 `XMP ` XMP payload to embed, if any.
    pub xmp: Option<Vec<u8>>,
}

impl WebpMetadataOwned {
    /// Borrow this owned metadata as a [`WebpMetadata`] for an encode call.
    pub fn as_borrowed(&self) -> WebpMetadata<'_> {
        WebpMetadata {
            icc: self.icc.as_deref(),
            exif: self.exif.as_deref(),
            xmp: self.xmp.as_deref(),
        }
    }

    /// True if every field is `None`.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none()
    }
}

impl From<WebpMetadataOwned> for WebpFileMetadata {
    fn from(m: WebpMetadataOwned) -> Self {
        WebpFileMetadata {
            icc: m.icc,
            exif: m.exif,
            xmp: m.xmp,
        }
    }
}

/// The published-API error type for the flat [`decode_webp`] /
/// [`extract_metadata`] decode paths.
///
/// This is intentionally coarse-grained — the stable shape downstream
/// consumers match on. The internal [`Error`] enum (with its per-module
/// variants) remains the richer surface for the low-level
/// [`decode_webp_image`] / [`decode_lossless_image`] helpers; it maps into
/// `WebpError` via the `From<Error>` impl.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebpError {
    /// The input is not a well-formed WebP file (bad magic, malformed
    /// chunk structure, a sub-decoder rejected the bitstream, …).
    InvalidData,
    /// The file is well-formed but carries an image kind this build does
    /// not decode yet — currently the §2.5 `VP8 ` lossy bitstream and
    /// animation frame assembly.
    Unsupported,
    /// The input ended before a complete image could be read.
    Eof,
    /// More input is required to complete the decode (streaming callers).
    NeedMore,
}

impl core::fmt::Display for WebpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::InvalidData => "oxideav-webp: invalid WebP data",
            Self::Unsupported => "oxideav-webp: unsupported WebP feature",
            Self::Eof => "oxideav-webp: unexpected end of input",
            Self::NeedMore => "oxideav-webp: more input required",
        };
        f.write_str(s)
    }
}

impl std::error::Error for WebpError {}

/// Map the rich internal [`Error`] onto the coarse published [`WebpError`].
///
/// `Unsupported` / `NotImplemented` collapse to [`WebpError::Unsupported`];
/// every other variant (a malformed container or a sub-decoder rejecting
/// the bitstream) is [`WebpError::InvalidData`].
impl From<Error> for WebpError {
    fn from(e: Error) -> Self {
        match e {
            Error::Unsupported(_) | Error::NotImplemented => WebpError::Unsupported,
            _ => WebpError::InvalidData,
        }
    }
}

/// Decode a WebP file to the published flat-RGBA [`WebpImage`] shape.
///
/// This is the `image`-crate-compatible entry point: every returned
/// [`WebpFrame::rgba`] is `width * height * 4` tightly packed `[R, G, B, A]`
/// bytes (no stride padding), so a frame wraps zero-copy as
/// `image::ImageBuffer::from_raw(frame.width, frame.height, frame.rgba)`.
///
/// Supported today (built on the rebuilt §4–§6 VP8L decoder):
///
/// * **Simple / extended lossless** (`VP8L`, optionally `VP8X`-fronted,
///   with optional `ALPH`-over-`VP8L` alpha) → a single-frame `WebpImage`.
///
/// Not yet rebuilt (returns [`WebpError::Unsupported`], never a fake
/// decode):
///
/// * **§2.5 `VP8 ` lossy** — needs the `oxideav-vp8` bitstream path.
/// * **Animation** — `ANMF` frame assembly.
///
/// The low-level [`decode_webp_image`] / [`decode_lossless_image`] helpers
/// expose the rebuild's internal [`DecodedWebp`] / [`vp8l_decode::DecodedImage`]
/// surfaces for callers that want them.
pub fn decode_webp(bytes: &[u8]) -> Result<WebpImage, WebpError> {
    // Parse the container once so we can read both pixels and metadata.
    let c = container::parse(bytes).map_err(|_| WebpError::InvalidData)?;

    // Animated file: an ANIM chunk introduces a sequence of ANMF frames.
    // Decode every frame's VP8L-lossless bitstream into a separate WebpFrame.
    if c.first_chunk_with_fourcc(container::fourcc::ANIM).is_some() {
        return decode_animation(bytes, &c);
    }

    // Pixel data: the rebuilt path decodes VP8L (simple or extended).
    // VP8 lossy and animation are reported Unsupported, not faked.
    let decoded = decode_webp_image(bytes).map_err(WebpError::from)?;

    let frame = WebpFrame {
        rgba: decoded.rgba,
        width: decoded.width,
        height: decoded.height,
        duration_ms: 0,
    };

    let metadata = metadata_from_container(bytes, &c);

    // Non-animated file: ANIM fields are None. (Animation assembly is not
    // rebuilt yet, so any animated file already errored Unsupported above
    // via the NoImageData path.)
    Ok(WebpImage {
        frames: vec![frame],
        metadata,
        anim_background_rgba: None,
        anim_loop_count: None,
    })
}

/// Decode an animated WebP (`ANIM` + per-frame `ANMF`) to a multi-frame
/// [`WebpImage`].
///
/// Each §2.7.1.1 `ANMF` chunk carries a 16-byte header
/// ([`anmf::AnmfHeader`]) followed by its "Frame Data" — a padded §2.3
/// sub-RIFF holding the frame's bitstream. This decoder handles the
/// §2.6 `VP8L` lossless sub-chunk (the path the in-crate animation encoder
/// produces); an `ANMF` carrying only a §2.5 `VP8 ` lossy sub-chunk is
/// [`WebpError::Unsupported`] (the VP8 lossy decoder is not rebuilt yet).
///
/// The §2.7.1.1 `ANIM` background colour and loop count populate
/// [`WebpImage::anim_background_rgba`] / [`WebpImage::anim_loop_count`]; the
/// per-frame `Frame Duration` populates each [`WebpFrame::duration_ms`].
fn decode_animation(bytes: &[u8], c: &container::WebpContainer) -> Result<WebpImage, WebpError> {
    // §2.7.1.1 ANIM: global background colour + loop count.
    let anim_chunk = c
        .first_chunk_with_fourcc(container::fourcc::ANIM)
        .ok_or(WebpError::InvalidData)?;
    let anim =
        anim::AnimHeader::parse(anim_chunk.payload(bytes)).map_err(|_| WebpError::InvalidData)?;
    let bg = anim.background_color;

    let mut frames = Vec::new();
    for anmf_chunk in c.chunks_with_fourcc(container::fourcc::ANMF) {
        let payload = anmf_chunk.payload(bytes);
        let header = anmf::AnmfHeader::parse(payload).map_err(|_| WebpError::InvalidData)?;
        let frame_data = &payload[header.frame_data_offset()..];

        // The Frame Data sub-RIFF is a flat list of §2.3 padded chunks. Find
        // the VP8L bitstream sub-chunk (lossy VP8 is not decoded here).
        let vp8l = find_subchunk(frame_data, container::fourcc::VP8L);
        let Some(vp8l_payload) = vp8l else {
            // A VP8 lossy sub-chunk is recognized-but-unsupported.
            if find_subchunk(frame_data, container::fourcc::VP8).is_some() {
                return Err(WebpError::Unsupported);
            }
            return Err(WebpError::InvalidData);
        };

        let chunk = vp8l_chunk::WebpLosslessChunk::from_payload(vp8l_payload)
            .map_err(|e| WebpError::from(Error::from(e)))?;
        let width = chunk.width();
        let height = chunk.height();
        let image = vp8l_transform::decode_lossless(chunk.bitstream(), width, height)
            .map_err(|e| WebpError::from(Error::from(e)))?;

        // An optional ALPH sub-chunk overrides the VP8L per-pixel alpha.
        let mut pixels = image;
        if let Some(alph_payload) = find_subchunk(frame_data, container::fourcc::ALPH) {
            if let Ok(plane) = alph::decode_alpha(alph_payload, width, height) {
                let px = pixels.pixels_mut();
                if plane.len() == px.len() {
                    for (p, &a) in px.iter_mut().zip(plane.iter()) {
                        *p = (*p & 0x00ff_ffff) | (u32::from(a) << 24);
                    }
                }
            }
        }

        frames.push(WebpFrame {
            rgba: argb_to_rgba(pixels.pixels()),
            width,
            height,
            duration_ms: header.duration_ms,
        });
    }

    if frames.is_empty() {
        return Err(WebpError::InvalidData);
    }

    Ok(WebpImage {
        frames,
        metadata: metadata_from_container(bytes, c),
        anim_background_rgba: Some([bg.red, bg.green, bg.blue, bg.alpha]),
        anim_loop_count: Some(anim.loop_count),
    })
}

/// Walk a flat §2.3 sub-chunk list (the `ANMF` "Frame Data" sub-RIFF — no
/// outer `RIFF`/`WEBP` header) and return the payload of the first chunk with
/// `target` FourCC. Returns `None` on a truncated header or no match.
fn find_subchunk(mut data: &[u8], target: container::FourCc) -> Option<&[u8]> {
    while data.len() >= 8 {
        let fourcc: container::FourCc = data[0..4].try_into().ok()?;
        let size = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let payload_start = 8usize;
        let payload_end = payload_start.checked_add(size)?;
        if payload_end > data.len() {
            return None;
        }
        if fourcc == target {
            return Some(&data[payload_start..payload_end]);
        }
        // §2.3: odd Size is followed by one pad byte not counted in Size.
        let advance = payload_end + (size & 1);
        if advance > data.len() {
            return None;
        }
        data = &data[advance..];
    }
    None
}

/// Read the file-level metadata chunks (ICC / Exif / XMP) without
/// decoding any pixels.
///
/// Walks the container and lifts the raw payloads of the §2.7.1.4 `ICCP`,
/// §2.7.1.5 `EXIF`, and §2.7.1.5 `XMP ` chunks (each `None` when absent).
pub fn extract_metadata(bytes: &[u8]) -> Result<WebpFileMetadata, WebpError> {
    let c = container::parse(bytes).map_err(|_| WebpError::InvalidData)?;
    Ok(metadata_from_container(bytes, &c))
}

/// Lift the ICC / Exif / XMP payloads out of an already-parsed container.
fn metadata_from_container(bytes: &[u8], c: &container::WebpContainer) -> WebpFileMetadata {
    let payload_of = |fourcc| {
        c.first_chunk_with_fourcc(fourcc)
            .map(|chunk| chunk.payload(bytes).to_vec())
    };
    WebpFileMetadata {
        icc: payload_of(container::fourcc::ICCP),
        exif: payload_of(container::fourcc::EXIF),
        xmp: payload_of(container::fourcc::XMP),
    }
}

/// Encode an interleaved 8-bit RGBA image to a complete RIFF/WEBP file
/// carrying a §2.6 simple-lossless `VP8L` chunk.
///
/// `rgba` is `width * height * 4` bytes in scan-line (top-to-bottom,
/// left-to-right) order, each pixel `[R, G, B, A]` — exactly the
/// [`DecodedWebp::rgba`] layout [`decode_webp_image`] returns. The encoded
/// file decodes back to the same bytes through [`decode_webp`], a
/// pixel-exact round trip.
///
/// This is the round-115 encoder: it takes the simplest spec-conformant
/// VP8L path — no §3.8.2 transform, no §3.8.3 color cache, a single
/// meta-prefix code, and a literal-only image (no LZ77 backward references)
/// — building the §3.7.2 canonical prefix codes per-image from the pixel
/// frequencies. See [`vp8l_encode::encode_webp_lossless`].
pub fn encode_webp_lossless(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, Error> {
    vp8l_encode::encode_webp_lossless(rgba, width, height).map_err(Into::into)
}

// ─────────────────────── Published-shape VP8L encode API ───────────────────────
//
// The published-0.1.5 lossless-encode public names, mapped onto the
// round-115 in-crate VP8L encoder. `encode_vp8l_argb` / `_with` produce a
// **bare** VP8L bitstream (no RIFF wrapper); `encode_vp8l_argb_with_metadata`
// produces a complete `.webp`, auto-promoting to the §2.7 `VP8X` layout when
// alpha or any metadata field is set. See `API-COMPAT.md`.

/// Encode an ARGB image to a **bare** §2.6 / §3.4 `VP8L` bitstream — the
/// chunk payload (image-header + image stream), with **no** RIFF/WEBP
/// wrapper.
///
/// `argb` is `width * height` packed ARGB values in scan-line order, each
/// `(alpha << 24) | (red << 16) | (green << 8) | blue` — the same layout
/// [`vp8l_decode::DecodedImage::pixels`] produces. The §3.4 `alpha_is_used`
/// header bit is auto-detected (set iff any pixel's alpha is not `0xff`);
/// use [`encode_vp8l_argb_with`] to set it explicitly.
///
/// Wrapping the returned bytes in `build::build_webp_file(.., ImageKind::Lossless, ..)`
/// (or `build::build_chunk(fourcc::VP8L, ..)`) yields a complete `.webp` that
/// decodes back to the input pixels exactly via [`decode_webp`].
pub fn encode_vp8l_argb(argb: &[u32], width: u32, height: u32) -> Result<Vec<u8>, WebpError> {
    vp8l_encode::encode_vp8l_argb(argb, width, height)
        .map_err(Error::from)
        .map_err(WebpError::from)
}

/// Encode an ARGB image to a bare §2.6 / §3.4 `VP8L` bitstream with the
/// §3.4 `alpha_is_used` header bit set **explicitly** by the caller.
///
/// The fixed (non-RDO) form of [`encode_vp8l_argb`]: `has_alpha` becomes the
/// header bit verbatim instead of being scanned from the pixels. The alpha
/// values are carried in the §3.7.3 ARGB literals regardless of the bit, so
/// the round trip is exact either way.
pub fn encode_vp8l_argb_with(
    argb: &[u32],
    width: u32,
    height: u32,
    has_alpha: bool,
) -> Result<Vec<u8>, WebpError> {
    vp8l_encode::encode_vp8l_argb_with(argb, width, height, has_alpha)
        .map_err(Error::from)
        .map_err(WebpError::from)
}

/// Encode an ARGB image to a complete `.webp` file carrying a §2.6 `VP8L`
/// lossless bitstream, embedding any supplied file-level metadata.
///
/// `argb` is `width * height` packed ARGB values in scan-line order. The
/// output layout is chosen automatically:
///
/// * **Simple `VP8L`** (`RIFF`/`WEBP` + `VP8L`) when `has_alpha` is `false`
///   **and** `meta` is empty — the smallest spec-conformant still image.
/// * **Extended `VP8X`** (`RIFF`/`WEBP` + `VP8X` + `ICCP` + `VP8L` +
///   `EXIF` + `XMP `) when `has_alpha` is `true` **or** any metadata
///   field is set. The §2.7.1 `VP8X` flag octet declares exactly the
///   features present (`L`/`I`/`E`/`X`), and the metadata chunks are emitted
///   in the §2.7 order (`ICCP` before the image, `EXIF`/`XMP ` after).
///
/// The bitstream's own §3.4 `alpha_is_used` header bit is set from
/// `has_alpha`. Decoding the result through [`decode_webp`] reproduces the
/// input pixels exactly; [`extract_metadata`] reads back the embedded
/// ICC / Exif / XMP payloads.
pub fn encode_vp8l_argb_with_metadata(
    width: u32,
    height: u32,
    argb: &[u32],
    has_alpha: bool,
    meta: &WebpMetadata<'_>,
) -> Result<Vec<u8>, WebpError> {
    // Bare VP8L bitstream (image-header + image stream).
    let payload = encode_vp8l_argb_with(argb, width, height, has_alpha)?;

    // Simple layout when there is nothing to declare in a VP8X.
    if !has_alpha && meta.is_empty() {
        return build::build_webp_file(&payload, build::ImageKind::Lossless, width, height)
            .map_err(Error::from)
            .map_err(WebpError::from);
    }

    // Extended layout: VP8X header declaring exactly the present features,
    // then ICCP, the VP8L image, EXIF, XMP — the §2.7 order.
    let flags = build::Vp8xFlags {
        has_iccp: meta.icc.is_some(),
        has_alpha,
        has_exif: meta.exif.is_some(),
        has_xmp: meta.xmp.is_some(),
        has_animation: false,
    };
    let vp8x_payload = build::build_vp8x_chunk(width, height, flags)
        .map_err(Error::from)
        .map_err(WebpError::from)?;

    let mut body = Vec::new();
    let mut push_chunk = |fourcc, payload: &[u8]| -> Result<(), WebpError> {
        let chunk = build::build_chunk(fourcc, payload)
            .map_err(Error::from)
            .map_err(WebpError::from)?;
        body.extend_from_slice(&chunk);
        Ok(())
    };

    push_chunk(container::fourcc::VP8X, &vp8x_payload)?;
    if let Some(icc) = meta.icc {
        push_chunk(container::fourcc::ICCP, icc)?;
    }
    push_chunk(container::fourcc::VP8L, &payload)?;
    if let Some(exif) = meta.exif {
        push_chunk(container::fourcc::EXIF, exif)?;
    }
    if let Some(xmp) = meta.xmp {
        push_chunk(container::fourcc::XMP, xmp)?;
    }

    // §2.4 file framing around the assembled body.
    let file_size = (body.len() as u64) + 4;
    if file_size > u64::from(u32::MAX) {
        return Err(WebpError::InvalidData);
    }
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&container::fourcc::RIFF);
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&container::fourcc::WEBP);
    out.extend_from_slice(&body);
    Ok(out)
}

// ─────────────────────── Published-shape animation encode API ───────────────────────
//
// The published-0.1.5 `build_animated_webp` surface, rebuilt on top of the
// in-crate VP8L encoder + the §2.7.1.1 ANIM / ANMF container framing. Only the
// VP8L-lossless path (`AnimFrameMode::Lossless`) is wired up; `Auto` / `Delta`
// return `WebpError::Unsupported` (the VP8 lossy + delta paths are blocked on
// `oxideav-vp8`, workspace task #1041). See `API-COMPAT.md`.

#[doc(inline)]
pub use anim_encode::{
    build_animated_webp, build_animated_webp_with_options, AnimEncoderOptions, AnimFrame,
    AnimFrameMode, DeltaConfig, DownsampleKernel,
};

/// Stable codec identifier the VP8L lossless encoder registers under in the
/// codec registry — the published `"webp_vp8l"` name.
pub const CODEC_ID_VP8L: &str = "webp_vp8l";

/// Repack a scan-line-order ARGB pixel buffer (`(a<<24)|(r<<16)|(g<<8)|b`)
/// into interleaved 8-bit `[R, G, B, A]` bytes — the
/// `oxideav_core::PixelFormat::Rgba` layout.
fn argb_to_rgba(pixels: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() * 4);
    for &argb in pixels {
        out.push((argb >> 16) as u8); // R
        out.push((argb >> 8) as u8); // G
        out.push(argb as u8); // B
        out.push((argb >> 24) as u8); // A
    }
    out
}

/// Install the WebP decoder factory and the `.webp` extension hint into
/// `ctx` per round 112.
///
/// Wraps [`registry::register`]; see that module for the full breakdown
/// of what lands in the codec / container sub-registries. The decoder
/// covers the §2.6 / §3.4 `VP8L` lossless image (simple or
/// `VP8X`-extended) with optional §2.7.1.2 `ALPH`-over-`VP8L` alpha
/// override. The §2.5 `VP8 ` lossy path surfaces as a clean
/// `oxideav_core::Error::Unsupported` — callers route lossy chunks via
/// [`extract_lossy_chunk`] to a downstream VP8 decoder.
#[cfg(feature = "registry")]
pub fn register(ctx: &mut RuntimeContext) {
    registry::register(ctx);
}

#[cfg(feature = "registry")]
oxideav_core::register!("webp", register);
