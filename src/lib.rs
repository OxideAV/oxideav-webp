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
pub mod anmf;
pub mod build;
pub mod container;
pub mod meta_prefix;
#[cfg(feature = "registry")]
pub mod registry;
pub mod vp8_chunk;
pub mod vp8l_chunk;
pub mod vp8l_decode;
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

/// Decode a still WebP file to a tightly packed 8-bit RGBA pixel buffer.
///
/// Convenience over [`decode_webp_image`] that drops the dimensions and
/// returns only the `width * height * 4` interleaved `[R, G, B, A]`
/// bytes. See [`decode_webp_image`] for the supported image kinds and
/// the [`Error::Unsupported`] cases (`VP8 ` lossy / animation /
/// header-only).
pub fn decode_webp(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Ok(decode_webp_image(bytes)?.rgba)
}

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
