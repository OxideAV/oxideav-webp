//! RIFF/WEBP container *builder* helpers per RFC 9649 §2.3–§2.7.
//!
//! Where [`crate::container`] **walks** an existing `RIFF/WEBP` byte
//! stream into a typed [`crate::container::WebpContainer`], this module
//! is its inverse: it **assembles** a well-formed `RIFF/WEBP` byte
//! stream from a single bitstream payload.
//!
//! The round-5 surface is deliberately minimal — just enough container
//! plumbing for an external encoder (e.g. the workspace's `cli-convert`
//! tool) to bolt a `VP8 ` / `VP8L` / `VP8X`-extended file header around
//! payload bytes it computed elsewhere. The actual `VP8 ` / `VP8L`
//! bitstream encoders are *not* in this crate yet; the builders here
//! treat the codec payload as opaque bytes.
//!
//! ## What lives in this module
//!
//! * [`build_chunk`] — the §2.3 generic chunk writer: `FourCC` + 4-byte
//!   little-endian `Size` + payload bytes + (if `Size` is odd) a single
//!   `0x00` pad byte. The §2.4 file-header writer and the §2.7.1 `VP8X`
//!   payload writer are both expressed in terms of this primitive.
//!
//! * [`build_vp8x_chunk`] — the §2.7.1 Figure 7 typed writer:
//!   `flags(1) + Reserved(3) + (canvas_width-1)(3 LE) + (canvas_height-1)(3 LE)`.
//!   Returns the **payload** only (10 bytes); wrap with [`build_chunk`]
//!   passing [`crate::container::fourcc::VP8X`] for the on-disk chunk.
//!
//! * [`build_webp_file`] — the §2.4 file writer for a *simple* layout
//!   (lossy / lossless / extended-with-VP8X-only). Given a single
//!   codec payload + image kind + canvas dimensions it produces:
//!
//!     ```text
//!     RIFF | <File Size LE u32> | WEBP | <chunk> ...
//!     ```
//!
//!   For [`ImageKind::Lossy`] / [`ImageKind::Lossless`] the body is the
//!   single `VP8 ` / `VP8L` chunk per §2.5 / §2.6. For
//!   [`ImageKind::ExtendedLossy`] / [`ImageKind::ExtendedLossless`]
//!   the body is a §2.7.1 `VP8X` chunk followed by the `VP8 ` / `VP8L`
//!   chunk per §2.7's chunk-ordering rule.
//!
//! ## What is intentionally *not* here
//!
//! * No `ANIM` / `ANMF` / `ALPH` writers — the round-5 still-image fast
//!   path plus the §2.7 metadata-wrapper writer ([`build_webp_file_with_metadata`])
//!   cover ICCP / EXIF / XMP. Animation chunks live in [`crate::anim_encode`]
//!   because they need additional global parameters (`ANIM` background +
//!   loop count, per-frame `ANMF` placement).
//! * No payload validation. `build_webp_file` does not parse the bytes
//!   the caller hands it as `payload`. A nonsense payload still
//!   produces a structurally-valid RIFF — the responsibility for the
//!   payload's correctness sits with whoever computed it.
//! * No registry dependency. Every public function in this module
//!   compiles cleanly under `--no-default-features` (no `oxideav-core`
//!   in the dependency tree) because the builders are plain
//!   byte-pushing functions over `std::vec::Vec<u8>`.
//!
//! ## §2.4 `File Size` rule
//!
//! RFC 9649 §2.4: "The file size in the header is the total size of
//! the chunks that follow plus 4 bytes for the `WEBP` FourCC." So if
//! the body (concatenated chunks, each already including its own §2.3
//! pad byte) is `N` bytes long, the `File Size` field is `N + 4`. The
//! `RIFF` FourCC and the `File Size` field itself are *not* counted.
//!
//! ## Round-trip guarantee
//!
//! Every byte stream produced by [`build_webp_file`] / [`build_chunk`]
//! parses successfully through [`crate::container::parse`] and
//! [`crate::parse_vp8x_header`]. The
//! [`crate::container::WebpChunk::payload`] of the resulting chunks
//! equals the input payload byte-for-byte (the §2.3 pad byte is added
//! to the chunk *stream* on disk but is not counted in `Size` and is
//! therefore not included in the `payload` slice). This is enforced
//! by the round-trip tests at the bottom of this module.

use crate::container::{fourcc, FourCc};

/// Maximum 1-based canvas dimension representable in the §2.7.1 24-bit
/// `Minus One` field — `2^24` (because the on-disk value is `dim - 1`
/// and the field is 24 bits wide, the largest representable dim is
/// `0x00FF_FFFF + 1 = 0x0100_0000`).
pub const MAX_VP8X_CANVAS_DIM: u32 = 0x0100_0000;

/// Maximum payload bytes a single §2.3 chunk can carry. The `Size`
/// field is a `uint32`; a chunk whose payload is exactly this size
/// fills the field; an odd value of this size would also require a
/// `0x00` pad byte. We additionally subtract 1 to leave room for that
/// pad byte without overflowing `u32` when callers compute total chunk
/// sizes downstream. (Practical WebP files are nowhere near this.)
pub const MAX_CHUNK_PAYLOAD: u32 = u32::MAX - 1;

/// Which §2.4 / §2.5 / §2.6 / §2.7 file layout [`build_webp_file`]
/// should emit around the caller's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    /// §2.5 simple-lossy layout: just a `VP8 ` chunk after the §2.4
    /// file header. No `VP8X`. The `width`/`height` arguments to
    /// [`build_webp_file`] are *ignored* — the canvas dimensions in a
    /// simple-lossy WebP are derived by the decoder from the `VP8 `
    /// bitstream itself per §2.5's "VP8 frame header contains the VP8
    /// frame width and height" note.
    Lossy,
    /// §2.6 simple-lossless layout: just a `VP8L` chunk after the
    /// §2.4 file header. No `VP8X`. Same note re. canvas dimensions:
    /// the `width`/`height` arguments are *ignored* because the `VP8L`
    /// bitstream carries its own image-width / image-height fields
    /// per §2.6.
    Lossless,
    /// §2.7 extended layout: `VP8X` + `VP8 `. Use this when the file
    /// needs a `VP8X`-declared canvas (e.g. ahead of future ALPH /
    /// metadata / animation extensions, or to declare a canvas size
    /// that disagrees with the `VP8 ` bitstream's intrinsic
    /// dimensions). Round-5 emits **no** feature flags in the VP8X
    /// byte; downstream rounds can add an enum variant with feature
    /// flags once `ALPH` / animation writers exist.
    ExtendedLossy,
    /// §2.7 extended layout: `VP8X` + `VP8L`. Symmetric with
    /// [`ImageKind::ExtendedLossy`] but the bitstream chunk is `VP8L`.
    ExtendedLossless,
}

impl ImageKind {
    /// FourCC of the single bitstream chunk this kind wraps the
    /// payload in.
    pub fn bitstream_fourcc(self) -> FourCc {
        match self {
            Self::Lossy | Self::ExtendedLossy => fourcc::VP8,
            Self::Lossless | Self::ExtendedLossless => fourcc::VP8L,
        }
    }

    /// True if this kind emits a `VP8X` chunk ahead of the bitstream
    /// per §2.7.
    pub fn is_extended(self) -> bool {
        matches!(self, Self::ExtendedLossy | Self::ExtendedLossless)
    }
}

/// Errors raised by the §2.3–§2.7 builder helpers. Builders refuse
/// inputs that would produce a file the corresponding parser would
/// reject — the symmetry is deliberate so round-trips can't be
/// constructed-but-unparseable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// A §2.7.1 canvas dimension is zero. The on-disk field is
    /// `dim - 1`, so a 0 dimension would underflow.
    CanvasDimZero {
        /// Which dimension was zero — `"width"` or `"height"`.
        which: &'static str,
    },
    /// A §2.7.1 canvas dimension exceeds the 24-bit `Minus One` field's
    /// maximum representable value (`2^24`).
    CanvasDimTooLarge {
        /// Which dimension overflowed.
        which: &'static str,
        /// The offending 1-based dimension.
        got: u32,
    },
    /// The product `canvas_width * canvas_height` exceeds the §2.7.1
    /// `2^32 - 1` cap.
    CanvasTooLarge {
        /// 1-based canvas width.
        canvas_width: u32,
        /// 1-based canvas height.
        canvas_height: u32,
    },
    /// The caller-supplied payload is larger than a §2.3 chunk's
    /// `Size` field (a `uint32`) can address.
    PayloadTooLargeForChunk {
        /// Payload length the caller passed in.
        got: usize,
    },
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CanvasDimZero { which } => {
                write!(f, "§2.7.1 VP8X canvas {which} must be ≥ 1 (got 0)")
            }
            Self::CanvasDimTooLarge { which, got } => write!(
                f,
                "§2.7.1 VP8X canvas {which} {got} exceeds the 24-bit Minus-One field's 2^24 cap",
            ),
            Self::CanvasTooLarge {
                canvas_width,
                canvas_height,
            } => write!(
                f,
                "§2.7.1 VP8X canvas {canvas_width}x{canvas_height} \
                 exceeds the 2^32 - 1 product cap",
            ),
            Self::PayloadTooLargeForChunk { got } => write!(
                f,
                "§2.3 chunk payload of {got} bytes exceeds the uint32 Size field's range",
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// Emit a §2.3 RIFF chunk: 4-byte FourCC + 4-byte little-endian
/// `Size` + payload + (if `Size` is odd) one `0x00` pad byte.
///
/// The pad byte is **not** counted in `Size` per §2.3. Callers don't
/// need to think about it — this writer adds it when (and only when)
/// the payload length is odd.
///
/// The returned `Vec<u8>` is exactly `8 + payload.len() + (payload.len() & 1)`
/// bytes long.
pub fn build_chunk(fourcc: FourCc, payload: &[u8]) -> Result<Vec<u8>, BuildError> {
    if payload.len() as u64 > MAX_CHUNK_PAYLOAD as u64 {
        return Err(BuildError::PayloadTooLargeForChunk { got: payload.len() });
    }
    let size = payload.len() as u32;
    let needs_pad = (size & 1) == 1;
    let total = 8 + payload.len() + if needs_pad { 1 } else { 0 };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&fourcc);
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(payload);
    if needs_pad {
        out.push(0);
    }
    Ok(out)
}

/// Emit the 10-byte §2.7.1 Figure 7 `VP8X` chunk **payload**
/// (i.e. *not* including the 8-byte chunk header that
/// [`build_chunk`] would prepend).
///
/// Layout (matches [`crate::vp8x::Vp8xHeader::parse`]'s inverse):
///
/// | offset | width | field                                |
/// |--------|-------|--------------------------------------|
/// | 0      | 1 B   | flags byte `Rsv\|I\|L\|E\|X\|A\|R`     |
/// | 1..4   | 3 B   | reserved 24-bit field (zero-filled)  |
/// | 4..7   | 3 B   | `(canvas_width - 1)`  uint24 LE      |
/// | 7..10  | 3 B   | `(canvas_height - 1)` uint24 LE      |
///
/// Where the `flags` byte is built from the named feature flags using
/// the same bit-position table as the parser:
///
/// | feature flag      | bit (LSB=0) |
/// |-------------------|-------------|
/// | `has_iccp` (`I`)  | 5           |
/// | `has_alpha` (`L`) | 4           |
/// | `has_exif` (`E`)  | 3           |
/// | `has_xmp` (`X`)   | 2           |
/// | `has_animation` (`A`) | 1       |
///
/// The two `Rsv` bits (7..6), the trailing `R` bit (0), and the 24-bit
/// reserved field at bytes 1..4 are zero-filled — §2.7.1 requires
/// reserved positions to be 0 on write even though readers MUST
/// ignore non-zero values.
///
/// Returns the 10-byte payload; callers wrap it via
/// `build_chunk(fourcc::VP8X, &payload)` to obtain the on-disk chunk.
pub fn build_vp8x_chunk(
    canvas_width: u32,
    canvas_height: u32,
    flags: Vp8xFlags,
) -> Result<Vec<u8>, BuildError> {
    if canvas_width == 0 {
        return Err(BuildError::CanvasDimZero { which: "width" });
    }
    if canvas_height == 0 {
        return Err(BuildError::CanvasDimZero { which: "height" });
    }
    if canvas_width > MAX_VP8X_CANVAS_DIM {
        return Err(BuildError::CanvasDimTooLarge {
            which: "width",
            got: canvas_width,
        });
    }
    if canvas_height > MAX_VP8X_CANVAS_DIM {
        return Err(BuildError::CanvasDimTooLarge {
            which: "height",
            got: canvas_height,
        });
    }
    if (canvas_width as u64) * (canvas_height as u64) > u64::from(u32::MAX) {
        return Err(BuildError::CanvasTooLarge {
            canvas_width,
            canvas_height,
        });
    }

    let cwm1 = canvas_width - 1;
    let chm1 = canvas_height - 1;

    // §2.7.1 byte 0 bit positions (LSB=0): I=5, L=4, E=3, X=2, A=1.
    let mut flag_byte: u8 = 0;
    if flags.has_iccp {
        flag_byte |= 1 << 5;
    }
    if flags.has_alpha {
        flag_byte |= 1 << 4;
    }
    if flags.has_exif {
        flag_byte |= 1 << 3;
    }
    if flags.has_xmp {
        flag_byte |= 1 << 2;
    }
    if flags.has_animation {
        flag_byte |= 1 << 1;
    }

    let mut payload = Vec::with_capacity(10);
    payload.push(flag_byte);
    // §2.7.1 24-bit Reserved field — MUST be 0 on write.
    payload.extend_from_slice(&[0u8, 0u8, 0u8]);
    // §2.7.1 Canvas Width Minus One — 24-bit little-endian.
    payload.push((cwm1 & 0xFF) as u8);
    payload.push(((cwm1 >> 8) & 0xFF) as u8);
    payload.push(((cwm1 >> 16) & 0xFF) as u8);
    // §2.7.1 Canvas Height Minus One — 24-bit little-endian.
    payload.push((chm1 & 0xFF) as u8);
    payload.push(((chm1 >> 8) & 0xFF) as u8);
    payload.push(((chm1 >> 16) & 0xFF) as u8);
    Ok(payload)
}

/// Feature flags for the §2.7.1 `VP8X` flag octet.
///
/// `Default` is all-zero — i.e. a `VP8X` chunk that declares no
/// optional features. That matches the round-5 fast path: the
/// builder emits a `VP8X` only to express a canvas size that the
/// `VP8 ` / `VP8L` bitstream wouldn't otherwise carry.
///
/// Once `ALPH` / `ANIM` / `ICCP` / `EXIF` / `XMP ` writers land in
/// later rounds, those writers will set the corresponding flag here
/// so the §2.7.1 declaration matches the chunks the builder actually
/// emitted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Vp8xFlags {
    /// §2.7.1 `I` bit — an `ICCP` chunk follows.
    pub has_iccp: bool,
    /// §2.7.1 `L` bit — any frame contains alpha.
    pub has_alpha: bool,
    /// §2.7.1 `E` bit — an `EXIF` chunk follows.
    pub has_exif: bool,
    /// §2.7.1 `X` bit — an `XMP ` chunk follows.
    pub has_xmp: bool,
    /// §2.7.1 `A` bit — this is an animated image.
    pub has_animation: bool,
}

/// Build a `RIFF/WEBP` file around a single bitstream payload per
/// RFC 9649 §2.4 + §2.5 / §2.6 / §2.7.
///
/// Arguments:
///
/// * `payload` — the opaque `VP8 ` or `VP8L` bitstream bytes. The
///   builder copies these into the output without inspecting them.
/// * `image_kind` — which file layout to emit; see [`ImageKind`]. The
///   selection determines the bitstream FourCC (`VP8 ` for
///   `Lossy` / `ExtendedLossy`, `VP8L` for the lossless pair) and
///   whether a §2.7.1 `VP8X` chunk is emitted ahead of the bitstream.
/// * `canvas_width`, `canvas_height` — 1-based pixel dimensions, used
///   only for [`ImageKind::ExtendedLossy`] / [`ImageKind::ExtendedLossless`].
///   For [`ImageKind::Lossy`] / [`ImageKind::Lossless`] the canvas
///   dimensions are encoded in the bitstream's own frame header per
///   §2.5 / §2.6 and the arguments here are *ignored* (the builder
///   does not validate them against the bitstream).
///
/// Returns the complete on-disk byte stream, including the 12-byte
/// §2.4 file header and any §2.3 pad bytes the chunks needed.
///
/// ```text
/// [ 'RIFF' | <File Size LE u32> | 'WEBP' | <VP8X chunk?> | <VP8 / VP8L chunk> ]
/// ```
///
/// §2.4 `File Size` = `4` (the 'WEBP' FourCC) `+ body.len()`, where
/// `body` is the concatenation of every chunk after 'WEBP' (each
/// chunk already includes its own §2.3 pad byte where required). The
/// `RIFF` FourCC and the `File Size` field itself are not counted.
pub fn build_webp_file(
    payload: &[u8],
    image_kind: ImageKind,
    canvas_width: u32,
    canvas_height: u32,
) -> Result<Vec<u8>, BuildError> {
    if payload.len() as u64 > MAX_CHUNK_PAYLOAD as u64 {
        return Err(BuildError::PayloadTooLargeForChunk { got: payload.len() });
    }

    let bitstream_chunk = build_chunk(image_kind.bitstream_fourcc(), payload)?;

    let body = if image_kind.is_extended() {
        let vp8x_payload = build_vp8x_chunk(canvas_width, canvas_height, Vp8xFlags::default())?;
        let vp8x_chunk = build_chunk(fourcc::VP8X, &vp8x_payload)?;
        let mut b = Vec::with_capacity(vp8x_chunk.len() + bitstream_chunk.len());
        b.extend_from_slice(&vp8x_chunk);
        b.extend_from_slice(&bitstream_chunk);
        b
    } else {
        bitstream_chunk
    };

    // §2.4: File Size = 4 ('WEBP' FourCC) + body length.
    let file_size = (body.len() as u64) + 4;
    // The §2.4 File Size field is uint32 with a documented maximum of
    // 2^32 - 10, so a body up to ~4 GiB - 14 fits. We bound by u32::MAX
    // here for the cast; in practice MAX_CHUNK_PAYLOAD already caps
    // each chunk far below that.
    if file_size > u64::from(u32::MAX) {
        return Err(BuildError::PayloadTooLargeForChunk { got: payload.len() });
    }
    let file_size = file_size as u32;

    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&fourcc::RIFF);
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(&fourcc::WEBP);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Borrowed §2.7 file-level metadata payloads for the metadata-aware
/// container writer [`build_webp_file_with_metadata`].
///
/// Each field is the raw payload bytes of the corresponding §2.7.1.4 /
/// §2.7.1.5 chunk, or `None` to omit the chunk entirely. The writer
/// derives the §2.7.1 `VP8X` flag bits from which fields are `Some`
/// (`I` for `iccp`, `E` for `exif`, `X` for `xmp`) so the declared
/// feature set always matches the chunks actually emitted.
///
/// The `Default` impl is all-`None` — equivalent to a non-metadata
/// extended-layout file. A caller with no metadata to embed and no
/// alpha to declare should use [`build_webp_file`] directly instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileMetadata<'a> {
    /// §2.7.1.4 `ICCP` ICC color-profile payload to embed, or `None` to
    /// omit the chunk. The writer sets the §2.7.1 `I` flag iff
    /// `Some(_)`.
    pub iccp: Option<&'a [u8]>,
    /// §2.7.1.5 `EXIF` Exif payload to embed, or `None` to omit the
    /// chunk. The writer sets the §2.7.1 `E` flag iff `Some(_)`.
    pub exif: Option<&'a [u8]>,
    /// §2.7.1.5 `XMP ` XMP payload to embed, or `None` to omit the
    /// chunk. The writer sets the §2.7.1 `X` flag iff `Some(_)`.
    pub xmp: Option<&'a [u8]>,
}

impl FileMetadata<'_> {
    /// `true` when every metadata field is `None`.
    pub fn is_empty(&self) -> bool {
        self.iccp.is_none() && self.exif.is_none() && self.xmp.is_none()
    }
}

/// Build a `RIFF/WEBP` file in the §2.7 *extended* layout, wrapping a
/// single bitstream payload with a §2.7.1 `VP8X` chunk + optional
/// §2.7.1.4 `ICCP` / §2.7.1.5 `EXIF` / §2.7.1.5 `XMP ` metadata chunks.
///
/// The on-disk chunk order is the §2.7 canonical order:
///
/// ```text
/// RIFF | <File Size LE u32> | WEBP | VP8X | ICCP? | <VP8 | VP8L> | EXIF? | XMP ?
/// ```
///
/// (Each `?` chunk is emitted only when the corresponding
/// [`FileMetadata`] field is `Some`.)
///
/// Arguments:
///
/// * `payload` — the opaque `VP8 ` or `VP8L` bitstream bytes.
/// * `image_kind` — selects the bitstream chunk's FourCC; the *simple*
///   variants ([`ImageKind::Lossy`] / [`ImageKind::Lossless`]) and the
///   *extended* variants ([`ImageKind::ExtendedLossy`] /
///   [`ImageKind::ExtendedLossless`]) both produce the same extended
///   on-disk layout here — this writer *always* emits a `VP8X`
///   ahead of the bitstream because metadata chunks require an
///   `Extended File Format` per §2.7.
/// * `canvas_width`, `canvas_height` — 1-based pixel dimensions written
///   into the §2.7.1 canvas-minus-one fields (subject to the §2.7.1
///   range / product caps enforced by [`build_vp8x_chunk`]).
/// * `has_alpha` — value of the §2.7.1 `L` ("Alpha") flag. `true`
///   when any frame contains transparency; carried verbatim into the
///   `VP8X` flag byte. (Whether the bitstream payload itself carries
///   alpha is the caller's responsibility — this writer treats the
///   payload as opaque bytes.)
/// * `metadata` — see [`FileMetadata`]. Setting `iccp` / `exif` / `xmp`
///   to `Some(..)` both (a) sets the corresponding §2.7.1 flag bit
///   (`I` / `E` / `X`) and (b) emits the chunk in §2.7 canonical
///   position.
///
/// ## Round-trip guarantee
///
/// Every byte stream produced by this function parses successfully
/// through [`crate::container::parse`], and its §2.7.1.4 `ICCP` /
/// §2.7.1.5 `EXIF` / §2.7.1.5 `XMP ` payloads round-trip byte-for-byte
/// through [`crate::extract_metadata`]. The §2.7.1 `VP8X` flag octet
/// also round-trips through [`crate::vp8x::Vp8xHeader::parse`] —
/// `has_iccp` / `has_exif` / `has_xmp` reflect exactly which
/// [`FileMetadata`] fields were `Some(..)`, and `has_alpha` reflects
/// the `has_alpha` argument.
///
/// ## §2.3 odd-payload pad bytes
///
/// Each chunk emitted here goes through [`build_chunk`], so any
/// metadata payload of odd length receives the §2.3 `0x00` pad byte
/// automatically. The pad byte is *not* counted in the chunk's `Size`
/// field and *is* counted in the §2.4 file `File Size` total, mirroring
/// what [`crate::container::parse`] expects on the read side.
pub fn build_webp_file_with_metadata(
    payload: &[u8],
    image_kind: ImageKind,
    canvas_width: u32,
    canvas_height: u32,
    has_alpha: bool,
    metadata: FileMetadata<'_>,
) -> Result<Vec<u8>, BuildError> {
    if payload.len() as u64 > MAX_CHUNK_PAYLOAD as u64 {
        return Err(BuildError::PayloadTooLargeForChunk { got: payload.len() });
    }

    // §2.7.1 VP8X flag byte — derive from which metadata fields are
    // Some plus the explicit has_alpha argument.
    let flags = Vp8xFlags {
        has_iccp: metadata.iccp.is_some(),
        has_alpha,
        has_exif: metadata.exif.is_some(),
        has_xmp: metadata.xmp.is_some(),
        has_animation: false,
    };
    let vp8x_payload = build_vp8x_chunk(canvas_width, canvas_height, flags)?;
    let vp8x_chunk = build_chunk(fourcc::VP8X, &vp8x_payload)?;
    let bitstream_chunk = build_chunk(image_kind.bitstream_fourcc(), payload)?;

    // §2.7 canonical order: VP8X, ICCP (before image data), image data,
    // EXIF, XMP.
    let mut body = Vec::with_capacity(
        vp8x_chunk.len()
            + metadata.iccp.map_or(0, |b| 8 + b.len() + (b.len() & 1))
            + bitstream_chunk.len()
            + metadata.exif.map_or(0, |b| 8 + b.len() + (b.len() & 1))
            + metadata.xmp.map_or(0, |b| 8 + b.len() + (b.len() & 1)),
    );
    body.extend_from_slice(&vp8x_chunk);
    if let Some(iccp) = metadata.iccp {
        let c = build_chunk(fourcc::ICCP, iccp)?;
        body.extend_from_slice(&c);
    }
    body.extend_from_slice(&bitstream_chunk);
    if let Some(exif) = metadata.exif {
        let c = build_chunk(fourcc::EXIF, exif)?;
        body.extend_from_slice(&c);
    }
    if let Some(xmp) = metadata.xmp {
        let c = build_chunk(fourcc::XMP, xmp)?;
        body.extend_from_slice(&c);
    }

    // §2.4: File Size = 4 ('WEBP' FourCC) + body length.
    let file_size = (body.len() as u64) + 4;
    if file_size > u64::from(u32::MAX) {
        return Err(BuildError::PayloadTooLargeForChunk { got: payload.len() });
    }
    let file_size = file_size as u32;

    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&fourcc::RIFF);
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(&fourcc::WEBP);
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{parse, ContainerError};
    use crate::vp8x::Vp8xHeader;

    /// §2.3 sanity: a chunk with even-length payload is exactly
    /// `8 + payload.len()` bytes — no pad byte.
    #[test]
    fn build_chunk_even_payload_has_no_pad_byte() {
        let bytes = build_chunk(fourcc::VP8, &[0u8; 8]).unwrap();
        assert_eq!(bytes.len(), 8 + 8);
        assert_eq!(&bytes[0..4], b"VP8 ");
        assert_eq!(&bytes[4..8], &8u32.to_le_bytes());
    }

    /// §2.3 sanity: odd-length payload gets a trailing `0x00` byte
    /// that is *not* counted in `Size`.
    #[test]
    fn build_chunk_odd_payload_appends_one_zero_pad_byte() {
        let bytes = build_chunk(fourcc::ICCP, &[0xAA, 0xBB, 0xCC]).unwrap();
        // 8-byte header + 3 payload bytes + 1 pad byte.
        assert_eq!(bytes.len(), 8 + 3 + 1);
        assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
        assert_eq!(bytes[bytes.len() - 1], 0u8);
        // The pad byte sits past the declared Size.
        assert_eq!(&bytes[8..8 + 3], &[0xAA, 0xBB, 0xCC]);
    }

    /// §2.7.1 Figure 7 layout: byte 0 is flags, bytes 1..4 reserved
    /// (zero), bytes 4..7 width-minus-one LE, bytes 7..10
    /// height-minus-one LE.
    #[test]
    fn build_vp8x_payload_layout_matches_figure_7_byte_for_byte() {
        let payload = build_vp8x_chunk(
            128,
            64,
            Vp8xFlags {
                has_alpha: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(payload.len(), 10);
        // L bit (4) set, all others clear.
        assert_eq!(payload[0], 0b0001_0000);
        // Reserved 24-bit field zero.
        assert_eq!(&payload[1..4], &[0u8, 0u8, 0u8]);
        // 128 → minus-one = 127 = 0x7F.
        assert_eq!(&payload[4..7], &[0x7F, 0x00, 0x00]);
        // 64 → minus-one = 63 = 0x3F.
        assert_eq!(&payload[7..10], &[0x3F, 0x00, 0x00]);
    }

    /// All five named feature flags map to the bit positions the
    /// parser already locked down in [`crate::vp8x`].
    #[test]
    fn build_vp8x_flag_bits_match_parser_table() {
        let cases: &[(Vp8xFlags, u8)] = &[
            (
                Vp8xFlags {
                    has_iccp: true,
                    ..Default::default()
                },
                0b0010_0000,
            ),
            (
                Vp8xFlags {
                    has_alpha: true,
                    ..Default::default()
                },
                0b0001_0000,
            ),
            (
                Vp8xFlags {
                    has_exif: true,
                    ..Default::default()
                },
                0b0000_1000,
            ),
            (
                Vp8xFlags {
                    has_xmp: true,
                    ..Default::default()
                },
                0b0000_0100,
            ),
            (
                Vp8xFlags {
                    has_animation: true,
                    ..Default::default()
                },
                0b0000_0010,
            ),
        ];
        for (flags, expected) in cases {
            let payload = build_vp8x_chunk(1, 1, *flags).unwrap();
            assert_eq!(payload[0], *expected, "flags={flags:?}");
        }

        // All five together — every bit position above OR'd.
        let all = build_vp8x_chunk(
            1,
            1,
            Vp8xFlags {
                has_iccp: true,
                has_alpha: true,
                has_exif: true,
                has_xmp: true,
                has_animation: true,
            },
        )
        .unwrap();
        assert_eq!(all[0], 0b0011_1110);
    }

    /// 24-bit Minus-One width / height: exercise all three octets so
    /// we catch any LE / BE confusion in the encoder.
    #[test]
    fn build_vp8x_canvas_dims_are_24bit_little_endian() {
        let payload = build_vp8x_chunk(0x00ABCD, 0x000124, Vp8xFlags::default()).unwrap();
        // 0x00ABCD - 1 = 0x00ABCC
        assert_eq!(&payload[4..7], &[0xCC, 0xAB, 0x00]);
        // 0x000124 - 1 = 0x000123
        assert_eq!(&payload[7..10], &[0x23, 0x01, 0x00]);
    }

    /// §2.7.1 0-dimension is impossible to encode (the field is
    /// `dim - 1` and would underflow). The builder refuses up front.
    #[test]
    fn build_vp8x_rejects_zero_canvas_dim() {
        assert_eq!(
            build_vp8x_chunk(0, 1, Vp8xFlags::default()).unwrap_err(),
            BuildError::CanvasDimZero { which: "width" }
        );
        assert_eq!(
            build_vp8x_chunk(1, 0, Vp8xFlags::default()).unwrap_err(),
            BuildError::CanvasDimZero { which: "height" }
        );
    }

    /// §2.7.1 dim above 2^24 doesn't fit in the 24-bit Minus-One field.
    #[test]
    fn build_vp8x_rejects_canvas_dim_above_2_pow_24() {
        let too_big = MAX_VP8X_CANVAS_DIM + 1;
        assert_eq!(
            build_vp8x_chunk(too_big, 1, Vp8xFlags::default()).unwrap_err(),
            BuildError::CanvasDimTooLarge {
                which: "width",
                got: too_big
            }
        );
        assert_eq!(
            build_vp8x_chunk(1, too_big, Vp8xFlags::default()).unwrap_err(),
            BuildError::CanvasDimTooLarge {
                which: "height",
                got: too_big
            }
        );

        // Exactly 2^24 still fits.
        let ok = build_vp8x_chunk(MAX_VP8X_CANVAS_DIM, 1, Vp8xFlags::default()).unwrap();
        assert_eq!(&ok[4..7], &[0xFF, 0xFF, 0xFF]); // 2^24 - 1 = 0x00FF_FFFF
    }

    /// §2.7.1 product cap: w*h > 2^32 - 1 is rejected, mirroring the
    /// parser's `CanvasTooLarge`.
    #[test]
    fn build_vp8x_rejects_canvas_above_product_cap() {
        let err = build_vp8x_chunk(65_536, 65_536, Vp8xFlags::default()).unwrap_err();
        assert_eq!(
            err,
            BuildError::CanvasTooLarge {
                canvas_width: 65_536,
                canvas_height: 65_536,
            }
        );
    }

    /// §2.4 file: simple-lossy round-trip through the parser. The
    /// chunk list, FourCCs, and payload bytes survive intact.
    #[test]
    fn build_webp_file_simple_lossy_round_trips_through_parser() {
        let payload = b"\xDE\xAD\xBE\xEF\x01\x02\x03"; // 7 bytes, odd → pad byte needed
        let bytes = build_webp_file(payload, ImageKind::Lossy, 0, 0).unwrap();
        // 12 (file header) + 8 (chunk header) + 7 (payload) + 1 (pad)
        assert_eq!(bytes.len(), 12 + 8 + 7 + 1);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WEBP");
        let c = parse(&bytes).expect("simple-lossy file built by builder parses");
        assert_eq!(c.chunks.len(), 1);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8);
        assert_eq!(c.chunks[0].size, 7);
        assert_eq!(c.chunks[0].payload(&bytes), payload);
        assert!(!c.is_extended());
        // §2.4: File Size = 4 ('WEBP') + body (16 bytes incl. pad) = 20.
        assert_eq!(c.riff_file_size, 20);
    }

    /// §2.4 file: simple-lossless layout produces a single `VP8L`
    /// chunk; even-payload exercises the no-pad path.
    #[test]
    fn build_webp_file_simple_lossless_uses_vp8l_chunk() {
        let payload = vec![0x2F, 0x00, 0x00, 0x00]; // 4 bytes, even
        let bytes = build_webp_file(&payload, ImageKind::Lossless, 0, 0).unwrap();
        // 12 + 8 + 4 + 0 (no pad).
        assert_eq!(bytes.len(), 12 + 8 + 4);
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 1);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8L);
        assert_eq!(c.chunks[0].payload(&bytes), payload.as_slice());
    }

    /// §2.7 extended-lossy: emits VP8X first, then VP8.
    #[test]
    fn build_webp_file_extended_lossy_emits_vp8x_then_vp8() {
        let payload = vec![0u8; 6]; // even, no pad
        let bytes = build_webp_file(&payload, ImageKind::ExtendedLossy, 320, 240).unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 2);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::VP8);
        assert!(c.is_extended());

        // VP8X payload decodes to the canvas dims we passed in.
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert_eq!(vp8x.canvas_width, 320);
        assert_eq!(vp8x.canvas_height, 240);
        // Default flags — none set.
        assert!(!vp8x.has_iccp);
        assert!(!vp8x.has_alpha);
        assert!(!vp8x.has_exif);
        assert!(!vp8x.has_xmp);
        assert!(!vp8x.has_animation);
        assert!(!vp8x.has_unknown);
    }

    /// §2.7 extended-lossless: emits VP8X first, then VP8L. Also
    /// exercises a 1x1 canvas at the §2.7.1 lower bound.
    #[test]
    fn build_webp_file_extended_lossless_emits_vp8x_then_vp8l() {
        let payload = vec![0u8; 5]; // odd → pad byte on VP8L
        let bytes = build_webp_file(&payload, ImageKind::ExtendedLossless, 1, 1).unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 2);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::VP8L);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert_eq!(vp8x.canvas_width, 1);
        assert_eq!(vp8x.canvas_height, 1);
        assert_eq!(c.chunks[1].payload(&bytes), &[0u8; 5]);
    }

    /// §2.4 File Size accounting matches the field the parser sees —
    /// counts `WEBP` (4) plus every byte of the chunk body (including
    /// pad bytes), and nothing else.
    #[test]
    fn build_webp_file_file_size_field_matches_parsed_value() {
        // Simple lossy with 7-byte payload (odd → +1 pad). Body =
        // 8 header + 7 payload + 1 pad = 16. File Size = 20.
        let bytes = build_webp_file(&[0u8; 7], ImageKind::Lossy, 0, 0).unwrap();
        let declared = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(declared, 20);

        // Extended lossy with 8-byte payload (even, no pad). Body =
        // (8 + 10 VP8X) + (8 + 8 VP8) = 18 + 16 = 34. File Size = 38.
        let bytes = build_webp_file(&[0u8; 8], ImageKind::ExtendedLossy, 100, 100).unwrap();
        let declared = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(declared, 38);
        // Sanity: declared payload-end matches the byte slice length.
        assert_eq!((declared as usize) + 8, bytes.len());
    }

    /// Canvas-validation errors on the extended layouts bubble out of
    /// `build_webp_file` (rather than panicking or producing a file
    /// the parser would reject).
    #[test]
    fn build_webp_file_extended_propagates_canvas_validation_errors() {
        // 0-width canvas.
        assert_eq!(
            build_webp_file(&[0u8; 4], ImageKind::ExtendedLossy, 0, 1).unwrap_err(),
            BuildError::CanvasDimZero { which: "width" }
        );
        // Product cap exceeded.
        assert_eq!(
            build_webp_file(&[0u8; 4], ImageKind::ExtendedLossless, 65_536, 65_536).unwrap_err(),
            BuildError::CanvasTooLarge {
                canvas_width: 65_536,
                canvas_height: 65_536,
            }
        );
    }

    /// Empty payloads are *allowed* — they produce a well-formed RIFF
    /// with a zero-size bitstream chunk. The parser accepts it; what
    /// the decoder does next is the decoder's problem.
    #[test]
    fn build_webp_file_empty_payload_is_a_well_formed_empty_chunk() {
        let bytes = build_webp_file(&[], ImageKind::Lossy, 0, 0).unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 1);
        assert_eq!(c.chunks[0].size, 0);
        assert!(c.chunks[0].payload(&bytes).is_empty());
    }

    /// Round-trip with a longer (multi-block) payload to catch any
    /// off-by-one issues in the Vec growth / cursor arithmetic.
    #[test]
    fn build_webp_file_round_trip_preserves_64kib_payload_byte_for_byte() {
        let mut payload = Vec::with_capacity(65_535);
        for i in 0..65_535 {
            payload.push((i & 0xFF) as u8);
        }
        let bytes = build_webp_file(&payload, ImageKind::Lossless, 0, 0).unwrap();
        let c = parse(&bytes).expect("64 KiB payload parses");
        assert_eq!(c.chunks.len(), 1);
        assert_eq!(c.chunks[0].size as usize, payload.len());
        assert_eq!(c.chunks[0].payload(&bytes), payload.as_slice());
        // Odd payload → §2.3 pad byte; the parser still walks cleanly.
    }

    // ─────────────────────── build_webp_file_with_metadata ───────────────────────

    /// §2.7 canonical chunk order when no metadata is present: VP8X
    /// then the bitstream chunk, nothing else.
    #[test]
    fn build_with_metadata_emits_vp8x_then_payload_when_no_metadata() {
        let payload = vec![0u8; 6];
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossy,
            64,
            32,
            false,
            FileMetadata::default(),
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 2);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::VP8);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert!(!vp8x.has_iccp);
        assert!(!vp8x.has_alpha);
        assert!(!vp8x.has_exif);
        assert!(!vp8x.has_xmp);
        assert!(!vp8x.has_animation);
    }

    /// §2.7.1.4 ICCP-only round trip: chunk lands before the bitstream,
    /// §2.7.1 `I` flag set, payload survives.
    #[test]
    fn build_with_metadata_iccp_only_round_trips() {
        let payload = vec![0u8; 4];
        let iccp = b"icc-profile-bytes".to_vec();
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            16,
            16,
            false,
            FileMetadata {
                iccp: Some(&iccp),
                ..Default::default()
            },
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        // Order: VP8X, ICCP, VP8L.
        assert_eq!(c.chunks.len(), 3);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::ICCP);
        assert_eq!(c.chunks[2].fourcc, fourcc::VP8L);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert!(vp8x.has_iccp);
        assert!(!vp8x.has_exif);
        assert!(!vp8x.has_xmp);
        // Extracted ICCP payload matches.
        let m = crate::extract_metadata(&bytes).unwrap();
        assert_eq!(m.icc.as_deref(), Some(&iccp[..]));
        assert_eq!(m.exif, None);
        assert_eq!(m.xmp, None);
    }

    /// §2.7.1.5 EXIF-only round trip: §2.7 says EXIF lands *after* the
    /// bitstream.
    #[test]
    fn build_with_metadata_exif_only_round_trips() {
        let payload = vec![0u8; 4];
        let exif = b"Exif\x00\x00MM\x00*".to_vec();
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            8,
            8,
            false,
            FileMetadata {
                exif: Some(&exif),
                ..Default::default()
            },
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 3);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::VP8L);
        assert_eq!(c.chunks[2].fourcc, fourcc::EXIF);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert!(!vp8x.has_iccp);
        assert!(vp8x.has_exif);
        assert!(!vp8x.has_xmp);
        let m = crate::extract_metadata(&bytes).unwrap();
        assert_eq!(m.icc, None);
        assert_eq!(m.exif.as_deref(), Some(&exif[..]));
        assert_eq!(m.xmp, None);
    }

    /// §2.7.1.5 XMP-only round trip.
    #[test]
    fn build_with_metadata_xmp_only_round_trips() {
        let payload = vec![0u8; 4];
        let xmp = b"<?xpacket begin?>".to_vec();
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            8,
            8,
            false,
            FileMetadata {
                xmp: Some(&xmp),
                ..Default::default()
            },
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 3);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::VP8L);
        assert_eq!(c.chunks[2].fourcc, fourcc::XMP);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert!(vp8x.has_xmp);
        let m = crate::extract_metadata(&bytes).unwrap();
        assert_eq!(m.xmp.as_deref(), Some(&xmp[..]));
    }

    /// ICCP + EXIF together: ICCP before the bitstream, EXIF after,
    /// both flag bits set.
    #[test]
    fn build_with_metadata_iccp_plus_exif_round_trips() {
        let payload = vec![0u8; 4];
        let iccp = b"icc".to_vec();
        let exif = b"Exif\x00\x00MM\x00*more".to_vec();
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            8,
            8,
            false,
            FileMetadata {
                iccp: Some(&iccp),
                exif: Some(&exif),
                ..Default::default()
            },
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 4);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::ICCP);
        assert_eq!(c.chunks[2].fourcc, fourcc::VP8L);
        assert_eq!(c.chunks[3].fourcc, fourcc::EXIF);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert!(vp8x.has_iccp);
        assert!(vp8x.has_exif);
        assert!(!vp8x.has_xmp);
        let m = crate::extract_metadata(&bytes).unwrap();
        assert_eq!(m.icc.as_deref(), Some(&iccp[..]));
        assert_eq!(m.exif.as_deref(), Some(&exif[..]));
    }

    /// ICCP + XMP together: ICCP before, XMP after, no EXIF.
    #[test]
    fn build_with_metadata_iccp_plus_xmp_round_trips() {
        let payload = vec![0u8; 4];
        let iccp = b"icc-bytes-here".to_vec();
        let xmp = b"<xmp/>".to_vec();
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            8,
            8,
            false,
            FileMetadata {
                iccp: Some(&iccp),
                xmp: Some(&xmp),
                ..Default::default()
            },
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 4);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::ICCP);
        assert_eq!(c.chunks[2].fourcc, fourcc::VP8L);
        assert_eq!(c.chunks[3].fourcc, fourcc::XMP);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert!(vp8x.has_iccp);
        assert!(!vp8x.has_exif);
        assert!(vp8x.has_xmp);
        let m = crate::extract_metadata(&bytes).unwrap();
        assert_eq!(m.icc.as_deref(), Some(&iccp[..]));
        assert_eq!(m.xmp.as_deref(), Some(&xmp[..]));
    }

    /// EXIF + XMP together: both land after the bitstream, in §2.7 order
    /// (EXIF before XMP).
    #[test]
    fn build_with_metadata_exif_plus_xmp_round_trips() {
        let payload = vec![0u8; 4];
        let exif = b"E".to_vec();
        let xmp = b"X".to_vec();
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            8,
            8,
            false,
            FileMetadata {
                exif: Some(&exif),
                xmp: Some(&xmp),
                ..Default::default()
            },
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 4);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::VP8L);
        assert_eq!(c.chunks[2].fourcc, fourcc::EXIF);
        assert_eq!(c.chunks[3].fourcc, fourcc::XMP);
        let m = crate::extract_metadata(&bytes).unwrap();
        assert_eq!(m.exif.as_deref(), Some(&exif[..]));
        assert_eq!(m.xmp.as_deref(), Some(&xmp[..]));
    }

    /// All three metadata kinds together: VP8X | ICCP | bitstream |
    /// EXIF | XMP, every flag bit set.
    #[test]
    fn build_with_metadata_all_three_round_trip_in_canonical_order() {
        let payload = vec![0u8; 4];
        let iccp = b"ICC-profile-blob".to_vec();
        let exif = b"Exif\x00\x00II*\x00".to_vec();
        let xmp = b"<x:xmpmeta/>".to_vec();
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            16,
            16,
            true, // also flips the L bit
            FileMetadata {
                iccp: Some(&iccp),
                exif: Some(&exif),
                xmp: Some(&xmp),
            },
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 5);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::ICCP);
        assert_eq!(c.chunks[2].fourcc, fourcc::VP8L);
        assert_eq!(c.chunks[3].fourcc, fourcc::EXIF);
        assert_eq!(c.chunks[4].fourcc, fourcc::XMP);
        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
        assert!(vp8x.has_iccp);
        assert!(vp8x.has_alpha);
        assert!(vp8x.has_exif);
        assert!(vp8x.has_xmp);
        assert!(!vp8x.has_animation);
        // §2.7.1 canvas dims survive.
        assert_eq!(vp8x.canvas_width, 16);
        assert_eq!(vp8x.canvas_height, 16);
        let m = crate::extract_metadata(&bytes).unwrap();
        assert_eq!(m.icc.as_deref(), Some(&iccp[..]));
        assert_eq!(m.exif.as_deref(), Some(&exif[..]));
        assert_eq!(m.xmp.as_deref(), Some(&xmp[..]));
    }

    /// §2.3 odd-length metadata payloads trigger a single `0x00` pad
    /// byte per chunk (not counted in `Size`), and the parser still
    /// walks cleanly.
    #[test]
    fn build_with_metadata_odd_payloads_get_pad_bytes() {
        // Three odd-length metadata payloads + an odd-length bitstream.
        let payload = vec![0xABu8, 0xCD, 0xEF, 0x01, 0x02]; // 5 bytes (odd)
        let iccp = b"AAA".to_vec(); // 3 bytes (odd)
        let exif = b"BBBBB".to_vec(); // 5 bytes (odd)
        let xmp = b"CCCCCCC".to_vec(); // 7 bytes (odd)
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            8,
            8,
            false,
            FileMetadata {
                iccp: Some(&iccp),
                exif: Some(&exif),
                xmp: Some(&xmp),
            },
        )
        .unwrap();
        // Each odd chunk contributes 8 (header) + len + 1 (pad).
        // VP8X is 10 bytes (even → no pad). Total body:
        // (8+10) + (8+3+1) + (8+5+1) + (8+5+1) + (8+7+1) = 18+12+14+14+16 = 74.
        // File Size = 4 ('WEBP') + 74 = 78.
        let declared = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(declared, 78);
        // Parser-level Size fields exclude the pad byte each.
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks[1].size, 3, "ICCP §Size excludes pad byte");
        assert_eq!(c.chunks[2].size, 5, "VP8L §Size excludes pad byte");
        assert_eq!(c.chunks[3].size, 5, "EXIF §Size excludes pad byte");
        assert_eq!(c.chunks[4].size, 7, "XMP §Size excludes pad byte");
        // Payload survives intact through the parser (the pad byte sits
        // outside `payload()`).
        assert_eq!(c.chunks[1].payload(&bytes), &iccp[..]);
        assert_eq!(c.chunks[2].payload(&bytes), &payload[..]);
        assert_eq!(c.chunks[3].payload(&bytes), &exif[..]);
        assert_eq!(c.chunks[4].payload(&bytes), &xmp[..]);
    }

    /// §2.7.1 flag-bit derivation: each `Some(..)` field independently
    /// flips the corresponding bit, every other combination clears it.
    /// Exhaustively covers the 2^3 metadata-presence states (× the
    /// `has_alpha` × 2 axis for the L bit).
    #[test]
    fn build_with_metadata_flag_bits_match_field_presence() {
        let payload = vec![0u8; 2];
        for has_icc in [false, true] {
            for has_exif_p in [false, true] {
                for has_xmp_p in [false, true] {
                    for has_alpha in [false, true] {
                        let icc_blob: &[u8] = &[0xAA];
                        let exif_blob: &[u8] = &[0xBB];
                        let xmp_blob: &[u8] = &[0xCC];
                        let metadata = FileMetadata {
                            iccp: has_icc.then_some(icc_blob),
                            exif: has_exif_p.then_some(exif_blob),
                            xmp: has_xmp_p.then_some(xmp_blob),
                        };
                        let bytes = build_webp_file_with_metadata(
                            &payload,
                            ImageKind::ExtendedLossless,
                            4,
                            4,
                            has_alpha,
                            metadata,
                        )
                        .unwrap();
                        let c = parse(&bytes).unwrap();
                        let vp8x = Vp8xHeader::parse(c.chunks[0].payload(&bytes)).unwrap();
                        assert_eq!(
                            vp8x.has_iccp, has_icc,
                            "I flag (has_icc={has_icc}, has_exif={has_exif_p}, has_xmp={has_xmp_p}, has_alpha={has_alpha})"
                        );
                        assert_eq!(
                            vp8x.has_alpha, has_alpha,
                            "L flag (has_icc={has_icc}, has_exif={has_exif_p}, has_xmp={has_xmp_p}, has_alpha={has_alpha})"
                        );
                        assert_eq!(
                            vp8x.has_exif, has_exif_p,
                            "E flag (has_icc={has_icc}, has_exif={has_exif_p}, has_xmp={has_xmp_p}, has_alpha={has_alpha})"
                        );
                        assert_eq!(
                            vp8x.has_xmp, has_xmp_p,
                            "X flag (has_icc={has_icc}, has_exif={has_exif_p}, has_xmp={has_xmp_p}, has_alpha={has_alpha})"
                        );
                        // Animation always off — this writer never emits ANIM/ANMF.
                        assert!(!vp8x.has_animation);
                    }
                }
            }
        }
    }

    /// Canvas-validation failures propagate out of the metadata writer
    /// without producing a half-baked file.
    #[test]
    fn build_with_metadata_propagates_canvas_validation_errors() {
        assert_eq!(
            build_webp_file_with_metadata(
                &[0u8; 4],
                ImageKind::ExtendedLossless,
                0,
                1,
                false,
                FileMetadata::default(),
            )
            .unwrap_err(),
            BuildError::CanvasDimZero { which: "width" }
        );
        assert_eq!(
            build_webp_file_with_metadata(
                &[0u8; 4],
                ImageKind::ExtendedLossless,
                65_536,
                65_536,
                false,
                FileMetadata::default(),
            )
            .unwrap_err(),
            BuildError::CanvasTooLarge {
                canvas_width: 65_536,
                canvas_height: 65_536,
            }
        );
    }

    /// FileMetadata::is_empty mirrors "every field is None" — and the
    /// writer with an empty metadata struct on a VP8L payload still
    /// emits a parseable file (just VP8X + bitstream, no metadata
    /// chunks).
    #[test]
    fn build_with_metadata_empty_metadata_omits_optional_chunks() {
        assert!(FileMetadata::default().is_empty());
        let payload = vec![0u8; 8];
        let bytes = build_webp_file_with_metadata(
            &payload,
            ImageKind::ExtendedLossless,
            4,
            4,
            false,
            FileMetadata::default(),
        )
        .unwrap();
        let c = parse(&bytes).unwrap();
        assert_eq!(c.chunks.len(), 2);
        assert_eq!(c.chunks[0].fourcc, fourcc::VP8X);
        assert_eq!(c.chunks[1].fourcc, fourcc::VP8L);
        let m = crate::extract_metadata(&bytes).unwrap();
        assert!(m.icc.is_none());
        assert!(m.exif.is_none());
        assert!(m.xmp.is_none());
    }

    /// Negative: a hand-crafted file with a chunk Size that runs past
    /// the buffer is rejected by the parser. (Sanity check that we're
    /// actually exercising the same parser the builders need to round-
    /// trip through.)
    #[test]
    fn parser_still_rejects_corrupt_size_field_after_builder_round_trip() {
        let mut bytes = build_webp_file(&[0u8; 8], ImageKind::Lossy, 0, 0).unwrap();
        // Corrupt the VP8 chunk's Size to a huge value.
        let chunk_size_off = 12 + 4;
        bytes[chunk_size_off..chunk_size_off + 4].copy_from_slice(&100_000u32.to_le_bytes());
        match parse(&bytes) {
            Err(ContainerError::ChunkPayloadOverflowsRiff { .. }) => {}
            other => panic!("expected ChunkPayloadOverflowsRiff, got {other:?}"),
        }
    }
}
