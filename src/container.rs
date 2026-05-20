//! RIFF/WEBP container walker per RFC 9649 (WebP Image Format).
//!
//! This module covers only the **structural** layer of WebP:
//!
//! * §2.3 — Generic RIFF chunk: 4-byte FourCC, 4-byte little-endian
//!   uint32 size, payload, and (if size is odd) a single 0 padding
//!   byte that is not counted in the size field.
//! * §2.4 — WebP file header: 12 bytes total — the ASCII tag `RIFF`,
//!   a 32-bit little-endian *File Size* counting everything after
//!   offset 8, then the ASCII tag `WEBP`.
//! * §2.5 / §2.6 / §2.7 — the layouts that follow: simple lossy
//!   (`VP8 `), simple lossless (`VP8L`), and extended (`VP8X` plus
//!   any of `ICCP`, `ANIM`, `ANMF`, `ALPH`, `VP8 `/`VP8L`, `EXIF`,
//!   `XMP `, plus unknown chunks per §2.7.1.6).
//!
//! Decoding of `VP8 ` / `VP8L` / `ALPH` payloads is **out of scope**
//! for this layer; the walker only records FourCC + payload range.

use core::fmt;

/// Fixed 4-byte FourCC tag as carried on disk (preserving the
/// trailing space in `"VP8 "` and `"XMP "`).
pub type FourCc = [u8; 4];

/// FourCC tags called out by name in RFC 9649 §2.4–§2.7.
pub mod fourcc {
    use super::FourCc;

    /// `"RIFF"` — opening tag of the §2.4 WebP file header.
    pub const RIFF: FourCc = *b"RIFF";
    /// `"WEBP"` — form-type tag of the §2.4 WebP file header.
    pub const WEBP: FourCc = *b"WEBP";

    /// `"VP8 "` (with trailing 0x20) — §2.5 / §2.7.1.3 lossy bitstream.
    pub const VP8: FourCc = *b"VP8 ";
    /// `"VP8L"` — §2.6 / §2.7.1.3 lossless bitstream.
    pub const VP8L: FourCc = *b"VP8L";
    /// `"VP8X"` — §2.7 extended-format flags + canvas dimensions.
    pub const VP8X: FourCc = *b"VP8X";
    /// `"ALPH"` — §2.7.1.2 alpha plane (used with `VP8 `).
    pub const ALPH: FourCc = *b"ALPH";
    /// `"ANIM"` — §2.7.1.1 animation control.
    pub const ANIM: FourCc = *b"ANIM";
    /// `"ANMF"` — §2.7.1.1 per-frame chunk.
    pub const ANMF: FourCc = *b"ANMF";
    /// `"ICCP"` — §2.7.1.4 ICC color profile.
    pub const ICCP: FourCc = *b"ICCP";
    /// `"EXIF"` — §2.7.1.5 Exif metadata.
    pub const EXIF: FourCc = *b"EXIF";
    /// `"XMP "` (with trailing 0x20) — §2.7.1.5 XMP metadata.
    pub const XMP: FourCc = *b"XMP ";
}

/// Errors raised by the RIFF/WEBP walker. The walker reports the
/// *first* structural problem it sees and stops — it is not a
/// recovery layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerError {
    /// The buffer is shorter than the 12-byte §2.4 file header.
    TooShortForHeader { got: usize },
    /// Bytes 0..4 are not the ASCII tag `RIFF`.
    NotRiff { got: FourCc },
    /// Bytes 8..12 are not the ASCII tag `WEBP`.
    NotWebp { got: FourCc },
    /// The §2.4 `File Size` field says the payload extends past the
    /// end of the buffer. The header `File Size` counts the
    /// `WEBP` FourCC plus everything after it.
    RiffSizeOverflowsBuffer {
        /// `File Size` value as parsed from bytes 4..8.
        declared: u32,
        /// Total buffer length the walker was given.
        buffer_len: usize,
    },
    /// A chunk header was truncated — the 8 bytes required to read
    /// `FourCC + Size` do not fit in what remains of the RIFF
    /// payload at `offset`.
    TruncatedChunkHeader {
        /// Absolute offset (from the start of the buffer) where the
        /// truncated header begins.
        offset: usize,
    },
    /// A chunk's declared `Size` value runs past the end of the RIFF
    /// payload — i.e. payload `Size` bytes would extend beyond the
    /// region delimited by the outer §2.4 file header.
    ChunkPayloadOverflowsRiff {
        /// Absolute offset where the offending chunk's header starts.
        offset: usize,
        /// `Size` value as parsed from the chunk header.
        declared: u32,
        /// Number of payload bytes the walker actually had room for.
        available: usize,
    },
    /// A chunk's declared `Size` value is odd, and the §2.3 padding
    /// byte that would follow it is missing because there are no
    /// further bytes in the RIFF payload.
    MissingPadByte {
        /// Absolute offset of the chunk header whose payload ends
        /// without its required pad byte.
        offset: usize,
    },
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShortForHeader { got } => write!(
                f,
                "WebP buffer too short for §2.4 file header (12 bytes), got {got}"
            ),
            Self::NotRiff { got } => write!(
                f,
                "WebP buffer does not start with §2.4 'RIFF' tag (got {:02x?})",
                got
            ),
            Self::NotWebp { got } => write!(
                f,
                "WebP buffer is RIFF but not 'WEBP' form (got {:02x?})",
                got
            ),
            Self::RiffSizeOverflowsBuffer {
                declared,
                buffer_len,
            } => write!(
                f,
                "§2.4 RIFF File Size {declared} overflows buffer length {buffer_len}"
            ),
            Self::TruncatedChunkHeader { offset } => write!(
                f,
                "§2.3 chunk header at offset {offset} is truncated (need 8 bytes)"
            ),
            Self::ChunkPayloadOverflowsRiff {
                offset,
                declared,
                available,
            } => write!(
                f,
                "§2.3 chunk at offset {offset} declares Size {declared} \
                 but only {available} bytes remain in the RIFF payload"
            ),
            Self::MissingPadByte { offset } => write!(
                f,
                "§2.3 chunk at offset {offset} has odd Size but no trailing pad byte"
            ),
        }
    }
}

impl std::error::Error for ContainerError {}

/// One §2.3 RIFF chunk inside a WebP file.
///
/// The walker records the FourCC, the declared payload size, and the
/// absolute `(start, end)` byte range of the payload inside the input
/// buffer. Borrowing the payload bytes is left to the caller to keep
/// this struct cheap to move and copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebpChunk {
    /// Four-byte FourCC tag.
    pub fourcc: FourCc,
    /// `Size` field as declared in the §2.3 chunk header.
    pub size: u32,
    /// Absolute offset (from buffer start) of the first payload byte.
    pub payload_start: usize,
    /// Absolute offset (from buffer start) of one past the last
    /// payload byte. `payload_end - payload_start == size as usize`.
    pub payload_end: usize,
}

impl WebpChunk {
    /// Borrow the payload bytes out of the original input slice.
    pub fn payload<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        &buf[self.payload_start..self.payload_end]
    }

    /// True if the FourCC is `"VP8 "` (note the trailing space).
    pub fn is_vp8_lossy(&self) -> bool {
        self.fourcc == fourcc::VP8
    }

    /// True if the FourCC is `"VP8L"`.
    pub fn is_vp8_lossless(&self) -> bool {
        self.fourcc == fourcc::VP8L
    }

    /// True if the FourCC is `"VP8X"`.
    pub fn is_extended(&self) -> bool {
        self.fourcc == fourcc::VP8X
    }
}

/// Output of the §2.3–§2.7 RIFF walk.
///
/// Holds the §2.4 declared `File Size` plus the ordered list of
/// chunks discovered inside the RIFF payload. The walker does not
/// re-order chunks — they appear in the order on disk so that
/// downstream code can apply §2.7's ordering rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebpContainer {
    /// `File Size` field from bytes 4..8 of the §2.4 file header.
    pub riff_file_size: u32,
    /// Chunks parsed from the §2.4 file header's payload, in the
    /// order they appear on disk.
    pub chunks: Vec<WebpChunk>,
}

impl WebpContainer {
    /// Iterate over chunks matching a given FourCC.
    pub fn chunks_with_fourcc(&self, fourcc: FourCc) -> impl Iterator<Item = &WebpChunk> + '_ {
        self.chunks.iter().filter(move |c| c.fourcc == fourcc)
    }

    /// Find the first chunk matching a given FourCC, if any.
    pub fn first_chunk_with_fourcc(&self, fourcc: FourCc) -> Option<&WebpChunk> {
        self.chunks.iter().find(|c| c.fourcc == fourcc)
    }

    /// True if the container starts with `VP8X`, indicating §2.7
    /// extended layout. (The §2.7 ordering rule places `VP8X` first
    /// among the structural chunks when present.)
    pub fn is_extended(&self) -> bool {
        self.chunks
            .first()
            .map(|c| c.is_extended())
            .unwrap_or(false)
    }
}

/// Walk a `RIFF/WEBP` container per RFC 9649 §2.3–§2.7 and return
/// the list of chunks discovered.
///
/// The walker enforces structural invariants only:
///
/// * `buf` is at least 12 bytes.
/// * Bytes `0..4` are `"RIFF"` and bytes `8..12` are `"WEBP"`.
/// * The §2.4 `File Size` field does not overflow `buf`.
/// * Each subsequent chunk header is 8 bytes; the declared `Size`
///   fits inside the remaining RIFF payload; and if `Size` is odd
///   the required §2.3 pad byte is present.
///
/// Per-chunk *content* validation (e.g. the VP8X reserved bits,
/// VP8 frame width/height, animation counts) is the responsibility
/// of layers above this walker.
pub fn parse(buf: &[u8]) -> Result<WebpContainer, ContainerError> {
    // §2.4 file header — 12 bytes.
    if buf.len() < 12 {
        return Err(ContainerError::TooShortForHeader { got: buf.len() });
    }
    let riff_tag: FourCc = buf[0..4]
        .try_into()
        .expect("12-byte slice always has 4 bytes at offset 0");
    if riff_tag != fourcc::RIFF {
        return Err(ContainerError::NotRiff { got: riff_tag });
    }
    let riff_file_size = u32::from_le_bytes(
        buf[4..8]
            .try_into()
            .expect("12-byte slice always has 4 bytes at offset 4"),
    );
    let webp_tag: FourCc = buf[8..12]
        .try_into()
        .expect("12-byte slice always has 4 bytes at offset 8");
    if webp_tag != fourcc::WEBP {
        return Err(ContainerError::NotWebp { got: webp_tag });
    }

    // §2.4: "The file size in the header is the total size of the
    // chunks that follow plus 4 bytes for the 'WEBP' FourCC."
    //
    // So the RIFF payload (the bytes after the 8-byte 'RIFF' +
    // File Size header) is exactly `riff_file_size` bytes long, of
    // which the first 4 are the 'WEBP' FourCC and the remainder are
    // the chunk stream. The walker tolerates trailing data beyond
    // `riff_file_size` per §2.4 ("Readers MAY parse such files,
    // ignoring the trailing data") but it never *reads* past that
    // declared limit when walking chunks.
    let declared_payload_end = 8usize.saturating_add(riff_file_size as usize);
    if declared_payload_end > buf.len() {
        return Err(ContainerError::RiffSizeOverflowsBuffer {
            declared: riff_file_size,
            buffer_len: buf.len(),
        });
    }
    let chunk_stream_end = declared_payload_end;

    let mut chunks: Vec<WebpChunk> = Vec::new();
    let mut cursor: usize = 12; // first byte after the §2.4 header
    while cursor < chunk_stream_end {
        // Need 8 bytes for FourCC + Size.
        if chunk_stream_end - cursor < 8 {
            return Err(ContainerError::TruncatedChunkHeader { offset: cursor });
        }
        let fourcc: FourCc = buf[cursor..cursor + 4]
            .try_into()
            .expect("bounds checked above");
        let size = u32::from_le_bytes(
            buf[cursor + 4..cursor + 8]
                .try_into()
                .expect("bounds checked above"),
        );

        let payload_start = cursor + 8;
        let payload_avail = chunk_stream_end - payload_start;
        if (size as usize) > payload_avail {
            return Err(ContainerError::ChunkPayloadOverflowsRiff {
                offset: cursor,
                declared: size,
                available: payload_avail,
            });
        }
        let payload_end = payload_start + size as usize;

        chunks.push(WebpChunk {
            fourcc,
            size,
            payload_start,
            payload_end,
        });

        // §2.3 padding: if Size is odd, a single 0 byte follows that
        // is *not* counted in Size. The walker requires that byte to
        // be present (but does not check its value — §2.3 says it
        // MUST be 0; that's a writer constraint, not a reader
        // refusal mode).
        let needs_pad = (size & 1) == 1;
        let total = if needs_pad {
            (size as usize).checked_add(1)
        } else {
            Some(size as usize)
        }
        .expect("size+1 cannot overflow because size <= payload_avail < usize::MAX");
        let after_chunk =
            payload_start
                .checked_add(total)
                .ok_or(ContainerError::ChunkPayloadOverflowsRiff {
                    offset: cursor,
                    declared: size,
                    available: payload_avail,
                })?;
        if after_chunk > chunk_stream_end {
            return Err(ContainerError::MissingPadByte { offset: cursor });
        }
        cursor = after_chunk;
    }

    Ok(WebpContainer {
        riff_file_size,
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a §2.3 chunk header + payload + (if odd) one pad byte.
    fn chunk(fourcc: &FourCc, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 + payload.len() + 1);
        v.extend_from_slice(fourcc);
        v.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        v.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            v.push(0);
        }
        v
    }

    /// Wrap a sequence of already-formed chunks in a §2.4 WebP file
    /// header, setting `File Size` to `4 + sum_of_chunk_bytes`.
    fn webp(chunks: &[u8]) -> Vec<u8> {
        let file_size = 4u32 + chunks.len() as u32;
        let mut v = Vec::with_capacity(12 + chunks.len());
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&file_size.to_le_bytes());
        v.extend_from_slice(b"WEBP");
        v.extend_from_slice(chunks);
        v
    }

    #[test]
    fn simple_lossy_walks_to_one_vp8_chunk() {
        // §2.5: WebP file header + a single 'VP8 ' chunk with a
        // 7-byte payload (odd, exercises the §2.3 pad byte).
        let body = chunk(&fourcc::VP8, &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03]);
        let buf = webp(&body);
        let c = parse(&buf).expect("simple lossy parses");
        assert_eq!(c.riff_file_size, 4 + body.len() as u32);
        assert_eq!(c.chunks.len(), 1);
        let only = &c.chunks[0];
        assert!(only.is_vp8_lossy());
        assert_eq!(only.size, 7);
        assert_eq!(
            only.payload(&buf),
            &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03]
        );
        assert!(!c.is_extended());
    }

    #[test]
    fn simple_lossless_walks_to_one_vp8l_chunk() {
        // §2.6: WebP file header + a single 'VP8L' chunk with a
        // 4-byte payload (even, no §2.3 pad byte).
        let body = chunk(&fourcc::VP8L, &[0x2F, 0x00, 0x00, 0x00]);
        let buf = webp(&body);
        let c = parse(&buf).expect("simple lossless parses");
        assert_eq!(c.chunks.len(), 1);
        let only = &c.chunks[0];
        assert!(only.is_vp8_lossless());
        assert_eq!(only.size, 4);
        assert_eq!(only.payload(&buf), &[0x2F, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn extended_layout_walks_all_chunks_in_order() {
        // §2.7 example: VP8X + ICCP + ANIM + ANMF (+ inner VP8 ) +
        // EXIF + XMP . The walker should record them in the order
        // they appear on disk and surface every FourCC.
        let vp8x_payload = vec![
            0x10, 0x00, 0x00, 0x00, // Rsv|I|L|E|X|A|R + 24 bits reserved
            0x07, 0x00, 0x00, // Canvas Width Minus One = 7  (width 8)
            0x07, 0x00, 0x00, // Canvas Height Minus One = 7 (height 8)
        ];
        let mut body = Vec::new();
        body.extend(chunk(&fourcc::VP8X, &vp8x_payload));
        body.extend(chunk(&fourcc::ICCP, &[0xAA; 5])); // odd payload exercises pad
        body.extend(chunk(&fourcc::ANIM, &[0; 6]));
        body.extend(chunk(&fourcc::ANMF, &[0; 9])); // odd payload
        body.extend(chunk(&fourcc::VP8, &[0; 8]));
        body.extend(chunk(&fourcc::EXIF, b"Exif\x00\x00MM*\x00"));
        body.extend(chunk(&fourcc::XMP, b"<?xpacket?>"));
        let buf = webp(&body);

        let c = parse(&buf).expect("extended layout parses");
        let order: Vec<FourCc> = c.chunks.iter().map(|c| c.fourcc).collect();
        assert_eq!(
            order,
            vec![
                fourcc::VP8X,
                fourcc::ICCP,
                fourcc::ANIM,
                fourcc::ANMF,
                fourcc::VP8,
                fourcc::EXIF,
                fourcc::XMP,
            ]
        );
        assert!(c.is_extended());
        assert_eq!(c.first_chunk_with_fourcc(fourcc::ICCP).unwrap().size, 5);
        assert_eq!(c.chunks_with_fourcc(fourcc::VP8).count(), 1);

        // Spot-check the ICCP payload survived the §2.3 pad byte.
        let iccp = c.first_chunk_with_fourcc(fourcc::ICCP).unwrap();
        assert_eq!(iccp.payload(&buf), &[0xAA, 0xAA, 0xAA, 0xAA, 0xAA]);
    }

    #[test]
    fn rejects_buffer_shorter_than_file_header() {
        // §2.4 requires 12 bytes; supply only 11.
        let buf = b"RIFF\x00\x00\x00\x00WEB";
        assert_eq!(
            parse(buf),
            Err(ContainerError::TooShortForHeader { got: 11 })
        );
    }

    #[test]
    fn rejects_wrong_riff_or_form_tag() {
        // First the 'RIFF' tag itself is wrong.
        let mut buf = b"riff\x04\x00\x00\x00WEBP".to_vec();
        match parse(&buf) {
            Err(ContainerError::NotRiff { got }) => assert_eq!(&got, b"riff"),
            other => panic!("expected NotRiff, got {other:?}"),
        }

        // Now 'RIFF' but a non-'WEBP' form type — §2.4 demands WEBP.
        buf[0..4].copy_from_slice(b"RIFF");
        buf[8..12].copy_from_slice(b"AVI ");
        match parse(&buf) {
            Err(ContainerError::NotWebp { got }) => assert_eq!(&got, b"AVI "),
            other => panic!("expected NotWebp, got {other:?}"),
        }
    }

    #[test]
    fn rejects_chunk_whose_size_overflows_riff_payload() {
        // A 'VP8 ' header that claims Size = 100 in a RIFF whose
        // payload only has 8 + 0 bytes of room for the chunk.
        let mut bad = Vec::new();
        bad.extend_from_slice(b"VP8 ");
        bad.extend_from_slice(&100u32.to_le_bytes()); // declared size 100
                                                      // Wrap in a §2.4 header that says File Size = 4 (just WEBP)
                                                      // + 8 (the bad chunk header). The chunk's declared 100-byte
                                                      // payload doesn't fit in the 0 remaining bytes.
        let buf = webp(&bad);
        match parse(&buf) {
            Err(ContainerError::ChunkPayloadOverflowsRiff {
                offset,
                declared,
                available,
            }) => {
                assert_eq!(offset, 12);
                assert_eq!(declared, 100);
                assert_eq!(available, 0);
            }
            other => panic!("expected ChunkPayloadOverflowsRiff, got {other:?}"),
        }
    }

    #[test]
    fn rejects_odd_chunk_missing_pad_byte() {
        // Hand-craft a RIFF whose declared File Size accounts for an
        // odd-length chunk **without** including its §2.3 pad byte.
        // The walker should refuse rather than read past the end of
        // the declared payload.
        let mut chunk_bytes = Vec::new();
        chunk_bytes.extend_from_slice(b"ICCP");
        chunk_bytes.extend_from_slice(&3u32.to_le_bytes()); // odd size
        chunk_bytes.extend_from_slice(&[0xDE, 0xAD, 0xBE]); // 3 payload bytes, NO pad

        // File Size = 4 ('WEBP') + len(chunk_bytes); deliberately
        // no extra trailing pad byte beyond what we wrote.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(4u32 + chunk_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&chunk_bytes);

        match parse(&buf) {
            Err(ContainerError::MissingPadByte { offset }) => assert_eq!(offset, 12),
            other => panic!("expected MissingPadByte, got {other:?}"),
        }
    }

    #[test]
    fn rejects_riff_size_that_runs_past_buffer() {
        // Header says File Size = 1000 but we only supply the
        // 12-byte header itself.
        let mut buf = b"RIFF".to_vec();
        buf.extend_from_slice(&1000u32.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        match parse(&buf) {
            Err(ContainerError::RiffSizeOverflowsBuffer {
                declared,
                buffer_len,
            }) => {
                assert_eq!(declared, 1000);
                assert_eq!(buffer_len, 12);
            }
            other => panic!("expected RiffSizeOverflowsBuffer, got {other:?}"),
        }
    }
}
