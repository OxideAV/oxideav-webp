//! Typed §2.5 `VP8 ` chunk handle.
//!
//! RFC 9649 §2.5 ("Simple File Format (Lossy)") wraps a single
//! `VP8 ` chunk whose payload is an RFC 6386 VP8 bitstream. §2.5
//! also notes:
//!
//! > Note that the VP8 frame header contains the VP8 frame width
//! > and height. That is assumed to be the width and height of
//! > the canvas.
//!
//! This module gives container-layer callers a way to route the
//! `VP8 ` payload to a downstream VP8 decoder **without** the
//! `oxideav-webp` crate taking a runtime dependency on
//! `oxideav-vp8`. A [`WebpLossyChunk`] is a thin typed handle that:
//!
//! * borrows the chunk payload byte-for-byte (no copy);
//! * peeks the 10-byte RFC 6386 §9.1 keyframe header so the
//!   container layer can surface `(width, height)` to the caller
//!   without descending into the entropy-coded portion of the VP8
//!   bitstream;
//! * refuses non-keyframe inputs, which §2.5 implicitly forbids
//!   (a WebP file carries a single intra-only frame; the §9.1
//!   3-byte frame tag's `frame_type` bit MUST be 0).
//!
//! The actual VP8 decode — DCT, intra prediction, loop filter,
//! boolean entropy coder, etc. — is **not** in scope here. That
//! work lives in the `oxideav-vp8` sibling crate. The contract is
//! one-way: this module emits a typed payload-bytes handle, and
//! the caller routes those bytes to whatever VP8 decoder it
//! wishes.
//!
//! ## RFC 6386 §9.1 frame tag layout
//!
//! Per RFC 6386 §9.1, the first 3 payload bytes are the VP8 frame
//! tag, packed little-endian:
//!
//! ```text
//!   byte 0 bit 0       — frame type (0 = key frame, 1 = interframe)
//!   byte 0 bits 1..3   — version (0..3 defined; others reserved)
//!   byte 0 bit 4       — show_frame (1 = display, 0 = not for display)
//!   byte 0 bits 5..7   |
//!   byte 1 bits 0..7   |  19-bit first-partition-size
//!   byte 2 bits 0..2   |
//! ```
//!
//! For key frames (the only frame type §2.5 allows), bytes 3..6
//! are the start code `0x9D 0x01 0x2A` and bytes 6..10 are two
//! 16-bit fields `(2-bit hscale << 14) | width` and
//! `(2-bit vscale << 14) | height`, both little-endian.
//!
//! ## What this module does NOT do
//!
//! * No VP8 bitstream decode (entropy coder, residuals, prediction,
//!   loop filter). The payload is borrowed verbatim.
//! * No cross-validation between the §2.5 canvas dims and a §2.7.1
//!   `VP8X`-declared canvas. The container layer surfaces both and
//!   leaves the policy decision to the caller.
//! * No `oxideav-vp8` runtime dependency. Routing the payload to a
//!   VP8 decoder is the caller's responsibility.

use crate::container::{fourcc, WebpChunk, WebpContainer};

/// RFC 6386 §9.1 start code byte 0 — the first of the three sync
/// bytes that follow the 3-byte frame tag in a key frame.
pub const VP8_START_CODE_0: u8 = 0x9D;
/// RFC 6386 §9.1 start code byte 1.
pub const VP8_START_CODE_1: u8 = 0x01;
/// RFC 6386 §9.1 start code byte 2.
pub const VP8_START_CODE_2: u8 = 0x2A;

/// Bytes the §9.1 key-frame header occupies in the payload.
pub const VP8_KEYFRAME_HEADER_LEN: usize = 10;

/// Errors raised when peeking a §2.5 `VP8 ` chunk's framing.
///
/// The chunk *walker* (the parent
/// [`crate::container::parse`] call) is responsible for the chunk
/// boundary; these errors only surface when the bytes that the
/// walker handed us don't satisfy RFC 9649 §2.5 / RFC 6386 §9.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebpLossyError {
    /// The chunk's FourCC was not `"VP8 "`. The `extract`
    /// helpers refuse to dress up a non-lossy chunk as a lossy
    /// one.
    NotVp8Chunk {
        /// The FourCC actually carried on the chunk.
        got: [u8; 4],
    },
    /// The payload is shorter than the RFC 6386 §9.1 key-frame
    /// header (10 bytes — 3-byte frame tag + 3-byte start code +
    /// 4 bytes of width/height).
    PayloadTooShortForKeyframe {
        /// Bytes actually present in the chunk payload.
        got: usize,
    },
    /// RFC 6386 §9.1 frame-type bit was 1 (interframe). RFC 9649
    /// §2.5 only allows key frames in a `VP8 ` WebP chunk.
    NotAKeyframe,
    /// The §9.1 sync bytes 3..6 did not match the constant
    /// `0x9D 0x01 0x2A`. The walker reports what it saw so callers
    /// can distinguish "wrong codec" from "corrupted payload".
    BadStartCode {
        /// Bytes actually found at payload offsets 3..6.
        got: [u8; 3],
    },
}

impl core::fmt::Display for WebpLossyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotVp8Chunk { got } => write!(
                f,
                "§2.5 lossy-chunk handle wraps a non-'VP8 ' FourCC (got {:02x?})",
                got
            ),
            Self::PayloadTooShortForKeyframe { got } => write!(
                f,
                "RFC 6386 §9.1 keyframe header needs {} bytes; chunk payload has {got}",
                VP8_KEYFRAME_HEADER_LEN
            ),
            Self::NotAKeyframe => write!(
                f,
                "RFC 9649 §2.5 / RFC 6386 §9.1: frame_type=1 (interframe) \
                 is not permitted inside a 'VP8 ' WebP chunk"
            ),
            Self::BadStartCode { got } => write!(
                f,
                "RFC 6386 §9.1 sync bytes mismatch — expected {:02x?}, got {:02x?}",
                [VP8_START_CODE_0, VP8_START_CODE_1, VP8_START_CODE_2],
                got
            ),
        }
    }
}

impl std::error::Error for WebpLossyError {}

/// Typed §2.5 `VP8 ` chunk handle.
///
/// Construct via [`WebpLossyChunk::from_chunk`] or
/// [`WebpLossyChunk::from_payload`]. The handle borrows the
/// payload bytes out of the input slice — it does not own them.
/// Cloning is cheap (a `(width, height, version, &[u8])` copy).
///
/// The intended use is **routing**: a container-layer caller walks
/// the WebP file with [`crate::container::parse`], pulls the
/// `VP8 ` chunk out of the chunk list, dresses it up as a
/// [`WebpLossyChunk`], and hands the resulting
/// [`WebpLossyChunk::bitstream`] slice to a VP8 decoder in a
/// downstream crate (e.g. `oxideav-vp8`) without
/// `oxideav-webp` itself taking a runtime dependency on that
/// decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebpLossyChunk<'a> {
    /// The full `VP8 ` chunk payload, as the walker recorded it.
    /// Length matches `WebpChunk::size as usize`.
    payload: &'a [u8],
    /// RFC 6386 §9.1 version field (0..3 defined; bits 1..3 of
    /// frame-tag byte 0).
    version: u8,
    /// RFC 6386 §9.1 show_frame flag (bit 4 of frame-tag byte 0).
    show_frame: bool,
    /// RFC 6386 §9.1 19-bit first-partition-size field. Surfaced
    /// raw so callers can sanity-check `partition_length` against
    /// the remaining payload.
    first_partition_size: u32,
    /// RFC 6386 §9.1 14-bit width — the canvas width per §2.5.
    width: u16,
    /// RFC 6386 §9.1 14-bit height — the canvas height per §2.5.
    height: u16,
    /// RFC 6386 §9.1 2-bit horizontal-scale field.
    horizontal_scale: u8,
    /// RFC 6386 §9.1 2-bit vertical-scale field.
    vertical_scale: u8,
}

impl<'a> WebpLossyChunk<'a> {
    /// Peek the §9.1 keyframe header out of an already-walked
    /// `VP8 ` chunk and return a typed handle.
    ///
    /// The `buf` argument is the entire WebP file buffer the
    /// walker was given; the chunk's `payload_start` / `payload_end`
    /// index into it.
    ///
    /// Returns [`WebpLossyError::NotVp8Chunk`] if the chunk's
    /// FourCC is not `"VP8 "`, and the various §9.1 refusal modes
    /// if the payload fails the frame-tag / start-code / keyframe
    /// checks.
    pub fn from_chunk(buf: &'a [u8], chunk: &WebpChunk) -> Result<Self, WebpLossyError> {
        if chunk.fourcc != fourcc::VP8 {
            return Err(WebpLossyError::NotVp8Chunk { got: chunk.fourcc });
        }
        Self::from_payload(chunk.payload(buf))
    }

    /// Peek the §9.1 keyframe header out of a raw `VP8 ` chunk
    /// payload (the bytes that follow the 8-byte §2.3 chunk
    /// header). Use [`from_chunk`](Self::from_chunk) when you also
    /// want a FourCC sanity check.
    pub fn from_payload(payload: &'a [u8]) -> Result<Self, WebpLossyError> {
        if payload.len() < VP8_KEYFRAME_HEADER_LEN {
            return Err(WebpLossyError::PayloadTooShortForKeyframe { got: payload.len() });
        }
        // RFC 6386 §9.1 frame tag — bytes 0..3, little-endian:
        //   bit 0      : frame type (0=key)
        //   bits 1..3  : version
        //   bit 4      : show_frame
        //   bits 5..23 : first partition size (19 bits)
        let tag = (payload[0] as u32) | ((payload[1] as u32) << 8) | ((payload[2] as u32) << 16);
        let frame_type = (tag & 0x1) as u8;
        if frame_type != 0 {
            return Err(WebpLossyError::NotAKeyframe);
        }
        let version = ((tag >> 1) & 0x7) as u8;
        let show_frame = ((tag >> 4) & 0x1) == 1;
        let first_partition_size = (tag >> 5) & 0x7_FFFF;

        // §9.1 start code — bytes 3..6.
        let sc = [payload[3], payload[4], payload[5]];
        if sc != [VP8_START_CODE_0, VP8_START_CODE_1, VP8_START_CODE_2] {
            return Err(WebpLossyError::BadStartCode { got: sc });
        }

        // §9.1 width / height fields — each 16 bits little-endian,
        // with the high 2 bits being the scale and the low 14 the
        // dimension.
        let w_raw = (payload[6] as u16) | ((payload[7] as u16) << 8);
        let h_raw = (payload[8] as u16) | ((payload[9] as u16) << 8);
        let width = w_raw & 0x3FFF;
        let height = h_raw & 0x3FFF;
        let horizontal_scale = ((w_raw >> 14) & 0x3) as u8;
        let vertical_scale = ((h_raw >> 14) & 0x3) as u8;

        Ok(Self {
            payload,
            version,
            show_frame,
            first_partition_size,
            width,
            height,
            horizontal_scale,
            vertical_scale,
        })
    }

    /// The full `VP8 ` chunk payload — the bitstream bytes that
    /// downstream consumers route to a VP8 decoder.
    ///
    /// This is the byte-for-byte payload the walker recorded; the
    /// §9.1 keyframe header is included at offset 0. A downstream
    /// VP8 decoder consumes the whole slice.
    pub fn bitstream(&self) -> &'a [u8] {
        self.payload
    }

    /// RFC 6386 §9.1 version field (0..3 defined).
    pub fn version(&self) -> u8 {
        self.version
    }

    /// RFC 6386 §9.1 show_frame flag. WebP's `VP8 ` chunk is
    /// effectively always for display, but the field is surfaced
    /// raw rather than refused so callers can choose policy.
    pub fn show_frame(&self) -> bool {
        self.show_frame
    }

    /// RFC 6386 §9.1 19-bit first-partition-size field. The §2.5
    /// container does not constrain this; it's surfaced for
    /// observability and sanity checks.
    pub fn first_partition_size(&self) -> u32 {
        self.first_partition_size
    }

    /// RFC 9649 §2.5 canvas width — the §9.1 14-bit width field
    /// from the VP8 frame header.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// RFC 9649 §2.5 canvas height — the §9.1 14-bit height field
    /// from the VP8 frame header.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// RFC 6386 §9.1 horizontal-scale field (top 2 bits of the
    /// 16-bit width word).
    pub fn horizontal_scale(&self) -> u8 {
        self.horizontal_scale
    }

    /// RFC 6386 §9.1 vertical-scale field (top 2 bits of the 16-bit
    /// height word).
    pub fn vertical_scale(&self) -> u8 {
        self.vertical_scale
    }
}

/// Walk the container and pull out the first `VP8 ` chunk as a
/// typed [`WebpLossyChunk`].
///
/// Returns `Ok(None)` if the container is well-formed but carries
/// no `VP8 ` chunk (e.g. it's a simple-lossless or extended
/// `VP8L`-only file). Returns `Err(WebpLossyError)` if a `VP8 `
/// chunk is present but its §9.1 header is malformed.
pub fn extract_lossy<'a>(
    buf: &'a [u8],
    container: &WebpContainer,
) -> Result<Option<WebpLossyChunk<'a>>, WebpLossyError> {
    match container.first_chunk_with_fourcc(fourcc::VP8) {
        Some(chunk) => WebpLossyChunk::from_chunk(buf, chunk).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::parse;

    /// Build a §9.1 keyframe header given the named fields, plus
    /// `extra` bytes of opaque entropy-coded payload tacked on the
    /// end.
    #[allow(clippy::too_many_arguments)]
    fn keyframe(
        version: u8,
        show_frame: bool,
        first_partition_size: u32,
        h_scale: u8,
        width: u16,
        v_scale: u8,
        height: u16,
        extra: &[u8],
    ) -> Vec<u8> {
        let frame_type: u32 = 0; // key frame
        let mut tag: u32 = 0;
        tag |= frame_type & 0x1;
        tag |= ((version as u32) & 0x7) << 1;
        tag |= (if show_frame { 1 } else { 0 } & 0x1) << 4;
        tag |= (first_partition_size & 0x7_FFFF) << 5;

        let w_word = ((h_scale as u16 & 0x3) << 14) | (width & 0x3FFF);
        let h_word = ((v_scale as u16 & 0x3) << 14) | (height & 0x3FFF);

        let mut out = Vec::with_capacity(VP8_KEYFRAME_HEADER_LEN + extra.len());
        out.push((tag & 0xFF) as u8);
        out.push(((tag >> 8) & 0xFF) as u8);
        out.push(((tag >> 16) & 0xFF) as u8);
        out.push(VP8_START_CODE_0);
        out.push(VP8_START_CODE_1);
        out.push(VP8_START_CODE_2);
        out.push((w_word & 0xFF) as u8);
        out.push(((w_word >> 8) & 0xFF) as u8);
        out.push((h_word & 0xFF) as u8);
        out.push(((h_word >> 8) & 0xFF) as u8);
        out.extend_from_slice(extra);
        out
    }

    #[test]
    fn from_payload_decodes_minimal_1x1_keyframe_header() {
        // 1x1 keyframe, version 0, show_frame=1, tiny first
        // partition. No entropy bytes — the handle doesn't care
        // about the post-header payload.
        let payload = keyframe(0, true, 11, 0, 1, 0, 1, &[]);
        let h = WebpLossyChunk::from_payload(&payload).expect("1x1 keyframe parses");
        assert_eq!(h.width(), 1);
        assert_eq!(h.height(), 1);
        assert_eq!(h.version(), 0);
        assert!(h.show_frame());
        assert_eq!(h.first_partition_size(), 11);
        assert_eq!(h.horizontal_scale(), 0);
        assert_eq!(h.vertical_scale(), 0);
        assert_eq!(h.bitstream(), payload.as_slice());
    }

    #[test]
    fn from_payload_extracts_14bit_width_and_height_at_extremes() {
        // 0x3FFF = 16383 is the largest 14-bit width / height.
        let payload = keyframe(2, true, 0x7_FFFE, 3, 0x3FFF, 1, 0x2222, &[]);
        let h = WebpLossyChunk::from_payload(&payload).unwrap();
        assert_eq!(h.width(), 0x3FFF);
        assert_eq!(h.height(), 0x2222);
        assert_eq!(h.horizontal_scale(), 3);
        assert_eq!(h.vertical_scale(), 1);
        assert_eq!(h.version(), 2);
        assert_eq!(h.first_partition_size(), 0x7_FFFE);
    }

    #[test]
    fn from_payload_refuses_short_payload() {
        // 9 bytes — one short of the §9.1 keyframe header length.
        let payload = vec![0u8; 9];
        match WebpLossyChunk::from_payload(&payload) {
            Err(WebpLossyError::PayloadTooShortForKeyframe { got }) => assert_eq!(got, 9),
            other => panic!("expected PayloadTooShortForKeyframe, got {other:?}"),
        }
    }

    #[test]
    fn from_payload_refuses_interframes_per_section_2_5() {
        // Frame tag with frame_type=1 (interframe). RFC 9649 §2.5
        // forbids this in a WebP `VP8 ` chunk.
        let mut payload = keyframe(0, true, 0, 0, 1, 0, 1, &[]);
        payload[0] |= 0x01; // set frame_type bit
        assert_eq!(
            WebpLossyChunk::from_payload(&payload).unwrap_err(),
            WebpLossyError::NotAKeyframe,
        );
    }

    #[test]
    fn from_payload_refuses_bad_start_code() {
        // Right length, frame_type=0, but mangle the sync bytes.
        let mut payload = keyframe(0, true, 0, 0, 1, 0, 1, &[]);
        payload[3] = 0xDE;
        payload[4] = 0xAD;
        payload[5] = 0xBE;
        match WebpLossyChunk::from_payload(&payload) {
            Err(WebpLossyError::BadStartCode { got }) => assert_eq!(got, [0xDE, 0xAD, 0xBE]),
            other => panic!("expected BadStartCode, got {other:?}"),
        }
    }

    #[test]
    fn from_chunk_refuses_non_vp8_fourcc() {
        // Wrap a (valid-looking) keyframe payload in a 'VP8L'
        // chunk and walk that. The lossy handle should refuse.
        let payload = keyframe(0, true, 0, 0, 1, 0, 1, &[]);
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8L");
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(&payload);
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);
        let c = parse(&buf).expect("valid RIFF parses");
        let chunk = &c.chunks[0];
        match WebpLossyChunk::from_chunk(&buf, chunk) {
            Err(WebpLossyError::NotVp8Chunk { got }) => assert_eq!(&got, b"VP8L"),
            other => panic!("expected NotVp8Chunk, got {other:?}"),
        }
    }

    #[test]
    fn from_chunk_round_trips_payload_bytes_through_walker() {
        // Build a real RIFF/WEBP file containing a 'VP8 ' chunk
        // whose payload is a synthetic keyframe header + 4 random
        // bytes of entropy-coded payload. The walker → handle path
        // must surface the same bytes.
        let header = keyframe(1, true, 7, 0, 12, 0, 8, &[]);
        let mut payload = header.clone();
        payload.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8 ");
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(&payload);
        // Pad if odd.
        if payload.len() % 2 == 1 {
            body.push(0);
        }
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);

        let c = parse(&buf).expect("synthetic lossy file walks");
        let handle = WebpLossyChunk::from_chunk(&buf, &c.chunks[0]).expect("keyframe handle peeks");
        assert_eq!(handle.width(), 12);
        assert_eq!(handle.height(), 8);
        assert_eq!(handle.version(), 1);
        assert_eq!(handle.first_partition_size(), 7);
        // bitstream() must include the entropy payload verbatim.
        assert_eq!(handle.bitstream(), payload.as_slice());
        assert_eq!(
            &handle.bitstream()[VP8_KEYFRAME_HEADER_LEN..],
            &[0x11, 0x22, 0x33, 0x44]
        );
    }

    #[test]
    fn extract_lossy_returns_none_for_lossless_container() {
        // §2.6 simple-lossless file — has a 'VP8L', no 'VP8 '.
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8L");
        body.extend_from_slice(&4u32.to_le_bytes());
        body.extend_from_slice(&[0x2F, 0x00, 0x00, 0x00]);
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);

        let c = parse(&buf).unwrap();
        assert!(extract_lossy(&buf, &c).unwrap().is_none());
    }

    #[test]
    fn extract_lossy_returns_some_for_simple_lossy_container() {
        // §2.5 simple-lossy file — handle.width / height must
        // match the keyframe we wrote.
        let payload = keyframe(0, true, 13, 0, 1, 0, 1, &[]);
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8 ");
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(&payload);
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);

        let c = parse(&buf).unwrap();
        let handle = extract_lossy(&buf, &c).unwrap().expect("VP8 chunk present");
        assert_eq!(handle.width(), 1);
        assert_eq!(handle.height(), 1);
    }
}
