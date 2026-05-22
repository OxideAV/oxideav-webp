//! Typed §2.6 `VP8L` chunk handle.
//!
//! RFC 9649 §2.6 ("Simple File Format (Lossless)") wraps a single
//! `VP8L` chunk whose payload begins with a 5-byte WebP-Lossless
//! *image header* (the §3.4 / §7.1 "Basic Structure" `image-header`)
//! followed by the entropy-coded transform / spatially-coded image
//! stream. RFC 9649 §2.6 also notes:
//!
//! > Note that the VP8L header contains the VP8L image width and
//! > height.  That is assumed to be the width and height of the
//! > canvas.
//!
//! This module gives container-layer callers a way to route the
//! `VP8L` payload to a downstream VP8L decoder **without** the
//! `oxideav-webp` crate taking a runtime dependency on
//! `oxideav-vp8l` (or any other lossless-WebP decoder). A
//! [`WebpLosslessChunk`] is a thin typed handle that:
//!
//! * borrows the chunk payload byte-for-byte (no copy);
//! * decodes the first 5 bytes of the VP8L bitstream — the
//!   one-byte 0x2F signature plus the 28 bits that pack
//!   `(width - 1)`, `(height - 1)`, `alpha_is_used`, `version`;
//! * refuses payloads that don't start with the §3.4 / §7.1
//!   `0x2F` signature byte, or that are shorter than the 5 bytes
//!   needed for the image-header bit packing;
//! * surfaces the resolved 1-based `width()` / `height()` plus the
//!   raw `alpha_is_used()` and `version()` flags.
//!
//! The actual VP8L decode — transforms (predictor / color / subtract-
//! green / color-indexing), Huffman / prefix-coded image stream,
//! LZ77 backref, color cache — is **not** in scope here. That work
//! lives in a downstream lossless-WebP decoder. The contract is
//! one-way: this module emits a typed payload-bytes handle, and the
//! caller routes those bytes to whatever VP8L decoder it wishes.
//!
//! ## §3.4 / §7.1 image-header bit layout
//!
//! Per the WebP Lossless Bitstream Specification §7.1 (Basic
//! Structure), the `image-header` is a least-significant-bit-first
//! packed sequence on top of the VP8L byte stream:
//!
//! ```text
//!   byte 0        — 0x2F signature (§3.4 calls it `%x2F`).
//!   byte 1 bits 0..7 |
//!   byte 2 bits 0..5 |  14-bit `width - 1`     (LSB first).
//!   byte 2 bits 6..7 |
//!   byte 3 bits 0..7 |
//!   byte 4 bit 0     |  14-bit `height - 1`    (LSB first).
//!   byte 4 bit 1     —  `alpha_is_used`        (1 bit).
//!   byte 4 bits 2..4 —  `version`              (3 bits, MUST be 0).
//! ```
//!
//! `width` and `height` are reported in the "as you would index a
//! pixel" form — `(field + 1)`, range `1..=16384`. `alpha_is_used`
//! and `version` are surfaced as their on-wire 1-bit / 3-bit
//! values; the §3.4 statement that `version` SHOULD be `0` is
//! observable (so callers can act on it) but **not** enforced as a
//! refusal mode — §3.4 says non-zero versions "should be treated as
//! an error" by a *decoder*, which is the layer below this typed
//! handle. The handle is a routing surface; the downstream decoder
//! is responsible for the version-mismatch policy.
//!
//! ## What this module does NOT do
//!
//! * No VP8L bitstream decode (transforms, prefix-coded image stream,
//!   LZ77, color cache). The payload is borrowed verbatim.
//! * No cross-validation between the §2.6 / §3.4 canvas dims and a
//!   §2.7.1 `VP8X`-declared canvas. The container layer surfaces
//!   both and leaves the policy decision to the caller.
//! * No `oxideav-vp8l` runtime dependency. Routing the payload to a
//!   VP8L decoder is the caller's responsibility.

use crate::container::{fourcc, WebpChunk, WebpContainer};

/// WebP Lossless §3.4 / §7.1 signature byte — the first byte of
/// every `VP8L` chunk's payload.
pub const VP8L_SIGNATURE: u8 = 0x2F;

/// Number of bytes the §3.4 / §7.1 image-header occupies at the
/// start of a `VP8L` payload: 1 signature byte + 28 packed bits
/// (rounded up to 4 bytes).
pub const VP8L_IMAGE_HEADER_LEN: usize = 5;

/// Errors raised when peeking a §2.6 `VP8L` chunk's framing.
///
/// The chunk *walker* (the parent
/// [`crate::container::parse`] call) is responsible for the chunk
/// boundary; these errors only surface when the bytes that the
/// walker handed us don't satisfy RFC 9649 §2.6 / §3.4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebpLosslessError {
    /// The chunk's FourCC was not `"VP8L"`. The `extract`
    /// helpers refuse to dress up a non-lossless chunk as a
    /// lossless one.
    NotVp8lChunk {
        /// The FourCC actually carried on the chunk.
        got: [u8; 4],
    },
    /// The payload is shorter than the §7.1 5-byte image-header
    /// (1 signature byte + 4 bytes packing the 14+14+1+3 = 32 bits
    /// of width / height / alpha / version).
    PayloadTooShortForHeader {
        /// Bytes actually present in the chunk payload.
        got: usize,
    },
    /// Payload byte 0 was not the §3.4 / §7.1 `0x2F` signature
    /// byte.
    BadSignature {
        /// Byte actually found at offset 0 of the chunk payload.
        got: u8,
    },
}

impl core::fmt::Display for WebpLosslessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotVp8lChunk { got } => write!(
                f,
                "§2.6 lossless-chunk handle wraps a non-'VP8L' FourCC (got {:02x?})",
                got
            ),
            Self::PayloadTooShortForHeader { got } => write!(
                f,
                "§3.4 / §7.1 VP8L image-header needs {} bytes; chunk payload has {got}",
                VP8L_IMAGE_HEADER_LEN
            ),
            Self::BadSignature { got } => write!(
                f,
                "§3.4 / §7.1 VP8L signature mismatch — expected 0x{:02x}, got 0x{:02x}",
                VP8L_SIGNATURE, got
            ),
        }
    }
}

impl std::error::Error for WebpLosslessError {}

/// Typed §2.6 `VP8L` chunk handle.
///
/// Construct via [`WebpLosslessChunk::from_chunk`] or
/// [`WebpLosslessChunk::from_payload`]. The handle borrows the
/// payload bytes out of the input slice — it does not own them.
/// Cloning is cheap (a `(width, height, alpha_is_used, version,
/// &[u8])` copy).
///
/// The intended use is **routing**: a container-layer caller walks
/// the WebP file with [`crate::container::parse`], pulls the
/// `VP8L` chunk out of the chunk list, dresses it up as a
/// [`WebpLosslessChunk`], and hands the resulting
/// [`WebpLosslessChunk::bitstream`] slice to a VP8L decoder in a
/// downstream crate without `oxideav-webp` itself taking a runtime
/// dependency on that decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebpLosslessChunk<'a> {
    /// The full `VP8L` chunk payload, as the walker recorded it.
    /// Length matches `WebpChunk::size as usize`.
    payload: &'a [u8],
    /// Resolved §3.4 image width — `(width_minus_one + 1)`,
    /// always in `1..=16384`.
    width: u32,
    /// Resolved §3.4 image height — `(height_minus_one + 1)`,
    /// always in `1..=16384`.
    height: u32,
    /// §3.4 `alpha_is_used` bit. Per §3.4 this is a hint only;
    /// surfaced raw.
    alpha_is_used: bool,
    /// §3.4 `version_number` 3-bit field. §3.4 says it MUST be
    /// `0`; surfaced raw rather than refused so the routing
    /// handle stays observability-friendly.
    version: u8,
}

impl<'a> WebpLosslessChunk<'a> {
    /// Peek the §3.4 / §7.1 image-header out of an already-walked
    /// `VP8L` chunk and return a typed handle.
    ///
    /// The `buf` argument is the entire WebP file buffer the
    /// walker was given; the chunk's `payload_start` / `payload_end`
    /// index into it.
    ///
    /// Returns [`WebpLosslessError::NotVp8lChunk`] if the chunk's
    /// FourCC is not `"VP8L"`, and the various §3.4 refusal modes
    /// if the payload fails the signature / length checks.
    pub fn from_chunk(buf: &'a [u8], chunk: &WebpChunk) -> Result<Self, WebpLosslessError> {
        if chunk.fourcc != fourcc::VP8L {
            return Err(WebpLosslessError::NotVp8lChunk { got: chunk.fourcc });
        }
        Self::from_payload(chunk.payload(buf))
    }

    /// Peek the §3.4 / §7.1 image-header out of a raw `VP8L` chunk
    /// payload (the bytes that follow the 8-byte §2.3 chunk
    /// header). Use [`from_chunk`](Self::from_chunk) when you also
    /// want a FourCC sanity check.
    pub fn from_payload(payload: &'a [u8]) -> Result<Self, WebpLosslessError> {
        if payload.len() < VP8L_IMAGE_HEADER_LEN {
            return Err(WebpLosslessError::PayloadTooShortForHeader { got: payload.len() });
        }
        // §3.4 / §7.1: byte 0 is the `0x2F` signature.
        if payload[0] != VP8L_SIGNATURE {
            return Err(WebpLosslessError::BadSignature { got: payload[0] });
        }

        // §7.1 packs the next 28 bits (then 4 bits of version-or-
        // -reserved-padding for the alignment) LSB-first across
        // bytes 1..5:
        //
        //   bits 0..13   width - 1
        //   bits 14..27  height - 1
        //   bit  28      alpha_is_used
        //   bits 29..31  version (3 bits, MUST be 0 per §3.4)
        //
        // Concatenate bytes 1..5 LE into a u32, then unpack.
        let packed = (payload[1] as u32)
            | ((payload[2] as u32) << 8)
            | ((payload[3] as u32) << 16)
            | ((payload[4] as u32) << 24);
        let width_minus_one = packed & 0x3FFF; // 14 bits
        let height_minus_one = (packed >> 14) & 0x3FFF; // 14 bits
        let alpha_is_used = ((packed >> 28) & 0x1) == 1;
        let version = ((packed >> 29) & 0x7) as u8;

        Ok(Self {
            payload,
            width: width_minus_one + 1,
            height: height_minus_one + 1,
            alpha_is_used,
            version,
        })
    }

    /// The full `VP8L` chunk payload — the bitstream bytes that
    /// downstream consumers route to a VP8L decoder.
    ///
    /// This is the byte-for-byte payload the walker recorded; the
    /// §3.4 / §7.1 5-byte image-header is included at offset 0. A
    /// downstream VP8L decoder consumes the whole slice (the
    /// §7.1 `image-stream` follows immediately after the
    /// `image-header`).
    pub fn bitstream(&self) -> &'a [u8] {
        self.payload
    }

    /// Resolved §3.4 image width — `(width_minus_one + 1)`. Always
    /// in `1..=16384` (`0x4000`, the maximum after the 14-bit
    /// `width - 1` field).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Resolved §3.4 image height — `(height_minus_one + 1)`. Always
    /// in `1..=16384`.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// §3.4 `alpha_is_used` bit. Per the spec, a hint only — a
    /// strict reader does not refuse a file because the hint was
    /// off; the routing handle surfaces it for downstream policy.
    pub fn alpha_is_used(&self) -> bool {
        self.alpha_is_used
    }

    /// §3.4 `version_number` 3-bit field. §3.4 states the value
    /// MUST be `0`; non-zero is "should be treated as an error" by
    /// a *decoder*, which is the next layer down. This handle
    /// surfaces it raw.
    pub fn version(&self) -> u8 {
        self.version
    }
}

/// Walk the container and pull out the first `VP8L` chunk as a
/// typed [`WebpLosslessChunk`].
///
/// Returns `Ok(None)` if the container is well-formed but carries
/// no `VP8L` chunk (e.g. it's a simple-lossy or extended `VP8 `-only
/// file). Returns `Err(WebpLosslessError)` if a `VP8L` chunk is
/// present but its §3.4 / §7.1 image-header is malformed.
pub fn extract_lossless<'a>(
    buf: &'a [u8],
    container: &WebpContainer,
) -> Result<Option<WebpLosslessChunk<'a>>, WebpLosslessError> {
    match container.first_chunk_with_fourcc(fourcc::VP8L) {
        Some(chunk) => WebpLosslessChunk::from_chunk(buf, chunk).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::parse;

    /// Build a §3.4 / §7.1 VP8L image-header given the named
    /// fields, plus `extra` bytes of opaque entropy-coded payload
    /// tacked on the end.
    fn vp8l_header(
        width: u32,
        height: u32,
        alpha_is_used: bool,
        version: u8,
        extra: &[u8],
    ) -> Vec<u8> {
        assert!((1..=0x4000).contains(&width));
        assert!((1..=0x4000).contains(&height));
        assert!(version <= 0x7);
        let width_minus_one = width - 1;
        let height_minus_one = height - 1;
        let mut packed: u32 = 0;
        packed |= width_minus_one & 0x3FFF;
        packed |= (height_minus_one & 0x3FFF) << 14;
        packed |= (if alpha_is_used { 1 } else { 0 }) << 28;
        packed |= ((version as u32) & 0x7) << 29;

        let mut out = Vec::with_capacity(VP8L_IMAGE_HEADER_LEN + extra.len());
        out.push(VP8L_SIGNATURE);
        out.push((packed & 0xFF) as u8);
        out.push(((packed >> 8) & 0xFF) as u8);
        out.push(((packed >> 16) & 0xFF) as u8);
        out.push(((packed >> 24) & 0xFF) as u8);
        out.extend_from_slice(extra);
        out
    }

    #[test]
    fn from_payload_decodes_minimal_1x1_header() {
        // 1x1 lossless image, no alpha, version 0.
        let payload = vp8l_header(1, 1, false, 0, &[]);
        let h = WebpLosslessChunk::from_payload(&payload).expect("1x1 VP8L parses");
        assert_eq!(h.width(), 1);
        assert_eq!(h.height(), 1);
        assert!(!h.alpha_is_used());
        assert_eq!(h.version(), 0);
        assert_eq!(h.bitstream(), payload.as_slice());
    }

    #[test]
    fn from_payload_decodes_max_dims_with_alpha_and_version_zero() {
        // 16384 = 0x4000 — the largest 14-bit (width - 1) + 1.
        let payload = vp8l_header(0x4000, 0x4000, true, 0, &[]);
        let h = WebpLosslessChunk::from_payload(&payload).unwrap();
        assert_eq!(h.width(), 0x4000);
        assert_eq!(h.height(), 0x4000);
        assert!(h.alpha_is_used());
        assert_eq!(h.version(), 0);
    }

    #[test]
    fn from_payload_surfaces_nonstandard_version_without_refusing() {
        // §3.4 says version MUST be 0, but the routing handle is
        // *observability* — non-zero is surfaced for the downstream
        // decoder to act on.
        let payload = vp8l_header(2, 3, false, 5, &[]);
        let h = WebpLosslessChunk::from_payload(&payload).unwrap();
        assert_eq!(h.width(), 2);
        assert_eq!(h.height(), 3);
        assert_eq!(h.version(), 5);
    }

    #[test]
    fn from_payload_refuses_short_payload() {
        // 4 bytes — one short of the §7.1 image-header length.
        let payload = [VP8L_SIGNATURE, 0, 0, 0];
        match WebpLosslessChunk::from_payload(&payload) {
            Err(WebpLosslessError::PayloadTooShortForHeader { got }) => assert_eq!(got, 4),
            other => panic!("expected PayloadTooShortForHeader, got {other:?}"),
        }
    }

    #[test]
    fn from_payload_refuses_bad_signature() {
        // Right length, wrong signature byte. §3.4 demands 0x2F.
        let mut payload = vp8l_header(1, 1, false, 0, &[]);
        payload[0] = 0x55;
        match WebpLosslessChunk::from_payload(&payload) {
            Err(WebpLosslessError::BadSignature { got }) => assert_eq!(got, 0x55),
            other => panic!("expected BadSignature, got {other:?}"),
        }
    }

    #[test]
    fn from_payload_borrows_trailing_image_stream_verbatim() {
        // The handle's bitstream() slice must include the
        // entropy-coded payload that follows the 5-byte header.
        let extra = [0xAA, 0xBB, 0xCC];
        let payload = vp8l_header(7, 5, true, 0, &extra);
        let h = WebpLosslessChunk::from_payload(&payload).unwrap();
        assert_eq!(h.width(), 7);
        assert_eq!(h.height(), 5);
        assert!(h.alpha_is_used());
        assert_eq!(h.bitstream(), payload.as_slice());
        assert_eq!(&h.bitstream()[VP8L_IMAGE_HEADER_LEN..], &extra);
    }

    #[test]
    fn from_chunk_refuses_non_vp8l_fourcc() {
        // Wrap a (valid-looking) VP8L payload in a 'VP8 ' chunk
        // and walk that. The lossless handle should refuse.
        let payload = vp8l_header(1, 1, false, 0, &[]);
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8 ");
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(&payload);
        if payload.len() % 2 == 1 {
            body.push(0);
        }
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);
        let c = parse(&buf).expect("valid RIFF parses");
        let chunk = &c.chunks[0];
        match WebpLosslessChunk::from_chunk(&buf, chunk) {
            Err(WebpLosslessError::NotVp8lChunk { got }) => assert_eq!(&got, b"VP8 "),
            other => panic!("expected NotVp8lChunk, got {other:?}"),
        }
    }

    #[test]
    fn from_chunk_round_trips_payload_bytes_through_walker() {
        // Build a real RIFF/WEBP file containing a 'VP8L' chunk
        // whose payload is a synthetic image header + 4 random
        // bytes of entropy-coded payload. The walker → handle path
        // must surface the same bytes.
        let mut payload = vp8l_header(12, 8, true, 0, &[]);
        payload.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8L");
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(&payload);
        if payload.len() % 2 == 1 {
            body.push(0);
        }
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);

        let c = parse(&buf).expect("synthetic lossless file walks");
        let handle = WebpLosslessChunk::from_chunk(&buf, &c.chunks[0]).expect("VP8L handle peeks");
        assert_eq!(handle.width(), 12);
        assert_eq!(handle.height(), 8);
        assert!(handle.alpha_is_used());
        assert_eq!(handle.version(), 0);
        // bitstream() must include the trailing entropy bytes verbatim.
        assert_eq!(handle.bitstream(), payload.as_slice());
        assert_eq!(
            &handle.bitstream()[VP8L_IMAGE_HEADER_LEN..],
            &[0x11, 0x22, 0x33, 0x44]
        );
    }

    #[test]
    fn extract_lossless_returns_none_for_lossy_container() {
        // §2.5 simple-lossy file — has a 'VP8 ', no 'VP8L'.
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8 ");
        // 10-byte synthetic VP8 keyframe header (zeroed; the walker
        // doesn't decode it).
        body.extend_from_slice(&10u32.to_le_bytes());
        body.extend_from_slice(&[0u8; 10]);
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);

        let c = parse(&buf).unwrap();
        assert!(extract_lossless(&buf, &c).unwrap().is_none());
    }

    #[test]
    fn extract_lossless_returns_some_for_simple_lossless_container() {
        // §2.6 simple-lossless file — handle.width / height must
        // match what we wrote into the header.
        let payload = vp8l_header(1, 1, false, 0, &[]);
        let mut body = Vec::new();
        body.extend_from_slice(b"VP8L");
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(&payload);
        if payload.len() % 2 == 1 {
            body.push(0);
        }
        let file_size = (body.len() as u32) + 4;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WEBP");
        buf.extend_from_slice(&body);

        let c = parse(&buf).unwrap();
        let handle = extract_lossless(&buf, &c)
            .unwrap()
            .expect("VP8L chunk present");
        assert_eq!(handle.width(), 1);
        assert_eq!(handle.height(), 1);
    }
}
