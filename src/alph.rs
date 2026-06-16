//! Typed parser for the `ALPH` chunk **info byte** per RFC 9649
//! §2.7.1.2 (Figure 10).
//!
//! The §2.3 walker in [`crate::container`] surfaces an `ALPH` chunk as
//! an opaque payload whose first byte packs four 2-bit fields:
//!
//! ```text
//!  0 1 2 3 4 5 6 7
//! +-+-+-+-+-+-+-+-+
//! |Rsv| P | F | C |
//! +-+-+-+-+-+-+-+-+
//! ```
//!
//! * `Rsv` — Reserved, 2 bits. MUST be 0; readers MUST ignore.
//! * `P`   — Preprocessing, 2 bits. 0 = none, 1 = level reduction.
//!   Other values are informational (decoders are not required to act
//!   on this hint).
//! * `F`   — Filtering method, 2 bits. 0 = none, 1 = horizontal,
//!   2 = vertical, 3 = gradient.
//! * `C`   — Compression method, 2 bits. 0 = uncompressed raw,
//!   1 = WebP lossless format. Other values are not defined by RFC
//!   9649 §2.7.1.2.
//!
//! This module decodes the info byte into a typed [`AlphHeader`]; it
//! also decodes the full Alpha Bitstream that follows
//! ([`decode_alpha`]) into a width × height plane of 8-bit alpha
//! values, covering both compression methods and all four §2.7.1.2
//! filtering methods.
//!
//! ## Alpha bitstream decode ([`decode_alpha`])
//!
//! Per RFC 9649 §2.7.1.2, the alpha bitstream is either:
//!
//! * **Compression method 0** — raw, uncompressed 8-bit alpha values
//!   in scan order, of length `width * height`.
//! * **Compression method 1** — a §3 WebP-lossless *image-stream* of
//!   implicit dimensions `width x height` (no 5-byte image header).
//!   Once decoded into ARGB, "the transparency information must be
//!   extracted from the green channel of the ARGB quadruplet."
//!
//! After de-compression, the §2.7.1.2 inverse filter
//! (none / horizontal / vertical / gradient) is applied over the
//! reconstructed plane: each output alpha is
//! `(predictor + decompressed) % 256`, with the per-method predictor
//! and the documented left-most / top-most edge cases.
//!
//! ## Bit layout anchor
//!
//! The RFC's ASCII-art `|Rsv|P|F|C|` reads MSB-first within the byte,
//! giving:
//!
//! | bit (LSB=0) | field |
//! |-------------|-------|
//! | 7..6        | Rsv   |
//! | 5..4        | P     |
//! | 3..2        | F     |
//! | 1..0        | C     |
//!
//! Cross-checked against `docs/image/webp/fixtures/lossy-with-alpha-128x128/trace.txt`
//! which reports `header_byte=0x01 method=1 filter=0 pre_processing=0` for a
//! reference-encoder-produced fixture — only the C nibble's LSB is set, matching
//! `compression = 1` (lossless) with everything else 0.

use core::fmt;

/// Compression method (`C`) per RFC 9649 §2.7.1.2.
///
/// The spec enumerates `0` (no compression) and `1` (WebP lossless
/// format). Higher values are not defined; we preserve them in
/// [`Self::Reserved`] so callers can refuse on encounter without the
/// parser itself imposing that policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphCompression {
    /// 0: No compression — the alpha bitstream is raw 8-bit values in
    /// scan order, of length `width * height`.
    None,
    /// 1: Lossless — the alpha bitstream is a §3 VP8L image-stream
    /// with implicit dimensions `width x height` (no header).
    Lossless,
    /// 2 or 3 — undefined by §2.7.1.2.
    Reserved(u8),
}

impl AlphCompression {
    fn from_bits(c: u8) -> Self {
        match c & 0b11 {
            0 => Self::None,
            1 => Self::Lossless,
            other => Self::Reserved(other),
        }
    }
}

/// Filtering method (`F`) per RFC 9649 §2.7.1.2.
///
/// The four values are exhaustive within the 2-bit field; the spec
/// defines a prediction rule for each (None / A / B / clip(A+B-C)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphFiltering {
    /// 0: predictor = 0 for every pixel (no filter).
    None,
    /// 1: predictor = A (the pixel to the left).
    Horizontal,
    /// 2: predictor = B (the pixel above).
    Vertical,
    /// 3: predictor = clip(A + B - C) — the gradient predictor.
    Gradient,
}

impl AlphFiltering {
    fn from_bits(f: u8) -> Self {
        match f & 0b11 {
            0 => Self::None,
            1 => Self::Horizontal,
            2 => Self::Vertical,
            3 => Self::Gradient,
            _ => unreachable!("masked to 2 bits"),
        }
    }
}

/// Preprocessing hint (`P`) per RFC 9649 §2.7.1.2.
///
/// Only `0` and `1` are named in the spec; the other two 2-bit values
/// are reserved. §2.7.1.2: "Decoders are not required to use this
/// information in any specified way." — i.e. this is purely
/// informational metadata, not a refusal trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphPreprocessing {
    /// 0: No preprocessing was applied.
    None,
    /// 1: Level reduction was applied prior to compression.
    LevelReduction,
    /// 2 or 3 — undefined by §2.7.1.2.
    Reserved(u8),
}

impl AlphPreprocessing {
    fn from_bits(p: u8) -> Self {
        match p & 0b11 {
            0 => Self::None,
            1 => Self::LevelReduction,
            other => Self::Reserved(other),
        }
    }
}

/// Errors raised by the §2.7.1.2 ALPH info-byte parser and the
/// [`decode_alpha`] bitstream decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlphError {
    /// The ALPH payload is empty — at minimum one info byte is
    /// required per §2.7.1.2 Figure 10, even if the alpha bitstream
    /// itself is zero-length (which §2.7.1.2 does not forbid).
    EmptyPayload,
    /// `width * height` overflowed `usize` (or `u32`), so the plane
    /// cannot be addressed on this platform.
    DimensionsOverflow {
        /// The implicit width passed by the caller.
        width: u32,
        /// The implicit height passed by the caller.
        height: u32,
    },
    /// Compression method 0 (raw) but the alpha bitstream length does
    /// not equal `width * height` (§2.7.1.2: "a byte sequence of
    /// length = width * height").
    RawLengthMismatch {
        /// The expected `width * height` byte count.
        expected: usize,
        /// The actual number of bytes available in the bitstream.
        actual: usize,
    },
    /// Compression method `C` was `2` or `3` — undefined by §2.7.1.2.
    UnsupportedCompression(u8),
    /// The compression-method-1 §3 VP8L image-stream failed to decode.
    Vp8l(crate::vp8l_decode::DecodeError),
}

impl fmt::Display for AlphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPayload => {
                f.write_str("ALPH payload missing the §2.7.1.2 info byte (payload length 0)")
            }
            Self::DimensionsOverflow { width, height } => write!(
                f,
                "ALPH alpha-plane dimensions {width}x{height} overflow the addressable range"
            ),
            Self::RawLengthMismatch { expected, actual } => write!(
                f,
                "ALPH raw (method 0) bitstream length {actual} != width*height {expected}"
            ),
            Self::UnsupportedCompression(c) => write!(
                f,
                "ALPH compression method {c} is undefined by §2.7.1.2 (only 0 and 1 exist)"
            ),
            Self::Vp8l(e) => write!(f, "ALPH method-1 VP8L image-stream decode: {e}"),
        }
    }
}

impl std::error::Error for AlphError {}

impl From<crate::vp8l_decode::DecodeError> for AlphError {
    fn from(e: crate::vp8l_decode::DecodeError) -> Self {
        Self::Vp8l(e)
    }
}

/// Decoded §2.7.1.2 `ALPH` info byte plus the offset at which the
/// alpha bitstream begins inside the chunk payload.
///
/// Constructed via [`AlphHeader::parse`]. The actual alpha bitstream
/// (raw or VP8L-compressed) is **not** decoded — this layer's job is
/// to surface the 2-bit `Rsv` / `P` / `F` / `C` decomposition. The
/// payload after byte 0 — `payload[1..]` — is the §2.7.1.2 "Alpha
/// bitstream" of `Chunk Size - 1` bytes; callers that need it should
/// slice the chunk payload at [`Self::bitstream_offset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphHeader {
    /// `C` field — compression method (§2.7.1.2).
    pub compression: AlphCompression,
    /// `F` field — filtering method (§2.7.1.2).
    pub filtering: AlphFiltering,
    /// `P` field — preprocessing hint (§2.7.1.2).
    pub preprocessing: AlphPreprocessing,
    /// `Rsv` field — raw 2-bit value from bits 7..6 of the info byte.
    /// §2.7.1.2 says "MUST be 0. Readers MUST ignore this field." —
    /// we surface the raw value for observability without rejecting.
    pub reserved: u8,
    /// Raw info byte, preserved for round-trip and trace assertions.
    pub info_byte: u8,
}

impl AlphHeader {
    /// Parse the `ALPH` chunk payload's §2.7.1.2 info byte.
    ///
    /// `payload` is the whole §2.3 chunk payload (i.e. the slice
    /// returned by [`crate::container::WebpChunk::payload`] for a
    /// chunk whose FourCC is [`crate::container::fourcc::ALPH`]). Only
    /// the first byte is consumed by this layer; the remainder is the
    /// alpha bitstream callers must hand off to a later VP8L or raw
    /// decode pass.
    pub fn parse(payload: &[u8]) -> Result<Self, AlphError> {
        let info = *payload.first().ok_or(AlphError::EmptyPayload)?;

        // §2.7.1.2 Figure 10: byte 0 packs Rsv|P|F|C, MSB-first.
        let reserved = (info >> 6) & 0b11;
        let p_bits = (info >> 4) & 0b11;
        let f_bits = (info >> 2) & 0b11;
        let c_bits = info & 0b11;

        Ok(Self {
            compression: AlphCompression::from_bits(c_bits),
            filtering: AlphFiltering::from_bits(f_bits),
            preprocessing: AlphPreprocessing::from_bits(p_bits),
            reserved,
            info_byte: info,
        })
    }

    /// Offset (within the ALPH chunk payload) at which the alpha
    /// bitstream begins. Always 1 per §2.7.1.2 — the info byte is
    /// followed immediately by the bitstream.
    pub const fn bitstream_offset(&self) -> usize {
        1
    }
}

/// `clip(v)` per §2.7.1.2: 0 if `v < 0`, 255 if `v > 255`, else `v`.
#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Decode a complete `ALPH` chunk payload to a `width * height` plane of
/// 8-bit alpha values, in scan order.
///
/// `payload` is the **whole** §2.3 ALPH chunk payload (the §2.7.1.2 info
/// byte followed by the alpha bitstream). `width` / `height` are the
/// implicit alpha-plane dimensions — for a still image these are the
/// `VP8X` canvas dimensions (or the `VP8 ` keyframe dimensions); for an
/// animation frame they are the `ANMF` frame dimensions.
///
/// The decode follows §2.7.1.2 in two stages:
///
/// 1. **De-compression** (`C` field): method 0 copies the raw bytes;
///    method 1 decodes the headerless §3 VP8L image-stream and lifts the
///    alpha values out of the **green** channel of each ARGB pixel.
/// 2. **Inverse filtering** (`F` field): the per-pixel predictor
///    (none / A / B / clip(A+B-C)) is added to the de-compressed value
///    modulo 256, with the §2.7.1.2 left-most / top-most edge cases.
///
/// Returns the reconstructed alpha plane (`width * height` bytes). The
/// §2.7.1.2 preprocessing (`P`) hint is informational and is **not**
/// applied here (the spec: "Decoders are not required to use this
/// information in any specified way.").
pub fn decode_alpha(payload: &[u8], width: u32, height: u32) -> Result<Vec<u8>, AlphError> {
    let header = AlphHeader::parse(payload)?;

    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(AlphError::DimensionsOverflow { width, height })?;

    // The alpha bitstream proper is everything after the info byte.
    let bitstream = &payload[header.bitstream_offset()..];

    // Stage 1 — de-compression into the raw (still-filtered) plane.
    let filtered: Vec<u8> = match header.compression {
        AlphCompression::None => {
            if bitstream.len() != count {
                return Err(AlphError::RawLengthMismatch {
                    expected: count,
                    actual: bitstream.len(),
                });
            }
            bitstream.to_vec()
        }
        AlphCompression::Lossless => {
            // §2.7.1.2: a headerless §3 image-stream of implicit
            // dimensions width x height; the alpha values live in the
            // GREEN channel of the decoded ARGB quadruplets.
            let image =
                crate::vp8l_transform::decode_lossless_headerless(bitstream, width, height)?;
            image
                .pixels()
                .iter()
                .map(|argb| (argb >> 8) as u8)
                .collect()
        }
        AlphCompression::Reserved(c) => return Err(AlphError::UnsupportedCompression(c)),
    };

    // A zero-area plane has nothing to filter.
    if count == 0 {
        return Ok(filtered);
    }

    // Stage 2 — inverse filter into the final plane.
    let w = width as usize;
    let h = height as usize;

    Ok(inverse_filter(filtered, w, h, header.filtering))
}

/// §2.7.1.2 Stage-2 inverse filter: reconstruct the alpha plane from the
/// de-compressed residual `filtered` (length `w * h`, scan order) under
/// the given §2.7.1.2 filtering method.
///
/// `alpha = (predictor + residual) % 256`, where the predictor reads the
/// already-reconstructed `out` plane for the A = left / B = above /
/// C = above-left neighbours (RFC 9649 §2.7.1.2 Figure 11). The
/// per-method §2.7.1.2 edge cases — `(0,0)` always predicts 0; the first
/// column and first row each fall back to the single in-bounds neighbour
/// — are evaluated **once outside the interior loop** (one specialised
/// border pass + one branch-free interior loop per method) rather than
/// re-tested on every pixel. This is the same border-rule hoist the
/// lossless §3.5.2 inverse predictor received; it does not change a
/// single emitted byte (the per-pixel arithmetic is bit-for-bit the prior
/// `match (x, y)` / `match filtering` form, just with the constant
/// dispatch lifted out of the hot loop).
fn inverse_filter(filtered: Vec<u8>, w: usize, h: usize, filtering: AlphFiltering) -> Vec<u8> {
    // §2.7.1.2 method 0 (None): predictor = 0 for every pixel, so the
    // reconstruction is the identity `out = filtered`. No border special
    // case is needed — `(0,0)` already predicts 0 under None.
    if filtering == AlphFiltering::None {
        return filtered;
    }

    let mut out = vec![0u8; w * h];

    // (0,0) always predicts 0, for every filter method.
    out[0] = filtered[0];

    match filtering {
        AlphFiltering::None => unreachable!("handled above"),
        AlphFiltering::Horizontal => {
            // First row (x>0, y=0): predictor = A = left = out[x-1].
            for x in 1..w {
                out[x] = ((out[x - 1] as i32 + filtered[x] as i32) & 0xff) as u8;
            }
            for y in 1..h {
                let row = y * w;
                let above = row - w;
                // Left-most (0, y>0): predicted by (0, y-1) = out[above].
                out[row] = ((out[above] as i32 + filtered[row] as i32) & 0xff) as u8;
                // Interior (x>0, y>0): predictor = A = left = out[row+x-1].
                for x in 1..w {
                    let i = row + x;
                    out[i] = ((out[i - 1] as i32 + filtered[i] as i32) & 0xff) as u8;
                }
            }
        }
        AlphFiltering::Vertical => {
            // First row (x>0, y=0): predictor = (x-1, 0) = out[x-1].
            for x in 1..w {
                out[x] = ((out[x - 1] as i32 + filtered[x] as i32) & 0xff) as u8;
            }
            // Interior + left-most (any x, y>0): predictor = B = above.
            for y in 1..h {
                let row = y * w;
                let above = row - w;
                for x in 0..w {
                    let i = row + x;
                    out[i] = ((out[above + x] as i32 + filtered[i] as i32) & 0xff) as u8;
                }
            }
        }
        AlphFiltering::Gradient => {
            // First row (x>0, y=0): predictor = (x-1, 0) = out[x-1].
            for x in 1..w {
                out[x] = ((out[x - 1] as i32 + filtered[x] as i32) & 0xff) as u8;
            }
            for y in 1..h {
                let row = y * w;
                let above = row - w;
                // Left-most (0, y>0): predicted by (0, y-1) = out[above].
                out[row] = ((out[above] as i32 + filtered[row] as i32) & 0xff) as u8;
                // Interior (x>0, y>0): predictor = clip(A + B − C).
                for x in 1..w {
                    let i = row + x;
                    let a = out[i - 1] as i32;
                    let b = out[above + x] as i32;
                    let c = out[above + x - 1] as i32;
                    let pred = clip(a + b - c) as i32;
                    out[i] = ((pred + filtered[i] as i32) & 0xff) as u8;
                }
            }
        }
    }

    out
}

/// §2.7.1.2 **forward** filter: the algebraic inverse of [`inverse_filter`].
///
/// Produces the residual plane the §2.7.1.2 decoder reconstructs from:
/// `residual = (alpha - predictor) mod 256`, read over the *original*
/// `plane` (length `w * h`, scan order) under the given filtering method.
/// The predictor is the same per-method rule the inverse pass uses
/// (None = 0 / Horizontal = A = left / Vertical = B = above /
/// Gradient = clip(A + B − C)), with the identical §2.7.1.2 border cases:
/// `(0,0)` predicts 0; the first column and first row each fall back to
/// the single in-bounds neighbour.
///
/// Because the filter is causal — every predictor reads only neighbours
/// that precede the current pixel in scan order — the predictor the
/// encoder reads from the original plane equals the one the decoder reads
/// from the reconstructed plane on a correct decode. So
/// `inverse_filter(forward_filter(plane, …), …)` is the identity for every
/// method (this is what [`encode_alpha`]'s round-trip rests on).
fn forward_filter(plane: &[u8], w: usize, h: usize, filtering: AlphFiltering) -> Vec<u8> {
    // None: predictor = 0 everywhere → residual is the plane unchanged.
    if filtering == AlphFiltering::None || w == 0 || h == 0 {
        return plane.to_vec();
    }

    let mut residual = vec![0u8; w * h];

    // (0,0) always predicts 0, for every method.
    residual[0] = plane[0];

    match filtering {
        AlphFiltering::None => unreachable!("handled above"),
        AlphFiltering::Horizontal => {
            // First row (x>0, y=0): predictor = A = left.
            for x in 1..w {
                residual[x] = (plane[x] as i32 - plane[x - 1] as i32) as u8;
            }
            for y in 1..h {
                let row = y * w;
                let above = row - w;
                // Left-most (0, y>0): predictor = (0, y-1).
                residual[row] = (plane[row] as i32 - plane[above] as i32) as u8;
                // Interior (x>0, y>0): predictor = A = left.
                for x in 1..w {
                    let i = row + x;
                    residual[i] = (plane[i] as i32 - plane[i - 1] as i32) as u8;
                }
            }
        }
        AlphFiltering::Vertical => {
            // First row (x>0, y=0): predictor = (x-1, 0).
            for x in 1..w {
                residual[x] = (plane[x] as i32 - plane[x - 1] as i32) as u8;
            }
            // Interior + left-most (any x, y>0): predictor = B = above.
            for y in 1..h {
                let row = y * w;
                let above = row - w;
                for x in 0..w {
                    let i = row + x;
                    residual[i] = (plane[i] as i32 - plane[above + x] as i32) as u8;
                }
            }
        }
        AlphFiltering::Gradient => {
            // First row (x>0, y=0): predictor = (x-1, 0).
            for x in 1..w {
                residual[x] = (plane[x] as i32 - plane[x - 1] as i32) as u8;
            }
            for y in 1..h {
                let row = y * w;
                let above = row - w;
                // Left-most (0, y>0): predictor = (0, y-1).
                residual[row] = (plane[row] as i32 - plane[above] as i32) as u8;
                // Interior (x>0, y>0): predictor = clip(A + B − C).
                for x in 1..w {
                    let i = row + x;
                    let a = plane[i - 1] as i32;
                    let b = plane[above + x] as i32;
                    let c = plane[above + x - 1] as i32;
                    let pred = clip(a + b - c) as i32;
                    residual[i] = (plane[i] as i32 - pred) as u8;
                }
            }
        }
    }

    residual
}

/// §2.7.1.2 forward-filter selection cost: the sum of residual byte
/// *magnitudes* interpreted as signed two's-complement (-128..=127).
///
/// A residual close to 0 (or 255, i.e. -1) means the predictor tracked the
/// alpha plane well; the filter whose residual has the smallest total
/// magnitude is the one whose residual stream is the flattest — the
/// standard predictor-selection heuristic (RFC 9649 §2.7.1.2 leaves the
/// *choice* of filter to the encoder; only the per-method reconstruction
/// is normative). For a method-0 (raw) ALPH the payload length is
/// `w*h + 1` for every filter, so this cost is the only thing that
/// distinguishes the candidates; it does not affect decoded bytes — every
/// candidate round-trips to the same plane.
fn forward_filter_cost(residual: &[u8]) -> u64 {
    residual
        .iter()
        .map(|&r| u64::from((r as i8).unsigned_abs()))
        .sum()
}

/// Encode an 8-bit alpha plane into a complete §2.7.1.2 `ALPH` chunk
/// payload (info byte + raw, **compression-method-0** alpha bitstream).
///
/// `plane` is `width * height` alpha bytes in scan order. The encoder
/// runs the §2.7.1.2 forward filter for all four `F` methods
/// (None / Horizontal / Vertical / Gradient) and keeps the one whose
/// residual stream is flattest by [`forward_filter_cost`], emitting that
/// filter's residual as the raw alpha bitstream with the matching `F`
/// field set in the info byte (`C` = 0 raw, `P` = 0, `Rsv` = 0). The
/// returned bytes are exactly the §2.3 chunk *payload* — the caller wraps
/// them in the `ALPH` FourCc + size header.
///
/// The output is bit-exactly round-trippable: for every selected filter,
/// [`decode_alpha`] reconstructs `plane` byte-for-byte (the forward filter
/// is the algebraic inverse of the §2.7.1.2 inverse filter, and method-0
/// stores the residual verbatim). Which filter wins only changes the `F`
/// nibble and the residual bytes, never the reconstructed plane — the same
/// "cost model picks the mode, not the result" invariant the lossless
/// encoder upholds.
///
/// # Errors
///
/// Returns [`AlphError::RawLengthMismatch`] if `plane.len()` does not equal
/// `width * height`, or [`AlphError::DimensionsOverflow`] if `width *
/// height` overflows `usize`. A zero-area plane (`width == 0` or
/// `height == 0`) is valid and yields a single info byte with no bitstream.
pub fn encode_alpha(plane: &[u8], width: u32, height: u32) -> Result<Vec<u8>, AlphError> {
    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(AlphError::DimensionsOverflow { width, height })?;

    if plane.len() != count {
        return Err(AlphError::RawLengthMismatch {
            expected: count,
            actual: plane.len(),
        });
    }

    let w = width as usize;
    let h = height as usize;

    // Pick the §2.7.1.2 F method whose forward residual is flattest. For a
    // zero-area plane the residual is empty under every method, so None
    // (the no-op, smallest info byte) is the natural pick.
    let mut best_f = AlphFiltering::None;
    let mut best_residual = forward_filter(plane, w, h, AlphFiltering::None);
    let mut best_cost = forward_filter_cost(&best_residual);

    for f in [
        AlphFiltering::Horizontal,
        AlphFiltering::Vertical,
        AlphFiltering::Gradient,
    ] {
        let residual = forward_filter(plane, w, h, f);
        let cost = forward_filter_cost(&residual);
        if cost < best_cost {
            best_cost = cost;
            best_f = f;
            best_residual = residual;
        }
    }

    let f_bits = match best_f {
        AlphFiltering::None => 0u8,
        AlphFiltering::Horizontal => 1,
        AlphFiltering::Vertical => 2,
        AlphFiltering::Gradient => 3,
    };
    // Info byte: Rsv=0, P=0, F=best_f, C=0 (raw). MSB-first Rsv|P|F|C.
    let info_byte = f_bits << 2;

    let mut payload = Vec::with_capacity(1 + best_residual.len());
    payload.push(info_byte);
    payload.extend_from_slice(&best_residual);
    Ok(payload)
}

/// Encode an alpha plane under one specific §2.7.1.2 filtering method
/// (raw, compression-method-0), without the [`encode_alpha`] best-filter
/// search. The returned bytes are the §2.3 `ALPH` chunk payload (info byte
/// with the given `F` field + raw residual). Exposed so callers that
/// already know the optimal filter — or want a fixed one for testing /
/// reproducibility — can skip the four-way sweep; the round-trip guarantee
/// is identical (`decode_alpha` reconstructs the input plane exactly).
///
/// # Errors
///
/// Same as [`encode_alpha`]: length-mismatch and dimension-overflow.
pub fn encode_alpha_with_filter(
    plane: &[u8],
    width: u32,
    height: u32,
    filtering: AlphFiltering,
) -> Result<Vec<u8>, AlphError> {
    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(AlphError::DimensionsOverflow { width, height })?;

    if plane.len() != count {
        return Err(AlphError::RawLengthMismatch {
            expected: count,
            actual: plane.len(),
        });
    }

    let residual = forward_filter(plane, width as usize, height as usize, filtering);
    let f_bits = match filtering {
        AlphFiltering::None => 0u8,
        AlphFiltering::Horizontal => 1,
        AlphFiltering::Vertical => 2,
        AlphFiltering::Gradient => 3,
    };
    let info_byte = f_bits << 2;

    let mut payload = Vec::with_capacity(1 + residual.len());
    payload.push(info_byte);
    payload.extend_from_slice(&residual);
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compose an ALPH info byte from its four 2-bit fields, MSB-first.
    fn info(rsv: u8, p: u8, f: u8, c: u8) -> u8 {
        ((rsv & 0b11) << 6) | ((p & 0b11) << 4) | ((f & 0b11) << 2) | (c & 0b11)
    }

    #[test]
    fn empty_payload_is_rejected_with_named_error() {
        // §2.7.1.2 Figure 10 mandates one info byte at minimum.
        assert_eq!(AlphHeader::parse(&[]), Err(AlphError::EmptyPayload));
    }

    #[test]
    fn all_zero_info_decodes_to_none_none_none_zero() {
        // info = 0x00 → C=0, F=0, P=0, Rsv=0. The simplest legal ALPH.
        let h = AlphHeader::parse(&[0x00]).unwrap();
        assert_eq!(h.compression, AlphCompression::None);
        assert_eq!(h.filtering, AlphFiltering::None);
        assert_eq!(h.preprocessing, AlphPreprocessing::None);
        assert_eq!(h.reserved, 0);
        assert_eq!(h.info_byte, 0);
        assert_eq!(h.bitstream_offset(), 1);
    }

    #[test]
    fn compression_field_decodes_all_four_values() {
        // C nibble at bits 1..0.
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 0)]).unwrap().compression,
            AlphCompression::None
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 1)]).unwrap().compression,
            AlphCompression::Lossless
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 2)]).unwrap().compression,
            AlphCompression::Reserved(2)
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 3)]).unwrap().compression,
            AlphCompression::Reserved(3)
        );
    }

    #[test]
    fn filtering_field_decodes_all_four_methods() {
        // F nibble at bits 3..2. All four are named in §2.7.1.2.
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 0)]).unwrap().filtering,
            AlphFiltering::None
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 1, 0)]).unwrap().filtering,
            AlphFiltering::Horizontal
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 2, 0)]).unwrap().filtering,
            AlphFiltering::Vertical
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 3, 0)]).unwrap().filtering,
            AlphFiltering::Gradient
        );
    }

    #[test]
    fn preprocessing_field_decodes_both_named_values_plus_reserved() {
        // P nibble at bits 5..4. §2.7.1.2 names 0 + 1.
        assert_eq!(
            AlphHeader::parse(&[info(0, 0, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::None
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 1, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::LevelReduction
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 2, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::Reserved(2)
        );
        assert_eq!(
            AlphHeader::parse(&[info(0, 3, 0, 0)])
                .unwrap()
                .preprocessing,
            AlphPreprocessing::Reserved(3)
        );
    }

    #[test]
    fn reserved_field_surfaces_raw_two_bit_value_without_rejection() {
        // §2.7.1.2: "MUST be 0. Readers MUST ignore this field." So a
        // non-zero Rsv must parse, with the raw value carried through.
        for rsv in 0u8..=3 {
            let h = AlphHeader::parse(&[info(rsv, 0, 0, 0)]).unwrap();
            assert_eq!(h.reserved, rsv, "Rsv={rsv}");
            // Named fields stay clean.
            assert_eq!(h.compression, AlphCompression::None);
            assert_eq!(h.filtering, AlphFiltering::None);
            assert_eq!(h.preprocessing, AlphPreprocessing::None);
        }
    }

    #[test]
    fn fields_decode_independently_across_a_full_combination() {
        // Hand-pick a byte where every nibble is non-zero & distinct:
        // Rsv=2, P=3, F=1, C=2  →  10 11 01 10  =  0xB6
        let h = AlphHeader::parse(&[0xB6]).unwrap();
        assert_eq!(h.reserved, 0b10);
        assert_eq!(h.preprocessing, AlphPreprocessing::Reserved(0b11));
        assert_eq!(h.filtering, AlphFiltering::Horizontal);
        assert_eq!(h.compression, AlphCompression::Reserved(0b10));
        assert_eq!(h.info_byte, 0xB6);
    }

    #[test]
    fn fixture_lossy_with_alpha_info_byte_decodes_to_lossless_no_filter_no_pre() {
        // docs/image/webp/fixtures/lossy-with-alpha-128x128/trace.txt
        //   ALPH method=1 filter=0 pre_processing=0 header_byte=0x01
        let h = AlphHeader::parse(&[0x01]).unwrap();
        assert_eq!(h.compression, AlphCompression::Lossless);
        assert_eq!(h.filtering, AlphFiltering::None);
        assert_eq!(h.preprocessing, AlphPreprocessing::None);
        assert_eq!(h.reserved, 0);
        assert_eq!(h.info_byte, 0x01);
    }

    #[test]
    fn bitstream_offset_is_always_one_past_the_info_byte() {
        // §2.7.1.2 "Alpha bitstream: _Chunk Size_ bytes - 1" — i.e.
        // payload[1..] for any payload that survives parse().
        let h = AlphHeader::parse(&[0x01, 0xAA, 0xBB]).unwrap();
        assert_eq!(h.bitstream_offset(), 1);
    }

    #[test]
    fn trailing_bytes_are_not_consumed_by_the_info_byte_parse() {
        // Extra bytes (the actual bitstream) must NOT change the
        // decoded info-byte fields; the parser only reads byte 0.
        let baseline = AlphHeader::parse(&[0x01]).unwrap();
        let with_tail = AlphHeader::parse(&[0x01, 0xFF, 0x00, 0x55, 0xAA]).unwrap();
        assert_eq!(baseline, with_tail);
    }

    // ---- decode_alpha: §2.7.1.2 bitstream decode ----

    /// Build an ALPH payload with filter method `f` and compression
    /// method 0 (raw): the info byte followed by the residual stream.
    fn raw_alph(f: u8, residual: &[u8]) -> Vec<u8> {
        let mut v = vec![info(0, 0, f, 0)];
        v.extend_from_slice(residual);
        v
    }

    #[test]
    fn decode_raw_uncompressed_no_filter_is_identity() {
        // §2.7.1.2 method 0 + filter 0: alpha = (0 + X) % 256 = X.
        let residual = [10u8, 5, 250, 3, 100, 200];
        let payload = raw_alph(0, &residual);
        let plane = decode_alpha(&payload, 3, 2).unwrap();
        assert_eq!(plane, residual.to_vec());
    }

    #[test]
    fn decode_raw_length_mismatch_is_rejected() {
        // method 0 requires exactly width*height residual bytes.
        let payload = raw_alph(0, &[1, 2, 3]); // 3 bytes for a 2x2 (=4) plane.
        assert_eq!(
            decode_alpha(&payload, 2, 2),
            Err(AlphError::RawLengthMismatch {
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn decode_unsupported_compression_method_is_rejected() {
        // C = 2 → Reserved(2) → UnsupportedCompression.
        let payload = vec![info(0, 0, 0, 2), 0, 0, 0, 0];
        assert_eq!(
            decode_alpha(&payload, 2, 2),
            Err(AlphError::UnsupportedCompression(2))
        );
    }

    #[test]
    fn decode_horizontal_filter_inverse() {
        // §2.7.1.2 method 1 (horizontal): predictor = A (left); the
        // left-most pixel (0, y>0) uses (0, y-1); (0,0) uses 0.
        //   X     = [10,  5, 250,   3, 100, 200]   (3x2)
        //   out   = [10, 15,   9,  13, 113,  57]
        let residual = [10u8, 5, 250, 3, 100, 200];
        let payload = raw_alph(1, &residual);
        let plane = decode_alpha(&payload, 3, 2).unwrap();
        assert_eq!(plane, vec![10, 15, 9, 13, 113, 57]);
    }

    #[test]
    fn decode_vertical_filter_inverse() {
        // §2.7.1.2 method 2 (vertical): predictor = B (above); the
        // top-most pixel (x>0, 0) uses (x-1, 0); (0,0) uses 0.
        //   X   = [10,  5, 250,  3, 100, 200]   (3x2)
        //   out = [10, 15,   9, 13, 115, 209]
        let residual = [10u8, 5, 250, 3, 100, 200];
        let payload = raw_alph(2, &residual);
        let plane = decode_alpha(&payload, 3, 2).unwrap();
        assert_eq!(plane, vec![10, 15, 9, 13, 115, 209]);
    }

    #[test]
    fn decode_gradient_filter_inverse() {
        // §2.7.1.2 method 3 (gradient): predictor = clip(A+B-C) for
        // interior pixels; left-most uses above, top-most uses left,
        // (0,0) uses 0.
        //   X   = [10,  5,  7,   3, 100,  50,  20,   8,   9]   (3x3)
        //   out = [10, 15, 22,  13, 118, 175,  33, 146, 212]
        let residual = [10u8, 5, 7, 3, 100, 50, 20, 8, 9];
        let payload = raw_alph(3, &residual);
        let plane = decode_alpha(&payload, 3, 3).unwrap();
        assert_eq!(plane, vec![10, 15, 22, 13, 118, 175, 33, 146, 212]);
    }

    #[test]
    fn decode_modulo_256_wraps_into_0_255() {
        // §2.7.1.2: "modulo-256 arithmetic to wrap the [256..511] range
        // into the [0..255] one." Horizontal, single row.
        //   X   = [200, 200]   →   out = [200, (200+200)%256 = 144]
        let payload = raw_alph(1, &[200, 200]);
        let plane = decode_alpha(&payload, 2, 1).unwrap();
        assert_eq!(plane, vec![200, 144]);
    }

    #[test]
    fn decode_gradient_clip_clamps_predictor() {
        // Force clip() to clamp high: A=255, B=255, C=0 → A+B-C=510 →
        // clip=255. Build a 2x2 whose reconstruction reaches that.
        //   X   = [255, 0, 0, 5]   (2x2)
        //   (0,0)=255; (1,0) top-most pred=255 → 255; (0,1) left-most
        //   pred=out(0,0)=255 → 255; (1,1) interior A=255 B=255 C=255 →
        //   clip(255)=255 → (255+5)%256 = 4.
        let payload = raw_alph(3, &[255, 0, 0, 5]);
        let plane = decode_alpha(&payload, 2, 2).unwrap();
        assert_eq!(plane, vec![255, 255, 255, 4]);
    }

    #[test]
    fn decode_zero_area_plane_is_empty() {
        // A 0xN or Nx0 plane decodes to an empty raw plane (length 0).
        let payload = raw_alph(0, &[]);
        assert_eq!(decode_alpha(&payload, 0, 4).unwrap(), Vec::<u8>::new());
        assert_eq!(decode_alpha(&payload, 4, 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn decode_empty_payload_is_rejected() {
        assert_eq!(decode_alpha(&[], 1, 1), Err(AlphError::EmptyPayload));
    }

    /// Straight per-pixel transcription of the §2.7.1.2 inverse filter as
    /// the round-291 `match (x, y)` / `match filtering` form read, kept
    /// here as the byte-identity oracle for the round-293 border-rule
    /// hoist. If the hoisted [`inverse_filter`] ever diverges from this
    /// reference on any plane / method, the test below fails.
    fn inverse_filter_reference(filtered: &[u8], w: usize, h: usize, f: AlphFiltering) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        let idx = |x: usize, y: usize| y * w + x;
        for y in 0..h {
            for x in 0..w {
                let xv = filtered[idx(x, y)] as i32;
                let predictor: i32 = match (x, y) {
                    (0, 0) => 0,
                    _ => match f {
                        AlphFiltering::None => 0,
                        AlphFiltering::Horizontal => {
                            if x == 0 {
                                out[idx(0, y - 1)] as i32
                            } else {
                                out[idx(x - 1, y)] as i32
                            }
                        }
                        AlphFiltering::Vertical => {
                            if y == 0 {
                                out[idx(x - 1, 0)] as i32
                            } else {
                                out[idx(x, y - 1)] as i32
                            }
                        }
                        AlphFiltering::Gradient => {
                            if x == 0 {
                                out[idx(0, y - 1)] as i32
                            } else if y == 0 {
                                out[idx(x - 1, 0)] as i32
                            } else {
                                let a = out[idx(x - 1, y)] as i32;
                                let b = out[idx(x, y - 1)] as i32;
                                let c = out[idx(x - 1, y - 1)] as i32;
                                clip(a + b - c) as i32
                            }
                        }
                    },
                };
                out[idx(x, y)] = ((predictor + xv) & 0xff) as u8;
            }
        }
        out
    }

    #[test]
    fn hoisted_inverse_filter_matches_per_pixel_reference_across_methods_and_dims() {
        // Deterministic LCG residual so the predictor sees a spread of
        // neighbour values across every dimension/method combination; the
        // hoisted Stage-2 loop must equal the per-pixel reference exactly.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for &(w, h) in &[
            (1usize, 1usize),
            (1, 7),
            (7, 1),
            (2, 2),
            (3, 5),
            (5, 3),
            (16, 16),
            (13, 17),
            (128, 128),
        ] {
            let residual: Vec<u8> = (0..w * h).map(|_| next()).collect();
            for f in [
                AlphFiltering::None,
                AlphFiltering::Horizontal,
                AlphFiltering::Vertical,
                AlphFiltering::Gradient,
            ] {
                let got = inverse_filter(residual.clone(), w, h, f);
                let want = inverse_filter_reference(&residual, w, h, f);
                assert_eq!(got, want, "mismatch at {w}x{h} method {f:?}");
            }
        }
    }

    /// §2.7.1.2 forward filter `X = (alpha - predictor) mod 256`, the
    /// algebraic inverse of [`inverse_filter`]'s `alpha = (predictor +
    /// X) % 256`, written in the simplest per-pixel `match (x, y)` form.
    /// Kept as the independent oracle the production [`forward_filter`]
    /// (border-hoisted) is proven byte-for-byte against. Reads the
    /// *original* plane in scan order; each predictor sees only
    /// already-emitted neighbours, so on a correct decode the predictor
    /// the encoder reads equals the one the decoder reads from the
    /// reconstructed plane. Mirrors RFC 9649 §2.7.1.2 Figure 11 + the
    /// per-method border rules.
    fn forward_filter_reference(plane: &[u8], w: usize, h: usize, f: AlphFiltering) -> Vec<u8> {
        let pred = |p: &[u8], x: usize, y: usize| -> i32 {
            if x == 0 && y == 0 {
                return 0;
            }
            let idx = |xx: usize, yy: usize| p[yy * w + xx] as i32;
            match f {
                AlphFiltering::None => 0,
                AlphFiltering::Horizontal => {
                    if x == 0 {
                        idx(0, y - 1)
                    } else {
                        idx(x - 1, y)
                    }
                }
                AlphFiltering::Vertical => {
                    if y == 0 {
                        idx(x - 1, 0)
                    } else {
                        idx(x, y - 1)
                    }
                }
                AlphFiltering::Gradient => {
                    if x == 0 {
                        idx(0, y - 1)
                    } else if y == 0 {
                        idx(x - 1, 0)
                    } else {
                        clip(idx(x - 1, y) + idx(x, y - 1) - idx(x - 1, y - 1)) as i32
                    }
                }
            }
        };
        let mut residual = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                residual[i] = ((plane[i] as i32 - pred(plane, x, y)) & 0xff) as u8;
            }
        }
        residual
    }

    /// Round 318 — full §2.7.1.2 **compression-method-1** decode chain
    /// oracle: forward-filter a known plane, pack the residual into the
    /// GREEN channel of a §3 headerless VP8L stream (built by the
    /// crate's own encoder), wrap as a method-1 ALPH payload, and assert
    /// `decode_alpha` (VP8L decode → green extract → inverse filter)
    /// reproduces the original plane byte-for-byte across all four F
    /// methods and a spread of dimensions including the §2.7.1.2 border
    /// shapes (1-wide / 1-tall / corner).
    #[test]
    fn decode_method1_lossless_round_trips_all_filters() {
        let mut state: u32 = 0x9e37_79b9;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for &(w, h) in &[
            (1usize, 1usize),
            (1, 9),
            (9, 1),
            (2, 2),
            (3, 5),
            (16, 16),
            (13, 17),
        ] {
            let plane: Vec<u8> = (0..w * h).map(|_| next()).collect();
            for f in [
                AlphFiltering::None,
                AlphFiltering::Horizontal,
                AlphFiltering::Vertical,
                AlphFiltering::Gradient,
            ] {
                let residual = forward_filter(&plane, w, h, f);
                // Method-1 carrier: residual byte → green channel of an
                // opaque ARGB pixel. The VP8L stream is lossless, so the
                // decoder's green channel is bit-for-bit this residual.
                let argb: Vec<u32> = residual
                    .iter()
                    .map(|&g| 0xff00_0000u32 | ((g as u32) << 8))
                    .collect();
                let bitstream =
                    crate::vp8l_encode::encode_argb_literals_with_width(&argb, w as u32);
                let f_bits = match f {
                    AlphFiltering::None => 0,
                    AlphFiltering::Horizontal => 1,
                    AlphFiltering::Vertical => 2,
                    AlphFiltering::Gradient => 3,
                };
                let mut payload = vec![info(0, 0, f_bits, 1)];
                payload.extend_from_slice(&bitstream);

                let decoded = decode_alpha(&payload, w as u32, h as u32)
                    .expect("method-1 ALPH from a valid VP8L stream must decode");
                assert_eq!(
                    decoded, plane,
                    "method-1 chain mismatch at {w}x{h} filter {f:?}"
                );
            }
        }
    }

    /// The border-hoisted production [`forward_filter`] must equal the
    /// simple per-pixel [`forward_filter_reference`] byte-for-byte across
    /// every F method and a spread of even/odd + degenerate dimensions
    /// (including the §2.7.1.2 border shapes).
    #[test]
    fn forward_filter_matches_per_pixel_reference() {
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        };
        for &(w, h) in &[
            (1usize, 1usize),
            (1, 7),
            (7, 1),
            (2, 2),
            (3, 4),
            (5, 5),
            (16, 9),
            (13, 13),
        ] {
            let plane: Vec<u8> = (0..w * h).map(|_| next()).collect();
            for f in [
                AlphFiltering::None,
                AlphFiltering::Horizontal,
                AlphFiltering::Vertical,
                AlphFiltering::Gradient,
            ] {
                assert_eq!(
                    forward_filter(&plane, w, h, f),
                    forward_filter_reference(&plane, w, h, f),
                    "forward_filter != reference at {w}x{h} method {f:?}"
                );
            }
        }
    }

    /// `encode_alpha` → `decode_alpha` is the identity for arbitrary alpha
    /// planes: whichever §2.7.1.2 filter the cost model selects, the raw
    /// (method-0) payload reconstructs the exact input plane.
    #[test]
    fn encode_alpha_round_trips_to_input_plane() {
        let mut state: u32 = 0xdead_beef;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for &(w, h) in &[
            (1u32, 1u32),
            (1, 11),
            (11, 1),
            (2, 2),
            (4, 3),
            (16, 16),
            (17, 13),
            (33, 7),
        ] {
            let n = (w * h) as usize;
            // A few content regimes: flat, gradient-ish, and noisy.
            for plane in [
                vec![0xffu8; n],                                     // fully opaque
                (0..n).map(|i| (i % 256) as u8).collect::<Vec<_>>(), // ramp
                (0..n).map(|_| next()).collect::<Vec<_>>(),          // noise
            ] {
                let payload = encode_alpha(&plane, w, h).expect("encode_alpha must succeed");
                // Payload is info byte + w*h raw residual bytes (method 0).
                assert_eq!(payload.len(), 1 + n, "method-0 payload length at {w}x{h}");
                let header = AlphHeader::parse(&payload).unwrap();
                assert_eq!(header.compression, AlphCompression::None);
                assert_eq!(header.preprocessing, AlphPreprocessing::None);
                assert_eq!(header.reserved, 0);

                let decoded = decode_alpha(&payload, w, h).expect("decode_alpha must succeed");
                assert_eq!(decoded, plane, "encode→decode mismatch at {w}x{h}");
            }
        }
    }

    /// `encode_alpha_with_filter` round-trips under every explicitly
    /// requested filter, and the selected `F` field is written into the
    /// info byte.
    #[test]
    fn encode_alpha_with_filter_round_trips_each_method() {
        let w = 9u32;
        let h = 6u32;
        let n = (w * h) as usize;
        let plane: Vec<u8> = (0..n).map(|i| ((i * 31 + 5) & 0xff) as u8).collect();
        for (f, f_bits) in [
            (AlphFiltering::None, 0u8),
            (AlphFiltering::Horizontal, 1),
            (AlphFiltering::Vertical, 2),
            (AlphFiltering::Gradient, 3),
        ] {
            let payload = encode_alpha_with_filter(&plane, w, h, f).unwrap();
            let header = AlphHeader::parse(&payload).unwrap();
            assert_eq!(header.filtering, f, "F field not the requested method");
            assert_eq!(header.info_byte, f_bits << 2);
            let decoded = decode_alpha(&payload, w, h).unwrap();
            assert_eq!(decoded, plane, "round-trip failed for {f:?}");
        }
    }

    /// A flat (constant) alpha plane filters to all-zero residual under
    /// Horizontal / Vertical / Gradient, so the best-filter chooser drops
    /// to a non-None predictive filter (cost 0 beats raw's nonzero cost on
    /// any non-zero constant).
    #[test]
    fn encode_alpha_prefers_predictive_filter_on_flat_plane() {
        let w = 8u32;
        let h = 8u32;
        let plane = vec![200u8; (w * h) as usize];
        let payload = encode_alpha(&plane, w, h).unwrap();
        let header = AlphHeader::parse(&payload).unwrap();
        assert_ne!(
            header.filtering,
            AlphFiltering::None,
            "a flat non-zero plane should pick a predictive filter (zero residual)"
        );
        // The residual after the info byte must be all zero except (0,0).
        let residual = &payload[1..];
        assert_eq!(residual[0], 200, "(0,0) stores the raw value");
        assert!(
            residual[1..].iter().all(|&r| r == 0),
            "flat plane residual must be zero past (0,0)"
        );
        assert_eq!(decode_alpha(&payload, w, h).unwrap(), plane);
    }

    /// Length / dimension error surfaces match the decoder's.
    #[test]
    fn encode_alpha_rejects_length_mismatch() {
        let err = encode_alpha(&[0u8; 5], 2, 3).unwrap_err();
        assert_eq!(
            err,
            AlphError::RawLengthMismatch {
                expected: 6,
                actual: 5
            }
        );
    }

    /// A zero-area plane is valid and yields a bare info byte (no bitstream).
    #[test]
    fn encode_alpha_zero_area_is_bare_info_byte() {
        let payload = encode_alpha(&[], 0, 5).unwrap();
        assert_eq!(payload, vec![0u8]); // F=None, C=0 → info byte 0x00
        assert_eq!(decode_alpha(&payload, 0, 5).unwrap(), Vec::<u8>::new());
    }
}
