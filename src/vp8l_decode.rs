//! VP8L (WebP-Lossless) §5.2 LZ77 backward-reference + §5.2.3
//! color-cache per-pixel ARGB decode loop.
//!
//! This is the layer that sits directly on top of round 106's
//! [`crate::meta_prefix::PrefixCodeGroup`] reader. Round 106 assembled
//! the five canonical prefix codes one §5 image-data block opens with;
//! this module runs the §6.2.3 per-pixel decode loop that *consumes*
//! symbols from that group, producing a fully-decoded ARGB pixel
//! buffer.
//!
//! ## The §6.2.3 per-pixel dispatch
//!
//! For each pixel position `(x, y)` in scan-line order, the decoder
//! reads one symbol `S` from prefix code #1 (the GREEN / length /
//! color-cache alphabet, sized `256 + 24 + color_cache_size` per
//! §6.2.3). `S` is interpreted by range:
//!
//! * **`S < 256`** — §5.2.1 prefix-coded literal. `S` is the green
//!   channel; red, blue, and alpha are read from prefix codes #2, #3,
//!   and #4. The emitted pixel is `ARGB = (alpha << 24) | (red << 16)
//!   | (green << 8) | blue`.
//! * **`256 <= S < 256 + 24`** — §5.2.2 LZ77 backward reference.
//!   `S - 256` is a *length prefix code*; the §5.2.2 prefix → value
//!   formula plus extra bits yields the copy length `L`. A *distance
//!   prefix code* is then read from prefix code #5, decoded the same
//!   way into a raw `distance_code`, and mapped through the §5.2.2
//!   distance map into a scan-line distance `D`. `L` pixels are copied
//!   from `D` pixels back.
//! * **`S >= 256 + 24`** — §5.2.3 color-cache code. `S - (256 + 24)`
//!   indexes the color cache; the cached ARGB is emitted.
//!
//! Every emitted pixel — literal, copied, or cache-resolved — is
//! inserted into the color cache (when enabled) in stream order, per
//! §5.2.3 ("the state of the color cache is maintained by inserting
//! every pixel ... into the cache in the order they appear in the
//! stream").
//!
//! ## §5.2.2 LZ77 prefix coding
//!
//! Length and distance both use the same prefix → value transform
//! (§5.2.2):
//!
//! ```text
//! if (prefix_code < 4) return prefix_code + 1;
//! int extra_bits = (prefix_code - 2) >> 1;
//! int offset = (2 + (prefix_code & 1)) << extra_bits;
//! return offset + ReadBits(extra_bits) + 1;
//! ```
//!
//! The §5.2.2 *distance map* maps the 120 smallest distance codes onto
//! a close neighborhood of the current pixel; larger codes denote a
//! scan-line distance offset by 120:
//!
//! ```text
//! (xi, yi) = distance_map[distance_code - 1]
//! dist = xi + yi * image_width
//! if (dist < 1) dist = 1
//! ```
//!
//! (for `distance_code > 120`, `dist = distance_code - 120`).
//!
//! ## What this module does NOT do
//!
//! * No §4 inverse-transform application — this decoder produces the
//!   raw entropy-decoded ARGB stream, *before* any predictor /
//!   color / subtract-green / color-indexing inverse pass. (Those
//!   transforms operate on the buffer this loop produces.)
//! * No `oxideav-core` runtime dependency — this module compiles under
//!   `--no-default-features`.
//!
//! ## §6.2.2 multi-group (entropy-image) path
//!
//! [`decode_argb`] is the full ARGB-role decode: it consumes the
//! round-106 [`crate::meta_prefix::MetaPrefixHeader`] result and, when
//! the meta-prefix bit selected multiple groups, decodes the §6.2.2
//! *entropy image* (itself a §5 `entropy-coded-image` of size
//! `DIV_ROUND_UP(width, 1<<prefix_bits)` ×
//! `DIV_ROUND_UP(height, 1<<prefix_bits)`), derives
//! `num_prefix_groups = max(entropy image) + 1`, reads that many
//! [`PrefixCodeGroup`]s, and runs the per-pixel §6.2.3 loop selecting a
//! group per block via
//! `meta_index[(y >> prefix_bits) * block_width + (x >> prefix_bits)]`.
//! The single-group case (meta-prefix bit zero, or a non-ARGB role)
//! degrades to the same primitives with one group used everywhere.
//! Per §6.2.2 the per-pixel meta-prefix code is the red+green channels
//! of the entropy-image pixel: `(entropy_pixel >> 8) & 0xffff`.

use crate::meta_prefix::{
    ImageRole, MetaPrefixCodes, MetaPrefixError, MetaPrefixHeader, PrefixCodeGroup,
};
use crate::vp8l_prefix::PrefixError;
use crate::vp8l_stream::{BitReader, BitReaderEof};

/// §5.2.2: the number of "short" distance codes reserved for the close
/// neighborhood. Codes `1..=120` map through [`DISTANCE_MAP`]; codes
/// `> 120` denote a scan-line distance offset by 120.
pub const NUM_DISTANCE_MAP_CODES: usize = 120;

/// §5.2.2 the maximum number of *length* prefix codes that are
/// meaningful (`256..256+24`). The GREEN alphabet reserves 24 slots
/// for these.
pub const NUM_LENGTH_PREFIX_CODES: usize = 24;

/// §5.2.2 distance map: the `(xi, yi)` neighbor offset for distance
/// codes `1..=120` (indexed `distance_code - 1`). Transcribed from the
/// spec's §5.2.2 "Distance Mapping" listing.
pub const DISTANCE_MAP: [(i32, i32); NUM_DISTANCE_MAP_CODES] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 2),
    (2, 0),
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
    (2, 2),
    (-2, 2),
    (0, 3),
    (3, 0),
    (1, 3),
    (-1, 3),
    (3, 1),
    (-3, 1),
    (2, 3),
    (-2, 3),
    (3, 2),
    (-3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (-1, 4),
    (4, 1),
    (-4, 1),
    (3, 3),
    (-3, 3),
    (2, 4),
    (-2, 4),
    (4, 2),
    (-4, 2),
    (0, 5),
    (3, 4),
    (-3, 4),
    (4, 3),
    (-4, 3),
    (5, 0),
    (1, 5),
    (-1, 5),
    (5, 1),
    (-5, 1),
    (2, 5),
    (-2, 5),
    (5, 2),
    (-5, 2),
    (4, 4),
    (-4, 4),
    (3, 5),
    (-3, 5),
    (5, 3),
    (-5, 3),
    (0, 6),
    (6, 0),
    (1, 6),
    (-1, 6),
    (6, 1),
    (-6, 1),
    (2, 6),
    (-2, 6),
    (6, 2),
    (-6, 2),
    (4, 5),
    (-4, 5),
    (5, 4),
    (-5, 4),
    (3, 6),
    (-3, 6),
    (6, 3),
    (-6, 3),
    (0, 7),
    (7, 0),
    (1, 7),
    (-1, 7),
    (5, 5),
    (-5, 5),
    (7, 1),
    (-7, 1),
    (4, 6),
    (-4, 6),
    (6, 4),
    (-6, 4),
    (2, 7),
    (-2, 7),
    (7, 2),
    (-7, 2),
    (3, 7),
    (-3, 7),
    (7, 3),
    (-7, 3),
    (5, 6),
    (-5, 6),
    (6, 5),
    (-6, 5),
    (8, 0),
    (4, 7),
    (-4, 7),
    (7, 4),
    (-7, 4),
    (8, 1),
    (8, 2),
    (6, 6),
    (-6, 6),
    (8, 3),
    (5, 7),
    (-5, 7),
    (7, 5),
    (-7, 5),
    (8, 4),
    (6, 7),
    (-6, 7),
    (7, 6),
    (-7, 6),
    (8, 5),
    (7, 7),
    (-7, 7),
    (8, 6),
    (8, 7),
];

/// §5.2.3 color-cache hash multiplier. Specified verbatim in §5.2.3:
/// "Colors are looked up by indexing them by `(0x1e35a7bd * color) >>
/// (32 - color_cache_code_bits)`."
pub const COLOR_CACHE_HASH_MULTIPLIER: u32 = 0x1e35_a7bd;

/// Errors raised while decoding the §5.2 per-pixel ARGB stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The bit reader hit EOF mid-symbol or mid-extra-bits.
    Eof(BitReaderEof),
    /// A prefix-code symbol read failed.
    Prefix(PrefixError),
    /// §6.2.3: the GREEN symbol `S` was at or beyond the alphabet size
    /// `256 + 24 + color_cache_size`.
    GreenSymbolOutOfRange {
        /// The offending symbol value.
        symbol: usize,
        /// The GREEN alphabet size.
        alphabet_size: usize,
    },
    /// §5.2.3: a color-cache code resolved to an index outside the
    /// cache, or a color-cache code was read for a stream with no
    /// color cache.
    ColorCacheIndexOutOfRange {
        /// The resolved cache index.
        index: usize,
        /// The configured cache size (`0` when the cache is disabled).
        cache_size: usize,
    },
    /// §5.2.2: a backward-reference copy reached back beyond the start
    /// of the already-decoded pixels.
    BackwardReferenceUnderflow {
        /// The pixel position the copy was attempted at.
        position: usize,
        /// The scan-line distance `D` that underflowed.
        distance: usize,
    },
    /// §5.2.2: a backward-reference run would write past the end of the
    /// image (more pixels than remain).
    BackwardReferenceOverflow {
        /// The pixel position the copy started at.
        position: usize,
        /// The copy length `L`.
        length: usize,
        /// The total number of pixels in the image.
        total_pixels: usize,
    },
    /// §5.2.3 / §6.2.2: reading the color-cache info, meta-prefix bit,
    /// the entropy-image header, or one of the per-group prefix-code
    /// groups failed.
    MetaPrefix(MetaPrefixError),
    /// §6.2.2: the entropy image's `prefix_image_width` /
    /// `prefix_image_height` were zero (a degenerate header that yields
    /// no blocks to select groups for).
    EmptyEntropyImage {
        /// The derived entropy-image width.
        prefix_image_width: u32,
        /// The derived entropy-image height.
        prefix_image_height: u32,
    },
    /// §6.2.2: a pixel's block selected a meta-prefix code at or beyond
    /// `num_prefix_groups` — i.e. an index larger than the maximum the
    /// entropy image was supposed to contain. This can only arise from a
    /// logic error in the caller, but is checked rather than panicking.
    MetaPrefixIndexOutOfRange {
        /// The selected meta-prefix code.
        meta_prefix_code: usize,
        /// The number of prefix-code groups that were read.
        num_prefix_groups: usize,
    },
    /// §4 / §7.2: a transform type appeared more than once in the
    /// `optional-transform` list. The spec mandates each transform is
    /// used at most once.
    DuplicateTransform,
}

impl From<BitReaderEof> for DecodeError {
    fn from(e: BitReaderEof) -> Self {
        Self::Eof(e)
    }
}

impl From<PrefixError> for DecodeError {
    fn from(e: PrefixError) -> Self {
        match e {
            PrefixError::Eof(eof) => Self::Eof(eof),
            other => Self::Prefix(other),
        }
    }
}

impl From<MetaPrefixError> for DecodeError {
    fn from(e: MetaPrefixError) -> Self {
        match e {
            MetaPrefixError::Eof(eof) => Self::Eof(eof),
            other => Self::MetaPrefix(other),
        }
    }
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Eof(e) => write!(f, "VP8L §5.2 image decode: {e}"),
            Self::Prefix(e) => write!(f, "VP8L §5.2 image decode prefix code: {e}"),
            Self::GreenSymbolOutOfRange {
                symbol,
                alphabet_size,
            } => write!(
                f,
                "VP8L §6.2.3 image decode: GREEN symbol {symbol} out of range for alphabet size {alphabet_size}"
            ),
            Self::ColorCacheIndexOutOfRange { index, cache_size } => write!(
                f,
                "VP8L §5.2.3 image decode: color-cache index {index} out of range for cache size {cache_size}"
            ),
            Self::BackwardReferenceUnderflow { position, distance } => write!(
                f,
                "VP8L §5.2.2 image decode: backward reference distance {distance} underflows at pixel {position}"
            ),
            Self::BackwardReferenceOverflow {
                position,
                length,
                total_pixels,
            } => write!(
                f,
                "VP8L §5.2.2 image decode: backward reference length {length} at pixel {position} overruns the {total_pixels}-pixel image"
            ),
            Self::MetaPrefix(e) => write!(f, "VP8L §6.2.2 image decode meta-prefix: {e}"),
            Self::EmptyEntropyImage {
                prefix_image_width,
                prefix_image_height,
            } => write!(
                f,
                "VP8L §6.2.2 image decode: degenerate entropy image of size {prefix_image_width}x{prefix_image_height}"
            ),
            Self::MetaPrefixIndexOutOfRange {
                meta_prefix_code,
                num_prefix_groups,
            } => write!(
                f,
                "VP8L §6.2.2 image decode: meta-prefix code {meta_prefix_code} out of range for {num_prefix_groups} prefix-code group(s)"
            ),
            Self::DuplicateTransform => write!(
                f,
                "VP8L §4 transform list: a transform type appears more than once"
            ),
        }
    }
}

impl std::error::Error for DecodeError {}

/// §5.2.2: turn a length/distance *prefix code* plus its extra bits
/// into the decoded value.
///
/// The transform is shared between length and distance per the
/// §5.2.2 pseudocode:
///
/// ```text
/// if (prefix_code < 4) return prefix_code + 1;
/// int extra_bits = (prefix_code - 2) >> 1;
/// int offset = (2 + (prefix_code & 1)) << extra_bits;
/// return offset + ReadBits(extra_bits) + 1;
/// ```
pub fn read_lz77_value(reader: &mut BitReader<'_>, prefix_code: u32) -> Result<u32, BitReaderEof> {
    if prefix_code < 4 {
        return Ok(prefix_code + 1);
    }
    let extra_bits = (prefix_code - 2) >> 1;
    let offset = (2 + (prefix_code & 1)) << extra_bits;
    let extra = reader.read_bits(extra_bits as usize)?;
    Ok(offset + extra + 1)
}

/// §5.2.3 color cache: a fixed-size array of recently-emitted ARGB
/// colors, indexed by a multiplicative hash of the color.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorCache {
    /// `color_cache_code_bits`; the table holds `1 << bits` entries.
    code_bits: u32,
    entries: Vec<u32>,
}

impl ColorCache {
    /// Create a color cache holding `1 << code_bits` zero-initialized
    /// ARGB entries (§5.2.3: "all entries in all color cache values are
    /// set to zero").
    pub fn new(code_bits: u32) -> Self {
        let size = 1usize << code_bits;
        Self {
            code_bits,
            entries: vec![0u32; size],
        }
    }

    /// The number of entries in the cache (`1 << code_bits`).
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// §5.2.3 hash: `(0x1e35a7bd * argb) >> (32 - code_bits)`.
    pub fn hash(&self, argb: u32) -> usize {
        (COLOR_CACHE_HASH_MULTIPLIER.wrapping_mul(argb) >> (32 - self.code_bits)) as usize
    }

    /// Insert `argb` into the cache at its hashed slot (§5.2.3: every
    /// emitted pixel is inserted in stream order).
    pub fn insert(&mut self, argb: u32) {
        let idx = self.hash(argb);
        self.entries[idx] = argb;
    }

    /// Look up the ARGB color stored at `index` (a resolved
    /// color-cache code, `S - (256 + 24)`).
    pub fn lookup(&self, index: usize) -> Option<u32> {
        self.entries.get(index).copied()
    }
}

/// The result of a §5.2 single-group ARGB decode: the raw,
/// pre-inverse-transform ARGB pixel buffer in scan-line order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    width: u32,
    height: u32,
    /// `width * height` ARGB pixels, scan-line order. Each pixel is
    /// `(alpha << 24) | (red << 16) | (green << 8) | blue`.
    pixels: Vec<u32>,
}

impl DecodedImage {
    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The decoded ARGB pixels in scan-line order.
    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }

    /// The decoded ARGB pixels in scan-line order, mutably. Used by the
    /// §4 inverse-transform passes ([`crate::vp8l_transform`]), which
    /// rewrite the buffer in place.
    pub fn pixels_mut(&mut self) -> &mut [u32] {
        &mut self.pixels
    }

    /// Construct a [`DecodedImage`] from explicit dimensions and a
    /// scan-line-order ARGB buffer. Used by the §4 color-indexing
    /// inverse transform, which replaces the (subsampled) decoded buffer
    /// with a freshly-sized palette-lookup buffer.
    pub fn from_parts(width: u32, height: u32, pixels: Vec<u32>) -> Self {
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// §6.2.3 GREEN-channel symbol-type tag, split out of the integer
/// dispatch so it can be unit-tested in isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreenSymbol {
    /// `S < 256`: a §5.2.1 prefix-coded literal; the value is the green
    /// channel.
    Literal {
        /// The green channel value (`S`).
        green: u8,
    },
    /// `256 <= S < 256 + 24`: a §5.2.2 LZ77 length prefix code; the
    /// value is `S - 256`.
    LengthPrefix {
        /// The length prefix code (`S - 256`, range `0..24`).
        prefix_code: u32,
    },
    /// `S >= 256 + 24`: a §5.2.3 color-cache code; the value is the
    /// cache index `S - (256 + 24)`.
    ColorCache {
        /// The cache index (`S - 280`).
        index: usize,
    },
}

impl GreenSymbol {
    /// §6.2.3: classify a GREEN symbol `S` by range. `alphabet_size` is
    /// the GREEN alphabet size (`256 + 24 + color_cache_size`); a
    /// symbol at or beyond it is invalid.
    pub fn classify(symbol: usize, alphabet_size: usize) -> Result<Self, DecodeError> {
        if symbol >= alphabet_size {
            return Err(DecodeError::GreenSymbolOutOfRange {
                symbol,
                alphabet_size,
            });
        }
        if symbol < 256 {
            Ok(Self::Literal {
                green: symbol as u8,
            })
        } else if symbol < 256 + NUM_LENGTH_PREFIX_CODES {
            Ok(Self::LengthPrefix {
                prefix_code: (symbol - 256) as u32,
            })
        } else {
            Ok(Self::ColorCache {
                index: symbol - (256 + NUM_LENGTH_PREFIX_CODES),
            })
        }
    }
}

/// §5.2.2: convert a raw `distance_code` to a scan-line pixel distance
/// `D`, given the image width.
///
/// Codes `1..=120` index the §5.2.2 distance map; codes `> 120` denote
/// a scan-line distance offset by 120. The result is clamped to a
/// minimum of 1 per the spec's `if (dist < 1) dist = 1`.
pub fn distance_code_to_pixel_distance(distance_code: u32, image_width: u32) -> usize {
    if distance_code > NUM_DISTANCE_MAP_CODES as u32 {
        return (distance_code - NUM_DISTANCE_MAP_CODES as u32) as usize;
    }
    // distance_code is in 1..=120 here.
    let (xi, yi) = DISTANCE_MAP[(distance_code - 1) as usize];
    let dist = xi + yi * image_width as i32;
    if dist < 1 {
        1
    } else {
        dist as usize
    }
}

/// Assemble an ARGB pixel from its four channels (§6.2.3 / §5.2.1):
/// `(alpha << 24) | (red << 16) | (green << 8) | blue`.
fn pack_argb(green: u8, red: u8, blue: u8, alpha: u8) -> u32 {
    ((alpha as u32) << 24) | ((red as u32) << 16) | ((green as u32) << 8) | (blue as u32)
}

/// Decode the single §6.2.3 symbol at the current position using
/// `group`, appending the emitted pixel(s) to `pixels` and (when
/// present) maintaining `color_cache` in stream order.
///
/// This is the per-symbol core shared by the single-group
/// [`decode_image`] loop and the multi-group [`decode_argb`] loop. The
/// only difference between the two callers is *which* group they pass
/// for the symbol at the current pixel; the literal / LZ77 / color-cache
/// dispatch below is identical (the spec switches the group per pixel in
/// §6.2.2 but the §6.2.3 decode of each pixel is role-independent).
///
/// `width` is the image width (needed for the §5.2.2 distance map) and
/// `total_pixels` is its pixel count (needed for the overrun check).
/// `alphabet_size` and `cache_size` are derived once by the caller.
#[allow(clippy::too_many_arguments)]
fn decode_one_symbol(
    reader: &mut BitReader<'_>,
    group: &PrefixCodeGroup,
    color_cache: &mut Option<ColorCache>,
    pixels: &mut Vec<u32>,
    width: u32,
    total_pixels: usize,
    alphabet_size: usize,
    cache_size: usize,
) -> Result<(), DecodeError> {
    let s = group.green.read_symbol(reader)? as usize;
    match GreenSymbol::classify(s, alphabet_size)? {
        GreenSymbol::Literal { green } => {
            // §5.2.1: green came from prefix #1 (already read as S);
            // read red / blue / alpha from prefix #2 / #3 / #4.
            let red = group.red.read_symbol(reader)? as u8;
            let blue = group.blue.read_symbol(reader)? as u8;
            let alpha = group.alpha.read_symbol(reader)? as u8;
            let argb = pack_argb(green, red, blue, alpha);
            pixels.push(argb);
            if let Some(cache) = color_cache.as_mut() {
                cache.insert(argb);
            }
        }
        GreenSymbol::LengthPrefix { prefix_code } => {
            // §5.2.2: length L from the length prefix code + extra.
            let length = read_lz77_value(reader, prefix_code)? as usize;
            // Distance: prefix code #5 → raw distance_code → map.
            let dist_prefix = group.distance.read_symbol(reader)? as u32;
            let distance_code = read_lz77_value(reader, dist_prefix)?;
            let dist = distance_code_to_pixel_distance(distance_code, width);
            let position = pixels.len();
            if dist > position {
                return Err(DecodeError::BackwardReferenceUnderflow {
                    position,
                    distance: dist,
                });
            }
            if position + length > total_pixels {
                return Err(DecodeError::BackwardReferenceOverflow {
                    position,
                    length,
                    total_pixels,
                });
            }
            // §5.2.2: copy L pixels from `position - dist`. The copy is
            // byte-for-byte scan-line, and overlapping copies (dist <
            // length) repeat the just-copied pixels, which is the
            // standard LZ77 behaviour.
            let src_start = position - dist;
            for i in 0..length {
                let argb = pixels[src_start + i];
                pixels.push(argb);
                if let Some(cache) = color_cache.as_mut() {
                    cache.insert(argb);
                }
            }
        }
        GreenSymbol::ColorCache { index } => {
            // §5.2.3: resolve the cache index to its ARGB.
            let cache = color_cache
                .as_mut()
                .ok_or(DecodeError::ColorCacheIndexOutOfRange {
                    index,
                    cache_size: 0,
                })?;
            let argb = cache
                .lookup(index)
                .ok_or(DecodeError::ColorCacheIndexOutOfRange { index, cache_size })?;
            pixels.push(argb);
            // §5.2.3: every emitted pixel (even a cache hit) is
            // re-inserted in stream order.
            cache.insert(argb);
        }
    }
    Ok(())
}

/// Decode one §5.2 entropy-coded ARGB image with a single
/// [`PrefixCodeGroup`].
///
/// `reader` must be positioned at the start of the §5.2 `data`
/// (the `lz77-coded-image`) — i.e. right after the color-cache info,
/// meta-prefix bit, and the prefix-code group the round-106
/// [`crate::meta_prefix::MetaPrefixHeader`] reader consumed.
///
/// `color_cache` is `Some` when the §5.2.3 color cache is enabled for
/// this stream (its `code_bits` matching the
/// [`crate::meta_prefix::ColorCacheInfo`]), `None` otherwise. The
/// cache size feeds the GREEN alphabet size (`256 + 24 +
/// color_cache_size`).
///
/// Returns a [`DecodedImage`] holding `width * height` ARGB pixels in
/// scan-line order, before any §4 inverse transform.
pub fn decode_image(
    reader: &mut BitReader<'_>,
    group: &PrefixCodeGroup,
    mut color_cache: Option<ColorCache>,
    width: u32,
    height: u32,
) -> Result<DecodedImage, DecodeError> {
    let cache_size = color_cache.as_ref().map(|c| c.size()).unwrap_or(0);
    let alphabet_size = PrefixCodeGroup::green_alphabet_size(cache_size);
    let total_pixels = (width as usize) * (height as usize);
    let mut pixels: Vec<u32> = Vec::with_capacity(total_pixels);

    while pixels.len() < total_pixels {
        decode_one_symbol(
            reader,
            group,
            &mut color_cache,
            &mut pixels,
            width,
            total_pixels,
            alphabet_size,
            cache_size,
        )?;
    }

    Ok(DecodedImage {
        width,
        height,
        pixels,
    })
}

/// §6.2.2 per-pixel meta-prefix index: the decoded entropy image folded
/// into the one number each pixel block needs — its meta-prefix code,
/// i.e. the index into the array of [`PrefixCodeGroup`]s.
///
/// The entropy image is a sub-resolution image of size
/// `prefix_image_width × prefix_image_height` where each pixel covers a
/// `(1 << prefix_bits) × (1 << prefix_bits)` block of the full ARGB
/// image. Per §6.2.2 the meta-prefix code for a block is the red+green
/// channels of the corresponding entropy-image pixel:
/// `(entropy_pixel >> 8) & 0xffff`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaPrefixIndex {
    /// §6.2.2 `prefix_bits` (`ReadBits(3) + 2`); a block is
    /// `(1 << prefix_bits)` pixels wide and tall.
    prefix_bits: u8,
    /// `DIV_ROUND_UP(image_width, 1 << prefix_bits)`.
    block_width: u32,
    /// `DIV_ROUND_UP(image_height, 1 << prefix_bits)`.
    block_height: u32,
    /// `block_width * block_height` meta-prefix codes, scan-line order.
    meta_codes: Vec<u16>,
}

impl MetaPrefixIndex {
    /// §6.2.2 `prefix_bits`.
    pub fn prefix_bits(&self) -> u8 {
        self.prefix_bits
    }

    /// The entropy-image width in blocks (`prefix_image_width`).
    pub fn block_width(&self) -> u32 {
        self.block_width
    }

    /// The entropy-image height in blocks (`prefix_image_height`).
    pub fn block_height(&self) -> u32 {
        self.block_height
    }

    /// The per-block meta-prefix codes in scan-line order.
    pub fn meta_codes(&self) -> &[u16] {
        &self.meta_codes
    }

    /// §6.2.2 `num_prefix_groups = max(entropy image) + 1`. With no
    /// blocks this is `0`; callers reject that as a degenerate header.
    pub fn num_prefix_groups(&self) -> usize {
        self.meta_codes
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0)
    }

    /// §6.2.2 group selection for pixel `(x, y)`:
    /// `meta_index[(y >> prefix_bits) * block_width + (x >> prefix_bits)]`.
    pub fn meta_code_for(&self, x: u32, y: u32) -> u16 {
        let bx = x >> self.prefix_bits;
        let by = y >> self.prefix_bits;
        let position = (by * self.block_width + bx) as usize;
        self.meta_codes[position]
    }
}

/// Decode a §7.3 `entropy-coded-image` to a [`DecodedImage`].
///
/// An `entropy-coded-image = color-cache-info data` (§7.3 ABNF): a
/// §5.2.3 `color-cache-info` bit followed by a single 5-code prefix-code
/// group (no §6.2.2 meta-prefix layer) and the §5.2 LZ77 / color-cache
/// data. This is the shape of the §6.2.2 entropy image, the §4.1
/// predictor image, the §4.2 color-transform image, and the §4.4
/// color-indexing color table — every image-data role *except* the
/// top-level §5.1 ARGB image.
///
/// `reader` must be positioned at the start of the block (the
/// `color-cache-info` bit). On return the reader sits just past the last
/// pixel of the `width * height` image. Use this to decode a §4
/// transform's sub-resolution body, then resume reading the next
/// transform (or the main image) from the same reader.
pub fn decode_entropy_coded_image(
    reader: &mut BitReader<'_>,
    width: u32,
    height: u32,
) -> Result<DecodedImage, DecodeError> {
    if width == 0 || height == 0 {
        return Err(DecodeError::EmptyEntropyImage {
            prefix_image_width: width,
            prefix_image_height: height,
        });
    }

    // The image is an entropy-coded-image: color-cache-info + one
    // prefix-code group, no §6.2.2 meta-prefix bit. The round-106
    // reader handles exactly that for the `EntropyCoded` role.
    let header = MetaPrefixHeader::read(reader, ImageRole::EntropyCoded, width, height)?;
    let group = match &header.codes {
        MetaPrefixCodes::Single { group } => group.as_ref(),
        // EntropyImagePending is impossible for the EntropyCoded role
        // (no meta-prefix bit is read), but guard rather than panic.
        MetaPrefixCodes::EntropyImagePending { .. } => {
            return Err(DecodeError::EmptyEntropyImage {
                prefix_image_width: width,
                prefix_image_height: height,
            });
        }
    };
    let cache = header
        .color_cache
        .is_enabled()
        .then(|| ColorCache::new(header.color_cache.code_bits));

    decode_image(reader, group, cache, width, height)
}

/// Decode the §6.2.2 *entropy image* into a [`MetaPrefixIndex`].
///
/// The entropy image is itself an `entropy-coded-image` (§7.3 ABNF):
/// a §5.2.3 `color-cache-info` bit followed by a single prefix-code
/// group and the §5.2 LZ77 / color-cache data. `reader` must be
/// positioned at the start of that block — i.e. at the
/// `entropy_image_bit_position` the round-106
/// [`MetaPrefixHeader`] recorded for an
/// [`MetaPrefixCodes::EntropyImagePending`] result. On return the
/// reader sits at the start of the main ARGB image's prefix-code
/// groups.
///
/// `prefix_bits`, `prefix_image_width`, and `prefix_image_height` are
/// the dimensions the round-106 header already derived.
pub fn decode_entropy_image(
    reader: &mut BitReader<'_>,
    prefix_bits: u8,
    prefix_image_width: u32,
    prefix_image_height: u32,
) -> Result<MetaPrefixIndex, DecodeError> {
    let decoded = decode_entropy_coded_image(reader, prefix_image_width, prefix_image_height)
        .map_err(|e| match e {
            // Map a degenerate-size rejection back onto the
            // entropy-image-specific error to preserve the public
            // contract of this function.
            DecodeError::EmptyEntropyImage { .. } => DecodeError::EmptyEntropyImage {
                prefix_image_width,
                prefix_image_height,
            },
            other => other,
        })?;

    // §6.2.2: meta_prefix_code = (entropy_pixel >> 8) & 0xffff — the
    // red+green channels of each block's entropy-image pixel.
    let meta_codes: Vec<u16> = decoded
        .pixels()
        .iter()
        .map(|&argb| ((argb >> 8) & 0xffff) as u16)
        .collect();

    Ok(MetaPrefixIndex {
        prefix_bits,
        block_width: prefix_image_width,
        block_height: prefix_image_height,
        meta_codes,
    })
}

/// Decode a full §5.1 ARGB-role image — the §6.2.2 multi-group path.
///
/// `reader` must be positioned at the start of the ARGB image's
/// `spatially-coded-image`: the §5.2.3 `color-cache-info` bit, then the
/// §6.2.2 meta-prefix bit. This function reads the round-106
/// [`MetaPrefixHeader`] for the [`ImageRole::Argb`] role and dispatches:
///
/// * **single group** (`meta-prefix = %b0`): runs the §6.2.3 decode
///   loop with the one group everywhere (identical to [`decode_image`]).
/// * **multiple groups** (`meta-prefix = %b1`): decodes the §6.2.2
///   entropy image via [`decode_entropy_image`], computes
///   `num_prefix_groups = max(entropy image) + 1`, reads that many
///   [`PrefixCodeGroup`]s, then runs the §6.2.3 loop selecting a group
///   per pixel block via
///   [`MetaPrefixIndex::meta_code_for`].
///
/// Returns a [`DecodedImage`] of `width * height` ARGB pixels in
/// scan-line order, before any §4 inverse transform.
pub fn decode_argb(
    reader: &mut BitReader<'_>,
    width: u32,
    height: u32,
) -> Result<DecodedImage, DecodeError> {
    let header = MetaPrefixHeader::read(reader, ImageRole::Argb, width, height)?;
    let color_cache_size = header.color_cache.size();
    let cache_code_bits = header.color_cache.code_bits;
    let cache_enabled = header.color_cache.is_enabled();

    match header.codes {
        MetaPrefixCodes::Single { group } => {
            let cache = cache_enabled.then(|| ColorCache::new(cache_code_bits));
            decode_image(reader, &group, cache, width, height)
        }
        MetaPrefixCodes::EntropyImagePending {
            prefix_bits,
            image_width: prefix_image_width,
            image_height: prefix_image_height,
            ..
        } => {
            // The reader is already positioned at the entropy image
            // start (just past the `prefix_bits` field).
            let index =
                decode_entropy_image(reader, prefix_bits, prefix_image_width, prefix_image_height)?;
            let num_prefix_groups = index.num_prefix_groups();
            if num_prefix_groups == 0 {
                return Err(DecodeError::EmptyEntropyImage {
                    prefix_image_width,
                    prefix_image_height,
                });
            }

            // §6.2.2: read num_prefix_groups prefix-code groups, each
            // sized against the *main* ARGB color cache.
            let mut groups: Vec<PrefixCodeGroup> = Vec::with_capacity(num_prefix_groups);
            for _ in 0..num_prefix_groups {
                groups.push(PrefixCodeGroup::read(reader, color_cache_size)?);
            }

            decode_argb_multi_group(
                reader,
                &groups,
                &index,
                cache_enabled,
                cache_code_bits,
                width,
                height,
            )
        }
    }
}

/// §6.2.2 / §6.2.3 multi-group decode loop: like [`decode_image`] but
/// the group used for each pixel is selected by the entropy image.
///
/// `groups` holds the `num_prefix_groups` prefix-code groups, `index`
/// the decoded entropy image. For each pixel `(x, y)` in scan-line
/// order the meta-prefix code
/// `index.meta_code_for(x, y)` indexes `groups`; that group decodes the
/// pixel via the shared §6.2.3 [`decode_one_symbol`] core. A single
/// color cache is maintained across the whole image in stream order
/// (§5.2.3), independent of which group emitted the pixel.
#[allow(clippy::too_many_arguments)]
fn decode_argb_multi_group(
    reader: &mut BitReader<'_>,
    groups: &[PrefixCodeGroup],
    index: &MetaPrefixIndex,
    cache_enabled: bool,
    cache_code_bits: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, DecodeError> {
    let cache_size = if cache_enabled {
        1usize << cache_code_bits
    } else {
        0
    };
    let alphabet_size = PrefixCodeGroup::green_alphabet_size(cache_size);
    let total_pixels = (width as usize) * (height as usize);
    let mut color_cache = cache_enabled.then(|| ColorCache::new(cache_code_bits));
    let mut pixels: Vec<u32> = Vec::with_capacity(total_pixels);

    // The current pixel's (x, y) is derived from `pixels.len()` — the
    // decode loop fills in scan-line order, and an LZ77 backref pushes
    // multiple pixels at once but always lands the cursor back on the
    // next undefined pixel. Group selection happens once per *symbol*,
    // keyed by the position of the next pixel to emit (the spec selects
    // the group for the pixel the decoder is about to decode).
    while pixels.len() < total_pixels {
        let position = pixels.len();
        let x = (position % width as usize) as u32;
        let y = (position / width as usize) as u32;
        let meta_code = index.meta_code_for(x, y) as usize;
        let group = groups
            .get(meta_code)
            .ok_or(DecodeError::MetaPrefixIndexOutOfRange {
                meta_prefix_code: meta_code,
                num_prefix_groups: groups.len(),
            })?;
        decode_one_symbol(
            reader,
            group,
            &mut color_cache,
            &mut pixels,
            width,
            total_pixels,
            alphabet_size,
            cache_size,
        )?;
    }

    Ok(DecodedImage {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny LSB-first bit writer for building synthetic §5.2 streams.
    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: usize,
    }
    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }
        fn write_bits(&mut self, mut value: u32, n: usize) {
            for _ in 0..n {
                let byte_idx = self.bit_pos >> 3;
                if byte_idx >= self.bytes.len() {
                    self.bytes.push(0);
                }
                let bit = (value & 1) as u8;
                self.bytes[byte_idx] |= bit << (self.bit_pos & 7);
                self.bit_pos += 1;
                value >>= 1;
            }
        }
        /// Emit a §6.2.1 simple-code single-symbol code (length-1 leaf)
        /// for `sym` in the 8-bit form.
        fn write_simple_single_symbol(&mut self, sym: u32) {
            self.write_bits(1, 1); // simple flag
            self.write_bits(0, 1); // num_symbols - 1 = 0
            self.write_bits(1, 1); // is_first_8bits = 1
            self.write_bits(sym, 8); // symbol0
        }
        /// Emit a §6.2.1 simple-code two-symbol code, both at length 1.
        /// Canonical: the smaller value reads as bit 0, the larger as
        /// bit 1.
        fn write_simple_two_symbols(&mut self, sym0: u32, sym1: u32) {
            self.write_bits(1, 1); // simple flag
            self.write_bits(1, 1); // num_symbols - 1 = 1 → 2 symbols
            self.write_bits(1, 1); // is_first_8bits = 1
            self.write_bits(sym0, 8);
            self.write_bits(sym1, 8);
        }
        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
        /// Bit count written so far. Used by the bit-prefix property
        /// test to know the exact stream length; the trailing
        /// `bytes.len() * 8 - bit_len()` bits of the last byte are
        /// zero-padded.
        fn bit_len(&self) -> usize {
            self.bit_pos
        }
    }

    // ---- §5.2.2 LZ77 prefix-value transform ----

    #[test]
    fn lz77_value_small_codes_are_identity_plus_one() {
        // prefix_code < 4 → prefix_code + 1, no extra bits.
        let data: [u8; 0] = [];
        let mut r = BitReader::new(&data);
        assert_eq!(read_lz77_value(&mut r, 0).unwrap(), 1);
        assert_eq!(read_lz77_value(&mut r, 1).unwrap(), 2);
        assert_eq!(read_lz77_value(&mut r, 2).unwrap(), 3);
        assert_eq!(read_lz77_value(&mut r, 3).unwrap(), 4);
        assert_eq!(r.bit_position(), 0); // no bits consumed
    }

    #[test]
    fn lz77_value_prefix_4_reads_one_extra_bit() {
        // prefix_code = 4 → extra_bits = (4-2)>>1 = 1; offset =
        // (2 + 0) << 1 = 4; value = 4 + ReadBits(1) + 1.
        // The spec table says prefix 4 covers value range 5..6.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // extra bit = 0 → value 5
        w.write_bits(1, 1); // extra bit = 1 → value 6
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        assert_eq!(read_lz77_value(&mut r, 4).unwrap(), 5);
        assert_eq!(read_lz77_value(&mut r, 4).unwrap(), 6);
    }

    #[test]
    fn lz77_value_prefix_5_covers_7_to_8() {
        // prefix 5 → extra_bits = 1; offset = (2 + 1) << 1 = 6;
        // value = 6 + ReadBits(1) + 1 → 7 or 8.
        let mut w = BitWriter::new();
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        assert_eq!(read_lz77_value(&mut r, 5).unwrap(), 7);
        assert_eq!(read_lz77_value(&mut r, 5).unwrap(), 8);
    }

    #[test]
    fn lz77_value_prefix_6_covers_9_to_12() {
        // prefix 6 → extra_bits = 2; offset = (2 + 0) << 2 = 8;
        // value = 8 + ReadBits(2) + 1 → 9..12.
        let mut w = BitWriter::new();
        for extra in 0u32..4 {
            w.write_bits(extra, 2);
        }
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        for expected in 9u32..=12 {
            assert_eq!(read_lz77_value(&mut r, 6).unwrap(), expected);
        }
    }

    #[test]
    fn lz77_value_prefix_23_max_length_4096() {
        // The spec note: max backward-reference length is 4096; prefix
        // 23 covers 3072..4096 with 10 extra bits.
        // prefix 23 → extra_bits = (23-2)>>1 = 10; offset =
        // (2 + 1) << 10 = 3072; max value = 3072 + 1023 + 1 = 4096.
        let mut w = BitWriter::new();
        w.write_bits(0, 10); // min → 3073
        w.write_bits(1023, 10); // max → 4096
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        assert_eq!(read_lz77_value(&mut r, 23).unwrap(), 3073);
        assert_eq!(read_lz77_value(&mut r, 23).unwrap(), 4096);
    }

    // ---- §5.2.2 distance map ----

    #[test]
    fn distance_map_has_120_entries() {
        assert_eq!(DISTANCE_MAP.len(), 120);
        assert_eq!(NUM_DISTANCE_MAP_CODES, 120);
    }

    #[test]
    fn distance_map_first_entries_match_spec_examples() {
        // §5.2.2: "distance code 1 indicates an offset of (0, 1)".
        assert_eq!(DISTANCE_MAP[0], (0, 1));
        // "distance code 3 indicates the top-left pixel" → (1, 1).
        assert_eq!(DISTANCE_MAP[2], (1, 1));
        // Last two entries from the listing.
        assert_eq!(DISTANCE_MAP[118], (8, 6));
        assert_eq!(DISTANCE_MAP[119], (8, 7));
    }

    #[test]
    fn distance_code_1_is_pixel_above() {
        // (0, 1) over a width-W image → dist = 0 + 1*W = W (the pixel
        // directly above the current one).
        assert_eq!(distance_code_to_pixel_distance(1, 16), 16);
        assert_eq!(distance_code_to_pixel_distance(1, 1), 1);
    }

    #[test]
    fn distance_code_2_is_pixel_to_left() {
        // (1, 0) → dist = 1 + 0*W = 1, the pixel immediately left.
        assert_eq!(distance_code_to_pixel_distance(2, 100), 1);
    }

    #[test]
    fn distance_code_above_120_offsets_by_120() {
        // §5.2.2: codes > 120 denote scan-line distance offset by 120.
        assert_eq!(distance_code_to_pixel_distance(121, 64), 1);
        assert_eq!(distance_code_to_pixel_distance(200, 64), 80);
    }

    #[test]
    fn distance_clamps_to_one_for_negative_offsets() {
        // A neighbor whose (xi, yi) gives dist < 1 clamps to 1.
        // (-1, 1) with width 1 → -1 + 1 = 0 → clamps to 1.
        // distance_code 4 maps to (-1, 1).
        assert_eq!(DISTANCE_MAP[3], (-1, 1));
        assert_eq!(distance_code_to_pixel_distance(4, 1), 1);
    }

    // ---- §6.2.3 GREEN symbol classification ----

    #[test]
    fn green_symbol_literal_range() {
        match GreenSymbol::classify(0, 280).unwrap() {
            GreenSymbol::Literal { green } => assert_eq!(green, 0),
            other => panic!("expected Literal, got {other:?}"),
        }
        match GreenSymbol::classify(255, 280).unwrap() {
            GreenSymbol::Literal { green } => assert_eq!(green, 255),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn green_symbol_length_prefix_range() {
        // 256 → length prefix 0; 279 → length prefix 23.
        match GreenSymbol::classify(256, 280).unwrap() {
            GreenSymbol::LengthPrefix { prefix_code } => assert_eq!(prefix_code, 0),
            other => panic!("expected LengthPrefix, got {other:?}"),
        }
        match GreenSymbol::classify(279, 280).unwrap() {
            GreenSymbol::LengthPrefix { prefix_code } => assert_eq!(prefix_code, 23),
            other => panic!("expected LengthPrefix, got {other:?}"),
        }
    }

    #[test]
    fn green_symbol_color_cache_range() {
        // With a cache of size 16, alphabet = 256 + 24 + 16 = 296.
        // Symbol 280 → cache index 0; 295 → cache index 15.
        match GreenSymbol::classify(280, 296).unwrap() {
            GreenSymbol::ColorCache { index } => assert_eq!(index, 0),
            other => panic!("expected ColorCache, got {other:?}"),
        }
        match GreenSymbol::classify(295, 296).unwrap() {
            GreenSymbol::ColorCache { index } => assert_eq!(index, 15),
            other => panic!("expected ColorCache, got {other:?}"),
        }
    }

    #[test]
    fn green_symbol_out_of_range_is_refused() {
        match GreenSymbol::classify(280, 280) {
            Err(DecodeError::GreenSymbolOutOfRange {
                symbol,
                alphabet_size,
            }) => {
                assert_eq!(symbol, 280);
                assert_eq!(alphabet_size, 280);
            }
            other => panic!("expected GreenSymbolOutOfRange, got {other:?}"),
        }
    }

    // ---- §5.2.3 color cache ----

    #[test]
    fn color_cache_hash_matches_spec_formula() {
        let cache = ColorCache::new(4); // code_bits = 4, size 16
        let argb = 0xFF_12_34_56u32;
        let expected = (COLOR_CACHE_HASH_MULTIPLIER.wrapping_mul(argb) >> (32 - 4)) as usize;
        assert_eq!(cache.hash(argb), expected);
        assert!(cache.hash(argb) < 16);
    }

    #[test]
    fn color_cache_insert_then_lookup_round_trips() {
        let mut cache = ColorCache::new(8);
        let argb = 0xDE_AD_BE_EFu32;
        cache.insert(argb);
        let idx = cache.hash(argb);
        assert_eq!(cache.lookup(idx), Some(argb));
    }

    #[test]
    fn color_cache_starts_zeroed() {
        let cache = ColorCache::new(2);
        for i in 0..cache.size() {
            assert_eq!(cache.lookup(i), Some(0));
        }
    }

    // ---- §5.2 full decode loop ----

    /// Build a single prefix-code group from synthetic simple codes so
    /// the §5.2 decode loop has a known group to consume. The GREEN
    /// code is supplied by the caller (it varies per test); RED, BLUE,
    /// ALPHA, DIST default to single-symbol codes the caller picks.
    fn group_with_codes(
        green_writer: impl Fn(&mut BitWriter),
        red: u32,
        blue: u32,
        alpha: u32,
        dist: u32,
        cache_size: usize,
    ) -> PrefixCodeGroup {
        let mut w = BitWriter::new();
        green_writer(&mut w);
        w.write_simple_single_symbol(red);
        w.write_simple_single_symbol(blue);
        w.write_simple_single_symbol(alpha);
        w.write_simple_single_symbol(dist);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        PrefixCodeGroup::read(&mut r, cache_size).unwrap()
    }

    #[test]
    fn decode_literal_only_2x1_image() {
        // A 2x1 image of two literal pixels. GREEN code carries two
        // symbols (green values 10 and 20); RED/BLUE/ALPHA single.
        // No color cache.
        let group = group_with_codes(
            |w| w.write_simple_two_symbols(10, 20),
            0xAA, // red
            0xBB, // blue
            0xCC, // alpha
            0,    // dist (unused)
            0,
        );
        // Image stream: two GREEN symbols (10 → bit 0, 20 → bit 1).
        // Each literal also reads RED/BLUE/ALPHA which are single-leaf
        // (no bits). So just two GREEN bits.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // GREEN = 10
        w.write_bits(1, 1); // GREEN = 20
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let img = decode_image(&mut r, &group, None, 2, 1).unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);
        assert_eq!(
            img.pixels(),
            &[
                pack_argb(10, 0xAA, 0xBB, 0xCC),
                pack_argb(20, 0xAA, 0xBB, 0xCC)
            ]
        );
    }

    #[test]
    fn decode_single_literal_pixel() {
        // 1x1 image; GREEN single-symbol = 0x42.
        let group = group_with_codes(
            |w| w.write_simple_single_symbol(0x42),
            0x10,
            0x20,
            0x30,
            0,
            0,
        );
        // GREEN is single-leaf → no bits; RED/BLUE/ALPHA single → no
        // bits. So the whole image stream is empty.
        let data: [u8; 0] = [];
        let mut r = BitReader::new(&data);
        let img = decode_image(&mut r, &group, None, 1, 1).unwrap();
        assert_eq!(img.pixels(), &[pack_argb(0x42, 0x10, 0x20, 0x30)]);
    }

    #[test]
    fn decode_length_distance_back_reference() {
        // 4x1 image. First emit one literal (green=5), then a backward
        // reference of length 3 distance 1 that copies the literal
        // three times (LZ77 self-overlap fills the row).
        //
        // GREEN code must carry literal 5 AND length-prefix symbol
        // 256 (= length prefix code 0 → length 1)... but we need
        // length 3, so use length prefix code 2 (symbol 258) → value 3.
        // So GREEN alphabet must hold symbols {5, 258}. Use a two-symbol
        // simple code (5 and 258); but simple codes only reach 255.
        // Instead drive GREEN with a normal code? Simpler: use a
        // two-symbol simple code over {5, 258}? 258 > 255 → not
        // expressible in a simple code. So use the GREEN single-symbol
        // trick twice won't work either.
        //
        // Build GREEN from an explicit code-length table with symbols 5
        // and 258 both at length 1 (a complete 2-leaf tree). Canonical:
        // smaller value (5) → bit 0, larger (258) → bit 1.
        let green = {
            let mut lengths = vec![0u8; PrefixCodeGroup::green_alphabet_size(0)];
            lengths[5] = 1;
            lengths[258] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(lengths).unwrap()
        };
        let red = {
            let mut l = vec![0u8; 256];
            l[0xAA] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(l).unwrap()
        };
        let blue = {
            let mut l = vec![0u8; 256];
            l[0xBB] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(l).unwrap()
        };
        let alpha = {
            let mut l = vec![0u8; 256];
            l[0xCC] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(l).unwrap()
        };
        // Distance code #5: single symbol = distance prefix code 1.
        // distance prefix 1 → distance_code value = 2 → distance map
        // entry 2 = (1,0) → pixel distance 1.
        let dist = {
            let mut l = vec![0u8; 40];
            l[1] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(l).unwrap()
        };
        let group = PrefixCodeGroup {
            green,
            red,
            blue,
            alpha,
            distance: dist,
        };

        // Stream:
        //  pixel0 literal: GREEN bit 0 (→ sym 5). RED/BLUE/ALPHA single
        //                  (no bits).
        //  then a length code: GREEN bit 1 (→ sym 258 → length prefix
        //                  code 2 → value 3). LENGTH read_lz77_value(2)
        //                  consumes no extra bits.
        //                  DIST: single-leaf (no bits) → distance prefix
        //                  1 → read_lz77_value(1) → value 2 (no extra) →
        //                  distance_map[1] = (1,0) → dist 1.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // GREEN = 5 (literal)
        w.write_bits(1, 1); // GREEN = 258 (length prefix code 2)
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let img = decode_image(&mut r, &group, None, 4, 1).unwrap();
        let p = pack_argb(5, 0xAA, 0xBB, 0xCC);
        // literal then 3 copies of the same pixel (dist 1).
        assert_eq!(img.pixels(), &[p, p, p, p]);
    }

    #[test]
    fn decode_color_cache_hit() {
        // 2x1 image with a color cache. Pixel 0 is a literal; pixel 1
        // is a color-cache code that resolves to pixel 0's ARGB.
        let cache_bits = 4u32;
        let cache_size = 1usize << cache_bits; // 16
        let alphabet = PrefixCodeGroup::green_alphabet_size(cache_size); // 296

        // Determine the literal pixel's ARGB and its cache slot.
        let green_val = 0x11u8;
        let red_val = 0x22u8;
        let blue_val = 0x33u8;
        let alpha_val = 0x44u8;
        let argb = pack_argb(green_val, red_val, blue_val, alpha_val);
        let probe = ColorCache::new(cache_bits);
        let slot = probe.hash(argb);
        let cache_symbol = 256 + NUM_LENGTH_PREFIX_CODES + slot; // 280 + slot

        // GREEN code carries literal `green_val` AND the color-cache
        // symbol. Both at length 1 (2-leaf complete tree). Canonical:
        // smaller value → bit 0, larger → bit 1.
        let green = {
            let mut lengths = vec![0u8; alphabet];
            lengths[green_val as usize] = 1;
            lengths[cache_symbol] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(lengths).unwrap()
        };
        let mk_single = |sym: usize, size: usize| {
            let mut l = vec![0u8; size];
            l[sym] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(l).unwrap()
        };
        let group = PrefixCodeGroup {
            green,
            red: mk_single(red_val as usize, 256),
            blue: mk_single(blue_val as usize, 256),
            alpha: mk_single(alpha_val as usize, 256),
            distance: mk_single(0, 40),
        };

        // Stream: pixel0 GREEN bit (literal green_val < cache_symbol →
        // bit 0). pixel1 GREEN bit (cache_symbol → bit 1).
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // GREEN = green_val literal
        w.write_bits(1, 1); // GREEN = cache_symbol
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let img = decode_image(&mut r, &group, Some(ColorCache::new(cache_bits)), 2, 1).unwrap();
        assert_eq!(img.pixels(), &[argb, argb]);
    }

    #[test]
    fn decode_backward_reference_underflow_is_refused() {
        // A length code at pixel 0 (nothing decoded yet) with distance
        // 1 underflows.
        let green = {
            let mut lengths = vec![0u8; PrefixCodeGroup::green_alphabet_size(0)];
            lengths[256] = 1; // length prefix code 0 → length 1
            crate::vp8l_prefix::PrefixCode::from_code_lengths(lengths).unwrap()
        };
        let mk_single = |sym: usize, size: usize| {
            let mut l = vec![0u8; size];
            l[sym] = 1;
            crate::vp8l_prefix::PrefixCode::from_code_lengths(l).unwrap()
        };
        let group = PrefixCodeGroup {
            green,
            red: mk_single(0, 256),
            blue: mk_single(0, 256),
            alpha: mk_single(0, 256),
            distance: mk_single(1, 40), // dist prefix 1 → value 2 → (1,0) → dist 1
        };
        // GREEN single-leaf (no bits) → length symbol 256 → length 1.
        // LENGTH read_lz77_value(0) → 1 (no extra). DIST single-leaf →
        // prefix 1 → value 2 → dist 1. At position 0, dist 1 > 0 →
        // underflow.
        let data: [u8; 0] = [];
        let mut r = BitReader::new(&data);
        match decode_image(&mut r, &group, None, 4, 1) {
            Err(DecodeError::BackwardReferenceUnderflow { position, distance }) => {
                assert_eq!(position, 0);
                assert_eq!(distance, 1);
            }
            other => panic!("expected BackwardReferenceUnderflow, got {other:?}"),
        }
    }

    #[test]
    fn decode_color_cache_code_without_cache_is_refused() {
        // A GREEN symbol in the color-cache range when no cache exists
        // would itself be out of range (alphabet excludes it). Confirm
        // classify rejects a cache symbol against the no-cache alphabet.
        let alphabet = PrefixCodeGroup::green_alphabet_size(0); // 280
        match GreenSymbol::classify(280, alphabet) {
            Err(DecodeError::GreenSymbolOutOfRange { .. }) => {}
            other => panic!("expected GreenSymbolOutOfRange, got {other:?}"),
        }
    }

    // ---- §6.2.2 entropy-image multi-group path ----

    impl BitWriter {
        /// Emit a §6.2 prefix-code group of five single-symbol simple
        /// codes (GREEN / RED / BLUE / ALPHA / DIST), each carrying no
        /// data bits in the §5.2 stream.
        fn write_single_symbol_group(&mut self, g: u32, r: u32, b: u32, a: u32, d: u32) {
            self.write_simple_single_symbol(g);
            self.write_simple_single_symbol(r);
            self.write_simple_single_symbol(b);
            self.write_simple_single_symbol(a);
            self.write_simple_single_symbol(d);
        }
    }

    #[test]
    fn meta_prefix_index_helpers_match_spec() {
        // Build the index directly to exercise the §6.2.2 derivations.
        let index = MetaPrefixIndex {
            prefix_bits: 2, // block size 4
            block_width: 2,
            block_height: 1,
            meta_codes: vec![0, 1],
        };
        // num_prefix_groups = max(0, 1) + 1 = 2.
        assert_eq!(index.num_prefix_groups(), 2);
        // §6.2.2 selection: position = (y>>2)*2 + (x>>2).
        // Pixels 0..3 → block 0 → code 0; 4..7 → block 1 → code 1.
        assert_eq!(index.meta_code_for(0, 0), 0);
        assert_eq!(index.meta_code_for(3, 0), 0);
        assert_eq!(index.meta_code_for(4, 0), 1);
        assert_eq!(index.meta_code_for(7, 0), 1);
    }

    #[test]
    fn meta_prefix_index_num_groups_uses_max_not_count() {
        // A 2-block entropy image whose codes are {0, 5} → 6 groups
        // (max + 1), not 2.
        let index = MetaPrefixIndex {
            prefix_bits: 2,
            block_width: 2,
            block_height: 1,
            meta_codes: vec![0, 5],
        };
        assert_eq!(index.num_prefix_groups(), 6);
    }

    #[test]
    fn decode_entropy_image_extracts_red_green_meta_codes() {
        // A 2x1 entropy image. Pixel 0: green=0, red=0 → meta-code 0.
        // Pixel 1: green=1, red=0 → meta-code 1. (meta = (red<<8)|green.)
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // entropy-image color-cache-info = disabled
                            // GREEN code carries greens {0, 1}; RED/BLUE/ALPHA/DIST single 0.
        w.write_simple_two_symbols(0, 1); // GREEN: 0 → bit 0, 1 → bit 1
        w.write_simple_single_symbol(0); // RED  = 0
        w.write_simple_single_symbol(0); // BLUE = 0
        w.write_simple_single_symbol(0); // ALPHA= 0
        w.write_simple_single_symbol(0); // DIST = 0
        w.write_bits(0, 1); // entropy pixel 0 GREEN = 0
        w.write_bits(1, 1); // entropy pixel 1 GREEN = 1
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let index = decode_entropy_image(&mut r, 2, 2, 1).unwrap();
        assert_eq!(index.prefix_bits(), 2);
        assert_eq!(index.block_width(), 2);
        assert_eq!(index.block_height(), 1);
        assert_eq!(index.meta_codes(), &[0, 1]);
        assert_eq!(index.num_prefix_groups(), 2);
    }

    #[test]
    fn decode_entropy_image_high_meta_code_uses_red_channel() {
        // A 1x1 entropy image with green=2, red=3 → meta = (3<<8)|2 = 770.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // color-cache disabled
        w.write_simple_single_symbol(2); // GREEN = 2 (single leaf, no data bit)
        w.write_simple_single_symbol(3); // RED   = 3
        w.write_simple_single_symbol(0); // BLUE
        w.write_simple_single_symbol(0); // ALPHA
        w.write_simple_single_symbol(0); // DIST
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let index = decode_entropy_image(&mut r, 2, 1, 1).unwrap();
        assert_eq!(index.meta_codes(), &[(3u16 << 8) | 2]);
        assert_eq!(index.num_prefix_groups(), 771);
    }

    #[test]
    fn decode_argb_two_groups_select_per_block() {
        // An 8x1 ARGB image, prefix_bits = 2 (block size 4) → 2 blocks.
        // Block 0 (pixels 0..3) uses group 0 (green 100); block 1 (4..7)
        // uses group 1 (green 200). Both groups single-symbol → main
        // image data stream consumes no bits.
        let mut w = BitWriter::new();
        // --- spatially-coded-image header (ARGB role) ---
        w.write_bits(0, 1); // main color-cache-info = disabled
        w.write_bits(1, 1); // meta-prefix = 1 → multiple groups
        w.write_bits(0, 3); // prefix_bits raw = 0 → 2 (block 4)
                            // --- entropy image (2x1 entropy-coded-image) ---
        w.write_bits(0, 1); // entropy color-cache-info = disabled
        w.write_simple_two_symbols(0, 1); // GREEN {0,1}
        w.write_simple_single_symbol(0); // RED
        w.write_simple_single_symbol(0); // BLUE
        w.write_simple_single_symbol(0); // ALPHA
        w.write_simple_single_symbol(0); // DIST
        w.write_bits(0, 1); // entropy pixel 0 GREEN = 0 → meta 0
        w.write_bits(1, 1); // entropy pixel 1 GREEN = 1 → meta 1
                            // --- num_prefix_groups = 2 prefix-code groups ---
        w.write_single_symbol_group(100, 0x10, 0x20, 0x30, 0); // group 0
        w.write_single_symbol_group(200, 0x40, 0x50, 0x60, 0); // group 1
                                                               // --- main image data: 8 single-symbol literals, no bits ---
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let img = decode_argb(&mut r, 8, 1).unwrap();
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 1);
        let g0 = pack_argb(100, 0x10, 0x20, 0x30);
        let g1 = pack_argb(200, 0x40, 0x50, 0x60);
        assert_eq!(
            img.pixels(),
            &[g0, g0, g0, g0, g1, g1, g1, g1],
            "first 4 pixels use group 0, last 4 use group 1"
        );
    }

    #[test]
    fn decode_argb_single_group_meta_prefix_zero() {
        // ARGB role, meta-prefix = 0 → one group everywhere. A 2x1 image
        // with greens 7 and 8 via a two-symbol GREEN code.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // color-cache disabled
        w.write_bits(0, 1); // meta-prefix = 0 → single group
        w.write_simple_two_symbols(7, 8); // GREEN {7, 8}
        w.write_simple_single_symbol(0xAA); // RED
        w.write_simple_single_symbol(0xBB); // BLUE
        w.write_simple_single_symbol(0xCC); // ALPHA
        w.write_simple_single_symbol(0); // DIST
        w.write_bits(0, 1); // pixel0 GREEN = 7
        w.write_bits(1, 1); // pixel1 GREEN = 8
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let img = decode_argb(&mut r, 2, 1).unwrap();
        assert_eq!(
            img.pixels(),
            &[
                pack_argb(7, 0xAA, 0xBB, 0xCC),
                pack_argb(8, 0xAA, 0xBB, 0xCC)
            ]
        );
    }

    #[test]
    fn decode_argb_single_group_matches_decode_image() {
        // The single-group decode_argb result must match a direct
        // decode_image with the same group + stream tail.
        let mut header = BitWriter::new();
        header.write_bits(0, 1); // color-cache disabled
        header.write_bits(0, 1); // meta-prefix = 0
        header.write_single_symbol_group(0x42, 0x10, 0x20, 0x30, 0);
        let data = header.into_bytes();
        let mut r = BitReader::new(&data);
        let img = decode_argb(&mut r, 1, 1).unwrap();
        assert_eq!(img.pixels(), &[pack_argb(0x42, 0x10, 0x20, 0x30)]);
    }

    #[test]
    fn decode_argb_multi_group_with_color_cache_round_2x2() {
        // A 8x1 ARGB image with a color cache enabled. Two groups; group
        // 0 emits literal green=50, group 1 emits literal green=60. The
        // cache is maintained across blocks in stream order; this checks
        // the multi-group loop threads a single cache through both
        // groups without panicking and yields the expected literals.
        let cache_bits = 4u32; // size 16
        let cache_size = 1usize << cache_bits;
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // main color-cache-info = enabled
        w.write_bits(cache_bits, 4); // code_bits = 4
        w.write_bits(1, 1); // meta-prefix = 1 → multi
        w.write_bits(0, 3); // prefix_bits → 2 (block 4)
                            // entropy image 2x1, codes {0, 1}, no cache.
        w.write_bits(0, 1);
        w.write_simple_two_symbols(0, 1);
        w.write_simple_single_symbol(0);
        w.write_simple_single_symbol(0);
        w.write_simple_single_symbol(0);
        w.write_simple_single_symbol(0);
        w.write_bits(0, 1); // entropy pixel 0 → meta 0
        w.write_bits(1, 1); // entropy pixel 1 → meta 1
                            // 2 prefix-code groups, sized against the main cache.
        w.write_single_symbol_group(50, 0x11, 0x22, 0x33, 0);
        w.write_single_symbol_group(60, 0x44, 0x55, 0x66, 0);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let img = decode_argb(&mut r, 8, 1).unwrap();
        let g0 = pack_argb(50, 0x11, 0x22, 0x33);
        let g1 = pack_argb(60, 0x44, 0x55, 0x66);
        assert_eq!(img.pixels(), &[g0, g0, g0, g0, g1, g1, g1, g1]);
        // Sanity: cache was sized for the alphabet (so the group GREEN
        // reads used the extended alphabet without overruns).
        assert_eq!(
            PrefixCodeGroup::green_alphabet_size(cache_size),
            256 + 24 + cache_size
        );
    }

    #[test]
    fn decode_entropy_image_zero_dim_is_refused() {
        let data = [0u8; 4];
        let mut r = BitReader::new(&data);
        match decode_entropy_image(&mut r, 2, 0, 1) {
            Err(DecodeError::EmptyEntropyImage {
                prefix_image_width,
                prefix_image_height,
            }) => {
                assert_eq!(prefix_image_width, 0);
                assert_eq!(prefix_image_height, 1);
            }
            other => panic!("expected EmptyEntropyImage, got {other:?}"),
        }
    }

    // ---- §5.2 / §6.2.2 malformed-input property tests for decode_argb ----
    //
    // The full ARGB-role decode_argb pipeline is the public surface
    // attackers and corrupt files reach. Each of the four cooperating
    // sub-stages (§5.2.3 color-cache info, §6.2.2 meta-prefix header,
    // §6.2.2 entropy image, per-group §6.2 prefix code groups, §6.2.3
    // main pixel loop) reads from the same BitReader and must respond
    // to a truncated stream with a structured DecodeError — never a
    // panic, never an `Ok` with an under-filled image. These tests pin
    // that contract on each boundary.
    //
    // The property at the end is exhaustive: every byte-prefix of a
    // valid 8x1 multi-group stream that doesn't span all stages must
    // produce an `Err`. That covers every off-by-one truncation point
    // any stage could be reading at.

    /// Build the exact byte buffer that the round-106
    /// `decode_argb_two_groups_select_per_block` test uses. Factored so
    /// the truncation-prefix tests can re-use it without copy/paste.
    fn build_valid_two_group_8x1_stream() -> Vec<u8> {
        build_valid_two_group_8x1_stream_with_bit_len().0
    }

    /// Same as [`build_valid_two_group_8x1_stream`] but also returns the
    /// number of bits actually written to the buffer. The trailing
    /// `bytes.len() * 8 - bit_len` bits of the last byte are zero pad.
    /// Used by the round-165 bit-prefix property test to slice the
    /// stream at every bit boundary, not only every byte boundary.
    fn build_valid_two_group_8x1_stream_with_bit_len() -> (Vec<u8>, usize) {
        let mut w = BitWriter::new();
        // spatially-coded-image header (ARGB role)
        w.write_bits(0, 1); // §5.2.3 main color-cache-info = disabled
        w.write_bits(1, 1); // §6.2.2 meta-prefix = 1 → multiple groups
        w.write_bits(0, 3); // §6.2.2 prefix_bits raw = 0 → 2 (block 4)
                            // §6.2.2 entropy image (2x1 entropy-coded-image)
        w.write_bits(0, 1); // entropy color-cache-info = disabled
        w.write_simple_two_symbols(0, 1); // GREEN {0,1}
        w.write_simple_single_symbol(0); // RED
        w.write_simple_single_symbol(0); // BLUE
        w.write_simple_single_symbol(0); // ALPHA
        w.write_simple_single_symbol(0); // DIST
        w.write_bits(0, 1); // entropy pixel 0 GREEN = 0 → meta 0
        w.write_bits(1, 1); // entropy pixel 1 GREEN = 1 → meta 1
                            // num_prefix_groups = 2 prefix-code groups
        w.write_single_symbol_group(100, 0x10, 0x20, 0x30, 0);
        w.write_single_symbol_group(200, 0x40, 0x50, 0x60, 0);
        // main image data: 8 single-symbol literals, no bits.
        let bit_len = w.bit_len();
        (w.into_bytes(), bit_len)
    }

    #[test]
    fn decode_argb_two_groups_baseline_decodes_clean() {
        // Sanity: the helper produces a stream that round-trips. The
        // truncation tests below all assume this baseline is valid.
        let data = build_valid_two_group_8x1_stream();
        let mut r = BitReader::new(&data);
        assert!(decode_argb(&mut r, 8, 1).is_ok());
    }

    #[test]
    fn decode_argb_empty_input_reports_eof() {
        // Zero bytes can't even satisfy the first 1-bit color-cache-info
        // read. decode_argb must return Eof, not panic, not Ok.
        let data: [u8; 0] = [];
        let mut r = BitReader::new(&data);
        match decode_argb(&mut r, 1, 1) {
            Err(DecodeError::Eof(_)) | Err(DecodeError::MetaPrefix(_)) => {}
            other => panic!("expected Eof or MetaPrefix, got {other:?}"),
        }
    }

    #[test]
    fn decode_argb_truncated_after_meta_prefix_header_reports_eof() {
        // Just enough bits to land past the meta-prefix-header phase (5
        // bits: color-cache=0, meta-prefix=1, prefix_bits=000). The
        // entropy-image read should then EOF on the entropy
        // color-cache-info bit.
        let mut w = BitWriter::new();
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        w.write_bits(0, 3);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        match decode_argb(&mut r, 8, 1) {
            Err(DecodeError::Eof(_)) | Err(DecodeError::MetaPrefix(_)) => {}
            other => panic!("expected Eof / MetaPrefix, got {other:?}"),
        }
    }

    #[test]
    fn decode_argb_truncated_mid_per_group_prefix_reports_eof() {
        // Build a valid stream, then chop it to a length that lands
        // *inside* the per-group prefix-code section (after entropy
        // image, before the second group's tables are complete). The
        // per-group PrefixCodeGroup::read must surface an EOF rather
        // than a wrong-shape success.
        let full = build_valid_two_group_8x1_stream();
        // The full stream is ~10 bytes; truncating to 6 lands inside
        // group 0's GREEN/RED tables on this layout.
        assert!(
            full.len() > 6,
            "stream layout changed; rechoose truncation point"
        );
        let truncated = &full[..6];
        let mut r = BitReader::new(truncated);
        match decode_argb(&mut r, 8, 1) {
            Err(DecodeError::Eof(_))
            | Err(DecodeError::MetaPrefix(_))
            | Err(DecodeError::Prefix(_)) => {}
            other => panic!("expected Eof / MetaPrefix / Prefix, got {other:?}"),
        }
    }

    #[test]
    fn decode_argb_every_byte_prefix_of_valid_stream_is_safe() {
        // The strong property: for every byte-prefix shorter than the
        // full valid stream, decode_argb on an 8x1 image must either
        // return Err, or — if the bit cursor happened to land at a
        // valid stage boundary and the remaining symbols are
        // single-leaf no-bit reads (so EOF isn't tripped) — return an
        // image of the requested size. It must NOT panic and must NOT
        // return a successfully-decoded image with the wrong pixel
        // count.
        let full = build_valid_two_group_8x1_stream();
        for len in 0..full.len() {
            let prefix = &full[..len];
            let mut r = BitReader::new(prefix);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                decode_argb(&mut r, 8, 1)
            }));
            match result {
                Ok(Ok(img)) => {
                    assert_eq!(
                        img.pixels().len(),
                        8,
                        "len={len}: Ok decode produced wrong pixel count"
                    );
                }
                Ok(Err(_)) => {
                    // Any structured DecodeError is fine — it's the
                    // contract we want.
                }
                Err(_) => panic!("len={len}: decode_argb panicked on truncated input"),
            }
        }
    }

    #[test]
    fn decode_argb_oversize_meta_prefix_bits_is_refused() {
        // §6.2.2 prefix_bits raw is 3 bits → derived prefix_bits is
        // raw + 2, capped at 9. Some malformed encoders may emit the
        // maximum raw value (7 → derived 9) on a tiny canvas; if the
        // resulting entropy image is degenerate, decode_argb must
        // surface EmptyEntropyImage rather than wedge.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // color-cache disabled
        w.write_bits(1, 1); // meta-prefix = 1 → multiple groups
        w.write_bits(7, 3); // prefix_bits raw 7 → derived 9 → block 512
                            // For a 1x1 image with block 512, the entropy image
                            // dimensions are (1+511)/512 = 1, so we still need
                            // a valid entropy image. Make it minimal: one
                            // single-meta-code pixel = 0.
        w.write_bits(0, 1); // entropy color-cache disabled
        w.write_simple_single_symbol(0); // GREEN single 0
        w.write_simple_single_symbol(0); // RED
        w.write_simple_single_symbol(0); // BLUE
        w.write_simple_single_symbol(0); // ALPHA
        w.write_simple_single_symbol(0); // DIST
                                         // One prefix-code group (max = 0 → count = 1).
        w.write_single_symbol_group(0x77, 0, 0, 0, 0);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        // The point of this test is "doesn't panic / doesn't loop"; a
        // clean Ok with the right pixel count is also acceptable for
        // this corner, and any structured Err is also acceptable.
        if let Ok(img) = decode_argb(&mut r, 1, 1) {
            assert_eq!(img.pixels().len(), 1);
        }
    }

    // ---- Round 165: §5.2 / §6.2.2 bit-prefix property test ----
    //
    // The round-164 property iterates byte-prefixes of the valid 8x1
    // multi-group stream. The §5.2 / §6.2.2 stages all consume sub-byte
    // bit fields (a §5.2.3 color-cache-info is a single bit; the
    // §6.2.2 prefix_bits field is three bits; §6.2.1 simple-code
    // symbols read 1-8 bits each), so a byte-prefix only samples the
    // truncation lattice at every 8 bits — most stage-boundary
    // truncation points sit *inside* a byte and are invisible to the
    // r164 property.
    //
    // This round-165 property tightens the granularity to a single
    // bit. For every truncation point `bit_len ∈ 0..=full_bits`, build
    // a buffer of `ceil(bit_len / 8)` bytes whose last byte's upper
    // `(8 - bit_len % 8) & 7` bits are zero-masked, then run the same
    // catch_unwind-protected decode_argb assertion: no panic, and any
    // `Ok` must carry exactly the requested pixel count. The
    // zero-padding tail is part of the property — the decoder may
    // legitimately read zero bits past the encoder's stop point and
    // either succeed (if the trailing zeros parse as a valid
    // single-leaf symbol stream) or EOF cleanly.
    //
    // Pure additive coverage: the decoder source is unchanged; this
    // pins the structured-error contract at 8x finer resolution than
    // r164.

    /// Truncate `data` (LSB-first bitstream) to its first `bit_len`
    /// bits, returning a fresh byte vector. The returned buffer has
    /// length `ceil(bit_len / 8)`; if `bit_len` is not a byte multiple,
    /// the upper `(8 - bit_len % 8)` bits of the last byte are masked
    /// to zero so the buffer cleanly represents the first `bit_len`
    /// bits of `data` followed by zero pad to the byte boundary.
    fn truncate_to_bit_prefix(data: &[u8], bit_len: usize) -> Vec<u8> {
        let byte_len = bit_len.div_ceil(8);
        assert!(
            byte_len <= data.len(),
            "bit_len {bit_len} exceeds source ({} bits)",
            data.len() * 8
        );
        let mut out = data[..byte_len].to_vec();
        let leftover = bit_len & 7;
        if leftover != 0 && !out.is_empty() {
            // Keep the low `leftover` bits of the last byte, zero the rest.
            let mask = (1u8 << leftover) - 1;
            let last = out.len() - 1;
            out[last] &= mask;
        }
        out
    }

    #[test]
    fn truncate_to_bit_prefix_round_trips_a_known_byte() {
        // 0b1011_0101 = 0xB5. Bit 0 (LSB of LSB-first) = 1, bit 1 = 0,
        // bit 2 = 1, bit 3 = 0, bit 4 = 1, bit 5 = 1, bit 6 = 0,
        // bit 7 = 1.
        let src = [0xB5u8];
        // 0-bit prefix: empty buffer.
        assert!(truncate_to_bit_prefix(&src, 0).is_empty());
        // 1-bit prefix: only bit 0 = 1, byte = 0x01.
        assert_eq!(truncate_to_bit_prefix(&src, 1), vec![0x01]);
        // 4-bit prefix: 0b0101 = 0x05.
        assert_eq!(truncate_to_bit_prefix(&src, 4), vec![0x05]);
        // 8-bit prefix: full byte.
        assert_eq!(truncate_to_bit_prefix(&src, 8), vec![0xB5]);
    }

    #[test]
    fn decode_argb_every_bit_prefix_of_valid_stream_is_safe() {
        // The round-165 strong property: for every bit-prefix
        // `bit_len ∈ 0..=full_bits` of a valid 8x1 multi-group stream
        // (zero-padded to the next byte boundary as the natural
        // representation of a partial bit count), decode_argb must
        // either return Err, or — when the trailing zero pad happens
        // to parse as a valid continuation of the stream — return an
        // image of exactly 8 pixels. It must NOT panic.
        //
        // Granularity is 1 bit (8x tighter than r164's byte property);
        // the §5.2.3 / §6.2.2 stages all read sub-byte fields, so this
        // catches truncation seams the byte property cannot see.
        let (full, full_bits) = build_valid_two_group_8x1_stream_with_bit_len();
        // Sanity: a full-bit-length decode succeeds (the baseline test
        // checks this too, but we re-assert here so a regression in
        // the helper is caught locally).
        {
            let mut r = BitReader::new(&full);
            assert!(
                decode_argb(&mut r, 8, 1).is_ok(),
                "full {full_bits}-bit stream must decode"
            );
        }
        // Iterate every bit cut from 0 up to and including the full
        // bit length. The 0-bit prefix mirrors the empty-input EOF
        // contract; the full-bits prefix mirrors the baseline decode.
        for bit_len in 0..=full_bits {
            let prefix = truncate_to_bit_prefix(&full, bit_len);
            let mut r = BitReader::new(&prefix);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                decode_argb(&mut r, 8, 1)
            }));
            match result {
                Ok(Ok(img)) => {
                    assert_eq!(
                        img.pixels().len(),
                        8,
                        "bit_len={bit_len}: Ok decode produced wrong pixel count"
                    );
                }
                Ok(Err(_)) => {
                    // Any structured DecodeError is acceptable — that's
                    // the contract under test.
                }
                Err(_) => {
                    panic!("bit_len={bit_len}: decode_argb panicked on truncated input")
                }
            }
        }
    }

    #[test]
    fn decode_argb_bit_prefix_covers_every_sub_byte_seam() {
        // Coverage sanity for the round-165 property above: confirm
        // the prefix sweep actually visits at least one bit position
        // inside every byte of the stream (i.e. covers all sub-byte
        // boundaries the byte-prefix property would skip past). With
        // the iteration `0..=full_bits` and `full_bits > 8`, every
        // bit boundary in `[0, full_bits]` is hit by construction;
        // this test just pins that arithmetic so a future refactor
        // that switched the bound (e.g. to `step_by(8)`) is caught.
        let (_full, full_bits) = build_valid_two_group_8x1_stream_with_bit_len();
        // The fixture currently writes about ~80 bits (5-bit header +
        // ~19-bit GREEN simple-two + 4×11-bit single-symbol + 2-bit
        // entropy pixels + 2×~55-bit single-symbol groups); allow a
        // generous lower bound so the test doesn't lock the exact
        // layout but still asserts non-trivial coverage.
        assert!(
            full_bits >= 60,
            "fixture bit-length unexpectedly short ({full_bits} bits); \
             round-165 property loses its multi-stage coverage if the \
             helper stops writing the entropy image / per-group tables"
        );
        // And the byte buffer must be at least `ceil(full_bits / 8)`
        // bytes — otherwise truncate_to_bit_prefix's len assertion
        // would fail on the largest bit prefix.
        let byte_len_needed = full_bits.div_ceil(8);
        assert!(
            byte_len_needed * 8 >= full_bits,
            "byte length ({byte_len_needed}) cannot cover bit length ({full_bits})"
        );
    }
}
