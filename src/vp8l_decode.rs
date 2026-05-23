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
//! * No §6.2.2 entropy-image / multi-group selection — this loop runs
//!   a single [`PrefixCodeGroup`] over the whole image. The
//!   multi-group path (one group per pixel block selected by an
//!   entropy image) reuses the same per-symbol primitives but is a
//!   future round.
//! * No `oxideav-core` runtime dependency — this module compiles under
//!   `--no-default-features`.

use crate::meta_prefix::PrefixCodeGroup;
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
                // §5.2.2: copy L pixels from `position - dist`. The copy
                // is byte-for-byte scan-line, and overlapping copies
                // (dist < length) repeat the just-copied pixels, which
                // is the standard LZ77 behaviour.
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
}
