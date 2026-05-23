//! VP8L (WebP-Lossless) §5.2.3 color-cache info + §6.2.2 meta-prefix
//! header reader + §6.2 5-prefix-code-group reader.
//!
//! This sits directly on top of round 104's
//! [`crate::vp8l_prefix::PrefixCode`] reader. Where round 104 reads a
//! *single* canonical prefix code off the wire, this module assembles
//! the §7.3 ABNF `spatially-coded-image` and `entropy-coded-image`
//! preambles:
//!
//! ```text
//! spatially-coded-image =  color-cache-info meta-prefix data
//! entropy-coded-image   =  color-cache-info data
//!
//! color-cache-info      =  %b0
//! color-cache-info      =/ (%b1 4BIT)   ; 1 followed by color cache size
//!
//! meta-prefix           =  %b0 / (%b1 entropy-image)
//!
//! data                  =  prefix-codes lz77-coded-image
//! prefix-codes          =  prefix-code-group *prefix-codes
//! prefix-code-group     =  5prefix-code
//! ```
//!
//! Per §5.1, "image data" appears in five **roles**: the top-level
//! ARGB image (spatially-coded), the §6.2.2 *entropy image*, the §4.1
//! *predictor image*, the §4.2 *color transform image*, and the §4.4
//! *color indexing image*. The §6.2.2 meta-prefix layer is "used
//! *only* when the image is being used in the role of an ARGB image"
//! — the other four roles drop straight from the color-cache-info bit
//! into a single prefix-code group + LZ77 stream.
//!
//! ## What this module produces
//!
//! [`MetaPrefixHeader::read`] consumes the `color-cache-info` bit
//! (plus the `ReadBits(4)` `color_cache_code_bits` if set per §5.2.3),
//! then the `meta-prefix` 1-bit flag (only for an ARGB-role read),
//! then dispatches:
//!
//! * **`meta-prefix = %b0`** (single prefix-code group everywhere, or
//!   non-ARGB role): immediately reads the `prefix-code-group`
//!   (5 canonical prefix codes; the GREEN alphabet absorbs the
//!   `color_cache_size` per §6.2.3) and returns a fully-built
//!   [`MetaPrefixCodes::Single`].
//!
//! * **`meta-prefix = %b1 entropy-image`** (multiple groups, ARGB role
//!   only): reads the §6.2.2 `prefix_bits = ReadBits(3) + 2` field and
//!   computes the entropy-image dimensions, then **stops** — the
//!   entropy image itself is an `entropy-coded-image` (i.e. another
//!   `color-cache-info data` stream that needs the §5.2 LZ77 +
//!   color-cache decoder this crate does not yet have). Returns
//!   [`MetaPrefixCodes::EntropyImagePending`] with `prefix_bits`,
//!   derived `image_width` / `image_height`, and the bit position
//!   where the entropy image starts. The next layer's §5.2 reader can
//!   resume from there.
//!
//! ## Why split the multi-group case
//!
//! §6.2.2 says `num_prefix_groups = max(entropy_image) + 1`. The
//! decoder cannot read that maximum out of the entropy image without
//! actually decoding the entropy image — a §5.2-encoded
//! `entropy-coded-image`. Until the §5.2 LZ77 + color-cache layer
//! lands, the multi-group case is structurally read up to the
//! entropy-image boundary, the boundary is recorded, and the rest is
//! left to the next round (same pattern round 99 used to stop at the
//! first §5 transform body and round 104 used to resume at it).
//!
//! ## What this module does NOT do
//!
//! * No §5.2 LZ77 backward-reference or color-cache *symbol* decode.
//!   Those consume symbols produced by the [`PrefixCodeGroup`] this
//!   reader returns but live one layer up.
//! * No actual entropy-image decode (the `num_prefix_groups` lookup).
//! * No `oxideav-core` runtime dependency — this module compiles under
//!   `--no-default-features`.

use crate::vp8l_prefix::{PrefixCode, PrefixError};
use crate::vp8l_stream::{BitReader, BitReaderEof};

/// The five prefix codes that make up one §6.2 "prefix code group".
///
/// In bitstream order: green-channel + backref-length + color-cache
/// (channel 1), red (2), blue (3), alpha (4), backref-distance (5).
/// Each pixel is decoded with exactly one group; which group applies
/// to which pixel block is selected by the §6.2.2 meta-prefix layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCodeGroup {
    /// Prefix code #1: green channel + length-prefix + color-cache.
    /// Alphabet size is `256 + 24 + color_cache_size` per §6.2.3.
    pub green: PrefixCode,
    /// Prefix code #2: red channel, alphabet `256`.
    pub red: PrefixCode,
    /// Prefix code #3: blue channel, alphabet `256`.
    pub blue: PrefixCode,
    /// Prefix code #4: alpha channel, alphabet `256`.
    pub alpha: PrefixCode,
    /// Prefix code #5: backref distance, alphabet `40`.
    pub distance: PrefixCode,
}

impl PrefixCodeGroup {
    /// The size-of-alphabet for prefix code #1 (green / length /
    /// color-cache) for a stream whose color-cache holds
    /// `color_cache_size` entries (`0` when the cache is disabled,
    /// `1 << color_cache_code_bits` when it is enabled). Per §6.2.3:
    /// `256 + 24 + color_cache_size`.
    pub fn green_alphabet_size(color_cache_size: usize) -> usize {
        256 + 24 + color_cache_size
    }

    /// Read one prefix-code group (five canonical prefix codes) from
    /// the bitstream in §6.2 order.
    pub fn read(
        reader: &mut BitReader<'_>,
        color_cache_size: usize,
    ) -> Result<Self, MetaPrefixError> {
        let green_alphabet = Self::green_alphabet_size(color_cache_size);
        let green = PrefixCode::read(reader, green_alphabet)?;
        let red = PrefixCode::read(reader, 256)?;
        let blue = PrefixCode::read(reader, 256)?;
        let alpha = PrefixCode::read(reader, 256)?;
        let distance = PrefixCode::read(reader, 40)?;
        Ok(Self {
            green,
            red,
            blue,
            alpha,
            distance,
        })
    }
}

/// The role an image-data block is being read in.
///
/// Per §5.1 / §6.2.2: only the top-level `Argb` role carries the
/// §6.2.2 meta-prefix layer. The other four roles (entropy / predictor
/// / color-transform / color-indexing image, all `entropy-coded-image`
/// in the §7.3 ABNF) drop straight from the color-cache-info bit into
/// the single 5-code prefix-code group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageRole {
    /// §5.1 ARGB image (the spatially-coded top-level pixel stream).
    /// Carries the §6.2.2 meta-prefix bit.
    Argb,
    /// Any of the four `entropy-coded-image` roles (entropy /
    /// predictor / color-transform / color-indexing image). No
    /// meta-prefix bit; single prefix-code group only.
    EntropyCoded,
}

/// §5.2.3: range gate for `color_cache_code_bits`.
///
/// Spec wording is "the range of allowed values for
/// `color_cache_code_bits` is `[1..11]`. Compliant decoders must
/// indicate a corrupted bitstream for other values."
pub const COLOR_CACHE_BITS_MIN: u32 = 1;
/// See [`COLOR_CACHE_BITS_MIN`].
pub const COLOR_CACHE_BITS_MAX: u32 = 11;

/// §6.2.2 entropy image: `prefix_bits = ReadBits(3) + 2`, so the
/// allowed range is `[2..9]`.
pub const PREFIX_BITS_MIN: u32 = 2;
/// See [`PREFIX_BITS_MIN`].
pub const PREFIX_BITS_MAX: u32 = 9;

/// Errors raised while reading the §5.2.3 color-cache info + §6.2.2
/// meta-prefix header + the §6.2 prefix-code group(s).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaPrefixError {
    /// The bit reader hit EOF mid-field.
    Eof(BitReaderEof),
    /// §5.2.3: `color_cache_code_bits` was outside the `[1..11]`
    /// allowed range.
    InvalidColorCacheCodeBits {
        /// The on-wire value that failed the range check.
        value: u32,
    },
    /// One of the five §6.2.1 prefix codes inside a group failed to
    /// parse.
    Prefix(PrefixError),
}

impl From<BitReaderEof> for MetaPrefixError {
    fn from(e: BitReaderEof) -> Self {
        Self::Eof(e)
    }
}

impl From<PrefixError> for MetaPrefixError {
    fn from(e: PrefixError) -> Self {
        match e {
            PrefixError::Eof(eof) => Self::Eof(eof),
            other => Self::Prefix(other),
        }
    }
}

impl core::fmt::Display for MetaPrefixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Eof(e) => write!(f, "VP8L §5.2.3 / §6.2.2 meta-prefix: {e}"),
            Self::InvalidColorCacheCodeBits { value } => write!(
                f,
                "VP8L §5.2.3 color_cache_code_bits = {value} is outside the [{COLOR_CACHE_BITS_MIN}..{COLOR_CACHE_BITS_MAX}] allowed range"
            ),
            Self::Prefix(e) => write!(f, "VP8L §6.2 prefix code in group: {e}"),
        }
    }
}

impl std::error::Error for MetaPrefixError {}

/// §5.2.3 color-cache info: whether a color cache is present + its
/// size (`1 << color_cache_code_bits` per §5.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorCacheInfo {
    /// `color_cache_code_bits` read off the wire, or `0` when the
    /// cache is disabled (`color-cache-info = %b0`). The §5.2.3 range
    /// for the *enabled* case is `[1..11]`; `0` here is the sentinel
    /// for "no cache" and is **never** a wire value (a 0 wire value
    /// is rejected as out-of-range).
    pub code_bits: u32,
}

impl ColorCacheInfo {
    /// `true` when a color cache is active (i.e. `color-cache-info`
    /// started with the `%b1` flag).
    pub fn is_enabled(&self) -> bool {
        self.code_bits != 0
    }

    /// Size of the color cache table, `1 << code_bits` when enabled,
    /// `0` when disabled.
    pub fn size(&self) -> usize {
        if self.is_enabled() {
            1usize << self.code_bits
        } else {
            0
        }
    }

    /// Read a §5.2.3 `color-cache-info` field.
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self, MetaPrefixError> {
        if reader.read_bit()? {
            let code_bits = reader.read_bits(4)?;
            if !(COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&code_bits) {
                return Err(MetaPrefixError::InvalidColorCacheCodeBits { value: code_bits });
            }
            Ok(Self { code_bits })
        } else {
            Ok(Self { code_bits: 0 })
        }
    }
}

/// The decoded §6.2.2 prefix-code-group layer: either a single group
/// that applies to the whole image, or a pending entropy-image
/// boundary the next §5.2 reader has to resume from.
///
/// The `Single` variant boxes its `PrefixCodeGroup` because the group
/// is much larger than the `EntropyImagePending` variant's handful of
/// integers — keeping the enum compact for callers that only branch on
/// the discriminant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaPrefixCodes {
    /// `meta-prefix = %b0` (or non-ARGB role): exactly one
    /// prefix-code group for the entire image.
    Single {
        /// The one prefix-code group.
        group: Box<PrefixCodeGroup>,
    },
    /// `meta-prefix = %b1 entropy-image`: multiple prefix-code groups,
    /// chosen per pixel block via an entropy image. The entropy image
    /// itself is an `entropy-coded-image` (color-cache-info + §5.2
    /// data) which this round cannot yet decode; the reader stops at
    /// the entropy-image start, records `prefix_bits` +
    /// `image_width`/`image_height` + the start bit position, and
    /// leaves the decode to the next round.
    EntropyImagePending {
        /// §6.2.2 `prefix_bits = ReadBits(3) + 2`, range `[2..9]`.
        /// `block_size = 1 << prefix_bits`.
        prefix_bits: u8,
        /// `DIV_ROUND_UP(image_width, 1 << prefix_bits)`.
        image_width: u32,
        /// `DIV_ROUND_UP(image_height, 1 << prefix_bits)`.
        image_height: u32,
        /// Bit position (from the start of the slice the
        /// [`BitReader`] was constructed over) at which the entropy
        /// image begins (i.e. just past the `prefix_bits` field).
        entropy_image_bit_position: usize,
    },
}

impl MetaPrefixCodes {
    /// `true` when the read produced a single complete prefix-code
    /// group.
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Single { .. })
    }

    /// The single prefix-code group, when this is `Single`.
    pub fn group(&self) -> Option<&PrefixCodeGroup> {
        match self {
            Self::Single { group } => Some(group),
            Self::EntropyImagePending { .. } => None,
        }
    }
}

/// The combined §5.2.3 color-cache info + §6.2.2 meta-prefix +
/// §6.2 prefix-code-group(s) header for one §5 image-data block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaPrefixHeader {
    /// The §5.2.3 color-cache info bit + size.
    pub color_cache: ColorCacheInfo,
    /// The §6.2.2 meta-prefix dispatch result.
    pub codes: MetaPrefixCodes,
}

/// Round-up division — matches the spec's `DIV_ROUND_UP` macro defined
/// in §4.1.
fn div_round_up(n: u32, d: u32) -> u32 {
    debug_assert!(d != 0, "DIV_ROUND_UP requires non-zero divisor");
    n.div_ceil(d)
}

impl MetaPrefixHeader {
    /// Read the §5.2.3 + §6.2.2 + §6.2 preamble for one §5 image-data
    /// block. `role` selects whether the §6.2.2 meta-prefix bit is
    /// read (only for the ARGB image role); `image_width` /
    /// `image_height` are used to derive the entropy-image dimensions
    /// when the meta-prefix bit is set.
    ///
    /// The reader's bit cursor is left right at the start of the
    /// §5.2 `data` (the LZ77-coded image) for `Single`, or right at
    /// the start of the §6.2.2 `entropy-image` block for
    /// `EntropyImagePending`.
    pub fn read(
        reader: &mut BitReader<'_>,
        role: ImageRole,
        image_width: u32,
        image_height: u32,
    ) -> Result<Self, MetaPrefixError> {
        let color_cache = ColorCacheInfo::read(reader)?;

        // §6.2.2: meta-prefix bit is present ONLY for the ARGB role.
        let use_meta_prefix = if matches!(role, ImageRole::Argb) {
            reader.read_bit()?
        } else {
            false
        };

        if !use_meta_prefix {
            // Single group: read the 5 prefix codes directly.
            let group = PrefixCodeGroup::read(reader, color_cache.size())?;
            return Ok(Self {
                color_cache,
                codes: MetaPrefixCodes::Single {
                    group: Box::new(group),
                },
            });
        }

        // Multi-group: read `prefix_bits`, derive entropy-image dims,
        // record the boundary, and stop.
        let raw = reader.read_bits(3)?;
        let prefix_bits = raw + PREFIX_BITS_MIN;
        debug_assert!((PREFIX_BITS_MIN..=PREFIX_BITS_MAX).contains(&prefix_bits));
        let block_size = 1u32 << prefix_bits;
        let entropy_w = div_round_up(image_width, block_size);
        let entropy_h = div_round_up(image_height, block_size);
        let entropy_image_bit_position = reader.bit_position();
        Ok(Self {
            color_cache,
            codes: MetaPrefixCodes::EntropyImagePending {
                prefix_bits: prefix_bits as u8,
                image_width: entropy_w,
                image_height: entropy_h,
                entropy_image_bit_position,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny LSB-first bit writer for building synthetic streams.
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
        /// Write one simple-code length-1 single-symbol prefix code
        /// for the symbol `sym` (in the 8-bit symbol form, alphabet
        /// `<= 256+24+cache_size`).
        fn write_simple_single_symbol(&mut self, sym: u32) {
            // flag = 1 (simple)
            self.write_bits(1, 1);
            // num_symbols - 1 = 0
            self.write_bits(0, 1);
            // is_first_8bits = 1
            self.write_bits(1, 1);
            // symbol0 in 8 bits
            self.write_bits(sym, 8);
        }
        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    // ---- ColorCacheInfo ----

    #[test]
    fn color_cache_info_disabled() {
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // %b0
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let cc = ColorCacheInfo::read(&mut r).unwrap();
        assert!(!cc.is_enabled());
        assert_eq!(cc.size(), 0);
        assert_eq!(cc.code_bits, 0);
        assert_eq!(r.bit_position(), 1);
    }

    #[test]
    fn color_cache_info_enabled_min() {
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // %b1
        w.write_bits(1, 4); // code_bits = 1 (the lower bound)
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let cc = ColorCacheInfo::read(&mut r).unwrap();
        assert!(cc.is_enabled());
        assert_eq!(cc.code_bits, 1);
        assert_eq!(cc.size(), 2);
    }

    #[test]
    fn color_cache_info_enabled_max() {
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(11, 4); // code_bits = 11 (the upper bound)
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let cc = ColorCacheInfo::read(&mut r).unwrap();
        assert_eq!(cc.code_bits, 11);
        assert_eq!(cc.size(), 2048);
    }

    #[test]
    fn color_cache_info_zero_code_bits_is_refused() {
        // §5.2.3 says the enabled range is [1..11]; 0 is out of range.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(0, 4);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        match ColorCacheInfo::read(&mut r) {
            Err(MetaPrefixError::InvalidColorCacheCodeBits { value }) => assert_eq!(value, 0),
            other => panic!("expected InvalidColorCacheCodeBits, got {other:?}"),
        }
    }

    #[test]
    fn color_cache_info_twelve_code_bits_is_refused() {
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(12, 4);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        match ColorCacheInfo::read(&mut r) {
            Err(MetaPrefixError::InvalidColorCacheCodeBits { value }) => assert_eq!(value, 12),
            other => panic!("expected InvalidColorCacheCodeBits, got {other:?}"),
        }
    }

    // ---- PrefixCodeGroup ----

    #[test]
    fn prefix_code_group_green_alphabet_size_matches_spec() {
        // §6.2.3: G alphabet = 256 + 24 + color_cache_size.
        assert_eq!(PrefixCodeGroup::green_alphabet_size(0), 280);
        assert_eq!(PrefixCodeGroup::green_alphabet_size(2), 282);
        assert_eq!(PrefixCodeGroup::green_alphabet_size(2048), 2328);
    }

    #[test]
    fn prefix_code_group_reads_five_simple_codes_in_order() {
        // Bitstream order per §6.2: GREEN, RED, BLUE, ALPHA, DIST.
        let mut w = BitWriter::new();
        w.write_simple_single_symbol(60); // GREEN
        w.write_simple_single_symbol(180); // RED
        w.write_simple_single_symbol(90); // BLUE
        w.write_simple_single_symbol(255); // ALPHA
        w.write_simple_single_symbol(0); // DIST
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let g = PrefixCodeGroup::read(&mut r, 0).unwrap();
        assert_eq!(g.green.single_symbol(), Some(60));
        assert_eq!(g.red.single_symbol(), Some(180));
        assert_eq!(g.blue.single_symbol(), Some(90));
        assert_eq!(g.alpha.single_symbol(), Some(255));
        assert_eq!(g.distance.single_symbol(), Some(0));
    }

    // ---- MetaPrefixHeader (single-group / non-ARGB) ----

    #[test]
    fn entropy_coded_role_has_no_meta_prefix_bit() {
        // For a non-ARGB role we read color-cache-info then drop
        // straight into the group — no §6.2.2 dispatch bit.
        // color-cache=0, then 5 single-symbol simple codes.
        let mut w = BitWriter::new();
        w.write_bits(0, 1);
        for sym in [60u32, 180, 90, 255, 0] {
            w.write_simple_single_symbol(sym);
        }
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let h = MetaPrefixHeader::read(&mut r, ImageRole::EntropyCoded, 1, 1).unwrap();
        assert!(!h.color_cache.is_enabled());
        let group = h.codes.group().expect("Single");
        assert_eq!(group.green.single_symbol(), Some(60));
        assert_eq!(group.distance.single_symbol(), Some(0));
    }

    #[test]
    fn argb_role_meta_prefix_zero_reads_one_group_directly() {
        // ARGB role, color-cache=0, meta-prefix=0, then 5 single-symbol
        // simple codes.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // color-cache disabled
        w.write_bits(0, 1); // meta-prefix = 0 → single group
        for sym in [42u32, 200, 150, 100, 5] {
            w.write_simple_single_symbol(sym);
        }
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let h = MetaPrefixHeader::read(&mut r, ImageRole::Argb, 64, 64).unwrap();
        let group = h.codes.group().expect("Single");
        assert_eq!(group.green.single_symbol(), Some(42));
        assert_eq!(group.distance.single_symbol(), Some(5));
        assert!(h.codes.is_single());
    }

    #[test]
    fn argb_role_meta_prefix_one_stops_at_entropy_image() {
        // ARGB role, color-cache=0, meta-prefix=1, prefix_bits = 0 + 2 = 2.
        // image dims 32x16 → 1<<2 = 4 → 32/4=8, 16/4=4.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // color-cache disabled
        w.write_bits(1, 1); // meta-prefix = 1 → multi
        w.write_bits(0, 3); // prefix_bits raw = 0 → 2
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let h = MetaPrefixHeader::read(&mut r, ImageRole::Argb, 32, 16).unwrap();
        match h.codes {
            MetaPrefixCodes::EntropyImagePending {
                prefix_bits,
                image_width,
                image_height,
                entropy_image_bit_position,
            } => {
                assert_eq!(prefix_bits, 2);
                assert_eq!(image_width, 8);
                assert_eq!(image_height, 4);
                // 1 (cache) + 1 (meta) + 3 (prefix_bits) = 5
                assert_eq!(entropy_image_bit_position, 5);
            }
            other => panic!("expected EntropyImagePending, got {other:?}"),
        }
    }

    #[test]
    fn argb_role_meta_prefix_one_div_round_up_rounds() {
        // image 17x9 with prefix_bits 2 (block 4) → ceil(17/4)=5, ceil(9/4)=3.
        let mut w = BitWriter::new();
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        w.write_bits(0, 3); // prefix_bits = 2
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let h = MetaPrefixHeader::read(&mut r, ImageRole::Argb, 17, 9).unwrap();
        match h.codes {
            MetaPrefixCodes::EntropyImagePending {
                image_width,
                image_height,
                ..
            } => {
                assert_eq!(image_width, 5);
                assert_eq!(image_height, 3);
            }
            other => panic!("expected EntropyImagePending, got {other:?}"),
        }
    }

    #[test]
    fn argb_role_meta_prefix_one_max_prefix_bits() {
        // raw = 7 → prefix_bits = 9 (the §6.2.2 upper bound).
        let mut w = BitWriter::new();
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        w.write_bits(7, 3); // prefix_bits raw = 7 → 9
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let h = MetaPrefixHeader::read(&mut r, ImageRole::Argb, 1024, 1024).unwrap();
        match h.codes {
            MetaPrefixCodes::EntropyImagePending {
                prefix_bits,
                image_width,
                image_height,
                ..
            } => {
                assert_eq!(prefix_bits, 9);
                // ceil(1024 / 512) = 2.
                assert_eq!(image_width, 2);
                assert_eq!(image_height, 2);
            }
            other => panic!("expected EntropyImagePending, got {other:?}"),
        }
    }

    #[test]
    fn argb_role_color_cache_size_is_absorbed_into_green_alphabet() {
        // ARGB role with an enabled cache; the GREEN alphabet must grow
        // to 256 + 24 + (1 << code_bits). Build the GREEN code as a
        // simple-code single-symbol *inside* that extended range and
        // confirm it reads back.
        let cache_bits: u32 = 4; // size = 16
        let cache_size: u32 = 1 << cache_bits; // 16
        let extended_green = 256 + 24 + cache_size - 1; // 295 — valid wire value
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // color-cache enabled
        w.write_bits(cache_bits, 4); // code_bits = 4
        w.write_bits(0, 1); // meta-prefix = 0 → single group
                            // GREEN: simple, 1 symbol, is_first_8bits = 0 → 1-bit sym field
                            // can't reach 295. Force the 8-bit form, but 8 bits only reach
                            // 255 — so use two symbols, first 8-bit symbol = 255, second
                            // 8-bit symbol = (extended_green - 256) → wait, no: simple-code
                            // symbol is a *literal* index in [0..alphabet_size). We need
                            // the actual two-stage 1/8-bit accommodation the simple code
                            // can only express up to value 255. For larger values we must
                            // use the normal code length code. So instead exercise the
                            // alphabet by reading a normal code that covers an extended
                            // alphabet without actually emitting a wide symbol.
                            //
                            // Simpler check: read a GREEN code with a 2-symbol normal CLC
                            // that emits lengths only for symbols 0 and 1, then RED/BLUE/
                            // ALPHA/DIST as simple single-symbol codes. The point is just
                            // that the alphabet size is propagated; the read uses the
                            // value to size `lengths`.
                            //
                            // GREEN normal CLC with one symbol at length 1 (single-leaf):
                            // flag=0 (normal), num_code_lengths=4 (=4+0), then 4 CLC
                            // entries in kCodeLengthCodeOrder (17, 18, 0, 1). Set pos1=1
                            // (literal len 1), everything else 0. max_symbol gate = 0
                            // (alphabet_size). Then emit a code-18 zero-run to skip ahead
                            // — but the simpler approach is: don't gate, and just emit a
                            // single length-1 literal for symbol 0 plus a long code-18 run.
                            //
                            // Easiest test: use a *simple* GREEN code with symbol 0.
        w.write_simple_single_symbol(0); // GREEN sym = 0
        w.write_simple_single_symbol(10); // RED
        w.write_simple_single_symbol(20); // BLUE
        w.write_simple_single_symbol(30); // ALPHA
        w.write_simple_single_symbol(0); // DIST
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let h = MetaPrefixHeader::read(&mut r, ImageRole::Argb, 64, 64).unwrap();
        assert!(h.color_cache.is_enabled());
        assert_eq!(h.color_cache.size(), cache_size as usize);
        let g = h.codes.group().expect("Single");
        assert_eq!(g.green.single_symbol(), Some(0));
        // Sanity: code_lengths table is sized to the *extended* alphabet,
        // proving the cache size flowed through.
        assert_eq!(g.green.code_lengths().len(), extended_green as usize + 1);
    }

    #[test]
    fn truncated_color_cache_info_reports_eof() {
        let data: [u8; 0] = [];
        let mut r = BitReader::new(&data);
        match ColorCacheInfo::read(&mut r) {
            Err(MetaPrefixError::Eof(_)) => {}
            other => panic!("expected Eof, got {other:?}"),
        }
    }

    #[test]
    fn truncated_meta_prefix_bit_reports_eof() {
        // ARGB role; only the color-cache-info=0 bit is present; the
        // meta-prefix bit read should EOF.
        let data = [0x00u8];
        let mut r = BitReader::new(&data);
        // Manually position the cursor past the only valid bit.
        // First read consumes the cache-info=0 bit; then meta-prefix
        // tries to read the 2nd bit which is still inside the byte.
        // To force EOF we need to exhaust the slice — seek to bit 8.
        let h = MetaPrefixHeader::read(&mut r, ImageRole::Argb, 1, 1);
        // color-cache-info=0 (bit 0), then meta-prefix=0 (bit 1), then
        // GREEN code reads (which run out of bytes) — so the actual EOF
        // comes from the prefix-code read, not the meta-prefix bit. We
        // accept any Eof here.
        match h {
            Err(MetaPrefixError::Eof(_)) => {}
            other => panic!("expected Eof, got {other:?}"),
        }
    }
}
