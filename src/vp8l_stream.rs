//! VP8L (WebP-Lossless) bit-reader + §4 transform-list header reader.
//!
//! This is the section that sits directly on top of the round-7 typed
//! [`crate::vp8l_chunk::WebpLosslessChunk`] image-header peek. Where the
//! round-7 handle decodes only the fixed 5-byte §3.4 / §7.1
//! `image-header` (signature + 14-bit dims + alpha hint + version), this
//! module provides:
//!
//! 1. A [`BitReader`] implementing the WebP-Lossless §2 `ReadBits(n)`
//!    primitive: bytes are consumed in stream order and bits of each
//!    byte are read **least-significant-bit-first**. When several bits
//!    are read at once, the integer is assembled so that the first bit
//!    read is bit 0 of the result (the "most significant bits of the
//!    returned integer are also the most significant bits of the
//!    original data" rule from the spec).
//! 2. A [`TransformList`] reader implementing the §4 transform-presence
//!    loop:
//!
//!    ```text
//!    while (ReadBits(1)) {              // Transform present.
//!      enum TransformType t = ReadBits(2);
//!      ...                              // transform data
//!    }
//!    ```
//!
//!    For each present transform this reader decodes the *fixed-size*
//!    leading fields that are pure `ReadBits` and do **not** require the
//!    §5 entropy decoder:
//!
//!    * `PREDICTOR_TRANSFORM` (0) / `COLOR_TRANSFORM` (1): the §4.1 /
//!      §4.2 `size_bits = ReadBits(3) + 2` block-size field.
//!    * `SUBTRACT_GREEN_TRANSFORM` (2): no data (§4.3).
//!    * `COLOR_INDEXING_TRANSFORM` (3): the §4.4
//!      `color_table_size = ReadBits(8) + 1` field plus the derived
//!      pixel-bundling `width_bits`.
//!
//! ## Where the reader stops
//!
//! The §4 transform *data* for the predictor / color / color-indexing
//! transforms is a sub-resolution image (or color table) encoded with
//! the §5 entropy machinery (prefix codes, LZ77, color cache). That
//! decoder does not exist in this crate yet, so [`TransformList::read`]
//! reads each transform's leading fixed fields and then **stops** at
//! the byte/bit boundary where the first §5-encoded body begins. The
//! returned [`TransformList`] records that boundary
//! ([`TransformList::body_bit_position`]) so the next round's §5 reader
//! can pick up exactly where this one left off.
//!
//! Concretely, for a transform that carries an entropy-coded body
//! (`PREDICTOR`, `COLOR`, `COLOR_INDEXING`) this reader records the
//! transform's type + leading fields and then halts the loop, because
//! it cannot skip the §5 body to look for the next transform-presence
//! bit. A `SUBTRACT_GREEN` transform has no body, so the loop continues
//! past it. This matches every fixture in `docs/image/webp/fixtures/`:
//! e.g. `lossless-32x32-rgba` is `SUBTRACT_GREEN` (no body) followed by
//! `PREDICTOR` (entropy body) — the reader returns both, stopping at
//! the predictor's sub-resolution image.
//!
//! ## What this module does NOT do
//!
//! * No §5 entropy decode (prefix codes / Huffman code groups / LZ77 /
//!   color cache). That is the next section.
//! * No actual inverse-transform application to pixels.
//! * No `oxideav-core` runtime dependency — this module compiles under
//!   `--no-default-features`.

/// Least-significant-bit-first bit reader over a VP8L byte stream.
///
/// Implements the WebP-Lossless §2 `ReadBits(n)` contract: bytes are
/// read in natural stream order, bits within a byte are read
/// least-significant-bit-first, and a multi-bit read returns an integer
/// whose bit 0 is the first bit read off the wire.
///
/// The reader is positioned in **bits** from the start of the slice it
/// was constructed over. Construct it over the *whole* VP8L chunk
/// payload and seek past the image-header (see
/// [`BitReader::new_after_image_header`]) to line up with the §4
/// transform list.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit cursor from the start of `data`.
    bit_pos: usize,
}

/// Error raised when the bit reader runs past the end of its slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitReaderEof {
    /// Bit position the reader was at when the read was attempted.
    pub bit_pos: usize,
    /// Number of bits the failing read wanted.
    pub wanted: usize,
    /// Number of bits available from `bit_pos` to the end of the slice.
    pub available: usize,
}

impl core::fmt::Display for BitReaderEof {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "VP8L bit-reader EOF at bit {}: wanted {} bit(s), {} available",
            self.bit_pos, self.wanted, self.available
        )
    }
}

impl std::error::Error for BitReaderEof {}

impl<'a> BitReader<'a> {
    /// Create a bit reader positioned at bit 0 of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Create a bit reader over a full VP8L chunk payload, positioned
    /// just past the §3.4 / §7.1 5-byte `image-header` so the first
    /// [`read_bits`](Self::read_bits) lines up with the §4 transform
    /// list.
    ///
    /// The image-header occupies the first
    /// [`crate::vp8l_chunk::VP8L_IMAGE_HEADER_LEN`] bytes (40 bits) of
    /// the payload.
    pub fn new_after_image_header(payload: &'a [u8]) -> Self {
        Self {
            data: payload,
            bit_pos: crate::vp8l_chunk::VP8L_IMAGE_HEADER_LEN * 8,
        }
    }

    /// The current bit cursor, counted from the start of the slice the
    /// reader was constructed over.
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Reposition the bit cursor to an absolute bit offset from the
    /// start of the slice.
    ///
    /// Used to resume reading at a boundary recorded earlier — e.g.
    /// [`TransformList::body_bit_position`], where the §5 entropy-coded
    /// body of a transform (or the main image stream) begins. The offset
    /// is clamped to the end of the slice.
    pub fn seek_to_bit(&mut self, bit_pos: usize) {
        self.bit_pos = bit_pos.min(self.data.len() * 8);
    }

    /// Bits remaining from the cursor to the end of the slice.
    pub fn bits_remaining(&self) -> usize {
        self.data.len() * 8 - self.bit_pos
    }

    /// Read `n` bits (0 ≤ `n` ≤ 32) least-significant-bit-first and
    /// return them as a `u32` whose bit 0 is the first bit read.
    ///
    /// Reading 0 bits is a no-op that returns 0 (mirrors the spec's
    /// `ReadBits(0)` corner used by, e.g., a 0-bit color cache).
    pub fn read_bits(&mut self, n: usize) -> Result<u32, BitReaderEof> {
        debug_assert!(n <= 32, "read_bits supports up to 32 bits");
        if n == 0 {
            return Ok(0);
        }
        let available = self.bits_remaining();
        if n > available {
            return Err(BitReaderEof {
                bit_pos: self.bit_pos,
                wanted: n,
                available,
            });
        }
        let mut value: u32 = 0;
        for i in 0..n {
            let byte = self.data[self.bit_pos >> 3];
            let bit_in_byte = self.bit_pos & 7;
            let bit = (byte >> bit_in_byte) & 1;
            // First bit read becomes bit 0 of the result.
            value |= (bit as u32) << i;
            self.bit_pos += 1;
        }
        Ok(value)
    }

    /// Read a single bit as a `bool` (LSB-first), used by the §4
    /// transform-presence loop.
    pub fn read_bit(&mut self) -> Result<bool, BitReaderEof> {
        Ok(self.read_bits(1)? != 0)
    }
}

/// The four §4 transform types. The discriminant matches the on-wire
/// 2-bit `TransformType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    /// §4.1 predictor (spatial) transform.
    Predictor = 0,
    /// §4.2 color transform (the fixture traces label it `CROSS_COLOR`).
    Color = 1,
    /// §4.3 subtract-green transform — carries no data.
    SubtractGreen = 2,
    /// §4.4 color-indexing (palette) transform.
    ColorIndexing = 3,
}

impl TransformType {
    /// Map the on-wire 2-bit value onto a [`TransformType`]. The value
    /// is always in `0..=3` (it comes from a `ReadBits(2)`), so this is
    /// total.
    pub fn from_bits(bits: u32) -> Self {
        match bits & 0x3 {
            0 => Self::Predictor,
            1 => Self::Color,
            2 => Self::SubtractGreen,
            _ => Self::ColorIndexing,
        }
    }
}

/// One parsed §4 transform entry: the type plus its decoded leading
/// fixed-size fields. The entropy-coded body (sub-resolution image or
/// color table) is **not** decoded here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// §4.1 predictor transform. `size_bits = ReadBits(3) + 2`; the
    /// per-block prediction modes live in a sub-resolution image that
    /// follows (§5-encoded, not decoded here).
    Predictor {
        /// `ReadBits(3) + 2`, range `2..=9`. `block_width = block_height
        /// = 1 << size_bits`.
        size_bits: u8,
    },
    /// §4.2 color transform. Same `size_bits` layout as the predictor;
    /// the per-block `ColorTransformElement`s live in a sub-resolution
    /// image that follows (§5-encoded, not decoded here).
    Color {
        /// `ReadBits(3) + 2`, range `2..=9`.
        size_bits: u8,
    },
    /// §4.3 subtract-green transform — no data.
    SubtractGreen,
    /// §4.4 color-indexing transform. `color_table_size = ReadBits(8) +
    /// 1`; the color table itself is a width-`color_table_size`,
    /// height-1 §5-encoded image that follows (not decoded here).
    ColorIndexing {
        /// `ReadBits(8) + 1`, range `1..=256`.
        color_table_size: u16,
        /// §4.4 pixel-bundling width: 0 (≥17 colors), 1 (≤16), 2 (≤4),
        /// 3 (≤2). Derived from `color_table_size`.
        width_bits: u8,
    },
}

impl Transform {
    /// The transform type tag for this entry.
    pub fn transform_type(&self) -> TransformType {
        match self {
            Self::Predictor { .. } => TransformType::Predictor,
            Self::Color { .. } => TransformType::Color,
            Self::SubtractGreen => TransformType::SubtractGreen,
            Self::ColorIndexing { .. } => TransformType::ColorIndexing,
        }
    }

    /// Whether this transform is followed by a §5 entropy-coded body
    /// (a sub-resolution image or a color table). `SubtractGreen` is
    /// the only transform with no body.
    pub fn has_entropy_body(&self) -> bool {
        !matches!(self, Self::SubtractGreen)
    }
}

/// §4.4: derive the pixel-bundling `width_bits` from a color table
/// size, per the spec's threshold table.
fn color_indexing_width_bits(color_table_size: u16) -> u8 {
    if color_table_size <= 2 {
        3
    } else if color_table_size <= 4 {
        2
    } else if color_table_size <= 16 {
        1
    } else {
        0
    }
}

/// Errors raised while reading the §4 transform list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformListError {
    /// The bit reader hit EOF mid-field.
    Eof(BitReaderEof),
    /// §4 says "each transform is allowed to be used only once". A
    /// transform type appeared twice in the list.
    DuplicateTransform {
        /// The transform type that was repeated.
        transform_type: TransformType,
    },
}

impl From<BitReaderEof> for TransformListError {
    fn from(e: BitReaderEof) -> Self {
        Self::Eof(e)
    }
}

impl core::fmt::Display for TransformListError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Eof(e) => write!(f, "VP8L §4 transform list: {e}"),
            Self::DuplicateTransform { transform_type } => write!(
                f,
                "VP8L §4 transform list: {transform_type:?} transform appears more than once"
            ),
        }
    }
}

impl std::error::Error for TransformListError {}

/// The result of reading the §4 transform-presence loop.
///
/// Records the transforms in **read order** (the spec applies inverse
/// transforms in reverse read order) plus the bit position where the
/// reader stopped — either at the end of the transform list (the `0`
/// transform-presence bit was consumed) or at the start of the first
/// §5-encoded transform body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformList {
    /// Transforms in read order.
    transforms: Vec<Transform>,
    /// Bit position at which the reader stopped.
    body_bit_position: usize,
    /// `true` if the reader stopped because it reached a §5-encoded
    /// transform body it cannot decode (the last entry in
    /// [`transforms`](Self::transforms) is that transform's header).
    /// `false` if it consumed the terminating `0` presence bit and the
    /// transform list is structurally complete.
    stopped_at_entropy_body: bool,
}

impl TransformList {
    /// Read the §4 transform-presence loop from `reader`.
    ///
    /// `reader` must be positioned just past the §3.4 image-header (use
    /// [`BitReader::new_after_image_header`]). The reader decodes each
    /// present transform's leading fixed fields. It stops when it
    /// either consumes the terminating `0` presence bit (a transform
    /// list with no §5 body — e.g. an all-`SUBTRACT_GREEN` list, or no
    /// transforms at all) or reaches the first transform that carries a
    /// §5 entropy-coded body (which it cannot skip without the §5
    /// decoder).
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self, TransformListError> {
        let mut transforms: Vec<Transform> = Vec::new();
        let mut seen = [false; 4];
        loop {
            if !reader.read_bit()? {
                // Terminating presence bit — list complete.
                return Ok(Self {
                    transforms,
                    body_bit_position: reader.bit_position(),
                    stopped_at_entropy_body: false,
                });
            }
            let ttype = TransformType::from_bits(reader.read_bits(2)?);
            let idx = ttype as usize;
            if seen[idx] {
                return Err(TransformListError::DuplicateTransform {
                    transform_type: ttype,
                });
            }
            seen[idx] = true;

            let transform = match ttype {
                TransformType::Predictor => {
                    let size_bits = (reader.read_bits(3)? + 2) as u8;
                    Transform::Predictor { size_bits }
                }
                TransformType::Color => {
                    let size_bits = (reader.read_bits(3)? + 2) as u8;
                    Transform::Color { size_bits }
                }
                TransformType::SubtractGreen => Transform::SubtractGreen,
                TransformType::ColorIndexing => {
                    let color_table_size = (reader.read_bits(8)? + 1) as u16;
                    Transform::ColorIndexing {
                        color_table_size,
                        width_bits: color_indexing_width_bits(color_table_size),
                    }
                }
            };
            let has_body = transform.has_entropy_body();
            transforms.push(transform);
            if has_body {
                // The transform body is §5-encoded; we cannot advance
                // past it to find the next presence bit. Stop here and
                // report the boundary for the §5 reader.
                return Ok(Self {
                    transforms,
                    body_bit_position: reader.bit_position(),
                    stopped_at_entropy_body: true,
                });
            }
            // SUBTRACT_GREEN has no body — keep reading the loop.
        }
    }

    /// The transforms in read order. The spec applies inverse
    /// transforms in **reverse** read order.
    pub fn transforms(&self) -> &[Transform] {
        &self.transforms
    }

    /// Bit position (from the start of the VP8L payload) at which the
    /// reader stopped. If [`stopped_at_entropy_body`](Self::stopped_at_entropy_body)
    /// is `true`, this is the start of the last transform's §5 body;
    /// otherwise it is just past the terminating `0` presence bit.
    pub fn body_bit_position(&self) -> usize {
        self.body_bit_position
    }

    /// `true` when the reader halted at a §5-encoded transform body it
    /// could not decode; `false` when the transform list was read to
    /// its terminating `0` presence bit.
    pub fn stopped_at_entropy_body(&self) -> bool {
        self.stopped_at_entropy_body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BitReader ----

    #[test]
    fn read_bits_is_lsb_first() {
        // byte 0b1010_0110 = 0xA6. Reading one bit at a time should
        // return the LSB first: 0,1,1,0,0,1,0,1.
        let data = [0xA6];
        let mut r = BitReader::new(&data);
        let expected = [0, 1, 1, 0, 0, 1, 0, 1];
        for &e in &expected {
            assert_eq!(r.read_bits(1).unwrap(), e);
        }
        assert_eq!(r.bit_position(), 8);
    }

    #[test]
    fn read_bits_multi_bit_assembles_first_bit_as_lsb() {
        // Spec example: b = ReadBits(2) == (ReadBits(1) | ReadBits(1)<<1).
        // byte 0xA6 = 0b1010_0110; first two bits LSB-first are 0 then 1
        // → value 0b10 = 2.
        let data = [0xA6];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(2).unwrap(), 0b10);
        // Equivalence with the two-statement form.
        let mut r2 = BitReader::new(&data);
        let b0 = r2.read_bits(1).unwrap();
        let b1 = r2.read_bits(1).unwrap();
        assert_eq!(b0 | (b1 << 1), 0b10);
    }

    #[test]
    fn read_bits_crosses_byte_boundary() {
        // bytes [0xFF, 0x01]: 8 ones then 1, then 7 zeros. Reading 9
        // bits LSB-first → 0b1_1111_1111 = 0x1FF.
        let data = [0xFF, 0x01];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(9).unwrap(), 0x1FF);
        assert_eq!(r.bit_position(), 9);
    }

    #[test]
    fn read_bits_full_u32() {
        let data = [0x78, 0x56, 0x34, 0x12];
        let mut r = BitReader::new(&data);
        // LSB-first byte order assembles to little-endian u32.
        assert_eq!(r.read_bits(32).unwrap(), 0x1234_5678);
    }

    #[test]
    fn read_bits_zero_is_noop() {
        let data = [0xFF];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(0).unwrap(), 0);
        assert_eq!(r.bit_position(), 0);
    }

    #[test]
    fn read_bits_eof_reports_position_and_demand() {
        let data = [0x00]; // 8 bits available
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(4).unwrap(), 0);
        match r.read_bits(8) {
            Err(BitReaderEof {
                bit_pos,
                wanted,
                available,
            }) => {
                assert_eq!(bit_pos, 4);
                assert_eq!(wanted, 8);
                assert_eq!(available, 4);
            }
            other => panic!("expected EOF, got {other:?}"),
        }
        // Position must not advance on a failed read.
        assert_eq!(r.bit_position(), 4);
    }

    #[test]
    fn seek_to_bit_repositions_and_clamps() {
        let data = [0x00u8, 0xFF];
        let mut r = BitReader::new(&data);
        r.seek_to_bit(8);
        assert_eq!(r.bit_position(), 8);
        // The byte at offset 8..16 is 0xFF.
        assert_eq!(r.read_bits(8).unwrap(), 0xFF);
        // Seeking past the end clamps to the slice length.
        r.seek_to_bit(1000);
        assert_eq!(r.bit_position(), 16);
        assert_eq!(r.bits_remaining(), 0);
    }

    #[test]
    fn new_after_image_header_skips_40_bits() {
        let payload = [0u8; 8];
        let r = BitReader::new_after_image_header(&payload);
        assert_eq!(
            r.bit_position(),
            crate::vp8l_chunk::VP8L_IMAGE_HEADER_LEN * 8
        );
        assert_eq!(r.bit_position(), 40);
    }

    // ---- TransformType ----

    #[test]
    fn transform_type_from_bits_total() {
        assert_eq!(TransformType::from_bits(0), TransformType::Predictor);
        assert_eq!(TransformType::from_bits(1), TransformType::Color);
        assert_eq!(TransformType::from_bits(2), TransformType::SubtractGreen);
        assert_eq!(TransformType::from_bits(3), TransformType::ColorIndexing);
    }

    #[test]
    fn color_indexing_width_bits_thresholds() {
        // §4.4 threshold table.
        assert_eq!(color_indexing_width_bits(1), 3);
        assert_eq!(color_indexing_width_bits(2), 3);
        assert_eq!(color_indexing_width_bits(3), 2);
        assert_eq!(color_indexing_width_bits(4), 2);
        assert_eq!(color_indexing_width_bits(5), 1);
        assert_eq!(color_indexing_width_bits(16), 1);
        assert_eq!(color_indexing_width_bits(17), 0);
        assert_eq!(color_indexing_width_bits(256), 0);
    }

    // ---- TransformList ----

    /// Tiny LSB-first bit *writer* for building synthetic transform
    /// lists in tests (mirror of the reader).
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
        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    #[test]
    fn empty_transform_list_consumes_one_zero_bit() {
        // Single 0 presence bit → no transforms, complete list.
        let mut w = BitWriter::new();
        w.write_bits(0, 1);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let list = TransformList::read(&mut r).unwrap();
        assert!(list.transforms().is_empty());
        assert!(!list.stopped_at_entropy_body());
        assert_eq!(list.body_bit_position(), 1);
    }

    #[test]
    fn subtract_green_only_then_terminator() {
        // present=1, type=SUBTRACT_GREEN(2), present=0.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(2, 2);
        w.write_bits(0, 1);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let list = TransformList::read(&mut r).unwrap();
        assert_eq!(list.transforms(), &[Transform::SubtractGreen]);
        assert!(!list.stopped_at_entropy_body());
        assert_eq!(list.body_bit_position(), 4);
    }

    #[test]
    fn predictor_stops_at_entropy_body() {
        // present=1, type=PREDICTOR(0), size_bits field = ReadBits(3).
        // size_bits=9 → ReadBits(3) returns 7.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(0, 2);
        w.write_bits(7, 3); // 7 + 2 = 9
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let list = TransformList::read(&mut r).unwrap();
        assert_eq!(list.transforms(), &[Transform::Predictor { size_bits: 9 }]);
        assert!(list.stopped_at_entropy_body());
        // 1 + 2 + 3 = 6 bits consumed.
        assert_eq!(list.body_bit_position(), 6);
    }

    #[test]
    fn color_transform_size_bits_min() {
        // present=1, type=COLOR(1), ReadBits(3)=1 → size_bits=3.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(1, 2);
        w.write_bits(1, 3);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let list = TransformList::read(&mut r).unwrap();
        assert_eq!(list.transforms(), &[Transform::Color { size_bits: 3 }]);
        assert!(list.stopped_at_entropy_body());
    }

    #[test]
    fn color_indexing_reads_table_size_and_width_bits() {
        // present=1, type=COLOR_INDEXING(3), ReadBits(8)=0 → size=1,
        // width_bits=3.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(3, 2);
        w.write_bits(0, 8); // num_colors = 1
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let list = TransformList::read(&mut r).unwrap();
        assert_eq!(
            list.transforms(),
            &[Transform::ColorIndexing {
                color_table_size: 1,
                width_bits: 3,
            }]
        );
        assert!(list.stopped_at_entropy_body());
        // 1 + 2 + 8 = 11 bits.
        assert_eq!(list.body_bit_position(), 11);
    }

    #[test]
    fn subtract_green_then_predictor_matches_fixture_shape() {
        // Mirrors lossless-32x32-rgba's first two transforms:
        // SUBTRACT_GREEN (no body) then PREDICTOR (size_bits=9, body).
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // present
        w.write_bits(2, 2); // SUBTRACT_GREEN
        w.write_bits(1, 1); // present
        w.write_bits(0, 2); // PREDICTOR
        w.write_bits(7, 3); // size_bits = 9
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let list = TransformList::read(&mut r).unwrap();
        assert_eq!(
            list.transforms(),
            &[
                Transform::SubtractGreen,
                Transform::Predictor { size_bits: 9 }
            ]
        );
        assert!(list.stopped_at_entropy_body());
    }

    #[test]
    fn duplicate_transform_is_refused() {
        // SUBTRACT_GREEN twice — §4 forbids reuse.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(2, 2);
        w.write_bits(1, 1);
        w.write_bits(2, 2);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        match TransformList::read(&mut r) {
            Err(TransformListError::DuplicateTransform { transform_type }) => {
                assert_eq!(transform_type, TransformType::SubtractGreen);
            }
            other => panic!("expected DuplicateTransform, got {other:?}"),
        }
    }

    #[test]
    fn truncated_transform_list_reports_eof() {
        // present=1 but no type bits follow.
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        let data = w.into_bytes();
        // Trim to exactly 1 bit of available data by using a 1-byte
        // slice — but bit 1..8 are 0, so type would read as PREDICTOR
        // and then size_bits would EOF. Instead start the reader at the
        // very end to force EOF on the first read.
        let mut r = BitReader::new(&data);
        r.read_bits(8).unwrap(); // consume the whole byte
        match TransformList::read(&mut r) {
            Err(TransformListError::Eof(_)) => {}
            other => panic!("expected Eof, got {other:?}"),
        }
    }

    #[test]
    fn transform_helpers_report_type_and_body() {
        assert_eq!(
            Transform::Predictor { size_bits: 4 }.transform_type(),
            TransformType::Predictor
        );
        assert!(Transform::Predictor { size_bits: 4 }.has_entropy_body());
        assert!(Transform::Color { size_bits: 4 }.has_entropy_body());
        assert!(Transform::ColorIndexing {
            color_table_size: 8,
            width_bits: 1
        }
        .has_entropy_body());
        assert!(!Transform::SubtractGreen.has_entropy_body());
    }
}
