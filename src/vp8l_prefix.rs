//! VP8L (WebP-Lossless) §6.2.1 prefix-code reader + canonical decoder.
//!
//! This sits directly on top of the round-99
//! [`crate::vp8l_stream::BitReader`] / [`crate::vp8l_stream::TransformList`]
//! layer. Where round 99 reads the §2 bit-reader foundation and the §4
//! transform list (stopping at the first §5 entropy body), this module
//! implements the first piece of the §5 / §6 entropy machinery: reading
//! a single canonical prefix code's *code lengths* off the wire and
//! building a decoder that can pull one symbol at a time.
//!
//! Per the WebP-Lossless spec §6.2.1 ("Decoding and Building the Prefix
//! Codes"):
//!
//! > Most of the data is coded using a canonical prefix code. Hence, the
//! > codes are transmitted by sending the prefix code lengths, as opposed
//! > to the actual prefix codes.
//!
//! A code's lengths can be transmitted two ways, selected by a leading
//! 1-bit flag:
//!
//! * **Simple code length code** (flag `1`, §6.2.1 "Simple Code Length
//!   Code"): 1 or 2 symbols in `[0..255]`, each at code length 1; every
//!   other length is implicitly 0.
//! * **Normal code length code** (flag `0`, §6.2.1 "Normal Code Length
//!   Code"): the code lengths are themselves entropy-coded with a
//!   19-symbol *code-length-code* whose own lengths are read first (in
//!   the fixed `kCodeLengthCodeOrder`), then run through a small
//!   canonical decoder that emits up to `max_symbol` literal lengths
//!   (codes `0..15`) and repeat/zero-run codes (`16`, `17`, `18`).
//!
//! Once the per-symbol code lengths are known, a **canonical prefix
//! code** is built. The canonical assignment is the one the spec's
//! `[Huffman]` reference fixes: symbols are ordered first by ascending
//! code length, then by ascending symbol value, and codes are assigned
//! sequentially; reading proceeds most-significant-bit-first *within a
//! code* (each freshly read bit appends to the low end of the
//! accumulated code value). The §6.2.1 completeness rule is enforced:
//! the sum of `2^(-length)` over all non-zero-length symbols must be
//! exactly one, with the single documented exception of the
//! single-leaf-node tree (one symbol at length 1, all others 0), which
//! consumes no bits when read.
//!
//! ## What this module does NOT do
//!
//! * No meta-Huffman / entropy-image decode (§6.2.2). That selects
//!   *which* prefix-code group applies to a given pixel block; it is the
//!   next section.
//! * No LZ77 backward-reference or color-cache decode (§5.2). Those
//!   consume symbols produced by this decoder but live one layer up.
//! * No `oxideav-core` runtime dependency — this module compiles under
//!   `--no-default-features`.

use crate::vp8l_stream::{BitReader, BitReaderEof};

/// §6.2.1: the number of code-length-code symbols (the small alphabet
/// used by the *normal* code length code).
pub const NUM_CODE_LENGTH_CODES: usize = 19;

/// §6.2.1 `kCodeLengthCodeOrder`: the order in which the (up to 19)
/// code-length-code lengths are transmitted. The first
/// `num_code_lengths` entries here are read; the remaining positions
/// keep their implicit length 0.
pub const CODE_LENGTH_CODE_ORDER: [usize; NUM_CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// The largest code length a VP8L canonical prefix code may use. The
/// spec stores literal code lengths in the `[0..15]` range, so 15 bits
/// is the hard ceiling.
pub const MAX_CODE_LENGTH: usize = 15;

/// Width (in bits) of the primary [`PrefixCode::read_symbol`] lookup
/// table. Codes of length ≤ `LOOKUP_BITS` resolve with one peek + one
/// table load; longer codes fall back to the per-bit row walk. 8 bits
/// keeps the table at 1 KiB (one `u32` per entry) — small enough that
/// building it stays cheap for the per-image table churn (five codes
/// per §6.2 prefix-code group plus one code-length code per normal
/// read) while covering the overwhelming share of symbols: a canonical
/// code assigns its most frequent symbols the shortest codes, so the
/// ≤ 8-bit slice of the alphabet carries most of the stream.
const LOOKUP_BITS: usize = 8;

/// Number of entries in the primary lookup table (`1 << LOOKUP_BITS`).
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;

/// Minimum used-symbol count below which
/// [`PrefixCode::from_code_lengths`] skips building the lookup table.
/// The table is an investment paid at build time (allocation + zero
/// fill + stamping) and on first touches (16 cold cache lines): a code
/// with few used symbols has short codes (≤ `used - 1` bits by the
/// §6.2.1 Kraft equality) that the per-bit walk already resolves in a
/// couple of predictable iterations, and such codes are typical of
/// header-dominated streams (tiny animation delta frames, sub-resolution
/// transform images, the 19-symbol code-length code) where the table
/// would be built, touched a handful of times, and thrown away.
const MIN_LOOKUP_USED: usize = 32;

/// Errors raised while reading or building a §6.2.1 prefix code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixError {
    /// The bit reader hit EOF mid-field.
    Eof(BitReaderEof),
    /// A symbol value read by the simple/normal code length code is at
    /// or beyond the alphabet size.
    SymbolOutOfRange {
        /// The offending symbol value.
        symbol: usize,
        /// The alphabet size for this code.
        alphabet_size: usize,
    },
    /// §6.2.1: `max_symbol` exceeded the alphabet size.
    MaxSymbolTooLarge {
        /// The `max_symbol` value read from the bitstream.
        max_symbol: usize,
        /// The alphabet size for this code.
        alphabet_size: usize,
    },
    /// A code-length-code repeat (`16`) was used before any non-zero
    /// length had been emitted in a context that forbids it, or a
    /// repeat/zero-run code overran the alphabet.
    LengthRunOverflow,
    /// §6.2.1: the assembled code is over-subscribed — the sum of
    /// `2^(-length)` exceeds one (more leaves than the tree can hold).
    OverSubscribed,
    /// §6.2.1: the assembled code is incomplete — the sum of
    /// `2^(-length)` is below one and the code is not the documented
    /// single-leaf exception.
    Incomplete,
    /// A code length larger than [`MAX_CODE_LENGTH`] was produced.
    LengthTooLarge {
        /// The offending length.
        length: usize,
    },
    /// The bitstream presented a bit pattern with no matching symbol in
    /// a (necessarily complete) code while decoding a symbol.
    NoMatchingSymbol,
}

impl From<BitReaderEof> for PrefixError {
    fn from(e: BitReaderEof) -> Self {
        Self::Eof(e)
    }
}

impl core::fmt::Display for PrefixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Eof(e) => write!(f, "VP8L §6.2.1 prefix code: {e}"),
            Self::SymbolOutOfRange {
                symbol,
                alphabet_size,
            } => write!(
                f,
                "VP8L §6.2.1 prefix code: symbol {symbol} out of range for alphabet size {alphabet_size}"
            ),
            Self::MaxSymbolTooLarge {
                max_symbol,
                alphabet_size,
            } => write!(
                f,
                "VP8L §6.2.1 prefix code: max_symbol {max_symbol} exceeds alphabet size {alphabet_size}"
            ),
            Self::LengthRunOverflow => {
                f.write_str("VP8L §6.2.1 prefix code: code-length run overran the alphabet")
            }
            Self::OverSubscribed => {
                f.write_str("VP8L §6.2.1 prefix code: over-subscribed (sum 2^-len > 1)")
            }
            Self::Incomplete => {
                f.write_str("VP8L §6.2.1 prefix code: incomplete (sum 2^-len < 1)")
            }
            Self::LengthTooLarge { length } => write!(
                f,
                "VP8L §6.2.1 prefix code: code length {length} exceeds the {MAX_CODE_LENGTH}-bit ceiling"
            ),
            Self::NoMatchingSymbol => {
                f.write_str("VP8L §6.2.1 prefix code: bit pattern matched no symbol")
            }
        }
    }
}

impl std::error::Error for PrefixError {}

/// A built canonical prefix code over an alphabet of `alphabet_size`
/// symbols.
///
/// Constructed from a per-symbol code-length table via
/// [`PrefixCode::from_code_lengths`], or read straight off the
/// bitstream with [`PrefixCode::read`]. Symbols are decoded one at a
/// time with [`PrefixCode::read_symbol`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCode {
    /// Per-symbol code length (0 = symbol absent from the code).
    code_lengths: Vec<u8>,
    /// `(length, first_code, first_symbol_index)` rows, one per
    /// distinct used length, in ascending length order. Within a
    /// length, symbols are stored in ascending value order in
    /// [`sorted_symbols`](Self::sorted_symbols) starting at
    /// `first_symbol_index`.
    length_rows: Vec<LengthRow>,
    /// Symbols sorted by `(length, value)`; used-length symbols only.
    sorted_symbols: Vec<u16>,
    /// `true` when the code is the §6.2.1 single-leaf-node tree (exactly
    /// one symbol, at length 1, reading consumes no bits).
    single_symbol: Option<u16>,
    /// Primary decode table: `LOOKUP_SIZE` entries indexed by the next
    /// `LOOKUP_BITS` bits of the stream **in wire order** (first bit
    /// read = bit 0 of the index, matching `BitReader::peek_bits`).
    /// Each entry packs `(code_length << 16) | symbol` for the unique
    /// code (of length ≤ `LOOKUP_BITS`) that is a prefix of that bit
    /// pattern; `0` (impossible length) marks indices whose matching
    /// code is longer than `LOOKUP_BITS` — those resolve through the
    /// per-bit slow path. Empty for the single-leaf-node tree and for
    /// codes below the `MIN_LOOKUP_USED` amortization gate (which
    /// decode through the per-bit walk, as before the table existed).
    lookup: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LengthRow {
    length: u8,
    /// Canonical code value of the first symbol at this length.
    first_code: u32,
    /// Index into [`PrefixCode::sorted_symbols`] of the first symbol at
    /// this length.
    first_symbol_index: usize,
    /// Number of symbols at this length.
    count: usize,
}

impl PrefixCode {
    /// Build a canonical prefix code from a per-symbol code-length
    /// table.
    ///
    /// `code_lengths[s]` is the number of bits assigned to symbol `s`
    /// (0 means symbol `s` is not in the code). The canonical
    /// assignment orders symbols by `(length, value)` and assigns codes
    /// sequentially, mirroring the DEFLATE/`[Huffman]` canonical rule
    /// the spec references.
    ///
    /// Enforces the §6.2.1 completeness rule (`sum 2^-len == 1`) with
    /// the single-leaf-node exception.
    pub fn from_code_lengths(code_lengths: Vec<u8>) -> Result<Self, PrefixError> {
        // Count symbols per length and validate the ceiling.
        let mut bl_count = [0usize; MAX_CODE_LENGTH + 1];
        let mut used = 0usize;
        let mut single_candidate: Option<u16> = None;
        for (sym, &len) in code_lengths.iter().enumerate() {
            let len = len as usize;
            if len > MAX_CODE_LENGTH {
                return Err(PrefixError::LengthTooLarge { length: len });
            }
            if len != 0 {
                bl_count[len] += 1;
                used += 1;
                single_candidate = Some(sym as u16);
            }
        }

        // Degenerate code: no symbols at all. The spec treats an empty
        // prefix code as a single-symbol-0 code (§6.2.1 note), so we
        // refuse a truly empty table here and let callers encode the
        // single-symbol-0 form explicitly.
        if used == 0 {
            return Err(PrefixError::Incomplete);
        }

        // §6.2.1 single-leaf-node exception: exactly one used symbol.
        // The spec marks it with length 1 and reading consumes no bits.
        if used == 1 {
            let sym = single_candidate.expect("used == 1");
            return Ok(Self {
                code_lengths,
                length_rows: Vec::new(),
                sorted_symbols: vec![sym],
                single_symbol: Some(sym),
                lookup: Vec::new(),
            });
        }

        // Canonical first-code per length (DEFLATE bit-reversed-free
        // form: codes read MSB-first within a code).
        let mut next_code = [0u32; MAX_CODE_LENGTH + 2];
        let mut code: u32 = 0;
        // Kraft sum over a common denominator 2^MAX_CODE_LENGTH to keep
        // the completeness check in integer arithmetic.
        let mut kraft: u64 = 0;
        for len in 1..=MAX_CODE_LENGTH {
            code = (code + bl_count[len - 1] as u32) << 1;
            next_code[len] = code;
            if bl_count[len] > 0 {
                kraft += (bl_count[len] as u64) << (MAX_CODE_LENGTH - len);
            }
        }
        let full: u64 = 1u64 << MAX_CODE_LENGTH;
        match kraft.cmp(&full) {
            core::cmp::Ordering::Greater => return Err(PrefixError::OverSubscribed),
            core::cmp::Ordering::Less => return Err(PrefixError::Incomplete),
            core::cmp::Ordering::Equal => {}
        }

        // Assign codes in (length, value) order and record per-length
        // decode rows. The `(length, value)` sort is a counting sort:
        // `bl_count` already gives each length's bucket size, so prefix
        // sums fix every bucket's start index up front and ONE rescan of
        // `code_lengths` drops each used symbol at its bucket cursor —
        // symbols are visited in ascending value order, so each bucket
        // stays value-sorted internally. (Previously this rescanned the
        // full table once per used length.)
        let mut length_rows: Vec<LengthRow> = Vec::new();
        let mut cursor = [0usize; MAX_CODE_LENGTH + 1];
        let mut first_symbol_index = 0usize;
        for len in 1..=MAX_CODE_LENGTH {
            if bl_count[len] == 0 {
                continue;
            }
            cursor[len] = first_symbol_index;
            length_rows.push(LengthRow {
                length: len as u8,
                first_code: next_code[len],
                first_symbol_index,
                count: bl_count[len],
            });
            first_symbol_index += bl_count[len];
        }
        let mut sorted_symbols: Vec<u16> = vec![0; used];
        for (sym, &slen) in code_lengths.iter().enumerate() {
            if slen != 0 {
                let slot = &mut cursor[slen as usize];
                sorted_symbols[*slot] = sym as u16;
                *slot += 1;
            }
        }

        // Primary lookup table (built only past the MIN_LOOKUP_USED
        // amortization gate). For every code of length `len <=
        // LOOKUP_BITS` the matching indices are exactly those whose low
        // `len` bits equal the code's bits *in wire order*. The wire
        // order is the reverse of the canonical code value: the first
        // bit read is the code's MSB (the slow path appends each fresh
        // bit at the low end of the accumulator) but bit 0 of a peeked
        // index (LSB-first `ReadBits`). So the base index is the
        // canonical value bit-reversed across `len` bits, and the
        // remaining `LOOKUP_BITS - len` high index bits are free —
        // stamped over every `1 << len` stride. Indices left at 0 are
        // prefixes of longer codes only (the §6.2.1 Kraft completeness
        // check guarantees the code is prefix-free and complete, so no
        // two stamps collide).
        let mut lookup = Vec::new();
        if used >= MIN_LOOKUP_USED {
            lookup = vec![0u32; LOOKUP_SIZE];
            for row in &length_rows {
                let len = row.length as usize;
                if len > LOOKUP_BITS {
                    // Rows are in ascending length order; the rest are
                    // slow-path codes.
                    break;
                }
                for k in 0..row.count {
                    let code = row.first_code + k as u32;
                    let symbol = sorted_symbols[row.first_symbol_index + k];
                    let base = (code.reverse_bits() >> (32 - len)) as usize;
                    let entry = ((len as u32) << 16) | symbol as u32;
                    let mut idx = base;
                    while idx < LOOKUP_SIZE {
                        lookup[idx] = entry;
                        idx += 1 << len;
                    }
                }
            }
        }

        Ok(Self {
            code_lengths,
            length_rows,
            sorted_symbols,
            single_symbol: None,
            lookup,
        })
    }

    /// Read one symbol from `reader` using this code.
    ///
    /// For the single-leaf-node tree this consumes no bits and returns
    /// the lone symbol. Otherwise the next `LOOKUP_BITS` stream bits
    /// are peeked (zero-padded at EOF, cursor unmoved) and resolved
    /// through the primary lookup table: one load gives `(symbol,
    /// code_length)` for any code of length ≤ `LOOKUP_BITS`, and the
    /// cursor advances by exactly that length. Longer codes — and
    /// near-EOF reads where the matched length exceeds the bits
    /// actually remaining (so the match could have leaned on the zero
    /// padding) — fall back to the per-bit row walk, which reproduces
    /// the pre-table behaviour bit for bit, including the EOF error's
    /// position fields.
    pub fn read_symbol(&self, reader: &mut BitReader<'_>) -> Result<u16, PrefixError> {
        if let Some(sym) = self.single_symbol {
            return Ok(sym);
        }
        if self.lookup.is_empty() {
            // Below the MIN_LOOKUP_USED gate: small codes resolve in a
            // couple of per-bit iterations.
            return self.read_symbol_walk(reader);
        }
        let peeked = reader.peek_bits(LOOKUP_BITS);
        let entry = self.lookup[peeked as usize];
        let len = (entry >> 16) as usize;
        let available = reader.bits_remaining();
        if len != 0 {
            if len <= available {
                reader.advance_bits(len);
                return Ok(entry as u16);
            }
            // The match leaned on the zero padding past EOF; replay the
            // per-bit walk so the EOF error carries the exact pre-table
            // position fields.
            return self.read_symbol_walk(reader);
        }
        if available >= LOOKUP_BITS {
            // The peeked LOOKUP_BITS bits are all real stream bits and —
            // by the table's completeness — a strict prefix of a longer
            // code. Consume them and continue the walk from there: the
            // accumulated MSB-first code value of the first LOOKUP_BITS
            // bits is the bit-reversal of the peeked (wire-order) value.
            let code = peeked.reverse_bits() >> (32 - LOOKUP_BITS);
            reader.advance_bits(LOOKUP_BITS);
            return self.read_symbol_long(reader, code);
        }
        // Fewer than LOOKUP_BITS real bits remain and none of their
        // prefixes completes a code (the table entry would have carried
        // it): the per-bit walk runs the stream dry exactly as before.
        self.read_symbol_walk(reader)
    }

    /// Continuation of [`read_symbol`](Self::read_symbol) for codes
    /// longer than `LOOKUP_BITS`: `code` is the accumulated MSB-first
    /// value of the `LOOKUP_BITS` bits already consumed (none of whose
    /// prefixes matched a row — guaranteed by the lookup table), so the
    /// walk resumes at length `LOOKUP_BITS + 1` with decisions, bit
    /// consumption, and error positions identical to the pre-table
    /// per-bit loop.
    fn read_symbol_long(
        &self,
        reader: &mut BitReader<'_>,
        mut code: u32,
    ) -> Result<u16, PrefixError> {
        let mut len = LOOKUP_BITS;
        loop {
            len += 1;
            if len > MAX_CODE_LENGTH {
                return Err(PrefixError::NoMatchingSymbol);
            }
            // A canonical code reads MSB-first within the code: append
            // the freshly read bit to the low end.
            code = (code << 1) | reader.read_bits(1)?;
            // Is there a row at this length whose code range contains
            // `code`?
            if let Some(row) = self.length_rows.iter().find(|r| r.length as usize == len) {
                let offset = code.wrapping_sub(row.first_code);
                if (offset as usize) < row.count {
                    return Ok(self.sorted_symbols[row.first_symbol_index + offset as usize]);
                }
            }
        }
    }

    /// Reference decoder: the pre-table per-bit walk, bypassing the
    /// primary lookup table entirely (the single-leaf-node tree still
    /// consumes no bits, as in [`read_symbol`](Self::read_symbol)).
    ///
    /// Behaviour-neutral oracle for the `read_symbol_lut_diff` fuzz
    /// harness and the in-tree differential tests: for every code and
    /// every bit stream it must produce exactly the symbol sequence,
    /// cursor positions, and error fields [`read_symbol`](Self::read_symbol)
    /// produces through the table fast path. Not part of the supported
    /// API surface.
    #[doc(hidden)]
    pub fn read_symbol_reference(&self, reader: &mut BitReader<'_>) -> Result<u16, PrefixError> {
        if let Some(sym) = self.single_symbol {
            return Ok(sym);
        }
        self.read_symbol_walk(reader)
    }

    /// The pre-table per-bit walk, bit for bit: the only decode path
    /// for codes below the `MIN_LOOKUP_USED` gate (no table built), and
    /// the fallback for table reads close enough to EOF that the table
    /// outcome cannot be used — so a raised [`PrefixError::Eof`]
    /// carries exactly the position / wanted / available fields the
    /// pre-table implementation produced.
    fn read_symbol_walk(&self, reader: &mut BitReader<'_>) -> Result<u16, PrefixError> {
        let mut code: u32 = 0;
        let mut len: usize = 0;
        loop {
            len += 1;
            if len > MAX_CODE_LENGTH {
                return Err(PrefixError::NoMatchingSymbol);
            }
            code = (code << 1) | reader.read_bits(1)?;
            if let Some(row) = self.length_rows.iter().find(|r| r.length as usize == len) {
                let offset = code.wrapping_sub(row.first_code);
                if (offset as usize) < row.count {
                    return Ok(self.sorted_symbols[row.first_symbol_index + offset as usize]);
                }
            }
        }
    }

    /// The per-symbol code length table this code was built from.
    pub fn code_lengths(&self) -> &[u8] {
        &self.code_lengths
    }

    /// `Some(symbol)` when this is the §6.2.1 single-leaf-node tree
    /// (reading consumes no bits); `None` otherwise.
    pub fn single_symbol(&self) -> Option<u16> {
        self.single_symbol
    }

    /// Read a §6.2.1 prefix code straight off `reader`, given the
    /// alphabet size for this symbol type.
    ///
    /// Dispatches on the leading 1-bit flag: `1` selects the simple
    /// code length code, `0` selects the normal code length code. The
    /// returned [`PrefixCode`] is ready for [`read_symbol`](Self::read_symbol).
    pub fn read(reader: &mut BitReader<'_>, alphabet_size: usize) -> Result<Self, PrefixError> {
        let lengths = read_code_lengths(reader, alphabet_size)?;
        Self::from_code_lengths(lengths)
    }
}

/// Read the per-symbol code-length table for one §6.2.1 prefix code.
///
/// Returns a `Vec<u8>` of length `alphabet_size` (lengths for symbols
/// at or beyond `max_symbol` are 0). Dispatches on the leading
/// simple/normal flag.
pub fn read_code_lengths(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
) -> Result<Vec<u8>, PrefixError> {
    if reader.read_bit()? {
        read_simple_code_lengths(reader, alphabet_size)
    } else {
        read_normal_code_lengths(reader, alphabet_size)
    }
}

/// §6.2.1 "Simple Code Length Code": 1 or 2 symbols, each at length 1.
fn read_simple_code_lengths(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
) -> Result<Vec<u8>, PrefixError> {
    let mut lengths = vec![0u8; alphabet_size];
    let num_symbols = reader.read_bits(1)? + 1;
    let is_first_8bits = reader.read_bits(1)?;
    let symbol0 = reader.read_bits(1 + 7 * is_first_8bits as usize)? as usize;
    if symbol0 >= alphabet_size {
        return Err(PrefixError::SymbolOutOfRange {
            symbol: symbol0,
            alphabet_size,
        });
    }
    lengths[symbol0] = 1;
    if num_symbols == 2 {
        let symbol1 = reader.read_bits(8)? as usize;
        if symbol1 >= alphabet_size {
            return Err(PrefixError::SymbolOutOfRange {
                symbol: symbol1,
                alphabet_size,
            });
        }
        // §6.2.1: "The two symbols should be different. Duplicate
        // symbols are allowed, but inefficient." Setting length 1 twice
        // is idempotent, so a duplicate collapses to a single-symbol
        // code — which is exactly the spec's allowed (inefficient)
        // behaviour.
        lengths[symbol1] = 1;
    }
    Ok(lengths)
}

/// §6.2.1 "Normal Code Length Code": the code lengths are themselves
/// entropy-coded with a 19-symbol code-length-code.
fn read_normal_code_lengths(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
) -> Result<Vec<u8>, PrefixError> {
    // Read the (up to 19) code-length-code lengths in kCodeLengthCodeOrder.
    let num_code_lengths = 4 + reader.read_bits(4)? as usize;
    let mut clcl = [0u8; NUM_CODE_LENGTH_CODES];
    for &pos in CODE_LENGTH_CODE_ORDER.iter().take(num_code_lengths) {
        clcl[pos] = reader.read_bits(3)? as u8;
    }
    // Build the small canonical decoder for the code-length-code.
    let cl_code = PrefixCode::from_code_lengths(clcl.to_vec())?;

    // §6.2.1 max_symbol gate.
    let max_symbol = if reader.read_bits(1)? == 0 {
        alphabet_size
    } else {
        let length_nbits = 2 + 2 * reader.read_bits(3)? as usize;
        let ms = 2 + reader.read_bits(length_nbits)? as usize;
        if ms > alphabet_size {
            return Err(PrefixError::MaxSymbolTooLarge {
                max_symbol: ms,
                alphabet_size,
            });
        }
        ms
    };

    // Decode up to max_symbol literal code lengths, expanding the
    // repeat/zero-run codes 16/17/18.
    let mut lengths = vec![0u8; alphabet_size];
    let mut symbol = 0usize;
    let mut prev_nonzero: u8 = 8; // §6.2.1: code 16 before any nonzero repeats 8.
    let mut emitted = 0usize;
    while symbol < alphabet_size && emitted < max_symbol {
        emitted += 1;
        let code = cl_code.read_symbol(reader)? as usize;
        match code {
            0..=15 => {
                lengths[symbol] = code as u8;
                if code != 0 {
                    prev_nonzero = code as u8;
                }
                symbol += 1;
            }
            16 => {
                // Repeat the previous non-zero length 3 + ReadBits(2)
                // times.
                let repeat = 3 + reader.read_bits(2)? as usize;
                if symbol + repeat > alphabet_size {
                    return Err(PrefixError::LengthRunOverflow);
                }
                for _ in 0..repeat {
                    lengths[symbol] = prev_nonzero;
                    symbol += 1;
                }
            }
            17 => {
                // Streak of zeros, 3 + ReadBits(3).
                let repeat = 3 + reader.read_bits(3)? as usize;
                if symbol + repeat > alphabet_size {
                    return Err(PrefixError::LengthRunOverflow);
                }
                symbol += repeat;
            }
            18 => {
                // Streak of zeros, 11 + ReadBits(7).
                let repeat = 11 + reader.read_bits(7)? as usize;
                if symbol + repeat > alphabet_size {
                    return Err(PrefixError::LengthRunOverflow);
                }
                symbol += repeat;
            }
            _ => return Err(PrefixError::LengthRunOverflow),
        }
    }

    Ok(lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny LSB-first bit writer for building synthetic prefix codes
    /// in tests (mirror of [`BitReader`]).
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

    // ---- canonical code construction ----

    #[test]
    fn single_leaf_node_consumes_no_bits() {
        // §6.2.1 single-leaf exception: one symbol at length 1.
        let mut lengths = vec![0u8; 256];
        lengths[42] = 1;
        let code = PrefixCode::from_code_lengths(lengths).unwrap();
        assert_eq!(code.single_symbol(), Some(42));
        // Reading consumes no bits.
        let data = [0u8; 0];
        let mut r = BitReader::new(&data);
        assert_eq!(code.read_symbol(&mut r).unwrap(), 42);
        assert_eq!(r.bit_position(), 0);
    }

    #[test]
    fn two_symbol_length_one_code_is_canonical() {
        // Two symbols both at length 1 → complete tree. Symbol with the
        // smaller value gets code 0, the larger gets code 1.
        let mut lengths = vec![0u8; 4];
        lengths[1] = 1;
        lengths[3] = 1;
        let code = PrefixCode::from_code_lengths(lengths).unwrap();
        assert_eq!(code.single_symbol(), None);
        // bit 0 → symbol 1, bit 1 → symbol 3.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // reads MSB-first within code: value 0
        w.write_bits(1, 1); // value 1
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        assert_eq!(code.read_symbol(&mut r).unwrap(), 1);
        assert_eq!(code.read_symbol(&mut r).unwrap(), 3);
    }

    #[test]
    fn canonical_lengths_1_2_3_3_decode_in_value_order() {
        // Classic canonical example: lengths [1,2,3,3] over symbols
        // 0..3. Canonical codes: sym0=0 (len1), sym1=10 (len2),
        // sym2=110 (len3), sym3=111 (len3). Completeness: 1/2 + 1/4 +
        // 1/8 + 1/8 == 1.
        let lengths = vec![1u8, 2, 3, 3];
        let code = PrefixCode::from_code_lengths(lengths).unwrap();

        // Build a bitstream of sym0, sym1, sym2, sym3 in MSB-first form.
        let mut w = BitWriter::new();
        // sym0: code "0" (1 bit)
        w.write_bits(0, 1);
        // sym1: code "10" → MSB-first bits 1 then 0
        w.write_bits(1, 1);
        w.write_bits(0, 1);
        // sym2: code "110" → 1,1,0
        w.write_bits(1, 1);
        w.write_bits(1, 1);
        w.write_bits(0, 1);
        // sym3: code "111" → 1,1,1
        w.write_bits(1, 1);
        w.write_bits(1, 1);
        w.write_bits(1, 1);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        assert_eq!(code.read_symbol(&mut r).unwrap(), 0);
        assert_eq!(code.read_symbol(&mut r).unwrap(), 1);
        assert_eq!(code.read_symbol(&mut r).unwrap(), 2);
        assert_eq!(code.read_symbol(&mut r).unwrap(), 3);
    }

    #[test]
    fn over_subscribed_code_is_refused() {
        // Three symbols at length 1 → sum 2^-1 * 3 = 1.5 > 1.
        let lengths = vec![1u8, 1, 1];
        match PrefixCode::from_code_lengths(lengths) {
            Err(PrefixError::OverSubscribed) => {}
            other => panic!("expected OverSubscribed, got {other:?}"),
        }
    }

    #[test]
    fn incomplete_code_is_refused() {
        // Two symbols at length 2 → sum 2^-2 * 2 = 0.5 < 1.
        let lengths = vec![2u8, 2, 0, 0];
        match PrefixCode::from_code_lengths(lengths) {
            Err(PrefixError::Incomplete) => {}
            other => panic!("expected Incomplete, got {other:?}"),
        }
    }

    #[test]
    fn empty_table_is_refused() {
        let lengths = vec![0u8; 8];
        match PrefixCode::from_code_lengths(lengths) {
            Err(PrefixError::Incomplete) => {}
            other => panic!("expected Incomplete, got {other:?}"),
        }
    }

    #[test]
    fn length_too_large_is_refused() {
        let mut lengths = vec![0u8; 4];
        lengths[0] = (MAX_CODE_LENGTH + 1) as u8;
        match PrefixCode::from_code_lengths(lengths) {
            Err(PrefixError::LengthTooLarge { length }) => {
                assert_eq!(length, MAX_CODE_LENGTH + 1);
            }
            other => panic!("expected LengthTooLarge, got {other:?}"),
        }
    }

    // ---- simple code length code ----

    #[test]
    fn simple_one_symbol_8bit() {
        // flag=1 (simple), num_symbols=1, is_first_8bits=1, symbol=60.
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // simple
        w.write_bits(0, 1); // num_symbols-1 = 0
        w.write_bits(1, 1); // is_first_8bits = 1
        w.write_bits(60, 8); // symbol0
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let code = PrefixCode::read(&mut r, 256 + 24).unwrap();
        assert_eq!(code.single_symbol(), Some(60));
    }

    #[test]
    fn simple_one_symbol_1bit() {
        // is_first_8bits=0 → symbol in [0..1] coded with 1 bit.
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // simple
        w.write_bits(0, 1); // num_symbols-1 = 0
        w.write_bits(0, 1); // is_first_8bits = 0
        w.write_bits(1, 1); // symbol0 = 1
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let code = PrefixCode::read(&mut r, 256).unwrap();
        assert_eq!(code.single_symbol(), Some(1));
    }

    #[test]
    fn simple_two_symbols() {
        // num_symbols=2, symbol0=5 (8 bits), symbol1=200 (8 bits).
        let mut w = BitWriter::new();
        w.write_bits(1, 1); // simple
        w.write_bits(1, 1); // num_symbols-1 = 1 → 2 symbols
        w.write_bits(1, 1); // is_first_8bits = 1
        w.write_bits(5, 8); // symbol0
        w.write_bits(200, 8); // symbol1
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let code = PrefixCode::read(&mut r, 256).unwrap();
        assert_eq!(code.single_symbol(), None);
        // Both at length 1; canonical: smaller value (5) → bit 0,
        // larger (200) → bit 1.
        let mut w2 = BitWriter::new();
        w2.write_bits(0, 1);
        w2.write_bits(1, 1);
        let d2 = w2.into_bytes();
        let mut r2 = BitReader::new(&d2);
        assert_eq!(code.read_symbol(&mut r2).unwrap(), 5);
        assert_eq!(code.read_symbol(&mut r2).unwrap(), 200);
    }

    #[test]
    fn simple_symbol_out_of_range_is_refused() {
        // symbol0 = 200 but alphabet size is 40 (distance alphabet).
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        w.write_bits(200, 8);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        match PrefixCode::read(&mut r, 40) {
            Err(PrefixError::SymbolOutOfRange {
                symbol,
                alphabet_size,
            }) => {
                assert_eq!(symbol, 200);
                assert_eq!(alphabet_size, 40);
            }
            other => panic!("expected SymbolOutOfRange, got {other:?}"),
        }
    }

    // ---- normal code length code ----

    /// Build a *normal* code length code header that transmits a given
    /// set of literal code lengths directly (no repeat codes), using a
    /// code-length-code that maps each used CLC length to a length-1
    /// or length-2 symbol. To keep the test tractable we use a
    /// code-length-code where symbols 0 and 1 (i.e. literal lengths 0
    /// and 1) are the only ones present, both at length 1.
    #[test]
    fn normal_two_symbol_alphabet_with_clc() {
        // Target: a 4-symbol literal alphabet with code lengths
        // [1, 1, 0, 0] (symbols 0 and 1 each length 1).
        //
        // Code-length-code: we need to emit literal lengths 0 and 1.
        // Use a CLC where CLC-symbol 0 (literal length 0) and CLC-symbol
        // 1 (literal length 1) are both at CLC-length 1. That CLC is a
        // 2-leaf complete tree: CLC-sym0 → bit 0, CLC-sym1 → bit 1.
        //
        // kCodeLengthCodeOrder = [17,18,0,1,2,...]; to set positions 0
        // and 1 we must transmit at least the first 4 entries
        // (17,18,0,1). num_code_lengths = 4.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // normal
        w.write_bits(0, 4); // num_code_lengths - 4 = 0 → 4
                            // CLC lengths in order [17,18,0,1]: we want pos0=1, pos1=1,
                            // pos17=0, pos18=0.
        w.write_bits(0, 3); // pos 17 → 0
        w.write_bits(0, 3); // pos 18 → 0
        w.write_bits(1, 3); // pos 0  → 1
        w.write_bits(1, 3); // pos 1  → 1
                            // max_symbol gate: use full alphabet.
        w.write_bits(0, 1); // use_max = 0 → max_symbol = alphabet_size
                            // Now read up to alphabet_size (4) literal code lengths via the
                            // CLC. CLC-sym0 → bit 0, CLC-sym1 → bit 1.
                            // We want literal lengths [1, 1, 0, 0] → CLC symbols [1,1,0,0].
        w.write_bits(1, 1); // literal len for sym0 = 1
        w.write_bits(1, 1); // sym1 = 1
        w.write_bits(0, 1); // sym2 = 0
        w.write_bits(0, 1); // sym3 = 0
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let code = PrefixCode::read(&mut r, 4).unwrap();
        assert_eq!(code.code_lengths(), &[1, 1, 0, 0]);
        // Decode: bit 0 → sym0, bit 1 → sym1.
        let mut w2 = BitWriter::new();
        w2.write_bits(0, 1);
        w2.write_bits(1, 1);
        let d2 = w2.into_bytes();
        let mut r2 = BitReader::new(&d2);
        assert_eq!(code.read_symbol(&mut r2).unwrap(), 0);
        assert_eq!(code.read_symbol(&mut r2).unwrap(), 1);
    }

    #[test]
    fn normal_zero_run_code_18() {
        // Use CLC code 18 (zero streak 11 + ReadBits(7)) to skip a long
        // run of zeros, then two literal length-1 symbols. Alphabet size
        // 16; we want lengths[0..11]=0, lengths[12]=lengths[13]=1, and
        // lengths[14..15]=0 (a 2-leaf complete code).
        //
        // The CLC only carries symbols for literal-length 1 and the
        // code-18 zero run, so it cannot encode a literal-length-0
        // symbol directly. We gate the read with `max_symbol` so the
        // decode loop stops right after sym13 — symbols 14 and 15 are
        // never read and keep their implicit length 0 (§6.2.1: "used to
        // read up to max_symbol code lengths").
        //
        // CLC: pos1=1 (literal len 1), pos18=1 (zero run). Order
        // [17,18,0,1,...]; the first 4 entries reach pos1.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // normal
        w.write_bits(0, 4); // num_code_lengths = 4
        w.write_bits(0, 3); // pos17 → 0
        w.write_bits(1, 3); // pos18 → 1
        w.write_bits(0, 3); // pos0  → 0
        w.write_bits(1, 3); // pos1  → 1
                            // max_symbol gate: §6.2.1 says the CLC is "used to read up to
                            // max_symbol code lengths" — i.e. max_symbol counts CLC symbols
                            // *read*, not the literal-symbol cursor. We make exactly 3 CLC
                            // reads below (one code-18 zero run + two literal-1 codes), so
                            // gate at 3. use_max = 1, length_nbits = 2 + 2*ReadBits(3); pick
                            // ReadBits(3) = 0 → length_nbits = 2; max_symbol = 2 +
                            // ReadBits(2). Want max_symbol = 3 → ReadBits(2) = 1.
        w.write_bits(1, 1); // use_max = 1
        w.write_bits(0, 3); // length_nbits = 2
        w.write_bits(1, 2); // max_symbol = 2 + 1 = 3
                            // CLC is 2 leaves: CLC-sym1 and CLC-sym18, both length 1.
                            // Canonical (value order): sym1 → bit 0, sym18 → bit 1.
                            // CLC read #1: code 18 with run = 11 + ReadBits(7). We want 12
                            // zeros, so ReadBits(7) = 1.
        w.write_bits(1, 1); // CLC-sym18 (bit 1)
        w.write_bits(1, 7); // run = 11 + 1 = 12 zeros (symbols 0..11)
                            // Cursor at 12. CLC reads #2 and #3: two literal-len-1 codes for
                            // sym12, sym13.
        w.write_bits(0, 1); // CLC-sym1 → literal len 1 for sym12
        w.write_bits(0, 1); // CLC-sym1 → literal len 1 for sym13
                            // After 3 CLC reads `emitted` hits max_symbol, so the loop stops
                            // with sym14/sym15 at length 0.
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let code = PrefixCode::read(&mut r, 16).unwrap();
        let mut expected = [0u8; 16];
        expected[12] = 1;
        expected[13] = 1;
        assert_eq!(code.code_lengths(), &expected[..]);
    }

    #[test]
    fn normal_repeat_code_16() {
        // CLC code 16 repeats the previous non-zero length 3 + ReadBits(2)
        // times. Build a 6-symbol alphabet with lengths all = 3 (a
        // complete 8-leaf tree needs 8 symbols, so use lengths so the
        // Kraft sum is exactly 1).
        //
        // Lengths target: [2,2,2,2] over 4 symbols (sum 4 * 1/4 = 1).
        // Emit literal len 2 for sym0, then code 16 repeat ×3.
        //
        // CLC needs symbols 2 (literal len 2) and 16 (repeat). Positions
        // pos2 and pos16. Order [17,18,0,1,2,3,4,5,16,...]; pos16 is the
        // 9th entry, so num_code_lengths must be ≥ 9.
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // normal
        w.write_bits(5, 4); // num_code_lengths = 4 + 5 = 9
                            // Order entries 0..8: 17,18,0,1,2,3,4,5,16.
        for pos in [17usize, 18, 0, 1, 2, 3, 4, 5, 16] {
            // pos2 → length 1, pos16 → length 1, everything else 0.
            let l = if pos == 2 || pos == 16 { 1 } else { 0 };
            w.write_bits(l, 3);
        }
        w.write_bits(0, 1); // max_symbol = alphabet_size (4)
                            // CLC: 2 leaves, CLC-sym2 and CLC-sym16, both length 1.
                            // Value order: sym2 → bit 0, sym16 → bit 1.
                            // Emit literal len 2 for sym0 (CLC bit 0).
        w.write_bits(0, 1); // → literal length 2, sym0 = 2
                            // Emit code 16 (CLC bit 1) repeat = 3 + ReadBits(2). Want 3
                            // repeats for sym1,sym2,sym3 → ReadBits(2) = 0.
        w.write_bits(1, 1); // CLC-sym16
        w.write_bits(0, 2); // repeat = 3
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        let code = PrefixCode::read(&mut r, 4).unwrap();
        assert_eq!(code.code_lengths(), &[2, 2, 2, 2]);
    }

    #[test]
    fn normal_max_symbol_too_large_is_refused() {
        let mut w = BitWriter::new();
        w.write_bits(0, 1); // normal
        w.write_bits(0, 4); // num_code_lengths = 4
        for pos in [17usize, 18, 0, 1] {
            let l = if pos == 0 || pos == 1 { 1 } else { 0 };
            w.write_bits(l, 3);
        }
        // use_max = 1, length_nbits = 2 + 2*ReadBits(3); pick ReadBits(3)
        // = 0 → length_nbits = 2; max_symbol = 2 + ReadBits(2). Want a
        // value exceeding alphabet_size 4 → ReadBits(2) = 3 → max = 5.
        w.write_bits(1, 1);
        w.write_bits(0, 3);
        w.write_bits(3, 2);
        let data = w.into_bytes();
        let mut r = BitReader::new(&data);
        match PrefixCode::read(&mut r, 4) {
            Err(PrefixError::MaxSymbolTooLarge {
                max_symbol,
                alphabet_size,
            }) => {
                assert_eq!(max_symbol, 5);
                assert_eq!(alphabet_size, 4);
            }
            other => panic!("expected MaxSymbolTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn read_symbol_reference_matches_fast_path_on_lut_built_code() {
        // A 256-symbol code well above the MIN_LOOKUP_USED gate mixing
        // ≤ 8-bit codes (primary-table hits) and > 8-bit codes (the
        // continuation path). Built programmatically so the Kraft sum
        // is exactly 2^15: start with every symbol at length 10, then
        // shorten symbols from the front (each shortening of a len-L
        // symbol to L-1 adds 2^(15-L) to the sum) until complete —
        // yielding a front of short codes and a long >8-bit tail.
        let mut lengths = vec![10u8; 256];
        let full: u64 = 1 << MAX_CODE_LENGTH;
        let mut kraft: u64 = 256 << (MAX_CODE_LENGTH - 10);
        let mut i = 0usize;
        while kraft < full {
            let l = lengths[i] as usize;
            let gain = 1u64 << (MAX_CODE_LENGTH - l);
            if kraft + gain <= full && l > 1 {
                lengths[i] -= 1;
                kraft += gain;
            } else {
                i += 1;
            }
        }
        let code = PrefixCode::from_code_lengths(lengths).unwrap();

        // Pseudo-random bit soup (LCG), long enough for hundreds of
        // symbols, ending mid-code so the EOF arm is compared too.
        let mut bytes = Vec::with_capacity(257);
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..257 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bytes.push((state >> 33) as u8);
        }
        let mut fast = BitReader::new(&bytes);
        let mut reference = BitReader::new(&bytes);
        loop {
            let a = code.read_symbol(&mut fast);
            let b = code.read_symbol_reference(&mut reference);
            assert_eq!(a, b, "fast path and reference walk diverged");
            assert_eq!(fast.bit_position(), reference.bit_position());
            if a.is_err() {
                break;
            }
        }
    }

    #[test]
    fn truncated_prefix_code_reports_eof() {
        // flag bit only, then EOF.
        let data = [0x00u8]; // one byte
        let mut r = BitReader::new(&data);
        r.read_bits(8).unwrap(); // consume the byte
        match PrefixCode::read(&mut r, 256) {
            Err(PrefixError::Eof(_)) => {}
            other => panic!("expected Eof, got {other:?}"),
        }
    }
}
