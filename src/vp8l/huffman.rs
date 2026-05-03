//! Canonical Huffman trees for VP8L.
//!
//! VP8L's Huffman encoding has two shapes:
//!
//! * **Simple code** — 1 or 2 symbols encoded in 1-3 bits each. Used for
//!   alphabets that collapse to a single literal or a binary choice.
//! * **Normal code** — RFC 1951-style canonical Huffman. First the "code
//!   lengths of the code lengths" alphabet is read with fixed ordering,
//!   then the actual per-symbol code lengths via that meta-tree (with
//!   repeat/zero-run codes), then canonical codes are assembled.
//!
//! The implementation stores the tree as a flat `(left, right)` link
//! vector; each internal node is indexed by a u32 offset from the root.
//! Leaves store the decoded symbol. Decoding is a straight bit-by-bit
//! walk — this is plenty fast for still-image sized alphabets and keeps
//! the code short.

use oxideav_core::{Error, Result};

use super::bit_reader::BitReader;

/// Fixed order in which code lengths for the "meta alphabet" are read
/// (spec §6.2.5).
const CODE_LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// Decoded symbol type. Kept as a plain u16 — VP8L alphabets fit
/// comfortably in 16 bits (largest is ~2072 = 256 + 24 + cache).
pub type HuffmanCode = u16;

/// Bits used to index the primary lookup table in `HuffmanTree`. Larger
/// values trade memory (and one-time build cost) for fewer fall-back
/// tree walks. 8 bits ⇒ a 256-entry × 4-byte = 1 KiB table per tree,
/// which decodes any code of length ≤ 8 in a single load + shift.
///
/// VP8L permits codes up to length 15, so codes of length 9..15 still
/// fall back to the bit-by-bit tree walk. In practice the tree walks
/// are rare: the most-frequent literal-green and length codes nearly
/// always land at lengths ≤ 7, and a 2× bigger LUT (LUT_BITS = 9, 2 KiB
/// per tree) doesn't hit deep enough into the tail to pay back its
/// build cost on small frames.
const LUT_BITS: u8 = 8;
const LUT_SIZE: usize = 1 << LUT_BITS;

/// One LUT entry. Packs (length, symbol) into a u32 to keep the table
/// dense (cache-line friendly):
///   * bits  0..16 — symbol (HuffmanCode = u16, alphabets fit in 16 bits)
///   * bits 16..24 — code length in bits (0 ⇒ "fall back to tree walk")
///
/// Encoding length=0 as the fall-back marker is safe because canonical-
/// Huffman codes always have length ≥ 1.
type LutEntry = u32;

#[inline]
fn lut_pack(symbol: HuffmanCode, length: u8) -> LutEntry {
    ((length as u32) << 16) | (symbol as u32)
}

#[inline]
fn lut_length(e: LutEntry) -> u8 {
    ((e >> 16) & 0xff) as u8
}

#[inline]
fn lut_symbol(e: LutEntry) -> HuffmanCode {
    (e & 0xffff) as HuffmanCode
}

/// A Huffman tree ready for bit-by-bit decode.
///
/// Decode strategy is two-tier:
///
/// 1. **Primary LUT** (`lut`) indexed by `LUT_BITS` peeked bits — one
///    load + shift covers every code of length ≤ `LUT_BITS`. This is
///    the textbook canonical-Huffman speedup; for an alphabet whose
///    most-frequent codes are short (the literal-green / length /
///    cache codes typically are) it eliminates the per-bit branch in
///    the hot path entirely.
/// 2. **Fall-back tree walk** for the rare ≥`LUT_BITS+1`-bit codes —
///    the same flat `nodes` array the original bit-by-bit decoder
///    used. Correctness is verified by the existing 50+ Huffman edge-
///    case tests in `tests/vp8l_huffman_unit.rs` plus the in-module
///    `tests` suite below.
#[derive(Debug)]
pub struct HuffmanTree {
    /// Single-symbol shortcut: if present, every read emits this symbol
    /// (consumes no bits).
    only_symbol: Option<HuffmanCode>,
    /// Flat node array. The root is node 0. Each non-leaf `Node::Internal`
    /// stores indices to its 0/1 children. Leaves store the decoded
    /// symbol. Only consulted when the LUT entry has length 0
    /// (fall-back path).
    nodes: Vec<Node>,
    /// Primary lookup table — indexed by `LUT_BITS` peeked bits in the
    /// bit-reader's LSB-first order. Each entry packs (symbol, length)
    /// per `lut_pack`. Empty for the single-symbol shortcut path.
    lut: Vec<LutEntry>,
}

#[derive(Clone, Copy, Debug)]
enum Node {
    Leaf(HuffmanCode),
    Internal { zero: u32, one: u32 },
}

impl HuffmanTree {
    /// Read a Huffman tree from the bitstream; `alphabet` is the number of
    /// symbols in the alphabet.
    pub fn read(br: &mut BitReader<'_>, alphabet: usize) -> Result<Self> {
        let simple = br.read_bit()?;
        if simple == 1 {
            Self::read_simple(br, alphabet)
        } else {
            Self::read_normal(br, alphabet)
        }
    }

    fn read_simple(br: &mut BitReader<'_>, alphabet: usize) -> Result<Self> {
        let num_symbols = br.read_bit()? + 1; // 1 or 2
        let is_first_8bits = br.read_bit()?;
        let sym0 = br.read_bits(if is_first_8bits != 0 { 8 } else { 1 })? as HuffmanCode;
        // Per spec §3.7.2.1.1, the symbol must be a valid index into the
        // alphabet for the symbol type (e.g. `[0..40)` for distance,
        // `[0..256)` for A/R/B literals). The 1-bit / 8-bit read width is
        // independent of the alphabet, so distance can also produce sym0=255
        // out of a 1-bit field if `is_first_8bits=1` — but per the spec that
        // is an invalid bitstream.
        if (sym0 as usize) >= alphabet {
            return Err(Error::invalid("VP8L: simple huffman symbol out of range"));
        }
        if num_symbols == 1 {
            return Ok(Self {
                only_symbol: Some(sym0),
                nodes: vec![Node::Leaf(sym0)],
                lut: Vec::new(),
            });
        }
        let sym1 = br.read_bits(8)? as HuffmanCode;
        if (sym1 as usize) >= alphabet {
            return Err(Error::invalid("VP8L: simple huffman symbol out of range"));
        }
        // 1-bit: 0 -> sym0, 1 -> sym1. Build the LUT explicitly here
        // rather than going through `build_from_lengths` — the simple
        // 2-symbol shape is already canonical and a length table over
        // the alphabet would be sparse and wasteful (alphabet can be
        // 256+ for a green tree).
        let mut lut = vec![0 as LutEntry; LUT_SIZE];
        let entry0 = lut_pack(sym0, 1);
        let entry1 = lut_pack(sym1, 1);
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = if i & 1 == 0 { entry0 } else { entry1 };
        }
        Ok(Self {
            only_symbol: None,
            nodes: vec![
                Node::Internal { zero: 1, one: 2 },
                Node::Leaf(sym0),
                Node::Leaf(sym1),
            ],
            lut,
        })
    }

    fn read_normal(br: &mut BitReader<'_>, alphabet: usize) -> Result<Self> {
        // Read the code-length-tree's own lengths.
        let num_code_lengths = (br.read_bits(4)? + 4) as usize;
        if num_code_lengths > CODE_LENGTH_ORDER.len() {
            return Err(Error::invalid("VP8L: too many code-length lengths"));
        }
        let mut code_length_code_lengths = [0u8; 19];
        for i in 0..num_code_lengths {
            code_length_code_lengths[CODE_LENGTH_ORDER[i]] = br.read_bits(3)? as u8;
        }
        let meta_tree = build_from_lengths(&code_length_code_lengths)?;

        // Read the per-symbol code lengths, possibly truncated.
        let (max_symbol, use_length) = if br.read_bit()? == 1 {
            // Length-bound mode.
            let length_nbits = 2 + 2 * br.read_bits(3)? as usize;
            let max = 2 + br.read_bits(length_nbits as u8)? as usize;
            // Per spec §3.7.2.1.2: "If max_symbol is larger than the size
            // of the alphabet for the symbol type, the bitstream is
            // invalid." Reject rather than silently clamping.
            if max > alphabet {
                return Err(Error::invalid("VP8L: max_symbol > alphabet"));
            }
            (max, true)
        } else {
            (alphabet, false)
        };

        let mut code_lengths = vec![0u8; alphabet];
        let mut sym = 0usize;
        // Spec §3.7.2.1.2: "If code 16 is used before a nonzero value has
        // been emitted, a value of 8 is repeated." So we initialize
        // `prev_len` to 8 — this is normative, not empirical.
        let mut prev_len = 8u8;
        // `max_symbol` (when use_length is set) caps the number of meta-tree
        // codes decoded — i.e. one increment per `meta_tree.decode()` call
        // regardless of whether the code is a literal length (0..15) or a
        // run code (16/17/18). A run code emits multiple per-position
        // lengths in one go but still counts as a single meta-symbol read,
        // matching the spec's "read up to max_symbol code lengths" wording
        // (§3.7.2.1.2). Treating the run-emitted lengths as separate
        // "symbols" lets the loop overshoot and produces over-determined
        // alphabets that then fail canonical-Huffman assignment with the
        // spurious "self-collides" diagnostic — the bug fixed in 14d7715.
        let mut count = 0usize;
        while sym < alphabet {
            if use_length && count >= max_symbol {
                break;
            }
            let code = meta_tree.decode(br)?;
            count += 1;
            match code {
                0..=15 => {
                    code_lengths[sym] = code as u8;
                    if code != 0 {
                        prev_len = code as u8;
                    }
                    sym += 1;
                }
                16 => {
                    let repeat = 3 + br.read_bits(2)? as usize;
                    if sym + repeat > alphabet {
                        return Err(Error::invalid("VP8L: huffman repeat past alphabet"));
                    }
                    for _ in 0..repeat {
                        code_lengths[sym] = prev_len;
                        sym += 1;
                    }
                }
                17 => {
                    let repeat = 3 + br.read_bits(3)? as usize;
                    if sym + repeat > alphabet {
                        return Err(Error::invalid("VP8L: huffman zero-run past alphabet"));
                    }
                    for _ in 0..repeat {
                        code_lengths[sym] = 0;
                        sym += 1;
                    }
                }
                18 => {
                    let repeat = 11 + br.read_bits(7)? as usize;
                    if sym + repeat > alphabet {
                        return Err(Error::invalid("VP8L: huffman long-zero-run past alphabet"));
                    }
                    for _ in 0..repeat {
                        code_lengths[sym] = 0;
                        sym += 1;
                    }
                }
                _ => return Err(Error::invalid("VP8L: bad code length code")),
            }
        }
        build_from_lengths(&code_lengths)
    }

    /// Build a tree directly from a code-length table. Exposed for the
    /// criterion bench harness (`benches/vp8l_decode.rs`) which needs to
    /// construct trees outside the bitstream parser. Not part of the
    /// stable surface — the wire format is the only supported entry.
    #[doc(hidden)]
    pub fn from_code_lengths_for_bench(lengths: &[u8]) -> Self {
        build_from_lengths(lengths).expect("test/bench-only helper, lengths must be valid")
    }

    /// Decode a single symbol.
    ///
    /// Two-tier strategy:
    ///   1. Refill so ≥ `LUT_BITS` bits are buffered, peek them, and
    ///      look up the (symbol, length) pair in O(1). Consume `length`
    ///      bits and return — covers every code of length ≤ `LUT_BITS`.
    ///   2. If the LUT entry has length 0 (the "fall-back" sentinel —
    ///      the prefix is consistent with multiple distinct codes,
    ///      i.e. the actual code length exceeds `LUT_BITS`), fall back
    ///      to the original bit-by-bit tree walk over `nodes`.
    ///
    /// Past end-of-buffer the bit-reader injects zero bits (libwebp
    /// trailing-zero contract), so peek/refill never errors on EOF —
    /// we just keep decoding into well-defined zero-bit territory and
    /// rely on the outer pixel-loop's pixel count to terminate.
    #[inline]
    pub fn decode(&self, br: &mut BitReader<'_>) -> Result<HuffmanCode> {
        if let Some(s) = self.only_symbol {
            return Ok(s);
        }
        // Primary LUT fast-path. The bit-reader's `read_bits` would also
        // refill, but inlining the refill+peek+consume pair here avoids
        // the conditional-on-n branch and keeps the symbol load close
        // to the bit-shift in the generated code.
        br.refill(LUT_BITS);
        let key = br.peek_bits(LUT_BITS) as usize;
        let entry = self.lut[key];
        let length = lut_length(entry);
        if length != 0 {
            br.consume(length);
            return Ok(lut_symbol(entry));
        }
        // Fall-back: the actual code is longer than LUT_BITS bits.
        // Walk the tree from the root one bit at a time. (We can't
        // start from where the LUT prefix points without storing
        // sub-tree roots in the LUT — a possible future refinement.)
        let mut node = 0u32;
        loop {
            match self.nodes[node as usize] {
                Node::Leaf(s) => return Ok(s),
                Node::Internal { zero, one } => {
                    let b = br.read_bit()?;
                    node = if b == 0 { zero } else { one };
                }
            }
        }
    }
}

/// Build a canonical-Huffman tree from an array of code lengths (one per
/// symbol). Lengths of 0 mean "absent".
fn build_from_lengths(lengths: &[u8]) -> Result<HuffmanTree> {
    // Count symbols by length.
    let mut max_len = 0u8;
    let mut total_nonzero = 0usize;
    let mut lone_symbol: Option<u16> = None;
    for (i, &l) in lengths.iter().enumerate() {
        if l != 0 {
            total_nonzero += 1;
            if l > max_len {
                max_len = l;
            }
            lone_symbol = Some(i as u16);
        }
    }
    if total_nonzero == 0 {
        return Ok(HuffmanTree {
            only_symbol: Some(0),
            nodes: vec![Node::Leaf(0)],
            lut: Vec::new(),
        });
    }
    if total_nonzero == 1 {
        let s = lone_symbol.unwrap_or(0);
        return Ok(HuffmanTree {
            only_symbol: Some(s),
            nodes: vec![Node::Leaf(s)],
            lut: Vec::new(),
        });
    }

    // Canonical Huffman: assign codes in ascending (length, symbol) order.
    let mut bl_count = vec![0u32; (max_len + 1) as usize];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = vec![0u32; (max_len + 1) as usize];
    let mut code = 0u32;
    for bits in 1..=max_len as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Insert each symbol into a flat tree AND populate the primary LUT
    // for codes of length ≤ LUT_BITS. Walk the canonical code MSB-
    // first, allocating internal nodes on demand and then a final leaf.
    //
    // LUT population: the bit-reader returns bits LSB-first (the first
    // emitted bit ends up in bit 0 of the peek). The encoder emits the
    // MSB of `code_val` first, so the LSB-first key prefix is the
    // bit-reverse of `code_val` over `len` bits. We pre-compute that
    // prefix here, then stride through every LUT slot whose low `len`
    // bits match — `1 << (LUT_BITS - len)` slots in total per symbol,
    // each one representing all possible "bits beyond the code" suffixes.
    let mut nodes: Vec<Node> = vec![Node::Internal { zero: 0, one: 0 }];
    let mut lut: Vec<LutEntry> = vec![0 as LutEntry; LUT_SIZE];
    for (sym, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let code_val = next_code[len as usize];
        next_code[len as usize] += 1;

        // Populate the LUT for short codes. (Long codes are served by
        // the tree walk; their LUT slots stay at length=0, the
        // fall-back marker.)
        if len <= LUT_BITS {
            // bit-reverse code_val over `len` bits to get the LSB-first prefix.
            let mut prefix = 0u32;
            for b in 0..len {
                if ((code_val >> b) & 1) != 0 {
                    prefix |= 1u32 << (len - 1 - b);
                }
            }
            let stride = 1usize << len;
            let entry = lut_pack(sym as u16, len);
            let mut k = prefix as usize;
            while k < LUT_SIZE {
                lut[k] = entry;
                k += stride;
            }
        }

        // Build the fall-back tree node-by-node, bit-by-bit MSB-first.
        let mut node = 0u32;
        for b in (0..len).rev() {
            let bit = (code_val >> b) & 1;
            if b == 0 {
                let leaf_idx = nodes.len() as u32;
                nodes.push(Node::Leaf(sym as u16));
                match &mut nodes[node as usize] {
                    Node::Internal { zero, one } => {
                        if bit == 0 {
                            *zero = leaf_idx;
                        } else {
                            *one = leaf_idx;
                        }
                    }
                    Node::Leaf(_) => {
                        return Err(Error::invalid(
                            "VP8L: canonical Huffman length table self-collides",
                        ))
                    }
                }
            } else {
                let child = match nodes[node as usize] {
                    Node::Internal { zero, one } => {
                        if bit == 0 {
                            zero
                        } else {
                            one
                        }
                    }
                    Node::Leaf(_) => {
                        return Err(Error::invalid(
                            "VP8L: canonical Huffman length table self-collides",
                        ))
                    }
                };
                let next = if child == 0 {
                    let new_idx = nodes.len() as u32;
                    nodes.push(Node::Internal { zero: 0, one: 0 });
                    match &mut nodes[node as usize] {
                        Node::Internal { zero, one } => {
                            if bit == 0 {
                                *zero = new_idx;
                            } else {
                                *one = new_idx;
                            }
                        }
                        _ => unreachable!(),
                    }
                    new_idx
                } else {
                    child
                };
                node = next;
            }
        }
    }
    Ok(HuffmanTree {
        only_symbol: None,
        nodes,
        lut,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────────────────────────────────────────────────────────
    // build_from_lengths — direct unit tests on the canonical-Huffman
    // assembler. These exercise every shape the function distinguishes:
    //   * all-zero (returns Leaf(0))
    //   * single-symbol (returns lone-symbol shortcut)
    //   * Kraft-equality (the common "fully tiles the codespace" case)
    //   * Kraft-under-equality (sparse but still well-formed)
    //   * Kraft-over-equality (must error)
    //   * Length-15 (per-symbol max per spec)
    //   * Length-1 (the 2-symbol fast tree)
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn canonical_two_symbols() {
        // Two symbols of length 1 each: 0 -> sym0, 1 -> sym1.
        let tree = build_from_lengths(&[1, 1]).unwrap();
        let buf = [0b0000_0010u8];
        let mut br = BitReader::new(&buf);
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
        assert_eq!(tree.decode(&mut br).unwrap(), 1);
    }

    #[test]
    fn build_all_zeros_returns_leaf0() {
        // All-zero length table → degenerate tree returning symbol 0
        // unconditionally. Spec §3.7.2.1.1 calls out this empty-prefix
        // case explicitly ("can be coded as those containing a single
        // symbol 0").
        let tree = build_from_lengths(&[0u8; 40]).unwrap();
        let buf = [0u8];
        let mut br = BitReader::new(&buf);
        // Should consume zero bits and emit symbol 0.
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
        assert_eq!(br.byte_pos(), 0);
        // Repeated reads keep yielding 0 (single-symbol shortcut).
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
    }

    #[test]
    fn build_one_nonzero_returns_lone_symbol() {
        // Exactly one symbol present (at index 17, length 1) → degenerate
        // single-symbol tree consuming zero bits.
        let mut lens = vec![0u8; 40];
        lens[17] = 1;
        let tree = build_from_lengths(&lens).unwrap();
        let buf = [0u8];
        let mut br = BitReader::new(&buf);
        assert_eq!(tree.decode(&mut br).unwrap(), 17);
        assert_eq!(br.byte_pos(), 0);
    }

    #[test]
    fn build_one_nonzero_with_long_length_still_lone() {
        // Single non-zero entry, even at length 15, collapses to the
        // lone-symbol shortcut (no bits consumed on decode).
        let mut lens = vec![0u8; 256];
        lens[200] = 15;
        let tree = build_from_lengths(&lens).unwrap();
        let buf = [0u8];
        let mut br = BitReader::new(&buf);
        assert_eq!(tree.decode(&mut br).unwrap(), 200);
        assert_eq!(br.byte_pos(), 0);
    }

    #[test]
    fn build_kraft_equality_three_symbols() {
        // Lengths 1, 2, 2 → Kraft sum = 1/2 + 1/4 + 1/4 = 1.0 (equality).
        // Canonical assignment: sym0 = "0", sym1 = "10", sym2 = "11".
        let tree = build_from_lengths(&[1, 2, 2]).unwrap();
        // The decoder walks the tree MSB-first per canonical encoding (the
        // first bit decoded is the top bit of the canonical code). The
        // BitReader is LSB-first within bytes, so the bit at byte0[0] is
        // the first bit emitted by the encoder.
        //   sym0 "0"  → read order [0]
        //   sym1 "10" → read order [1, 0]
        //   sym2 "11" → read order [1, 1]
        // Concatenated read-order bits: 0, 1, 0, 1, 1.
        // Byte 0 LSB→MSB: 0,1,0,1,1,_,_,_ → 0b0001_1010 = 0x1A.
        let buf = [0x1Au8];
        let mut br = BitReader::new(&buf);
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
        assert_eq!(tree.decode(&mut br).unwrap(), 1);
        assert_eq!(tree.decode(&mut br).unwrap(), 2);
    }

    #[test]
    fn build_kraft_under_equality_is_accepted() {
        // Lengths 2, 2 → Kraft sum = 1/4 + 1/4 = 0.5 (under-equality).
        // The codespace isn't fully tiled, but the two listed codes are
        // still unambiguously decodable. Canonical: sym0 = "00", sym1 = "01".
        let tree = build_from_lengths(&[2, 2]).unwrap();
        // Read order: sym0 "00" → 0,0; sym1 "01" → 0,1.
        // Bits (LSB-first): 0,0,0,1,_,_,_,_ → 0b0000_1000 = 0x08.
        let buf = [0x08u8];
        let mut br = BitReader::new(&buf);
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
        assert_eq!(tree.decode(&mut br).unwrap(), 1);
    }

    #[test]
    fn build_kraft_over_equality_errors() {
        // Lengths 1, 1, 2 → Kraft = 1/2 + 1/4 = 1.25 > 1. The two
        // length-1 codes fill both root children with leaves; the
        // length-2 code's first bit walks into one of those leaves
        // (where build_from_lengths expects an Internal) and the
        // intermediate-walk match-arm raises "self-collides".
        //
        // Note: not every Kraft-over case is caught — see
        // `build_kraft_over_equality_three_length_one_silently_truncates`.
        let err = build_from_lengths(&[1, 1, 2]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("self-collides"),
            "expected self-collide error, got {msg}"
        );
    }

    #[test]
    fn build_kraft_over_equality_three_length_one_silently_truncates() {
        // Documenting current behaviour: [1, 1, 1] has Kraft = 3/2 > 1
        // but the current build_from_lengths silently overwrites the
        // first sym-0 leaf with sym 2 (both at canonical code 0). This
        // is technically invalid bitstream content per spec, but the
        // current detector doesn't catch it. Pin the behaviour so a
        // future tightening of the validator surfaces as a test diff.
        //
        // TODO: tighten build_from_lengths to reject any Kraft sum > 1.
        let tree = build_from_lengths(&[1, 1, 1]).expect("currently accepted");
        let buf = [0u8];
        let mut br = BitReader::new(&buf);
        // First-bit-zero ends up as sym 2 (last assigned), not sym 0.
        let s = tree.decode(&mut br).unwrap();
        assert!(s == 0 || s == 2, "got sym {s}");
    }

    #[test]
    fn build_length_15_max_per_spec() {
        // Spec sets the per-symbol code-length max at 15 (Code 0..15 in
        // §3.7.2.1.2). Build a tree where one code uses the maximum.
        // Lengths: [1, 2, 3, ..., 15, 15] — Kraft sum should be 1.
        let mut lens = vec![0u8; 16];
        for (i, l) in (1u8..=15).enumerate() {
            lens[i] = l;
        }
        // Add a duplicate length-15 so Kraft = 1/2 + 1/4 + ... + 1/(2^15) +
        // 1/(2^15) = 1 (geometric).
        lens[15] = 15;
        // Sum check: 1 - (1/2^15) + (1/2^15) = 1. Builds.
        let tree = build_from_lengths(&lens).expect("length-15 tree should build");
        // Smoke-test: feed plenty of zero bits — first emit must be sym0
        // (length 1, code "0").
        let buf = [0u8; 4];
        let mut br = BitReader::new(&buf);
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
    }

    #[test]
    fn build_length_1_two_symbol_fast_tree() {
        // The canonical 1-bit tree: sym A on '0', sym B on '1'.
        let tree = build_from_lengths(&[1, 1]).unwrap();
        let buf = [0b1010_1010u8]; // alternating bits, LSB-first
        let mut br = BitReader::new(&buf);
        // LSB-first → 0,1,0,1,0,1,0,1
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
        assert_eq!(tree.decode(&mut br).unwrap(), 1);
        assert_eq!(tree.decode(&mut br).unwrap(), 0);
        assert_eq!(tree.decode(&mut br).unwrap(), 1);
    }

    // ────────────────────────────────────────────────────────────────────
    // read_simple — these exercise the simple-Huffman parser at the API
    // boundary. Each test crafts the exact bit pattern described in
    // spec §3.7.2.1.1 and asserts the resulting tree.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn simple_one_symbol() {
        // Simple encoding: bit0=1 (simple), bit1=0 (num_symbols=1),
        // bit2=0 (is_first_8bits=0), bit3=1 (sym0 in 1-bit field).
        let buf = [0b0000_1001u8];
        let mut br = BitReader::new(&buf);
        let tree = HuffmanTree::read(&mut br, 256).unwrap();
        // Single-symbol tree — any read returns 1.
        assert_eq!(tree.decode(&mut br).unwrap(), 1);
    }

    #[test]
    fn simple_one_symbol_8bit_field() {
        // bit0=1 (simple), bit1=0 (num_syms=1), bit2=1 (is_first_8bits=1),
        // bits 3..10 = 0xa5.
        // Stream LSB-first: 1, 0, 1, then 8 bits of 0xa5 (= 1010_0101 LSB
        // first → bit values 1,0,1,0,0,1,0,1).
        // Bits laid out byte-by-byte:
        //   b0=1,b1=0,b2=1,b3=1,b4=0,b5=1,b6=0,b7=0  → byte 0 = 0010_1101 = 0x2d
        //   b8=1,b9=0,b10=1                          → byte 1 = 0000_0101 = 0x05
        let buf = [0x2du8, 0x05];
        let mut br = BitReader::new(&buf);
        let tree = HuffmanTree::read(&mut br, 256).unwrap();
        assert_eq!(tree.decode(&mut br).unwrap(), 0xa5);
    }

    #[test]
    fn simple_two_symbols() {
        // bit0=1 simple, bit1=1 num_syms=2, bit2=1 8-bit, bits 3..10=sym0=10,
        // bits 11..18=sym1=20. Then 1 bit on decode picks sym0/sym1.
        let mut bw = TestBitWriter::new();
        bw.write(1, 1); // simple
        bw.write(1, 1); // num_syms = 2
        bw.write(1, 1); // is_first_8bits
        bw.write(10, 8); // sym0
        bw.write(20, 8); // sym1
        bw.write(0, 1); // pick sym0
        bw.write(1, 1); // pick sym1
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let tree = HuffmanTree::read(&mut br, 256).unwrap();
        assert_eq!(tree.decode(&mut br).unwrap(), 10);
        assert_eq!(tree.decode(&mut br).unwrap(), 20);
    }

    #[test]
    fn simple_two_symbols_degenerate_duplicate() {
        // Spec note: "Duplicate symbols are allowed, but inefficient." We
        // accept them — the tree still builds, and reads are degenerate
        // (both branches lead to the same value).
        // build_from_lengths sees two non-zero lengths at the same index,
        // which collapses to one (since we walk by index). In practice the
        // simple-Huffman parser builds the tree directly without going
        // through build_from_lengths, so each branch is its own leaf.
        let mut bw = TestBitWriter::new();
        bw.write(1, 1); // simple
        bw.write(1, 1); // num_syms = 2
        bw.write(1, 1); // 8-bit
        bw.write(42, 8); // sym0 = 42
        bw.write(42, 8); // sym1 = 42 (duplicate)
        bw.write(0, 1);
        bw.write(1, 1);
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let tree = HuffmanTree::read(&mut br, 256).unwrap();
        assert_eq!(tree.decode(&mut br).unwrap(), 42);
        assert_eq!(tree.decode(&mut br).unwrap(), 42);
    }

    #[test]
    fn simple_one_symbol_distance_alphabet_in_range() {
        // num_syms=1, 1-bit field, sym0=1 → valid for distance (alphabet=40).
        let mut bw = TestBitWriter::new();
        bw.write(1, 1); // simple
        bw.write(0, 1); // num_syms = 1
        bw.write(0, 1); // is_first_8bits = 0 → 1-bit symbol
        bw.write(1, 1); // sym0 = 1
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let tree = HuffmanTree::read(&mut br, 40).unwrap();
        assert_eq!(tree.decode(&mut br).unwrap(), 1);
    }

    #[test]
    fn simple_one_symbol_distance_alphabet_8bit_out_of_range_errors() {
        // Regression for round-2 deferred bug: distance alphabet has 40
        // symbols, but the simple-Huffman parser used to silently accept
        // any sym0 < max(alphabet, 256), letting sym0 = 100 through into a
        // distance tree. That value can never be a legal distance code
        // (codes are [0..40)) — the bitstream is malformed.
        let mut bw = TestBitWriter::new();
        bw.write(1, 1); // simple
        bw.write(0, 1); // num_syms = 1
        bw.write(1, 1); // is_first_8bits = 1 → 8-bit symbol
        bw.write(100, 8); // sym0 = 100, out of [0..40)
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let err = HuffmanTree::read(&mut br, 40).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("out of range"),
            "expected out-of-range error, got {msg}"
        );
    }

    #[test]
    fn simple_two_symbols_sym1_out_of_range_errors() {
        // sym1 must be < alphabet. Build a tree with alphabet=40 and
        // sym1=200 — should error.
        let mut bw = TestBitWriter::new();
        bw.write(1, 1); // simple
        bw.write(1, 1); // num_syms = 2
        bw.write(1, 1); // 8-bit
        bw.write(0, 8); // sym0 = 0
        bw.write(200, 8); // sym1 = 200, out of range
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let err = HuffmanTree::read(&mut br, 40).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("out of range"), "got {msg}");
    }

    // ────────────────────────────────────────────────────────────────────
    // read_normal — uses HuffmanTree::read which dispatches on the first
    // bit (simple=0 → normal). Tests construct full normal-Huffman bit
    // streams via the TestBitWriter helper below.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn normal_max_symbol_greater_than_alphabet_errors() {
        // Per spec §3.7.2.1.2: "If max_symbol is larger than the size of
        // the alphabet for the symbol type, the bitstream is invalid."
        // We previously silently clamped via `.min(alphabet)`; this test
        // pins the corrected behaviour.
        let mut bw = TestBitWriter::new();
        bw.write(0, 1); // not simple → normal
        bw.write(0, 4); // num_code_lengths_minus4 = 0 → 4 entries in CODE_LENGTH_ORDER
                        // First 4 entries: 17, 18, 0, 1. Set length 1 for code 0
                        // (literal-zero) only — degenerate single-symbol meta-tree.
        bw.write(0, 3); // CODE_LENGTH_ORDER[0]=17 → length 0
        bw.write(0, 3); // CODE_LENGTH_ORDER[1]=18 → length 0
        bw.write(1, 3); // CODE_LENGTH_ORDER[2]=0  → length 1
        bw.write(0, 3); // CODE_LENGTH_ORDER[3]=1  → length 0
        bw.write(1, 1); // use_length = 1
        bw.write(0, 3); // length_nbits = 2 + 2*0 = 2
        bw.write(3, 2); // max = 2 + 3 = 5
                        // Alphabet size we'll pass to read() = 4, so 5 > 4 → must error.
        let buf = bw.finish();
        let mut br = BitReader::new(&buf);
        let err = HuffmanTree::read(&mut br, 4).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("max_symbol") || msg.to_lowercase().contains("invalid"),
            "expected max_symbol error, got {msg}"
        );
    }

    #[test]
    fn build_lengths_in_distance_alphabet_size() {
        // 40-symbol alphabet with one entry → lone-symbol shortcut at the
        // distance alphabet size. Verifies build_from_lengths handles
        // alphabets that aren't powers of two and don't equal 256.
        let mut lens = vec![0u8; 40];
        lens[39] = 1;
        let tree = build_from_lengths(&lens).unwrap();
        let buf = [0u8];
        let mut br = BitReader::new(&buf);
        assert_eq!(tree.decode(&mut br).unwrap(), 39);
    }

    #[test]
    fn build_lengths_full_two_codes_at_distance_alphabet_size() {
        // 40-symbol alphabet with two equal-length codes — exercises the
        // standard 1-bit canonical case at a non-power-of-two alphabet.
        let mut lens = vec![0u8; 40];
        lens[5] = 1;
        lens[37] = 1;
        let tree = build_from_lengths(&lens).unwrap();
        let buf = [0b1010_1010u8];
        let mut br = BitReader::new(&buf);
        // bit 0 (= 0) → sym0 = 5; bit 1 (= 1) → sym1 = 37.
        assert_eq!(tree.decode(&mut br).unwrap(), 5);
        assert_eq!(tree.decode(&mut br).unwrap(), 37);
        assert_eq!(tree.decode(&mut br).unwrap(), 5);
        assert_eq!(tree.decode(&mut br).unwrap(), 37);
    }

    // ────────────────────────────────────────────────────────────────────
    // Local LSB-first bit writer mirroring the public BitReader. Kept
    // private to this test mod so we don't expose it; the integration
    // tests have their own copy.
    // ────────────────────────────────────────────────────────────────────
    struct TestBitWriter {
        out: Vec<u8>,
        cur: u32,
        nbits: u32,
    }
    impl TestBitWriter {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                cur: 0,
                nbits: 0,
            }
        }
        fn write(&mut self, value: u32, n: u32) {
            debug_assert!(n <= 24);
            let mask = ((1u64 << n) - 1) as u32;
            self.cur |= (value & mask) << self.nbits;
            self.nbits += n;
            while self.nbits >= 8 {
                self.out.push((self.cur & 0xff) as u8);
                self.cur >>= 8;
                self.nbits -= 8;
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                self.out.push((self.cur & 0xff) as u8);
            }
            self.out
        }
    }
}
