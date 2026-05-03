//! Unit tests for VP8L Huffman parsing — exercising the surface area
//! that the just-fixed `max_symbol`/RLE-code interaction lives on, plus
//! adjacent edge cases that the same fuzzer family is likely to probe
//! next. Each test builds a minimal bit stream by hand and decodes it
//! through `HuffmanTree::read` (the public entry point that dispatches
//! to `read_simple` / `read_normal` based on the leading bit).
//!
//! These tests deliberately keep the bit-stream constructions
//! declarative — `build_normal_with_decode_bits`, `write_meta_lens`,
//! `write_meta_code` and `canonical_codes` let new tests be written
//! as "build a tree with these per-symbol lengths, then decode this
//! sequence of meta-symbols". Hand-editing bit fields per test was
//! the original blocker to broader coverage.

use oxideav_webp::vp8l::bit_reader::BitReader;
use oxideav_webp::vp8l::huffman::HuffmanTree;

/// LSB-first bit writer matching the VP8L bit reader's wire convention.
///
/// Identical in shape to the writer in `tests/vp8l_handcrafted.rs`; kept
/// local so each test file is self-contained.
pub(crate) struct BitWriter {
    out: Vec<u8>,
    cur: u32,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    /// Append `n` (≤24) bits of `value` LSB-first to the stream.
    pub fn write(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 24);
        let mask = if n == 0 {
            0u32
        } else {
            ((1u64 << n) - 1) as u32
        };
        self.cur |= (value & mask) << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.out.push((self.cur & 0xff) as u8);
            self.cur >>= 8;
            self.nbits -= 8;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.cur & 0xff) as u8);
        }
        self.out
    }
}

/// Fixed code-length-code order from spec §3.7.2.1.2.
const CODE_LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// Emit the meta-tree header for a normal Huffman tree where the only
/// non-zero meta-symbol lengths are at positions in `meta_lens` (indexed
/// by raw meta-code, NOT by `CODE_LENGTH_ORDER` slot). The first slot
/// not used in `CODE_LENGTH_ORDER[..num_code_lengths]` is implicitly
/// length 0.
///
/// Caller must already have written the leading `0` bit indicating
/// "not simple Huffman".
pub(crate) fn write_meta_lens(bw: &mut BitWriter, meta_lens: &[u8; 19]) {
    // Find the highest-index used slot in CODE_LENGTH_ORDER.
    let mut last_used = 0usize;
    for (slot, &meta_code) in CODE_LENGTH_ORDER.iter().enumerate() {
        if meta_lens[meta_code] != 0 {
            last_used = slot + 1;
        }
    }
    let num_code_lengths = last_used.max(4);
    bw.write((num_code_lengths - 4) as u32, 4);
    for &meta_code in CODE_LENGTH_ORDER.iter().take(num_code_lengths) {
        bw.write(meta_lens[meta_code] as u32, 3);
    }
}

/// Compute canonical-Huffman codes for a length array. Returns
/// (codes, lens) — both 19-element arrays for the meta alphabet.
/// Codes are MSB-first natural integers; the test must MSB-first walk
/// them when emitting.
pub(crate) fn canonical_codes(lens: &[u8]) -> Vec<u32> {
    let max_len = *lens.iter().max().unwrap_or(&0);
    let mut codes = vec![0u32; lens.len()];
    if max_len == 0 {
        return codes;
    }
    let mut bl_count = vec![0u32; max_len as usize + 1];
    for &l in lens {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = vec![0u32; max_len as usize + 1];
    let mut code = 0u32;
    for bits in 1..=max_len as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (sym, &l) in lens.iter().enumerate() {
        if l > 0 {
            codes[sym] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
    }
    codes
}

/// Emit the canonical bit pattern for `meta_code` using the meta-tree
/// with lengths/codes computed from `meta_lens`. The bits are written
/// MSB-first as the canonical-Huffman walk dictates (top bit first =
/// first read).
pub(crate) fn write_meta_code(
    bw: &mut BitWriter,
    meta_lens: &[u8; 19],
    meta_codes: &[u32],
    meta_code: usize,
) {
    let l = meta_lens[meta_code];
    let code = meta_codes[meta_code];
    assert!(l > 0, "meta_code {meta_code} has zero length");
    for i in (0..l).rev() {
        let bit = (code >> i) & 1;
        bw.write(bit, 1);
    }
}

/// Common helper used by tests that need to (a) emit a normal Huffman
/// tree via the uniform-meta-tree helper and then (b) immediately
/// follow it with explicit decode bits — without worrying about byte
/// alignment between the two halves. With `decode_bits = []` this is
/// just the tree on its own.
fn build_normal_with_decode_bits(lens: &[u8], decode_bits: &[u32]) -> Vec<u8> {
    // Use a meta-tree that gives every literal code length 0..15 a
    // 6-bit code (Kraft = 16/64 = 0.25, valid). Wasteful but uniformly
    // easy to encode without thinking about per-test code-length
    // assignment. build_from_lengths accepts under-Kraft tables.
    let mut meta_lens = [0u8; 19];
    meta_lens[..16].fill(6);
    let meta_codes = canonical_codes(&meta_lens);
    let mut bw = BitWriter::new();
    bw.write(0, 1); // not simple
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(0, 1); // use_length = 0
    for &l in lens {
        write_meta_code(&mut bw, &meta_lens, &meta_codes, l as usize);
    }
    for &b in decode_bits {
        bw.write(b, 1);
    }
    bw.finish()
}

// ──────────────────────────────────────────────────────────────────────
// Tests for read_normal — the bug class that 14d7715 fixed.
// ──────────────────────────────────────────────────────────────────────

#[test]
fn normal_uniform_lengths_roundtrip() {
    // Build a simple 4-symbol tree by emitting per-symbol literal lengths
    // through the helper, then decode each canonical code.
    // Lengths [2,2,2,2] → 4 codes at length 2, Kraft = 1.
    // Canonical: sym0="00", sym1="01", sym2="10", sym3="11".
    // Decode in order 0,1,2,3 → bit stream (read order):
    //   0,0, 0,1, 1,0, 1,1
    let blob = build_normal_with_decode_bits(&[2, 2, 2, 2], &[0, 0, 0, 1, 1, 0, 1, 1]);
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, 4).expect("tree should build");
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
    assert_eq!(tree.decode(&mut br).unwrap(), 1);
    assert_eq!(tree.decode(&mut br).unwrap(), 2);
    assert_eq!(tree.decode(&mut br).unwrap(), 3);
}

#[test]
fn normal_full_alphabet_no_max_symbol() {
    // Decode a 256-literal tree where the only present symbols are
    // 0 and 1 (each at length 1) — `use_length=0` so the helper writes
    // one meta-code per alphabet entry. This is the "no length-bound"
    // case at full alphabet scale.
    let mut lens = vec![0u8; 256];
    lens[0] = 1;
    lens[1] = 1;
    let blob = build_normal_with_decode_bits(&lens, &[0, 1, 0, 1]);
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, 256).expect("tree should build");
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
    assert_eq!(tree.decode(&mut br).unwrap(), 1);
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
    assert_eq!(tree.decode(&mut br).unwrap(), 1);
}

#[test]
fn normal_max_symbol_with_run_code_17_first() {
    // Emit a tree via the use_length path where the first emitted
    // meta-code is a run-17 (zero-run [3..10]). Per spec §3.7.2.1.2 a
    // run-17 produces multiple zero lengths but counts as ONE meta-symbol
    // toward `max_symbol`. With max_symbol=2 and the first code being
    // 17(repeat=3) followed by literal length 1, we expect:
    //   lengths = [0, 0, 0, 1, 0, 0, 0, ...]
    // and a tree that decodes to symbol 3 only.
    let mut meta_lens = [0u8; 19];
    meta_lens[17] = 1;
    meta_lens[1] = 1; // literal-code-length 1 — produces a 1-bit code for symbol 3.
    let meta_codes = canonical_codes(&meta_lens);

    let alphabet = 8usize;
    let max_symbol = 2usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1); // not simple
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(1, 1); // use_length = 1
    bw.write(0, 3); // length_nbits raw = 0 → length_nbits = 2
    bw.write((max_symbol - 2) as u32, 2); // max = 2
                                          // Body: meta-code 17 (3 zeros) then meta-code 1 (length 1 at sym 3).
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 17);
    bw.write(0, 3); // 17's extra: repeat = 3 + 0 = 3 → fills syms 0,1,2 with zero
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 1);
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, alphabet).expect("normal tree should build");
    // The resulting alphabet has a single non-zero entry at sym 3 → tree
    // is a lone-symbol shortcut returning 3.
    assert_eq!(tree.decode(&mut br).unwrap(), 3);
}

#[test]
fn normal_max_symbol_2_with_code_17_only() {
    // max_symbol = 2 with first code = 17 (3-zero run). Per the corrected
    // counting, after the single run-17 the loop sees count=1 < 2, so it
    // pulls one more code; we send a literal-length 0 to give a clean
    // empty alphabet. Without the fix in 14d7715 the loop counted run
    // codes by the number of positions filled (3), exceeded max_symbol,
    // and bailed early — producing a different alphabet than intended.
    //
    // For this test we expect the resulting tree to be the all-zero
    // alphabet → degenerate single-symbol-0 shortcut.
    let mut meta_lens = [0u8; 19];
    meta_lens[17] = 1;
    meta_lens[0] = 1;
    let meta_codes = canonical_codes(&meta_lens);

    let alphabet = 16usize;
    let max_symbol = 2usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(1, 1); // use_length = 1
    bw.write(0, 3); // length_nbits = 2
    bw.write((max_symbol - 2) as u32, 2);
    // Body: 17(repeat=3) then literal-0 (sym 3 = 0).
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 17);
    bw.write(0, 3); // repeat = 3
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 0); // literal length 0 at sym 3
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, alphabet).expect("normal tree should build");
    // Empty alphabet — every decode returns 0 without consuming bits.
    let zero_buf = [0u8; 4];
    let mut br2 = BitReader::new(&zero_buf);
    assert_eq!(tree.decode(&mut br2).unwrap(), 0);
    assert_eq!(br2.byte_pos(), 0);
}

#[test]
fn normal_max_symbol_3_with_code_18_long_run_then_one() {
    // max_symbol=3 with sequence: literal-1 (sym0=1), 18(repeat=11), then
    // one final literal-1 (sym12=1). Without the fix, run-18 would have
    // counted as 11 and overshot max_symbol=3; with the fix it counts as
    // 1, leaving room for the final literal.
    //
    // Resulting alphabet: lens[0]=1, lens[1..12]=0, lens[12]=1 → two
    // length-1 codes → standard 1-bit tree returning 0 on bit 0 and 12
    // on bit 1.
    let mut meta_lens = [0u8; 19];
    meta_lens[18] = 2;
    meta_lens[1] = 2;
    // Need the meta-tree to be valid. Two codes at length 2 = Kraft 1/2
    // (under-equality, accepted).
    let meta_codes = canonical_codes(&meta_lens);

    let alphabet = 16usize;
    let max_symbol = 3usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(1, 1); // use_length = 1
    bw.write(0, 3); // length_nbits = 2
    bw.write((max_symbol - 2) as u32, 2);
    // Body: literal-1, 18(repeat=11), literal-1.
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 1);
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 18);
    bw.write(0, 7); // repeat = 11 + 0 = 11 → fills syms 1..12 with 0
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 1);
    // Append decoder bits: 0 → sym0; 1 → sym12; 0 → sym0.
    bw.write(0, 1);
    bw.write(1, 1);
    bw.write(0, 1);
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, alphabet).expect("normal tree should build");
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
    assert_eq!(tree.decode(&mut br).unwrap(), 12);
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
}

#[test]
fn normal_run_code_18_overshooting_alphabet_errors() {
    // Code 18 with a repeat that pushes past `alphabet` must error
    // ("VP8L: huffman long-zero-run past alphabet"). Use a small
    // alphabet (4) and a max-length (138) run.
    //
    // Setup: meta-tree contains two codes (18 and 0, each length 1) so
    // build_from_lengths produces a real bit-consuming tree (not a
    // lone-symbol shortcut). Code 18 = canonical "0", code 0 = canonical
    // "1" — see canonical_codes.
    let mut meta_lens = [0u8; 19];
    meta_lens[18] = 1;
    meta_lens[0] = 1;
    let _meta_codes = canonical_codes(&meta_lens);

    let alphabet = 4usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(0, 1); // use_length = 0 → max_symbol = alphabet
                    // First meta-code: code 18 → canonical "1" (write_meta_code emits MSB-
                    // first; we hand-emit because the helper picks the wrong-direction bit).
                    // Actually canonical_codes for [1, 0, 0, ...0_at_18=1] gives:
                    //   bl_count[1] = 2 → next_code[1] = 0
                    //   sym 0 (smallest): code 0 → "0"
                    //   sym 18           : code 1 → "1"
                    // So decoding bit 0 → meta-code 0 (literal length 0), bit 1 → meta-code 18.
                    // We want meta-code 18 first → write bit 1.
    bw.write(1, 1);
    bw.write(127, 7); // repeat = 11 + 127 = 138, far past alphabet=4
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let err = HuffmanTree::read(&mut br, alphabet).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("past alphabet") || msg.to_lowercase().contains("zero-run"),
        "got {msg}"
    );
}

#[test]
fn normal_run_code_17_overshooting_alphabet_errors() {
    // Code 17 with repeat that overshoots. Same two-code meta-tree shape
    // as the 18-overshoot test so the meta-tree consumes a bit per
    // decode rather than collapsing to a shortcut.
    let mut meta_lens = [0u8; 19];
    meta_lens[17] = 1;
    meta_lens[0] = 1;
    let _meta_codes = canonical_codes(&meta_lens);
    // canonical_codes for sym 0 (length 1) → "0"; sym 17 (length 1) → "1".
    let alphabet = 4usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(0, 1);
    bw.write(1, 1); // bit 1 → meta-code 17
    bw.write(7, 3); // repeat = 3 + 7 = 10, past alphabet=4
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let err = HuffmanTree::read(&mut br, alphabet).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.to_lowercase().contains("zero-run") || msg.contains("past"),
        "got {msg}"
    );
}

#[test]
fn normal_run_code_16_before_nonzero_uses_8() {
    // Per spec §3.7.2.1.2: "If code 16 is used before a nonzero value
    // has been emitted, a value of 8 is repeated." Build a tree that
    // starts with a run-16 and verify the resulting lengths are [8;3]
    // by checking the Kraft-impossible state (3 codes of length 8 has
    // Kraft = 3/256 — valid, but build_from_lengths produces a 3-symbol
    // tree all at length 8). We then decode to confirm.
    let mut meta_lens = [0u8; 19];
    meta_lens[16] = 1;
    meta_lens[0] = 1;
    let meta_codes = canonical_codes(&meta_lens);

    let alphabet = 4usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(0, 1); // no max_symbol
                    // Body: 16(repeat=3) then literal-0 (one zero to fill alphabet=4).
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 16);
    bw.write(0, 2); // repeat = 3 + 0 = 3
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 0); // sym3 = 0
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, alphabet).expect("normal tree should build");
    // Lengths should be [8, 8, 8, 0]. Canonical codes: sym0="00000000",
    // sym1="00000001", sym2="00000010". Decode all-zero bits → sym0.
    let zero_buf = [0u8; 8];
    let mut br2 = BitReader::new(&zero_buf);
    assert_eq!(tree.decode(&mut br2).unwrap(), 0);
}

#[test]
fn normal_use_length_max_symbol_equals_alphabet() {
    // Edge case: use_length=1 with max_symbol == alphabet. Should behave
    // identically to use_length=0. Build the bit stream directly so the
    // decode bits are bit-aligned with the tree body.
    let mut meta_lens = [0u8; 19];
    meta_lens[..16].fill(6);
    let meta_codes = canonical_codes(&meta_lens);

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(1, 1); // use_length = 1
    bw.write(0, 3); // length_nbits = 2
    bw.write(2u32, 2); // max_symbol - 2 = 2 → max = 4 = alphabet
    let lens = [1u8, 1u8, 0, 0];
    for &l in &lens {
        write_meta_code(&mut bw, &meta_lens, &meta_codes, l as usize);
    }
    bw.write(0, 1); // decode → sym 0
    bw.write(1, 1); // decode → sym 1
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, 4).expect("normal tree should build");
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
    assert_eq!(tree.decode(&mut br).unwrap(), 1);
}

#[test]
fn normal_max_symbol_one_emits_single_length() {
    // max_symbol=2 (the minimum) with one literal length followed by
    // implicit zeros for the rest of the alphabet. Lone-symbol tree.
    let mut meta_lens = [0u8; 19];
    meta_lens[1] = 2;
    meta_lens[0] = 2;
    meta_lens[17] = 2;
    let meta_codes = canonical_codes(&meta_lens);
    let alphabet = 8usize;
    let max_symbol = 2usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(1, 1); // use_length = 1
    bw.write(0, 3); // length_nbits = 2
    bw.write((max_symbol - 2) as u32, 2);
    // Two meta-symbols (max_symbol=2): literal-1 at sym0, then literal-0
    // at sym1. The remainder of the alphabet (sym 2..8) is implicitly 0.
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 1);
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 0);
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, alphabet).expect("normal tree should build");
    // Single non-zero length → lone-symbol shortcut at sym 0.
    let zero_buf = [0u8; 4];
    let mut br2 = BitReader::new(&zero_buf);
    assert_eq!(tree.decode(&mut br2).unwrap(), 0);
    assert_eq!(br2.byte_pos(), 0);
}

#[test]
fn normal_meta_tree_kraft_over_equality_caught_when_walk_hits_leaf() {
    // build_from_lengths catches Kraft over-equality only when a longer
    // canonical code's walk lands on an already-assigned leaf at an
    // intermediate depth. Construct a meta-tree where three length-1
    // codes (at low alphabet indices) cause the third length-1 code to
    // overwrite the second leaf at root.zero — and then a length-2
    // code at alphabet index 17 walks into root.one, which is now a
    // Leaf, raising "self-collides".
    //
    // Meta_lens layout: indices [0,1,2]=1, [17]=2.
    // CODE_LENGTH_ORDER slot mapping: code 0→slot 2, code 1→slot 3,
    // code 2→slot 4, code 17→slot 0.
    // → write num_code_lengths=5 (raw 1), slot bits: 2,0,1,1,1.
    let mut bw = BitWriter::new();
    bw.write(0, 1); // not simple
    bw.write(1, 4); // num_code_lengths = 5
    bw.write(2, 3); // slot 0 = code 17 → len 2
    bw.write(0, 3); // slot 1 = code 18 → len 0
    bw.write(1, 3); // slot 2 = code 0  → len 1
    bw.write(1, 3); // slot 3 = code 1  → len 1
    bw.write(1, 3); // slot 4 = code 2  → len 1
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let err = HuffmanTree::read(&mut br, 32).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("self-collides"), "got {msg}");
}

#[test]
fn normal_meta_tree_single_meta_code_shortcut() {
    // Single non-zero meta-code length → meta-tree is a lone-symbol
    // shortcut. Every meta-symbol read returns the same code.
    //
    // With meta-tree returning literal-0 always, and use_length=0, the
    // body emits N copies of "literal length 0" → all-zero alphabet →
    // single-symbol shortcut returning 0.
    let mut meta_lens = [0u8; 19];
    meta_lens[0] = 1;
    let alphabet = 8usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(0, 1);
    // No meta-code bits needed — the meta-tree is a lone-symbol shortcut
    // that returns 0 without consuming bits. The loop reads `alphabet`
    // copies of "literal length 0".
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, alphabet).expect("normal tree should build");
    // All-zero alphabet → tree returns 0 always.
    let zero_buf = [0u8; 1];
    let mut br2 = BitReader::new(&zero_buf);
    assert_eq!(tree.decode(&mut br2).unwrap(), 0);
}

#[test]
fn normal_max_symbol_count_runs_for_each_code() {
    // Re-derivation of the libwebp regression at the unit level.
    // max_symbol=4 with sequence: 1, 2, 17(+2)=2-zeros, 2, 18(+0)=11-zeros, 1
    // — six meta-decodes, six count increments, max_symbol cap reached
    // at four. The first four metas are: 1, 2, 17, 2 — leaving the loop
    // before we read the would-be 5th and 6th literals.
    //
    // Correct lengths array with the fix (lens[0..n] up through
    // max_symbol meta-decodes consumed):
    //   meta1 (literal 1):   lens[0] = 1
    //   meta2 (literal 2):   lens[1] = 2
    //   meta3 (17,repeat=3): lens[2..5] = 0,0,0
    //   meta4 (literal 2):   lens[5] = 2
    // → lens = [1, 2, 0, 0, 0, 2, 0, 0, ...]
    // Kraft sum: 1/2 + 1/4 + 1/4 = 1.0 — valid canonical tree.
    let mut meta_lens = [0u8; 19];
    // Need: codes 1, 2, 17. Three codes; pick lengths so they fit
    // canonically. Lengths 2, 2, 1 → Kraft = 1/4 + 1/4 + 1/2 = 1.
    meta_lens[1] = 2;
    meta_lens[2] = 2;
    meta_lens[17] = 1;
    let meta_codes = canonical_codes(&meta_lens);
    let alphabet = 16usize;
    let max_symbol = 4usize;

    let mut bw = BitWriter::new();
    bw.write(0, 1);
    write_meta_lens(&mut bw, &meta_lens);
    bw.write(1, 1);
    bw.write(0, 3); // length_nbits = 2
    bw.write((max_symbol - 2) as u32, 2);
    // meta1: literal 1 → lens[0] = 1
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 1);
    // meta2: literal 2 → lens[1] = 2
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 2);
    // meta3: 17, repeat=3 → lens[2..5] = 0,0,0
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 17);
    bw.write(0, 3);
    // meta4: literal 2 → lens[5] = 2
    write_meta_code(&mut bw, &meta_lens, &meta_codes, 2);
    // Verify the tree decodes the three live symbols (0, 1, 5) using
    // their canonical codes.
    //   sym0 length 1 → code "0"
    //   sym1 length 2 → code "10"
    //   sym5 length 2 → code "11"
    // Decode bit stream: "0" "10" "11" "0" — emit via the same writer
    // so the bits are bit-aligned with the tree body (push-a-byte would
    // skip pad bits between the tree's partial-byte tail and the
    // start of the next byte).
    bw.write(0, 1); // sym0
    bw.write(1, 1); // first bit of sym1 "10"
    bw.write(0, 1); // second bit of sym1
    bw.write(1, 1); // first bit of sym5 "11"
    bw.write(1, 1); // second bit of sym5
    bw.write(0, 1); // sym0
    let blob = bw.finish();
    let mut br = BitReader::new(&blob);
    let tree = HuffmanTree::read(&mut br, alphabet).expect("normal tree should build");
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
    assert_eq!(tree.decode(&mut br).unwrap(), 1);
    assert_eq!(tree.decode(&mut br).unwrap(), 5);
    assert_eq!(tree.decode(&mut br).unwrap(), 0);
}
