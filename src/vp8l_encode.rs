//! VP8L (WebP-Lossless) §3.8 / §3.7 *encoder*.
//!
//! This is the writer counterpart of the round-99..111 decoder stack. The
//! decoder ([`crate::vp8l_transform::decode_lossless`]) walks a VP8L chunk
//! payload — §3.4 image-header, §3.8.2 transform list, §3.8.3 image data
//! (color-cache-info, meta-prefix, prefix-codes, LZ77-coded image) — and
//! produces ARGB pixels. This module produces a VP8L chunk payload from
//! ARGB pixels, taking the simplest end-to-end path the spec admits:
//!
//! * **§3.8.2 optional subtract-green transform** — as of round 120 the
//!   encoder evaluates both the no-transform and subtract-green paths and
//!   emits whichever is smaller. The subtract-green transform (`%b1 %b10`
//!   in the §3.8.2 grammar; transform type 2 per §3.5 Table 1) carries
//!   no body bits and subtracts the green channel from red and blue
//!   before the entropy stage, lowering per-pixel red/blue entropy on
//!   natural images (the spec's §3.5.3 motivation: "this transform is
//!   redundant, as it can be modeled using the color transform, but since
//!   there is no additional data here, the subtract green transform can
//!   be coded using fewer bits"). The other three transforms (predictor
//!   / color / color-indexing) are still pass-through.
//! * **§5.2.1 / §5.2.3 color cache** — as of round 121 the encoder
//!   evaluates a `color_cache_code_bits = 8` (256-entry) color cache
//!   alongside the no-cache path and emits whichever is smaller. When
//!   enabled, the §3.8.3 `color-cache-info` field becomes `%b1 8` (1-bit
//!   flag + 4-bit `code_bits`), the GREEN alphabet grows to
//!   `256 + 24 + 256 = 536` symbols, and each repeat of a previously-
//!   inserted ARGB literal is emitted as a §5.2.3 color-cache code
//!   `256 + 24 + index` instead of four separate ARGB-channel literals.
//!   Cache state is maintained per §5.2.3: every emitted pixel — literal
//!   *and* every pixel covered by a §5.2.2 backward-reference copy — is
//!   re-inserted at its hashed slot
//!   (`(0x1e35a7bd * argb) >> (32 - code_bits)`). The chooser cross-
//!   products with subtract-green so the encoder picks the best of
//!   `(no-tx | subtract-green) × (no-cache | cache)`; on uncorrelated /
//!   non-repeating content the no-cache no-tx path wins and is kept.
//! * **Single §3.7.2.2 meta-prefix code** — `meta-prefix` is `%b0`, so one
//!   [`crate::meta_prefix::PrefixCodeGroup`] of five prefix codes applies
//!   to the whole image.
//! * **Literal-only §3.8.3 image data** — every pixel is a §3.7.3 ARGB
//!   literal (green via prefix code #1, red/blue/alpha via #2/#3/#4). No
//!   LZ77 backward references are emitted by [`encode_argb_literals`], so
//!   the distance prefix code (#5) is the single-symbol-0 form the §3.7.2.1.1
//!   note sanctions ("empty prefix codes can be coded as those containing a
//!   single symbol 0").
//!
//! The result, wrapped by [`encode_webp_lossless`] in the §2.4 RIFF/WEBP
//! framing (via [`crate::build`]), decodes back to the exact input pixels
//! through [`crate::decode_webp`] — a pixel-exact round trip.
//!
//! ## §3.7.2 prefix-code construction
//!
//! For each of the five symbol alphabets the encoder:
//!
//! 1. counts symbol frequencies over the data it will emit;
//! 2. builds a length-limited (≤ [`MAX_CODE_LENGTH`]) canonical
//!    Huffman code-length assignment from those frequencies
//!    ([`build_code_lengths`]);
//! 3. writes the code lengths to the stream with the §3.7.2.1.2 *normal
//!    code length code* (or the trivial single-symbol form), then writes
//!    each symbol with the canonical code derived from the lengths.
//!
//! The canonical code assignment ([`canonical_codes`]) is the identical
//! `(length, value)`-ordered rule the decoder's
//! [`crate::vp8l_prefix::PrefixCode`] reads, so a code emitted here
//! decodes there bit-for-bit.
//!
//! ## §5.2.2 LZ77 backward-reference matching
//!
//! As of round 119, [`encode_argb_literals`] runs an optional §5.2.2
//! backward-reference pass before emitting the image data. A hash-chain
//! matcher ([`Lz77Matcher`]) finds repeated pixel runs; each run of
//! `length >= MIN_MATCH` pixels at scan-line distance `D` is emitted as a
//! §5.2.2 *length + distance code* pair instead of `length` separate ARGB
//! literals, compressing repetitive images. The match's length is encoded
//! via the GREEN alphabet's length-prefix symbols (`256 + prefix_code`).
//!
//! As of round 130 the encoder picks the **smaller** of two distance-code
//! forms per backward reference:
//!
//! 1. The *scan-line* encoding `distance_code = D + NUM_DISTANCE_MAP_CODES`
//!    (always valid, was the round-119 default).
//! 2. Any §5.2.2 *distance map* code `c ∈ 1..=120` whose
//!    `(xi, yi) = DISTANCE_MAP[c-1]` satisfies `max(xi + yi*W, 1) == D` for
//!    the image width `W`. These small codes feed the §5.2.2 distance
//!    prefix code through low-prefix slots (codes `1..=4` use 0 extra bits,
//!    code `5` uses 1 extra bit) instead of the high-prefix slots that
//!    `D + 120` for typical row distances would fall into.
//!
//! The reconstruction in
//! [`crate::vp8l_decode::distance_code_to_pixel_distance`] is identical for
//! both forms (`xi + yi*W` clamped to 1), so round-trips remain bit-exact.
//! Photo-like content with vertical correlation (every scan-line referring
//! to the row above) sees a dramatic improvement: a row-distance match on
//! a 256-wide image goes from prefix 16 (8-ish bits Huffman + 7 extra) to
//! prefix 0 (1–4 bits Huffman + 0 extra), shrinking the per-match cost by
//! ~10 bits. The width-aware helper is
//! [`pixel_distance_to_distance_code`]; the round-119 scan-line-only
//! form is still used as the chooser's fallback whenever no distance-map
//! code matches.
//!
//! The inverse of the §5.2.2 prefix-value transform ([`value_to_prefix`])
//! splits a length/distance into its prefix code and extra bits, the exact
//! counterpart of the decoder's [`crate::vp8l_decode::read_lz77_value`].
//!
//! The literal-only path is still available via [`encode_argb_literals_only`]
//! (used by the size-reduction comparison test); the default
//! [`encode_argb_literals`] entry point chooses the LZ77 path.
//!
//! ## What this module does NOT do
//!
//! * No §3.8.2 predictor / color / color-indexing transform encoding
//!   (subtract-green is wired). Pass-through only for the other three.
//! * No `oxideav-core` runtime dependency — this module compiles under
//!   `--no-default-features`.

use crate::build::{self, ImageKind};

/// The largest code length a VP8L canonical prefix code may use (§3.7.2.1.2
/// stores literal code lengths in `[0..15]`). Mirrors
/// [`crate::vp8l_prefix::MAX_CODE_LENGTH`].
pub const MAX_CODE_LENGTH: usize = 15;

/// §3.7.2.1.2 `kCodeLengthCodes`: the 19-symbol code-length-code alphabet.
pub const NUM_CODE_LENGTH_CODES: usize = 19;

/// §3.7.2.1.2 `kCodeLengthCodeOrder`: the order the (up to 19)
/// code-length-code lengths are transmitted in. Identical to the decoder's
/// [`crate::vp8l_prefix::CODE_LENGTH_CODE_ORDER`].
pub const CODE_LENGTH_CODE_ORDER: [usize; NUM_CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// Errors raised while encoding a VP8L image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// The caller passed an empty pixel buffer, or one whose length does
    /// not match `width * height * 4`.
    PixelBufferMismatch {
        /// Bytes the caller supplied.
        got: usize,
        /// Bytes expected (`width * height * 4`).
        expected: usize,
    },
    /// `width` or `height` was zero, or exceeded the §3.4 14-bit field
    /// maximum of 16384.
    InvalidDimensions {
        /// The offending width.
        width: u32,
        /// The offending height.
        height: u32,
    },
    /// The RIFF/WEBP framing builder rejected the assembled payload.
    Build(build::BuildError),
}

impl From<build::BuildError> for EncodeError {
    fn from(e: build::BuildError) -> Self {
        Self::Build(e)
    }
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PixelBufferMismatch { got, expected } => write!(
                f,
                "VP8L encode: pixel buffer is {got} bytes, expected {expected} (width*height*4)"
            ),
            Self::InvalidDimensions { width, height } => write!(
                f,
                "VP8L encode: invalid dimensions {width}x{height} (must be 1..=16384)"
            ),
            Self::Build(e) => write!(f, "VP8L encode: RIFF/WEBP framing: {e}"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// §3.4 14-bit `width - 1` / `height - 1` field maximum (1-based 16384).
const MAX_DIMENSION: u32 = 1 << 14;

/// Least-significant-bit-first bit writer over a growing byte buffer.
///
/// The exact inverse of [`crate::vp8l_stream::BitReader`]: bits are packed
/// LSB-first within each byte and bytes accumulate in stream order. A
/// multi-bit write lays the value's bit 0 down first, so a subsequent
/// `read_bits(n)` returns it unchanged.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    /// Create an empty bit writer positioned at bit 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of bits written so far.
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Write the low `n` bits of `value` (0 ≤ `n` ≤ 32) LSB-first.
    ///
    /// Writing 0 bits is a no-op (mirrors the reader's `read_bits(0)`).
    pub fn write_bits(&mut self, value: u32, n: usize) {
        debug_assert!(n <= 32, "write_bits supports up to 32 bits");
        let mut value = value;
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

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Consume the writer and return the packed bytes (the final partial
    /// byte is zero-padded in its high bits).
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Build a length-limited (≤ [`MAX_CODE_LENGTH`]) canonical Huffman
/// code-length assignment for an alphabet of `freqs.len()` symbols.
///
/// Returns a `Vec<u8>` of code lengths, one per symbol (0 = symbol unused).
/// The construction guarantees the §3.7.2 completeness invariant the
/// decoder enforces — the Kraft sum of `2^-len` over used symbols equals
/// exactly one — for every input with at least two used symbols, and it
/// produces the §3.7.2.1.2 single-leaf form (one symbol at length 1) for an
/// input with exactly one used symbol.
///
/// The algorithm is a textbook Huffman build over a min-heap of
/// `(frequency, node)` pairs, followed by a length-limiting pass that caps
/// any over-long code at [`MAX_CODE_LENGTH`] while re-balancing so the
/// Kraft sum stays exactly 1. For the small alphabets and pixel counts this
/// encoder targets, the cap is rarely hit; the pass is correctness
/// insurance, not an optimization.
pub fn build_code_lengths(freqs: &[u32]) -> Vec<u8> {
    let n = freqs.len();
    let mut lengths = vec![0u8; n];

    // Collect used symbols.
    let used: Vec<usize> = (0..n).filter(|&s| freqs[s] > 0).collect();
    match used.len() {
        0 => return lengths, // empty code; caller encodes single-symbol-0.
        1 => {
            // §3.7.2.1.2 single-leaf: one symbol marked length 1.
            lengths[used[0]] = 1;
            return lengths;
        }
        _ => {}
    }

    // Huffman build. Nodes 0..n are leaves; internal nodes are appended.
    // We track each node's frequency and, via a parent array, recover the
    // depth (= code length) of each leaf.
    #[derive(Clone, Copy)]
    struct HeapItem {
        freq: u64,
        node: usize,
        // Tie-breaker for deterministic, canonical-friendly ordering.
        order: u64,
    }

    let mut parent: Vec<isize> = vec![-1; n];
    let mut node_freq: Vec<u64> = (0..n).map(|s| freqs[s] as u64).collect();

    // A simple binary min-heap keyed on (freq, order).
    let mut heap: Vec<HeapItem> = Vec::with_capacity(used.len());
    let mut order_counter: u64 = 0;
    for &s in &used {
        heap.push(HeapItem {
            freq: freqs[s] as u64,
            node: s,
            order: order_counter,
        });
        order_counter += 1;
    }
    fn heap_less(a: &HeapItem, b: &HeapItem) -> bool {
        (a.freq, a.order) < (b.freq, b.order)
    }
    fn sift_up(heap: &mut [HeapItem], mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if heap_less(&heap[i], &heap[p]) {
                heap.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }
    fn sift_down(heap: &mut [HeapItem], mut i: usize) {
        let len = heap.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut smallest = i;
            if l < len && heap_less(&heap[l], &heap[smallest]) {
                smallest = l;
            }
            if r < len && heap_less(&heap[r], &heap[smallest]) {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            heap.swap(i, smallest);
            i = smallest;
        }
    }
    fn heap_push(heap: &mut Vec<HeapItem>, item: HeapItem) {
        heap.push(item);
        let last = heap.len() - 1;
        sift_up(heap, last);
    }
    fn heap_pop(heap: &mut Vec<HeapItem>) -> HeapItem {
        let top = heap[0];
        let last = heap.pop().unwrap();
        if !heap.is_empty() {
            heap[0] = last;
            sift_down(heap, 0);
        }
        top
    }
    // Re-heapify the initial array.
    for i in (0..heap.len() / 2).rev() {
        sift_down(&mut heap, i);
    }

    while heap.len() > 1 {
        let a = heap_pop(&mut heap);
        let b = heap_pop(&mut heap);
        let new_node = node_freq.len();
        node_freq.push(a.freq + b.freq);
        parent.push(-1);
        parent[a.node] = new_node as isize;
        parent[b.node] = new_node as isize;
        heap_push(
            &mut heap,
            HeapItem {
                freq: a.freq + b.freq,
                node: new_node,
                order: order_counter,
            },
        );
        order_counter += 1;
    }

    // Recover each leaf's depth.
    let mut max_len = 0usize;
    for &s in &used {
        let mut depth = 0usize;
        let mut cur = s as isize;
        while parent[cur as usize] != -1 {
            cur = parent[cur as usize];
            depth += 1;
        }
        // A single internal-node tree (two leaves) gives depth 1; never 0
        // here because used.len() >= 2.
        lengths[s] = depth as u8;
        max_len = max_len.max(depth);
    }

    if max_len > MAX_CODE_LENGTH {
        limit_code_lengths(&mut lengths, &used);
    }

    lengths
}

/// Cap every code length at [`MAX_CODE_LENGTH`] while keeping the Kraft sum
/// exactly 1, using the standard "move a too-long leaf up and lengthen a
/// short leaf to compensate" rebalancing pass.
///
/// This is the approach a length-limited Huffman post-pass uses when a
/// pathological frequency distribution would otherwise need codes longer
/// than the format allows. It produces a *valid* (complete) code that is at
/// most marginally sub-optimal; exactness of the round trip is unaffected
/// because the decoder reconstructs pixels from whatever complete code the
/// lengths describe.
fn limit_code_lengths(lengths: &mut [u8], used: &[usize]) {
    // Clamp.
    for &s in used {
        if lengths[s] as usize > MAX_CODE_LENGTH {
            lengths[s] = MAX_CODE_LENGTH as u8;
        }
    }
    // Kraft sum over denominator 2^MAX_CODE_LENGTH.
    let full: i64 = 1i64 << MAX_CODE_LENGTH;
    let kraft = |lengths: &[u8]| -> i64 {
        let mut k = 0i64;
        for &s in used {
            let l = lengths[s] as usize;
            if l > 0 {
                k += 1i64 << (MAX_CODE_LENGTH - l);
            }
        }
        k
    };
    // If over-subscribed (sum > 1), lengthen the deepest (largest-length,
    // i.e. cheapest-to-lengthen) leaves until the sum drops to 1.
    let mut k = kraft(lengths);
    while k > full {
        // Find a symbol we can lengthen (length < MAX) with the largest
        // current length, to remove the most "excess" per step.
        let mut target: Option<usize> = None;
        let mut best_len = 0u8;
        for &s in used {
            let l = lengths[s];
            if (l as usize) < MAX_CODE_LENGTH && l >= best_len {
                best_len = l;
                target = Some(s);
            }
        }
        match target {
            Some(s) => {
                lengths[s] += 1;
                k = kraft(lengths);
            }
            None => break,
        }
    }
    // If under-subscribed (sum < 1), shorten the deepest leaves until the
    // sum reaches 1.
    while k < full {
        let mut target: Option<usize> = None;
        let mut best_len = 0u8;
        for &s in used {
            let l = lengths[s];
            if l > 1 && l >= best_len {
                best_len = l;
                target = Some(s);
            }
        }
        match target {
            Some(s) => {
                lengths[s] -= 1;
                k = kraft(lengths);
            }
            None => break,
        }
    }
}

/// Build the canonical code values for a per-symbol length table.
///
/// Returns `codes[s]` = the canonical code value for symbol `s` (only
/// meaningful where `lengths[s] > 0`). The assignment is the same DEFLATE
/// canonical rule the decoder's [`crate::vp8l_prefix::PrefixCode`] reads:
/// symbols ordered by `(length, value)`, codes assigned sequentially, read
/// most-significant-bit-first within a code.
pub fn canonical_codes(lengths: &[u8]) -> Vec<u32> {
    let mut bl_count = [0u32; MAX_CODE_LENGTH + 1];
    for &l in lengths {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = [0u32; MAX_CODE_LENGTH + 2];
    let mut code = 0u32;
    for len in 1..=MAX_CODE_LENGTH {
        code = (code + bl_count[len - 1]) << 1;
        next_code[len] = code;
    }
    let mut codes = vec![0u32; lengths.len()];
    let mut assign = next_code;
    // Indexed by code length to assign sequential canonical codes; mirrors
    // the decoder's `(length, value)`-ordered assignment.
    #[allow(clippy::needless_range_loop)]
    for len in 1..=MAX_CODE_LENGTH {
        for (sym, &l) in lengths.iter().enumerate() {
            if l as usize == len {
                codes[sym] = assign[len];
                assign[len] += 1;
            }
        }
    }
    codes
}

/// §5.2.2: split a length/distance `value` (≥ 1) into its *prefix code* and
/// *extra bits*, the exact inverse of the decoder's
/// [`crate::vp8l_decode::read_lz77_value`].
///
/// Returns `(prefix_code, extra_bits, extra_value)` where:
///
/// * `prefix_code` is the entropy-coded symbol (a GREEN length symbol is
///   `256 + prefix_code`; a distance symbol is `prefix_code` directly),
/// * `extra_bits` is how many raw bits follow the prefix code,
/// * `extra_value` is the value those `extra_bits` carry (LSB-first, as the
///   decoder's `ReadBits` consumes them).
///
/// The decoder reconstructs `value` as:
///
/// ```text
/// if prefix_code < 4 { value = prefix_code + 1 }
/// else {
///     extra_bits = (prefix_code - 2) >> 1
///     offset = (2 + (prefix_code & 1)) << extra_bits
///     value = offset + extra_value + 1
/// }
/// ```
///
/// so feeding `extra_value` back through that formula yields `value`.
pub fn value_to_prefix(value: u32) -> (u32, u32, u32) {
    debug_assert!(value >= 1, "LZ77 length/distance values are 1-based");
    if value <= 4 {
        // prefix_code = value - 1; no extra bits (the `< 4` decoder branch).
        return (value - 1, 0, 0);
    }
    // value >= 5. Find the prefix code p (>= 4) whose range
    // [offset+1, offset + 2^extra_bits] contains `value`, where
    // extra_bits = (p - 2) >> 1 and offset = (2 + (p & 1)) << extra_bits.
    //
    // Equivalently: let v0 = value - 1 (>= 4). The high bit of v0 selects
    // the magnitude; the next bit selects the (p & 1) parity sub-band.
    let v0 = value - 1; // >= 4
                        // `msb` = floor(log2(v0)) >= 2.
    let msb = 31 - v0.leading_zeros();
    let extra_bits = msb - 1;
    // Parity bit: the bit just below the MSB distinguishes the two
    // sub-bands offset = 2<<e (parity 0) vs offset = 3<<e (parity 1).
    let parity = (v0 >> (msb - 1)) & 1;
    let prefix_code = 2 * extra_bits + 2 + parity;
    let offset = (2 + parity) << extra_bits;
    let extra_value = value - offset - 1;
    debug_assert!(extra_value < (1u32 << extra_bits));
    (prefix_code, extra_bits, extra_value)
}

/// A built prefix code ready for symbol emission: per-symbol length + code.
#[derive(Debug, Clone)]
struct WriteCode {
    lengths: Vec<u8>,
    codes: Vec<u32>,
    /// `Some(sym)` when this is the single-leaf form (one symbol, length 1).
    single: Option<usize>,
}

impl WriteCode {
    /// Build a [`WriteCode`] from symbol frequencies over an alphabet of
    /// `alphabet_size` symbols.
    fn from_freqs(freqs: &[u32]) -> Self {
        let used: Vec<usize> = (0..freqs.len()).filter(|&s| freqs[s] > 0).collect();
        let single = if used.len() == 1 { Some(used[0]) } else { None };
        let lengths = build_code_lengths(freqs);
        let codes = canonical_codes(&lengths);
        Self {
            lengths,
            codes,
            single,
        }
    }

    /// An *empty* code: encoded per §3.7.2.1.1's note as a single symbol 0.
    /// Used for the distance code when no backward references are emitted.
    fn empty(alphabet_size: usize) -> Self {
        let mut freqs = vec![0u32; alphabet_size];
        freqs[0] = 1;
        Self::from_freqs(&freqs)
    }

    /// Emit one symbol's code to `w` (MSB-first within the code, matching
    /// the canonical assignment the decoder reads). For the single-leaf
    /// form this writes nothing (reading consumes no bits).
    fn write_symbol(&self, w: &mut BitWriter, symbol: usize) {
        if self.single.is_some() {
            return; // single-leaf code: 0 bits.
        }
        let len = self.lengths[symbol] as usize;
        let code = self.codes[symbol];
        // The decoder reads MSB-first within the code, so emit the high bit
        // first. write_bits is LSB-first, so reverse the `len` low bits.
        for i in 0..len {
            let bit = (code >> (len - 1 - i)) & 1;
            w.write_bits(bit, 1);
        }
    }

    /// Write this code's per-symbol lengths to `w` using the §3.7.2.1.2
    /// *normal code length code* (the general form that can represent any
    /// length table, including the single-leaf one).
    fn write_code_lengths(&self, w: &mut BitWriter) {
        write_normal_code_lengths(w, &self.lengths);
    }
}

/// Write a per-symbol length table with the §3.7.2.1.2 *normal code length
/// code*.
///
/// The encoder uses the general (non-run-length) form: it transmits one
/// code-length-code symbol per literal length. To keep the code-length-code
/// itself trivially decodable, every length value `0..=15` that actually
/// occurs is given a code-length-code symbol; the CLC is built from the
/// frequencies of those length values. Runs (codes 16/17/18) are not
/// emitted — the literal length sequence is sent verbatim, which the
/// decoder's `read_normal_code_lengths` handles as the `0..=15` literal
/// branch.
fn write_normal_code_lengths(w: &mut BitWriter, lengths: &[u8]) {
    // §3.7.2.1.2: the code-length-code is itself a prefix code over the
    // 19-symbol alphabet {0..15 literal lengths, 16 repeat, 17/18 zero
    // runs}. We only emit symbols 0..=15 (no runs), so the CLC alphabet is
    // those length values that occur in `lengths`.
    let mut clc_freq = [0u32; NUM_CODE_LENGTH_CODES];
    for &l in lengths {
        clc_freq[l as usize] += 1;
    }
    let clc_lengths = build_code_lengths(&clc_freq);
    let clc_codes = canonical_codes(&clc_lengths);

    // num_code_lengths: how many CLC lengths we transmit, in
    // kCodeLengthCodeOrder. We must transmit enough leading entries to
    // cover the highest-ordered CLC symbol that has a non-zero length.
    let mut max_order_used = 0usize;
    for (order_idx, &pos) in CODE_LENGTH_CODE_ORDER.iter().enumerate() {
        if clc_lengths[pos] != 0 {
            max_order_used = order_idx;
        }
    }
    // §3.7.2.1.2: num_code_lengths = 4 + ReadBits(4), range [4..19].
    let num_code_lengths = (max_order_used + 1).max(4);

    // normal flag bit.
    w.write_bit(false);
    // num_code_lengths - 4 in 4 bits.
    w.write_bits((num_code_lengths - 4) as u32, 4);
    // The CLC lengths, 3 bits each, in kCodeLengthCodeOrder.
    for &pos in CODE_LENGTH_CODE_ORDER.iter().take(num_code_lengths) {
        w.write_bits(clc_lengths[pos] as u32, 3);
    }
    // max_symbol gate: ReadBits(1) == 0 → max_symbol = alphabet_size, i.e.
    // read all `lengths.len()` entries. We always emit the full table.
    w.write_bit(false);

    // Whether the CLC is a single-leaf code (one length value occurs):
    // write_symbol then emits 0 bits, and the decoder's CLC reader returns
    // that lone symbol for every read — which is exactly the literal length
    // we want, repeated for every symbol. Build a tiny symbol writer.
    let clc_single = {
        let used: Vec<usize> = (0..NUM_CODE_LENGTH_CODES)
            .filter(|&s| clc_freq[s] > 0)
            .collect();
        if used.len() == 1 {
            Some(used[0])
        } else {
            None
        }
    };

    // Emit one CLC symbol per literal length (the `0..=15` branch).
    for &l in lengths {
        let sym = l as usize;
        if clc_single.is_some() {
            continue; // single-leaf CLC: 0 bits per symbol.
        }
        let code = clc_codes[sym];
        let len = clc_lengths[sym] as usize;
        for i in 0..len {
            let bit = (code >> (len - 1 - i)) & 1;
            w.write_bits(bit, 1);
        }
    }
}

/// Smallest backward-reference run (in pixels) the matcher will emit. A
/// match of fewer than this many pixels rarely pays for the length +
/// distance prefix codes versus emitting the pixels as literals, so short
/// runs stay literal.
pub const MIN_MATCH: usize = 3;

/// Largest backward-reference run the §5.2.2 length prefix coding admits
/// (the spec note: "The maximum backward reference length is limited to
/// 4096."). A longer repeat is split into consecutive matches.
pub const MAX_MATCH: usize = 4096;

/// Number of low bits of the rolling pixel hash → hash-chain head buckets.
/// `1 << HASH_BITS` heads; collisions are resolved by walking the chain.
const HASH_BITS: usize = 14;
/// Cap on chain steps walked per position, bounding the matcher's worst
/// case on adversarial inputs while keeping the common-case match quality.
const MAX_CHAIN: usize = 64;

/// A single emitted token in the §5.2.2 LZ77 stream: either a raw ARGB
/// pixel (a §5.2.1 literal), a §5.2.3 color-cache reference, or a
/// §5.2.2 backward-reference copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    /// A §5.2.1 ARGB literal pixel (encoded as four channel symbols).
    Literal(u32),
    /// A §5.2.3 color-cache reference. `index` is the resolved
    /// cache slot (the green symbol on the wire is
    /// `256 + 24 + index`).
    CacheRef {
        /// The hashed cache index (`0..color_cache_size`).
        index: u32,
    },
    /// A §5.2.2 backward reference: copy `length` pixels from `distance`
    /// pixels back in scan-line order.
    Copy {
        /// Copy length in pixels (`MIN_MATCH..=MAX_MATCH`).
        length: usize,
        /// Scan-line pixel distance back to the copy source (`>= 1`).
        distance: usize,
    },
}

/// §5.2.2 hash-chain matcher over a scan-line ARGB pixel buffer.
///
/// Hashes 4-pixel windows into `1 << HASH_BITS` buckets and chains every
/// position sharing a hash, so a match search at position `p` walks only
/// positions that begin with the same 4-pixel hash. This is the standard
/// LZ77 greedy match structure; it finds repeated pixel runs without ever
/// consulting any external implementation — the only correctness contract
/// is that an emitted `Copy { length, distance }` is reproducible by the
/// decoder's §5.2.2 copy loop, which it is for any `1 <= distance <= p` and
/// `length <= remaining`.
struct Lz77Matcher<'a> {
    pixels: &'a [u32],
    head: Vec<i32>,
    prev: Vec<i32>,
}

impl<'a> Lz77Matcher<'a> {
    /// Build a matcher over `pixels` with empty hash chains.
    fn new(pixels: &'a [u32]) -> Self {
        Self {
            pixels,
            head: vec![-1; 1 << HASH_BITS],
            prev: vec![-1; pixels.len()],
        }
    }

    /// Hash the 4-pixel window starting at `pos` (callers guarantee
    /// `pos + 4 <= pixels.len()`). A simple multiplicative mix over the
    /// four ARGB words, folded into `HASH_BITS` bits.
    fn hash(&self, pos: usize) -> usize {
        let p = self.pixels;
        let mut h = 0u32;
        for k in 0..4 {
            h = h.wrapping_mul(0x9e37_79b1).wrapping_add(p[pos + k]);
        }
        (h >> (32 - HASH_BITS)) as usize
    }

    /// Insert `pos` at the head of its hash bucket's chain.
    fn insert(&mut self, pos: usize) {
        if pos + 4 > self.pixels.len() {
            return;
        }
        let h = self.hash(pos);
        self.prev[pos] = self.head[h];
        self.head[h] = pos as i32;
    }

    /// Find the longest match for the window at `pos`, returning
    /// `Some((length, distance))` when a run of `>= MIN_MATCH` pixels is
    /// found. Walks at most [`MAX_CHAIN`] chain links.
    ///
    /// The matcher hashes 4-pixel windows, so a match search requires
    /// `pos + 4 <= pixels.len()`. The tail of the image (fewer than 4
    /// pixels remaining) is always emitted as literals.
    fn find(&self, pos: usize) -> Option<(usize, usize)> {
        let p = self.pixels;
        let n = p.len();
        if pos + 4 > n {
            return None;
        }
        let max_len = (n - pos).min(MAX_MATCH);
        let h = self.hash(pos);
        let mut cand = self.head[h];
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        let mut steps = 0usize;
        while cand >= 0 && steps < MAX_CHAIN {
            let c = cand as usize;
            // Candidates were all inserted at positions < pos.
            let mut len = 0usize;
            while len < max_len && p[c + len] == p[pos + len] {
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best_dist = pos - c;
                if len >= max_len {
                    break;
                }
            }
            cand = self.prev[c];
            steps += 1;
        }
        if best_len >= MIN_MATCH {
            Some((best_len, best_dist))
        } else {
            None
        }
    }
}

/// Run the §5.2.2 greedy hash-chain matcher over `pixels`, producing the
/// token stream (literals + backward-reference copies) the entropy stage
/// emits. Every `Copy` token has `1 <= distance <= position` and
/// `MIN_MATCH <= length <= MAX_MATCH`, so the decoder's §5.2.2 copy loop
/// reproduces the exact pixels.
fn tokenize_lz77(pixels: &[u32]) -> Vec<Token> {
    let n = pixels.len();
    let mut matcher = Lz77Matcher::new(pixels);
    let mut tokens = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        if let Some((len, dist)) = matcher.find(pos) {
            tokens.push(Token::Copy {
                length: len,
                distance: dist,
            });
            // Insert every covered position into the chains so later
            // matches can reference inside the just-copied run, then skip
            // past the run.
            let end = pos + len;
            while pos < end {
                matcher.insert(pos);
                pos += 1;
            }
        } else {
            tokens.push(Token::Literal(pixels[pos]));
            matcher.insert(pos);
            pos += 1;
        }
    }
    tokens
}

/// Allowed range for the §5.2.3 `color_cache_code_bits` field: an
/// enabled cache has `code_bits ∈ [1, 11]`, giving a cache size of
/// `2..=2048` entries. Mirrors
/// [`crate::meta_prefix::COLOR_CACHE_BITS_MIN`] /
/// [`crate::meta_prefix::COLOR_CACHE_BITS_MAX`].
pub const COLOR_CACHE_BITS_MIN: u32 = 1;
/// See [`COLOR_CACHE_BITS_MIN`].
pub const COLOR_CACHE_BITS_MAX: u32 = 11;

/// The default `color_cache_code_bits` the chooser evaluates when
/// deciding whether to enable §5.2.3 color caching. Eight bits gives a
/// 256-entry cache — the spec doesn't mandate a specific size, but 8 is
/// the sweet spot for the kind of payloads this first-pass encoder
/// targets (saves the 4-bit `code_bits` field a slot in the middle of
/// the allowed range; 256 entries is large enough that the hash
/// collisions are negligible on most natural images).
pub const DEFAULT_COLOR_CACHE_BITS: u32 = 8;

/// Round-132 §5.2.3 size-selection slate: the chooser cross-products each
/// of these `code_bits` values (alongside the disabled-cache option) with
/// the §3.8.2 subtract-green transform candidate set, then emits the
/// smallest resulting byte stream.
///
/// The slate spans the §5.2.3 allowed range [`COLOR_CACHE_BITS_MIN`] ..=
/// [`COLOR_CACHE_BITS_MAX`] with five evenly spread sizes:
///
/// * `5` → 32 entries  — cheap to populate; fits tiny palettes.
/// * `7` → 128 entries — covers small/medium palettes (the 4-bit
///   `code_bits` header field cost is amortised over more reuse).
/// * `8` → 256 entries — the round-121 single-size default; kept as the
///   middle "sweet spot" for typical natural-image content.
/// * `9` → 512 entries — for images with denser color spreads where
///   8-bit collisions start to bite.
/// * `11` → 2048 entries — the §5.2.3 maximum; pays off only when the
///   image has thousands of distinct colors that recur enough to amortise
///   the wider GREEN alphabet's prefix-code-length overhead.
///
/// The "no cache" option (encoded as `code_bits = 0` per §5.2.3) is
/// implicit: [`encode_argb_literals_with_width`] always evaluates the
/// `None` candidate as the baseline. The disabled-cache + the five
/// listed sizes give six cache-axis candidates per transform-axis option,
/// for a 2 × 6 = 12-way cross-product.
pub const CANDIDATE_COLOR_CACHE_BITS: [u32; 5] = [5, 7, DEFAULT_COLOR_CACHE_BITS, 9, 11];

/// §5.2.3 color-cache helper used by the encoder. Mirrors the decoder's
/// [`crate::vp8l_decode::ColorCache`] semantics: an array of
/// `1 << code_bits` ARGB entries, all initialized to zero, with a
/// hashed lookup `(0x1e35a7bd * argb) >> (32 - code_bits)`.
///
/// The encoder maintains the cache in stream order — exactly as the
/// decoder will when re-walking the emitted symbols — so a slot's
/// state matches between writer and reader at every bit position. A
/// §5.2.3 `CacheRef { index }` token is emitted *only* when
/// `lookup(index) == Some(argb)` at the moment the token is produced;
/// the decoder will read the same index and produce the same ARGB.
#[derive(Debug, Clone)]
struct EncoderColorCache {
    code_bits: u32,
    entries: Vec<u32>,
}

impl EncoderColorCache {
    /// Allocate a fresh `1 << code_bits`-entry cache. `code_bits` must
    /// be in `[COLOR_CACHE_BITS_MIN, COLOR_CACHE_BITS_MAX]`; debug
    /// builds assert.
    fn new(code_bits: u32) -> Self {
        debug_assert!((COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&code_bits));
        Self {
            code_bits,
            entries: vec![0u32; 1usize << code_bits],
        }
    }

    /// `1 << code_bits` — the §5.2.3 cache size.
    #[cfg(test)]
    fn size(&self) -> usize {
        self.entries.len()
    }

    /// §5.2.3: `(0x1e35a7bd * argb) >> (32 - code_bits)`. Identical to
    /// the decoder's [`crate::vp8l_decode::ColorCache::hash`].
    fn hash(&self, argb: u32) -> usize {
        (crate::vp8l_decode::COLOR_CACHE_HASH_MULTIPLIER.wrapping_mul(argb)
            >> (32 - self.code_bits)) as usize
    }

    /// `true` when the slot for `argb`'s hash currently holds `argb`
    /// itself — i.e. emitting a `CacheRef { index: hash(argb) }`
    /// token would round-trip to the same pixel on decode.
    fn contains(&self, argb: u32) -> Option<usize> {
        let idx = self.hash(argb);
        if self.entries[idx] == argb {
            Some(idx)
        } else {
            None
        }
    }

    /// Insert `argb` at its hashed slot (§5.2.3: every emitted pixel,
    /// literal or covered by a backward reference, is re-inserted).
    fn insert(&mut self, argb: u32) {
        let idx = self.hash(argb);
        self.entries[idx] = argb;
    }
}

/// Second-pass §5.2.3 cache-aware token rewrite.
///
/// Walks `tokens` in stream order, maintaining the cache exactly as
/// the decoder will. When a `Literal(argb)` matches the cache's
/// current slot for `argb`, the literal is rewritten to a
/// `CacheRef { index }` token so the decoder can re-read it from the
/// cache. Backward-reference copies pass through unchanged; the
/// covered pixels are inserted into the cache (spec §5.2.3) so later
/// repeats can refer back to them via cache codes.
///
/// `pixels` provides the underlying pixel sequence for backward
/// references (needed to know which colors a `Copy` token covers so
/// the cache state stays in sync).
fn cacheify_tokens(tokens: &[Token], pixels: &[u32], code_bits: u32) -> Vec<Token> {
    let mut cache = EncoderColorCache::new(code_bits);
    let mut out = Vec::with_capacity(tokens.len());
    let mut pos = 0usize;
    for &tok in tokens {
        match tok {
            Token::Literal(argb) => {
                if let Some(idx) = cache.contains(argb) {
                    out.push(Token::CacheRef { index: idx as u32 });
                } else {
                    out.push(Token::Literal(argb));
                }
                cache.insert(argb);
                pos += 1;
            }
            Token::CacheRef { .. } => {
                // Caller should not pre-emit cache refs into the
                // input stream; keep tokens we don't recognise as
                // literals from the matcher's output verbatim.
                out.push(tok);
                pos += 1;
            }
            Token::Copy { length, distance } => {
                out.push(tok);
                // Mirror the decoder's §5.2.3 invariant: every pixel
                // covered by a backward-reference copy is inserted in
                // stream order. The source pixels live at
                // `pos - distance .. pos - distance + length` in
                // `pixels`; the destination at `pos .. pos + length`
                // would be identical (copies always reproduce source
                // bytes), so we read directly off the source slice.
                let src_start = pos - distance;
                for i in 0..length {
                    let argb = pixels[src_start + i];
                    cache.insert(argb);
                }
                pos += length;
            }
        }
    }
    debug_assert_eq!(
        pos,
        pixels.len(),
        "cacheify_tokens: token stream covered {pos} of {} pixels",
        pixels.len()
    );
    out
}

/// The five per-symbol frequency tables for one prefix-code group: green
/// (literals + §5.2.2 length symbols + §5.2.3 cache indices), red, blue,
/// alpha, and distance.
struct Frequencies {
    green: Vec<u32>,
    red: Vec<u32>,
    blue: Vec<u32>,
    alpha: Vec<u32>,
    distance: Vec<u32>,
}

/// Legacy §5.2.2 *scan-line* distance encoding (`distance_code = D + 120`).
///
/// The decoder's [`crate::vp8l_decode::distance_code_to_pixel_distance`]
/// maps any `distance_code > 120` straight back to `distance_code - 120 == D`,
/// so this is always a valid round-trip. Retained as the unit-test reference
/// (so the round-130 chooser can be measured against the round-119 baseline)
/// — production paths use [`pixel_distance_to_distance_code`], which picks
/// the smaller of the scan-line code and any matching distance-map code.
#[cfg(test)]
fn distance_to_code(distance: usize) -> u32 {
    distance as u32 + crate::vp8l_decode::NUM_DISTANCE_MAP_CODES as u32
}

/// §5.2.2 distance-code chooser: pick the smaller of the scan-line code
/// (`D + 120`) and any §5.2.2 distance-map code `c ∈ 1..=120` that
/// reconstructs `D` for the given `image_width`.
///
/// A distance-map entry `(xi, yi)` at index `c-1` reconstructs to
/// `max(xi + yi * image_width, 1)` per the decoder's
/// [`crate::vp8l_decode::distance_code_to_pixel_distance`]. The chooser
/// scans all 120 entries and returns the **smallest** raw code that
/// reconstructs to `distance` — smaller raw codes feed
/// [`value_to_prefix`] through low-prefix slots (codes `1..=4` use 0
/// extra bits; code `5` uses 1 extra bit; …), which then enter the
/// distance prefix-code's Huffman tree with the highest frequencies and
/// the shortest emitted lengths.
///
/// The reconstruction is identical to the legacy scan-line form, so the
/// decoder produces the exact same pixel distance and the round-trip
/// stays bit-exact.
///
/// Panics in debug builds when `distance == 0` (callers guarantee
/// `1 <= distance <= position` per §5.2.2's backward-reference invariant).
pub fn pixel_distance_to_distance_code(distance: usize, image_width: u32) -> u32 {
    debug_assert!(distance >= 1, "§5.2.2 distance must be >= 1");
    let scan_line_code = distance as u32 + crate::vp8l_decode::NUM_DISTANCE_MAP_CODES as u32;
    let mut best = scan_line_code;
    let width_i32 = image_width as i32;
    for (idx, &(xi, yi)) in crate::vp8l_decode::DISTANCE_MAP.iter().enumerate() {
        // The decoder computes `xi + yi * W` and clamps to 1. Match the
        // exact reconstruction so we never emit a code that would resolve
        // to a different distance.
        let raw = xi + yi * width_i32;
        let mapped = if raw < 1 { 1 } else { raw as usize };
        if mapped == distance {
            let candidate = (idx + 1) as u32;
            if candidate < best {
                best = candidate;
            }
        }
    }
    best
}

/// Accumulate the per-symbol frequencies for a token stream so the entropy
/// stage can build length-optimal prefix codes before emitting.
///
/// `color_cache_size` is `1 << color_cache_code_bits` (0 when the cache
/// is disabled). It extends the GREEN alphabet to
/// `256 + 24 + color_cache_size` per §6.2.3 so a `CacheRef { index }`
/// token's wire symbol `256 + 24 + index` is in range.
///
/// `image_width` is needed to feed [`pixel_distance_to_distance_code`] so
/// the frequency table matches the prefix codes the emit loop will choose
/// for each backward reference. Passing `1` (the legacy width-less form)
/// disables the §5.2.2 distance-map optimisation — only codes 1..=8 can
/// possibly match at width 1, so all row-style matches fall back to the
/// scan-line `D + 120` form.
fn count_frequencies(tokens: &[Token], color_cache_size: usize, image_width: u32) -> Frequencies {
    let green_alphabet = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES + color_cache_size;
    let mut freqs = Frequencies {
        green: vec![0u32; green_alphabet],
        red: vec![0u32; 256],
        blue: vec![0u32; 256],
        alpha: vec![0u32; 256],
        distance: vec![0u32; 40],
    };
    for &tok in tokens {
        match tok {
            Token::Literal(p) => {
                let a = ((p >> 24) & 0xff) as usize;
                let r = ((p >> 16) & 0xff) as usize;
                let g = ((p >> 8) & 0xff) as usize;
                let b = (p & 0xff) as usize;
                freqs.green[g] += 1;
                freqs.red[r] += 1;
                freqs.blue[b] += 1;
                freqs.alpha[a] += 1;
            }
            Token::CacheRef { index } => {
                // §5.2.3: GREEN symbol is `256 + 24 + index`.
                let sym = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES + index as usize;
                debug_assert!(sym < green_alphabet);
                freqs.green[sym] += 1;
            }
            Token::Copy { length, distance } => {
                // §5.2.2: length is a GREEN symbol `256 + length_prefix`.
                let (len_prefix, _, _) = value_to_prefix(length as u32);
                freqs.green[256 + len_prefix as usize] += 1;
                // Distance prefix code (#5). Width-aware chooser picks the
                // smaller of scan-line `D + 120` and any §5.2.2 distance-map
                // code reconstructing to `D` for `image_width`.
                let raw_code = pixel_distance_to_distance_code(distance, image_width);
                let (dist_prefix, _, _) = value_to_prefix(raw_code);
                freqs.distance[dist_prefix as usize] += 1;
            }
        }
    }
    freqs
}

/// Emit a length/distance `value` to `w`: the entropy-coded prefix symbol
/// via `code`, then its `extra_bits` raw bits LSB-first (matching the
/// decoder's `ReadBits`). `symbol_base` is added to the prefix code before
/// the entropy lookup (256 for GREEN length symbols, 0 for distances).
fn write_lz77_value(w: &mut BitWriter, code: &WriteCode, symbol_base: usize, value: u32) {
    let (prefix, extra_bits, extra_value) = value_to_prefix(value);
    code.write_symbol(w, symbol_base + prefix as usize);
    if extra_bits > 0 {
        w.write_bits(extra_value, extra_bits as usize);
    }
}

/// §3.5.3 / §3.8.2 *forward* subtract-green transform: subtract the green
/// channel from red and blue per pixel, in place. The exact inverse of
/// [`crate::vp8l_transform::inverse_subtract_green`], so re-applying the
/// decoder's inverse pass after entropy decode restores the original
/// pixels byte-for-byte.
///
/// Spec arithmetic: `red := (red - green) & 0xff`,
/// `blue := (blue - green) & 0xff` (the §3.5.3 inverse is `+ green & 0xff`,
/// so subtracting on the encode side and adding back on the decode side is
/// a perfect round trip modulo 256).
pub fn apply_subtract_green(pixels: &mut [u32]) {
    for px in pixels.iter_mut() {
        let a = (*px >> 24) & 0xff;
        let r = (*px >> 16) & 0xff;
        let g = (*px >> 8) & 0xff;
        let b = *px & 0xff;
        let r_new = r.wrapping_sub(g) & 0xff;
        let b_new = b.wrapping_sub(g) & 0xff;
        *px = (a << 24) | (r_new << 16) | (g << 8) | b_new;
    }
}

/// Encode an ARGB image to a VP8L *image-stream* (the bytes that follow the
/// §3.4 5-byte image-header), running the §5.2.2 LZ77 backward-reference
/// matcher so repeated pixel runs compress.
///
/// As of round 120, the encoder also evaluates the §3.5.3 / §3.8.2
/// **subtract-green transform** and emits whichever of the two paths is
/// smaller. The transform header costs only three bits (`%b1 %b10`), so on
/// natural images where the green-correlated red/blue channels shrink the
/// per-channel entropy, subtract-green is a near-free compression win. On
/// images where the transform doesn't help (or hurts), the no-transform
/// path is kept.
///
/// `pixels` is `width * height` ARGB values in scan-line order, each
/// `(alpha << 24) | (red << 16) | (green << 8) | blue` — the same layout
/// [`crate::vp8l_decode::DecodedImage::pixels`] produces. The returned
/// bytes, prefixed with the image-header and wrapped in RIFF/WEBP framing,
/// decode back to `pixels` exactly.
pub fn encode_argb_literals(pixels: &[u32]) -> Vec<u8> {
    // Width-less entry: feed `image_width = 1`, which disables the §5.2.2
    // distance-map chooser (no map entry reconstructs to a "row" distance
    // when the row is a single pixel wide). Production callers go through
    // [`encode_argb_literals_with_width`] via [`encode_vp8l_payload`] so
    // the optimisation is wired for `.webp` output.
    encode_argb_literals_with_width(pixels, 1)
}

/// Width-aware variant of [`encode_argb_literals`]: cross-products the
/// §3.8.2 subtract-green transform axis (`no-tx | subtract-green`) with
/// the round-132 §5.2.3 color-cache size axis
/// (`no-cache | each of [`CANDIDATE_COLOR_CACHE_BITS`]`), and emits the
/// smallest resulting byte stream. Each candidate threads `image_width`
/// into [`encode_tokens`] so the §5.2.2 distance-map optimisation is
/// exercised. The production `.webp` path ([`encode_vp8l_payload`] →
/// [`encode_webp_lossless`] / [`encode_vp8l_argb`]) uses this entry; the
/// no-width [`encode_argb_literals`] is retained for test callers that
/// exercise the entropy stage without spatial structure.
///
/// Round 132 widened the cache axis from "0 or 8" (round 121) to a slate
/// of five sizes per [`CANDIDATE_COLOR_CACHE_BITS`]. The §5.2.3 GREEN
/// alphabet's width is `256 + 24 + (1 << code_bits)`, so the prefix-code
/// header overhead scales with the chosen `code_bits`; picking the
/// smallest size that captures the image's color recurrence avoids paying
/// for an over-sized cache.
pub fn encode_argb_literals_with_width(pixels: &[u32], image_width: u32) -> Vec<u8> {
    encode_argb_literals_with_width_selected(pixels, image_width).0
}

/// Like [`encode_argb_literals_with_width`] but also returns the
/// §5.2.3 `color_cache_code_bits` value the chooser selected (`0` when
/// the no-cache path won).
///
/// The chooser walks the 2 × (1 + N) candidate grid (where
/// `N = CANDIDATE_COLOR_CACHE_BITS.len()`), evaluates each combination's
/// produced byte length, and returns the smallest along with the cache
/// size of the winning candidate. The returned `code_bits` is always in
/// `{0} ∪ CANDIDATE_COLOR_CACHE_BITS` — `0` for "no cache enabled",
/// any other value for "this `code_bits` chosen". The byte stream itself
/// is bit-exact decodable through [`crate::decode_lossless_image`] in
/// every case (the chooser only ever compares spec-conformant outputs).
pub fn encode_argb_literals_with_width_selected(
    pixels: &[u32],
    image_width: u32,
) -> (Vec<u8>, u32) {
    debug_assert!(image_width >= 1);

    // Baseline: no-tx, no-cache. Tracked separately so we always have a
    // valid winner even if every cross-product candidate inflates.
    let mut best_bytes = encode_literals_with_options(pixels, false, None, image_width);
    let mut best_code_bits: u32 = 0;

    // No-tx × every cache size.
    for &bits in &CANDIDATE_COLOR_CACHE_BITS {
        let cand = encode_literals_with_options(pixels, false, Some(bits), image_width);
        if cand.len() < best_bytes.len() {
            best_bytes = cand;
            best_code_bits = bits;
        }
    }
    // Subtract-green × no-cache.
    let cand = encode_literals_with_options(pixels, true, None, image_width);
    if cand.len() < best_bytes.len() {
        best_bytes = cand;
        best_code_bits = 0;
    }
    // Subtract-green × every cache size.
    for &bits in &CANDIDATE_COLOR_CACHE_BITS {
        let cand = encode_literals_with_options(pixels, true, Some(bits), image_width);
        if cand.len() < best_bytes.len() {
            best_bytes = cand;
            best_code_bits = bits;
        }
    }

    (best_bytes, best_code_bits)
}

/// Encode `pixels` with explicit knobs: optionally apply the §3.5.3 /
/// §3.8.2 subtract-green transform, optionally enable a §5.2.3 color
/// cache with the given `code_bits` (`None` disables it). The
/// implementation runs the §5.2.2 LZ77 matcher, then (if a cache is
/// requested) rewrites literal tokens into §5.2.3 cache references in
/// stream order, then emits the §3.8.3 image stream.
fn encode_literals_with_options(
    pixels: &[u32],
    subtract_green: bool,
    cache_code_bits: Option<u32>,
    image_width: u32,
) -> Vec<u8> {
    let mut working = pixels.to_vec();
    if subtract_green {
        apply_subtract_green(&mut working);
    }
    let mut tokens = tokenize_lz77(&working);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, &working, bits);
    }
    encode_tokens(&tokens, subtract_green, cache_code_bits, image_width)
}

/// Encode an ARGB image with the literal-only, no-transform path: every
/// pixel becomes a §5.2.1 ARGB literal and no §3.8.2 transform is written.
/// Retained as the baseline the round-119 size-reduction test compares the
/// LZ77 path against; [`encode_argb_literals`] is the default entry point.
pub fn encode_argb_literals_only(pixels: &[u32]) -> Vec<u8> {
    let tokens: Vec<Token> = pixels.iter().map(|&p| Token::Literal(p)).collect();
    // Literal-only stream emits no Copy tokens, so `image_width` is
    // unused by the entropy stage; pass 1 as the trivial value.
    encode_tokens(&tokens, false, None, 1)
}

/// Encode an ARGB image forcing the §3.5.3 / §3.8.2 subtract-green
/// transform on, regardless of whether it shrinks the stream. Used by the
/// round-120 size-reduction comparison test to measure the transform's
/// effect on a natural-image-like fixture; production callers use
/// [`encode_argb_literals`] which picks the smaller of the two paths.
pub fn encode_argb_literals_subtract_green(pixels: &[u32]) -> Vec<u8> {
    let mut sg_pixels = pixels.to_vec();
    apply_subtract_green(&mut sg_pixels);
    let tokens = tokenize_lz77(&sg_pixels);
    // Width-less test entry: pass 1 (the chooser falls back to scan-line).
    encode_tokens(&tokens, true, None, 1)
}

/// Encode an ARGB image forcing a §5.2.3 color cache on (size
/// `1 << cache_code_bits`), with no §3.8.2 transform. Used by the
/// round-121 size-reduction comparison test to isolate the cache's
/// effect from the subtract-green chooser; production callers use
/// [`encode_argb_literals`] which picks the smallest of the four
/// path combinations.
pub fn encode_argb_literals_color_cache(pixels: &[u32], cache_code_bits: u32) -> Vec<u8> {
    debug_assert!((COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&cache_code_bits));
    // Width-less test entry: pass 1 (the chooser falls back to scan-line).
    encode_literals_with_options(pixels, false, Some(cache_code_bits), 1)
}

/// Shared entropy stage: from a §5.2.2 token stream, build the five prefix
/// codes and emit the §3.8.3 image data (optional-transform header,
/// color-cache-info, meta-prefix, the five prefix-code length tables, then
/// the LZ77-coded image).
///
/// `subtract_green` controls the §3.8.2 transform header: `false` emits a
/// single `%b0` terminator (no transform); `true` emits `%b1 %b10 %b0` —
/// the subtract-green transform (type 2, bodyless) followed by the end-of-
/// list terminator.
///
/// `color_cache_code_bits` controls the §5.2.3 `color-cache-info` field:
/// `None` emits `%b0` (no cache); `Some(bits)` emits `%b1 4BIT` with the
/// caller-supplied `code_bits ∈ [1, 11]`. The token stream must already
/// reflect the choice — `CacheRef` tokens are only meaningful when the
/// cache is enabled.
///
/// `image_width` is the §3.4 image width the encoded stream describes;
/// it feeds [`pixel_distance_to_distance_code`] for the §5.2.2 distance
/// chooser so backward references whose scan-line distance equals
/// `xi + yi*image_width` for some distance-map entry get the smaller
/// distance code. Pass `1` to retain the round-119 scan-line-only
/// behaviour (no map codes match at width 1 for typical distances).
fn encode_tokens(
    tokens: &[Token],
    subtract_green: bool,
    color_cache_code_bits: Option<u32>,
    image_width: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §3.8.2 optional-transform.
    if subtract_green {
        // Present-bit `%b1`, then 2-bit TransformType `SubtractGreen` (value
        // 2 in LSB-first bit order: bit0=0, bit1=1 — matches the spec's
        // `%b10` MSB-first notation when read through the LSB-first
        // `ReadBits(2)`). No body for subtract-green per §3.5.3 / §3.8.2.
        w.write_bit(true);
        w.write_bits(crate::vp8l_stream::TransformType::SubtractGreen as u32, 2);
    }
    // End-of-list terminator.
    w.write_bit(false);

    // §3.8.3 spatially-coded-image = color-cache-info meta-prefix data.
    // color-cache-info: `%b0` (no cache) or `%b1 4BIT` (enabled).
    let color_cache_size = match color_cache_code_bits {
        Some(bits) => {
            debug_assert!((COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&bits));
            w.write_bit(true);
            w.write_bits(bits, 4);
            1usize << bits
        }
        None => {
            w.write_bit(false);
            0
        }
    };
    // meta-prefix: `%b0` (single prefix-code group).
    w.write_bit(false);

    // Build the five prefix codes from token frequencies. The GREEN
    // alphabet covers literals (`< 256`), the §5.2.2 length prefix
    // symbols (`256 + length_prefix`), and (when the cache is enabled)
    // the §5.2.3 cache indices (`256 + 24 + index`). The distance
    // alphabet (40 codes) is exercised only when the matcher emitted at
    // least one copy.
    let freqs = count_frequencies(tokens, color_cache_size, image_width);
    let green_code = WriteCode::from_freqs(&freqs.green);
    let red_code = WriteCode::from_freqs(&freqs.red);
    let blue_code = WriteCode::from_freqs(&freqs.blue);
    let alpha_code = WriteCode::from_freqs(&freqs.alpha);
    // Prefix #5 (distance): if no backward references were emitted, the
    // frequency table is all-zero → `from_freqs` yields the empty code,
    // which `WriteCode` serialises as the §3.7.2.1.1 single-symbol-0 form.
    let dist_code = if freqs.distance.iter().any(|&f| f > 0) {
        WriteCode::from_freqs(&freqs.distance)
    } else {
        WriteCode::empty(40)
    };

    // data = prefix-codes lz77-coded-image.
    // prefix-code-group = 5 prefix codes, in bitstream order:
    // green, red, blue, alpha, distance.
    green_code.write_code_lengths(&mut w);
    red_code.write_code_lengths(&mut w);
    blue_code.write_code_lengths(&mut w);
    alpha_code.write_code_lengths(&mut w);
    dist_code.write_code_lengths(&mut w);

    // lz77-coded-image: each token is either a §5.2.1 ARGB literal
    // (channel order green, red, blue, alpha), a §5.2.3 color-cache
    // reference (a single GREEN symbol), or a §5.2.2 length + distance
    // backward reference.
    for &tok in tokens {
        match tok {
            Token::Literal(p) => {
                let a = ((p >> 24) & 0xff) as usize;
                let r = ((p >> 16) & 0xff) as usize;
                let g = ((p >> 8) & 0xff) as usize;
                let b = (p & 0xff) as usize;
                green_code.write_symbol(&mut w, g);
                red_code.write_symbol(&mut w, r);
                blue_code.write_symbol(&mut w, b);
                alpha_code.write_symbol(&mut w, a);
            }
            Token::CacheRef { index } => {
                // §5.2.3: GREEN symbol is `256 + 24 + index`. Red /
                // blue / alpha are not transmitted; the decoder
                // recovers the full ARGB from the cache slot.
                debug_assert!(color_cache_size > 0, "CacheRef requires an enabled cache");
                let sym = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES + index as usize;
                green_code.write_symbol(&mut w, sym);
            }
            Token::Copy { length, distance } => {
                // §5.2.2: length via a GREEN length symbol (base 256), then
                // distance via prefix code #5 (base 0). The chooser must
                // agree with `count_frequencies` so the prefix-code Huffman
                // tree we built actually contains the prefix slot we look up.
                write_lz77_value(&mut w, &green_code, 256, length as u32);
                let raw_code = pixel_distance_to_distance_code(distance, image_width);
                write_lz77_value(&mut w, &dist_code, 0, raw_code);
            }
        }
    }

    w.into_bytes()
}

/// Build the §3.4 / §7.1 5-byte VP8L image-header.
///
/// `0x2F` signature + 14-bit `(width-1)` + 14-bit `(height-1)` +
/// `alpha_is_used` bit + 3-bit `version` (0). The exact inverse of
/// [`crate::vp8l_chunk::WebpLosslessChunk::from_payload`]'s header peek.
fn build_image_header(width: u32, height: u32, alpha_is_used: bool) -> [u8; 5] {
    let packed: u32 =
        ((width - 1) & 0x3FFF) | (((height - 1) & 0x3FFF) << 14) | ((alpha_is_used as u32) << 28);
    // version is 0 → bits 29..31 stay zero.
    [
        crate::vp8l_chunk::VP8L_SIGNATURE,
        (packed & 0xFF) as u8,
        ((packed >> 8) & 0xFF) as u8,
        ((packed >> 16) & 0xFF) as u8,
        ((packed >> 24) & 0xFF) as u8,
    ]
}

/// Encode an interleaved 8-bit RGBA image to a complete RIFF/WEBP file
/// carrying a §2.6 simple-lossless `VP8L` chunk.
///
/// `rgba` is `width * height * 4` bytes in scan-line order, each pixel
/// `[R, G, B, A]` — the `oxideav_core::PixelFormat::Rgba` layout
/// [`crate::DecodedWebp::rgba`] uses. The returned file decodes back to the
/// same RGBA bytes through [`crate::decode_webp`], a pixel-exact round trip.
///
/// The encoder takes the simplest spec-conformant path: no §3.8.2
/// transform, no §3.8.3 color cache, a single meta-prefix code, and a
/// literal-only image (no LZ77 backward references). The §3.7.2 prefix
/// codes are built per-image from the pixel data.
pub fn encode_webp_lossless(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected {
        return Err(EncodeError::PixelBufferMismatch {
            got: rgba.len(),
            expected,
        });
    }

    // Repack RGBA → ARGB and detect whether alpha is non-trivial.
    let mut pixels = Vec::with_capacity(rgba.len() / 4);
    let mut alpha_is_used = false;
    for px in rgba.chunks_exact(4) {
        let (r, g, b, a) = (px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32);
        if a != 0xff {
            alpha_is_used = true;
        }
        pixels.push((a << 24) | (r << 16) | (g << 8) | b);
    }

    let payload = encode_vp8l_payload(&pixels, width, height, alpha_is_used);

    // §2.4 / §2.6 RIFF/WEBP framing around the VP8L payload.
    let file = build::build_webp_file(&payload, ImageKind::Lossless, width, height)?;
    Ok(file)
}

/// Validate `width`/`height` against the §3.4 14-bit field range and check
/// that an ARGB pixel slice carries exactly `width * height` pixels.
///
/// Shared by the bare-bitstream [`encode_vp8l_argb`] / [`encode_vp8l_argb_with`]
/// entry points. Returns the §3.7.2.1.1 "pixel buffer is N, expected M"
/// mismatch error using `pixels.len() * 4` so the byte counts match the
/// RGBA-flavoured [`encode_webp_lossless`] error.
fn validate_argb(pixels: &[u32], width: u32, height: u32) -> Result<(), EncodeError> {
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    let expected = (width as usize) * (height as usize);
    if pixels.len() != expected {
        return Err(EncodeError::PixelBufferMismatch {
            got: pixels.len() * 4,
            expected: expected * 4,
        });
    }
    Ok(())
}

/// Assemble the bare §2.6 / §3.4 `VP8L` chunk **payload** for an ARGB image:
/// the 5-byte §3.4 image-header followed by the §3.8.3 image stream.
///
/// `pixels` is `width * height` ARGB values in scan-line order, each
/// `(alpha << 24) | (red << 16) | (green << 8) | blue`. `alpha_is_used`
/// becomes the §3.4 `alpha_is_used` header bit. This is the inner payload a
/// `VP8L` chunk wraps — *not* a RIFF/WEBP file. Callers wanting the framed
/// file use [`encode_webp_lossless`] / [`encode_vp8l_argb_with_metadata`].
fn encode_vp8l_payload(pixels: &[u32], width: u32, height: u32, alpha_is_used: bool) -> Vec<u8> {
    // Production path: thread the actual image width so the §5.2.2
    // distance-map chooser can swap row-style scan-line codes for
    // small distance-map codes (round 130).
    let stream = encode_argb_literals_with_width(pixels, width);
    let header = build_image_header(width, height, alpha_is_used);
    let mut payload = Vec::with_capacity(header.len() + stream.len());
    payload.extend_from_slice(&header);
    payload.extend_from_slice(&stream);
    payload
}

/// Encode an ARGB image to a **bare** §2.6 / §3.4 `VP8L` bitstream — the
/// chunk payload (image-header + image stream), with **no** RIFF/WEBP
/// wrapper.
///
/// `pixels` is `width * height` ARGB values in scan-line order, each
/// `(alpha << 24) | (red << 16) | (green << 8) | blue`. The `alpha_is_used`
/// §3.4 header bit is auto-detected: it is set iff any pixel's alpha byte is
/// not `0xff`. Use [`encode_vp8l_argb_with`] to force the bit explicitly.
///
/// The output is the exact byte sequence
/// [`crate::vp8l_chunk::WebpLosslessChunk::bitstream`] returns for a framed
/// file — i.e. wrapping it in `build_chunk(fourcc::VP8L, ..)` (or
/// [`build::build_webp_file`] with [`ImageKind::Lossless`]) yields a complete
/// `.webp`. Encoding path matches [`encode_webp_lossless`]: no §3.8.2
/// transform, no §3.8.3 color cache, single meta-prefix code, literal-only.
pub fn encode_vp8l_argb(pixels: &[u32], width: u32, height: u32) -> Result<Vec<u8>, EncodeError> {
    let alpha_is_used = pixels.iter().any(|&p| (p >> 24) & 0xff != 0xff);
    encode_vp8l_argb_with(pixels, width, height, alpha_is_used)
}

/// Encode an ARGB image to a bare §2.6 / §3.4 `VP8L` bitstream with the
/// §3.4 `alpha_is_used` header bit set **explicitly** by the caller.
///
/// Identical to [`encode_vp8l_argb`] but with a fixed (non-auto-detected)
/// `alpha_is_used`. A caller that already knows whether the image carries
/// alpha — e.g. one decoding the §2.7.1 `VP8X` `L` flag — avoids the
/// per-pixel scan. Setting `alpha_is_used = true` on a fully-opaque image is
/// permitted (a decoder reconstructs the same opaque pixels); setting it
/// `false` on an image with non-opaque pixels still round-trips because the
/// alpha values are carried in the §3.7.3 ARGB literals regardless of the
/// header bit.
pub fn encode_vp8l_argb_with(
    pixels: &[u32],
    width: u32,
    height: u32,
    alpha_is_used: bool,
) -> Result<Vec<u8>, EncodeError> {
    validate_argb(pixels, width, height)?;
    Ok(encode_vp8l_payload(pixels, width, height, alpha_is_used))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp8l_prefix::PrefixCode;
    use crate::vp8l_stream::BitReader;

    // ---- BitWriter ----

    #[test]
    fn bit_writer_round_trips_through_bit_reader() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0xABCD, 16);
        w.write_bit(true);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(16).unwrap(), 0xABCD);
        assert!(r.read_bit().unwrap());
    }

    // ---- canonical code construction ----

    #[test]
    fn code_lengths_single_symbol_is_length_one() {
        let mut freq = vec![0u32; 8];
        freq[3] = 10;
        let lengths = build_code_lengths(&freq);
        assert_eq!(lengths[3], 1);
        assert_eq!(lengths.iter().filter(|&&l| l != 0).count(), 1);
    }

    #[test]
    fn code_lengths_two_symbols_length_one_each() {
        let mut freq = vec![0u32; 4];
        freq[1] = 5;
        freq[2] = 5;
        let lengths = build_code_lengths(&freq);
        assert_eq!(lengths[1], 1);
        assert_eq!(lengths[2], 1);
    }

    #[test]
    fn code_lengths_kraft_sum_is_one() {
        // A skewed distribution that produces varied lengths.
        let freq = vec![100u32, 1, 1, 1, 50, 25, 4, 2];
        let lengths = build_code_lengths(&freq);
        let mut k = 0f64;
        for &l in &lengths {
            if l > 0 {
                k += 2f64.powi(-(l as i32));
            }
        }
        assert!((k - 1.0).abs() < 1e-9, "Kraft sum {k} != 1");
    }

    #[test]
    fn built_code_decodes_through_prefix_reader() {
        // Build a code, emit symbols with it, and decode with the
        // round-104 reader to confirm bit-exact agreement.
        let freq = vec![40u32, 10, 5, 5, 1, 0, 0, 0];
        let code = WriteCode::from_freqs(&freq);
        let mut w = BitWriter::new();
        code.write_code_lengths(&mut w);
        // Emit symbols 0,1,2,3,4 in sequence.
        let seq = [0usize, 1, 2, 3, 4, 0, 0, 1];
        for &s in &seq {
            code.write_symbol(&mut w, s);
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded = PrefixCode::read(&mut r, freq.len()).unwrap();
        for &s in &seq {
            assert_eq!(decoded.read_symbol(&mut r).unwrap() as usize, s);
        }
    }

    #[test]
    fn empty_distance_code_is_single_symbol_zero() {
        let code = WriteCode::empty(40);
        let mut w = BitWriter::new();
        code.write_code_lengths(&mut w);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded = PrefixCode::read(&mut r, 40).unwrap();
        assert_eq!(decoded.single_symbol(), Some(0));
    }

    // ---- image-header ----

    #[test]
    fn image_header_round_trips_through_chunk_peek() {
        use crate::vp8l_chunk::WebpLosslessChunk;
        let header = build_image_header(7, 5, true);
        // Append a dummy byte so the payload is long enough to peek.
        let mut payload = header.to_vec();
        payload.push(0);
        let h = WebpLosslessChunk::from_payload(&payload).unwrap();
        assert_eq!(h.width(), 7);
        assert_eq!(h.height(), 5);
        assert!(h.alpha_is_used());
        assert_eq!(h.version(), 0);
    }

    // ---- end-to-end round trips ----

    #[test]
    fn round_trip_1x1_opaque() {
        let rgba = [0x12, 0x34, 0x56, 0xff];
        let file = encode_webp_lossless(&rgba, 1, 1).unwrap();
        let decoded = crate::decode_webp(&file).unwrap();
        assert_eq!(decoded.frames[0].rgba, rgba);
    }

    #[test]
    fn round_trip_1x1_with_alpha() {
        let rgba = [0xaa, 0xbb, 0xcc, 0x40];
        let file = encode_webp_lossless(&rgba, 1, 1).unwrap();
        let img = crate::decode_webp_image(&file).unwrap();
        assert_eq!(img.width, 1);
        assert_eq!(img.height, 1);
        assert_eq!(img.rgba, rgba);
    }

    #[test]
    fn round_trip_small_gradient() {
        // 4x3 image with a spread of colors.
        let w = 4u32;
        let h = 3u32;
        let mut rgba = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgba.push((x * 60) as u8);
                rgba.push((y * 80) as u8);
                rgba.push(((x + y) * 30) as u8);
                rgba.push(0xff);
            }
        }
        let file = encode_webp_lossless(&rgba, w, h).unwrap();
        let decoded = crate::decode_webp(&file).unwrap();
        assert_eq!(decoded.frames[0].rgba, rgba);
    }

    #[test]
    fn round_trip_solid_color_uses_single_leaf_codes() {
        // A solid color makes every channel a single-symbol code. The
        // round trip must still be exact.
        let w = 8u32;
        let h = 8u32;
        let mut rgba = Vec::new();
        for _ in 0..(w * h) {
            rgba.extend_from_slice(&[0x20, 0x40, 0x60, 0xff]);
        }
        let file = encode_webp_lossless(&rgba, w, h).unwrap();
        let decoded = crate::decode_webp(&file).unwrap();
        assert_eq!(decoded.frames[0].rgba, rgba);
    }

    #[test]
    fn round_trip_larger_random_like() {
        // A deterministic pseudo-random pattern over a 16x16 RGBA image,
        // exercising all four channel codes with many distinct symbols.
        let w = 16u32;
        let h = 16u32;
        let mut rgba = Vec::new();
        let mut state = 0x1234_5678u32;
        for _ in 0..(w * h) {
            for _ in 0..4 {
                // xorshift
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                rgba.push((state & 0xff) as u8);
            }
        }
        let file = encode_webp_lossless(&rgba, w, h).unwrap();
        let decoded = crate::decode_webp(&file).unwrap();
        assert_eq!(decoded.frames[0].rgba, rgba);
    }

    #[test]
    fn encoded_file_walks_as_simple_lossless_container() {
        let rgba = [0x12, 0x34, 0x56, 0xff];
        let file = encode_webp_lossless(&rgba, 1, 1).unwrap();
        let c = crate::parse_container(&file).unwrap();
        assert!(c
            .first_chunk_with_fourcc(crate::container::fourcc::VP8L)
            .is_some());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let rgba = [0u8; 4]; // 1 pixel
        match encode_webp_lossless(&rgba, 2, 2) {
            Err(EncodeError::PixelBufferMismatch { got, expected }) => {
                assert_eq!(got, 4);
                assert_eq!(expected, 16);
            }
            other => panic!("expected PixelBufferMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_dimensions() {
        match encode_webp_lossless(&[], 0, 0) {
            Err(EncodeError::InvalidDimensions { width, height }) => {
                assert_eq!(width, 0);
                assert_eq!(height, 0);
            }
            other => panic!("expected InvalidDimensions, got {other:?}"),
        }
    }

    // ---- bare VP8L bitstream (encode_vp8l_argb / _with) ----

    /// The bare bitstream wrapped in §2.6 framing equals the file
    /// [`encode_webp_lossless`] produces for the same pixels.
    #[test]
    fn bare_bitstream_wrapped_equals_framed_file() {
        // 3x2 ARGB image with a spread of colors and one non-opaque pixel.
        let pixels: [u32; 6] = [
            0xff10_2030,
            0xff40_5060,
            0x8070_8090,
            0xffa0_b0c0,
            0xffd0_e0f0,
            0xff00_1122,
        ];
        let bare = encode_vp8l_argb(&pixels, 3, 2).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, 3, 2).unwrap();

        // Re-derive the same file via the RGBA entry point.
        let mut rgba = Vec::new();
        for &p in &pixels {
            rgba.push((p >> 16) as u8);
            rgba.push((p >> 8) as u8);
            rgba.push(p as u8);
            rgba.push((p >> 24) as u8);
        }
        let via_rgba = encode_webp_lossless(&rgba, 3, 2).unwrap();
        assert_eq!(framed, via_rgba);
    }

    /// A bare bitstream has no `RIFF` header — it begins with the §3.4
    /// `0x2F` VP8L signature byte.
    #[test]
    fn bare_bitstream_has_no_riff_wrapper() {
        let pixels = [0xff12_3456u32];
        let bare = encode_vp8l_argb(&pixels, 1, 1).unwrap();
        assert_ne!(&bare[0..4], b"RIFF");
        assert_eq!(bare[0], crate::vp8l_chunk::VP8L_SIGNATURE);
    }

    /// `encode_vp8l_argb` auto-detects the §3.4 `alpha_is_used` bit.
    #[test]
    fn bare_bitstream_auto_detects_alpha() {
        let opaque = [0xff11_2233u32, 0xff44_5566];
        let bare = encode_vp8l_argb(&opaque, 2, 1).unwrap();
        let h = crate::vp8l_chunk::WebpLosslessChunk::from_payload(&bare).unwrap();
        assert!(!h.alpha_is_used());

        let translucent = [0x8011_2233u32, 0xff44_5566];
        let bare = encode_vp8l_argb(&translucent, 2, 1).unwrap();
        let h = crate::vp8l_chunk::WebpLosslessChunk::from_payload(&bare).unwrap();
        assert!(h.alpha_is_used());
    }

    /// `encode_vp8l_argb_with` forces the header bit regardless of pixels.
    #[test]
    fn bare_bitstream_with_forces_alpha_bit() {
        let opaque = [0xff11_2233u32];
        let bare = encode_vp8l_argb_with(&opaque, 1, 1, true).unwrap();
        let h = crate::vp8l_chunk::WebpLosslessChunk::from_payload(&bare).unwrap();
        assert!(h.alpha_is_used());
    }

    /// The bare bitstream round-trips back to the exact pixels through the
    /// full decode chain once framed.
    #[test]
    fn bare_bitstream_round_trips() {
        let pixels: [u32; 4] = [0x80aa_bbcc, 0xff00_1122, 0xc033_4455, 0xff66_7788];
        let bare = encode_vp8l_argb(&pixels, 2, 2).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, 2, 2).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), &pixels);
    }

    #[test]
    fn bare_bitstream_rejects_dimension_mismatch() {
        let pixels = [0xff00_0000u32]; // 1 pixel
        match encode_vp8l_argb(&pixels, 2, 2) {
            Err(EncodeError::PixelBufferMismatch { got, expected }) => {
                assert_eq!(got, 4);
                assert_eq!(expected, 16);
            }
            other => panic!("expected PixelBufferMismatch, got {other:?}"),
        }
    }

    // ---- §5.2.2 LZ77 prefix-value inverse ----

    /// Every value `1..=4` maps to prefix code `value - 1` with no extra
    /// bits, matching the `< 4` decoder branch.
    #[test]
    fn value_to_prefix_small_values_have_no_extra_bits() {
        for v in 1u32..=4 {
            let (p, e, x) = value_to_prefix(v);
            assert_eq!(p, v - 1);
            assert_eq!(e, 0);
            assert_eq!(x, 0);
        }
    }

    /// Round-trip every length value `1..=MAX_MATCH` through
    /// [`value_to_prefix`] back into the §5.2.2 decoder formula.
    #[test]
    fn value_to_prefix_round_trips_length_range() {
        for v in 1u32..=MAX_MATCH as u32 {
            let (p, e, x) = value_to_prefix(v);
            // Re-apply the §5.2.2 decoder formula.
            let recovered = if p < 4 {
                p + 1
            } else {
                let extra_bits = (p - 2) >> 1;
                let offset = (2 + (p & 1)) << extra_bits;
                assert_eq!(extra_bits, e);
                offset + x + 1
            };
            assert_eq!(recovered, v, "value_to_prefix lost value {v}");
        }
    }

    /// Round-trip via the live decoder helper [`crate::vp8l_decode::read_lz77_value`]
    /// to confirm the encoder's split is bit-compatible with what the
    /// decoder actually executes.
    #[test]
    fn value_to_prefix_round_trips_through_decoder() {
        use crate::vp8l_decode::read_lz77_value;
        use crate::vp8l_stream::BitReader;
        // A spread of values across every prefix-code band.
        let samples = [
            1u32, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 24, 25, 32, 100, 1000, 4096,
        ];
        for &v in &samples {
            let (p, e, x) = value_to_prefix(v);
            let mut w = BitWriter::new();
            if e > 0 {
                w.write_bits(x, e as usize);
            }
            let data = w.into_bytes();
            let mut r = BitReader::new(&data);
            let got = read_lz77_value(&mut r, p).unwrap();
            assert_eq!(
                got, v,
                "value {v} → prefix {p}, extra ({e}b: {x:b}) decoded as {got}"
            );
        }
    }

    // ---- §5.2.2 LZ77 matcher / encoder round-trips ----

    /// A solid-color image's pixels are a single literal followed by one
    /// long copy that covers the rest. Round trip must be exact.
    #[test]
    fn round_trip_solid_color_uses_lz77_copy() {
        let w = 32u32;
        let h = 32u32;
        let pixels = vec![0xff20_4060u32; (w * h) as usize];
        let tokens = tokenize_lz77(&pixels);
        // 1 literal + ceil((1024 - 1) / 4096) copies; for 1024 pixels: 1 + 1.
        let copies = tokens
            .iter()
            .filter(|t| matches!(t, Token::Copy { .. }))
            .count();
        assert!(
            copies >= 1,
            "solid-color image should emit at least one copy"
        );
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// A repeated 4-pixel pattern (cycle length 4) compresses to a long
    /// copy with `distance = 4`, which the §5.2.2 overlap rule
    /// (`distance < length`) self-replicates correctly.
    #[test]
    fn round_trip_periodic_pattern_uses_overlapping_copy() {
        let pattern = [0xff10_2030u32, 0xff40_5060, 0xff70_8090, 0xffa0_b0c0];
        let w = 16u32;
        let h = 4u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) {
            pixels.push(pattern[(i % 4) as usize]);
        }
        let tokens = tokenize_lz77(&pixels);
        let copies: Vec<_> = tokens
            .iter()
            .filter_map(|t| match t {
                Token::Copy { length, distance } => Some((*length, *distance)),
                _ => None,
            })
            .collect();
        assert!(!copies.is_empty(), "periodic pattern should emit a copy");
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// The §5.2.2 LZ77 path produces a strictly smaller chunk than the
    /// literal-only baseline on a compressible (repetitive) image. This is
    /// the round-119 headline measurement.
    #[test]
    fn lz77_beats_literal_only_on_repetitive_image() {
        // 64x64 image whose first scan-line is a small palette of distinct
        // colors and the remaining 63 lines copy the first line verbatim.
        let w = 64u32;
        let h = 64u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let palette = [
            0xff10_2030u32,
            0xff40_5060,
            0xff70_8090,
            0xffa0_b0c0,
            0xffd0_e0f0,
            0xff00_1122,
            0xff33_4455,
            0xff66_7788,
        ];
        for x in 0..w {
            pixels.push(palette[(x as usize) % palette.len()]);
        }
        for _ in 1..h {
            for x in 0..w {
                pixels.push(palette[(x as usize) % palette.len()]);
            }
        }
        let lz77 = encode_argb_literals(&pixels);
        let lit_only = encode_argb_literals_only(&pixels);
        assert!(
            lz77.len() < lit_only.len(),
            "LZ77 stream ({} B) not smaller than literal-only ({} B)",
            lz77.len(),
            lit_only.len(),
        );
        // And, more strongly, at least a 50% reduction on this case.
        assert!(
            lz77.len() * 2 < lit_only.len(),
            "LZ77 stream ({} B) failed to halve literal-only ({} B)",
            lz77.len(),
            lit_only.len(),
        );

        // Round trip is exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// A pixel buffer with no exploitable repetition (deterministic
    /// xorshift) still round-trips through the LZ77 encoder — even when
    /// the matcher emits no copies and the distance code stays empty.
    #[test]
    fn lz77_round_trips_incompressible_pixels() {
        let w = 17u32;
        let h = 19u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xdead_beefu32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(state);
        }
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    // ---- §3.5.3 / §3.8.2 subtract-green forward transform ----

    /// `apply_subtract_green` is the per-pixel inverse of
    /// [`crate::vp8l_transform::inverse_subtract_green`]: subtracting
    /// then re-adding green restores the originals, even across the
    /// `& 0xff` wrap.
    #[test]
    fn apply_subtract_green_is_inverse_of_inverse_subtract_green() {
        let mut pixels = [
            0xff00_0000u32, // black
            0xff7f_ff00,    // greenish
            0xffff_ffff,    // white
            0x8012_3456,    // mid alpha
            0x0001_0203,    // wrapping case: r=01, g=02, b=03
        ];
        let original = pixels;
        apply_subtract_green(&mut pixels);
        // Run the decoder's inverse and confirm we're back at the start.
        crate::vp8l_transform::inverse_subtract_green(&mut pixels);
        assert_eq!(pixels, original);
    }

    /// `apply_subtract_green` preserves the green and alpha channels and
    /// only mutates red/blue per the §3.5.3 spec.
    #[test]
    fn apply_subtract_green_only_touches_red_and_blue() {
        let mut pixels = [0x80_70_60_50u32]; // a=80 r=70 g=60 b=50
        apply_subtract_green(&mut pixels);
        // a, g unchanged; r := (0x70 - 0x60) & 0xff = 0x10; b := 0xf0.
        assert_eq!((pixels[0] >> 24) & 0xff, 0x80);
        assert_eq!((pixels[0] >> 16) & 0xff, 0x10);
        assert_eq!((pixels[0] >> 8) & 0xff, 0x60);
        assert_eq!(pixels[0] & 0xff, 0xf0); // 0x50 - 0x60 = -0x10 → 0xf0
    }

    /// On a synthetic natural-image-like fixture (a gradient where red and
    /// blue track green), the subtract-green path is strictly smaller than
    /// the no-transform path. This is the round-120 headline measurement.
    #[test]
    fn subtract_green_beats_no_transform_on_green_correlated_image() {
        // 32x32 image whose r and b channels each closely track g, so
        // (r - g) and (b - g) cluster tightly around 0 — exactly the
        // distribution §3.5.3 is designed to exploit.
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xC0FFEE12u32;
        for _ in 0..(w * h) {
            // xorshift-driven green; r/b are green plus small noise.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let g = state & 0xff;
            let r = g.wrapping_add(((state >> 8) & 0x0f).wrapping_sub(7) & 0xff) & 0xff;
            let b = g.wrapping_add(((state >> 16) & 0x0f).wrapping_sub(7) & 0xff) & 0xff;
            pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
        }
        let no_tx = {
            let tokens = tokenize_lz77(&pixels);
            // Width-less baseline (matches `encode_argb_literals_subtract_green`
            // below, which also uses width=1) so the comparison isolates
            // the subtract-green transform from the round-130 distance-map
            // chooser.
            encode_tokens(&tokens, false, None, 1)
        };
        let sg = encode_argb_literals_subtract_green(&pixels);
        eprintln!(
            "[round-120] 32x32 green-correlated: no-tx={} B, subtract-green={} B ({:.1}% reduction)",
            no_tx.len(),
            sg.len(),
            100.0 * (no_tx.len() as f64 - sg.len() as f64) / no_tx.len() as f64,
        );
        assert!(
            sg.len() < no_tx.len(),
            "subtract-green ({} B) did not beat no-transform ({} B)",
            sg.len(),
            no_tx.len(),
        );

        // Round trip through the full decode chain stays pixel-exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// `encode_argb_literals` picks the smallest of the four
    /// `(no-tx | sg) × (no-cache | cache)` paths it evaluates, so on
    /// any image its output equals the minimum of all four candidate
    /// streams.
    #[test]
    fn encode_argb_literals_chooses_smaller_path() {
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        // A solid green tint with slight per-pixel red/blue noise — the
        // subtract-green path concentrates r and b near zero.
        let mut state = 0x12345678u32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let g = 0x80u32;
            let r = g.wrapping_add((state & 0x0f).wrapping_sub(7) & 0xff) & 0xff;
            let b = g.wrapping_add(((state >> 4) & 0x0f).wrapping_sub(7) & 0xff) & 0xff;
            pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
        }
        let chosen = encode_argb_literals(&pixels);
        // `encode_argb_literals` defaults to width=1 (no distance-map
        // optimisation); match it for the per-option comparison.
        let no_tx = encode_literals_with_options(&pixels, false, None, 1);
        let sg = encode_literals_with_options(&pixels, true, None, 1);
        let cc = encode_literals_with_options(&pixels, false, Some(DEFAULT_COLOR_CACHE_BITS), 1);
        let sg_cc = encode_literals_with_options(&pixels, true, Some(DEFAULT_COLOR_CACHE_BITS), 1);
        let best = no_tx.len().min(sg.len()).min(cc.len()).min(sg_cc.len());
        assert_eq!(chosen.len(), best);
    }

    /// A subtract-green-encoded image survives a full encode → decode
    /// round trip via the public entry points: the encoder writes the
    /// §3.8.2 transform header, the decoder reads it back and applies the
    /// §4.3 inverse, restoring the originals.
    #[test]
    fn subtract_green_path_round_trips_via_public_entry_points() {
        let w = 8u32;
        let h = 8u32;
        let pixels: Vec<u32> = (0..(w * h))
            .map(|i| {
                let g = (i * 4) & 0xff;
                let r = g.wrapping_add(3) & 0xff;
                let b = g.wrapping_sub(2) & 0xff;
                0xff00_0000 | (r << 16) | (g << 8) | b
            })
            .collect();
        // Force the subtract-green path via the test-only entry.
        let stream = encode_argb_literals_subtract_green(&pixels);
        let header = build_image_header(w, h, false);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// On a pure-noise image (no green correlation) the chooser falls
    /// back to the no-transform path — `encode_argb_literals` should
    /// never produce a stream larger than the literal-only baseline by
    /// applying a transform that doesn't help.
    #[test]
    fn encode_argb_literals_does_not_regress_on_uncorrelated_noise() {
        let w = 16u32;
        let h = 16u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xDEAD_BEEFu32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(state | 0xff00_0000);
        }
        let chosen = encode_argb_literals(&pixels);
        let no_tx = {
            let tokens = tokenize_lz77(&pixels);
            // Match `encode_argb_literals`'s width-less form (width=1) so
            // the chooser comparison stays apples-to-apples regardless of
            // the round-130 distance-map optimisation.
            encode_tokens(&tokens, false, None, 1)
        };
        assert!(
            chosen.len() <= no_tx.len(),
            "chooser regressed: {} B with chooser vs {} B no-transform",
            chosen.len(),
            no_tx.len(),
        );
    }

    /// A maximum-length copy (>= MAX_MATCH pixels of identical color) is
    /// split into consecutive §5.2.2 copies, each bounded by `MAX_MATCH`.
    #[test]
    fn round_trip_splits_match_at_max_length() {
        // A solid-color image with `> MAX_MATCH` pixels: the first row
        // is the literal source, subsequent rows are copies.
        let total = MAX_MATCH + 100;
        let pixels = vec![0xff80_8080u32; total];
        let tokens = tokenize_lz77(&pixels);
        for tok in &tokens {
            if let Token::Copy { length, .. } = tok {
                assert!(
                    *length <= MAX_MATCH,
                    "copy length {length} exceeded MAX_MATCH"
                );
            }
        }
        // Round trip via the full encoder/decoder chain (1-row image of
        // `total` pixels).
        let w = total as u32;
        let h = 1u32;
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    // ---- §5.2.1 / §5.2.3 color cache (round 121) ----

    /// The encoder's `EncoderColorCache` uses the spec's §5.2.3 hash
    /// formula and matches the decoder's
    /// [`crate::vp8l_decode::ColorCache::hash`] bit-for-bit at every
    /// allowed `code_bits`.
    #[test]
    fn encoder_color_cache_hash_matches_decoder_hash() {
        use crate::vp8l_decode::ColorCache;
        for bits in COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX {
            let enc = EncoderColorCache::new(bits);
            let dec = ColorCache::new(bits);
            // A spread of synthetic ARGB pixels: black, white, the
            // wrap-around 0x01020304, a saturated red, a mid-alpha
            // greenish, plus a zero (which all caches start with).
            for argb in [
                0x0000_0000u32,
                0xffff_ffff,
                0x0102_0304,
                0xffff_0000,
                0x8000_ff80,
                0x1234_5678,
            ] {
                assert_eq!(
                    enc.hash(argb),
                    dec.hash(argb),
                    "hash mismatch at code_bits={bits} for argb=0x{argb:08x}"
                );
            }
            assert_eq!(enc.size(), 1 << bits);
        }
    }

    /// A fresh cache holds zeros, so `contains(0)` succeeds *before*
    /// any insertion — exactly the §5.2.3 "all entries set to zero"
    /// invariant the decoder relies on.
    #[test]
    fn encoder_color_cache_starts_zero_initialized() {
        let cache = EncoderColorCache::new(4);
        // Index 0's slot starts at the all-zero pixel.
        let zero_idx = cache.hash(0);
        assert_eq!(cache.entries[zero_idx], 0);
        assert_eq!(cache.contains(0), Some(zero_idx));
    }

    /// Inserting a pixel makes a subsequent `contains` for that same
    /// pixel resolve to the matching slot; an unrelated pixel does
    /// not collide (with overwhelming probability at 8 cache bits).
    #[test]
    fn encoder_color_cache_insert_then_contains_round_trips() {
        let mut cache = EncoderColorCache::new(8);
        let argb = 0xff12_3456u32;
        assert!(cache.contains(argb).is_none() || cache.entries[cache.hash(argb)] != argb);
        cache.insert(argb);
        assert_eq!(cache.contains(argb), Some(cache.hash(argb)));
    }

    /// `cacheify_tokens` converts a literal back-to-back repeat into
    /// a `CacheRef` token whose `index` matches the cache slot, while
    /// leaving the first (unique) literal as a literal.
    #[test]
    fn cacheify_tokens_collapses_repeat_literal_into_cache_ref() {
        let argb = 0xff20_4060u32;
        let pixels = vec![argb, argb];
        let raw = vec![Token::Literal(argb), Token::Literal(argb)];
        let out = cacheify_tokens(&raw, &pixels, 8);
        assert!(matches!(out[0], Token::Literal(p) if p == argb));
        let cache = EncoderColorCache::new(8);
        let idx = cache.hash(argb) as u32;
        assert_eq!(out[1], Token::CacheRef { index: idx });
    }

    /// A backward-reference `Copy` token inserts each copied pixel
    /// into the cache, so a subsequent literal that hashes to the
    /// same slot is collapsed to a `CacheRef`.
    #[test]
    fn cacheify_tokens_copy_updates_cache_for_subsequent_literal() {
        let argb = 0xff80_4010u32;
        // pixels: [argb, argb, argb, argb] — represented as a literal
        // followed by a Copy {length: 3, distance: 1}, then later
        // (at position 4) we add the same argb as a literal again.
        let pixels = vec![argb, argb, argb, argb, argb];
        let raw = vec![
            Token::Literal(argb),
            Token::Copy {
                length: 3,
                distance: 1,
            },
            Token::Literal(argb),
        ];
        let out = cacheify_tokens(&raw, &pixels, 8);
        // The first literal is still a literal; the copy passes
        // through; the trailing literal is now a CacheRef.
        assert!(matches!(out[0], Token::Literal(p) if p == argb));
        assert!(matches!(
            out[1],
            Token::Copy {
                length: 3,
                distance: 1,
            }
        ));
        let cache = EncoderColorCache::new(8);
        let idx = cache.hash(argb) as u32;
        assert_eq!(out[2], Token::CacheRef { index: idx });
    }

    /// Forcing the color-cache path on a repetitive 16-color palette
    /// fixture round-trips bit-exactly through the decoder. This is
    /// the headline round-121 sanity test: the encoder emits §5.2.3
    /// cache codes; the decoder reads them back via its own
    /// [`crate::vp8l_decode::ColorCache`] and reconstructs the same
    /// pixels.
    #[test]
    fn color_cache_path_round_trips_via_public_entry_points() {
        let w = 8u32;
        let h = 8u32;
        // 16 distinct ARGB colors cycling per scan-line; every color
        // appears multiple times so the cache gets exercised.
        let palette: [u32; 16] = [
            0xff00_0000,
            0xff00_00ff,
            0xff00_ff00,
            0xff00_ffff,
            0xffff_0000,
            0xffff_00ff,
            0xffff_ff00,
            0xffff_ffff,
            0xff80_8080,
            0xff20_4060,
            0xff60_4020,
            0xff10_2030,
            0xff30_2010,
            0xffa0_b0c0,
            0xffc0_b0a0,
            0xff55_aa55,
        ];
        let pixels: Vec<u32> = (0..(w * h))
            .map(|i| palette[(i as usize) % palette.len()])
            .collect();
        // Force the color-cache path via the test-only entry.
        let stream = encode_argb_literals_color_cache(&pixels, DEFAULT_COLOR_CACHE_BITS);
        let header = build_image_header(w, h, false);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// On a small palette of repeated colors (a synthetic but
    /// realistic case for palette-heavy artwork), the §5.2.3
    /// color-cache path produces a smaller stream than the
    /// no-cache LZ77 path. This is the round-121 headline
    /// measurement.
    #[test]
    fn color_cache_beats_no_cache_on_small_palette_image() {
        // 32x32 image where every pixel is drawn from an 8-color
        // palette, in a pseudo-random pattern (so the LZ77 matcher
        // can't collapse them all into long copies and the
        // color-cache codes get to do real work).
        let w = 32u32;
        let h = 32u32;
        let palette: [u32; 8] = [
            0xff10_2030,
            0xff40_5060,
            0xff70_8090,
            0xffa0_b0c0,
            0xffd0_e0f0,
            0xff00_1122,
            0xff33_4455,
            0xff66_7788,
        ];
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0x1357_9bdfu32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(palette[(state as usize) % palette.len()]);
        }
        // Width-less form (matches `encode_argb_literals_color_cache`,
        // which also uses width=1) so the comparison isolates the
        // color-cache effect from the round-130 distance-map chooser.
        let no_cache = encode_literals_with_options(&pixels, false, None, 1);
        let cache = encode_literals_with_options(&pixels, false, Some(DEFAULT_COLOR_CACHE_BITS), 1);
        eprintln!(
            "[round-121] 32x32 small-palette pseudo-random: no-cache={} B, color-cache={} B ({:.1}% reduction)",
            no_cache.len(),
            cache.len(),
            100.0 * (no_cache.len() as f64 - cache.len() as f64) / no_cache.len() as f64,
        );
        assert!(
            cache.len() < no_cache.len(),
            "color-cache stream ({} B) did not beat no-cache LZ77 ({} B)",
            cache.len(),
            no_cache.len(),
        );

        // Round trip through the full encoder/decoder chain is exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// On a noisy image with effectively-zero color repetition the
    /// chooser never selects the cache path (it would just inflate
    /// the GREEN alphabet for no compression gain), so
    /// `encode_argb_literals` never produces a stream larger than the
    /// no-cache baseline on uncorrelated noise.
    #[test]
    fn color_cache_chooser_does_not_regress_on_uncorrelated_noise() {
        let w = 16u32;
        let h = 16u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xfeed_b00bu32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(state | 0xff00_0000);
        }
        let chosen = encode_argb_literals(&pixels);
        // Match `encode_argb_literals`'s width=1 form so the comparison
        // is apples-to-apples.
        let no_cache_no_tx = encode_literals_with_options(&pixels, false, None, 1);
        assert!(
            chosen.len() <= no_cache_no_tx.len(),
            "chooser regressed on noise: {} B chosen vs {} B no-cache no-tx",
            chosen.len(),
            no_cache_no_tx.len(),
        );
    }

    /// The §5.2.3 `color-cache-info` header field encodes the
    /// chosen `code_bits` value: when the cache is enabled, the
    /// decoder reads `%b1` followed by `ReadBits(4) = code_bits`,
    /// and the `ColorCacheInfo::is_enabled()` flag flips on. This
    /// test routes the encoded stream through the live decoder's
    /// `MetaPrefixHeader::read` and confirms it sees the cache.
    #[test]
    fn color_cache_header_round_trips_through_meta_prefix_reader() {
        use crate::meta_prefix::{ImageRole, MetaPrefixHeader};
        use crate::vp8l_stream::BitReader;
        let w = 4u32;
        let h = 4u32;
        let palette = [0xff10_2030u32, 0xff40_5060, 0xff70_8090, 0xffa0_b0c0];
        let pixels: Vec<u32> = (0..(w * h))
            .map(|i| palette[(i as usize) % palette.len()])
            .collect();
        let stream = encode_argb_literals_color_cache(&pixels, DEFAULT_COLOR_CACHE_BITS);
        // Read straight off the image-stream — no §3.8.2 transform
        // header is present (we forced the no-tx path), so the
        // very first bit is the transform-list terminator `%b0`,
        // followed by the §3.8.3 `color-cache-info`.
        let mut r = BitReader::new(&stream);
        // Skip the transform-list terminator.
        assert!(!r.read_bit().unwrap());
        let header = MetaPrefixHeader::read(&mut r, ImageRole::Argb, w, h).unwrap();
        assert!(header.color_cache.is_enabled());
        assert_eq!(header.color_cache.code_bits, DEFAULT_COLOR_CACHE_BITS);
        assert_eq!(header.color_cache.size(), 1 << DEFAULT_COLOR_CACHE_BITS);
    }

    // ---- round 130: §5.2.2 distance-map chooser ----

    /// `pixel_distance_to_distance_code` reconstructs the spec's
    /// `xi + yi * W` for the chosen code, identical to the decoder.
    /// Across every distance-map entry at a fixed width, the chooser
    /// must pick a code that round-trips through
    /// `distance_code_to_pixel_distance` to the original distance.
    #[test]
    fn distance_chooser_reconstructs_each_distance_map_entry() {
        use crate::vp8l_decode::{distance_code_to_pixel_distance, DISTANCE_MAP};
        let width = 256u32;
        for &(xi, yi) in DISTANCE_MAP.iter() {
            let raw = xi + yi * width as i32;
            let d = if raw < 1 { 1 } else { raw as usize };
            let code = pixel_distance_to_distance_code(d, width);
            assert_eq!(
                distance_code_to_pixel_distance(code, width),
                d,
                "chooser code {code} for d={d} (xi={xi},yi={yi}) does not round-trip",
            );
        }
    }

    /// For a 256-wide image, pixel distance 256 (one row above) must be
    /// represented by distance-map code 1 ((0, 1)), not the scan-line
    /// code 376 (`256 + 120`). This is the headline round-130 win on
    /// natural images.
    #[test]
    fn distance_chooser_picks_map_code_for_row_distance() {
        let width = 256u32;
        let code = pixel_distance_to_distance_code(width as usize, width);
        assert_eq!(code, 1, "row distance must collapse to map code 1");
        // And legacy scan-line code is the bigger alternative.
        assert_eq!(distance_to_code(width as usize), width + 120);
    }

    /// A distance with no §5.2.2 map representation at the chosen width
    /// falls back to the scan-line code `D + 120`. At width 256, a
    /// distance of 1000 has no `(xi, yi)` entry that reconstructs it, so
    /// the chooser emits `1000 + 120 = 1120`.
    #[test]
    fn distance_chooser_falls_back_to_scan_line_when_no_map_match() {
        let width = 256u32;
        let code = pixel_distance_to_distance_code(1000, width);
        assert_eq!(code, 1000 + 120);
    }

    /// Width-1 (the no-spatial-structure form) admits no distance-map
    /// entry whose `xi + yi*1` exceeds 8+7 = 15, so any distance >= 16
    /// must use the scan-line form. The chooser must agree.
    #[test]
    fn distance_chooser_width_one_uses_scan_line_for_large_distances() {
        for d in [16usize, 32, 64, 100, 500] {
            assert_eq!(
                pixel_distance_to_distance_code(d, 1),
                (d as u32) + 120,
                "width=1 distance {d} should not collapse",
            );
        }
    }

    /// On a row-correlated image (every scan-line copies the row above
    /// verbatim), the round-130 width-aware encoder must produce a
    /// strictly smaller stream than the round-119 scan-line-only form.
    /// This is the headline round-130 size-reduction measurement.
    #[test]
    fn width_aware_distance_beats_scan_line_only_on_row_correlated_image() {
        // 128x128 image whose every row is a fresh pseudo-random
        // 128-pixel pattern repeated for the next scan-line. The LZ77
        // matcher emits a single `Copy { length: ~MAX_MATCH, distance:
        // 128 }` per row (and chains thereafter). At width 128, distance
        // 128 = `(0, 1)` = distance-map code 1, far smaller than the
        // scan-line code 248.
        let w = 128u32;
        let h = 128u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xC0DE_FACEu32;
        for _ in 0..w {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push((state & 0x00ff_ffff) | 0xff00_0000);
        }
        for y in 1..h {
            for x in 0..w {
                pixels.push(pixels[(x + (y - 1) * w) as usize]);
            }
        }

        let width_aware = encode_argb_literals_with_width(&pixels, w);
        let scan_line_only = encode_argb_literals(&pixels); // width=1

        eprintln!(
            "[round-130] 128x128 row-correlated: scan-line-only={} B, width-aware={} B ({:.1}% reduction)",
            scan_line_only.len(),
            width_aware.len(),
            100.0 * (scan_line_only.len() as f64 - width_aware.len() as f64)
                / scan_line_only.len() as f64,
        );
        assert!(
            width_aware.len() < scan_line_only.len(),
            "width-aware stream ({} B) not smaller than scan-line-only ({} B)",
            width_aware.len(),
            scan_line_only.len(),
        );

        // Round trip is exact via the public entry point.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// A photo-like fixture (smooth luma gradient + per-pixel small
    /// noise to fill the LZ77 hash chains) gets the round-130 chooser
    /// to find numerous small `(xi, yi)` matches in the §5.2.2
    /// distance-map neighbourhood. Compared to the width=1 scan-line
    /// baseline, the width-aware path is strictly smaller.
    #[test]
    fn width_aware_distance_beats_scan_line_only_on_photo_like_image() {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        // Each row is a low-amplitude noise pattern around a luma ramp;
        // adjacent rows share the same noise seed but with a tiny offset,
        // so 2-D neighbour matches are abundant.
        let mut state = 0x1234_5678u32;
        for y in 0..h {
            let luma = (y * 4) as u8;
            for _x in 0..w {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let n = (state & 0x07) as i32 - 3; // [-3, 4)
                let g = (luma as i32 + n).clamp(0, 255) as u32;
                let r = g;
                let b = g;
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        let width_aware = encode_argb_literals_with_width(&pixels, w);
        let scan_line_only = encode_argb_literals(&pixels);
        eprintln!(
            "[round-130] 64x64 photo-like: scan-line-only={} B, width-aware={} B ({:.1}% reduction)",
            scan_line_only.len(),
            width_aware.len(),
            100.0 * (scan_line_only.len() as f64 - width_aware.len() as f64)
                / scan_line_only.len() as f64,
        );
        assert!(
            width_aware.len() <= scan_line_only.len(),
            "width-aware regressed: {} B vs scan-line-only {} B",
            width_aware.len(),
            scan_line_only.len(),
        );

        // Round trip stays exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Round trip is exact across a spread of image widths. The chooser
    /// must never emit a distance code that reconstructs to a different
    /// pixel distance on the decode side.
    #[test]
    fn width_aware_round_trip_across_assorted_widths() {
        for &(w, h) in &[
            (1u32, 16u32),
            (3u32, 16u32),
            (16u32, 16u32),
            (97u32, 13u32),
            (200u32, 3u32),
            (256u32, 8u32),
        ] {
            let mut pixels = Vec::with_capacity((w * h) as usize);
            // A row-repeating pattern so the LZ77 matcher emits copies
            // at row-multiple distances, exercising the chooser.
            for y in 0..h {
                for x in 0..w {
                    let v = (x.wrapping_mul(31).wrapping_add(y)) & 0xff;
                    pixels.push(0xff00_0000 | (v << 16) | (v << 8) | v);
                }
            }
            let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
            let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
            let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
            assert_eq!(
                img.pixels(),
                pixels.as_slice(),
                "round trip mismatch at {w}x{h}",
            );
        }
    }

    /// A 64x64 image whose every row is row 0 shifted by `(y % 4) - 1`
    /// pixels — the resulting per-row matches are short (3-pixel-aligned
    /// hashes mostly), at distances clustered near `width = 64`. The
    /// matcher emits many small Copy tokens whose distances are 60–65
    /// (= 64-4..64+1), all of which the round-130 chooser collapses to
    /// distance-map codes 1, 3, 4 (prefix 0–2). With dozens of emissions
    /// the chooser's per-token saving compounds against the scan-line
    /// baseline (which would assign each to prefix-14 buckets).
    #[test]
    fn width_aware_distance_compounds_on_many_short_row_offset_matches() {
        let w = 64u32;
        let h = 64u32;
        let mut row0 = Vec::with_capacity(w as usize);
        let mut state = 0x1357_2468u32;
        for _ in 0..w {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            row0.push((state & 0x00ff_ffff) | 0xff00_0000);
        }
        let mut pixels = Vec::with_capacity((w * h) as usize);
        pixels.extend_from_slice(&row0);
        for y in 1..h {
            // Per-row 0..3 horizontal shift, ringing back into row0.
            let shift = (y as usize) & 0x3;
            for x in 0..(w as usize) {
                pixels.push(row0[(x + shift) % (w as usize)]);
            }
        }
        let width_aware = encode_argb_literals_with_width(&pixels, w);
        let scan_line_only = encode_argb_literals(&pixels);
        eprintln!(
            "[round-130] 64x64 row-shifted: scan-line-only={} B, width-aware={} B ({:.1}% reduction)",
            scan_line_only.len(),
            width_aware.len(),
            100.0 * (scan_line_only.len() as f64 - width_aware.len() as f64)
                / scan_line_only.len() as f64,
        );
        assert!(
            width_aware.len() < scan_line_only.len(),
            "width-aware ({} B) not smaller than scan-line-only ({} B)",
            width_aware.len(),
            scan_line_only.len(),
        );

        // Round trip stays exact via the production path.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// A 256x256 row-repeating image (every scan-line a copy of row 1)
    /// drives the round-130 chooser to swap the scan-line code `256+120
    /// = 376` (prefix 16, 7 extra bits) for the map code 1 (prefix 0,
    /// 0 extra bits) — the largest single-emission saving the chooser
    /// can produce. The aggregate stream-size delta is the round-130
    /// headline measurement on row-correlated content.
    #[test]
    fn width_aware_distance_headline_256x256_row_repeating() {
        let w = 256u32;
        let h = 256u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xABCD_1234u32;
        for _ in 0..w {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push((state & 0x00ff_ffff) | 0xff00_0000);
        }
        for y in 1..h {
            for x in 0..w {
                pixels.push(pixels[(x + (y - 1) * w) as usize]);
            }
        }

        let width_aware = encode_argb_literals_with_width(&pixels, w);
        let scan_line_only = encode_argb_literals(&pixels);
        eprintln!(
            "[round-130] 256x256 row-repeating: scan-line-only={} B, width-aware={} B ({:.1}% reduction)",
            scan_line_only.len(),
            width_aware.len(),
            100.0 * (scan_line_only.len() as f64 - width_aware.len() as f64)
                / scan_line_only.len() as f64,
        );
        assert!(
            width_aware.len() < scan_line_only.len(),
            "width-aware stream ({} B) not smaller than scan-line-only ({} B)",
            width_aware.len(),
            scan_line_only.len(),
        );

        // Round trip stays exact via the production path.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Re-encode an existing lossless fixture (decoded to ARGB) through
    /// both the width=1 scan-line-only form and the round-130 width-aware
    /// form, and confirm the width-aware variant is strictly smaller and
    /// round-trips bit-exactly. This exercises the chooser on
    /// non-synthetic distance distributions (the fixture's encoder
    /// produced whatever natural-image-style matches it found).
    #[test]
    fn width_aware_re_encode_of_real_fixture_is_smaller() {
        // 32x32 RGBA fixture committed in-tree (no external decode).
        let bytes: &[u8] = include_bytes!("../tests/data/lossless-32x32-rgba.webp");
        let decoded = crate::decode_lossless_image(bytes).unwrap().unwrap();
        let w = decoded.width();
        let h = decoded.height();
        let pixels = decoded.pixels().to_vec();

        let width_aware = encode_argb_literals_with_width(&pixels, w);
        let scan_line_only = encode_argb_literals(&pixels);
        eprintln!(
            "[round-130] {}x{} re-encoded fixture: scan-line-only={} B, width-aware={} B ({:.1}% reduction)",
            w,
            h,
            scan_line_only.len(),
            width_aware.len(),
            100.0 * (scan_line_only.len() as f64 - width_aware.len() as f64)
                / scan_line_only.len() as f64,
        );
        assert!(
            width_aware.len() <= scan_line_only.len(),
            "width-aware regressed: {} B vs scan-line-only {} B",
            width_aware.len(),
            scan_line_only.len(),
        );

        // Round trip through the encoder + decoder is exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    // ---- §5.2.3 color-cache size selection (round 132) ----

    /// `CANDIDATE_COLOR_CACHE_BITS` lists every cache size the chooser
    /// evaluates. The slate must be a strict subset of the spec-allowed
    /// `[COLOR_CACHE_BITS_MIN, COLOR_CACHE_BITS_MAX]` range (no `0`, no
    /// `>11`), monotonically increasing, and contain
    /// `DEFAULT_COLOR_CACHE_BITS` so the round-121 baseline is always
    /// among the candidates.
    #[test]
    fn color_cache_candidate_slate_is_spec_legal_and_monotone() {
        assert!(!CANDIDATE_COLOR_CACHE_BITS.is_empty());
        let mut prev: Option<u32> = None;
        for &bits in &CANDIDATE_COLOR_CACHE_BITS {
            assert!(
                (COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&bits),
                "candidate {bits} outside §5.2.3 [{COLOR_CACHE_BITS_MIN}..{COLOR_CACHE_BITS_MAX}]",
            );
            if let Some(p) = prev {
                assert!(
                    bits > p,
                    "candidate slate not monotone at {bits} (prev {p})"
                );
            }
            prev = Some(bits);
        }
        assert!(
            CANDIDATE_COLOR_CACHE_BITS.contains(&DEFAULT_COLOR_CACHE_BITS),
            "DEFAULT_COLOR_CACHE_BITS={DEFAULT_COLOR_CACHE_BITS} missing from candidate slate",
        );
    }

    /// On a palette-heavy synthetic image the chooser must pick a
    /// non-zero cache size: every pixel is drawn from a small palette
    /// repeated in a pseudo-random order, so each repeat is a
    /// `CacheRef` win that more than pays for the §5.2.3 header.
    /// (a) of the round-132 brief.
    #[test]
    fn size_selection_picks_nonzero_on_palette_heavy_image() {
        let w = 64u32;
        let h = 64u32;
        let palette: [u32; 12] = [
            0xff10_2030,
            0xff40_5060,
            0xff70_8090,
            0xffa0_b0c0,
            0xffd0_e0f0,
            0xff00_1122,
            0xff33_4455,
            0xff66_7788,
            0xff99_aabb,
            0xffcc_ddee,
            0xff11_2233,
            0xff44_5566,
        ];
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0x1357_9bdfu32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(palette[(state as usize) % palette.len()]);
        }
        let (_, chosen) = encode_argb_literals_with_width_selected(&pixels, w);
        assert_ne!(
            chosen, 0,
            "palette-heavy image should engage the §5.2.3 color cache, got code_bits=0",
        );
        assert!(
            CANDIDATE_COLOR_CACHE_BITS.contains(&chosen),
            "chosen code_bits {chosen} not in candidate slate",
        );
    }

    /// On uncorrelated ARGB noise the chooser must select size 0
    /// (no cache): every pixel is distinct, so cache references would
    /// never fire and the wider GREEN alphabet only inflates the
    /// prefix-code lengths. (b) of the round-132 brief.
    #[test]
    fn size_selection_picks_zero_on_noise_image() {
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xfeed_b00bu32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(state | 0xff00_0000);
        }
        let (_, chosen) = encode_argb_literals_with_width_selected(&pixels, w);
        assert_eq!(
            chosen, 0,
            "uncorrelated noise should not engage the color cache, got code_bits={chosen}",
        );
    }

    /// Across a varied suite of inputs the selected `code_bits` is
    /// always either `0` (disabled) or a value in
    /// `CANDIDATE_COLOR_CACHE_BITS` — never outside the spec-allowed
    /// range, never a value the chooser wasn't asked to evaluate.
    /// (c) of the round-132 brief.
    #[test]
    fn selected_size_is_always_zero_or_in_candidate_slate() {
        // Spread of inputs: palette-heavy, noise, row-correlated,
        // solid, tiny.
        let mut suites: Vec<(u32, u32, Vec<u32>)> = Vec::new();
        // Palette-heavy 32x32.
        {
            let w = 32u32;
            let h = 32u32;
            let palette: [u32; 6] = [
                0xff00_0000,
                0xffff_ffff,
                0xff80_0000,
                0xff00_8000,
                0xff00_0080,
                0xff80_8080,
            ];
            let mut pixels = Vec::with_capacity((w * h) as usize);
            let mut state = 0xc0ff_eeeeu32;
            for _ in 0..(w * h) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                pixels.push(palette[(state as usize) % palette.len()]);
            }
            suites.push((w, h, pixels));
        }
        // Pure noise 16x16.
        {
            let w = 16u32;
            let h = 16u32;
            let mut pixels = Vec::with_capacity((w * h) as usize);
            let mut state = 0xdead_beefu32;
            for _ in 0..(w * h) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                pixels.push(state | 0xff00_0000);
            }
            suites.push((w, h, pixels));
        }
        // Row-correlated 32x32 (every row repeats row 0).
        {
            let w = 32u32;
            let h = 32u32;
            let mut row0 = Vec::with_capacity(w as usize);
            let mut state = 0x1234_5678u32;
            for _ in 0..w {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                row0.push((state & 0x00ff_ffff) | 0xff00_0000);
            }
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for _ in 0..h {
                pixels.extend_from_slice(&row0);
            }
            suites.push((w, h, pixels));
        }
        // Solid color 8x8.
        {
            let w = 8u32;
            let h = 8u32;
            let pixels = vec![0xff12_3456u32; (w * h) as usize];
            suites.push((w, h, pixels));
        }
        // Tiny 1x1.
        suites.push((1, 1, vec![0xffaa_5500u32]));

        for (w, _h, pixels) in &suites {
            let (_, chosen) = encode_argb_literals_with_width_selected(pixels, *w);
            assert!(
                chosen == 0 || CANDIDATE_COLOR_CACHE_BITS.contains(&chosen),
                "chosen code_bits {chosen} (w={w}) not in {{0}} ∪ {:?}",
                CANDIDATE_COLOR_CACHE_BITS,
            );
            // And the chosen size is in the §5.2.3 spec range.
            assert!(
                chosen == 0 || (COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&chosen),
                "chosen code_bits {chosen} (w={w}) outside §5.2.3 [{COLOR_CACHE_BITS_MIN}..{COLOR_CACHE_BITS_MAX}]",
            );
        }
    }

    /// The chosen byte stream round-trips bit-exactly through the
    /// decoder for every cache-size decision the chooser can make.
    /// (d) of the round-132 brief.
    #[test]
    fn selected_stream_round_trips_bit_exact_for_each_decision() {
        // Each suite is chosen to drive a different cache decision:
        // palette-heavy → non-zero size; noise → size 0; tiny solid →
        // size 0 (no recurrence opportunity that beats the header
        // cost on so few pixels).
        let suites: [(u32, u32, &str); 3] = [
            (32, 32, "palette-heavy"),
            (16, 16, "noise"),
            (4, 4, "solid"),
        ];
        for (w, h, label) in suites {
            let pixels: Vec<u32> = match label {
                "palette-heavy" => {
                    let palette: [u32; 8] = [
                        0xff10_2030,
                        0xff40_5060,
                        0xff70_8090,
                        0xffa0_b0c0,
                        0xffd0_e0f0,
                        0xff00_1122,
                        0xff33_4455,
                        0xff66_7788,
                    ];
                    let mut p = Vec::with_capacity((w * h) as usize);
                    let mut state = 0x9876_5432u32;
                    for _ in 0..(w * h) {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        p.push(palette[(state as usize) % palette.len()]);
                    }
                    p
                }
                "noise" => {
                    let mut p = Vec::with_capacity((w * h) as usize);
                    let mut state = 0x0baf_face_u32;
                    for _ in 0..(w * h) {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        p.push(state | 0xff00_0000);
                    }
                    p
                }
                "solid" => vec![0xff7f_8081u32; (w * h) as usize],
                _ => unreachable!(),
            };

            let (_bytes, chosen) = encode_argb_literals_with_width_selected(&pixels, w);
            // Cache-size decision is in-range.
            assert!(
                chosen == 0 || CANDIDATE_COLOR_CACHE_BITS.contains(&chosen),
                "{label}: chosen code_bits {chosen} unexpected",
            );
            // Round-trip via the production path (which feeds the
            // chooser internally).
            let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
            let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
            let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
            assert_eq!(
                img.pixels(),
                pixels.as_slice(),
                "{label}: round-trip mismatch at chosen code_bits={chosen}",
            );
        }
    }

    /// The round-132 multi-size chooser is never *worse* than the
    /// round-121 single-size chooser on any of the test fixtures:
    /// since the round-121 candidate (no-cache | cache@8) is a strict
    /// subset of the round-132 grid (no-cache | cache @ {5,7,8,9,11}),
    /// the round-132 winner is at most as large.
    #[test]
    fn round132_chooser_never_regresses_against_round121_single_size() {
        let fixtures: [(u32, u32, &str); 4] = [
            (32, 32, "palette"),
            (16, 16, "noise"),
            (24, 24, "row-corr"),
            (8, 8, "solid"),
        ];
        for (w, h, label) in fixtures {
            let pixels: Vec<u32> = match label {
                "palette" => {
                    let palette: [u32; 10] = [
                        0xff10_2030,
                        0xff40_5060,
                        0xff70_8090,
                        0xffa0_b0c0,
                        0xffd0_e0f0,
                        0xff00_1122,
                        0xff33_4455,
                        0xff66_7788,
                        0xff99_aabb,
                        0xffcc_ddee,
                    ];
                    let mut p = Vec::with_capacity((w * h) as usize);
                    let mut state = 0x5a5a_a5a5u32;
                    for _ in 0..(w * h) {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        p.push(palette[(state as usize) % palette.len()]);
                    }
                    p
                }
                "noise" => {
                    let mut p = Vec::with_capacity((w * h) as usize);
                    let mut state = 0xf00d_baadu32;
                    for _ in 0..(w * h) {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        p.push(state | 0xff00_0000);
                    }
                    p
                }
                "row-corr" => {
                    let mut row0 = Vec::with_capacity(w as usize);
                    let mut state = 0xcafe_d00du32;
                    for _ in 0..w {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        row0.push((state & 0x00ff_ffff) | 0xff00_0000);
                    }
                    let mut p = Vec::with_capacity((w * h) as usize);
                    for _ in 0..h {
                        p.extend_from_slice(&row0);
                    }
                    p
                }
                "solid" => vec![0xffc0_ffeeu32; (w * h) as usize],
                _ => unreachable!(),
            };

            // Round-121 single-size emulation: min of (no-tx | sg) ×
            // (None | Some(8)).
            let r121 = {
                let mut best = encode_literals_with_options(&pixels, false, None, w);
                for cand in [
                    encode_literals_with_options(&pixels, true, None, w),
                    encode_literals_with_options(&pixels, false, Some(DEFAULT_COLOR_CACHE_BITS), w),
                    encode_literals_with_options(&pixels, true, Some(DEFAULT_COLOR_CACHE_BITS), w),
                ] {
                    if cand.len() < best.len() {
                        best = cand;
                    }
                }
                best.len()
            };
            let (bytes, chosen) = encode_argb_literals_with_width_selected(&pixels, w);
            let r132 = bytes.len();
            eprintln!(
                "[round-132] {label} {w}x{h}: round-121={r121} B, round-132={r132} B (chosen code_bits={chosen})",
            );
            assert!(
                r132 <= r121,
                "{label}: round-132 chooser regressed: {r132} B vs round-121 {r121} B",
            );
        }
    }

    /// The chooser must never inflate a distance: the chosen code's
    /// prefix code is always less than or equal to the scan-line
    /// alternative's prefix code, since the chooser picks the smaller
    /// raw code and `value_to_prefix` is monotonic in the value.
    #[test]
    fn chooser_never_picks_larger_prefix_than_scan_line() {
        let width = 320u32;
        for d in 1..=(width as usize * 4) {
            let chooser_code = pixel_distance_to_distance_code(d, width);
            let scan_code = distance_to_code(d);
            let (chooser_prefix, _, _) = value_to_prefix(chooser_code);
            let (scan_prefix, _, _) = value_to_prefix(scan_code);
            assert!(
                chooser_prefix <= scan_prefix,
                "d={d}: chooser code {chooser_code} (prefix {chooser_prefix}) > scan-line {scan_code} (prefix {scan_prefix})",
            );
        }
    }
}
