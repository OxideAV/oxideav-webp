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
//! * **No §3.8.3 color cache** — `color-cache-info` is the single `%b0`
//!   bit. Every pixel is emitted directly.
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
//! via the GREEN alphabet's length-prefix symbols (`256 + prefix_code`),
//! and the distance via prefix code #5 using the *scan-line* encoding
//! `distance_code = D + NUM_DISTANCE_MAP_CODES` (the §5.2.2 distance map is
//! an optional decoder convenience for nearby pixels; emitting
//! `D + 120` is always valid and the in-crate decoder's
//! [`crate::vp8l_decode::distance_code_to_pixel_distance`] reconstructs `D`
//! exactly). The inverse of the §5.2.2 prefix-value transform
//! ([`value_to_prefix`]) splits a length/distance into its prefix code and
//! extra bits, the exact counterpart of the decoder's
//! [`crate::vp8l_decode::read_lz77_value`].
//!
//! The literal-only path is still available via [`encode_argb_literals_only`]
//! (used by the size-reduction comparison test); the default
//! [`encode_argb_literals`] entry point chooses the LZ77 path.
//!
//! ## What this module does NOT do
//!
//! * No §3.8.2 transform encoding (predictor / color / subtract-green /
//!   color-indexing). Pass-through only.
//! * No §3.8.3 color cache. (LZ77 backward references *are* emitted.)
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
/// pixel (a §5.2.1 literal) or a backward-reference copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    /// A §5.2.1 ARGB literal pixel.
    Literal(u32),
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

/// The five per-symbol frequency tables for one prefix-code group: green
/// (literals + §5.2.2 length symbols), red, blue, alpha, and distance.
struct Frequencies {
    green: Vec<u32>,
    red: Vec<u32>,
    blue: Vec<u32>,
    alpha: Vec<u32>,
    distance: Vec<u32>,
}

/// The §5.2.2 distance-code form this encoder uses: the *scan-line*
/// encoding (`distance_code = D + 120`). The decoder's
/// [`crate::vp8l_decode::distance_code_to_pixel_distance`] maps any
/// `distance_code > 120` straight back to `distance_code - 120 == D`, so
/// every distance round-trips without touching the §5.2.2 distance map.
fn distance_to_code(distance: usize) -> u32 {
    distance as u32 + crate::vp8l_decode::NUM_DISTANCE_MAP_CODES as u32
}

/// Accumulate the per-symbol frequencies for a token stream so the entropy
/// stage can build length-optimal prefix codes before emitting.
fn count_frequencies(tokens: &[Token]) -> Frequencies {
    let green_alphabet = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES;
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
            Token::Copy { length, distance } => {
                // §5.2.2: length is a GREEN symbol `256 + length_prefix`.
                let (len_prefix, _, _) = value_to_prefix(length as u32);
                freqs.green[256 + len_prefix as usize] += 1;
                // Distance prefix code (#5).
                let (dist_prefix, _, _) = value_to_prefix(distance_to_code(distance));
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
    let tokens = tokenize_lz77(pixels);
    let no_tx = encode_tokens(&tokens, false);
    // Subtract-green path: apply the forward transform, re-tokenize the
    // residual pixels (matches change because the per-pixel values do),
    // and prefix the §3.8.2 transform header.
    let mut sg_pixels = pixels.to_vec();
    apply_subtract_green(&mut sg_pixels);
    let sg_tokens = tokenize_lz77(&sg_pixels);
    let sg = encode_tokens(&sg_tokens, true);
    if sg.len() < no_tx.len() {
        sg
    } else {
        no_tx
    }
}

/// Encode an ARGB image with the literal-only, no-transform path: every
/// pixel becomes a §5.2.1 ARGB literal and no §3.8.2 transform is written.
/// Retained as the baseline the round-119 size-reduction test compares the
/// LZ77 path against; [`encode_argb_literals`] is the default entry point.
pub fn encode_argb_literals_only(pixels: &[u32]) -> Vec<u8> {
    let tokens: Vec<Token> = pixels.iter().map(|&p| Token::Literal(p)).collect();
    encode_tokens(&tokens, false)
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
    encode_tokens(&tokens, true)
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
fn encode_tokens(tokens: &[Token], subtract_green: bool) -> Vec<u8> {
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
    // color-cache-info: `%b0` (no color cache).
    w.write_bit(false);
    // meta-prefix: `%b0` (single prefix-code group).
    w.write_bit(false);

    // Build the five prefix codes from token frequencies. The GREEN
    // alphabet covers literals (`< 256`) *and* the §5.2.2 length prefix
    // symbols (`256 + length_prefix`). The distance alphabet (40 codes) is
    // exercised only when the matcher emitted at least one copy.
    let freqs = count_frequencies(tokens);
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
    // (channel order green, red, blue, alpha) or a §5.2.2 length + distance
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
            Token::Copy { length, distance } => {
                // §5.2.2: length via a GREEN length symbol (base 256), then
                // distance via prefix code #5 (base 0).
                write_lz77_value(&mut w, &green_code, 256, length as u32);
                write_lz77_value(&mut w, &dist_code, 0, distance_to_code(distance));
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
    let stream = encode_argb_literals(pixels);
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
            encode_tokens(&tokens, false)
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

    /// `encode_argb_literals` picks the smaller of the two evaluated
    /// paths, so on a green-correlated image its output equals the
    /// subtract-green path (or smaller).
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
        let sg = encode_argb_literals_subtract_green(&pixels);
        let no_tx = {
            let tokens = tokenize_lz77(&pixels);
            encode_tokens(&tokens, false)
        };
        assert_eq!(chosen.len(), sg.len().min(no_tx.len()));
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
            encode_tokens(&tokens, false)
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
}
