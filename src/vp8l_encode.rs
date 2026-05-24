//! VP8L (WebP-Lossless) §3.8 / §3.7 *encoder*.
//!
//! This is the writer counterpart of the round-99..111 decoder stack. The
//! decoder ([`crate::vp8l_transform::decode_lossless`]) walks a VP8L chunk
//! payload — §3.4 image-header, §3.8.2 transform list, §3.8.3 image data
//! (color-cache-info, meta-prefix, prefix-codes, LZ77-coded image) — and
//! produces ARGB pixels. This module produces a VP8L chunk payload from
//! ARGB pixels, taking the simplest end-to-end path the spec admits:
//!
//! * **No §3.8.2 transform** — the `optional-transform` loop emits a
//!   single `%b0` terminator (pass-through). The four transforms
//!   (predictor / color / subtract-green / color-indexing) are all
//!   optional; a decoder reconstructs the exact pixels without them.
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
//! ## What this module does NOT do
//!
//! * No §3.8.2 transform encoding (predictor / color / subtract-green /
//!   color-indexing). Pass-through only.
//! * No §3.8.3 LZ77 match search or color cache. Literal-only.
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

/// Encode an ARGB image to a VP8L *image-stream* (the bytes that follow the
/// §3.4 5-byte image-header), using the literal-only / no-transform path.
///
/// `pixels` is `width * height` ARGB values in scan-line order, each
/// `(alpha << 24) | (red << 16) | (green << 8) | blue` — the same layout
/// [`crate::vp8l_decode::DecodedImage::pixels`] produces. The returned
/// bytes, prefixed with the image-header and wrapped in RIFF/WEBP framing,
/// decode back to `pixels` exactly.
pub fn encode_argb_literals(pixels: &[u32]) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §3.8.2 optional-transform: none. Single `%b0` terminator.
    w.write_bit(false);

    // §3.8.3 spatially-coded-image = color-cache-info meta-prefix data.
    // color-cache-info: `%b0` (no color cache).
    w.write_bit(false);
    // meta-prefix: `%b0` (single prefix-code group).
    w.write_bit(false);

    // Build the five prefix codes from literal frequencies.
    // Prefix #1 (green): alphabet 256 + 24 (length codes) + 0 (no cache).
    // We only emit literal green symbols (< 256), no length codes.
    let green_alphabet = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES;
    let mut green_freq = vec![0u32; green_alphabet];
    let mut red_freq = vec![0u32; 256];
    let mut blue_freq = vec![0u32; 256];
    let mut alpha_freq = vec![0u32; 256];
    for &p in pixels {
        let a = (p >> 24) & 0xff;
        let r = (p >> 16) & 0xff;
        let g = (p >> 8) & 0xff;
        let b = p & 0xff;
        green_freq[g as usize] += 1;
        red_freq[r as usize] += 1;
        blue_freq[b as usize] += 1;
        alpha_freq[a as usize] += 1;
    }

    let green_code = WriteCode::from_freqs(&green_freq);
    let red_code = WriteCode::from_freqs(&red_freq);
    let blue_code = WriteCode::from_freqs(&blue_freq);
    let alpha_code = WriteCode::from_freqs(&alpha_freq);
    // Prefix #5 (distance): no backward references → empty code (single
    // symbol 0), alphabet 40.
    let dist_code = WriteCode::empty(40);

    // data = prefix-codes lz77-coded-image.
    // prefix-code-group = 5 prefix codes, in bitstream order:
    // green, red, blue, alpha, distance.
    green_code.write_code_lengths(&mut w);
    red_code.write_code_lengths(&mut w);
    blue_code.write_code_lengths(&mut w);
    alpha_code.write_code_lengths(&mut w);
    dist_code.write_code_lengths(&mut w);

    // lz77-coded-image: one ARGB literal per pixel (§3.7.3 order:
    // green, red, blue, alpha).
    for &p in pixels {
        let a = ((p >> 24) & 0xff) as usize;
        let r = ((p >> 16) & 0xff) as usize;
        let g = ((p >> 8) & 0xff) as usize;
        let b = (p & 0xff) as usize;
        green_code.write_symbol(&mut w, g);
        red_code.write_symbol(&mut w, r);
        blue_code.write_symbol(&mut w, b);
        alpha_code.write_symbol(&mut w, a);
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
}
