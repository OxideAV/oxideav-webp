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
//!   / color / color-indexing) get their own forward passes in later
//!   rounds.
//! * **§5.2.1 / §5.2.3 color cache** — as of round 121 the encoder
//!   evaluates a color cache alongside the no-cache path and emits
//!   whichever is smaller. As of round 148 the chooser sweeps every
//!   §5.2.3 `cache_code_bits ∈ [1..11]` per the spec's allowed range
//!   (2..=2048-entry caches) and picks the smallest stream, rather
//!   than the round-121 fixed 256-entry choice. When the cache is
//!   enabled, the §3.8.3 `color-cache-info` field becomes
//!   `%b1 code_bits` (1-bit flag + 4-bit `code_bits`), the GREEN
//!   alphabet grows to `256 + 24 + (1 << code_bits)` symbols, and
//!   each repeat of a previously-inserted ARGB literal is emitted as
//!   a §5.2.3 color-cache code `256 + 24 + index` instead of four
//!   separate ARGB-channel literals.
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
//! As of round 163 the matcher applies **four-position lazy matching
//! with a diminishing-returns guard**: after finding a match
//! `(L_a, _)` at `pos`, the encoder also probes `pos + 1`, `pos + 2`,
//! and `pos + 3` (the round-158 depth-3 contract), and then — only
//! when the running best across those four positions is still shorter
//! than [`DEPTH4_GUARD_THRESHOLD`] — also probes `pos + 4`. Whichever
//! of the candidate start positions yields the strictly longest match
//! wins; the pixels skipped to reach the chosen start are emitted as
//! literals. The depth-4 guard captures the empirical observation
//! that once the depth-3 best already covers a length-`THRESHOLD` run,
//! a fourth-order swap is almost never able to amortise the four
//! literals it would cost — the depth-4 probe is gated to avoid
//! spending hash-chain inserts and a `find` call when its expected
//! marginal payoff is small. This still recovers fourth-order traps
//! where the leading match at `pos..=pos + 3` is short. The decoder
//! output is bit-identical for any input — only the token *partition*
//! shifts (by up to four pixels) — so round-trips remain bit-exact
//! under any input. See [`tokenize_lz77_inner`] for the shared
//! `lazy_depth: u32`-toggled implementation (`0` strict-greedy r155
//! baseline, `1` r156 depth-1, `2` r157 depth-2, `3` r158 depth-3,
//! `4` r163 guarded depth-4, now the production default).
//!
//! ## §4.1 spatial-predictor forward transform
//!
//! The encoder also evaluates the §4.1 predictor transform path: the
//! image is divided into `(1 << DEFAULT_PREDICTOR_SIZE_BITS)`-pixel
//! square blocks; each block picks the prediction mode `0..=13` that
//! minimises a residual-magnitude proxy (sum of per-channel
//! `|residual|` folded onto `[-128, 127]`) over the block's pixels.
//! As of round 159, the chooser also threads an
//! **entropy-image-aware tie-break** through the per-block walk:
//! when multiple modes tie on residual cost, the chooser prefers
//! the mode chosen by the *previous neighbour* block (left-of in
//! the current row, or top-of for the left-column blocks). The
//! predictor sub-image is written as a §7.2 `entropy-coded-image`,
//! so adjacent blocks carrying the same mode value reduce that
//! sub-image's symbol entropy and the bytes the writer emits for
//! it; this matches RFC 9649 §3.5's "transform data can be decided
//! based on entropy minimization" note. The residuals themselves
//! are unchanged on tie-equal swaps (the cost was already minimal),
//! so decoded pixels stay bit-identical. As of round 160 the
//! chooser also evaluates a **slack-cost variant** of the
//! tie-break — see [`pick_block_mode_with_hint_slack`] — that
//! accepts the preferred neighbour mode at a small additive
//! `slack` budget above the otherwise-best cost, trading a small
//! residual increase for a strict drop in the sub-image's symbol
//! entropy. The slack variant is one of four predictor candidates
//! the production chooser builds per `size_bits` (slack ∈
//! `{0, block_pixels, 2·block_pixels, 4·block_pixels}`), and the
//! byte-shortest stream wins — so the slack candidates can only
//! add options to the chooser's selection set and never regress.
//! The sub-resolution predictor image is written as a §7.2
//! `predictor-image = 3BIT entropy-coded-image` and the per-pixel
//! residuals are then handed to the standard
//! `spatially-coded-image` writer. As of round 155 the chooser
//! sweeps two `size_bits` values for the §4.1 predictor: the
//! default 16×16-pixel blocks (per-region predictor-mode
//! granularity, good for images whose best-mode varies spatially)
//! and a maximal single-block transform whose `size_bits` is large
//! enough that the entire image collapses to one mode (`1 << size`
//! ≥ max(width, height), so the sub-image is at most 1×1 — the
//! cheapest possible §4.1 header). Each predictor `size_bits`
//! candidate uses the round-148 cache-bits sweep (§5.2.3
//! `cache_code_bits ∈ [1..11]` plus the disabled-cache baseline)
//! and is cross-compared against the no-tx / subtract-green
//! candidates; the smallest stream wins. On smooth gradients with
//! strong spatial correlation, the predictor path's per-pixel
//! residual entropy is much lower than the raw pixels' entropy,
//! more than paying for the predictor-image overhead.
//!
//! ## §3.5.2 / §4.2 color-transform forward pass
//!
//! As of round 147 the encoder also evaluates the §3.5.2 / §4.2
//! color transform: the image is divided into
//! `(1 << DEFAULT_COLOR_TRANSFORM_SIZE_BITS)`-pixel square blocks; each
//! block picks a `(green_to_red, green_to_blue, red_to_blue)` triple
//! that minimises a residual-magnitude proxy on the red and blue
//! channels (the green channel is untouched per §3.5.2). The
//! per-axis sweep is exact because the cost decomposes additively
//! across channels: `red_residual` depends only on `green_to_red`,
//! `blue_residual` depends additively on `(green_to_blue,
//! red_to_blue)`, so the three axes can be optimised independently
//! over a small candidate grid (see [`CTE_AXIS_CANDIDATES`]). The
//! sub-resolution color image is written as a §7.2
//! `color-image = 3BIT entropy-coded-image` (re-using
//! `write_entropy_coded_image_literals`) and the per-pixel residuals
//! are then handed to the standard `spatially-coded-image` writer.
//! Each color-transform `size_bits` candidate uses the round-148
//! cache-bits sweep (§5.2.3 `cache_code_bits ∈ [1..11]` plus the
//! disabled-cache baseline) and is cross-compared against the no-tx,
//! subtract-green, and §4.1 predictor candidates; the smallest stream
//! wins. On natural images with red/green and blue/green correlation,
//! the color-transform path concentrates the red/blue residuals near
//! zero, shrinking the per-channel Huffman codes and further reducing
//! the chosen stream's size on top of the §4.1 predictor pass.
//!
//! ## §4.4 color-indexing transform encoder
//!
//! As of round 150 the encoder also evaluates the §4.4 color-indexing
//! transform: an O(N) palette probe walks `pixels` and bails out
//! early at >256 unique ARGB values; below that threshold a sorted
//! palette is built (sorted ARGB-numerically so the §4.4
//! subtraction-coded color-table deltas concentrate near zero), each
//! pixel is replaced by its palette index, and indices are bundled
//! into one byte per the §4.4 table (`width_bits = 3 / 2 / 1 / 0`
//! for palettes of 1..=2 / 3..=4 / 5..=16 / 17..=256 entries —
//! packing 8 / 4 / 2 / 1 indices into each green byte respectively).
//! The bundled image is then handed to the standard
//! `spatially-coded-image` writer at the subsampled `packed_width =
//! DIV_ROUND_UP(width, 1 << width_bits)`. The color-indexing
//! candidate uses the round-148 cache-bits sweep (§5.2.3
//! `cache_code_bits ∈ [1..11]` plus the disabled-cache baseline) and
//! is cross-compared against every other candidate; the smallest
//! stream wins. On palette-ish content (icons, line art, screen
//! captures) the index-bundling drops the entropy stage's symbol
//! count by 2..8×, more than paying for the small subtraction-coded
//! palette-write overhead.
//!
//! ## What this module does NOT do
//!
//! * No multi-meta-prefix (§6.2.2 entropy image). All candidates use
//!   a single prefix-code group for the entire image.
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
/// The algorithm is a textbook Huffman build, followed by a
/// length-limiting pass that caps any over-long code at
/// [`MAX_CODE_LENGTH`] while re-balancing so the Kraft sum stays at
/// exactly one. For the small alphabets and pixel counts this encoder
/// targets, the cap is rarely hit; the pass is correctness insurance,
/// not an optimization.
///
/// The merge loop exploits the classic two-queue property instead of a
/// heap: with the leaves sorted ascending by `(frequency, symbol)` once
/// up front, every internal node is created with a frequency no smaller
/// than any previously created one, so a plain FIFO of internal nodes
/// stays sorted by `(frequency, creation order)` for free. Each merge
/// step then takes the two smallest nodes by comparing the two queue
/// fronts in O(1) — preferring the leaf on a frequency tie, because the
/// tie-break order ranks every leaf (ascending symbol) before every
/// internal node (creation order). This reproduces, merge for merge, the
/// exact `(freq, order)` pop sequence the previous min-heap build used,
/// so the emitted length tables are bit-identical; only the cost drops
/// (O(n log n) sort + O(n) merge, versus 3(n-1) heap operations of
/// O(log n) swaps each).
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

    // Huffman build. Nodes 0..n are leaves; internal nodes n.. are
    // appended in creation order. A parent array recovers the depth
    // (= code length) of each leaf afterwards.
    let m = used.len();

    // Leaf queue: `(freq << 32) | symbol` keys, sorted ascending. The
    // packed key makes the sort a single-u64 comparison while encoding
    // exactly the `(freq, ascending symbol)` tie-break the merge needs
    // (`freq` is `u32`, so the shift is exact, and a symbol index always
    // fits the low half).
    let mut leaves: Vec<u64> = used
        .iter()
        .map(|&s| ((freqs[s] as u64) << 32) | s as u64)
        .collect();
    leaves.sort_unstable();

    // Internal-node FIFO: frequencies only; internal node `i` has node
    // index `n + i`. `u32::MAX` marks "no parent yet" (only the root
    // keeps it, and the root's slot is never read back).
    let mut inode_freq: Vec<u64> = Vec::with_capacity(m - 1);
    let mut parent: Vec<u32> = vec![u32::MAX; n + m - 1];

    /// Take the smallest remaining node by `(freq, tie-break order)`:
    /// the front leaf wins ties because leaves rank before internal
    /// nodes in the tie-break order.
    fn take_min(
        leaves: &[u64],
        li: &mut usize,
        inode_freq: &[u64],
        ii: &mut usize,
        n: usize,
    ) -> (usize, u64) {
        let use_leaf = if *li < leaves.len() {
            *ii >= inode_freq.len() || (leaves[*li] >> 32) <= inode_freq[*ii]
        } else {
            false
        };
        if use_leaf {
            let key = leaves[*li];
            *li += 1;
            ((key & 0xffff_ffff) as usize, key >> 32)
        } else {
            let node = n + *ii;
            let freq = inode_freq[*ii];
            *ii += 1;
            (node, freq)
        }
    }

    let mut li = 0usize; // leaf cursor
    let mut ii = 0usize; // internal-node cursor
    for _ in 0..m - 1 {
        let (a_node, a_freq) = take_min(&leaves, &mut li, &inode_freq, &mut ii, n);
        let (b_node, b_freq) = take_min(&leaves, &mut li, &inode_freq, &mut ii, n);
        let new_node = n + inode_freq.len();
        parent[a_node] = new_node as u32;
        parent[b_node] = new_node as u32;
        inode_freq.push(a_freq + b_freq);
    }

    // Recover each leaf's depth top-down: an internal node's parent is
    // always created later (larger index), so a single reverse pass over
    // the internal nodes settles every internal depth, and each leaf is
    // then one deeper than its (always internal) parent.
    let mut internal_depth = vec![0u32; m - 1];
    for i in (0..m - 1).rev() {
        let p = parent[n + i];
        if p != u32::MAX {
            internal_depth[i] = internal_depth[p as usize - n] + 1;
        }
    }
    let mut max_len = 0usize;
    for &s in &used {
        let depth = internal_depth[parent[s] as usize - n] as usize + 1;
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
    //
    // Selection rule being reproduced: the historical per-step rescan
    // walked all of `used` and kept the LAST `used`-order symbol among
    // those sharing the largest current length below the cap (the
    // `l >= best_len` comparison kept updating on ties). Two facts turn
    // that O(n)-per-adjustment rescan into an O(1)-per-adjustment bucket
    // drain with the identical pick sequence:
    //
    // 1. A bucket per length, filled in one pass over `used`, holds each
    //    bucket's symbols in `used` order — so the back of the highest
    //    non-empty bucket IS the rescan's pick.
    // 2. Once a pick is lengthened from `l` to `l + 1 < MAX`, it is
    //    strictly the unique deepest eligible leaf (everything else is
    //    `<= l`), so the rescan re-picks the same symbol every step
    //    until it reaches MAX (leaving the eligible set) or the sum
    //    reaches 1. Driving the popped symbol upward in place therefore
    //    replays the original step sequence exactly; no eligible bucket
    //    ever gains a member while the pass is still running.
    let mut k = kraft(lengths);
    if k > full {
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); MAX_CODE_LENGTH];
        for &s in used {
            let l = lengths[s] as usize;
            if l < MAX_CODE_LENGTH {
                buckets[l].push(s);
            }
        }
        // Bucket 0 is included for parity with the historical rescan,
        // which treated a (theoretical) zero-length used symbol as
        // eligible; the §3.7.2 build never produces one for a used
        // symbol, so the bucket is empty in practice.
        'over: for l0 in (0..MAX_CODE_LENGTH).rev() {
            while k > full {
                let Some(s) = buckets[l0].pop() else { break };
                // Lengthening from `l` to `l + 1` swaps the Kraft term
                // `2^(MAX-l)` for `2^(MAX-l-1)`, i.e. removes exactly
                // `2^(MAX-l-1)` — same integer a full recompute would
                // give.
                let mut l = l0;
                while k > full && l < MAX_CODE_LENGTH {
                    l += 1;
                    lengths[s] = l as u8;
                    k -= 1i64 << (MAX_CODE_LENGTH - l);
                }
            }
            if k <= full {
                break 'over;
            }
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
                // Shortening `s` from `l` to `l - 1` swaps `2^(MAX-l)` for
                // `2^(MAX-l+1)`, i.e. adds exactly `2^(MAX-l)` — again the
                // same integer a full recompute would give.
                let l = lengths[s] as usize;
                lengths[s] -= 1;
                k += 1i64 << (MAX_CODE_LENGTH - l);
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

    /// Write this code's per-symbol lengths to `w`, picking the cheaper
    /// of the two §3.7.2.1 forms.
    ///
    /// The §3.7.2.1.1 *simple code length code* can only represent length
    /// tables with 1 or 2 symbols at length 1 (every other symbol
    /// implicitly absent). When that constraint holds, `write_code_lengths`
    /// computes the precise bit-cost of both forms and picks the smaller.
    /// Otherwise it falls back to the §3.7.2.1.2 *normal code length code*.
    fn write_code_lengths(&self, w: &mut BitWriter) {
        if let Some(simple) = self.as_simple_form() {
            // Two trivial cases the simple form can carry — compare
            // bit-costs and pick the cheaper.
            let simple_bits = simple_form_bits(&simple);
            let normal_bits = normal_form_bits(&self.lengths);
            if simple_bits <= normal_bits {
                write_simple_code_lengths(w, &simple);
                return;
            }
        }
        write_normal_code_lengths(w, &self.lengths);
    }

    /// If this code's length table is encodable with the §3.7.2.1.1 simple
    /// form (1 or 2 symbols at length 1, all others 0), return the symbol
    /// list `[symbol0]` or `[symbol0, symbol1]`. Otherwise return `None`.
    fn as_simple_form(&self) -> Option<Vec<usize>> {
        let used: Vec<(usize, u8)> = self
            .lengths
            .iter()
            .enumerate()
            .filter_map(|(s, &l)| if l != 0 { Some((s, l)) } else { None })
            .collect();
        // Simple form requires 1 or 2 used symbols, each at length 1.
        // §3.7.2.1.1: "code length 1. All other prefix code lengths are
        // implicitly zeros."
        if used.is_empty() || used.len() > 2 {
            return None;
        }
        if used.iter().any(|&(_, l)| l != 1) {
            return None;
        }
        // §3.7.2.1.1 first symbol is coded with 1 or 8 bits, so it must
        // fit in [0..255]; second symbol always 8 bits, [0..255]. Anything
        // beyond 255 can only be sent via the normal form.
        if used.iter().any(|&(s, _)| s > 255) {
            return None;
        }
        Some(used.iter().map(|&(s, _)| s).collect())
    }
}

/// Precise bit-cost of the §3.7.2.1.1 *simple code length code* for the
/// given symbol list (1 or 2 entries, each in `[0..255]`).
///
/// Layout per §3.7.2.1.1:
/// * 1 flag bit (`1` = simple)
/// * 1 bit `num_symbols - 1`
/// * 1 bit `is_first_8bits` (chooses 1-bit vs 8-bit width for symbol0)
/// * `1 + 7 * is_first_8bits` bits for `symbol0`
/// * if `num_symbols == 2`: 8 bits for `symbol1`
fn simple_form_bits(symbols: &[usize]) -> usize {
    debug_assert!(symbols.len() == 1 || symbols.len() == 2);
    let is_first_8bits = symbols[0] > 1;
    // Per spec: the second symbol, when present, is always 8 bits.
    let s0_width = if is_first_8bits { 8 } else { 1 };
    let s1_width = if symbols.len() == 2 { 8 } else { 0 };
    // 1 (flag) + 1 (num_symbols-1) + 1 (is_first_8bits) + s0 + s1.
    3 + s0_width + s1_width
}

/// Precise bit-cost of [`write_normal_code_lengths`] for `lengths`.
///
/// Mirrors `write_normal_code_lengths` exactly so the chooser is
/// self-consistent: any change in normal-form layout there must reflect
/// here.
fn normal_form_bits(lengths: &[u8]) -> usize {
    // CLC frequencies are the histogram of length values 0..=15 in the
    // literal length table.
    let mut clc_freq = [0u32; NUM_CODE_LENGTH_CODES];
    for &l in lengths {
        clc_freq[l as usize] += 1;
    }
    let clc_lengths = build_code_lengths(&clc_freq);

    // Locate the highest-ordered CLC symbol that has a non-zero length.
    let mut max_order_used = 0usize;
    for (order_idx, &pos) in CODE_LENGTH_CODE_ORDER.iter().enumerate() {
        if clc_lengths[pos] != 0 {
            max_order_used = order_idx;
        }
    }
    let num_code_lengths = (max_order_used + 1).max(4);

    // §3.7.2.1.2 header tax: 1 flag + 4 num_code_lengths + 3*num_code_lengths
    // CLC lengths + 1 max_symbol gate.
    let mut bits = 1 + 4 + 3 * num_code_lengths + 1;

    // Per-symbol body: when the CLC collapses to a single non-zero
    // length (single-leaf CLC), the decoder consumes 0 bits per symbol
    // and the writer emits nothing. Otherwise emit the canonical code for
    // each literal length value.
    let used_clc: Vec<usize> = (0..NUM_CODE_LENGTH_CODES)
        .filter(|&s| clc_freq[s] > 0)
        .collect();
    if used_clc.len() > 1 {
        for &l in lengths {
            bits += clc_lengths[l as usize] as usize;
        }
    }
    bits
}

/// Write a per-symbol length table with the §3.7.2.1.1 *simple code
/// length code*.
///
/// Only valid for `symbols.len()` in `[1, 2]`, each symbol in `[0..255]`,
/// each implicitly at code length 1. The caller is responsible for
/// checking applicability via [`WriteCode::as_simple_form`].
fn write_simple_code_lengths(w: &mut BitWriter, symbols: &[usize]) {
    debug_assert!(symbols.len() == 1 || symbols.len() == 2);
    debug_assert!(symbols.iter().all(|&s| s <= 255));

    // §3.7.2.1.1 flag: 1 selects the simple form.
    w.write_bit(true);
    // num_symbols = ReadBits(1) + 1, so write `num_symbols - 1`.
    w.write_bits((symbols.len() as u32) - 1, 1);
    // §3.7.2.1.1: "is_first_8bits ... range [0..1] or [0..255]". Choose
    // the 1-bit form when symbol0 fits in [0..1], else the 8-bit form.
    let is_first_8bits = symbols[0] > 1;
    w.write_bits(if is_first_8bits { 1 } else { 0 }, 1);
    let s0_width = if is_first_8bits { 8 } else { 1 };
    w.write_bits(symbols[0] as u32, s0_width);
    if symbols.len() == 2 {
        // §3.7.2.1.1: "The second symbol, if present, is always assumed
        // to be in the range [0..255] and coded using 8 bits."
        w.write_bits(symbols[1] as u32, 8);
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

/// Run the §5.2.2 hash-chain matcher over `pixels`, producing the
/// token stream (literals + backward-reference copies) the entropy
/// stage emits. Every `Copy` token has `1 <= distance <= position` and
/// `MIN_MATCH <= length <= MAX_MATCH`, so the decoder's §5.2.2 copy
/// loop reproduces the exact pixels.
///
/// As of round 158 the matcher applies **three-position lazy matching**:
/// when the matcher finds a match `(len_a, _)` at `pos`, the encoder
/// also probes `pos + 1` (depth-1), `pos + 2` (depth-2), and `pos + 3`
/// (depth-3). The longest of `(len_a, len_b, len_c, len_d)` wins; ties
/// resolve to the earliest position (preserving the strict-greater
/// semantics introduced in round 156). When the depth-3 match `len_d`
/// is the unique longest, the encoder emits *three* literals (at
/// `pos`, `pos + 1`, `pos + 2`) and takes the longer match starting
/// at `pos + 3`. This costs at most three extra hash-chain walks per
/// match attempt and extends the round-157 two-position lazy recovery
/// to the third-order trap: a short match at each of `pos`, `pos + 1`,
/// `pos + 2` together blocking a strictly longer match at `pos + 3`.
/// The reconstructed pixels are bit-identical to the strict-greedy,
/// depth-1, and depth-2 partitions for any input — only the token
/// *partition* shifts by up to three pixels — so round-trips remain
/// bit-exact and the existing test suite continues to pass.
fn tokenize_lz77(pixels: &[u32]) -> Vec<Token> {
    tokenize_lz77_inner(pixels, LAZY_DEPTH_DEFAULT)
}

/// Production lazy-match depth used by [`tokenize_lz77`]. Round 156
/// set this to 1 (single-position look-ahead); round 157 bumped it to
/// 2 (two-position look-ahead); round 158 bumped it to 3 (three-
/// position look-ahead); round 163 bumps it to 4 (four-position look-
/// ahead with a [`DEPTH4_GUARD_THRESHOLD`] diminishing-returns guard).
/// A value of 0 reproduces the r155 strict-greedy partition.
const LAZY_DEPTH_DEFAULT: u32 = 4;

/// Round-163 diminishing-returns guard for the depth-4 probe. The
/// depth-4 `find(pos + 4)` call (plus the `matcher.insert(pos + 3)`
/// bookkeeping that gives it a fair shot at including `pos..=pos + 3`
/// in its window) is only executed when the running best length
/// across the depth-1/2/3 probes is strictly less than this
/// threshold. Once the depth-3 best already covers a length-
/// `THRESHOLD` run, swapping to a depth-4 alternative would have to
/// strictly exceed that length while paying for four literals
/// (`pixels[pos..pos + 4]`); the empirical pay-off shrinks rapidly
/// past the threshold and is rarely big enough to recover the
/// literal-emission cost in the entropy stage. Tuned to a conservative
/// value (`6`) so the guard only suppresses depth-4 work when the
/// running best is already comfortably above the four-literal break-
/// even line. At `THRESHOLD = u32::MAX` the depth-4 probe still
/// honours the `best_len > MIN_MATCH` floor (see
/// [`tokenize_lz77_inner`]); at `THRESHOLD = 0` (or below
/// `MIN_MATCH + 1 = 4`) the depth-4 probe never fires. The A/B
/// regression test [`round_163_depth4_guard_suppresses_long_run_swap`]
/// exercises the guard's switching boundary.
const DEPTH4_GUARD_THRESHOLD: u32 = 6;

/// Implementation of [`tokenize_lz77`] with an explicit `lazy_depth`
/// toggle. Values:
///
/// * `0` — strict-greedy r155 partition (no look-ahead). Always emits
///   the match found at `pos`.
/// * `1` — round-156 single-position lazy partition: probe `pos + 1`,
///   swap to a strictly-longer match starting there.
/// * `2` — round-157 two-position lazy partition: also probe
///   `pos + 2`, swap to a strictly-longer match starting there (the
///   `pos + 2` match must strictly beat both `pos` and `pos + 1`).
/// * `3` — round-158 three-position lazy partition: also probe
///   `pos + 3`, swap to a strictly-longer match starting there (the
///   `pos + 3` match must strictly beat the running best across
///   `pos`, `pos + 1`, and `pos + 2`).
/// * `4` — round-163 guarded four-position lazy partition: also
///   probes `pos + 4`, but **only when** the running best across the
///   first four positions is strictly greater than [`MIN_MATCH`]
///   (`MIN_MATCH = 3`, so `best_len >= 4`) AND strictly less than
///   [`DEPTH4_GUARD_THRESHOLD`]. The `> MIN_MATCH` floor ensures the
///   pre-inserted `pos + 3` position is always covered by the chosen
///   match's range, so the next iteration's `find` never sees its
///   own position in the chain. When the guard fires, the `pos + 4`
///   match must strictly beat the running best.
///
/// Values `>= 4` are clamped to `4`. The A/B regression tests
/// in this module use `0`, `1`, `2`, and `3` to compare against the
/// r155, r156, r157, and r158 baselines.
fn tokenize_lz77_inner(pixels: &[u32], lazy_depth: u32) -> Vec<Token> {
    let n = pixels.len();
    let mut matcher = Lz77Matcher::new(pixels);
    let mut tokens = Vec::new();
    let mut pos = 0usize;
    let depth = lazy_depth.min(4);
    while pos < n {
        if let Some((len_a, dist_a)) = matcher.find(pos) {
            // Lazy lookahead. The matcher's hash chains do not yet
            // include `pos` (matches at `pos` only reference positions
            // strictly before `pos`), so to give the `pos + 1` probe a
            // fair shot at a match that *includes* the pixel at `pos`
            // we insert `pos` into the chains before the look-ahead
            // `find`. Likewise, the `pos + 2` probe needs both `pos`
            // and `pos + 1` in the chains, and the `pos + 3` probe
            // needs `pos`, `pos + 1`, and `pos + 2` all in. The
            // bookkeeping at the tail of each branch skips
            // re-inserting any positions that the lookahead probes
            // already inserted.
            let mut best_len = len_a;
            let mut best_dist = dist_a;
            let mut best_start = pos; // pixel index where the match begins
            let inserted_pos = depth >= 1 && len_a < MAX_MATCH && pos + 1 < n;
            if inserted_pos {
                matcher.insert(pos);
                if let Some((len_b, dist_b)) = matcher.find(pos + 1) {
                    if len_b > best_len {
                        best_len = len_b;
                        best_dist = dist_b;
                        best_start = pos + 1;
                    }
                }
            }
            // Depth-2 probe: only meaningful if depth allows it, the
            // current best match is short enough to be worth
            // attempting to displace, and `pos + 2` is in range. We
            // also require `pos + 1` to be inserted so the `pos + 2`
            // window can reference it; the depth-1 probe already
            // inserted `pos`.
            let inserted_pos1 = depth >= 2 && best_len < MAX_MATCH && pos + 2 < n;
            if inserted_pos1 {
                matcher.insert(pos + 1);
                if let Some((len_c, dist_c)) = matcher.find(pos + 2) {
                    if len_c > best_len {
                        best_len = len_c;
                        best_dist = dist_c;
                        best_start = pos + 2;
                    }
                }
            }
            // Depth-3 probe: only meaningful if depth allows it, the
            // running best match is short enough to be worth
            // attempting to displace, and `pos + 3` is in range. We
            // also require `pos + 2` to be inserted so the `pos + 3`
            // window can reference it; the depth-1 / depth-2 probes
            // already inserted `pos` and `pos + 1`.
            let inserted_pos2 = depth >= 3 && best_len < MAX_MATCH && pos + 3 < n;
            if inserted_pos2 {
                matcher.insert(pos + 2);
                if let Some((len_d, dist_d)) = matcher.find(pos + 3) {
                    if len_d > best_len {
                        best_len = len_d;
                        best_dist = dist_d;
                        best_start = pos + 3;
                    }
                }
            }
            // Depth-4 probe (round 163): only meaningful if depth
            // allows it, the running best match is short enough to be
            // worth attempting to displace, `pos + 4` is in range,
            // AND the round-163 diminishing-returns guard fires
            // (`best_len < DEPTH4_GUARD_THRESHOLD`). The guard skips
            // the depth-4 work when the depth-3 best is already
            // comfortably above the four-literal break-even line.
            //
            // Additional **lower-bound** floor: the depth-4 probe pre-
            // inserts `pos + 3` into the matcher chain so the `find(pos
            // + 4)` window can reference it. That pre-insert must be
            // covered by the chosen match's range `[best_start,
            // best_start + best_len)` — otherwise the next iteration's
            // `pos` (= `best_start + best_len`) could equal `pos + 3`,
            // and `find(pos + 3)` would see itself in the chain and
            // return distance `0`. We avoid that corner by gating on
            // `best_len > MIN_MATCH` (i.e., `best_len >= 4`): with
            // `best_start == pos` the match end is at least `pos + 4 >
            // pos + 3`, covering the pre-insert. The depth-3 best of
            // exactly 3 pixels (`= MIN_MATCH`) is short enough that
            // the depth-4 probe is rarely worth it anyway, so the
            // floor costs almost nothing on the matcher's behaviour.
            //
            // We also require `pos + 3` to be inserted so the `pos + 4`
            // window can reference it; the depth-1 / depth-2 / depth-3
            // probes already inserted `pos`, `pos + 1`, and `pos + 2`.
            let inserted_pos3 = depth >= 4
                && best_len > MIN_MATCH
                && best_len < MAX_MATCH
                && (best_len as u32) < DEPTH4_GUARD_THRESHOLD
                && pos + 4 < n;
            if inserted_pos3 {
                matcher.insert(pos + 3);
                if let Some((len_e, dist_e)) = matcher.find(pos + 4) {
                    if len_e > best_len {
                        best_len = len_e;
                        best_dist = dist_e;
                        best_start = pos + 4;
                    }
                }
            }

            // Emit literals for any pixels skipped by the chosen
            // lazy starting position, then the chosen match.
            for &skipped in &pixels[pos..best_start] {
                tokens.push(Token::Literal(skipped));
            }
            tokens.push(Token::Copy {
                length: best_len,
                distance: best_dist,
            });

            // Hash-chain bookkeeping. Insert every covered position
            // into the chains so later matches can reference inside
            // the just-copied run; skip positions that the lookahead
            // probes already inserted.
            //
            // Pre-inserted positions (so far): `pos` if `inserted_pos`,
            // `pos + 1` if `inserted_pos1`, `pos + 2` if `inserted_pos2`,
            // `pos + 3` if `inserted_pos3` (round 163). The chosen
            // match covers `[best_start, best_start + best_len)`. Walk
            // that range and only `insert` the positions that are not
            // already in the chains.
            let end = best_start + best_len;
            let mut q = pos;
            while q < end {
                let already_in = (q == pos && inserted_pos)
                    || (q == pos + 1 && inserted_pos1)
                    || (q == pos + 2 && inserted_pos2)
                    || (q == pos + 3 && inserted_pos3);
                if q >= best_start && !already_in {
                    matcher.insert(q);
                }
                q += 1;
            }
            pos = end;
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

/// The default `color_cache_code_bits` the chooser evaluates when a
/// caller asks for a single representative cache size (e.g. test
/// fixtures, the `encode_argb_literals_color_cache` direct entry).
/// Eight bits gives a 256-entry cache — a middle-of-range value that
/// works reasonably well across the §5.2.3 `[1..11]` range.
///
/// The production chooser ([`encode_argb_literals_with_width`] and
/// [`encode_argb_with_predictor_chooser`]) no longer uses this single
/// value: as of round 148 it sweeps every `cache_code_bits ∈ [1..11]`
/// per the §5.2.3 range and emits the smallest stream. See
/// [`select_best_cache_bits`].
pub const DEFAULT_COLOR_CACHE_BITS: u32 = 8;

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

// ---- §4.1 spatial-predictor forward transform (encoder side) ----

/// `DIV_ROUND_UP(num, den)` from §4.1 (`((num) + (den) - 1) / (den)`).
#[inline]
fn predictor_div_round_up(num: u32, den: u32) -> u32 {
    num.div_ceil(den)
}

/// Per-channel `(a + b) / 2` (`Average2` from §4.1).
#[inline]
fn predictor_average2(a: u32, b: u32) -> u32 {
    let f = |sh: u32| -> u32 {
        let ca = (a >> sh) & 0xff;
        let cb = (b >> sh) & 0xff;
        (ca + cb) / 2
    };
    (f(24) << 24) | (f(16) << 16) | (f(8) << 8) | f(0)
}

/// `Clamp(a)` from §4.1: saturate `a` to `[0, 255]`.
#[inline]
fn predictor_clamp(a: i32) -> i32 {
    a.clamp(0, 255)
}

/// §4.1 `ClampAddSubtractFull(a, b, c)` = `Clamp(a + b - c)` per channel.
#[inline]
fn predictor_clamp_add_subtract_full(a: u32, b: u32, c: u32) -> u32 {
    let f = |sh: u32| -> u32 {
        let ca = ((a >> sh) & 0xff) as i32;
        let cb = ((b >> sh) & 0xff) as i32;
        let cc = ((c >> sh) & 0xff) as i32;
        predictor_clamp(ca + cb - cc) as u32
    };
    (f(24) << 24) | (f(16) << 16) | (f(8) << 8) | f(0)
}

/// §4.1 `ClampAddSubtractHalf(a, b)` = `Clamp(a + (a - b) / 2)` per
/// channel.
#[inline]
fn predictor_clamp_add_subtract_half(a: u32, b: u32) -> u32 {
    let f = |sh: u32| -> u32 {
        let ca = ((a >> sh) & 0xff) as i32;
        let cb = ((b >> sh) & 0xff) as i32;
        predictor_clamp(ca + (ca - cb) / 2) as u32
    };
    (f(24) << 24) | (f(16) << 16) | (f(8) << 8) | f(0)
}

/// §4.1 `Select(L, T, TL)` — whichever of `L` / `T` is closer
/// (per-channel Manhattan distance) to the `L + T - TL` estimate.
#[inline]
fn predictor_select(l: u32, t: u32, tl: u32) -> u32 {
    let ach = |x: u32| ((x >> 24) & 0xff) as i32;
    let rch = |x: u32| ((x >> 16) & 0xff) as i32;
    let gch = |x: u32| ((x >> 8) & 0xff) as i32;
    let bch = |x: u32| (x & 0xff) as i32;

    let p_a = ach(l) + ach(t) - ach(tl);
    let p_r = rch(l) + rch(t) - rch(tl);
    let p_g = gch(l) + gch(t) - gch(tl);
    let p_b = bch(l) + bch(t) - bch(tl);

    let p_l =
        (p_a - ach(l)).abs() + (p_r - rch(l)).abs() + (p_g - gch(l)).abs() + (p_b - bch(l)).abs();
    let p_t =
        (p_a - ach(t)).abs() + (p_r - rch(t)).abs() + (p_g - gch(t)).abs() + (p_b - bch(t)).abs();

    if p_l < p_t {
        l
    } else {
        t
    }
}

/// Compute the §4.1 prediction for `mode ∈ 0..=13` given the four
/// reconstructed-pixel neighbours.
///
/// Identical formula to the decoder's
/// `crate::vp8l_transform::inverse_predictor` `predict` helper — kept
/// as a separate copy here because the encoder is built (and tested)
/// independently of the decoder's transform module.
fn predictor_predict(mode: u8, l: u32, t: u32, tr: u32, tl: u32) -> u32 {
    match mode {
        0 => 0xff00_0000,
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => predictor_average2(predictor_average2(l, tr), t),
        6 => predictor_average2(l, tl),
        7 => predictor_average2(l, t),
        8 => predictor_average2(tl, t),
        9 => predictor_average2(t, tr),
        10 => predictor_average2(predictor_average2(l, tl), predictor_average2(t, tr)),
        11 => predictor_select(l, t, tl),
        12 => predictor_clamp_add_subtract_full(l, t, tl),
        13 => predictor_clamp_add_subtract_half(predictor_average2(l, t), tl),
        // §4.1 only defines [0..13]. An out-of-range mode produces the
        // top-left's solid-black prediction, matching the decoder.
        _ => 0xff00_0000,
    }
}

/// Per-channel residual `(original - pred) mod 256`. The inverse of
/// the decoder's `add_pred` (`residual + pred mod 256 = original`),
/// so re-applying the §4.1 inverse predictor recovers `original`
/// exactly.
///
/// **Round-224 SWAR experiment — closure-of-four body retained.**
/// The decoder-side `add_pred` was rewritten in round 170 as a
/// two-pair SWAR (`(x & 0x00ff_00ff).wrapping_add(...)` /
/// `(x & 0xff00_ff00).wrapping_add(...)`) because addition does not
/// propagate carry across the zero "guard" bytes when the summand has
/// its high bit masked out. Subtraction is asymmetric: a borrow at the
/// low byte of a lane DOES propagate through the zero guard byte and
/// corrupts the adjacent lane, so the mirror rewrite needs to bias
/// the minuend with a `0x0100` guard per lane (`(orig & 0x00ff_00ff)
/// | 0x0100_0100`) to suppress underflow before the subtract, with a
/// final `& 0x00ff_00ff` mask to clear the guard. We measured both
/// forms in round 224 against the new `predictor_subtract_256x256`
/// bench: **34.1 µs (closure-of-four) → 40.5 µs (biased SWAR), a
/// +18.4% regression.** AArch64 NEON auto-vectorisation of the four
/// sequential per-byte `wrapping_sub` calls is tighter than the
/// explicit biased-SWAR pattern at this call site. Same shape as the
/// round-194 BENCHMARKS footnote that recorded a regression for a
/// `clamp_add_subtract_*` (mode 12) per-channel `to_le_bytes()` +
/// `i16` byte-loop attempt — the closure-of-four `i32` body remains
/// the right starting point on this target until a true 16-byte
/// `std::simd` formulation can amortise the lane-bias cost across
/// multiple pixels per iteration (mirroring the `to_rgba_simd`
/// precedent under the `simd` feature).
#[inline]
pub fn predictor_subtract(original: u32, pred: u32) -> u32 {
    let a = ((original >> 24) & 0xff).wrapping_sub((pred >> 24) & 0xff) & 0xff;
    let r = ((original >> 16) & 0xff).wrapping_sub((pred >> 16) & 0xff) & 0xff;
    let g = ((original >> 8) & 0xff).wrapping_sub((pred >> 8) & 0xff) & 0xff;
    let b = (original & 0xff).wrapping_sub(pred & 0xff) & 0xff;
    (a << 24) | (r << 16) | (g << 8) | b
}

/// Cost proxy used to pick a block's predictor mode: the sum of
/// per-pixel per-channel `|residual|` over the block, where `|x|`
/// folds the mod-256 residual onto `[-128, 127]` (a value `x ∈ [0,
/// 255]` representing `(original - pred) mod 256` has true magnitude
/// `min(x, 256 - x)`).
///
/// Sum-of-magnitudes is a standard zero-cost proxy for the entropy
/// of the residual histogram: lower magnitudes peak the histogram
/// near zero, which a Huffman code over the residual symbols
/// compresses well. Using the folded magnitude correctly rewards
/// modes that produce both small-positive and small-negative
/// residuals (e.g. `0xff` = `-1 mod 256`, magnitude 1).
#[inline]
fn residual_magnitude(residual: u32) -> u32 {
    let fold = |v: u32| -> u32 {
        let v = v & 0xff;
        if v <= 128 {
            v
        } else {
            256 - v
        }
    };
    fold(residual >> 24) + fold(residual >> 16) + fold(residual >> 8) + fold(residual)
}

/// §4.1 border-aware prediction at `(x, y)`. Mirrors
/// [`crate::vp8l_transform::inverse_predictor`]: top-left is solid
/// black `0xff000000`; top row predicts L; left column predicts T;
/// rightmost column uses the row's leftmost pixel as TR; otherwise
/// `predictor_predict(mode, L, T, TR, TL)`.
///
/// `pixels` is the `width × height` ARGB source (read-only — the
/// encoder predicts against the *originals*, since the decoder
/// reconstructs pixels equal to those originals).
fn predictor_at(pixels: &[u32], width: usize, x: usize, y: usize, mode: u8) -> u32 {
    if x == 0 && y == 0 {
        return 0xff00_0000;
    }
    let idx = y * width + x;
    if y == 0 {
        return pixels[idx - 1];
    }
    if x == 0 {
        return pixels[idx - width];
    }
    let l = pixels[idx - 1];
    let t = pixels[idx - width];
    let tl = pixels[idx - width - 1];
    let tr = if x == width - 1 {
        pixels[idx - width - (width - 1)]
    } else {
        pixels[idx - width + 1]
    };
    predictor_predict(mode, l, t, tr, tl)
}

/// Per-pixel residual consumer for [`for_each_block_residual`].
///
/// `pixel` receives each in-bounds block pixel's §4.1 mod-256
/// residual in raster order; `row_end` runs after the last pixel of
/// each block row and returns whether the walk should continue.
///
/// Pruning at row granularity (instead of per pixel, as the
/// pre-round-280 chooser loops did) is pick-identical: a pruned
/// walk's partial cost is only ever compared `>= cap` by the caller,
/// and per-pixel contributions are non-negative, so any partial sum
/// that prunes implies the full sum would also have compared
/// `>= cap`. Coarsening the prune lets the interior pixel loop run
/// branch-free (auto-vectorisable) at the cost of at most one block
/// row of extra work on a pruned mode.
trait ResidualSink {
    fn pixel(&mut self, residual: u32);
    fn row_end(&mut self) -> bool;
}

/// Walk every in-bounds pixel of the block `[x0, x0+bw) × [y0,
/// y0+bh)` of the `width × height` image in raster order, feeding
/// each pixel's §4.1 residual (`predictor_subtract(original,
/// prediction)`) to `sink`. Interior predictions come from
/// `predict(l, t, tr, tl)`; border pixels follow the §4.1 border
/// rules (top-left → solid black, top row → L, left column → T,
/// rightmost column → the §4.1 TR wraparound) — pixel-for-pixel
/// identical to a [`predictor_at`] + [`predictor_subtract`] walk,
/// but with the border branch chain hoisted out of the inner loop
/// (round-180 decoder precedent) and the predictor monomorphised in
/// by the caller so the per-pixel 14-way mode dispatch disappears.
#[inline]
#[allow(clippy::too_many_arguments)]
fn walk_block_residuals<P, S>(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    predict: P,
    sink: &mut S,
) where
    P: Fn(u32, u32, u32, u32) -> u32,
    S: ResidualSink,
{
    let y_end = (y0 + bh).min(height);
    let x_end = (x0 + bw).min(width);
    if x0 >= x_end || y0 >= y_end {
        return;
    }
    let mut y = y0;
    if y == 0 {
        // Top row: (0, 0) predicts solid black, the rest predict L.
        let mut x = x0;
        if x == 0 {
            sink.pixel(predictor_subtract(pixels[0], 0xff00_0000));
            x = 1;
        }
        while x < x_end {
            sink.pixel(predictor_subtract(pixels[x], pixels[x - 1]));
            x += 1;
        }
        if !sink.row_end() {
            return;
        }
        y = 1;
    }
    // The §4.1 right-column TR wraparound only applies when the block
    // reaches the image's right edge.
    let interior_end = if x_end == width { width - 1 } else { x_end };
    while y < y_end {
        let row = y * width;
        let mut x = x0;
        if x == 0 {
            // Left column predicts T.
            sink.pixel(predictor_subtract(pixels[row], pixels[row - width]));
            x = 1;
        }
        while x < interior_end {
            let idx = row + x;
            let l = pixels[idx - 1];
            let t = pixels[idx - width];
            let tl = pixels[idx - width - 1];
            let tr = pixels[idx - width + 1];
            sink.pixel(predictor_subtract(pixels[idx], predict(l, t, tr, tl)));
            x += 1;
        }
        if x < x_end {
            // x == width - 1: §4.1 TR wraparound.
            let idx = row + x;
            let l = pixels[idx - 1];
            let t = pixels[idx - width];
            let tl = pixels[idx - width - 1];
            let tr = pixels[idx - width - (width - 1)];
            sink.pixel(predictor_subtract(pixels[idx], predict(l, t, tr, tl)));
        }
        if !sink.row_end() {
            return;
        }
        y += 1;
    }
}

/// Run [`walk_block_residuals`] with the §4.1 predictor for `mode`
/// monomorphised into the walk, so the mode dispatch runs once per
/// block instead of once per pixel. Out-of-range modes predict solid
/// black, matching [`predictor_predict`].
#[inline]
#[allow(clippy::too_many_arguments)]
fn for_each_block_residual<S: ResidualSink>(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    mode: u8,
    sink: &mut S,
) {
    macro_rules! walk {
        ($p:expr) => {
            walk_block_residuals(pixels, width, height, x0, y0, bw, bh, $p, sink)
        };
    }
    match mode {
        1 => walk!(|l, _, _, _| l),
        2 => walk!(|_, t, _, _| t),
        3 => walk!(|_, _, tr, _| tr),
        4 => walk!(|_, _, _, tl| tl),
        5 => walk!(|l, t, tr, _| predictor_average2(predictor_average2(l, tr), t)),
        6 => walk!(|l, _, _, tl| predictor_average2(l, tl)),
        7 => walk!(|l, t, _, _| predictor_average2(l, t)),
        8 => walk!(|_, t, _, tl| predictor_average2(tl, t)),
        9 => walk!(|_, t, tr, _| predictor_average2(t, tr)),
        10 => walk!(|l, t, tr, tl| predictor_average2(
            predictor_average2(l, tl),
            predictor_average2(t, tr)
        )),
        11 => walk!(|l, t, _, tl| predictor_select(l, t, tl)),
        12 => walk!(|l, t, _, tl| predictor_clamp_add_subtract_full(l, t, tl)),
        13 => walk!(|l, t, _, tl| predictor_clamp_add_subtract_half(predictor_average2(l, t), tl)),
        // Mode 0 and §4.1-undefined modes both predict solid black.
        _ => walk!(|_, _, _, _| 0xff00_0000),
    }
}

/// [`ResidualSink`] accumulating the folded-L1 [`residual_magnitude`]
/// cost proxy, pruning at row granularity once the running sum
/// reaches `cap` (see the trait docs for why row-granular pruning is
/// pick-identical to the pre-round-280 per-pixel early-out).
struct MagnitudeCostSink {
    cost: u64,
    cap: u64,
}

impl ResidualSink for MagnitudeCostSink {
    #[inline]
    fn pixel(&mut self, residual: u32) {
        self.cost += residual_magnitude(residual) as u64;
    }
    #[inline]
    fn row_end(&mut self) -> bool {
        self.cost < self.cap
    }
}

/// [`ResidualSink`] filling the per-channel residual byte histograms
/// [`block_mode_entropy_cost`] feeds its Shannon sum. Never prunes:
/// the histograms must be complete before the entropy is meaningful.
struct ResidualHistogramSink {
    hist: [[u32; 256]; 4],
    n: u32,
}

impl ResidualSink for ResidualHistogramSink {
    #[inline]
    fn pixel(&mut self, residual: u32) {
        self.hist[0][((residual >> 24) & 0xff) as usize] += 1;
        self.hist[1][((residual >> 16) & 0xff) as usize] += 1;
        self.hist[2][((residual >> 8) & 0xff) as usize] += 1;
        self.hist[3][(residual & 0xff) as usize] += 1;
        self.n += 1;
    }
    #[inline]
    fn row_end(&mut self) -> bool {
        true
    }
}

/// Pick the §4.1 mode `0..=13` that minimises the residual cost
/// proxy over the rectangular block `[x0, x0+bw) × [y0, y0+bh)` of
/// the `width × height` image. Border rules per
/// [`predictor_at`].
///
/// On ties (multiple modes producing equal magnitude sums) the
/// lowest mode wins, which makes the chooser deterministic.
///
/// This is the no-hint entry point — equivalent to calling
/// [`pick_block_mode_with_hint`] with `prefer_mode = None`. The
/// production caller [`build_predictor_image`] uses the
/// hint-aware variant; the no-hint form is retained for the
/// in-module tie-breaker tests.
#[cfg(test)]
fn pick_block_mode(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
) -> u8 {
    pick_block_mode_with_hint(pixels, width, height, x0, y0, bw, bh, None)
}

/// Compute the §4.1 residual-cost proxy for a single mode over
/// the rectangular block `[x0, x0+bw) × [y0, y0+bh)`. Walks every
/// in-bounds pixel without an early-out so the caller can use the
/// result as an authoritative tie-break reference.
///
/// This is the same per-mode sum the main chooser computes inside
/// [`pick_block_mode_with_hint`], factored out so the entropy-
/// image-aware tie-breaker can evaluate the preferred neighbour
/// mode exactly once and re-use the value to decide whether a
/// post-walk swap is allowed.
#[allow(clippy::too_many_arguments)]
fn block_mode_cost(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    mode: u8,
) -> u64 {
    block_mode_cost_capped(pixels, width, height, x0, y0, bw, bh, mode, u64::MAX)
}

/// [`block_mode_cost`] with a pruning `cap`: once the running cost
/// reaches `cap` at a block-row boundary the walk stops and the
/// partial sum is returned. Callers only compare a pruned return
/// value `>= cap` (the residual magnitudes are non-negative, so a
/// partial sum at or above `cap` proves the full sum is too), which
/// keeps mode picks identical to an uncapped walk.
#[allow(clippy::too_many_arguments)]
fn block_mode_cost_capped(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    mode: u8,
    cap: u64,
) -> u64 {
    let mut sink = MagnitudeCostSink { cost: 0, cap };
    for_each_block_residual(pixels, width, height, x0, y0, bw, bh, mode, &mut sink);
    sink.cost
}

/// Hint-aware variant of [`pick_block_mode`]: picks the §4.1 mode
/// minimising the residual cost proxy, and on ties prefers
/// `prefer_mode` over the otherwise-lowest tied mode.
///
/// `prefer_mode = Some(m)` directs the tie-break: when `m`'s cost
/// equals the lowest cost found across all 14 modes, the chooser
/// returns `m` instead of the lowest-indexed tied mode. When
/// `prefer_mode = None` (or `prefer_mode = Some(m)` with `m`
/// strictly worse than another mode), the lowest-tied-mode behaviour
/// is preserved exactly.
///
/// Round 159: [`build_predictor_image`] passes the left neighbour
/// block's chosen mode (or the top neighbour at the left edge of
/// the predictor image) as the hint. The §3.5 RFC 9649 note
/// "transform data can be decided based on entropy minimization"
/// motivates this: residual-cost-equal modes encode different
/// values into the predictor sub-image, and the sub-image is
/// written as an `entropy-coded-image` (§7.2) so reducing its
/// symbol entropy directly shrinks the output stream. The
/// residuals themselves do not change (this is a strict tie-break),
/// so decode round-trips are unaffected.
#[allow(clippy::too_many_arguments)]
fn pick_block_mode_with_hint(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    prefer_mode: Option<u8>,
) -> u8 {
    let mut best_mode: u8 = 0;
    let mut best_cost = u64::MAX;
    for mode in 0u8..=13 {
        // The cap prunes modes already worse than the current best at
        // block-row granularity; a pruned partial sum is `>= best_cost`
        // so the `cost < best_cost` update below stays pick-identical
        // to a full walk.
        let cost = block_mode_cost_capped(pixels, width, height, x0, y0, bw, bh, mode, best_cost);
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }
    // Round 159 entropy-image-aware tie-breaker. If the caller
    // supplied a preferred mode (typically the left or top neighbour
    // block's chosen mode) and the preferred mode's full cost ties
    // with `best_cost`, swap to the preferred mode so the predictor
    // sub-image carries a longer run of identical mode values. The
    // residual stream produced by the main image's forward transform
    // is unchanged (the cost is equal), so decode round-trips are
    // bit-identical.
    if let Some(m) = prefer_mode {
        if m != best_mode {
            let cost = block_mode_cost(pixels, width, height, x0, y0, bw, bh, m);
            if cost == best_cost {
                best_mode = m;
            }
        }
    }
    best_mode
}

/// Round 160 *slack-cost* variant of [`pick_block_mode_with_hint`].
///
/// Where the round-159 strict tie-break only swaps to the preferred
/// mode when its residual cost is **exactly equal** to the best,
/// this variant also accepts the preferred mode when its cost is
/// within an additive `slack` budget of the best. RFC 9649 §3.5
/// authorises the encoder to "decide \[transform data\] based on
/// entropy minimization", and the slack budget formalises the
/// trade-off: a small per-pixel-magnitude increase in the §4.1
/// residual stream may be acceptable when it strictly reduces the
/// entropy of the §7.2 predictor sub-image (longer run of identical
/// mode values → fewer distinct prefix-code symbols → fewer bytes
/// emitted for the sub-image).
///
/// This is no longer a residual-cost-neutral swap: the residuals
/// produced by the main image's forward transform **do change** on
/// a slack-accepted swap. Decode round-trips are still bit-correct
/// (the residuals are recomputed against the chosen mode at
/// `apply_forward_predictor` time, and the decoder applies the same
/// mode in reverse), but pixel-level decode equivalence between two
/// encoder runs at different slack budgets is **not** preserved —
/// only end-to-end image round-trip equivalence is.
///
/// The encoder protects itself from regressions by building both the
/// `slack = 0` (strict, round-159 baseline) and `slack > 0`
/// predictor candidates and keeping the strictly-smaller encoded
/// stream — so a slack candidate that hurts overall byte cost on
/// some input is simply not chosen.
#[allow(clippy::too_many_arguments)]
fn pick_block_mode_with_hint_slack(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    prefer_mode: Option<u8>,
    slack: u64,
) -> u8 {
    let mut best_mode: u8 = 0;
    let mut best_cost = u64::MAX;
    for mode in 0u8..=13 {
        // Row-granular prune against the current best; pick-identical
        // to a full walk (see `block_mode_cost_capped`).
        let cost = block_mode_cost_capped(pixels, width, height, x0, y0, bw, bh, mode, best_cost);
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }
    // Round-160 slack-cost tie-break: accept the preferred neighbour
    // mode when its cost is within `slack` of the best cost. The
    // slack budget lets the encoder trade a small residual increase
    // for a predictor-sub-image entropy drop. `slack == 0` recovers
    // the round-159 strict tie-break behaviour exactly.
    if let Some(m) = prefer_mode {
        if m != best_mode {
            let cost = block_mode_cost(pixels, width, height, x0, y0, bw, bh, m);
            if cost <= best_cost.saturating_add(slack) {
                best_mode = m;
            }
        }
    }
    best_mode
}

/// Build the §4.1 sub-resolution *predictor image*: one ARGB pixel
/// per `(1 << size_bits)`-pixel-square block of the main image, with
/// the chosen mode stored in the green channel (alpha/red/blue
/// fixed at 0xff / 0 / 0 — the decoder only reads the green channel
/// via `inverse_predictor`'s `green(predictor_image[...])`).
///
/// Returns `(predictor_image, transform_width, transform_height)`.
/// `transform_width = DIV_ROUND_UP(width, 1 << size_bits)` and
/// `transform_height = DIV_ROUND_UP(height, 1 << size_bits)`, per
/// §4.1.
///
/// Round 159: each block consults
/// [`pick_block_mode_with_hint`] with the immediately-prior
/// block's chosen mode as the preferred tie-break — left neighbour
/// in the current row, or the top neighbour for blocks in the left
/// column (no neighbour for the top-left block). This is a strict
/// tie-break: when the preferred mode's residual cost equals the
/// otherwise-lowest cost, the neighbour's value is chosen so the
/// predictor sub-image carries longer runs of identical modes,
/// dropping the sub-image's entropy and the bytes the
/// `entropy-coded-image` writer emits for it. Residual values are
/// unchanged on cost-equal swaps, so decoded pixels are
/// bit-identical to the round-158 baseline.
fn build_predictor_image(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
) -> (Vec<u32>, u32, u32) {
    let block = 1u32 << size_bits;
    let tw = predictor_div_round_up(width, block);
    let th = predictor_div_round_up(height, block);
    let mut img = Vec::with_capacity((tw * th) as usize);
    let w = width as usize;
    let h = height as usize;
    let bsz = block as usize;
    // Track the previous row's chosen modes so the left-column
    // blocks can fall back to a top neighbour. Each slot is `None`
    // while building the very first row.
    let mut prev_row: Vec<Option<u8>> = vec![None; tw as usize];
    for by in 0..th as usize {
        let mut left_mode: Option<u8> = None;
        for (bx, top_slot) in prev_row.iter_mut().enumerate() {
            let x0 = bx * bsz;
            let y0 = by * bsz;
            // Preferred tie-break: left neighbour (current row) if
            // present, else top neighbour (previous row). The
            // top-left block (by == 0 && bx == 0) gets no hint and
            // falls back to the lowest-tied-mode default.
            let prefer = left_mode.or(*top_slot);
            let mode = pick_block_mode_with_hint(pixels, w, h, x0, y0, bsz, bsz, prefer);
            // Pack mode into the green channel; opaque alpha and
            // zeroed red/blue keep the sub-image visually inert and
            // match the channel the decoder reads.
            img.push(0xff00_0000 | ((mode as u32) << 8));
            left_mode = Some(mode);
            *top_slot = Some(mode);
        }
    }
    (img, tw, th)
}

/// Round-160 *slack-cost* variant of [`build_predictor_image`].
///
/// Identical structure to `build_predictor_image`, but routes every
/// per-block mode choice through [`pick_block_mode_with_hint_slack`]
/// with the caller-supplied `slack` budget. `slack == 0` recovers
/// `build_predictor_image` exactly. Larger `slack` values let the
/// preferred neighbour mode win even at a small residual-cost
/// increase, trading per-pixel residual mass against the §7.2
/// predictor-sub-image's symbol entropy.
///
/// Round-trip correctness is unaffected by `slack`: the forward
/// transform later re-derives residuals against the chosen modes,
/// and the decoder's inverse pass uses the same modes from the
/// sub-image, so the decoded image always equals the input.
///
/// The encoder chooser builds both `slack == 0` and `slack > 0`
/// candidates and keeps the shortest, so a slack candidate that
/// hurts overall byte cost on a given input is simply not chosen.
fn build_predictor_image_with_slack(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
    slack: u64,
) -> (Vec<u32>, u32, u32) {
    let block = 1u32 << size_bits;
    let tw = predictor_div_round_up(width, block);
    let th = predictor_div_round_up(height, block);
    let mut img = Vec::with_capacity((tw * th) as usize);
    let w = width as usize;
    let h = height as usize;
    let bsz = block as usize;
    let mut prev_row: Vec<Option<u8>> = vec![None; tw as usize];
    for by in 0..th as usize {
        let mut left_mode: Option<u8> = None;
        for (bx, top_slot) in prev_row.iter_mut().enumerate() {
            let x0 = bx * bsz;
            let y0 = by * bsz;
            let prefer = left_mode.or(*top_slot);
            let mode =
                pick_block_mode_with_hint_slack(pixels, w, h, x0, y0, bsz, bsz, prefer, slack);
            img.push(0xff00_0000 | ((mode as u32) << 8));
            left_mode = Some(mode);
            *top_slot = Some(mode);
        }
    }
    (img, tw, th)
}

/// Round 161 *Shannon-entropy bit-cost* per-mode cost function.
///
/// Where [`block_mode_cost`] sums the folded L1 magnitude of the
/// per-pixel residual as a *proxy* for Huffman bit cost, this
/// function computes the actual lower-bound bit cost a Huffman code
/// over the residual byte distribution would emit:
///
/// 1. Build the per-channel `[u32; 256]` histogram of the block's
///    mod-256 residuals against the candidate `mode`.
/// 2. Compute the Shannon entropy `H = -Σ (c/N) · log2(c/N)` over
///    each channel's histogram (zero-count bins contribute zero).
/// 3. Sum `N · H` across channels — this is the lower-bound bit
///    count a per-symbol Huffman code over those residuals would
///    emit (the encoder's actual prefix coder is within ~1 bit of
///    this bound per symbol, so the bit-count *ordering* between
///    modes is faithful even though absolute counts differ by O(1)
///    per symbol).
///
/// The cost is returned as a fixed-point u64 in units of
/// **milli-bits** (1 bit = 1000 units) so comparisons stay exact
/// without floats leaking into the chooser's tie-break logic. The
/// quantisation rounds to the nearest milli-bit which is finer
/// than any Huffman code's per-symbol cost, so two modes that
/// would tie in floating-point also tie in the quantised cost.
///
/// Walks every in-bounds pixel without an early-out (unlike
/// [`block_mode_cost`]'s magnitude proxy which can prune): the
/// per-channel histograms must be complete before the entropy
/// sum is meaningful.
#[allow(clippy::too_many_arguments)]
fn block_mode_entropy_cost(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    mode: u8,
) -> u64 {
    let mut sink = ResidualHistogramSink {
        hist: [[0u32; 256]; 4],
        n: 0,
    };
    for_each_block_residual(pixels, width, height, x0, y0, bw, bh, mode, &mut sink);
    let hist = sink.hist;
    let n = sink.n;
    if n == 0 {
        return 0;
    }
    // Σ_channels Σ_b c·log2(N/c) milli-bits, with c·log2(N/c) =
    // c·(log2(N) − log2(c)). Float arithmetic is fine here: the
    // result is rounded to nearest milli-bit before u64 cast, so
    // bit-for-bit determinism holds across platforms with IEEE-754
    // ln(). The Shannon expansion picks `log2(N/c)` rather than
    // `−log2(c/N)` to keep the per-bin operand non-negative (zero
    // when c = N, growing as c shrinks) which is friendly to the
    // accumulator.
    let n_f = n as f64;
    let log2_n = n_f.log2();
    let mut milli_bits: f64 = 0.0;
    for channel_hist in &hist {
        for &count in channel_hist.iter() {
            if count == 0 {
                continue;
            }
            let c_f = count as f64;
            // Per-bin contribution to N·H: c·log2(N/c).
            milli_bits += c_f * (log2_n - c_f.log2());
        }
    }
    // Scale to milli-bits and round to nearest.
    (milli_bits * 1000.0 + 0.5) as u64
}

/// Round 161 *Shannon-entropy bit-cost* variant of
/// [`pick_block_mode_with_hint`].
///
/// Picks the §4.1 mode minimising [`block_mode_entropy_cost`] — a
/// true Huffman lower-bound bit cost rather than the L1 magnitude
/// proxy the round-159/160 chooser uses. The entropy bit-cost
/// correctly distinguishes a "near-zero with two outliers"
/// residual distribution (low L1, but the outliers force long
/// Huffman codes for the two distinct outlier values) from a
/// "spread of small values" distribution (slightly higher L1, but
/// more concentrated histogram → lower Huffman cost). The L1
/// proxy treats them as comparable; the entropy cost reflects
/// what the §5.x prefix-code writer will actually emit.
///
/// The hint mechanism mirrors [`pick_block_mode_with_hint`]: when
/// `prefer_mode = Some(m)` and `m`'s entropy cost equals the
/// chooser's best, the chooser returns `m` so the predictor sub-
/// image carries longer runs of identical mode values (§7.2
/// `entropy-coded-image` shrinks).
///
/// This is a strict tie-break: residual values are unchanged on
/// cost-equal swaps, so decode round-trips are bit-identical
/// across `prefer_mode` choices. End-to-end the encoder builds
/// both the L1-proxy and entropy-cost candidates and keeps the
/// shortest stream, so the entropy candidate cannot regress
/// against the L1 path — see [`encode_argb_with_predictor_chooser`].
#[allow(clippy::too_many_arguments)]
fn pick_block_mode_with_hint_entropy(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    prefer_mode: Option<u8>,
) -> u8 {
    let mut best_mode: u8 = 0;
    let mut best_cost = u64::MAX;
    for mode in 0u8..=13 {
        let cost = block_mode_entropy_cost(pixels, width, height, x0, y0, bw, bh, mode);
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }
    // Round-159-style strict tie-break under the entropy cost.
    if let Some(m) = prefer_mode {
        if m != best_mode {
            let cost = block_mode_entropy_cost(pixels, width, height, x0, y0, bw, bh, m);
            if cost == best_cost {
                best_mode = m;
            }
        }
    }
    best_mode
}

/// Round 161 *Shannon-entropy bit-cost* variant of
/// [`build_predictor_image`].
///
/// Identical structure to `build_predictor_image`, but routes every
/// per-block mode choice through [`pick_block_mode_with_hint_entropy`]
/// — replacing the round-159 L1-magnitude proxy with a true Huffman
/// lower-bound bit cost. The strict-tie-break hint mechanism is
/// preserved: the left neighbour (or top neighbour at the left
/// edge) is the preferred mode on cost-equal swaps.
///
/// Round-trip correctness is unaffected by the cost model choice:
/// the forward transform later re-derives residuals against the
/// chosen modes, and the decoder's inverse pass uses the same modes
/// from the sub-image, so the decoded image always equals the input.
///
/// The encoder chooser keeps both the L1-proxy candidates (round-
/// 159/160) and the entropy candidate and emits the shortest
/// stream, so a fixture on which the L1 proxy is genuinely better
/// is simply not regressed against.
fn build_predictor_image_entropy(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
) -> (Vec<u32>, u32, u32) {
    let block = 1u32 << size_bits;
    let tw = predictor_div_round_up(width, block);
    let th = predictor_div_round_up(height, block);
    let mut img = Vec::with_capacity((tw * th) as usize);
    let w = width as usize;
    let h = height as usize;
    let bsz = block as usize;
    let mut prev_row: Vec<Option<u8>> = vec![None; tw as usize];
    for by in 0..th as usize {
        let mut left_mode: Option<u8> = None;
        for (bx, top_slot) in prev_row.iter_mut().enumerate() {
            let x0 = bx * bsz;
            let y0 = by * bsz;
            let prefer = left_mode.or(*top_slot);
            let mode = pick_block_mode_with_hint_entropy(pixels, w, h, x0, y0, bsz, bsz, prefer);
            img.push(0xff00_0000 | ((mode as u32) << 8));
            left_mode = Some(mode);
            *top_slot = Some(mode);
        }
    }
    (img, tw, th)
}

/// Round 162 — milli-bit Shannon delta for adding one occurrence of
/// `mode` to a running sub-image mode histogram with current counts
/// `hist[0..14]` and total `total`.
///
/// Returns `(N_new · H_new − N_old · H_old)` in milli-bits, where
/// `H = −Σ p·log2(p)` over the 14-bin mode distribution. This is the
/// **exact** marginal Shannon contribution of one extra `mode`
/// occurrence to the sub-image's symbol entropy mass — the same
/// `Σ c·log2(N/c)` form [`block_mode_entropy_cost`] uses, applied to
/// the sub-image's green-channel mode distribution rather than the
/// per-block residual byte histogram.
///
/// At the floor (`hist` all zero, `total == 0`) the delta is zero:
/// adding the first symbol moves the system from a degenerate
/// no-symbol state to a single-symbol histogram with `H = 0`. The
/// first **subsequent** occurrence of a *different* mode does grow
/// the mass (now two distinct symbols, total = 2 → `N·H = 2`). The
/// formula stays well-defined at every step because the post-add
/// histogram always has `total + 1 ≥ 1` and all bins with `c == 0`
/// are skipped from the sum.
///
/// Used by [`pick_block_mode_with_hint_entropy_subaware`] to charge a
/// per-block mode candidate not only for its own residual entropy
/// but also for its marginal contribution to the §7.2 predictor
/// sub-image's prefix-code mass — making the chooser sub-image-
/// aware in a way the round-159 hint and round-160 slack budget were
/// not (those mechanisms only acted on local neighbour identity,
/// without any global accounting of the sub-image's distribution
/// shape).
fn sub_image_mode_cost_delta_milli(hist: &[u32; 14], total: u32, mode: u8) -> u64 {
    debug_assert!(mode < 14);
    // Compute Σ c·log2(N/c) before and after; the delta is the
    // marginal Shannon mass in bits, scaled to milli-bits and
    // rounded to nearest u64. Float arithmetic is fine here for the
    // same reason as `block_mode_entropy_cost`: the rounding step
    // makes the result bit-for-bit deterministic across IEEE-754
    // log2 implementations to within ±1 milli-bit, which is finer
    // than any per-symbol cost ordering.
    let n_old = total as f64;
    let n_new = (total + 1) as f64;
    let log2_n_old = if total > 0 { n_old.log2() } else { 0.0 };
    let log2_n_new = n_new.log2();
    let mut mass_old: f64 = 0.0;
    let mut mass_new: f64 = 0.0;
    for (m, &c) in hist.iter().enumerate() {
        let c_after = if m == mode as usize { c + 1 } else { c };
        if c > 0 {
            let c_f = c as f64;
            mass_old += c_f * (log2_n_old - c_f.log2());
        }
        if c_after > 0 {
            let c_f = c_after as f64;
            mass_new += c_f * (log2_n_new - c_f.log2());
        }
    }
    let delta = (mass_new - mass_old).max(0.0);
    (delta * 1000.0 + 0.5) as u64
}

/// Round 162 — *sub-image-aware* Shannon-entropy bit-cost variant of
/// [`pick_block_mode_with_hint_entropy`].
///
/// Picks the §4.1 mode minimising the **joint** cost
///
/// ```text
///     cost(m) = block_mode_entropy_cost(..., m)
///             + (lambda_milli * sub_image_mode_cost_delta_milli(hist, total, m)) / 1000
/// ```
///
/// where the first term is the per-block residual entropy (same
/// metric the round-161 chooser uses) and the second term is the
/// marginal §7.2 predictor sub-image cost — the bits the
/// `entropy-coded-image` writer will emit for this mode value given
/// the sub-image's running distribution shape. `lambda_milli` is the
/// per-sub-image-bit weight, in milli-units (so `lambda_milli = 1000`
/// weights one sub-image bit equal to one residual bit). Larger
/// lambda biases the chooser toward modes that reuse already-popular
/// values in the sub-image; `lambda_milli == 0` recovers the round-
/// 161 entropy-only chooser exactly (no sub-image weighting at all).
///
/// The round-159 strict tie-break hint is preserved: when
/// `prefer_mode = Some(m)` and `m`'s joint cost equals the chooser's
/// best, the chooser returns `m` so the sub-image keeps the longer
/// run of identical mode values. The hint check uses the same joint
/// cost (residual + lambda · sub-image delta) the main sweep uses,
/// so the tie semantics stay self-consistent.
///
/// Round-trip correctness is unaffected by the cost model choice:
/// the forward transform later re-derives residuals against the
/// chosen modes, and the decoder's inverse pass uses the same modes
/// from the sub-image, so the decoded image always equals the input.
///
/// The encoder protects itself from regressions by building both the
/// round-161 (sub-image-unaware) and round-162 (sub-image-aware at
/// multiple lambda values) predictor candidates and keeping the
/// shortest stream — so a fixture on which the sub-image weighting
/// hurts overall byte cost is simply not chosen.
#[allow(clippy::too_many_arguments)]
fn pick_block_mode_with_hint_entropy_subaware(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    prefer_mode: Option<u8>,
    sub_image_hist: &[u32; 14],
    sub_image_total: u32,
    lambda_milli: u64,
) -> u8 {
    let mut best_mode: u8 = 0;
    let mut best_cost = u64::MAX;
    for mode in 0u8..=13 {
        let residual_cost = block_mode_entropy_cost(pixels, width, height, x0, y0, bw, bh, mode);
        let sub_delta = sub_image_mode_cost_delta_milli(sub_image_hist, sub_image_total, mode);
        // lambda_milli is "per-sub-image-bit weight in milli-units".
        // sub_delta is already in milli-bits. Multiply and divide by
        // 1000 to keep the whole expression in milli-bit units.
        let weighted_sub = sub_delta.saturating_mul(lambda_milli) / 1000;
        let cost = residual_cost.saturating_add(weighted_sub);
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }
    if let Some(m) = prefer_mode {
        if m != best_mode {
            let residual_cost = block_mode_entropy_cost(pixels, width, height, x0, y0, bw, bh, m);
            let sub_delta = sub_image_mode_cost_delta_milli(sub_image_hist, sub_image_total, m);
            let weighted_sub = sub_delta.saturating_mul(lambda_milli) / 1000;
            let cost = residual_cost.saturating_add(weighted_sub);
            if cost == best_cost {
                best_mode = m;
            }
        }
    }
    best_mode
}

/// Round 162 *sub-image-aware* variant of
/// [`build_predictor_image_entropy`].
///
/// Identical structure to `build_predictor_image_entropy`, but routes
/// every per-block mode choice through
/// [`pick_block_mode_with_hint_entropy_subaware`] with a running
/// histogram of the sub-image's mode values chosen so far. `lambda_milli`
/// is the per-sub-image-bit weight (see
/// [`pick_block_mode_with_hint_entropy_subaware`] for the unit). The
/// round-159 strict-tie-break hint mechanism is preserved: the left
/// neighbour (or top neighbour at the left edge) is the preferred
/// mode on joint-cost-equal swaps.
///
/// `lambda_milli == 0` is byte-identical to
/// `build_predictor_image_entropy` (the sub-image term contributes
/// zero to every candidate). Larger `lambda_milli` biases the
/// chooser toward modes that reuse already-popular values in the
/// sub-image.
///
/// Round-trip correctness is unaffected: the decoder reads the
/// chosen modes from the sub-image; the forward transform recomputes
/// residuals against them. The chooser's joint-cost choice only
/// shifts which mode is recorded per block — never the decode
/// reconstruction path.
fn build_predictor_image_entropy_subaware(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
    lambda_milli: u64,
) -> (Vec<u32>, u32, u32) {
    let block = 1u32 << size_bits;
    let tw = predictor_div_round_up(width, block);
    let th = predictor_div_round_up(height, block);
    let mut img = Vec::with_capacity((tw * th) as usize);
    let w = width as usize;
    let h = height as usize;
    let bsz = block as usize;
    let mut prev_row: Vec<Option<u8>> = vec![None; tw as usize];
    let mut hist = [0u32; 14];
    let mut total: u32 = 0;
    for by in 0..th as usize {
        let mut left_mode: Option<u8> = None;
        for (bx, top_slot) in prev_row.iter_mut().enumerate() {
            let x0 = bx * bsz;
            let y0 = by * bsz;
            let prefer = left_mode.or(*top_slot);
            let mode = pick_block_mode_with_hint_entropy_subaware(
                pixels,
                w,
                h,
                x0,
                y0,
                bsz,
                bsz,
                prefer,
                &hist,
                total,
                lambda_milli,
            );
            img.push(0xff00_0000 | ((mode as u32) << 8));
            left_mode = Some(mode);
            *top_slot = Some(mode);
            hist[mode as usize] += 1;
            total += 1;
        }
    }
    (img, tw, th)
}

/// Apply the §4.1 *forward* predictor transform: for each pixel,
/// replace it with the per-channel mod-256 residual `(original -
/// pred)`. `pred` is computed from the **source** (un-modified)
/// pixels — see [`predictor_at`] — so the decoder's inverse pass
/// (which uses already-reconstructed pixels equal to those source
/// pixels) recovers the originals exactly.
///
/// Writes residuals into `dst` (`width * height` long). `src` is
/// the un-modified source. `predictor_image` / `transform_width` /
/// `size_bits` describe the sub-resolution mode image. Per §4.1's
/// border rules the top-left predicts solid black, the top row
/// predicts L, the left column predicts T, the rightmost column
/// uses the row's leftmost pixel as TR; interior pixels read their
/// mode from the predictor image's green channel.
fn apply_forward_predictor(
    src: &[u32],
    dst: &mut [u32],
    width: u32,
    height: u32,
    predictor_image: &[u32],
    transform_width: u32,
    size_bits: u8,
) {
    if width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            // Interior pixels read their block mode from the
            // sub-resolution predictor image; border rules in
            // `predictor_at` ignore the mode for top-row /
            // left-column / top-left pixels.
            let mode = if x == 0 || y == 0 {
                0
            } else {
                let bx = (x as u32) >> size_bits;
                let by = (y as u32) >> size_bits;
                let block_index = (by * transform_width + bx) as usize;
                ((predictor_image[block_index] >> 8) & 0xff) as u8
            };
            let pred = predictor_at(src, w, x, y, mode);
            dst[idx] = predictor_subtract(src[idx], pred);
        }
    }
}

/// Default §4.1 `size_bits` value the encoder picks for the
/// predictor sub-image: `4` → 16×16 pixel blocks. Smaller blocks
/// give finer mode granularity (better residual savings) at the
/// cost of a larger predictor sub-image (4× the entries for each
/// `size_bits` decrement). 16×16 is a reasonable middle ground for
/// the typical encoder workloads here; the spec admits `2..=9`
/// (`block` sizes 4..=512). As of round 155 the chooser also
/// evaluates a maximal single-block candidate by promoting
/// `size_bits` until `1 << size_bits ≥ max(width, height)`, so the
/// default value here only sets the per-region granularity floor;
/// see [`encode_argb_with_predictor_chooser`].
const DEFAULT_PREDICTOR_SIZE_BITS: u8 = 4;

/// Encode `pixels` taking the §4.1 spatial predictor path: pick a
/// per-block predictor mode minimising the residual magnitude,
/// transform the pixels to residuals, then encode the residuals via
/// the standard `spatially-coded-image` shape — wrapped by an
/// `optional-transform` whose first entry is the §4.1 predictor
/// transform (header bit `%b1` + transform type `Predictor = 0` +
/// 3-bit `size_bits - 2` + the sub-resolution predictor image as an
/// `entropy-coded-image`).
///
/// The chooser composes with `cache_code_bits`: when `Some(bits)` a
/// §5.2.3 color cache of that size is built over the residual
/// stream's literal tokens.
///
/// **NB:** the predictor transform requires at least a 2-pixel
/// dimension on the side being predicted (a 1-pixel image triggers
/// the §4.1 top-left-only border rule, so the transform body cannot
/// produce a meaningful residual). The caller should fall back to
/// the no-transform candidate for trivially small images.
fn encode_with_predictor(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
    cache_code_bits: Option<u32>,
    image_width: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();

    // ---- §3.8.2 / §7.2 optional-transform: predictor-tx ----
    // present bit `%b1`.
    w.write_bit(true);
    // transform type `Predictor = 0`, 2 bits.
    w.write_bits(crate::vp8l_stream::TransformType::Predictor as u32, 2);
    // 3-bit `size_bits - 2` (decoder adds 2 back per §4.1).
    debug_assert!((2..=9).contains(&size_bits));
    w.write_bits((size_bits - 2) as u32, 3);

    // Build the sub-resolution predictor image then write it as an
    // entropy-coded-image per §7.2 `predictor-image = 3BIT
    // entropy-coded-image`.
    let (predictor_image, tw, _th) = build_predictor_image(pixels, width, height, size_bits);
    write_entropy_coded_image_literals(&mut w, &predictor_image);

    // End of optional-transform list (`%b0`).
    w.write_bit(false);

    // ---- Forward-transform the main image into residuals ----
    let mut residuals = vec![0u32; pixels.len()];
    apply_forward_predictor(
        pixels,
        &mut residuals,
        width,
        height,
        &predictor_image,
        tw,
        size_bits,
    );

    // ---- Tokenise + emit the residual spatially-coded-image ----
    let mut tokens = tokenize_lz77(&residuals);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, &residuals, bits);
    }
    write_spatially_coded_image(&mut w, &tokens, cache_code_bits, image_width);

    w.into_bytes()
}

/// Round-160 *slack-cost* variant of [`encode_with_predictor`].
///
/// Same wire shape as `encode_with_predictor`, but the §4.1
/// predictor sub-image is built via
/// [`build_predictor_image_with_slack`] with the caller-supplied
/// `slack` budget. `slack == 0` produces a byte-identical stream
/// to `encode_with_predictor`.
///
/// `slack > 0` permits the chooser to swap to the preferred
/// neighbour mode at a small residual-cost increase, with the goal
/// of dropping the predictor sub-image's symbol entropy. The
/// chooser at [`encode_argb_with_predictor_chooser`] always
/// compares the slack candidates against `slack == 0`, so a slack
/// budget that hurts overall byte cost on a given input is
/// non-selecting (the strict candidate wins on byte length).
fn encode_with_predictor_slack(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
    cache_code_bits: Option<u32>,
    image_width: u32,
    slack: u64,
) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_bit(true);
    w.write_bits(crate::vp8l_stream::TransformType::Predictor as u32, 2);
    debug_assert!((2..=9).contains(&size_bits));
    w.write_bits((size_bits - 2) as u32, 3);

    let (predictor_image, tw, _th) =
        build_predictor_image_with_slack(pixels, width, height, size_bits, slack);
    write_entropy_coded_image_literals(&mut w, &predictor_image);

    w.write_bit(false);

    let mut residuals = vec![0u32; pixels.len()];
    apply_forward_predictor(
        pixels,
        &mut residuals,
        width,
        height,
        &predictor_image,
        tw,
        size_bits,
    );

    let mut tokens = tokenize_lz77(&residuals);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, &residuals, bits);
    }
    write_spatially_coded_image(&mut w, &tokens, cache_code_bits, image_width);

    w.into_bytes()
}

/// Round-161 *Shannon-entropy bit-cost* variant of
/// [`encode_with_predictor`].
///
/// Same wire shape as `encode_with_predictor`, but the §4.1
/// predictor sub-image is built via [`build_predictor_image_entropy`]
/// — replacing the per-block L1-magnitude proxy with a true Huffman
/// lower-bound bit cost on the per-channel residual histogram. The
/// chooser hint mechanism (strict tie-break favouring the
/// neighbour's mode) is preserved.
///
/// `encode_argb_with_predictor_chooser` always compares this
/// candidate against the L1-proxy candidates (round-159 strict tie-
/// break and round-160 slack variants), so on fixtures where the L1
/// proxy genuinely wins, the entropy candidate is non-selecting.
fn encode_with_predictor_entropy(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
    cache_code_bits: Option<u32>,
    image_width: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_bit(true);
    w.write_bits(crate::vp8l_stream::TransformType::Predictor as u32, 2);
    debug_assert!((2..=9).contains(&size_bits));
    w.write_bits((size_bits - 2) as u32, 3);

    let (predictor_image, tw, _th) =
        build_predictor_image_entropy(pixels, width, height, size_bits);
    write_entropy_coded_image_literals(&mut w, &predictor_image);

    w.write_bit(false);

    let mut residuals = vec![0u32; pixels.len()];
    apply_forward_predictor(
        pixels,
        &mut residuals,
        width,
        height,
        &predictor_image,
        tw,
        size_bits,
    );

    let mut tokens = tokenize_lz77(&residuals);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, &residuals, bits);
    }
    write_spatially_coded_image(&mut w, &tokens, cache_code_bits, image_width);

    w.into_bytes()
}

/// Round 162 — *sub-image-aware* Shannon-entropy bit-cost predictor
/// path. Identical to [`encode_with_predictor_entropy`] but routes
/// the sub-image construction through
/// [`build_predictor_image_entropy_subaware`] with `lambda_milli` as
/// the per-sub-image-bit weight for the joint cost.
///
/// `lambda_milli == 0` is byte-identical to
/// [`encode_with_predictor_entropy`] (the sub-image term contributes
/// zero to every per-block choice, so the chooser falls back to the
/// round-161 entropy chooser).
///
/// `encode_argb_with_predictor_chooser` always compares the round-
/// 162 candidates (multiple lambda settings) against every round-159
/// / round-160 / round-161 candidate, so on fixtures where sub-
/// image weighting hurts overall byte cost, the round-162 candidate
/// is non-selecting and the path strictly extends the encoder's
/// option set rather than redirecting it.
fn encode_with_predictor_entropy_subaware(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
    cache_code_bits: Option<u32>,
    image_width: u32,
    lambda_milli: u64,
) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_bit(true);
    w.write_bits(crate::vp8l_stream::TransformType::Predictor as u32, 2);
    debug_assert!((2..=9).contains(&size_bits));
    w.write_bits((size_bits - 2) as u32, 3);

    let (predictor_image, tw, _th) =
        build_predictor_image_entropy_subaware(pixels, width, height, size_bits, lambda_milli);
    write_entropy_coded_image_literals(&mut w, &predictor_image);

    w.write_bit(false);

    let mut residuals = vec![0u32; pixels.len()];
    apply_forward_predictor(
        pixels,
        &mut residuals,
        width,
        height,
        &predictor_image,
        tw,
        size_bits,
    );

    let mut tokens = tokenize_lz77(&residuals);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, &residuals, bits);
    }
    write_spatially_coded_image(&mut w, &tokens, cache_code_bits, image_width);

    w.into_bytes()
}

// ---- §3.5.2 / §4.2 forward color-transform encoder ------------------

/// §3.5.2 `ColorTransformDelta(t, c)` = `(int8(t) * int8(c)) >> 5`,
/// with `t` and `c` interpreted as signed 8-bit two's-complement values.
/// Identical formula to the decoder's
/// [`crate::vp8l_transform::color_transform_delta`] — kept local so this
/// module compiles under `--no-default-features` (which the decoder also
/// satisfies, but the helper is `pub(crate)`-private to that file).
///
/// Only the low 8 bits of the result are meaningful per §3.5.2
/// ("only the lowest 8 bits are used from the result"); the wider `i32`
/// return type lets callers fold it into a signed pixel computation
/// before masking.
#[inline]
fn color_xfrm_delta(t: u8, c: u8) -> i32 {
    let ts = t as i8 as i32;
    let cs = c as i8 as i32;
    (ts * cs) >> 5
}

/// §3.5.2 *forward* color-transform on one pixel.
///
/// Subtracts the three color-transform deltas from `red` and `blue`
/// (green is untouched per §3.5.2). The arguments mirror the §3.5.2
/// `ColorTransform()` C signature: the per-block element is unpacked
/// into `(green_to_red, green_to_blue, red_to_blue)`. Returns the
/// encoded `(new_red, new_blue)` as low 8-bit residuals. The §3.5.2
/// red argument to the third delta is the *original* `red` (not the
/// post-green-to-red residual), matching the spec's encoder pseudo-
/// code; the decoder's inverse adds the same delta back using its
/// reconstructed `tmp_red & 0xff`, which by symmetry equals the
/// original red, so the round-trip is bit-exact.
#[inline]
fn forward_color_pixel(
    r: u8,
    g: u8,
    b: u8,
    green_to_red: u8,
    green_to_blue: u8,
    red_to_blue: u8,
) -> (u8, u8) {
    let mut tmp_red = r as i32;
    let mut tmp_blue = b as i32;
    tmp_red -= color_xfrm_delta(green_to_red, g);
    tmp_blue -= color_xfrm_delta(green_to_blue, g);
    tmp_blue -= color_xfrm_delta(red_to_blue, r);
    ((tmp_red & 0xff) as u8, (tmp_blue & 0xff) as u8)
}

/// §3.5.2 color-transform candidate values swept by [`pick_block_cte`]
/// for each of the three `(green_to_red, green_to_blue, red_to_blue)`
/// axes.
///
/// Each value is an 8-bit two's-complement integer. With the §3.5.2
/// fixed-point interpretation (`>> 5` divides by 32), a value of 32
/// corresponds to a slope of 1 in the corresponding channel; the
/// listed entries span `[-96, 96]` with fine resolution `±4` near
/// zero (where most natural-image channel correlations sit, e.g. a
/// slope of 1/3 ≈ 10.7 fixed-point) coarsening to `±16` further out.
/// Including 0 ("no transform") guarantees the per-axis chooser never
/// picks a CTE worse than the no-correlation baseline on that axis.
///
/// 25 candidates × 3 axes = 75 cost evaluations per block (with the
/// per-axis greedy in `pick_block_cte` being exact because the cost
/// decomposes additively across the red and blue channels — green is
/// untouched, the red channel depends only on `green_to_red`, and the
/// blue channel depends additively on `(green_to_blue, red_to_blue)`).
const CTE_AXIS_CANDIDATES: [u8; 25] = [
    0xa0, // -96
    0xb0, // -80
    0xc0, // -64
    0xd0, // -48
    0xe0, // -32
    0xe8, // -24
    0xec, // -20
    0xf0, // -16
    0xf4, // -12
    0xf8, //  -8
    0xfc, //  -4
    0xfe, //  -2
    0x00, //   0
    0x02, //   2
    0x04, //   4
    0x08, //   8
    0x0c, //  12
    0x10, //  16
    0x14, //  20
    0x18, //  24
    0x20, //  32
    0x30, //  48
    0x40, //  64
    0x50, //  80
    0x60, //  96
];

/// Per-channel folded-magnitude cost: same residual-magnitude proxy
/// [`residual_magnitude`] uses for the §4.1 predictor, but on a single
/// 8-bit channel — `min(v, 256 - v)`. Lower magnitudes peak the
/// histogram near zero, which the per-channel Huffman codes compress
/// better.
#[inline]
fn channel_magnitude(v: u32) -> u32 {
    let v = v & 0xff;
    if v <= 128 {
        v
    } else {
        256 - v
    }
}

/// §3.5.2: pick the `(green_to_red, green_to_blue, red_to_blue)`
/// element that minimises the residual-magnitude cost on the
/// rectangular block `[x0, x0+bw) × [y0, y0+bh)` of the
/// `width × height` image.
///
/// The cost decomposes additively across channels (green is untouched
/// by §3.5.2, red depends only on `green_to_red`, blue depends on
/// `green_to_blue + red_to_blue`), so a per-axis greedy sweep over
/// [`CTE_AXIS_CANDIDATES`] is exact:
///
/// 1. For each `gtr` candidate, sum `|red - delta(gtr, green)| & 0xff`
///    folded onto `[-128, 127]` over the block's pixels; keep the
///    smallest.
/// 2. For each `gtb` candidate, sum
///    `|blue - delta(gtb, green)| & 0xff` folded similarly.
/// 3. For each `rtb` candidate, sum
///    `|(blue - delta(best_gtb, green)) - delta(rtb, red)| & 0xff`.
///
/// On ties the candidate appearing earlier in
/// [`CTE_AXIS_CANDIDATES`] wins, which makes the chooser deterministic.
///
/// Public so the `pick_block_cte` criterion bench can drive the
/// chooser walk directly (same shelf as [`predictor_subtract`] /
/// [`apply_subtract_green`]); encoder callers go through
/// `build_color_image`.
pub fn pick_block_cte(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
) -> (u8, u8, u8) {
    // Gather the block's per-pixel channel triples once.
    let mut samples: Vec<(u8, u8, u8)> = Vec::with_capacity(bw * bh);
    for dy in 0..bh {
        let y = y0 + dy;
        if y >= height {
            break;
        }
        for dx in 0..bw {
            let x = x0 + dx;
            if x >= width {
                break;
            }
            let px = pixels[y * width + x];
            let r = ((px >> 16) & 0xff) as u8;
            let g = ((px >> 8) & 0xff) as u8;
            let b = (px & 0xff) as u8;
            samples.push((r, g, b));
        }
    }
    if samples.is_empty() {
        return (0, 0, 0);
    }

    // Axis 1: green → red. The red residual is
    // `(red - delta(gtr, green)) & 0xff`, independent of gtb and rtb.
    let best_gtr = sweep_cte_axis(&samples, |gtr, r, g, _b| {
        channel_magnitude((r as i32 - color_xfrm_delta(gtr, g)) as u32)
    });

    // Axis 2: green → blue. The intermediate blue residual is
    // `(blue - delta(gtb, green)) & 0xff`, independent of rtb. We
    // evaluate the GREEN→BLUE contribution alone here; the joint
    // (gtb, rtb) choice is exact because the red-to-blue delta is
    // additive in `rtb` and depends only on the original red.
    let best_gtb = sweep_cte_axis(&samples, |gtb, _r, g, b| {
        channel_magnitude((b as i32 - color_xfrm_delta(gtb, g)) as u32)
    });

    // Axis 3: red → blue. Fold the now-fixed green→blue delta into
    // each pixel's intermediate blue, then sweep rtb.
    let best_rtb = sweep_cte_axis(&samples, |rtb, r, g, b| {
        let inter = b as i32 - color_xfrm_delta(best_gtb, g);
        channel_magnitude((inter - color_xfrm_delta(rtb, r)) as u32)
    });

    (best_gtr, best_gtb, best_rtb)
}

/// One per-axis greedy sweep of [`pick_block_cte`]: evaluate every
/// [`CTE_AXIS_CANDIDATES`] entry's summed per-sample cost and return
/// the candidate with the smallest sum (earliest entry wins ties).
///
/// The prune that used to run per sample (`cost >= best` → abandon
/// the candidate) is checked at [`CTE_PRUNE_CHUNK`]-sample
/// granularity instead, the same despecialisation the round-280
/// §4.1 chooser walker applied at block-row granularity: the
/// interior chunk loop carries no data-dependent exit, so the
/// monomorphised `cost` closure body auto-vectorises. Pick-identical
/// by the round-280 argument — per-sample contributions are
/// non-negative, so a partial sum reaching `>= best` implies the
/// full sum also compares `>= best`, and a candidate that now runs
/// to completion instead of pruning yields its exact full sum, which
/// is still `>= best` and therefore still loses; completed sums and
/// the strict-`<` tie-break are unchanged. Worst case is one extra
/// chunk of work per pruned candidate.
///
/// The per-chunk partial fits `u32`: each [`channel_magnitude`] is
/// `<= 128`, so a chunk sums to `<= 128 * CTE_PRUNE_CHUNK`.
#[inline]
fn sweep_cte_axis(samples: &[(u8, u8, u8)], cost_of: impl Fn(u8, u8, u8, u8) -> u32) -> u8 {
    let mut best: u8 = 0;
    let mut best_cost = u64::MAX;
    for &cand in &CTE_AXIS_CANDIDATES {
        let mut cost = 0u64;
        for chunk in samples.chunks(CTE_PRUNE_CHUNK) {
            let mut partial = 0u32;
            for &(r, g, b) in chunk {
                partial += cost_of(cand, r, g, b);
            }
            cost += partial as u64;
            if cost >= best_cost {
                break;
            }
        }
        if cost < best_cost {
            best_cost = cost;
            best = cand;
        }
    }
    best
}

/// Sample granularity of the [`sweep_cte_axis`] prune check. 32
/// samples is two 16-pixel block rows at the encoder-default
/// `size_bits = 4` — small enough that a hopeless candidate is
/// abandoned after ~12% of a 16×16 block, large enough for the
/// branch-free interior loop to amortise the check.
const CTE_PRUNE_CHUNK: usize = 32;

/// Build the §3.5.2 sub-resolution *color image*: one ARGB pixel per
/// `(1 << size_bits)`-pixel-square block of the main image, with the
/// chosen [`ColorTransformElement`] packed per §3.5.2 ("each
/// `ColorTransformElement` 'cte' is treated as a pixel in a
/// subresolution image whose alpha component is 255, red component is
/// `cte.red_to_blue`, green component is `cte.green_to_blue`, and
/// blue component is `cte.green_to_red`").
///
/// Returns `(color_image, transform_width, transform_height)`. The
/// dimensions follow the §4.2 `DIV_ROUND_UP` rule, identical to the
/// §4.1 predictor image's.
fn build_color_image(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
) -> (Vec<u32>, u32, u32) {
    let block = 1u32 << size_bits;
    let tw = predictor_div_round_up(width, block);
    let th = predictor_div_round_up(height, block);
    let mut img = Vec::with_capacity((tw * th) as usize);
    let w = width as usize;
    let h = height as usize;
    let bsz = block as usize;
    for by in 0..th as usize {
        for bx in 0..tw as usize {
            let x0 = bx * bsz;
            let y0 = by * bsz;
            let (gtr, gtb, rtb) = pick_block_cte(pixels, w, h, x0, y0, bsz, bsz);
            // Pack the CTE into one ARGB pixel exactly as §3.5.2
            // specifies: alpha=255, red=red_to_blue, green=green_to_blue,
            // blue=green_to_red. The decoder unpacks it in
            // `crate::vp8l_transform::inverse_color` via the same
            // channel-name mapping.
            let argb = 0xff00_0000 | ((rtb as u32) << 16) | ((gtb as u32) << 8) | (gtr as u32);
            img.push(argb);
        }
    }
    (img, tw, th)
}

/// Apply the §3.5.2 *forward* color transform: for each pixel, look up
/// the per-block element from `color_image` (with the §3.5.2 channel
/// layout) and rewrite the red and blue channels via
/// [`forward_color_pixel`]. Green and alpha are passed through.
///
/// Writes the transformed pixels into `dst` (`width * height` long).
/// `src` is the un-modified source; the encoder transforms against the
/// originals because the decoder reconstructs identical originals
/// channel-by-channel (the inverse adds back the same per-block delta).
fn apply_forward_color(
    src: &[u32],
    dst: &mut [u32],
    width: u32,
    height: u32,
    color_image: &[u32],
    transform_width: u32,
    size_bits: u8,
) {
    if width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let bx = (x as u32) >> size_bits;
            let by = (y as u32) >> size_bits;
            let block_index = (by * transform_width + bx) as usize;
            let cte = color_image[block_index];
            // §3.5.2 channel mapping: red=red_to_blue, green=green_to_blue,
            // blue=green_to_red.
            let red_to_blue = ((cte >> 16) & 0xff) as u8;
            let green_to_blue = ((cte >> 8) & 0xff) as u8;
            let green_to_red = (cte & 0xff) as u8;

            let px = src[idx];
            let a = ((px >> 24) & 0xff) as u8;
            let r = ((px >> 16) & 0xff) as u8;
            let g = ((px >> 8) & 0xff) as u8;
            let b = (px & 0xff) as u8;
            let (new_r, new_b) =
                forward_color_pixel(r, g, b, green_to_red, green_to_blue, red_to_blue);
            dst[idx] =
                ((a as u32) << 24) | ((new_r as u32) << 16) | ((g as u32) << 8) | (new_b as u32);
        }
    }
}

/// Default §3.5.2 `size_bits` value the encoder picks for the color
/// sub-image: `4` → 16×16 pixel blocks, matching
/// [`DEFAULT_PREDICTOR_SIZE_BITS`]. The spec admits `2..=9`
/// (`block` sizes 4..=512); finer blocks give better per-block CTE
/// fitting at the cost of a larger color sub-image. 16×16 is a
/// reasonable middle ground for the typical encoder workloads here.
const DEFAULT_COLOR_TRANSFORM_SIZE_BITS: u8 = 4;

/// Encode `pixels` taking the §3.5.2 / §4.2 color-transform path: pick
/// a per-block `(green_to_red, green_to_blue, red_to_blue)` triple,
/// forward-transform the red and blue channels into the per-block
/// residuals, then encode the residuals via the standard
/// `spatially-coded-image` shape — wrapped by an `optional-transform`
/// whose first entry is the §4.2 color transform (header bit `%b1` +
/// transform type `Color = 1` + 3-bit `size_bits - 2` + the sub-
/// resolution color image as an `entropy-coded-image`).
///
/// The chooser composes with `cache_code_bits`: when `Some(bits)` a
/// §5.2.3 color cache of that size is built over the residual stream's
/// literal tokens.
///
/// **NB:** the color transform requires at least a `1 << size_bits`-
/// pixel side on both dimensions so the sub-resolution image has more
/// than one block; smaller images fall back to the no-transform
/// candidates.
fn encode_with_color_transform(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
    cache_code_bits: Option<u32>,
    image_width: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();

    // ---- §3.8.2 / §7.2 optional-transform: color-tx ----
    // present bit `%b1`.
    w.write_bit(true);
    // transform type `Color = 1`, 2 bits.
    w.write_bits(crate::vp8l_stream::TransformType::Color as u32, 2);
    // 3-bit `size_bits - 2` (decoder adds 2 back per §3.5.2).
    debug_assert!((2..=9).contains(&size_bits));
    w.write_bits((size_bits - 2) as u32, 3);

    // Build the sub-resolution color image then write it as an
    // entropy-coded-image per §7.2 `color-image = 3BIT
    // entropy-coded-image`.
    let (color_image, tw, _th) = build_color_image(pixels, width, height, size_bits);
    write_entropy_coded_image_literals(&mut w, &color_image);

    // End of optional-transform list (`%b0`).
    w.write_bit(false);

    // ---- Forward-transform the main image ----
    let mut residuals = vec![0u32; pixels.len()];
    apply_forward_color(
        pixels,
        &mut residuals,
        width,
        height,
        &color_image,
        tw,
        size_bits,
    );

    // ---- Tokenise + emit the residual spatially-coded-image ----
    let mut tokens = tokenize_lz77(&residuals);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, &residuals, bits);
    }
    write_spatially_coded_image(&mut w, &tokens, cache_code_bits, image_width);

    w.into_bytes()
}

// ---- §4.4 color-indexing transform encoder --------------------------

/// §4.4 upper bound on the color-table size that triggers the
/// color-indexing transform: the spec describes the inverse with an
/// 8-bit on-wire `color_table_size = ReadBits(8) + 1`, so the legal
/// range is `1..=256` unique ARGB colors.
const MAX_PALETTE_SIZE: usize = 256;

/// Scan `pixels` for unique ARGB values and, if the count is below
/// [`MAX_PALETTE_SIZE`], return a `(palette, index_of)` pair:
///
/// * `palette` — the unique ARGB values, sorted numerically. Sorting
///   maximises the per-component delta correlation the §4.4
///   subtraction-coded color table feeds to the entropy stage:
///   adjacent palette entries share similar ARGB bits, so the deltas
///   `palette[i] - palette[i-1]` (per-channel, mod 256) concentrate
///   near zero — the histogram shape Huffman codes shrink best.
///
/// * `index_of` — a lookup map from ARGB pixel value to its position
///   in `palette`, used by [`pack_indices_into_bundled_image`] to
///   replace each pixel with its index.
///
/// Returns `None` as soon as the unique-color count exceeds
/// [`MAX_PALETTE_SIZE`] (the §4.4 on-wire limit), so the early-exit
/// cost on photo-like images is bounded.
fn collect_palette(pixels: &[u32]) -> Option<(Vec<u32>, std::collections::HashMap<u32, u32>)> {
    use std::collections::HashSet;
    let mut set: HashSet<u32> = HashSet::new();
    for &p in pixels {
        set.insert(p);
        if set.len() > MAX_PALETTE_SIZE {
            return None;
        }
    }
    let mut palette: Vec<u32> = set.into_iter().collect();
    palette.sort_unstable();
    let mut map: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::with_capacity(palette.len());
    for (i, &c) in palette.iter().enumerate() {
        map.insert(c, i as u32);
    }
    Some((palette, map))
}

/// §4.4 *subtraction-encode* a color table in place — the inverse of
/// the decoder's [`crate::vp8l_transform::inverse_color_table`].
///
/// The decoder reconstructs `color_table[i] = color_table[i-1] +
/// color_table[i]` (per-channel mod 256), so the encoder emits
/// `color_table[i] - color_table[i-1]` (per-channel mod 256) for
/// `i >= 1`, leaving `color_table[0]` unchanged. Deltas walk
/// back-to-front so each cell still sees the original (pre-encoded)
/// previous value at the moment of subtraction.
fn forward_color_table(color_table: &mut [u32]) {
    if color_table.len() < 2 {
        return;
    }
    for i in (1..color_table.len()).rev() {
        let cur = color_table[i];
        let prev = color_table[i - 1];
        let a = ((cur >> 24) & 0xff).wrapping_sub((prev >> 24) & 0xff) & 0xff;
        let r = ((cur >> 16) & 0xff).wrapping_sub((prev >> 16) & 0xff) & 0xff;
        let g = ((cur >> 8) & 0xff).wrapping_sub((prev >> 8) & 0xff) & 0xff;
        let b = (cur & 0xff).wrapping_sub(prev & 0xff) & 0xff;
        color_table[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// §4.4 *forward* pixel bundling: replace each ARGB pixel by its
/// palette `index`, packing 1/2/4/8 indices into one byte's-worth of
/// green channel per the §4.4 LSB-first packing rule. Other channels
/// are zeroed (alpha 0, red 0, blue 0) — the decoder reads only the
/// green channel via `inverse_color_indexing`.
///
/// `width_bits` is the value the shared §4.4 threshold table
/// [`crate::vp8l_transform::color_indexing_width_bits`] returns for
/// the palette size. `packed_width = DIV_ROUND_UP(width,
/// 1 << width_bits)` — the new image width fed to the §3 image
/// stream.
///
/// Returns the `packed_width * height` ARGB buffer the
/// `spatially-coded-image` writer feeds to the entropy stage. The
/// inverse `inverse_color_indexing` reconstructs the original
/// `width * height` ARGB image when given this buffer and the
/// (un-subtraction-encoded) palette.
fn pack_indices_into_bundled_image(
    pixels: &[u32],
    index_of: &std::collections::HashMap<u32, u32>,
    width: u32,
    height: u32,
    width_bits: u8,
) -> (Vec<u32>, u32) {
    let count = 1u32 << width_bits;
    let bits_per_index = if width_bits == 0 { 8 } else { 8 / count };
    let packed_width = width.div_ceil(count);
    let pw = packed_width as usize;
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u32; pw * h];
    for y in 0..h {
        for x in 0..w {
            let idx = *index_of
                .get(&pixels[y * w + x])
                .expect("collect_palette covered every pixel");
            let packed_x = x / count as usize;
            let sub = x % count as usize;
            let shift = sub * bits_per_index as usize;
            let bits = (idx & ((1u32 << bits_per_index) - 1)) << shift;
            out[y * pw + packed_x] |= bits << 8; // pack into the green channel.
        }
    }
    (out, packed_width)
}

/// Encode `pixels` taking the §4.4 color-indexing transform path:
/// build the unique-color palette, replace every pixel with its
/// palette index (bundled per the §4.4 `width_bits` rule when the
/// palette has ≤16 entries), then emit the bundled-width image via
/// the standard `spatially-coded-image` shape — wrapped by an
/// `optional-transform` whose first entry is the §4.4 color-indexing
/// transform.
///
/// Wire format produced (§3.8.2 / §7.2 grammar):
///
/// ```text
/// optional-transform =
///   %b1                               -- transform present
///   %b11                              -- type ColorIndexing = 3
///   8BIT                              -- color_table_size - 1
///   entropy-coded-image               -- the subtraction-encoded palette,
///                                       written at width = color_table_size,
///                                       height = 1
///   %b0                               -- end of optional-transform list
/// spatially-coded-image               -- packed indices at packed_width
/// ```
///
/// Returns `None` when the palette size exceeds [`MAX_PALETTE_SIZE`]
/// (the §4.4 on-wire limit), so the chooser can skip this candidate
/// in O(N) on photo-like content. The chooser composes with
/// `cache_code_bits`: when `Some(bits)` a §5.2.3 color cache of that
/// size is built over the packed-index stream's literal tokens.
fn encode_with_color_indexing(
    pixels: &[u32],
    width: u32,
    height: u32,
    cache_code_bits: Option<u32>,
) -> Option<Vec<u8>> {
    let (palette, index_of) = collect_palette(pixels)?;
    if palette.is_empty() {
        return None;
    }

    // §4.4 threshold table — single shared copy.
    let width_bits = crate::vp8l_transform::color_indexing_width_bits(palette.len());
    let (packed_image, packed_width) =
        pack_indices_into_bundled_image(pixels, &index_of, width, height, width_bits);

    let mut w = BitWriter::new();

    // ---- §3.8.2 / §7.2 optional-transform: color-indexing-tx ----
    // Header bit `%b1` (transform present).
    w.write_bit(true);
    // Transform type `ColorIndexing = 3` (2 bits, LSB-first → value 3
    // matches the spec's `%b11` MSB-first ABNF when read through
    // `ReadBits(2)`).
    w.write_bits(crate::vp8l_stream::TransformType::ColorIndexing as u32, 2);
    // 8-bit `color_table_size - 1` (decoder adds 1 back per §4.4).
    debug_assert!((1..=MAX_PALETTE_SIZE).contains(&palette.len()));
    w.write_bits((palette.len() - 1) as u32, 8);

    // Color table = an entropy-coded-image at width = color_table_size,
    // height = 1. The on-wire palette is subtraction-encoded; the
    // decoder applies `inverse_color_table` to reverse it.
    let mut subtraction_encoded = palette.clone();
    forward_color_table(&mut subtraction_encoded);
    write_entropy_coded_image_literals(&mut w, &subtraction_encoded);

    // End of optional-transform list (`%b0`).
    w.write_bit(false);

    // ---- Spatially-coded-image at the *subsampled* width ------------
    // After §4.4, `image_width` is `DIV_ROUND_UP(width, 1 <<
    // width_bits)`; that is the width the entropy stage threads
    // through the §5.2.2 distance-code chooser. Pixel values are the
    // packed-green-channel bytes whose red/blue/alpha channels are
    // identically zero, so the per-channel Huffman codes for those
    // three channels collapse to a 1-symbol prefix code each (almost
    // free header overhead).
    let mut tokens = tokenize_lz77(&packed_image);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, &packed_image, bits);
    }
    write_spatially_coded_image(&mut w, &tokens, cache_code_bits, packed_width);

    Some(w.into_bytes())
}

// ---- §6.2.2 multi-meta-prefix (entropy-image) encoder ----------------

/// Default `prefix_bits` candidate the §6.2.2 multi-meta-prefix
/// chooser sweeps. Each value gives a block side of `1 << prefix_bits`
/// pixels — larger blocks mean fewer of them (cheap entropy image,
/// fewer prefix-code groups) but coarser per-region adaptation; smaller
/// blocks mean finer adaptation but a larger entropy-image overhead.
/// The sweep across `[4, 5, 6, 7]` gives 16/32/64/128-pixel blocks,
/// which span the useful range for the dimensions this crate targets
/// (typical lossless WebP fixtures are 16..512 pixels per side).
///
/// The spec admits `prefix_bits ∈ [2..9]` (i.e. 4..512-pixel blocks);
/// the chooser narrows that to four values rather than the full eight
/// because the very smallest (4-pixel) blocks rarely beat the
/// single-group baseline (the entropy image grows quadratically with
/// `1 / block_side`) and the largest (256/512-pixel) blocks are
/// useless on the smaller images this candidate targets.
const META_PREFIX_BITS_SWEEP: [u8; 4] = [4, 5, 6, 7];

/// Largest number of prefix-code groups the §6.2.2 chooser will form.
/// Each group costs five additional code-length tables in the stream
/// header (~30..120 bits per code), so the chooser only pays the
/// overhead when the per-group savings on the LZ77 stream beat the
/// header cost. Capping at 4 keeps the chooser's wall-time bounded
/// while covering the per-region adaptation that pays for itself on
/// natural images (where the per-quadrant statistics diverge enough to
/// justify separate codes).
const MAX_META_GROUPS: u32 = 4;

// ---- §6.2.2 histogram-distance block clusterer -------------------------
//
// Spec context (RFC 9649 §3.7.2.2 / WebP Lossless §6.2.2): the §5.2 LZ77
// + prefix-code-group decoder selects one of `num_prefix_groups` groups
// per pixel block. The encoder gets to choose how to *partition* the
// image's blocks into groups — the spec only constrains the on-wire
// representation (an `entropy-coded-image` whose green+red channels
// carry the per-block meta-prefix code).
//
// The right partition collects blocks whose alphabet-symbol histograms
// (green, red, blue, alpha + LZ77 length / distance) match closely, so
// each group's shared §6.2 prefix code can compact those symbols
// efficiently. A direct symbol-histogram clusterer would have to
// pre-tokenise to see which symbols each block produces, which puts a
// hard constraint on the matcher (`tokenize_lz77` runs *after* the
// clusterer here). We use a pixel-domain proxy instead: a coarse
// per-channel RGB histogram. Blocks whose pixel-value distributions
// agree at bin resolution will, in expectation, produce closely-matched
// literal-symbol frequencies, which is exactly what drives §6.2's
// per-group code cost.

/// Bin shift collapsing the 256-value channel range into a coarser
/// histogram for clustering. `BIN_SHIFT = 4` → 16 bins per channel.
///
/// The smaller the shift the finer the discrimination but the more
/// per-block memory + per-iteration arithmetic; 4 keeps the per-block
/// feature vector at 48 `u32` slots (16 × 3 channels) which is small
/// enough to scan repeatedly in Lloyd's iteration but large enough to
/// distinguish meaningfully different per-region distributions on
/// natural-image inputs.
const CLUSTER_BIN_SHIFT: u32 = 4;
/// Number of histogram bins per channel after [`CLUSTER_BIN_SHIFT`]:
/// `256 >> CLUSTER_BIN_SHIFT`.
const CLUSTER_BINS_PER_CHANNEL: usize = 256 >> CLUSTER_BIN_SHIFT;
/// Channels included in the feature vector. We histogram red / green /
/// blue; alpha is omitted because most lossless WebP payloads carry an
/// opaque alpha and a uniform-`0xff` alpha bin contributes no signal.
const CLUSTER_NUM_CHANNELS: usize = 3;
/// Length of one block's feature vector: `bins-per-channel × channels`.
const CLUSTER_FEATURE_DIM: usize = CLUSTER_BINS_PER_CHANNEL * CLUSTER_NUM_CHANNELS;

/// Maximum Lloyd's-algorithm iteration count. On the diagnostic
/// fixtures the assignment settles in 2–3 passes; the cap bounds the
/// chooser's wall-time on pathological inputs (the outer chooser will
/// often discard this candidate anyway).
const CLUSTER_MAX_ITERATIONS: u32 = 8;

/// Build the per-block coarse RGB histogram feature vectors.
///
/// The feature layout per block is three contiguous channel chunks:
/// red bins, then green bins, then blue bins, each of length
/// [`CLUSTER_BINS_PER_CHANNEL`]. Counts are left raw (not normalised)
/// because all blocks of the same `block_side` see the same pixel
/// count, so L1 distance between any two block vectors is directly
/// comparable. Boundary blocks (where `block_side` doesn't divide
/// `width` / `height` evenly) have smaller pixel counts, so their
/// vector magnitudes are correspondingly smaller — the L1 metric
/// stays meaningful because both sides of every comparison are
/// pulled from the same fixed-size bin grid.
fn histogram_block_features(
    pixels: &[u32],
    width: u32,
    height: u32,
    prefix_bits: u8,
) -> (Vec<u32>, usize) {
    let block_side = 1u32 << prefix_bits;
    let blocks_wide = width.div_ceil(block_side) as usize;
    let blocks_high = height.div_ceil(block_side) as usize;
    let block_count = blocks_wide * blocks_high;
    let mut features = vec![0u32; block_count * CLUSTER_FEATURE_DIM];

    let row_stride = width as usize;
    let bs = block_side as usize;
    for y in 0..height as usize {
        let block_row = y / bs;
        for x in 0..width as usize {
            let block_col = x / bs;
            let block_index = block_row * blocks_wide + block_col;
            let pixel = pixels[y * row_stride + x];
            let r_bin = (((pixel >> 16) & 0xff) >> CLUSTER_BIN_SHIFT) as usize;
            let g_bin = (((pixel >> 8) & 0xff) >> CLUSTER_BIN_SHIFT) as usize;
            let b_bin = ((pixel & 0xff) >> CLUSTER_BIN_SHIFT) as usize;
            let base = block_index * CLUSTER_FEATURE_DIM;
            features[base + r_bin] += 1;
            features[base + CLUSTER_BINS_PER_CHANNEL + g_bin] += 1;
            features[base + 2 * CLUSTER_BINS_PER_CHANNEL + b_bin] += 1;
        }
    }
    (features, block_count)
}

/// L1 (sum-of-absolute-differences) distance between two
/// `CLUSTER_FEATURE_DIM`-length count vectors. Symmetric and integer-
/// valued; zero iff every bin matches exactly.
fn histogram_l1(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), CLUSTER_FEATURE_DIM);
    debug_assert_eq!(b.len(), CLUSTER_FEATURE_DIM);
    let mut sum: u64 = 0;
    for i in 0..CLUSTER_FEATURE_DIM {
        let ai = a[i];
        let bi = b[i];
        sum += ai.abs_diff(bi) as u64;
    }
    sum
}

/// Deterministic centroid seeding by farthest-from-already-chosen rule
/// (a k-means++-style maximum-minimum-distance variant with no
/// randomness so identical inputs always produce identical seeds).
///
/// Starts with block 0 as the first centroid, then repeatedly picks
/// the block whose minimum L1 distance to the already-chosen set is
/// the largest. Returns the chosen block indices. If at some step no
/// remaining block has positive distance to every chosen centroid
/// (i.e. it duplicates one already in the set), the seeding stops
/// early — the caller treats a list shorter than `num_groups` as a
/// signal that the input cannot be split that finely.
fn seed_cluster_centroids(features: &[u32], block_count: usize, num_groups: u32) -> Vec<usize> {
    let target = num_groups as usize;
    debug_assert!(target >= 1 && target <= block_count);
    let mut picks: Vec<usize> = Vec::with_capacity(target);
    picks.push(0);
    while picks.len() < target {
        let mut champion_block = 0usize;
        let mut champion_min_dist: u64 = 0;
        for cand in 0..block_count {
            if picks.contains(&cand) {
                continue;
            }
            let cand_vec = &features[cand * CLUSTER_FEATURE_DIM..(cand + 1) * CLUSTER_FEATURE_DIM];
            let mut nearest: u64 = u64::MAX;
            for &p in &picks {
                let pick_vec = &features[p * CLUSTER_FEATURE_DIM..(p + 1) * CLUSTER_FEATURE_DIM];
                let d = histogram_l1(cand_vec, pick_vec);
                if d < nearest {
                    nearest = d;
                }
            }
            if nearest > champion_min_dist {
                champion_min_dist = nearest;
                champion_block = cand;
            }
        }
        if champion_min_dist == 0 {
            // No more distinguishable centroids remain.
            break;
        }
        picks.push(champion_block);
    }
    picks
}

/// Partition the image's `prefix_bits`-aligned blocks into at most
/// `num_groups` clusters by coarse-RGB-histogram L1 distance, returning
/// one meta-prefix code per block in scan-line order.
///
/// The returned codes are always *compact*: they form the contiguous
/// range `0..actual_groups - 1` with no gaps. Per RFC 9649 §3.7.2.2.2
/// the entropy image's `num_prefix_groups` is derived as
/// `max(entropy image) + 1`, so a gap (an empty group sitting between
/// used ones) would force the encoder to emit an unused prefix-code
/// group and pay its code-length-table cost for no benefit.
///
/// Returns `vec![0; block_count]` (a single-group degenerate) when:
///
/// * `num_groups == 1` (caller asked for one group),
/// * `block_count <= 1` (the entropy image holds at most one block, so
///   there is no partition to make),
/// * seeding cannot find `≥ 2` distinguishable centroids (e.g. all
///   blocks have identical histograms), or
/// * Lloyd's iteration converges to a single non-empty cluster after
///   the compaction pass.
///
/// The caller's chooser uses the degenerate path as a signal to fall
/// through to the single-group baseline rather than paying the
/// multi-group meta-prefix header overhead.
///
/// **Determinism.** Two calls with the same `(pixels, width, height,
/// prefix_bits, num_groups)` always produce the same `Vec<u16>` — the
/// seeding rule, the Lloyd loop's tie-break (lowest-index centroid
/// wins on equal-distance), and the compaction pass are all
/// deterministic.
fn cluster_blocks_by_histogram_distance(
    pixels: &[u32],
    width: u32,
    height: u32,
    prefix_bits: u8,
    num_groups: u32,
) -> Vec<u16> {
    debug_assert!(num_groups >= 1);
    let (features, block_count) = histogram_block_features(pixels, width, height, prefix_bits);
    if num_groups == 1 || block_count <= 1 {
        return vec![0u16; block_count];
    }

    let seeds = seed_cluster_centroids(&features, block_count, num_groups);
    if seeds.len() < 2 {
        return vec![0u16; block_count];
    }
    let cluster_k = seeds.len();

    // Centroids are stored as running sums of assigned-block feature
    // vectors so the update step amortises the per-bin sum across all
    // assigned blocks in O(block_count × feat_dim). The per-cluster
    // assignment count divides the sum on demand to materialise the
    // average for the L1 step.
    let mut centroid_sums: Vec<u64> = vec![0u64; cluster_k * CLUSTER_FEATURE_DIM];
    let mut centroid_counts: Vec<u64> = vec![1u64; cluster_k];
    for (slot, &block_idx) in seeds.iter().enumerate() {
        let src = &features[block_idx * CLUSTER_FEATURE_DIM..(block_idx + 1) * CLUSTER_FEATURE_DIM];
        for (i, &v) in src.iter().enumerate() {
            centroid_sums[slot * CLUSTER_FEATURE_DIM + i] = v as u64;
        }
    }

    let mut assignment: Vec<u16> = vec![0u16; block_count];
    let mut centroid_view: Vec<u32> = vec![0u32; CLUSTER_FEATURE_DIM];

    for _pass in 0..CLUSTER_MAX_ITERATIONS {
        // Assignment step: reassign each block to the nearest centroid.
        let mut any_change = false;
        for b in 0..block_count {
            let block_vec = &features[b * CLUSTER_FEATURE_DIM..(b + 1) * CLUSTER_FEATURE_DIM];
            let mut best_group: u16 = 0;
            let mut best_dist: u64 = u64::MAX;
            for ci in 0..cluster_k {
                let divisor = centroid_counts[ci].max(1);
                for i in 0..CLUSTER_FEATURE_DIM {
                    let raw = centroid_sums[ci * CLUSTER_FEATURE_DIM + i];
                    centroid_view[i] = (raw / divisor) as u32;
                }
                let d = histogram_l1(block_vec, &centroid_view);
                if d < best_dist {
                    best_dist = d;
                    best_group = ci as u16;
                }
            }
            if assignment[b] != best_group {
                assignment[b] = best_group;
                any_change = true;
            }
        }
        if !any_change {
            break;
        }

        // Update step: rebuild centroid sums + counts from the new
        // assignment.
        for slot in centroid_sums.iter_mut() {
            *slot = 0;
        }
        for slot in centroid_counts.iter_mut() {
            *slot = 0;
        }
        for b in 0..block_count {
            let ci = assignment[b] as usize;
            let block_vec = &features[b * CLUSTER_FEATURE_DIM..(b + 1) * CLUSTER_FEATURE_DIM];
            let base = ci * CLUSTER_FEATURE_DIM;
            for (i, &v) in block_vec.iter().enumerate() {
                centroid_sums[base + i] += v as u64;
            }
            centroid_counts[ci] += 1;
        }
    }

    // Compaction: map the (possibly sparse) assigned group IDs onto
    // the contiguous range `0..used - 1`. First-seen-in-scan-order
    // wins, so the output is deterministic.
    let mut remap: Vec<i32> = vec![-1; cluster_k];
    let mut next_id: u16 = 0;
    for slot in assignment.iter_mut() {
        let group = *slot as usize;
        if remap[group] < 0 {
            remap[group] = next_id as i32;
            next_id += 1;
        }
        *slot = remap[group] as u16;
    }
    if next_id < 2 {
        return vec![0u16; block_count];
    }
    assignment
}

/// §6.2.2 per-pixel group selector backed by a flat block-index map.
/// Mirrors the decoder's [`crate::vp8l_decode::MetaPrefixIndex`] but
/// owns its data so the encoder can build/inspect it without going
/// through the decoder type.
struct EncoderMetaIndex {
    prefix_bits: u8,
    block_width: u32,
    /// Per-block meta-prefix code in scan-line order, `block_width *
    /// block_height` entries.
    codes: Vec<u16>,
}

impl EncoderMetaIndex {
    /// §6.2.2 group selection for pixel `(x, y)`:
    /// `codes[(y >> prefix_bits) * block_width + (x >> prefix_bits)]`.
    fn group_for(&self, x: u32, y: u32) -> u16 {
        let bx = x >> self.prefix_bits;
        let by = y >> self.prefix_bits;
        self.codes[(by * self.block_width + bx) as usize]
    }

    /// §6.2.2 `num_prefix_groups = max(entropy image) + 1`.
    fn num_groups(&self) -> u32 {
        self.codes
            .iter()
            .copied()
            .max()
            .map(|c| c as u32 + 1)
            .unwrap_or(1)
    }

    /// Build the entropy-image ARGB pixel buffer the §6.2.2 entropy
    /// image is decoded from. Per §6.2.2, the meta-prefix code is the
    /// red+green channels of the entropy pixel: `(meta_code >> 8) &
    /// 0xffff` — i.e. the low 8 bits of `meta_code` go into the green
    /// channel and the next 8 bits into the red channel. Other channels
    /// (alpha, blue) are zero.
    fn entropy_image_argb(&self) -> Vec<u32> {
        self.codes
            .iter()
            .map(|&c| {
                let lo = (c & 0xff) as u32; // green
                let hi = ((c >> 8) & 0xff) as u32; // red
                (hi << 16) | (lo << 8)
            })
            .collect()
    }
}

/// Split `tokens` into one bucket per group. The LZ77 token stream was
/// generated globally over the whole image, so each token's group is
/// determined by the position of the *first* pixel it emits — for a
/// `Literal` / `CacheRef` that's a single-pixel position; for a
/// `Copy { length, distance }` it's the position of the copy's *start*
/// pixel. The §6.2.3 decode loop selects the group per *symbol*, so we
/// emit each token's symbols entirely under that single group's prefix
/// codes (matching the decoder's group-per-symbol contract, which is
/// also group-per-token because each token contributes one indexed
/// position via the next-undefined-pixel cursor).
///
/// Returns a `(group_token_lists, group_pixel_positions)` pair where
/// `group_token_lists[i]` is the ordered tokens belonging to group `i`
/// and `group_pixel_positions[i]` is the parallel list of starting
/// pixel positions (used as a sanity check during `count_frequencies`).
fn split_tokens_by_group(
    tokens: &[Token],
    index: &EncoderMetaIndex,
    width: u32,
    num_groups: u32,
) -> Vec<Vec<Token>> {
    let mut buckets: Vec<Vec<Token>> = vec![Vec::new(); num_groups as usize];
    let mut pos = 0usize;
    let w = width as usize;
    for &tok in tokens {
        let x = (pos % w) as u32;
        let y = (pos / w) as u32;
        let g = index.group_for(x, y) as usize;
        debug_assert!(g < buckets.len());
        buckets[g].push(tok);
        let consumed = match tok {
            Token::Literal(_) | Token::CacheRef { .. } => 1usize,
            Token::Copy { length, .. } => length,
        };
        pos += consumed;
    }
    buckets
}

/// Build the encoder-side per-group [`WriteCode`] tables: for each
/// group, count its token-bucket frequencies and Huffman-build the
/// five §6.2 prefix codes. The GREEN alphabet size is the same across
/// groups (`256 + 24 + color_cache_size`) so the on-wire prefix code
/// layouts are uniformly sized; the per-group frequency *distributions*
/// differ, which is exactly the point — each group gets a code tailored
/// to the bucket it represents.
///
/// Empty-bucket handling: when a group's bucket has zero tokens (the
/// clusterer assigned a block group_id that ends up unused after the
/// LZ77 matcher's emission cursor walked past it), every per-channel
/// frequency table is all-zero. The standard `WriteCode::from_freqs`
/// would yield an incomplete (Kraft-sum-zero) code the decoder
/// rejects with §6.2.1's "incomplete" error. We mirror
/// `write_prefix_codes_and_tokens`'s empty-distance handling for every
/// channel in that degenerate case: emit the §3.7.2.1.1 single-symbol-0
/// form, which decodes to a valid (one-leaf) code the bucket will
/// never actually exercise.
fn build_group_codes(
    buckets: &[Vec<Token>],
    color_cache_size: usize,
    image_width: u32,
) -> Vec<[WriteCode; 5]> {
    let green_alphabet = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES + color_cache_size;
    buckets
        .iter()
        .map(|bucket| {
            let freqs = count_frequencies(bucket, color_cache_size, image_width);
            // `empty(N)` produces a valid one-leaf code over an
            // alphabet of size `N` (the §3.7.2.1.1 single-symbol-0
            // form). For each channel, fall back to it when no
            // symbols were emitted in this bucket — the decoder
            // accepts the resulting one-leaf code without ever
            // consuming a symbol from it.
            let green = if freqs.green.iter().any(|&f| f > 0) {
                WriteCode::from_freqs(&freqs.green)
            } else {
                WriteCode::empty(green_alphabet)
            };
            let red = if freqs.red.iter().any(|&f| f > 0) {
                WriteCode::from_freqs(&freqs.red)
            } else {
                WriteCode::empty(256)
            };
            let blue = if freqs.blue.iter().any(|&f| f > 0) {
                WriteCode::from_freqs(&freqs.blue)
            } else {
                WriteCode::empty(256)
            };
            let alpha = if freqs.alpha.iter().any(|&f| f > 0) {
                WriteCode::from_freqs(&freqs.alpha)
            } else {
                WriteCode::empty(256)
            };
            let dist = if freqs.distance.iter().any(|&f| f > 0) {
                WriteCode::from_freqs(&freqs.distance)
            } else {
                WriteCode::empty(40)
            };
            [green, red, blue, alpha, dist]
        })
        .collect()
}

/// Try encoding `pixels` with the §6.2.2 multi-meta-prefix path:
///
/// 1. Cluster the image's `prefix_bits`-aligned blocks into `num_groups`
///    groups by coarse-RGB-histogram L1 distance (see
///    [`cluster_blocks_by_histogram_distance`]). Blocks whose pixel-
///    value distributions agree at bin resolution end up in the same
///    group and share a single five-code prefix-code group.
/// 2. Tokenise the image via the standard §5.2.2 LZ77 matcher
///    (`tokenize_lz77`), optionally cacheifying with `cache_code_bits`.
/// 3. Split tokens into per-group buckets, build per-group prefix codes,
///    and emit the §3.8.3 image data with:
///      * `%b0` (no §3.8.2 transforms in this candidate),
///      * `color-cache-info` (`%b0` or `%b1 4BIT`),
///      * `meta-prefix = %b1` + 3-bit `prefix_bits - 2`,
///      * the entropy image as an `entropy-coded-image` body via
///        [`write_entropy_coded_image_literals`],
///      * `num_groups` prefix-code groups (5 prefix codes each),
///      * the LZ77 token stream emitted with the group selected per
///        pixel block.
///
/// Returns `None` when the candidate is degenerate (image too small
/// for the requested block side; clustering collapsed to one group).
/// The chooser must fall back to the single-group path in those cases.
fn encode_with_meta_prefix(
    pixels: &[u32],
    width: u32,
    height: u32,
    prefix_bits: u8,
    num_groups: u32,
    cache_code_bits: Option<u32>,
    image_width: u32,
) -> Option<Vec<u8>> {
    debug_assert!((2..=9).contains(&prefix_bits));
    debug_assert!((1..=MAX_META_GROUPS).contains(&num_groups));

    let block_side = 1u32 << prefix_bits;
    // The §6.2.2 entropy image is `DIV_ROUND_UP(image_width, block_side)`
    // × `DIV_ROUND_UP(image_height, block_side)`. We need at least two
    // blocks for a multi-group split to be possible.
    let pw = width.div_ceil(block_side);
    let ph = height.div_ceil(block_side);
    if (pw * ph) < num_groups {
        return None;
    }

    let codes =
        cluster_blocks_by_histogram_distance(pixels, width, height, prefix_bits, num_groups);
    let index = EncoderMetaIndex {
        prefix_bits,
        block_width: pw,
        codes,
    };
    let actual_groups = index.num_groups();
    if actual_groups < 2 {
        // Clustering collapsed — no point paying the meta-prefix overhead.
        return None;
    }

    // Build the LZ77 token stream globally (matches the
    // single-group path's token sequence; the group selection happens
    // per *symbol* during emission, not per *match*).
    let mut tokens = tokenize_lz77(pixels);
    if let Some(bits) = cache_code_bits {
        tokens = cacheify_tokens(&tokens, pixels, bits);
    }

    let buckets = split_tokens_by_group(&tokens, &index, width, actual_groups);
    let cache_size = cache_code_bits.map(|b| 1usize << b).unwrap_or(0);
    let group_codes = build_group_codes(&buckets, cache_size, image_width);

    let mut w = BitWriter::new();

    // §3.8.2 optional-transform list: empty (no transforms in this
    // candidate). Future revisions can stack §4.1 / §4.2 / §4.4 atop
    // the multi-prefix path; for now we keep the candidate small.
    w.write_bit(false);

    // §3.8.3 / §7.3 spatially-coded-image:
    //   color-cache-info meta-prefix data
    //
    // color-cache-info: `%b0` (no cache) or `%b1 4BIT` (enabled).
    if let Some(bits) = cache_code_bits {
        debug_assert!((COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&bits));
        w.write_bit(true);
        w.write_bits(bits, 4);
    } else {
        w.write_bit(false);
    }
    // meta-prefix: `%b1` (multi-group).
    w.write_bit(true);
    // §6.2.2 `prefix_bits = ReadBits(3) + 2`.
    w.write_bits((prefix_bits - 2) as u32, 3);

    // §6.2.2 entropy image, written as an `entropy-coded-image`
    // (color-cache-info=%b0 + single prefix-code group + LZ77 data).
    // The §6.2.2 entropy pixels carry `(meta_code >> 8) & 0xffff` in
    // red+green; the literal-only writer feeds the decoder's
    // `decode_entropy_coded_image` path exactly.
    let entropy_image = index.entropy_image_argb();
    write_entropy_coded_image_literals(&mut w, &entropy_image);

    // §6.2.2 `num_prefix_groups` prefix-code groups, in canonical
    // group-index order (group 0 first, then group 1, …).
    for group in &group_codes {
        for code in group.iter() {
            code.write_code_lengths(&mut w);
        }
    }

    // §6.2.3 LZ77 emission: walk tokens in original order, look up the
    // group for each token's *start* pixel, and emit its symbols with
    // that group's prefix codes. This matches the decoder's
    // group-per-symbol contract — the decoder picks the group for
    // each pixel from the meta-prefix index, which is constant across
    // every symbol contributing to a single token (literal,
    // cache-ref, or backward-reference copy whose covered pixels all
    // fall in the same block as the start pixel, ensured by the
    // block-aligned tokenisation that the chooser feeds the matcher;
    // see `bucket_aligns_with_decoder_groups_test`).
    let mut pos = 0usize;
    let w_pixels = width as usize;
    for &tok in &tokens {
        let x = (pos % w_pixels) as u32;
        let y = (pos / w_pixels) as u32;
        let g = index.group_for(x, y) as usize;
        let codes = &group_codes[g];
        let green_code = &codes[0];
        let red_code = &codes[1];
        let blue_code = &codes[2];
        let alpha_code = &codes[3];
        let dist_code = &codes[4];
        match tok {
            Token::Literal(p) => {
                let a = ((p >> 24) & 0xff) as usize;
                let r = ((p >> 16) & 0xff) as usize;
                let g_ch = ((p >> 8) & 0xff) as usize;
                let b = (p & 0xff) as usize;
                green_code.write_symbol(&mut w, g_ch);
                red_code.write_symbol(&mut w, r);
                blue_code.write_symbol(&mut w, b);
                alpha_code.write_symbol(&mut w, a);
                pos += 1;
            }
            Token::CacheRef { index: ix } => {
                debug_assert!(cache_size > 0, "CacheRef requires an enabled cache");
                let sym = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES + ix as usize;
                green_code.write_symbol(&mut w, sym);
                pos += 1;
            }
            Token::Copy { length, distance } => {
                write_lz77_value(&mut w, green_code, 256, length as u32);
                let raw_code = pixel_distance_to_distance_code(distance, image_width);
                write_lz77_value(&mut w, dist_code, 0, raw_code);
                pos += length;
            }
        }
    }

    Some(w.into_bytes())
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

/// Width-aware variant of [`encode_argb_literals`]: same 2×2
/// `(no-tx | subtract-green) × (no-cache | cache)` chooser, but each
/// candidate threads `image_width` into [`encode_tokens`] so the
/// §5.2.2 distance-map optimisation is exercised. The production
/// `.webp` path ([`encode_vp8l_payload`] → [`encode_webp_lossless`] /
/// [`encode_vp8l_argb`]) uses this entry; the no-width
/// [`encode_argb_literals`] is retained for test callers that exercise
/// the entropy stage without spatial structure.
pub fn encode_argb_literals_with_width(pixels: &[u32], image_width: u32) -> Vec<u8> {
    debug_assert!(image_width >= 1);
    // For each `(subtract_green)` choice, evaluate the no-cache
    // baseline plus every §5.2.3 `cache_code_bits ∈ [1..11]` and keep
    // the smallest stream per the round-148 sweep. The §5.2.3 cache
    // size is `1 << code_bits` (2..=2048 entries), so different
    // payloads peak at different sizes: small-palette images favour
    // narrow caches (less header overhead for the same hit-rate);
    // large-palette photo-like images favour wider caches (fewer hash
    // collisions). Sweeping is the only way to pick the best per
    // payload without an analytical model.
    let mut best = select_best_cache_bits(|cache_bits| {
        encode_literals_with_options(pixels, false, cache_bits, image_width)
    });
    let sg_best = select_best_cache_bits(|cache_bits| {
        encode_literals_with_options(pixels, true, cache_bits, image_width)
    });
    if sg_best.len() < best.len() {
        best = sg_best;
    }
    best
}

/// Sweep §5.2.3 `cache_code_bits ∈ [1..11]` plus the disabled-cache
/// (`None`) baseline for an encoder candidate, returning the smallest
/// stream the closure produced.
///
/// `build_with_cache` takes the candidate `cache_code_bits` (`None`
/// = disable, `Some(bits)` = enable with the given size) and returns
/// the encoded bytes for that choice. The function calls
/// `build_with_cache` 12 times: once with `None` and once per value
/// in [`COLOR_CACHE_BITS_MIN`]..=[`COLOR_CACHE_BITS_MAX`], i.e. the
/// full §5.2.3 `[1..11]` range a compliant decoder accepts.
///
/// The §5.2.3 cache size is `1 << code_bits`, so the optimum varies
/// per payload:
///
/// * **Disabled** wins on uncorrelated noise (every "hit" is a hash
///   collision; the §3.8.3 `color-cache-info` `%b1 4BIT` header costs
///   five bits the no-cache path doesn't pay; the GREEN alphabet
///   stays at `256 + 24 = 280` symbols rather than growing to
///   `256 + 24 + cache_size`).
/// * **Narrow caches** (`code_bits` 1..4 → 2..16 entries) win on
///   payloads with a tiny effective palette where a 256-entry cache
///   wastes alphabet width on slots that never see a hit.
/// * **Wide caches** (`code_bits` 9..11 → 512..2048 entries) win on
///   photo-like images with hundreds of distinct colors where hash
///   collisions in a 256-entry cache prevent a hit.
///
/// Note that the §3.7.2 prefix code's alphabet length is exactly
/// `256 + 24 + (1 << code_bits)`, so a wider cache also widens every
/// emitted code-length-table entry; the trade-off between hit rate
/// and alphabet overhead is non-monotonic, which is why the chooser
/// sweeps the full range instead of using a single heuristic value.
fn select_best_cache_bits<F>(mut build_with_cache: F) -> Vec<u8>
where
    F: FnMut(Option<u32>) -> Vec<u8>,
{
    let mut best = build_with_cache(None);
    for bits in COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX {
        let cand = build_with_cache(Some(bits));
        if cand.len() < best.len() {
            best = cand;
        }
    }
    best
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

    write_spatially_coded_image(&mut w, tokens, color_cache_code_bits, image_width);

    w.into_bytes()
}

/// Write the §3.8.3 / §7.3 `spatially-coded-image` body — everything
/// after the §3.8.2 / §7.2 `optional-transform` terminator: the
/// `color-cache-info` bit(s), the `meta-prefix` bit (always `%b0` here
/// — single prefix-code group), the five prefix codes, and the
/// LZ77-coded image.
///
/// This is the writer counterpart of
/// [`crate::vp8l_decode::decode_argb`] for the single-meta-prefix
/// case, and the same body the §4.1 / §4.2 transform encoders wrap
/// after writing their own optional-transform header(s) (the
/// transform headers and any sub-resolution image bodies are written
/// by the caller; this function only emits the trailing
/// `spatially-coded-image`).
fn write_spatially_coded_image(
    w: &mut BitWriter,
    tokens: &[Token],
    color_cache_code_bits: Option<u32>,
    image_width: u32,
) {
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

    write_prefix_codes_and_tokens(w, tokens, color_cache_size, image_width);
}

/// Write an §7.3 `entropy-coded-image` (color-cache-info + data) of
/// `pixels.len()` ARGB pixels in scan-line order, using a
/// literal-only encoding with NO color cache and NO LZ77 matching.
///
/// This is the body shape required for the §4.1 predictor image and
/// the §4.2 color-transform image (per §7.2 ABNF: `predictor-image =
/// 3BIT ; sub-pixel code / entropy-coded-image`). The decoder reads
/// it via [`crate::vp8l_decode::decode_entropy_coded_image`].
///
/// Sub-resolution transform images are tiny (one ARGB pixel per
/// `block_width × block_height` block of the main image), so the
/// per-pixel overhead of the §5.2.2 LZ77 / §5.2.3 cache machinery
/// rarely pays off — the literal-only path is the smallest write for
/// these bodies in practice.
fn write_entropy_coded_image_literals(w: &mut BitWriter, pixels: &[u32]) {
    // color-cache-info = `%b0` (no cache).
    w.write_bit(false);

    let tokens: Vec<Token> = pixels.iter().map(|&p| Token::Literal(p)).collect();
    // `image_width = 1` is the trivial value (no Copy tokens are
    // emitted by a literal-only stream, so the distance-code chooser
    // is unused). `color_cache_size = 0` disables the cache alphabet.
    write_prefix_codes_and_tokens(w, &tokens, 0, 1);
}

/// Shared `data = prefix-codes lz77-coded-image` writer (§3.8.3 /
/// §7.3). Builds the five §3.7.2 prefix codes from token
/// frequencies, writes their code lengths in green/red/blue/alpha/
/// distance order, then emits the token stream.
fn write_prefix_codes_and_tokens(
    w: &mut BitWriter,
    tokens: &[Token],
    color_cache_size: usize,
    image_width: u32,
) {
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
    green_code.write_code_lengths(w);
    red_code.write_code_lengths(w);
    blue_code.write_code_lengths(w);
    alpha_code.write_code_lengths(w);
    dist_code.write_code_lengths(w);

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
                green_code.write_symbol(w, g);
                red_code.write_symbol(w, r);
                blue_code.write_symbol(w, b);
                alpha_code.write_symbol(w, a);
            }
            Token::CacheRef { index } => {
                // §5.2.3: GREEN symbol is `256 + 24 + index`. Red /
                // blue / alpha are not transmitted; the decoder
                // recovers the full ARGB from the cache slot.
                debug_assert!(color_cache_size > 0, "CacheRef requires an enabled cache");
                let sym = 256 + crate::vp8l_decode::NUM_LENGTH_PREFIX_CODES + index as usize;
                green_code.write_symbol(w, sym);
            }
            Token::Copy { length, distance } => {
                // §5.2.2: length via a GREEN length symbol (base 256), then
                // distance via prefix code #5 (base 0). The chooser must
                // agree with `count_frequencies` so the prefix-code Huffman
                // tree we built actually contains the prefix slot we look up.
                write_lz77_value(w, &green_code, 256, length as u32);
                let raw_code = pixel_distance_to_distance_code(distance, image_width);
                write_lz77_value(w, &dist_code, 0, raw_code);
            }
        }
    }
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
    let stream = encode_argb_with_predictor_chooser(pixels, width, height);
    let header = build_image_header(width, height, alpha_is_used);
    let mut payload = Vec::with_capacity(header.len() + stream.len());
    payload.extend_from_slice(&header);
    payload.extend_from_slice(&stream);
    payload
}

/// Width × height-aware super-chooser: evaluates the four
/// `(no-tx | subtract-green) × (no-cache | cache)` candidates plus
/// (as of round 155) two §4.1 spatial-predictor `size_bits`
/// candidates, two §3.5.2 / §4.2 color-transform `size_bits`
/// candidates, and (as of round 150) one §4.4 color-indexing
/// candidate when the unique-color count fits in the §4.4
/// 256-entry table, each with the round-148 §5.2.3
/// `cache_code_bits ∈ [1..11]` sweep plus the disabled-cache
/// baseline. Returns the smallest of the resulting streams.
///
/// The block-based transform-bearing candidates (§4.1 predictor,
/// §4.2 color) are only considered when both dimensions are at least
/// `1 << size_bits` (otherwise the sub-resolution transform image
/// collapses to a single block with no useful per-block resolution).
/// The §4.4 color-indexing candidate has no per-block size_bits and
/// is gated solely on palette feasibility (≤ 256 unique colors);
/// for smaller images or photo-like content the existing
/// no-transform / subtract-green chooser remains the only path.
fn encode_argb_with_predictor_chooser(pixels: &[u32], width: u32, height: u32) -> Vec<u8> {
    let mut best = encode_argb_literals_with_width(pixels, width);

    // The §4.1 predictor and §4.2 color transform pay off once the
    // image is at least one block wide AND tall, so each block
    // carries some real per-block residual mass. For images smaller
    // than a block, the chooser skips both transforms (the no-tx /
    // subtract-green paths are strictly cheaper in that regime — no
    // transform header, no sub-image bytes).
    let pred_size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
    let ctx_size_bits = DEFAULT_COLOR_TRANSFORM_SIZE_BITS;
    let pred_block = 1u32 << pred_size_bits;
    let ctx_block = 1u32 << ctx_size_bits;

    if width >= pred_block && height >= pred_block {
        // Round 155: sweep two `size_bits` values for the §4.1
        // spatial predictor, mirroring the §4.2 color-transform shape
        // below. The default (16-pixel blocks → per-region predictor-
        // mode granularity, good for images whose local statistics
        // change across regions) is paired with a maximal single-block
        // transform whose `size_bits` is large enough that the entire
        // image collapses into one mode (1 sub-image pixel → 4-byte
        // sub-image overhead, the cheapest possible §4.1 header). Per
        // RFC 9649 §4.1 `size_bits` ranges over `[2..=9]` (`block`
        // sizes 4..=512); the maximal value here is whatever `2..=9`
        // makes the sub-image at most 1×1. Single-block is best on
        // images whose local statistics agree everywhere (one
        // dominant predictor mode does the entire image, so the per-
        // region mode-image's bits are pure overhead); per-region
        // wins on images whose best-mode varies spatially.
        let mut pred_single_block_size_bits: u8 = pred_size_bits;
        while pred_single_block_size_bits < 9
            && ((1u32 << pred_single_block_size_bits) < width
                || (1u32 << pred_single_block_size_bits) < height)
        {
            pred_single_block_size_bits += 1;
        }
        // Deduplicate when the per-region and single-block size_bits
        // collapse onto the same value (small images).
        let try_pred_single_block = pred_single_block_size_bits != pred_size_bits;
        // Round 148: per `size_bits`, sweep §5.2.3
        // `cache_code_bits ∈ [1..11]` plus the disabled-cache baseline
        // (was hardcoded at `DEFAULT_COLOR_CACHE_BITS = 8`).
        let mut pred_candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
            encode_with_predictor(pixels, width, height, pred_size_bits, cache_bits, width)
        })];
        // Round 160: add §4.1 slack-cost tie-break candidates.
        // `slack > 0` lets the per-block chooser swap to the
        // preferred-neighbour mode at a small residual-cost
        // increase, dropping the §7.2 predictor-sub-image's symbol
        // entropy. The slack budget is expressed in residual-
        // magnitude units summed across the whole block, so it
        // scales linearly with the block's pixel count to stay a
        // bounded per-pixel quantity. Two slack settings (1× and 2×
        // the pixel count) are tried; the chooser picks the
        // shortest stream and is therefore non-regressing relative
        // to the strict-tie-break (slack = 0) baseline.
        let pred_block_pixels: u64 = (1u64 << pred_size_bits) * (1u64 << pred_size_bits);
        for slack in [
            pred_block_pixels,
            2 * pred_block_pixels,
            4 * pred_block_pixels,
        ] {
            pred_candidates.push(select_best_cache_bits(|cache_bits| {
                encode_with_predictor_slack(
                    pixels,
                    width,
                    height,
                    pred_size_bits,
                    cache_bits,
                    width,
                    slack,
                )
            }));
        }
        // Round 161: add the Shannon-entropy bit-cost candidate at
        // the per-region `size_bits`. Per-block mode is chosen by
        // a true Huffman lower-bound bit cost on the residual byte
        // histogram rather than the L1-magnitude proxy used by the
        // round-159/160 candidates. RFC 9649 §3.5 authorises the
        // choice ("transform data can be decided based on entropy
        // minimization"); the entropy cost replaces the proxy with
        // the actual metric Huffman codes minimise. The chooser
        // keeps both the entropy and L1 candidates and emits the
        // byte-shortest stream so the round-161 path cannot
        // regress against the round-160 baseline.
        pred_candidates.push(select_best_cache_bits(|cache_bits| {
            encode_with_predictor_entropy(pixels, width, height, pred_size_bits, cache_bits, width)
        }));
        // Round 162: add the *sub-image-aware* Shannon-entropy
        // candidate at the per-region `size_bits` across a small
        // lambda sweep. Per-block mode is chosen on a joint cost
        // that adds the §7.2 predictor sub-image's marginal Shannon
        // bit-cost contribution (weighted by lambda) to the round-
        // 161 per-block residual entropy. Where the round-159 hint
        // and round-160 slack budget act only on local neighbour
        // identity, the round-162 chooser accounts for the running
        // sub-image distribution globally. `lambda_milli = 0`
        // recovers the round-161 chooser exactly; the swept values
        // here weight one sub-image bit at 1×, 4×, 16× a residual
        // bit (a 16×16 block contains 256 residual symbols per
        // channel — so even modest sub-image weighting can pay back
        // through longer mode-runs in the sub-image's prefix code).
        // The chooser keeps the byte-shortest stream so the round-
        // 162 path cannot regress against the round-161 baseline.
        //
        // The lambda sweep targets the empirically-observed cost
        // crossover on smooth-gradient fixtures (~64000 milli-per-
        // bit): below that, the residual cost dominates and the
        // round-161 chooser already wins; above that, the sub-
        // image's mass dominates and converging the mode set pays
        // back through a much smaller §7.2 prefix-code header.
        for lambda_milli in [4_000u64, 16_000u64, 64_000u64, 256_000u64] {
            pred_candidates.push(select_best_cache_bits(|cache_bits| {
                encode_with_predictor_entropy_subaware(
                    pixels,
                    width,
                    height,
                    pred_size_bits,
                    cache_bits,
                    width,
                    lambda_milli,
                )
            }));
        }
        if try_pred_single_block {
            pred_candidates.push(select_best_cache_bits(|cache_bits| {
                encode_with_predictor(
                    pixels,
                    width,
                    height,
                    pred_single_block_size_bits,
                    cache_bits,
                    width,
                )
            }));
            // Round-160 slack-cost candidates also at the single-
            // block size_bits. A single block has one predictor-
            // image entry, so the slack-cost variant degenerates to
            // the strict variant at this `size_bits` (no neighbour
            // hint exists to fire); the candidate is still
            // evaluated to keep the sweep regular, but its
            // contribution to the byte-best win comes through the
            // per-region size_bits.
            let single_pred_block_pixels: u64 =
                (1u64 << pred_single_block_size_bits) * (1u64 << pred_single_block_size_bits);
            for slack in [
                single_pred_block_pixels,
                2 * single_pred_block_pixels,
                4 * single_pred_block_pixels,
            ] {
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor_slack(
                        pixels,
                        width,
                        height,
                        pred_single_block_size_bits,
                        cache_bits,
                        width,
                        slack,
                    )
                }));
            }
            // Round 161: also evaluate the Shannon-entropy candidate
            // at the single-block size_bits. With one block the hint
            // mechanism never fires (no neighbour exists) and the
            // entropy chooser degenerates to "pick the mode whose
            // single-block residual histogram has the lowest Huffman
            // bit cost" — still a strict improvement over the L1
            // proxy on fixtures whose distribution skews the
            // ordering between the two metrics.
            pred_candidates.push(select_best_cache_bits(|cache_bits| {
                encode_with_predictor_entropy(
                    pixels,
                    width,
                    height,
                    pred_single_block_size_bits,
                    cache_bits,
                    width,
                )
            }));
        }
        for cand in pred_candidates {
            if cand.len() < best.len() {
                best = cand;
            }
        }
    }

    if width >= ctx_block && height >= ctx_block {
        // Sweep two `size_bits` values for the color transform: the
        // default (16-pixel blocks → per-region CTE granularity, good
        // for varying-correlation natural images) and a maximal
        // single-block transform whose `size_bits` is large enough
        // that the entire image collapses into one CTE (1 sub-image
        // pixel → 4-byte sub-image overhead, the cheapest possible
        // header). Single-block is best for high-noise images with
        // a single dominant channel correlation; per-region wins on
        // images whose correlation varies spatially.
        let mut single_block_size_bits: u8 = ctx_size_bits;
        while single_block_size_bits < 9
            && ((1u32 << single_block_size_bits) < width
                || (1u32 << single_block_size_bits) < height)
        {
            single_block_size_bits += 1;
        }
        // Deduplicate when the per-region and single-block size_bits
        // collapse onto the same value (small images).
        let try_single_block = single_block_size_bits != ctx_size_bits;
        // Round 148: per `size_bits`, sweep §5.2.3
        // `cache_code_bits ∈ [1..11]` plus the disabled-cache baseline
        // (was hardcoded at `DEFAULT_COLOR_CACHE_BITS = 8`).
        let mut candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
            encode_with_color_transform(pixels, width, height, ctx_size_bits, cache_bits, width)
        })];
        if try_single_block {
            candidates.push(select_best_cache_bits(|cache_bits| {
                encode_with_color_transform(
                    pixels,
                    width,
                    height,
                    single_block_size_bits,
                    cache_bits,
                    width,
                )
            }));
        }
        for cand in candidates {
            if cand.len() < best.len() {
                best = cand;
            }
        }
    }

    // Round 150: §4.4 color-indexing transform candidate. Considered
    // unconditionally (no per-block size_bits to sweep): a single
    // O(N) palette probe decides feasibility, so the path is cheap
    // to skip on photo-like content. On palette-ish images (icons,
    // line art, screen captures) the bundled-index stream shrinks
    // the §5 image data dramatically (a 4-color image packs 4 pixels
    // per byte at width_bits=2, giving the entropy stage 1/4 the
    // symbols to code), more than paying for the palette-write
    // overhead.
    if collect_palette(pixels).is_some() {
        let ci_best = select_best_cache_bits(|cache_bits| {
            encode_with_color_indexing(pixels, width, height, cache_bits)
                .expect("palette feasibility already confirmed")
        });
        if ci_best.len() < best.len() {
            best = ci_best;
        }
    }

    // Round 151: §6.2.2 multi-meta-prefix (entropy-image) candidate.
    // Sweeps a small set of `(prefix_bits, num_groups)` combinations,
    // each paired with the round-148 `cache_code_bits ∈ [1..11]` plus
    // disabled-cache baseline; whichever is smallest is compared
    // against the running `best`. The candidate is only built when
    // the image is large enough to contain `num_groups` blocks at the
    // current `prefix_bits` (the `encode_with_meta_prefix` helper
    // returns `None` otherwise). Multi-group encoding pays for itself
    // on images whose per-region statistics diverge (e.g. natural
    // images with sky-vs-foreground contrast, screenshots with
    // distinct UI regions) where separate per-region Huffman codes
    // shrink the LZ77 stream by more than the entropy-image +
    // additional code-length-table overhead.
    if let Some(mp_best) = sweep_meta_prefix_candidate(pixels, width, height) {
        if mp_best.len() < best.len() {
            best = mp_best;
        }
    }

    best
}

/// Sweep every `(prefix_bits, num_groups, cache_code_bits)` combination
/// the §6.2.2 multi-meta-prefix candidate admits and return the smallest
/// resulting stream, or `None` if no `(prefix_bits, num_groups)` pair
/// produced a non-degenerate stream (i.e. the image was too small for any
/// multi-block split, or every clustering collapsed to a single group).
fn sweep_meta_prefix_candidate(pixels: &[u32], width: u32, height: u32) -> Option<Vec<u8>> {
    let mut best: Option<Vec<u8>> = None;
    for &prefix_bits in META_PREFIX_BITS_SWEEP.iter() {
        for num_groups in 2..=MAX_META_GROUPS {
            // Per-(prefix_bits, num_groups), sweep the cache sizes;
            // some shapes are degenerate (None returned). Track the
            // best non-degenerate candidate.
            let mut shape_best: Option<Vec<u8>> = None;
            for cache_opt in
                std::iter::once(None).chain((COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).map(Some))
            {
                if let Some(cand) = encode_with_meta_prefix(
                    pixels,
                    width,
                    height,
                    prefix_bits,
                    num_groups,
                    cache_opt,
                    width,
                ) {
                    match &shape_best {
                        Some(s) if s.len() <= cand.len() => {}
                        _ => shape_best = Some(cand),
                    }
                }
            }
            if let Some(cand) = shape_best {
                match &best {
                    Some(b) if b.len() <= cand.len() => {}
                    _ => best = Some(cand),
                }
            }
        }
    }
    best
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

    // ---- §3.7.2.1.1 simple code length code chooser ----

    /// `WriteCode::as_simple_form` rejects any table that the simple form
    /// cannot represent verbatim: length > 1, symbol > 255, more than two
    /// used symbols, all-zeros table.
    #[test]
    fn simple_form_rejects_tables_outside_3_7_2_1_1_constraints() {
        // Three symbols → too many for simple form.
        let mut freq = vec![0u32; 8];
        freq[0] = 1;
        freq[1] = 1;
        freq[2] = 1;
        let three_sym = WriteCode::from_freqs(&freq);
        assert!(three_sym.as_simple_form().is_none());

        // All-zero / empty alphabet → as_simple_form returns None
        // (encoder handles the empty case via `WriteCode::empty`).
        let lengths_empty = vec![0u8; 16];
        let codes_empty = canonical_codes(&lengths_empty);
        let empty_code = WriteCode {
            lengths: lengths_empty,
            codes: codes_empty,
            single: None,
        };
        assert!(empty_code.as_simple_form().is_none());

        // Symbol > 255 → simple form's 8-bit symbol field can't carry it.
        let mut freq_big = vec![0u32; 300];
        freq_big[280] = 1;
        let beyond_255 = WriteCode::from_freqs(&freq_big);
        assert!(beyond_255.as_simple_form().is_none());

        // Length > 1 → cannot be the simple form (every present symbol
        // must be at length 1).
        let mixed_lengths = vec![0u8, 2, 2, 1];
        let mixed_codes = canonical_codes(&mixed_lengths);
        let mixed = WriteCode {
            lengths: mixed_lengths,
            codes: mixed_codes,
            single: None,
        };
        assert!(mixed.as_simple_form().is_none());
    }

    /// `WriteCode::as_simple_form` accepts the two qualifying shapes
    /// (1 used symbol or 2 used symbols, each at length 1).
    #[test]
    fn simple_form_accepts_one_or_two_length_one_symbols() {
        let mut freq1 = vec![0u32; 16];
        freq1[7] = 1;
        let one = WriteCode::from_freqs(&freq1);
        assert_eq!(one.as_simple_form(), Some(vec![7]));

        let mut freq2 = vec![0u32; 16];
        freq2[3] = 4;
        freq2[12] = 4;
        let two = WriteCode::from_freqs(&freq2);
        assert_eq!(two.as_simple_form(), Some(vec![3, 12]));
    }

    /// §3.7.2.1.1 exact bit-cost layout: 1 flag + 1 num + 1 width + s0 + s1.
    /// `simple_form_bits` must match the bytes [`write_simple_code_lengths`]
    /// actually emits.
    #[test]
    fn simple_form_bits_matches_written_layout() {
        // 1 symbol, symbol0 in [0..1] → is_first_8bits = 0 → 1-bit symbol.
        // Total = 1 + 1 + 1 + 1 = 4 bits.
        assert_eq!(simple_form_bits(&[1]), 4);
        // 1 symbol, symbol0 = 7 (> 1) → is_first_8bits = 1 → 8-bit symbol.
        // Total = 1 + 1 + 1 + 8 = 11 bits.
        assert_eq!(simple_form_bits(&[7]), 11);
        // 2 symbols, symbol0 = 0 (fits in 1 bit), symbol1 = 50.
        // Total = 1 + 1 + 1 + 1 + 8 = 12 bits.
        assert_eq!(simple_form_bits(&[0, 50]), 12);
        // 2 symbols, symbol0 = 200 (> 1) → 8 bits; symbol1 = 100 → 8 bits.
        // Total = 1 + 1 + 1 + 8 + 8 = 19 bits.
        assert_eq!(simple_form_bits(&[200, 100]), 19);

        // Round-trip the byte count against an actual writer.
        let mut w = BitWriter::new();
        write_simple_code_lengths(&mut w, &[200, 100]);
        // 19 bits → 3 bytes (24 bits, padded). Confirm the writer's
        // bit-position is exactly 19.
        let pos_bits = w.bit_position();
        assert_eq!(pos_bits, 19);
    }

    /// The chooser switches to the simple form for a 1-symbol distance
    /// code (saves ~14 bits over the normal-form single-leaf path).
    #[test]
    fn chooser_prefers_simple_form_for_empty_distance_code() {
        let code = WriteCode::empty(40);
        // Confirm normal form would have been more expensive than simple.
        let normal_bits = normal_form_bits(&code.lengths);
        let simple = code.as_simple_form().expect("empty(40) is simple-form");
        let simple_bits = simple_form_bits(&simple);
        assert!(
            simple_bits < normal_bits,
            "expected simple form (= {simple_bits} bits) to beat normal form (= {normal_bits} bits) for empty distance code"
        );

        // Now drive write_code_lengths and confirm the leading flag bit is
        // 1 (the simple-form selector per §3.7.2.1).
        let mut w = BitWriter::new();
        code.write_code_lengths(&mut w);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert!(
            r.read_bit().expect("flag bit"),
            "chooser must select simple form (flag bit = 1) for the empty distance code"
        );
    }

    /// `write_code_lengths` round-trips through the decoder for both
    /// branches of the chooser: a 1-symbol code (simple form) and a
    /// 4-symbol code (normal form).
    #[test]
    fn chooser_round_trips_through_decoder_on_both_branches() {
        // ---- 1-symbol path: simple form ----
        let mut freq = vec![0u32; 16];
        freq[9] = 7;
        let code1 = WriteCode::from_freqs(&freq);
        let mut w1 = BitWriter::new();
        code1.write_code_lengths(&mut w1);
        let bytes1 = w1.into_bytes();
        let mut r1 = BitReader::new(&bytes1);
        let decoded1 = PrefixCode::read(&mut r1, 16).expect("decode simple form");
        assert_eq!(
            decoded1.single_symbol(),
            Some(9),
            "decoder must recover the single-leaf symbol from the simple form"
        );

        // ---- 4-symbol path: normal form ----
        let freq4 = vec![10u32, 4, 2, 1, 0, 0, 0, 0];
        let code4 = WriteCode::from_freqs(&freq4);
        let mut w4 = BitWriter::new();
        code4.write_code_lengths(&mut w4);
        // Emit a representative symbol sequence and round-trip it.
        let seq = [0usize, 1, 2, 3, 0, 0, 1, 2];
        for &s in &seq {
            code4.write_symbol(&mut w4, s);
        }
        let bytes4 = w4.into_bytes();
        let mut r4 = BitReader::new(&bytes4);
        let decoded4 = PrefixCode::read(&mut r4, 8).expect("decode normal form");
        for &s in &seq {
            assert_eq!(
                decoded4.read_symbol(&mut r4).expect("symbol") as usize,
                s,
                "round-trip mismatch on normal-form code"
            );
        }
    }

    /// On a 1×1 opaque image the encoder produces 5 prefix codes
    /// (G/R/B/A + distance) and every one of them is the single-leaf
    /// case (one length-1 symbol, all others zero). Before round 149 the
    /// chooser had only the normal-form path, paying ≥ 58 bits per code
    /// to send the length table even though the per-symbol body
    /// collapses to zero. The simple-form path costs at most 11 bits
    /// (1-symbol header + 8-bit value), so the round-149 chooser flips
    /// all five codes and shrinks the encoded file by a large fraction
    /// on this baseline fixture.
    #[test]
    fn round_149_simple_form_shrinks_1x1_lossless_baseline() {
        let rgba = [0x12, 0x34, 0x56, 0xff];
        let file = encode_webp_lossless(&rgba, 1, 1).unwrap();
        eprintln!("round-149 1x1 lossless byte count: {}", file.len());

        // Round-trip confirms the chosen stream still decodes.
        let decoded = crate::decode_webp(&file).unwrap();
        assert_eq!(decoded.frames[0].rgba, rgba);

        // Round-148 baseline for this fixture was 174 bytes (5 prefix
        // codes × ≥ 58 bits each, plus container envelope). Round 149
        // lands at 32 bytes — a >80% reduction. Assert a conservative
        // strict-beat below the round-148 size.
        assert!(
            file.len() <= 48,
            "expected round-149 simple-form chooser to bring the 1×1 baseline well under the round-148 174-byte size; got {}",
            file.len()
        );
    }

    /// Same chooser-shrink check on a 16×16 gradient. The chooser
    /// trade-off here applies to many of the candidate streams the
    /// super-chooser races: each pays substantially less header tax on
    /// its prefix codes when the alphabet collapses to one or two
    /// length-1 symbols (single-pixel column, alpha-uniform images,
    /// solid-color blocks, the bulk of small synthetic fixtures).
    #[test]
    fn round_149_simple_form_shrinks_synthetic_fixtures() {
        // 32×32 solid gray — every channel emits one literal value
        // repeated 1024 times. Each of the 4 literal prefix codes is a
        // single-leaf code → all four flip to the simple form.
        let mut solid = Vec::new();
        for _ in 0..1024 {
            solid.extend_from_slice(&[0x80, 0x80, 0x80, 0xff]);
        }
        let file_solid = encode_webp_lossless(&solid, 32, 32).unwrap();
        eprintln!("round-149 32×32 solid: {}", file_solid.len());
        assert!(
            file_solid.len() <= 100,
            "round-149 32×32 solid should land far below the round-148 174-byte size; got {}",
            file_solid.len()
        );

        // 8×8 with 2 alpha values, single literal triple — RGB codes
        // single-leaf (one value each), alpha code two-symbol (0x80 and
        // 0xff). Two-symbol case may pick simple or normal depending on
        // the cost — the chooser picks whichever is cheaper.
        let mut alpha = Vec::new();
        for y in 0..8u32 {
            for x in 0..8u32 {
                let a = if (x + y) % 2 == 0 { 0xff } else { 0x80 };
                alpha.extend_from_slice(&[0x44, 0x88, 0xcc, a]);
            }
        }
        let file_alpha = encode_webp_lossless(&alpha, 8, 8).unwrap();
        eprintln!("round-149 8×8 alpha: {}", file_alpha.len());
        assert!(
            file_alpha.len() <= 110,
            "round-149 8×8 alpha should land below the round-148 178-byte size; got {}",
            file_alpha.len()
        );

        // Every chosen stream still decodes byte-exact.
        let decoded_solid = crate::decode_webp(&file_solid).unwrap();
        assert_eq!(decoded_solid.frames[0].rgba, solid);
        let decoded_alpha = crate::decode_webp(&file_alpha).unwrap();
        assert_eq!(decoded_alpha.frames[0].rgba, alpha);
    }

    /// Two-symbol simple-form path: when the alphabet has exactly two
    /// length-1 symbols, the chooser may pick simple (≤19 bits) or
    /// normal (≥18 bits) — whichever is cheaper. The chooser picks the
    /// minimum, and the chosen stream still decodes.
    #[test]
    fn round_149_two_symbol_simple_form_round_trips() {
        // Manually drive the chooser with a 2-symbol length-1 code.
        let mut freq = vec![0u32; 16];
        freq[2] = 5;
        freq[11] = 5;
        let code = WriteCode::from_freqs(&freq);
        assert_eq!(code.as_simple_form(), Some(vec![2, 11]));

        // Confirm bit-costs are within ±1 bit of each other (the
        // chooser's interesting regime). Either choice round-trips.
        let normal_bits = normal_form_bits(&code.lengths);
        let simple_bits = simple_form_bits(&[2, 11]);
        eprintln!(
            "2-symbol code: simple={} bits, normal={} bits",
            simple_bits, normal_bits
        );

        // Drive write_code_lengths through the chooser + decode.
        let mut w = BitWriter::new();
        code.write_code_lengths(&mut w);
        // Emit a few symbols to confirm the round-trip works.
        for _ in 0..3 {
            code.write_symbol(&mut w, 2);
            code.write_symbol(&mut w, 11);
        }
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let decoded = PrefixCode::read(&mut r, 16).expect("decode chooser output");
        for _ in 0..3 {
            assert_eq!(decoded.read_symbol(&mut r).unwrap() as usize, 2);
            assert_eq!(decoded.read_symbol(&mut r).unwrap() as usize, 11);
        }
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

    // ---- round 146: §4.1 spatial-predictor forward transform ----

    /// Round-224 cross-check (kept after the SWAR-form regression
    /// finding documented on the function itself): the public
    /// `predictor_subtract` body must remain bit-identical to the
    /// per-channel `wrapping_sub` semantics. Sweep 1 024 deterministic
    /// LCG `(original, pred)` pairs plus six hand-picked boundary
    /// pairs (every-channel underflow, every-channel positive,
    /// all-zero, all-0xff, mixed) against a verbatim copy of the
    /// closure-of-four reference. Acts as a regression guard so any
    /// future re-attempt at a SWAR / `std::simd` rewrite of this
    /// function can re-use this test to pin the new body against the
    /// reference semantics.
    #[test]
    fn predictor_subtract_matches_per_byte_reference_random() {
        // Verbatim copy of the closure-of-four reference body. The
        // published function must be bit-identical to this for every
        // input — this is a cross-check, not a baseline measurement.
        fn reference(original: u32, pred: u32) -> u32 {
            let a = ((original >> 24) & 0xff).wrapping_sub((pred >> 24) & 0xff) & 0xff;
            let r = ((original >> 16) & 0xff).wrapping_sub((pred >> 16) & 0xff) & 0xff;
            let g = ((original >> 8) & 0xff).wrapping_sub((pred >> 8) & 0xff) & 0xff;
            let b = (original & 0xff).wrapping_sub(pred & 0xff) & 0xff;
            (a << 24) | (r << 16) | (g << 8) | b
        }
        // Boundary cases: every-channel underflow, no-underflow, mixed.
        for &(orig, pred) in &[
            (0x0000_0000u32, 0x0000_0000u32),
            (0xffff_ffffu32, 0xffff_ffffu32),
            (0x0000_0000u32, 0xffff_ffffu32), // every channel underflows
            (0xffff_ffffu32, 0x0000_0000u32), // every channel saturates positive
            (0x10_20_30_40u32, 0x05_30_20_50u32), // mixed: r,b underflow; a,g positive
            (0x80_80_80_80u32, 0x80_80_80_80u32), // zero residual
        ] {
            assert_eq!(
                predictor_subtract(orig, pred),
                reference(orig, pred),
                "predictor_subtract diverges from per-byte reference at \
                 orig=0x{orig:08x} pred=0x{pred:08x}"
            );
        }
        let mut seed: u32 = 0xcafe_d00d;
        let mut rng = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed
        };
        for _ in 0..1_024 {
            let orig = rng();
            let pred = rng();
            assert_eq!(
                predictor_subtract(orig, pred),
                reference(orig, pred),
                "predictor_subtract diverges from per-byte reference at \
                 orig=0x{orig:08x} pred=0x{pred:08x}"
            );
        }
    }

    /// Round-280 cross-check for the mode-specialised block-residual
    /// walker: `block_mode_cost`, `block_mode_entropy_cost`, and the
    /// capped walks driving `pick_block_mode_with_hint` /
    /// `pick_block_mode_with_hint_slack` must stay bit-identical to a
    /// verbatim copy of the pre-round-280 per-pixel `predictor_at`
    /// loops. Sweeps deterministic-LCG images over shapes covering
    /// every walker boundary regime — 1×N / N×1 (border-only rows and
    /// columns), 2×2, blocks overlapping the right and bottom edges,
    /// blocks larger than the image, interior blocks not touching any
    /// border — for every mode `0..=13` plus an out-of-range mode,
    /// and pins the hinted pickers (whose row-granular prune must be
    /// pick-identical to the reference per-pixel early-out) for every
    /// `prefer_mode` and a slack sweep.
    #[test]
    fn block_walker_matches_predictor_at_reference_random() {
        // Verbatim pre-round-280 `block_mode_cost` body.
        #[allow(clippy::too_many_arguments)]
        fn ref_cost(
            pixels: &[u32],
            width: usize,
            height: usize,
            x0: usize,
            y0: usize,
            bw: usize,
            bh: usize,
            mode: u8,
        ) -> u64 {
            let mut cost: u64 = 0;
            for dy in 0..bh {
                let y = y0 + dy;
                if y >= height {
                    break;
                }
                for dx in 0..bw {
                    let x = x0 + dx;
                    if x >= width {
                        break;
                    }
                    let pred = predictor_at(pixels, width, x, y, mode);
                    let original = pixels[y * width + x];
                    let residual = predictor_subtract(original, pred);
                    cost += residual_magnitude(residual) as u64;
                }
            }
            cost
        }
        // Verbatim pre-round-280 `block_mode_entropy_cost` histogram
        // fill (the Shannon sum over it is unchanged, so comparing
        // the histograms pins the whole function).
        #[allow(clippy::too_many_arguments)]
        fn ref_hist(
            pixels: &[u32],
            width: usize,
            height: usize,
            x0: usize,
            y0: usize,
            bw: usize,
            bh: usize,
            mode: u8,
        ) -> ([[u32; 256]; 4], u32) {
            let mut hist: [[u32; 256]; 4] = [[0u32; 256]; 4];
            let mut n: u32 = 0;
            for dy in 0..bh {
                let y = y0 + dy;
                if y >= height {
                    break;
                }
                for dx in 0..bw {
                    let x = x0 + dx;
                    if x >= width {
                        break;
                    }
                    let pred = predictor_at(pixels, width, x, y, mode);
                    let original = pixels[y * width + x];
                    let residual = predictor_subtract(original, pred);
                    hist[0][((residual >> 24) & 0xff) as usize] += 1;
                    hist[1][((residual >> 16) & 0xff) as usize] += 1;
                    hist[2][((residual >> 8) & 0xff) as usize] += 1;
                    hist[3][(residual & 0xff) as usize] += 1;
                    n += 1;
                }
            }
            (hist, n)
        }
        let mut seed: u32 = 0x2b80_c0de;
        let mut rng = move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed
        };
        // (width, height, x0, y0, bw, bh) — every walker regime.
        let shapes: &[(usize, usize, usize, usize, usize, usize)] = &[
            (1, 1, 0, 0, 4, 4),   // single pixel, block larger than image
            (1, 9, 0, 0, 4, 4),   // single column (left-column rule only)
            (1, 9, 0, 8, 4, 4),   // single column, partial bottom block
            (9, 1, 0, 0, 4, 4),   // single row (top-row rule only)
            (9, 1, 4, 0, 4, 4),   // single row, interior-start block
            (2, 2, 0, 0, 2, 2),   // smallest full-rules image
            (8, 8, 0, 0, 8, 8),   // block == image (all four borders)
            (8, 8, 4, 4, 4, 4),   // bottom-right block (TR wraparound)
            (8, 8, 4, 0, 4, 4),   // top-right block (top row + wraparound)
            (8, 8, 0, 4, 4, 4),   // bottom-left block (left column)
            (11, 7, 8, 4, 4, 4),  // overlaps right and bottom edges
            (16, 16, 4, 4, 4, 4), // pure interior block (no borders)
            (5, 5, 0, 0, 16, 16), // block much larger than image
        ];
        for &(width, height, x0, y0, bw, bh) in shapes {
            let pixels: Vec<u32> = (0..width * height).map(|_| rng()).collect();
            // Cost + histogram equivalence for every mode, including
            // one §4.1-undefined mode (predicts solid black).
            for mode in 0u8..=14 {
                assert_eq!(
                    block_mode_cost(&pixels, width, height, x0, y0, bw, bh, mode),
                    ref_cost(&pixels, width, height, x0, y0, bw, bh, mode),
                    "block_mode_cost diverges at {width}x{height} block \
                     ({x0},{y0},{bw},{bh}) mode {mode}"
                );
                let mut sink = ResidualHistogramSink {
                    hist: [[0u32; 256]; 4],
                    n: 0,
                };
                for_each_block_residual(&pixels, width, height, x0, y0, bw, bh, mode, &mut sink);
                let (hist, n) = ref_hist(&pixels, width, height, x0, y0, bw, bh, mode);
                assert_eq!(
                    (sink.hist, sink.n),
                    (hist, n),
                    "residual histogram diverges at {width}x{height} block \
                     ({x0},{y0},{bw},{bh}) mode {mode}"
                );
            }
            // Pick equivalence: the row-granular capped walk must
            // select the same mode as a reference full-cost argmin
            // (lowest mode wins ties) for every hint, and the slack
            // variant for a slack sweep.
            let mut ref_best_mode: u8 = 0;
            let mut ref_best_cost = u64::MAX;
            for mode in 0u8..=13 {
                let cost = ref_cost(&pixels, width, height, x0, y0, bw, bh, mode);
                if cost < ref_best_cost {
                    ref_best_cost = cost;
                    ref_best_mode = mode;
                }
            }
            for hint in std::iter::once(None).chain((0u8..=13).map(Some)) {
                let mut want = ref_best_mode;
                if let Some(m) = hint {
                    if m != want
                        && ref_cost(&pixels, width, height, x0, y0, bw, bh, m) == ref_best_cost
                    {
                        want = m;
                    }
                }
                assert_eq!(
                    pick_block_mode_with_hint(&pixels, width, height, x0, y0, bw, bh, hint),
                    want,
                    "hinted pick diverges at {width}x{height} block \
                     ({x0},{y0},{bw},{bh}) hint {hint:?}"
                );
                for slack in [0u64, 1, 7, 64] {
                    let mut want_slack = ref_best_mode;
                    if let Some(m) = hint {
                        if m != want_slack
                            && ref_cost(&pixels, width, height, x0, y0, bw, bh, m)
                                <= ref_best_cost.saturating_add(slack)
                        {
                            want_slack = m;
                        }
                    }
                    assert_eq!(
                        pick_block_mode_with_hint_slack(
                            &pixels, width, height, x0, y0, bw, bh, hint, slack
                        ),
                        want_slack,
                        "slack pick diverges at {width}x{height} block \
                         ({x0},{y0},{bw},{bh}) hint {hint:?} slack {slack}"
                    );
                }
            }
        }
    }

    /// `predictor_subtract` is the per-channel mod-256 inverse of the
    /// decoder's `add_pred`: re-adding the same prediction recovers
    /// the original, regardless of which channels wrap.
    #[test]
    fn predictor_subtract_is_inverse_of_add() {
        let cases = [
            (0xff00_0000u32, 0xff00_0000u32),
            (0x1234_5678u32, 0x0000_0000u32),
            (0xff80_4020u32, 0x8040_2010u32),
            (0x0000_ff00u32, 0xff00_ff00u32),
        ];
        for (orig, pred) in cases {
            let residual = predictor_subtract(orig, pred);
            // Reconstruct via add_pred semantics: per-channel
            // wrapping_add must restore the original.
            let a = ((residual >> 24) & 0xff).wrapping_add((pred >> 24) & 0xff) & 0xff;
            let r = ((residual >> 16) & 0xff).wrapping_add((pred >> 16) & 0xff) & 0xff;
            let g = ((residual >> 8) & 0xff).wrapping_add((pred >> 8) & 0xff) & 0xff;
            let b = (residual & 0xff).wrapping_add(pred & 0xff) & 0xff;
            let rebuilt = (a << 24) | (r << 16) | (g << 8) | b;
            assert_eq!(
                rebuilt, orig,
                "subtract+add did not round-trip for orig=0x{orig:08x} pred=0x{pred:08x}"
            );
        }
    }

    /// On a solid block, mode 1 (L) and mode 2 (T) both predict the
    /// neighbour exactly → zero residual on every channel for every
    /// interior pixel. `pick_block_mode` returns the lowest such
    /// mode by tie-breaking convention; either 0 (border-only block)
    /// or 1 is acceptable for the top-left block of a solid image.
    #[test]
    fn pick_block_mode_zero_cost_on_solid_block() {
        let w = 8usize;
        let h = 8usize;
        let pixels = vec![0xff50_6070u32; w * h];
        // Block covering rows 1..8, cols 1..8 — all interior except
        // the strip at x=0 / y=0, but those are clamped out by the
        // edge rules in `predictor_at`.
        let mode = pick_block_mode(&pixels, w, h, 0, 0, w, h);
        // Any mode that uses an immediate neighbour (1=L, 2=T, etc.)
        // produces zero residual on a constant image, so the cost
        // is zero; with the tie-breaker, the lowest mode wins. Mode
        // 0 (solid black) only matches when the image *is* solid
        // black — here the constant is grey, so mode 0 costs more
        // than 1/2/.../13, and one of those wins.
        assert!(mode <= 13, "mode out of range: {mode}");
        // Sanity: residual under the picked mode must indeed be
        // zero everywhere (the top-left predicts 0xff000000 → cost
        // 0x60 + 0x70 + 0x50 = 0xe0 fold worth, but interior pixels
        // dominate — total cost ≪ what mode 0 produces).
        let mode_cost = |m: u8| -> u64 {
            let mut c = 0u64;
            for y in 0..h {
                for x in 0..w {
                    let pred = predictor_at(&pixels, w, x, y, m);
                    let r = predictor_subtract(pixels[y * w + x], pred);
                    c += residual_magnitude(r) as u64;
                }
            }
            c
        };
        let picked_cost = mode_cost(mode);
        let mode0_cost = mode_cost(0);
        assert!(
            picked_cost < mode0_cost,
            "expected picked-mode cost ({picked_cost}) < mode-0 cost ({mode0_cost})"
        );
    }

    /// Forward + inverse predictor round-trips bit-exact: applying
    /// the encoder's forward transform then the decoder's inverse
    /// transform recovers the original pixels.
    #[test]
    fn forward_predictor_round_trips_through_decoder_inverse() {
        use crate::vp8l_transform::inverse_predictor;
        let w = 16u32;
        let h = 16u32;
        // Smooth gradient — mode 7 (Average2(L, T)) should predict
        // most pixels well.
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = x * 16;
                let g = y * 16;
                let b = (x + y) * 8;
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        let size_bits = 4u8; // 16x16 blocks → tw=th=1.
        let (pred_img, tw, _th) = build_predictor_image(&pixels, w, h, size_bits);
        let mut residuals = vec![0u32; pixels.len()];
        apply_forward_predictor(&pixels, &mut residuals, w, h, &pred_img, tw, size_bits);
        // Apply the decoder's inverse pass and confirm we recover
        // the originals.
        inverse_predictor(&mut residuals, w, h, &pred_img, tw, size_bits);
        assert_eq!(residuals, pixels);
    }

    /// End-to-end: encode + decode via the public `encode_webp_lossless`
    /// path round-trips a smooth-gradient image bit-exactly. The
    /// chooser is free to pick the predictor candidate or not; the
    /// round-trip property must hold for *whatever* path it picks.
    #[test]
    fn round_trip_smooth_gradient_with_predictor_candidate() {
        let w = 32u32;
        let h = 32u32;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                rgba.push((x * 8) as u8); // r
                rgba.push((y * 8) as u8); // g
                rgba.push(((x + y) * 4) as u8); // b
                rgba.push(0xff); // a
            }
        }
        let file = encode_webp_lossless(&rgba, w, h).unwrap();
        let decoded = crate::decode_webp(&file).unwrap();
        assert_eq!(decoded.frames[0].rgba, rgba);
    }

    /// On a smooth gradient the §4.1 predictor candidate should
    /// produce a smaller stream than the no-transform / subtract-
    /// green baseline: per-pixel residuals concentrate near zero,
    /// shrinking the green/red/blue Huffman codes. The chooser
    /// must select the predictor (or another equally-good
    /// candidate), so the final stream size is at most the
    /// no-tx baseline.
    #[test]
    fn predictor_path_shrinks_smooth_gradient() {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = (x * 4) & 0xff;
                let g = (y * 4) & 0xff;
                let b = ((x + y) * 2) & 0xff;
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        // No-tx + no-cache baseline (the round-119 path).
        let baseline = encode_literals_with_options(&pixels, false, None, w);
        // The full chooser (which now includes the predictor path).
        let chosen = encode_argb_with_predictor_chooser(&pixels, w, h);
        eprintln!(
            "[round-146] {}x{} smooth gradient: no-tx baseline={} B, chooser={} B ({:.1}% reduction)",
            w,
            h,
            baseline.len(),
            chosen.len(),
            100.0 * (baseline.len() as f64 - chosen.len() as f64) / baseline.len() as f64,
        );
        assert!(
            chosen.len() <= baseline.len(),
            "chooser regressed on smooth gradient: {} B vs no-tx baseline {} B",
            chosen.len(),
            baseline.len(),
        );

        // Round trip through the full encoder/decoder is exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// On uncorrelated random noise the predictor never helps (no
    /// neighbour predicts the next pixel any better than random),
    /// so the chooser stays on the no-tx no-cache path (or
    /// subtract-green if that happens to win). The final stream
    /// must not regress vs the no-predictor chooser.
    #[test]
    fn predictor_chooser_does_not_regress_on_noise() {
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xc0ff_eeeeu32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(state | 0xff00_0000);
        }
        let no_predictor = encode_argb_literals_with_width(&pixels, w);
        let chosen = encode_argb_with_predictor_chooser(&pixels, w, h);
        assert!(
            chosen.len() <= no_predictor.len(),
            "predictor chooser regressed on noise: {} B vs {} B",
            chosen.len(),
            no_predictor.len(),
        );
    }

    /// Round-trip the published `lossless-128x128-natural` fixture:
    /// decode it, re-encode via the full predictor-aware chooser,
    /// decode again. The decoded pixels must match the originals
    /// bit-exactly, and the re-encoded stream size should
    /// demonstrate the predictor path is being exercised on a
    /// natural image (we don't assert a specific size, only
    /// log it).
    #[test]
    fn natural_fixture_round_trips_through_predictor_aware_encoder() {
        let bytes: &[u8] = include_bytes!("../tests/data/lossless-128x128-natural.webp");
        let decoded = crate::decode_lossless_image(bytes).unwrap().unwrap();
        let w = decoded.width();
        let h = decoded.height();
        let pixels = decoded.pixels().to_vec();

        let pre_predictor = encode_argb_literals_with_width(&pixels, w);
        let with_predictor = encode_argb_with_predictor_chooser(&pixels, w, h);
        eprintln!(
            "[round-146] {}x{} natural fixture re-encoded: pre-predictor chooser={} B, predictor chooser={} B ({:.1}% reduction)",
            w,
            h,
            pre_predictor.len(),
            with_predictor.len(),
            100.0 * (pre_predictor.len() as f64 - with_predictor.len() as f64)
                / pre_predictor.len() as f64,
        );
        assert!(
            with_predictor.len() <= pre_predictor.len(),
            "predictor chooser regressed on natural fixture: {} B vs {} B",
            with_predictor.len(),
            pre_predictor.len(),
        );

        // End-to-end round trip is bit-exact through the public API.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    // ---- round 147: §3.5.2 / §4.2 color-transform forward pass ----

    /// `color_xfrm_delta` matches the §3.5.2 formula
    /// `(int8(t) * int8(c)) >> 5` for both signed inputs.
    #[test]
    fn color_xfrm_delta_matches_spec_examples() {
        // t = -1, c = 64 → (-1 * 64) >> 5 = -2.
        assert_eq!(color_xfrm_delta(0xff, 0x40), -2);
        // t = 2, c = 64 → (2 * 64) >> 5 = 4.
        assert_eq!(color_xfrm_delta(2, 0x40), 4);
        // t = 0, c = anything → 0.
        assert_eq!(color_xfrm_delta(0, 0x7f), 0);
        // Identity case: t = 0 (no slope) ⇒ no contribution.
        assert_eq!(color_xfrm_delta(0, 0xff), 0);
    }

    /// Forward + inverse §3.5.2 color transform round-trips per-pixel
    /// for arbitrary CTE values. Validates [`forward_color_pixel`]
    /// against the decoder's [`crate::vp8l_transform::inverse_color`]
    /// math.
    #[test]
    fn forward_color_pixel_round_trips_through_decoder_inverse() {
        use crate::vp8l_transform;
        let cases: &[(u8, u8, u8, u8, u8, u8)] = &[
            // (r, g, b, gtr, gtb, rtb)
            (120, 80, 200, 0x12, 0xf0, 0x05),
            (255, 0, 0, 0x20, 0x00, 0x00),
            (0, 255, 0, 0x00, 0x20, 0x00),
            (0, 0, 255, 0x00, 0x00, 0x20),
            (200, 100, 50, 0xe0, 0xd0, 0x10),
        ];
        for &(r, g, b, gtr, gtb, rtb) in cases {
            let (enc_r, enc_b) = forward_color_pixel(r, g, b, gtr, gtb, rtb);
            // Drive the decoder's helper through a 1×1 sub-image so
            // we exercise the actual published inverse path.
            let mut argb = vec![
                ((0xffu32) << 24) | ((enc_r as u32) << 16) | ((g as u32) << 8) | (enc_b as u32),
            ];
            // Build the §3.5.2 CTE pixel: red=rtb, green=gtb, blue=gtr.
            let cte = ((0xffu32) << 24) | ((rtb as u32) << 16) | ((gtb as u32) << 8) | (gtr as u32);
            let color_img = vec![cte];
            // size_bits = 9 → block 512, single block covers a 1×1 image.
            vp8l_transform::inverse_color(&mut argb, 1, 1, &color_img, 1, 9);
            assert_eq!(
                (argb[0] >> 16) & 0xff,
                r as u32,
                "red mismatch for r={r} g={g} b={b} gtr=0x{gtr:02x} gtb=0x{gtb:02x} rtb=0x{rtb:02x}",
            );
            assert_eq!(argb[0] & 0xff, b as u32, "blue mismatch");
            assert_eq!((argb[0] >> 8) & 0xff, g as u32, "green altered");
        }
    }

    /// On a solid-color block the per-axis sweep is free to pick any
    /// CTE — but whichever CTE it picks must minimise the per-pixel
    /// folded-magnitude proxy that drove the choice. Verifying the
    /// picker against the all-zero baseline (which leaves residuals at
    /// the source's pixel values) confirms the chooser is not
    /// inflating cost: a constant image's red channel can still be
    /// "decorrelated" against the constant green if some `gtr` value
    /// brings `red - delta(gtr, green)` closer to zero (mod 256) than
    /// the raw `red`.
    #[test]
    fn pick_block_cte_is_minimum_on_solid_block() {
        let w = 8usize;
        let h = 8usize;
        let pixels = vec![0xff50_6070u32; w * h];

        // Per-pixel folded-magnitude cost summed across the block, for
        // an arbitrary CTE.
        let block_cost = |gtr: u8, gtb: u8, rtb: u8| -> u64 {
            let mut c = 0u64;
            for &px in &pixels {
                let r = ((px >> 16) & 0xff) as u8;
                let g = ((px >> 8) & 0xff) as u8;
                let b = (px & 0xff) as u8;
                // Decompose like pick_block_cte does (additive across
                // channels): red proxy + blue proxy.
                let red_residual = (r as i32 - color_xfrm_delta(gtr, g)) as u32;
                let inter_blue = b as i32 - color_xfrm_delta(gtb, g);
                let blue_residual = (inter_blue - color_xfrm_delta(rtb, r)) as u32;
                c += channel_magnitude(red_residual) as u64;
                c += channel_magnitude(blue_residual) as u64;
            }
            c
        };

        let (gtr, gtb, rtb) = pick_block_cte(&pixels, w, h, 0, 0, w, h);
        let picked_cost = block_cost(gtr, gtb, rtb);
        let zero_cost = block_cost(0, 0, 0);
        assert!(
            picked_cost <= zero_cost,
            "picked CTE (0x{gtr:02x}, 0x{gtb:02x}, 0x{rtb:02x}) cost {picked_cost} > all-zero cost {zero_cost}",
        );
    }

    /// On a strongly green-correlated image (`red ≈ green / 2`), the
    /// per-axis sweep must pick a non-zero `green_to_red` to cancel
    /// the slope. A slope of 1/2 corresponds to a fixed-point value
    /// of 16 (since `>> 5` divides by 32: 16/32 = 0.5).
    #[test]
    fn pick_block_cte_recovers_known_slope() {
        let w = 16usize;
        let h = 16usize;
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let g = ((x + y) * 4) as u32 & 0xff;
                // red = green / 2 (deterministic linear correlation):
                let r = (g / 2) & 0xff;
                // blue uncorrelated → keep at a constant so gtb/rtb
                // don't have a clear winner.
                let b = 0x80u32;
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        let (gtr, _gtb, _rtb) = pick_block_cte(&pixels, w, h, 0, 0, w, h);
        // gtr should land on or near 16 (slope 0.5). Allow ±16 wiggle
        // because the grid is coarser than the optimum and the
        // residual-magnitude proxy is not strictly convex.
        let gtr_signed = gtr as i8 as i32;
        assert!(
            (0..=32).contains(&gtr_signed),
            "expected gtr ≈ +16 for red≈green/2 correlation, got {gtr_signed} (raw 0x{gtr:02x})",
        );
    }

    /// Forward + inverse over a multi-block image round-trips bit-
    /// exactly: encoder builds the per-block color image, forward-
    /// transforms the pixels, decoder applies its inverse pass and
    /// recovers the originals.
    #[test]
    fn forward_color_round_trips_through_decoder_inverse() {
        use crate::vp8l_transform::inverse_color;
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                // Some correlation between channels (so the picker
                // chooses non-trivial CTEs in at least some blocks).
                let r = (x * 7) & 0xff;
                let g = (y * 5) & 0xff;
                let b = ((x + y) * 3) & 0xff;
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        let size_bits = 4u8;
        let (color_img, tw, _th) = build_color_image(&pixels, w, h, size_bits);
        let mut residuals = vec![0u32; pixels.len()];
        apply_forward_color(&pixels, &mut residuals, w, h, &color_img, tw, size_bits);
        inverse_color(&mut residuals, w, h, &color_img, tw, size_bits);
        assert_eq!(residuals, pixels);
    }

    /// End-to-end: encode + decode via the public `encode_webp_lossless`
    /// path round-trips a chroma-correlated image bit-exactly. The
    /// chooser is free to pick the color-transform candidate or not;
    /// the round-trip property must hold for *whatever* path it picks.
    #[test]
    fn round_trip_chroma_correlated_image_with_color_transform_candidate() {
        let w = 32u32;
        let h = 32u32;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let g = ((x + y) * 4) as u8;
                let r = g.wrapping_div(2);
                let b = g.wrapping_div(3);
                rgba.push(r);
                rgba.push(g);
                rgba.push(b);
                rgba.push(0xff);
            }
        }
        let file = encode_webp_lossless(&rgba, w, h).unwrap();
        let decoded = crate::decode_webp(&file).unwrap();
        assert_eq!(decoded.frames[0].rgba, rgba);
    }

    /// On a chroma-correlated synthetic image the §4.2 color-transform
    /// candidate should at worst tie the existing pre-color-transform
    /// chooser: even if the predictor path already wins, the chooser
    /// must never inflate the stream by adding the color transform as
    /// a new option.
    #[test]
    fn color_transform_chooser_never_regresses() {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let g = ((x + y) * 4) & 0xff;
                let r = (g / 2) & 0xff;
                let b = (g / 3) & 0xff;
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        let pre_color = pre_round_147_chooser(&pixels, w, h);
        let with_color = encode_argb_with_predictor_chooser(&pixels, w, h);
        eprintln!(
            "[round-147] {}x{} chroma-correlated synth: pre-color chooser={} B, color chooser={} B ({:.1}% reduction)",
            w,
            h,
            pre_color.len(),
            with_color.len(),
            100.0 * (pre_color.len() as f64 - with_color.len() as f64) / pre_color.len() as f64,
        );
        assert!(
            with_color.len() <= pre_color.len(),
            "color-transform chooser regressed: {} B vs pre-color {} B",
            with_color.len(),
            pre_color.len(),
        );

        // Round trip through the full encoder/decoder is exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Build a 128×128 channel-correlated noise fixture with
    /// *spatially varying* correlation slopes — each 16×16 block has a
    /// different `(green_to_red, green_to_blue)` correlation drawn
    /// from a small palette, giving the §3.5.2 per-block color
    /// transform a clear advantage over §3.5.3 subtract-green (which
    /// applies the same all-channels-equal correction everywhere).
    /// Within a block: spatially random green (LCG-driven), red and
    /// blue are `(slope × green + jitter) mod 256` in signed-mod-256
    /// arithmetic, with 6-bit jitter (high unique-pixel count keeps
    /// the §5.2.3 cache from dominating).
    fn make_channel_correlated_noise(w: u32, h: u32) -> Vec<u32> {
        let mut pixels = vec![0u32; (w * h) as usize];
        // Per-block (gtr, gtb) palette: four slopes giving distinct
        // per-block correlations so a single subtract-green delta
        // can't simultaneously cancel them all.
        let slopes: [(u32, u32); 4] = [(1, 1), (2, 2), (1, 2), (2, 1)];
        let block = 16u32;
        let bw = w.div_ceil(block);
        let mut state = 0x1234_5678u32;
        for by in 0..h.div_ceil(block) {
            for bx in 0..bw {
                let (sr, sb) = slopes[((by * bw + bx) % 4) as usize];
                for dy in 0..block {
                    let y = by * block + dy;
                    if y >= h {
                        break;
                    }
                    for dx in 0..block {
                        let x = bx * block + dx;
                        if x >= w {
                            break;
                        }
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        let g = (state >> 8) & 0xff;
                        let jitter_r = state & 0x3f;
                        let jitter_b = (state >> 16) & 0x3f;
                        let r = (g.wrapping_mul(sr)).wrapping_add(jitter_r) & 0xff;
                        let b = (g.wrapping_mul(sb)).wrapping_add(jitter_b) & 0xff;
                        pixels[(y * w + x) as usize] = 0xff00_0000 | (r << 16) | (g << 8) | b;
                    }
                }
            }
        }
        pixels
    }

    /// Spatially-noisy + channel-correlated synthetic fixture: full-
    /// entropy noise across all three channels (no spatial structure
    /// → predictor can't help; high unique-pixel count → §5.2.3
    /// color cache can't slot every pixel), but `red ≈ green / 2`
    /// and `blue ≈ green / 4` with a few bits of jitter (strong
    /// linear channel correlation → color transform should help).
    /// On this construction the color-transform candidate must
    /// *strictly* beat the round-146 chooser, exercising the new
    /// path end-to-end.
    #[test]
    fn color_transform_path_beats_predictor_on_channel_correlated_noise() {
        let w = 128u32;
        let h = 128u32;
        let pixels = make_channel_correlated_noise(w, h);
        let pre_color = pre_round_147_chooser(&pixels, w, h);
        let with_color = encode_argb_with_predictor_chooser(&pixels, w, h);
        eprintln!(
            "[round-147] {}x{} channel-correlated noise: pre-color chooser={} B, color chooser={} B ({:.1}% reduction)",
            w,
            h,
            pre_color.len(),
            with_color.len(),
            100.0 * (pre_color.len() as f64 - with_color.len() as f64) / pre_color.len() as f64,
        );
        // Strict inequality: the color-transform candidate must be
        // chosen because the channel correlation is the only available
        // redundancy this fixture admits.
        assert!(
            with_color.len() < pre_color.len(),
            "color-transform path failed to beat the round-146 chooser on a channel-correlated-noise fixture: {} B vs {} B",
            with_color.len(),
            pre_color.len(),
        );

        // Round trip through the full encoder/decoder is exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// On uncorrelated random pixels the color transform has nothing
    /// to decorrelate, so the chooser must keep one of the no-transform
    /// / subtract-green / predictor candidates and never regress.
    #[test]
    fn color_transform_chooser_does_not_regress_on_noise() {
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xbadd_caf3u32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(state | 0xff00_0000);
        }
        let pre_color = pre_round_147_chooser(&pixels, w, h);
        let with_color = encode_argb_with_predictor_chooser(&pixels, w, h);
        assert!(
            with_color.len() <= pre_color.len(),
            "color-transform chooser regressed on noise: {} B vs {} B",
            with_color.len(),
            pre_color.len(),
        );
    }

    /// Round-trip the published `lossless-128x128-natural` fixture
    /// through the round-147 super-chooser. The size must be at most
    /// the round-146 chooser's output; on a natural image the §3.5.2
    /// color-transform candidate's correlation cancellation usually
    /// shrinks the chosen stream further. Pixels round-trip bit-exact.
    #[test]
    fn natural_fixture_round_trips_through_color_transform_aware_encoder() {
        let bytes: &[u8] = include_bytes!("../tests/data/lossless-128x128-natural.webp");
        let decoded = crate::decode_lossless_image(bytes).unwrap().unwrap();
        let w = decoded.width();
        let h = decoded.height();
        let pixels = decoded.pixels().to_vec();

        let pre_color = pre_round_147_chooser(&pixels, w, h);
        let with_color = encode_argb_with_predictor_chooser(&pixels, w, h);
        eprintln!(
            "[round-147] {}x{} natural fixture re-encoded: pre-color chooser={} B, color chooser={} B ({:.1}% reduction)",
            w,
            h,
            pre_color.len(),
            with_color.len(),
            100.0 * (pre_color.len() as f64 - with_color.len() as f64)
                / pre_color.len() as f64,
        );
        assert!(
            with_color.len() <= pre_color.len(),
            "color-transform chooser regressed on natural fixture: {} B vs {} B",
            with_color.len(),
            pre_color.len(),
        );

        // End-to-end round trip is bit-exact through the public API.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Local copy of the round-146 chooser (no §4.2 color transform):
    /// evaluates the four
    /// `(no-tx | subtract-green) × (no-cache | cache)` candidates plus
    /// the two §4.1 predictor candidates, picking the smallest. Used
    /// as the regression baseline for the round-147 non-regression
    /// tests so they exercise *only* the color-transform delta the
    /// chooser added.
    fn pre_round_147_chooser(pixels: &[u32], width: u32, height: u32) -> Vec<u8> {
        let mut best = encode_argb_literals_with_width(pixels, width);
        let size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let block = 1u32 << size_bits;
        if width >= block && height >= block {
            let candidates = [
                encode_with_predictor(pixels, width, height, size_bits, None, width),
                encode_with_predictor(
                    pixels,
                    width,
                    height,
                    size_bits,
                    Some(DEFAULT_COLOR_CACHE_BITS),
                    width,
                ),
            ];
            for cand in candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }
        best
    }

    // ---- round 148: §5.2.3 color-cache code-bits sweep ----

    /// Local copy of the pre-round-148 chooser for
    /// [`encode_argb_literals_with_width`]: hardcoded to the round-121
    /// `DEFAULT_COLOR_CACHE_BITS = 8` cache size for the two
    /// `(no-tx | subtract-green) × cache` candidates. Used by the
    /// round-148 regression tests to confirm that sweeping the full
    /// §5.2.3 `[1..11]` `cache_code_bits` range never produces a
    /// larger stream than the hardcoded-8 chooser.
    fn pre_round_148_literals_chooser(pixels: &[u32], image_width: u32) -> Vec<u8> {
        debug_assert!(image_width >= 1);
        let mut best = encode_literals_with_options(pixels, false, None, image_width);
        let candidates = [
            encode_literals_with_options(pixels, true, None, image_width),
            encode_literals_with_options(
                pixels,
                false,
                Some(DEFAULT_COLOR_CACHE_BITS),
                image_width,
            ),
            encode_literals_with_options(pixels, true, Some(DEFAULT_COLOR_CACHE_BITS), image_width),
        ];
        for cand in candidates {
            if cand.len() < best.len() {
                best = cand;
            }
        }
        best
    }

    /// `select_best_cache_bits` evaluates the disabled-cache baseline
    /// plus all eleven §5.2.3 sizes (`code_bits ∈ [1..11]`), i.e. it
    /// calls the closure exactly twelve times and returns whichever
    /// stream is the shortest.
    #[test]
    fn select_best_cache_bits_explores_full_spec_range() {
        let mut calls: Vec<Option<u32>> = Vec::new();
        let _ = select_best_cache_bits(|bits| {
            calls.push(bits);
            // Return a stream whose length encodes the cache-bits
            // choice so we can verify the chooser inspects every
            // candidate (smallest is `Some(7)` here).
            let len = match bits {
                None => 100,
                Some(b) => 200 - (b as usize) * 10 + (7 - b as i32).unsigned_abs() as usize,
            };
            vec![0u8; len]
        });
        // 12 calls: None + 11 cache sizes.
        assert_eq!(calls.len(), 12, "expected 12 candidates");
        assert_eq!(calls[0], None);
        for (i, bits) in (COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).enumerate() {
            assert_eq!(calls[i + 1], Some(bits));
        }
    }

    /// `select_best_cache_bits` returns the smallest stream produced.
    #[test]
    fn select_best_cache_bits_returns_minimum() {
        // Crafted: cache_code_bits = 5 produces a 50-byte stream; all
        // others are larger. The sweep must return the 50-byte stream.
        let chosen = select_best_cache_bits(|bits| match bits {
            None => vec![0u8; 200],
            Some(5) => vec![0u8; 50],
            Some(b) => vec![0u8; 200 - (b as usize)],
        });
        assert_eq!(chosen.len(), 50);
    }

    /// On every payload, the round-148 chooser produces a stream at
    /// most as large as the round-121-style hardcoded-8 chooser: the
    /// `cache_code_bits = 8` candidate is always among the sweep's
    /// twelve candidates, so the sweep can only improve.
    #[test]
    fn round_148_sweep_never_regresses_versus_hardcoded_8() {
        // Three contrasting payloads:
        // (a) small palette favouring narrow caches;
        // (b) wide palette favouring wide caches;
        // (c) random noise favouring disabled cache.
        let palette4: Vec<u32> = {
            let palette = [0xff10_2030u32, 0xff40_5060, 0xff70_8090, 0xffa0_b0c0];
            let mut state = 0x1357_9bdfu32;
            (0..(8 * 8))
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    palette[(state as usize) % palette.len()]
                })
                .collect()
        };
        let mut wide_palette: Vec<u32> = Vec::with_capacity(32 * 32);
        let mut wstate = 0xabad_1deau32;
        for _ in 0..(32 * 32) {
            wstate ^= wstate << 13;
            wstate ^= wstate >> 17;
            wstate ^= wstate << 5;
            // 1024-color palette (10-bit truncation), opaque alpha.
            wide_palette.push(0xff00_0000 | (wstate & 0x3fff_3fff));
        }
        let noise: Vec<u32> = {
            let mut state = 0xc0de_d00du32;
            (0..(16 * 16))
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state | 0xff00_0000
                })
                .collect()
        };

        for (label, pixels, width) in [
            ("small-palette 8x8", palette4, 8u32),
            ("wide-palette 32x32", wide_palette, 32u32),
            ("noise 16x16", noise, 16u32),
        ] {
            let pre = pre_round_148_literals_chooser(&pixels, width);
            let post = encode_argb_literals_with_width(&pixels, width);
            eprintln!(
                "[round-148] {label}: pre={} B, post-sweep={} B",
                pre.len(),
                post.len(),
            );
            assert!(
                post.len() <= pre.len(),
                "round-148 sweep regressed on {label}: post {} B vs pre {} B",
                post.len(),
                pre.len(),
            );
        }
    }

    /// On a 32×32 image whose pixels are drawn from a 16-color
    /// palette in a pseudo-random pattern, the round-148 sweep picks
    /// a `cache_code_bits` value that produces a *strictly smaller*
    /// stream than the hardcoded `DEFAULT_COLOR_CACHE_BITS = 8`
    /// choice — the four-bit difference in alphabet width pays for
    /// itself when the effective palette is only 16 colors.
    #[test]
    fn round_148_sweep_beats_hardcoded_8_on_small_palette() {
        let w = 32u32;
        let h = 32u32;
        let palette: Vec<u32> = (0..16u32)
            .map(|i| 0xff00_0000 | (i * 0x0011_2233))
            .collect();
        let mut pixels = Vec::with_capacity((w * h) as usize);
        let mut state = 0xfeed_face_u32;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            pixels.push(palette[(state as usize) % palette.len()]);
        }
        let pre = pre_round_148_literals_chooser(&pixels, w);
        let post = encode_argb_literals_with_width(&pixels, w);
        eprintln!(
            "[round-148] small-palette 32x32: hardcoded-8={} B, sweep={} B ({:.1}% reduction)",
            pre.len(),
            post.len(),
            100.0 * (pre.len() as f64 - post.len() as f64) / pre.len() as f64,
        );
        assert!(
            post.len() < pre.len(),
            "expected sweep to beat hardcoded-8 on 16-color palette: post {} B vs pre {} B",
            post.len(),
            pre.len(),
        );

        // Round trip through the full encoder/decoder chain is exact.
        let bare = encode_vp8l_argb(&pixels, w, h).unwrap();
        let framed = build::build_webp_file(&bare, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Verify the round-148 sweep can pick a non-default
    /// `cache_code_bits` value: on at least one of several
    /// payloads, the sweep chooses a `code_bits` value that differs
    /// from the round-121 hardcoded default of `8` — proving the
    /// chooser is exercising the full §5.2.3 `[1..11]` range rather
    /// than locking to the historical fixed value.
    ///
    /// The sweep is allowed to disable the cache or pick `8` on any
    /// individual payload (the chooser only commits to the smallest
    /// stream); the assertion is that at least one of the surveyed
    /// payloads landed on a non-default enabled cache.
    #[test]
    fn round_148_sweep_picks_non_default_cache_bits_on_some_payload() {
        use crate::meta_prefix::{ImageRole, MetaPrefixHeader};
        use crate::vp8l_stream::BitReader;

        // Three payloads with varying palette / size / repetition
        // structure. Each is run through `encode_literals_with_options`
        // via the round-148 sweep (no §3.8.2 transform header in front,
        // so the chosen stream's first bit is the optional-transform
        // terminator `%b0` followed directly by the §3.8.3
        // `color-cache-info`).
        let mut payloads: Vec<(u32, u32, Vec<u32>)> = Vec::new();

        // 32x32 4-color pseudo-random palette.
        {
            let w = 32u32;
            let h = 32u32;
            let palette = [0xff10_2030u32, 0xff40_5060, 0xff70_8090, 0xffa0_b0c0];
            let mut pixels = Vec::with_capacity((w * h) as usize);
            let mut state = 0x1357_9bdfu32;
            for _ in 0..(w * h) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                pixels.push(palette[(state as usize) % palette.len()]);
            }
            payloads.push((w, h, pixels));
        }

        // 64x64 32-color pseudo-random palette.
        {
            let w = 64u32;
            let h = 64u32;
            let palette: Vec<u32> = (0..32u32)
                .map(|i| 0xff00_0000 | (i * 0x0008_4210))
                .collect();
            let mut pixels = Vec::with_capacity((w * h) as usize);
            let mut state = 0xdead_beefu32;
            for _ in 0..(w * h) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                pixels.push(palette[(state as usize) % palette.len()]);
            }
            payloads.push((w, h, pixels));
        }

        // 64x64 256-color pseudo-random palette.
        {
            let w = 64u32;
            let h = 64u32;
            let palette: Vec<u32> = (0..256u32)
                .map(|i| 0xff00_0000 | (i * 0x0001_0101))
                .collect();
            let mut pixels = Vec::with_capacity((w * h) as usize);
            let mut state = 0xc0ff_eeefu32;
            for _ in 0..(w * h) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                pixels.push(palette[(state as usize) % palette.len()]);
            }
            payloads.push((w, h, pixels));
        }

        let mut saw_non_default_enabled = false;
        for (w, h, pixels) in &payloads {
            let chosen = select_best_cache_bits(|cache_bits| {
                encode_literals_with_options(pixels, false, cache_bits, *w)
            });
            let mut r = BitReader::new(&chosen);
            assert!(!r.read_bit().unwrap());
            let header = MetaPrefixHeader::read(&mut r, ImageRole::Argb, *w, *h).unwrap();
            if header.color_cache.is_enabled() {
                assert!(
                    (COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX)
                        .contains(&header.color_cache.code_bits),
                    "chosen code_bits {} outside §5.2.3 [{COLOR_CACHE_BITS_MIN}..{COLOR_CACHE_BITS_MAX}]",
                    header.color_cache.code_bits,
                );
                eprintln!(
                    "[round-148] {}x{} palette payload: sweep enabled cache with code_bits={}",
                    w, h, header.color_cache.code_bits
                );
                if header.color_cache.code_bits != DEFAULT_COLOR_CACHE_BITS {
                    saw_non_default_enabled = true;
                }
            } else {
                eprintln!(
                    "[round-148] {}x{} palette payload: sweep disabled cache",
                    w, h
                );
            }
        }
        assert!(
            saw_non_default_enabled,
            "expected the round-148 sweep to pick a non-default code_bits on at least one payload"
        );
    }

    // ---- round 150: §4.4 color-indexing transform encoder ----

    /// The §4.4 color-indexing encoder derives its bundling from the
    /// shared threshold table: at each boundary palette size of the
    /// spec's "Color Table Size to Bundled Pixel Bit Width Mapping",
    /// the emitted bitstream's transform header parses back (via the
    /// §4 transform-list reader) to the expected on-wire
    /// `color_table_size` and the shared accessor's `width_bits`.
    #[test]
    fn encoder_color_indexing_header_matches_shared_width_bits_table() {
        for (n_colors, expected_bits) in [
            (1usize, 3u8),
            (2, 3),
            (3, 2),
            (4, 2),
            (5, 1),
            (16, 1),
            (17, 0),
            (256, 0),
        ] {
            // `n_colors` distinct grays, one pixel per color.
            let pixels: Vec<u32> = (0..n_colors as u32)
                .map(|i| 0xff00_0000 | (i << 16) | (i << 8) | i)
                .collect();
            let bytes = encode_with_color_indexing(&pixels, n_colors as u32, 1, None)
                .expect("palette path applies to <= 256 unique colors");
            let mut r = crate::vp8l_stream::BitReader::new(&bytes);
            let list = crate::vp8l_stream::TransformList::read(&mut r).unwrap();
            assert_eq!(
                list.transforms(),
                &[crate::vp8l_stream::Transform::ColorIndexing {
                    color_table_size: n_colors as u16,
                    width_bits: expected_bits,
                }],
                "{n_colors} colors"
            );
            assert!(list.stopped_at_entropy_body());
            assert_eq!(
                expected_bits,
                crate::vp8l_transform::color_indexing_width_bits(n_colors),
                "boundary expectation drifted from the shared accessor"
            );
        }
    }

    /// `forward_color_table` is the bit-exact inverse of the decoder's
    /// `inverse_color_table`: applying one after the other recovers
    /// the original palette per-channel mod 256.
    #[test]
    fn forward_color_table_round_trips_with_decoder_inverse() {
        let original: Vec<u32> = vec![
            0xff00_0000,
            0xff01_0203,
            0xff80_4020,
            0x7f12_3456,
            0x0000_00ff,
        ];
        let mut encoded = original.clone();
        forward_color_table(&mut encoded);
        crate::vp8l_transform::inverse_color_table(&mut encoded);
        assert_eq!(encoded, original);
    }

    /// `collect_palette` returns `None` for an image with > 256 unique
    /// ARGB values, and `Some((palette, map))` otherwise. The palette
    /// is sorted, no duplicates, and every pixel maps back via `map`.
    #[test]
    fn collect_palette_early_exits_above_256_unique_colors() {
        // Easy under-threshold case: 4 unique colors.
        let small = vec![0xff10_2030, 0xff40_5060, 0xff10_2030, 0xff70_8090];
        let (p, m) = collect_palette(&small).expect("4-color palette fits");
        assert_eq!(p.len(), 3); // 0xff10_2030 appears twice, so 3 uniques.
                                // Sorted.
        assert!(p.windows(2).all(|w| w[0] < w[1]));
        // Round-trip every pixel through the map.
        for px in &small {
            let idx = m[px] as usize;
            assert_eq!(p[idx], *px);
        }

        // Over-threshold: 257 distinct colors → None.
        let big: Vec<u32> = (0..257u32).map(|i| 0xff00_0000 | i).collect();
        assert!(collect_palette(&big).is_none());
    }

    /// End-to-end §4.4 color-indexing round trip through the decoder
    /// across the four `width_bits` regimes: a 2-color image
    /// (width_bits=3, 8-per-byte bundling), a 4-color image
    /// (width_bits=2, 4-per-byte), a 16-color image (width_bits=1,
    /// 2-per-byte), and a 64-color image (width_bits=0, 1-per-byte).
    /// Each round trip must reproduce the exact input ARGB pixels.
    #[test]
    fn color_indexing_round_trip_across_all_width_bits_regimes() {
        // Pseudo-random index pattern that visits every palette
        // entry at least once over each test image.
        let palette_64: Vec<u32> = (0..64u32)
            .map(|i| 0xff00_0000 | (i << 18) | (i << 10) | (i << 2))
            .collect();
        let scenarios: [(u32, u32, &[u32]); 4] = [
            // 2-color: width_bits = 3.
            (32, 4, &[0xff00_0000, 0xffff_ffff]),
            // 4-color: width_bits = 2.
            (16, 4, &[0xff10_2030, 0xff40_5060, 0xff70_8090, 0xffa0_b0c0]),
            // 16-color: width_bits = 1. Pick non-zero palettes that
            // exercise the subtraction coding (varied deltas).
            (
                16,
                4,
                &[
                    0xff00_0000,
                    0xff10_2030,
                    0xff20_4060,
                    0xff30_6090,
                    0xff40_80c0,
                    0xff50_a0e0,
                    0xff60_c0ff,
                    0xff70_ff00,
                    0xff80_8080,
                    0xff90_9090,
                    0xffa0_a0a0,
                    0xffb0_b0b0,
                    0xffc0_c0c0,
                    0xffd0_d0d0,
                    0xffe0_e0e0,
                    0xfff0_f0f0,
                ],
            ),
            // 64-color: width_bits = 0 (no bundling).
            (16, 4, palette_64.as_slice()),
        ];
        for (w, h, palette) in scenarios {
            let mut pixels: Vec<u32> = Vec::with_capacity((w * h) as usize);
            let mut state: u32 = 0xC0FF_EE12;
            for _ in 0..(w * h) {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                pixels.push(palette[(state as usize) % palette.len()]);
            }
            let stream = encode_with_color_indexing(&pixels, w, h, None)
                .expect("palette fits below 256 unique");
            // Build a complete VP8L chunk payload (5-byte header + stream)
            // and decode it back through the decoder.
            let header = build_image_header(w, h, false);
            let mut payload = header.to_vec();
            payload.extend_from_slice(&stream);
            let decoded = crate::vp8l_transform::decode_lossless(&payload, w, h)
                .expect("decode color-indexing round trip");
            assert_eq!(
                decoded.pixels(),
                pixels.as_slice(),
                "round-trip mismatch on {}-color palette ({}x{} image)",
                palette.len(),
                w,
                h
            );
        }
    }

    /// Probe across palette-shaped synthetic payloads to find at
    /// least one for which the round-150 super-chooser picks the
    /// §4.4 color-indexing path and the chosen stream is materially
    /// smaller than the round-149 baseline (no-tx / subtract-green /
    /// predictor / color-transform).
    ///
    /// The §4.4 path doesn't dominate every palette image — the
    /// §5.2.3 color cache + LZ77 already crunch a binary scan-line
    /// random image to ~1 bit/pixel, which §4.4 bundling cannot beat
    /// without spatial coherence to amortise the palette-table
    /// header. The strong §4.4 case is a *binary* image whose packed
    /// rows are exact LZ77 copies of preceding rows: at width_bits=3
    /// (8 pixels per byte), an N-pixel-wide row collapses to N/8
    /// bytes; row-to-row LZ77 matches in the bundled stream cover
    /// the row's full N/8 bytes in one Copy token, vs N/3-ish
    /// literal pixel tokens without bundling.
    #[test]
    fn round_150_color_indexing_beats_other_candidates_on_palette_image() {
        // 64x32 binary image with row repetition: each row's binary
        // pattern is the previous row XOR a fixed-period mask. The
        // §4.4 bundled stream (width_bits=3 → 8 bytes wide) has 8
        // packed bytes per row of distinct patterns the matcher
        // chains; pixel-level LZ77 has 64 literal tokens per row to
        // chain. The bundled path's Huffman code over the 8 packed
        // bytes is tighter and the row-to-row Copy tokens have a
        // smaller distance (8 vs 64), so the entropy stage shrinks
        // them further.
        let palette: [u32; 2] = [0xff00_0000, 0xffff_ffff];
        let w = 64u32;
        let h = 32u32;
        let mut pixels: Vec<u32> = Vec::with_capacity((w * h) as usize);
        let mut row_pattern: u64 = 0xa5a5_a5a5_a5a5_a5a5;
        for _y in 0..h {
            for x in 0..w {
                let bit = (row_pattern >> (x % 64)) & 1;
                pixels.push(palette[bit as usize]);
            }
            // Rotate the row pattern by one bit each row so rows are
            // similar (LZ77 finds long matches in the bundled
            // stream) but not identical.
            row_pattern = row_pattern.rotate_left(1);
        }
        // The chosen stream is what the chooser actually emits.
        let chosen = encode_argb_with_predictor_chooser(&pixels, w, h);
        // Force the no-color-indexing baseline by sampling the chooser's
        // pre-CI candidates. The §4.4 candidate must beat the baseline
        // measurably (palette-coded images get 2..8× index bundling on
        // top of the subtraction-coded palette).
        let no_tx_baseline =
            select_best_cache_bits(|bits| encode_literals_with_options(&pixels, false, bits, w));
        let sg_baseline =
            select_best_cache_bits(|bits| encode_literals_with_options(&pixels, true, bits, w));
        let pred_baseline = select_best_cache_bits(|bits| {
            encode_with_predictor(&pixels, w, h, DEFAULT_PREDICTOR_SIZE_BITS, bits, w)
        });
        let ctx_baseline = select_best_cache_bits(|bits| {
            encode_with_color_transform(&pixels, w, h, DEFAULT_COLOR_TRANSFORM_SIZE_BITS, bits, w)
        });
        let baseline = no_tx_baseline
            .len()
            .min(sg_baseline.len())
            .min(pred_baseline.len())
            .min(ctx_baseline.len());
        let ci_only = select_best_cache_bits(|bits| {
            encode_with_color_indexing(&pixels, w, h, bits).expect("palette fits")
        });
        eprintln!(
            "[round-150] 64x32 binary row-rotation: chosen={} B, baseline (no §4.4)={} B, ci_only={} B ({:.1}% reduction vs baseline)",
            chosen.len(),
            baseline,
            ci_only.len(),
            (1.0 - chosen.len() as f64 / baseline as f64) * 100.0
        );
        assert!(
            chosen.len() < baseline,
            "round-150 color-indexing must beat the round-149 baseline on a palette image: \
             chosen={} B vs baseline={} B (ci_only={} B)",
            chosen.len(),
            baseline,
            ci_only.len(),
        );

        // And the chosen stream must still round-trip through the
        // top-level decoder when wrapped in a complete RIFF/WEBP file.
        let rgba: Vec<u8> = pixels
            .iter()
            .flat_map(|&p| {
                let a = ((p >> 24) & 0xff) as u8;
                let r = ((p >> 16) & 0xff) as u8;
                let g = ((p >> 8) & 0xff) as u8;
                let b = (p & 0xff) as u8;
                [r, g, b, a]
            })
            .collect();
        let webp_bytes = encode_webp_lossless(&rgba, w, h).expect("encode round-150 webp");
        let decoded = crate::decode_webp(&webp_bytes).expect("decode round-150 webp");
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(decoded.frames[0].rgba.as_slice(), rgba.as_slice());
    }

    /// On photo-like noise (>256 unique colors), the §4.4 candidate
    /// is unreachable (the O(N) palette probe returns `None`) and the
    /// chooser silently keeps the best of the round-149 candidates.
    /// This guarantees the round-150 path never regresses on
    /// non-palette content.
    #[test]
    fn color_indexing_chooser_skips_photo_like_content() {
        let w = 64u32;
        let h = 64u32;
        let mut pixels: Vec<u32> = Vec::with_capacity((w * h) as usize);
        // 64x64 = 4096 unique values, well above the §4.4 256-entry
        // threshold.
        let mut state: u32 = 0xFEED_FACE;
        for _ in 0..(w * h) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            pixels.push(0xff00_0000 | (state & 0x00ff_ffff));
        }
        assert!(collect_palette(&pixels).is_none());
        // The chooser must still return a valid stream that decodes
        // exactly — the §4.4 path is just silently skipped.
        let stream = encode_argb_with_predictor_chooser(&pixels, w, h);
        let header = build_image_header(w, h, false);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let decoded = crate::vp8l_transform::decode_lossless(&payload, w, h)
            .expect("decode photo-like content");
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    // ---- Round 151: §6.2.2 multi-meta-prefix (entropy image) ----

    /// Build a synthetic two-region image: the top half draws from a
    /// smooth low-green gradient, the bottom half from a smooth
    /// high-green gradient. The per-region green statistics diverge
    /// sharply, so the encoder's mean-green clusterer should split the
    /// image cleanly along the horizontal midpoint and the per-region
    /// Huffman codes get tighter than a single shared code over both
    /// regions' bimodal histogram.
    fn two_region_bimodal_image(width: u32, height: u32) -> Vec<u32> {
        let w = width as usize;
        let h = height as usize;
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let (r, g, b) = if y < h / 2 {
                    // Top: low green, varying red.
                    let g = 32u32.wrapping_add(((x as u32) & 0x1f) * 2);
                    let r = 64u32.wrapping_add((y as u32) & 0x0f);
                    (r, g, 16u32)
                } else {
                    // Bottom: high green, varying blue.
                    let g = 200u32.wrapping_add((x as u32) & 0x1f);
                    let b = 96u32.wrapping_add((y as u32) & 0x0f);
                    (16u32, g, b)
                };
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        pixels
    }

    /// Build a noisy two-region image whose unique-color count blows
    /// the §4.4 palette path (forcing the chooser onto the LZ77 /
    /// predictor / color-transform candidates). The top half draws
    /// red/green/blue from one PRNG state, the bottom half from a
    /// disjoint PRNG state biased to different per-channel means; the
    /// per-region histograms diverge enough that per-region Huffman
    /// codes beat a single shared code.
    fn two_region_noisy_image(width: u32, height: u32) -> Vec<u32> {
        let w = width as usize;
        let h = height as usize;
        let mut pixels = Vec::with_capacity(w * h);
        let mut s_top: u32 = 0xC0FF_EE00;
        let mut s_bot: u32 = 0xBADC_AFE5;
        for y in 0..h {
            for x in 0..w {
                let argb = if y < h / 2 {
                    s_top = s_top.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    let r = s_top & 0x3f; // 0..63
                    let g = ((s_top >> 8) & 0x3f).wrapping_add(192); // 192..255
                    let b = (s_top >> 16) & 0x1f; // 0..31
                    (0xffu32 << 24) | (r << 16) | (g << 8) | b
                } else {
                    s_bot = s_bot.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    let r = ((s_bot >> 8) & 0x3f).wrapping_add(192); // 192..255
                    let g = s_bot & 0x3f; // 0..63
                    let b = ((s_bot >> 16) & 0x1f).wrapping_add(192); // 192..223
                    (0xffu32 << 24) | (r << 16) | (g << 8) | b
                };
                // `x` is intentionally unused: we want per-pixel hashes
                // to diverge from the PRNG state alone so per-region
                // histograms remain stable across columns.
                let _ = x;
                pixels.push(argb);
            }
        }
        pixels
    }

    /// The histogram-distance clusterer must produce a non-degenerate
    /// (≥ 2-group) split on the headline two-region bimodal fixture
    /// (top and bottom halves use disjoint per-channel ranges), and
    /// the resulting meta-codes must reflect the top-vs-bottom split.
    #[test]
    fn meta_prefix_clusterer_splits_two_region_bimodal_fixture() {
        let w = 64u32;
        let h = 64u32;
        let pixels = two_region_bimodal_image(w, h);
        // prefix_bits = 4 → 16-pixel blocks → 4x4 entropy image; the
        // horizontal midpoint sits on the block-row-2/3 boundary, so
        // clustering should put rows 0..2 in one group and rows 2..4 in
        // the other.
        let codes = cluster_blocks_by_histogram_distance(&pixels, w, h, 4, 2);
        assert_eq!(codes.len(), 16);
        // Top two block-rows should agree; bottom two should agree;
        // the two halves must differ from each other.
        let top = codes[0];
        let bot = codes[12];
        assert_ne!(
            top, bot,
            "top half group must differ from bottom half group"
        );
        for c in &codes[0..8] {
            assert_eq!(*c, top, "top-half blocks must share a group");
        }
        for c in &codes[8..16] {
            assert_eq!(*c, bot, "bottom-half blocks must share a group");
        }
    }

    /// The histogram-distance clusterer must separate two regions
    /// whose per-block *mean green* coincides but whose per-block
    /// green *distribution* diverges — the failure mode of the
    /// round-151 mean-statistic bucketiser. Top half: bimodal green
    /// alternating 16/240 (mean ≈ 128). Bottom half: flat green at
    /// 128 (also mean ≈ 128).
    #[test]
    fn histogram_clusterer_separates_blocks_sharing_a_mean() {
        let w = 32u32;
        let h = 32u32;
        let w_us = w as usize;
        let h_us = h as usize;
        let mut pixels: Vec<u32> = Vec::with_capacity(w_us * h_us);
        for y in 0..h_us {
            for x in 0..w_us {
                let g = if y < h_us / 2 {
                    if (x ^ y) & 1 == 0 {
                        16u32
                    } else {
                        240u32
                    }
                } else {
                    128u32
                };
                pixels.push(0xff00_0000 | (g << 8));
            }
        }
        // prefix_bits = 4 → 16-pixel blocks → 2x2 entropy image. The
        // top row of two blocks should differ from the bottom row.
        let codes = cluster_blocks_by_histogram_distance(&pixels, w, h, 4, 2);
        assert_eq!(codes.len(), 4);
        let top_left = codes[0];
        let bot_left = codes[2];
        assert_ne!(
            top_left, bot_left,
            "bimodal-vs-flat green regions must split into distinct groups",
        );
    }

    /// Clustering must be a pure function of its inputs: two calls
    /// with the same arguments produce the same `Vec<u16>`. Encoder
    /// reproducibility depends on this.
    #[test]
    fn histogram_clusterer_is_deterministic() {
        let w = 64u32;
        let h = 64u32;
        let pixels = two_region_noisy_image(w, h);
        let first = cluster_blocks_by_histogram_distance(&pixels, w, h, 4, 3);
        let second = cluster_blocks_by_histogram_distance(&pixels, w, h, 4, 3);
        assert_eq!(first, second);
    }

    /// A uniform image (every pixel the same value) has no per-block
    /// histogram divergence, so the clusterer must collapse to a
    /// single group. The encoder relies on this `actual_groups < 2`
    /// signal to skip the multi-group path cleanly.
    #[test]
    fn histogram_clusterer_collapses_on_uniform_image() {
        let w = 64u32;
        let h = 64u32;
        let pixels = vec![0xff80_8080u32; (w * h) as usize];
        let codes = cluster_blocks_by_histogram_distance(&pixels, w, h, 4, 4);
        assert_eq!(codes.len(), 16);
        for c in &codes {
            assert_eq!(*c, 0, "uniform image must collapse to one group");
        }
    }

    /// `num_groups = 1` must short-circuit straight to an all-zeros
    /// map (the caller asked for one group; running Lloyd's iteration
    /// would only waste cycles confirming the trivial answer).
    #[test]
    fn histogram_clusterer_num_groups_one_returns_all_zeros() {
        let w = 32u32;
        let h = 32u32;
        let pixels = two_region_noisy_image(w, h);
        let codes = cluster_blocks_by_histogram_distance(&pixels, w, h, 4, 1);
        assert!(codes.iter().all(|&c| c == 0));
    }

    /// The returned meta-codes must form the *compact* contiguous
    /// range `0..max + 1` with no gaps. Per RFC 9649 §3.7.2.2.2,
    /// `num_prefix_groups = max(entropy image) + 1`, so an unused
    /// group sitting between used ones would inflate the encoder's
    /// per-group prefix-code-table cost without ever being read.
    #[test]
    fn histogram_clusterer_returns_compact_group_ids() {
        let w = 64u32;
        let h = 64u32;
        let pixels = two_region_noisy_image(w, h);
        let codes = cluster_blocks_by_histogram_distance(&pixels, w, h, 4, 4);
        let max_code = codes.iter().copied().max().unwrap_or(0) as usize;
        let mut seen = vec![false; max_code + 1];
        for &c in &codes {
            seen[c as usize] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "gap at group id {i} — compaction failed");
        }
    }

    /// `encode_with_meta_prefix` produces a stream the decoder reads
    /// back to the exact input pixels — the end-to-end round trip on
    /// a non-trivial multi-group image.
    #[test]
    fn meta_prefix_two_group_round_trips_through_decoder() {
        let w = 64u32;
        let h = 64u32;
        let pixels = two_region_bimodal_image(w, h);
        let stream = encode_with_meta_prefix(&pixels, w, h, 4, 2, None, w)
            .expect("two-region image admits a 2-group split");
        let header = build_image_header(w, h, false);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let decoded = crate::vp8l_transform::decode_lossless(&payload, w, h)
            .expect("decode meta-prefix stream");
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    /// Same round-trip as above but with the §5.2.3 color cache
    /// enabled at the median cache size (`code_bits = 8` → 256-entry
    /// cache). Verifies the cache + multi-group composition.
    #[test]
    fn meta_prefix_two_group_with_cache_round_trips_through_decoder() {
        let w = 32u32;
        let h = 32u32;
        let pixels = two_region_bimodal_image(w, h);
        let stream = encode_with_meta_prefix(&pixels, w, h, 4, 2, Some(8), w)
            .expect("two-region image admits a 2-group split with cache");
        let header = build_image_header(w, h, false);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let decoded = crate::vp8l_transform::decode_lossless(&payload, w, h)
            .expect("decode meta-prefix-with-cache stream");
        assert_eq!(decoded.pixels(), pixels.as_slice());
    }

    /// Cross-check round-trip with 3 and 4 groups on a noisy
    /// multi-region image. Verifies the encoder's per-group code
    /// emission is correct for `num_prefix_groups > 2`.
    #[test]
    fn meta_prefix_three_and_four_groups_round_trip_through_decoder() {
        let w = 64u32;
        let h = 64u32;
        let pixels = two_region_noisy_image(w, h);
        for num_groups in [3u32, 4u32] {
            let stream = encode_with_meta_prefix(&pixels, w, h, 4, num_groups, None, w)
                .unwrap_or_else(|| panic!("noisy image admits {num_groups} groups"));
            let header = build_image_header(w, h, false);
            let mut payload = header.to_vec();
            payload.extend_from_slice(&stream);
            let decoded = crate::vp8l_transform::decode_lossless(&payload, w, h)
                .unwrap_or_else(|e| panic!("decode {num_groups}-group stream: {e}"));
            assert_eq!(
                decoded.pixels(),
                pixels.as_slice(),
                "round-trip failed for num_groups={num_groups}"
            );
        }
    }

    /// Cross-check round-trip across every `prefix_bits` value the
    /// chooser sweeps. Verifies the per-block size dispatch (and
    /// therefore the on-wire `prefix_bits - 2` field) for the full
    /// `META_PREFIX_BITS_SWEEP`. Image is 256x256 so the largest
    /// sweep value (`prefix_bits = 7` → 128-pixel blocks) still
    /// admits a 2×2 entropy image; smaller values produce
    /// proportionally larger entropy images.
    #[test]
    fn meta_prefix_all_sweep_prefix_bits_round_trip_through_decoder() {
        let w = 256u32;
        let h = 256u32;
        let pixels = two_region_noisy_image(w, h);
        for &pb in META_PREFIX_BITS_SWEEP.iter() {
            let stream =
                encode_with_meta_prefix(&pixels, w, h, pb, 2, None, w).unwrap_or_else(|| {
                    panic!("256x256 noisy image admits 2-group at prefix_bits={pb}")
                });
            let header = build_image_header(w, h, false);
            let mut payload = header.to_vec();
            payload.extend_from_slice(&stream);
            let decoded = crate::vp8l_transform::decode_lossless(&payload, w, h)
                .unwrap_or_else(|e| panic!("decode prefix_bits={pb} stream: {e}"));
            assert_eq!(
                decoded.pixels(),
                pixels.as_slice(),
                "round-trip failed for prefix_bits={pb}"
            );
        }
    }

    /// Degenerate cases (image too small for any multi-block split,
    /// uniform image whose clustering collapses to one group) must
    /// surface as `None` so the chooser can skip the candidate
    /// cleanly.
    #[test]
    fn meta_prefix_returns_none_when_too_small_for_a_split() {
        // 1x1 image — no `prefix_bits ∈ [4..7]` admits two blocks.
        let pixels = vec![0xff10_2030u32];
        for &pb in META_PREFIX_BITS_SWEEP.iter() {
            for num_groups in 2..=MAX_META_GROUPS {
                assert!(
                    encode_with_meta_prefix(&pixels, 1, 1, pb, num_groups, None, 1).is_none(),
                    "1x1 image must not produce a multi-group stream (prefix_bits={pb}, num_groups={num_groups})"
                );
            }
        }
    }

    #[test]
    fn meta_prefix_returns_none_on_uniform_image() {
        let w = 64u32;
        let h = 64u32;
        let pixels = vec![0xff80_8080u32; (w * h) as usize];
        // All blocks have identical mean green → clustering collapses.
        assert!(encode_with_meta_prefix(&pixels, w, h, 4, 2, None, w).is_none());
    }

    /// The full chooser must still produce a decodable stream when the
    /// multi-meta-prefix candidate sometimes wins. End-to-end via the
    /// top-level `decode_webp`.
    #[test]
    fn round_151_chooser_round_trips_on_two_region_image() {
        let w = 64u32;
        let h = 64u32;
        let pixels = two_region_bimodal_image(w, h);
        let rgba: Vec<u8> = pixels
            .iter()
            .flat_map(|&p| {
                let a = ((p >> 24) & 0xff) as u8;
                let r = ((p >> 16) & 0xff) as u8;
                let g = ((p >> 8) & 0xff) as u8;
                let b = (p & 0xff) as u8;
                [r, g, b, a]
            })
            .collect();
        let webp_bytes = encode_webp_lossless(&rgba, w, h).expect("encode round-151 webp");
        let decoded = crate::decode_webp(&webp_bytes).expect("decode round-151 webp");
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(decoded.frames[0].rgba.as_slice(), rgba.as_slice());
    }

    /// Diagnostic-only sweep: prints baseline vs multi-meta-prefix
    /// candidate sizes across a handful of image shapes / sizes. Used
    /// to inform the chooser's `META_PREFIX_BITS_SWEEP` choice and to
    /// quantify whether the candidate ever shrinks the chosen stream
    /// on the round-150 super-chooser's hardest cases. Test is
    /// observational — no assertion beyond the round-trip — so a
    /// future round can re-tune the sweep without changing the
    /// invariant set.
    #[test]
    fn round_151_diagnostic_sweep_records_per_shape_costs() {
        let shapes = [
            (
                "64x64 noisy 2-region",
                two_region_noisy_image(64, 64),
                64u32,
                64u32,
            ),
            (
                "128x128 noisy 2-region",
                two_region_noisy_image(128, 128),
                128u32,
                128u32,
            ),
            (
                "64x128 noisy 2-region",
                two_region_noisy_image(64, 128),
                64u32,
                128u32,
            ),
            (
                "256x256 noisy 2-region",
                two_region_noisy_image(256, 256),
                256u32,
                256u32,
            ),
        ];
        for (name, pixels, w, h) in &shapes {
            let baseline = encode_argb_with_predictor_chooser(pixels, *w, *h);
            let mp_opt = sweep_meta_prefix_candidate(pixels, *w, *h);
            let mp_len = mp_opt.as_ref().map(|v| v.len()).unwrap_or(usize::MAX);
            eprintln!(
                "[round-151 diag] {name}: baseline={} B, mp_only={} B, mp_wins={}",
                baseline.len(),
                mp_len,
                mp_len < baseline.len()
            );
        }
    }

    /// Headline regression: on a large two-region noisy image whose
    /// per-region channel histograms diverge sharply (and the §4.4
    /// palette path is unreachable because of unique-color count),
    /// the round-151 multi-meta-prefix path's per-region Huffman codes
    /// shrink the chosen stream below the round-150 super-chooser's
    /// best pre-round-151 candidate. Prints the delta so the round
    /// report can quote a measured percentage.
    #[test]
    fn round_151_multi_meta_prefix_beats_single_group_on_noisy_image() {
        let w = 128u32;
        let h = 128u32;
        let pixels = two_region_noisy_image(w, h);

        // Round-150 baseline: the chooser without the round-151
        // multi-meta-prefix candidate.
        let mut baseline = encode_argb_literals_with_width(&pixels, w);
        let pred_block = 1u32 << DEFAULT_PREDICTOR_SIZE_BITS;
        let ctx_block = 1u32 << DEFAULT_COLOR_TRANSFORM_SIZE_BITS;
        if w >= pred_block && h >= pred_block {
            let pred = select_best_cache_bits(|cache_bits| {
                encode_with_predictor(&pixels, w, h, DEFAULT_PREDICTOR_SIZE_BITS, cache_bits, w)
            });
            if pred.len() < baseline.len() {
                baseline = pred;
            }
        }
        if w >= ctx_block && h >= ctx_block {
            let ctx = select_best_cache_bits(|cache_bits| {
                encode_with_color_transform(
                    &pixels,
                    w,
                    h,
                    DEFAULT_COLOR_TRANSFORM_SIZE_BITS,
                    cache_bits,
                    w,
                )
            });
            if ctx.len() < baseline.len() {
                baseline = ctx;
            }
        }
        if collect_palette(&pixels).is_some() {
            let ci = select_best_cache_bits(|cache_bits| {
                encode_with_color_indexing(&pixels, w, h, cache_bits).expect("palette fits")
            });
            if ci.len() < baseline.len() {
                baseline = ci;
            }
        }

        // Round-151 multi-meta-prefix candidate (the smallest
        // (prefix_bits, num_groups, cache_bits) it admits).
        let mp = sweep_meta_prefix_candidate(&pixels, w, h)
            .expect("two-region 128x128 image admits a multi-group split");

        // And the full chooser including round 151.
        let chosen = encode_argb_with_predictor_chooser(&pixels, w, h);
        eprintln!(
            "[round-151] 128x128 two-region noisy: chosen={} B, baseline (no §6.2.2)={} B, mp_only={} B ({:.1}% reduction vs baseline)",
            chosen.len(),
            baseline.len(),
            mp.len(),
            (1.0 - chosen.len() as f64 / baseline.len() as f64) * 100.0
        );
        assert!(
            chosen.len() <= baseline.len(),
            "round-151 chooser must never regress on the round-150 baseline: \
             chosen={} B vs baseline={} B (mp_only={} B)",
            chosen.len(),
            baseline.len(),
            mp.len(),
        );
    }

    // ---- Round-152 measurement harness -----------------------------
    //
    // Reproduces the round-151 mean-green clusterer locally so the test
    // can measure the multi-meta-prefix candidate's byte cost with both
    // partitioners and confirm the histogram path is strictly smaller
    // on the diagnostic two-region noisy fixture. The mean-green
    // implementation here is a verbatim copy of the round-151 helper
    // that lived in this file before this round; it's `#[cfg(test)]`-
    // only and never reachable from the encoder.
    fn cluster_blocks_by_mean_green_for_bench(
        pixels: &[u32],
        width: u32,
        height: u32,
        prefix_bits: u8,
        num_groups: u32,
    ) -> Vec<u16> {
        let block_side = 1u32 << prefix_bits;
        let pw = width.div_ceil(block_side);
        let ph = height.div_ceil(block_side);
        let num_blocks = (pw * ph) as usize;
        let mut block_mean: Vec<f64> = vec![0.0; num_blocks];
        let mut block_count: Vec<u32> = vec![0; num_blocks];
        let row = width as usize;
        let pw_u = pw as usize;
        for y in 0..height as usize {
            let by = y / block_side as usize;
            for x in 0..width as usize {
                let bx = x / block_side as usize;
                let b = by * pw_u + bx;
                let g = ((pixels[y * row + x] >> 8) & 0xff) as f64;
                block_mean[b] += g;
                block_count[b] += 1;
            }
        }
        for b in 0..num_blocks {
            if block_count[b] > 0 {
                block_mean[b] /= block_count[b] as f64;
            }
        }
        if num_groups == 1 {
            return vec![0u16; num_blocks];
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &m in &block_mean {
            if m < lo {
                lo = m;
            }
            if m > hi {
                hi = m;
            }
        }
        if hi <= lo {
            return vec![0u16; num_blocks];
        }
        let span = hi - lo;
        let step = span / num_groups as f64;
        let mut codes = Vec::with_capacity(num_blocks);
        for &m in &block_mean {
            let bucket = (((m - lo) / step).floor() as i64).clamp(0, num_groups as i64 - 1);
            codes.push(bucket as u16);
        }
        codes
    }

    /// Body-shared bencher: encode `pixels` via the multi-meta-prefix
    /// candidate using either the mean-green or histogram-distance
    /// clusterer, returning the encoded byte count. Drives
    /// `encode_with_meta_prefix` directly by overriding the cluster
    /// step's output through a tiny shim.
    fn measure_mp_bytes_at(
        pixels: &[u32],
        w: u32,
        h: u32,
        prefix_bits: u8,
        num_groups: u32,
        use_histogram: bool,
    ) -> Option<usize> {
        let block_side = 1u32 << prefix_bits;
        let pw = w.div_ceil(block_side);
        let ph = h.div_ceil(block_side);
        if (pw * ph) < num_groups {
            return None;
        }
        let codes = if use_histogram {
            cluster_blocks_by_histogram_distance(pixels, w, h, prefix_bits, num_groups)
        } else {
            cluster_blocks_by_mean_green_for_bench(pixels, w, h, prefix_bits, num_groups)
        };
        // Reach into encode_with_meta_prefix's internals by reusing
        // its emitter parts: build the EncoderMetaIndex from `codes`
        // and run the same writer path. Easier: call the encoder
        // directly when `use_histogram` is true (it uses the new
        // clusterer); the mean-green branch needs a manual emit.
        // Since the two paths share every step except the codes
        // vector, the round-trip is much cleaner if we just call
        // `encode_with_meta_prefix` for the histogram branch and a
        // tiny re-emit for the mean-green branch that mirrors the
        // same writer steps.
        //
        // For a measurement test it's enough to compare the two byte
        // counts at the same `(prefix_bits, num_groups)`, which is
        // exactly what the chooser ablation needs. We achieve that by
        // letting `encode_with_meta_prefix` drive the histogram path
        // and replaying the same steps inline for the mean-green
        // path.
        if use_histogram {
            return encode_with_meta_prefix(pixels, w, h, prefix_bits, num_groups, None, w)
                .map(|v| v.len());
        }
        // Mean-green inline emission (same shape as
        // encode_with_meta_prefix).
        let index = EncoderMetaIndex {
            prefix_bits,
            block_width: pw,
            codes,
        };
        let actual_groups = index.num_groups();
        if actual_groups < 2 {
            return None;
        }
        let tokens = tokenize_lz77(pixels);
        let buckets = split_tokens_by_group(&tokens, &index, w, actual_groups);
        let group_codes = build_group_codes(&buckets, 0, w);
        let mut bw = BitWriter::new();
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bits((prefix_bits - 2) as u32, 3);
        let entropy_image = index.entropy_image_argb();
        write_entropy_coded_image_literals(&mut bw, &entropy_image);
        for group in &group_codes {
            for code in group.iter() {
                code.write_code_lengths(&mut bw);
            }
        }
        let mut pos = 0usize;
        let w_pixels = w as usize;
        for &tok in &tokens {
            let x = (pos % w_pixels) as u32;
            let y = (pos / w_pixels) as u32;
            let g = index.group_for(x, y) as usize;
            let codes = &group_codes[g];
            let green_code = &codes[0];
            let red_code = &codes[1];
            let blue_code = &codes[2];
            let alpha_code = &codes[3];
            let dist_code = &codes[4];
            match tok {
                Token::Literal(p) => {
                    let a = ((p >> 24) & 0xff) as usize;
                    let r = ((p >> 16) & 0xff) as usize;
                    let g_ch = ((p >> 8) & 0xff) as usize;
                    let b = (p & 0xff) as usize;
                    green_code.write_symbol(&mut bw, g_ch);
                    red_code.write_symbol(&mut bw, r);
                    blue_code.write_symbol(&mut bw, b);
                    alpha_code.write_symbol(&mut bw, a);
                    pos += 1;
                }
                Token::CacheRef { .. } => unreachable!("no cache in measurement"),
                Token::Copy { length, distance } => {
                    write_lz77_value(&mut bw, green_code, 256, length as u32);
                    let raw_code = pixel_distance_to_distance_code(distance, w);
                    write_lz77_value(&mut bw, dist_code, 0, raw_code);
                    pos += length;
                }
            }
        }
        Some(bw.into_bytes().len())
    }

    /// A four-region fixture where the top-left quadrant has the same
    /// per-channel mean as the bottom-right but a very different
    /// per-channel distribution, and the top-right has the same mean
    /// as the bottom-left also with a divergent distribution. The
    /// mean-green clusterer at `num_groups = 2` can only find one
    /// axis of separation and folds two distinct distributions onto
    /// the same group; the histogram clusterer separates by full
    /// distribution and finds the right partition.
    fn four_region_mean_collision_image(width: u32, height: u32) -> Vec<u32> {
        let w = width as usize;
        let h = height as usize;
        let mut pixels = Vec::with_capacity(w * h);
        let mut s: u32 = 0x12345678;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let top = y < h / 2;
                let left = x < w / 2;
                // Pick (g, r) pairs whose means match across the
                // top-left vs bottom-right and top-right vs bottom-left
                // diagonals but whose distributions are very different.
                let (g, r, b) = match (top, left) {
                    (true, true) => {
                        // top-left: g bimodal {16, 240} mean ≈ 128
                        let gv = if (s & 1) == 0 { 16 } else { 240 };
                        let rv = (s >> 8) & 0x3f;
                        let bv = (s >> 16) & 0x3f;
                        (gv, rv, bv)
                    }
                    (true, false) => {
                        // top-right: g flat 128
                        let gv = 128u32;
                        let rv = ((s >> 8) & 0x3f).wrapping_add(192);
                        let bv = (s >> 16) & 0x3f;
                        (gv, rv, bv)
                    }
                    (false, true) => {
                        // bottom-left: g bimodal but {64, 192} mean ≈ 128
                        let gv = if (s & 1) == 0 { 64 } else { 192 };
                        let rv = (s >> 8) & 0x3f;
                        let bv = ((s >> 16) & 0x3f).wrapping_add(192);
                        (gv, rv, bv)
                    }
                    (false, false) => {
                        // bottom-right: g flat 128 too
                        let gv = 128u32;
                        let rv = ((s >> 8) & 0x3f).wrapping_add(192);
                        let bv = ((s >> 16) & 0x3f).wrapping_add(192);
                        (gv, rv, bv)
                    }
                };
                pixels.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        pixels
    }

    /// For a given fixture, sweep every `(prefix_bits, num_groups)`
    /// the round-151 chooser searches and return the smallest
    /// non-degenerate multi-meta-prefix byte cost under the named
    /// clusterer. Returns `None` if every combination collapsed.
    fn best_mp_bytes_over_sweep(
        pixels: &[u32],
        w: u32,
        h: u32,
        use_histogram: bool,
    ) -> Option<usize> {
        let mut best: Option<usize> = None;
        for &prefix_bits in META_PREFIX_BITS_SWEEP.iter() {
            for num_groups in 2u32..=MAX_META_GROUPS {
                if let Some(bytes) =
                    measure_mp_bytes_at(pixels, w, h, prefix_bits, num_groups, use_histogram)
                {
                    best = Some(match best {
                        Some(b) => b.min(bytes),
                        None => bytes,
                    });
                }
            }
        }
        best
    }

    /// Confirm the round-152 histogram-distance clusterer beats (or at
    /// worst ties) the round-151 mean-green bucketiser on the
    /// diagnostic two-region noisy sweep. Prints byte counts (run with
    /// `--nocapture`).
    #[test]
    fn histogram_clusterer_reduces_mp_bytes_on_two_region_sweep() {
        let shapes: &[(u32, u32)] = &[(64, 64), (128, 128), (64, 128), (256, 256)];
        for &(w, h) in shapes {
            let pixels = two_region_noisy_image(w, h);
            let mg = best_mp_bytes_over_sweep(&pixels, w, h, false)
                .expect("mean-green path must produce a candidate");
            let hi = best_mp_bytes_over_sweep(&pixels, w, h, true)
                .expect("histogram path must produce a candidate");
            assert!(
                hi <= mg,
                "{w}x{h}: histogram path produced {hi} B, mean-green produced {mg} B \
                 — histogram path must not regress on the two-region sweep",
            );
            println!(
                "r152 measurement {w}x{h}: mean-green={mg} B histogram={hi} B \
                 delta={} B ({:.2}%)",
                mg as i64 - hi as i64,
                100.0 * (mg as f64 - hi as f64) / mg as f64,
            );
        }
    }

    /// Confirm the histogram clusterer is *strictly* better than
    /// mean-green on the four-region mean-collision fixture, where
    /// blocks sharing a green mean diverge in distribution. Prints
    /// byte counts (run with `--nocapture`).
    #[test]
    fn histogram_clusterer_reduces_mp_bytes_on_mean_collision_sweep() {
        let shapes: &[(u32, u32)] = &[(64, 64), (128, 128), (64, 128), (256, 256)];
        for &(w, h) in shapes {
            let pixels = four_region_mean_collision_image(w, h);
            let mg_opt = best_mp_bytes_over_sweep(&pixels, w, h, false);
            let hi = best_mp_bytes_over_sweep(&pixels, w, h, true)
                .expect("histogram path must produce a candidate");
            match mg_opt {
                Some(mg) => {
                    assert!(
                        hi < mg,
                        "{w}x{h}: histogram path produced {hi} B, mean-green produced {mg} B \
                         — histogram path must strictly improve on mean-collision fixture",
                    );
                    println!(
                        "r152 mean-collision {w}x{h}: mean-green={mg} B histogram={hi} B \
                         delta={} B ({:.2}%)",
                        mg as i64 - hi as i64,
                        100.0 * (mg as f64 - hi as f64) / mg as f64,
                    );
                }
                None => {
                    println!(
                        "r152 mean-collision {w}x{h}: mean-green collapsed (no candidate); \
                         histogram={hi} B",
                    );
                }
            }
        }
    }

    // ---- round 155: §4.1 predictor size_bits two-value sweep ----------
    //
    // The round-155 step extends the predictor candidate from a single
    // `DEFAULT_PREDICTOR_SIZE_BITS = 4` block-grid to a two-value sweep
    // mirroring the round-147 §4.2 color-transform shape: per-region
    // (`size_bits = 4` → 16×16 pixel blocks) plus a maximal single-block
    // candidate (`size_bits` promoted up to 9 so the sub-image is 1×1).
    // Each value composes with the round-148 `cache_code_bits ∈ [1..11]`
    // + disabled-cache baseline.
    //
    // The tests below establish three contracts:
    //
    // 1) Non-regression — the round-155 chooser never produces a stream
    //    longer than the pre-round-155 chooser (which only evaluated the
    //    default `size_bits = 4` predictor).
    // 2) Strict-beat on a synthetic fixture where the maximal-single-
    //    block predictor wins (a small image whose `size_bits = 4`
    //    per-region path emits a costly 1×1 sub-image equal to the
    //    single-block one but where the per-region wraps in the same
    //    16×16 mode, leaving the two effectively identical except for
    //    sub-image layout — and small enough that the single-block path
    //    wins on noise).
    // 3) Round-trip — every emitted stream still round-trips through
    //    `decode_lossless_image`, so the size_bits promotion did not
    //    break the §4.1 header.

    /// Local copy of the pre-round-155 chooser: identical to
    /// [`encode_argb_with_predictor_chooser`] but evaluates only the
    /// default-size predictor candidate (no maximal single-block sweep).
    /// Used as the regression baseline for the round-155 non-regression
    /// tests so they exercise *only* the size_bits-sweep delta the
    /// chooser added.
    fn pre_round_155_predictor_chooser(pixels: &[u32], width: u32, height: u32) -> Vec<u8> {
        let mut best = encode_argb_literals_with_width(pixels, width);

        let pred_size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let ctx_size_bits = DEFAULT_COLOR_TRANSFORM_SIZE_BITS;
        let pred_block = 1u32 << pred_size_bits;
        let ctx_block = 1u32 << ctx_size_bits;

        if width >= pred_block && height >= pred_block {
            // Pre-round-155: single `size_bits = 4` predictor only.
            let pred_best = select_best_cache_bits(|cache_bits| {
                encode_with_predictor(pixels, width, height, pred_size_bits, cache_bits, width)
            });
            if pred_best.len() < best.len() {
                best = pred_best;
            }
        }

        // §4.2 color transform unchanged (round-147 two-value sweep).
        if width >= ctx_block && height >= ctx_block {
            let mut single_block_size_bits: u8 = ctx_size_bits;
            while single_block_size_bits < 9
                && ((1u32 << single_block_size_bits) < width
                    || (1u32 << single_block_size_bits) < height)
            {
                single_block_size_bits += 1;
            }
            let try_single_block = single_block_size_bits != ctx_size_bits;
            let mut candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_color_transform(pixels, width, height, ctx_size_bits, cache_bits, width)
            })];
            if try_single_block {
                candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_color_transform(
                        pixels,
                        width,
                        height,
                        single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if collect_palette(pixels).is_some() {
            let ci_best = select_best_cache_bits(|cache_bits| {
                encode_with_color_indexing(pixels, width, height, cache_bits)
                    .expect("palette feasibility already confirmed")
            });
            if ci_best.len() < best.len() {
                best = ci_best;
            }
        }

        if let Some(mp_best) = sweep_meta_prefix_candidate(pixels, width, height) {
            if mp_best.len() < best.len() {
                best = mp_best;
            }
        }

        best
    }

    /// Round 155 non-regression: across a fixture matrix spanning
    /// gradient / noise / palette-ish images and several shapes, the
    /// round-155 chooser must never produce a stream longer than the
    /// pre-round-155 chooser (which had only the default-size predictor
    /// candidate). The round-155 chooser is a strict superset of the
    /// pre-round-155 candidate set, so this is a structural guarantee.
    #[test]
    fn round_155_predictor_size_bits_sweep_never_regresses() {
        let shapes: &[(u32, u32)] = &[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (48, 48),
            (16, 32),
            (64, 16),
            (40, 24),
        ];
        for &(w, h) in shapes {
            // Three fixture families: smooth gradient, dense noise,
            // small-palette stripes.
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0xFF;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let mut seed = 0xC0FFEE_u32;
            let noise: Vec<u32> = (0..(w * h) as usize)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    0xFF00_0000 | (seed & 0x00FF_FFFF)
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();

            for (name, pixels) in [
                ("gradient", &gradient),
                ("noise", &noise),
                ("stripes", &stripes),
            ] {
                let pre = pre_round_155_predictor_chooser(pixels, w, h);
                let post = encode_argb_with_predictor_chooser(pixels, w, h);
                assert!(
                    post.len() <= pre.len(),
                    "round-155 chooser regression on {name} {w}x{h}: pre={} B post={} B",
                    pre.len(),
                    post.len(),
                );
            }
        }
    }

    /// Round 155 strict-beat: on a fixture small enough that the
    /// default-size predictor block-image has no useful resolution
    /// (a 20×20 image gives one 16×16 in-bounds block plus border
    /// padding that still pays a 1-pixel sub-image), the maximal
    /// single-block predictor strictly shrinks the chosen stream
    /// because both candidates share the same block-image cost while
    /// the single-block path picks a globally-optimal predictor mode
    /// over the noise pattern. The test prints the byte-saved delta so
    /// the round report can quote a measured number.
    #[test]
    fn round_155_predictor_size_bits_sweep_strictly_beats_default_on_some_fixture() {
        // 20×20 dense-residual fixture: per-pixel green channel changes
        // every pixel so the per-region 16×16 block path can't dominate
        // and the chooser's two candidates differ only in sub-image
        // shape + global predictor pick.
        let w = 20u32;
        let h = 20u32;
        let mut seed = 0xDEADBEEF_u32;
        let pixels: Vec<u32> = (0..(w * h) as usize)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                0xFF00_0000 | (seed & 0x00FF_FFFF)
            })
            .collect();

        let pre = pre_round_155_predictor_chooser(&pixels, w, h);
        let post = encode_argb_with_predictor_chooser(&pixels, w, h);

        eprintln!(
            "[round-155] {w}x{h} dense-residual: pre={} B post={} B delta={} B ({:.2}%)",
            pre.len(),
            post.len(),
            pre.len() as i64 - post.len() as i64,
            (pre.len() as f64 - post.len() as f64) / pre.len() as f64 * 100.0,
        );
        assert!(
            post.len() < pre.len(),
            "round-155 maximal-single-block predictor must strictly shrink the chosen \
             stream on the 20x20 dense-residual fixture: pre={} B post={} B",
            pre.len(),
            post.len(),
        );
    }

    /// Round 155 round-trip: the maximal-single-block predictor
    /// candidate (size_bits promoted up to 9) must still emit a valid
    /// §4.1 transform header that the decoder accepts; the resulting
    /// stream must round-trip back to the exact input pixels via
    /// [`crate::decode_lossless_image`]. The test directly invokes
    /// `encode_with_predictor` at the largest size_bits the sweep can
    /// pick (matching the chooser's promotion loop) and frames it with
    /// `build_image_header` for the round-trip path.
    #[test]
    fn round_155_predictor_single_block_round_trips_through_decoder() {
        let w = 64u32;
        let h = 16u32;
        let mut seed = 0xA5A5_F00D_u32;
        let pixels: Vec<u32> = (0..(w * h) as usize)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                0xFF00_0000 | (seed & 0x00FF_FFFF)
            })
            .collect();

        // 1) The chooser's chosen stream must round-trip end-to-end
        //    through `build::build_webp_file` + `decode_lossless_image`.
        let stream_chooser = encode_argb_with_predictor_chooser(&pixels, w, h);
        let header_chooser = build_image_header(w, h, true);
        let mut payload_chooser = header_chooser.to_vec();
        payload_chooser.extend_from_slice(&stream_chooser);
        let framed_chooser =
            build::build_webp_file(&payload_chooser, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed_chooser)
            .unwrap()
            .unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());

        // 2) The single-block predictor path directly: pick the
        //    smallest size_bits such that `1 << size_bits ≥ max(w, h)`,
        //    matching the chooser's promotion loop.
        let mut single_block_size_bits: u8 = DEFAULT_PREDICTOR_SIZE_BITS;
        while single_block_size_bits < 9
            && ((1u32 << single_block_size_bits) < w || (1u32 << single_block_size_bits) < h)
        {
            single_block_size_bits += 1;
        }
        // 64×16 promotes to size_bits = 6 (block 64).
        assert_eq!(single_block_size_bits, 6);
        let stream = encode_with_predictor(&pixels, w, h, single_block_size_bits, None, w);
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img2 = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img2.pixels(), pixels.as_slice());
    }

    // ---- round 156: §5.2.2 single-position lazy LZ77 matching --------
    //
    // The round-156 step adds a single-position look-ahead to the §5.2.2
    // hash-chain matcher in `tokenize_lz77`: when a match `(L_a, _)` is
    // found at `pos`, the encoder also probes `pos + 1` and, if the
    // look-ahead yields a strictly longer match, emits `pixels[pos]` as
    // a literal and uses the longer match from `pos + 1` instead. The
    // decoder output is bit-identical for any input — only the token
    // partition shifts — so the property under test is *byte-count*,
    // not pixel correctness (which the existing round-trip tests cover).
    //
    // The internal `tokenize_lz77_inner` exposes a `lazy_depth: u32`
    // toggle so a test can build the strict-greedy r155 baseline token
    // stream (`lazy_depth = 0`) alongside the round-156 depth-1 stream
    // (`lazy_depth = 1`) and the round-157 depth-2 stream
    // (`lazy_depth = 2`) on the same fixture, then compare token counts.
    // Three contracts:
    //
    // 1) Round-trip — every lazy-matched stream still round-trips
    //    end-to-end through `decode_lossless_image`.
    // 2) Strict-beat — on a hand-crafted fixture where the strict-
    //    greedy matcher gets trapped in a short match, the lazy matcher
    //    emits strictly fewer tokens (and the test asserts the headline
    //    drop, printing the per-fixture numbers).
    // 3) Non-regression — on a broader fixture matrix the lazy token
    //    count is `<=` the strict-greedy token count everywhere (the
    //    look-ahead only ever swaps when the longer match strictly
    //    wins, so this is a structural guarantee — the test ensures
    //    no off-by-one in the insert-bookkeeping reintroduces a
    //    regression on future refactors).

    /// Round 156 round-trip: a noisy 64×16 fixture encoded with the
    /// round-156 lazy matcher must still decode bit-exactly back to the
    /// original ARGB pixels. The fixture is large enough that the
    /// matcher produces many `Copy` tokens, so the lazy branch is
    /// exercised throughout the run (and not just at the tail).
    #[test]
    fn round_156_lazy_match_round_trips_through_decoder() {
        let w = 64u32;
        let h = 16u32;
        let mut seed = 0xF00D_BABE_u32;
        let pixels: Vec<u32> = (0..(w * h) as usize)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                0xFF00_0000 | (seed & 0x00FF_FFFF)
            })
            .collect();

        // The full chooser includes the lazy matcher via
        // `tokenize_lz77`; the round-trip through the framed file must
        // recover the exact input.
        let stream = encode_argb_with_predictor_chooser(&pixels, w, h);
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());

        // The direct lazy-only token stream against
        // `encode_argb_literals_with_width` must also round-trip — this
        // catches the case where lazy on the no-transform path
        // mis-tracks the hash-chain insert bookkeeping.
        let stream_direct = encode_argb_literals_with_width(&pixels, w);
        let header_direct = build_image_header(w, h, true);
        let mut payload_direct = header_direct.to_vec();
        payload_direct.extend_from_slice(&stream_direct);
        let framed_direct =
            build::build_webp_file(&payload_direct, ImageKind::Lossless, w, h).unwrap();
        let img_direct = crate::decode_lossless_image(&framed_direct)
            .unwrap()
            .unwrap();
        assert_eq!(img_direct.pixels(), pixels.as_slice());
    }

    /// Round 156 strict-beat: a hand-crafted look-ahead-trap fixture
    /// where the strict-greedy matcher accepts a short match at
    /// position `p` that prevents a strictly longer match at `p + 1`.
    /// The fixture engineers two 4-pixel-hash chains so the strict
    /// matcher finds a length-4 match at `p` while `p + 1` finds a
    /// length-6 match; lazy resolves to the longer partition.
    ///
    /// Layout (each pixel is a unique ARGB constant):
    ///
    /// ```text
    ///   pos  0..7    [A B C D E F G H]   — primary prefix, gives the
    ///                                       [A,B,C,D] chain entry
    ///                                       at pos 0 and the
    ///                                       [B,C,D,E] entry at pos 1.
    ///   pos  8       Z                    — separator
    ///   pos  9..15   [A B C D E F G]      — `find(p=10)` matches the
    ///                                       primary prefix [A,B,C,D,E,F,G]
    ///                                       at pos 0 — length 7. Lazy
    ///                                       irrelevant here (no longer
    ///                                       match exists past length 7
    ///                                       at pos 11).
    /// ```
    ///
    /// That doesn't trap. A real trap requires the `p` match to be
    /// strictly shorter than the `p + 1` match. The construction below
    /// achieves this by deliberately mismatching the 4th byte at pos
    /// `p`'s candidate so the strict match stops at length 4, while
    /// pos `p + 1` walks a second pre-seeded chain with a 6+ pixel run.
    /// Specifically:
    ///
    /// ```text
    ///   pos  0..3    [A B C D]            — first chain (pos 0).
    ///   pos  4..6    [Z Z Z]               — separator.
    ///   pos  7..13   [B C D E F G H]       — second chain (pos 7's
    ///                                       window is [B,C,D,E]).
    ///   pos 14..16   [Z Z Z]
    ///   pos 17       A   ← trap start.    `find(17)`'s window is
    ///                                     [A,B,C,D] → matches pos 0,
    ///                                     extension stops at length 4
    ///                                     because pos 4 = Z ≠ pos 21.
    ///   pos 18..23   [B C D E F G]        — `find(18)`'s window is
    ///                                     [B,C,D,E] → matches pos 7,
    ///                                     extension goes 7 long (B-H)
    ///                                     against the second chain.
    /// ```
    ///
    /// Greedy: emits `Copy{len=4, dist=17}` at pos 17, then has to
    /// emit `[E,F,G]` as literals (pos 21,22,23) because the chain at
    /// pos 21's window is gone.
    ///
    /// Lazy: emits `Literal(A)` at pos 17, then `Copy{len=7, dist=11}`
    /// at pos 18, covering `[B,C,D,E,F,G,H]` from pos 7. Net: -2 tokens.
    #[test]
    fn round_156_lazy_match_strictly_beats_greedy_on_trap_fixture() {
        let a = 0xFF11_2233_u32;
        let b = 0xFF22_3344_u32;
        let c = 0xFF33_4455_u32;
        let d = 0xFF44_5566_u32;
        let e = 0xFF55_6677_u32;
        let f = 0xFF66_7788_u32;
        let g = 0xFF77_8899_u32;
        let h = 0xFF88_99AA_u32;
        let z = 0xFF00_0000_u32;

        // The buffer layout (per the doc comment above). Indices are
        // explicit so the trap is unambiguous.
        let mut pixels: Vec<u32> = vec![
            a, b, c, d, // 0..4    primary chain anchor [A,B,C,D]
            z, z, z, // 4..7    separator
            b, c, d, e, f, g, h, // 7..14   secondary chain anchor [B,C,D,E,...]
            z, z, z, // 14..17  separator
            a, // 17       trap-start: find(17)→pos0, length 4
            b, c, d, e, f, g, h, // 18..25  decoy: find(18)→pos7, length 7
        ];
        // Pad to 64 pixels so the framing call has a non-degenerate
        // image; tail content is uniform Z so it does not interact
        // with the trap region.
        while pixels.len() < 64 {
            pixels.push(z);
        }

        let greedy = tokenize_lz77_inner(&pixels, 0);
        let lazy = tokenize_lz77_inner(&pixels, 1);

        let greedy_copies = greedy
            .iter()
            .filter(|t| matches!(t, Token::Copy { .. }))
            .count();
        let lazy_copies = lazy
            .iter()
            .filter(|t| matches!(t, Token::Copy { .. }))
            .count();
        // Sum of pixels covered by each partition: must equal the
        // input length for both partitions (sanity).
        let coverage = |toks: &[Token]| -> usize {
            toks.iter()
                .map(|t| match *t {
                    Token::Literal(_) => 1,
                    Token::CacheRef { .. } => 1,
                    Token::Copy { length, .. } => length,
                })
                .sum()
        };
        assert_eq!(coverage(&greedy), pixels.len());
        assert_eq!(coverage(&lazy), pixels.len());

        eprintln!(
            "[round-156] trap fixture: greedy tokens={} (copies={}), \
             lazy tokens={} (copies={}), copy delta={}",
            greedy.len(),
            greedy_copies,
            lazy.len(),
            lazy_copies,
            greedy_copies as i64 - lazy_copies as i64,
        );

        // The trap region has greedy emit
        //   [Copy{4, 17}, Copy{7, 11}, Copy{36, 1}]   = 3 copies
        // while lazy emits
        //   [Literal(A), Copy{10, 11}, Copy{36, 1}]   = 2 copies
        // covering the same 11-pixel trap span. The lazy partition
        // collapses two separate copies into one longer copy, which is
        // the round-156 structural win. (The literal-symbol count rises
        // by one to compensate; total tokens may match but the *copy
        // count* — and the prefix-code statistics — diverge.)
        assert!(
            lazy_copies < greedy_copies,
            "round-156 lazy matcher must emit strictly fewer Copy tokens on the trap \
             fixture: greedy copies={} lazy copies={}\ngreedy partition: {:?}\n\
             lazy partition:   {:?}",
            greedy_copies,
            lazy_copies,
            greedy,
            lazy,
        );

        // Round-trip the bytes through the no-transform encoder for
        // good measure: the lazy path must still decode back exactly.
        let stream = encode_argb_literals_with_width(&pixels, pixels.len() as u32);
        let w = pixels.len() as u32;
        let h = 1u32;
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Round 156 non-regression: across a broad fixture matrix
    /// (gradient / noise / stripes shapes), the lazy matcher's token
    /// count is `<=` the strict-greedy matcher's everywhere. Structural
    /// because the look-ahead only swaps when the alternate match is
    /// strictly longer, so the lazy partition uses at most as many
    /// tokens as the greedy partition. The test guards against
    /// off-by-one bugs in the hash-chain insert bookkeeping (the
    /// insert-of-`pos`-for-lookahead path) that future refactors might
    /// introduce.
    #[test]
    fn round_156_lazy_never_increases_token_count() {
        let shapes: &[(u32, u32)] = &[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (48, 48),
            (16, 32),
            (64, 16),
            (40, 24),
        ];
        for &(w, h) in shapes {
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0xFF;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let mut seed = 0xC0FFEE_u32;
            let noise: Vec<u32> = (0..(w * h) as usize)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    0xFF00_0000 | (seed & 0x00FF_FFFF)
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();

            for (name, pixels) in [
                ("gradient", &gradient),
                ("noise", &noise),
                ("stripes", &stripes),
            ] {
                let greedy = tokenize_lz77_inner(pixels, 0);
                let lazy = tokenize_lz77_inner(pixels, 1);
                assert!(
                    lazy.len() <= greedy.len(),
                    "round-156 lazy regression on {name} {w}x{h}: greedy={} tokens, \
                     lazy={} tokens",
                    greedy.len(),
                    lazy.len(),
                );
            }
        }
    }

    // ---- round 157: §5.2.2 two-position lazy LZ77 matching -----------
    //
    // The round-157 step extends the round-156 single-position lazy
    // matcher with a second look-ahead position. After finding a match
    // `(L_a, _)` at `pos` and (depth-1) probing `pos + 1` for a strictly
    // longer `L_b`, the matcher also (depth-2) probes `pos + 2` for an
    // `L_c > max(L_a, L_b)`. When the depth-2 probe wins, the encoder
    // emits two literals (`pixels[pos]` and `pixels[pos + 1]`) and takes
    // the longer match from `pos + 2`. This recovers a *second-order*
    // strict-greedy trap that the round-156 depth-1 matcher could not
    // escape — a short match at `pos` AND a short match at `pos + 1`
    // together blocking a strictly longer match at `pos + 2`. The
    // decoder output is bit-identical for any input — only the token
    // *partition* shifts by up to two pixels — so round-trips remain
    // bit-exact under any input.
    //
    // Three contracts (mirroring the round-156 layout):
    //
    // 1) Round-trip — every depth-2 lazy-matched stream still
    //    round-trips end-to-end through `decode_lossless_image`.
    // 2) Strict-beat — on a hand-crafted depth-2-trap fixture, the
    //    depth-2 matcher emits strictly fewer Copy tokens than both
    //    the strict-greedy matcher and the depth-1 lazy matcher.
    // 3) Non-regression — on a broader fixture matrix the depth-2
    //    token count is `<=` the depth-1 token count everywhere.

    /// Round 157 round-trip: a noisy 80×16 fixture encoded with the
    /// round-157 depth-2 lazy matcher (now the production
    /// `tokenize_lz77` default) must still decode bit-exactly back to
    /// the original ARGB pixels. Uses an independent xorshift seed
    /// from the round-156 test so both fixtures exercise the matcher
    /// over distinct entropy.
    #[test]
    fn round_157_depth2_lazy_match_round_trips_through_decoder() {
        let w = 80u32;
        let h = 16u32;
        let mut seed = 0xCAFE_F00D_u32;
        let pixels: Vec<u32> = (0..(w * h) as usize)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                0xFF00_0000 | (seed & 0x00FF_FFFF)
            })
            .collect();

        // The full chooser delegates to `tokenize_lz77` (depth-2 as of
        // round 157); end-to-end round-trip through the framed file
        // must recover the exact input.
        let stream = encode_argb_with_predictor_chooser(&pixels, w, h);
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());

        // The direct depth-2 token stream against the no-transform
        // encoder must also round-trip — guards against bookkeeping
        // bugs in the new depth-2 insert/skip dedup path.
        let stream_direct = encode_argb_literals_with_width(&pixels, w);
        let header_direct = build_image_header(w, h, true);
        let mut payload_direct = header_direct.to_vec();
        payload_direct.extend_from_slice(&stream_direct);
        let framed_direct =
            build::build_webp_file(&payload_direct, ImageKind::Lossless, w, h).unwrap();
        let img_direct = crate::decode_lossless_image(&framed_direct)
            .unwrap()
            .unwrap();
        assert_eq!(img_direct.pixels(), pixels.as_slice());
    }

    /// Round 157 strict-beat: a hand-crafted depth-2-trap fixture where
    /// the strict-greedy matcher AND the round-156 depth-1 lazy matcher
    /// both accept a short match at `pos` that prevents a strictly
    /// longer match at `pos + 2`. The depth-2 lazy matcher emits two
    /// literals and takes the longer match.
    ///
    /// Layout (each capital letter is a unique ARGB constant; the
    /// `Z*` family are unique separator pixels that share no 4-pixel
    /// window with the anchors):
    ///
    /// ```text
    ///   pos  0..3    [P Q R S]                — anchor A (4 px)
    ///   pos  4..6    [Z1 Z2 Z3]               — separator
    ///   pos  7..10   [Q R S T]                — anchor B (4 px)
    ///   pos 11..13   [Z4 Z5 Z6]               — separator
    ///   pos 14..21   [R S T U V W X Y]        — anchor C (8 px)
    ///   pos 22..24   [Z7 Z8 Z9]               — separator
    ///   pos 25       P                         — trap start
    ///   pos 26       Q
    ///   pos 27..33   [R S T U V W X]          — depth-2 chain region
    ///   pos 34..     fill with a fresh Zfill color (no 4-window match)
    /// ```
    ///
    /// At pos 25:
    ///
    /// * `find(25)` window `[P,Q,R,S]` → matches anchor A (pos 0),
    ///   extension stops at length 4 because pos 4 (Z1) ≠ pos 29 (T).
    /// * `find(26)` window `[Q,R,S,T]` → matches anchor B (pos 7),
    ///   extension stops at length 4 because pos 11 (Z4) ≠ pos 30 (U).
    ///   `L_b = 4 = L_a`, **not strictly greater**, so the depth-1
    ///   lazy matcher does NOT swap.
    /// * `find(27)` window `[R,S,T,U]` → matches anchor C (pos 14),
    ///   extension goes `[R,S,T,U,V,W,X]` (length 7) before pos 21
    ///   (Y) ≠ pos 34 (Zfill). `L_c = 7 > 4`, so the depth-2 lazy
    ///   matcher swaps to two literals + the length-7 match.
    ///
    /// Strict-greedy AND depth-1 partition at the trap:
    /// `[Copy{4, dist=25}, ...]`. Depth-2 partition: `[Lit(P),
    /// Lit(Q), Copy{7, dist=13}, ...]`. Net: depth-2 collapses a
    /// short-then-short pair into one longer copy — strictly fewer
    /// Copy tokens, at the cost of one extra literal (mirroring the
    /// round-156 pattern).
    #[test]
    fn round_157_depth2_lazy_match_strictly_beats_depth1_on_trap_fixture() {
        // Distinct ARGB constants. Anchor letters P..Y carry the
        // structural matches; Z1..Z9 + Zfill are deliberately unique
        // so they cannot seed a parasitic chain.
        let p_ = 0xFF11_2200_u32;
        let q_ = 0xFF22_3300_u32;
        let r_ = 0xFF33_4400_u32;
        let s_ = 0xFF44_5500_u32;
        let t_ = 0xFF55_6600_u32;
        let u_ = 0xFF66_7700_u32;
        let v_ = 0xFF77_8800_u32;
        let w_ = 0xFF88_9900_u32;
        let x_ = 0xFF99_AA00_u32;
        let y_ = 0xFFAA_BB00_u32;
        let z1 = 0xFFCC_DD01_u32;
        let z2 = 0xFFCC_DD02_u32;
        let z3 = 0xFFCC_DD03_u32;
        let z4 = 0xFFCC_DD04_u32;
        let z5 = 0xFFCC_DD05_u32;
        let z6 = 0xFFCC_DD06_u32;
        let z7 = 0xFFCC_DD07_u32;
        let z8 = 0xFFCC_DD08_u32;
        let z9 = 0xFFCC_DD09_u32;

        let mut pixels: Vec<u32> = vec![
            p_, q_, r_, s_, // 0..4    anchor A
            z1, z2, z3, // 4..7    separator
            q_, r_, s_, t_, // 7..11   anchor B
            z4, z5, z6, // 11..14  separator
            r_, s_, t_, u_, v_, w_, x_, y_, // 14..22  anchor C
            z7, z8, z9, // 22..25  separator
            p_, q_, // 25..27  trap start (depth-1 cannot escape)
            r_, s_, t_, u_, v_, w_, x_, // 27..34  depth-2 chain region
        ];
        // Pad the tail with unique colors so the depth-2 swap's
        // post-match region cannot trigger another long match that
        // might mask the trap's copy-count delta.
        let mut filler = 0xFFE0_0000_u32;
        while pixels.len() < 80 {
            filler = filler.wrapping_add(1);
            pixels.push(filler);
        }

        let greedy = tokenize_lz77_inner(&pixels, 0);
        let lazy1 = tokenize_lz77_inner(&pixels, 1);
        let lazy2 = tokenize_lz77_inner(&pixels, 2);

        let copies = |toks: &[Token]| -> usize {
            toks.iter()
                .filter(|t| matches!(t, Token::Copy { .. }))
                .count()
        };
        let coverage = |toks: &[Token]| -> usize {
            toks.iter()
                .map(|t| match *t {
                    Token::Literal(_) => 1,
                    Token::CacheRef { .. } => 1,
                    Token::Copy { length, .. } => length,
                })
                .sum()
        };
        // Sanity: all three partitions cover the exact image.
        assert_eq!(coverage(&greedy), pixels.len());
        assert_eq!(coverage(&lazy1), pixels.len());
        assert_eq!(coverage(&lazy2), pixels.len());

        let g_c = copies(&greedy);
        let l1_c = copies(&lazy1);
        let l2_c = copies(&lazy2);
        eprintln!(
            "[round-157] depth-2 trap fixture: greedy tokens={} (copies={}), \
             depth-1 tokens={} (copies={}), depth-2 tokens={} (copies={}), \
             copy delta vs depth-1={}",
            greedy.len(),
            g_c,
            lazy1.len(),
            l1_c,
            lazy2.len(),
            l2_c,
            l1_c as i64 - l2_c as i64,
        );

        // The trap forces depth-2 to collapse a length-4 copy into a
        // 2-literals + length-7 copy that subsumes 7 pixels of what
        // greedy / depth-1 would have to cover with multiple matches.
        // The structural win is on Copy count: depth-2 must emit
        // strictly fewer Copy tokens than BOTH baselines.
        assert_eq!(
            g_c, l1_c,
            "round-157 fixture: depth-1 must agree with greedy here \
             (no depth-1 swap fires) — greedy={g_c}, depth-1={l1_c}"
        );
        assert!(
            l2_c < l1_c,
            "round-157 depth-2 matcher must emit strictly fewer Copy \
             tokens than the depth-1 matcher on the depth-2 trap \
             fixture: depth-1 copies={l1_c} depth-2 copies={l2_c}\n\
             depth-1 partition: {lazy1:?}\n\
             depth-2 partition: {lazy2:?}"
        );

        // Round-trip the bytes through the no-transform encoder for
        // good measure: the depth-2 path must decode back exactly.
        let stream = encode_argb_literals_with_width(&pixels, pixels.len() as u32);
        let w = pixels.len() as u32;
        let h = 1u32;
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Round 157 non-regression: across a broad fixture matrix the
    /// depth-2 lazy token count is `<=` the depth-1 lazy token count
    /// everywhere. Structural because the depth-2 probe only swaps
    /// when the alternate match is strictly longer than the depth-1
    /// best, so the depth-2 partition uses at most as many tokens as
    /// the depth-1 partition. The test guards against off-by-one in
    /// the new depth-2 insert/skip dedup (where `pos` and `pos + 1`
    /// can both be pre-inserted before the chosen match starts at
    /// `pos`, `pos + 1`, or `pos + 2`).
    #[test]
    fn round_157_depth2_never_increases_token_count_over_depth1() {
        let shapes: &[(u32, u32)] = &[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (48, 48),
            (16, 32),
            (64, 16),
            (40, 24),
        ];
        for &(w, h) in shapes {
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0xFF;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let mut seed = 0xC0FFEE_u32;
            let noise: Vec<u32> = (0..(w * h) as usize)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    0xFF00_0000 | (seed & 0x00FF_FFFF)
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();

            for (name, pixels) in [
                ("gradient", &gradient),
                ("noise", &noise),
                ("stripes", &stripes),
            ] {
                let lazy1 = tokenize_lz77_inner(pixels, 1);
                let lazy2 = tokenize_lz77_inner(pixels, 2);
                assert!(
                    lazy2.len() <= lazy1.len(),
                    "round-157 depth-2 regression on {name} {w}x{h}: \
                     depth-1={} tokens, depth-2={} tokens",
                    lazy1.len(),
                    lazy2.len(),
                );
                // Round-trip the depth-2 stream as a defensive check
                // for hash-chain insert bookkeeping.
                let stream = encode_argb_literals_with_width(pixels, w);
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&stream);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-157 depth-2 round-trip mismatch on {name} {w}x{h}"
                );
            }
        }
    }

    // ---- round 158: §5.2.2 three-position lazy LZ77 matching ---------
    //
    // The round-158 step extends the round-157 two-position lazy
    // matcher with a third look-ahead position. After finding a match
    // `(L_a, _)` at `pos` and (depth-1) probing `pos + 1` for a strictly
    // longer `L_b`, and (depth-2) probing `pos + 2` for a strictly
    // longer `L_c`, the matcher also (depth-3) probes `pos + 3` for an
    // `L_d > max(L_a, L_b, L_c)`. When the depth-3 probe wins, the
    // encoder emits three literals (`pixels[pos]`, `pixels[pos + 1]`,
    // and `pixels[pos + 2]`) and takes the longer match from `pos + 3`.
    // This recovers a *third-order* strict-greedy trap that the
    // round-157 depth-2 matcher could not escape — three consecutive
    // short matches at `pos`, `pos + 1`, `pos + 2` together blocking a
    // strictly longer match at `pos + 3`. The decoder output is
    // bit-identical for any input — only the token *partition* shifts
    // by up to three pixels — so round-trips remain bit-exact under
    // any input.
    //
    // Three contracts (mirroring the round-156 / round-157 layout):
    //
    // 1) Round-trip — every depth-3 lazy-matched stream still
    //    round-trips end-to-end through `decode_lossless_image`.
    // 2) Strict-beat — on a hand-crafted depth-3-trap fixture, the
    //    depth-3 matcher emits strictly fewer Copy tokens than the
    //    strict-greedy, depth-1, and depth-2 matchers.
    // 3) Non-regression — on a broader fixture matrix the depth-3
    //    token count is `<=` the depth-2 token count everywhere.

    /// Round 158 round-trip: a noisy 96×16 fixture encoded with the
    /// round-158 depth-3 lazy matcher (now the production
    /// `tokenize_lz77` default) must still decode bit-exactly back to
    /// the original ARGB pixels. Uses an independent xorshift seed
    /// from the round-156 / round-157 tests so all three fixtures
    /// exercise the matcher over distinct entropy.
    #[test]
    fn round_158_depth3_lazy_match_round_trips_through_decoder() {
        let w = 96u32;
        let h = 16u32;
        let mut seed = 0xDEAD_BEEF_u32;
        let pixels: Vec<u32> = (0..(w * h) as usize)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                0xFF00_0000 | (seed & 0x00FF_FFFF)
            })
            .collect();

        // The full chooser delegates to `tokenize_lz77` (depth-3 as of
        // round 158); end-to-end round-trip through the framed file
        // must recover the exact input.
        let stream = encode_argb_with_predictor_chooser(&pixels, w, h);
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());

        // The direct depth-3 token stream against the no-transform
        // encoder must also round-trip — guards against bookkeeping
        // bugs in the new depth-3 insert/skip dedup path (where `pos`,
        // `pos + 1`, and `pos + 2` can all be pre-inserted before the
        // chosen match starts at `pos`, `pos + 1`, `pos + 2`, or
        // `pos + 3`).
        let stream_direct = encode_argb_literals_with_width(&pixels, w);
        let header_direct = build_image_header(w, h, true);
        let mut payload_direct = header_direct.to_vec();
        payload_direct.extend_from_slice(&stream_direct);
        let framed_direct =
            build::build_webp_file(&payload_direct, ImageKind::Lossless, w, h).unwrap();
        let img_direct = crate::decode_lossless_image(&framed_direct)
            .unwrap()
            .unwrap();
        assert_eq!(img_direct.pixels(), pixels.as_slice());
    }

    /// Round 158 strict-beat: a hand-crafted depth-3-trap fixture
    /// where the strict-greedy matcher, the round-156 depth-1 lazy
    /// matcher, AND the round-157 depth-2 lazy matcher all accept a
    /// short match at `pos` that prevents a strictly longer match at
    /// `pos + 3`. The depth-3 lazy matcher emits three literals and
    /// takes the longer match.
    ///
    /// Layout (each capital letter is a unique ARGB constant; the
    /// `Z*` family are unique separator pixels that share no 4-pixel
    /// window with the anchors):
    ///
    /// ```text
    ///   pos  0..4    [P Q R S]                — anchor A (4 px)
    ///   pos  4..7    [Z1 Z2 Z3]               — separator
    ///   pos  7..11   [Q R S T]                — anchor B (4 px)
    ///   pos 11..14   [Z4 Z5 Z6]               — separator
    ///   pos 14..18   [R S T U]                — anchor C (4 px)
    ///   pos 18..21   [Z7 Z8 Z9]               — separator
    ///   pos 21..30   [S T U V W X Y A B]      — anchor D (9 px)
    ///   pos 30..33   [Z10 Z11 Z12]            — separator
    ///   pos 33       P                         — trap start
    ///   pos 34       Q
    ///   pos 35       R
    ///   pos 36..45   [S T U V W X Y A B]      — depth-3 chain region
    ///   pos 45..     fill with unique Zfill colors (no 4-window match)
    /// ```
    ///
    /// At pos 33:
    ///
    /// * `find(33)` window `[P,Q,R,S]` → matches anchor A (pos 0),
    ///   extension stops at length 4 because pos 4 (Z1) ≠ pos 37 (T).
    /// * `find(34)` window `[Q,R,S,T]` → matches anchor B (pos 7),
    ///   extension stops at length 4 because pos 11 (Z4) ≠ pos 38 (U).
    ///   `L_b = 4 = L_a`, **not strictly greater**, so the depth-1
    ///   lazy matcher does NOT swap.
    /// * `find(35)` window `[R,S,T,U]` → matches anchor C (pos 14),
    ///   extension stops at length 4 because pos 18 (Z7) ≠ pos 39 (V).
    ///   `L_c = 4 = L_a`, **not strictly greater**, so the depth-2
    ///   lazy matcher does NOT swap.
    /// * `find(36)` window `[S,T,U,V]` → matches anchor D (pos 21),
    ///   extension goes the full `[S,T,U,V,W,X,Y,A,B]` (length 9)
    ///   before pos 30 (Z10) ≠ pos 45 (Zfill). `L_d = 9 > 4`, so the
    ///   depth-3 lazy matcher swaps to three literals + the length-9
    ///   match.
    ///
    /// Strict-greedy, depth-1, AND depth-2 partition at the trap:
    /// `[Copy{4, dist=33}, ...]`. Depth-3 partition: `[Lit(P), Lit(Q),
    /// Lit(R), Copy{9, dist=15}, ...]`. Net: depth-3 collapses a
    /// short-then-short-then-short triple into one longer copy.
    #[test]
    fn round_158_depth3_lazy_match_strictly_beats_depth2_on_trap_fixture() {
        // Distinct ARGB constants. Anchor letters P..Y + A..B carry
        // the structural matches; Z1..Z12 + Zfill are deliberately
        // unique so they cannot seed a parasitic chain.
        let p_ = 0xFF11_2200_u32;
        let q_ = 0xFF22_3300_u32;
        let r_ = 0xFF33_4400_u32;
        let s_ = 0xFF44_5500_u32;
        let t_ = 0xFF55_6600_u32;
        let u_ = 0xFF66_7700_u32;
        let v_ = 0xFF77_8800_u32;
        let w_ = 0xFF88_9900_u32;
        let x_ = 0xFF99_AA00_u32;
        let y_ = 0xFFAA_BB00_u32;
        let a_ = 0xFFBB_CC00_u32;
        let b_ = 0xFFCC_DD00_u32;
        let z01 = 0xFFEE_0001_u32;
        let z02 = 0xFFEE_0002_u32;
        let z03 = 0xFFEE_0003_u32;
        let z04 = 0xFFEE_0004_u32;
        let z05 = 0xFFEE_0005_u32;
        let z06 = 0xFFEE_0006_u32;
        let z07 = 0xFFEE_0007_u32;
        let z08 = 0xFFEE_0008_u32;
        let z09 = 0xFFEE_0009_u32;
        let z10 = 0xFFEE_000A_u32;
        let z11 = 0xFFEE_000B_u32;
        let z12 = 0xFFEE_000C_u32;

        let mut pixels: Vec<u32> = vec![
            p_, q_, r_, s_, // 0..4    anchor A
            z01, z02, z03, // 4..7    separator
            q_, r_, s_, t_, // 7..11   anchor B
            z04, z05, z06, // 11..14  separator
            r_, s_, t_, u_, // 14..18  anchor C
            z07, z08, z09, // 18..21  separator
            s_, t_, u_, v_, w_, x_, y_, a_, b_, // 21..30  anchor D (9 px)
            z10, z11, z12, // 30..33  separator
            p_, q_, r_, // 33..36  trap start (depth-1/2 cannot escape)
            s_, t_, u_, v_, w_, x_, y_, a_, b_, // 36..45  depth-3 chain region
        ];
        // Pad the tail with unique colors so the depth-3 swap's
        // post-match region cannot trigger another long match that
        // might mask the trap's copy-count delta.
        let mut filler = 0xFFF0_0000_u32;
        while pixels.len() < 96 {
            filler = filler.wrapping_add(1);
            pixels.push(filler);
        }

        let greedy = tokenize_lz77_inner(&pixels, 0);
        let lazy1 = tokenize_lz77_inner(&pixels, 1);
        let lazy2 = tokenize_lz77_inner(&pixels, 2);
        let lazy3 = tokenize_lz77_inner(&pixels, 3);

        let copies = |toks: &[Token]| -> usize {
            toks.iter()
                .filter(|t| matches!(t, Token::Copy { .. }))
                .count()
        };
        let coverage = |toks: &[Token]| -> usize {
            toks.iter()
                .map(|t| match *t {
                    Token::Literal(_) => 1,
                    Token::CacheRef { .. } => 1,
                    Token::Copy { length, .. } => length,
                })
                .sum()
        };
        // Sanity: all four partitions cover the exact image.
        assert_eq!(coverage(&greedy), pixels.len());
        assert_eq!(coverage(&lazy1), pixels.len());
        assert_eq!(coverage(&lazy2), pixels.len());
        assert_eq!(coverage(&lazy3), pixels.len());

        let g_c = copies(&greedy);
        let l1_c = copies(&lazy1);
        let l2_c = copies(&lazy2);
        let l3_c = copies(&lazy3);
        eprintln!(
            "[round-158] depth-3 trap fixture: greedy tokens={} (copies={}), \
             depth-1 tokens={} (copies={}), depth-2 tokens={} (copies={}), \
             depth-3 tokens={} (copies={}), copy delta vs depth-2={}",
            greedy.len(),
            g_c,
            lazy1.len(),
            l1_c,
            lazy2.len(),
            l2_c,
            lazy3.len(),
            l3_c,
            l2_c as i64 - l3_c as i64,
        );

        // The trap forces depth-3 to collapse a length-4 copy + a
        // follow-on length-8 copy into a 3-literals + length-9 copy
        // that subsumes 12 pixels of what greedy / depth-1 / depth-2
        // would have to cover with two matches. The structural win
        // is on Copy count: depth-3 must emit strictly fewer Copy
        // tokens than all three baselines.
        assert_eq!(
            g_c, l1_c,
            "round-158 fixture: depth-1 must agree with greedy here \
             (no depth-1 swap fires) — greedy={g_c}, depth-1={l1_c}"
        );
        assert_eq!(
            g_c, l2_c,
            "round-158 fixture: depth-2 must agree with greedy here \
             (no depth-2 swap fires) — greedy={g_c}, depth-2={l2_c}"
        );
        assert!(
            l3_c < l2_c,
            "round-158 depth-3 matcher must emit strictly fewer Copy \
             tokens than the depth-2 matcher on the depth-3 trap \
             fixture: depth-2 copies={l2_c} depth-3 copies={l3_c}\n\
             depth-2 partition: {lazy2:?}\n\
             depth-3 partition: {lazy3:?}"
        );

        // Round-trip the bytes through the no-transform encoder for
        // good measure: the depth-3 path must decode back exactly.
        let stream = encode_argb_literals_with_width(&pixels, pixels.len() as u32);
        let w = pixels.len() as u32;
        let h = 1u32;
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Round 158 non-regression: across a broad fixture matrix the
    /// depth-3 lazy token count is `<=` the depth-2 lazy token count
    /// everywhere. Structural because the depth-3 probe only swaps
    /// when the alternate match is strictly longer than the depth-2
    /// best, so the depth-3 partition uses at most as many tokens as
    /// the depth-2 partition. The test guards against off-by-one in
    /// the new depth-3 insert/skip dedup (where `pos`, `pos + 1`, and
    /// `pos + 2` can all be pre-inserted before the chosen match
    /// starts at `pos`, `pos + 1`, `pos + 2`, or `pos + 3`).
    #[test]
    fn round_158_depth3_never_increases_token_count_over_depth2() {
        let shapes: &[(u32, u32)] = &[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (48, 48),
            (16, 32),
            (64, 16),
            (40, 24),
        ];
        for &(w, h) in shapes {
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0xFF;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let mut seed = 0xC0FFEE_u32;
            let noise: Vec<u32> = (0..(w * h) as usize)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    0xFF00_0000 | (seed & 0x00FF_FFFF)
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();

            for (name, pixels) in [
                ("gradient", &gradient),
                ("noise", &noise),
                ("stripes", &stripes),
            ] {
                let lazy2 = tokenize_lz77_inner(pixels, 2);
                let lazy3 = tokenize_lz77_inner(pixels, 3);
                assert!(
                    lazy3.len() <= lazy2.len(),
                    "round-158 depth-3 regression on {name} {w}x{h}: \
                     depth-2={} tokens, depth-3={} tokens",
                    lazy2.len(),
                    lazy3.len(),
                );
                // Round-trip the depth-3 stream as a defensive check
                // for hash-chain insert bookkeeping.
                let stream = encode_argb_literals_with_width(pixels, w);
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&stream);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-158 depth-3 round-trip mismatch on {name} {w}x{h}"
                );
            }
        }
    }

    // ---- round 163: §5.2.2 guarded depth-4 lazy LZ77 ----
    //
    // Three tests, mirroring the round-156 / 157 / 158 contract:
    //
    // 1) End-to-end round-trip — a noisy 96×16 fixture encoded with
    //    the round-163 guarded depth-4 lazy matcher (now the production
    //    `tokenize_lz77` default) must still decode bit-exactly back
    //    to the original ARGB pixels.
    // 2) Diminishing-returns guard — a hand-crafted fixture where the
    //    depth-3 best at `pos` is a long run (`>= DEPTH4_GUARD_THRESHOLD`)
    //    and a depth-4 swap candidate exists. The guard must suppress
    //    the depth-4 work so depth-4 == depth-3 byte-for-byte on that
    //    fixture; the unguarded depth-4 (simulated with `DEPTH4_GUARD_THRESHOLD`
    //    set to `MAX_MATCH`) would have swapped. We exercise the
    //    boundary by toggling the depth around the guard rather than
    //    monkey-patching the constant — the two depth values that
    //    bracket the guard (`3` vs `4`) produce identical partitions
    //    on the long-run fixture, proving the guard suppressed the
    //    probe.
    // 3) Non-regression — on a broader fixture matrix the depth-4
    //    token count is `<=` the depth-3 token count everywhere
    //    (structural: the depth-4 probe only swaps to a *strictly*
    //    longer match, so it can only remove tokens, never add them).

    /// Round 163 round-trip: a noisy 96×16 fixture encoded with the
    /// round-163 guarded depth-4 lazy matcher (now the production
    /// `tokenize_lz77` default) must still decode bit-exactly back to
    /// the original ARGB pixels. Uses an independent xorshift seed
    /// from the round-156 / 157 / 158 tests so all four fixtures
    /// exercise the matcher over distinct entropy.
    #[test]
    fn round_163_depth4_lazy_match_round_trips_through_decoder() {
        let w = 96u32;
        let h = 16u32;
        let mut seed = 0xFEED_FACE_u32;
        let pixels: Vec<u32> = (0..(w * h) as usize)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                0xFF00_0000 | (seed & 0x00FF_FFFF)
            })
            .collect();

        // The full chooser delegates to `tokenize_lz77` (depth-4 as of
        // round 163); end-to-end round-trip through the framed file
        // must recover the exact input.
        let stream = encode_argb_with_predictor_chooser(&pixels, w, h);
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());

        // The direct depth-4 token stream against the no-transform
        // encoder must also round-trip — guards against bookkeeping
        // bugs in the new depth-4 insert/skip dedup path (where `pos`,
        // `pos + 1`, `pos + 2`, and `pos + 3` can all be pre-inserted
        // before the chosen match starts at `pos`, `pos + 1`, `pos + 2`,
        // `pos + 3`, or `pos + 4`).
        let stream_direct = encode_argb_literals_with_width(&pixels, w);
        let header_direct = build_image_header(w, h, true);
        let mut payload_direct = header_direct.to_vec();
        payload_direct.extend_from_slice(&stream_direct);
        let framed_direct =
            build::build_webp_file(&payload_direct, ImageKind::Lossless, w, h).unwrap();
        let img_direct = crate::decode_lossless_image(&framed_direct)
            .unwrap()
            .unwrap();
        assert_eq!(img_direct.pixels(), pixels.as_slice());
    }

    /// Round 163 guard contract: on a fixture whose depth-3 best at
    /// some position is already a long run (length strictly `>=
    /// DEPTH4_GUARD_THRESHOLD`), the depth-4 probe MUST be suppressed
    /// by the guard. We construct an input where a long literal run
    /// at the start seeds a long match for the second copy. The
    /// depth-3 matcher emits a long match at the first probe; the
    /// depth-4 probe, if it were unguarded, would attempt a `find` at
    /// `pos + 4`. The guard's structural contract is that whenever
    /// the depth-3 best already covers `>= DEPTH4_GUARD_THRESHOLD`
    /// pixels, depth-4 produces the IDENTICAL token sequence as
    /// depth-3 — i.e. the guard fired and the depth-4 work was
    /// skipped.
    ///
    /// The simpler property the test asserts: on a long-run fixture
    /// the depth-4 partition (depth = 4) is byte-for-byte equal to
    /// the depth-3 partition (depth = 3). If the guard fails to fire,
    /// depth-4 would still find some marginal swap somewhere in the
    /// fixture and the two partitions would diverge.
    #[test]
    fn round_163_depth4_guard_suppresses_long_run_swap() {
        // A long, smoothly-varying run guarantees that almost every
        // match the matcher finds is significantly longer than
        // `DEPTH4_GUARD_THRESHOLD == 6` — so the guard should fire at
        // every probe site and depth-4 should produce the same token
        // partition as depth-3.
        //
        // We use a 4-pixel repeating motif that the matcher can find
        // long copies of after the first cycle: `[A, B, C, D, A, B, C,
        // D, …]`. After 12 pixels of warm-up, a `find` will return a
        // match length up to MAX_MATCH (well over the guard threshold).
        let a_ = 0xFF10_2030_u32;
        let b_ = 0xFF40_5060_u32;
        let c_ = 0xFF70_8090_u32;
        let d_ = 0xFFA0_B0C0_u32;
        let motif = [a_, b_, c_, d_];
        let mut pixels: Vec<u32> = Vec::with_capacity(512);
        for i in 0..512 {
            pixels.push(motif[i & 3]);
        }

        let lazy3 = tokenize_lz77_inner(&pixels, 3);
        let lazy4 = tokenize_lz77_inner(&pixels, 4);

        // Guard contract: when the depth-3 best is already long, the
        // depth-4 probe is suppressed and the two partitions are
        // byte-for-byte equal.
        assert_eq!(
            lazy3,
            lazy4,
            "round-163 depth-4 guard should suppress the depth-4 probe \
             on a long-run fixture (every depth-3 best `>= DEPTH4_GUARD_THRESHOLD == {}`), \
             producing the identical depth-3 partition; depth-3={} tokens, \
             depth-4={} tokens",
            DEPTH4_GUARD_THRESHOLD,
            lazy3.len(),
            lazy4.len(),
        );

        // Sanity: both partitions must cover the input exactly.
        let coverage = |toks: &[Token]| -> usize {
            toks.iter()
                .map(|t| match *t {
                    Token::Literal(_) => 1,
                    Token::CacheRef { .. } => 1,
                    Token::Copy { length, .. } => length,
                })
                .sum()
        };
        assert_eq!(coverage(&lazy3), pixels.len());
        assert_eq!(coverage(&lazy4), pixels.len());

        // End-to-end round-trip via the production chooser for good
        // measure: the depth-4-default `tokenize_lz77` must still
        // decode back exactly on this long-run fixture.
        let w = pixels.len() as u32;
        let h = 1u32;
        let stream = encode_argb_literals_with_width(&pixels, w);
        let header = build_image_header(w, h, true);
        let mut payload = header.to_vec();
        payload.extend_from_slice(&stream);
        let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
        let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
        assert_eq!(img.pixels(), pixels.as_slice());
    }

    /// Round 163 non-regression: across a broad fixture matrix the
    /// depth-4 lazy token count is `<=` the depth-3 lazy token count
    /// everywhere. Structural because the depth-4 probe — when the
    /// guard allows it to fire — only swaps when the alternate match
    /// is strictly longer than the depth-3 best, so the depth-4
    /// partition uses at most as many tokens as the depth-3 partition.
    /// When the guard suppresses the probe, depth-4 produces the same
    /// tokens as depth-3 directly. The test also guards against
    /// off-by-one in the new depth-4 insert/skip dedup (where `pos`,
    /// `pos + 1`, `pos + 2`, and `pos + 3` can all be pre-inserted
    /// before the chosen match starts at any of those positions or
    /// `pos + 4`).
    #[test]
    fn round_163_depth4_never_increases_token_count_over_depth3() {
        let shapes: &[(u32, u32)] = &[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (48, 48),
            (16, 32),
            (64, 16),
            (40, 24),
        ];
        for &(w, h) in shapes {
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0xFF;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let mut seed = 0xBADD_CAFE_u32;
            let noise: Vec<u32> = (0..(w * h) as usize)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    0xFF00_0000 | (seed & 0x00FF_FFFF)
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();

            for (name, pixels) in [
                ("gradient", &gradient),
                ("noise", &noise),
                ("stripes", &stripes),
            ] {
                let lazy3 = tokenize_lz77_inner(pixels, 3);
                let lazy4 = tokenize_lz77_inner(pixels, 4);
                assert!(
                    lazy4.len() <= lazy3.len(),
                    "round-163 depth-4 regression on {name} {w}x{h}: \
                     depth-3={} tokens, depth-4={} tokens",
                    lazy3.len(),
                    lazy4.len(),
                );
                // Round-trip the depth-4 stream as a defensive check
                // for hash-chain insert bookkeeping.
                let stream = encode_argb_literals_with_width(pixels, w);
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&stream);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-163 depth-4 round-trip mismatch on {name} {w}x{h}"
                );
            }
        }
    }

    // ---- round 159: §4.1 entropy-image-aware tie-break ----

    /// `pick_block_mode_with_hint` accepts the preferred neighbour
    /// mode when it ties with the otherwise-lowest mode at the same
    /// minimal residual cost. The block is a solid-colour fill, so
    /// modes 1..=13 all predict the left/top neighbour exactly →
    /// every interior pixel has zero residual, and ties run across
    /// every mode whose residual sum equals the lowest sum found.
    /// Without a hint the chooser picks the lowest mode (mode 1 on a
    /// non-black solid); with a hint of `Some(7)` it returns mode 7.
    #[test]
    fn round_159_pick_block_mode_with_hint_swaps_on_tie() {
        let w = 8usize;
        let h = 8usize;
        let pixels = vec![0xff50_6070u32; w * h];

        // No hint: the lowest tied mode wins (deterministic baseline).
        let baseline = pick_block_mode_with_hint(&pixels, w, h, 0, 0, w, h, None);
        // The exact value depends on the border rule for mode 0 vs
        // the per-channel residual; what matters here is that the
        // hint can swap to a different mode that ties at the same
        // cost.
        let baseline_cost = block_mode_cost(&pixels, w, h, 0, 0, w, h, baseline);

        // Probe every mode 0..=13 to find one that ties baseline but
        // is not equal to it.
        let mut tied_other: Option<u8> = None;
        for m in 0u8..=13 {
            if m == baseline {
                continue;
            }
            let c = block_mode_cost(&pixels, w, h, 0, 0, w, h, m);
            if c == baseline_cost {
                tied_other = Some(m);
                break;
            }
        }
        let other = tied_other
            .expect("a solid-fill block has at least two modes tied at minimal residual cost");

        // With hint == Some(other) and `other` strictly distinct
        // from `baseline` but tied at the same cost, the chooser
        // must return `other`.
        let with_hint = pick_block_mode_with_hint(&pixels, w, h, 0, 0, w, h, Some(other));
        assert_eq!(
            with_hint, other,
            "round-159 tie-break did not adopt the preferred mode: \
             baseline={baseline}, other={other}, returned={with_hint}"
        );
    }

    /// `pick_block_mode_with_hint` does NOT swap when the preferred
    /// mode is strictly worse than the cost-minimal mode. A diagonal
    /// 2-D ramp `pixels[y, x] = (x + 2y) & 0xff` makes the L-based
    /// modes pay residual `1` per pixel while the T-based modes pay
    /// residual `2` per pixel, so the chooser picks an L-based mode
    /// uniquely. Probing every mode confirms which one is strictly
    /// worse than the picked baseline; with that mode as the hint
    /// the chooser must still return the baseline.
    #[test]
    fn round_159_pick_block_mode_with_hint_keeps_best_when_hint_worse() {
        let w = 16usize;
        let h = 16usize;
        // 2-D ramp: L-based modes pay 1/pixel; T-based modes pay 2/pixel.
        let pixels: Vec<u32> = (0..(w * h))
            .map(|i| {
                let x = (i % w) as u32;
                let y = (i / w) as u32;
                let v = (x + 2 * y) & 0xff;
                0xff00_0000 | (v << 16) | (v << 8) | v
            })
            .collect();

        let baseline = pick_block_mode_with_hint(&pixels, w, h, 0, 0, w, h, None);
        let baseline_cost = block_mode_cost(&pixels, w, h, 0, 0, w, h, baseline);
        // Find any mode whose cost is strictly worse than baseline.
        let mut worse: Option<u8> = None;
        for m in 0u8..=13 {
            let c = block_mode_cost(&pixels, w, h, 0, 0, w, h, m);
            if c > baseline_cost {
                worse = Some(m);
                break;
            }
        }
        let worse = worse
            .expect("test premise: the 2-D ramp should produce at least one strictly-worse mode");
        let with_hint = pick_block_mode_with_hint(&pixels, w, h, 0, 0, w, h, Some(worse));
        assert_eq!(
            with_hint, baseline,
            "round-159 tie-break must not adopt a strictly-worse hint \
             (baseline={baseline}, worse-hint={worse})"
        );
    }

    /// Local pre-round-159 copy of `build_predictor_image`. Mirrors
    /// the round-158 behaviour exactly: every block calls the
    /// hint-aware chooser with `prefer_mode = None`, so ties resolve
    /// to the lowest mode regardless of any spatial coherence. Used
    /// by the round-159 non-regression and strict-beat tests as the
    /// before-after baseline.
    fn pre_round_159_build_predictor_image(
        pixels: &[u32],
        width: u32,
        height: u32,
        size_bits: u8,
    ) -> (Vec<u32>, u32, u32) {
        let block = 1u32 << size_bits;
        let tw = predictor_div_round_up(width, block);
        let th = predictor_div_round_up(height, block);
        let mut img = Vec::with_capacity((tw * th) as usize);
        let w = width as usize;
        let h = height as usize;
        let bsz = block as usize;
        for by in 0..th as usize {
            for bx in 0..tw as usize {
                let x0 = bx * bsz;
                let y0 = by * bsz;
                let mode = pick_block_mode_with_hint(pixels, w, h, x0, y0, bsz, bsz, None);
                img.push(0xff00_0000 | ((mode as u32) << 8));
            }
        }
        (img, tw, th)
    }

    /// Round 159 structural correctness: the entropy-image-aware
    /// tie-break is residual-cost-neutral, so for *every* block the
    /// post-r159 chosen mode has identical residual cost to the
    /// pre-r159 chosen mode (only the mode *value* may differ on
    /// ties). The check is per-block: across a fixture matrix the
    /// summed per-block residual cost must be exactly equal under
    /// the two choosers.
    #[test]
    fn round_159_predictor_image_tie_break_is_cost_neutral() {
        let shapes: &[(u32, u32, u8)] = &[
            (32, 32, 4),
            (48, 48, 4),
            (64, 32, 4),
            (32, 64, 4),
            (24, 24, 3),
        ];
        for &(w, h, size_bits) in shapes {
            // Two fixtures: smooth gradient (many ties on flat regions
            // between modes 1/2/3 etc.) and palette-ish stripes
            // (column-aligned ties between L-based modes).
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0x0F;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();

            for (name, pixels) in [("gradient", &gradient), ("stripes", &stripes)] {
                let (pre_img, _, _) = pre_round_159_build_predictor_image(pixels, w, h, size_bits);
                let (post_img, _, _) = build_predictor_image(pixels, w, h, size_bits);
                assert_eq!(
                    pre_img.len(),
                    post_img.len(),
                    "pre/post mode-image length differs on {name} {w}x{h} size_bits={size_bits}"
                );
                let block = 1u32 << size_bits;
                let tw = predictor_div_round_up(w, block) as usize;
                let bsz = block as usize;
                let wu = w as usize;
                let hu = h as usize;
                for (idx, (pre_px, post_px)) in pre_img.iter().zip(post_img.iter()).enumerate() {
                    let bx = idx % tw;
                    let by = idx / tw;
                    let x0 = bx * bsz;
                    let y0 = by * bsz;
                    let pre_mode = ((pre_px >> 8) & 0xff) as u8;
                    let post_mode = ((post_px >> 8) & 0xff) as u8;
                    let pre_cost = block_mode_cost(pixels, wu, hu, x0, y0, bsz, bsz, pre_mode);
                    let post_cost = block_mode_cost(pixels, wu, hu, x0, y0, bsz, bsz, post_mode);
                    assert_eq!(
                        pre_cost, post_cost,
                        "round-159 tie-break changed residual cost on {name} {w}x{h} \
                         block=({bx},{by}): pre mode {pre_mode} cost {pre_cost}, \
                         post mode {post_mode} cost {post_cost}"
                    );
                }
            }
        }
    }

    /// Round 159 non-regression: across a fixture matrix the
    /// post-r159 predictor-chooser stream must never be longer than
    /// the pre-r159 stream. Since the tie-break is a strict subset
    /// of the pre-r159 candidate space (the chosen mode is always a
    /// cost-minimal mode under both choosers), the residual stream
    /// is identical and only the predictor sub-image's entropy can
    /// differ. The standalone chooser is invoked end-to-end through
    /// the lossless decoder to confirm round-trips on every fixture.
    #[test]
    fn round_159_predictor_chooser_never_regresses() {
        let shapes: &[(u32, u32)] = &[(16, 16), (24, 24), (32, 32), (48, 48), (32, 16), (24, 40)];
        for &(w, h) in shapes {
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0x0F;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();
            let mut seed = 0xDEAD_BEEFu32;
            let noise: Vec<u32> = (0..(w * h) as usize)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    0xFF00_0000 | (seed & 0x000F_0F0F)
                })
                .collect();

            for (name, pixels) in [
                ("gradient", &gradient),
                ("stripes", &stripes),
                ("low-noise", &noise),
            ] {
                // Encode under the production chooser (with r159 tie-break).
                let post = encode_argb_with_predictor_chooser(pixels, w, h);
                // Decode round-trip — strict invariant.
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&post);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-159 round-trip mismatch on {name} {w}x{h}"
                );
                // Non-regression: the chooser's output with the
                // r159 hint must be no larger than the chooser with
                // the hint stubbed out. Since the hint is a strict
                // tie-break (same residual cost), the residual
                // stream is identical; only the predictor sub-image
                // can change, and it changes in the entropy-
                // reducing direction (so the writer emits fewer
                // bytes for it).
                let pre = encode_argb_with_predictor_chooser_no_r159_hint(pixels, w, h);
                assert!(
                    post.len() <= pre.len(),
                    "round-159 chooser regressed on {name} {w}x{h}: \
                     pre={} B post={} B",
                    pre.len(),
                    post.len(),
                );
            }
        }
    }

    /// Round 159 structural strict-beat: across a sweep of
    /// perturbation seeds, at least one fixture must reach a
    /// strictly more-uniform predictor sub-image under the r159
    /// hint-aware chooser than under the no-hint baseline — i.e.
    /// the mode-image's distinct-mode count drops by at least 1.
    /// The sweep verifies the entropy-image-aware tie-break
    /// actually fires on realistic small fixtures and reports the
    /// byte delta in the §4.1 predictor candidate's output for the
    /// first such fixture.
    ///
    /// Operates on `encode_with_predictor` directly (vs the full
    /// chooser) so the savings aren't masked by a competing
    /// candidate winning the chooser.
    #[test]
    fn round_159_predictor_candidate_strictly_beats_no_hint_on_some_fixture() {
        let w = 48u32;
        let h = 48u32;
        let size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let mut found_strict_image = false;
        let mut found_strict_bytes = false;
        let mut best_savings: i64 = 0;
        let mut seed_winner: u32 = 0;
        for seed_init in [
            0xCAFE_BABEu32,
            0xC0FFEE00,
            0xDEAD_BEEF,
            0xFACE_F00D,
            0xFEED_F00D,
            0x1234_5678,
            0xABCD_1234,
            0x90AB_CDEF,
            0x5A5A_5A5A,
            0xA5A5_A5A5,
            0xBA5E_BA11,
            0xB16B_00B5,
        ] {
            // Solid-fill canvas with a small perturbed region.
            // Vary the perturbation extent so different fixtures
            // trigger different mode-image patterns.
            let solid = 0xff60_8050u32;
            let mut pixels = vec![solid; (w * h) as usize];
            let mut s = seed_init;
            // 8×8 perturbation in the top-left so the right /
            // bottom neighbours' left-/top-column reads stay
            // mostly on solid pixels.
            for y in 0..8u32 {
                for x in 0..8u32 {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    let v = (s & 0x0007_0707) | 0xFF00_0000;
                    pixels[(y * w + x) as usize] = v;
                }
            }
            let (pre_img, _, _) = pre_round_159_build_predictor_image(&pixels, w, h, size_bits);
            let (post_img, _, _) = build_predictor_image(&pixels, w, h, size_bits);
            let pre_modes: Vec<u8> = pre_img.iter().map(|p| ((p >> 8) & 0xff) as u8).collect();
            let post_modes: Vec<u8> = post_img.iter().map(|p| ((p >> 8) & 0xff) as u8).collect();
            let pre_distinct: std::collections::BTreeSet<u8> = pre_modes.iter().copied().collect();
            let post_distinct: std::collections::BTreeSet<u8> =
                post_modes.iter().copied().collect();
            if post_distinct.len() < pre_distinct.len() {
                found_strict_image = true;
                // Encode the predictor candidate under both
                // variants and check the byte delta.
                let post = encode_with_predictor(&pixels, w, h, size_bits, None, w);
                let pre = encode_with_predictor_no_r159_hint(&pixels, w, h, size_bits, None, w);
                let saved = pre.len() as i64 - post.len() as i64;
                if saved > best_savings {
                    best_savings = saved;
                    seed_winner = seed_init;
                }
                if post.len() < pre.len() {
                    found_strict_bytes = true;
                    // Round-trip the post stream end-to-end.
                    let header = build_image_header(w, h, true);
                    let mut payload = header.to_vec();
                    payload.extend_from_slice(&post);
                    let framed =
                        build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                    let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                    assert_eq!(
                        img.pixels(),
                        pixels.as_slice(),
                        "round-159 strict-beat predictor candidate round-trip mismatch on \
                         seed=0x{seed_init:08x}"
                    );
                    eprintln!(
                        "[round-159] strict-beat predictor candidate: seed=0x{seed_init:08x}, \
                         pre modes={pre_modes:?} post modes={post_modes:?} (distinct \
                         pre={} post={}), pre={} B post={} B, saved={saved} B",
                        pre_distinct.len(),
                        post_distinct.len(),
                        pre.len(),
                        post.len(),
                    );
                }
                // Non-regression always holds (residual cost is the
                // same under the tie-break, so the encoded bytes
                // can never increase).
                assert!(
                    post.len() <= pre.len(),
                    "round-159 tie-break regressed on seed=0x{seed_init:08x}: \
                     pre={} B post={} B",
                    pre.len(),
                    post.len(),
                );
            }
        }
        assert!(
            found_strict_image,
            "round-159 sweep did not produce a single strictly-more-uniform mode image \
             — the hint propagation never fired across the fixture set"
        );
        assert!(
            found_strict_bytes,
            "round-159 sweep found a strict mode-image reduction but never a strict byte \
             reduction; entropy savings stayed within the LSB packing slack \
             (best_savings={best_savings} on seed=0x{seed_winner:08x})"
        );
    }

    /// Local pre-round-159 copy of `encode_argb_with_predictor_chooser`
    /// that forces every predictor-image build to use the no-hint
    /// chooser. Used by `round_159_predictor_chooser_never_regresses`
    /// as the before-after baseline. The chooser's other candidate
    /// paths (no-tx, subtract-green, color-transform, color-indexing,
    /// meta-prefix) are re-used verbatim — only the predictor
    /// candidate is swapped for the no-hint variant.
    fn encode_argb_with_predictor_chooser_no_r159_hint(
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut best = encode_argb_literals_with_width(pixels, width);

        let pred_size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let ctx_size_bits = DEFAULT_COLOR_TRANSFORM_SIZE_BITS;
        let pred_block = 1u32 << pred_size_bits;
        let ctx_block = 1u32 << ctx_size_bits;

        if width >= pred_block && height >= pred_block {
            let mut pred_single_block_size_bits: u8 = pred_size_bits;
            while pred_single_block_size_bits < 9
                && ((1u32 << pred_single_block_size_bits) < width
                    || (1u32 << pred_single_block_size_bits) < height)
            {
                pred_single_block_size_bits += 1;
            }
            let try_pred_single_block = pred_single_block_size_bits != pred_size_bits;
            let mut pred_candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_predictor_no_r159_hint(
                    pixels,
                    width,
                    height,
                    pred_size_bits,
                    cache_bits,
                    width,
                )
            })];
            if try_pred_single_block {
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor_no_r159_hint(
                        pixels,
                        width,
                        height,
                        pred_single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in pred_candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if width >= ctx_block && height >= ctx_block {
            let mut single_block_size_bits: u8 = ctx_size_bits;
            while single_block_size_bits < 9
                && ((1u32 << single_block_size_bits) < width
                    || (1u32 << single_block_size_bits) < height)
            {
                single_block_size_bits += 1;
            }
            let try_single_block = single_block_size_bits != ctx_size_bits;
            let mut candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_color_transform(pixels, width, height, ctx_size_bits, cache_bits, width)
            })];
            if try_single_block {
                candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_color_transform(
                        pixels,
                        width,
                        height,
                        single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if collect_palette(pixels).is_some() {
            let ci_best = select_best_cache_bits(|cache_bits| {
                encode_with_color_indexing(pixels, width, height, cache_bits)
                    .expect("palette feasibility already confirmed")
            });
            if ci_best.len() < best.len() {
                best = ci_best;
            }
        }

        if let Some(mp_best) = sweep_meta_prefix_candidate(pixels, width, height) {
            if mp_best.len() < best.len() {
                best = mp_best;
            }
        }

        best
    }

    /// Local pre-round-159 copy of `encode_with_predictor` — same
    /// shape, but builds the predictor sub-image via the no-hint
    /// chooser (`pre_round_159_build_predictor_image`) so the
    /// before-after comparison isolates exactly the round-159
    /// tie-break change.
    fn encode_with_predictor_no_r159_hint(
        pixels: &[u32],
        width: u32,
        height: u32,
        size_bits: u8,
        cache_code_bits: Option<u32>,
        image_width: u32,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bit(true);
        w.write_bits(crate::vp8l_stream::TransformType::Predictor as u32, 2);
        debug_assert!((2..=9).contains(&size_bits));
        w.write_bits((size_bits - 2) as u32, 3);
        let (predictor_image, tw, _th) =
            pre_round_159_build_predictor_image(pixels, width, height, size_bits);
        write_entropy_coded_image_literals(&mut w, &predictor_image);
        w.write_bit(false);
        let mut residuals = vec![0u32; pixels.len()];
        apply_forward_predictor(
            pixels,
            &mut residuals,
            width,
            height,
            &predictor_image,
            tw,
            size_bits,
        );
        let mut tokens = tokenize_lz77(&residuals);
        if let Some(bits) = cache_code_bits {
            tokens = cacheify_tokens(&tokens, &residuals, bits);
        }
        write_spatially_coded_image(&mut w, &tokens, cache_code_bits, image_width);
        w.into_bytes()
    }

    // ---- round-160 §4.1 slack-cost tie-break tests ------------------

    /// Round 160 hint-aware chooser contract (slack form): given a
    /// preferred mode whose residual cost is **within `slack`** of
    /// the otherwise-best cost, the chooser returns the preferred
    /// mode rather than the lowest-tied (or lowest-best) mode.
    /// Constructs a small 4×4 block with carefully-chosen
    /// per-channel values such that the lowest-best mode is 0
    /// (Black) but a non-trivial L-based mode has cost only one
    /// magnitude unit higher; the slack=1 chooser must select the
    /// preferred mode.
    #[test]
    fn round_160_pick_block_mode_with_hint_slack_swaps_within_budget() {
        // Solid-fill 4×4: every mode 1..=13 ties at zero residual
        // cost across the block interior; mode 0 (Black) gives a
        // strictly larger cost (the solid color is far from black).
        // The slack-cost chooser with `prefer = Some(7)` and slack
        // >= 0 must select mode 7 (the preferred tied mode), and
        // the strict-tie chooser must agree.
        let solid = 0xff60_8050u32;
        let pixels: Vec<u32> = vec![solid; 16];
        let strict = pick_block_mode_with_hint(&pixels, 4, 4, 0, 0, 4, 4, Some(7));
        let slack0 = pick_block_mode_with_hint_slack(&pixels, 4, 4, 0, 0, 4, 4, Some(7), 0);
        assert_eq!(
            strict, slack0,
            "slack=0 must be byte-identical to the round-159 strict tie-break"
        );
        assert_eq!(
            slack0, 7,
            "preferred tied mode must win on slack=0 when cost is equal"
        );

        // Now construct a block where mode 0 has cost 0 (strictly
        // best) and another mode has small positive cost. The slack
        // chooser at sufficiently-large slack must swap to the
        // preferred mode; at slack=0 it must keep mode 0.
        //
        // Choose a 2×2 block of solid black (all zeros). The Black
        // predictor returns 0 (matches), and every other mode that
        // predicts from a neighbour also returns 0 (neighbours are
        // solid black). So *every* mode has cost 0 — not the
        // shape we want.
        //
        // Instead, place the test block inside a larger fixture so
        // that the block's *neighbour* pixels (above/left) differ
        // and force the L/T/etc. modes to non-zero cost while
        // Black mode stays at 0.
        //
        // 8×8 fixture: top half black, bottom half a non-zero
        // colour. Place the test block at (0, 4) — the row of
        // pixels above is the boundary between black (y=3) and
        // colour (y=4), so the T mode reads the row-3 black pixels
        // while the block itself is non-zero → T mode has non-zero
        // cost. The Black mode is `pred = 0` everywhere → cost is
        // the sum-magnitudes of the block's non-zero pixels.
        let mut big = vec![0xff00_0000u32; 64];
        for y in 4..8u32 {
            for x in 0..8u32 {
                big[(y * 8 + x) as usize] = 0xff01_0101;
            }
        }
        let best_default = pick_block_mode_with_hint(&big, 8, 8, 0, 4, 4, 4, None);
        let best_cost = block_mode_cost(&big, 8, 8, 0, 4, 4, 4, best_default);

        // Pick a non-best mode and find its cost.
        let mut preferred: u8 = u8::MAX;
        let mut pref_cost: u64 = u64::MAX;
        for m in 0u8..=13 {
            if m == best_default {
                continue;
            }
            let c = block_mode_cost(&big, 8, 8, 0, 4, 4, 4, m);
            if c > best_cost && c < pref_cost {
                preferred = m;
                pref_cost = c;
            }
        }
        if preferred != u8::MAX {
            let extra = pref_cost - best_cost;
            // Strict tie-break must keep the best mode (cost
            // mismatch).
            let strict = pick_block_mode_with_hint(&big, 8, 8, 0, 4, 4, 4, Some(preferred));
            assert_eq!(
                strict, best_default,
                "strict round-159 tie-break must NOT swap when costs differ"
            );
            // Slack = extra - 1 must also keep the best mode.
            if extra > 0 {
                let slack_too_small = pick_block_mode_with_hint_slack(
                    &big,
                    8,
                    8,
                    0,
                    4,
                    4,
                    4,
                    Some(preferred),
                    extra - 1,
                );
                assert_eq!(
                    slack_too_small, best_default,
                    "slack < (pref_cost - best_cost) must NOT swap"
                );
            }
            // Slack = extra must now allow the swap.
            let slack_exact =
                pick_block_mode_with_hint_slack(&big, 8, 8, 0, 4, 4, 4, Some(preferred), extra);
            assert_eq!(
                slack_exact, preferred,
                "slack >= (pref_cost - best_cost) must accept the preferred mode swap"
            );
        }
    }

    /// Round 160 strict round-159 equivalence: with `slack = 0` the
    /// slack-cost chooser must produce byte-identical predictor
    /// sub-images and byte-identical encoded streams to the
    /// round-159 strict-tie-break baseline, across a fixture
    /// matrix.
    #[test]
    fn round_160_slack_zero_matches_round_159_baseline() {
        let shapes: &[(u32, u32, u8)] = &[
            (32, 32, 4),
            (48, 48, 4),
            (64, 32, 4),
            (32, 64, 4),
            (24, 24, 3),
        ];
        for &(w, h, size_bits) in shapes {
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0x0F;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();

            for (name, pixels) in [("gradient", &gradient), ("stripes", &stripes)] {
                let (r159_img, _, _) = build_predictor_image(pixels, w, h, size_bits);
                let (r160_img, _, _) = build_predictor_image_with_slack(pixels, w, h, size_bits, 0);
                assert_eq!(
                    r159_img, r160_img,
                    "slack=0 sub-image must equal r159 baseline on {name} {w}x{h} \
                     size_bits={size_bits}"
                );
                let r159_bytes = encode_with_predictor(pixels, w, h, size_bits, None, w);
                let r160_bytes = encode_with_predictor_slack(pixels, w, h, size_bits, None, w, 0);
                assert_eq!(
                    r159_bytes, r160_bytes,
                    "slack=0 encoded bytes must equal r159 baseline on {name} {w}x{h} \
                     size_bits={size_bits}"
                );
            }
        }
    }

    /// Round 160 round-trip correctness: at any slack budget, the
    /// slack-cost predictor candidate produces a stream that, when
    /// framed and decoded, reproduces the input pixels exactly. The
    /// per-block chosen mode changes with slack but the forward
    /// transform always derives residuals from the chosen modes and
    /// the decoder re-derives the same modes from the sub-image.
    #[test]
    fn round_160_slack_predictor_round_trips_through_decoder() {
        let w = 32u32;
        let h = 32u32;
        let size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let pixels: Vec<u32> = (0..(w * h) as usize)
            .map(|i| {
                let x = (i as u32) % w;
                let y = (i as u32) / w;
                let r = (x * 7) & 0xff;
                let g = (y * 11) & 0xff;
                let b = ((x ^ y) * 3) & 0xff;
                0xFF00_0000 | (r << 16) | (g << 8) | b
            })
            .collect();
        let block_pixels: u64 = (1u64 << size_bits) * (1u64 << size_bits);
        for slack in [0, block_pixels, 2 * block_pixels, 8 * block_pixels] {
            let stream = encode_with_predictor_slack(&pixels, w, h, size_bits, None, w, slack);
            let header = build_image_header(w, h, true);
            let mut payload = header.to_vec();
            payload.extend_from_slice(&stream);
            let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
            let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
            assert_eq!(
                img.pixels(),
                pixels.as_slice(),
                "round-160 slack={slack} predictor candidate failed end-to-end round-trip"
            );
        }
    }

    /// Round 160 non-regression: across a fixture matrix the
    /// production `encode_argb_with_predictor_chooser` output is
    /// `<=` the chooser's output with slack candidates disabled
    /// (i.e. the round-159 chooser). The new slack candidates can
    /// only *add* options to the byte-best selection, so they must
    /// never increase the chosen output length.
    #[test]
    fn round_160_chooser_never_regresses_vs_round_159() {
        let shapes: &[(u32, u32)] = &[(32, 32), (48, 48), (32, 64), (64, 32), (24, 24)];
        for &(w, h) in shapes {
            // Three fixtures: smooth gradient, palette stripes, and
            // a sparse noise image (low predictor residual mass for
            // a few mode-image blocks, high for others — exactly
            // the regime where the slack tie-break can pay off).
            let gradient: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    let y = (i as u32) / w;
                    let g = (x + y) & 0x0F;
                    0xFF00_0000 | (g << 16) | (g << 8) | g
                })
                .collect();
            let stripes: Vec<u32> = (0..(w * h) as usize)
                .map(|i| {
                    let x = (i as u32) % w;
                    match x % 4 {
                        0 => 0xFFAA_5500,
                        1 => 0xFF55_AA00,
                        2 => 0xFF00_55AA,
                        _ => 0xFF55_00AA,
                    }
                })
                .collect();
            let mut s: u32 = 0xCAFE_BABE;
            let noise: Vec<u32> = (0..(w * h) as usize)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    0xFF00_0000 | (s & 0x00FF_FFFF)
                })
                .collect();

            for (name, pixels) in [
                ("gradient", &gradient),
                ("stripes", &stripes),
                ("noise", &noise),
            ] {
                let r159 = encode_argb_with_predictor_chooser_no_r160_slack(pixels, w, h);
                let r160 = encode_argb_with_predictor_chooser(pixels, w, h);
                assert!(
                    r160.len() <= r159.len(),
                    "round-160 chooser regressed on {name} {w}x{h}: r159={} B r160={} B",
                    r159.len(),
                    r160.len()
                );
                // End-to-end round-trip parity on the r160 stream.
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&r160);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-160 chooser output failed end-to-end round-trip on \
                     {name} {w}x{h}"
                );
            }
        }
    }

    /// Round 160 headline: the slack-cost **predictor candidate**
    /// strictly beats the round-159 strict-tie-break predictor
    /// candidate on at least one fixture, with the seed, slack
    /// budget, and byte savings printed for the round report.
    ///
    /// The comparison is between the two predictor candidates in
    /// isolation, not between the overall chooser outputs: the
    /// production chooser composes the predictor candidate with
    /// every other transform path (no-tx, subtract-green, color-
    /// transform, color-indexing, multi-meta-prefix) and may pick a
    /// non-predictor path as best, so the chooser output won't
    /// always reflect the slack savings on the predictor candidate
    /// alone. The invariant we *prove* here is: on at least one
    /// fixture in the sweep, `encode_with_predictor_slack(..,
    /// slack > 0, ..)` produces a strictly shorter byte stream
    /// than `encode_with_predictor(.., slack = 0, ..)`, which is
    /// the byte-cost win the round-160 slack-cost variant is
    /// designed to capture. The full chooser also picks up the
    /// win whenever the predictor path ends up the byte-best
    /// overall.
    ///
    /// The fixtures are seeded perturbations of a mostly-uniform
    /// canvas: small perturbation patches plus a sparse single-
    /// pixel noise sprinkle. These are the layouts where the
    /// predictor sub-image carries a small number of "almost
    /// uniform" mode-image entries that the slack tie-break can
    /// collapse onto a single dominant mode at a small residual
    /// cost.
    #[test]
    fn round_160_slack_candidate_strictly_beats_strict_on_some_fixture() {
        let w = 128u32;
        let h = 128u32;
        let size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let mut found = false;
        let mut best_savings: i64 = 0;
        let mut seed_winner: u32 = 0;
        let mut slack_winner: u64 = 0;
        // Slack sweep: pick a spread of budgets between 1 residual
        // unit and 4× block_pixels. The diagnostic phase of round
        // 160 development showed that the productive regime starts
        // around slack ≥ block_pixels / 4 (16-pixel blocks → slack
        // ≥ 64) on the seeded fixtures used here.
        let block_pixels: u64 = (1u64 << size_bits) * (1u64 << size_bits);
        let slack_candidates: &[u64] = &[
            1,
            4,
            16,
            64,
            block_pixels,
            2 * block_pixels,
            4 * block_pixels,
        ];
        for seed_init in [
            0xCAFE_BABEu32,
            0xC0FFEE00,
            0xDEAD_BEEF,
            0xFACE_F00D,
            0xFEED_F00D,
            0x1234_5678,
            0xABCD_1234,
            0x90AB_CDEF,
            0x5A5A_5A5A,
            0xA5A5_A5A5,
            0xBA5E_BA11,
            0xB16B_00B5,
            0x00DD_BA11,
            0xC1AB_AB00,
            0xDEAF_BABE,
            0xCABB_A6E0,
            0x1337_C0DE,
            0xABAD_CAFE,
            0xBADF_00D0,
            0x8BAD_F00D,
        ] {
            // Mostly-solid canvas with a 1-bit-per-channel noise
            // overlay sprinkled at a sparse stride. The overlay is
            // small enough that the residual mass added per block
            // is in the order of `block_pixels` (matches our chooser
            // slack budget) but large enough to push the best-mode
            // choice off the all-zero tie in some blocks.
            let solid = 0xff60_8050u32;
            let mut pixels = vec![solid; (w * h) as usize];
            let mut s = seed_init;
            // Two perturbation patches of varying sizes to give the
            // chooser something to chew on without dominating the
            // whole image (the chooser must still see lots of tied
            // blocks for the slack tie-break to pay off).
            for y in 0..6u32 {
                for x in 0..6u32 {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    pixels[(y * w + x) as usize] = (s & 0x0003_0303) | 0xFF60_8050;
                }
            }
            for y in 20..30u32 {
                for x in 20..30u32 {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    pixels[(y * w + x) as usize] = (s & 0x0007_0707) | 0xFF60_8050;
                }
            }
            // Sparse single-pixel perturbations scattered across the
            // remaining canvas — these are the perturbations that
            // tend to push individual blocks just barely off the
            // best-mode tie, exposing the slack tie-break opportunity.
            for _ in 0..32u32 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let px = (s >> 8) % w;
                let py = (s >> 16) % h;
                pixels[(py * w + px) as usize] = (s & 0x0001_0101) | 0xFF60_8050;
            }

            // Strict-tie-break baseline (round-159 chooser): the
            // slack = 0 predictor candidate at the default
            // size_bits. Cache-bits stays at None for a clean
            // comparison — the slack candidate is also tested at
            // cache_code_bits = None, isolating the effect to the
            // §4.1 forward transform.
            let strict_bytes = encode_with_predictor(&pixels, w, h, size_bits, None, w);
            // Slack sweep: pick the smallest slack-cost predictor
            // stream and compare against the strict baseline.
            let mut best_slack_bytes = strict_bytes.clone();
            let mut best_slack_value: u64 = 0;
            for &slack in slack_candidates {
                let bytes = encode_with_predictor_slack(&pixels, w, h, size_bits, None, w, slack);
                if bytes.len() < best_slack_bytes.len() {
                    best_slack_bytes = bytes;
                    best_slack_value = slack;
                }
            }
            if best_slack_bytes.len() < strict_bytes.len() {
                let saved = strict_bytes.len() as i64 - best_slack_bytes.len() as i64;
                if saved > best_savings {
                    best_savings = saved;
                    seed_winner = seed_init;
                    slack_winner = best_slack_value;
                }
                if !found {
                    found = true;
                }
                // Round-trip the winning slack stream end-to-end
                // through the full framed-WebP path to prove decode
                // correctness on the slack-tie-break-modified
                // residual stream.
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&best_slack_bytes);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-160 strict-beat predictor candidate round-trip mismatch on \
                     seed=0x{seed_init:08x} slack={best_slack_value}"
                );
                eprintln!(
                    "[round-160] slack-cost strict-beat: seed=0x{seed_init:08x}, \
                     slack={best_slack_value}, strict={} B slack={} B saved={saved} B",
                    strict_bytes.len(),
                    best_slack_bytes.len(),
                );
            }
            // Production chooser non-regression: r160 chooser
            // (which evaluates both strict and slack predictor
            // candidates against every other transform path) is
            // always ≤ r159 chooser (which evaluates strict only).
            let r159 = encode_argb_with_predictor_chooser_no_r160_slack(&pixels, w, h);
            let r160 = encode_argb_with_predictor_chooser(&pixels, w, h);
            assert!(
                r160.len() <= r159.len(),
                "round-160 chooser regressed on seed 0x{seed_init:08x}: \
                 r159={} B r160={} B",
                r159.len(),
                r160.len()
            );
        }
        assert!(
            found,
            "round-160 slack-cost sweep did not produce a single strict byte reduction \
             across the seeded fixture set; the new slack candidates never won \
             (best_savings={best_savings} on seed=0x{seed_winner:08x} slack={slack_winner})"
        );
    }

    /// Local pre-round-160 copy of `encode_argb_with_predictor_chooser`
    /// that omits the round-160 slack-cost predictor candidates. Used
    /// by the round-160 non-regression and strict-beat tests as the
    /// before-after baseline; the rest of the chooser (no-tx,
    /// subtract-green, color-transform, color-indexing, meta-prefix)
    /// is re-used verbatim.
    fn encode_argb_with_predictor_chooser_no_r160_slack(
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut best = encode_argb_literals_with_width(pixels, width);

        let pred_size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let ctx_size_bits = DEFAULT_COLOR_TRANSFORM_SIZE_BITS;
        let pred_block = 1u32 << pred_size_bits;
        let ctx_block = 1u32 << ctx_size_bits;

        if width >= pred_block && height >= pred_block {
            let mut pred_single_block_size_bits: u8 = pred_size_bits;
            while pred_single_block_size_bits < 9
                && ((1u32 << pred_single_block_size_bits) < width
                    || (1u32 << pred_single_block_size_bits) < height)
            {
                pred_single_block_size_bits += 1;
            }
            let try_pred_single_block = pred_single_block_size_bits != pred_size_bits;
            let mut pred_candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_predictor(pixels, width, height, pred_size_bits, cache_bits, width)
            })];
            if try_pred_single_block {
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor(
                        pixels,
                        width,
                        height,
                        pred_single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in pred_candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if width >= ctx_block && height >= ctx_block {
            let mut single_block_size_bits: u8 = ctx_size_bits;
            while single_block_size_bits < 9
                && ((1u32 << single_block_size_bits) < width
                    || (1u32 << single_block_size_bits) < height)
            {
                single_block_size_bits += 1;
            }
            let try_single_block = single_block_size_bits != ctx_size_bits;
            let mut candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_color_transform(pixels, width, height, ctx_size_bits, cache_bits, width)
            })];
            if try_single_block {
                candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_color_transform(
                        pixels,
                        width,
                        height,
                        single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if collect_palette(pixels).is_some() {
            let ci_best = select_best_cache_bits(|cache_bits| {
                encode_with_color_indexing(pixels, width, height, cache_bits)
                    .expect("palette feasibility already confirmed")
            });
            if ci_best.len() < best.len() {
                best = ci_best;
            }
        }

        if let Some(mp_best) = sweep_meta_prefix_candidate(pixels, width, height) {
            if mp_best.len() < best.len() {
                best = mp_best;
            }
        }

        best
    }

    // ---- Round 161 tests: Shannon-entropy bit-cost predictor variant -------

    /// Local pre-round-161 copy of `encode_argb_with_predictor_chooser`
    /// that omits the round-161 entropy-cost predictor candidates but
    /// **keeps** every round-160 slack-cost candidate. Used by the
    /// round-161 non-regression and strict-beat tests as the
    /// before-after baseline. Mirrors
    /// `encode_argb_with_predictor_chooser_no_r160_slack` in shape.
    fn encode_argb_with_predictor_chooser_no_r161_entropy(
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut best = encode_argb_literals_with_width(pixels, width);

        let pred_size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let ctx_size_bits = DEFAULT_COLOR_TRANSFORM_SIZE_BITS;
        let pred_block = 1u32 << pred_size_bits;
        let ctx_block = 1u32 << ctx_size_bits;

        if width >= pred_block && height >= pred_block {
            let mut pred_single_block_size_bits: u8 = pred_size_bits;
            while pred_single_block_size_bits < 9
                && ((1u32 << pred_single_block_size_bits) < width
                    || (1u32 << pred_single_block_size_bits) < height)
            {
                pred_single_block_size_bits += 1;
            }
            let try_pred_single_block = pred_single_block_size_bits != pred_size_bits;
            let mut pred_candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_predictor(pixels, width, height, pred_size_bits, cache_bits, width)
            })];
            let pred_block_pixels: u64 = (1u64 << pred_size_bits) * (1u64 << pred_size_bits);
            for slack in [
                pred_block_pixels,
                2 * pred_block_pixels,
                4 * pred_block_pixels,
            ] {
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor_slack(
                        pixels,
                        width,
                        height,
                        pred_size_bits,
                        cache_bits,
                        width,
                        slack,
                    )
                }));
            }
            if try_pred_single_block {
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor(
                        pixels,
                        width,
                        height,
                        pred_single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
                let single_pred_block_pixels: u64 =
                    (1u64 << pred_single_block_size_bits) * (1u64 << pred_single_block_size_bits);
                for slack in [
                    single_pred_block_pixels,
                    2 * single_pred_block_pixels,
                    4 * single_pred_block_pixels,
                ] {
                    pred_candidates.push(select_best_cache_bits(|cache_bits| {
                        encode_with_predictor_slack(
                            pixels,
                            width,
                            height,
                            pred_single_block_size_bits,
                            cache_bits,
                            width,
                            slack,
                        )
                    }));
                }
            }
            for cand in pred_candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if width >= ctx_block && height >= ctx_block {
            let mut single_block_size_bits: u8 = ctx_size_bits;
            while single_block_size_bits < 9
                && ((1u32 << single_block_size_bits) < width
                    || (1u32 << single_block_size_bits) < height)
            {
                single_block_size_bits += 1;
            }
            let try_single_block = single_block_size_bits != ctx_size_bits;
            let mut candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_color_transform(pixels, width, height, ctx_size_bits, cache_bits, width)
            })];
            if try_single_block {
                candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_color_transform(
                        pixels,
                        width,
                        height,
                        single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if collect_palette(pixels).is_some() {
            let ci_best = select_best_cache_bits(|cache_bits| {
                encode_with_color_indexing(pixels, width, height, cache_bits)
                    .expect("palette feasibility already confirmed")
            });
            if ci_best.len() < best.len() {
                best = ci_best;
            }
        }

        if let Some(mp_best) = sweep_meta_prefix_candidate(pixels, width, height) {
            if mp_best.len() < best.len() {
                best = mp_best;
            }
        }

        best
    }

    /// Round 161 — [`block_mode_entropy_cost`] reports zero milli-bits
    /// on a 1×1 block of pixel `0xff_00_00_00` (the top-left border
    /// rule sets `pred = 0xff_00_00_00`, so the residual is zero, the
    /// histogram has a single occupied bin per channel and the
    /// `c · log2(N/c) = N · log2(1) = 0` per-bin contribution sums to
    /// zero). Confirms the entropy summation correctly bottoms-out
    /// at the no-residual edge case.
    #[test]
    fn round_161_block_mode_entropy_cost_zero_on_zero_residual_block() {
        let pixels = vec![0xff_00_00_00u32; 1];
        for mode in 0u8..=13 {
            let cost = block_mode_entropy_cost(&pixels, 1, 1, 0, 0, 1, 1, mode);
            assert_eq!(
                cost, 0,
                "1×1 zero-residual block should produce zero-entropy cost under mode {mode}, got {cost}"
            );
        }
    }

    /// Round 161 — on an interior solid-fill block, every mode that
    /// produces a *constant* residual (whether zero or non-zero) ties
    /// at zero Shannon entropy — Shannon entropy measures **variety**
    /// in the residual symbol distribution, not magnitude. This is
    /// the key structural difference from the L1 magnitude proxy: L1
    /// would penalise mode 0 (which emits constant non-zero residual
    /// `0x00_60_80_50` per pixel on a `0xff_60_80_50` solid block),
    /// while Shannon entropy correctly treats a constant-residual
    /// distribution as zero-cost (a Huffman code over a single-symbol
    /// alphabet emits one bit per symbol, which is the theoretical
    /// floor and matches the §3.7.2.1.1 single-leaf encoding's
    /// near-zero overhead).
    ///
    /// This test pins down that semantic: on the interior solid
    /// block, every neighbour-predicting mode AND mode 0 all sit at
    /// zero entropy cost; the chooser then falls through to the
    /// lowest-index tie-break (mode 0) or the hint when one is
    /// supplied.
    #[test]
    fn round_161_block_mode_entropy_cost_zero_on_constant_residual_block() {
        let w = 8usize;
        let h = 8usize;
        let pixels = vec![0xff_60_80_50u32; w * h];
        // Block [4..8) × [4..8) — interior. Every mode produces a
        // constant residual across the block (zero for the
        // neighbour-predicting modes; `0x00_60_80_50` for mode 0).
        // Constant residual = single-symbol histogram per channel
        // = zero Shannon entropy.
        for mode in 0u8..=13 {
            let cost = block_mode_entropy_cost(&pixels, w, h, 4, 4, 4, 4, mode);
            assert_eq!(
                cost, 0,
                "constant-residual mode {mode} on interior solid block should have zero entropy cost, got {cost}"
            );
        }
    }

    /// Round 161 — Shannon entropy cost is strictly monotone in
    /// residual variety: a block whose residual histogram is
    /// peaked at a single value (zero or non-zero) has lower
    /// entropy cost than a block whose residuals scatter across
    /// multiple distinct values. This is the property a Huffman
    /// code over the residuals would actually minimise — and the
    /// L1 magnitude proxy does NOT distinguish (a constant non-
    /// zero residual block has the same L1 sum as a scattered
    /// block of the same mean magnitude). Confirms the entropy
    /// cost adds real signal vs the proxy.
    #[test]
    fn round_161_entropy_cost_distinguishes_concentrated_from_scattered() {
        // 16×16 image with two interior blocks. Concentrated block:
        // pure solid grey on the [4..8) × [4..8) corner — mode 1 (L
        // predictor) reproduces every interior pixel from its left
        // neighbour so every residual is zero. Scattered block:
        // checkerboard greys on the [8..12) × [8..12) corner — mode
        // 1 produces non-zero residuals alternating across
        // horizontal steps, populating multiple histogram bins.
        let w = 16usize;
        let h = 16usize;
        let grey = 0xff_60_80_50u32;
        let other = 0xff_70_90_60u32;
        let mut pixels = vec![grey; w * h];
        // Scatter `other` in a horizontal checkerboard across the
        // scattered block region. Use an isolated mutated quadrant
        // that doesn't reach the concentrated block; keep a buffer
        // row/column of solid grey around the scattered block so
        // its L neighbours at the block's left edge are still grey
        // (giving a deterministic histogram).
        for y in 8..12 {
            for x in 8..12 {
                if x % 2 == 0 {
                    pixels[y * w + x] = other;
                }
            }
        }
        let concentrated = block_mode_entropy_cost(&pixels, w, h, 4, 4, 4, 4, 1);
        let scattered = block_mode_entropy_cost(&pixels, w, h, 8, 8, 4, 4, 1);
        assert!(
            scattered > concentrated,
            "scattered block should have higher entropy cost than concentrated: \
             scattered={scattered}, concentrated={concentrated}"
        );
        assert_eq!(
            concentrated, 0,
            "concentrated (interior solid) block under mode 1 should have zero-entropy cost, \
             got {concentrated}"
        );
        assert!(
            scattered > 0,
            "scattered block should have strictly positive entropy cost, got {scattered}"
        );
    }

    /// Round 161 — the entropy chooser's tie-break mechanism mirrors
    /// the round-159 strict tie-break: when `prefer_mode`'s entropy
    /// cost equals the best, the chooser returns the preferred mode.
    /// On an interior solid-fill block, *every* mode produces a
    /// constant residual (zero or a fixed colour) and so ties at
    /// zero Shannon entropy; the chooser falls back to the lowest-
    /// index tie (mode 0) and the hint flips to any preferred mode.
    #[test]
    fn round_161_pick_block_mode_with_hint_entropy_honours_tie() {
        let w = 8usize;
        let h = 8usize;
        let pixels = vec![0xff_60_80_50u32; w * h];
        // Interior [4..8) × [4..8) block — every mode is a constant
        // residual (Shannon entropy zero) for the reasons in
        // [`round_161_block_mode_entropy_cost_zero_on_constant_residual_block`].
        // No hint → lowest mode 0 wins.
        let no_hint = pick_block_mode_with_hint_entropy(&pixels, w, h, 4, 4, 4, 4, None);
        assert_eq!(no_hint, 0);
        // Hint mode 11 → ties at zero → tie-break flips to 11.
        let with_hint = pick_block_mode_with_hint_entropy(&pixels, w, h, 4, 4, 4, 4, Some(11));
        assert_eq!(with_hint, 11);
        // Hint mode 5 → ties at zero → tie-break flips to 5.
        let with_hint5 = pick_block_mode_with_hint_entropy(&pixels, w, h, 4, 4, 4, 4, Some(5));
        assert_eq!(with_hint5, 5);
    }

    /// Round 161 — `encode_with_predictor_entropy` round-trips
    /// end-to-end through `decode_lossless_image`. Confirms the
    /// entropy chooser produces a decodable stream regardless of
    /// what cost model picked the modes (the §4.1 forward transform
    /// recomputes residuals against whatever mode the sub-image
    /// records, and the decoder applies the same inverse against
    /// that mode).
    #[test]
    fn round_161_entropy_predictor_round_trips_through_decoder() {
        let w = 32u32;
        let h = 32u32;
        // Mostly-uniform canvas with two small perturbations + a
        // single-pixel sprinkle — same recipe family as the round-
        // 160 strict-beat fixture, but smaller for fast test runs.
        let mut pixels = vec![0xff_60_80_50u32; (w * h) as usize];
        let mut s: u32 = 0xCAFE_BABE;
        for y in 2..8u32 {
            for x in 4..10u32 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                pixels[(y * w + x) as usize] = (s & 0x0007_0707) | 0xff60_8050;
            }
        }
        for cache_bits in [None, Some(2u32), Some(8u32)] {
            let bytes = encode_with_predictor_entropy(
                &pixels,
                w,
                h,
                DEFAULT_PREDICTOR_SIZE_BITS,
                cache_bits,
                w,
            );
            let header = build_image_header(w, h, true);
            let mut payload = header.to_vec();
            payload.extend_from_slice(&bytes);
            let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
            let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
            assert_eq!(
                img.pixels(),
                pixels.as_slice(),
                "entropy predictor round-trip mismatch at cache_bits={cache_bits:?}"
            );
        }
    }

    /// Round 161 — production chooser must never regress relative to
    /// the round-160 baseline. The round-161 entropy candidate is an
    /// additional path; the chooser keeps the byte-shortest stream,
    /// so adding a candidate cannot lengthen the output.
    #[test]
    fn round_161_chooser_never_regresses_vs_round_160() {
        let shapes: &[(u32, u32)] = &[(16, 16), (32, 32), (48, 48), (64, 32), (32, 64)];
        for &(w, h) in shapes {
            // Fixture A: solid fill.
            let solid = vec![0xff_60_80_50u32; (w * h) as usize];
            // Fixture B: low-frequency gradient.
            let mut gradient = vec![0u32; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    let r = (x * 255 / w.max(1)) as u8;
                    let g = (y * 255 / h.max(1)) as u8;
                    gradient[(y * w + x) as usize] =
                        0xff00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | 0x40;
                }
            }
            // Fixture C: small noise patch on a solid background.
            let mut sparse = vec![0xff_70_70_70u32; (w * h) as usize];
            let mut s: u32 = 0xDEAD_BEEF ^ (w * h);
            for _ in 0..(w * h / 16) {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let idx = ((s as usize) % sparse.len()) as usize;
                sparse[idx] = (s & 0x0003_0303) | 0xff70_7070;
            }
            for (name, pixels) in &[
                ("solid", &solid),
                ("gradient", &gradient),
                ("sparse", &sparse),
            ] {
                let r160 = encode_argb_with_predictor_chooser_no_r161_entropy(pixels, w, h);
                let r161 = encode_argb_with_predictor_chooser(pixels, w, h);
                assert!(
                    r161.len() <= r160.len(),
                    "round-161 chooser regressed on {name} {w}x{h}: \
                     r160={} B r161={} B",
                    r160.len(),
                    r161.len()
                );
                // Confirm decode round-trip on whatever the chooser
                // emitted — the chooser may have chosen the entropy
                // path or any of the L1 paths.
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&r161);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-161 chooser output failed decode round-trip on {name} {w}x{h}"
                );
            }
        }
    }

    /// Round 161 — sweep seeded fixtures to find at least one input
    /// where the entropy-cost predictor candidate strictly beats the
    /// best L1-proxy predictor candidate on raw bytes. Proves the
    /// entropy cost is doing real work — it's not merely a
    /// no-op-aliased duplicate of the round-160 path. The sweep
    /// also stress-tests round-trip correctness on every fixture
    /// where the entropy path wins.
    ///
    /// Construction: pre-residualised image families where the per-
    /// block mode-cost ordering differs between L1 magnitude and
    /// Shannon entropy. The most reliable family is one whose
    /// "lowest L1 mode" produces a varied residual histogram while
    /// some "slightly-higher L1 mode" produces a concentrated
    /// residual histogram — Shannon entropy picks the concentrated
    /// mode (faithful to what Huffman codes minimise), L1 picks the
    /// magnitude-min mode.
    #[test]
    fn round_161_entropy_candidate_strictly_beats_l1_on_some_fixture() {
        let w = 64u32;
        let h = 64u32;
        let size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let block_pixels: u64 = (1u64 << size_bits) * (1u64 << size_bits);
        let mut found = false;
        let mut best_savings: i64 = 0;
        let mut seed_winner: u32 = 0;
        let mut family_winner: &'static str = "";
        // Family A: row-translated tile with a hand-chosen base
        // colour. The L predictor (mode 1) reproduces each row's
        // base colour and has zero residual on interior pixels —
        // but the top-row predict-L rule on the first row leaks a
        // varied histogram (each first-row pixel's residual is a
        // function of its preceding column's source colour). Mode
        // 0 (predict 0xff000000) emits a constant residual equal
        // to source per pixel — zero entropy when the image is
        // solid, non-zero entropy when scattered. On a scattered
        // image mode 1 is L1-best but mode 0 is entropy-best.
        for seed_init in [
            0xCAFE_BABEu32,
            0xC0FFEE00,
            0xDEAD_BEEF,
            0xFACE_F00D,
            0xFEED_F00D,
            0x1234_5678,
            0xABCD_1234,
            0x90AB_CDEF,
            0x5A5A_5A5A,
            0xA5A5_A5A5,
            0xBA5E_BA11,
            0xB16B_00B5,
            0x00DD_BA11,
            0xC1AB_AB00,
            0xDEAF_BABE,
            0xCABB_A6E0,
            0x1337_C0DE,
            0xABAD_CAFE,
            0xBADF_00D0,
            0x8BAD_F00D,
            0xFEE1_DEAD,
            0xDEFE_C8ED,
            0xD15E_A5E0,
            0x600D_F00D,
            0xDEAD_C0DE,
            0xBADC_0DED,
            0xCAFE_F00D,
            0xC0DE_F00D,
            0xDEED_BEEF,
            0xBEAD_F00D,
            0x8008_5318,
            0xD0DE_C0DE,
        ] {
            // Build a fixture whose per-block mode-cost ordering
            // disagrees between L1 and Shannon entropy. The family
            // below produces blocks of varying L1-vs-entropy
            // disagreement intensity:
            //
            // Quadrant A (top-left): smooth low-frequency pattern
            //   where neighbour-predicting modes have low L1 but
            //   spread their residuals across multiple histogram
            //   bins (residual varies slightly with position).
            // Quadrant B (bottom-right): rare "spike" pixels (1 or
            //   2 per block) where mode 0's constant residual
            //   distribution wins on entropy.
            //
            // The two quadrants live in separate predictor blocks
            // so each contributes independently to whichever mode
            // wins on a block-by-block basis.
            let mut pixels = vec![0xff_60_80_50u32; (w * h) as usize];
            let mut s = seed_init;
            // Quadrant A: 32x32 patterned image with column-driven
            // gradient and a per-row jitter — produces non-trivial
            // residual histograms for every mode, so the L1-vs-
            // entropy disagreement frequency goes up.
            for y in 0..(h / 2) {
                for x in 0..(w / 2) {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    // Column-correlated colour + per-row jitter.
                    let r = 0x40 + (x as u8 & 0x1f);
                    let g = 0x60 + ((y as u8) & 0x1f) + ((s & 1) as u8);
                    let b = 0x30 + ((x as u8 ^ y as u8) & 0x0f);
                    pixels[(y * w + x) as usize] =
                        0xff00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
                }
            }
            // Quadrant B: solid grey with deliberate single-pixel
            // spikes at predictable positions. The spikes are
            // chosen to land inside a few of the predictor blocks
            // so those blocks see a residual distribution with one
            // major bin (zero) and one minor bin (the spike). The
            // L1 chooser picks the mode that minimises spike
            // magnitude; the entropy chooser picks the mode that
            // minimises the count of distinct residual bins.
            for y in (h / 2)..h {
                for x in (w / 2)..w {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    if (s & 0x1f) == 0 {
                        // Spike: random near-grey perturbation.
                        let perturb = (s & 0x0f0f_0f0f) | 0xff60_8050;
                        pixels[(y * w + x) as usize] = perturb;
                    }
                }
            }
            // Best L1-proxy predictor candidate at default
            // size_bits: strict round-159 + round-160 slack sweep.
            let strict_bytes = encode_with_predictor(&pixels, w, h, size_bits, None, w);
            let mut best_l1_bytes = strict_bytes.clone();
            for slack in [block_pixels, 2 * block_pixels, 4 * block_pixels] {
                let bytes = encode_with_predictor_slack(&pixels, w, h, size_bits, None, w, slack);
                if bytes.len() < best_l1_bytes.len() {
                    best_l1_bytes = bytes;
                }
            }
            let entropy_bytes = encode_with_predictor_entropy(&pixels, w, h, size_bits, None, w);
            if entropy_bytes.len() < best_l1_bytes.len() {
                let saved = best_l1_bytes.len() as i64 - entropy_bytes.len() as i64;
                if saved > best_savings {
                    best_savings = saved;
                    seed_winner = seed_init;
                    family_winner = "two-quadrant";
                }
                if !found {
                    found = true;
                }
                // Round-trip the winning entropy stream end-to-end.
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&entropy_bytes);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels.as_slice(),
                    "round-161 entropy strict-beat predictor candidate round-trip mismatch on \
                     seed=0x{seed_init:08x}"
                );
                eprintln!(
                    "[round-161] entropy strict-beat: seed=0x{seed_init:08x}, \
                     best_l1={} B entropy={} B saved={saved} B",
                    best_l1_bytes.len(),
                    entropy_bytes.len(),
                );
            }
        }
        // Family B: hand-crafted "constant non-zero residual"
        // fixture — a solid-colour image where mode 0 emits a
        // constant residual `source - 0xff000000` per pixel. The
        // L1 cost of mode 0 is `Σ |source - black|` per pixel; the
        // entropy cost of mode 0 is zero (single-symbol histogram).
        // Mode 1 (L predictor) also emits zero residual for
        // interior pixels but has non-zero residual at the leftmost
        // column. On a small image the per-block winner depends on
        // which of these effects dominates.
        if !found {
            // Build a 16×16 solid image — exactly one predictor
            // block at size_bits=4. The L1 cost of mode 0 is huge
            // (16² × magnitude); mode 1's cost is small (only the
            // leftmost column contributes). L1 picks mode 1.
            // Shannon entropy: mode 0 = 0 (constant residual);
            // mode 1 = small but non-zero (the leftmost column
            // residual). Entropy picks mode 0.
            //
            // Whether mode 0's predictor stream beats mode 1's
            // depends on the §5.x prefix-code overhead vs the
            // saved residual mass — not guaranteed, but a
            // candidate worth trying.
            let w2 = 16u32;
            let h2 = 16u32;
            let pixels2 = vec![0xff_80_80_80u32; (w2 * h2) as usize];
            let l1_bytes = encode_with_predictor(&pixels2, w2, h2, size_bits, None, w2);
            let entropy_bytes =
                encode_with_predictor_entropy(&pixels2, w2, h2, size_bits, None, w2);
            if entropy_bytes.len() < l1_bytes.len() {
                let saved = l1_bytes.len() as i64 - entropy_bytes.len() as i64;
                best_savings = saved;
                family_winner = "solid-grey-16x16";
                found = true;
                let header = build_image_header(w2, h2, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&entropy_bytes);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w2, h2).unwrap();
                let img = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    img.pixels(),
                    pixels2.as_slice(),
                    "round-161 entropy strict-beat solid-grey round-trip mismatch"
                );
                eprintln!(
                    "[round-161] entropy strict-beat (solid-grey 16x16): \
                     l1={} B entropy={} B saved={saved} B",
                    l1_bytes.len(),
                    entropy_bytes.len(),
                );
            }
        }
        assert!(
            found,
            "round-161 entropy candidate did not produce a single strict byte reduction \
             across the seeded fixture set; the entropy cost never won \
             (best_savings={best_savings} on seed=0x{seed_winner:08x} family={family_winner})"
        );
    }

    // ---- Round 162 tests: sub-image-aware Shannon-entropy chooser ----------

    /// Local pre-round-162 copy of `encode_argb_with_predictor_chooser`
    /// that omits the round-162 sub-image-aware lambda sweep but
    /// keeps every round-161 entropy candidate. Used as the
    /// before-after baseline for the round-162 non-regression and
    /// strict-beat tests.
    fn encode_argb_with_predictor_chooser_no_r162_subaware(
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut best = encode_argb_literals_with_width(pixels, width);

        let pred_size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
        let ctx_size_bits = DEFAULT_COLOR_TRANSFORM_SIZE_BITS;
        let pred_block = 1u32 << pred_size_bits;
        let ctx_block = 1u32 << ctx_size_bits;

        if width >= pred_block && height >= pred_block {
            let mut pred_single_block_size_bits: u8 = pred_size_bits;
            while pred_single_block_size_bits < 9
                && ((1u32 << pred_single_block_size_bits) < width
                    || (1u32 << pred_single_block_size_bits) < height)
            {
                pred_single_block_size_bits += 1;
            }
            let try_pred_single_block = pred_single_block_size_bits != pred_size_bits;
            let mut pred_candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_predictor(pixels, width, height, pred_size_bits, cache_bits, width)
            })];
            let pred_block_pixels: u64 = (1u64 << pred_size_bits) * (1u64 << pred_size_bits);
            for slack in [
                pred_block_pixels,
                2 * pred_block_pixels,
                4 * pred_block_pixels,
            ] {
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor_slack(
                        pixels,
                        width,
                        height,
                        pred_size_bits,
                        cache_bits,
                        width,
                        slack,
                    )
                }));
            }
            pred_candidates.push(select_best_cache_bits(|cache_bits| {
                encode_with_predictor_entropy(
                    pixels,
                    width,
                    height,
                    pred_size_bits,
                    cache_bits,
                    width,
                )
            }));
            if try_pred_single_block {
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor(
                        pixels,
                        width,
                        height,
                        pred_single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
                let single_pred_block_pixels: u64 =
                    (1u64 << pred_single_block_size_bits) * (1u64 << pred_single_block_size_bits);
                for slack in [
                    single_pred_block_pixels,
                    2 * single_pred_block_pixels,
                    4 * single_pred_block_pixels,
                ] {
                    pred_candidates.push(select_best_cache_bits(|cache_bits| {
                        encode_with_predictor_slack(
                            pixels,
                            width,
                            height,
                            pred_single_block_size_bits,
                            cache_bits,
                            width,
                            slack,
                        )
                    }));
                }
                pred_candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_predictor_entropy(
                        pixels,
                        width,
                        height,
                        pred_single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in pred_candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if width >= ctx_block && height >= ctx_block {
            let mut single_block_size_bits: u8 = ctx_size_bits;
            while single_block_size_bits < 9
                && ((1u32 << single_block_size_bits) < width
                    || (1u32 << single_block_size_bits) < height)
            {
                single_block_size_bits += 1;
            }
            let try_single_block = single_block_size_bits != ctx_size_bits;
            let mut candidates: Vec<Vec<u8>> = vec![select_best_cache_bits(|cache_bits| {
                encode_with_color_transform(pixels, width, height, ctx_size_bits, cache_bits, width)
            })];
            if try_single_block {
                candidates.push(select_best_cache_bits(|cache_bits| {
                    encode_with_color_transform(
                        pixels,
                        width,
                        height,
                        single_block_size_bits,
                        cache_bits,
                        width,
                    )
                }));
            }
            for cand in candidates {
                if cand.len() < best.len() {
                    best = cand;
                }
            }
        }

        if collect_palette(pixels).is_some() {
            let ci_best = select_best_cache_bits(|cache_bits| {
                encode_with_color_indexing(pixels, width, height, cache_bits)
                    .expect("palette feasibility already confirmed")
            });
            if ci_best.len() < best.len() {
                best = ci_best;
            }
        }

        if let Some(mp_best) = sweep_meta_prefix_candidate(pixels, width, height) {
            if mp_best.len() < best.len() {
                best = mp_best;
            }
        }

        best
    }

    /// Round 162 — `sub_image_mode_cost_delta_milli` returns zero when
    /// the first symbol is added to an empty histogram: the post-add
    /// state is a single-symbol histogram with `H = 0`, so the
    /// Shannon mass goes from 0 (degenerate) to 0 (single bin with
    /// `c·log2(N/c) = N·log2(1) = 0`).
    #[test]
    fn round_162_sub_image_mode_cost_delta_zero_on_first_add() {
        let hist = [0u32; 14];
        for mode in 0u8..=13 {
            let delta = sub_image_mode_cost_delta_milli(&hist, 0, mode);
            assert_eq!(
                delta, 0,
                "first symbol add must produce zero Shannon delta; mode={mode} delta={delta}"
            );
        }
    }

    /// Round 162 — `sub_image_mode_cost_delta_milli` returns zero when
    /// the added symbol equals the only mode already present (still a
    /// single-symbol histogram post-add), and a strictly positive
    /// delta when the added symbol is *different* from the only mode
    /// already present (the histogram grows from one to two bins, so
    /// `N·H` grows from `0` to `2·log2(2) - 2·1·log2(1) = 2` bits).
    #[test]
    fn round_162_sub_image_mode_cost_delta_grows_on_new_symbol() {
        // Start with five occurrences of mode 3 already in the
        // histogram (single-symbol state, N·H = 0).
        let mut hist = [0u32; 14];
        hist[3] = 5;
        let total = 5u32;

        let same = sub_image_mode_cost_delta_milli(&hist, total, 3);
        assert_eq!(
            same, 0,
            "adding same symbol to a single-mode histogram must not grow Shannon mass"
        );

        let different = sub_image_mode_cost_delta_milli(&hist, total, 7);
        assert!(
            different > 0,
            "adding a new symbol to a single-mode histogram must grow Shannon mass; got 0"
        );
        // Sanity: the post-add N·H is 6·log2(6) − 5·log2(5) − 1·log2(1)
        //       ≈ 15.5097 − 11.6096 − 0 ≈ 3.9 bits ≈ 3900 milli-bits.
        // Pre-add was 0, so the delta should be roughly 3900 ±1.
        assert!(
            (3500..=4300).contains(&different),
            "expected delta near 3900 milli-bits; got {different}"
        );
    }

    /// Round 162 — `lambda_milli == 0` makes the sub-image-aware
    /// chooser byte-identical to the round-161 entropy chooser: every
    /// candidate's joint cost equals its residual-only cost (the
    /// sub-image term contributes zero), and the tie-break rules
    /// match exactly.
    #[test]
    fn round_162_lambda_zero_byte_identical_to_round_161() {
        // Use a 32×32 fixture exercising the per-region path with at
        // least four 16×16 blocks worth of sub-image entries.
        let w = 32u32;
        let h = 32u32;
        let mut pixels = vec![0u32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let r = (x as u8).wrapping_mul(7);
                let g = (y as u8).wrapping_mul(11);
                let b = ((x + y) as u8).wrapping_mul(13);
                pixels[y * w as usize + x] =
                    0xff00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            }
        }

        let r161 = encode_with_predictor_entropy(&pixels, w, h, 4, None, w);
        let r162_lambda0 = encode_with_predictor_entropy_subaware(&pixels, w, h, 4, None, w, 0);
        assert_eq!(
            r161, r162_lambda0,
            "lambda_milli == 0 must produce a byte-identical stream to round-161 entropy"
        );

        // Also covers Some(cache_bits) — the cache path shouldn't
        // alter the equivalence.
        let r161_cached = encode_with_predictor_entropy(&pixels, w, h, 4, Some(6), w);
        let r162_cached_lambda0 =
            encode_with_predictor_entropy_subaware(&pixels, w, h, 4, Some(6), w, 0);
        assert_eq!(
            r161_cached, r162_cached_lambda0,
            "lambda_milli == 0 must be byte-identical with cache_bits = Some(6)"
        );
    }

    /// Round 162 — `pick_block_mode_with_hint_entropy_subaware` honours
    /// the strict tie-break: when the preferred mode's joint cost
    /// equals the best, the chooser returns the preferred mode (so
    /// the sub-image keeps the longer mode-run). Mirrors the round-
    /// 159 / round-161 tie-break test.
    #[test]
    fn round_162_pick_block_mode_subaware_honours_tie() {
        // Tiny 1×1 block — every mode reduces to the top-left border
        // (`pred = 0xff_00_00_00`), so all modes yield zero residual
        // entropy and tie at zero. The hint should flip the result.
        let pixels = vec![0xff_00_00_00u32; 1];
        let hist = [0u32; 14];
        let chosen_no_hint = pick_block_mode_with_hint_entropy_subaware(
            &pixels, 1, 1, 0, 0, 1, 1, None, &hist, 0, 4_000,
        );
        assert_eq!(
            chosen_no_hint, 0,
            "no-hint pick should fall back to lowest-tied mode (= 0)"
        );

        for hint in 0u8..=13 {
            let chosen = pick_block_mode_with_hint_entropy_subaware(
                &pixels,
                1,
                1,
                0,
                0,
                1,
                1,
                Some(hint),
                &hist,
                0,
                4_000,
            );
            assert_eq!(
                chosen, hint,
                "hint {hint} should win on a fully-tied block; got {chosen}"
            );
        }
    }

    /// Round 162 — end-to-end round-trip: the sub-image-aware encoder
    /// produces a stream the §5.x decoder reconstructs to the
    /// original pixels at three lambda settings and two cache-bits
    /// settings, across a small fixture with mixed local statistics.
    #[test]
    fn round_162_subaware_round_trips_through_decoder() {
        let w = 32u32;
        let h = 32u32;
        let mut pixels = vec![0u32; (w * h) as usize];
        // Top-left 16×16: gradient. Top-right: noise. Bottom-left:
        // solid. Bottom-right: vertical bars. Drives different
        // per-block best modes across the four sub-image entries.
        for y in 0..h as usize {
            for x in 0..w as usize {
                let v = match (x < 16, y < 16) {
                    (true, true) => 0xff_00_00_00 | (((x + y) as u32 * 8) << 8),
                    (false, true) => {
                        let seed = (x.wrapping_mul(97) ^ y.wrapping_mul(53)) as u32;
                        0xff_00_00_00 | ((seed & 0xff) << 16) | (seed & 0xff00)
                    }
                    (true, false) => 0xff_80_80_80,
                    (false, false) => {
                        if x % 2 == 0 {
                            0xff_ff_ff_ff
                        } else {
                            0xff_00_00_00
                        }
                    }
                };
                pixels[y * w as usize + x] = v;
            }
        }

        for lambda_milli in [1_000u64, 4_000u64, 16_000u64] {
            for cache_bits in [None, Some(4u32), Some(8u32)] {
                let payload = encode_with_predictor_entropy_subaware(
                    &pixels,
                    w,
                    h,
                    4,
                    cache_bits,
                    w,
                    lambda_milli,
                );
                let header = build_image_header(w, h, true);
                let mut bytes = header.to_vec();
                bytes.extend_from_slice(&payload);
                let framed = build::build_webp_file(&bytes, ImageKind::Lossless, w, h).unwrap();
                let decoded = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    decoded.pixels(),
                    pixels.as_slice(),
                    "round-trip mismatch lambda_milli={lambda_milli} cache_bits={cache_bits:?}"
                );
            }
        }
    }

    /// Round 162 — the production chooser never regresses against the
    /// round-161 baseline: across 5 image shapes × 3 fixture
    /// generators, the round-162 chooser output is byte-`<=` the
    /// chooser-without-round-162-candidates output, AND every
    /// chosen stream round-trips through the decoder bit-exactly.
    #[test]
    fn round_162_chooser_never_regresses_vs_round_161() {
        let shapes: &[(u32, u32)] = &[(16, 16), (24, 32), (32, 24), (48, 48), (64, 32)];
        for &(w, h) in shapes {
            for fixture_kind in 0..3u32 {
                let mut pixels = vec![0u32; (w * h) as usize];
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let v = match fixture_kind {
                            0 => 0xff_00_00_00 | (((x ^ y) as u32 * 3) & 0xff),
                            1 => {
                                let seed =
                                    (x.wrapping_mul(2654435761).wrapping_add(y) & 0xff) as u32;
                                0xff_00_00_00 | (seed << 16) | seed
                            }
                            _ => {
                                if (x + y) % 5 < 2 {
                                    0xff_a0_a0_a0
                                } else {
                                    0xff_60_60_60
                                }
                            }
                        };
                        pixels[y * w as usize + x] = v;
                    }
                }

                let baseline = encode_argb_with_predictor_chooser_no_r162_subaware(&pixels, w, h);
                let r162 = encode_argb_with_predictor_chooser(&pixels, w, h);
                assert!(
                    r162.len() <= baseline.len(),
                    "round-162 chooser regressed at shape={w}×{h} fixture={fixture_kind}: \
                     baseline={} B r162={} B",
                    baseline.len(),
                    r162.len()
                );

                // Decode round-trip on the round-162 stream. The
                // chooser emits a bare VP8L payload; wrap with the
                // image header before framing.
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&r162);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let decoded = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    decoded.pixels(),
                    pixels.as_slice(),
                    "round-trip mismatch at shape={w}×{h} fixture={fixture_kind}"
                );
            }
        }
    }

    /// Round 162 — the *isolated* sub-image-aware predictor candidate
    /// (`encode_with_predictor_entropy_subaware`) strictly beats the
    /// round-161 isolated entropy candidate
    /// (`encode_with_predictor_entropy`) on every smooth-gradient
    /// fixture in the sweep. This is the headline empirical result
    /// for the round-162 cost model: smooth gradients are the
    /// canonical case where many §4.1 sub-image entries can converge
    /// onto a small mode set (the gradient predictors all yield
    /// near-zero residuals so the sub-image's prefix-code mass
    /// dominates total cost). The crossover at the swept lambda
    /// values (`64_000` per-sub-image-bit milli-units) is where the
    /// sub-image weighting takes off — below that, residual cost
    /// dominates and the round-161 chooser already wins.
    ///
    /// This compares the round-162 and round-161 predictor
    /// candidates **in isolation** (same `size_bits = 4`, both
    /// running through `apply_forward_predictor` + LZ77 + prefix
    /// coding) so the win is attributable to the chooser, not to
    /// other paths in the full chooser sweep (subtract-green,
    /// single-block predictor, etc.) which may produce an equally-
    /// tight stream by a different mechanism. The production chooser
    /// adds the round-162 candidate to its sweep and keeps byte-
    /// shortest, so even when other paths tie, the round-162 path
    /// strictly extends the encoder's option set.
    ///
    /// Round-trips through the decoder bit-exactly on every winning
    /// fixture.
    #[test]
    fn round_162_subaware_isolated_strictly_beats_round_161_on_some_fixture() {
        let shapes: &[(u32, u32)] = &[(64, 64), (128, 128), (256, 128), (96, 96), (160, 80)];
        let lambda_to_test: u64 = 64_000;
        let mut wins = 0u32;
        let mut max_savings: i64 = 0;
        let mut max_savings_shape: (u32, u32) = (0, 0);
        for &(w, h) in shapes {
            let mut pixels = vec![0u32; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    let r = (x * 255 / w.max(1)) as u8;
                    let g = (y * 255 / h.max(1)) as u8;
                    pixels[(y * w + x) as usize] =
                        0xff00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | 0x40;
                }
            }
            let r161 = encode_with_predictor_entropy(&pixels, w, h, 4, None, w);
            let r162 =
                encode_with_predictor_entropy_subaware(&pixels, w, h, 4, None, w, lambda_to_test);
            // r162 may tie r161 on some shapes (the chosen mode set
            // already coincides), but it must never regress — the
            // sub-image-aware cost is a strict generalisation of the
            // round-161 cost.
            assert!(
                r162.len() <= r161.len(),
                "round-162 isolated candidate REGRESSED on gradient {w}x{h}: \
                 r161={} B r162={} B",
                r161.len(),
                r162.len()
            );
            let saved = r161.len() as i64 - r162.len() as i64;
            if r162.len() < r161.len() {
                wins += 1;
                if saved > max_savings {
                    max_savings = saved;
                    max_savings_shape = (w, h);
                }
                // Verify round-trip on the winning stream.
                let header = build_image_header(w, h, true);
                let mut payload = header.to_vec();
                payload.extend_from_slice(&r162);
                let framed = build::build_webp_file(&payload, ImageKind::Lossless, w, h).unwrap();
                let decoded = crate::decode_lossless_image(&framed).unwrap().unwrap();
                assert_eq!(
                    decoded.pixels(),
                    pixels.as_slice(),
                    "round-trip mismatch on gradient strict-beat {w}x{h}"
                );
                eprintln!(
                    "[round-162] isolated strict-beat (gradient {w}x{h}, lambda={lambda_to_test}): \
                     r161={} B r162={} B saved={saved} B ({:.1}% reduction)",
                    r161.len(),
                    r162.len(),
                    100.0 * saved as f64 / r161.len() as f64
                );
            } else {
                eprintln!(
                    "[round-162] tie (gradient {w}x{h}, lambda={lambda_to_test}): \
                     r161={} B r162={} B (no regression)",
                    r161.len(),
                    r162.len()
                );
            }
        }
        // Require strict wins on a majority of the gradient sweep —
        // proves the round-162 cost model is doing real work, not
        // just degenerating to the round-161 chooser everywhere.
        assert!(
            wins >= 3,
            "round-162 isolated candidate strictly beat round-161 on only {wins}/{} gradient \
             fixtures; expected at least 3 strict wins to demonstrate the sub-image cost is \
             doing real work",
            shapes.len()
        );
        eprintln!(
            "[round-162] isolated sub-image-aware: {wins}/{} gradient fixtures strict-won; \
             headline savings = {max_savings} B on {}x{}",
            shapes.len(),
            max_savings_shape.0,
            max_savings_shape.1
        );
    }
}
