//! VP8L lossless encoder.
//!
//! Pure-Rust VP8L encoder emitting a valid bitstream decodable by the
//! in-crate [`super::decode`]. Compared to libwebp the output is coarser
//! — the compression ratio gap is documented below — but we cover the
//! bits that matter most in practice:
//!
//! * **Subtract-green transform** (always on). Removes the common
//!   photographic correlation between the G/R and G/B channels by
//!   sending `r-g` and `b-g` instead of `r` and `b`.
//! * **Colour transform** (always on, tile-based G↔R/B decorrelation).
//!   For each 32×32 tile we search a coarse grid of 256 coefficient
//!   combinations (16 × 16 over the `g→r` / `g→b` pair with a
//!   post-optimisation pass on `r→b`) and keep the one that minimises
//!   sum-of-abs residuals. Runs after subtract-green, so the inverse
//!   order on decode is predictor → colour → add-green — matching
//!   libwebp.
//! * **Predictor transform** (always on, tile-based). Each 16×16 tile
//!   picks the best of all 14 VP8L predictor modes (RFC 9649 §4.1) by
//!   forward-pass sum-of-abs-residuals cost; the tile modes ride in a
//!   sub-image pixel stream.
//! * **Colour cache** (always on, 256 entries). Every literal pixel is
//!   also addressable by its hashed cache index, which shortens the
//!   green alphabet on repeat colours.
//!
//! * **Colour-indexing (palette) transform.** Triggered automatically
//!   when the image has ≤ 256 unique ARGB colours. Replaces every pixel
//!   with a small palette index (1, 2, 4, or 8 bits per index, packed
//!   into the green channel of the index image) and ships a delta-coded
//!   palette out of band. Wins 2-5× on icons, line art, and screenshots
//!   — and the index image often compresses *further* via subtract-green
//!   + LZ77 + colour-cache once the channel-decorrelation is no longer
//!   the bottleneck.
//!
//! What we still don't do (compared to libwebp):
//!
//! * **No meta-Huffman image.** A single Huffman group covers the
//!   whole picture.
//!
//! What *is* implemented end-to-end:
//!
//! * Length-limited canonical Huffman tree builder (≤15 bits per code,
//!   matching the VP8L spec's §5 limit) using a frequency-driven sort +
//!   depth-capping redistribution pass.
//! * Canonical-Huffman code-length tree emission, reusing the 19-symbol
//!   meta-alphabet + run-length codes 16/17/18 expected by the decoder.
//! * A 4 KB sliding-window, hash-chain LZ77 matcher over the residual
//!   pixel sequence. Matches of length ≥ 3 are emitted as (length,
//!   distance) pairs using the VP8L length-or-distance symbol scheme.
//!   Distances are always emitted in the `code = d + 120` form, so the
//!   short-distance diamond table isn't consulted on the encoder side.
//!
//! The entry point is [`encode_vp8l_argb`]: a bare VP8L bitstream (no
//! RIFF wrapper) sized for a given `width × height` ARGB pixel buffer.

use crate::error::{Result, WebpError as Error};

use super::VP8L_SIGNATURE;

/// Maximum Huffman code length allowed by the VP8L spec.
const MAX_CODE_LENGTH: u8 = 15;

/// LZ77 window size (in pixels). 4K pixels is plenty for the small-image
/// roundtrip tests and keeps the hash chain tight.
const LZ_WINDOW: usize = 4096;

/// Minimum LZ77 match length we're willing to emit. Shorter matches lose
/// to simple literals once the length/distance bits are counted.
const MIN_MATCH: usize = 3;

/// Maximum LZ77 match length. The VP8L length alphabet tops out well
/// above this but long runs are rare in ARGB data and short-chain hash
/// searches get expensive past a few hundred pixels.
const MAX_MATCH: usize = 4096;

/// Colour-cache bit width. 8 bits = 256-entry cache — small enough to
/// keep the green alphabet compact (256 + 24 + 256 = 536 symbols) yet
/// large enough to pay for itself on most natural images. Always on;
/// degenerate images still compress correctly because the cache-index
/// alphabet's Huffman tree collapses to a handful of symbols.
const COLOR_CACHE_BITS: u32 = 8;

/// Side length (in pixels) of a predictor tile. VP8L carries `tile_bits
/// = 2..=9`, i.e. 4..=512 pixels per side. 16 is the libwebp default
/// and strikes a reasonable balance between side-image overhead and
/// per-block mode accuracy.
const PREDICTOR_TILE_BITS: u32 = 4; // 16-pixel tiles

/// Side length (in pixels) of a colour-transform tile. The spec allows
/// 2..=9 (4..=512). 32 is libwebp's default for the colour transform:
/// large enough that the 256-combo per-tile search stays cheap (one scan
/// per candidate over 1 K pixels) yet small enough that coefficient
/// choice can track real spatial variation.
const COLOR_TRANSFORM_TILE_BITS: u32 = 5; // 32-pixel tiles

/// Candidate values for the `g→r` and `g→b` colour-transform
/// coefficients. Spec §3.6.6 stores each coefficient as a signed int8
/// and weights the per-pixel delta by `>> 5`, so every step of 1 is a
/// ~3 % correction — the grid below covers the useful range with
/// 16 entries (-24..=21, step 3). 16 × 16 = 256 per tile, matching the
/// per-tile search budget set by the colour-transform design note.
const COLOR_COEFF_GRID: [i8; 16] = [
    -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21,
];

/// Coarse sweep for the `r→b` coefficient, run once per tile *after*
/// the (g→r, g→b) grid has picked a best pair. Five values are enough
/// in practice — `r→b` wins are smaller than the green-axis ones.
const COLOR_R2B_GRID: [i8; 5] = [-12, -6, 0, 6, 12];

/// Predictor modes we're willing to pick between on the encoder side.
/// We probe all 14 VP8L predictor modes (RFC 9649 §4.1) and let the
/// per-tile sum-of-abs-residuals scan pick the cheapest. The earlier
/// pool was `[0, 1, 2, 11]` — fine on flat / left-correlated / top-
/// correlated / "select"-friendly content but blind to:
///
/// * 3 — top-right (handles diagonal stripes leaning the wrong way).
/// * 4 — top-left (catches strong NW-SE correlation).
/// * 5..10 — neighbour averages (smooth content where mean-of-2 or
///   mean-of-3 beats every single-neighbour mode, especially on the
///   green-decorrelated residual stream).
/// * 12 — clamped L+T-TL (Paeth-like; another natural-image staple).
/// * 13 — clamped (avg(L,T)) + half delta (handles content that's
///   "almost an average" of two neighbours but with a slight bias).
///
/// Cost per tile is one residual-sum scan per candidate, so the budget
/// scales linearly with the pool size — going from 4 modes to 14 is a
/// 3.5× per-tile cost which still amortises cheaply against entropy
/// coding (the sub-image is one byte per tile per the spec).
const PREDICTOR_MODES: &[u32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

/// LSB-first bit writer matching the VP8L decoder's bit-reader convention.
struct BitWriter {
    out: Vec<u8>,
    cur: u64,
    nbits: u32,
}

/// Snapshot of [`BitWriter`] state. Used by the meta-Huffman trial path
/// to roll back a speculative encode when the single-group baseline
/// turns out to be smaller.
#[derive(Clone, Copy)]
struct BitWriterMark {
    out_len: usize,
    cur: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    fn write(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        let mask = if n == 0 {
            0u64
        } else if n == 32 {
            0xFFFF_FFFFu64
        } else {
            (1u64 << n) - 1
        };
        self.cur |= ((value as u64) & mask) << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.out.push((self.cur & 0xff) as u8);
            self.cur >>= 8;
            self.nbits -= 8;
        }
    }

    /// Capture the writer's current position so a speculative chunk can
    /// be rolled back later via [`Self::restore`]. Cheap (one `usize` +
    /// scalar copies).
    fn mark(&self) -> BitWriterMark {
        BitWriterMark {
            out_len: self.out.len(),
            cur: self.cur,
            nbits: self.nbits,
        }
    }

    /// Roll the writer back to a prior [`BitWriterMark`]. Truncates
    /// `out` and restores the staging buffer + bit count to the saved
    /// values. Required for the meta-Huffman trial path: we encode both
    /// candidates speculatively into the live writer, pick the smaller,
    /// and rewind to re-emit the winner cleanly.
    fn restore(&mut self, mark: BitWriterMark) {
        self.out.truncate(mark.out_len);
        self.cur = mark.cur;
        self.nbits = mark.nbits;
    }

    /// Bit position of the writer's tail, in bits since the start of
    /// the output. Used to score speculative encodes — `bits_emitted -
    /// mark.bit_pos()` gives the bit length of the chunk written since
    /// `mark`.
    fn bit_pos(&self) -> u64 {
        (self.out.len() as u64) * 8 + self.nbits as u64
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.cur & 0xff) as u8);
        }
        self.out
    }
}

impl BitWriterMark {
    fn bit_pos(&self) -> u64 {
        (self.out_len as u64) * 8 + self.nbits as u64
    }
}

/// Encode `width × height` ARGB pixels (one u32 per pixel: `a<<24 | r<<16 |
/// g<<8 | b`) as a bare VP8L bitstream (no RIFF wrapper).
///
/// `has_alpha` sets the `alpha_is_used` header bit. It's purely advisory
/// — the alpha channel is transmitted either way.
///
/// Internally runs a 32-trial RDO sweep over the four optional VP8L
/// transforms × four colour-cache widths and keeps the smallest
/// encoded variant. Pixels with `alpha == 0` get their RGB stripped to
/// zero by default (visually identical, compresses better — matches
/// libwebp's default). Callers who need fully deterministic behaviour
/// for a fixed transform configuration or who need to preserve the RGB
/// bytes of fully-transparent pixels should use [`encode_vp8l_argb_with`].
pub fn encode_vp8l_argb(
    width: u32,
    height: u32,
    pixels: &[u32],
    has_alpha: bool,
) -> Result<Vec<u8>> {
    encode_vp8l_argb_rdo(width, height, pixels, has_alpha)
}

/// Replace the RGB bytes of every alpha-zero pixel with `0`. The ARGB
/// layout puts alpha in the top 8 bits and the masked region (RGB) in
/// the bottom 24, so a single `& 0xff00_0000` per such pixel does it.
///
/// Choosing `0` is the simplest deterministic fill: the predictor +
/// LZ77 + colour cache then collapse runs of fully-transparent pixels
/// into a handful of cache hits or backreferences. libwebp uses a more
/// elaborate "predict from neighbours" heuristic for marginally better
/// compression; that's a future enhancement.
fn strip_transparent_rgb(pixels: &mut [u32]) {
    for p in pixels.iter_mut() {
        if (*p >> 24) & 0xff == 0 {
            *p &= 0xff00_0000;
        }
    }
}

/// Translate the public `near_lossless` knob (0..=100, libwebp scale)
/// into a per-channel low-bit shift, or `None` when preprocessing is
/// disabled. The shift is applied to each R/G/B byte by rounding to the
/// nearest multiple of `1 << shift`, which collapses 2..16 source values
/// onto a single quantised representative — making the residual stream
/// more compressible.
///
/// Calibration follows libwebp's spirit: 100 is a no-op, 60 quantises
/// LSBs only (1-bit shift), 40 ≈ 2-bit, 20 ≈ 3-bit, 0 = maximum (4-bit).
/// Anything in between is rounded down to the next breakpoint:
///
/// | level    | shift |
/// |----------|-------|
/// | 100      | off   |
/// | 80..=99  | 1     |
/// | 60..=79  | 1     |
/// | 40..=59  | 2     |
/// | 20..=39  | 3     |
/// | 0..=19   | 4     |
///
/// 4 bits is the max we apply: above that the visual drift becomes
/// obvious (steps of 32 on each channel) and gains from extra
/// quantisation tail off because the predictor and colour-cache already
/// fold runs of repeated values.
fn near_lossless_shift(level: u8) -> Option<u32> {
    // Treat anything > 100 as "off" (matches the libwebp default).
    // Callers who pick 100 or anything above it pay no quantisation cost.
    match level {
        100..=u8::MAX => None,
        60..=99 => Some(1),
        40..=59 => Some(2),
        20..=39 => Some(3),
        0..=19 => Some(4),
    }
}

/// Apply a per-channel near-lossless quantisation pass in place. Each
/// R/G/B byte is rounded to the nearest multiple of `step = 1 << shift`,
/// using a half-step bias (`+ step/2`) so the rounding is to-nearest
/// rather than to-floor. The result is then clamped to `[0, 255]` —
/// adding the bias to a value near 255 can otherwise push it past the
/// representable range.
///
/// Alpha is left untouched: keeping transparency exact matters more
/// than the marginal gain from rounding it, and the strip-transparent
/// pass (run earlier when enabled) already collapses RGB on alpha=0
/// pixels.
///
/// This pass intentionally does *not* try to be neighbourhood-aware —
/// the expensive smoothing libwebp does on top of the bit-shift is a
/// further refinement; the bare quantisation step alone already
/// captures most of the size win on photographic content. Smoothing
/// can be layered on later as a second pass if benchmarks justify it.
fn apply_near_lossless(pixels: &mut [u32], shift: u32) {
    debug_assert!((1..=7).contains(&shift));
    let step = 1u32 << shift;
    let bias = step >> 1;
    let mask = !(step - 1);
    for p in pixels.iter_mut() {
        let a = (*p >> 24) & 0xff;
        let r = (*p >> 16) & 0xff;
        let g = (*p >> 8) & 0xff;
        let b = *p & 0xff;
        let nr = quantise_channel(r, bias, mask);
        let ng = quantise_channel(g, bias, mask);
        let nb = quantise_channel(b, bias, mask);
        *p = (a << 24) | (nr << 16) | (ng << 8) | nb;
    }
}

#[inline]
fn quantise_channel(v: u32, bias: u32, mask: u32) -> u32 {
    // (v + bias) & mask, clamped at 255. The bias can push values past
    // 255 (e.g. 254 + 8 = 262 with shift=4), so clamp before applying
    // the mask so the result stays in `[0, 255]`.
    let raised = (v + bias).min(255);
    raised & mask
}

/// Encoder tuning knobs. Hidden from the public docs — primarily a
/// testing surface for sizing transforms on/off against each other.
///
/// `cache_bits` is honoured only when `use_color_cache` is `true`. Valid
/// values are 1..=11 per the VP8L spec; the encoder's
/// [`COLOR_CACHE_BITS`] default lands on 8 (256-entry cache).
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct EncoderOptions {
    pub use_subtract_green: bool,
    pub use_color_transform: bool,
    pub use_predictor: bool,
    pub use_color_cache: bool,
    /// Width of the colour-cache index in bits. Ignored when
    /// `use_color_cache == false`. Outside 1..=11 falls back to
    /// [`COLOR_CACHE_BITS`].
    pub cache_bits: u32,
    /// When `true`, every pixel whose alpha channel is zero has its
    /// RGB bytes replaced with `0` before encoding. Fully transparent
    /// pixels are visually invisible, so the RGB component is free —
    /// collapsing it to a constant lets the predictor + entropy coder
    /// pack the alpha-zero regions much more tightly. Mirrors libwebp's
    /// `WebPConfig::exact == false` default; set to `false` to preserve
    /// the input bytes exactly (raises file size, useful for callers
    /// that round-trip RGB out-of-band of the alpha channel).
    pub strip_transparent_color: bool,
    /// When `true`, the encoder probes the unique-colour count and emits
    /// a [colour-indexing transform](super::transform::Transform::ColorIndex)
    /// whenever it can fit in 256 entries. The pixel stream is replaced
    /// with palette indices (1/2/4/8 bits per pixel depending on the
    /// palette size). Setting this to `false` forces the full ARGB
    /// path even on palettised images — useful for testing the
    /// non-palette transforms in isolation. When the image has > 256
    /// unique colours the encoder transparently falls back to the ARGB
    /// path regardless of this flag.
    pub use_color_index: bool,
    /// Near-lossless preprocessing intensity, on libwebp's `cwebp
    /// -near_lossless N` scale. `100` = OFF (default; bit-identical
    /// lossless). Lower values quantize per-channel pixel values to
    /// nearest multiples of `1 << shift`, where `shift` grows as the
    /// level falls (60 → 1 bit, 40 → 2 bits, 20 → 3 bits, 0 → 4 bits).
    /// Quantization is rounded to the closest representable value (with
    /// ties to even) and clamped to `[0, 255]`; alpha is left untouched
    /// so transparency is exact. The output is still a fully-spec-
    /// compliant lossless VP8L stream — the lossy step is purely a
    /// pre-encode pixel rewrite that improves the entropy coder's
    /// statistics. Recommended for photographic content where 1-2 LSBs
    /// of colour drift are imperceptible; leave at `100` for
    /// pixel-perfect content (icons, screenshots, line art).
    pub near_lossless: u8,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            use_subtract_green: true,
            use_color_transform: true,
            use_predictor: true,
            use_color_cache: true,
            cache_bits: COLOR_CACHE_BITS,
            strip_transparent_color: true,
            use_color_index: true,
            near_lossless: 100,
        }
    }
}

impl EncoderOptions {
    /// All transforms off — the pre-transform "literals only" baseline.
    #[doc(hidden)]
    pub fn bare() -> Self {
        Self {
            use_subtract_green: false,
            use_color_transform: false,
            use_predictor: false,
            use_color_cache: false,
            cache_bits: COLOR_CACHE_BITS,
            strip_transparent_color: true,
            use_color_index: false,
            near_lossless: 100,
        }
    }

    /// Subtract-green only. Used by the per-transform shrinkage tests so
    /// the colour-transform contribution can be isolated from the other
    /// passes.
    #[doc(hidden)]
    pub fn subtract_green_only() -> Self {
        Self {
            use_subtract_green: true,
            use_color_transform: false,
            use_predictor: false,
            use_color_cache: false,
            cache_bits: COLOR_CACHE_BITS,
            strip_transparent_color: true,
            use_color_index: false,
            near_lossless: 100,
        }
    }
}

#[doc(hidden)]
pub fn encode_vp8l_argb_with(
    width: u32,
    height: u32,
    pixels: &[u32],
    has_alpha: bool,
    opts: EncoderOptions,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(Error::invalid("VP8L encoder: zero-size image"));
    }
    if width > 16384 || height > 16384 {
        return Err(Error::invalid("VP8L encoder: max dimension 16384"));
    }
    if (pixels.len() as u64) != (width as u64) * (height as u64) {
        return Err(Error::invalid("VP8L encoder: pixel count mismatch"));
    }

    let mut bw = BitWriter::new();
    // Signature.
    bw.write(VP8L_SIGNATURE as u32, 8);
    // 14-bit width-minus-1 / 14-bit height-minus-1. Decoder masks ×0x3fff
    // and reconstructs via `+1`, so cap at 16384 (done above).
    bw.write(width - 1, 14);
    bw.write(height - 1, 14);
    bw.write(if has_alpha { 1 } else { 0 }, 1);
    bw.write(0, 3); // version

    // ── Transform chain ──────────────────────────────────────────────
    //
    // Transforms are written in the order the encoder applies them;
    // the decoder iterates them front-to-back and applies in reverse,
    // so the LAST transform we write is the FIRST the decoder inverts.
    // Subtract-green is written before predictor so the predictor works
    // on the already green-decorrelated residuals (matching libwebp).

    let mut working = pixels.to_vec();

    // Optional pre-transform pass: collapse the RGB of fully-transparent
    // pixels to `0`. Lossless from the consumer's point of view (alpha=0
    // hides the colour anyway) and a clear win for the entropy coder.
    if opts.strip_transparent_color {
        strip_transparent_rgb(&mut working);
    }

    // Optional near-lossless preprocessing. Lossy on the colour
    // channels (alpha is preserved); the resulting bitstream is still
    // a fully-spec-compliant lossless VP8L stream — this is a pure
    // pre-encode pixel rewrite that improves entropy-coder statistics.
    if let Some(shift) = near_lossless_shift(opts.near_lossless) {
        apply_near_lossless(&mut working, shift);
    }

    // ── Optional colour-indexing (palette) transform ────────────────
    //
    // Detect ≤ 256 unique ARGB colours and, when present, emit the
    // palette transform first (so the decoder applies it last —
    // expanding the index back into ARGB at the very end). The pixel
    // stream that follows then carries small integer indices, which
    // the rest of the transform chain (subtract-green / predictor /
    // colour-cache) can compress further.
    //
    // Per RFC 9649 §3.6.5 the palette index goes into the green
    // channel of the index image, with R=B=A masked to a known
    // constant (the decoder ignores R/B and uses the palette entry's
    // own alpha). When num_colors ≤ 16 the indices are bit-packed —
    // the spec's `bits_per_pixel` derivation matches what
    // [`super::transform::Transform::read`] reconstructs.
    //
    // Subtract-green is suppressed when the palette is in use because
    // the green channel is now an index (not a colour) and the
    // decorrelation it relies on no longer applies.
    let mut current_width = width;
    let mut palette_active = false;
    if opts.use_color_index {
        if let Some(palette) = build_palette(&working) {
            palette_active = true;
            let bits_per_pixel = bits_per_pixel_for(palette.len() as u32);
            let pack = 8u32 / bits_per_pixel;
            let packed_w = (width + pack - 1) / pack;

            // Transform header: present + type 3 (ColorIndexing).
            bw.write(1, 1);
            bw.write(3, 2);
            // num_colors - 1, 8-bit.
            bw.write(palette.len() as u32 - 1, 8);
            // Delta-encode the palette along the row (decoder undoes
            // this with a forward `add_argb` walk per spec §3.6.5).
            let palette_delta = delta_encode_palette(&palette);
            // The palette ships as an `image stream` of (num_colors)×1
            // pixels. Sub-image, no cache, no meta-Huffman.
            encode_image_stream(&mut bw, &palette_delta, palette.len() as u32, 1, false, 0)?;

            // Replace the working pixel buffer with the packed index
            // image: each output pixel carries `pack` palette indices
            // in its green channel low bits.
            working =
                pack_palette_indices(&working, width, height, &palette, bits_per_pixel, packed_w);
            current_width = packed_w;
        }
    }

    if !palette_active && opts.use_subtract_green {
        // Transform header: present + type 2 (SubtractGreen).
        bw.write(1, 1);
        bw.write(2, 2);
        apply_subtract_green_forward(&mut working);
    }

    // Colour-transform and predictor are skipped on palette-encoded
    // images: the green channel of the index image is a small integer
    // (0..palette.len()-1), not a colour, so the per-channel
    // decorrelation that the colour transform models doesn't apply, and
    // the predictor's "expand the residual to a full byte" step would
    // typically inflate (small palettes leave the high bits of the
    // green byte zero — predictor residuals fill those bits with noise
    // and break the bit-packing prerequisite anyway). Cache-only on
    // palette streams already gives most of the LZ77 / repeat-symbol
    // gains without the side-image overhead.
    if !palette_active && opts.use_color_transform {
        // Forward colour transform: per-tile search over the
        // [`COLOR_COEFF_GRID`] × [`COLOR_COEFF_GRID`] grid (256 combos)
        // plus a follow-up `r→b` sweep. Emits a predictor-shaped
        // sub-image with one ARGB pixel per tile.
        //
        // Coefficient packing per WebP lossless spec §4.2:
        //   A = 255 (unused), R = red_to_blue, G = green_to_blue,
        //   B = green_to_red.
        // The previous version had R and B swapped (g2r in R, r2b in
        // B); rust round-tripped fine but libwebp couldn't decode the
        // result. Fixed in lockstep with the matching decoder change
        // — see transform.rs::apply_color_transform.
        let tile_bits = COLOR_TRANSFORM_TILE_BITS;
        let tile_side = 1u32 << tile_bits;
        let sub_w = (width + tile_side - 1) / tile_side;
        let sub_h = (height + tile_side - 1) / tile_side;
        let coeffs = choose_color_transform(&working, width, height, tile_bits, sub_w, sub_h);

        // Transform header: present + type 1 (ColorTransform) + tile_bits-2.
        bw.write(1, 1);
        bw.write(1, 2);
        bw.write(tile_bits - 2, 3);

        // Sub-image: 0xff alpha so the decoder's generic ARGB decode
        // matches what the decoder's `apply_color_transform` then reads
        // out of `(coeffs >> 16) / (coeffs >> 8) / coeffs`.
        let sub_pixels: Vec<u32> = coeffs
            .iter()
            .map(|c| {
                let g2r = (c.g2r as u8) as u32;
                let g2b = (c.g2b as u8) as u32;
                let r2b = (c.r2b as u8) as u32;
                0xff00_0000 | (r2b << 16) | (g2b << 8) | g2r
            })
            .collect();
        encode_image_stream(&mut bw, &sub_pixels, sub_w, sub_h, false, 0)?;

        // Apply the forward colour transform to the working pixels.
        working = apply_color_transform_forward(&working, width, height, tile_bits, &coeffs, sub_w);
    }

    if !palette_active && opts.use_predictor {
        // Forward predictor: pick a mode per tile, then subtract
        // predictions. The sub-image we ship carries one mode per
        // tile — stored in the green channel's low 4 bits per spec.
        let tile_bits = PREDICTOR_TILE_BITS;
        let tile_side = 1u32 << tile_bits;
        let sub_w = (width + tile_side - 1) / tile_side;
        let sub_h = (height + tile_side - 1) / tile_side;
        let modes = choose_predictor_modes(&working, width, height, tile_bits, sub_w, sub_h);

        // Transform header: present + type 0 (Predictor) + tile_bits-2.
        bw.write(1, 1);
        bw.write(0, 2);
        bw.write(tile_bits - 2, 3);

        // Emit the mode sub-image as an ARGB image stream (alpha 0xff,
        // red/blue 0, green = mode). No cache, no meta-huffman — the
        // sub-image is tiny and the decoder reads it with main_image=false.
        let sub_pixels: Vec<u32> = modes
            .iter()
            .map(|&m| 0xff00_0000 | ((m & 0xff) << 8))
            .collect();
        encode_image_stream(&mut bw, &sub_pixels, sub_w, sub_h, false, 0)?;

        // Residuals (main image payload): pixel - predicted-from-decoded-
        // neighbours. The decode side recomputes the same prediction
        // from its own already-decoded neighbourhood and re-adds the
        // residual modulo 256 per channel.
        working = apply_predictor_forward(&working, width, height, tile_bits, &modes, sub_w);
    }

    // No more transforms.
    bw.write(0, 1);

    // ── Main image stream ────────────────────────────────────────────
    let cache_bits = if opts.use_color_cache {
        if (1..=11).contains(&opts.cache_bits) {
            opts.cache_bits
        } else {
            COLOR_CACHE_BITS
        }
    } else {
        0
    };
    let stream_h = height;
    encode_image_stream(&mut bw, &working, current_width, stream_h, true, cache_bits)?;

    Ok(bw.finish())
}

/// Encode `pixels` via the smallest of a coarse parameter grid: each
/// trial flips the four optional VP8L transforms on/off and tries a
/// short list of colour-cache widths. Returns the smallest encoded
/// bitstream — bit-identical lossless behaviour, just tuned for size.
///
/// Search space (per spec §3 transforms + §5 colour cache):
///
/// | knob               | values                |
/// |--------------------|----------------------|
/// | subtract-green     | off, on               |
/// | colour-transform   | off, on               |
/// | predictor          | off, on               |
/// | colour-cache       | off, 6 bits, 8 bits, 10 bits |
/// | colour-indexing    | off, on (auto-skipped if > 256 unique colours) |
///
/// 2 × 2 × 2 × 4 × 2 = 64 trials. Each trial runs the existing
/// per-transform pipeline so the cost is one full-image encode per
/// candidate. Trials are independent and small, so we encode all of
/// them and then pick.
///
/// Note that with `use_color_index = true` the encoder internally
/// suppresses the subtract-green / colour-transform / predictor steps
/// (they don't apply to a packed-index pixel stream); the
/// `use_subtract_green` / `use_color_transform` / `use_predictor`
/// flags become no-ops for those trials, giving the same encode
/// regardless of how they're set. The redundant work is cheap (palette
/// builder is O(N log K) for K ≤ 256) but the early winner is usually
/// the cache-only variant on palettised content.
///
/// Used as the production path under [`encode_vp8l_argb`] — callers that
/// want a single fixed configuration should still use
/// [`encode_vp8l_argb_with`].
fn encode_vp8l_argb_rdo(
    width: u32,
    height: u32,
    pixels: &[u32],
    has_alpha: bool,
) -> Result<Vec<u8>> {
    // Cache widths to probe. 0 means "no cache". 6/8/10 cover the
    // useful spread: 6 for very small palettes, 8 (the previous fixed
    // default) for typical photos, 10 for highly-repeated colour
    // images. Wider than 10 rarely pays for the extra header overhead.
    const CACHE_BITS_GRID: [u32; 4] = [0, 6, 8, 10];

    // Apply the alpha-zero RGB strip once up front (default-on, matches
    // libwebp). Each per-trial encode then runs with
    // `strip_transparent_color: false` so we don't pay for the pass
    // 32 times — the result is identical to letting the per-trial
    // encoder do its own strip.
    let mut stripped: Vec<u32> = pixels.to_vec();
    strip_transparent_rgb(&mut stripped);

    let mut best: Option<Vec<u8>> = None;

    for &use_palette in &[true, false] {
        for &use_sg in &[true, false] {
            for &use_ct in &[true, false] {
                for &use_pr in &[true, false] {
                    for &cb in CACHE_BITS_GRID.iter() {
                        let opts = EncoderOptions {
                            use_subtract_green: use_sg,
                            use_color_transform: use_ct,
                            use_predictor: use_pr,
                            use_color_cache: cb > 0,
                            cache_bits: if cb > 0 { cb } else { COLOR_CACHE_BITS },
                            // Strip already applied above on `stripped`.
                            strip_transparent_color: false,
                            use_color_index: use_palette,
                            // RDO is the lossless production path: don't
                            // sneak a quantising preprocess in here.
                            // Callers that want near-lossless go through
                            // [`encode_vp8l_argb_with`] explicitly.
                            near_lossless: 100,
                        };
                        let bytes =
                            encode_vp8l_argb_with(width, height, &stripped, has_alpha, opts)?;
                        if best.as_ref().map(|b| bytes.len() < b.len()).unwrap_or(true) {
                            best = Some(bytes);
                        }
                    }
                }
            }
        }
    }
    // `best` is always Some — we ran at least one trial above.
    Ok(best.expect("RDO produced at least one candidate"))
}

/// Encode a `width × height` VP8L image stream (post-transform residuals)
/// into `bw`. Used both for the main picture and for the tiny predictor/
/// colour sub-images. Sub-images always pass `main_image = false` + zero
/// cache bits; the decoder-side parse at [`super::decode_image_stream`]
/// matches that calling convention.
///
/// On the main image (and only there) this entry point now also tries
/// the **meta-Huffman per-tile grouping** path. Tiles are clustered
/// into 2 Huffman groups and the shorter-bitstream variant (single-
/// group baseline vs 2-group meta-Huffman) wins. The single-group
/// path is always tried first, so the meta-Huffman attempt is strictly
/// non-regressing — it just gives the encoder a second shot.
fn encode_image_stream(
    bw: &mut BitWriter,
    pixels: &[u32],
    width: u32,
    height: u32,
    main_image: bool,
    cache_bits: u32,
) -> Result<()> {
    let cache_size = if cache_bits == 0 {
        0u32
    } else {
        1u32 << cache_bits
    };

    // Single-pass build of the symbol stream — shared by every encode
    // attempt below (single-group baseline and the 2-group meta-Huffman
    // variant).
    let stream = build_symbol_stream(pixels, width, height, cache_bits);

    // Speculatively emit the single-group baseline into the live writer,
    // measure the bit length, and remember the rollback point. If the
    // meta-Huffman variant turns out smaller we'll rewind and re-emit;
    // otherwise we keep what's already there.
    let baseline_mark = bw.mark();
    encode_image_stream_single_group(bw, &stream, cache_bits, cache_size, main_image)?;
    let baseline_bits = bw.bit_pos() - baseline_mark.bit_pos();

    // For sub-images (e.g. predictor/colour mode maps) the spec only
    // allows a single group — meta-Huffman lives on the main image
    // alone. Skip the trial entirely on sub-images.
    if !main_image {
        return Ok(());
    }

    // Speculatively emit the meta-Huffman variant after rewinding to
    // the same starting position. Compare bit lengths and keep the
    // shorter one in the live writer.
    //
    // Performance note: the rewind-then-meta-trial-then-maybe-rewind-
    // again dance can in the worst case cost two full single-group
    // encodes plus one meta-Huffman encode for the main image. The
    // RDO sweep runs this 32+ times per `encode_vp8l_argb_rdo`, so we
    // gate on the early-bail flags inside [`try_encode_meta_huffman`]
    // (image too small / single-cluster degeneration) before paying
    // the full meta-Huffman cost.
    bw.restore(baseline_mark);
    let meta_emitted = try_encode_meta_huffman(bw, &stream, width, height, cache_bits, cache_size)?;
    if meta_emitted {
        let meta_bits = bw.bit_pos() - baseline_mark.bit_pos();
        if meta_bits < baseline_bits {
            // Meta-Huffman wins — leave its bytes in place.
            return Ok(());
        }
        // Meta-Huffman lost. Rewind whatever the trial wrote and
        // re-emit the baseline.
        bw.restore(baseline_mark);
        encode_image_stream_single_group(bw, &stream, cache_bits, cache_size, main_image)?;
        debug_assert_eq!(bw.bit_pos() - baseline_mark.bit_pos(), baseline_bits);
    } else {
        // The trial returned without writing — the writer is still at
        // `baseline_mark`. Re-emit the baseline (no rewind needed
        // because the trial left no trace).
        encode_image_stream_single_group(bw, &stream, cache_bits, cache_size, main_image)?;
        debug_assert_eq!(bw.bit_pos() - baseline_mark.bit_pos(), baseline_bits);
    }
    Ok(())
}

/// Single-Huffman-group baseline: one set of {green, red, blue, alpha,
/// distance} trees covering the entire image. Mirrors the original
/// encoder shape; factored out so the meta-Huffman trial path can call
/// the same code with a per-group symbol partition.
fn encode_image_stream_single_group(
    bw: &mut BitWriter,
    stream: &[StreamSym],
    cache_bits: u32,
    cache_size: u32,
    main_image: bool,
) -> Result<()> {
    if cache_bits > 0 {
        bw.write(1, 1);
        bw.write(cache_bits, 4);
    } else {
        bw.write(0, 1);
    }

    if main_image {
        // Single Huffman group → meta-Huffman absent.
        bw.write(0, 1);
    }

    let green_alpha = 256 + 24 + cache_size as usize;
    let mut green_freq = vec![0u32; green_alpha];
    let mut red_freq = vec![0u32; 256];
    let mut blue_freq = vec![0u32; 256];
    let mut alpha_freq = vec![0u32; 256];
    let mut dist_freq = vec![0u32; 40];

    for sym in stream {
        accumulate_symbol_freq(
            sym,
            &mut green_freq,
            &mut red_freq,
            &mut blue_freq,
            &mut alpha_freq,
            &mut dist_freq,
        );
    }

    let green_lens = build_limited_lengths(&green_freq, MAX_CODE_LENGTH)?;
    let red_lens = build_limited_lengths(&red_freq, MAX_CODE_LENGTH)?;
    let blue_lens = build_limited_lengths(&blue_freq, MAX_CODE_LENGTH)?;
    let alpha_lens = build_limited_lengths(&alpha_freq, MAX_CODE_LENGTH)?;
    let dist_lens = build_limited_lengths(&dist_freq, MAX_CODE_LENGTH)?;

    let green_codes = canonical_codes(&green_lens);
    let red_codes = canonical_codes(&red_lens);
    let blue_codes = canonical_codes(&blue_lens);
    let alpha_codes = canonical_codes(&alpha_lens);
    let dist_codes = canonical_codes(&dist_lens);

    emit_huffman_tree(bw, &green_lens)?;
    emit_huffman_tree(bw, &red_lens)?;
    emit_huffman_tree(bw, &blue_lens)?;
    emit_huffman_tree(bw, &alpha_lens)?;
    emit_huffman_tree(bw, &dist_lens)?;

    for sym in stream {
        emit_symbol(
            bw,
            sym,
            &green_codes,
            &green_lens,
            &red_codes,
            &red_lens,
            &blue_codes,
            &blue_lens,
            &alpha_codes,
            &alpha_lens,
            &dist_codes,
            &dist_lens,
        );
    }
    Ok(())
}

/// Push the frequencies of one symbol onto the per-alphabet histograms.
/// Hoisted out so the single-group and meta-Huffman paths share the same
/// accounting code.
#[inline]
fn accumulate_symbol_freq(
    sym: &StreamSym,
    green: &mut [u32],
    red: &mut [u32],
    blue: &mut [u32],
    alpha: &mut [u32],
    dist: &mut [u32],
) {
    match *sym {
        StreamSym::Literal { a, r, g, b } => {
            green[g as usize] += 1;
            red[r as usize] += 1;
            blue[b as usize] += 1;
            alpha[a as usize] += 1;
        }
        StreamSym::Backref {
            len_sym, dist_sym, ..
        } => {
            green[256 + len_sym as usize] += 1;
            dist[dist_sym as usize] += 1;
        }
        StreamSym::CacheRef { index } => {
            green[256 + 24 + index as usize] += 1;
        }
    }
}

/// Emit one symbol using the supplied per-alphabet codes/lengths. As
/// with [`accumulate_symbol_freq`], shared between the two encoder
/// paths so the wire-format details live in exactly one place.
#[inline]
#[allow(clippy::too_many_arguments)]
fn emit_symbol(
    bw: &mut BitWriter,
    sym: &StreamSym,
    green_codes: &[u32],
    green_lens: &[u8],
    red_codes: &[u32],
    red_lens: &[u8],
    blue_codes: &[u32],
    blue_lens: &[u8],
    alpha_codes: &[u32],
    alpha_lens: &[u8],
    dist_codes: &[u32],
    dist_lens: &[u8],
) {
    match *sym {
        StreamSym::Literal { a, r, g, b } => {
            write_code(bw, green_codes, green_lens, g as usize);
            write_code(bw, red_codes, red_lens, r as usize);
            write_code(bw, blue_codes, blue_lens, b as usize);
            write_code(bw, alpha_codes, alpha_lens, a as usize);
        }
        StreamSym::Backref {
            len_sym,
            len_extra_bits,
            len_extra,
            dist_sym,
            dist_extra_bits,
            dist_extra,
        } => {
            write_code(bw, green_codes, green_lens, 256 + len_sym as usize);
            if len_extra_bits > 0 {
                bw.write(len_extra, len_extra_bits);
            }
            write_code(bw, dist_codes, dist_lens, dist_sym as usize);
            if dist_extra_bits > 0 {
                bw.write(dist_extra, dist_extra_bits);
            }
        }
        StreamSym::CacheRef { index } => {
            write_code(bw, green_codes, green_lens, 256 + 24 + index as usize);
        }
    }
}

/// Number of pixels consumed by one symbol when the decoder applies it.
/// Literal/CacheRef advance one pixel; a backref advances `length` (which
/// we recover from the prefix-symbol + extra-bits factoring inverse).
#[inline]
fn symbol_pixel_span(sym: &StreamSym) -> usize {
    match *sym {
        StreamSym::Literal { .. } | StreamSym::CacheRef { .. } => 1,
        StreamSym::Backref {
            len_sym,
            len_extra_bits: _,
            len_extra,
            ..
        } => decode_len_or_dist_value(len_sym, len_extra) as usize,
    }
}

/// Inverse of [`encode_len_or_dist_value`]. Mirrors the decoder's
/// `decode_length_or_distance` — kept here so the encoder can compute
/// per-symbol pixel spans without re-deriving the formula in the
/// meta-Huffman tile-assignment loop.
#[inline]
fn decode_len_or_dist_value(symbol: u32, extra: u32) -> u32 {
    if symbol < 4 {
        symbol + 1
    } else {
        let extra_bits = (symbol - 2) >> 1;
        let offset = (2 + (symbol & 1)) << extra_bits;
        offset + extra + 1
    }
}

/// Try the 2-group meta-Huffman path. Writes the candidate bitstream
/// into `bw` and returns `Ok(true)` when something was emitted, or
/// `Ok(false)` when the trial decided meta-Huffman wasn't worth
/// attempting (e.g. the picture is too small to make per-tile clustering
/// pay for the extra Huffman trees + meta-image overhead).
///
/// The decision to keep or discard the result is left to the caller —
/// pair this with [`BitWriter::mark`] / [`BitWriter::restore`] so the
/// candidate can be rolled back when a previously-tried variant was
/// shorter.
fn try_encode_meta_huffman(
    bw: &mut BitWriter,
    stream: &[StreamSym],
    width: u32,
    height: u32,
    cache_bits: u32,
    cache_size: u32,
) -> Result<bool> {
    // Meta-Huffman tile size. The spec allows `meta_bits ∈ 2..=9`
    // (4..=512 px per side). 4 (16-pixel tiles) is libwebp's default
    // and gives a reasonable signal-to-overhead ratio.
    const META_BITS: u32 = 4;
    let tile_side = 1u32 << META_BITS;
    let meta_w = (width + tile_side - 1) / tile_side;
    let meta_h = (height + tile_side - 1) / tile_side;
    let num_tiles = (meta_w * meta_h) as usize;

    // No grouping makes sense with a single tile.
    if num_tiles < 2 {
        return Ok(false);
    }

    // Bail out early on tiny images — the two extra Huffman-tree sets
    // plus the meta-image stream overhead easily outweigh any per-group
    // savings on an image with few hundred symbols. 1024 pixels (32×32)
    // is a conservative floor; smaller images stick with the single-
    // group baseline.
    if (width as u64) * (height as u64) < 1024 {
        return Ok(false);
    }

    // ── Step 1: compute per-tile histograms over each of the five
    // alphabets. Each symbol's tile is determined by its starting pixel
    // position in raster order; backrefs are bound to the tile they
    // *start* in (matching the decoder, which looks up the meta-image
    // once at the start of every prefix-coded symbol).
    let green_alpha = 256 + 24 + cache_size as usize;
    let mut tile_green: Vec<Vec<u32>> = (0..num_tiles).map(|_| vec![0u32; green_alpha]).collect();
    let mut tile_red: Vec<Vec<u32>> = (0..num_tiles).map(|_| vec![0u32; 256]).collect();
    let mut tile_blue: Vec<Vec<u32>> = (0..num_tiles).map(|_| vec![0u32; 256]).collect();
    let mut tile_alpha: Vec<Vec<u32>> = (0..num_tiles).map(|_| vec![0u32; 256]).collect();
    let mut tile_dist: Vec<Vec<u32>> = (0..num_tiles).map(|_| vec![0u32; 40]).collect();
    // Per-symbol index → tile index. We need this twice: once to seed
    // the clustering, once to emit the per-symbol writes after groups
    // have been assigned. Store it here so we don't re-walk positions.
    let mut sym_tile: Vec<usize> = Vec::with_capacity(stream.len());

    let mut x: u32 = 0;
    let mut y: u32 = 0;
    for sym in stream {
        let mx = x >> META_BITS;
        let my = y >> META_BITS;
        let tile_idx = (my * meta_w + mx) as usize;
        sym_tile.push(tile_idx);
        accumulate_symbol_freq(
            sym,
            &mut tile_green[tile_idx],
            &mut tile_red[tile_idx],
            &mut tile_blue[tile_idx],
            &mut tile_alpha[tile_idx],
            &mut tile_dist[tile_idx],
        );
        let span = symbol_pixel_span(sym) as u32;
        let new_x = x + span;
        y += new_x / width;
        x = new_x % width;
    }

    // ── Step 2: cluster tiles into 2 groups by histogram similarity.
    // The "fingerprint" we cluster on is the green-alphabet histogram —
    // it carries the most discriminative info (mix of literal-greens,
    // length codes, cache hits varies wildly between e.g. flat and
    // textured tiles).
    //
    // We use a 2-iteration k-means with K=2: pick two seed tiles
    // (first non-empty + the one farthest from it), assign every tile
    // to its nearest seed, recompute centroids, reassign. Two
    // iterations is enough for the histograms to stabilise on natural
    // images and keeps the worst-case cost bounded at
    // O(num_tiles × green_alpha × K × iters).
    let assignments = cluster_tiles_kmeans2(&tile_green, num_tiles);

    // If every tile landed in one group there's no point continuing —
    // the meta-Huffman variant degenerates back to the single-group
    // baseline plus pure overhead. Skip.
    let used_groups: std::collections::BTreeSet<u32> = assignments.iter().copied().collect();
    if used_groups.len() < 2 {
        return Ok(false);
    }

    // ── Step 3: build per-group histograms by summing over the tiles
    // assigned to each group.
    let mut g0_green = vec![0u32; green_alpha];
    let mut g0_red = vec![0u32; 256];
    let mut g0_blue = vec![0u32; 256];
    let mut g0_alpha = vec![0u32; 256];
    let mut g0_dist = vec![0u32; 40];
    let mut g1_green = vec![0u32; green_alpha];
    let mut g1_red = vec![0u32; 256];
    let mut g1_blue = vec![0u32; 256];
    let mut g1_alpha = vec![0u32; 256];
    let mut g1_dist = vec![0u32; 40];
    for t in 0..num_tiles {
        let (g, r, b, a, d) = if assignments[t] == 0 {
            (
                &mut g0_green,
                &mut g0_red,
                &mut g0_blue,
                &mut g0_alpha,
                &mut g0_dist,
            )
        } else {
            (
                &mut g1_green,
                &mut g1_red,
                &mut g1_blue,
                &mut g1_alpha,
                &mut g1_dist,
            )
        };
        for (i, v) in tile_green[t].iter().enumerate() {
            g[i] += v;
        }
        for (i, v) in tile_red[t].iter().enumerate() {
            r[i] += v;
        }
        for (i, v) in tile_blue[t].iter().enumerate() {
            b[i] += v;
        }
        for (i, v) in tile_alpha[t].iter().enumerate() {
            a[i] += v;
        }
        for (i, v) in tile_dist[t].iter().enumerate() {
            d[i] += v;
        }
    }

    // ── Step 4: emit. Header → cache flag → meta-Huffman flag → meta
    // sub-image → per-group Huffman trees → interleaved symbols.

    if cache_bits > 0 {
        bw.write(1, 1);
        bw.write(cache_bits, 4);
    } else {
        bw.write(0, 1);
    }

    // Meta-Huffman present.
    bw.write(1, 1);
    bw.write(META_BITS - 2, 3);

    // Build the meta-image: one ARGB pixel per tile. The decoder reads
    // the group index from `(p >> 8) & 0xffff` (green byte = low 8 bits
    // of group_idx; we keep K ≤ 256 so no need for the red byte). Alpha
    // 0xff so the decode path treats it as a normal opaque pixel.
    let meta_pixels: Vec<u32> = assignments
        .iter()
        .map(|&g| 0xff00_0000 | ((g & 0xff) << 8))
        .collect();
    encode_image_stream(bw, &meta_pixels, meta_w, meta_h, false, 0)?;

    // Build per-group Huffman trees.
    let g0_green_lens = build_limited_lengths(&g0_green, MAX_CODE_LENGTH)?;
    let g0_red_lens = build_limited_lengths(&g0_red, MAX_CODE_LENGTH)?;
    let g0_blue_lens = build_limited_lengths(&g0_blue, MAX_CODE_LENGTH)?;
    let g0_alpha_lens = build_limited_lengths(&g0_alpha, MAX_CODE_LENGTH)?;
    let g0_dist_lens = build_limited_lengths(&g0_dist, MAX_CODE_LENGTH)?;

    let g1_green_lens = build_limited_lengths(&g1_green, MAX_CODE_LENGTH)?;
    let g1_red_lens = build_limited_lengths(&g1_red, MAX_CODE_LENGTH)?;
    let g1_blue_lens = build_limited_lengths(&g1_blue, MAX_CODE_LENGTH)?;
    let g1_alpha_lens = build_limited_lengths(&g1_alpha, MAX_CODE_LENGTH)?;
    let g1_dist_lens = build_limited_lengths(&g1_dist, MAX_CODE_LENGTH)?;

    let g0_green_codes = canonical_codes(&g0_green_lens);
    let g0_red_codes = canonical_codes(&g0_red_lens);
    let g0_blue_codes = canonical_codes(&g0_blue_lens);
    let g0_alpha_codes = canonical_codes(&g0_alpha_lens);
    let g0_dist_codes = canonical_codes(&g0_dist_lens);

    let g1_green_codes = canonical_codes(&g1_green_lens);
    let g1_red_codes = canonical_codes(&g1_red_lens);
    let g1_blue_codes = canonical_codes(&g1_blue_lens);
    let g1_alpha_codes = canonical_codes(&g1_alpha_lens);
    let g1_dist_codes = canonical_codes(&g1_dist_lens);

    // Emit groups in numeric order (group 0 then group 1), each as a
    // run of 5 trees (green/red/blue/alpha/distance). Matches what
    // [`super::HuffmanGroup::read`] expects.
    emit_huffman_tree(bw, &g0_green_lens)?;
    emit_huffman_tree(bw, &g0_red_lens)?;
    emit_huffman_tree(bw, &g0_blue_lens)?;
    emit_huffman_tree(bw, &g0_alpha_lens)?;
    emit_huffman_tree(bw, &g0_dist_lens)?;

    emit_huffman_tree(bw, &g1_green_lens)?;
    emit_huffman_tree(bw, &g1_red_lens)?;
    emit_huffman_tree(bw, &g1_blue_lens)?;
    emit_huffman_tree(bw, &g1_alpha_lens)?;
    emit_huffman_tree(bw, &g1_dist_lens)?;

    // Per-symbol writes: pick the group from the tile assignment.
    for (idx, sym) in stream.iter().enumerate() {
        let group = assignments[sym_tile[idx]];
        if group == 0 {
            emit_symbol(
                bw,
                sym,
                &g0_green_codes,
                &g0_green_lens,
                &g0_red_codes,
                &g0_red_lens,
                &g0_blue_codes,
                &g0_blue_lens,
                &g0_alpha_codes,
                &g0_alpha_lens,
                &g0_dist_codes,
                &g0_dist_lens,
            );
        } else {
            emit_symbol(
                bw,
                sym,
                &g1_green_codes,
                &g1_green_lens,
                &g1_red_codes,
                &g1_red_lens,
                &g1_blue_codes,
                &g1_blue_lens,
                &g1_alpha_codes,
                &g1_alpha_lens,
                &g1_dist_codes,
                &g1_dist_lens,
            );
        }
    }

    Ok(true)
}

/// Cluster `num_tiles` per-tile green-alphabet histograms into 2 groups
/// using a tiny 2-iteration k-means. Returns one group id (0 or 1) per
/// tile in row-major order.
///
/// The seeding picks the busiest tile (highest total count — usually the
/// one with the most diverse content) as group 0's centroid and the tile
/// most distant from it (L1 over the histogram) as group 1's centroid.
/// Two reassignment passes are enough for the histograms to settle on
/// most natural images; further iterations rarely change cluster
/// membership and would only burn CPU.
fn cluster_tiles_kmeans2(tile_green: &[Vec<u32>], num_tiles: usize) -> Vec<u32> {
    let alpha = tile_green.first().map(|h| h.len()).unwrap_or(0);
    if num_tiles < 2 || alpha == 0 {
        return vec![0u32; num_tiles];
    }

    // Pick seed 0: the tile with the largest total count.
    let mut seed0 = 0usize;
    let mut max_total = 0u64;
    for (i, h) in tile_green.iter().enumerate() {
        let total: u64 = h.iter().map(|&v| v as u64).sum();
        if total > max_total {
            max_total = total;
            seed0 = i;
        }
    }
    // Pick seed 1: the tile farthest from seed 0 by L1 distance over
    // the green histogram.
    let mut seed1 = if seed0 == 0 { 1 } else { 0 };
    let mut max_dist: u64 = 0;
    for (i, h) in tile_green.iter().enumerate() {
        if i == seed0 {
            continue;
        }
        let d = l1_dist_u32(&tile_green[seed0], h);
        if d > max_dist {
            max_dist = d;
            seed1 = i;
        }
    }

    // Centroids carry the **mean** histogram for each cluster — divided
    // by the cluster size so that L1 distance against a single tile's
    // histogram is meaningful (otherwise the sum of e.g. 50 tile
    // histograms would dwarf the per-tile counts and the comparison
    // would degenerate to "which cluster has more tiles").
    let mut centroid0 = tile_green[seed0].clone();
    let mut centroid1 = tile_green[seed1].clone();
    let mut assignments = vec![0u32; num_tiles];

    for _iter in 0..2 {
        // Reassign every tile to its nearest centroid.
        for (t, h) in tile_green.iter().enumerate() {
            let d0 = l1_dist_u32(&centroid0, h);
            let d1 = l1_dist_u32(&centroid1, h);
            assignments[t] = if d1 < d0 { 1 } else { 0 };
        }
        // Recompute centroids as per-bucket means over the assigned
        // tiles. Empty-cluster pathology: leave a centroid at its
        // previous value if nothing was assigned to it (keeps the
        // seeding stable rather than collapsing to the all-zero
        // histogram).
        let mut sum0 = vec![0u64; alpha];
        let mut sum1 = vec![0u64; alpha];
        let mut n0 = 0u64;
        let mut n1 = 0u64;
        for (t, h) in tile_green.iter().enumerate() {
            if assignments[t] == 0 {
                for (i, &v) in h.iter().enumerate() {
                    sum0[i] += v as u64;
                }
                n0 += 1;
            } else {
                for (i, &v) in h.iter().enumerate() {
                    sum1[i] += v as u64;
                }
                n1 += 1;
            }
        }
        if n0 > 0 {
            for (i, s) in sum0.iter().enumerate() {
                centroid0[i] = (s / n0) as u32;
            }
        }
        if n1 > 0 {
            for (i, s) in sum1.iter().enumerate() {
                centroid1[i] = (s / n1) as u32;
            }
        }
    }

    assignments
}

/// L1 distance between two equal-length u32 histograms. `abs_diff`
/// avoids signed-arithmetic underflow; the per-bucket diff sums into a
/// u64 — never overflows for typical tile sizes (16×16 tiles cap each
/// bucket at ~256, summing across <512 buckets stays well under u32
/// anyway, but u64 keeps the helper general).
#[inline]
fn l1_dist_u32(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0u64;
    for (av, bv) in a.iter().zip(b.iter()) {
        s += (*av).abs_diff(*bv) as u64;
    }
    s
}

/// Parsed-pixel symbol. Either a literal ARGB quadruplet, an LZ77
/// backreference (length + distance with their extra-bit fields already
/// factored out), or a colour-cache reference by index.
#[derive(Clone, Copy)]
enum StreamSym {
    Literal {
        a: u8,
        r: u8,
        g: u8,
        b: u8,
    },
    Backref {
        len_sym: u32,
        len_extra_bits: u32,
        len_extra: u32,
        dist_sym: u32,
        dist_extra_bits: u32,
        dist_extra: u32,
    },
    CacheRef {
        index: u32,
    },
}

/// Factor a VP8L length/distance *value* (≥1) into a prefix symbol +
/// trailing extra bits. Inverse of `decode_length_or_distance` in
/// `super`.
fn encode_len_or_dist_value(value: u32) -> (u32, u32, u32) {
    debug_assert!(value >= 1);
    if value <= 4 {
        return (value - 1, 0, 0);
    }
    let v = value - 1; // ≥ 4
    let msb = 31 - v.leading_zeros(); // ≥ 2
    let extra_bits = msb - 1;
    let sym_sub = (v >> extra_bits) & 1; // 0 (even) or 1 (odd)
    let symbol = 2 * extra_bits + 2 + sym_sub;
    let offset = (2 + sym_sub) << extra_bits;
    let extra = v - offset;
    (symbol, extra_bits, extra)
}

/// Walk `pixels` and emit literals + LZ77 backreferences + colour-cache
/// refs. Uses a simple prefix-hash chain with head + next-pointer arrays;
/// the chain is bounded by [`LZ_WINDOW`].
fn build_symbol_stream(
    pixels: &[u32],
    _width: u32,
    _height: u32,
    cache_bits: u32,
) -> Vec<StreamSym> {
    let mut out: Vec<StreamSym> = Vec::with_capacity(pixels.len());
    let n = pixels.len();
    // Hash table: 12-bit table, heads index into `pixels`. `next` is a
    // per-pixel chain pointer (usize::MAX = terminator).
    const HASH_BITS: u32 = 12;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    let mut head: Vec<usize> = vec![usize::MAX; HASH_SIZE];
    let mut next: Vec<usize> = vec![usize::MAX; n];

    let hash3 = |p0: u32, p1: u32, p2: u32| -> usize {
        // Cheap multiplicative hash over 3 pixels (12 bytes worth of data).
        let k = p0
            .wrapping_mul(0x9E3779B9)
            .wrapping_add(p1.wrapping_mul(0x85EBCA77))
            .wrapping_add(p2.wrapping_mul(0xC2B2AE3D));
        (k >> (32 - HASH_BITS)) as usize
    };

    // Colour-cache mirror. The decoder updates the cache on every
    // emitted/decoded pixel, so we maintain the same state during
    // parsing and can emit cache-index codes when a literal's hash
    // slot already holds that exact colour.
    let cache_size = if cache_bits == 0 {
        0usize
    } else {
        1usize << cache_bits
    };
    let mut cache: Vec<u32> = vec![0u32; cache_size];

    let mut i = 0usize;
    while i < n {
        // Find best match starting at i, if at least MIN_MATCH pixels
        // remain.
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + MIN_MATCH <= n {
            let h = hash3(pixels[i], pixels[i + 1], pixels[i + 2]);
            let mut candidate = head[h];
            let mut tries = 64usize;
            while candidate != usize::MAX && tries > 0 {
                let dist = i - candidate;
                if dist == 0 || dist > LZ_WINDOW {
                    break;
                }
                let max_len = (n - i).min(MAX_MATCH);
                let mut l = 0usize;
                while l < max_len && pixels[candidate + l] == pixels[i + l] {
                    l += 1;
                }
                if l >= MIN_MATCH && l > best_len {
                    best_len = l;
                    best_dist = dist;
                    if l >= 64 {
                        break;
                    }
                }
                candidate = next[candidate];
                tries -= 1;
            }
        }

        if best_len >= MIN_MATCH {
            let (len_sym, len_eb, len_ex) = encode_len_or_dist_value(best_len as u32);
            let (dist_sym, dist_eb, dist_ex) = encode_len_or_dist_value((best_dist as u32) + 120);
            out.push(StreamSym::Backref {
                len_sym,
                len_extra_bits: len_eb,
                len_extra: len_ex,
                dist_sym,
                dist_extra_bits: dist_eb,
                dist_extra: dist_ex,
            });
            for k in 0..best_len {
                let pos = i + k;
                if pos + 2 < n {
                    let h = hash3(pixels[pos], pixels[pos + 1], pixels[pos + 2]);
                    next[pos] = head[h];
                    head[h] = pos;
                }
                if cache_size > 0 {
                    cache_add(&mut cache, cache_bits, pixels[pos]);
                }
            }
            i += best_len;
        } else {
            let p = pixels[i];
            // Try a cache hit first. The decoder's hash is deterministic
            // (`0x1e35a7bd * argb >> (32 - cache_bits)`), so we can look
            // up the current slot and emit a cache-index code if it
            // already holds this exact colour. A hit saves the R/B/A
            // literal codes entirely (only the green symbol is written).
            let mut emitted_cache = false;
            if cache_size > 0 {
                let idx = (0x1e35_a7bd_u32.wrapping_mul(p) >> (32 - cache_bits)) as usize;
                if idx < cache.len() && cache[idx] == p {
                    out.push(StreamSym::CacheRef { index: idx as u32 });
                    emitted_cache = true;
                }
            }
            if !emitted_cache {
                out.push(StreamSym::Literal {
                    a: ((p >> 24) & 0xff) as u8,
                    r: ((p >> 16) & 0xff) as u8,
                    g: ((p >> 8) & 0xff) as u8,
                    b: (p & 0xff) as u8,
                });
            }
            if cache_size > 0 {
                cache_add(&mut cache, cache_bits, p);
            }
            if i + 2 < n {
                let h = hash3(pixels[i], pixels[i + 1], pixels[i + 2]);
                next[i] = head[h];
                head[h] = i;
            }
            i += 1;
        }
    }
    out
}

fn cache_add(cache: &mut [u32], cache_bits: u32, argb: u32) {
    if cache.is_empty() {
        return;
    }
    let idx = (0x1e35_a7bd_u32.wrapping_mul(argb) >> (32 - cache_bits)) as usize;
    if idx < cache.len() {
        cache[idx] = argb;
    }
}

// ── Transforms (encoder side) ─────────────────────────────────────────

// ── Colour-indexing (palette) transform ───────────────────────────────
//
// Spec §3.6.5: when the image has ≤ 256 unique ARGB colours we can ship
// a 1D palette out of band and replace every pixel with a small integer
// index into it. The index image carries one index per pixel in the
// green channel; for palettes ≤ 16 entries multiple indices are
// bit-packed into one green byte (`pack = 8 / bits_per_pixel`), which
// also shrinks the image dimensions the entropy coder sees.

/// Maximum palette size that VP8L can encode. The transform header
/// stores `num_colors - 1` in 8 bits.
const MAX_PALETTE_SIZE: usize = 256;

/// Walk `pixels` and build a sorted, de-duplicated palette of unique
/// ARGB values. Returns `None` if the image has > 256 unique colours
/// (palette transform doesn't apply) or if it's degenerate (1 colour
/// — a single-entry palette is technically legal but the bit-packing
/// math becomes awkward; the regular path with one literal + a giant
/// LZ77 run already handles this case cheaply).
///
/// Sorting by ARGB value gives the delta-coded palette entries small
/// component differences on average — important because the palette
/// itself ships through the same prefix-coded image stream as the rest
/// of the bitstream and benefits from a low-entropy delta sequence.
fn build_palette(pixels: &[u32]) -> Option<Vec<u32>> {
    // Use a tiny set built from a sorted Vec — std HashSet would be
    // overkill here and pulls in hashing overhead for a search space
    // bounded at 256. We early-exit the moment the unique count
    // exceeds MAX_PALETTE_SIZE.
    let mut palette: Vec<u32> = Vec::with_capacity(MAX_PALETTE_SIZE + 1);
    for &p in pixels {
        match palette.binary_search(&p) {
            Ok(_) => {}
            Err(pos) => {
                if palette.len() >= MAX_PALETTE_SIZE {
                    return None;
                }
                palette.insert(pos, p);
            }
        }
    }
    if palette.len() < 2 {
        // Single-colour image — the regular path handles this with a
        // single literal + colour-cache hits + a long LZ77 backref,
        // and avoids the palette header overhead.
        return None;
    }
    Some(palette)
}

/// Per spec §3.6.5: bits-per-pixel selection for the index image.
/// Smaller palettes get bit-packed for better main-stream compression.
fn bits_per_pixel_for(num_colors: u32) -> u32 {
    if num_colors <= 2 {
        1
    } else if num_colors <= 4 {
        2
    } else if num_colors <= 16 {
        4
    } else {
        8
    }
}

/// Encode the palette as a delta-coded ARGB row. The decoder walks the
/// row left-to-right doing per-channel `add_argb` to recover the
/// originals (see `Transform::read` in `transform.rs`); the forward
/// step is the matching per-channel `sub_argb`.
///
/// Returns `palette.len()`-many u32s, ready to feed into
/// [`encode_image_stream`] as a sub-image of size (num_colors × 1).
fn delta_encode_palette(palette: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(palette.len());
    out.push(palette[0]);
    for i in 1..palette.len() {
        out.push(sub_argb(palette[i], palette[i - 1]));
    }
    out
}

/// Build the packed index image. Each ARGB pixel of the input is
/// looked up in `palette` (binary search — palette is sorted) and the
/// resulting index is shifted into the green byte of the packed pixel.
///
/// For `bits_per_pixel ∈ {1, 2, 4}` multiple indices share a single
/// packed pixel: spec §3.6.5 packs them low-bits-first across an
/// 8-bit green channel. Output width is `(width + pack - 1) / pack`,
/// matching the decoder's `image_width_or_default`.
///
/// Padding pixels in the rightmost packed column (if `width` doesn't
/// divide evenly by `pack`) are zero-filled — the decoder will read
/// them but its bounds check (`ox < orig_xsize`) drops them on
/// expansion, so the value is don't-care.
fn pack_palette_indices(
    pixels: &[u32],
    width: u32,
    height: u32,
    palette: &[u32],
    bits_per_pixel: u32,
    packed_w: u32,
) -> Vec<u32> {
    let w = width as usize;
    let h = height as usize;
    let pack = (8 / bits_per_pixel) as usize;
    let pw = packed_w as usize;
    let mut out: Vec<u32> = vec![0u32; pw * h];
    for y in 0..h {
        for xp in 0..pw {
            let mut g_byte: u32 = 0;
            for sub in 0..pack {
                let x_orig = xp * pack + sub;
                if x_orig >= w {
                    break;
                }
                let pixel = pixels[y * w + x_orig];
                // Palette is sorted, so binary search is O(log n) per
                // pixel — for n ≤ 256 that's at most 8 comparisons.
                let idx = palette
                    .binary_search(&pixel)
                    .expect("palette must contain every input pixel by construction");
                g_byte |= (idx as u32) << (sub as u32 * bits_per_pixel);
            }
            // Index image: A=0xff, R=0, G=indices, B=0. The decoder
            // pulls the indices out of `(p >> 8) & 0xff` per spec.
            out[y * pw + xp] = 0xff00_0000 | (g_byte << 8);
        }
    }
    out
}

/// Subtract the green channel from R and B in-place. Mirrors
/// `apply_subtract_green` in [`super::transform`] (the decoder reverses
/// this by adding G back).
fn apply_subtract_green_forward(pixels: &mut [u32]) {
    for p in pixels.iter_mut() {
        let a = (*p >> 24) & 0xff;
        let r = (*p >> 16) & 0xff;
        let g = (*p >> 8) & 0xff;
        let b = *p & 0xff;
        let nr = r.wrapping_sub(g) & 0xff;
        let nb = b.wrapping_sub(g) & 0xff;
        *p = (a << 24) | (nr << 16) | (g << 8) | nb;
    }
}

/// Compute the per-channel prediction used by the VP8L predictor
/// transform. Mirrors the decoder's `predict_argb` — kept in sync here
/// because we need the exact same prediction on both ends.
fn predict_argb(out: &[u32], w: usize, x: usize, y: usize, mode: u32) -> u32 {
    let l = out[y * w + x - 1];
    let t = out[(y - 1) * w + x];
    let tl = out[(y - 1) * w + x - 1];
    // RFC 9649 §4.1: TR for the rightmost column is the LEFTMOST pixel of
    // the current row (column 0 of row y), not the LEFT neighbour. Kept
    // in sync with the decoder's `predict_argb` — they must produce the
    // exact same prediction or our self-roundtrip would diverge from a
    // libwebp-encoded stream's predictor sub-image. (See issue #8.)
    let tr = if x + 1 < w {
        out[(y - 1) * w + x + 1]
    } else {
        out[y * w]
    };
    match mode {
        0 => 0xff00_0000,
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => avg3(l, tr, t),
        6 => avg2(l, tl),
        7 => avg2(l, t),
        8 => avg2(tl, t),
        9 => avg2(t, tr),
        10 => avg2(avg2(l, tl), avg2(t, tr)),
        11 => select_argb(l, t, tl),
        12 => clamp_add_sub_argb(l, t, tl),
        13 => clamp_add_sub_half_argb(avg2(l, t), tl),
        _ => 0xff00_0000,
    }
}

fn avg2(a: u32, b: u32) -> u32 {
    let mut out = 0u32;
    for c in 0..4 {
        let sh = c * 8;
        let av = (a >> sh) & 0xff;
        let bv = (b >> sh) & 0xff;
        out |= ((av + bv) >> 1) << sh;
    }
    out
}

fn avg3(a: u32, b: u32, c: u32) -> u32 {
    avg2(a, avg2(b, c))
}

fn select_argb(l: u32, t: u32, tl: u32) -> u32 {
    let mut out = 0u32;
    let mut dl = 0i32;
    let mut dt = 0i32;
    for c in 0..4 {
        let sh = c * 8;
        let lv = ((l >> sh) & 0xff) as i32;
        let tv = ((t >> sh) & 0xff) as i32;
        let tlv = ((tl >> sh) & 0xff) as i32;
        dl += (tv - tlv).abs();
        dt += (lv - tlv).abs();
    }
    for c in 0..4 {
        let sh = c * 8;
        let lv = (l >> sh) & 0xff;
        let tv = (t >> sh) & 0xff;
        let v = if dl < dt { lv } else { tv };
        out |= v << sh;
    }
    out
}

fn clamp_add_sub_argb(l: u32, t: u32, tl: u32) -> u32 {
    let mut out = 0u32;
    for c in 0..4 {
        let sh = c * 8;
        let lv = ((l >> sh) & 0xff) as i32;
        let tv = ((t >> sh) & 0xff) as i32;
        let tlv = ((tl >> sh) & 0xff) as i32;
        let v = (lv + tv - tlv).clamp(0, 255) as u32;
        out |= v << sh;
    }
    out
}

fn clamp_add_sub_half_argb(a: u32, b: u32) -> u32 {
    let mut out = 0u32;
    for c in 0..4 {
        let sh = c * 8;
        let av = ((a >> sh) & 0xff) as i32;
        let bv = ((b >> sh) & 0xff) as i32;
        let v = (av + (av - bv) / 2).clamp(0, 255) as u32;
        out |= v << sh;
    }
    out
}

/// Per-channel ARGB subtraction modulo 256 — inverse of `add_argb` in
/// the decoder.
fn sub_argb(a: u32, b: u32) -> u32 {
    let aa = (a >> 24) & 0xff;
    let ar = (a >> 16) & 0xff;
    let ag = (a >> 8) & 0xff;
    let ab = a & 0xff;
    let ba = (b >> 24) & 0xff;
    let br = (b >> 16) & 0xff;
    let bg = (b >> 8) & 0xff;
    let bb = b & 0xff;
    (((aa.wrapping_sub(ba)) & 0xff) << 24)
        | (((ar.wrapping_sub(br)) & 0xff) << 16)
        | (((ag.wrapping_sub(bg)) & 0xff) << 8)
        | ((ab.wrapping_sub(bb)) & 0xff)
}

/// Score a predictor mode on a tile. The "cost" is sum-of-abs per-
/// channel residuals over the tile: a crude but monotonic proxy for
/// entropy that doesn't require building a second Huffman pass. Chooses
/// between modes purely by residual magnitude, which correlates well
/// enough with final code length on natural images.
fn score_predictor_on_tile(
    originals: &[u32],
    decoded: &[u32],
    width: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
    mode: u32,
) -> u64 {
    let mut cost = 0u64;
    for y in tile_y0..tile_y1 {
        for x in tile_x0..tile_x1 {
            let idx = y * width + x;
            let pred = if x == 0 && y == 0 {
                0xff00_0000
            } else if y == 0 {
                decoded[idx - 1]
            } else if x == 0 {
                decoded[idx - width]
            } else {
                predict_argb(decoded, width, x, y, mode)
            };
            let p = originals[idx];
            for c in 0..4 {
                let sh = c * 8;
                let pv = ((p >> sh) & 0xff) as i32;
                let prv = ((pred >> sh) & 0xff) as i32;
                let d = (pv - prv).unsigned_abs() as u64;
                // Fold the wrap-around: residual is mod 256, so the
                // "real" magnitude is min(d, 256-d) — that matches what
                // the Huffman alphabet will see.
                cost += d.min(256 - d);
            }
        }
    }
    cost
}

/// Pick one predictor mode per tile by minimising sum-of-abs residuals
/// over the [`PREDICTOR_MODES`] pool. Returns one entry per tile in
/// row-major order.
fn choose_predictor_modes(
    pixels: &[u32],
    width: u32,
    height: u32,
    tile_bits: u32,
    sub_w: u32,
    sub_h: u32,
) -> Vec<u32> {
    let tile_side = 1usize << tile_bits;
    let w = width as usize;
    let h = height as usize;
    let mut modes = Vec::with_capacity((sub_w * sub_h) as usize);
    // The predictor lookup on the decoder side references already-decoded
    // pixels, not the residuals. Since we're forward-computing, we use
    // the original pixel buffer here — the causal neighbourhood is
    // identical pre- and post-inversion because residual + prediction =
    // original mod 256.
    for ty in 0..sub_h as usize {
        for tx in 0..sub_w as usize {
            let x0 = tx * tile_side;
            let y0 = ty * tile_side;
            let x1 = (x0 + tile_side).min(w);
            let y1 = (y0 + tile_side).min(h);
            let mut best = PREDICTOR_MODES[0];
            let mut best_cost = u64::MAX;
            for &m in PREDICTOR_MODES {
                let c = score_predictor_on_tile(pixels, pixels, w, x0, y0, x1, y1, m);
                if c < best_cost {
                    best_cost = c;
                    best = m;
                }
            }
            modes.push(best);
        }
    }
    modes
}

// ── Colour transform (encoder side) ────────────────────────────────────
//
// Spec §3.6.6. Each tile carries three signed int8 coefficients:
//
//   * `g→r`  — pulled out of red:   r_enc = r - ((g2r * sext(g)) >> 5)
//   * `g→b`  — pulled out of blue:  b_enc = b - ((g2b * sext(g)) >> 5)
//   * `r→b`  — pulled out of blue *after* the green correction, using
//              the ORIGINAL (un-encoded) red: b_enc2 = b_enc - ((r2b * sext(r_orig)) >> 5).
//
// The decoder inverts this by first adding the `g2r` / `g2b` corrections
// back (using the already-decoded green), then adding the `r2b`
// correction (using the already-decoded red). Forward and inverse are
// strict inverses per channel modulo 256.

/// Per-tile colour-transform coefficients. Stored as signed int8 — the
/// decoder sign-extends the same way.
#[derive(Clone, Copy, Default)]
struct ColorCoeffs {
    g2r: i8,
    g2b: i8,
    r2b: i8,
}

/// Cost of a single (pixel, coeffs) pair: sum-of-abs residual over the
/// three mutable channels (green is unchanged). Wrap-around is folded so
/// a large positive residual matches its negative equivalent — tracks
/// Huffman-alphabet magnitude.
fn residual_cost(r: i32, g: i32, b: i32, c: ColorCoeffs) -> u64 {
    let nr = r - (((c.g2r as i32) * sign_extend_i8(g)) >> 5);
    let nb1 = b - (((c.g2b as i32) * sign_extend_i8(g)) >> 5);
    let nb2 = nb1 - (((c.r2b as i32) * sign_extend_i8(r)) >> 5);
    let rr = nr.rem_euclid(256) as u64;
    let bb = nb2.rem_euclid(256) as u64;
    rr.min(256 - rr) + bb.min(256 - bb)
}

/// Sign-extend the low 8 bits of `v` to a signed i32. Mirrors the
/// `coeff as i8 as i32` trick the decoder uses; kept as a function so
/// unit tests can lean on it.
fn sign_extend_i8(v: i32) -> i32 {
    ((v & 0xff) as i8) as i32
}

/// Pick a colour-transform coefficient triple per tile by scanning the
/// [`COLOR_COEFF_GRID`] × [`COLOR_COEFF_GRID`] grid with `r2b=0`, then
/// refining `r2b` around the best (g2r, g2b).
fn choose_color_transform(
    pixels: &[u32],
    width: u32,
    height: u32,
    tile_bits: u32,
    sub_w: u32,
    sub_h: u32,
) -> Vec<ColorCoeffs> {
    let tile_side = 1usize << tile_bits;
    let w = width as usize;
    let h = height as usize;
    let mut out = Vec::with_capacity((sub_w * sub_h) as usize);
    for ty in 0..sub_h as usize {
        for tx in 0..sub_w as usize {
            let x0 = tx * tile_side;
            let y0 = ty * tile_side;
            let x1 = (x0 + tile_side).min(w);
            let y1 = (y0 + tile_side).min(h);

            // Materialise the tile's (r, g, b) triplets — we'll reuse
            // them across every candidate coefficient tuple.
            let mut tile: Vec<(i32, i32, i32)> = Vec::with_capacity((x1 - x0) * (y1 - y0));
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = pixels[y * w + x];
                    let r = ((p >> 16) & 0xff) as i32;
                    let g = ((p >> 8) & 0xff) as i32;
                    let b = (p & 0xff) as i32;
                    tile.push((r, g, b));
                }
            }

            let baseline_score = tile.iter().fold(0u64, |acc, &(r, _g, b)| {
                let rr = r.rem_euclid(256) as u64;
                let bb = b.rem_euclid(256) as u64;
                acc + rr.min(256 - rr) + bb.min(256 - bb)
            });

            let mut best = ColorCoeffs::default();
            let mut best_cost = baseline_score;
            // 16 × 16 grid over (g2r, g2b) with r2b = 0.
            for &g2r in COLOR_COEFF_GRID.iter() {
                for &g2b in COLOR_COEFF_GRID.iter() {
                    let c = ColorCoeffs { g2r, g2b, r2b: 0 };
                    let mut total = 0u64;
                    for &(r, g, b) in &tile {
                        total += residual_cost(r, g, b, c);
                        // Early-abort: if we're already past the best,
                        // further pixels can only make it worse.
                        if total >= best_cost {
                            break;
                        }
                    }
                    if total < best_cost {
                        best_cost = total;
                        best = c;
                    }
                }
            }
            // Second pass: sweep r2b around the best (g2r, g2b).
            for &r2b in COLOR_R2B_GRID.iter() {
                let c = ColorCoeffs {
                    g2r: best.g2r,
                    g2b: best.g2b,
                    r2b,
                };
                let mut total = 0u64;
                for &(r, g, b) in &tile {
                    total += residual_cost(r, g, b, c);
                    if total >= best_cost {
                        break;
                    }
                }
                if total < best_cost {
                    best_cost = total;
                    best = c;
                }
            }
            out.push(best);
        }
    }
    out
}

/// Apply the forward colour transform using per-tile coefficients.
/// Inverse of [`super::transform::apply_color_transform`].
fn apply_color_transform_forward(
    pixels: &[u32],
    width: u32,
    height: u32,
    tile_bits: u32,
    coeffs: &[ColorCoeffs],
    sub_w: u32,
) -> Vec<u32> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let p = pixels[idx];
            let tx = x >> tile_bits;
            let ty = y >> tile_bits;
            let c = coeffs[ty * sub_w as usize + tx];
            let a = (p >> 24) & 0xff;
            let r_orig = ((p >> 16) & 0xff) as i32;
            let g = ((p >> 8) & 0xff) as i32;
            let b_orig = (p & 0xff) as i32;

            // Forward: undo the decoder's green additions, then the
            // `r2b` addition — using the *pre-transform* r as the
            // decoder-side "decoded r" (because forward + inverse
            // cancel out mod 256).
            let new_r = (r_orig - (((c.g2r as i32) * sign_extend_i8(g)) >> 5)).rem_euclid(256);
            let b_after_g = (b_orig - (((c.g2b as i32) * sign_extend_i8(g)) >> 5)).rem_euclid(256);
            let new_b =
                (b_after_g - (((c.r2b as i32) * sign_extend_i8(r_orig)) >> 5)).rem_euclid(256);

            out[idx] = (a << 24) | ((new_r as u32) << 16) | ((g as u32) << 8) | (new_b as u32);
        }
    }
    out
}

/// Apply the predictor transform in the forward (encoder) direction:
/// produce residuals such that the decoder's `apply_predictor` pass
/// recovers the original pixels. The decoder's causal neighbourhood
/// uses *already decoded* pixels (post-residual-add), which by the
/// reconstruction identity equals the original pixel buffer. So we can
/// compute predictions from the originals directly.
fn apply_predictor_forward(
    pixels: &[u32],
    width: u32,
    height: u32,
    tile_bits: u32,
    modes: &[u32],
    sub_w: u32,
) -> Vec<u32> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let pred = if x == 0 && y == 0 {
                0xff00_0000
            } else if y == 0 {
                pixels[idx - 1]
            } else if x == 0 {
                pixels[idx - w]
            } else {
                let tx = x >> tile_bits;
                let ty = y >> tile_bits;
                let mode = modes[ty * sub_w as usize + tx];
                predict_argb(pixels, w, x, y, mode)
            };
            out[idx] = sub_argb(pixels[idx], pred);
        }
    }
    out
}

// ── Huffman tree plumbing (unchanged from the pre-transform encoder) ──

/// Canonical Huffman code length builder with a 15-bit length cap.
fn build_limited_lengths(freqs: &[u32], max_len: u8) -> Result<Vec<u8>> {
    let n = freqs.len();
    let mut lens = vec![0u8; n];
    let nonzero: Vec<usize> = (0..n).filter(|&i| freqs[i] > 0).collect();

    if nonzero.is_empty() {
        if n >= 2 {
            lens[0] = 1;
            lens[1] = 1;
        } else {
            lens[0] = 1;
        }
        return Ok(lens);
    }
    if nonzero.len() == 1 {
        let s = nonzero[0];
        lens[s] = 1;
        let d = if s == 0 { 1.min(n - 1) } else { 0 };
        lens[d] = 1;
        return Ok(lens);
    }

    #[derive(Clone)]
    struct Node {
        freq: u64,
        left: i32,
        right: i32,
        symbol: i32,
    }
    let mut nodes: Vec<Node> = Vec::with_capacity(n * 2);
    for (i, &f) in freqs.iter().enumerate() {
        nodes.push(Node {
            freq: f as u64,
            left: -1,
            right: -1,
            symbol: i as i32,
        });
    }
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> =
        std::collections::BinaryHeap::new();
    for &i in &nonzero {
        heap.push(std::cmp::Reverse((nodes[i].freq, i)));
    }
    while heap.len() > 1 {
        let std::cmp::Reverse((fa, a)) = heap.pop().unwrap();
        let std::cmp::Reverse((fb, b)) = heap.pop().unwrap();
        let idx = nodes.len();
        nodes.push(Node {
            freq: fa + fb,
            left: a as i32,
            right: b as i32,
            symbol: -1,
        });
        heap.push(std::cmp::Reverse((fa + fb, idx)));
    }
    let root = heap.pop().unwrap().0 .1;

    fn walk(nodes: &[Node], idx: usize, depth: u8, lens: &mut [u8]) {
        let n = &nodes[idx];
        if n.symbol >= 0 {
            lens[n.symbol as usize] = depth.max(1);
        } else {
            walk(nodes, n.left as usize, depth + 1, lens);
            walk(nodes, n.right as usize, depth + 1, lens);
        }
    }
    walk(&nodes, root, 0, &mut lens);

    limit_code_lengths(&mut lens, max_len);
    Ok(lens)
}

fn limit_code_lengths(lens: &mut [u8], max_len: u8) {
    // Length-limit a Huffman code: clamp every code length down to
    // `max_len` while keeping the result a complete binary tree (Σ
    // 2^-l_i == 1).
    //
    // Algorithm:
    //   (1) Collapse all symbols at depth > max_len to depth max_len.
    //       This raises the Kraft sum above 1.
    //   (2) Bleed the excess back down by promoting one short code at
    //       a time (depth d → d+1, which subtracts 2^-(d+1) from Kraft).
    //   (3) If we overshot (Kraft < 1, which can happen because phase 2
    //       moves are quantised), fill the remainder by demoting one
    //       deep code at a time (depth d → d-1, which adds 2^-d to
    //       Kraft) — picking depths small enough not to re-overshoot.
    //
    // Kraft is tracked in *exact integer* units of 2^-max_len so the
    // arithmetic is loss-free. A previous version mixed two unit
    // systems (one for phase 1, another for phase 2) and clamped a
    // negative overflow to zero, silently producing an under-complete
    // tree — libwebp rejected those streams with `BITSTREAM_ERROR`
    // (fuzz crash 7bd80cbd, 1×190 image, predictor + colour-cache).
    let max_observed = *lens.iter().max().unwrap_or(&0);
    if max_observed <= max_len {
        return;
    }
    let mut bl_count: Vec<u32> = vec![0; (max_observed as usize + 1).max(1)];
    for &l in lens.iter() {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    // Compute initial Kraft sum (in units of 2^-max_len). Symbols at
    // depth d contribute 2^-d = 2^(max_len-d) units when d <= max_len,
    // and 2^(max_len-d) for d > max_len too — but for d > max_len that
    // is a fractional value (< 1). We only need the integer
    // post-collapse Kraft, so just collapse everything > max_len down
    // to max_len first and compute Kraft on the result.
    for l in (max_len as usize + 1)..bl_count.len() {
        bl_count[max_len as usize] += bl_count[l];
        bl_count[l] = 0;
    }
    // Kraft sum in units of 2^-max_len: depth d contributes 2^(max_len-d).
    let kraft = |bl: &[u32]| -> i64 {
        let mut s: i64 = 0;
        for (d, &c) in bl.iter().enumerate() {
            if d == 0 || d > max_len as usize {
                continue;
            }
            s += (c as i64) << (max_len as usize - d);
        }
        s
    };
    let target: i64 = 1i64 << (max_len as u32);

    // Phase 2: while over-complete, promote a short code (smallest
    // possible move that still bleeds excess). Choose the deepest
    // available depth d < max_len so the promotion granularity is
    // smallest (subtracts 2^(max_len-d-1) units).
    while kraft(&bl_count) > target {
        let mut d = max_len as i32 - 1;
        while d > 0 && bl_count[d as usize] == 0 {
            d -= 1;
        }
        if d <= 0 {
            break;
        }
        bl_count[d as usize] -= 1;
        bl_count[(d + 1) as usize] += 1;
    }

    // Phase 3: if we now under-shot, demote one code at a time —
    // shallower add is bigger, so walk DEEPEST first (smallest add)
    // and only demote if `add ≤ deficit`. Repeat until balanced or
    // no safe move remains.
    loop {
        let k = kraft(&bl_count);
        if k >= target {
            break;
        }
        let deficit = target - k;
        // Pick the deepest depth d > 1 with bl_count[d] > 0 such that
        // moving one symbol from d to d-1 adds 2^(max_len-d) ≤ deficit.
        // (Going from d to d-1 changes contribution from 2^(max_len-d)
        // to 2^(max_len-d+1), a delta of 2^(max_len-d).)
        let mut chosen: Option<i32> = None;
        let mut d = max_len as i32;
        while d > 1 {
            if bl_count[d as usize] > 0 {
                let add = 1i64 << ((max_len as i32 - d).max(0) as u32);
                if add <= deficit {
                    chosen = Some(d);
                    break;
                }
            }
            d -= 1;
        }
        let Some(d) = chosen else {
            break;
        };
        bl_count[d as usize] -= 1;
        bl_count[(d - 1) as usize] += 1;
    }

    let mut by_depth: Vec<(u8, usize)> = lens
        .iter()
        .enumerate()
        .filter(|(_, &l)| l > 0)
        .map(|(i, &l)| (l, i))
        .collect();
    by_depth.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for l in lens.iter_mut() {
        *l = 0;
    }
    let mut idx = 0usize;
    for l in (1..=max_len as usize).rev() {
        let cnt = bl_count[l] as usize;
        for _ in 0..cnt {
            if idx >= by_depth.len() {
                break;
            }
            let (_, sym) = by_depth[idx];
            lens[sym] = l as u8;
            idx += 1;
        }
    }
    while idx < by_depth.len() {
        let (_, sym) = by_depth[idx];
        lens[sym] = max_len;
        idx += 1;
    }
}

fn canonical_codes(lens: &[u8]) -> Vec<u32> {
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

fn write_code(bw: &mut BitWriter, codes: &[u32], lens: &[u8], sym: usize) {
    let l = lens[sym];
    let code = codes[sym];
    let mut rev = 0u32;
    for i in 0..l {
        if (code >> i) & 1 != 0 {
            rev |= 1 << (l - 1 - i);
        }
    }
    bw.write(rev, l as u32);
}

fn emit_huffman_tree(bw: &mut BitWriter, lens: &[u8]) -> Result<()> {
    bw.write(0, 1); // simple-code = 0

    const CODE_LENGTH_ORDER: [usize; 19] = [
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];

    let meta_stream = compress_lengths(lens);

    let mut meta_freq = vec![0u32; 19];
    for (code, _extra) in &meta_stream {
        meta_freq[*code as usize] += 1;
    }
    let meta_lens = build_limited_lengths(&meta_freq, 7)?;
    let meta_codes = canonical_codes(&meta_lens);

    let mut last_used = 0usize;
    for i in 0..19 {
        let sym = CODE_LENGTH_ORDER[i];
        if meta_lens[sym] != 0 {
            last_used = i + 1;
        }
    }
    let num_code_lengths = last_used.max(4);
    bw.write((num_code_lengths - 4) as u32, 4);
    for i in 0..num_code_lengths {
        let sym = CODE_LENGTH_ORDER[i];
        bw.write(meta_lens[sym] as u32, 3);
    }

    bw.write(0, 1);

    for (code, extra) in &meta_stream {
        write_code(bw, &meta_codes, &meta_lens, *code as usize);
        match *code {
            16 => bw.write(*extra, 2),
            17 => bw.write(*extra, 3),
            18 => bw.write(*extra, 7),
            _ => {}
        }
    }
    Ok(())
}

fn compress_lengths(lens: &[u8]) -> Vec<(u8, u32)> {
    let mut out: Vec<(u8, u32)> = Vec::new();
    let mut i = 0usize;
    while i < lens.len() {
        let v = lens[i];
        if v == 0 {
            let mut j = i;
            while j < lens.len() && lens[j] == 0 {
                j += 1;
            }
            let mut run = j - i;
            while run >= 11 {
                let take = run.min(138);
                out.push((18, (take - 11) as u32));
                run -= take;
            }
            while run >= 3 {
                let take = run.min(10);
                out.push((17, (take - 3) as u32));
                run -= take;
            }
            for _ in 0..run {
                out.push((0, 0));
            }
            i = j;
        } else {
            out.push((v, 0));
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_len_dist_roundtrip_small() {
        for value in 1u32..=200 {
            let (sym, eb, extra) = encode_len_or_dist_value(value);
            let decoded = if sym < 4 {
                sym + 1
            } else {
                let eb_d = (sym - 2) >> 1;
                let off = (2 + (sym & 1)) << eb_d;
                off + extra + 1
            };
            assert_eq!(decoded, value, "round-trip failed for value {value}");
            assert!(eb <= 14, "extra_bits too big for value {value}");
            let _ = extra;
            let _ = eb;
        }
    }

    #[test]
    fn canonical_codes_match_decoder_shape() {
        let lens = [2u8, 1u8, 3u8, 3u8];
        let codes = canonical_codes(&lens);
        assert_eq!(codes[1], 0b0);
        assert_eq!(codes[0], 0b10);
        assert_eq!(codes[2], 0b110);
        assert_eq!(codes[3], 0b111);
    }
}
