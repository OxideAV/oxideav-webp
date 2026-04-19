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
//!   picks one of a small set of VP8L predictor modes (0 = opaque
//!   black, 1 = left, 2 = top, 11 = select) by forward-pass cost
//!   estimation; the tile modes ride in a sub-image pixel stream.
//! * **Colour cache** (always on, 256 entries). Every literal pixel is
//!   also addressable by its hashed cache index, which shortens the
//!   green alphabet on repeat colours.
//!
//! What we still don't do (compared to libwebp):
//!
//! * **No colour-indexing (palette) transform.** Palette images still
//!   go through the full ARGB path — inefficient for palettised art but
//!   correct.
//! * **No meta-Huffman image.** A single Huffman group covers the
//!   whole picture.
//! * **Single predictor-mode pool** (0/1/2/11). libwebp probes all 14.
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

use oxideav_core::{Error, Result};

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
/// Limiting the pool keeps the per-tile mode search cheap (a linear
/// sum-of-abs-residuals pass per candidate) and still hits the common
/// correlations for photographic + flat content:
///
/// * 0 — opaque black (good for the top-left tile, alpha-only rows).
/// * 1 — left.
/// * 2 — top.
/// * 11 — "select" (libwebp's workhorse for natural images).
const PREDICTOR_MODES: &[u32] = &[0, 1, 2, 11];

/// LSB-first bit writer matching the VP8L decoder's bit-reader convention.
struct BitWriter {
    out: Vec<u8>,
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

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.cur & 0xff) as u8);
        }
        self.out
    }
}

/// Encode `width × height` ARGB pixels (one u32 per pixel: `a<<24 | r<<16 |
/// g<<8 | b`) as a bare VP8L bitstream (no RIFF wrapper).
///
/// `has_alpha` sets the `alpha_is_used` header bit. It's purely advisory
/// — the alpha channel is transmitted either way.
pub fn encode_vp8l_argb(
    width: u32,
    height: u32,
    pixels: &[u32],
    has_alpha: bool,
) -> Result<Vec<u8>> {
    encode_vp8l_argb_with(width, height, pixels, has_alpha, EncoderOptions::default())
}

/// Encoder tuning knobs. Hidden from the public docs — primarily a
/// testing surface for sizing transforms on/off against each other.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct EncoderOptions {
    pub use_subtract_green: bool,
    pub use_color_transform: bool,
    pub use_predictor: bool,
    pub use_color_cache: bool,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            use_subtract_green: true,
            use_color_transform: true,
            use_predictor: true,
            use_color_cache: true,
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

    if opts.use_subtract_green {
        // Transform header: present + type 2 (SubtractGreen).
        bw.write(1, 1);
        bw.write(2, 2);
        apply_subtract_green_forward(&mut working);
    }

    if opts.use_color_transform {
        // Forward colour transform: per-tile search over the
        // [`COLOR_COEFF_GRID`] × [`COLOR_COEFF_GRID`] grid (256 combos)
        // plus a follow-up `r→b` sweep. Emits a predictor-shaped
        // sub-image (one ARGB pixel per tile, coeffs packed as
        // R = g2r, G = g2b, B = r2b, A = 0xff).
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
                0xff00_0000 | (g2r << 16) | (g2b << 8) | r2b
            })
            .collect();
        encode_image_stream(&mut bw, &sub_pixels, sub_w, sub_h, false, 0)?;

        // Apply the forward colour transform to the working pixels.
        working = apply_color_transform_forward(&working, width, height, tile_bits, &coeffs, sub_w);
    }

    if opts.use_predictor {
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
        COLOR_CACHE_BITS
    } else {
        0
    };
    encode_image_stream(&mut bw, &working, width, height, true, cache_bits)?;

    Ok(bw.finish())
}

/// Encode a `width × height` VP8L image stream (post-transform residuals)
/// into `bw`. Used both for the main picture and for the tiny predictor/
/// colour sub-images. Sub-images always pass `main_image = false` + zero
/// cache bits; the decoder-side parse at [`super::decode_image_stream`]
/// matches that calling convention.
fn encode_image_stream(
    bw: &mut BitWriter,
    pixels: &[u32],
    width: u32,
    height: u32,
    main_image: bool,
    cache_bits: u32,
) -> Result<()> {
    if cache_bits > 0 {
        bw.write(1, 1);
        bw.write(cache_bits, 4);
    } else {
        bw.write(0, 1);
    }

    // Meta-Huffman "present" bit is only read by the decoder on the
    // outermost (main) image; sub-images (predictor mode map, colour
    // sub-image) skip this entirely. We always emit 0 (single group)
    // when we do write it.
    if main_image {
        bw.write(0, 1);
    }

    let cache_size = if cache_bits == 0 {
        0u32
    } else {
        1u32 << cache_bits
    };

    let stream = build_symbol_stream(pixels, width, height, cache_bits);

    let green_alpha = 256 + 24 + cache_size as usize;
    let mut green_freq = vec![0u32; green_alpha];
    let mut red_freq = vec![0u32; 256];
    let mut blue_freq = vec![0u32; 256];
    let mut alpha_freq = vec![0u32; 256];
    let mut dist_freq = vec![0u32; 40];

    for sym in &stream {
        match *sym {
            StreamSym::Literal { a, r, g, b } => {
                green_freq[g as usize] += 1;
                red_freq[r as usize] += 1;
                blue_freq[b as usize] += 1;
                alpha_freq[a as usize] += 1;
            }
            StreamSym::Backref {
                len_sym, dist_sym, ..
            } => {
                green_freq[256 + len_sym as usize] += 1;
                dist_freq[dist_sym as usize] += 1;
            }
            StreamSym::CacheRef { index } => {
                green_freq[256 + 24 + index as usize] += 1;
            }
        }
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

    for sym in &stream {
        match *sym {
            StreamSym::Literal { a, r, g, b } => {
                write_code(bw, &green_codes, &green_lens, g as usize);
                write_code(bw, &red_codes, &red_lens, r as usize);
                write_code(bw, &blue_codes, &blue_lens, b as usize);
                write_code(bw, &alpha_codes, &alpha_lens, a as usize);
            }
            StreamSym::Backref {
                len_sym,
                len_extra_bits,
                len_extra,
                dist_sym,
                dist_extra_bits,
                dist_extra,
            } => {
                write_code(bw, &green_codes, &green_lens, 256 + len_sym as usize);
                if len_extra_bits > 0 {
                    bw.write(len_extra, len_extra_bits);
                }
                write_code(bw, &dist_codes, &dist_lens, dist_sym as usize);
                if dist_extra_bits > 0 {
                    bw.write(dist_extra, dist_extra_bits);
                }
            }
            StreamSym::CacheRef { index } => {
                write_code(bw, &green_codes, &green_lens, 256 + 24 + index as usize);
            }
        }
    }
    Ok(())
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
    let tr = if x + 1 < w {
        out[(y - 1) * w + x + 1]
    } else {
        out[y * w + x - 1]
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
    let mut overflow: u64 = 0;
    for l in (max_len as usize + 1)..bl_count.len() {
        overflow += (bl_count[l] as u64) * ((1u64 << (l - max_len as usize)) - 1);
        bl_count[max_len as usize] += bl_count[l];
        bl_count[l] = 0;
    }

    while overflow > 0 {
        let mut d = max_len as i32 - 1;
        while d > 0 && bl_count[d as usize] == 0 {
            d -= 1;
        }
        if d <= 0 {
            break;
        }
        bl_count[d as usize] -= 1;
        bl_count[(d + 1) as usize] += 1;
        let freed = 1u64 << ((max_len as i32 - d - 1).max(0) as u32);
        if freed >= overflow {
            overflow = 0;
        } else {
            overflow -= freed;
        }
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
