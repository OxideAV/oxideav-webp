//! VP8L transforms — predictor, colour, subtract-green, colour-indexing.
//!
//! Each transform is parsed once from the bitstream and later applied in
//! reverse order during the final image assembly. The predictor and
//! colour transforms carry their own sub-image (a small tiled image of
//! transform parameters); colour-indexing carries a 1D palette; subtract-
//! green has no parameters.

use oxideav_core::{Error, Result};

use super::bit_reader::BitReader;
use super::decode_image_stream;

#[derive(Debug)]
pub enum Transform {
    Predictor {
        tile_bits: u32,
        sub_image: Vec<u32>,
        sub_w: u32,
        #[allow(dead_code)]
        sub_h: u32,
        xsize: u32,
    },
    Color {
        tile_bits: u32,
        sub_image: Vec<u32>,
        sub_w: u32,
        #[allow(dead_code)]
        sub_h: u32,
        xsize: u32,
    },
    SubtractGreen,
    ColorIndex {
        colors: Vec<u32>,
        bits_per_pixel: u32,
        orig_xsize: u32,
    },
}

impl Transform {
    pub fn read(br: &mut BitReader<'_>, xsize: u32, ysize: u32) -> Result<Self> {
        let ty = br.read_bits(2)?;
        match ty {
            0 => {
                // Predictor.
                let tile_bits = br.read_bits(3)? + 2;
                let sub_w = subsampled_size(xsize, tile_bits);
                let sub_h = subsampled_size(ysize, tile_bits);
                let sub = decode_image_stream(br, sub_w, sub_h, false)?;
                Ok(Transform::Predictor {
                    tile_bits,
                    sub_image: sub,
                    sub_w,
                    sub_h,
                    xsize,
                })
            }
            1 => {
                // Colour.
                let tile_bits = br.read_bits(3)? + 2;
                let sub_w = subsampled_size(xsize, tile_bits);
                let sub_h = subsampled_size(ysize, tile_bits);
                let sub = decode_image_stream(br, sub_w, sub_h, false)?;
                Ok(Transform::Color {
                    tile_bits,
                    sub_image: sub,
                    sub_w,
                    sub_h,
                    xsize,
                })
            }
            2 => Ok(Transform::SubtractGreen),
            3 => {
                // Colour indexing.
                let num_colors = br.read_bits(8)? + 1;
                let mut colors_raw = decode_image_stream(br, num_colors, 1, false)?;
                // Colour table is delta-coded along the row (each entry
                // differs from the previous by a per-channel value in
                // modulo 256 arithmetic).
                for i in 1..colors_raw.len() {
                    colors_raw[i] = add_argb(colors_raw[i], colors_raw[i - 1]);
                }
                let bits_per_pixel = if num_colors <= 2 {
                    1
                } else if num_colors <= 4 {
                    2
                } else if num_colors <= 16 {
                    4
                } else {
                    8
                };
                Ok(Transform::ColorIndex {
                    colors: colors_raw,
                    bits_per_pixel,
                    orig_xsize: xsize,
                })
            }
            _ => Err(Error::invalid("VP8L: invalid transform type")),
        }
    }

    /// Width of the image stream produced *after* this transform's parse
    /// step. Used while parsing subsequent transforms. For colour-
    /// indexing the pixel stream is packed: its width shrinks by the
    /// packing factor. Other transforms keep `default_w` unchanged —
    /// the caller passes the current xsize as the default.
    pub fn image_width_or_default(&self, default_w: u32) -> u32 {
        match self {
            Transform::ColorIndex {
                bits_per_pixel,
                orig_xsize,
                ..
            } => {
                let pack = 8 / *bits_per_pixel;
                (orig_xsize + pack - 1) / pack
            }
            _ => default_w,
        }
    }

    /// Width of the image after this transform is *applied* in the
    /// reverse pass. For colour-indexing it expands back to `orig_xsize`;
    /// every other transform is width-neutral.
    pub fn output_width(&self, input_w: u32) -> u32 {
        match self {
            Transform::ColorIndex { orig_xsize, .. } => *orig_xsize,
            _ => input_w,
        }
    }

    pub fn apply(&self, pixels: &[u32], width: u32, height: u32) -> Result<Vec<u32>> {
        match self {
            Transform::Predictor {
                tile_bits,
                sub_image,
                sub_w,
                ..
            } => Ok(apply_predictor(
                pixels, width, height, *tile_bits, sub_image, *sub_w,
            )),
            Transform::Color {
                tile_bits,
                sub_image,
                sub_w,
                ..
            } => Ok(apply_color_transform(
                pixels, width, height, *tile_bits, sub_image, *sub_w,
            )),
            Transform::SubtractGreen => Ok(apply_subtract_green(pixels)),
            Transform::ColorIndex {
                colors,
                bits_per_pixel,
                orig_xsize,
            } => apply_color_index(pixels, width, height, colors, *bits_per_pixel, *orig_xsize),
        }
    }
}

fn subsampled_size(size: u32, bits: u32) -> u32 {
    (size + (1 << bits) - 1) >> bits
}

/// ARGB addition per-component (modulo 256). Used by transforms that
/// encode residuals.
///
/// Implemented with the standard SWAR trick: mask out bit 7 of every
/// byte, add (which can no longer carry into the next byte because the
/// per-byte sum is ≤ 0x7f+0x7f = 0xfe), then re-derive bit 7 of each
/// result byte from `a ^ b` masked to the bit-7 lane. The previous
/// per-byte unpack/add/repack cost ~12 shifts + 8 masks + 4 adds + 4
/// shifts + 3 ORs per call; the SWAR version is 3 ANDs + 1 add + 1
/// XOR + 1 XOR. Called per pixel in the predictor transform inner
/// loop and per palette entry during colour-index delta-decode.
#[inline]
fn add_argb(a: u32, b: u32) -> u32 {
    let masked_sum = (a & 0x7f7f_7f7f).wrapping_add(b & 0x7f7f_7f7f);
    masked_sum ^ ((a ^ b) & 0x8080_8080)
}

// ── Predictor transform ───────────────────────────────────────────────
//
// Each tile gets a predictor mode 0..13 from the sub-image's green
// channel. The decoded pixel is `pred + residual` per-component mod 256,
// where `pred` is computed from the already-decoded neighbourhood.

fn apply_predictor(
    residual: &[u32],
    width: u32,
    height: u32,
    tile_bits: u32,
    sub_image: &[u32],
    sub_w: u32,
) -> Vec<u32> {
    // Build the decoded image into a fresh buffer in raster order. The
    // previous implementation `to_vec()`'d `residual` up-front so it could
    // index `out[idx] = ...` cheaply, but that's a wasted memcpy of the
    // entire residual image (~64 KiB on the 128×128-natural fixture) plus
    // a write-then-overwrite of every cell. Since `predict_argb` only
    // reads the already-decoded neighbourhood (L / T / TL / TR — all at
    // indices strictly less than `idx` in raster order), `Vec::push`
    // works directly: at the moment we compute pixel `idx`, every
    // earlier slot is filled and every later slot is logically untouched.
    //
    // Tile hoisting: the predictor `mode` only varies between tiles
    // (`1 << tile_bits` pixels wide/tall, typically 4..32). Iterating
    // column-tile-by-column-tile lets us look up `mode` once per tile
    // row segment instead of once per pixel — eliminating two shifts +
    // a multiply + a sub-image load from the hot inner loop.
    let pixel_count = residual.len();
    let mut out: Vec<u32> = Vec::with_capacity(pixel_count);
    let w_usize = width as usize;
    let sub_w_usize = sub_w as usize;

    // ── Row 0 (top row): predictor is `out[idx-1]` for x>0 and
    // 0xff00_0000 for the top-left pixel. No tile lookup needed.
    if height > 0 {
        out.push(add_argb(residual[0], 0xff00_0000));
        for x in 1..width as usize {
            let pred = out[x - 1];
            out.push(add_argb(residual[x], pred));
        }
    }

    // ── Rows 1..height: column 0 uses the top neighbour, columns
    // 1..width walk through tiles using a fixed `mode` per tile.
    for y in 1..height {
        let ty = (y >> tile_bits) as usize;
        let row_base = (y * width) as usize;
        // Column 0 — top-only predictor.
        let pred0 = out[row_base - w_usize];
        out.push(add_argb(residual[row_base], pred0));

        // Columns 1..width: walk in tile-sized spans so `mode` is
        // only loaded when crossing a tile boundary.
        let mut x: u32 = 1;
        while x < width {
            let tx = (x >> tile_bits) as usize;
            let mode = (sub_image[ty * sub_w_usize + tx] >> 8) & 0x0f;
            // End of this tile column: next multiple of `tile_size`
            // strictly greater than `x`. Capped at `width`.
            let tile_end = (((x >> tile_bits) + 1) << tile_bits).min(width);
            // Per-mode specialisation: dispatch once per tile-row
            // span, not once per pixel.
            apply_predictor_tile_row(&mut out, residual, w_usize, y as usize, x, tile_end, mode);
            x = tile_end;
        }
    }
    out
}

/// Apply the predictor for one tile-row span — pixels `(x..x_end, y)` —
/// with a fixed `mode`. The mode dispatch is hoisted out of the per-pixel
/// loop so each mode's inner body sees only one branch.
#[inline]
fn apply_predictor_tile_row(
    out: &mut Vec<u32>,
    residual: &[u32],
    w: usize,
    y: usize,
    x_start: u32,
    x_end: u32,
    mode: u32,
) {
    // y >= 1 and x_start >= 1 by construction (caller handles the
    // first-row + first-column special cases). All four neighbours
    // (L, T, TL, TR) are therefore in-bounds.
    let row_base = y * w;
    match mode {
        0 => {
            // Constant 0xff00_0000 — no neighbour reads.
            for x in x_start..x_end {
                let idx = row_base + x as usize;
                out.push(add_argb(residual[idx], 0xff00_0000));
            }
        }
        1 => {
            for x in x_start..x_end {
                let idx = row_base + x as usize;
                let pred = out[idx - 1];
                out.push(add_argb(residual[idx], pred));
            }
        }
        2 => {
            for x in x_start..x_end {
                let idx = row_base + x as usize;
                let pred = out[idx - w];
                out.push(add_argb(residual[idx], pred));
            }
        }
        4 => {
            for x in x_start..x_end {
                let idx = row_base + x as usize;
                let pred = out[idx - w - 1];
                out.push(add_argb(residual[idx], pred));
            }
        }
        _ => {
            // Modes 3, 5..13 all need TR which is column-boundary
            // sensitive (the "leftmost pixel on the same row" wrap),
            // so keep them on the generic path.
            for x in x_start..x_end {
                let idx = row_base + x as usize;
                let pred = predict_argb(out, w, x as usize, y, mode);
                out.push(add_argb(residual[idx], pred));
            }
        }
    }
}

fn predict_argb(out: &[u32], w: usize, x: usize, y: usize, mode: u32) -> u32 {
    let l = out[y * w + x - 1];
    let t = out[(y - 1) * w + x];
    let tl = out[(y - 1) * w + x - 1];
    let tr = if x + 1 < w {
        out[(y - 1) * w + x + 1]
    } else {
        // RFC 9649 §4.1: "Addressing the TR-pixel for pixels on the
        // rightmost column is exceptional. … the leftmost pixel on the
        // same row as the current pixel is instead used as the TR-pixel."
        // (Note: that is column 0 of the *current* row — NOT the LEFT
        // neighbour at column x-1, which is what we previously had and
        // which produced the issue-#8 regression where pixel (1, 53) of
        // a libwebp-encoded 5×78 image cascaded a wrong TR through every
        // row's column-4 predictor into adjacent columns' L/T/TL/TR.)
        out[y * w]
    };
    match mode {
        0 => 0xff00_0000, // opaque black
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

/// Per-byte floor-average of two ARGB pixels: result_byte = (a_byte +
/// b_byte) >> 1 per channel. Used by predictor modes 5..10 which mix
/// neighbour pixels; called multiple times per output pixel for those
/// modes (mode 10 nests two `avg2` calls under another `avg2`).
///
/// SWAR identity: `(a + b) / 2 = (a^b)/2 + (a&b)`. Masking `a^b` with
/// 0xfefefefe before the right-shift drops bit 0 of every byte (which
/// would otherwise be the LSB of the next byte after `>> 1`), keeping
/// the half-shift inside its lane. Lane sums are bounded by 0xff —
/// floor((0xff + 0xff) / 2) — so the wrapping_add can't carry into the
/// next byte.
#[inline]
fn avg2(a: u32, b: u32) -> u32 {
    (a & b).wrapping_add(((a ^ b) & 0xfefe_fefe) >> 1)
}

fn avg3(a: u32, b: u32, c: u32) -> u32 {
    avg2(a, avg2(b, c))
}

/// Predictor mode 11 ("select"): a Paeth-like decision that picks the
/// pixel whose per-channel L1 distance to TL is smaller. The decision is
/// global to the pixel — the result is *either* `l` or `t` whole, never
/// a per-channel mix — so we only need the channel-summed L1 distances.
///
/// Re-shaped from the previous double-loop into:
///   1. Unpack the three pixels into byte arrays once. LLVM lowers this
///      to a single 4-byte load; the compiler then vectorises the
///      summed-abs-diffs into SAD-style instructions on x86 (PSADBW)
///      and the equivalent on aarch64.
///   2. Compute `dl` (summed |t-tl|) and `dt` (summed |l-tl|) in one
///      pass over the four channel positions.
///   3. Branch once on `dl < dt` and return `l` or `t` directly — no
///      per-channel re-pack.
///
/// Avoids 8 shifts + 8 ands + 8 sign-extends + 4 `i32::abs()` + 4
/// shifts + 4 ors that the previous shape forced on every call.
#[inline]
fn select_argb(l: u32, t: u32, tl: u32) -> u32 {
    let lb = l.to_le_bytes();
    let tb = t.to_le_bytes();
    let tlb = tl.to_le_bytes();
    let mut dl: u32 = 0;
    let mut dt: u32 = 0;
    for c in 0..4 {
        // |a - b| on u8 lanes via the unsigned `abs_diff` intrinsic
        // (one instruction on most targets).
        dl += tb[c].abs_diff(tlb[c]) as u32;
        dt += lb[c].abs_diff(tlb[c]) as u32;
    }
    if dl < dt {
        l
    } else {
        t
    }
}

/// Predictor mode 12: per-channel `clamp(l + t - tl, 0, 255)`. Lane
/// arithmetic is a signed sum-then-clamp; the previous shape did this
/// with explicit shift/and/sign-extend per channel plus one `i32::clamp`
/// then a re-pack `or-shift`. The byte-array variant lets LLVM hoist
/// the per-byte unpack/repack and saturate instructions on platforms
/// that have them.
#[inline]
fn clamp_add_sub_argb(l: u32, t: u32, tl: u32) -> u32 {
    let lb = l.to_le_bytes();
    let tb = t.to_le_bytes();
    let tlb = tl.to_le_bytes();
    let mut out = [0u8; 4];
    for c in 0..4 {
        let v = (lb[c] as i32) + (tb[c] as i32) - (tlb[c] as i32);
        out[c] = v.clamp(0, 255) as u8;
    }
    u32::from_le_bytes(out)
}

/// Predictor mode 13: `clamp(a + (a - b) / 2, 0, 255)` per channel.
/// Same byte-array re-shape as `clamp_add_sub_argb` for the same reason.
#[inline]
fn clamp_add_sub_half_argb(a: u32, b: u32) -> u32 {
    let ab = a.to_le_bytes();
    let bb = b.to_le_bytes();
    let mut out = [0u8; 4];
    for c in 0..4 {
        let av = ab[c] as i32;
        let bv = bb[c] as i32;
        let v = av + (av - bv) / 2;
        out[c] = v.clamp(0, 255) as u8;
    }
    u32::from_le_bytes(out)
}

// ── Colour transform ──────────────────────────────────────────────────
//
// Spec §4.2. Removes correlation between R/B channels by subtracting
// scaled versions of G and of (post-subtract) R.

fn apply_color_transform(
    pixels: &[u32],
    width: u32,
    height: u32,
    tile_bits: u32,
    sub_image: &[u32],
    sub_w: u32,
) -> Vec<u32> {
    // Tile hoisting: `coeffs` (and therefore the unpacked r2b / g2b /
    // g2r values + their sign-extends) only changes between tiles. The
    // previous shape did `(x >> tile_bits)` + `(y >> tile_bits)` + a
    // sub-image load + three byte-extracts + three sign-extends per
    // pixel; tiling lifts all of that to once per tile-row span (≥ 4
    // pixels at minimum tile_bits=2, typically 16-32).
    let mut out = Vec::with_capacity(pixels.len());
    let sub_w_usize = sub_w as usize;
    for y in 0..height {
        let ty = (y >> tile_bits) as usize;
        let row_base = (y * width) as usize;
        let mut x: u32 = 0;
        while x < width {
            let tx = (x >> tile_bits) as usize;
            let coeffs = sub_image[ty * sub_w_usize + tx];
            // Coeff packing per WebP lossless spec §4.2 (the "Color
            // Transform" section): each `ColorTransformElement` is
            // stored as an ARGB pixel where
            //   A = 255 (unused)
            //   R = red_to_blue
            //   G = green_to_blue
            //   B = green_to_red
            let r2b = ((coeffs >> 16) & 0xff) as i8 as i32;
            let g2b = ((coeffs >> 8) & 0xff) as i8 as i32;
            let g2r = (coeffs & 0xff) as i8 as i32;
            let tile_end = (((x >> tile_bits) + 1) << tile_bits).min(width);
            for xi in x..tile_end {
                let p = pixels[row_base + xi as usize];
                let a = (p >> 24) & 0xff;
                let mut r = ((p >> 16) & 0xff) as i32;
                let g = ((p >> 8) & 0xff) as i32;
                let mut b = (p & 0xff) as i32;

                // g2r / g2b / r2b are sign-extended 8-bit values; per
                // spec the correction is `((coeff * sign_extend(green)) >> 5)`.
                r = (r + ((g2r * (g as i8 as i32)) >> 5)) & 0xff;
                b = (b + ((g2b * (g as i8 as i32)) >> 5)) & 0xff;
                b = (b + ((r2b * (r as i8 as i32)) >> 5)) & 0xff;

                let argb = (a << 24)
                    | ((r as u32 & 0xff) << 16)
                    | ((g as u32 & 0xff) << 8)
                    | (b as u32 & 0xff);
                out.push(argb);
            }
            x = tile_end;
        }
    }
    out
}

// ── Subtract-green transform ──────────────────────────────────────────

/// Inverse of the encoder's "subtract green" — re-adds the green
/// channel into R and B (per-byte mod 256), leaving A and G untouched.
///
/// SWAR form: broadcast G into the R and B byte lanes (the A and G
/// lanes stay zero), then SWAR-add to the original pixel via the same
/// bit-7-XOR trick as `add_argb`. This collapses the per-byte
/// unpack/add/repack to roughly 5 bitwise ops + 1 add per pixel.
fn apply_subtract_green(pixels: &[u32]) -> Vec<u32> {
    pixels
        .iter()
        .map(|&p| {
            let g = (p >> 8) & 0xff;
            // Broadcast G into the R (bits 16..24) and B (bits 0..8) lanes.
            let g_rb = (g << 16) | g;
            // Per-byte add mod 256 — same SWAR identity as add_argb.
            let masked_sum = (p & 0x7f7f_7f7f).wrapping_add(g_rb & 0x7f7f_7f7f);
            masked_sum ^ ((p ^ g_rb) & 0x8080_8080)
        })
        .collect()
}

// ── Colour indexing transform ─────────────────────────────────────────
//
// The decoded pixel stream is an "index image": each pixel's green
// channel is an index into `colors`. When there are ≤16 colours the
// stream is bit-packed — `bits_per_pixel` indices per green byte.

fn apply_color_index(
    packed: &[u32],
    width: u32,
    _height: u32,
    colors: &[u32],
    bits_per_pixel: u32,
    orig_xsize: u32,
) -> Result<Vec<u32>> {
    let num_colors = colors.len() as u32;
    let pack = 8 / bits_per_pixel;
    let mask = (1u32 << bits_per_pixel) - 1;
    let rows = packed.len() / width as usize;
    let mut out = Vec::with_capacity((orig_xsize as usize) * rows.max(1));
    for y in 0..rows {
        for xp in 0..width as usize {
            let p = packed[y * width as usize + xp];
            let g = (p >> 8) & 0xff;
            for sub in 0..pack {
                let ox = xp * pack as usize + sub as usize;
                if ox >= orig_xsize as usize {
                    break;
                }
                let idx = (g >> (bits_per_pixel * sub)) & mask;
                let color = if idx < num_colors {
                    colors[idx as usize]
                } else {
                    0
                };
                out.push(color);
            }
        }
    }
    Ok(out)
}
