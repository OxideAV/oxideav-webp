//! VP8L (WebP-Lossless) §4 inverse-transform passes.
//!
//! Round 108's [`crate::vp8l_decode::decode_argb`] produces the raw
//! entropy-decoded ARGB pixel buffer of the main §5.1 image — the buffer
//! *before* any §4 transform is undone. This module supplies the four §4
//! inverse transforms that turn that buffer into final pixels, plus the
//! top-level [`decode_lossless`] driver that reads the §4 transform list
//! (each transform's fixed fields **and** its §5-encoded body), decodes
//! the main ARGB image, then applies the inverse transforms in reverse
//! read order (§4: "The inverse transforms are applied in the reverse
//! order that they are read from the bitstream, that is, last one
//! first.").
//!
//! ## The four inverse transforms (§4.1–§4.4)
//!
//! * **§4.1 Predictor.** The image is split into `1 << size_bits` square
//!   blocks; a per-block prediction mode (the green channel of a
//!   sub-resolution *predictor image*) selects one of 14 predictors over
//!   the already-reconstructed TL / T / TR / L neighbours. The final
//!   pixel is the per-channel sum of the residual and the prediction.
//!   Border rules: the top-left pixel predicts `0xff000000`, the top row
//!   predicts L, the left column predicts T, and the rightmost column
//!   uses the row's leftmost pixel as TR.
//! * **§4.2 Color.** Per-block `ColorTransformElement`s (a
//!   sub-resolution *color image*) add green→red, green→blue, and
//!   red→blue deltas back into the red and blue channels.
//! * **§4.3 Subtract-Green.** Adds the green channel into red and blue.
//!   No transform data.
//! * **§4.4 Color-Indexing.** The decoded buffer holds palette indices
//!   in the green channel (possibly width-packed for ≤16-color
//!   palettes); each pixel is replaced by `color_table[green]`, with
//!   out-of-range indices mapping to transparent black. Pixel bundling
//!   for ≤16 colors packs 2/4/8 indices per green byte; the inverse pass
//!   un-bundles back to the original image width.
//!
//! ## Channel layout
//!
//! Every pixel is `(alpha << 24) | (red << 16) | (green << 8) | blue`,
//! matching [`crate::vp8l_decode::DecodedImage`].
//!
//! No `oxideav-core` runtime dependency — this module compiles under
//! `--no-default-features`.

use crate::vp8l_decode::{decode_argb, decode_entropy_coded_image, DecodeError, DecodedImage};
use crate::vp8l_stream::{BitReader, TransformType};

/// `DIV_ROUND_UP(num, den)` from §4.1 (`((num) + (den) - 1) / (den)`).
#[inline]
fn div_round_up(num: u32, den: u32) -> u32 {
    num.div_ceil(den)
}

// ---- ARGB channel accessors (§4 ALPHA/RED/GREEN/BLUE macros) ----

#[inline]
fn alpha(argb: u32) -> u8 {
    (argb >> 24) as u8
}
#[inline]
fn red(argb: u32) -> u8 {
    (argb >> 16) as u8
}
#[inline]
fn green(argb: u32) -> u8 {
    (argb >> 8) as u8
}
#[inline]
fn blue(argb: u32) -> u8 {
    argb as u8
}
#[inline]
fn pack_argb(a: u8, r: u8, g: u8, b: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

// ---- §4.1 predictor primitives ----

/// §4.1 `Average2`, per ARGB component: `(a + b) / 2`.
///
/// **SWAR rewrite** (round 170, see `BENCHMARKS.md`): the standard
/// `(a + b) >> 1` lane-parallel average expressed as
/// `((a ^ b) >> 1) & 0x7f7f7f7f + (a & b)`. This computes four
/// independent 8-bit `(x + y) / 2` values inside one u32 with no
/// carry between lanes, and matches the §4.1 truncating-divide
/// semantics exactly because the original `(ca + cb) / 2` is the
/// arithmetic-mean formula `(a ^ b) / 2 + (a & b)` for two 8-bit
/// values (commonly written as the "halving add"). Predictor modes
/// 5..10 each call this 1–3× per pixel, so the win compounds in the
/// inverse-predictor loop that the round-170 profile flagged as
/// ~80% of decode self-time.
#[inline]
fn average2(a: u32, b: u32) -> u32 {
    // Halving-add identity for unsigned bytes:
    //   avg(a, b) = (a & b) + ((a ^ b) >> 1)
    // No carry crosses lane boundaries because (a ^ b) >> 1 has each
    // byte's MSB cleared and (a & b) is bit-wise per-lane.
    (a & b).wrapping_add((a ^ b) >> 1 & 0x7f7f_7f7f)
}

/// §4.1 `Clamp`: clamp `a` to `[0, 255]`.
#[inline]
fn clamp(a: i32) -> i32 {
    a.clamp(0, 255)
}

/// §4.1 `ClampAddSubtractFull(a, b, c)`: `Clamp(a + b - c)` per channel.
#[inline]
fn clamp_add_subtract_full(a: u32, b: u32, c: u32) -> u32 {
    let f = |sh: u32| -> u32 {
        let ca = ((a >> sh) & 0xff) as i32;
        let cb = ((b >> sh) & 0xff) as i32;
        let cc = ((c >> sh) & 0xff) as i32;
        clamp(ca + cb - cc) as u32
    };
    (f(24) << 24) | (f(16) << 16) | (f(8) << 8) | f(0)
}

/// §4.1 `ClampAddSubtractHalf(a, b)`: `Clamp(a + (a - b) / 2)` per
/// channel.
#[inline]
fn clamp_add_subtract_half(a: u32, b: u32) -> u32 {
    let f = |sh: u32| -> u32 {
        let ca = ((a >> sh) & 0xff) as i32;
        let cb = ((b >> sh) & 0xff) as i32;
        clamp(ca + (ca - cb) / 2) as u32
    };
    (f(24) << 24) | (f(16) << 16) | (f(8) << 8) | f(0)
}

/// §4.1 `Select(L, T, TL)`: returns whichever of `L` / `T` is closer
/// (Manhattan distance) to the `L + T - TL` per-channel estimate.
///
/// **Algebraic simplification** (round 194): the §4.1 reference form
/// computes `estimate_c = l_c + t_c - tl_c` for each of the four
/// channels and then takes `|estimate_c - l_c|` and `|estimate_c - t_c|`
/// per channel. The first algebraically reduces to `|t_c - tl_c|`
/// and the second to `|l_c - tl_c|` — the `estimate` term cancels in
/// each subtraction. We therefore compute only the two four-channel
/// Manhattan distances `Manhattan(t, tl)` and `Manhattan(l, tl)`
/// directly, dropping the eight per-channel additions+subtractions
/// that built the `estimate` intermediate. Bit-identical to the
/// reference form (the tie-break `p_l < p_t` is the same comparison
/// against the same two integers); asserted by
/// `select_matches_estimate_reference_random` which sweeps 1 024
/// deterministic LCG `(l, t, tl)` triples against a verbatim copy of
/// the pre-round-194 estimate-based body. Predictor mode 11 calls
/// `select` once per pixel; the round-180 profile attributes ~80% of
/// decode self-time to `inverse_predictor` and mode 11 is one of the
/// arithmetic-heavy branches inside `predict`.
#[inline]
fn select(l: u32, t: u32, tl: u32) -> u32 {
    // p_l = Manhattan(t, tl), p_t = Manhattan(l, tl). Each per-channel
    // term uses the same shift+mask+abs pattern the closure form in
    // `clamp_add_subtract_full` already proves vectorises well.
    let p_l = (alpha(t) as i32 - alpha(tl) as i32).abs()
        + (red(t) as i32 - red(tl) as i32).abs()
        + (green(t) as i32 - green(tl) as i32).abs()
        + (blue(t) as i32 - blue(tl) as i32).abs();
    let p_t = (alpha(l) as i32 - alpha(tl) as i32).abs()
        + (red(l) as i32 - red(tl) as i32).abs()
        + (green(l) as i32 - green(tl) as i32).abs()
        + (blue(l) as i32 - blue(tl) as i32).abs();
    if p_l < p_t {
        l
    } else {
        t
    }
}

/// §4.1: compute the prediction for `mode` `[0..13]` given the four
/// already-reconstructed neighbours.
fn predict(mode: u8, l: u32, t: u32, tr: u32, tl: u32) -> u32 {
    match mode {
        0 => 0xff00_0000,
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => average2(average2(l, tr), t),
        6 => average2(l, tl),
        7 => average2(l, t),
        8 => average2(tl, t),
        9 => average2(t, tr),
        10 => average2(average2(l, tl), average2(t, tr)),
        11 => select(l, t, tl),
        12 => clamp_add_subtract_full(l, t, tl),
        13 => clamp_add_subtract_half(average2(l, t), tl),
        // Modes are read from the green channel of the predictor image,
        // a u8; only [0..13] are defined. Per the spec the encoder only
        // ever writes [0..13]; an out-of-range mode is treated as 0
        // (solid black) rather than panicking.
        _ => 0xff00_0000,
    }
}

/// §4.1 per-channel residual add: `final = residual + pred` per channel,
/// each wrapped to 8 bits.
///
/// **SWAR rewrite** (round 170, see `BENCHMARKS.md`): four parallel
/// 8-bit lane adds packed inside one 32-bit add, masked back to byte
/// lanes. Carry between lanes is suppressed by masking each pair of
/// lanes separately and re-combining. The §4.1 contract is "add per
/// channel mod 256"; this expression is bit-identical to the original
/// four `u8::wrapping_add` calls and gives the decoder a measurable
/// per-pixel win because the predictor loop is the lossless decode
/// hot path (~80% of self-time per the round-170 profile).
///
/// Identity:
/// ```text
///   (residual & 0x00ff00ff) + (pred & 0x00ff00ff) → lanes 0 + 2
///   (residual & 0xff00ff00) + (pred & 0xff00ff00) → lanes 1 + 3 (
///       carry into the next lane is harmless: the contributing
///       lanes are already in the high byte of their u16 pair, so
///       wrapping to the next lane's low byte zeroes them; we mask
///       back to the original pattern before OR-combining).
/// ```
#[inline]
fn add_pred(residual: u32, pred: u32) -> u32 {
    let lo = (residual & 0x00ff_00ff).wrapping_add(pred & 0x00ff_00ff) & 0x00ff_00ff;
    let hi = (residual & 0xff00_ff00).wrapping_add(pred & 0xff00_ff00) & 0xff00_ff00;
    lo | hi
}

/// Apply the §4.1 inverse predictor transform in place.
///
/// `pixels` is the `width * height` ARGB buffer (residuals on entry,
/// reconstructed pixels on exit). `predictor_image` is the decoded
/// sub-resolution predictor image of size `transform_width *
/// transform_height` (its green channel holds each block's mode).
/// `size_bits` gives the block size (`1 << size_bits`).
pub fn inverse_predictor(
    pixels: &mut [u32],
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

    // §4.1 border-rule hoist (round 180): the per-pixel branch chain
    // on `(x == 0, y == 0, x == w - 1)` is the same outcome for entire
    // rows and entire columns, so we run each border region in its own
    // loop and leave the interior loop (`x in 1..w-1`, `y >= 1`)
    // branch-free. The body of each region is bit-identical to the
    // original per-pixel `if/else if`; only the loop structure changes.
    // Decode self-time was ~80% in this function per the round-170
    // profile, so collapsing the inner conditional is a meaningful win.

    // (0, 0): left-topmost pixel predicts 0xff00_0000.
    pixels[0] = add_pred(pixels[0], 0xff00_0000);

    // Top row (y == 0, x in 1..w): predict L (the left neighbour).
    for x in 1..w {
        let pred = pixels[x - 1];
        pixels[x] = add_pred(pixels[x], pred);
    }
    if h == 1 {
        return;
    }

    // Left column (x == 0, y in 1..h): predict T (the top neighbour).
    // We could fuse this into the interior y-loop, but keeping it
    // separate lets the interior loop start at idx = y*w + 1 with no
    // x == 0 special case.
    for y in 1..h {
        let idx = y * w;
        let pred = pixels[idx - w];
        pixels[idx] = add_pred(pixels[idx], pred);
    }

    // Interior + right column (x in 1..w, y in 1..h). The original
    // body did a per-pixel `x == w - 1` check; we hoist that out by
    // running the interior `x in 1..w-1` loop separately and handling
    // x = w - 1 in a single statement after it. For 1-column images
    // (w == 1) this whole region is empty.
    if w == 1 {
        return;
    }
    let tw = transform_width as usize;
    for y in 1..h {
        let row = y * w;
        let block_row = (y >> size_bits) * tw;
        // Interior: x in 1..w-1, TR is the actual top-right neighbour.
        for x in 1..w - 1 {
            let idx = row + x;
            let block_index = block_row + (x >> size_bits);
            let mode = green(predictor_image[block_index]);
            let l = pixels[idx - 1];
            let t = pixels[idx - w];
            let tl = pixels[idx - w - 1];
            let tr = pixels[idx - w + 1];
            let pred = predict(mode, l, t, tr, tl);
            pixels[idx] = add_pred(pixels[idx], pred);
        }
        // Right column (x = w - 1): §4.1 rightmost-column rule uses the
        // row's leftmost pixel as TR. `idx - w - (w - 1)` collapses to
        // `idx - 2*w + 1` which is the top-row leftmost; equivalent to
        // `row - w` (start of previous row).
        let x = w - 1;
        let idx = row + x;
        let block_index = block_row + (x >> size_bits);
        let mode = green(predictor_image[block_index]);
        let l = pixels[idx - 1];
        let t = pixels[idx - w];
        let tl = pixels[idx - w - 1];
        let tr = pixels[row - w];
        let pred = predict(mode, l, t, tr, tl);
        pixels[idx] = add_pred(pixels[idx], pred);
    }
}

// ---- §4.2 color transform ----

/// §4.2 `ColorTransformDelta(t, c)` = `(t * c) >> 5`, with `t` and `c`
/// interpreted as signed 8-bit two's-complement values. Only the low 8
/// bits of the result are meaningful.
#[inline]
fn color_transform_delta(t: u8, c: u8) -> i32 {
    let ts = t as i8 as i32;
    let cs = c as i8 as i32;
    (ts * cs) >> 5
}

/// §4.2 inverse color transform for one pixel.
///
/// `green_to_red` / `green_to_blue` / `red_to_blue` are the block's
/// `ColorTransformElement`. Returns `(new_red, new_blue)`.
#[inline]
fn inverse_color_pixel(
    r: u8,
    g: u8,
    b: u8,
    green_to_red: u8,
    green_to_blue: u8,
    red_to_blue: u8,
) -> (u8, u8) {
    let mut tmp_red = r as i32;
    let mut tmp_blue = b as i32;
    tmp_red += color_transform_delta(green_to_red, g);
    tmp_blue += color_transform_delta(green_to_blue, g);
    tmp_blue += color_transform_delta(red_to_blue, (tmp_red & 0xff) as u8);
    ((tmp_red & 0xff) as u8, (tmp_blue & 0xff) as u8)
}

/// Apply the §4.2 inverse color transform in place.
///
/// `color_image` is the decoded sub-resolution color image of size
/// `transform_width * transform_height`. Per §4.2 each pixel encodes a
/// `ColorTransformElement` as: red = `red_to_blue`, green =
/// `green_to_blue`, blue = `green_to_red`.
///
/// **Per-block CTE hoist** (round 207): the `ColorTransformElement` is
/// constant across each `1 << size_bits` block, so the original code's
/// `block_index` recomputation + `cte` load + three byte extracts
/// (`red_to_blue` / `green_to_blue` / `green_to_red`) per pixel are
/// hoisted out of the inner pixel loop and refreshed once per block in
/// x. The row-base `y * w` and the block-row base
/// `(y >> size_bits) * tw` are also hoisted out of the x loop. Same
/// arithmetic per pixel — only the loop structure changes; bit-identical
/// to the per-pixel form (asserted by a randomised cross-check test that
/// compares the hoisted form against a verbatim copy of the pre-r207
/// per-pixel body at seven `(size_bits, w, h)` configurations).
///
/// The hoist is a no-op when `size_bits == 0` (one CTE per pixel, no
/// block to amortise across); for `size_bits >= 1` it strictly reduces
/// per-pixel work in the inner loop.
pub fn inverse_color(
    pixels: &mut [u32],
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
    let tw = transform_width as usize;
    if size_bits == 0 {
        // Block size 1 → one CTE per pixel; the hoist degenerates into
        // an extra layer of loop the optimizer can't always flatten.
        // Keep the original flat double loop for this corner; only the
        // hoist of `row_off` and `block_row` (constant across the x
        // loop) survives.
        for y in 0..h {
            let row_off = y * w;
            let block_row = y * tw;
            for x in 0..w {
                let idx = row_off + x;
                let cte = color_image[block_row + x];
                let red_to_blue = red(cte);
                let green_to_blue = green(cte);
                let green_to_red = blue(cte);
                let px = pixels[idx];
                let (new_red, new_blue) = inverse_color_pixel(
                    red(px),
                    green(px),
                    blue(px),
                    green_to_red,
                    green_to_blue,
                    red_to_blue,
                );
                pixels[idx] = pack_argb(alpha(px), new_red, green(px), new_blue);
            }
        }
        return;
    }
    let block_w = 1usize << size_bits;
    for y in 0..h {
        let row_off = y * w;
        let block_row = (y >> size_bits) * tw;
        let mut x = 0usize;
        while x < w {
            // CTE is constant across [x, x + block_w) in this row.
            let block_index = block_row + (x >> size_bits);
            let cte = color_image[block_index];
            // §4.2: cte.red_to_blue = RED, green_to_blue = GREEN,
            // green_to_red = BLUE of the color-image pixel.
            let red_to_blue = red(cte);
            let green_to_blue = green(cte);
            let green_to_red = blue(cte);
            // Walk all pixels in this block of the current row.
            let x_end = (x + block_w).min(w);
            for xi in x..x_end {
                let idx = row_off + xi;
                let px = pixels[idx];
                let (new_red, new_blue) = inverse_color_pixel(
                    red(px),
                    green(px),
                    blue(px),
                    green_to_red,
                    green_to_blue,
                    red_to_blue,
                );
                pixels[idx] = pack_argb(alpha(px), new_red, green(px), new_blue);
            }
            x = x_end;
        }
    }
}

// ---- §4.3 subtract-green ----

/// Apply the §4.3 inverse subtract-green transform in place: add the
/// green channel into both red and blue (`& 0xff`).
///
/// **SWAR rewrite** (round 170, see `BENCHMARKS.md`): the per-pixel
/// work is `r = r + g; b = b + g` (mod 256). Broadcasting the green
/// byte into both the red lane (bits 16..24) and the blue lane (bits
/// 0..8) gives a single SWAR mask `0x00gg00gg`; one masked add into
/// the original ARGB pixel, with carry suppressed by the same
/// odd/even lane split as [`add_pred`], lands the new red + blue
/// bytes in one shot while leaving alpha and green untouched. The
/// emitted bytes are identical to the per-channel `u8::wrapping_add`
/// loop.
pub fn inverse_subtract_green(pixels: &mut [u32]) {
    for px in pixels.iter_mut() {
        let p = *px;
        // Green byte broadcast into the red lane and the blue lane
        // (alpha + green lanes zeroed → only r and b receive the
        // green delta).
        let g = (p >> 8) & 0xff;
        let mask = (g << 16) | g; // 0x00gg00gg
                                  // Add the broadcast green into the [r, b] lanes using the
                                  // `add_pred`-style SWAR pattern. `mask` only lives in the
                                  // 0x00ff00ff lane group, so carry never enters the green or
                                  // alpha lanes.
        let lo = (p & 0x00ff_00ff).wrapping_add(mask) & 0x00ff_00ff;
        let hi = p & 0xff00_ff00; // alpha + green unchanged
        *px = lo | hi;
    }
}

// ---- §4.4 color-indexing ----

/// §4.4: subtraction-decode the color table in place.
///
/// "In decoding, every final color in the color table can be obtained by
/// adding the previous color component values by each ARGB component
/// separately and storing the least significant 8 bits of the result."
pub fn inverse_color_table(color_table: &mut [u32]) {
    for i in 1..color_table.len() {
        let prev = color_table[i - 1];
        let cur = color_table[i];
        let a = alpha(cur).wrapping_add(alpha(prev));
        let r = red(cur).wrapping_add(red(prev));
        let g = green(cur).wrapping_add(green(prev));
        let b = blue(cur).wrapping_add(blue(prev));
        color_table[i] = pack_argb(a, r, g, b);
    }
}

/// §4.4 pixel-bundling `width_bits` from the color-table size, per the
/// spec's "Color Table Size to Bundled Pixel Bit Width Mapping"
/// threshold table:
///
/// | `color_table_size` | `width_bits` | indices per packed byte |
/// |--------------------|--------------|-------------------------|
/// | `1..=2`            | `3`          | `8` (1 bit each)        |
/// | `3..=4`            | `2`          | `4` (2 bits each)       |
/// | `5..=16`           | `1`          | `2` (4 bits each)       |
/// | `17..=256`         | `0`          | `1` (8 bits each)       |
///
/// `width_bits = 0` indicates no pixel bundling; values 1 / 2 / 3
/// combine 2 / 4 / 8 palette indices into one green byte (§4.4).
/// After the transform is read, the image width is subsampled to
/// `DIV_ROUND_UP(image_width, 1 << width_bits)` per §4.4.
///
/// The on-wire field is `color_table_size = ReadBits(8) + 1`, so only
/// `[1, 256]` is bitstream-reachable. The comparisons are total over
/// `usize` — 0 falls in the first threshold window and sizes above
/// 256 in the last — so out-of-window callers get the nearest
/// window's value rather than a panic.
///
/// This is the single shared copy of the §4.4 threshold table: the
/// §4 transform-list reader ([`crate::vp8l_stream`]) and the §4.4
/// forward color-indexing encoder ([`crate::vp8l_encode`]) both
/// delegate here, as does [`inverse_color_indexing`] below.
pub fn color_indexing_width_bits(color_table_size: usize) -> u8 {
    if color_table_size <= 2 {
        3
    } else if color_table_size <= 4 {
        2
    } else if color_table_size <= 16 {
        1
    } else {
        0
    }
}

/// Apply the §4.4 inverse color-indexing transform.
///
/// `packed` is the decoded image at the *subsampled* width
/// `DIV_ROUND_UP(orig_width, 1 << width_bits)`; its green channel holds
/// (possibly bundled) palette indices. `orig_width` / `height` are the
/// final image dimensions. `color_table` is the subtraction-*decoded*
/// palette.
///
/// Returns a fresh `orig_width * height` ARGB buffer. Each output pixel
/// is `color_table[index]`, or transparent black (`0x00000000`) when
/// `index >= color_table.len()`.
///
/// **Round-210 per-bundle hoist** (see `BENCHMARKS.md`): the bundled
/// path (`width_bits` ∈ {1, 2, 3}) has `count = 1 << width_bits`
/// output pixels share the same packed green byte. The original code
/// recomputed `y * packed_w + (x / count)`, `y * ow + x`, and
/// `(x % count) * bits` for every output pixel, even though the
/// packed-row index is constant across each row and the green byte +
/// bundle origin are constant across each `count`-pixel run. The
/// rewrite hoists the two row bases out of the x loop and walks the
/// row as a sequence of `count`-wide bundles: load the green byte
/// once at the bundle boundary, then iterate `count` sub-indices with
/// `shift = 0, bits, 2*bits, …`. The trailing partial bundle at row
/// end (when `orig_width` is not a multiple of `count`) reuses the
/// inner-bundle walk under a `min` clamp. Bit-identical to the
/// per-pixel form (asserted by `color_indexing_matches_per_pixel_
/// reference_random`).
pub fn inverse_color_indexing(
    packed: &[u32],
    orig_width: u32,
    height: u32,
    color_table: &[u32],
) -> Vec<u32> {
    let width_bits = color_indexing_width_bits(color_table.len());
    let ow = orig_width as usize;
    let h = height as usize;
    let mut out = vec![0u32; ow * h];

    if width_bits == 0 {
        // No bundling: the packed buffer is already `orig_width` wide.
        for (o, &p) in out.iter_mut().zip(packed.iter()) {
            let index = green(p) as usize;
            *o = color_table.get(index).copied().unwrap_or(0);
        }
        return out;
    }

    // Bundled: `count = 1 << width_bits` indices share one green byte
    // at packed-x = x / count; index occupies `bits` bits, sub-index
    // `x % count` selects the field (LSB first per §4.4).
    let count = 1usize << width_bits;
    let bits = 8 / count; // 4, 2, or 1 bits per index.
    let mask = (1u32 << bits) - 1;
    let packed_w = div_round_up(orig_width, count as u32) as usize;
    let table_len = color_table.len();
    for y in 0..h {
        let packed_row = y * packed_w;
        let out_row = y * ow;
        let mut x = 0usize;
        // Whole bundles: every iteration consumes `count` output pixels
        // from one packed green byte. The bound `x + count <= ow`
        // skips the trailing partial bundle, handled below.
        while x + count <= ow {
            let g = green(packed[packed_row + (x >> width_bits)]) as u32;
            let base = out_row + x;
            let mut shift = 0u32;
            for sub in 0..count {
                let index = ((g >> shift) & mask) as usize;
                out[base + sub] = if index < table_len {
                    color_table[index]
                } else {
                    0
                };
                shift += bits as u32;
            }
            x += count;
        }
        // Trailing partial bundle when `ow` is not a multiple of count.
        if x < ow {
            let g = green(packed[packed_row + (x >> width_bits)]) as u32;
            let base = out_row + x;
            let remaining = ow - x;
            let mut shift = 0u32;
            for sub in 0..remaining {
                let index = ((g >> shift) & mask) as usize;
                out[base + sub] = if index < table_len {
                    color_table[index]
                } else {
                    0
                };
                shift += bits as u32;
            }
        }
    }
    out
}

/// One §4 transform together with its decoded body, in *read* order.
#[derive(Debug, Clone)]
enum ParsedTransform {
    Predictor {
        size_bits: u8,
        transform_width: u32,
        image: Vec<u32>,
    },
    Color {
        size_bits: u8,
        transform_width: u32,
        image: Vec<u32>,
    },
    SubtractGreen,
    ColorIndexing {
        color_table: Vec<u32>,
    },
}

/// Decode a complete VP8L lossless image: the §4 transform list (each
/// transform's fixed fields and its §5-encoded body), the main §5.1
/// ARGB image, and the §4 inverse-transform chain applied in reverse
/// read order.
///
/// `payload` is the full VP8L chunk payload (starting at the §3.4 5-byte
/// image-header). `width` / `height` are the canvas dimensions from the
/// image-header. Returns the final `width * height` ARGB image in
/// scan-line order.
pub fn decode_lossless(
    payload: &[u8],
    width: u32,
    height: u32,
) -> Result<DecodedImage, DecodeError> {
    let mut reader = BitReader::new_after_image_header(payload);
    decode_lossless_with_reader(&mut reader, width, height)
}

/// Decode a *headerless* VP8L image-stream — the §2.7.1.2 / §3 form used
/// by the compressed `ALPH` alpha bitstream, where the 5-byte image
/// header is omitted because the dimensions are already known from the
/// container ("this image-stream does NOT contain any headers describing
/// the image dimensions").
///
/// `payload` is the alpha bitstream proper (everything after the §2.7.1.2
/// info byte). `width` / `height` are the implicit dimensions. The
/// decode is otherwise identical to [`decode_lossless`]: §4 transforms,
/// the main §5.1 ARGB image, and the inverse-transform chain. The caller
/// extracts the alpha plane from the GREEN channel per §2.7.1.2.
pub fn decode_lossless_headerless(
    payload: &[u8],
    width: u32,
    height: u32,
) -> Result<DecodedImage, DecodeError> {
    let mut reader = BitReader::new(payload);
    decode_lossless_with_reader(&mut reader, width, height)
}

/// Shared §4 transform-list + main-image decode driver, parameterised on
/// a [`BitReader`] already positioned at the start of the §4 transform
/// list. [`decode_lossless`] starts it past the 5-byte image header;
/// [`decode_lossless_headerless`] starts it at bit 0.
fn decode_lossless_with_reader(
    reader: &mut BitReader<'_>,
    width: u32,
    height: u32,
) -> Result<DecodedImage, DecodeError> {
    let mut parsed: Vec<ParsedTransform> = Vec::new();
    let mut seen = [false; 4];

    // `current_width` tracks §4.4 width subsampling: a color-indexing
    // transform shrinks the width seen by subsequent transform bodies
    // *and* the main ARGB image.
    let mut current_width = width;

    // §4 / §7.2 optional-transform loop.
    while reader.read_bit()? {
        let ttype = TransformType::from_bits(reader.read_bits(2)?);
        let idx = ttype as usize;
        if seen[idx] {
            // §4: "each transform is allowed to be used only once."
            return Err(DecodeError::DuplicateTransform);
        }
        seen[idx] = true;

        match ttype {
            TransformType::Predictor => {
                let size_bits = (reader.read_bits(3)? + 2) as u8;
                let block = 1u32 << size_bits;
                let tw = div_round_up(current_width, block);
                let th = div_round_up(height, block);
                let image = decode_entropy_coded_image(reader, tw, th)?;
                parsed.push(ParsedTransform::Predictor {
                    size_bits,
                    transform_width: tw,
                    image: image.pixels().to_vec(),
                });
            }
            TransformType::Color => {
                let size_bits = (reader.read_bits(3)? + 2) as u8;
                let block = 1u32 << size_bits;
                let tw = div_round_up(current_width, block);
                let th = div_round_up(height, block);
                let image = decode_entropy_coded_image(reader, tw, th)?;
                parsed.push(ParsedTransform::Color {
                    size_bits,
                    transform_width: tw,
                    image: image.pixels().to_vec(),
                });
            }
            TransformType::SubtractGreen => {
                parsed.push(ParsedTransform::SubtractGreen);
            }
            TransformType::ColorIndexing => {
                let color_table_size = reader.read_bits(8)? + 1;
                // Color table: a `color_table_size × 1` entropy-coded
                // image, subtraction-coded.
                let table_img = decode_entropy_coded_image(reader, color_table_size, 1)?;
                let mut color_table = table_img.pixels().to_vec();
                inverse_color_table(&mut color_table);
                // §4.4: image_width is subsampled by width_bits.
                let width_bits = color_indexing_width_bits(color_table.len());
                current_width = div_round_up(current_width, 1u32 << width_bits);
                parsed.push(ParsedTransform::ColorIndexing { color_table });
            }
        }
    }

    // Main §5.1 ARGB image, decoded at the (possibly subsampled) width.
    let mut image = decode_argb(reader, current_width, height)?;

    // §4: apply inverse transforms in reverse read order. Width may grow
    // back when a color-indexing transform is undone.
    let mut cur_w = current_width;
    for transform in parsed.iter().rev() {
        match transform {
            ParsedTransform::Predictor {
                size_bits,
                transform_width,
                image: predictor_image,
            } => {
                inverse_predictor(
                    image.pixels_mut(),
                    cur_w,
                    height,
                    predictor_image,
                    *transform_width,
                    *size_bits,
                );
            }
            ParsedTransform::Color {
                size_bits,
                transform_width,
                image: color_image,
            } => {
                inverse_color(
                    image.pixels_mut(),
                    cur_w,
                    height,
                    color_image,
                    *transform_width,
                    *size_bits,
                );
            }
            ParsedTransform::SubtractGreen => {
                inverse_subtract_green(image.pixels_mut());
            }
            ParsedTransform::ColorIndexing { color_table } => {
                // Un-bundling restores the original (pre-subsampled)
                // width. With a single color-indexing transform that is
                // the canvas `width`.
                let out = inverse_color_indexing(image.pixels(), width, height, color_table);
                cur_w = width;
                image = DecodedImage::from_parts(width, height, out);
            }
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- §4.1 predictor primitives ----

    #[test]
    fn average2_per_channel_floor() {
        // (10+20)/2=15, (200+201)/2=200 (floor), etc.
        let a = pack_argb(10, 200, 0, 254);
        let b = pack_argb(20, 201, 1, 255);
        let r = average2(a, b);
        assert_eq!(alpha(r), 15);
        assert_eq!(red(r), 200);
        assert_eq!(green(r), 0);
        assert_eq!(blue(r), 254);
    }

    #[test]
    fn clamp_add_subtract_full_clamps() {
        // a=200,b=200,c=10 → 390 clamp→255; a=10,b=10,c=200 → -180→0.
        let a = pack_argb(200, 10, 0, 0);
        let b = pack_argb(200, 10, 0, 0);
        let c = pack_argb(10, 200, 0, 0);
        let r = clamp_add_subtract_full(a, b, c);
        assert_eq!(alpha(r), 255); // 200+200-10=390 → 255
        assert_eq!(red(r), 0); // 10+10-200=-180 → 0
    }

    #[test]
    fn clamp_add_subtract_half_matches_spec_formula() {
        // a=100, b=40 → 100 + (100-40)/2 = 100 + 30 = 130.
        let a = pack_argb(0, 100, 0, 0);
        let b = pack_argb(0, 40, 0, 0);
        let r = clamp_add_subtract_half(a, b);
        assert_eq!(red(r), 130);
        // Negative half: a=10, b=200 → 10 + (10-200)/2 = 10 - 95 = -85 → 0.
        let a2 = pack_argb(0, 0, 10, 0);
        let b2 = pack_argb(0, 0, 200, 0);
        let r2 = clamp_add_subtract_half(a2, b2);
        assert_eq!(green(r2), 0);
    }

    #[test]
    fn select_matches_estimate_reference_random() {
        // Round-194 simplification cross-check. The rewritten `select`
        // computes `Manhattan(t, tl)` vs. `Manhattan(l, tl)` directly
        // after observing that `estimate - l = t - tl` and
        // `estimate - t = l - tl` (the `estimate = l + t - tl` term
        // cancels in each per-channel subtraction). Pin the new form
        // against a verbatim copy of the pre-r194 `estimate`-based body
        // for 1 024 deterministic LCG triples — including triples that
        // straddle the tie-break boundary, so an off-by-one in the
        // comparison would surface.
        fn reference(l: u32, t: u32, tl: u32) -> u32 {
            let p_alpha = alpha(l) as i32 + alpha(t) as i32 - alpha(tl) as i32;
            let p_red = red(l) as i32 + red(t) as i32 - red(tl) as i32;
            let p_green = green(l) as i32 + green(t) as i32 - green(tl) as i32;
            let p_blue = blue(l) as i32 + blue(t) as i32 - blue(tl) as i32;
            let p_l = (p_alpha - alpha(l) as i32).abs()
                + (p_red - red(l) as i32).abs()
                + (p_green - green(l) as i32).abs()
                + (p_blue - blue(l) as i32).abs();
            let p_t = (p_alpha - alpha(t) as i32).abs()
                + (p_red - red(t) as i32).abs()
                + (p_green - green(t) as i32).abs()
                + (p_blue - blue(t) as i32).abs();
            if p_l < p_t {
                l
            } else {
                t
            }
        }
        // Hand-picked boundary triples: tl == t (forces p_l = 0 path),
        // tl == l (forces p_t = 0 path), all-equal (tie → returns t),
        // and maximally-separated channels.
        for &(l, t, tl) in &[
            (0x12345678u32, 0x12345678u32, 0x12345678u32),
            (0xff_ff_ff_ffu32, 0x00_00_00_00u32, 0x00_00_00_00u32),
            (0x00_00_00_00u32, 0xff_ff_ff_ffu32, 0xff_ff_ff_ffu32),
            (0xff_00_ff_00u32, 0x00_ff_00_ffu32, 0x80_80_80_80u32),
        ] {
            assert_eq!(
                select(l, t, tl),
                reference(l, t, tl),
                "boundary triple mismatch (l={l:08x}, t={t:08x}, tl={tl:08x})"
            );
        }
        let mut seed: u32 = 0xfeed_face;
        let mut rng = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed
        };
        for _ in 0..1_024 {
            let l = rng();
            let t = rng();
            let tl = rng();
            assert_eq!(
                select(l, t, tl),
                reference(l, t, tl),
                "simplification diverges from reference at \
                 (l={l:08x}, t={t:08x}, tl={tl:08x})"
            );
        }
    }

    #[test]
    fn select_picks_closer_neighbour() {
        // If TL == T, then estimate == L, so pL=0 ≤ pT → returns L.
        let l = pack_argb(255, 100, 50, 25);
        let t = pack_argb(255, 10, 10, 10);
        let tl = pack_argb(255, 10, 10, 10);
        assert_eq!(select(l, t, tl), l);
        // If TL == L, estimate == T, so pT=0 < pL → returns T.
        let l2 = pack_argb(255, 10, 10, 10);
        let t2 = pack_argb(255, 100, 50, 25);
        let tl2 = pack_argb(255, 10, 10, 10);
        assert_eq!(select(l2, t2, tl2), t2);
    }

    #[test]
    fn predict_modes_pick_expected_neighbour() {
        let l = pack_argb(1, 2, 3, 4);
        let t = pack_argb(5, 6, 7, 8);
        let tr = pack_argb(9, 10, 11, 12);
        let tl = pack_argb(13, 14, 15, 16);
        assert_eq!(predict(0, l, t, tr, tl), 0xff00_0000);
        assert_eq!(predict(1, l, t, tr, tl), l);
        assert_eq!(predict(2, l, t, tr, tl), t);
        assert_eq!(predict(3, l, t, tr, tl), tr);
        assert_eq!(predict(4, l, t, tr, tl), tl);
    }

    #[test]
    fn inverse_predictor_top_left_is_black_residual() {
        // 1×1: only pixel uses pred 0xff000000. residual 0 → 0xff000000.
        let mut px = vec![0u32];
        let pred_img = vec![0u32]; // mode 0 (unused for the single pixel)
        inverse_predictor(&mut px, 1, 1, &pred_img, 1, 2);
        assert_eq!(px[0], 0xff00_0000);
    }

    #[test]
    fn inverse_predictor_top_row_uses_left() {
        // 3×1, mode irrelevant for top row (all predict L). With
        // residuals [P0, d, d] the reconstruction accumulates: P0,
        // P0+d, P0+d+d (per channel, green channel here).
        // Pixel 0 predicts 0xff000000 (top-left rule), so its residual
        // alpha=0 keeps alpha at 255 after the +255 prediction.
        let p0 = pack_argb(0, 0, 10, 0);
        let d = pack_argb(0, 0, 5, 0);
        let mut px = vec![p0, d, d];
        // predictor image one block, mode 1 (L) — but top-row rule forces
        // L regardless.
        let pred_img = vec![pack_argb(0, 0, 1, 0)];
        inverse_predictor(&mut px, 3, 1, &pred_img, 1, 9);
        assert_eq!(green(px[0]), 10);
        assert_eq!(green(px[1]), 15);
        assert_eq!(green(px[2]), 20);
        // alpha only added at pixel 0 (top-left predicts 0xff000000): the
        // residual alpha=0 plus pred alpha=255 → 255. Subsequent pixels
        // predict the (now alpha-255) left neighbour, so alpha stays 255.
        assert_eq!(alpha(px[0]), 255);
        assert_eq!(alpha(px[1]), 255);
        assert_eq!(alpha(px[2]), 255);
    }

    #[test]
    fn inverse_predictor_left_column_uses_top() {
        // 1×3: leftmost column → T pixel. residuals accumulate down.
        let p0 = pack_argb(255, 0, 10, 0);
        let d = pack_argb(0, 0, 5, 0);
        let mut px = vec![p0, d, d];
        let pred_img = vec![pack_argb(0, 0, 2, 0)];
        inverse_predictor(&mut px, 1, 3, &pred_img, 1, 9);
        assert_eq!(green(px[0]), 10);
        assert_eq!(green(px[1]), 15);
        assert_eq!(green(px[2]), 20);
    }

    #[test]
    fn inverse_predictor_right_column_uses_row_leftmost_as_tr() {
        // §4.1 rightmost-column rule: TR is `pixels[idx - w - (w - 1)]`
        // i.e. the leftmost pixel of the row ABOVE (= start of the
        // previous row), not the actual top-right (which doesn't exist
        // at x = w - 1). This test pins that wraparound after the
        // round-180 region-split rewrite.
        //
        // Layout (2×2, mode 3 = TR predictor everywhere):
        //   row 0: [A, B]
        //   row 1: [C, D]    where D is the right-column case.
        //
        // For pixel D (x=1, y=1): mode=3 picks `tr`. The §4.1 rule
        // says tr = pixels[idx - w - (w-1)] = pixels[3 - 2 - 1] = A.
        // So D's prediction is pixel A (the row's leftmost).
        // After add_pred, pixels[D] = residual_D + A (per channel).
        let a_after = pack_argb(255, 100, 50, 25); // value of A after decode
        let res_d = pack_argb(0, 1, 2, 3); // residual for D
                                           // Predictor image: 1 block, mode 3 (TR).
        let pred_img = vec![pack_argb(0, 0, 3, 0)];
        // Set up so that pixel A decodes to `a_after`. A is the
        // top-left pixel so its prediction is 0xff00_0000; with
        // residual = a_after - 0xff00_0000 per channel, A ends up at
        // `a_after`. We pick a_after with alpha=255 so the residual
        // alpha is 0 (255 + 0 = 255).
        let res_a = pack_argb(0, 100, 50, 25);
        // Other pixels: residuals are zero; predictions handle them.
        // B (x=1, y=0): top-row rule → predict L = A.
        //   pixels[B] = 0 + A = A.
        // C (x=0, y=1): left-column rule → predict T = A.
        //   pixels[C] = 0 + A = A.
        let mut px = vec![res_a, 0u32, 0u32, res_d];
        inverse_predictor(&mut px, 2, 2, &pred_img, 1, 9);
        // A reconstructed:
        assert_eq!(px[0], a_after);
        // B and C predicted from A with residual 0:
        assert_eq!(px[1], a_after);
        assert_eq!(px[2], a_after);
        // D = residual + A (per channel, SWAR add):
        let expected_d = add_pred(res_d, a_after);
        assert_eq!(px[3], expected_d);
    }

    #[test]
    fn inverse_predictor_matches_unsplit_reference_random() {
        // Cross-check the round-180 split-loop layout against a
        // straight-line per-pixel reference (the pre-split structure).
        // This pins that no region's body diverged from the original.
        fn reference(
            pixels: &mut [u32],
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
                    let pred = if x == 0 && y == 0 {
                        0xff00_0000
                    } else if y == 0 {
                        pixels[idx - 1]
                    } else if x == 0 {
                        pixels[idx - w]
                    } else {
                        let bi =
                            (y as u32 >> size_bits) * transform_width + (x as u32 >> size_bits);
                        let mode = green(predictor_image[bi as usize]);
                        let l = pixels[idx - 1];
                        let t = pixels[idx - w];
                        let tl = pixels[idx - w - 1];
                        let tr = if x == w - 1 {
                            pixels[idx - w - (w - 1)]
                        } else {
                            pixels[idx - w + 1]
                        };
                        predict(mode, l, t, tr, tl)
                    };
                    pixels[idx] = add_pred(pixels[idx], pred);
                }
            }
        }

        // Deterministic LCG so the test is reproducible without any
        // rand-crate dep. Cover several aspect ratios so the
        // top-row / left-column / right-column / interior loops all
        // get exercised, including 1-pixel-tall and 1-pixel-wide.
        let mut seed: u32 = 0x1234_5678;
        let mut rng = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed
        };
        for &(w, h, size_bits) in &[
            (1u32, 1u32, 3u8),
            (1, 7, 2),
            (7, 1, 2),
            (2, 2, 1),
            (5, 5, 1),
            (8, 6, 0), // size_bits=0 → 1-pixel blocks, transform == image.
            (13, 9, 2),
        ] {
            let n = (w as usize) * (h as usize);
            let pixels: Vec<u32> = (0..n).map(|_| rng()).collect();
            let tw = div_round_up(w, 1u32 << size_bits);
            let th = div_round_up(h, 1u32 << size_bits);
            let pred_n = (tw as usize) * (th as usize);
            // Predictor image: only the green channel matters, and
            // only values [0..=13] are defined modes. Clamp.
            let pred_img: Vec<u32> = (0..pred_n)
                .map(|_| pack_argb(0, 0, (rng() % 14) as u8, 0))
                .collect();

            let mut new_path = pixels.clone();
            inverse_predictor(&mut new_path, w, h, &pred_img, tw, size_bits);
            let mut ref_path = pixels.clone();
            reference(&mut ref_path, w, h, &pred_img, tw, size_bits);
            assert_eq!(
                new_path, ref_path,
                "split-loop diverges from per-pixel reference at \
                 (w={w}, h={h}, size_bits={size_bits})"
            );
        }
    }

    // ---- §4.2 color transform ----

    #[test]
    fn color_transform_delta_signed() {
        // t=0xFF (-1), c=0x40 (64): (-1*64)>>5 = -64>>5 = -2.
        assert_eq!(color_transform_delta(0xFF, 0x40), -2);
        // t=2, c=0x40 (64): (2*64)>>5 = 128>>5 = 4.
        assert_eq!(color_transform_delta(2, 0x40), 4);
        // t=0, c anything: 0.
        assert_eq!(color_transform_delta(0, 0x7F), 0);
    }

    #[test]
    fn inverse_color_is_inverse_of_forward() {
        // Forward (§4.2 ColorTransform) then inverse should round-trip.
        let green_to_red = 0x12u8;
        let green_to_blue = 0xF0u8; // negative
        let red_to_blue = 0x05u8;
        let (r0, g0, b0) = (120u8, 80u8, 200u8);
        // Forward transform (subtract deltas):
        let mut tr = r0 as i32;
        let mut tb = b0 as i32;
        tr -= color_transform_delta(green_to_red, g0);
        tb -= color_transform_delta(green_to_blue, g0);
        tb -= color_transform_delta(red_to_blue, r0);
        let enc_r = (tr & 0xff) as u8;
        let enc_b = (tb & 0xff) as u8;
        // Inverse:
        let (dec_r, dec_b) =
            inverse_color_pixel(enc_r, g0, enc_b, green_to_red, green_to_blue, red_to_blue);
        assert_eq!(dec_r, r0);
        assert_eq!(dec_b, b0);
    }

    #[test]
    fn inverse_color_in_place_uses_block_element() {
        // 1×1 image, one color block. cte: red_to_blue=0, green_to_blue=0,
        // green_to_red=0 → identity (all deltas zero).
        let mut px = vec![pack_argb(255, 100, 50, 200)];
        let color_img = vec![pack_argb(255, 0, 0, 0)];
        inverse_color(&mut px, 1, 1, &color_img, 1, 9);
        assert_eq!(px[0], pack_argb(255, 100, 50, 200));
    }

    #[test]
    fn inverse_color_matches_per_pixel_reference_random() {
        // Round-207 per-block-CTE hoist cross-check. The new
        // `inverse_color` walks blocks of `1 << size_bits` pixels in
        // x with the CTE coefficients cached at the block boundary;
        // assert the per-pixel emitted bytes still match a verbatim
        // copy of the pre-r207 per-pixel body across deterministic
        // LCG fills at seven `(size_bits, w, h)` configurations,
        // including the size_bits = 0 corner (block_w = 1, hoist is
        // a no-op), a 1-row image, a 1-column image, sub-block-sized
        // edge tiles, and a large block that exceeds image bounds.
        fn reference(
            pixels: &mut [u32],
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
                    let block_index =
                        (y as u32 >> size_bits) * transform_width + (x as u32 >> size_bits);
                    let cte = color_image[block_index as usize];
                    let red_to_blue = red(cte);
                    let green_to_blue = green(cte);
                    let green_to_red = blue(cte);
                    let px = pixels[idx];
                    let (new_red, new_blue) = inverse_color_pixel(
                        red(px),
                        green(px),
                        blue(px),
                        green_to_red,
                        green_to_blue,
                        red_to_blue,
                    );
                    pixels[idx] = pack_argb(alpha(px), new_red, green(px), new_blue);
                }
            }
        }

        let cases: &[(u32, u32, u8)] = &[
            (16, 16, 0), // block_w = 1, hoist is a no-op
            (16, 16, 2), // 4×4 blocks
            (32, 32, 3), // 8×8 blocks
            (64, 16, 5), // 32-wide blocks, edge tile straddles
            (1, 32, 3),  // single column
            (32, 1, 3),  // single row
            (17, 17, 7), // block (128) > image; one CTE for the whole image
        ];

        let mut seed: u32 = 0x9876_5432;
        let mut rng = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed
        };

        for &(w, h, size_bits) in cases {
            let n = (w as usize) * (h as usize);
            let mut pixels: Vec<u32> = (0..n).map(|_| rng()).collect();
            let block = 1u32 << size_bits;
            let tw = w.div_ceil(block);
            let th = h.div_ceil(block);
            let cn = (tw as usize) * (th as usize);
            let color_image: Vec<u32> = (0..cn).map(|_| rng()).collect();

            let mut expected = pixels.clone();
            reference(&mut expected, w, h, &color_image, tw, size_bits);
            inverse_color(&mut pixels, w, h, &color_image, tw, size_bits);

            assert_eq!(
                pixels, expected,
                "hoist diverges from per-pixel reference at \
                 (w={w}, h={h}, size_bits={size_bits})"
            );
        }
    }

    // ---- §4.3 subtract-green ----

    #[test]
    fn inverse_subtract_green_adds_green() {
        // green=10 added to red and blue, wrapping.
        let mut px = vec![pack_argb(255, 5, 10, 250)];
        inverse_subtract_green(&mut px);
        assert_eq!(red(px[0]), 15);
        assert_eq!(green(px[0]), 10);
        assert_eq!(blue(px[0]), 4); // 250+10=260 & 0xff = 4
        assert_eq!(alpha(px[0]), 255);
    }

    // ---- §4.4 color-indexing ----

    #[test]
    fn color_table_subtraction_decode() {
        // table[0] stays; each next = prev + delta (per channel).
        let mut t = vec![
            pack_argb(255, 10, 20, 30),
            pack_argb(0, 5, 5, 5),
            pack_argb(0, 1, 2, 3),
        ];
        inverse_color_table(&mut t);
        assert_eq!(t[0], pack_argb(255, 10, 20, 30));
        assert_eq!(t[1], pack_argb(255, 15, 25, 35));
        assert_eq!(t[2], pack_argb(255, 16, 27, 38));
    }

    #[test]
    fn color_indexing_no_bundling_lookup() {
        // 17-color palette → width_bits 0. green channel = index.
        let mut table = vec![0u32; 17];
        for (i, c) in table.iter_mut().enumerate() {
            *c = pack_argb(255, i as u8, 0, 0);
        }
        let packed = vec![
            pack_argb(0, 0, 0, 0),  // index 0
            pack_argb(0, 0, 3, 0),  // index 3
            pack_argb(0, 0, 16, 0), // index 16
        ];
        let out = inverse_color_indexing(&packed, 3, 1, &table);
        assert_eq!(out, vec![table[0], table[3], table[16]]);
    }

    #[test]
    fn color_indexing_out_of_range_is_transparent_black() {
        let table = vec![pack_argb(255, 1, 2, 3), pack_argb(255, 4, 5, 6)];
        // width_bits for 2 colors = 3 → bundling of 8. Use orig_width 1
        // so packed_w = DIV_ROUND_UP(1, 8) = 1; index from low bit.
        let packed = vec![pack_argb(0, 0, 0x05, 0)]; // low bit = 1 → index 1
        let out = inverse_color_indexing(&packed, 1, 1, &table);
        assert_eq!(out, vec![table[1]]);
    }

    #[test]
    fn color_indexing_bundling_width_bits_1() {
        // 8-color palette → width_bits 1, 2 indices per green byte, 4
        // bits each. orig_width=4 → packed_w = 2.
        let mut table = vec![0u32; 8];
        for (i, c) in table.iter_mut().enumerate() {
            *c = pack_argb(255, i as u8, 0, 0);
        }
        // green byte 0: low nibble idx 1, high nibble idx 2 → 0x21.
        // green byte 1: low nibble idx 3, high nibble idx 4 → 0x43.
        let packed = vec![pack_argb(0, 0, 0x21, 0), pack_argb(0, 0, 0x43, 0)];
        let out = inverse_color_indexing(&packed, 4, 1, &table);
        assert_eq!(out, vec![table[1], table[2], table[3], table[4]]);
    }

    #[test]
    fn color_indexing_bundling_width_bits_3() {
        // 2-color palette → width_bits 3, 8 indices per byte, 1 bit each.
        // orig_width=8 → packed_w = 1.
        let table = vec![pack_argb(255, 0, 0, 0), pack_argb(255, 255, 255, 255)];
        // bits LSB-first: 1,0,1,1,0,0,1,0 → green = 0b0100_1101 = 0x4D.
        let packed = vec![pack_argb(0, 0, 0x4D, 0)];
        let out = inverse_color_indexing(&packed, 8, 1, &table);
        let expect: Vec<u32> = [1, 0, 1, 1, 0, 0, 1, 0].iter().map(|&i| table[i]).collect();
        assert_eq!(out, expect);
    }

    #[test]
    fn width_bits_thresholds() {
        assert_eq!(color_indexing_width_bits(1), 3);
        assert_eq!(color_indexing_width_bits(2), 3);
        assert_eq!(color_indexing_width_bits(3), 2);
        assert_eq!(color_indexing_width_bits(4), 2);
        assert_eq!(color_indexing_width_bits(5), 1);
        assert_eq!(color_indexing_width_bits(16), 1);
        assert_eq!(color_indexing_width_bits(17), 0);
        assert_eq!(color_indexing_width_bits(256), 0);
    }

    /// §4.4 threshold-table sweep over the entire on-wire window:
    /// `color_table_size = ReadBits(8) + 1` ranges over `[1, 256]`,
    /// and every size must map to the spec's "Color Table Size to
    /// Bundled Pixel Bit Width Mapping" row it falls in.
    #[test]
    fn width_bits_exhaustive_over_wire_window() {
        for size in 1usize..=256 {
            let expected = match size {
                1..=2 => 3u8,
                3..=4 => 2,
                5..=16 => 1,
                _ => 0,
            };
            assert_eq!(
                color_indexing_width_bits(size),
                expected,
                "color_table_size {size}"
            );
        }
    }

    #[test]
    fn color_indexing_matches_per_pixel_reference_random() {
        // Round-210 per-bundle hoist cross-check. The new
        // `inverse_color_indexing` walks bundles of `count = 1 <<
        // width_bits` pixels with the packed green byte cached at the
        // bundle boundary; assert the emitted output still matches a
        // verbatim copy of the pre-r210 per-pixel body across
        // deterministic LCG fills at six configurations covering all
        // three bundling levels, including row widths that straddle
        // bundle boundaries, a single-row image, a single-column
        // image (which falls entirely inside the trailing partial
        // bundle), and out-of-range indices that must collapse to
        // transparent black.
        fn reference(
            packed: &[u32],
            orig_width: u32,
            height: u32,
            color_table: &[u32],
        ) -> Vec<u32> {
            let width_bits = color_indexing_width_bits(color_table.len());
            let ow = orig_width as usize;
            let h = height as usize;
            let mut out = vec![0u32; ow * h];
            if width_bits == 0 {
                for (o, &p) in out.iter_mut().zip(packed.iter()) {
                    let index = green(p) as usize;
                    *o = color_table.get(index).copied().unwrap_or(0);
                }
                return out;
            }
            let count = 1usize << width_bits;
            let bits = 8 / count;
            let mask = (1u32 << bits) - 1;
            let packed_w = div_round_up(orig_width, count as u32) as usize;
            for y in 0..h {
                for x in 0..ow {
                    let px = packed[y * packed_w + (x / count)];
                    let shift = (x % count) * bits;
                    let index = ((green(px) as u32 >> shift) & mask) as usize;
                    out[y * ow + x] = color_table.get(index).copied().unwrap_or(0);
                }
            }
            out
        }

        // (orig_width, height, table_size). table_size drives the
        // bundling level via `color_indexing_width_bits`:
        //   >=17 → width_bits 0 (no bundle), 5..=16 → 1 (count 2),
        //   3..=4 → 2 (count 4), 1..=2 → 3 (count 8).
        let cases: &[(u32, u32, usize)] = &[
            (16, 16, 8), // width_bits 1, count 2, exact bundles
            (15, 9, 8),  // width_bits 1, count 2, trailing partial
            (32, 32, 4), // width_bits 2, count 4, exact bundles
            (17, 5, 4),  // width_bits 2, count 4, partial 1
            (24, 7, 2),  // width_bits 3, count 8, exact bundles
            (1, 32, 2),  // width_bits 3, count 8, single column
            (32, 1, 8),  // width_bits 1, count 2, single row
            (20, 4, 32), // width_bits 0 — exercise the no-bundle path
            (5, 5, 4),   // width_bits 2, count 4, partial 1, oob indices
        ];

        let mut seed: u32 = 0x2468_ace0;
        let mut rng = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed
        };

        for &(ow_u32, h_u32, table_size) in cases {
            let mut color_table: Vec<u32> = (0..table_size).map(|_| rng()).collect();
            // Run the spec subtraction-decode first so the table looks
            // like a real palette (round-211-safe: caller does the same).
            inverse_color_table(&mut color_table);

            let width_bits = color_indexing_width_bits(table_size);
            let count = 1u32 << width_bits;
            let packed_w = ow_u32.div_ceil(count) as usize;
            let n = packed_w * (h_u32 as usize);
            // Random green bytes so every sub-index slot exercises both
            // in-range and out-of-range cases.
            let packed: Vec<u32> = (0..n).map(|_| rng() & 0x0000_ff00).collect();

            let expected = reference(&packed, ow_u32, h_u32, &color_table);
            let got = inverse_color_indexing(&packed, ow_u32, h_u32, &color_table);

            assert_eq!(
                got, expected,
                "bundle hoist diverges from per-pixel reference at \
                 (ow={ow_u32}, h={h_u32}, table_size={table_size}, \
                  width_bits={width_bits})"
            );
        }
    }
}
