#![no_main]

//! Probe the §4.1 inverse-predictor + §4.2 inverse-color in-place
//! transform passes
//! `oxideav_webp::vp8l_transform::{inverse_predictor, inverse_color}`
//! with arbitrary attacker-controlled pixel + sub-resolution
//! predictor/color images.
//!
//! After the §5 entropy stream has emitted the raw §5.1 ARGB residual
//! buffer, every §4 transform in the read-order list runs in reverse
//! against that buffer. The two arithmetic transforms — §4.1 Predictor
//! and §4.2 Color — read a sub-resolution image (the "predictor image"
//! or the "color image") whose green/red/blue channels encode the
//! per-block prediction mode or the per-block
//! `ColorTransformElement`, and they walk the main `width * height`
//! ARGB buffer applying the inverse derivation in place. Sibling
//! harnesses already cover every surface that **feeds** these
//! transforms — `parse_transform_list` (§4 transform-presence loop
//! that lays out the read-order list), `parse_meta_prefix` (§5.2.3 +
//! §6.2.2 preamble for the §5 entropy bodies), `color_cache` (§5.2.3
//! cache primitives), `distance_code` (§5.2.2 distance-code mapping),
//! `parse_container` (§2.3 / §2.4 RIFF walk to the §2.6 VP8L chunk),
//! `decode` (full §2 RIFF + §3..§5 entry), `roundtrip_lossless`
//! (encode→decode equality oracle on the full §3 lossless contract),
//! `roundtrip_animated` (the §2.7.1.1 animation widening of the same
//! round-trip oracle) — but **none** of them reaches the §4.1 / §4.2
//! in-place arithmetic passes directly: they reach them only via
//! whichever residual buffer the upstream §5 entropy decoder produces
//! and whichever predictor/color image the upstream sub-resolution
//! `decode_entropy_coded_image` produces, both of which are bounded
//! by the entropy stream itself. This fifteenth harness drives the
//! §4.1 inverse predictor and the §4.2 inverse color transforms
//! directly across the full attacker-reachable `(width, height,
//! size_bits, residual_pixels, sub_resolution_image)` cross-product
//! within bounded sizes, with the §4.1 border rules / §4.2
//! alpha-and-green preservation contract / both passes' single-pixel
//! degenerate behaviour cross-checked against the RFC 9649 §4.1 +
//! §4.2 spec text.
//!
//! The contract under test, per RFC 9649 §4.1 + §4.2:
//!
//! * **§4.1 Predictor (`inverse_predictor`).** The pass walks the
//!   `width * height` ARGB buffer in scan-line order. For each pixel
//!   `(x, y)`:
//!     * The top-left pixel `(0, 0)` is reconstructed by adding the
//!       constant prediction `0xff000000` to the residual (per channel,
//!       mod 256). This is the spec text: "the predicted value for the
//!       left-topmost pixel of the image is 0xff000000".
//!     * The remaining top row `(x, 0)` for `x in 1..width` predicts
//!       `L` (the already-reconstructed left neighbour).
//!     * The remaining left column `(0, y)` for `y in 1..height`
//!       predicts `T` (the already-reconstructed top neighbour).
//!     * Interior + right-column pixels read the per-block prediction
//!       mode from `predictor_image[block_index].green` where
//!       `block_index = (y >> size_bits) * transform_width +
//!       (x >> size_bits)`, then apply `predict(mode, L, T, TR, TL)`
//!       per §4.1 Table 2.
//!     * The right column uses the row's leftmost pixel as `TR`, per
//!       §4.1: "the leftmost pixel on the same row as the current
//!       pixel is instead used as the TR-pixel".
//!     * The reconstruction is `final = residual + prediction` per
//!       channel mod 256 (§4.1 `PredictorTransformOutput`).
//!     * The pass is total — every `(width, height, size_bits,
//!       residual, predictor_image)` is in-domain *iff* the
//!       sub-resolution image is at least
//!       `DIV_ROUND_UP(width, 1 << size_bits) *
//!       DIV_ROUND_UP(height, 1 << size_bits)` long and the residual
//!       buffer is at least `width * height` long. The harness only
//!       calls the function when both invariants hold (we size both
//!       buffers from the fuzz-derived dimensions).
//!     * Degenerate dimensions (`width == 0` or `height == 0`) leave
//!       the residual buffer untouched per the §4.1 early-return.
//!     * Single-pixel images (`width == 1 && height == 1`) apply only
//!       the (0, 0) constant — `result = residual + 0xff000000`
//!       independent of the predictor image (per the §4.1 left-topmost
//!       rule).
//! * **§4.2 Color (`inverse_color`).** The pass walks the
//!   `width * height` ARGB buffer; for each pixel `(x, y)` it reads the
//!   `ColorTransformElement` from `color_image[block_index]` (per the
//!   same `block_index` formula) and applies the §4.2
//!   `InverseTransform`:
//!     * `tmp_red += ColorTransformDelta(green_to_red, green)`
//!     * `tmp_blue += ColorTransformDelta(green_to_blue, green)`
//!     * `tmp_blue += ColorTransformDelta(red_to_blue,
//!        tmp_red & 0xff)`
//!       where `ColorTransformDelta(t, c) = (i8(t) * i8(c)) >> 5` and
//!       only the low 8 bits of the result are kept.
//!     * "The alpha and green channels are left as is" — the inverse
//!       color transform NEVER modifies the alpha or green channel of
//!       any output pixel (§4.2 spec text, just below the `InverseTransform`
//!       block).
//!     * Per-block constancy: every pixel inside a `(1 << size_bits) ×
//!       (1 << size_bits)` block reads the *same* `ColorTransformElement`,
//!       so two same-block pixels with identical (red, green, blue) inputs
//!       produce identical outputs.
//!     * Degenerate dimensions (`width == 0` or `height == 0`) leave
//!       the buffer untouched per the §4.2 early-return.
//!     * A zero `ColorTransformElement` (all three fields == 0) is a
//!       no-op: every delta term is `(0 * c) >> 5 == 0` and the
//!       inverse transform reduces to identity.
//!
//! Every assertion below is a real §4.1 / §4.2 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer. The same is true
//! for any out-of-bounds index, integer overflow, or other unexpected
//! abort raised inside the in-place pass.
//!
//! ## Iteration cost bound
//!
//! Each pass is `O(width * height)` arithmetic with the constants
//! `1` (Predictor: at most 4 channel adds + at most ~6 averages +
//! at most 1 `Select` Manhattan distance) and `1` (Color: 3 8-bit
//! signed multiplies + 3 additions). The harness caps `width <= 32`
//! and `height <= 32` (max `1024` pixels), `size_bits ∈ [0, 9]`
//! (the full §4.1 / §4.2 `ReadBits(3) + 2` window plus the size_bits = 0
//! sentinel used by the §4.2 hoist no-op corner). The total per-iter
//! work bound is ~10 K cycles regardless of input length.
//!
//! ## Input layout
//!
//! * Byte `[0]` — `width_raw`. Masked to `width = (raw & 0x1F) | 0x01`
//!   so `width ∈ [1, 32]` (the §4.1 / §4.2 contract is undefined when
//!   `width == 0`; the function's early-return is exercised separately
//!   below).
//! * Byte `[1]` — `height_raw`. Same masking → `height ∈ [1, 32]`.
//! * Byte `[2]` — `size_bits_raw`. Masked to `size_bits = raw % 10`
//!   so `size_bits ∈ [0, 9]`. The §4.1 / §4.2 wire field is
//!   `ReadBits(3) + 2` → `[2, 9]`, plus we also exercise `[0, 1]` to
//!   cover the §4.2 `size_bits == 0` hoist branch and the small-block
//!   corners.
//! * Bytes `[3..3 + 4 * width * height]` — residual ARGB pixels as
//!   little-endian u32 words (zero-padded if the fuzz buffer is short).
//! * Bytes after that — sub-resolution predictor/color image ARGB
//!   pixels as little-endian u32 words (zero-padded if short). Sized
//!   to `DIV_ROUND_UP(width, 1 << size_bits) * DIV_ROUND_UP(height,
//!   1 << size_bits)` (the §4.1 / §4.2 transform-image dimensions).
//!
//! Both transforms then run on independent clones of the residual
//! buffer + the same fuzz-derived sub-resolution image, and the
//! §4.1 border invariants + §4.2 alpha/green preservation invariant +
//! per-block constancy invariant are cross-checked against the
//! pre-pass copy.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_transform::{inverse_color, inverse_predictor};

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

/// `DIV_ROUND_UP(num, den)` from RFC 9649 §4.1.
#[inline]
fn div_round_up(num: u32, den: u32) -> u32 {
    num.div_ceil(den)
}

/// Read the next little-endian u32 from `data` starting at byte
/// `offset`, zero-padding if fewer than 4 bytes remain. Returns the
/// decoded word and the advanced offset.
fn read_u32_le(data: &[u8], offset: usize) -> (u32, usize) {
    let mut buf = [0u8; 4];
    let end = (offset + 4).min(data.len());
    if offset < data.len() {
        let slice = &data[offset..end];
        buf[..slice.len()].copy_from_slice(slice);
    }
    (u32::from_le_bytes(buf), offset + 4)
}

fuzz_target!(|data: &[u8]| {
    // The header is three carrier bytes (width, height, size_bits). We
    // still want to exercise the (width == 0 || height == 0) early-return
    // path the §4.1 / §4.2 functions take, so we do that separately on
    // every short input.
    if data.is_empty() {
        // Both passes accept an empty `pixels` slice with width == 0
        // && height == 0 (their early-return); confirm no panic. We
        // pass an empty sub-resolution image too — the §4.1 / §4.2
        // contract does not read it when the early-return fires.
        let mut pixels: Vec<u32> = Vec::new();
        let sub_image: Vec<u32> = Vec::new();
        inverse_predictor(&mut pixels, 0, 0, &sub_image, 0, 2);
        let mut pixels2: Vec<u32> = Vec::new();
        inverse_color(&mut pixels2, 0, 0, &sub_image, 0, 2);
        return;
    }

    // §4.1 / §4.2 width / height come from the §3.4 image-header but
    // the harness exercises the inverse passes directly, so we cap
    // them locally to keep each iteration microseconds.
    let width = (u32::from(data[0]) & 0x1F) | 0x01; // [1, 32]
    if data.len() < 2 {
        // Too short to fuzz the (width >= 1, height >= 1) path; run the
        // §4.1 (width >= 1, height == 0) early-return to widen
        // coverage on the degenerate branch.
        let mut pixels: Vec<u32> = Vec::new();
        let sub_image: Vec<u32> = Vec::new();
        inverse_predictor(&mut pixels, width, 0, &sub_image, 0, 2);
        return;
    }
    let height = (u32::from(data[1]) & 0x1F) | 0x01; // [1, 32]
    let size_bits = if data.len() < 3 { 2 } else { data[2] % 10 };

    let pixel_count = (width as usize) * (height as usize);
    let block = 1u32 << size_bits;
    let transform_width = div_round_up(width, block);
    let transform_height = div_round_up(height, block);
    let sub_count = (transform_width as usize) * (transform_height as usize);

    // Decode the residual ARGB buffer from the fuzz bytes.
    let mut offset = 3usize;
    let mut residual = Vec::with_capacity(pixel_count);
    for _ in 0..pixel_count {
        let (word, next) = read_u32_le(data, offset);
        residual.push(word);
        offset = next;
    }

    // Decode the sub-resolution predictor/color image from the next
    // little-endian u32 words. The §4.1 / §4.2 transform image is at
    // least 1 pixel by 1 pixel when width/height are both >= 1 (the
    // `DIV_ROUND_UP` always produces >= 1).
    let mut sub_image = Vec::with_capacity(sub_count);
    for _ in 0..sub_count {
        let (word, next) = read_u32_le(data, offset);
        sub_image.push(word);
        offset = next;
    }

    // ---- §4.1 inverse predictor ----

    // Snapshot the residual before the pass so we can cross-check the
    // §4.1 border rules.
    let pre_pred = residual.clone();
    let mut pred_pixels = residual.clone();
    inverse_predictor(
        &mut pred_pixels,
        width,
        height,
        &sub_image,
        transform_width,
        size_bits,
    );

    // §4.1: the in-place pass must leave the buffer the same length —
    // it never reallocates, just mutates.
    assert_eq!(
        pred_pixels.len(),
        pixel_count,
        "§4.1 inverse_predictor must not change the pixel buffer length"
    );

    // §4.1: the left-topmost pixel is reconstructed by adding the
    // constant prediction 0xff000000 to the residual.
    // Per-channel mod 256: alpha = a + 0xff, red = r + 0, green = g + 0,
    // blue = b + 0.
    let r0 = pre_pred[0];
    let expected_00 = ((alpha(r0).wrapping_add(0xff) as u32) << 24)
        | ((red(r0) as u32) << 16)
        | ((green(r0) as u32) << 8)
        | (blue(r0) as u32);
    assert_eq!(
        pred_pixels[0], expected_00,
        "§4.1 left-topmost pixel must equal residual + 0xff000000 (per channel mod 256)",
    );

    // §4.1: a single-pixel image has only the (0, 0) constant rule —
    // the predictor image is never read. The expected output is
    // exactly the (0, 0) rule above; no further assertion needed
    // beyond the equality just done. For 1xH or Wx1 images, the
    // border-row / border-column rules dominate; rather than re-derive
    // the §4.1 predict() table here (which would duplicate the
    // implementation it is supposed to oracle), the harness asserts
    // the §4.1 invariants that are derivation-free.

    // §4.1: when width == 1 (single column), there is no interior
    // pixel, so every pixel `(0, y)` for y >= 1 predicts T (the top
    // neighbour). The reconstruction is bottom-up: pixel `(0, y)` =
    // residual `(0, y)` + pred_pixels `(0, y - 1)` (per channel mod
    // 256). This is a derivation-free cross-check: each pixel of the
    // left column is the sum of the residual at that row and the
    // already-reconstructed pixel at the row above.
    if width == 1 {
        for y in 1..(height as usize) {
            let resid = pre_pred[y];
            let prev = pred_pixels[y - 1];
            let expected = ((alpha(resid).wrapping_add(alpha(prev)) as u32) << 24)
                | ((red(resid).wrapping_add(red(prev)) as u32) << 16)
                | ((green(resid).wrapping_add(green(prev)) as u32) << 8)
                | (blue(resid).wrapping_add(blue(prev)) as u32);
            assert_eq!(
                pred_pixels[y], expected,
                "§4.1 single-column row {} must equal residual + T (per channel mod 256)",
                y,
            );
        }
    }

    // §4.1: when height == 1 (single row), every pixel `(x, 0)` for
    // x >= 1 predicts L (the left neighbour). Same derivation-free
    // cross-check as above, left-to-right.
    if height == 1 {
        for x in 1..(width as usize) {
            let resid = pre_pred[x];
            let prev = pred_pixels[x - 1];
            let expected = ((alpha(resid).wrapping_add(alpha(prev)) as u32) << 24)
                | ((red(resid).wrapping_add(red(prev)) as u32) << 16)
                | ((green(resid).wrapping_add(green(prev)) as u32) << 8)
                | (blue(resid).wrapping_add(blue(prev)) as u32);
            assert_eq!(
                pred_pixels[x], expected,
                "§4.1 single-row col {} must equal residual + L (per channel mod 256)",
                x,
            );
        }
    }

    // ---- §4.2 inverse color ----

    // Snapshot the residual before the pass so we can cross-check the
    // §4.2 alpha-and-green preservation.
    let pre_color = residual.clone();
    let mut color_pixels = residual.clone();
    inverse_color(
        &mut color_pixels,
        width,
        height,
        &sub_image,
        transform_width,
        size_bits,
    );

    // §4.2: the in-place pass must leave the buffer the same length —
    // it never reallocates.
    assert_eq!(
        color_pixels.len(),
        pixel_count,
        "§4.2 inverse_color must not change the pixel buffer length"
    );

    // §4.2 invariant: "The alpha and green channels are left as is."
    // Every output pixel's alpha + green byte must match the input
    // pixel's alpha + green byte exactly — regardless of the
    // ColorTransformElement, regardless of size_bits, regardless of
    // position. This is the central §4.2 contract.
    for i in 0..pixel_count {
        let pre = pre_color[i];
        let post = color_pixels[i];
        assert_eq!(
            alpha(post),
            alpha(pre),
            "§4.2 inverse_color must preserve the alpha channel at pixel {}",
            i,
        );
        assert_eq!(
            green(post),
            green(pre),
            "§4.2 inverse_color must preserve the green channel at pixel {}",
            i,
        );
    }

    // §4.2: a zero ColorTransformElement (the all-zero sub-image
    // pixel) gives a no-op inverse. We cross-check that branch by
    // running the same residual through the pass against a fresh
    // sub-resolution image of all zeros (the alpha byte of every CTE
    // pixel is unused per §4.2, but we zero everything to lock down
    // the contract). The residual must come out byte-identical to the
    // pre-pass copy.
    let zero_sub = vec![0u32; sub_count];
    let mut noop_pixels = residual.clone();
    inverse_color(
        &mut noop_pixels,
        width,
        height,
        &zero_sub,
        transform_width,
        size_bits,
    );
    for i in 0..pixel_count {
        assert_eq!(
            noop_pixels[i], pre_color[i],
            "§4.2 inverse_color with a zero ColorTransformElement must be a no-op at pixel {}",
            i,
        );
    }

    // §4.2 per-block constancy: every pixel inside a `(1 << size_bits)
    // × (1 << size_bits)` block reads the *same* ColorTransformElement,
    // so two same-block pixels whose pre-pass (red, green, blue) bytes
    // are byte-identical must end up with byte-identical post-pass
    // (red, blue) bytes. (Alpha + green preservation is already
    // asserted above; we're additionally locking down that the two
    // arithmetic outputs are equal under equal inputs and equal CTE.)
    //
    // This is N^2 in the worst case so we cap the comparison at the
    // first block only (the upper-left block of `block_w * block_w`
    // pixels), which is sufficient to catch any per-pixel divergence
    // from the per-block constancy without ballooning iteration cost.
    if size_bits >= 1 {
        let block_w = 1usize << size_bits;
        let w = width as usize;
        let h = height as usize;
        let block_h_actual = block_w.min(h);
        let block_w_actual = block_w.min(w);
        for y in 0..block_h_actual {
            for x in 0..block_w_actual {
                for y2 in 0..block_h_actual {
                    for x2 in 0..block_w_actual {
                        let idx1 = y * w + x;
                        let idx2 = y2 * w + x2;
                        if idx1 >= idx2 {
                            continue;
                        }
                        let pre1 = pre_color[idx1];
                        let pre2 = pre_color[idx2];
                        if red(pre1) == red(pre2)
                            && green(pre1) == green(pre2)
                            && blue(pre1) == blue(pre2)
                        {
                            let post1 = color_pixels[idx1];
                            let post2 = color_pixels[idx2];
                            assert_eq!(
                                red(post1),
                                red(post2),
                                "§4.2 per-block constancy: pixels ({},{}) and ({},{}) with equal \
                                 input RGB must produce equal output red within the same block",
                                x,
                                y,
                                x2,
                                y2,
                            );
                            assert_eq!(
                                blue(post1),
                                blue(post2),
                                "§4.2 per-block constancy: pixels ({},{}) and ({},{}) with equal \
                                 input RGB must produce equal output blue within the same block",
                                x,
                                y,
                                x2,
                                y2,
                            );
                        }
                    }
                }
            }
        }
    }

    // ---- §4.1 / §4.2 idempotence under empty dimensions ----

    // Re-run both passes with `width == 0`: the §4.1 / §4.2 early-return
    // must leave the pixel buffer untouched.
    let mut zero_w_pixels = residual.clone();
    inverse_predictor(
        &mut zero_w_pixels,
        0,
        height,
        &sub_image,
        transform_width,
        size_bits,
    );
    assert_eq!(
        zero_w_pixels, pre_pred,
        "§4.1 inverse_predictor with width == 0 must be a no-op"
    );
    let mut zero_w_pixels2 = residual.clone();
    inverse_color(
        &mut zero_w_pixels2,
        0,
        height,
        &sub_image,
        transform_width,
        size_bits,
    );
    assert_eq!(
        zero_w_pixels2, pre_color,
        "§4.2 inverse_color with width == 0 must be a no-op"
    );

    // And with `height == 0`: same contract.
    let mut zero_h_pixels = residual.clone();
    inverse_predictor(
        &mut zero_h_pixels,
        width,
        0,
        &sub_image,
        transform_width,
        size_bits,
    );
    assert_eq!(
        zero_h_pixels, pre_pred,
        "§4.1 inverse_predictor with height == 0 must be a no-op"
    );
    let mut zero_h_pixels2 = residual.clone();
    inverse_color(
        &mut zero_h_pixels2,
        width,
        0,
        &sub_image,
        transform_width,
        size_bits,
    );
    assert_eq!(
        zero_h_pixels2, pre_color,
        "§4.2 inverse_color with height == 0 must be a no-op"
    );
});
