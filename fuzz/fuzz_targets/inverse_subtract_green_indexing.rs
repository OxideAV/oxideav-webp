#![no_main]

//! Probe the §4.3 inverse-subtract-green + §4.4 inverse-color-table +
//! §4.4 inverse-color-indexing transform passes
//! `oxideav_webp::vp8l_transform::{inverse_subtract_green,
//! inverse_color_table, inverse_color_indexing}` with arbitrary
//! attacker-controlled pixel + color-table + packed-index data.
//!
//! After the §5 entropy stream has emitted the raw §5.1 ARGB residual
//! buffer, every §4 transform in the read-order list runs in reverse
//! against that buffer. The fifteenth harness (`inverse_predictor_color`)
//! covers the two §4 *arithmetic* transforms — §4.1 Predictor and §4.2
//! Color — that read a sub-resolution image and walk the main buffer
//! per-pixel. This sixteenth harness covers the remaining three §4
//! primitives that have no sub-resolution image:
//!
//! * **§4.3 Subtract Green (`inverse_subtract_green`).** A per-pixel
//!   in-place pass that adds the green byte into both the red and blue
//!   bytes (mod 256). No carrier fields, no sub-image, no border
//!   special-case. Alpha and green channels MUST be preserved exactly;
//!   red and blue MUST receive the same green delta.
//! * **§4.4 Color Table Subtraction-Decode (`inverse_color_table`).**
//!   The §4.4 color table is transmitted as a subtraction-coded delta
//!   stream: `table[0]` is the seed, every subsequent entry is the
//!   previous entry plus the wire-coded delta (per channel mod 256).
//!   This in-place pass undoes that delta coding. `table[0]` MUST be
//!   left untouched; every later entry MUST equal the per-channel
//!   running sum of the original input bytes.
//! * **§4.4 Color Indexing (`inverse_color_indexing`).** A pure-function
//!   pass that walks a sub-sampled packed image whose green channel
//!   carries (possibly bundled) palette indices and emits the final
//!   `orig_width * height` ARGB buffer. The bundling level is derived
//!   from the color-table size via the §4.4 threshold table
//!   (1..=2 colors → width_bits 3, count 8, bits 1; 3..=4 → 2, count 4,
//!   bits 2; 5..=16 → 1, count 2, bits 4; >=17 → 0, no bundling). Each
//!   output pixel is `color_table[index]` (or transparent black
//!   `0x00000000` when the index is out of range, per the §4.4 carrier
//!   "any unused indices map to transparent black" rule).
//!
//! Sibling harnesses already cover every surface that **feeds** these
//! transforms — `parse_transform_list` (§4 transform-presence loop that
//! lays out the read-order list, including the §4.4 color-table size +
//! threshold-table `width_bits` derivation), `parse_meta_prefix`
//! (§5.2.3 + §6.2.2 preamble for the §5 entropy bodies that produce
//! the residual + color-table-delta + packed-index buffers),
//! `color_cache` (§5.2.3 cache primitives), `distance_code` (§5.2.2
//! distance-code mapping), `parse_container` (§2.3 / §2.4 RIFF walk to
//! the §2.6 VP8L chunk), `decode` (full §2 RIFF + §3..§5 entry),
//! `roundtrip_lossless` (encode→decode equality oracle on the full §3
//! lossless contract), `roundtrip_animated` (the §2.7.1.1 animation
//! widening of the same round-trip oracle), `inverse_predictor_color`
//! (§4.1 / §4.2 direct passes) — but **none** of them reaches the §4.3
//! subtract-green or the two §4.4 passes directly: they reach them only
//! via whichever residual buffer the upstream §5 entropy decoder produces
//! and whichever color-table-delta + packed-index buffers the upstream
//! sub-image decoders produce, all of which are bounded by the entropy
//! stream itself. This sixteenth harness drives the §4.3 subtract-green
//! + §4.4 inverse-color-table + §4.4 inverse-color-indexing transforms
//! directly across their full attacker-reachable input cross-products
//! within bounded sizes, with the §4.3 alpha/green preservation contract,
//! the §4.4 color-table seed-preservation contract, and the §4.4
//! color-indexing palette-lookup contract cross-checked against the
//! RFC 9649 §4.3 + §4.4 spec text.
//!
//! The contract under test, per RFC 9649 §4.3 + §4.4:
//!
//! * **§4.3 (`inverse_subtract_green`).** The in-place per-pixel pass
//!   applies `red = (red + green) & 0xff` and `blue = (blue + green) &
//!   0xff` for every pixel. Alpha and green are unchanged. The pass is
//!   total (every `&mut [u32]` is in-domain) and pure: the output of
//!   pixel `i` depends only on the input pixel `i`. The buffer length
//!   is preserved.
//! * **§4.4 (`inverse_color_table`).** The in-place pass leaves
//!   `table[0]` alone; for every `i >= 1`, the new `table[i]` is the
//!   per-channel sum of the previous (already-updated) `table[i - 1]`
//!   and the original `table[i]`. Buffer length is preserved. The pass
//!   is sequential — later entries depend on earlier ones — but pure in
//!   the sense that the output for a given input is deterministic. The
//!   single-element table is a no-op.
//! * **§4.4 (`inverse_color_indexing`).** Returns a fresh `orig_width *
//!   height` ARGB `Vec<u32>`. Per output pixel `(x, y)`, the function
//!   reads `packed[y * packed_w + (x / count)].green`, extracts the
//!   `width_bits`-th sub-field at offset `(x % count) * bits`, and emits
//!   `color_table[index]` if `index < color_table.len()` else
//!   transparent black. The buffer is `orig_width * height` long.
//!
//! Every assertion below is a real §4.3 / §4.4 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer. The same is true for
//! any out-of-bounds index, integer overflow, or other unexpected abort
//! raised inside any of the three passes.
//!
//! ## Iteration cost bound
//!
//! Each pass is `O(input_len)` arithmetic. The harness caps
//! `orig_width <= 32`, `height <= 32` (max `1024` pixels), and
//! `color_table_size ∈ [1, 256]` (the §4.4 spec window for the wire
//! `Color cache code bits` field's carrier). The total per-iter work
//! bound is ~10 K cycles regardless of input length.
//!
//! ## Input layout
//!
//! * Byte `[0]` — `orig_width_raw`. Masked to `orig_width = (raw & 0x1F)
//!   | 0x01` so `orig_width ∈ [1, 32]`.
//! * Byte `[1]` — `height_raw`. Same masking → `height ∈ [1, 32]`.
//! * Byte `[2]` — `table_size_raw`. Mapped to `table_size = (raw as
//!   usize) + 1` so `table_size ∈ [1, 256]` (the §4.4 spec window for
//!   the wire `color_table_size`).
//! * Bytes `[3..3 + 4 * width * height]` — pixels for the §4.3
//!   subtract-green pass as little-endian u32 words (zero-padded if the
//!   fuzz buffer is short).
//! * Bytes after that — `table_size` `u32` little-endian color-table
//!   delta entries for the §4.4 subtraction-decode pass.
//! * Bytes after that — packed-index pixels for the §4.4 color-indexing
//!   pass, sized to `DIV_ROUND_UP(orig_width, 1 << width_bits) * height`.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_transform::{
    inverse_color_indexing, inverse_color_table, inverse_subtract_green,
};

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

/// `DIV_ROUND_UP(num, den)` from RFC 9649 §4.4.
#[inline]
fn div_round_up(num: u32, den: u32) -> u32 {
    num.div_ceil(den)
}

/// §4.4 `width_bits` from a color-table size, per the spec's
/// threshold table. Mirror of the private function in
/// `src/vp8l_transform.rs`; the harness MUST NOT call the impl under
/// test as its own oracle, so the threshold derivation is re-stated
/// here from the RFC 9649 §4.4 spec text directly.
#[inline]
fn spec_width_bits(color_table_size: usize) -> u8 {
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
    // -------- §4.3 inverse_subtract_green degenerate: empty buffer --------
    //
    // The §4.3 pass on an empty pixel slice is a no-op; confirm no
    // panic. We always exercise this branch — it costs nothing and
    // gives libFuzzer one more edge to discover.
    {
        let mut empty: Vec<u32> = Vec::new();
        inverse_subtract_green(&mut empty);
        assert!(
            empty.is_empty(),
            "§4.3 inverse_subtract_green must not extend an empty buffer",
        );
    }

    // -------- §4.4 inverse_color_table degenerate: single-element --------
    //
    // A single-element color table has no `i >= 1` entry to update;
    // the §4.4 subtraction-decode is a no-op. Confirm no panic.
    {
        let mut single = vec![0x1234_5678u32];
        inverse_color_table(&mut single);
        assert_eq!(
            single,
            vec![0x1234_5678u32],
            "§4.4 inverse_color_table on a single-element table must be a no-op",
        );
    }

    // We need at least the three carrier bytes (width, height,
    // table_size) for the rest of the harness. Short inputs only
    // exercise the degenerate branches above.
    if data.len() < 3 {
        return;
    }

    let orig_width = (u32::from(data[0]) & 0x1F) | 0x01; // [1, 32]
    let height = (u32::from(data[1]) & 0x1F) | 0x01; // [1, 32]
                                                     // table_size ∈ [1, 256] per the §4.4 spec range. Use the full byte
                                                     // value + 1 to get [1, 256] from [0, 255].
    let table_size: usize = (data[2] as usize) + 1;
    debug_assert!((1..=256).contains(&table_size));

    let pixel_count = (orig_width as usize) * (height as usize);

    // Decode the §4.3 input pixel buffer from the fuzz bytes.
    let mut offset = 3usize;
    let mut subgreen_input = Vec::with_capacity(pixel_count);
    for _ in 0..pixel_count {
        let (word, next) = read_u32_le(data, offset);
        subgreen_input.push(word);
        offset = next;
    }

    // Decode the §4.4 color-table delta buffer.
    let mut delta_table = Vec::with_capacity(table_size);
    for _ in 0..table_size {
        let (word, next) = read_u32_le(data, offset);
        delta_table.push(word);
        offset = next;
    }

    // Decode the §4.4 packed-index buffer. The packed buffer width is
    // `DIV_ROUND_UP(orig_width, 1 << width_bits)`.
    let width_bits = spec_width_bits(table_size);
    let count = 1u32 << width_bits;
    let packed_w = div_round_up(orig_width, count) as usize;
    let packed_count = packed_w * (height as usize);
    let mut packed = Vec::with_capacity(packed_count);
    for _ in 0..packed_count {
        let (word, next) = read_u32_le(data, offset);
        packed.push(word);
        offset = next;
    }

    // ============================================================
    // §4.3 inverse_subtract_green
    // ============================================================

    // Snapshot the input so we can cross-check every §4.3 invariant.
    let pre_subgreen = subgreen_input.clone();
    let mut subgreen_pixels = subgreen_input.clone();
    inverse_subtract_green(&mut subgreen_pixels);

    // §4.3: the in-place pass must leave the buffer the same length —
    // it never reallocates, just mutates.
    assert_eq!(
        subgreen_pixels.len(),
        pixel_count,
        "§4.3 inverse_subtract_green must not change the pixel buffer length",
    );

    // §4.3 alpha + green preservation invariant. Across every pixel:
    // alpha and green are untouched; red and blue receive `+ green`
    // mod 256. This is a derivation-free cross-check (per-channel
    // arithmetic, no neighbours, no carrier image).
    for i in 0..pixel_count {
        let pre = pre_subgreen[i];
        let post = subgreen_pixels[i];
        let g = green(pre);
        assert_eq!(
            alpha(post),
            alpha(pre),
            "§4.3 inverse_subtract_green must preserve the alpha channel at pixel {}",
            i,
        );
        assert_eq!(
            green(post),
            green(pre),
            "§4.3 inverse_subtract_green must preserve the green channel at pixel {}",
            i,
        );
        assert_eq!(
            red(post),
            red(pre).wrapping_add(g),
            "§4.3 inverse_subtract_green red byte must equal red + green (mod 256) at pixel {}",
            i,
        );
        assert_eq!(
            blue(post),
            blue(pre).wrapping_add(g),
            "§4.3 inverse_subtract_green blue byte must equal blue + green (mod 256) at pixel {}",
            i,
        );
    }

    // §4.3 per-pixel locality: the output of pixel `i` depends only on
    // the input pixel `i`. Cross-check by running the pass on a
    // single-pixel buffer at each position and comparing to the
    // multi-pixel output at the matching position. We cap this at the
    // first 8 pixels to keep iteration cost bounded.
    let locality_cap = 8usize.min(pixel_count);
    for i in 0..locality_cap {
        let mut solo = vec![pre_subgreen[i]];
        inverse_subtract_green(&mut solo);
        assert_eq!(
            solo[0], subgreen_pixels[i],
            "§4.3 inverse_subtract_green output at pixel {} must depend only on input pixel {}",
            i, i,
        );
    }

    // §4.3 idempotence under a zero-green input: every pixel whose
    // green byte is zero is unchanged by the pass (the delta added to
    // red and blue is `0 + green == 0`). Run a zero-green replay and
    // confirm byte-identity. The ARGB layout is
    // `alpha:24..32 | red:16..24 | green:8..16 | blue:0..8`, so the
    // green byte is cleared with `& 0xff_ff_00_ff` (keep alpha + red +
    // blue, zero green).
    let mut zero_green: Vec<u32> = pre_subgreen.iter().map(|&p| p & 0xff_ff_00_ff).collect();
    let zero_green_pre = zero_green.clone();
    inverse_subtract_green(&mut zero_green);
    assert_eq!(
        zero_green, zero_green_pre,
        "§4.3 inverse_subtract_green with green byte == 0 must be a no-op",
    );

    // ============================================================
    // §4.4 inverse_color_table
    // ============================================================

    // Snapshot the input so we can cross-check the §4.4 invariants.
    let pre_table = delta_table.clone();
    let mut decoded_table = delta_table.clone();
    inverse_color_table(&mut decoded_table);

    // §4.4: the in-place pass must leave the buffer the same length.
    assert_eq!(
        decoded_table.len(),
        table_size,
        "§4.4 inverse_color_table must not change the color-table length",
    );

    // §4.4 seed-preservation: `table[0]` is left untouched.
    assert_eq!(
        decoded_table[0], pre_table[0],
        "§4.4 inverse_color_table must leave the seed entry (index 0) untouched",
    );

    // §4.4 running-sum: every later entry is the per-channel running
    // sum of the original input bytes (the decoded `table[i]` is the
    // running sum mod 256 of `pre_table[0..=i]`).
    let mut acc_a: u8 = alpha(pre_table[0]);
    let mut acc_r: u8 = red(pre_table[0]);
    let mut acc_g: u8 = green(pre_table[0]);
    let mut acc_b: u8 = blue(pre_table[0]);
    for i in 1..table_size {
        acc_a = acc_a.wrapping_add(alpha(pre_table[i]));
        acc_r = acc_r.wrapping_add(red(pre_table[i]));
        acc_g = acc_g.wrapping_add(green(pre_table[i]));
        acc_b = acc_b.wrapping_add(blue(pre_table[i]));
        assert_eq!(
            alpha(decoded_table[i]),
            acc_a,
            "§4.4 inverse_color_table alpha running sum mismatch at table entry {}",
            i,
        );
        assert_eq!(
            red(decoded_table[i]),
            acc_r,
            "§4.4 inverse_color_table red running sum mismatch at table entry {}",
            i,
        );
        assert_eq!(
            green(decoded_table[i]),
            acc_g,
            "§4.4 inverse_color_table green running sum mismatch at table entry {}",
            i,
        );
        assert_eq!(
            blue(decoded_table[i]),
            acc_b,
            "§4.4 inverse_color_table blue running sum mismatch at table entry {}",
            i,
        );
    }

    // §4.4 determinism: re-running the pass against a fresh clone of
    // the original delta-coded input MUST produce the same output
    // bytes. Cross-checks pure-function behaviour.
    let mut replay_table = pre_table.clone();
    inverse_color_table(&mut replay_table);
    assert_eq!(
        replay_table, decoded_table,
        "§4.4 inverse_color_table must be deterministic across replays",
    );

    // ============================================================
    // §4.4 inverse_color_indexing
    // ============================================================

    let indexed = inverse_color_indexing(&packed, orig_width, height, &decoded_table);

    // §4.4: the output buffer is `orig_width * height` pixels long.
    assert_eq!(
        indexed.len(),
        pixel_count,
        "§4.4 inverse_color_indexing output must be exactly orig_width * height pixels",
    );

    // §4.4: every output pixel is either an entry from the color
    // table or transparent black (`0x00000000`) when the wire index
    // is out of range. Cross-check by re-deriving the per-pixel index
    // from the packed buffer using the spec-text formula and comparing
    // to the impl's output.
    let bits_per_index: u32 = if width_bits == 0 {
        8
    } else {
        (8 / count) as u32
    };
    for y in 0..(height as usize) {
        for x in 0..(orig_width as usize) {
            let packed_x = x / (count as usize);
            let sub = x % (count as usize);
            let packed_pixel = packed[y * packed_w + packed_x];
            let green_byte = green(packed_pixel) as u32;
            let shift = (sub as u32) * bits_per_index;
            let mask: u32 = if width_bits == 0 {
                0xff
            } else {
                (1u32 << bits_per_index) - 1
            };
            let index = ((green_byte >> shift) & mask) as usize;
            let expected = if index < decoded_table.len() {
                decoded_table[index]
            } else {
                // §4.4: "any unused indices map to transparent black".
                0x0000_0000
            };
            let actual = indexed[y * (orig_width as usize) + x];
            assert_eq!(
                actual,
                expected,
                "§4.4 inverse_color_indexing at (x={}, y={}) (width_bits={}, count={}, \
                 bits={}, table_len={}, packed_x={}, sub={}, index={}) must match the §4.4 \
                 palette-lookup spec",
                x,
                y,
                width_bits,
                count,
                bits_per_index,
                decoded_table.len(),
                packed_x,
                sub,
                index,
            );
        }
    }

    // §4.4 determinism: re-running the pass against the same inputs
    // MUST produce byte-identical output.
    let indexed_replay = inverse_color_indexing(&packed, orig_width, height, &decoded_table);
    assert_eq!(
        indexed_replay, indexed,
        "§4.4 inverse_color_indexing must be deterministic across replays",
    );

    // §4.4 transparent-black sweep: when the color table is empty
    // (zero-length), every index is out of range and every output
    // pixel must be `0x00000000`. (Note: the §4.4 wire `color_table_size`
    // is in [1, 256] so a zero-length table is not reachable from the
    // wire — but the impl is reachable from callers directly, and the
    // out-of-range invariant is the same. We cross-check it here.)
    let empty_table: Vec<u32> = Vec::new();
    // With table_len == 0, spec_width_bits returns 3 (since 0 <= 2)
    // → count == 8, so packed_w == DIV_ROUND_UP(orig_width, 8).
    let empty_packed_w = div_round_up(orig_width, 8) as usize;
    let empty_packed = vec![0xffff_ffffu32; empty_packed_w * (height as usize)];
    let oob_only = inverse_color_indexing(&empty_packed, orig_width, height, &empty_table);
    assert_eq!(
        oob_only.len(),
        pixel_count,
        "§4.4 inverse_color_indexing with empty color table must still return orig_width * height pixels",
    );
    for (i, &p) in oob_only.iter().enumerate() {
        assert_eq!(
            p, 0x0000_0000,
            "§4.4 inverse_color_indexing with empty color table must emit transparent black at pixel {}",
            i,
        );
    }
});
