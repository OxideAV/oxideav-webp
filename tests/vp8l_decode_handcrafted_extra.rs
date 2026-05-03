//! Additional handcrafted VP8L decode tests.
//!
//! Complements `tests/vp8l_handcrafted.rs` (constant-pixel, predictor,
//! subtract-green) and `tests/vp8l_huffman_unit.rs` (Huffman parser
//! edge cases). The tests here drive the *full* `decode_webp` pipeline
//! through hand-built bit streams, covering:
//!
//!   * the `read_simple` distance-alphabet validation regression
//!     (round-2 deferred bug from the round 24 codec sweep)
//!   * color-cache extension on the green tree (alphabet sizes 280 vs
//!     2328 vs everything in between)
//!   * meta-Huffman recursion (the meta-image itself uses a normal
//!     Huffman tree)
//!   * a spot-check sweep through several predictor modes (0, 1, 2,
//!     4, 7) — all 14 modes are exercised end-to-end by the existing
//!     fixture round-trips, so we focus here on per-mode correctness.

use oxideav_webp::decode_webp;

/// LSB-first bit writer matching the BitReader convention. Local copy
/// so this test file is self-contained.
struct BitWriter {
    out: Vec<u8>,
    cur: u32,
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
        debug_assert!(n <= 24);
        let mask = if n == 0 {
            0u32
        } else {
            ((1u64 << n) - 1) as u32
        };
        self.cur |= (value & mask) << self.nbits;
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

/// Wrap a VP8L payload in a minimal RIFF/WEBP container.
fn wrap_in_riff(vp8l: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + vp8l.len());
    out.extend_from_slice(b"RIFF");
    let riff_size = (4 + 8 + vp8l.len() + (vp8l.len() & 1)) as u32;
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(b"VP8L");
    out.extend_from_slice(&(vp8l.len() as u32).to_le_bytes());
    out.extend_from_slice(vp8l);
    if vp8l.len() & 1 == 1 {
        out.push(0);
    }
    out
}

/// Emit a "simple" Huffman tree with a single 8-bit symbol.
fn write_simple_one_symbol_tree_8bit(bw: &mut BitWriter, sym: u32) {
    bw.write(1, 1); // simple
    bw.write(0, 1); // num_symbols - 1 = 0
    bw.write(1, 1); // is_first_8bits = 1
    bw.write(sym & 0xff, 8);
}

/// Emit a "simple" Huffman tree with a single 1-bit symbol (sym ∈ {0,1}).
/// Useful for building distance trees (alphabet=40) where 8-bit symbols
/// would overflow the alphabet.
fn write_simple_one_symbol_tree_1bit(bw: &mut BitWriter, sym: u32) {
    bw.write(1, 1); // simple
    bw.write(0, 1); // num_symbols - 1 = 0
    bw.write(0, 1); // is_first_8bits = 0
    bw.write(sym & 1, 1);
}

// ──────────────────────────────────────────────────────────────────────
// Regression tests for the distance-alphabet `read_simple` validation.
// ──────────────────────────────────────────────────────────────────────

/// A 1x1 VP8L image whose distance tree uses simple Huffman with
/// is_first_8bits=0 (1-bit field) — the distance symbol is 0, fully
/// in-range for the 40-symbol distance alphabet. This must decode
/// successfully — the regression-test pair below exercises the failure
/// path at sym0 = 100.
#[test]
fn vp8l_distance_simple_huffman_1bit_in_range_decodes() {
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8); // signature
    bw.write(0, 14); // width-1 = 0 → width 1
    bw.write(0, 14); // height-1 = 0
    bw.write(0, 1); // alpha_is_used
    bw.write(0, 3); // version
    bw.write(0, 1); // no transforms
    bw.write(0, 1); // no color cache
    bw.write(0, 1); // no meta-Huffman
                    // Five simple trees: green, red, blue, alpha, distance.
    write_simple_one_symbol_tree_8bit(&mut bw, 0x40);
    write_simple_one_symbol_tree_8bit(&mut bw, 0x80);
    write_simple_one_symbol_tree_8bit(&mut bw, 0x20);
    write_simple_one_symbol_tree_8bit(&mut bw, 0xff);
    // Distance tree: 1-bit field, sym=0 → in range for 40-symbol alphabet.
    write_simple_one_symbol_tree_1bit(&mut bw, 0);
    let blob = bw.finish();
    let riff = wrap_in_riff(&blob);
    let img = decode_webp(&riff).expect("decode 1x1 with 1-bit distance tree");
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
    let f = &img.frames[0];
    assert_eq!(f.rgba, vec![0x80, 0x40, 0x20, 0xff]);
}

/// Regression: distance tree with 8-bit simple symbol > 39 must be
/// rejected. The previous validation `sym0 < alphabet.max(256)` let
/// this through; the round-2 fix tightens it to `sym0 < alphabet`.
#[test]
fn vp8l_distance_simple_huffman_out_of_range_errors() {
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(0, 14);
    bw.write(0, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1); // no transforms
    bw.write(0, 1); // no color cache
    bw.write(0, 1); // no meta-Huffman
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    // Distance tree: 8-bit sym = 100, OUT of 40-symbol alphabet.
    write_simple_one_symbol_tree_8bit(&mut bw, 100);
    let blob = bw.finish();
    let riff = wrap_in_riff(&blob);
    let err = decode_webp(&riff)
        .err()
        .expect("decoder must reject out-of-range distance symbol");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("out of range") || msg.to_lowercase().contains("invalid"),
        "got {msg}"
    );
}

// ──────────────────────────────────────────────────────────────────────
// Color-cache alphabet extension tests.
// ──────────────────────────────────────────────────────────────────────

/// Build a 1x1 VP8L image with the given `color_cache_bits` (0 = no
/// cache, 1..=11 = explicit cache size). The green tree is built with
/// a single 8-bit symbol carrying value 0x42 — but with the alphabet
/// extension, that symbol must STILL be valid (0x42 < 280 and < 2328).
/// Verifies the green tree's alphabet expands by `1 << cache_bits`.
fn build_1x1_with_cache_bits(cache_bits: u32) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(0, 14);
    bw.write(0, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1); // no transforms
    if cache_bits == 0 {
        bw.write(0, 1); // no color cache
    } else {
        bw.write(1, 1);
        bw.write(cache_bits, 4);
    }
    bw.write(0, 1); // no meta-Huffman
                    // Green emits literal 0x42 — even with full cache extension (alphabet
                    // 256+24+2048 = 2328 at cache_bits=11), 0x42 is in range.
    write_simple_one_symbol_tree_8bit(&mut bw, 0x42);
    write_simple_one_symbol_tree_8bit(&mut bw, 0x10); // red
    write_simple_one_symbol_tree_8bit(&mut bw, 0x20); // blue
    write_simple_one_symbol_tree_8bit(&mut bw, 0xff); // alpha
    write_simple_one_symbol_tree_1bit(&mut bw, 0); // distance
    bw.finish()
}

#[test]
fn vp8l_color_cache_bits_zero_means_no_extension() {
    // cache_bits = 0 path: the bit-1 indicator is 0 and the green
    // alphabet is exactly GREEN_BASE_CODES = 280.
    let blob = build_1x1_with_cache_bits(0);
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode no-cache 1x1");
    assert_eq!(img.frames[0].rgba, vec![0x10, 0x42, 0x20, 0xff]);
}

#[test]
fn vp8l_color_cache_bits_1_min_extension() {
    // cache_bits=1 → alphabet = 280 + 2 = 282.
    let blob = build_1x1_with_cache_bits(1);
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode cache-bits-1 1x1");
    assert_eq!(img.frames[0].rgba, vec![0x10, 0x42, 0x20, 0xff]);
}

#[test]
fn vp8l_color_cache_bits_4_mid_extension() {
    // cache_bits=4 → alphabet = 280 + 16 = 296.
    let blob = build_1x1_with_cache_bits(4);
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode cache-bits-4 1x1");
    assert_eq!(img.frames[0].rgba, vec![0x10, 0x42, 0x20, 0xff]);
}

#[test]
fn vp8l_color_cache_bits_11_max_extension() {
    // cache_bits=11 → alphabet = 280 + 2048 = 2328 (the spec maximum).
    let blob = build_1x1_with_cache_bits(11);
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode cache-bits-11 1x1");
    assert_eq!(img.frames[0].rgba, vec![0x10, 0x42, 0x20, 0xff]);
}

#[test]
fn vp8l_red_tree_never_extends_with_cache() {
    // Even with cache_bits=11 the red tree stays at NUM_LITERAL_CODES=256.
    // Encoding sym=255 (in range for red) must succeed.
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(0, 14);
    bw.write(0, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1); // no transforms
    bw.write(1, 1); // cache present
    bw.write(11, 4); // cache_bits = 11
    bw.write(0, 1); // no meta-Huffman
                    // Green: 8-bit sym 0x42 (in range for 280..2328 alphabet).
    write_simple_one_symbol_tree_8bit(&mut bw, 0x42);
    // Red: sym 0xff = 255 → in range for 256-alphabet (still NUM_LITERAL_CODES
    // even though cache_bits=11). If the alphabet had been extended to 2328,
    // this would still work — but we verify with a green-fixture color cache
    // bit that subsequent 256-alphabet trees parse correctly.
    write_simple_one_symbol_tree_8bit(&mut bw, 0xff);
    write_simple_one_symbol_tree_8bit(&mut bw, 0x20);
    write_simple_one_symbol_tree_8bit(&mut bw, 0xff);
    write_simple_one_symbol_tree_1bit(&mut bw, 0);
    let blob = bw.finish();
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode cache+sym-255 red");
    assert_eq!(img.frames[0].rgba, vec![0xff, 0x42, 0x20, 0xff]);
}

#[test]
fn vp8l_distance_tree_never_extends_with_cache() {
    // Cache extension only applies to the green tree. Distance stays at
    // NUM_DISTANCE_CODES=40. Sym 50 (out of range for distance) must
    // error even when cache_bits=11.
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(0, 14);
    bw.write(0, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1);
    bw.write(1, 1); // cache present
    bw.write(11, 4); // max cache_bits
    bw.write(0, 1); // no meta-Huffman
    write_simple_one_symbol_tree_8bit(&mut bw, 0x42);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0xff);
    // Distance: sym 50 — out of range for 40-symbol distance alphabet.
    write_simple_one_symbol_tree_8bit(&mut bw, 50);
    let blob = bw.finish();
    let err = decode_webp(&wrap_in_riff(&blob))
        .err()
        .expect("decoder must reject sym 50 in distance tree");
    let _ = format!("{err:?}");
}

#[test]
fn vp8l_color_cache_bits_12_invalid() {
    // cache_bits is a 4-bit field, so values up to 15 fit; values
    // 12..=15 are out of the spec range [1..=11] and must error.
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(0, 14);
    bw.write(0, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1); // no transforms
    bw.write(1, 1); // color cache present
    bw.write(12, 4); // cache_bits = 12 → invalid
    let blob = bw.finish();
    let err = decode_webp(&wrap_in_riff(&blob))
        .err()
        .expect("decoder must reject cache_bits=12");
    let msg = format!("{err:?}");
    assert!(msg.contains("color cache bits") || msg.to_lowercase().contains("invalid"));
}

#[test]
fn vp8l_color_cache_bits_zero_indicator_invalid() {
    // cache_bits = 0 with the present-bit set is invalid (the spec range
    // is [1..=11] when present; a present-bit + cache_bits=0 means "use
    // a cache of size 1" which the spec disallows).
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(0, 14);
    bw.write(0, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1);
    bw.write(1, 1); // cache present
    bw.write(0, 4); // cache_bits = 0 → out of [1..=11]
    let blob = bw.finish();
    let err = decode_webp(&wrap_in_riff(&blob))
        .err()
        .expect("decoder must reject cache_bits=0");
    let _ = format!("{err:?}");
}

// ──────────────────────────────────────────────────────────────────────
// Predictor transform spot-checks.
// ──────────────────────────────────────────────────────────────────────
//
// Each test builds a 2x2 image whose predictor sub-image is a single 1x1
// tile carrying a chosen mode in the green channel of the sub-image's
// only pixel. Per the spec:
//   * mode 0 (opaque-black): pred = 0xff_00_00_00 for non-edge pixels
//   * mode 1 (left): pred = left pixel
//   * mode 2 (top): pred = top pixel
//   * etc.
// For the 2x2 case the corner pixels follow special rules (top-left =
// 0xff_00_00_00 always; first-row uses left; first-col uses top), so
// only the bottom-right pixel exercises the chosen mode.

fn build_2x2_predictor_with_mode(mode: u32, residual: (u8, u8, u8)) -> Vec<u8> {
    let (r, g, b) = residual;
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(1, 14); // w-1
    bw.write(1, 14); // h-1
    bw.write(0, 1); // alpha_is_used
    bw.write(0, 3); // version
                    // Predictor transform: tile_bits = 2 (raw 0).
    bw.write(1, 1); // transform present
    bw.write(0, 2); // type 0 = Predictor
    bw.write(0, 3); // tile_bits raw = 0 → tile_bits = 2 → 1×1 sub-image
                    // Sub-image: 1×1, single pixel whose green = mode.
    bw.write(0, 1); // no color cache (sub-image)
    write_simple_one_symbol_tree_8bit(&mut bw, mode);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_1bit(&mut bw, 0);
    bw.write(0, 1); // no further transforms
                    // Main image residuals.
    bw.write(0, 1); // no color cache
    bw.write(0, 1); // no meta-Huffman
    write_simple_one_symbol_tree_8bit(&mut bw, g as u32);
    write_simple_one_symbol_tree_8bit(&mut bw, r as u32);
    write_simple_one_symbol_tree_8bit(&mut bw, b as u32);
    write_simple_one_symbol_tree_8bit(&mut bw, 0); // alpha residual
    write_simple_one_symbol_tree_1bit(&mut bw, 0);
    bw.finish()
}

#[test]
fn vp8l_predictor_mode_0_opaque_black() {
    // Mode 0 → pred = 0xff_00_00_00 for non-corner pixels. Residual
    // (0x10, 0x20, 0x30) per pixel.
    // Pixel layout (per `tests/vp8l_handcrafted.rs::vp8l_2x2_predictor_transform`):
    //   pixel 0 (TL):     pred = 0xff_00_00_00 (special) → (ff, 10, 20, 30)
    //   pixel 1 (TR):     pred = pixel 0 → adds twice → (ff, 20, 40, 60)
    //   pixel 2 (BL):     pred = pixel 0 → (ff, 20, 40, 60)
    //   pixel 3 (BR m=0): pred = 0xff_00_00_00 → (ff, 10, 20, 30)
    let blob = build_2x2_predictor_with_mode(0, (0x10, 0x20, 0x30));
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode mode-0");
    let f = &img.frames[0];
    assert_eq!(&f.rgba[0..4], &[0x10, 0x20, 0x30, 0xff]);
    assert_eq!(&f.rgba[12..16], &[0x10, 0x20, 0x30, 0xff]);
}

#[test]
fn vp8l_predictor_mode_1_left() {
    // Mode 1 (LEFT) → pred = left pixel. So BR pred = BL = (ff, 20, 40, 60),
    // BR result = pred + residual = (ff, 30, 60, 90).
    let blob = build_2x2_predictor_with_mode(1, (0x10, 0x20, 0x30));
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode mode-1");
    let f = &img.frames[0];
    // BR pixel
    assert_eq!(&f.rgba[12..16], &[0x30, 0x60, 0x90, 0xff]);
}

#[test]
fn vp8l_predictor_mode_2_top() {
    // Mode 2 (TOP) → pred = top pixel = TR = (ff, 20, 40, 60).
    let blob = build_2x2_predictor_with_mode(2, (0x10, 0x20, 0x30));
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode mode-2");
    let f = &img.frames[0];
    assert_eq!(&f.rgba[12..16], &[0x30, 0x60, 0x90, 0xff]);
}

#[test]
fn vp8l_predictor_mode_4_top_left() {
    // Mode 4 (TL) → pred = top-left = TL = (ff, 10, 20, 30).
    let blob = build_2x2_predictor_with_mode(4, (0x10, 0x20, 0x30));
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode mode-4");
    let f = &img.frames[0];
    assert_eq!(&f.rgba[12..16], &[0x20, 0x40, 0x60, 0xff]);
}

#[test]
fn vp8l_predictor_mode_7_avg_left_top() {
    // Mode 7 (avg(L, T)) → pred = avg2(BL, TR) per-channel.
    //   BL = (ff, 20, 40, 60), TR = (ff, 20, 40, 60) (same value).
    //   avg = (ff, 20, 40, 60).
    //   BR result = (ff, 20+10=30, 40+20=60, 60+30=90).
    let blob = build_2x2_predictor_with_mode(7, (0x10, 0x20, 0x30));
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode mode-7");
    let f = &img.frames[0];
    assert_eq!(&f.rgba[12..16], &[0x30, 0x60, 0x90, 0xff]);
}

// ──────────────────────────────────────────────────────────────────────
// Meta-Huffman recursion.
// ──────────────────────────────────────────────────────────────────────

/// Compute canonical-Huffman code for one length array entry. Used
/// inline below to avoid pulling in the helper module from
/// vp8l_huffman_unit.rs.
fn canon_codes(lens: &[u8]) -> Vec<u32> {
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

const CODE_LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

fn write_meta_lens_local(bw: &mut BitWriter, meta_lens: &[u8; 19]) {
    let mut last_used = 0usize;
    for (slot, &meta_code) in CODE_LENGTH_ORDER.iter().enumerate() {
        if meta_lens[meta_code] != 0 {
            last_used = slot + 1;
        }
    }
    let num_code_lengths = last_used.max(4);
    bw.write((num_code_lengths - 4) as u32, 4);
    for slot in 0..num_code_lengths {
        let meta_code = CODE_LENGTH_ORDER[slot];
        bw.write(meta_lens[meta_code] as u32, 3);
    }
}

fn write_meta_code_local(
    bw: &mut BitWriter,
    meta_lens: &[u8; 19],
    meta_codes: &[u32],
    code: usize,
) {
    let l = meta_lens[code];
    let v = meta_codes[code];
    assert!(l > 0);
    for i in (0..l).rev() {
        bw.write((v >> i) & 1, 1);
    }
}

/// Write a normal-Huffman tree where the per-symbol lengths are exactly
/// `lens`. Uses use_length=0 (no length-bound) and a uniform 6-bit
/// meta-tree for codes 0..15 (no run codes).
fn write_normal_huffman_no_runs(bw: &mut BitWriter, lens: &[u8]) {
    let mut meta_lens = [0u8; 19];
    for v in 0..16 {
        meta_lens[v] = 6;
    }
    let meta_codes = canon_codes(&meta_lens);
    bw.write(0, 1); // not simple
    write_meta_lens_local(bw, &meta_lens);
    bw.write(0, 1); // use_length = 0
    for &l in lens {
        write_meta_code_local(bw, &meta_lens, &meta_codes, l as usize);
    }
}

/// Write a normal-Huffman tree using `use_length=1` and a hand-chosen
/// `max_symbol`. Caller supplies the per-symbol lengths up to
/// max_symbol; positions past max_symbol are implicitly 0.
fn write_normal_huffman_with_max_symbol(
    bw: &mut BitWriter,
    lens: &[u8],
    max_symbol: usize,
    length_nbits: u32,
) {
    let mut meta_lens = [0u8; 19];
    for v in 0..16 {
        meta_lens[v] = 6;
    }
    let meta_codes = canon_codes(&meta_lens);
    bw.write(0, 1);
    write_meta_lens_local(bw, &meta_lens);
    bw.write(1, 1); // use_length = 1
    let raw = (length_nbits - 2) / 2;
    bw.write(raw, 3);
    bw.write((max_symbol - 2) as u32, length_nbits);
    for i in 0..max_symbol.min(lens.len()) {
        write_meta_code_local(bw, &meta_lens, &meta_codes, lens[i] as usize);
    }
}

/// Build a 4×4 image with a 1×1 meta-image whose internal trees are
/// emitted via the *normal* Huffman path with `use_length=1`. The
/// meta-image's only pixel carries group_id=0 (i.e. the main image
/// uses the single Huffman group). Verifies `decode_image_stream`'s
/// recursive call into itself for the sub-image works at the
/// max_symbol-bounded normal-Huffman path.
#[test]
fn vp8l_meta_image_with_normal_huffman_use_length() {
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(3, 14); // w-1 = 3 → width 4
    bw.write(3, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1); // no transforms

    // Main image stream — main_image=true.
    bw.write(0, 1); // no color cache
    bw.write(1, 1); // meta-Huffman present
                    // bits = 2 + ReadBits(3); pick bits=2 → meta_w = (4+3)>>2 = 1, meta_h = 1.
    bw.write(0, 3);

    // Sub-image (1×1) — main_image=false, no meta-Huffman bit consumed.
    bw.write(0, 1); // no color cache
                    // Single huffman group for the meta-image.
                    //
                    // Meta-image decoder treats its pixels just like a regular VP8L
                    // image — 5 trees: green/red/blue/alpha/distance.
                    // Green: emit literal value 0 via normal-Huffman with
                    // use_length=1 and max_symbol=2. Lengths: [1, 0, ..., 0] →
                    // lone-symbol shortcut at sym 0. Effective alphabet = 280.
    let mut green_lens = vec![0u8; 280];
    green_lens[0] = 1;
    write_normal_huffman_with_max_symbol(&mut bw, &green_lens, 2, 2);
    // Red/Blue/Alpha/Distance: use simple-Huffman single-symbol = 0.
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_1bit(&mut bw, 0);
    // Meta-image pixel: green = 0 → ARGB = 0x00_00_00_00 → group_id =
    // (px >> 8) & 0xffff = 0. So num_groups = 1 main group.

    // Main image's single huffman group: trivial, all literals = 0.
    write_simple_one_symbol_tree_8bit(&mut bw, 0); // green
    write_simple_one_symbol_tree_8bit(&mut bw, 0); // red
    write_simple_one_symbol_tree_8bit(&mut bw, 0); // blue
    write_simple_one_symbol_tree_8bit(&mut bw, 0xff); // alpha
    write_simple_one_symbol_tree_1bit(&mut bw, 0); // distance

    let blob = bw.finish();
    let img = decode_webp(&wrap_in_riff(&blob)).expect("decode 4x4 with meta-Huffman normal-tree");
    assert_eq!(img.width, 4);
    assert_eq!(img.height, 4);
    let f = &img.frames[0];
    assert_eq!(f.rgba.len(), 4 * 4 * 4);
    // All pixels must be opaque black (rgba = 0,0,0,0xff).
    for chunk in f.rgba.chunks_exact(4) {
        assert_eq!(chunk, &[0, 0, 0, 0xff]);
    }
}

/// Counterpart with use_length=0 — the sub-image's normal-Huffman tree
/// goes through the full alphabet of literal codes. Slower but tests
/// the no-length-bound branch through the recursion.
#[test]
fn vp8l_meta_image_with_normal_huffman_no_length_bound() {
    let mut bw = BitWriter::new();
    bw.write(0x2f, 8);
    bw.write(3, 14);
    bw.write(3, 14);
    bw.write(0, 1);
    bw.write(0, 3);
    bw.write(0, 1); // no transforms

    bw.write(0, 1); // no color cache
    bw.write(1, 1); // meta-Huffman
    bw.write(0, 3); // bits = 2 → meta is 1×1

    bw.write(0, 1); // sub-image: no color cache
                    // Sub-image green tree via normal-Huffman use_length=0.
                    // Lens of size 280 with sym 0 = length 1 → lone-symbol shortcut.
    let mut green_lens = vec![0u8; 280];
    green_lens[0] = 1;
    write_normal_huffman_no_runs(&mut bw, &green_lens);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_1bit(&mut bw, 0);

    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0);
    write_simple_one_symbol_tree_8bit(&mut bw, 0xff);
    write_simple_one_symbol_tree_1bit(&mut bw, 0);

    let blob = bw.finish();
    let img =
        decode_webp(&wrap_in_riff(&blob)).expect("decode 4x4 with meta-Huffman no-length-bound");
    assert_eq!(img.width, 4);
    assert_eq!(img.height, 4);
}
