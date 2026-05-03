//! Hand-crafted VP8L bitstream tests.
//!
//! Each test builds a minimal VP8L blob bit-by-bit and decodes it through
//! the public [`oxideav_webp::decode_webp`] entry point (wrapped in a
//! synthetic RIFF/WEBP container) so the bit reader, simple-Huffman path,
//! and at least one transform are all exercised end-to-end.
//!
//! The intent is "no fixture file required": the tests are reproducible
//! from source and double as documentation for the bitstream layout.

use oxideav_webp::decode_webp;

/// LSB-first bit writer matching the VP8L bit reader's convention.
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
        debug_assert!(n <= 32);
        self.cur |= (value & ((1u64 << n) as u32).wrapping_sub(1)) << self.nbits;
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

/// Wrap a VP8L payload in a minimal RIFF/WEBP container so we can drive it
/// through the public `decode_webp` entry point.
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

/// Emit a "simple" Huffman tree with a single 8-bit symbol. Layout
/// (per VP8L spec §6.2.4):
///   bit 0: simple = 1
///   bit 1: num_symbols-1 = 0  (= 1 symbol)
///   bit 2: is_first_8bits = 1
///   bits 3..10: symbol value
fn write_simple_one_symbol_tree(bw: &mut BitWriter, sym: u32) {
    bw.write(1, 1); // simple
    bw.write(0, 1); // num_symbols - 1 = 0 → one symbol
    bw.write(1, 1); // is_first_8bits = 1 → 8-bit symbol
    bw.write(sym & 0xff, 8);
}

/// Build a minimal 2x2 VP8L bitstream where every pixel literally decodes
/// to the same ARGB constant. No transforms, no color cache, no meta-
/// Huffman — just five single-symbol simple Huffman trees and four pixels.
///
/// `(a, r, g, b)` describes the expected per-channel constant.
fn build_constant_2x2_vp8l(a: u8, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // ── Header ─────────────────────────────────────────────────────────
    bw.write(0x2f, 8); // signature
    bw.write(2 - 1, 14); // width-1
    bw.write(2 - 1, 14); // height-1
    bw.write(if a != 0xff { 1 } else { 0 }, 1); // alpha_is_used
    bw.write(0, 3); // version
    bw.write(0, 1); // no transforms

    // ── Main image stream (main_image=true) ───────────────────────────
    bw.write(0, 1); // no color cache
    bw.write(0, 1); // no meta-Huffman image (single group)

    // Five single-symbol trees: green, red, blue, alpha, distance.
    write_simple_one_symbol_tree(&mut bw, g as u32);
    write_simple_one_symbol_tree(&mut bw, r as u32);
    write_simple_one_symbol_tree(&mut bw, b as u32);
    write_simple_one_symbol_tree(&mut bw, a as u32);
    write_simple_one_symbol_tree(&mut bw, 0); // unused distance code

    // No pixel-stream bits needed: every Huffman tree is a single-symbol
    // shortcut, so the decoder consumes zero bits per pixel and walks
    // through 4 literal-green emits to fill the 2x2 image.

    bw.finish()
}

/// Build a 2x2 VP8L bitstream that exercises the *subtract-green*
/// transform. Strategy:
///   * 1 transform: SubtractGreen (no parameters).
///   * Main image residual: every pixel ARGB = (a, r-g mod 256, g, b-g mod 256)
///     using single-symbol simple Huffman trees (so the value is constant
///     for every pixel and no bits are consumed in the pixel loop).
///   * On decode the transform recomputes `r += g; b += g`, restoring
///     `(a, r, g, b)`.
///
/// This proves the transform-parse + reverse-apply pipeline runs end-to-
/// end and that a parameter-less transform composes correctly with the
/// constant-Huffman fast path.
fn build_subtract_green_2x2_vp8l(a: u8, r: u8, g: u8, b: u8) -> Vec<u8> {
    let r_resid = r.wrapping_sub(g);
    let b_resid = b.wrapping_sub(g);
    let mut bw = BitWriter::new();
    // Header.
    bw.write(0x2f, 8);
    bw.write(1, 14); // w-1 = 1
    bw.write(1, 14); // h-1 = 1
    bw.write(if a != 0xff { 1 } else { 0 }, 1); // alpha_is_used
    bw.write(0, 3); // version

    // One transform, type 2 = SubtractGreen.
    bw.write(1, 1); // transform present
    bw.write(2, 2); // type 2
                    // SubtractGreen carries no parameters.
    bw.write(0, 1); // no further transforms

    // ── Main image stream ──────────────────────────────────────────────
    bw.write(0, 1); // no color cache
    bw.write(0, 1); // no meta-Huffman image

    // Five single-symbol trees emitting the residual values.
    write_simple_one_symbol_tree(&mut bw, g as u32);
    write_simple_one_symbol_tree(&mut bw, r_resid as u32);
    write_simple_one_symbol_tree(&mut bw, b_resid as u32);
    write_simple_one_symbol_tree(&mut bw, a as u32);
    write_simple_one_symbol_tree(&mut bw, 0);

    bw.finish()
}

#[test]
fn vp8l_2x2_constant_pixel() {
    // Decode a hand-built 2x2 image where every pixel is ARGB(ff, 80, 40, 20).
    let blob = build_constant_2x2_vp8l(0xff, 0x80, 0x40, 0x20);
    let riff = wrap_in_riff(&blob);
    let img = decode_webp(&riff).expect("decode 2x2 constant VP8L");
    assert_eq!(img.width, 2);
    assert_eq!(img.height, 2);
    assert_eq!(img.frames.len(), 1);
    let f = &img.frames[0];
    assert_eq!(f.rgba.len(), 4 * 4);
    for i in 0..4 {
        let r = f.rgba[i * 4];
        let g = f.rgba[i * 4 + 1];
        let b = f.rgba[i * 4 + 2];
        let a = f.rgba[i * 4 + 3];
        assert_eq!(
            (r, g, b, a),
            (0x80, 0x40, 0x20, 0xff),
            "pixel {i} mismatch: got rgba=({r:#04x}, {g:#04x}, {b:#04x}, {a:#04x})"
        );
    }
}

/// Build a 2x2 VP8L bitstream that exercises the *predictor* transform
/// with mode 0 ("opaque black"). Residual is constant `(0, R, G, B)` so
/// the top-left and bottom-right pixels end up at `(ff, R, G, B)` (their
/// pred is ff_00_00_00 — top-left by spec, bottom-right by mode 0). The
/// two edge pixels get pred = the *previously decoded* pixel and so end
/// up at `(ff, 2R, 2G, 2B)`.
fn build_predictor_2x2_vp8l(r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // Header.
    bw.write(0x2f, 8);
    bw.write(1, 14); // w-1 = 1
    bw.write(1, 14); // h-1 = 1
    bw.write(0, 1); // alpha_is_used
    bw.write(0, 3); // version

    // One transform: type 0 = Predictor, tile_bits = 0+2 = 2.
    bw.write(1, 1); // transform present
    bw.write(0, 2); // type 0 = Predictor
    bw.write(0, 3); // tile_bits raw = 0 → tile_bits = 2 → 1×1 sub-image

    // Predictor sub-image: 1×1, single pixel whose green = mode 0.
    bw.write(0, 1); // no color cache (sub-image)
    write_simple_one_symbol_tree(&mut bw, 0); // green = mode 0
    write_simple_one_symbol_tree(&mut bw, 0);
    write_simple_one_symbol_tree(&mut bw, 0);
    write_simple_one_symbol_tree(&mut bw, 0);
    write_simple_one_symbol_tree(&mut bw, 0);
    // 1×1 sub-image → 1 pixel decoded with single-symbol trees → no extra
    // bits.

    bw.write(0, 1); // no further transforms

    // ── Main image stream ──────────────────────────────────────────────
    bw.write(0, 1); // no color cache
    bw.write(0, 1); // no meta-Huffman image

    // Residuals: (alpha=0, red=R, green=G, blue=B), single-symbol.
    write_simple_one_symbol_tree(&mut bw, g as u32);
    write_simple_one_symbol_tree(&mut bw, r as u32);
    write_simple_one_symbol_tree(&mut bw, b as u32);
    write_simple_one_symbol_tree(&mut bw, 0); // alpha residual
    write_simple_one_symbol_tree(&mut bw, 0); // unused distance

    bw.finish()
}

#[test]
fn vp8l_2x2_predictor_transform() {
    // Decode a 2x2 image whose predictor pipeline yields a known
    // diagonal pattern.
    let blob = build_predictor_2x2_vp8l(0x10, 0x20, 0x30);
    let riff = wrap_in_riff(&blob);
    let img = decode_webp(&riff).expect("decode 2x2 predictor VP8L");
    assert_eq!(img.width, 2);
    assert_eq!(img.height, 2);
    let f = &img.frames[0];
    // Per the analysis in `build_predictor_2x2_vp8l`:
    //   pixel 0 (top-left)     → (ff, R, G, B)        = (ff, 10, 20, 30)
    //   pixel 1 (y=0, x=1)     → (ff, 2R, 2G, 2B)     = (ff, 20, 40, 60)
    //   pixel 2 (y=1, x=0)     → (ff, 2R, 2G, 2B)     = (ff, 20, 40, 60)
    //   pixel 3 (y=1, x=1) m=0 → (ff, R, G, B)        = (ff, 10, 20, 30)
    let want: [(u8, u8, u8, u8); 4] = [
        (0x10, 0x20, 0x30, 0xff),
        (0x20, 0x40, 0x60, 0xff),
        (0x20, 0x40, 0x60, 0xff),
        (0x10, 0x20, 0x30, 0xff),
    ];
    for (i, exp) in want.iter().enumerate() {
        let r = f.rgba[i * 4];
        let g = f.rgba[i * 4 + 1];
        let b = f.rgba[i * 4 + 2];
        let a = f.rgba[i * 4 + 3];
        assert_eq!(
            &(r, g, b, a),
            exp,
            "pixel {i} (predictor): got rgba=({r:#04x}, {g:#04x}, {b:#04x}, {a:#04x})"
        );
    }
}

#[test]
fn vp8l_2x2_subtract_green_transform() {
    // Decode a 2x2 image whose subtract-green pipeline yields
    // ARGB(ff, 0x90, 0x40, 0x60). The encoded residual is
    // (ff, 0x50, 0x40, 0x20); the transform restores the original.
    let blob = build_subtract_green_2x2_vp8l(0xff, 0x90, 0x40, 0x60);
    let riff = wrap_in_riff(&blob);
    let img = decode_webp(&riff).expect("decode 2x2 subtract-green VP8L");
    assert_eq!(img.width, 2);
    assert_eq!(img.height, 2);
    let f = &img.frames[0];
    for i in 0..4 {
        let r = f.rgba[i * 4];
        let g = f.rgba[i * 4 + 1];
        let b = f.rgba[i * 4 + 2];
        let a = f.rgba[i * 4 + 3];
        assert_eq!(
            (r, g, b, a),
            (0x90, 0x40, 0x60, 0xff),
            "pixel {i} (subtract-green): got rgba=({r:#04x}, {g:#04x}, {b:#04x}, {a:#04x})"
        );
    }
}

/// Regression: a libwebp-encoded 32x32 RGBA image whose alpha tree is
/// emitted via the Normal Code Length Code with `use_length=1` and
/// `max_symbol=4`. Earlier the decoder counted only literal codes
/// (0..=15) toward `max_symbol`, letting the run codes (16/17/18) push
/// the loop past the encoder-intended cutoff. The over-emitted lengths
/// produced a Kraft-sum > 1 alphabet that then failed canonical-tree
/// assignment with a misleading "self-collides" error. Fix: count
/// every meta-symbol (including 16/17/18) once, matching the spec's
/// "read up to max_symbol code lengths" wording.
#[test]
fn vp8l_decode_libwebp_normal_huffman_with_max_symbol() {
    // 142-byte fixture produced by `cwebp -lossless` on a 32x32 RGBA
    // PNG (see docs/image/webp/fixtures/lossless-32x32-rgba/).
    const FIXTURE: &[u8] = &[
        0x52, 0x49, 0x46, 0x46, 0x86, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4c, 0x79, 0x00, 0x00, 0x00, 0x2f, 0x1f, 0xc0, 0x07, 0x10, 0xcd, 0x95, 0x21, 0xa2, 0xff,
        0xb1, 0x00, 0x08, 0xc2, 0x7f, 0xb8, 0x86, 0x01, 0x19, 0xf0, 0xfe, 0x07, 0x01, 0xb8, 0x16,
        0x26, 0x8d, 0x24, 0x09, 0x5a, 0x78, 0xfe, 0x2d, 0x3f, 0x6a, 0x07, 0xed, 0x60, 0x48, 0xe3,
        0x8f, 0x90, 0x80, 0xa0, 0xf9, 0xbf, 0x56, 0x24, 0xd2, 0xb6, 0x29, 0x9c, 0xec, 0xcb, 0x3e,
        0x14, 0x12, 0xa4, 0x10, 0x9b, 0x13, 0x70, 0xe8, 0xda, 0x97, 0x61, 0x1f, 0x27, 0xfd, 0x9f,
        0x60, 0x37, 0x51, 0x2e, 0xca, 0x45, 0xb9, 0x28, 0x17, 0xf0, 0xe0, 0xe3, 0x9d, 0x28, 0x17,
        0xe5, 0xa2, 0x5c, 0x94, 0x8b, 0x72, 0x51, 0x2e, 0xca, 0x45, 0xb9, 0x28, 0x17, 0xe5, 0xa2,
        0x5c, 0x94, 0x8b, 0x72, 0x51, 0x2e, 0xca, 0x45, 0xb9, 0x28, 0x17, 0xe5, 0xa2, 0x5c, 0x94,
        0x8b, 0x72, 0x51, 0x2e, 0xca, 0x05, 0x00,
    ];
    let img = decode_webp(FIXTURE).expect("decode 32x32 RGBA libwebp fixture");
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.frames.len(), 1);
    let rgba = &img.frames[0].rgba;
    assert_eq!(rgba.len(), 32 * 32 * 4);
}
