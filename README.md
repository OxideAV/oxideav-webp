# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM +
ANMF). Decoder and encoder both at production status as of 2026-05-27.

* Full **decode** of every container variant: simple-lossy (VP8),
  simple-lossless (VP8L), extended (`VP8X`) with `ALPH` alpha plane,
  ICCP / EXIF / XMP metadata, and animated WebP (`ANIM` + `ANMF`).
* **Encode** of complete `.webp` files in both lossless (VP8L) and
  lossy (VP8) modes, plus complete animated `.webp` files.
* Decoded pixels land in a tightly-packed `Vec<u8>` of `width * height
  * 4` RGBA bytes — drops directly into [`image`](https://crates.io/crates/image)'s
  `ImageBuffer::from_raw` with zero copy.
* The full crates.io `0.1.2` public surface is reachable, both with
  the default `registry` build and under `--no-default-features`.
  [`tests/api_compat_0_1_2.rs`](./tests/api_compat_0_1_2.rs) is the
  29-test compile-only assertion suite that pins every published
  symbol in place.

## Install

```toml
# Standalone — flat RGBA in / flat RGBA out, no framework dep:
[dependencies]
oxideav-webp = { version = "0.1", default-features = false }

# With the OxideAV runtime:
[dependencies]
oxideav-webp = "0.1"
```

| Feature | Default | What it does |
|---|---|---|
| `registry` | ✅ on | Pulls `oxideav-core` plus the framework-trait factories. Cascades into `oxideav-vp8/registry` so the VP8-lossy encode delegation can reach the sibling crate's factories. With this off, **lossless encode/decode + animation + metadata extraction all still work**; only the VP8-lossy *encode* requires `registry`. |
| `simd` | off (nightly only) | Opt-in `std::simd` acceleration of the hottest pixel-repack loop (`Vp8lImage::to_rgba`). Requires a nightly rustc because it activates `#![feature(portable_simd)]`. Byte-identical to the scalar path (asserted by `vp8l::tests::to_rgba_simd_matches_scalar_byte_for_byte`); see [`BENCHMARKS.md`](./BENCHMARKS.md) for the round-170 before/after numbers. |

### Benchmarks

The crate ships sixteen criterion benches under `benches/`: the
original four end-to-end / hot-loop targets (`lossless_decode`,
`lossless_encode`, `lz77_match`, `argb_to_rgba`); the four §4.x
decoder-side inverse-transform per-pass benches added across rounds
194 / 207 / 210 / 217 (`inverse_predictor`, `inverse_color`,
`inverse_color_indexing`, `inverse_subtract_green`); the round-249
§4.4 palette subtraction-decode bench (`inverse_color_table`); the
two encoder-side §4.1 / §4.3 forward-transform per-pass benches
added in rounds 224 / 248 (`predictor_subtract`,
`apply_subtract_green`); the two encoder-side §3.7.2 per-pass
benches added in rounds 250 / 251 (`build_code_lengths`,
`canonical_codes`) covering each §3.7.1 prefix-code-group alphabet
(distance-40, literal-256, green-281, green-2328) in both dense and
sparse frequency regimes; the round-252 decoder-side §3.6.2.2 LZ77
prefix-code-to-value per-call bench (`read_lz77_value`) covering
the fast-path / short-extra / long-extra / max-extra regimes from
§3.6.2.2 Table 4; the round-253 decoder-side §3.6.2.3 color-
cache hash slot-index per-call bench (`color_cache_hash`) covering
the §3.6.2.3 `code_bits` allowed range `[1..11]` at four
representative points (1 / 4 / 8 / 11); and the round-254 encoder-
side §5.2.2 LZ77 value-to-prefix-split per-call bench
(`value_to_prefix`, mirror of the round-252 decoder-side cell
layout) covering the same fast-path / short-extra / long-extra /
max-extra regimes at value-side samples (3 / 40 / 40_000 / 900_000).
Numbers, profile findings, and the
optimization log live in [`BENCHMARKS.md`](./BENCHMARKS.md). Run
with:

```text
CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
  cargo bench --manifest-path crates/oxideav-webp/Cargo.toml \
    --bench <name> -- --quick
```

### Fuzzing

Seven [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
targets live under [`fuzz/fuzz_targets/`](./fuzz/fuzz_targets):
`decode` and `extract_metadata` feed arbitrary bytes through the two
public single-shot entry points; `roundtrip_lossless` synthesises a
≤64 × 64 RGBA tile from fuzz-controlled bytes and asserts the §3
lossless contract pixel-for-pixel across `encode_webp_lossless` →
`decode_webp`; `roundtrip_animated` (round 238) widens the same
contract across the §2.7.1.1 animation carrier — a fuzz-controlled
1..8-frame animation (canvas ≤ 32 × 32) goes through
`build_animated_webp` → `decode_webp` and the frame count + per-frame
width/height + per-frame `duration_ms` + per-frame RGBA bytes are
asserted byte-identical; `decode_alph` (round 255) drives the
§2.7.1.2 ALPH standalone entry point `alph::decode_alpha` directly
across the four filter methods (none / horizontal / vertical /
gradient) and the two compression methods (raw + headerless §3 VP8L)
with `plane.len() == width * height` asserted on success;
`parse_vp8x` (round 256) drives the §2.7.1 VP8X chunk parser
standalone entry point `vp8x::Vp8xHeader::parse` directly across the
full §2.7.1 Figure 7 flag-octet / reserved-field / canvas-dimension
cross-product with every successfully-decoded field cross-checked
against the input bytes the parser observed and every error branch
cross-checked against the §2.7.1 refusal triggers; `parse_anmf`
(round 257) drives the §2.7.1.1 ANMF chunk header parser standalone
entry point `anmf::AnmfHeader::parse` directly across the full
§2.7.1.1 Figure 9 5 × uint24 + info-byte cross-product (Frame X * 2
doubling, Frame W/H Minus One + 1 resolution, uint24 LE duration,
info-byte Reserved / B / D extraction at bits 7..2 / 1 / 0) with
every successfully-decoded field cross-checked against the input
bytes the parser observed and the `PayloadTooShort` branch
cross-checked against the §2.7.1.1 16-byte minimum. Run any one
with (nightly + `cargo-fuzz` installed):

```text
cargo +nightly fuzz run decode               --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run extract_metadata     --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run roundtrip_lossless   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run roundtrip_animated   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_alph          --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_vp8x           --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_anmf           --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
```

## Standalone use (no `oxideav-core`)

### Decode any `.webp` file

```rust
use oxideav_webp::{decode_webp, WebpImage};

let webp_bytes: &[u8] = /* file bytes from disk, HTTP, … */;
let image: WebpImage = decode_webp(webp_bytes)?;

println!("{} × {}, {} frame(s)", image.width, image.height, image.frames.len());
for frame in &image.frames {
    // frame.rgba is a tight Vec<u8> of width*height*4 RGBA bytes,
    // row-major, no per-row padding — drops into `image::ImageBuffer`:
    //
    //   let img = image::RgbaImage::from_raw(frame.width, frame.height,
    //                                        frame.rgba.clone()).unwrap();
    //
    println!("  frame: {}×{}, {} ms", frame.width, frame.height, frame.duration_ms);
}

// ICC / EXIF / XMP are on image.metadata.{icc, exif, xmp} (each Option<Vec<u8>>).
```

### Read metadata only (no pixel decode)

```rust
use oxideav_webp::extract_metadata;

let meta = extract_metadata(webp_bytes)?;
if let Some(icc) = meta.icc.as_deref()  { /* color-management profile */ }
if let Some(exif) = meta.exif.as_deref() { /* EXIF blob */ }
if let Some(xmp) = meta.xmp.as_deref()   { /* XMP UTF-8 XML */ }
```

### Encode a lossless `.webp` from RGBA bytes

The shortest path — flat RGBA in, complete `.webp` file out:

```rust
use oxideav_webp::encode_webp_lossless;

let rgba: Vec<u8> = /* width*height*4 RGBA bytes */;
let webp_bytes: Vec<u8> = encode_webp_lossless(&rgba, width, height)?;
// Write to disk:
std::fs::write("out.webp", &webp_bytes)?;
```

### Encode lossless with metadata (ICC / EXIF / XMP)

```rust
use oxideav_webp::{encode_vp8l_argb_with_metadata, WebpMetadata};

// VP8L works in ARGB, one u32/pixel.
let argb: Vec<u32> = /* width*height ARGB pixels */;

let meta = WebpMetadata {
    icc:  Some(&my_icc_profile),
    exif: Some(&my_exif_blob),
    xmp:  Some(&my_xmp_xml),
};
let webp_bytes = encode_vp8l_argb_with_metadata(
    width, height, &argb, /* has_alpha = */ true, &meta,
)?;
```

If `has_alpha` is `true` or any metadata field is set, the output
auto-promotes to the extended `VP8X` layout; otherwise it's the
simple lossless layout.

### Bare VP8L bitstream (no RIFF wrap)

For consumers that wrap the bitstream themselves:

```rust
use oxideav_webp::vp8l::encode_vp8l_argb;
let vp8l: Vec<u8> = encode_vp8l_argb(&argb, width, height)?;
```

### Build an animated `.webp`

```rust
use oxideav_webp::{build_animated_webp, build_animated_webp_with_options,
                   AnimFrame, AnimEncoderOptions};

// Each AnimFrame is a tile (width × height RGBA) at (x, y) on the
// canvas, with a duration in milliseconds.
let frames = vec![
    AnimFrame::new(/* w */ 64, /* h */ 64, /* rgba */ frame0_rgba, /* duration_ms */ 100),
    AnimFrame::new(64, 64, frame1_rgba, 100),
    AnimFrame::new(64, 64, frame2_rgba, 100),
];

// Defaults: per-frame Auto mode (picks byte-smallest of Lossless / Delta).
let webp = build_animated_webp(&frames)?;

// Or with options (loop count, background colour, file-level metadata):
let opts = AnimEncoderOptions {
    loop_count: 0,                      // 0 = infinite
    background_rgba: [0xff, 0xff, 0xff, 0xff],
    ..Default::default()
};
let webp = build_animated_webp_with_options(&frames, &opts)?;
```

## With the OxideAV runtime (`registry` feature on)

```rust
use oxideav_core::RuntimeContext;
use oxideav_webp::{CODEC_ID_VP8, CODEC_ID_VP8L};   // "webp_vp8" / "webp_vp8l"

let mut ctx = RuntimeContext::new();
oxideav_webp::register(&mut ctx);
// ctx now exposes the "webp" container plus "webp_vp8" + "webp_vp8l" codecs.
```

This is the only way to reach the **VP8-lossy encoder** — it delegates
to the `oxideav-vp8` sibling crate's framework factory family:

```rust
use oxideav_webp::encoder_vp8::{make_encoder_with_quality, make_encoder_with_qindex};

// Returns Box<dyn oxideav_core::Encoder>; emits RIFF/WEBP-wrapped output.
let enc = make_encoder_with_quality(&params, 75.0)?;
let enc = make_encoder_with_qindex(&params, 32)?;
```

(Lossless encode + decode + animation + metadata extraction all work
without `registry`; only the VP8 *lossy* encode path needs it.)

## Clean-room sources

Implementation is derived entirely from the public format specs:

* **RFC 9649** — WebP Image Format
  (`docs/image/webp/rfc9649-webp.txt`, also `rfc9649-webp.pdf`).
* **WebP Lossless Bitstream Specification** — the LZ77 + prefix-coded
  literals + color cache + spatial / color / color-indexing transforms
  (also reproduced in RFC 9649 §3).
* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`) for the VP8 lossy
  framing routed through the `oxideav-vp8` sibling.

The 18-fixture corpus at `docs/image/webp/fixtures/` is consumed as
opaque byte streams; end-to-end fixture tests validate against the
ARGB pixels of each fixture's committed `expected.png`. No third-party
codec library source is consulted.

## License

MIT. See [`LICENSE`](./LICENSE).
