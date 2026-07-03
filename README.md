# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM +
ANMF). Decoder and encoder are both at production status.

## Capabilities

* Full **decode** of every container variant: simple-lossy (VP8),
  simple-lossless (VP8L), extended (`VP8X`) with `ALPH` alpha plane,
  ICCP / EXIF / XMP metadata, and animated WebP (`ANIM` + `ANMF`) —
  with per-frame `ANMF` bitstreams of both kinds the §2.7.2 rendering
  loop permits (`VP8L` lossless and `VP8 ` lossy, each with an optional
  `ALPH` alpha plane).
* **Encode** of complete `.webp` files in both lossless (VP8L) and
  lossy (VP8) modes, plus complete animated `.webp` files.
* Decoded pixels land in a tightly-packed `Vec<u8>` of
  `width * height * 4` RGBA bytes — drops directly into
  [`image`](https://crates.io/crates/image)'s `ImageBuffer::from_raw`
  with zero copy.
* The full `0.1.2` public surface is reachable both with the default
  `registry` build and under `--no-default-features`.
  [`tests/api_compat_0_1_2.rs`](./tests/api_compat_0_1_2.rs) is the
  compile-only assertion suite pinning every published symbol in place.

The lossless encoder is a byte-cost super-chooser: it builds the §3
no-transform / subtract-green baseline plus every §4 single-transform
and §3.5 stacked-transform candidate (including subtract-green →
predictor and transform-stacked §6.2.2 multi-group main images) —
sweeping `size_bits`, the §5.2.3 color cache, §4.4 palette orderings,
and the §6.2.2 meta-prefix grouping — and emits the byte-shortest
stream, so adding a candidate can never enlarge the output. Three
round-383 mechanisms drive the compression density: run-length
§3.7.2.1.2 code-length tables (codes 16/17/18, chosen per table by
exact bit cost), cost-priced LZ77 token planning (a shortest-path
re-parse against per-symbol Huffman prices, kept only when an exact
writer-cost mirror says it is smaller), and agglomerative
entropy-merge clustering of the §6.2.2 entropy image (per-block symbol
histograms over the five §6.2.3 alphabets, one merge chain
snapshotting every group count). On a 10-image mixed corpus the output
is smaller than the reference encoder's best effort on 7 images (up to
−28%) and within 9% on the rest; every stream is re-verified bit-exact
through a black-box reference decode. The cost models only change
which spec-legal stream is emitted; round-trips stay bit-exact
regardless.

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
| `registry` | ✅ on | Pulls `oxideav-core` plus the framework-trait factories. Cascades into `oxideav-vp8/registry` so the VP8-lossy encode delegation can reach the sibling crate's factories. With this off, lossless encode/decode + animation + metadata extraction all still work; only the VP8-lossy *encode* requires `registry`. |
| `simd` | off (nightly only) | Opt-in `std::simd` acceleration of the hottest pixel-repack / inverse-transform loops. Requires nightly rustc (`#![feature(portable_simd)]`). Byte-identical to the scalar path; see [`BENCHMARKS.md`](./BENCHMARKS.md). |

## Standalone use (no `oxideav-core`)

### Decode any `.webp` file

```rust
use oxideav_webp::{decode_webp, WebpImage};

let webp_bytes: &[u8] = /* file bytes from disk, HTTP, … */;
let image: WebpImage = decode_webp(webp_bytes)?;

println!("{} × {}, {} frame(s)", image.width, image.height, image.frames.len());
for frame in &image.frames {
    // frame.rgba is a tight Vec<u8> of width*height*4 RGBA bytes,
    // row-major, no per-row padding — drops into `image::ImageBuffer`.
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

```rust
use oxideav_webp::encode_webp_lossless;

let rgba: Vec<u8> = /* width*height*4 RGBA bytes */;
let webp_bytes: Vec<u8> = encode_webp_lossless(&rgba, width, height)?;
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
auto-promotes to the extended `VP8X` layout; otherwise it's the simple
lossless layout. For consumers that wrap the bitstream themselves,
`vp8l::encode_vp8l_argb` emits the bare VP8L bitstream with no RIFF
wrap.

### Build an animated `.webp`

```rust
use oxideav_webp::{build_animated_webp, build_animated_webp_with_options,
                   AnimFrame, AnimEncoderOptions};

let frames = vec![
    AnimFrame::new(/* w */ 64, /* h */ 64, /* rgba */ frame0_rgba, /* duration_ms */ 100),
    AnimFrame::new(64, 64, frame1_rgba, 100),
];
// Defaults: per-frame Auto mode (picks byte-smallest of Lossless / Delta).
let webp = build_animated_webp(&frames)?;

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

let enc = make_encoder_with_quality(&params, 75.0)?;
let enc = make_encoder_with_qindex(&params, 32)?;
```

(Lossless encode + decode + animation + metadata extraction all work
without `registry`; only the VP8 *lossy* encode path needs it.)

## Benchmarks

The crate ships a Criterion suite under `benches/` covering the
end-to-end decode / encode / roundtrip paths plus the decoder inverse
transforms (predictor, color, color-indexing, subtract-green), the
encoder forward passes (LZ77 matcher, CTE chooser, meta-prefix
clustering, distance-code lookup), and the entropy / prefix-code chain
(length-then-code build, canonical codes, per-symbol reader). Each
scenario synthesises its fixtures in-process. Numbers, profile
findings, and the optimization log live in
[`BENCHMARKS.md`](./BENCHMARKS.md). Run:

```text
CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
  cargo bench --manifest-path crates/oxideav-webp/Cargo.toml \
    --bench <name> -- --quick
```

## Fuzzing

Over thirty [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
targets live under [`fuzz/fuzz_targets/`](./fuzz/fuzz_targets). They
fall into three groups:

* **Public entry points** — `decode`, `decode_lossless_image`,
  `decode_alpha_plane`, `extract_metadata`, and the differential
  `roundtrip_lossless` / `roundtrip_animated` / `roundtrip_anim_modes`
  / `roundtrip_metadata` oracles that assert the encode→decode contract
  pixel-for-pixel, plus two `ALPH` inverse-filter value oracles:
  `roundtrip_alpha_filter` (forward-filter → method-0 `ALPH` → decode)
  and `roundtrip_alpha_filter_lossless` (forward-filter → residual packed
  into a §3 headerless VP8L green channel → method-1 `ALPH` → decode),
  pinning the §2.7.1.2 reconstructed *values* across all four `F` methods
  and the interior / left-most-column / top-most-row / `(0,0)`-corner
  border cases — the second target additionally exercising the VP8L
  decode → green-extract chain that the method-1 path runs.
* **Standalone parsers** — one target per chunk/header parser
  (`parse_container`, `parse_vp8x`, `parse_vp8_chunk`, `parse_anmf`,
  `parse_anim`, `parse_alph`, `parse_transform_list`,
  `parse_meta_prefix`) cross-checking every decoded field against the
  bytes the parser observed and every error branch against its refusal
  trigger.
* **Inner decode primitives** — `decode_argb`, `decode_lossless`,
  `decode_entropy_image`, `decode_entropy_coded_image`, `prefix_code`,
  `prefix_code_group`, `read_symbol_lut_diff`, `distance_code`,
  `color_cache`, `backward_reference`, `meta_prefix_index`, the
  inverse-transform passes, and `decode_alph`.

Sustained ASan campaigns are crash-free; several targets surfaced (and
the crate fixed) real defenses — eager-allocation OOM bounds on
adversarial canvas / image dimensions, a `BitReader::bits_remaining`
underflow, and a distance-code add-overflow. Run any target with
(nightly + `cargo-fuzz`):

```text
cargo +nightly fuzz run <target> --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
```

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

The fixture corpus at `docs/image/webp/fixtures/` is consumed as opaque
byte streams; end-to-end fixture tests validate against the ARGB pixels
of each fixture's committed `expected.png`, and the §2.7.1 metadata
aux-chunk extraction paths (`ICCP` / `EXIF` / `XMP `) are each
value-validated end-to-end — `extract_metadata` over the
`extended-with-icc-profile` / `extended-with-exif` / `extended-with-xmp`
fixtures must return the exact embedded payload bytes (length +
whole-payload digest + chunk-body cross-check). No third-party codec
library source is consulted.

## License

MIT. See [`LICENSE`](./LICENSE).
