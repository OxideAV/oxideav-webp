# oxideav-webp

[![CI](https://github.com/OxideAV/oxideav-webp/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-webp/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-webp.svg)](https://crates.io/crates/oxideav-webp) [![docs.rs](https://docs.rs/oxideav-webp/badge.svg)](https://docs.rs/oxideav-webp) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust WebP image codec (RFC 9649: RIFF + VP8 + VP8L + VP8X + ALPH +
ANIM + ANMF). Decoder and encoder are both at production status, and the
crate follows the OxideAV **image-crate API contract**
(`IMAGE_CRATE_API.md` in the workspace): the same small standalone
vocabulary every OxideAV image crate exposes, usable without the
framework.

## Standalone use

```toml
[dependencies]
oxideav-webp = { version = "0.2", default-features = false }
```

```rust
let bytes = std::fs::read("in.webp")?;
if oxideav_webp::probe(&bytes) {
    let info = oxideav_webp::info(&bytes)?;         // header only: width, height, format, frames, alpha, icc/exif/xmp
    let img  = oxideav_webp::decode(&bytes)?;       // WebpImage in its native layout
    let rgba: Vec<u8> = img.to_rgba8();             // tightly packed RGBA, 4 × width bytes per row
    let (w, h) = (img.width(), img.height());

    let opts = oxideav_webp::EncodeOptions::default();            // lossless (VP8L)
    let out: Vec<u8> = oxideav_webp::encode_rgba8(w, h, &rgba, &opts)?;
    std::fs::write("out.webp", out)?;

    let lossy = oxideav_webp::EncodeOptions::default().with_quality(80.0); // lossy (VP8 + ALPH)
    let small = oxideav_webp::encode_rgba8(w, h, &rgba, &lossy)?;
}
```

Root items, all available with `default-features = false`:

| Item | Signature |
|---|---|
| `probe` | `fn(&[u8]) -> bool` — `RIFF????WEBP` sniff, never panics |
| `info` | `fn(&[u8]) -> Result<ImageInfo, Error>` — dimensions, native `PixelFormat`, `frames`, `has_alpha`, `color`, `has_icc` / `has_exif` / `has_xmp`, plus `is_animated`, `is_lossy`, `loop_count`, `background_rgba` |
| `decode` / `decode_with` | `fn(&[u8]) -> Result<WebpImage, Error>` / `fn(&[u8], &DecodeOptions) -> …` — the primary image (an animation's first composited frame), native layout, `color` + `metadata` filled |
| `decode_rgb8` / `decode_rgba8` | `fn(&[u8]) -> Result<RgbImage, Error>` / `Result<RgbaImage, Error>` — `{ width, height, data }`, 3 / 4 bytes per pixel, tightly packed |
| `decode_all` / `decode_all_with` | `fn(&[u8]) -> Result<Vec<Frame>, Error>` — every frame of an animation composited onto the canvas per §2.7.1.1, `Frame { image, delay: Option<Duration> }`; a still yields one frame |
| `decode_from` | `fn<R: Read>(R) -> Result<WebpImage, Error>` |
| `encode` | `fn(&WebpImage, &EncodeOptions) -> Result<Vec<u8>, Error>` — writes the image as given; `Error::Unsupported` for a layout WebP cannot carry |
| `encode_rgb8` / `encode_rgba8` | `fn(u32, u32, &[u8], &EncodeOptions) -> Result<Vec<u8>, Error>` |
| `encode_to` | `fn<W: Write>(&WebpImage, &EncodeOptions, W) -> Result<(), Error>` |
| `encode_animation` | `fn(&[Frame], &EncodeOptions) -> Result<Vec<u8>, Error>` — lossless `ANIM` + `ANMF` (and `encode_animation_frames` for positioned `AnimFrame`s with blend / dispose flags) |
| `read_metadata` | `fn(&[u8]) -> Result<Metadata, Error>` — `ICCP` / `EXIF` / `XMP ` payloads without decoding pixels |
| `WebpImage` | `{ width, height, format: PixelFormat, planes: Vec<Plane>, color: ColorInfo, metadata: Metadata, palette: None }` with `new` / `from_rgb8` / `from_rgba8` / `from_yuv420`, `as_bytes` (packed layouts), `into_raw`, `to_rgb8`, `to_rgba8` |
| `PixelFormat` | `= WebpPixelFormat { Rgb24, Rgba, Yuv420P, Yuva420P }` — names mirror `oxideav_core::PixelFormat` |
| `Error` | `= WebpError { InvalidData(String), Unsupported(String), LimitExceeded(String), Io(io::Error), Eof, NeedMore }` |

## Framework use

```toml
[dependencies]
oxideav-webp = "0.2"      # default `registry` feature: pulls oxideav-core
```

```rust
use oxideav_core::RuntimeContext;

let mut ctx = RuntimeContext::new();
oxideav_webp::register(&mut ctx);
// ctx now exposes the "webp" decoder (claiming the `WEBP` FourCC and the
// `.webp` extension) plus the "webp_vp8l" (lossless) and "webp_vp8"
// (lossy) encoders.  Piece-wise: register_codecs(&mut ctx.codecs) /
// register_containers(&mut ctx.containers).
```

The framework `Decoder` emits each still in its **native** layout — one
`Rgba` plane for lossless, three `Yuv420P` planes (four with `ALPH`,
`Yuva420P`) for lossy, with the Rec. 601 limited-range colour signal
attached to the frame — exactly what `decode` returns; `From<WebpImage>
for VideoFrame`, `WebpImage::from_video_frame(&frame, &params)` and
`TryFrom<(&VideoFrame, &CodecParameters)>` convert both ways and
the pixel-format enums map 1:1 by name (`From<WebpPixelFormat> for
oxideav_core::PixelFormat` / `TryFrom` back). `make_decoder` /
`make_encoder` are the direct factories; `encoder_vp8::make_encoder_with_quality`
/ `make_encoder_with_qindex` reach the lossy encoder's quantiser knobs.
The registry path is a thin adapter: one implementation, two entry
styles.

| Feature | Default | What it does |
|---|---|---|
| `registry` | ✅ on | Pulls `oxideav-core` plus the framework-trait factories and `register*`. Cascades into `oxideav-vp8/registry` for the `webp_vp8` framework encoder. Everything in the table above works with it off. |
| `simd` | off (nightly only) | Opt-in `std::simd` acceleration of the hottest pixel-repack / inverse-transform loops. Requires nightly rustc (`#![feature(portable_simd)]`). Byte-identical to the scalar path; see [`BENCHMARKS.md`](./BENCHMARKS.md). |

## Supported layouts

Decode — the native layout `decode` / `info` report:

| File | `PixelFormat` | Planes | `color` |
|---|---|---|---|
| Lossless `VP8L` (simple or `VP8X`), with or without alpha | `Rgba` | 1, stride `4 × width` | sRGB (`1 / 13 / 0`, full range) |
| Lossy `VP8 ` (simple or `VP8X`) | `Yuv420P` | Y `width × height`; Cb, Cr `⌈w/2⌉ × ⌈h/2⌉` | BT.601 limited (`1 / 13 / 6`, limited range) |
| Lossy `VP8 ` + `ALPH` | `Yuva420P` | the three above + alpha `width × height` | BT.601 limited |
| Animation (`ANIM` + `ANMF`, lossless or lossy frames, optional `ALPH`) | `Rgba` per composited frame | 1 | sRGB |

`to_rgb8` / `to_rgba8` are exact integer kernels: a copy for the packed
layouts; for Y′CbCr the limited-range Rec. ITU-R BT.601 inverse
(`R = 1.164384 (Y − 16) + 1.596027 (Cr − 128)`, …, Q16 fixed point,
round-half-up, clamped) with nearest-neighbour 4:2:0 chroma upsampling —
RFC 9649 §2.5 "To convert to RGB, Recommendation 601 SHOULD be used".
Against the reference decoder's non-fancy output every sample is within
±1 (its own fixed-point rounding; see `tests/external_oracle.rs`).

Encode — what `encode` accepts:

| `WebpImage.format` | `EncodeOptions::default()` (lossless) | `.with_quality(q)` (lossy) |
|---|---|---|
| `Rgb24` | `VP8L`, opaque | `VP8 ` (RGB → limited-range BT.601 4:2:0) |
| `Rgba` | `VP8L`, alpha in the bitstream | `VP8 ` + `ALPH` when any pixel is not opaque |
| `Yuv420P` | `Error::Unsupported` (WebP has no lossless Y′CbCr) | `VP8 ` straight through (`color.range` must not be `Full`) |
| `Yuva420P` | `Error::Unsupported` | `VP8 ` + `ALPH` |

Nothing is converted silently: a lossless request for Y′CbCr planes and
a lossy request for full-range planes are refused. Lossless round trips
are exact — `decode(encode(img)) == img` for planes and metadata, pinned
by `tests/contract_api.rs`. The `ALPH` plane of a lossy encode is itself
lossless (method 1, headerless VP8L, falling back to raw when smaller).

## Options

`DecodeOptions` (`Default` + `with_*`; every limit an `Option`, `None` =
unlimited): `max_width` / `max_height` (default `Some(16384)`, the VP8L
/ VP8 per-side ceiling), `max_pixels` (default `Some(16384²)`),
`max_bytes` (default `None`), `strict` (default off — on,
a `VP8X` canvas disagreeing with its bitstream, `VP8X` reserved bits, an
`ALPH` chunk next to `VP8L`, or an `ANMF` rectangle disagreeing with its
frame bitstream are refused). Limits are checked against the headers
before any pixel buffer is allocated; a hit is `Error::LimitExceeded`.

`EncodeOptions` (`Default` + `with_*`): `quality: Option<f32>` (`None` =
lossless; `0..=100`, `100` best, mapped to the VP8 qindex and trellis
strength), `embed_icc` / `embed_exif` / `embed_xmp` (default on — the
image's metadata is written when present), and for animations
`loop_count` (`0` = forever), `background_rgba`, `frame_mode`
(`Auto` / `Delta` / `Lossless` dirty-rectangle strategy) and `delta`.
One struct for stills and animations; behaviour variants are fields.

## Metadata and colour

`WebpImage.metadata` / `ImageInfo.has_*` carry the §2.7.1.4 `ICCP` and
§2.7.1.5 `EXIF` / `XMP ` payloads verbatim (`gamma` is always `None`;
WebP has no such field). `WebpImage.color` is a `ColorInfo { range,
primaries, transfer, matrix }` with H.273 code points: sRGB
(`Full, 1, 13, 0`) for every RGB(A) image — RFC 9649 §2.7.1.4 "If this
chunk is not present, sRGB SHOULD be assumed" — and BT.601 limited
(`Limited, 1, 13, 6`) for the lossy Y′CbCr planes. An embedded ICC
profile is carried, not applied.

## Limits

* Dimensions: 1..=16384 per side (VP8L header), 1..=16383 for a lossy
  `VP8 ` encode (RFC 6386 §9.1 14-bit size words). A `VP8X` canvas
  larger than 16384 per side is refused before allocation (no frame
  could fill it).
* Animation frames are composited onto a full canvas; each `Frame.image`
  is a canvas-sized `Rgba` snapshot (`width × height × 4` bytes per
  frame).
* Lossy animation *encode* is not implemented (`Error::Unsupported`);
  lossy animation *decode* is.
* Hostile input never panics: every entry point returns `Error`; the
  fuzz targets below cover `probe` / `info` / `decode` / `decode_all` and
  both encode paths.

## WebP specifics

* `read_metadata` reads the metadata chunks without decoding pixels;
  `encode_animation_frames` takes positioned `AnimFrame`s (even `x` / `y`
  offsets, `blend` / `dispose`, per-frame `AnimFrameMode`);
  `encode_vp8l_argb` emits a bare VP8L bitstream with no RIFF wrapper;
  `animation_params` returns the `ANIM` loop count and background.
* The pre-contract surface — `decode_webp` (→ `DecodedWebpFile` with
  `WebpFrame`s), `decode_webp_image` (→ `DecodedWebp`),
  `extract_metadata`, `encode_webp_lossless`, `build_animated_webp` /
  `build_animated_webp_with_options` / `AnimEncoderOptions`,
  `WebpFileMetadata` — is kept for one release as `#[deprecated]` thin
  wrappers over the contract functions.
* The lossless encoder is a byte-cost super-chooser over every §3 / §4 /
  §3.5 transform candidate, cost-priced LZ77 planning and §6.2.2
  entropy-image clustering; on a 10-image corpus its output is smaller
  than the reference encoder's best effort on 9 of 10 images (up to
  −28%). Every stream is re-verified bit-exact through a black-box
  reference decode. See [`BENCHMARKS.md`](./BENCHMARKS.md) for the
  optimisation log.

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

Thirty-eight [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
targets live under [`fuzz/fuzz_targets/`](./fuzz/fuzz_targets). They
fall into three groups:

* **Public entry points** — `decode` (the contract `decode` /
  `decode_rgba8` / `decode_all`), `extract_metadata` (`probe` / `info` /
  `read_metadata`), `contract_encode` (lossless-exact + lossy VP8/`ALPH`
  round trips through `encode_rgb8` / `encode_rgba8`),
  `decode_lossless_image`, `decode_alpha_plane`, and the differential
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
  trigger, plus `extract_routing`, which drives the public
  chunk-routing façade (`extract_lossless_chunk` / `extract_lossy_chunk`
  / `read_vp8l_transform_list`) over one buffer and cross-checks it
  against the layers it stitches together (refusal propagation,
  presence routing, §3.4/§9.1 wire-byte re-derivations, and a
  routed-vs-manual §4 read differential).
* **Inner decode primitives** — `decode_argb`, `decode_lossless`,
  `decode_entropy_image`, `decode_entropy_coded_image`, `prefix_code`,
  `prefix_code_group`, `read_symbol_lut_diff`, `distance_code`,
  `color_cache`, `backward_reference`, `meta_prefix_index`, the
  inverse-transform passes, and `decode_alph`.
* **Structure-aware hostility harnesses** — `compose_animation` assembles
  *raw* attacker-controlled RIFF/WEBP/VP8X/ANIM/ANMF containers whose
  §2.7.1.1 `ANMF` header fields (offsets, `Frame W/H`, dispose/blend info
  byte, duration) are free to contradict the valid inner bitstream (per
  frame either a valid `VP8L` stream or a fuzz-mutated §2.5 `VP8 ` key
  frame — the only harness reaching the compositor's lossy sub-chunk leg),
  driving the §2.7.2 compositor's defensive branches — out-of-canvas rect
  rejection + offset overflow, canvas over/under-cover, zero-frame `ANIM`,
  missing `VP8X`, every dispose×blend combination at adversarial placements,
  and a raw-bytes `ALPH` overlay — and asserts the §2.7.1.1 flat-canvas
  carrier invariant, the §2.7.1.1 field carry (durations, loop count,
  background colour) and decode determinism on every survivor.
* **Parameterised encoder oracle** — `encode_params_roundtrip` makes the
  encoder *parameters* fuzz input (caller-fixed §3.4 `alpha_is_used`,
  forced subtract-green, forced §5.2.3 `cache_code_bits ∈ [1, 11]`,
  width-threaded vs width-less literal entries) and asserts the exact
  lossless round trip at every parameter point, not just at the public
  façade's chooser winner.

The `decode` / `decode_still_paths` still-image targets run the §2.5
`VP8 ` lossy legs too (unskipped in round 408 after the sibling
`oxideav-vp8` decoder fixed its §14.4 inverse-DCT overflow on master),
so hostile lossy bitstreams reach the sibling decoder through this
crate's public entry points, seeded from the committed lossy fixtures.

Sustained ASan campaigns are crash-free; several targets surfaced (and
the crate fixed) real defenses — eager-allocation OOM bounds on
adversarial canvas / image dimensions, a `BitReader::bits_remaining`
underflow, a distance-code add-overflow, and (round 408, surfaced by
`encode_params_roundtrip` within minutes of its first run) two
degenerate §3.7.2 prefix-code table shapes the encoder emitted as
Kraft-incomplete, undecodable streams on all-cache-reference token
streams.

Round 432 added a *declared-pixel-load* budget shared by the whole-file
decode harnesses (`fuzz/src/lib.rs`): a spec-legal §5.2.2
backward-reference stream can expand a ~40-byte chunk into ~10^8
decoded pixels (a minimised 38-byte example — now a committed seed —
decodes in ~16 s at ~2.4 GiB peak RSS under ASan), so `decode`,
`decode_still_paths`, `decode_lossless_image`, and `decode_alpha_plane`
skip iterations whose containers *declare* more pixels (across VP8L /
VP8 / VP8X-canvas-×-frames / per-`ANMF` sub-bitstream headers) than a
fuzz iteration can afford; the library itself intentionally accepts
those files (only per-side dimensions are capped). The scheduled Fuzz
workflow also weights its daily per-target slices by measured exec/s —
the encoder-in-loop oracles run four slices each where the
10^4..10^6 exec/s parsers run one (see `.github/workflows/fuzz.yml` for
the measured numbers).

Run any target with (nightly + `cargo-fuzz`):

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
value-validated end-to-end — `read_metadata` over the
`extended-with-icc-profile` / `extended-with-exif` / `extended-with-xmp`
fixtures must return the exact embedded payload bytes (length +
whole-payload digest + chunk-body cross-check). No third-party codec
library source is consulted.

## License

MIT. See [`LICENSE`](./LICENSE).
