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
# Ok::<(), Box<dyn std::error::Error>>(())
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
| `encode_all` | `fn(&[Frame], &EncodeOptions) -> Result<Vec<u8>, Error>` — the mirror of `decode_all`: a still for one delay-less frame, otherwise an `ANIM` + `ANMF` animation of full-canvas `Rgba` frames — lossless by default (`decode_all(encode_all(f)) == f` for `Rgba` frames with whole-millisecond delays), lossy `VP8 ` (+ `ALPH`) frames with a `quality` |
| `encode_animation` | the WebP-specific depth name under `encode_all` — always an animation (and `encode_animation_frames` for positioned `AnimFrame`s with blend / dispose flags) |
| `read_metadata` | `fn(&[u8]) -> Result<Metadata, Error>` — `ICCP` / `EXIF` / `XMP ` payloads without decoding pixels |
| `WebpImage` | `{ width, height, format: PixelFormat, planes: Vec<Plane>, color: ColorInfo, metadata: Metadata, palette: None }` with `new` / `from_rgb8` / `from_rgba8` / `from_yuv420` (each `Result`, refusing a zero dimension or planes that do not fit the layout with `InvalidData`), `as_bytes` (packed layouts), `into_raw`, `to_rgb8`, `to_rgba8` |
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
// ctx now exposes the "webp" decoder (claiming the `WEBP` FourCC), the
// "webp_vp8l" (lossless) and "webp_vp8" (lossy) encoders, and the "webp"
// container — demuxer, muxer, probe and the `.webp` extension — so the
// framework (and `oxideav-image`) opens and writes WebP files through the
// registry.  Piece-wise: register_codecs(&mut ctx.codecs) /
// register_containers(&mut ctx.containers).
```

The container (`oxideav_webp::container_registry`) declares one video
stream, time base **1/1000 s** (the RFC 9649 §2.7.1.1 `Frame Duration`
unit), `codec_id` `webp_vp8l` for a `VP8L` bitstream and `webp_vp8` for
`VP8 ` (an animation: its first frame's; all three registered decoders
accept either):

| File | Stream | Packets |
|---|---|---|
| Still | `width` / `height` / `pixel_format` / `color_signal` = what `info` reports (`Rgba` sRGB; `Yuv420P` / `Yuva420P` with the §2.5 limited-range BT.601 signal) | one packet holding the whole file, `pts 0` |
| Animation (`VP8X` `A` / `ANIM`) | `Rgba` sRGB, the `VP8X` canvas; `extradata` marks the stream as an animation (`container_registry::is_animation_stream`) | one packet per `ANMF` chunk, each a complete one-frame animated file (`VP8X` + `ANIM` + the original `ANMF`; packet 0 also carries `ICCP` / `EXIF` / `XMP `); `pts` cumulative, `duration` = the frame's `Frame Duration` in ms; only the first is a keyframe |

`Demuxer::metadata()` carries `("loop_count", n)` and
`("background_color", "#RRGGBBAA")` from `ANIM`. The decoder reads the
`extradata` record: on an animation stream it composites every packet's
`ANMF` onto one persistent canvas, so the registry yields the same
frames as `decode_all`; without the record (any other producer of
`webp` packets) every packet is a whole file and an animated one yields
its first composited frame, as before. The muxer writes one packet
verbatim and merges several into one animated file at the chunk level
(no pixel is re-encoded): a demuxed packet contributes its `ANMF` as
is, a still from either encoder becomes a full-canvas `ANMF` with
blending off, packet durations become `Frame Duration`s, `ANIM` comes
from the first packet or from the stream's `loop_count` /
`background_color` options (`0` / `#00000000` by default), and `ICCP` /
`EXIF` / `XMP ` from the first packet. Lossy (`webp_vp8`), lossless and
mixed frames therefore mux as they are (`encode_animation` codes every
frame from RGBA, all lossless or all lossy). `demux(mux(frames)) == frames` for full-canvas `Rgba`
frames; demux → mux → demux of any file reproduces its composited
frames, delays, loop count and background.

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
lossless (compression 1, a headerless VP8L stream, filtered per
§2.7.1.2 at the quick efforts; the raw plane when that is smaller).

## Options

`DecodeOptions` (`Default` + `with_*`; every limit an `Option`, `None` =
unlimited): `max_width` / `max_height` (default `Some(16384)`, the VP8L
/ VP8 per-side ceiling), `max_pixels` (default `Some(16384²)`),
`max_bytes` (default `None`), `strict` (default off — on,
a `VP8X` canvas disagreeing with its bitstream, `VP8X` reserved bits, an
`ALPH` chunk next to `VP8L`, or an `ANMF` rectangle disagreeing with its
frame bitstream are refused), `threads` (default `1`; with two or more
the alpha plane of a lossy picture is decoded next to its `VP8 ` frame;
the framework decoder takes it from the `ExecutionContext`). Limits are
checked against the headers before any pixel buffer is allocated; a hit
is `Error::LimitExceeded`.

`EncodeOptions` (`Default` + `with_*`): `quality: Option<f32>` (`None` =
lossless; `0..=100`, `100` best, mapped to the VP8 qindex and trellis
strength), `method` / `with_effort` (`0..=6`, default `4`: encoder
effort, see the table below), `alpha_method` (lossy only: the alpha
plane's effort when it should differ from `method`), `threads` (default `0` = the host's
parallelism, `1` = serial; never changes the output bytes),
`embed_icc` / `embed_exif` / `embed_xmp` (default on — the image's
metadata is written when present), and for animations `loop_count`
(`0` = forever), `background_rgba`, `frame_mode` (`Auto` / `Delta` /
`Lossless` dirty-rectangle strategy) and `delta`. One struct for stills
and animations; behaviour variants are fields.

| Effort | Lossy (`VP8 `) | Lossless (`VP8L`, `ALPH`, animation frames) |
|---|---|---|
| `0` | prediction modes shortlisted by prediction error, plain coefficient rounding | two transform candidates screened with a reduced-effort parse, one prefix-code group |
| `1` | sub-block modes shortlisted, winners trellis-quantised | four candidates; multi-group entropy layouts tried |
| `2` | every mode compared with a cost model, only winners coded | + the colour-transform candidate |
| `3` | exhaustive rate-distortion mode search | + entropy-driven predictor choices |
| `4` (default) | as `3` | + the winner re-planned with the cost-priced parse |
| `5` | `4` + token probabilities fitted to the picture (second pass) | + more colour-transform candidates |
| `6` | `5` + loop-filter level chosen by reconstruction error | exhaustive transform / colour-cache / parse search (above 2 Mpx: `5`'s screening and a second, deeper parse) |

Every lossless level is exactly lossless; every lossy level writes a
conformant key frame. The alpha plane of a lossy encode is always exact:
at efforts `0..=2` it is coded quickly (the §2.7.1.2 filter with the
cheapest residuals, then one lossless stream), at `3..=6` by the
lossless search of the same level — several times slower than the `VP8 `
encode itself on a large picture, for an `ALPH` chunk that is usually
within a few percent either way. `with_alpha_method(2)` picks the quick
coder for the plane and leaves the picture at its effort. Measured times and sizes: [Performance](#performance).

The `webp_vp8l` encoder takes `Rgba` / `Rgb24` frames; the `webp_vp8`
encoder takes `Yuv420P` (limited-range BT.601, coded as given),
`Yuva420P` (the fourth plane becomes the `ALPH` chunk), and `Rgb24` /
`Rgba` (converted as `encode` does; an `Rgba` frame with a pixel that is
not opaque gets an `ALPH` chunk). Each frame is read in place at its
strides and becomes one `.webp` packet.

The registered encoders take the same controls as options
(`CodecParameters::options`, schemas published through the codec
registry): `webp_vp8` — `effort`, `threads`, `qindex`, `quality`,
`alpha_effort` (`encoder_vp8::WebpVp8EncoderOptions`); `webp_vp8l` — `effort`,
`threads` (`WebpVp8lEncoderOptions`). `threads = 0` follows the
`ExecutionContext` the framework announces (serial when none is).

```rust
# use oxideav_core::{CodecId, CodecParameters, ExecutionContext, RuntimeContext};
# let mut ctx = RuntimeContext::new();
# oxideav_webp::register(&mut ctx);
let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8));
params.width = Some(800);
params.height = Some(533);
params.options = params.options.set("quality", "80").set("effort", "2");
let mut encoder = ctx.codecs.first_encoder(&params)?;
encoder.set_execution_context(&ExecutionContext::with_threads(4));
# Ok::<(), oxideav_core::Error>(())
```

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
* An animation is coded all lossless or all lossy (`quality`); frames
  of both kinds in one file come from the container muxer, which merges
  existing bitstreams. Lossy frames are key frames (`VP8 ` has no
  inter-frame prediction inside WebP); what they save between frames is
  the dirty rectangle.
* Hostile input never panics: every entry point returns `Error`; the
  fuzz targets below cover `probe` / `info` / `decode` / `decode_all` and
  both encode paths.

## WebP specifics

* `read_metadata` reads the metadata chunks without decoding pixels;
  `encode_animation_frames` takes positioned `AnimFrame`s (even `x` / `y`
  offsets, `blend` / `dispose`, per-frame `AnimFrameMode`);
  `encode_vp8l_argb` emits a bare VP8L bitstream with no RIFF wrapper;
  `encoder_vp8::Vp8LossyEncoder` encodes 4:2:0 planes in place (any
  stride) to a lossy `.webp` without the `registry` feature;
  `animation_params` returns the `ANIM` loop count and background.
* The pre-contract surface — `decode_webp` (→ `DecodedWebpFile` with
  `WebpFrame`s), `decode_webp_image` (→ `DecodedWebp`),
  `extract_metadata`, `encode_webp_lossless`, `build_animated_webp` /
  `build_animated_webp_with_options` / `AnimEncoderOptions`,
  `WebpFileMetadata` — is kept for one release as `#[deprecated]` thin
  wrappers over the contract functions.
* The lossless encoder at `method` 6 is a byte-cost super-chooser over
  every §3 / §4 / §3.5 transform candidate, cost-priced LZ77 planning
  and §6.2.2 entropy-image clustering, run on worker threads (byte-
  identical to a one-by-one search); on a 10-image corpus its output is
  smaller than the reference encoder's best effort on 9 of 10 images (up
  to −28%). The default `method` 4 screens a short candidate list with
  one cheap parse each and plans only the winner: an 800 x 533 photo
  encodes in about 0.4 s at 1.2 % more bytes than method 6 (about 4 s);
  a 12 MP photo in 2.4 s (8-9 s on one thread, under 300 MB heap). Above
  2 Mpx method 6 runs the screened search with a second, exhaustive-shape
  parse of the winner (12 MP: 5.9 s). The framework encoder honours the
  `ExecutionContext` thread budget, and the output is byte-identical for
  every budget.
  Every level is re-verified bit-exact through black-box reference
  decodes. See [`BENCHMARKS.md`](./BENCHMARKS.md) for the
  optimisation log.

## Performance

Times in milliseconds for one call through the public entry points, on
a generated photo-like picture (smooth multi-octave noise, three shaded
discs, light sensor noise) at 800 x 533 and 2400 x 1600. "Before" is
0.3.2 plus the commits up to the API-hygiene pass; "after" is this
tree. Both were built in release mode and run on the same 18-core Apple
M-series machine, in alternating passes over the whole matrix; each
figure is the **minimum** over the passes (20 before / 10 after for the
lossy and decode tables, 9 / 4 for the lossless and animation tables,
2 for cells that take two seconds or more). The machine was shared
with other builds, which is why the minimum and not the median is
shown: single-thread timings of the same cell varied up to threefold
between passes. `cargo bench --bench still_matrix` prints these tables
(the "after" half) on your machine.

The `threads` columns are `EncodeOptions::with_threads(1 / 4 / 8)`;
the bytes are the same in every column of a row.

**Lossy encode**, `encode_rgb8`, quality 80. Before, `method` `0..=4`
were one encode and nothing but an alpha plane used a second thread.
The default (`4`) writes the same bytes as before.

| size | effort | before: 1 / 4 / 8 threads ms | bytes | after: 1 / 4 / 8 threads ms | bytes |
|---|---|---:|---:|---:|---:|
| 800x533 | 0 | 46.8 / 46.6 / 47.2 | 28 872 | 11.6 / 3.61 / 2.64 | 33 770 |
| 800x533 | 1 | 46.7 / 46.7 / 46.7 | 28 872 | 21.7 / 6.31 / 4.33 | 30 792 |
| 800x533 | 2 | 46.8 / 46.2 / 46.5 | 28 872 | 26.9 / 7.67 / 5.27 | 30 372 |
| 800x533 | 4 | 46.5 / 46.7 / 46.7 | 28 872 | 45.8 / 12.9 / 8.15 | 28 872 |
| 800x533 | 5 | 93.2 / 93.6 / 93.0 | 27 262 | 92.5 / 26.6 / 16.8 | 27 262 |
| 800x533 | 6 | 151 / 152 / 152 | 27 262 | 151 / 83.7 / 73.7 | 27 262 |
| 2400x1600 | 0 | 400 / 407 / 410 | 237 368 | 99.8 / 29.1 / 19.3 | 280 154 |
| 2400x1600 | 1 | 405 / 400 / 396 | 237 368 | 184 / 54.3 / 32.8 | 254 652 |
| 2400x1600 | 2 | 407 / 402 / 418 | 237 368 | 235 / 64.3 / 40.2 | 251 034 |
| 2400x1600 | 4 | 413 / 406 / 396 | 237 368 | 384 / 101 / 62.1 | 237 368 |
| 2400x1600 | 5 | 801 / 807 / 803 | 222 538 | 780 / 222 / 132 | 222 538 |
| 2400x1600 | 6 | 1283 / 1273 / 1274 | 222 538 | 1276 / 705 / 642 | 222 538 |

Effort `6` spends most of its time in the loop-filter level search,
which runs on one thread.

**Lossy encode with an alpha plane**, `encode_rgba8`, quality 80 (a
soft-edged disc, a quarter of it at half opacity). `4, alpha 2` is the
default picture effort with `with_alpha_method(2)`.

| size | effort | before: 1 / 4 / 8 threads ms | bytes | after: 1 / 4 / 8 threads ms | bytes |
|---|---|---:|---:|---:|---:|
| 800x533 | 0 | 65.4 / 46.7 / 46.3 | 39 112 | 16.6 / 6.74 / 4.73 | 42 728 |
| 800x533 | 2 | 100 / 55.4 / 55.7 | 37 750 | 41.6 / 15.1 / 9.46 | 38 710 |
| 800x533 | 4, alpha 2 | — | — | 59.9 / 25.5 / 14.0 | 37 210 |
| 800x533 | 4 | 213 / 168 / 167 | 36 486 | 199 / 119 / 75.4 | 36 486 |
| 2400x1600 | 0 | 547 / 394 / 401 | 269 132 | 144 / 58.1 / 36.1 | 305 932 |
| 2400x1600 | 2 | 835 / 453 / 451 | 264 790 | 361 / 132 / 73.9 | 274 750 |
| 2400x1600 | 4, alpha 2 | — | — | 523 / 207 / 115 | 261 084 |
| 2400x1600 | 4 | 1719 / 1349 / 1352 | 261 070 | 1567 / 916 / 586 | 261 070 |

**Lossless encode**, `encode_rgb8`. Same bytes as before at every
level; on one thread nothing changed, the gains are stages that now use
the thread budget.

| size | effort | before: 1 / 4 / 8 threads ms | bytes | after: 1 / 4 / 8 threads ms | bytes |
|---|---|---:|---:|---:|---:|
| 800x533 | 0 | 47.9 / 38.7 / 38.3 | 714 474 | 50.7 / 30.6 / 28.4 | 714 474 |
| 800x533 | 2 | 115 / 72.5 / 55.7 | 680 274 | 118 / 73.9 / 52.0 | 680 274 |
| 800x533 | 4 | 217 / 90.9 / 74.9 | 680 274 | 227 / 95.1 / 72.4 | 680 274 |
| 800x533 | 6 | 7940 / 2648 / 1729 | 669 092 | 8433 / 2743 / 1772 | 669 092 |
| 2400x1600 | 0 | 514 / 402 / 431 | 6 415 350 | 477 / 278 / 265 | 6 415 350 |
| 2400x1600 | 2 | 1258 / 800 / 535 | 6 031 110 | 1132 / 689 / 498 | 6 031 110 |
| 2400x1600 | 4 | 2229 / 942 / 748 | 5 994 720 | 2106 / 890 / 662 | 5 994 720 |
| 2400x1600 | 6 | 3162 / 1491 / 1296 | 5 994 720 | 3110 / 1381 / 1143 | 5 994 720 |

**Decode**, `decode_rgba8`, one thread. Each build decodes the file it
wrote (the same bytes for the lossy and lossless rows; the "before"
alpha file has its own `ALPH` chunk).

| size | file | before ms | after ms |
|---|---|---:|---:|
| 800x533 | lossy, quality 80, 28 872 bytes | 3.02 | 2.61 |
| 800x533 | lossy with alpha, quality 80, 38 710 bytes | 4.76 | 4.03 |
| 800x533 | lossless, 680 274 bytes | 8.45 | 5.22 |
| 2400x1600 | lossy, quality 80, 237 368 bytes | 24.7 | 21.3 |
| 2400x1600 | lossy with alpha, quality 80, 274 750 bytes | 39.4 | 33.9 |
| 2400x1600 | lossless, 6 031 110 bytes | 75.7 | 47.8 |

With `DecodeOptions::with_threads(2)` the 2400 x 1600 lossy picture
with alpha decodes to its native planes in 15 ms instead of 28 ms
(measured separately, minimum of nine runs).

**Animation encode**, `encode_animation`, eight frames of the same
picture with a square moving across it. Lossy animations are new.

| size | frames | before: 1 / 4 / 8 threads ms | bytes | after: 1 / 4 / 8 threads ms | bytes |
|---|---|---:|---:|---:|---:|
| 800x533 | lossless, effort 2 | 946 / 612 / 470 | 861 144 | 966 / 271 / 157 | 861 144 |
| 800x533 | lossy, quality 80, effort 4 | — | — | 402 / 111 / 57.3 | 36 672 |
| 800x533 | lossy, quality 80, effort 1 | — | — | 185 / 53.1 / 28.9 | 38 912 |
| 2400x1600 | lossless, effort 2 | 9042 / 6015 / 4663 | 7 626 936 | 9216 / 2984 / 1924 | 7 626 936 |
| 2400x1600 | lossy, quality 80, effort 4 | — | — | 3292 / 880 / 487 | 286 802 |
| 2400x1600 | lossy, quality 80, effort 1 | — | — | 1558 / 429 / 236 | 307 362 |

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

Forty-one [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
targets live under [`fuzz/fuzz_targets/`](./fuzz/fuzz_targets). They
fall into three groups:

* **Framework path** — `demux`: the bytes as a file into the container
  demuxer, every packet through the registered decoder (whole-file or
  per-frame animation mode, under a 1 Mpx `DecoderLimits` budget), then
  the packets back through the muxer.
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
* **Effort and thread oracles** — `encode_effort_threads` (lossless) and
  `lossy_effort_threads` (lossy, with and without alpha) encode a
  fuzz-controlled image at a fuzz-chosen effort once on one thread and
  once on a fuzz-chosen budget: the two files must be identical, the
  lossless one must decode to the input and the lossy one to the input's
  size with its alpha plane exact.
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
