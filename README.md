# oxideav-webp

Pure-Rust **WebP** image codec and container — RIFF/WEBP simple
(lossy `VP8 ` + lossless `VP8L`) + extended (`VP8X` with `ALPH`, `ICCP`,
`EXIF`, `XMP `) + animated (`ANIM`/`ANMF`) decode, plus single-frame
encode on both the VP8 lossy and VP8L lossless paths. Zero C
dependencies.

VP8 lossy decoding and encoding both go through
[`oxideav-vp8`](https://crates.io/crates/oxideav-vp8) (also pure-Rust).
VP8L lossless is a self-contained implementation of Google's Huffman +
LZ77 + colour-cache + four-transform bitstream in this crate.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-container = "0.1"
oxideav-webp = "0.0"
```

## Quick use

`.webp` files carry one or many frames, so the typical path is: open
the file as a container, pull packets, decode them. Output is always
`PixelFormat::Rgba` regardless of whether the source chunk was `VP8 `
(lossy YUV → RGB) or `VP8L` (native RGBA).

```rust
use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::Frame;

let mut codecs = CodecRegistry::new();
let mut containers = ContainerRegistry::new();
oxideav_webp::register(&mut codecs, &mut containers);

let input: Box<dyn oxideav_container::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("image.webp")?),
);
let mut dmx = containers.open("webp", input)?;
let stream = &dmx.streams()[0];
let mut dec = oxideav_webp::decoder::WebpDecoder::new(
    stream.params.width.unwrap_or(0),
    stream.params.height.unwrap_or(0),
);

loop {
    match dmx.next_packet() {
        Ok(pkt) => {
            oxideav_codec::Decoder::send_packet(&mut dec, &pkt)?;
            while let Ok(Frame::Video(vf)) = oxideav_codec::Decoder::receive_frame(&mut dec) {
                // vf.format == PixelFormat::Rgba
                // vf.planes[0].data holds width*height*4 bytes, row-major.
            }
        }
        Err(oxideav_core::Error::Eof) => break,
        Err(e) => return Err(e.into()),
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a one-shot decode of an in-memory buffer, skip the registry dance:

```rust
let bytes = std::fs::read("image.webp")?;
let img = oxideav_webp::decode_webp(&bytes)?;
for frame in &img.frames {
    // frame.rgba has width*height*4 bytes; frame.duration_ms is the
    // ANMF per-frame delay (0 for still images).
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Encoder — VP8L (lossless, RGBA / RGB in)

```rust
use oxideav_core::{CodecId, CodecParameters, PixelFormat};

let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8L));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Rgba); // or PixelFormat::Rgb24
let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(frame))?;
let pkt = enc.receive_packet()?; // complete .webp file
```

The registered `webp_vp8l` encoder accepts two input pixel formats:

* **`Rgba`** — the historical default. Fully-opaque frames use the
  simple `RIFF/WEBP/VP8L` layout; frames with any transparent pixel
  switch to the extended `RIFF/WEBP/VP8X + VP8L` layout so the VP8X
  header advertises the alpha flag (required by any spec-compliant
  reader).
* **`Rgb24`** — three bytes per pixel, no alpha. Useful when the
  upstream is a JPEG decode or a PNG-without-alpha decode (the common
  case on the [`image`](https://crates.io/crates/image) crate side).
  The conversion to the encoder's internal ARGB pixel buffer
  **streams** through the input three bytes at a time — no
  intermediate `Rgba` byte buffer is materialised, so re-encoding an
  RGB image to WebP costs only the encoder's own working memory, not
  a full 4-byte expansion. Always emits the simple layout (Rgb24 is
  implicitly opaque). Closes [#7][issue-7].

[issue-7]: https://github.com/OxideAV/oxideav-webp/issues/7

If you need a bare VP8L bitstream (for embedding in another container,
say), call `oxideav_webp::encode_vp8l_argb` directly — that entry
point still returns the header-to-data bytes with no RIFF wrapper.

### Encoder — VP8 (lossy)

```rust
let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8));
params.width = Some(w);
params.height = Some(h);
// One of: Yuv420P, Yuva420P, Rgba, Rgb24
params.pixel_format = Some(PixelFormat::Yuv420P);
let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(frame))?;
let pkt = enc.receive_packet()?; // complete .webp file
```

Four input pixel formats are accepted:

* **`Yuv420P`** — the native VP8 input. Emits the simple
  `RIFF/WEBP/VP8 ` layout.
* **`Yuva420P`** — Yuv420P with a side full-resolution alpha plane.
  The YUV planes feed straight into the keyframe (no RGB roundtrip)
  and the alpha plane goes straight into a VP8L-compressed `ALPH`
  sidecar. Emits the extended `RIFF/WEBP/VP8X + ALPH + VP8 ` layout
  with the VP8X `ALPHA` flag set.
* **`Rgba`** — converts RGB to YUV 4:2:0 (BT.601 limited range) for
  the VP8 keyframe and compresses the alpha plane into an `ALPH`
  sidecar chunk. Emits the extended `RIFF/WEBP/VP8X + ALPH + VP8 `
  layout with the VP8X `ALPHA` flag set.
* **`Rgb24`** — RGB without alpha. Streams the RGB→YUV conversion
  three bytes at a time without ever building a Rgba byte buffer
  (issue #7), and emits the simple `RIFF/WEBP/VP8 ` layout.

`Yuva420P` is the natural input shape if you already have a
YUV-with-alpha frame from a video decoder. It avoids the YUV→RGB→YUV
roundtrip the `Rgba` path goes through.

Quality control: the VP8 lossy encoder exposes two equivalent factory
entry points for picking a target compression level —

* `encoder_vp8::make_encoder_with_quality(&params, quality)` — takes
  a libwebp-style `quality: f32` in `0.0..=100.0` (higher = better
  quality / larger file; the libwebp default is `75.0`).
* `encoder_vp8::make_encoder_with_qindex(&params, qindex)` — takes
  the underlying VP8 qindex in `0..=127` (lower = better) for callers
  that already speak the libvpx scale.

The `quality → qindex` mapping is the linear inversion
`qindex = round((100 - quality) * 1.27)`, so the API surface lines up
with libwebp but the **perceptual** behaviour does not: libwebp also
adjusts its quantizer matrices, AC/DC deltas, and segment-level QP
based on quality, none of which we currently do. Round-2 work would
tune the quantizer matrix and segment QPs to track libwebp's
perceptual targets at matching `quality` values.

### Scope

Encoder scope (current):

- VP8L lossless from `Rgba` or `Rgb24` (single frame). Emits subtract-green +
  colour (G↔R/B decorrelation) + tile-based predictor transforms plus
  a tunable colour cache, which puts the ratio in the neighbourhood
  of libwebp on smooth/photographic content. The default `encode_vp8l_argb`
  entry point now runs a 32-trial RDO sweep over every combination of
  the four optional transforms × four colour-cache widths
  ({off, 6, 8, 10} bits) and keeps the smallest encoded variant —
  callers that want a fixed configuration can still call
  `encode_vp8l_argb_with` directly. Still missing vs libwebp:
  the palette (colour-indexing) transform, meta-Huffman grouping, and
  a wider predictor-mode pool (the encoder currently picks between
  modes 0/1/2/11 per tile rather than probing all 14).
- VP8 lossy from `Yuv420P`, `Yuva420P`, `Rgba`, or `Rgb24` (single
  frame). For `Yuva420P` and `Rgba` the alpha plane is emitted as a
  VP8L-compressed `ALPH` chunk inside the extended (`VP8X`)
  container; `Yuva420P` skips the YUV→RGB→YUV roundtrip the `Rgba`
  path forces. `Rgb24` streams the RGB→YUV conversion without a
  Rgba alloc (issue #7). Default qindex from `oxideav-vp8` is used
  unless the caller selects a specific one via
  `encoder_vp8::make_encoder_with_qindex` (VP8 qindex `0..=127`,
  lower = better) or the libwebp-style
  `encoder_vp8::make_encoder_with_quality` (`0.0..=100.0`,
  higher = better) — see the encoder section above for the mapping
  caveat.
- `VP8X` extended header is emitted automatically whenever the output
  carries an `ALPH` sidecar or optional ICC / EXIF / XMP metadata via
  the `riff::WebpMetadata` helper.
- Animated WebP encode via [`build_animated_webp`] — emits a
  `VP8X + ANIM + ANMF...ANMF` file from a slice of `AnimFrame`s with
  per-frame durations, x/y offsets, blend, and disposal flags. Each
  ANMF wraps a `VP8L` lossless sub-chunk (mixed lossy + lossless
  animations are not yet produced; the decoder accepts both shapes).

Decoder scope:

- `VP8 ` simple lossy (through `oxideav-vp8`), `VP8L` lossless,
  `VP8X` extended with `ALPH` (raw / filtered / VP8L-compressed alpha
  plane), and `ANIM`/`ANMF` animation with per-frame disposal + blend
  modes composited onto an internal RGBA canvas.
- `ICCP` / `EXIF` / `XMP ` chunks are surfaced on
  `WebpImage::metadata` (a `WebpFileMetadata` struct with optional
  `icc` / `exif` / `xmp` byte vectors). For metadata-only access
  without decoding any pixels, call `oxideav_webp::extract_metadata`
  on the file bytes directly.
- Default output pixel format is `Rgba`. For single-frame VP8+ALPH
  input, `WebpDecoder::new_yuva420p(w, h)` (or
  `set_prefer_yuva420p(true)` after construction) flips the output
  to a 4-plane `Yuva420P` frame — skipping the YUV→RGB conversion
  + alpha overlay the default path runs. VP8L and animated files
  always stay on the RGBA path (cross-frame composite needs a
  unified pixel format).

### Codec / container IDs

- Codec: `"webp_vp8l"` (VP8L encoder + standalone VP8L decoder);
  accepted input pixel formats `Rgba`, `Rgb24`. Decoded output is
  always `Rgba`.
- Codec: `"webp_vp8"` (VP8 lossy encoder path); accepted input pixel
  formats `Yuv420P`, `Yuva420P`, `Rgba`, `Rgb24`.
- Container: `"webp"`, matches `.webp` by extension + `RIFF`/`WEBP` magic.

Single-image WebPs decode to one `VideoFrame`; animated WebPs produce
N frames with PTS in milliseconds (the ANMF native unit).

For VP8+ALPH inputs the `WebpDecoder` defaults to `Rgba` output (the
historical behaviour). To opt into a 4-plane `Yuva420P` frame
straight from the VP8 + ALPH decoders — skipping the YUV→RGB
conversion + alpha overlay — construct the decoder with
`WebpDecoder::new_yuva420p(w, h)` (or set
`set_prefer_yuva420p(true)` after construction). VP8L and animated
files always go through the RGBA canvas because cross-frame
disposal/blend semantics need a unified pixel format.

#### `image` crate interop

If you already have a `RgbImage` (`image::ImageBuffer<Rgb<u8>, _>`)
from a JPEG decode or a PNG-without-alpha decode, you can feed its
backing `Vec<u8>` straight to the `webp_vp8l` or `webp_vp8` encoder
with `pixel_format = Some(PixelFormat::Rgb24)` and a single-plane
`VideoFrame { stride: w * 3, data: ... }` — no Rgba allocation
required.

## License

MIT — see [LICENSE](LICENSE).
