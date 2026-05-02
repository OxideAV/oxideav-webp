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

### Encoder — VP8L (lossless, RGBA in)

```rust
use oxideav_core::{CodecId, CodecParameters, PixelFormat};

let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8L));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Rgba);
let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(rgba_frame))?;
let pkt = enc.receive_packet()?; // complete .webp file
```

The registered `webp_vp8l` encoder returns a complete RIFF-wrapped
`.webp` file. Fully-opaque frames use the simple `RIFF/WEBP/VP8L`
layout; frames with any transparent pixel switch to the extended
`RIFF/WEBP/VP8X + VP8L` layout so the VP8X header advertises the
alpha flag (required by any spec-compliant reader).

If you need a bare VP8L bitstream (for embedding in another container,
say), call `oxideav_webp::encode_vp8l_argb` directly — that entry
point still returns the header-to-data bytes with no RIFF wrapper.

### Encoder — VP8 (lossy)

```rust
let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Yuv420P); // or PixelFormat::Rgba
let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(frame))?;
let pkt = enc.receive_packet()?; // complete .webp file
```

Two input pixel formats are accepted:

* **`Yuv420P`** — emits the simple `RIFF/WEBP/VP8 ` layout.
* **`Rgba`** — converts RGB to YUV 4:2:0 (BT.601 limited range) for
  the VP8 keyframe and compresses the alpha plane into an `ALPH`
  sidecar chunk. Emits the extended `RIFF/WEBP/VP8X + ALPH + VP8 `
  layout with the VP8X `ALPHA` flag set.

### Scope

Encoder scope (current):

- VP8L lossless from RGBA, single frame. Emits subtract-green +
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
- VP8 lossy from YUV420P or RGBA, single frame. When given RGBA the
  alpha plane is emitted as a VP8L-compressed `ALPH` chunk inside the
  extended (`VP8X`) container. Default qindex from `oxideav-vp8` is
  used unless the caller selects a specific one via
  `encoder_vp8::make_encoder_with_qindex`.
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
- `ICCP` / `EXIF` / `XMP ` chunks are recognised and skipped; their
  payloads are not surfaced on the public API.

### Codec / container IDs

- Codec: `"webp_vp8l"` (VP8L encoder + standalone VP8L decoder);
  accepted pixel format `Rgba`.
- Codec: `"webp_vp8"` (VP8 lossy encoder path); accepted pixel format
  `Yuv420P`.
- Container: `"webp"`, matches `.webp` by extension + `RIFF`/`WEBP` magic.

Single-image WebPs decode to one `VideoFrame`; animated WebPs produce
N frames with PTS in milliseconds (the ANMF native unit).

## License

MIT — see [LICENSE](LICENSE).
