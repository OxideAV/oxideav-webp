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
oxideav-core = "0.0"
oxideav-codec = "0.0"
oxideav-container = "0.0"
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
let pkt = enc.receive_packet()?; // bare VP8L bitstream (no RIFF wrapper)
```

The emitted bytes are a bare VP8L bitstream — decodable directly by
`oxideav_webp::vp8l::decode`. Wrap them in a `RIFF`/`WEBP`/`VP8L`
container if you need an on-disk `.webp` file.

### Encoder — VP8 (lossy, YUV420P in)

```rust
let mut params = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Yuv420P);
let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(yuv420_frame))?;
let pkt = enc.receive_packet()?; // complete `.webp` file, RIFF/WEBP/VP8  wrapped
```

The VP8 lossy encoder writes a complete simple-format `.webp` file
(RIFF header + `WEBP` form + one `VP8 ` chunk holding a single VP8
keyframe) so the output is directly consumable by any WebP reader.
Feeding RGBA into this encoder is rejected with `Error::Unsupported`
— convert to YUV420P first, or use the VP8L path for RGBA.

### Scope

Encoder scope (current):

- VP8L lossless from RGBA, single frame. Emits subtract-green +
  tile-based predictor transforms plus a 256-entry colour cache, which
  gets the ratio close to libwebp on smooth/photographic content.
  Still missing vs libwebp: the colour (G↔R/B decorrelation) transform,
  the palette (colour-indexing) transform, meta-Huffman grouping, and a
  wider predictor-mode pool (the encoder currently picks between modes
  0/1/2/11 per tile rather than probing all 14).
- VP8 lossy from YUV420P, single frame at the `oxideav-vp8` default
  qindex (or a caller-supplied one via `encoder_vp8::make_encoder_with_qindex`).
- No `VP8X` extended header from either encoder. No `ALPH` sidecar on
  VP8 output. No animated (`ANMF`) output.

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
