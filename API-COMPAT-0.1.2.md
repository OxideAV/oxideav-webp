# Public API compatibility target — crates.io `oxideav-webp 0.1.2`

This file pins the **minimum** public surface that any current release of
`oxideav-webp` must keep exposing so that historical consumers pinned to
`oxideav-webp = "0.1"` can upgrade transparently. It is the strict
contract; [API-COMPAT.md](./API-COMPAT.md) is the broader aspirational
target distilled from the last pre-orphan release (0.1.5). When the two
disagree on a name or shape, **0.1.2 wins** for backward compatibility.

> Source of every signature below: the per-version rustdoc at
> `https://docs.rs/oxideav-webp/0.1.2/`. Recovered field-by-field from
> module / type / function index pages. Not from the removed `src/`,
> not from any reference encoder source.

## Cargo features (verbatim from the 0.1.2 manifest)

```toml
[features]
default = ["registry"]
registry = ["dep:oxideav-core", "oxideav-vp8/registry"]
```

- `default-features = false` ⇒ `oxideav-core` is **not** linked, and
  `oxideav-vp8` is pulled with its `registry` feature off (cascade).
- This is the **no-oxideav-core standalone build**. Historical users
  who set `default-features = false` to embed the crate in a non-
  framework image pipeline MUST keep building after the upgrade.

## Public modules (must exist; may add more, must not rename)

```
oxideav_webp::decoder
oxideav_webp::demux
oxideav_webp::encoder
oxideav_webp::encoder_anim
oxideav_webp::encoder_vp8
oxideav_webp::error
oxideav_webp::registry         (registry feature ON only)
oxideav_webp::riff
oxideav_webp::vp8l
oxideav_webp::vp8l::bit_reader
oxideav_webp::vp8l::encoder
oxideav_webp::vp8l::huffman
oxideav_webp::vp8l::transform
```

## Crate-root re-exports (must be reachable at `oxideav_webp::<name>`)

```rust
pub use decoder::{decode_webp, WebpDecoder, WebpFrame, WebpImage};
pub use demux::{extract_metadata, WebpFileMetadata};
pub use encoder_anim::{
    build_animated_webp, build_animated_webp_with_options,
    AnimEncoderOptions, AnimFrame, AnimFrameMode,
};
pub use error::{Result, WebpError};
pub use vp8l::encode_vp8l_argb;

#[cfg(feature = "registry")]
pub use registry::{register, register_codecs, register_containers};

pub const CODEC_ID_VP8:  &str = "webp_vp8";
pub const CODEC_ID_VP8L: &str = "webp_vp8l";
```

## Type / function shapes (verbatim from rustdoc 0.1.2)

### `decoder`

```rust
pub fn decode_webp(buf: &[u8]) -> Result<WebpImage>;
pub fn make_vp8l_decoder(/* … framework-internal … */);

#[derive(Clone, Debug)]
pub struct WebpFrame {
    pub width: u32,
    pub height: u32,
    pub duration_ms: u32,
    pub rgba: Vec<u8>,        // len = width * height * 4, RGBA8 row-major
}

#[derive(Clone, Debug)]
pub struct WebpImage {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<WebpFrame>,
    pub metadata: WebpFileMetadata,
}

pub struct WebpDecoder { /* streaming; impl Decoder via registry feature */ }
```

### `demux`

```rust
pub fn extract_metadata(buf: &[u8]) -> Result<WebpFileMetadata>;

#[derive(Clone, Debug, Default)]
pub struct WebpFileMetadata {
    pub icc:  Option<Vec<u8>>,
    pub exif: Option<Vec<u8>>,
    pub xmp:  Option<Vec<u8>>,
}
```

(`Default` shown in 0.1.2 module page; ICCP / EXIF / XMP chunk payloads.)

### `encoder` (RIFF-wrapped output, framework path)

```rust
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>>;
```

Accepted input pixel formats per 0.1.2 module docs: `Yuv420P`, `Yuva420P`,
`Rgba`, `Rgb24`.

### `encoder_vp8` (VP8 lossy factories — direct API + registry)

```rust
pub fn make_encoder(params: &CodecParameters)
    -> Result<Box<dyn Encoder>>;

pub fn make_encoder_with_quality(
    params: &CodecParameters,
    quality: f32,                       // WebP-canonical scale 0.0..=100.0
) -> Result<Box<dyn Encoder>>;

pub fn make_encoder_with_qindex(
    params: &CodecParameters,
    qindex: u8,                         // 0..=127, lower = better
) -> Result<Box<dyn Encoder>>;

pub fn make_encoder_with_qindex_and_freq_deltas(
    params: &CodecParameters,
    qindex: u8,
    deltas: Vp8FreqDeltas,
) -> Result<Box<dyn Encoder>>;

pub fn make_encoder_with_quality_and_freq_deltas(
    params: &CodecParameters,
    quality: f32,
    deltas: Vp8FreqDeltas,
) -> Result<Box<dyn Encoder>>;

/// round((100 - quality) * 1.27); NaN -> 127.
pub fn quality_to_qindex(quality: f32) -> u8;

#[derive(Clone, Copy, Debug, Default)]
pub struct Vp8FreqDeltas {
    pub y_dc_delta:  i8,    // each 5-bit signed-magnitude clamped to [-15, 15]
    pub y2_dc_delta: i8,
    pub y2_ac_delta: i8,
    pub uv_dc_delta: i8,
    pub uv_ac_delta: i8,
}
```

### `encoder_anim`

```rust
pub fn build_animated_webp(
    /* frames: &[AnimFrame], canvas_w: u32, canvas_h: u32 */
) -> Result<Vec<u8>>;

pub fn build_animated_webp_with_options(
    frames: &[AnimFrame],
    canvas_w: u32, canvas_h: u32,
    opts: &AnimEncoderOptions,
) -> Result<Vec<u8>>;

#[derive(Clone, Debug)]
pub struct AnimFrame {
    pub width: u32, pub height: u32,
    pub x_offset: u32, pub y_offset: u32,
    pub duration_ms: u32,
    pub rgba: Vec<u8>,                  // tile size w*h*4
    /* blend + dispose fields per ANMF spec */
}

#[derive(Clone, Copy, Debug)]
pub enum AnimFrameMode {
    Auto,         // per-frame: pick byte-smallest of {Lossless, Lossy(VP8)}
    Lossless,
    Lossy,        // hint: encode each ANMF as VP8 lossy
}

#[derive(Clone, Copy, Debug)]
pub struct AnimEncoderOptions {
    pub mode: AnimFrameMode,            // default: AnimFrameMode::Auto
    pub lossy_quality: f32,             // 0.0..=100.0, default 75.0
}

impl Default for AnimEncoderOptions {
    fn default() -> Self {
        Self { mode: AnimFrameMode::Auto, lossy_quality: 75.0 }
    }
}
```

### `error`

```rust
pub type Result<T> = core::result::Result<T, WebpError>;

pub enum WebpError {
    InvalidData(String),
    Unsupported(String),
    Eof,
    NeedMore,
}

impl WebpError {
    pub fn invalid     (msg: impl Into<String>) -> Self;
    pub fn unsupported (msg: impl Into<String>) -> Self;
}

impl core::fmt::Debug   for WebpError { /* … */ }
impl core::fmt::Display for WebpError { /* … */ }
impl core::error::Error for WebpError { /* … */ }

impl From<oxideav_vp8::error::Vp8Error> for WebpError { /* variants map 1-1 */ }

#[cfg(feature = "registry")]
impl From<WebpError> for oxideav_core::error::Error { /* enables `?` */ }
```

### `vp8l`

```rust
pub const VP8L_SIGNATURE: u8 = 0x2F;

pub fn decode(buf: &[u8]) -> Result<Vp8lImage>;

pub fn encode_vp8l_argb(
    width: u32, height: u32,
    argb_pixels: &[u32],                // ARGB-native, one u32/pixel
) -> Result<Vec<u8>>;                   // bare VP8L bitstream, no RIFF wrap

#[derive(Clone, Debug)]
pub struct Vp8lImage {
    pub width:  u32,
    pub height: u32,
    pub pixels: Vec<u32>,               // ARGB raster
    pub has_alpha: bool,
}

impl Vp8lImage {
    pub fn to_rgba(&self) -> Vec<u8>;   // 4 B/px, R,G,B,A row order
}

pub struct HuffmanGroup { /* used by vp8l::encoder & decoder */ }
```

### `registry` (registry feature ON only)

```rust
pub fn register          (ctx: &mut oxideav_core::RuntimeContext);
pub fn register_codecs   (ctx: &mut oxideav_core::RuntimeContext);
pub fn register_containers(ctx: &mut oxideav_core::RuntimeContext);
```

Codec IDs registered: `"webp_vp8"`, `"webp_vp8l"` (both equal the
crate-root constants above). Container: `"webp"`, matched by `.webp`
extension and `RIFF` / `WEBP` magic.

## Standalone build is NON-NEGOTIABLE

```bash
cargo build -p oxideav-webp --no-default-features
cargo test  -p oxideav-webp --no-default-features
```

Both must pass. The standalone build:

1. MUST NOT pull `oxideav-core` (verify by `cargo tree -p oxideav-webp
   --no-default-features --edges normal` showing zero `oxideav-core` node).
2. MUST expose `decode_webp` / `extract_metadata` / `encode_vp8l_argb` /
   `build_animated_webp*` / `WebpImage` / `WebpFrame` / `WebpFileMetadata`
   / `WebpError` / `Result` / `AnimFrame` / `AnimFrameMode` /
   `AnimEncoderOptions` / `CODEC_ID_VP8` / `CODEC_ID_VP8L` /
   `vp8l::Vp8lImage` / `vp8l::VP8L_SIGNATURE` / `vp8l::decode` /
   `encoder_vp8::Vp8FreqDeltas` / `encoder_vp8::quality_to_qindex` /
   `encoder_vp8::make_encoder*` / `encoder::make_encoder`.
3. MAY hide `registry::*` and `WebpDecoder` (registry-trait impls are
   gated; the `WebpDecoder` *struct* itself can stay public as a typed
   handle, but its `impl oxideav_core::Decoder` lives behind the cfg).
4. The `image`-crate flat-buffer property must hold:
   `WebpFrame.rgba` is a tight `Vec<u8>` of `width*height*4` bytes,
   drops into `image::ImageBuffer::from_raw` zero-copy.

## Verification checklist for the finalize round

- [ ] `cargo build -p oxideav-webp` (default registry build, with core)
- [ ] `cargo build -p oxideav-webp --no-default-features` (standalone)
- [ ] `cargo test  -p oxideav-webp` (registry)
- [ ] `cargo test  -p oxideav-webp --no-default-features` (standalone)
- [ ] `cargo doc   -p oxideav-webp --no-default-features` shows every
      type / function listed under "Crate-root re-exports" above.
- [ ] Add a `tests/api_compat_0_1_2.rs` that imports every symbol in the
      Crate-root re-exports section by **fully qualified name** and
      asserts they exist with the documented signature (compile-only
      sufficient; use `let _: fn(&[u8]) -> Result<WebpImage> = decode_webp;`
      style assertions).
- [ ] No new symbols **removed** versus current master; only added/widened.
