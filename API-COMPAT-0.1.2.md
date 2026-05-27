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

> **Round-168**: wired against `oxideav-vp8 0.2.1`. The five factories
> below delegate to the underlying `oxideav_vp8::encoder` framework
> factories and wrap every emitted raw VP8 keyframe in a §2.5
> simple-lossy `RIFF/WEBP` container. The `_freq_deltas` variants pass
> through to the matching no-deltas factory in this round (the
> `Vp8FreqDeltas` argument is a hint; plumbing it into the underlying
> encoder's per-band `KeyframeParams` deltas is a follow-up). See
> `tests/vp8_lossy_roundtrip.rs` for end-to-end coverage.


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

> **Round-168 spec deviation (deliberately widened)**: the rebuild's
> `AnimFrame` is the *owned* `Vec<u8>` shape rather than the published
> 0.1.2 `AnimFrame<'a> { rgba: &'a [u8], … }` borrowed shape, and the
> per-frame `AnimFrameMode` / `BlendingMethod` / `DisposalMethod`
> typed enums replace 0.1.2's `blend: bool` / `dispose_to_background: bool`
> booleans. The `build_animated_webp` / `_with_options` signatures
> match this owned shape — `canvas_w` / `canvas_h` / `background_bgra` /
> `loop_count` are read off the frames and the [`AnimEncoderOptions`]
> respectively. The current shape is a strict superset of the 0.1.2
> capability set: every animation a 0.1.2 caller could build, the
> rebuild can build (and several more — explicit dirty-rect / `Auto`
> mode picking, alpha-blend disposal, file-level metadata). The
> trade-off is a `&[u8] → Vec<u8>` clone at the boundary. See
> `tests/published_anim_api.rs` for the locked-in current shape; see
> `https://docs.rs/oxideav-webp/0.1.2/oxideav_webp/encoder_anim/`
> for the historical borrowed shape this deviation supplants.

```rust
pub fn build_animated_webp(frames: &[AnimFrame]) -> Result<Vec<u8>>;

pub fn build_animated_webp_with_options(
    frames: &[AnimFrame],
    opts: &AnimEncoderOptions<'_>,
) -> Result<Vec<u8>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnimFrame {
    pub pixels: Vec<u8>,                // tile size w*h*4, owned
    pub width: u32, pub height: u32,
    pub x: u32, pub y: u32,             // even per §2.7.1.1
    pub duration: u32,                  // ms
    pub blend: BlendingMethod,          // enum (was: bool in 0.1.2)
    pub dispose: DisposalMethod,        // enum (was: bool in 0.1.2)
    pub mode: AnimFrameMode,            // per-frame mode (new vs 0.1.2)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnimFrameMode {
    #[default] Auto,    // per-frame: pick byte-smallest of {Lossless, Delta}
    Delta,              // dirty-rect sub-frame against the previous canvas
    Lossless,           // full-canvas VP8L keyframe
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AnimEncoderOptions<'a> {
    pub loop_count: u16,                // §2.7.1.1 ANIM loop count
    pub background_rgba: [u8; 4],       // §2.7.1.1 ANIM background colour
    pub metadata: WebpMetadata<'a>,     // ICC / Exif / XMP, borrowed
    pub delta: DeltaConfig,             // dirty-rect tuning knobs
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

// Round-168: wired against `oxideav-vp8 0.2.1` (Vp8Error exported at
// `oxideav_vp8::Vp8Error` and `oxideav_vp8::error::Vp8Error`). The four
// `Vp8Error` variants map 1-1 onto `WebpError`; the `String` payloads
// on `InvalidData` / `Unsupported` are dropped (rebuild collapses to
// unit variants — see `WebpError::invalid` / `unsupported` constructors
// for the same convention).
impl From<oxideav_vp8::Vp8Error> for WebpError { /* InvalidData→InvalidData,
    Unsupported→Unsupported, Eof→Eof, NeedMore→NeedMore */ }

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
