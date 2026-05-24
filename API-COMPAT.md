# Public API compatibility target

This crate's master is a **clean-room orphan rebuild** (the previously
published implementation was removed for a clean-room compliance
problem). The published crates.io releases `0.0.3`–`0.1.5` (last
published 2026-05-10, all pre-orphan) exposed a stable public API that
downstream consumers depend on. **The rebuild MUST reproduce the
*shape* of that public API** — the same free function names, types,
fields, enum variants, constants, registry IDs, and trait
implementations, with backward-compatible signatures — so existing
consumers keep compiling and behaving.

Reproduce the **API surface and semantics**, NOT the old
implementation. The internal algorithms (entropy coding, LZ77 search,
psy-RDO, animation delta detection, …) must be re-derived clean-room
from RFC 9649 (WebP/VP8L) + `oxideav-vp8` (VP8 lossy). Do **not** read
the removed source, any published `.crate` `src/`, libwebp, `cwebp`,
or `dwebp` source. Black-box validation against the `cwebp`/`dwebp`
*binaries* is allowed.

Provenance of this target: recovered from the `README.md` of crates.io
`oxideav-webp 0.1.5`. For exact signatures (arg order, field types),
the per-version rustdoc at `https://docs.rs/oxideav-webp/0.1.5/` is the
authoritative record; recover them there or from real consumer usage,
not from the old `src/`.

## PRIMARY requirement — `image`-crate-compatible RGB/RGBA in memory

The single most important consumer property: decode straight to, and
encode straight from, a **flat contiguous byte buffer** (`Vec<u8>`,
row-major, no per-row stride padding, no planar layout) that drops
directly into the [`image`](https://crates.io/crates/image) crate via
`image::ImageBuffer::from_raw(width, height, buf)`:

- **Decode → RGBA**: `decode_webp` yields `WebpFrame { rgba: Vec<u8>, … }`
  where `rgba.len() == width * height * 4`, tightly packed RGBA8 →
  wraps as `image::RgbaImage` (`ImageBuffer<Rgba<u8>, Vec<u8>>`) with
  zero copy / zero repack. This flat-buffer shape is the whole point of
  the free `decode_webp` path; the framework `Decoder` trait path
  (`Frame::Video(VideoFrame)` with planes+strides) is the *other*
  consumer and must not be the only option.
- **Encode ← RGB24 / RGBA**: accept an `image::RgbImage`'s
  (`ImageBuffer<Rgb<u8>, Vec<u8>>`) or `RgbaImage`'s backing `Vec<u8>`
  directly — `pixel_format = Rgb24` (3 B/px) or `Rgba` (4 B/px). The
  `Rgb24` path must **stream** the RGB→internal conversion without
  materialising an intermediate RGBA buffer (no 3→4 byte expansion
  alloc), for both the VP8L and VP8 encoders.

Both directions must be reachable on the **standalone** (no
`oxideav-core`) build, since the typical caller is an image-library
user with no framework. Keep this property green with a test that
round-trips an `image`-crate buffer through encode→decode.

## Standalone surface (must build with `default-features = false`)

With the default `registry` feature OFF, the crate must drop
`oxideav-core` (and cascade the off-switch to `oxideav-vp8`) and still
expose these free-standing entry points:

### Decode
- `pub fn decode_webp(bytes: &[u8]) -> Result<WebpImage, WebpError>`
  — one-shot full decode. **Shape conflict to fix:** current master
  returns `Result<Vec<u8>, Error>`. Restore the `WebpImage`/`WebpError`
  shape; the existing low-level `decode_webp_image -> DecodedWebp` /
  `decode_lossless_image` helpers may stay as additional internal/extra
  API but must not occupy the `decode_webp` name with a different
  signature.
- `pub fn extract_metadata(bytes: &[u8]) -> Result<WebpFileMetadata, WebpError>`
  — metadata-only, decodes no pixels.

### Decode types
- `pub struct WebpImage` with at least:
  - `frames: Vec<WebpFrame>`
  - `metadata: WebpFileMetadata`
  - `anim_background_rgba: Option<[u8; 4]>`  (None for non-animated)
  - `anim_loop_count: Option<u16>`           (0 = infinite; None for non-animated)
- `pub struct WebpFrame` with at least:
  - `rgba: Vec<u8>`        (length `width * height * 4`, row-major)
  - `width: u32`, `height: u32`
  - `duration_ms: …`       (ANMF per-frame delay; 0 for still images)
- `pub struct WebpFileMetadata { icc: Option<Vec<u8>>, exif: Option<Vec<u8>>, xmp: Option<Vec<u8>> }`
- `pub enum WebpError { InvalidData, Unsupported, Eof, NeedMore }`
  - `impl From<oxideav_vp8::Vp8Error> for WebpError`

### Lossless (VP8L) encode entry points
- `pub fn encode_vp8l_argb(...)` — bare VP8L bitstream, **no** RIFF wrapper.
- `pub fn encode_vp8l_argb_with(...)` — fixed (non-RDO) configuration.
- `pub fn encode_vp8l_argb_with_metadata(w: u32, h: u32, argb: &[…], has_alpha: bool, meta: &WebpMetadata) -> Result<Vec<u8>, WebpError>`
  — complete `.webp`; auto-promotes to extended `VP8X` layout iff alpha
  or any metadata field is set, else simple `VP8L`.

### Lossy (VP8) encode entry points  (route through `oxideav-vp8`)
- `pub fn encode_vp8_lossy_yuv420p(…, meta: &WebpMetadata) -> Result<Vec<u8>, WebpError>`
- `pub fn encode_vp8_lossy_yuva420p(…, meta: &WebpMetadata) -> Result<Vec<u8>, WebpError>`
- `pub fn encode_vp8_lossy_rgba(width: u32, height: u32, rgba: &[u8], quality: f32, meta: &WebpMetadata) -> Result<Vec<u8>, WebpError>`
- `pub fn encode_vp8_lossy_rgb24(…, meta: &WebpMetadata) -> Result<Vec<u8>, WebpError>`
  (`quality`: libwebp-style `0.0..=100.0`, default `75.0`)

### Animation encode
- `pub fn build_animated_webp(…) -> Result<Vec<u8>, WebpError>`
- `pub fn build_animated_webp_with_options(frames: &[AnimFrame], opts: &AnimEncoderOptions) -> Result<Vec<u8>, WebpError>`
- `pub struct AnimFrame` (pixels, `duration`, x/y offset, blend, dispose)
- `pub enum AnimFrameMode { Auto, Delta, Lossless }`
- `pub struct AnimEncoderOptions { metadata: …, … }`
- `pub struct DeltaConfig` with builder methods:
  `max_components_override(n)`, `auto_inner_threshold_bytes(Option<usize>)`,
  `msssim_downsample_kernel(DownsampleKernel)`
- `pub enum DownsampleKernel { Box, Gaussian }`

### Metadata types + constants
- `pub struct WebpMetadata` (borrowed form; `WebpMetadata::default()`)
- `pub struct WebpMetadataOwned` (owned form; registry side)
- `pub const CODEC_ID_VP8L` and `pub const CODEC_ID_VP8`

## Registry surface (default `registry` feature ON → pulls `oxideav-core`)

- `pub fn register(ctx: &mut RuntimeContext)` — registers codecs + container.
- Codec `"webp_vp8l"`: VP8L encoder + standalone VP8L decoder; accepts
  input `Rgba` / `Rgb24`; decoded output always `Rgba`.
- Codec `"webp_vp8"`: VP8 lossy path; accepts `Yuv420P` / `Yuva420P` /
  `Rgba` / `Rgb24`.
- Container `"webp"`: matches `.webp` by extension + `RIFF`/`WEBP` magic.
- `Decoder` / `Encoder` / `Demuxer` trait impls.
- `pub struct WebpDecoder` (streaming):
  - `WebpDecoder::new(w: u32, h: u32)`
  - `WebpDecoder::new_yuva420p(w: u32, h: u32)`  (4-plane Yuva420P out for VP8+ALPH)
  - `set_prefer_yuva420p(bool)`
  - drives via `oxideav_codec::Decoder::{send_packet, receive_frame}`;
    yields `Frame::Video(vf)` with `vf.format == PixelFormat::Rgba`
    (or `Yuva420P` when opted in).
- VP8 lossy factories (module `encoder_vp8`) — keep the dual-API
  convention (registry path + direct factories):
  - `encoder_vp8::make_encoder_with_quality(&params, quality: f32)`
  - `encoder_vp8::make_encoder_with_qindex(&params, qindex)`  (`0..=127`, lower = better)
  - `encoder_vp8::make_encoder_with_target_size(&params, target_bytes)`
  - `encoder_vp8::make_encoder_with_qindex_and_metadata(…)`
  - `encoder_vp8::make_encoder_with_quality_and_metadata(…)`
  - `*_and_freq_deltas` variants taking `encoder_vp8::Vp8FreqDeltas`
  - `encoder_vp8::Vp8FreqDeltas`, `encoder_vp8::compute_psy_stats(…)`,
    `encoder_vp8::freq_deltas_for_qindex(…)`
- Direct factory `make_decoder(&CodecParameters) -> Result<Box<dyn Decoder>>`
  (already present) plus the matching `make_encoder` factory.

## Behavioural invariants

- Default decoded output pixel format is `Rgba` (single-frame and
  animated). `Yuva420P` only via the `new_yuva420p` / `set_prefer_yuva420p`
  opt-in, and only for single-frame VP8+ALPH.
- Single-image WebP → one `VideoFrame`; animated WebP → N frames, PTS in
  milliseconds (ANMF native unit).
- `Rgb24` encode inputs are treated as opaque and emit the simple layout.
