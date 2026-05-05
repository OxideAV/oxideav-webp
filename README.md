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

## Standalone use (no `oxideav-core`)

Image-library consumers that just want to turn a `.webp` byte buffer
into RGBA pixels — no framework, no codec registry, no trait
objects — can depend on this crate with the default `registry`
feature off:

```toml
[dependencies]
oxideav-webp = { version = "0.0", default-features = false }
```

That drops the `oxideav-core` dependency entirely (and cascades the
same off-switch through to `oxideav-vp8`) and exposes the
free-standing decode/encode entry points:

```rust
use oxideav_webp::{decode_webp, WebpImage, WebpError};

let img: WebpImage = decode_webp(&bytes)?;
for frame in &img.frames {
    // frame.rgba is `Vec<u8>` of length width * height * 4
}
# Ok::<_, WebpError>(())
```

`WebpImage` / `WebpFrame` / `WebpFileMetadata` already use
std-primitive fields (`Vec<u8>` RGBA, `u32` dimensions). `WebpError`
covers `InvalidData` / `Unsupported` / `Eof` / `NeedMore` and
`From`-converts from `oxideav_vp8::Vp8Error` so the VP8 lossy path
composes through cleanly. Encoder entry points
(`encode_vp8l_argb` / `encode_vp8l_argb_with`,
`build_animated_webp`) likewise stay available without
`oxideav-core`. Turning the `registry` feature back on adds the
`Decoder` / `Encoder` / `Demuxer` trait implementations + the
`register` helpers + the `WebpDecoder` streaming type so the crate
plugs into the framework registry as before.

## Quick use

`.webp` files carry one or many frames, so the typical path is: open
the file as a container, pull packets, decode them. Output is always
`PixelFormat::Rgba` regardless of whether the source chunk was `VP8 `
(lossy YUV → RGB) or `VP8L` (native RGBA).

```rust
use oxideav_core::{Frame, RuntimeContext};

let mut ctx = RuntimeContext::new();
oxideav_webp::register(&mut ctx);
let codecs = &ctx.codecs;
let containers = &ctx.containers;

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

Quality control: the VP8 lossy encoder exposes three factory entry
points for picking a target compression level —

* `encoder_vp8::make_encoder_with_quality(&params, quality)` — takes
  a libwebp-style `quality: f32` in `0.0..=100.0` (higher = better
  quality / larger file; the libwebp default is `75.0`).
* `encoder_vp8::make_encoder_with_qindex(&params, qindex)` — takes
  the underlying VP8 qindex in `0..=127` (lower = better) for callers
  that already speak the libvpx scale.
* `encoder_vp8::make_encoder_with_target_size(&params, target_bytes)`
  — drives a per-frame rate-control bisection (≤ 5 trials over
  qindex) to hit a caller-supplied byte budget within ±10 %. Use
  this when you need a known output footprint regardless of source
  complexity (image-thumbnailers, fixed-quota uploads, etc.).

The `quality → qindex` mapping is the linear inversion
`qindex = round((100 - quality) * 1.27)`. As of #465 the per-quality
knob also drives the per-segment QP / LF deltas (§10 / §15.2) **and**
the per-frequency AC/DC quant deltas (§6.6 / §9.6) — at high quality
every delta collapses to zero, at low quality the high-frequency Y2
AC and chroma AC bins land on a coarser step while the macroblock-
mean (Y2 DC) bin holds finer to suppress visible block-mean banding.
A **psy-RDO** modulation runs on top: the encoder computes per-frame
[`PsyStats`] from the source luma (mean activity + high-variance MB
fraction, one O(W·H) MAD pass), and biases the per-frequency AC and
per-segment-3 quant deltas by ±1 step based on the resulting CSF
profile — high-activity frames trim more bits from the masked high-
freq bins, low-activity frames spend extra bits to suppress visible
banding on flat regions. Modulation strength scales with qindex so
high-quality (qindex=0) output stays byte-identical to the pre-psy
bitstream. File size is byte-strictly monotone with quality on
AC-rich content and bitstreams stay spec-compliant under libwebp's
`dwebp`. Callers that have already done their own perceptual tuning
should reach for the explicit `*_and_freq_deltas` factories, which
pass the supplied `Vp8FreqDeltas` through verbatim *and* skip the
psy modulation (no preset added on top).

### Scope

Encoder scope (current):

- VP8L lossless from `Rgba` or `Rgb24` (single frame). Emits
  subtract-green + colour (G↔R/B decorrelation) + tile-based predictor
  + colour-indexing (palette) transforms plus a tunable colour cache.
  The default `encode_vp8l_argb` entry point runs a per-image RDO sweep
  over every combination of the four optional transforms × eight
  colour-cache widths ({off, 4, 6, 7, 8, 9, 10, 11} bits) × three
  predictor tile sizes ({8, 16, 32} px) and keeps the smallest
  encoded variant. Each trial also tries meta-Huffman per-tile
  grouping at K = 1 / 2 / 4 / 8 / 16 (gated by image pixel count:
  K=4 ≥ 4096 px, K=8 ≥ 16384 px, K=16 ≥ 65536 px), and for each K
  sweeps the **entropy-image tile size** at `meta_bits ∈ {3, 4, 5}`
  (8 / 16 / 32-pixel tiles) — i.e. **smart tile-boundary switching**
  on the entropy-image side, picking the (K, meta_bits) tuple that
  bills the fewest bits. Per RFC 9649 §3.7.2.2 the entropy image is
  the per-tile prefix-code-group selector (the libwebp "EntropyImage"
  transform); the encoder builds it via k-means++ farthest-first
  clustering of per-tile green-alphabet histograms with K-scaled
  iteration count (2 passes for K ≤ 2, 3 for K = 4, 4 for K ≥ 8) so
  the cluster boundaries actually settle on multi-region content
  rather than collapsing to a smaller-K baseline. Predictor pool
  covers all 14 RFC 9649 §4.1 modes per tile. LZ77 backreference
  search uses a 16384-pixel sliding window with up to 256 hash-chain
  candidates per starting position; the matcher runs a **three-pass
  cost-modelled** scan on the main image — pass 1 is greedy
  first-match, pass 2 re-walks the chain with a per-symbol
  `-log2(p) × 16` bit-cost model derived from the pass-1 histogram
  and picks each match by lowest bit-cost-per-pixel (plus a one-step
  lazy lookahead that defers a match if literal-here +
  match-at-i+1 bills fewer model bits), and pass 3 runs a full
  **Viterbi-style optimal LZ77** dynamic-programming sweep over the
  backward-reference graph (gated on ≥ 65 536 px = 256×256 — below
  it the per-position chain-walk cost dominates the per-K-trial
  budget). Pass 3 considers, at each pixel position, every plausible
  edge out of that position (literal, cache-ref, every reachable
  `(len, dist)` backref candidate at multiple lengths per chain
  entry) and updates `dp[i + span]` with the minimum cumulative
  cost; backtrack from `dp[n]` recovers the optimal symbol sequence.
  A refit sub-step rebuilds the cost model from the pass-3a histogram
  and re-runs the DP, keeping whichever Viterbi candidate bills
  fewer modelled bits. Optional near-lossless preprocessing
  (libwebp-compatible `0..=100` knob) collapses near-identical
  pixels into longer LZ77 runs / richer cache hits. Callers that
  want a fixed configuration call `encode_vp8l_argb_with` directly.
  A multi-iteration Huffman refit (up to 3 extra Viterbi passes under
  the exact code-length cost model) further tightens the token
  sequence after the initial Viterbi selection. Predictor-mode
  selection uses **Shannon-entropy scoring** (`Σ -p_i log2(p_i)` over
  per-channel residual histograms) matching cwebp's criterion;
  **colour-transform coefficient selection** likewise uses Shannon-
  entropy (256-bin histograms over the transformed R and B channels),
  replacing the earlier sum-of-abs-residuals (SAD) criterion and
  keeping both selection loops consistent.
  Encoder ≈ 96–100 % libwebp parity on natural fixtures (≤ 1.13× cwebp
  on a 1024×768 photo, ≤ 1.06× on a 512×512 still, **beats cwebp by
  4.5 %** on the in-tree 128×128 natural fixture and by 14.4 % on the
  64×64 cache-stress fixture, **byte-parity** with cwebp on a synthetic
  three-region 256×256 photo, **sub-cwebp (0.995×)** on the 256×256
  landscape, **1.030×** on brick-wall-256, **1.040×** on
  portrait-textured-256 — all measured under full RDO vs cwebp
  `-lossless -m 6 -z 9`).
- VP8 lossy from `Yuv420P`, `Yuva420P`, `Rgba`, or `Rgb24` (single
  frame). For `Yuva420P` and `Rgba` the alpha plane is emitted as a
  VP8L-compressed `ALPH` chunk inside the extended (`VP8X`)
  container; `Yuva420P` skips the YUV→RGB→YUV roundtrip the `Rgba`
  path forces. `Rgb24` streams the RGB→YUV conversion without a
  Rgba alloc (issue #7). Per-segment quantiser deltas (RFC 6386 §10)
  + per-segment loop-filter deltas (§15.2) are wired in based on a
  source-luma variance classifier, so smooth / textured regions get
  finer / coarser quant + softer / stronger deblocking respectively.
  Per-frequency AC/DC quantiser deltas (`y_dc_delta` / `y2_dc_delta`
  / `y2_ac_delta` / `uv_dc_delta` / `uv_ac_delta`) are wired through
  `encoder_vp8::Vp8FreqDeltas` and driven by the libwebp-style
  `quality` knob via `freq_deltas_for_qindex`: zero at qindex=0,
  widening to `[0, -2, +4, 0, +4]` at qindex=127 so high-frequency
  bins compress harder and the macroblock-mean bin holds finer to
  suppress block-mean banding. **Psy-RDO source analysis**
  (`encoder_vp8::compute_psy_stats`) runs once per frame on the
  source luma plane (cost is one O(W·H) MAD pass) and modulates
  both the per-frequency deltas and the per-segment quant deltas
  from the actual content distribution: high-activity frames
  (`mean_activity ≥ 16`) get one step coarser high-freq AC bins
  (CSF activity-mask saving on visually-noisy content),
  low-activity frames (`mean_activity < 6`) get one step finer
  high-freq AC bins to suppress banding on flat regions, and
  variance-segment deltas widen on segment-3-heavy frames /
  narrow on segment-3-empty frames. **Per-frame rate control**
  (`encoder_vp8::make_encoder_with_target_size`) bisects qindex
  in ≤ 5 trials (worst-case 6× single-shot encode cost) to land
  within ±10 % of a caller-supplied byte budget — converges in
  2-3 iterations on natural-image content (size-vs-qindex curve
  is monotone). Default qindex from `oxideav-vp8` is used unless
  the caller selects one via `encoder_vp8::make_encoder_with_qindex`
  (VP8 qindex `0..=127`, lower = better) or the libwebp-style
  `encoder_vp8::make_encoder_with_quality` (`0.0..=100.0`,
  higher = better). Explicit `*_and_freq_deltas` factories pass
  user freq-deltas through verbatim *and* skip the psy modulation
  (zero argument reproduces the pre-#465 bitstream byte-for-byte).
  Encoder ≈ 92 % libwebp parity on natural fixtures (psy save:
  −1.1 % bytes / +0.01 dB at qindex 64 on the in-tree noisy
  128×128 fixture); the residual gap is fully-internal per-MB QP
  refinement (sub-MB rate control inside `oxideav-vp8`'s segment
  classifier).
- `VP8X` extended header is emitted automatically whenever the output
  carries an `ALPH` sidecar or optional ICC / EXIF / XMP metadata via
  the `riff::WebpMetadata` helper.
- Animated WebP encode via [`build_animated_webp`] /
  [`build_animated_webp_with_options`] — emits a
  `VP8X + ANIM + ANMF...ANMF` file from a slice of `AnimFrame`s with
  per-frame durations, x/y offsets, blend, and disposal flags. Per-frame
  `AnimFrameMode::Auto` runs both VP8L and VP8+ALPH encoders and picks
  whichever sub-chunk is byte-smaller — animations can mix lossless
  and lossy frames, matching libwebp's `WebPAnimEncoderAdd` behaviour.
  All four blend × dispose-to-background combinations round-trip
  through the in-crate decoder.

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
