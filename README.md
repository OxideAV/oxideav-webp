# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM + ANMF).

## Status

**Production-ready, both decoder and encoder at ✅ 100% as of 2026-05-27.**

* **Decoder** — RFC 9649 RIFF container; VP8 lossy (via the
  `oxideav-vp8` sibling); VP8L lossless (full §3–§7 path including
  LZ77, spatial / color / color-indexing transforms, color cache,
  multi-meta-prefix); `ALPH` alpha plane; animated WebP (`ANIM` +
  `ANMF`); `VP8X` extended container with ICCP / EXIF / XMP
  metadata. File-level metadata extraction without pixel decode.
* **Encoder** — VP8L lossless (LZ77 + spatial + color transform +
  color cache + color indexing + multi-meta-prefix + histogram-
  distance clusterer + Shannon-entropy chooser + depth-4 lazy
  matching); VP8 lossy wired through `oxideav-vp8 0.2.1` (a
  `WebpVp8LossyEncoder` adapter wraps every emitted raw VP8
  keyframe in a §2.5 simple-lossy `RIFF/WEBP` container).
  Animation encoder emits dirty-rect `ANMF` sub-frames with
  `Auto` / `Delta` / `Lossless` per-frame mode selection.
* **0.1.2 public-surface lock** — every symbol the published
  crates.io `0.1.2` release exposed is reachable, both with the
  default `registry` build and under `--no-default-features`. See
  [`API-COMPAT-0.1.2.md`](./API-COMPAT-0.1.2.md) for the
  per-symbol contract and [`tests/api_compat_0_1_2.rs`](./tests/api_compat_0_1_2.rs)
  for the 29-test compile-only assertion suite. One deliberate
  spec deviation — the rebuild's `AnimFrame` owned shape replaces
  0.1.2's borrowed shape — is documented in the spec.

## Cargo features

| Feature | Default | What it does |
|---|---|---|
| `registry` | ✅ on | Enables the `oxideav-core` dependency and the framework-trait factories. Cascades into `oxideav-vp8/registry` so the VP8-lossy encode delegation can reach the sibling crate's `make_encoder*` factories. |

The crate builds and tests cleanly under both `cargo build -p oxideav-webp`
and `cargo build -p oxideav-webp --no-default-features`; both
configurations are kept green in CI.

## Direct-API entry points

### Decode

```rust
use oxideav_webp::{decode_webp, extract_metadata, WebpImage};

// Full decode — image::ImageBuffer::from_raw consumes the RGBA buffer
// zero-copy because `WebpFrame.rgba` is a tight Vec<u8> of width*height*4.
let image: WebpImage = decode_webp(&webp_bytes)?;
for frame in &image.frames {
    let rgba = &frame.rgba;
    // ... assert_eq!(rgba.len(), (frame.width * frame.height * 4) as usize);
}

// Metadata only, no pixel decode.
let meta = extract_metadata(&webp_bytes)?;
let icc  = meta.icc.as_deref();
let exif = meta.exif.as_deref();
let xmp  = meta.xmp.as_deref();
```

### Encode

```rust
use oxideav_webp::{
    vp8l::encode_vp8l_argb,
    encoder_vp8::{make_encoder_with_quality, make_encoder_with_qindex, quality_to_qindex},
    build_animated_webp,
    build_animated_webp_with_options,
    AnimEncoderOptions,
};

// Bare VP8L bitstream (no RIFF wrap).
let bitstream: Vec<u8> = encode_vp8l_argb(&argb, width, height)?;

// VP8 lossy via the framework path (RIFF/WEBP wrapped output).
let mut ctx = oxideav_core::RuntimeContext::new();
oxideav_webp::register(&mut ctx);
let enc = make_encoder_with_quality(&params, 75.0)?;
// ... drive enc.send_frame(...) / enc.receive_packet(...) as usual.

// Animated WebP.
let file = build_animated_webp(&frames)?;
```

The `_freq_deltas` factories (`make_encoder_with_qindex_and_freq_deltas`,
`make_encoder_with_quality_and_freq_deltas`) accept a `Vp8FreqDeltas`
record of per-band quantiser deltas — currently a forwarded hint (the
qindex is honoured; per-band plumbing into `oxideav-vp8`'s
`KeyframeParams` is a follow-up).

## Registry path

```rust
let mut ctx = oxideav_core::RuntimeContext::new();
oxideav_webp::register(&mut ctx);
// ctx now has the "webp" container plus "webp_vp8" + "webp_vp8l" codecs.
```

Codec / container IDs:

| Constant | Value | Notes |
|---|---|---|
| `CODEC_ID_VP8`  | `"webp_vp8"`  | VP8 lossy bitstream. |
| `CODEC_ID_VP8L` | `"webp_vp8l"` | VP8L lossless bitstream. |

Container: `"webp"`, matched by the `.webp` extension and the
`RIFF`/`WEBP` magic.

## Clean-room sources

Implementation is derived entirely from the public format specs:

* **RFC 9649** — WebP Image Format
  (`docs/image/webp/rfc9649-webp.txt`, also `rfc9649-webp.pdf`).
* **WebP Lossless Bitstream Specification** — `docs/image/webp/
  google-webp-lossless-bitstream.html` (also reproduced in RFC 9649
  §3). Covers the VP8L LZ77 + prefix-coded literals + color cache +
  spatial / color / color-indexing transforms.
* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`) for the VP8 lossy
  framing routed through the `oxideav-vp8` sibling.

The 18-fixture corpus at `docs/image/webp/fixtures/` is consumed as
opaque byte streams; end-to-end fixture tests validate against the
ARGB pixels of each fixture's committed `expected.png` (a clean-room
PNG decode of the corpus' own ground-truth files). No third-party
codec library source is consulted.

## License

MIT. See [`LICENSE`](./LICENSE).
