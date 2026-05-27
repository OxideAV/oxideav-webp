# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM +
ANMF). Decoder and encoder both at production status as of 2026-05-27.

* Full **decode** of every container variant: simple-lossy (VP8),
  simple-lossless (VP8L), extended (`VP8X`) with `ALPH` alpha plane,
  ICCP / EXIF / XMP metadata, and animated WebP (`ANIM` + `ANMF`).
* **Encode** of complete `.webp` files in both lossless (VP8L) and
  lossy (VP8) modes, plus complete animated `.webp` files.
* Decoded pixels land in a tightly-packed `Vec<u8>` of `width * height
  * 4` RGBA bytes — drops directly into [`image`](https://crates.io/crates/image)'s
  `ImageBuffer::from_raw` with zero copy.
* The full crates.io `0.1.2` public surface is reachable, both with
  the default `registry` build and under `--no-default-features`.
  See [`API-COMPAT-0.1.2.md`](./API-COMPAT-0.1.2.md) for the
  per-symbol contract and [`tests/api_compat_0_1_2.rs`](./tests/api_compat_0_1_2.rs)
  for the 29-test compile-only assertion suite.

## Install

```toml
# Standalone — flat RGBA in / flat RGBA out, no framework dep:
[dependencies]
oxideav-webp = { version = "0.1", default-features = false }

# With the OxideAV runtime:
[dependencies]
oxideav-webp = "0.1"
```

| Feature | Default | What it does |
|---|---|---|
| `registry` | ✅ on | Pulls `oxideav-core` plus the framework-trait factories. Cascades into `oxideav-vp8/registry` so the VP8-lossy encode delegation can reach the sibling crate's factories. With this off, **lossless encode/decode + animation + metadata extraction all still work**; only the VP8-lossy *encode* requires `registry`. |

## Standalone use (no `oxideav-core`)

### Decode any `.webp` file

```rust
use oxideav_webp::{decode_webp, WebpImage};

let webp_bytes: &[u8] = /* file bytes from disk, HTTP, … */;
let image: WebpImage = decode_webp(webp_bytes)?;

println!("{} × {}, {} frame(s)", image.width, image.height, image.frames.len());
for frame in &image.frames {
    // frame.rgba is a tight Vec<u8> of width*height*4 RGBA bytes,
    // row-major, no per-row padding — drops into `image::ImageBuffer`:
    //
    //   let img = image::RgbaImage::from_raw(frame.width, frame.height,
    //                                        frame.rgba.clone()).unwrap();
    //
    println!("  frame: {}×{}, {} ms", frame.width, frame.height, frame.duration_ms);
}

// ICC / EXIF / XMP are on image.metadata.{icc, exif, xmp} (each Option<Vec<u8>>).
```

### Read metadata only (no pixel decode)

```rust
use oxideav_webp::extract_metadata;

let meta = extract_metadata(webp_bytes)?;
if let Some(icc) = meta.icc.as_deref()  { /* color-management profile */ }
if let Some(exif) = meta.exif.as_deref() { /* EXIF blob */ }
if let Some(xmp) = meta.xmp.as_deref()   { /* XMP UTF-8 XML */ }
```

### Encode a lossless `.webp` from RGBA bytes

The shortest path — flat RGBA in, complete `.webp` file out:

```rust
use oxideav_webp::encode_webp_lossless;

let rgba: Vec<u8> = /* width*height*4 RGBA bytes */;
let webp_bytes: Vec<u8> = encode_webp_lossless(&rgba, width, height)?;
// Write to disk:
std::fs::write("out.webp", &webp_bytes)?;
```

### Encode lossless with metadata (ICC / EXIF / XMP)

```rust
use oxideav_webp::{encode_vp8l_argb_with_metadata, WebpMetadata};

// VP8L works in ARGB, one u32/pixel.
let argb: Vec<u32> = /* width*height ARGB pixels */;

let meta = WebpMetadata {
    icc:  Some(&my_icc_profile),
    exif: Some(&my_exif_blob),
    xmp:  Some(&my_xmp_xml),
};
let webp_bytes = encode_vp8l_argb_with_metadata(
    width, height, &argb, /* has_alpha = */ true, &meta,
)?;
```

If `has_alpha` is `true` or any metadata field is set, the output
auto-promotes to the extended `VP8X` layout; otherwise it's the
simple lossless layout.

### Bare VP8L bitstream (no RIFF wrap)

For consumers that wrap the bitstream themselves:

```rust
use oxideav_webp::vp8l::encode_vp8l_argb;
let vp8l: Vec<u8> = encode_vp8l_argb(&argb, width, height)?;
```

### Build an animated `.webp`

```rust
use oxideav_webp::{build_animated_webp, build_animated_webp_with_options,
                   AnimFrame, AnimEncoderOptions};

// Each AnimFrame is a tile (width × height RGBA) at (x, y) on the
// canvas, with a duration in milliseconds.
let frames = vec![
    AnimFrame::new(/* w */ 64, /* h */ 64, /* rgba */ frame0_rgba, /* duration_ms */ 100),
    AnimFrame::new(64, 64, frame1_rgba, 100),
    AnimFrame::new(64, 64, frame2_rgba, 100),
];

// Defaults: per-frame Auto mode (picks byte-smallest of Lossless / Delta).
let webp = build_animated_webp(&frames)?;

// Or with options (loop count, background colour, file-level metadata):
let opts = AnimEncoderOptions {
    loop_count: 0,                      // 0 = infinite
    background_rgba: [0xff, 0xff, 0xff, 0xff],
    ..Default::default()
};
let webp = build_animated_webp_with_options(&frames, &opts)?;
```

## With the OxideAV runtime (`registry` feature on)

```rust
use oxideav_core::RuntimeContext;
use oxideav_webp::{CODEC_ID_VP8, CODEC_ID_VP8L};   // "webp_vp8" / "webp_vp8l"

let mut ctx = RuntimeContext::new();
oxideav_webp::register(&mut ctx);
// ctx now exposes the "webp" container plus "webp_vp8" + "webp_vp8l" codecs.
```

This is the only way to reach the **VP8-lossy encoder** — it delegates
to the `oxideav-vp8` sibling crate's framework factory family:

```rust
use oxideav_webp::encoder_vp8::{make_encoder_with_quality, make_encoder_with_qindex};

// Returns Box<dyn oxideav_core::Encoder>; emits RIFF/WEBP-wrapped output.
let enc = make_encoder_with_quality(&params, 75.0)?;
let enc = make_encoder_with_qindex(&params, 32)?;
```

(Lossless encode + decode + animation + metadata extraction all work
without `registry`; only the VP8 *lossy* encode path needs it.)

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

The 18-fixture corpus at `docs/image/webp/fixtures/` is consumed as
opaque byte streams; end-to-end fixture tests validate against the
ARGB pixels of each fixture's committed `expected.png`. No third-party
codec library source is consulted.

## License

MIT. See [`LICENSE`](./LICENSE).
