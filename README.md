# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM + ANMF).

## Status — 2026-05-22 (clean-room round 6)

* **Container walker:** RFC 9649 §2.3–§2.7 RIFF/WEBP chunk walk.
  Surfaces the chunk list (`VP8 ` / `VP8L` / `VP8X` / `ALPH` /
  `ANIM` / `ANMF` / `ICCP` / `EXIF` / `XMP ` plus unknowns) with
  per-chunk FourCC, size, and absolute payload range. Validates the
  §2.4 file header, the §2.4 `File Size` against buffer length, every
  chunk's `Size` against the remaining RIFF payload, and the §2.3
  odd-size pad byte. Order-on-disk is preserved so §2.7 ordering
  rules can be applied by callers.
* **VP8X header field parse:** RFC 9649 §2.7.1 typed decode of the
  10-byte `VP8X` payload. [`vp8x::Vp8xHeader::parse`] / the
  [`parse_vp8x_header`](src/lib.rs) convenience wrapper return
  `Vp8xHeader { canvas_width, canvas_height, has_iccp, has_alpha,
  has_exif, has_xmp, has_animation, has_unknown }`. The §2.7.1
  product cap (`width * height ≤ 2^32 - 1`) is enforced; reserved
  bits are surfaced via `has_unknown` but do **not** trigger refusal.
* **ALPH info-byte parse (round 3):** RFC 9649 §2.7.1.2 typed decode
  of the 1-byte `Rsv|P|F|C` field. [`alph::AlphHeader::parse`] / the
  [`parse_alph_header`](src/lib.rs) wrapper return
  `AlphHeader { compression: AlphCompression, filtering:
  AlphFiltering, preprocessing: AlphPreprocessing, reserved, info_byte }`.
  `AlphCompression` covers `None` / `Lossless` / `Reserved(u8)`,
  `AlphFiltering` covers all four named methods (`None` / `Horizontal`
  / `Vertical` / `Gradient`), and `AlphPreprocessing` covers `None` /
  `LevelReduction` / `Reserved(u8)`. The alpha bitstream itself is
  **not** decoded — `AlphHeader::bitstream_offset()` returns the
  fixed `1` so callers can slice it out.
* **ANIM payload parse (round 3):** RFC 9649 §2.7.1.1 typed decode of
  the 6-byte payload. [`anim::AnimHeader::parse`] / the
  [`parse_anim_header`](src/lib.rs) wrapper return
  `AnimHeader { background_color: BackgroundColor { blue, green, red,
  alpha }, loop_count }`. `loops_forever()` reports the §2.7.1.1
  "0 means infinite" sentinel.
* **ANMF per-frame header parse (round 4):** RFC 9649 §2.7.1.1
  Figure 9 typed decode of the 16-byte per-frame header.
  [`anmf::AnmfHeader::parse`] / the
  [`parse_anmf_header`](src/lib.rs) wrapper return
  `AnmfHeader { x, y, width, height, duration_ms, blend:
  BlendingMethod, dispose: DisposalMethod, reserved, info_byte }`.
  Resolved-form fields: `x` / `y` carry the canvas-pixel
  coordinates (already doubled per §2.7.1.1); `width` / `height`
  carry the 1-based pixel counts. `BlendingMethod` is `AlphaBlend`
  / `Overwrite`; `DisposalMethod` is `None` / `Background`. The
  per-frame `Frame Data` sub-RIFF is **not** decoded —
  `AnmfHeader::frame_data_offset()` returns the fixed `16` so
  callers can slice it out for the next layer.
* **RIFF/WEBP builders (round 5):** RFC 9649 §2.3 / §2.4 / §2.7.1
  *writer* counterpart of the walker. New module `build` exposes:
  * [`build::build_chunk(fourcc, payload)`](src/build.rs) — generic
    §2.3 chunk writer (FourCC + Size LE + payload + odd-size pad).
  * [`build::build_vp8x_chunk(canvas_width, canvas_height, flags)`](src/build.rs) /
    the [`build_vp8x_chunk`](src/lib.rs) wrapper — §2.7.1 Figure 7
    10-byte payload writer (flags byte + zero-filled 24-bit reserved
    + 24-bit LE `(w-1)` + 24-bit LE `(h-1)`).
  * [`build::build_webp_file(payload, image_kind, w, h)`](src/build.rs) /
    the [`build_webp_file`](src/lib.rs) wrapper — §2.4 file writer
    for `ImageKind::{Lossy, Lossless, ExtendedLossy, ExtendedLossless}`.
    `ImageKind::Lossy` / `Lossless` emit the simple §2.5 / §2.6 layout
    (one `VP8 ` / `VP8L` chunk); the `Extended*` variants emit `VP8X`
    + bitstream per §2.7's chunk-ordering rule. The §2.4 `File Size`
    field is computed as `4 + body.len()` per the RFC.
  * Canvas-dim validation matches the parser's MUSTs symmetrically
    (zero / above 2^24 / product cap > 2^32 - 1).
  * Standalone-friendly: every public function compiles cleanly
    under `--no-default-features` (no `oxideav-core` dependency).
* **Typed `VP8 ` chunk handle (round 6):** RFC 9649 §2.5 routing
  layer for the simple-lossy `VP8 ` chunk. New module `vp8_chunk`
  exposes [`vp8_chunk::WebpLossyChunk`] +
  [`extract_lossy_chunk`](src/lib.rs). The handle borrows the chunk
  payload byte-for-byte and peeks the RFC 6386 §9.1 keyframe header
  (3-byte frame tag + 3-byte start code + 14-bit width / 14-bit
  height + 2-bit horizontal/vertical scale). Surfaces `width`,
  `height`, `version`, `show_frame`, `first_partition_size`,
  `horizontal_scale`, `vertical_scale` and exposes the routing slice
  via `bitstream()`. Refuses non-keyframe inputs per §2.5 and bad
  `0x9D 0x01 0x2A` start codes per §9.1. **No** runtime dependency on
  `oxideav-vp8`: the typed chunk is a hand-off surface that lets a
  caller route the bitstream to whichever VP8 decoder it wants.
* **Pixel decode (VP8 / VP8L / ALPH bitstream):** not implemented yet —
  [`decode_webp`](src/lib.rs) returns [`Error::NotImplemented`].
  Callers route the `VP8 ` payload via
  [`extract_lossy_chunk`](src/lib.rs) to an external VP8 decoder.
* **Registry hook:** [`register`](src/lib.rs) is a no-op; round 6
  still ships no decoder/encoder to the runtime context.

## What round 6 lands

| Item                          | Status                                              |
| ----------------------------- | --------------------------------------------------- |
| §2.4 WebP file header check   | done (`RIFF` + `File Size` + `WEBP`)                |
| §2.3 chunk walker             | done (FourCC, Size, payload range, odd-size pad)    |
| §2.5 simple lossy             | structural pass — `VP8 ` chunk surfaced             |
| §2.5 typed `VP8 ` handle      | **new** — RFC 6386 §9.1 keyframe peek + routing slice |
| §2.6 simple lossless          | structural pass — `VP8L` chunk surfaced             |
| §2.7 extended (VP8X et al.)   | structural pass — every documented FourCC surfaced  |
| §2.7.1 VP8X flag-byte parse   | done (`I`/`L`/`E`/`X`/`A` + `Rsv`/`R` ignored)      |
| §2.7.1 canvas dim decode      | done (24-bit LE × 2, 1-based)                       |
| §2.7.1 canvas product cap     | done (rejects `w*h > 2^32 - 1`)                     |
| §2.7.1.1 `ANIM` field parse   | done (BGRA u8×4 + u16 LE loop count)                |
| §2.7.1.1 `ANMF` field parse   | done (5×u24 LE + 6-bit Rsv + B + D info byte)       |
| §2.7.1.1 `ANMF` Frame Data    | not yet — sub-RIFF bytes after 16-byte header opaque |
| §2.7.1.2 `ALPH` info byte     | done (Rsv/P/F/C 2-bit decompose, typed enums)       |
| §2.7.1.2 `ALPH` bitstream     | not yet — bytes after info byte are out of scope    |
| §2.7.1.4 `ICCP`               | surfaced as opaque chunk                            |
| §2.7.1.5 `EXIF` / `XMP `      | surfaced as opaque chunks                           |
| §2.7.1.6 unknown chunks       | surfaced (no special handling required by §2.7.1.6) |
| §2.3 `build_chunk` writer     | done — generic FourCC+Size+payload+odd-pad emit     |
| §2.7.1 `build_vp8x_chunk`     | done — typed flag-byte + 24-bit LE × 2 emit         |
| §2.4 `build_webp_file`        | done — simple + extended `RIFF/WEBP` envelope       |
| VP8 / VP8L bitstream decode   | not yet — typed handle routes payload out-of-crate  |
| VP8 / VP8L bitstream encode   | not yet — payload is opaque input to the builder    |

Test count: **97** (82 unit + 15 integration against the
`docs/image/webp/fixtures/` corpus, including four new round-6
integration tests that pull a typed `WebpLossyChunk` out of the
real `lossy-1x1.webp` / `lossy-with-alpha-128x128.webp` fixtures
and verify the §9.1 dims / partition / scale fields against the
fixtures' `trace.txt` golden output, plus a `builder ↔ extractor`
round-trip and a `from_chunk` direct construction test).

## Clean-room sources

Rounds 1 through 6 were implemented entirely against:

* **RFC 9649** — WebP Image Format (`docs/image/webp/rfc9649-webp.txt`,
  also available as `rfc9649-webp.pdf`). Round 6 cites §2.5 (the
  simple-lossy file layout + its informative note that the VP8 frame
  header carries the canvas dimensions). Earlier rounds cited §2.3
  (generic RIFF chunk including the odd-size pad rule), §2.4
  (`File Size` field accounting), §2.6 (simple-lossless layout),
  §2.7 (extended-format chunk ordering), §2.7.1 Figure 7 (the VP8X
  10-byte payload layout), §2.7.1.1 Figure 8 (`ANIM`), §2.7.1.1
  Figure 9 (`ANMF`), and §2.7.1.2 Figure 10 (`ALPH`).
* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`). Round 6 cites §9.1
  ("Uncompressed Data Chunk") for the 3-byte frame tag layout
  (frame type / version / show_frame / first_partition_size), the
  3-byte sync code `0x9D 0x01 0x2A`, and the two 16-bit (scale << 14
  | dim) words that carry width and height. The typed
  `WebpLossyChunk` handle decodes exactly these fields — nothing
  past offset 10 of the VP8 payload is consulted.
* The 18-fixture corpus at `docs/image/webp/fixtures/` — consumed
  as opaque byte streams. Round 6 cross-checks the `lossy-1x1`
  trace's reported `width=1 height=1 partition_length=11 xscale=0
  yscale=0` and the `lossy-with-alpha-128x128` trace's reported
  `width=128 height=128 partition_length=328`. Round 5 round-trips
  the `lossy-1x1.webp` / `lossless-1x1.webp` fixtures' own `VP8 ` /
  `VP8L` payloads back through the new builder; round 4 cross-checked
  the ANMF u24 LE decode against `animated-with-alpha/trace.txt`.

No external library source — libwebp, libvpx, image-rs, webp-rs,
etc. — was consulted. `cwebp` / `dwebp` would be permissible as
black-box validators; rounds 1 through 6 did not invoke them
directly (round 6 reads only the fixture bytes already committed
to `docs/` / the in-crate `tests/data/`).

## License

MIT. See `LICENSE`.
