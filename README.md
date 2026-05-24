# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM + ANMF).

## Status — 2026-05-24 (clean-room round 112)

**The codec is now registered into `oxideav_core::RuntimeContext`.**
[`register`](src/lib.rs) (round 112) installs a `Decoder` factory under
the canonical `webp` codec id plus the `.webp` file-extension hint, so
the pipeline / `oxideav probe file.webp` can route a still WebP through
it. The new [`registry::WebpDecoder`](src/registry.rs) wraps
[`decode_webp_image`](src/lib.rs): each `send_packet` carries one whole
`RIFF/WEBP` file and `receive_frame` returns a single-planar
`Frame::Video` of interleaved 8-bit RGBA (`PixelFormat::Rgba`, stride
`width * 4`). The decoded `width` / `height` / pixel format are surfaced
on the decoder's [`CodecParameters`](src/registry.rs) after the first
frame. A §2.5 `VP8 ` lossy file — and any animation / header-only file
with no `VP8L`/`VP8 ` image-data chunk — is a clean
`oxideav_core::Error::Unsupported`; lossy callers route the chunk via
[`extract_lossy_chunk`](src/lib.rs) to a downstream VP8 decoder. The
codec also claims the `WEBP` FourCC for tag-based resolution. The
default-on `registry` feature gates the whole module, so the standalone
(`--no-default-features`) build path stays free of `oxideav-core`.

**The top-level still-image decode was wired up in round 111.**
[`decode_webp_image`](src/lib.rs) walks a `RIFF/WEBP` file and decodes a
§2.6 / §3.4 `VP8L` lossless image — simple **or** `VP8X`-extended — all
the way to a `DecodedWebp { width, height, rgba }`, where `rgba` is
`width*height*4` interleaved `[R, G, B, A]` bytes (the
`oxideav_core::PixelFormat::Rgba` layout). When a (spec-discouraged, per
§2.7.1.2 "SHOULD NOT") `ALPH` chunk accompanies the `VP8L` image, its
decoded alpha plane overrides the per-pixel alpha.
[`decode_webp`](src/lib.rs) is the flat-buffer shorthand. A §2.5 `VP8 `
lossy file is a clean `Error::Unsupported(LossyVp8)` — routed onward via
[`extract_lossy_chunk`](src/lib.rs) rather than stub-decoded.

The **VP8L lossless decode path is complete end-to-end, and the
§2.7.1.2 `ALPH` alpha bitstream decodes end-to-end too.**
[`decode_lossless_image`](src/lib.rs) walks a `RIFF/WEBP` file, extracts
the `VP8L` chunk, runs the §5/§6 entropy decode, and applies the §4
inverse-transform chain — validated bit-exact against the `lossless-1x1`,
`lossless-color-indexing-paletted`, and `lossless-32x32-rgba` fixture
PNGs (the last exercising SUBTRACT_GREEN + PREDICTOR + CROSS_COLOR + a
color cache simultaneously).
[`decode_alpha_plane`](src/lib.rs) decodes the `ALPH` chunk to a
full-resolution 8-bit alpha plane — both compression methods (raw +
headerless VP8L, the latter reusing the lossless decoder and lifting
alpha from the green channel) and all four §2.7.1.2 inverse filters —
validated bit-exact (all 16384 bytes) against `dwebp -alpha` on the
`lossy-with-alpha-128x128` fixture.


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
  `LevelReduction` / `Reserved(u8)`.
* **ALPH alpha bitstream decode (round 110):** RFC 9649 §2.7.1.2
  end-to-end. [`alph::decode_alpha`](src/alph.rs) decodes a whole
  `ALPH` payload to a `width * height` 8-bit alpha plane: compression
  method 0 (raw bytes) and method 1 (a *headerless* §3 VP8L
  image-stream of implicit dimensions, decoded by
  [`vp8l_transform::decode_lossless_headerless`](src/vp8l_transform.rs)
  with the alpha taken from the green channel), then the §2.7.1.2
  inverse filter (none / horizontal / vertical / gradient,
  `alpha = (predictor + X) % 256`, with the top-left / left-most /
  top-most edge cases). [`decode_alpha_plane`](src/lib.rs) is the
  container-level entry point — it sources the plane dimensions from
  `VP8X` (or the `VP8 ` keyframe) and returns `Ok(None)` for files with
  no `ALPH` chunk. The §2.7.1.2 preprocessing hint is informational and
  is not applied.
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
* **Typed `VP8L` chunk handle (round 7):** RFC 9649 §2.6 routing
  layer for the simple-lossless `VP8L` chunk. New module `vp8l_chunk`
  exposes [`vp8l_chunk::WebpLosslessChunk`] +
  [`extract_lossless_chunk`](src/lib.rs). The handle borrows the
  chunk payload byte-for-byte and decodes the §3.4 / §7.1 5-byte
  image-header (one-byte `0x2F` signature + LE bit-packed 14-bit
  `width - 1` + 14-bit `height - 1` + `alpha_is_used` bit + 3-bit
  `version`). Surfaces resolved 1-based `width`, `height` plus raw
  `alpha_is_used`, `version` and exposes the routing slice via
  `bitstream()`. Refuses short payloads and bad `0x2F` signatures;
  surfaces non-zero `version` for downstream policy (§3.4 says the
  *decoder* "should treat as an error" — out of scope for this
  router). **No** runtime dependency on `oxideav-vp8l`: the typed
  chunk is a hand-off surface that lets a caller route the
  bitstream to whichever VP8L decoder it wants.
* **VP8L bit-reader + §4 transform-list reader (round 99):** new
  module `vp8l_stream`. [`vp8l_stream::BitReader`] implements the
  WebP-Lossless §2 `ReadBits(n)` primitive — bytes in stream order,
  bits least-significant-bit-first, the first bit read becoming bit 0
  of the returned integer. [`vp8l_stream::TransformList::read`] walks
  the §4 `while (ReadBits(1))` transform-presence loop and decodes
  each present transform's leading fixed-size fields: the §4.1 / §4.2
  `size_bits = ReadBits(3) + 2` block-size for `Predictor` / `Color`,
  nothing for `SubtractGreen` (§4.3), and the §4.4
  `color_table_size = ReadBits(8) + 1` plus the derived pixel-bundling
  `width_bits` for `ColorIndexing`. §4's "each transform used only
  once" rule is enforced (`DuplicateTransform`). The reader **stops**
  at the first transform that carries a §5 entropy-coded body (a
  sub-resolution image or color table it cannot yet decode) and
  records the bit offset via
  [`TransformList::body_bit_position`] so the next-round §5 reader
  resumes exactly there; `SubtractGreen` (bodyless) lets the loop
  continue. Top-level [`read_vp8l_transform_list`](src/lib.rs) walks
  the container, extracts the `VP8L` chunk, and returns the parsed
  list. Standalone-friendly (compiles under `--no-default-features`).
* **VP8L §5.2.3 color-cache info + §6.2.2 meta-prefix + §6.2 prefix-code
  group reader (round 106):** new module `meta_prefix`.
  [`meta_prefix::PrefixCodeGroup`] bundles the five §6.2 prefix codes
  (GREEN+length+color-cache / RED / BLUE / ALPHA / DIST) the §5.2.1 /
  §5.2.2 / §5.2.3 decode paths consume, and
  [`PrefixCodeGroup::read`](src/meta_prefix.rs) reads them in §6.2
  bitstream order via the round-104 [`vp8l_prefix::PrefixCode`].
  [`meta_prefix::ColorCacheInfo`] reads the §5.2.3 `color-cache-info`
  field (the leading 1-bit dispatch plus the optional 4-bit
  `color_cache_code_bits`), validates the §5.2.3 `[1..11]` range, and
  surfaces `is_enabled()` / `size()` (`1 << code_bits`).
  [`meta_prefix::MetaPrefixHeader::read`](src/meta_prefix.rs) is the
  combined preamble reader for one §5 image-data block: given the
  [`meta_prefix::ImageRole`] (`Argb` carries the §6.2.2 meta-prefix
  bit, `EntropyCoded` does not — matching §5.1 + §7.3 ABNF), it
  consumes the color-cache info, then dispatches:
  * `meta-prefix = %b0` (or non-ARGB role): immediately reads the
    single 5-code [`PrefixCodeGroup`] and returns
    [`meta_prefix::MetaPrefixCodes::Single`].
  * `meta-prefix = %b1 entropy-image` (ARGB only): reads the §6.2.2
    `prefix_bits = ReadBits(3) + 2` field (range `[2..9]`), derives
    entropy-image `DIV_ROUND_UP(image_dim, 1 << prefix_bits)`
    dimensions, records the bit position of the entropy-image start,
    and returns [`meta_prefix::MetaPrefixCodes::EntropyImagePending`].
    The entropy image itself is a §5.2-encoded
    `entropy-coded-image` whose decode lives in the next layer; this
    reader records the boundary for that layer to resume from (same
    pattern round 99 used to stop at the first §5 transform body and
    round 104 used to resume at it).
  Boxes the `PrefixCodeGroup` inside the `Single` variant to keep the
  enum compact (per `clippy::large_enum_variant`). Standalone-friendly
  (compiles under `--no-default-features`). The next remaining lossless
  layer is §5.2 LZ77 backward-reference + §5.2.3 color-cache *symbol*
  decode — the per-pixel decoder that consumes symbols from a
  `PrefixCodeGroup`.
* **VP8L §4 inverse-transform passes (round 109):** new module
  `vp8l_transform`. [`vp8l_transform::decode_lossless`] is the top-level
  driver — it reads the §4 / §7.2 `optional-transform` list (each
  transform's fixed fields **and** its §5-encoded `entropy-coded-image`
  body, decoded via the new
  [`vp8l_decode::decode_entropy_coded_image`]), tracks §4.4 width
  subsampling, decodes the main §5.1 ARGB image at the subsampled width
  via [`vp8l_decode::decode_argb`], then applies the inverse transforms
  in reverse read order (§4: "last one first"):
  * §4.1 predictor ([`vp8l_transform::inverse_predictor`]) — the 14
    prediction modes (`Average2` / `Select` / `ClampAddSubtractFull` /
    `ClampAddSubtractHalf`) over the TL/T/TR/L block grid, with the
    border rules (top-left → `0xff000000`, top row → L, left column →
    T, rightmost column uses the row's leftmost pixel as TR) and the
    per-channel residual add.
  * §4.2 color ([`vp8l_transform::inverse_color`]) — per-block
    `ColorTransformElement` add-back (`ColorTransformDelta(t,c) =
    (t*c) >> 5`, signed 8-bit), on red and blue only.
  * §4.3 subtract-green ([`vp8l_transform::inverse_subtract_green`]) —
    add green into red and blue.
  * §4.4 color-indexing ([`vp8l_transform::inverse_color_table`] +
    [`vp8l_transform::inverse_color_indexing`]) — subtraction-decode of
    the palette, palette lookup, ≤16-color pixel un-bundling (2/4/8
    indices per green byte), width un-subsample to the canvas width,
    out-of-range indices → transparent black.
  [`decode_lossless_image`](src/lib.rs) is the container-level entry
  point. Standalone-friendly (compiles under `--no-default-features`).
* **VP8L §6.2.2 entropy-image multi-group ARGB decode (round 108):**
  [`vp8l_decode::decode_argb`] is the full ARGB-role decode. It reads
  the round-106 [`meta_prefix::MetaPrefixHeader`] for the `Argb` role
  and, when the §6.2.2 meta-prefix bit is set, decodes the *entropy
  image* — itself a §5 `entropy-coded-image` of size
  `DIV_ROUND_UP(width, 1<<prefix_bits)` ×
  `DIV_ROUND_UP(height, 1<<prefix_bits)` — via
  [`vp8l_decode::decode_entropy_image`] into a
  [`vp8l_decode::MetaPrefixIndex`]. Per §6.2.2 each block's meta-prefix
  code is the red+green channels of its entropy-image pixel
  (`(argb >> 8) & 0xffff`); `num_prefix_groups = max(entropy image) + 1`
  (the *maximum* code plus one, not the block count). The decoder reads
  that many [`meta_prefix::PrefixCodeGroup`]s, then runs the §6.2.3
  per-pixel loop selecting a group per block via
  `meta_index[(y >> prefix_bits) * block_width + (x >> prefix_bits)]`.
  A single §5.2.3 color cache is threaded across all groups in stream
  order. The single-group case (meta-prefix bit zero) degrades to the
  round-107 [`vp8l_decode::decode_image`] path. The §6.2.3 per-pixel
  core is now a shared helper used by both the single- and multi-group
  loops, so the literal / LZ77 / color-cache dispatch is identical in
  both. Standalone-friendly (compiles under `--no-default-features`).
  This completes the §5 / §6 lossless entropy stage; the remaining
  lossless work is the §4 inverse-transform passes that consume this
  decode's ARGB buffer.
* **VP8L §5.2 per-pixel ARGB decode loop (round 107):** new module
  `vp8l_decode`. [`vp8l_decode::decode_image`] runs the §6.2.3
  per-pixel decode loop over a single
  [`meta_prefix::PrefixCodeGroup`], producing a [`vp8l_decode::DecodedImage`]
  of `width * height` ARGB pixels in scan-line order (before any §4
  inverse transform). The GREEN symbol `S` from prefix code #1 is
  dispatched by range: `S < 256` is a §5.2.1 literal (green=`S`; red /
  blue / alpha from prefix codes #2 / #3 / #4); `256 <= S < 280` is a
  §5.2.2 LZ77 length prefix code; `S >= 280` is a §5.2.3 color-cache
  code (`S - 280` indexes the cache). [`vp8l_decode::read_lz77_value`]
  implements the §5.2.2 prefix → value transform shared by length and
  distance (`prefix < 4 → prefix + 1`, else `offset + ReadBits(extra)
  + 1`). [`vp8l_decode::DISTANCE_MAP`] is the 120-entry §5.2.2
  neighbor-offset table and
  [`vp8l_decode::distance_code_to_pixel_distance`] maps a raw distance
  code to a scan-line distance (`dist = xi + yi*width`, clamped to 1;
  codes `> 120` are `code - 120`). [`vp8l_decode::ColorCache`]
  implements the §5.2.3 cache: zero-initialized, hashed by
  `(0x1e35a7bd * argb) >> (32 - code_bits)`, with every emitted pixel
  re-inserted in stream order. Backward references that underflow the
  decoded prefix or overrun the image are refused; overlapping copies
  (dist < length) repeat the just-copied pixels per standard LZ77.
  [`vp8l_decode::GreenSymbol::classify`] is the §6.2.3 range dispatch,
  unit-testable in isolation. Standalone-friendly (compiles under
  `--no-default-features`). This closes the §5.2 single-group decode
  path; the remaining lossless work is the §6.2.2 entropy-image
  *multi-group* path (one `PrefixCodeGroup` per pixel block) plus the
  §4 inverse-transform passes that consume this loop's ARGB buffer.
* **VP8L §6.2.1 prefix-code reader + canonical decoder (round 104):**
  new module `vp8l_prefix`. [`vp8l_prefix::PrefixCode`] is a built
  canonical prefix code over an alphabet of a given size.
  [`PrefixCode::read`](src/vp8l_prefix.rs) reads one code's lengths off
  the wire — dispatching on the §6.2.1 leading 1-bit flag between the
  "Simple Code Length Code" (1–2 symbols, each length 1) and the
  "Normal Code Length Code" (the 19-symbol code-length-code read in
  `kCodeLengthCodeOrder`, the `max_symbol` gate, and the literal
  `[0..15]` / repeat-`16` / zero-run-`17`/`18` expansion) — and builds
  the decoder; [`PrefixCode::from_code_lengths`](src/vp8l_prefix.rs)
  builds straight from a per-symbol length table.
  [`PrefixCode::read_symbol`](src/vp8l_prefix.rs) decodes one symbol at
  a time, MSB-first within a code, matching the canonical
  `(length, value)` assignment the spec's `[Huffman]` reference fixes.
  The §6.2.1 single-leaf-node tree (one symbol at length 1, reading
  consumes no bits) is handled, and the completeness rule
  (`sum 2^-len == 1`) is enforced with integer Kraft arithmetic —
  over-/under-subscribed codes are refused. A new
  [`vp8l_stream::BitReader::seek_to_bit`](src/vp8l_stream.rs) lets a
  caller resume reading at a recorded boundary (e.g.
  [`TransformList::body_bit_position`]). Standalone-friendly (compiles
  under `--no-default-features`). The §6.2.1 reader is the foundation
  every §5 / §6 consumer needs; §6.2.2 (meta prefix codes / entropy
  image) and §5.2 (the LZ77 + color-cache pixel stream) are next.
* **Pixel decode (lossless VP8L):** **done** —
  [`decode_lossless_image`](src/lib.rs) decodes a `VP8L` chunk all the
  way to ARGB pixels (round 109). The generic [`decode_webp`](src/lib.rs)
  / [`decode_webp_image`](src/lib.rs) entries are now wired (round 111):
  they decode a simple or `VP8X`-extended `VP8L` file to packed
  `[R, G, B, A]` bytes, applying an accompanying `ALPH` plane when
  present. A `VP8 ` lossy file returns `Error::Unsupported(LossyVp8)`;
  callers route the `VP8 ` payload via
  [`extract_lossy_chunk`](src/lib.rs) to an external VP8 decoder.
  Round 99 landed the §4 transform list (first step of the lossless
  pixel path); round 104 landed the §6.2.1 canonical-prefix-code
  reader (the entropy primitive every §5 / §6 consumer needs);
  round 106 landed the §5.2.3 + §6.2.2 + §6.2 preamble that
  assembles five-code prefix-code groups and resolves the
  ARGB-role meta-prefix dispatch; round 107 lands the §5.2 per-pixel
  ARGB decode loop (`vp8l_decode::decode_image`) — §6.2.3 GREEN
  dispatch, §5.2.2 LZ77 length/distance + the 120-element distance
  map, and the §5.2.3 color cache — which turns a single
  `PrefixCodeGroup` plus the §5.2 data into a decoded ARGB pixel
  buffer. The §6.2.2 entropy-image *multi-group* path and the §4
  inverse-transform passes (which consume this loop's output) are
  next.
* **Registry hook (round 112):** [`register`](src/lib.rs) now installs
  a real `oxideav_core::Decoder` factory under the `webp` codec id (plus
  the `.webp` extension hint and a `WEBP` FourCC tag claim). The
  [`registry::WebpDecoder`](src/registry.rs) decodes a still `VP8L`
  file — simple or `VP8X`-extended, with optional §2.7.1.2
  `ALPH`-over-`VP8L` alpha override — to a single-planar
  `PixelFormat::Rgba` `VideoFrame`; the §2.5 `VP8 ` lossy path and
  animation / header-only files surface as
  `oxideav_core::Error::Unsupported`. No encoder factory is registered
  (the builders stay framing-only). Gated behind the default-on
  `registry` feature.

## What round 112 lands

| Item                                              | Status                                                |
| ------------------------------------------------- | ----------------------------------------------------- |
| `Decoder` impl over `decode_webp_image`           | **new** — `registry::WebpDecoder` (packet → RGBA frame) |
| `register()` installs decoder factory + tag       | **new** — `CodecInfo` under `webp` id + `WEBP` FourCC |
| `.webp` extension hint                            | **new** — `register_containers`                       |
| RGBA `VideoFrame` (single plane, stride `w*4`)    | **new** — `PixelFormat::Rgba`                         |
| dims / pixel format on `CodecParameters`          | **new** — refreshed after first `receive_frame`       |
| `VP8 ` lossy → `Error::Unsupported`               | **new** — via the registered decoder path             |
| animation / header-only → `Error::Unsupported`    | **new** — `NoImageData` → core `Unsupported`          |
| `Error` → `oxideav_core::Error` bridge            | **new** — `Unsupported` maps to core `Unsupported`    |
| `decode_webp_to_frame` direct helper              | **new** — framework-flavoured wrapper                 |

## What round 111 lands

| Item                                              | Status                                                |
| ------------------------------------------------- | ----------------------------------------------------- |
| Top-level still decode → RGBA                      | **new** — `decode_webp` / `decode_webp_image`         |
| `DecodedWebp { width, height, rgba }`             | **new** — packed `[R, G, B, A]` (`PixelFormat::Rgba`) |
| Simple + `VP8X`-extended `VP8L` dispatch          | **new** — both walk to the same ARGB decode           |
| `ALPH`-over-`VP8L` alpha override                 | **new** — §2.7.1.2 plane replaces per-pixel alpha     |
| `VP8 ` lossy → clean refusal                       | **new** — `Error::Unsupported(LossyVp8)` (not stubbed) |
| ARGB → interleaved RGBA repack                     | **new** — `argb_to_rgba`                              |

## What round 109 lands

| Item                                              | Status                                                |
| ------------------------------------------------- | ----------------------------------------------------- |
| §4 transform list + bodies (driver)               | **new** — `decode_lossless` reads each tx + its body  |
| §7.3 `entropy-coded-image` decode helper          | **new** — `decode_entropy_coded_image`                |
| §4.1 predictor (14 modes + border rules)          | **new** — `inverse_predictor`                         |
| §4.2 color (`ColorTransformDelta` add-back)       | **new** — `inverse_color`                             |
| §4.3 subtract-green (add green to R/B)            | **new** — `inverse_subtract_green`                    |
| §4.4 color-table subtraction decode               | **new** — `inverse_color_table`                       |
| §4.4 palette lookup + ≤16-color un-bundling       | **new** — `inverse_color_indexing`                    |
| §4 reverse-order inverse application               | **new** — `decode_lossless` (last-read-first)         |
| §4.4 width subsampling across the chain            | **new** — tracked, then un-subsampled on inverse      |
| End-to-end `VP8L` → ARGB                           | **new** — `decode_lossless_image` (container entry)   |

## What round 108 lands

| Item                                              | Status                                                |
| ------------------------------------------------- | ----------------------------------------------------- |
| §6.2.2 entropy-image decode → meta-index          | **new** — `decode_entropy_image` → `MetaPrefixIndex`  |
| §6.2.2 meta-prefix code = `(argb >> 8) & 0xffff`  | **new** — red+green channels per block                |
| §6.2.2 `num_prefix_groups = max(entropy)+1`       | **new** — `MetaPrefixIndex::num_prefix_groups`        |
| §6.2.2 read `num_prefix_groups` prefix groups     | **new** — `decode_argb` group-array read              |
| §6.2.2 per-pixel group selection                  | **new** — `MetaPrefixIndex::meta_code_for`            |
| §6.2.2 full multi-group ARGB decode               | **new** — `decode_argb` (`EntropyImagePending` arm)   |
| §6.2.2 single-group ARGB decode (`%b0`)           | **new** — `decode_argb` (`Single` arm) → `decode_image` |
| Shared §6.2.3 per-pixel core (single + multi)     | **new** — `decode_one_symbol`                         |
| Degenerate / out-of-range meta-index refusal      | **new** — `EmptyEntropyImage` / `MetaPrefixIndexOutOfRange` |

## What round 107 lands

| Item                                              | Status                                                |
| ------------------------------------------------- | ----------------------------------------------------- |
| §6.2.3 GREEN symbol dispatch (`S<256` literal)    | **new** — `GreenSymbol::classify` (`vp8l_decode`)     |
| §5.2.1 literal R/B/A from prefix #2/#3/#4         | **new** — `decode_image` literal arm                  |
| §6.2.3 length codes `256..280`                    | **new** — `GreenSymbol::LengthPrefix`                 |
| §6.2.3 color-cache codes `>= 280`                 | **new** — `GreenSymbol::ColorCache`                   |
| §5.2.2 LZ77 prefix → value transform              | **new** — `read_lz77_value` (length + distance)       |
| §5.2.2 120-element distance map                   | **new** — `DISTANCE_MAP` + `distance_code_to_pixel_distance` |
| §5.2.2 distance code `> 120` (offset by 120)      | **new** — scan-line distance path                     |
| §5.2.2 backward copy + LZ77 self-overlap          | **new** — `decode_image` length arm                   |
| §5.2.3 color cache (`0x1e35a7bd` hash)            | **new** — `ColorCache::{new,hash,insert,lookup}`      |
| §5.2.3 insert-every-pixel in stream order         | **new** — literal / copy / cache-hit all re-insert    |
| §5.2 single-group ARGB decode → pixel buffer      | **new** — `decode_image` → `DecodedImage`             |
| Underflow / overflow backward-ref refusal         | **new** — `BackwardReference{Underflow,Overflow}`     |

## What round 106 lands

| Item                                          | Status                                                |
| --------------------------------------------- | ----------------------------------------------------- |
| §5.2.3 color-cache info (`%b0` / `%b1 4BIT`)  | **new** — `ColorCacheInfo::read` (`meta_prefix`)      |
| §5.2.3 `color_cache_code_bits` `[1..11]` gate | **new** — `InvalidColorCacheCodeBits` refusal         |
| §6.2 5-prefix-code-group reader               | **new** — `PrefixCodeGroup::read` in §6.2 order       |
| §6.2.3 GREEN alphabet `256 + 24 + cache_size` | **new** — `green_alphabet_size(cache_size)`           |
| §6.2.2 ARGB-role meta-prefix bit              | **new** — read iff `role == Argb`                     |
| §6.2.2 entropy-coded-image role drops bit     | **new** — drops straight to single group              |
| §6.2.2 single-group dispatch (`%b0`)          | **new** — `MetaPrefixCodes::Single`                   |
| §6.2.2 multi-group dispatch (`%b1`)           | **new** — `MetaPrefixCodes::EntropyImagePending`      |
| §6.2.2 `prefix_bits = ReadBits(3) + 2`        | **new** — range `[2..9]`                              |
| §6.2.2 `DIV_ROUND_UP` entropy-image dims      | **new** — `n.div_ceil(1 << prefix_bits)`              |
| Entropy-image §5.2 boundary recording         | **new** — `entropy_image_bit_position` for next round |

## What round 104 lands

| Item                                  | Status                                                |
| ------------------------------------- | ----------------------------------------------------- |
| §6.2.1 prefix-code dispatch (1-bit)   | **new** — simple/normal flag (`vp8l_prefix`)          |
| §6.2.1 Simple Code Length Code        | **new** — 1–2 symbols at length 1 (1-bit or 8-bit)    |
| §6.2.1 Normal Code Length Code        | **new** — 19-sym CLC in `kCodeLengthCodeOrder`        |
| §6.2.1 `max_symbol` gate              | **new** — alphabet default + 2 + ReadBits(length_nbits) |
| §6.2.1 literal lengths `[0..15]`      | **new** — direct length emit                          |
| §6.2.1 repeat code `16`               | **new** — `3 + ReadBits(2)` previous-non-zero replay  |
| §6.2.1 zero-run codes `17` / `18`     | **new** — `3 + ReadBits(3)` / `11 + ReadBits(7)`      |
| §6.2.1 canonical decoder              | **new** — `(length, value)` order, MSB-first reads    |
| §6.2.1 single-leaf-node exception     | **new** — reading consumes no bits                    |
| §6.2.1 completeness (Kraft = 1)       | **new** — integer arithmetic; over/under refused      |
| `BitReader::seek_to_bit`              | **new** — resume at recorded body boundaries          |

## What round 99 lands

| Item                          | Status                                              |
| ----------------------------- | --------------------------------------------------- |
| §2 VP8L `ReadBits(n)` reader  | **new** — LSB-first bit reader (`vp8l_stream`)      |
| §4 transform-presence loop    | **new** — `while (ReadBits(1))` + 2-bit type        |
| §4.1/§4.2 predictor/color `size_bits` | **new** — `ReadBits(3) + 2` leading field   |
| §4.3 subtract-green           | **new** — bodyless, loop continues past it          |
| §4.4 color-indexing fields    | **new** — `color_table_size` + `width_bits`         |
| §4 "used only once" rule      | **new** — `DuplicateTransform` refusal              |
| §4 transform §5 body          | **r104 reader** consumes the §6.2.1 prefix-code start |
| §2.4 WebP file header check   | done (`RIFF` + `File Size` + `WEBP`)                |
| §2.3 chunk walker             | done (FourCC, Size, payload range, odd-size pad)    |
| §2.5 simple lossy             | structural pass — `VP8 ` chunk surfaced             |
| §2.5 typed `VP8 ` handle      | **new** — RFC 6386 §9.1 keyframe peek + routing slice |
| §2.6 simple lossless          | structural pass — `VP8L` chunk surfaced             |
| §2.6 typed `VP8L` handle      | **new** — §3.4 / §7.1 image-header peek + routing slice |
| §2.7 extended (VP8X et al.)   | structural pass — every documented FourCC surfaced  |
| §2.7.1 VP8X flag-byte parse   | done (`I`/`L`/`E`/`X`/`A` + `Rsv`/`R` ignored)      |
| §2.7.1 canvas dim decode      | done (24-bit LE × 2, 1-based)                       |
| §2.7.1 canvas product cap     | done (rejects `w*h > 2^32 - 1`)                     |
| §2.7.1.1 `ANIM` field parse   | done (BGRA u8×4 + u16 LE loop count)                |
| §2.7.1.1 `ANMF` field parse   | done (5×u24 LE + 6-bit Rsv + B + D info byte)       |
| §2.7.1.1 `ANMF` Frame Data    | not yet — sub-RIFF bytes after 16-byte header opaque |
| §2.7.1.2 `ALPH` info byte     | done (Rsv/P/F/C 2-bit decompose, typed enums)       |
| §2.7.1.2 `ALPH` bitstream     | done — raw + headerless VP8L (green channel) + 4 inverse filters |
| §2.7.1.4 `ICCP`               | surfaced as opaque chunk                            |
| §2.7.1.5 `EXIF` / `XMP `      | surfaced as opaque chunks                           |
| §2.7.1.6 unknown chunks       | surfaced (no special handling required by §2.7.1.6) |
| §2.3 `build_chunk` writer     | done — generic FourCC+Size+payload+odd-pad emit     |
| §2.7.1 `build_vp8x_chunk`     | done — typed flag-byte + 24-bit LE × 2 emit         |
| §2.4 `build_webp_file`        | done — simple + extended `RIFF/WEBP` envelope       |
| VP8L bitstream decode         | done — `decode_webp` / `decode_lossless_image` → RGBA |
| `oxideav_core::Decoder` registration | done (r112) — `register()` installs the `webp` decoder factory |
| VP8 lossy bitstream decode    | not yet — typed handle / decoder route payload out-of-crate |
| VP8 / VP8L bitstream encode   | not yet — payload is opaque input to the builder    |

Test count: **257** (213 unit + 44 integration against the
`docs/image/webp/fixtures/` corpus). Round 112 adds 10 unit tests inside
`registry::tests` — `register(&mut ctx)` installs the decoder factory
(and not an encoder) plus the `.webp` extension hint; the `WEBP` FourCC
resolves to the `webp` codec id; `first_decoder` returns a `WebpDecoder`;
an end-to-end `RuntimeContext → first_decoder → send_packet →
receive_frame` decode of `lossless-1x1.webp` yields the expected RGBA
frame and then `NeedMore`; a `VP8 ` lossy packet surfaces as
`Error::Unsupported` through the registered decoder; the decoder's
`CodecParameters` carry the decoded dims + `PixelFormat::Rgba` after the
first frame; a double `send_packet` without a `receive_frame` is
rejected; post-`flush` with no pending packet returns `Eof`;
`decode_webp_to_frame` produces an RGBA `VideoFrame`; and the
`Error → oxideav_core::Error` conversion maps both `Unsupported`
variants to the core `Unsupported`. Round 111 adds 7 integration tests
covering the top-level `decode_webp` / `decode_webp_image` entries
(simple `VP8L`, color-indexed palette, RGBA-alpha repack, synthesized
`VP8X`+`VP8L`, hand-assembled `VP8X`+`VP8L`+`ALPH` override, and the two
`VP8 ` lossy → `Unsupported` refusals). Round 109 adds 18 unit tests
inside `vp8l_transform::tests` (each §4.1 predictor primitive —
`Average2` / `Clamp` / `ClampAddSubtract{Full,Half}` / `Select` /
`predict`; the predictor border rules for the top-left, top-row, and
left-column cases; the §4.2 signed `ColorTransformDelta` + a
forward↔inverse round-trip + in-place block use; §4.3 green add-back
with wrap; §4.4 color-table subtraction decode + no-bundling lookup +
out-of-range → transparent black + width_bits-1 and width_bits-3
bundling + the threshold table) plus 4 integration tests in
`fixture_walks` that decode three real fixtures *bit-exactly* against
their `expected.png` ARGB ground truth
(`round109_lossless_1x1_color_indexing_decodes_end_to_end` →
`0xFFB43C5A`,
`round109_lossless_color_indexing_paletted_decodes_end_to_end` (8-color
palette, width_bits=1),
`round109_lossless_32x32_rgba_full_transform_chain_decodes_end_to_end`
(SUBTRACT_GREEN + PREDICTOR + CROSS_COLOR + color cache), plus the
`returns_none_for_lossy_file` guard) and a new in-crate fixture
`tests/data/lossless-color-indexing-paletted.webp`. Round 108 adds 9 unit tests
inside `vp8l_decode::tests` (`MetaPrefixIndex` selection + max-based
`num_prefix_groups`; entropy-image red+green meta-code extraction
including the high-code red-channel path; two-group per-block decode;
single-group `decode_argb`; single-group parity with `decode_image`;
multi-group with a shared color cache; zero-dim entropy-image refusal)
plus 3 integration tests in `fixture_walks` (public `decode_argb`
multi-group + single-group, public `decode_entropy_image` with the
max-based group count). Round 107 adds 24 unit tests
inside `vp8l_decode::tests` (§5.2.2 LZ77 value transform across prefix
codes 0–6 and the length-4096 boundary at prefix 23; distance map
length / first-entry spec examples / above-120 offset / negative-offset
clamp; §6.2.3 GREEN symbol literal / length / color-cache range
classification + out-of-range refusal; §5.2.3 color-cache hash formula
/ insert-lookup round-trip / zero-initialization; full decode loop for
a literal-only 2×1 image, a single literal pixel, a length/distance
back-reference with LZ77 self-overlap, a color-cache hit, plus the
backward-reference-underflow and no-cache-cache-code refusals) plus 2
integration tests:
* `round107_lossless_1x1_color_table_decodes_end_to_end_to_palette_pixel`
  drives the full pipeline — container walk → §4 transform list →
  resume at the COLOR_INDEXING §5 body → §5.2.3 + §6.2 meta-prefix
  header → `decode_image` — over `lossless-1x1.webp`'s 1×1 color-table
  image, producing the single palette pixel ARGB `0xFFB43C5A`
  (255,180,60,90) straight from the fixture's own VP8L payload bytes.
* `round107_decode_error_surfaces_through_crate_error` locks the
  `DecodeError → oxideav_webp::Error::Vp8lDecode` `From` wiring.

Round 106 adds 15 unit tests
inside `meta_prefix::tests` (color-cache info disabled / enabled at
`code_bits` 1 / 11 / 0-refused / 12-refused, GREEN alphabet size
formula across cache sizes, group read order matches §6.2,
EntropyCoded role skips meta-prefix bit, ARGB-role single-group
read, ARGB-role multi-group entropy-image boundary + bit position,
ARGB-role `DIV_ROUND_UP` rounding, ARGB-role max `prefix_bits=9`,
ARGB-role color-cache propagates into GREEN alphabet size,
truncated `ColorCacheInfo` EOF, truncated `MetaPrefixHeader` EOF)
plus 3 integration tests:
* `round106_lossless_1x1_color_table_meta_prefix_header_reads_single_group`
  reads the COLOR_INDEXING transform's color-table image (an
  `entropy-coded-image` role) and asserts the meta-prefix header
  surfaces the same 5-code group r104 cracked open by hand —
  GREEN=60 / RED=180 / BLUE=90 / ALPHA=255 / DIST=0.
* `round106_meta_prefix_argb_single_group_synthetic_matches_trace_shape`
  exercises the ARGB-role single-group path (`color_cache_bits=0`,
  `meta_huffman=0`, `num_htree_groups=1`) — the shape every
  fixture trace shows when no entropy image is in play.
* `round106_meta_prefix_argb_multi_group_records_entropy_image_boundary`
  exercises the ARGB-role multi-group path (`prefix_bits=4` over a
  128×128 image), asserts the 8×8 entropy-image dimensions and the
  recorded entropy-image bit position.

Round 104 added 16 unit tests inside `vp8l_prefix::tests` (single-leaf
no-bit read, two-symbol canonical assignment, the classic `[1,2,3,3]`
canonical example decoded in value order, over-subscribed / incomplete
/ empty / length-too-large refusals, simple 1-bit / 8-bit / two-symbol
codes, simple symbol-out-of-range refusal, normal CLC with direct
lengths, normal zero-run `18`, normal repeat `16`, normal
max_symbol-too-large refusal, truncated-code EOF) + 1
`vp8l_stream::tests::seek_to_bit_repositions_and_clamps` + 1
integration test
(`round104_lossless_1x1_color_table_prefix_group_matches_fixture_bytes`)
that resumes at the COLOR_INDEXING §5 body of `lossless-1x1.webp`,
reads the §5 color-cache info bit (0, matching the fixture trace's
`color_cache_bits=0`) and the full 5-code prefix group, and asserts
the single symbols GREEN=60 / RED=180 / BLUE=90 / ALPHA=255 / DIST=0
(the single ARGB palette color 255,180,60,90) decoded purely from
the fixture's own VP8L payload bytes.

## Clean-room sources

Rounds 1 through 109 were implemented entirely against:

* **RFC 9649** — WebP Image Format (`docs/image/webp/rfc9649-webp.txt`,
  also available as `rfc9649-webp.pdf`). Round 7 cites §2.6 (the
  simple-lossless file layout + its informative note that the VP8L
  header carries the canvas dimensions). Round 6 cited §2.5 (the
  simple-lossy file layout). Earlier rounds cited §2.3 (generic
  RIFF chunk including the odd-size pad rule), §2.4 (`File Size`
  field accounting), §2.7 (extended-format chunk ordering), §2.7.1
  Figure 7 (the VP8X 10-byte payload layout), §2.7.1.1 Figure 8
  (`ANIM`), §2.7.1.1 Figure 9 (`ANMF`), and §2.7.1.2 Figure 10
  (`ALPH`).
* **WebP Lossless Bitstream Specification** — `docs/image/webp/
  google-webp-lossless-bitstream.html` (also reproduced in RFC 9649
  §3). Round 7 cites §3.4 ("RIFF Header") for the on-wire layout
  (one-byte `0x2F` signature; `width = ReadBits(14) + 1`;
  `height = ReadBits(14) + 1`; `alpha_is_used = ReadBits(1)`;
  `version_number = ReadBits(3)`) and §7.1 ("Basic Structure") for
  the ABNF `image-header = %x2F image-size alpha-is-used version`
  / `image-size = 14BIT 14BIT ; width - 1, height - 1`. The typed
  `WebpLosslessChunk` handle decodes exactly these fields —
  nothing past offset 5 of the VP8L payload is consulted. Round 99
  cites §2 ("Bit Reading") for the `ReadBits(n)`
  least-significant-bit-first contract (the `b = ReadBits(2)` ≡
  `b = ReadBits(1); b |= ReadBits(1) << 1` example), §4
  ("Transforms") for the `while (ReadBits(1))` transform-presence
  loop and the `TransformType` 2-bit enum, §4.1 / §4.2 for
  `size_bits = ReadBits(3) + 2`, §4.3 (subtract-green carries no
  data), and §4.4 for `color_table_size = ReadBits(8) + 1` plus the
  `width_bits` pixel-bundling threshold table. The §4 transform
  *data* (sub-resolution images / color table, all §5-encoded) is
  not decoded — the reader stops at that boundary. Round 104 cites
  §6.1 ("Most of the data is coded using a canonical prefix code"),
  §6.2.1 ("Decoding and Building the Prefix Codes") for the leading
  1-bit simple/normal flag dispatch, "Simple Code Length Code" for
  the `num_symbols = ReadBits(1) + 1` and
  `symbol0 = ReadBits(1 + 7*is_first_8bits)` layout, "Normal Code
  Length Code" for the `num_code_lengths = 4 + ReadBits(4)`,
  `kCodeLengthCodes = 19`, `kCodeLengthCodeOrder`, the
  `max_symbol = alphabet_size | 2 + ReadBits(length_nbits)` gate,
  the literal `[0..15]` code-length emission, the repeat `16`
  (`3 + ReadBits(2)`), the zero-run `17` (`3 + ReadBits(3)`) and
  `18` (`11 + ReadBits(7)`), and the single-leaf-node tree
  exception. The canonical-prefix-code construction itself is the
  `[Huffman]`-referenced canonical assignment fixed across the
  family of canonical-prefix-code formats (codes assigned in
  `(length, value)` order, read MSB-first within a code) and is
  implemented from first principles via integer Kraft completeness
  checking. Round 106 cites §5.1 ("Roles of Image Data") for the
  five-roles taxonomy (ARGB / entropy / predictor / color-transform
  / color-indexing image), §5.2.3 ("Color Cache Coding") for the
  `color-cache-info` 1-bit dispatch + `color_cache_code_bits =
  ReadBits(4)` + `color_cache_size = 1 << code_bits` + the
  `[1..11]` range MUST, §6.2 ("Details") for the per-pixel
  five-prefix-code group definition (GREEN+length+cache / RED /
  BLUE / ALPHA / DIST), §6.2.2 ("Decoding of Meta Prefix Codes")
  for the ARGB-role-only 1-bit dispatch, the "Entropy Image"
  `prefix_bits = ReadBits(3) + 2` + `prefix_image_width =
  DIV_ROUND_UP(image_width, 1 << prefix_bits)` + same for height,
  and §6.2.3 ("Decoding Entropy-Coded Image Data") for the
  `256 + 24 + color_cache_size` GREEN alphabet size; §7.3
  ("Structure of the Image Data") for the
  `spatially-coded-image = color-cache-info meta-prefix data` /
  `entropy-coded-image = color-cache-info data` ABNF split that
  determines whether the meta-prefix bit is present. The entropy
  *image* itself is a §5.2-encoded `entropy-coded-image` (LZ77 +
  color-cache + per-pixel prefix-coded symbols) which round 106 does
  not decode — the reader records the entropy-image start bit
  for the next layer, mirroring how round 99 stopped at the first
  §5 transform body. Round 107 cites §5.2 ("Encoding of Image Data")
  for the three per-pixel methods (prefix-coded literal / LZ77
  backward reference / color-cache code), §5.2.1 ("Prefix-Coded
  Literals") for the green / red / blue / alpha channel order,
  §5.2.2 ("LZ77 Backward Reference") for the prefix-code + extra-bits
  value transform (`if (prefix_code < 4) return prefix_code + 1;
  extra_bits = (prefix_code - 2) >> 1; offset = (2 + (prefix_code &
  1)) << extra_bits; return offset + ReadBits(extra_bits) + 1`), the
  note that the maximum backward-reference length is 4096 (the first
  24 length prefix codes), and the "Distance Mapping" 120-entry
  distance map plus the `(xi, yi) = distance_map[distance_code - 1];
  dist = xi + yi * image_width; if (dist < 1) dist = 1` conversion
  (with codes `> 120` denoting the scan-line distance offset by 120),
  §5.2.3 ("Color Cache Coding") for the `(0x1e35a7bd * color) >> (32
  - color_cache_code_bits)` hash, the zero-initialization, and the
  "insert every pixel ... in the order they appear in the stream"
  rule, and §6.2.3 ("Decoding Entropy-Coded Image Data") for the
  GREEN symbol `S` range dispatch (`S < 256` literal, `256 <= S <
  256 + 24` length prefix code, `S >= 256 + 24` color-cache index).
  Round 109 cites §4 ("Transforms") for the reverse-read-order inverse
  application ("last one first"), §4.1 ("Predictor Transform") for the
  `block_index = (y >> size_bits) * transform_width + (x >> size_bits)`
  block addressing, the 14-mode table, the `Average2` / `Select` /
  `ClampAddSubtractFull` / `ClampAddSubtractHalf` definitions, the
  border rules (left-topmost → `0xff000000`, top row → L, left column →
  T, rightmost column uses the row's leftmost pixel as TR), and the
  `PredictorTransformOutput` per-channel residual add; §4.2 ("Color
  Transform") for `ColorTransformDelta(t,c) = (t*c) >> 5` (signed 8-bit
  `t`/`c`), the `InverseTransform` add order (green→red, green→blue,
  then red→blue using the already-corrected red), and the
  `ColorTransformElement` ↔ sub-image channel mapping (red =
  `red_to_blue`, green = `green_to_blue`, blue = `green_to_red`); §4.3
  ("Subtract Green Transform") for `AddGreenToBlueAndRed`; and §4.4
  ("Color Indexing Transform") for the `color_table_size = ReadBits(8)
  + 1` field, the subtraction-coded color table ("adding the previous
  color component values by each ARGB component"), the `argb =
  color_table[GREEN(argb)]` lookup with the out-of-range → `0x00000000`
  rule, the `width_bits` threshold table, the LSB-first packing of
  2/4/8 indices into the green byte, and the `image_width =
  DIV_ROUND_UP(image_width, 1 << width_bits)` subsampling. §7.2
  ("Structure of Transforms") fixes the per-transform ABNF
  (`predictor-image` / `color-image` = `3BIT entropy-coded-image`,
  `color-indexing-image` = `8BIT entropy-coded-image`) and §7.3 the
  `entropy-coded-image = color-cache-info data` shape the transform
  bodies share. With round 109 the lossless decode path is complete;
  remaining work is the lossy VP8 path, the ALPH alpha bitstream, and
  output-format packing.
* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`). Round 6 cites §9.1
  ("Uncompressed Data Chunk") for the 3-byte frame tag layout
  (frame type / version / show_frame / first_partition_size), the
  3-byte sync code `0x9D 0x01 0x2A`, and the two 16-bit (scale << 14
  | dim) words that carry width and height. The typed
  `WebpLossyChunk` handle decodes exactly these fields — nothing
  past offset 10 of the VP8 payload is consulted.
* The 18-fixture corpus at `docs/image/webp/fixtures/` — consumed
  as opaque byte streams. Round 104 walks `lossless-1x1.webp`'s
  VP8L payload directly to derive a golden anchor — color-cache
  bit = 0, then a 5-code prefix group each carrying a single
  symbol (GREEN=60 / RED=180 / BLUE=90 / ALPHA=255 / DIST=0,
  i.e. ARGB = 255,180,60,90, the single palette color of this 1×1
  image). The fixture trace's `VP8L_COLOR_CACHE color_cache_bits=0`
  / `VP8L_HUFFMAN_GROUP num_htree_groups=1` lines confirm the
  derivation. Round 99 cross-checks the §4 transform list decoded
  from `lossless-1x1` against its trace's
  `VP8L_TRANSFORM type=3 COLOR_INDEXING` / `num_colors=1
  packed_bits=3`, and from `lossless-32x32-rgba` against its
  trace's `type=2 SUBTRACT_GREEN` then `type=0 PREDICTOR bits=9`
  prefix (the reader halts at the predictor's §5 body).
  Round 7 cross-checks the `lossless-1x1`
  trace's reported `width=1 height=1 alpha_used=0 version=0` and
  the newly-vendored `lossless-32x32-rgba` trace's reported
  `width=32 height=32 alpha_used=1 version=0`. Round 6 cross-checked
  the `lossy-1x1` and `lossy-with-alpha-128x128` `width` / `height`
  / `partition_length` / `xscale` / `yscale`. Round 5 round-trips
  the `lossy-1x1.webp` / `lossless-1x1.webp` fixtures' own `VP8 ` /
  `VP8L` payloads back through the new builder; round 4 cross-checked
  the ANMF u24 LE decode against `animated-with-alpha/trace.txt`.

No external library source — libwebp, libvpx, image-rs, webp-rs,
etc. — was consulted. `cwebp` / `dwebp` would be permissible as
black-box validators; rounds 1 through 109 did not invoke them
directly. Round 109's end-to-end fixture tests validate against the
ARGB pixels of each fixture's committed `expected.png` (a clean-room
PNG-decode of the corpus' own ground-truth files, not any WebP
reference output).

## License

MIT. See `LICENSE`.
