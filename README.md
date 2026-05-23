# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM + ANMF).

## Status — 2026-05-24 (clean-room round 104)

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
* **Pixel decode (VP8 / VP8L / ALPH bitstream):** not implemented yet —
  [`decode_webp`](src/lib.rs) returns [`Error::NotImplemented`].
  Callers route the `VP8 ` payload via
  [`extract_lossy_chunk`](src/lib.rs) to an external VP8 decoder.
  Round 99 landed the §4 transform list (first step of the lossless
  pixel path); round 104 lands the §6.2.1 canonical-prefix-code
  reader (the entropy primitive every §5 / §6 consumer needs).
  §6.2.2 meta prefix codes and §5.2 LZ77 + color-cache decode are
  next.
* **Registry hook:** [`register`](src/lib.rs) is a no-op; round 6
  still ships no decoder/encoder to the runtime context.

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
| §2.7.1.2 `ALPH` bitstream     | not yet — bytes after info byte are out of scope    |
| §2.7.1.4 `ICCP`               | surfaced as opaque chunk                            |
| §2.7.1.5 `EXIF` / `XMP `      | surfaced as opaque chunks                           |
| §2.7.1.6 unknown chunks       | surfaced (no special handling required by §2.7.1.6) |
| §2.3 `build_chunk` writer     | done — generic FourCC+Size+payload+odd-pad emit     |
| §2.7.1 `build_vp8x_chunk`     | done — typed flag-byte + 24-bit LE × 2 emit         |
| §2.4 `build_webp_file`        | done — simple + extended `RIFF/WEBP` envelope       |
| VP8 / VP8L bitstream decode   | not yet — typed handle routes payload out-of-crate  |
| VP8 / VP8L bitstream encode   | not yet — payload is opaque input to the builder    |

Test count: **151** (127 unit + 24 integration against the
`docs/image/webp/fixtures/` corpus). Round 104 adds 16 unit tests
inside `vp8l_prefix::tests` (single-leaf no-bit read, two-symbol
canonical assignment, the classic `[1,2,3,3]` canonical example
decoded in value order, over-subscribed / incomplete / empty /
length-too-large refusals, simple 1-bit / 8-bit / two-symbol codes,
simple symbol-out-of-range refusal, normal CLC with direct lengths,
normal zero-run `18`, normal repeat `16`, normal max_symbol-too-large
refusal, truncated-code EOF) + 1 new
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

Rounds 1 through 104 were implemented entirely against:

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
  checking. The §6.2.2 meta-prefix-code section, §3.7 color cache,
  and §5.2 LZ77 backward-reference layers all consume `PrefixCode`
  symbols but are *not* implemented here.
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
black-box validators; rounds 1 through 99 did not invoke them
directly (round 99 reads only the fixture bytes already committed
to `docs/` / the in-crate `tests/data/`).

## License

MIT. See `LICENSE`.
