# Changelog

All notable changes to `oxideav-webp` are recorded here.

## [Unreleased]

### Added

* **Clean-room round 115 (2026-05-24).** First **VP8L lossless encoder**.
  New `vp8l_encode` module (compiles standalone, no `oxideav-core` dep):
  * `encode_webp_lossless(rgba, width, height)` — encodes an interleaved
    8-bit RGBA image (`[R, G, B, A]` scan order, the `DecodedWebp::rgba`
    layout) to a complete RIFF/WEBP file carrying a §2.6 simple-lossless
    `VP8L` chunk. Re-exported at the crate root as
    `oxideav_webp::encode_webp_lossless`. The encoded file decodes back to
    the exact input bytes through `decode_webp` — a pixel-exact round trip.
  * Simplest spec-conformant path: §3.8.2 `optional-transform` = `%b0`
    (no transform / pass-through), §3.8.3 `color-cache-info` = `%b0`
    (no color cache), §3.7.2.2 `meta-prefix` = `%b0` (single prefix-code
    group), and a literal-only §3.8.3 image (every pixel a §3.7.3 ARGB
    literal, no LZ77 backward references). The distance prefix code (#5)
    is the §3.7.2.1.1 single-symbol-0 form ("empty prefix codes can be
    coded as those containing a single symbol 0").
  * §3.7.2 canonical prefix-code construction: per-channel symbol
    frequencies → length-limited (≤ 15-bit) Huffman code lengths
    (`build_code_lengths`, min-heap build + length-limiting rebalance) →
    `(length, value)`-ordered canonical codes (`canonical_codes`) — the
    identical assignment the round-104 `vp8l_prefix::PrefixCode` reader
    consumes. Code lengths are written with the §3.7.2.1.2 *normal code
    length code* (or the trivial single-leaf form for constant channels).
  * `BitWriter` — the LSB-first inverse of `vp8l_stream::BitReader`.
  * `EncodeError` (`PixelBufferMismatch` / `InvalidDimensions` / `Build`)
    with a `From<EncodeError>` into the crate-wide `Error`.
  * 15 unit tests + 4 integration round trips: encode→decode is pixel-exact
    on synthetic 1×1 / gradient / solid / 16×16-pseudo-random images and on
    the real `lossless-1x1`, `lossless-32x32-rgba`, and
    `lossless-color-indexing-paletted` fixtures (decoded by the independent
    decode path, re-encoded, re-decoded, compared byte-for-byte).
  * Encoder scope is decode-only-validated for now: no §3.8.2 transform
    encoding, no LZ77 / color-cache compression. Files are larger than a
    libwebp-encoded equivalent but spec-valid and round-trip-exact.

* **Clean-room round 112 (2026-05-24).** The codec is now registered into
  `oxideav_core::RuntimeContext` — `register()` is no longer a no-op. New
  `registry` module (gated behind the default-on `registry` feature):
  * `registry::WebpDecoder` — an `oxideav_core::Decoder` impl over
    `decode_webp_image`. Each `send_packet` carries one whole
    `RIFF/WEBP` file; `receive_frame` returns a single-planar
    `Frame::Video` of interleaved 8-bit RGBA (`PixelFormat::Rgba`,
    stride `width * 4`). Covers §2.6 / §3.4 `VP8L` lossless (simple or
    `VP8X`-extended) with optional §2.7.1.2 `ALPH`-over-`VP8L` alpha
    override. A §2.5 `VP8 ` lossy file, and any animation / header-only
    file with no `VP8L`/`VP8 ` image-data chunk, surface as
    `oxideav_core::Error::Unsupported` (lossy callers route the chunk
    via `extract_lossy_chunk`).
  * `register()` / `registry::register_codecs` install one `CodecInfo`
    under the `webp` codec id with the decoder factory and a `WEBP`
    FourCC tag claim; `registry::register_containers` installs the
    `.webp` file-extension hint. No encoder factory is registered.
  * The decoder's `CodecParameters` carry the decoded `width` /
    `height` / `PixelFormat::Rgba` after the first `receive_frame`
    (read via `WebpDecoder::params`).
  * `registry::decode_webp_to_frame(bytes, pts)` — a direct
    `VideoFrame`-flavoured wrapper around `decode_webp_image`.
  * `From<Error> for oxideav_core::Error` — `Unsupported` maps to the
    core `Unsupported`; every other variant flows through `InvalidData`
    carrying the sub-module's `Display` text.
  * 10 unit tests in `registry::tests` cover the RuntimeContext install,
    FourCC resolution, an end-to-end lossless decode through the
    registered factory, the `VP8 ` lossy `Unsupported` refusal, the
    params dim/format surfacing, the one-packet/one-frame contract, the
    post-flush `Eof`, and the error conversion.

* **Clean-room round 111 (2026-05-24).** Top-level still-image decode is
  wired up — `decode_webp` no longer returns `NotImplemented` for the
  cases the crate can decode. New surface:
  * `decode_webp_image(bytes) -> DecodedWebp` — walks the `RIFF/WEBP`
    container, decodes a §2.6 / §3.4 `VP8L` lossless image (simple **or**
    `VP8X`-extended) through the full §4–§6 chain, and returns the
    `DecodedWebp { width, height, rgba }` struct. `rgba` is
    `width*height*4` interleaved `[R, G, B, A]` bytes in scan order — the
    `oxideav_core::PixelFormat::Rgba` layout the workspace's image crates
    share. When a (spec-discouraged, per §2.7.1.2 "SHOULD NOT") `ALPH`
    chunk accompanies the `VP8L` image, its decoded alpha plane overrides
    the per-pixel alpha.
  * `decode_webp(bytes) -> Vec<u8>` — the flat-buffer shorthand: same
    decode, returns just the packed RGBA bytes.
  * `Error::Unsupported(UnsupportedKind)` — a §2.5 `VP8 ` lossy file is a
    clean `Unsupported(LossyVp8)` (route it onward with
    `extract_lossy_chunk`); a file with no `VP8L`/`VP8 ` image-data chunk
    (animation / header-only) is `Unsupported(NoImageData)`. Lossy is
    **not** stub-decoded.
  * End-to-end tests decode the `lossless-1x1`,
    `lossless-color-indexing-paletted`, and `lossless-32x32-rgba`
    fixtures all the way to RGBA (dims + pixel spot-checks against the
    round-109 ARGB ground truth, including the RGBA alpha-channel
    repack), a synthesized `VP8X`+`VP8L` extended file, and a
    hand-assembled `VP8X`+`VP8L`+`ALPH` file proving the `ALPH` alpha
    override.

* **Clean-room round 110 (2026-05-24).** §2.7.1.2 `ALPH` alpha-channel
  bitstream decode — the alpha plane is now produced end-to-end. New
  surface in `alph`:
  * `alph::decode_alpha(payload, width, height)` — decodes a whole
    `ALPH` chunk payload to a `width * height` plane of 8-bit alpha
    values. Covers both compression methods: method 0 (raw 8-bit
    values, length `width*height`) and method 1 (a *headerless* §3 VP8L
    image-stream of implicit dimensions, decoded via the new
    `vp8l_transform::decode_lossless_headerless`, with the alpha lifted
    from the **green** channel per §2.7.1.2). Then applies the
    §2.7.1.2 inverse filter — none / horizontal (A) / vertical (B) /
    gradient (`clip(A+B-C)`) — as `alpha = (predictor + X) % 256`, with
    the documented top-left (predictor 0), left-most (use pixel above),
    and top-most (use pixel left) edge cases.
  * `decode_alpha_plane(bytes)` — container-level entry point: walks the
    `RIFF/WEBP` file, takes the alpha-plane dimensions from the `VP8X`
    canvas (or the `VP8 ` keyframe header when no `VP8X` is present),
    locates the `ALPH` chunk, and decodes. Returns `Ok(None)` when the
    file carries no `ALPH` chunk.
  * `AlphError` gained `DimensionsOverflow`, `RawLengthMismatch`,
    `UnsupportedCompression`, and `Vp8l` variants.
  * `vp8l_transform::decode_lossless_headerless(payload, width, height)`
    — the headerless §3 image-stream decode (no 5-byte image header)
    that the compressed alpha path reuses; the existing
    `decode_lossless` now delegates to a shared driver.
  * Verified bit-exact against the black-box `dwebp -alpha` validator on
    the `lossy-with-alpha-128x128` fixture (all 16384 alpha bytes
    identical); filter inverses are unit-tested against hand-computed
    §2.7.1.2 vectors for all four methods.

* **Clean-room round 109 (2026-05-24).** VP8L §4 inverse-transform
  passes — the layer that consumes round-108's `decode_argb` ARGB buffer
  and produces final pixels, closing the lossless decode path
  end-to-end. New module `vp8l_transform` exposes:
  * `vp8l_transform::decode_lossless(payload, width, height)` — the
    top-level driver. Reads the §4 / §7.2 `optional-transform` list
    (each transform's fixed fields **and** its §5-encoded
    `entropy-coded-image` body), tracks §4.4 width subsampling, decodes
    the main §5.1 ARGB image at the (subsampled) width via `decode_argb`,
    then applies the inverse transforms in reverse read order (§4: "last
    one first").
  * `vp8l_transform::inverse_predictor` — §4.1: 14 prediction modes
    (`Average2` / `Select` / `ClampAddSubtractFull` /
    `ClampAddSubtractHalf`) over the TL/T/TR/L block grid, with the
    border rules (top-left → `0xff000000`, top row → L, left column → T,
    rightmost column uses the row's leftmost pixel as TR) and the
    per-channel residual add.
  * `vp8l_transform::inverse_color` — §4.2: per-block
    `ColorTransformElement` add-back (`ColorTransformDelta(t,c) =
    (t*c) >> 5` with signed-8-bit `t`/`c`), green→red / green→blue /
    red→blue, on the red and blue channels only.
  * `vp8l_transform::inverse_subtract_green` — §4.3: add green into red
    and blue (`& 0xff`).
  * `vp8l_transform::inverse_color_table` (§4.4 subtraction-decode of the
    palette) + `vp8l_transform::inverse_color_indexing` (palette lookup;
    ≤16-color pixel un-bundling of 2/4/8 indices per green byte; the
    width un-subsample back to the canvas width; out-of-range indices →
    transparent black `0x00000000`).
  * `vp8l_decode::decode_entropy_coded_image(reader, width, height)` —
    a generalized §7.3 `entropy-coded-image` decoder (color-cache-info +
    one prefix-code group + §5.2 data, no meta-prefix layer) used to
    decode each transform's sub-resolution body. `decode_entropy_image`
    now delegates to it.
  * `vp8l_decode::DecodedImage::pixels_mut` / `from_parts` (used by the
    in-place inverse passes and the color-indexing re-size) and a new
    `DecodeError::DuplicateTransform` variant.
  * `decode_lossless_image(bytes)` — container-level entry point: walks
    the file, extracts the `VP8L` chunk, and decodes it to a
    `DecodedImage`. Returns `Ok(None)` for `VP8 `-only files.
* 18 new unit tests in `vp8l_transform::tests` (each predictor
  primitive; predictor border rules for the top-left / top-row / left-
  column cases; the §4.2 signed delta + forward↔inverse round-trip + in-
  place block use; §4.3 green add-back with wrap; §4.4 subtraction
  decode + no-bundling lookup + out-of-range → transparent black +
  width_bits-1/3 bundling + the threshold table) + 4 integration tests
  in `fixture_walks` that decode three real fixtures *bit-exactly*
  against their `expected.png` ARGB ground truth:
  * `round109_lossless_1x1_color_indexing_decodes_end_to_end` →
    `0xFFB43C5A`.
  * `round109_lossless_color_indexing_paletted_decodes_end_to_end`
    (32×32, 8-color palette, width_bits=1 bundling).
  * `round109_lossless_32x32_rgba_full_transform_chain_decodes_end_to_end`
    (SUBTRACT_GREEN + PREDICTOR + CROSS_COLOR + level-1 color cache,
    real alpha).
  * `round109_decode_lossless_image_returns_none_for_lossy_file`.
  New in-crate fixture `tests/data/lossless-color-indexing-paletted.webp`
  (byte-for-byte copy of the docs corpus). Test count: **229** (was
  207).
* The decoder is **standalone-friendly** — `vp8l_transform` compiles
  under `--no-default-features` with no `oxideav-core` dependency.

### Notes (round 109)

The VP8L lossless decode path is now **complete end-to-end**: container
walk → §4 transform list (with bodies) → §5/§6 entropy decode → §4
inverse-transform chain → final ARGB pixels, validated bit-exact on the
`lossless-1x1`, `lossless-color-indexing-paletted`, and
`lossless-32x32-rgba` fixtures. `decode_webp` itself still returns
`Error::NotImplemented` (it would need ARGB→output-format packing +
the VP8 lossy + ALPH alpha paths); callers wanting lossless pixels use
`decode_lossless_image`.

* **Clean-room round 108 (2026-05-24).** VP8L §6.2.2 entropy-image
  multi-group ARGB decode — the piece that turns the round-106
  meta-prefix dispatch and the round-107 single-group §5.2 loop into a
  full multi-group ARGB decode. New `vp8l_decode` surface:
  * `vp8l_decode::decode_argb(reader, width, height)` — the full
    ARGB-role decode. Reads the round-106 `MetaPrefixHeader` for the
    `Argb` role and dispatches: `meta-prefix = %b0` runs the
    single-group `decode_image` path; `meta-prefix = %b1` decodes the
    §6.2.2 entropy image, derives `num_prefix_groups = max(entropy
    image) + 1`, reads that many `PrefixCodeGroup`s, and runs the
    §6.2.3 loop selecting a group per pixel block.
  * `vp8l_decode::decode_entropy_image(reader, prefix_bits,
    prefix_image_width, prefix_image_height)` — decodes the §6.2.2
    entropy image (itself a §5 `entropy-coded-image`) into a
    `MetaPrefixIndex`. Each block's meta-prefix code is the red+green
    channels of its entropy-image pixel: `(argb >> 8) & 0xffff`.
  * `vp8l_decode::MetaPrefixIndex` — the per-block meta-prefix codes
    plus `prefix_bits` / `block_width` / `block_height`. Exposes
    `num_prefix_groups()` (max-based, not block count) and
    `meta_code_for(x, y)` (`meta[(y >> prefix_bits) * block_width +
    (x >> prefix_bits)]`).
  * New `DecodeError` variants `MetaPrefix` / `EmptyEntropyImage` /
    `MetaPrefixIndexOutOfRange`, plus a `From<MetaPrefixError>` impl.
* 9 new unit tests in `vp8l_decode::tests` (meta-index helpers and
  max-based `num_prefix_groups`; entropy-image red+green meta-code
  extraction incl. the high-code red-channel path; two-group per-block
  selection; single-group `decode_argb`; single-group parity with
  `decode_image`; multi-group with a shared color cache; zero-dim
  entropy-image refusal) and 3 integration tests in `fixture_walks`
  (public `decode_argb` multi-group + single-group, public
  `decode_entropy_image` with max-based group count).
* **Clean-room round 107 (2026-05-24).** VP8L §5.2 LZ77
  backward-reference + §5.2.3 color-cache per-pixel ARGB decode loop —
  the §6.2.3 decoder that consumes symbols from a round-106
  `PrefixCodeGroup` and produces a decoded ARGB pixel buffer. New
  module `vp8l_decode` exposes:
  * `vp8l_decode::decode_image(reader, group, color_cache, width,
    height)` — the §6.2.3 per-pixel decode loop. Reads GREEN symbol
    `S` from prefix code #1 and dispatches by range (§5.2.1 literal /
    §5.2.2 LZ77 backward reference / §5.2.3 color-cache code) until
    `width * height` ARGB pixels are emitted. Returns a
    `vp8l_decode::DecodedImage` (scan-line ARGB, pre-inverse-transform).
  * `vp8l_decode::read_lz77_value(reader, prefix_code)` — the §5.2.2
    prefix-code → value transform shared by length and distance
    (`prefix < 4 → prefix + 1`, else `offset + ReadBits(extra) + 1`).
  * `vp8l_decode::DISTANCE_MAP` (the 120-element §5.2.2 neighbor-offset
    table) + `distance_code_to_pixel_distance(code, width)` (the
    `dist = xi + yi*width`, clamp-to-1, `> 120 → code - 120` mapping).
  * `vp8l_decode::ColorCache` — the §5.2.3 cache: zero-initialized,
    hashed by `(0x1e35a7bd * argb) >> (32 - code_bits)`; `new` /
    `hash` / `insert` / `lookup` / `size`. Every emitted pixel is
    re-inserted in stream order.
  * `vp8l_decode::GreenSymbol::classify(symbol, alphabet_size)` — the
    §6.2.3 GREEN range dispatch (`Literal` / `LengthPrefix` /
    `ColorCache`), unit-testable in isolation.
  * `vp8l_decode::DecodeError` plus public constants
    `NUM_DISTANCE_MAP_CODES` / `NUM_LENGTH_PREFIX_CODES` /
    `COLOR_CACHE_HASH_MULTIPLIER`.
* 24 new unit tests in `vp8l_decode::tests` (§5.2.2 LZ77 value
  transform across prefix codes 0–6 + the length-4096 boundary at
  prefix 23; distance-map length / spec-example first entries /
  above-120 offset / negative-offset clamp; §6.2.3 GREEN literal /
  length / color-cache classification + out-of-range refusal; §5.2.3
  color-cache hash formula / insert-lookup round-trip /
  zero-initialization; full decode loop for a literal-only 2×1 image,
  a single literal pixel, a length/distance back-reference with LZ77
  self-overlap, a color-cache hit, plus backward-reference-underflow
  and no-cache refusals) plus 2 integration tests:
  * `round107_lossless_1x1_color_table_decodes_end_to_end_to_palette_pixel`
    drives container walk → §4 transform list → resume at the
    COLOR_INDEXING §5 body → §5.2.3 + §6.2 meta-prefix header →
    `decode_image` over `lossless-1x1.webp`'s 1×1 color-table image,
    producing the single palette pixel ARGB `0xFFB43C5A`
    (255,180,60,90) straight from the fixture's own VP8L payload bytes.
  * `round107_decode_error_surfaces_through_crate_error` locks the
    `DecodeError → oxideav_webp::Error::Vp8lDecode` `From` wiring.
  Test count: **195** (was 169).
* The decoder is **standalone-friendly** — `vp8l_decode` compiles
  under `--no-default-features` with no `oxideav-core` dependency.

### Changed

* `Error` gained a `Vp8lDecode(vp8l_decode::DecodeError)` variant.

### Notes

`decode_webp` still returns `Error::NotImplemented`. Round 107 closes
the §5.2 single-group ARGB decode path: a single `PrefixCodeGroup`
plus the §5.2 data now decodes to a full ARGB pixel buffer. The
remaining lossless work is the §6.2.2 entropy-image *multi-group*
path (one group per pixel block, selected by an entropy image) and
the §4 inverse-transform passes (predictor / color / subtract-green /
color-indexing) that operate on the buffer this loop produces.

* **Clean-room round 106 (2026-05-24).** VP8L §5.2.3 color-cache info
  + §6.2.2 meta-prefix dispatch + §6.2 5-prefix-code-group reader —
  the preamble every §5 image-data block opens with, sitting on top of
  the round-104 single-prefix-code reader. New module `meta_prefix`
  exposes:
  * `meta_prefix::ColorCacheInfo` — the §5.2.3 `color-cache-info`
    field. `ColorCacheInfo::read(reader)` dispatches on the leading
    1-bit flag, reads the 4-bit `color_cache_code_bits` when set,
    validates the §5.2.3 `[1..11]` range MUST, and surfaces
    `is_enabled()` / `size()` (`1 << code_bits`).
  * `meta_prefix::PrefixCodeGroup` — the five-prefix-code group the
    §6.2 / §6.2.3 / §5.2 decode paths consume (GREEN+length+cache /
    RED / BLUE / ALPHA / DIST). `PrefixCodeGroup::read(reader,
    color_cache_size)` reads them in §6.2 bitstream order, sizing the
    GREEN alphabet at `256 + 24 + color_cache_size` per §6.2.3.
  * `meta_prefix::ImageRole` — the §5.1 image-data role tag (`Argb`
    vs. `EntropyCoded`). Per §6.2.2 + §7.3 ABNF, the §6.2.2
    meta-prefix dispatch bit is present ONLY for the ARGB role.
  * `meta_prefix::MetaPrefixHeader::read(reader, role, image_w,
    image_h)` — the combined §5.2.3 + §6.2.2 + §6.2 preamble reader.
    Returns either `MetaPrefixCodes::Single { group }` (single
    prefix-code group, single Huffman group case + every non-ARGB
    role) or `MetaPrefixCodes::EntropyImagePending { prefix_bits,
    image_width, image_height, entropy_image_bit_position }` (ARGB
    role + multi-group case; the entropy image is itself a
    §5.2-encoded `entropy-coded-image` that requires the next layer's
    LZ77 + color-cache decoder, so the reader records the boundary
    and stops — mirroring how round 99 stopped at the first §5
    transform body and round 104 resumed there).
  * `meta_prefix::MetaPrefixError` plus public constants
    `COLOR_CACHE_BITS_MIN` / `COLOR_CACHE_BITS_MAX` /
    `PREFIX_BITS_MIN` / `PREFIX_BITS_MAX`.
* 15 new unit tests in `meta_prefix::tests` (color-cache info
  disabled / enabled at `code_bits` 1 / 11 / 0-refused / 12-refused,
  GREEN alphabet size formula, group read order matches §6.2,
  EntropyCoded role skips meta-prefix bit, ARGB single-group read,
  ARGB multi-group entropy-image boundary + bit position, ARGB
  `DIV_ROUND_UP` rounding, ARGB max `prefix_bits=9`, ARGB
  color-cache propagation into GREEN alphabet, truncated
  `ColorCacheInfo` EOF, truncated `MetaPrefixHeader` EOF) plus 3
  integration tests:
  * `round106_lossless_1x1_color_table_meta_prefix_header_reads_single_group`
    reads the COLOR_INDEXING transform's color-table image with the
    `EntropyCoded` role and asserts the surfaced group matches r104's
    by-hand decode (GREEN=60 / RED=180 / BLUE=90 / ALPHA=255 /
    DIST=0).
  * `round106_meta_prefix_argb_single_group_synthetic_matches_trace_shape`
    exercises the ARGB-role single-group shape (`color_cache_bits=0`,
    `meta_huffman=0`, `num_htree_groups=1`) every fixture trace
    reports when no entropy image is in play.
  * `round106_meta_prefix_argb_multi_group_records_entropy_image_boundary`
    exercises the ARGB-role multi-group shape (`prefix_bits=4` over a
    128×128 image), asserts 8×8 entropy-image dimensions and the
    recorded entropy-image bit position.
  Test count: **169** (was 151).
* The reader is **standalone-friendly** — `meta_prefix` compiles
  under `--no-default-features` with no `oxideav-core` dependency.

### Changed

* `Error` gained a `Vp8lMetaPrefix(meta_prefix::MetaPrefixError)`
  variant.

### Notes

`decode_webp` still returns `Error::NotImplemented`. Round 106 lands
the §5.2.3 + §6.2.2 + §6.2 preamble every §5 image-data block opens
with. The remaining lossless-pixel-path work is §5.2 LZ77
backward-reference decode + §5.2.3 color-cache *symbol-lookup*
decode (the per-pixel decoder that pulls symbols from a
`PrefixCodeGroup`) — that pair will close out the ARGB-role single-
and entropy-coded-image-role paths in one round, with the
entropy-image §5.2 decode (which feeds the ARGB multi-group path)
following thereafter.

* **Clean-room round 104 (2026-05-24).** VP8L §6.2.1 prefix-code
  reader + canonical decoder — the first piece of the §5 / §6 entropy
  machinery that sits on top of the round-99 §4 transform list. New
  module `vp8l_prefix` exposes:
  * `vp8l_prefix::PrefixCode` — a built canonical prefix code over an
    alphabet. `PrefixCode::read(reader, alphabet_size)` reads one
    code's lengths off the wire (dispatching on the §6.2.1 leading
    simple/normal flag) and builds the decoder;
    `PrefixCode::from_code_lengths(lengths)` builds straight from a
    per-symbol length table; `read_symbol(reader)` decodes one symbol
    at a time (MSB-first within a code, matching the canonical
    `(length, value)` assignment). The §6.2.1 single-leaf-node tree is
    handled (one symbol at length 1, reading consumes no bits) and the
    completeness rule (`sum 2^-len == 1`) is enforced via integer
    Kraft arithmetic — over-/under-subscribed codes are refused.
  * `vp8l_prefix::read_code_lengths(reader, alphabet_size)` — the
    §6.2.1 "Simple Code Length Code" (flag 1: 1–2 symbols at length 1)
    and "Normal Code Length Code" (flag 0: the 19-symbol
    code-length-code read in `kCodeLengthCodeOrder`, the `max_symbol`
    gate, and the literal `[0..15]` / repeat-`16` / zero-run-`17`/`18`
    expansion).
  * `vp8l_prefix::PrefixError` + public `NUM_CODE_LENGTH_CODES` /
    `CODE_LENGTH_CODE_ORDER` / `MAX_CODE_LENGTH` constants.
  * `vp8l_stream::BitReader::seek_to_bit(bit_pos)` — repositions the
    cursor to an absolute bit offset (clamped to the slice end) so a
    caller can resume reading at a recorded boundary, e.g.
    `TransformList::body_bit_position()`.
* 16 new unit tests in `vp8l_prefix::tests` (single-leaf no-bit read,
  two-symbol canonical assignment, the classic `[1,2,3,3]` canonical
  example decoded in value order, over-subscribed / incomplete / empty
  / length-too-large refusals, simple 1-bit / 8-bit / two-symbol
  codes, simple symbol-out-of-range refusal, normal CLC with direct
  lengths, normal zero-run `18`, normal repeat `16`, normal
  max_symbol-too-large refusal, truncated-code EOF) + 1
  `vp8l_stream::tests::seek_to_bit_repositions_and_clamps` + 1
  integration test:
  * `round104_lossless_1x1_color_table_prefix_group_matches_fixture_bytes`
    resumes at the COLOR_INDEXING §5 body of `lossless-1x1.webp`,
    reads the §5 color-cache info bit (0, matching the fixture trace's
    `color_cache_bits=0`) and the full 5-code prefix group, and
    asserts the single symbols GREEN=60 / RED=180 / BLUE=90 /
    ALPHA=255 / DIST=0 (the single ARGB palette color 255,180,60,90)
    decoded purely from the fixture's own VP8L payload bytes.
  Test count: **151** (was 133).
* The reader is **standalone-friendly** — `vp8l_prefix` compiles
  under `--no-default-features` with no `oxideav-core` dependency.

### Changed

* `Error` gained a `Vp8lPrefix(vp8l_prefix::PrefixError)` variant.

### Notes

`decode_webp` still returns `Error::NotImplemented`. Round 104 builds
the canonical-prefix-code primitive every §5 / §6 consumer needs.
The next sections are §6.2.2 (meta prefix codes / entropy image —
which *prefix-code group* applies to a pixel block) and §5.2 (the
LZ77 + color-cache pixel stream that reads symbols from a group).

* **Clean-room round 99 (2026-05-24).** VP8L bit-reader + §4
  transform-list reader. New module `vp8l_stream` exposes:
  * `vp8l_stream::BitReader` — the WebP-Lossless §2 `ReadBits(n)`
    primitive. Bytes are consumed in stream order, bits of each byte
    least-significant-bit-first, and a multi-bit read returns a `u32`
    whose bit 0 is the first bit read off the wire (matching the
    spec's `b = ReadBits(2)` ≡ `b = ReadBits(1); b |= ReadBits(1) <<
    1` rule). `read_bits(n)` / `read_bit()` /
    `new_after_image_header(payload)` (seeks past the 5-byte §3.4
    image-header) / `bit_position()` / `bits_remaining()`. EOF is a
    typed `BitReaderEof { bit_pos, wanted, available }` that does not
    advance the cursor.
  * `vp8l_stream::TransformList::read(reader)` — the §4
    `while (ReadBits(1))` transform-presence loop. For each present
    transform it decodes the leading fixed `ReadBits` fields:
    `Predictor` / `Color` `size_bits = ReadBits(3) + 2` (§4.1 / §4.2),
    `SubtractGreen` (no data, §4.3), and `ColorIndexing`
    `color_table_size = ReadBits(8) + 1` plus the derived
    pixel-bundling `width_bits` (§4.4). §4's "each transform used
    only once" rule is enforced (`DuplicateTransform`). The reader
    **stops** at the first transform carrying a §5 entropy-coded body
    (sub-resolution image / color table) it cannot yet decode and
    records `body_bit_position()` + `stopped_at_entropy_body()` so the
    next-round §5 reader resumes there; `SubtractGreen` (bodyless)
    lets the loop continue.
  * `vp8l_stream::Transform` / `TransformType` enums +
    `Transform::transform_type()` / `has_entropy_body()` helpers.
  * `read_vp8l_transform_list(bytes)` — top-level convenience: walks
    the container, extracts the `VP8L` chunk, reads its §4 transform
    list; returns `Ok(None)` for `VP8 `-only files.
* 18 new unit tests in `vp8l_stream::tests` (LSB-first
  single/multi-bit reads, byte-boundary read, full-u32 read, 0-bit
  no-op, EOF position/demand reporting, image-header seek,
  `TransformType` mapping, `width_bits` thresholds, empty list,
  subtract-green-only list, predictor/color/color-indexing
  stop-at-body, subtract-green→predictor fixture shape,
  duplicate-transform refusal, truncated-list EOF, transform helpers)
  plus 3 integration tests:
  * `round99_lossless_1x1_transform_list_is_color_indexing_from_fixture`
    cross-checks the §4 list decoded from `lossless-1x1.webp` against
    its `trace.txt` (`COLOR_INDEXING num_colors=1 packed_bits=3`).
  * `round99_lossless_32x32_rgba_transform_list_matches_fixture_prefix`
    cross-checks the `SUBTRACT_GREEN` → `PREDICTOR size_bits=9`
    prefix and the bit-49 stop boundary against the fixture trace.
  * `round99_transform_list_returns_none_for_lossy_fixture`.
  Test count: **133** (was 112).
* The reader is **standalone-friendly** — `vp8l_stream` and
  `read_vp8l_transform_list` compile under `--no-default-features`
  with no `oxideav-core` dependency.

### Changed

* `Error` gained a `Vp8lTransform(vp8l_stream::TransformListError)`
  variant.

### Notes

`decode_webp` still returns `Error::NotImplemented`. Round 99 is the
first step of the lossless pixel path: it reads the §2 bit-reader
foundation and the §4 transform list, stopping at the §5 entropy
boundary. The §5 entropy decode (prefix codes / Huffman code groups
/ LZ77 / color cache) is the next section.

* **Clean-room round 7 (2026-05-22).** Typed §2.6 `VP8L` chunk
  routing handle. New module `vp8l_chunk` exposes:
  * `vp8l_chunk::WebpLosslessChunk` — a borrowed handle around a
    §2.6 `VP8L` chunk payload. Decodes the 5-byte WebP-Lossless
    §3.4 / §7.1 image-header (one-byte `0x2F` signature followed
    by LE bit-packed 14-bit `width-1` + 14-bit `height-1` + 1-bit
    `alpha_is_used` + 3-bit `version`) and surfaces resolved
    1-based `width()` / `height()` plus raw `alpha_is_used()` /
    `version()`. The chunk payload is exposed verbatim via
    `bitstream()` so a downstream VP8L decoder can consume it.
  * `vp8l_chunk::WebpLosslessChunk::from_chunk(buf, chunk)` /
    `from_payload(slice)` constructors.
  * `vp8l_chunk::extract_lossless(buf, container)` — pulls the
    first `VP8L` chunk out of an already-walked container;
    returns `Ok(None)` for `VP8 `-only files.
  * `extract_lossless_chunk(bytes)` — top-level convenience wrapper
    that walks the container and extracts in one call.
  * `VP8L_SIGNATURE` / `VP8L_IMAGE_HEADER_LEN` public constants.
  * Refusal modes: `NotVp8lChunk` / `PayloadTooShortForHeader` /
    `BadSignature`. §3.4 says `version` MUST be `0`; the typed
    handle surfaces it raw rather than refusing — the
    version-mismatch policy belongs to the downstream decoder.
* The handle is deliberately a **routing** surface — `oxideav-webp`
  takes no runtime dependency on a VP8L decoder. A caller routes
  the borrowed `bitstream()` slice to whichever lossless-WebP
  decoder it wants.
* 10 new unit tests inside `vp8l_chunk::tests` (minimal 1×1,
  16384×16384 max dims with alpha hint set, non-zero version
  surfacing, short-payload refusal, bad-signature refusal,
  trailing-image-stream borrow, non-VP8L FourCC refusal, walker
  round-trip, lossy-container returns None, simple-lossless
  returns Some) plus a new `lossless-32x32-rgba.webp` fixture
  in `tests/data/` (byte-for-byte copy of
  `docs/image/webp/fixtures/lossless-32x32-rgba/input.webp`) +
  5 new integration tests:
  * `round7_lossless_1x1_fixture_extracts_to_typed_lossless_chunk_with_trace_dims`
    cross-checks every §3.4 field against `lossless-1x1/trace.txt`.
  * `round7_lossless_32x32_rgba_fixture_extracts_with_alpha_used_bit_set`
    cross-checks the only `alpha_used=1` path in the in-crate
    fixture corpus against `lossless-32x32-rgba/trace.txt`.
  * `round7_lossy_fixture_extract_lossless_returns_none` confirms
    `extract_lossless_chunk` returns `Ok(None)` on a `VP8 `-only file.
  * `round7_lossless_chunk_payload_survives_round_trip_through_builder`
    routes the extracted payload back through the round-5 builder
    and re-extracts, locking down the writer ↔ router contract.
  * `round7_lossless_chunk_from_chunk_works_on_walker_output`
    exercises the `from_chunk` constructor directly.
  Test count: **112** (was 97).

### Changed

* `Error` gained a `Lossless(vp8l_chunk::WebpLosslessError)` variant.

### Notes

`decode_webp` still returns `Error::NotImplemented`; the round-7
typed handle is a hand-off layer, not a pixel decoder. The routing
contract is one-way: this crate emits a typed
`WebpLosslessChunk::bitstream()` slice, and the caller picks a
VP8L decoder to consume it. That keeps `oxideav-webp`
standalone-friendly — every public function still compiles under
`--no-default-features` with no `oxideav-core` dependency.

* **Clean-room round 6 (2026-05-22).** Typed §2.5 `VP8 ` chunk
  routing handle. New module `vp8_chunk` exposes:
  * `vp8_chunk::WebpLossyChunk` — a borrowed handle around a §2.5
    `VP8 ` chunk payload. Peeks the 10-byte RFC 6386 §9.1 keyframe
    header (3-byte frame tag carrying frame_type / version /
    show_frame / 19-bit first_partition_size, 3-byte sync code
    `0x9D 0x01 0x2A`, two 16-bit `(scale << 14) | dim` words) and
    surfaces `width()` / `height()` / `version()` / `show_frame()`
    / `first_partition_size()` / `horizontal_scale()` /
    `vertical_scale()`. The chunk payload is exposed verbatim via
    `bitstream()` so a downstream VP8 decoder can consume it.
  * `vp8_chunk::WebpLossyChunk::from_chunk(buf, chunk)` /
    `from_payload(slice)` constructors.
  * `vp8_chunk::extract_lossy(buf, container)` — pulls the first
    `VP8 ` chunk out of an already-walked container; returns
    `Ok(None)` for `VP8L`-only files.
  * `extract_lossy_chunk(bytes)` — top-level convenience wrapper
    that walks the container and extracts in one call.
  * Refusal modes: `NotVp8Chunk` / `PayloadTooShortForKeyframe` /
    `NotAKeyframe` / `BadStartCode`. §2.5 / §9.1 together imply a
    WebP `VP8 ` chunk MUST be a keyframe; `NotAKeyframe` enforces
    this. Bad `0x9D 0x01 0x2A` sync bytes are surfaced raw so
    callers can distinguish "wrong codec" from "corrupted payload".
* The handle is deliberately a **routing** surface — `oxideav-webp`
  takes no runtime dependency on `oxideav-vp8`. A caller routes the
  borrowed `bitstream()` slice to whichever VP8 decoder it wants.
* 9 new unit tests inside `vp8_chunk::tests` (minimal 1x1 / max
  14-bit dims / short-payload refusal / interframe refusal / bad
  start-code refusal / non-VP8 fourcc refusal / payload-bytes
  round-trip via walker / extract returns None on lossless /
  extract returns Some on lossy) + 5 new integration tests against
  the fixture corpus:
  * `round6_lossy_1x1_fixture_extracts_to_typed_lossy_chunk_with_trace_dims`
    cross-checks every §9.1 field against `lossy-1x1/trace.txt`.
  * `round6_lossy_with_alpha_extended_fixture_extracts_to_128x128_keyframe`
    cross-checks the extended-format `VP8 ` chunk's §9.1 dims and
    also asserts the §2.7.1 VP8X-declared canvas agrees with the
    §9.1-derived canvas for this fixture.
  * `round6_lossless_fixture_extract_returns_none` confirms
    `extract_lossy_chunk` returns `Ok(None)` on a `VP8L`-only file.
  * `round6_lossy_chunk_payload_survives_round_trip_through_builder`
    routes the extracted payload back through the round-5 builder
    and re-extracts, locking down the writer ↔ router contract.
  * `round6_lossy_chunk_from_chunk_works_on_walker_output` exercises
    the `from_chunk` constructor directly.
  Test count: **97** (was 83).

### Changed

* `Error` gained a `Lossy(vp8_chunk::WebpLossyError)` variant.

### Notes

`decode_webp` still returns `Error::NotImplemented`; the round-6
typed handle is a hand-off layer, not a pixel decoder. The routing
contract is one-way: this crate emits a typed
`WebpLossyChunk::bitstream()` slice, and the caller picks a VP8
decoder (e.g. `oxideav-vp8`) to consume it. That keeps
`oxideav-webp` standalone-friendly — every public function still
compiles under `--no-default-features` with no `oxideav-core`
dependency.

* **Clean-room round 5 (2026-05-22).** RIFF/WEBP container *builder*
  helpers — the inverse of the round-1 walker. New module `build`
  exposes:
  * `build::build_chunk(fourcc, payload) -> Result<Vec<u8>, BuildError>`
    — generic §2.3 chunk writer (4-byte FourCC + 4-byte little-endian
    `Size` + payload + odd-size `0x00` pad byte).
  * `build::build_vp8x_chunk(canvas_width, canvas_height, Vp8xFlags) ->
    Result<Vec<u8>, BuildError>` — §2.7.1 Figure 7 10-byte payload
    writer. Inverse of `vp8x::Vp8xHeader::parse`: same bit positions
    for the `I` / `L` / `E` / `X` / `A` feature flags, same 24-bit
    little-endian Minus-One width/height encoding, same 24-bit zero-
    filled Reserved field, same 2^32 - 1 product cap.
  * `build::build_webp_file(payload, image_kind, canvas_width,
    canvas_height) -> Result<Vec<u8>, BuildError>` — §2.4 file writer
    over four `ImageKind` variants:
    * `Lossy` / `Lossless` — §2.5 / §2.6 simple layouts (single
      `VP8 ` / `VP8L` chunk; canvas dims are ignored because the
      bitstream carries them).
    * `ExtendedLossy` / `ExtendedLossless` — §2.7 extended layout
      (`VP8X` chunk + bitstream chunk, in the §2.7-mandated order).
  * Convenience wrappers `build_webp_file` / `build_vp8x_chunk` at
    the crate root that return the crate-wide `Error`.
* `Vp8xFlags` (Default-able struct with `has_iccp` / `has_alpha` /
  `has_exif` / `has_xmp` / `has_animation`) drives the §2.7.1 flag
  byte. Round 5 defaults all flags off since this crate ships no
  encoder for the related bitstreams yet — once `ALPH` / `ANIM` /
  metadata writers land, those writers will set the corresponding
  flag here so the §2.7.1 declaration matches the chunks emitted.
* `BuildError` variants: `CanvasDimZero { which }`,
  `CanvasDimTooLarge { which, got }`, `CanvasTooLarge { canvas_width,
  canvas_height }`, `PayloadTooLargeForChunk { got }`.
* Public `MAX_VP8X_CANVAS_DIM` / `MAX_CHUNK_PAYLOAD` constants
  documenting the §2.7.1 24-bit and §2.3 32-bit field maxima.
* 18 new unit tests inside `build::tests` (chunk layout / pad byte /
  flag bit positions / dim LE byte order / boundary refusal modes /
  file round-trip / file-size accounting / 64 KiB round-trip /
  corrupt-after-build refusal) + 3 new integration tests
  (`round5_lossy_fixture_payload_rewraps_into_byte_identical_riff_envelope`,
  `round5_lossless_fixture_payload_rewraps_into_byte_identical_riff_envelope`,
  `round5_build_vp8x_chunk_round_trips_through_typed_parser_with_flags`)
  that close the writer ↔ walker / writer ↔ typed-parser loop on
  real `docs/image/webp/fixtures/` bytes. Test count: **83** (was
  63).

### Changed

* `Error` gained a `Build(BuildError)` variant.

### Notes

The builders are intentionally framing-only: they accept the `VP8 ` /
`VP8L` payload as opaque bytes the caller computed elsewhere. Pixel
decode and VP8 / VP8L encode remain not-implemented in this crate;
`decode_webp` still returns `Error::NotImplemented`. With this layer
in place, the workspace's `cli-convert` `encode_webp` path is
unblocked at the container layer — it can drive the builder once a
VP8L encoder lands.

* **Clean-room round 4 (2026-05-21).** Typed parser for the per-frame
  §2.7.1.1 `ANMF` chunk header (Figure 9). New module `anmf` exposes
  `anmf::AnmfHeader::parse(&[u8]) -> Result<AnmfHeader, AnmfError>`
  and the top-level convenience wrapper `parse_anmf_header`. The
  16-byte header decodes to:
  * `x: u32` — `Frame X * 2` per §2.7.1.1 (24-bit little-endian
    uint24 doubled).
  * `y: u32` — `Frame Y * 2`.
  * `width: u32` — `1 + Frame Width Minus One` (always ≥ 1).
  * `height: u32` — `1 + Frame Height Minus One` (always ≥ 1).
  * `duration_ms: u32` — literal Frame Duration in ms.
  * `blend: BlendingMethod` — `AlphaBlend` / `Overwrite` (bit 1 of
    the info byte).
  * `dispose: DisposalMethod` — `None` / `Background` (bit 0 of the
    info byte).
  * `reserved: u8` + `info_byte: u8` — surfaced raw for trace
    observability.
  `AnmfHeader::HEADER_LEN` constant + `frame_data_offset()` helper
  (always 16) lets callers slice the per-frame `Frame Data` sub-RIFF
  out of the chunk payload. The header parser stays **structural** —
  it does not descend into the per-frame `ALPH` / `VP8 ` / `VP8L`
  sub-chunks.
* 15 new unit tests + 1 new integration test cross-checking the
  bit-position and uint24 decodes against the
  `docs/image/webp/fixtures/animated-with-alpha/trace.txt`
  (`flags_byte=0x02 dispose=0 blend=1`, three identical ANMF frames
  at 64×64 / 100 ms / x=0 / y=0) golden output. Test count: **63**
  (was 45).

### Changed

* `Error` gained an `Anmf(AnmfError)` variant.

### Notes

Pixel decode (VP8 / VP8L bitstreams) and the actual ALPH alpha
bitstream are still not implemented; `decode_webp` still returns
`Error::NotImplemented`. Round 5+ targets bitstream decode of the
simplest VP8L paths against the lossless-1x1 / lossless-32x32-rgb
fixtures.

## [Earlier — Unreleased entries, retained]

### Added

* **Clean-room round 3 (2026-05-21).** Typed parsers for the two
  §2.7.1 metadata chunks that travel alongside `VP8X`:
  * `alph::AlphHeader::parse(&[u8]) -> Result<AlphHeader, AlphError>`
    decodes the §2.7.1.2 Figure 10 info byte (`Rsv|P|F|C`, 2 bits each,
    MSB-first) into typed `AlphCompression` / `AlphFiltering` /
    `AlphPreprocessing` enums plus a raw `reserved: u8` for
    observability. The alpha bitstream itself is not decoded —
    `AlphHeader::bitstream_offset()` reports the constant `1` so
    callers can slice the remainder out of the chunk payload.
  * `anim::AnimHeader::parse(&[u8]) -> Result<AnimHeader, AnimError>`
    decodes the §2.7.1.1 Figure 8 6-byte payload: a 4-byte BGRA
    `BackgroundColor` plus a little-endian u16 `loop_count`. A
    `loops_forever()` helper surfaces the §2.7.1.1 `loop_count == 0`
    sentinel.
  * Top-level convenience wrappers `parse_alph_header` and
    `parse_anim_header`.
* 18 new unit tests + 2 new integration tests cross-checking the
  bit-position and BGRA decodes against the
  `docs/image/webp/fixtures/lossy-with-alpha-128x128/trace.txt`
  (`header_byte=0x01`, `method=1 filter=0 pre_processing=0`) and
  `docs/image/webp/fixtures/animated-with-alpha/trace.txt`
  (`bgcolor=0xffffffff loop_count=0`) golden outputs. Test count:
  **45** (was 27).

### Changed

* `Error` gained `Alph(AlphError)` and `Anim(AnimError)` variants.

### Notes

Pixel decode (VP8 / VP8L bitstreams) and the actual ALPH alpha
bitstream are still not implemented; `decode_webp` still returns
`Error::NotImplemented`. Subsequent rounds will decode each
bitstream layer against the RFC-9649-referenced specifications and
the fixture corpus.

## [Earlier — Unreleased entries, retained]

### Added

* **Clean-room round 2 (2026-05-21).** Typed parser for the §2.7.1
  `VP8X` chunk payload. New module `vp8x` exposes
  `Vp8xHeader::parse(&[u8]) -> Result<Vp8xHeader, Vp8xError>` and a
  top-level `parse_vp8x_header` convenience wrapper. `Vp8xHeader`
  carries the §2.7.1 1-based canvas dimensions
  (`canvas_width`, `canvas_height`) plus the five named feature
  flags (`has_iccp` ↔ `I`, `has_alpha` ↔ `L`, `has_exif` ↔ `E`,
  `has_xmp` ↔ `X`, `has_animation` ↔ `A`) and a derived
  `has_unknown` summary that is true when any of the §2.7.1
  reserved positions (the `Rsv` pair, the `R` bit, or the 24-bit
  reserved field) is non-zero. The parser enforces only the §2.7.1
  MUSTs that aren't "MUST be ignored": payload length is exactly
  10 bytes and `canvas_width * canvas_height ≤ 2^32 - 1`.
* 15 new unit tests + 1 new integration test cross-checking the
  bit-position decode against the fixture corpus' `trace.txt`
  output. Test count: **27** (was 11).

### Changed

* `Error` gained a `Vp8x(Vp8xError)` variant.

### Notes

Pixel decode (VP8 / VP8L / ALPH bitstreams) is still not
implemented; `decode_webp` still returns `Error::NotImplemented`.
Subsequent rounds will decode each bitstream layer against the
RFC-9649-referenced specifications and the fixture corpus.

## [Earlier — Unreleased entries, retained]

### Added

* **Clean-room round 1 (2026-05-20).** Structural RIFF/WEBP
  container walker per RFC 9649 §2.3–§2.7. New module `container`
  exposes `parse(&[u8]) -> Result<WebpContainer, ContainerError>`,
  a top-level `parse_container` wrapper, and FourCC constants for
  every chunk type called out by name in §2.4–§2.7 (`VP8 `, `VP8L`,
  `VP8X`, `ALPH`, `ANIM`, `ANMF`, `ICCP`, `EXIF`, `XMP `). The
  walker validates the §2.4 file header, the declared `File Size`
  against the buffer, each chunk's `Size` against the remaining
  RIFF payload, and the §2.3 odd-size pad byte. Order-on-disk is
  preserved so §2.7 ordering rules can be enforced by callers.
* 8 unit tests + 3 integration tests against the
  `docs/image/webp/fixtures/` corpus (`lossy-1x1`, `lossless-1x1`,
  `extended-with-exif`).

### Changed

* `Error` gained a `Container(ContainerError)` variant for walker
  errors; `NotImplemented` remains for the still-unimplemented
  pixel decode path.

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Per the workspace's Implementer-Round
  procedure, such audit failures are unrecoverable via incremental
  cleanup and require an orphan rebuild.

  No `old` branch is retained; long-standing audit failures forfeit
  the archive per workspace policy.
