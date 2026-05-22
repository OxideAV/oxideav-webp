# Changelog

All notable changes to `oxideav-webp` are recorded here.

## [Unreleased]

### Added

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
