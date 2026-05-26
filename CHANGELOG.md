# Changelog

All notable changes to `oxideav-webp` are recorded here.

## [Unreleased]

## [0.1.6](https://github.com/OxideAV/oxideav-webp/compare/v0.1.5...v0.1.6) - 2026-05-26

### Other

- canonical-inverse round-trip test for build/extract metadata (round 144)
- AnimEncoderOptions::default_lossy_quality / with_default_lossy_quality (round 143)
- AnimFrame::with_blend / with_dispose chainable builders (round 142)
- animation-wide near-lossless default (round 141)
- wire per-frame near-lossless preprocessing into build_animated_webp
- add near-lossless encoder preprocessing pass
- Round 138: build::build_webp_file_with_metadata (ICCP/EXIF/XMP)
- optimal length-limited Huffman via Package-Merge
- emit §3.7.2.1.1 simple code length code for 1-2 symbol codes
- add §4.4 color-indexing (palette) transform encoding
- add §4.2 color (cross-channel) transform encoding
- round-133 §4.1 predictor (spatial) transform encoding
- round-132 §5.2.3 color-cache size-selection chooser
- published VP8 lossy encoder API surface (API-shape stub)
- round 130 — §5.2.2 width-aware distance-code chooser
- round 127: Auto/Delta lossless dirty-rect animation modes + canvas compositing
- wire lossy decode against published DecodeError (not unpublished Vp8Error)
- §2.5 VP8 (lossy) decode via oxideav-vp8 (round 124)
- §5.2.1 / §5.2.3 color-cache writer (round 121)
- §3.5.3 / §3.8.2 subtract-green forward transform (round 120)
- §5.2.2 LZ77 backward-reference matching (round 119)
- Round 118: restore published-0.1.5 animation-encode API (VP8L path)
- round 117 — restore published VP8L lossless-encode public names
- restore published-0.1.5 decode API shape (WebpImage/WebpFrame/WebpError)
- add API-COMPAT.md — public API shape the rebuild must reproduce
- round 115 — first VP8L lossless encoder (literal-only round trip)
- register oxideav_core::Decoder into RuntimeContext (round 112)
- wire top-level decode_webp to RGBA for VP8L (round 111)
- decode the §2.7.1.2 ALPH alpha-channel bitstream (round 110)
- round 109 — VP8L §4 inverse-transform passes (lossless decode complete)
- round 108 — VP8L §6.2.2 entropy-image multi-group ARGB decode
- round 107 — VP8L §5.2 LZ77 + §5.2.3 color-cache per-pixel ARGB decode loop
- round 106 — VP8L §5.2.3 + §6.2.2 + §6.2 meta-prefix header reader
- round 104 — VP8L §6.2.1 prefix-code reader + canonical decoder
- clean-room round 99 — VP8L bit-reader + §4 transform-list reader
- round 7: typed §2.6 `VP8L` chunk routing handle
- round 6: typed §2.5 `VP8 ` chunk routing handle
- round 5: RIFF/WEBP container builders (§2.3 / §2.4 / §2.7.1)
- round 4: ANMF §2.7.1.1 typed per-frame header parse
- round 3: ALPH §2.7.1.2 + ANIM §2.7.1.1 typed field parse
- round 2: VP8X §2.7.1 typed field parse (flags + canvas dims)
- in-crate copies of fixture inputs for standalone CI checkout
- round 1: RIFF/WEBP container walker per RFC 9649 §2.3–§2.7
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **Clean-room round 144 (2026-05-26).** **Canonical-inverse round-trip
  tests pinning `build_webp_file_with_metadata` as the byte-exact left
  inverse of `extract_metadata` over the §2.7 metadata chunks.** §2.7's
  "may appear out of order" carve-out for `EXIF` / `XMP ` is collapsed by
  the writer to a single canonical `EXIF` → `XMP ` order, the §2.3 pad
  bytes are always zero, and the §2.7.1 flag octet is recomputed from
  the input's `Some`-ness. These three deterministic-emission properties
  together make the round trip
  `build(payload, meta) == build(payload, extract(build(payload, meta)))`
  byte-identical. The new tests pin that identity so any future change
  to the writer's emission order, pad handling, or flag-octet
  computation surfaces as a byte-diff in CI.

  No new public API. Four new integration tests in
  [`tests/published_encode_api.rs`](tests/published_encode_api.rs) cover
  (a) the all-three-set case, (b) every single-kind and pair-kind
  subset (six combinations) under one parametric harness, (c) the
  odd-length-payload case (every metadata chunk gets a §2.3 pad byte
  that must be regenerated identically on re-build), and (d) the
  `ImageKind::ExtendedLossy` bitstream-FourCC variant. One new inline
  unit test in [`src/build.rs`](src/build.rs) anchors the same
  property at the writer level via the `container::parse` walker
  (i.e. without depending on the `extract_metadata` public path).

  Total: **500** tests green (was 495: +5 from this round). Default and
  `--no-default-features` builds both clippy + fmt clean.

* **Clean-room round 143 (2026-05-26).** **Chainable
  `AnimEncoderOptions::with_default_lossy_quality(Option<u8>) -> Self`
  builder and matching `default_lossy_quality: Option<u8>` field,
  symmetric to the round-141
  `AnimEncoderOptions::with_default_near_lossless_quality` /
  `default_near_lossless_quality` pair.** Lets callers configure their
  full encoder shape (lossy + near-lossless animation-wide defaults)
  through one chainable builder surface today, ahead of the VP8 lossy
  encode body landing.

  **API-shape stub.** The VP8 lossy encode path is blocked on the
  `oxideav-vp8` per-MB driver (workspace task #1041); the existing
  `AnimFrameMode::Lossless` / `Delta` / `Auto` emission paths are
  lossless-only and ignore this field. The contract pinned by the new
  tests: setting `default_lossy_quality` to any value (`Some(0)` …
  `Some(255)` / `None`) produces output **byte-exact-equal** to the
  baseline `None` bytes, on both the full-keyframe and dirty-rect
  Delta paths, and does not perturb the round-141 near-lossless
  default's existing behaviour when both knobs are set together.

  Six new integration tests in
  [`tests/published_anim_default_lossy_api.rs`](tests/published_anim_default_lossy_api.rs)
  cover (a) field defaults to `None`, (b) builder round-trips
  `Some(80)` / `None` and composes with `with_default_near_lossless_quality`
  in either order, (c) the two defaults are stored in independent
  fields (struct-literal + dual-builder equivalence), (d) lossy
  default is byte-exact no-op on the Lossless keyframe path across
  `Some(0)`/`50`/`75`/`100`/`255`/`None`, (e) byte-exact no-op on
  the Delta dirty-rect path, (f) byte-exact no-op when overlaid on a
  `default_near_lossless_quality = Some(60)` baseline. Four new inline
  unit tests confirm the default, the builder round-trip, independence
  from the near-lossless default (copy-paste-swap guard), and the
  encoder no-op contract on a 4×4 fixture. Total: **495** tests green
  (was 485: +10 from this round). Default and `--no-default-features`
  builds both clippy + fmt clean.

* **Clean-room round 142 (2026-05-26).** **Chainable
  `AnimFrame::with_blend(BlendingMethod)` and
  `AnimFrame::with_dispose(DisposalMethod)` builders for the §2.7.1.1
  `B` (blending) and `D` (disposal) info-byte bits.** The underlying
  fields were already public on `AnimFrame` since round 118, but the
  published-0.1.5 surface only exposed the literal-construction path;
  the new helpers complete the chainable-builder pattern alongside
  `with_near_lossless_quality` (r140) and the `AnimFrame::new(…)`
  constructor. Both helpers consume the receiver and return `Self`, so
  they compose with each other and with the existing builders in any
  order. No new public fields; no behavioural change to existing
  encoder output (the literal-construction path was already wired).

  Six new integration tests in
  [`tests/published_anim_blend_dispose_api.rs`](tests/published_anim_blend_dispose_api.rs)
  pin the end-to-end semantics through `decode_webp`'s §2.7.1.1
  compositor: (a) opaque-source alpha-blend short-circuits to byte-
  exact round-trip; (b) a 2×2 translucent (alpha=128) BLUE sub-frame
  over an 8×8 opaque RED keyframe blends to the spec-formula bit-
  exact `(127, 0, 128, 255)`; (c) the same setup with Overwrite blits
  the translucent source verbatim into the canvas; (d) a 2×2 GREEN
  frame with dispose=Background between RED keyframe and 2×2 BLUE
  frame produces a third-frame snapshot whose `(2..4, 2..4)` rect is
  the ANIM bg, not GREEN; (e) the mirror case with dispose=None leaves
  GREEN on the canvas under the next frame; (f) the integrated case
  where AlphaBlend + Background dispose compose on the same frame
  produces the blended pixels in the f1 snapshot and the bg-cleared
  rect in the f2 snapshot.

  Six new inline unit tests confirm the `new()` blend/dispose
  defaults (`Overwrite` / `None`), the per-method builder round-trip
  for each, the chaining behaviour against `with_near_lossless_quality`,
  and the §2.7.1.1 Figure 9 info-byte emission for each `B` and `D`
  bit. Total: **485** tests green (was 473: +12 from this round).
  Default and `--no-default-features` builds both clippy + fmt clean.

* **Clean-room round 141 (2026-05-26).** **Animation-wide near-lossless
  default in `AnimEncoderOptions`.** The new
  `AnimEncoderOptions::default_near_lossless_quality: Option<u8>` field
  lets callers set the VP8L near-lossless preprocessing knob **once for
  the whole animation** instead of repeating it on every
  [`AnimFrame::near_lossless_quality`]. Resolution order at encode time:
  per-frame `Some(q)` always wins; per-frame `None` falls back to the
  options-level default; both `None` is the pre-round-140 baseline
  (no quantization, byte-exact-equal to the baseline encoder, identical
  to `Some(100)` in either slot). A builder helper
  `AnimEncoderOptions::with_default_near_lossless_quality(Option<u8>) -> Self`
  matches the rest of the chainable options surface.

  The default is threaded through to *both* the full-keyframe
  ([`AnimFrameMode::Lossless`]) and dirty-rect
  ([`AnimFrameMode::Delta`] / [`AnimFrameMode::Auto`]) emission paths
  via a single `effective_quality = frame.q.or(opts.default_q)`
  resolution in `build_animated_webp_with_options`'s per-frame loop, so
  the round-140 per-frame-knob behaviour and the round-141
  options-default behaviour share the exact same downstream quantization
  call and produce bit-identical per-frame ANMFs given the same
  effective quality.

  **Measured deltas (3-frame 64×64 noisy fixture, deterministic
  xorshift32 seeds 0/1/2):** baseline (default `None`) 37,364 B (ANMFs
  ~12,432 each). Options default `Some(60)` with no per-frame
  overrides: 28,180 B (**−24.58 %** overall), each ANMF ~9,368 B
  (**−24.66 %** / **−24.58 %** / **−24.63 %**) — byte-for-byte
  equivalent to the round-140 per-frame `Some(60)` output. Mixed
  (default `Some(60)`, frame 1 overridden to `Some(100)`): 31,236 B
  (**−16.40 %** overall); frames 0 and 2 shrink to 9,368 B each
  (default applied), frame 1 stays at 12,432 B exactly (override wins,
  matches baseline ANMF byte-for-byte). The per-frame chunk sizes are
  cross-checked against the corresponding "all `Some(60)`" and "all
  `Some(100)`" files inside the test.

  Seven new integration tests in
  [`tests/published_anim_default_near_lossless_api.rs`](tests/published_anim_default_near_lossless_api.rs)
  cover (a) default-only equals per-frame-only byte-for-byte, (b)
  default `None` equals the convenience baseline, (c) default `Some(255)`
  clamps to the baseline no-op, (d) per-frame `Some(q)` overrides the
  default in either direction, (e) decoder round-trip recovers
  `near_lossless::quantize(src, q)` exactly when the default supplies
  the quality, and (f) the default flows through to the Delta
  dirty-rect path the same way the per-frame knob does. Two new inline
  unit tests confirm the new field defaults to `None` and the
  `with_default_near_lossless_quality` builder round-trips. Default +
  `--no-default-features` builds clippy + fmt clean.

* **Clean-room round 140 (2026-05-26).** **Per-frame near-lossless
  preprocessing in the animated-WebP encoder.** The round-139 still-image
  `near_lossless::apply` preprocessor is now wired into the
  [`build_animated_webp`](src/anim_encode.rs) /
  [`build_animated_webp_with_options`](src/anim_encode.rs) path via a new
  optional `AnimFrame::near_lossless_quality: Option<u8>` field
  (defaulting to `None`, the no-op identity). A builder helper
  `AnimFrame::with_near_lossless_quality(Option<u8>) -> Self` matches the
  rest of the per-frame chainable construction. The preprocessing is
  applied identically to the full-keyframe
  ([`AnimFrameMode::Lossless`]) and dirty-rect
  ([`AnimFrameMode::Delta`] / [`AnimFrameMode::Auto`]) emission paths —
  each frame's quality knob feeds into its per-frame VP8L bitstream
  independently, so an animation can mix lossless (`None` / `Some(100)`)
  and near-lossless frames freely.

  **Three round-140 guarantees** are pinned by integration tests in
  [`tests/published_anim_near_lossless_api.rs`](tests/published_anim_near_lossless_api.rs):
  (a) `None` (default) and `Some(100)` produce byte-exact-equal output
  to the pre-round-140 baseline encoder on every tested fixture, so
  existing animated-WebP test fixtures keep their bit pattern; the
  `None | Some(q ≥ 100)` no-op short-circuit lives in
  `apply_near_lossless_if_requested` and delegates to
  `near_lossless::apply`. (b) `Some(60)` produces a strictly smaller
  3-frame animated WebP than `Some(100)` on a deterministic 64×64
  high-entropy fixture (37,364 B → 28,180 B, **−24.58 %**) with every
  decoded per-frame PSNR ≥ 46.25 dB — well above the ≥ 40 dB floor the
  test enforces. (c) The decoded RGBA matches the still-image
  `near_lossless::quantize` of the source byte-for-byte (full-keyframe
  path) and matches the f0-with-quantized-sub-rect composite for the
  dirty-rect path; alpha round-trips unchanged at q=40 on a
  non-opaque-alpha fixture. Per-frame chunk-size monotonicity at
  q=80 < q=60 < q=40 < q=0 is verified on a single-frame 48×48 fixture.
  Mixed-quality test confirms that a per-frame knob is honoured frame
  by frame.

  11 new integration tests + 1 inline builder-round-trip test. Default
  + `--no-default-features` builds both clippy + fmt clean.

* **Clean-room round 139 (2026-05-26).** **VP8L lossless encoder
  *near-lossless* preprocessing.** A new
  [`near_lossless`](src/near_lossless.rs) module exposes the in-place
  RGB-channel quantizer `near_lossless::apply(&mut [u32], quality)` plus
  its out-of-place sibling `near_lossless::quantize`; both round each
  color channel to the nearest multiple of `2^n` where
  `n = bits_to_drop_for_quality(quality)` per the documented mapping
  `n = clamp((100 - q + 19) / 20, 0, 5)` — `q = 100` → 0 (no-op),
  `q ∈ 80..=99` → 1, `q ∈ 60..=79` → 2, `q ∈ 40..=59` → 3,
  `q ∈ 20..=39` → 4, `q ∈ 0..=19` → 5. Alpha is preserved bit-exactly
  (the preprocessing never touches it). The new public entry point
  [`encode_vp8l_argb_with_near_lossless(argb, w, h, has_alpha, quality)`](src/lib.rs)
  runs the preprocessing pass and hands the (quantized) pixels to the
  existing VP8L encoder — no decoder-side change is needed, the result
  is a perfectly normal `VP8L` chunk that decodes back through
  [`decode_webp`](src/lib.rs) to the quantized pixels bit-exactly.
  RFC 9649 does not normatively define the near-lossless formula
  (it is an encoder choice); the chosen mapping + per-channel
  round-half-up-with-clamp rounding rule is documented in the module-
  level docstring so reproducibility is a property of *our* encoder
  rather than a portable cross-encoder guarantee.

  **Three round-139 guarantees** are pinned by integration tests in
  [`tests/published_near_lossless_api.rs`](tests/published_near_lossless_api.rs):
  (a) `quality = 100` is byte-exact-equal to the baseline encoder on
  every tested fixture (1×1, 3×5, 7×4, 16×16, 64×64 natural-gradient,
  96×96 noisy); the no-op fast path returns from
  `encode_vp8l_argb_with` directly. (b) `quality = 60` produces a
  strictly smaller bitstream than `quality = 100` on a deterministic
  96×96 high-entropy fixture (27,770 B → 20,860 B, **−24.9 %**) with
  PSNR 46.25 dB — well above the test's ≥ 40 dB floor. (c) Decoder
  round-trip recovers the quantized ARGB exactly at q=60 and q=0; alpha
  round-trips unchanged through the full encode-decode cycle at q=40 on
  a non-opaque image. Quality table on the noisy fixture: q=100 →
  27,770 B (identity); q=95 / q=80 → 24,327 B / 51.20 dB; q=60 →
  20,860 B / 46.25 dB; q=40 → 17,394 B / 40.43 dB; q=20 → 13,916 B /
  34.09 dB; q=0 → 10,451 B / 27.45 dB.

  Twelve new inline tests cover: the `bits_to_drop_for_quality` bucket
  table, monotonic non-decreasing behaviour as quality drops, the
  `quantize_channel` identity at `n = 0`, round-to-nearest behaviour at
  `n = 1` and `n = 2`, the `255 → largest multiple of step ≤ 255` clamp
  rule (`n = 1` → 254, `n = 2` → 252, `n = 5` → 224), the documented
  error bounds (`step - 1` worst case across the upper clamp window,
  `step / 2` tighter bound outside it), and `result % step == 0` over
  every byte value × `n` combination. ARGB-level tests cover the
  byte-for-byte identity at `q = 100` and `q > 100`, alpha preservation
  across every quality level, the `q = 60 → multiples of 4` invariant,
  per-channel error bound across `[0..255]` for every quality, and
  apply/quantize equivalence at every quality step. Nine integration
  tests + 1 measurement helper exercise the three guarantees end-to-end
  plus dimension-mismatch error propagation and the alpha-through-the-
  full-decode-cycle case. Total: **453** tests green (+20 from round
  138).

* **Clean-room round 138 (2026-05-26).** **`build::build_webp_file_with_metadata`
  — typed §2.7.1.4 `ICCP` / §2.7.1.5 `EXIF` / §2.7.1.5 `XMP ` writer at
  the container-builder layer.** The high-level
  `encode_vp8l_argb_with_metadata` (round 115) already framed metadata
  inline for the VP8L pixel path; round 138 lifts the same logic into
  the standalone `build::` module so external encoders can wrap any
  `VP8 ` / `VP8L` bitstream payload with the §2.7 metadata chunks
  without re-implementing the chunk-ordering rule. A new
  [`build::MetadataPayloads`] borrowed bag carries the three optional
  payloads; the writer picks the simple §2.5 / §2.6 layout when the bag
  is empty (byte-for-byte identical to `build::build_webp_file`) and
  auto-promotes to the §2.7 extended layout when any kind is present.
  The emitted chunk order matches RFC 9649 §2.7 verbatim: `VP8X` first,
  `ICCP` before the bitstream, the `VP8 ` / `VP8L` bitstream, then
  `EXIF` and `XMP ` after — and the §2.7.1 flag octet (`I` / `E` / `X`
  bits) declares exactly the kinds that follow, never an extra
  unconditional OR. Twelve new tests in `build::tests` cover: empty
  metadata + simple kind = byte-for-byte equivalence to
  `build_webp_file`; simple kind + metadata auto-promotes to extended
  with the right flag set; all-three-set emits in §2.7 order; per-kind
  isolation (each of ICC / Exif / XMP individually sets only its own
  flag and emits only its own chunk); odd-length payloads trigger the
  §2.3 pad byte; empty metadata payloads (zero-length) still emit the
  chunk + set the flag; extended kind + empty metadata matches
  `build_webp_file`; canvas-validation errors propagate. Five new
  integration tests in `tests/published_encode_api.rs` cover the
  per-kind isolation at the published-API level
  (`encode_vp8l_argb_with_metadata` + `extract_metadata`) plus a
  round-trip through `build::build_webp_file_with_metadata` with a real
  VP8L bitstream.

* **Clean-room round 137 (2026-05-26).** **Optimal length-limited Huffman
  via Package-Merge.** The round-136 encoder built canonical Huffman from
  histograms, then post-passed with a heuristic depth-limiter ("lengthen
  the deepest leaf, shorten a short one to keep the Kraft sum at 1") when
  the unconstrained tree depth exceeded the spec's 15-bit cap. On
  cap-triggering histograms the heuristic is only *locally* optimal; the
  globally-optimal length assignment under the depth cap is given by the
  Package-Merge algorithm (Larmore–Hirschberg 1990) applied to the
  coin-collector reformulation of length-limited Huffman. `build_code_lengths`
  now invokes a from-scratch Package-Merge implementation when (and only
  when) the unconstrained tree exceeds [`MAX_CODE_LENGTH`]; histograms whose
  unconstrained tree already fits under the cap continue to use the
  unconstrained build (which is itself optimal). The result is a
  weakly-smaller bit cost ∑ freq[s]·len[s] on every input that triggers the
  cap, and identical output on every input that does not — so the
  round-trip stays bit-exact on every existing fixture (414/414 tests
  green, including all seven lossless fixtures). Headline measurement on
  a pathological Fibonacci(25) frequency vector (unconstrained tree depth
  24, well past the 15-bit cap): **−759 bits (~95 bytes) vs the round-136
  heuristic**, with both forms remaining complete (Kraft sum 1) and
  cap-honouring. The earlier Fibonacci(20) case yields a smaller but real
  −26-bit win. The heuristic limiter is retained as `#[cfg(test)]`-only
  scaffolding so the comparison tests can demonstrate the strict-win
  delta. Eight new inline tests cover: single-symbol / two-symbol
  short-circuit, Kraft completeness over four histograms (uniform, mild
  skew, ramp, Fibonacci), agreement with unconstrained Huffman when the
  cap is not triggered, the strict-win pathological cases (Fibonacci(20)
  and Fibonacci(25)), the `build_code_lengths` fallback dispatch, and a
  bit-exact round trip through the round-104 prefix reader.

* **Clean-room round 136 (2026-05-25).** §3.7.2.1.1 **simple code length
  code** emission for the VP8L lossless encoder. The encoder previously
  always wrote each of the five prefix codes' length tables with the
  §3.7.2.1.2 *normal code length code* (a 19-symbol code-length-code plus
  per-symbol lengths). It now dispatches to the cheaper §3.7.2.1.1
  *simple* form whenever the length table describes 1 or 2 symbols at
  code length 1, all in the `[0..255]` range the simple form admits —
  exactly the form `build_code_lengths` produces for single-color
  channels and the empty distance code. The simple form encodes the
  whole table in 3–11 bits (`%b1` flag + 1-bit `num_symbols-1` + 1-bit
  `is_first_8bits` + a 1- or 8-bit `symbol0` + an optional 8-bit
  `symbol1`), versus the normal form's ≥18-bit header, and picks the
  narrow 1-bit `symbol0` field for the very common single-symbol-0
  (empty) distance code. Both forms describe the identical length table,
  so per-symbol code emission and the decoder's reconstruction are
  unaffected; the choice only shrinks the meta-block header. Bit-exact
  round trip holds on every fixture. Headline measurement: re-encoding
  the seven lossless fixtures totals **1634 B with the simple path vs
  2658 B normal-only — a 1024 B (38.5 %) header reduction**, ranging
  from −81.6 % on `lossless-1x1` (174 → 32 B) and −75.6 % on
  `lossless-32x32-rgb` (328 → 80 B) down to −16.8 % on the natural
  128×128 image (1038 → 864 B). Six new inline tests cover the
  `simple_code_symbols` classifier (1-symbol / 2-symbol eligible;
  length≠1, 3-symbol, symbol>255, all-zero ineligible), the 1- and
  2-symbol round trips through the round-104 prefix reader, and the
  measured byte win over the normal form for the empty distance code.

* **Clean-room round 135 (2026-05-25).** §3.5.4 / §4.4 **color-indexing
  (palette) transform encoding** for the VP8L lossless encoder — the
  last unimplemented VP8L transform on the encode side. When the image
  has ≤ 256 distinct ARGB colors, the encoder builds the palette (in
  first-appearance order), writes the §3.8.2 `color-indexing-tx` header
  (`%b1` present, transform type 3, 8-bit `color_table_size - 1`), the
  palette as a `color_table_size × 1` `entropy-coded-image` that is
  per-channel subtraction-coded (the exact inverse of the decoder's
  `inverse_color_table`), then replaces every pixel with its palette
  index packed into the green channel and emits that index image as a
  §3.8.3 `spatially-coded-image` at the subsampled width. For palettes
  ≤ 16 colors it applies the spec's §4.4 pixel-bundling per Table 3 —
  bundling 2 / 4 / 8 indices LSB-first into one green byte at
  `width_bits` 1 / 2 / 3 (palette ≤ 16 / ≤ 4 / ≤ 2), subsampling the
  main-image width by `DIV_ROUND_UP(width, 1 << width_bits)`. The
  forward bundling field layout is bit-identical to the decoder's
  `inverse_color_indexing` un-bundler, and the trailing partial bundle
  on non-multiple widths round-trips exactly. The color-indexing path
  is a new candidate in `encode_argb_literals_with_width_selected`
  alongside the round-134 color, round-133 predictor, and round-132
  subtract-green × cache cross-product; it returns `None` (and the
  chooser skips it) on images with > 256 distinct colors, and only
  wins on low-color images where the single-index-byte representation
  beats the literal/LZ77 paths. Headline measurement: a 64×64
  8-color image (`width_bits = 1`, two indices per byte) shrinks from
  1982 B (best non-palette path) to 1858 B (palette) — a 6.3 % size
  reduction; spatially-coherent palette art shrinks far more. Five
  new inline tests cover the Table-3 `width_bits` mapping, the palette
  build + > 256-color rejection, the forward/inverse subtraction-coding
  round trip, the bundled size win + chooser selection + bit-exact
  round trip, an unbundled (17..256-color) round trip, the partial-row
  bundle on non-power-of-two widths, and the > 256-color rejection
  through the production chooser.

* **Clean-room round 134 (2026-05-25).** §4.2 **color (cross-channel
  decorrelation) transform encoding** for the VP8L lossless encoder.
  The encoder tiles the image into `1 << size_bits` square blocks
  (`size_bits = 4`, 16×16) and, for each block, picks the three
  signed-8-bit `ColorTransformElement` coefficients (`green_to_red` /
  `green_to_blue` / `red_to_blue`) by a coordinate-descent search over
  a coarse 3.5-fixed-point grid, scored with the same wrap-aware
  sum-of-absolute-residuals proxy the predictor uses; the all-zero
  (identity) element is always the search origin so a block with no
  usable correlation keeps the no-transform residual. It writes the
  §3.8.2 `color-tx` header (`%b1` present, transform type 1, 3-bit
  `size_bits - 2`), the sub-resolution color image as an
  `entropy-coded-image` (the §4.2 layout: alpha 255, red =
  `red_to_blue`, green = `green_to_blue`, blue = `green_to_red`), then
  the residual main image as a §3.8.3 `spatially-coded-image`. The
  forward `ColorTransform` (subtract the three deltas, using the
  *original* red for the `red_to_blue` term) is the exact inverse of
  the decoder's `inverse_color`, which restores red before re-adding
  the blue delta — bit-exact across the modulo-256 wrap. The color
  path is a new candidate in `encode_argb_literals_with_width_selected`
  alongside the round-133 predictor and round-132 subtract-green ×
  cache cross-product; the chooser emits whichever candidate is
  smallest and never regresses. Headline measurement: a 64×64 image
  with high-entropy green and fractional-slope red/blue shrinks from
  7904 B (best non-color path) to 7026 B (color) — an 11 % size
  reduction. Five new inline tests cover the forward/inverse per-pixel
  round trip (including the signed [128..255] coefficient range), the
  sub-image layout, the correlated-image size win + round trip, a noisy
  non-power-of-two round trip, and a no-regression check on
  uncorrelated noise.

* **Clean-room round 133 (2026-05-25).** §3.5.1 / §4.1 **predictor
  (spatial) transform encoding** for the VP8L lossless encoder. The
  encoder tiles the image into `1 << size_bits` square blocks
  (`size_bits = 4`, 16×16) and, for each block, scores all 14
  prediction modes `[0..13]` by a per-channel sum-of-absolute-residuals
  proxy (folding the modulo-256 wrap so small negative residuals score
  low) and picks the cheapest. It writes the §3.8.2 `predictor-tx`
  header (`%b1` present, transform type 0, 3-bit `size_bits - 2`), the
  sub-resolution predictor image as an `entropy-coded-image` (mode in
  the green channel, no meta-prefix per the §3.8.2 grammar), then the
  residual main image as a §3.8.3 `spatially-coded-image`. The forward
  predictor primitives (`Average2` / `Select` / `ClampAddSubtractFull`
  / `ClampAddSubtractHalf` and the §4.1 left-topmost / top-row /
  left-column / rightmost-column border rules) are bit-identical to the
  decoder's `inverse_predictor`, so `residual = actual − pred` is the
  exact inverse of the decoder's `final = residual + pred`. The
  predictor path is a new candidate in
  `encode_argb_literals_with_width_selected` alongside the round-132
  subtract-green × cache cross-product; the chooser emits whichever
  candidate is smallest and never regresses. Headline measurement:
  64×64 smooth 2-D gradient goes from 10377 B (no-predictor) to 308 B
  (predictor) — a 97 % size reduction. The residual main image and the
  predictor sub-image each run their own §5.2.3 color-cache evaluation.
  Round trip stays bit-exact through `decode_lossless_image`, validated
  on smooth, noisy, and non-power-of-two (partial-block) fixtures. Four
  new inline tests cover (a) `sub_pred`/`add_pred` inverse across the
  wrap; (b) every selected mode ∈ `0..=13`; (c) the gradient shrinks
  and round-trips; (d) the predictor path round-trips on noise +
  non-power-of-two dimensions.

* **Clean-room round 132 (2026-05-25).** §5.2.3 **color-cache
  size-selection chooser** for the VP8L lossless encoder. The
  round-121 chooser evaluated a single `code_bits = 8` candidate
  alongside the no-cache path; the round-132 chooser evaluates a
  five-size slate `{5, 7, 8, 9, 11}` (exposed as
  `CANDIDATE_COLOR_CACHE_BITS`) cross-producted with the §3.8.2
  subtract-green transform axis, and emits the smallest of the 2 × 6
  = 12 candidates. The §5.2.3 GREEN alphabet width is `256 + 24 +
  (1 << code_bits)`, so the prefix-code header overhead scales with
  the chosen size — picking the smallest cache that captures the
  image's color recurrence avoids paying for an over-sized cache.
  New public entry point `encode_argb_literals_with_width_selected`
  returns `(bytes, chosen_code_bits)`; the existing
  `encode_argb_literals_with_width` keeps its `Vec<u8>` signature
  (the `.webp` production path calls into the selected variant and
  drops the chosen-size scalar). Headline measurement: 32×32
  palette-heavy pseudo-random goes from 661 B (round-121, fixed
  `code_bits=8`) to 645 B (round-132, chosen `code_bits=7`) — a
  2.4 % saving on top of the round-121 cache writer. Noise /
  row-correlated / solid fixtures match the round-121 size byte-for-
  byte (the chooser correctly falls back to `code_bits=0`). The
  §5.2.3 header read on the decoder side is unchanged (it already
  accepted the full `[1, 11]` range per RFC 9649); round-trip stays
  bit-exact through `decode_lossless_image`. 6 new inline tests
  (slate-spec-legal, palette → non-zero, noise → zero, chosen-size-
  in-slate, per-decision round-trip, never-regress vs round 121).
  Total: 386 tests (was 380).

* **Clean-room round 131 (2026-05-25).** Published §2.5 `VP8 ` (lossy)
  **encoder API surface** — the `encode_vp8_lossy_rgba` /
  `encode_vp8_lossy_rgb24` / `encode_vp8_lossy_yuv420p` /
  `encode_vp8_lossy_yuva420p` free functions, the new `encoder_vp8`
  module with `make_encoder_with_quality` / `_with_qindex` /
  `_with_target_size` (and `_and_metadata` / `_and_freq_deltas`
  variants), `Vp8FreqDeltas`, `Vp8PsyStats`, `compute_psy_stats`,
  `freq_deltas_for_qindex`, `quality_to_qindex`,
  `DEFAULT_QUALITY` / `QUALITY_MIN..=QUALITY_MAX` / `QINDEX_MIN..=QINDEX_MAX`,
  and the registry-side `CODEC_ID_VP8 = "webp_vp8"` codec id with the
  `WebpVp8LossyEncoder` `Encoder` trait impl + `make_vp8_lossy_encoder`
  factory. **Status:** API-shape stubs. Every entry point validates
  input dims / buffer length and then returns
  `WebpError::Unsupported` (free functions) /
  `oxideav_core::Error::Unsupported` (trait impl). The wiring to a
  real VP8 lossy bitstream is blocked on the §13 / §14 pixel-driven
  encode round on the `oxideav-vp8` sibling crate: the current
  `oxideav-vp8 = "0.2"` encoder ships Phase 1
  ([`oxideav_vp8::encode_silent_keyframe`]) which emits a structurally
  valid VP8 keyframe but **ignores the caller's pixels** (every MB is
  `mb_skip_coeff = 1` with `DC_PRED` → constant-grey picture). Wiring
  that through to a WebP RIFF wrapper would produce garbage bytes —
  the exact failure mode the round-131 directive forbids — so this
  round lands the API shape only and reports the gap. The
  `encoder_vp8` module-level doc-comment enumerates the precise
  missing primitives on `oxideav-vp8` (forward WHT / forward DCT /
  forward quantization / per-MB pixel-driven encode driver / top-level
  I420 → keyframe driver). Once those land, the function bodies become
  a thin call into the new vp8 encoder + the existing
  `build::build_webp_file` RIFF wrapper — no API churn. 18 new
  published-API tests + 4 new inline unit tests; total 380 (was 347).

* **Clean-room round 130 (2026-05-25).** §5.2.2 **width-aware distance-code
  chooser** for the VP8L lossless encoder. Each backward reference now
  picks the smaller of the scan-line code (`D + 120`, the round-119
  default) and any §5.2.2 distance-map code `c ∈ 1..=120` whose
  `(xi, yi)` entry reconstructs to `D` for the image width — so a row-
  distance match (D = W) on a 256-wide image collapses from scan-line
  code 376 (prefix 16, 7 extra bits per emission) to map code 1
  (prefix 0, 0 extra bits). The reconstruction in
  `vp8l_decode::distance_code_to_pixel_distance` is identical for both
  forms, so the round trip stays bit-exact. New public helper
  `pixel_distance_to_distance_code(distance, image_width)`; new internal
  `encode_argb_literals_with_width(pixels, image_width)` that threads
  the actual image width into the chooser (wired by `encode_vp8l_payload`
  → `encode_webp_lossless` / `encode_vp8l_argb` / animation encoders).
  The legacy width-less `encode_argb_literals` is retained for test
  callers that exercise the entropy stage without spatial structure;
  it defaults to width = 1, which disables the chooser (no distance-map
  entry reconstructs typical distances at a single-pixel-wide row).
  Headline: a 256×256 row-repeating fixture shrinks from 972 B to 958 B
  (~1.4 % reduction); a 128×128 row-correlated fixture from 522 B to
  519 B (~0.6 %). Eight new tests cover chooser correctness
  (`distance_chooser_reconstructs_each_distance_map_entry`,
  `distance_chooser_picks_map_code_for_row_distance`,
  `distance_chooser_falls_back_to_scan_line_when_no_map_match`,
  `distance_chooser_width_one_uses_scan_line_for_large_distances`),
  per-prefix non-regression (`chooser_never_picks_larger_prefix_than_scan_line`),
  measured size-reduction
  (`width_aware_distance_beats_scan_line_only_on_row_correlated_image`,
  `width_aware_distance_compounds_on_many_short_row_offset_matches`,
  `width_aware_distance_headline_256x256_row_repeating`,
  `width_aware_distance_beats_scan_line_only_on_photo_like_image`,
  `width_aware_re_encode_of_real_fixture_is_smaller`), and round-trip
  bit-exactness across widths
  (`width_aware_round_trip_across_assorted_widths`). 356 tests total.

* **Clean-room round 127 (2026-05-25).** `AnimFrameMode::Auto` and
  `AnimFrameMode::Delta` are no longer `WebpError::Unsupported` — both
  now encode the caller's frames against the previous canvas using a
  **lossless dirty-rectangle delta** path on top of the existing VP8L
  encoder. `Delta` always emits the dirty-rect sub-frame (or, for the
  first frame / a frame whose dirty rect spans the whole canvas, a full
  keyframe); `Auto` evaluates both candidates and emits the smaller
  bitstream. Both honour the §2.7.1.1 `B = 1` / `D = 0`
  (overwrite, no dispose) ANMF semantics so the encoded file round-trips
  byte-for-byte through `decode_webp`'s canvas compositor. The
  even-offset constraint of §2.7.1.1 is preserved by aligning the dirty
  rect's top-left down to the nearest even coordinate. Identical
  consecutive frames emit a degenerate 2×2 sub-frame so duration timing
  is preserved without re-encoding. Headline: a 128×128 frame pair with
  an 8×8 changed block compresses from 87 476 B (all-Lossless) to
  43 986 B (Delta or Auto) — ~50 % size reduction with a byte-exact
  round trip. The original lossy-keyframe-vs-inter-frame-delta `Auto`
  semantics will return once `oxideav-vp8` ships a real lossy encoder;
  the dirty-rect path remains useful on lossless input regardless. New
  tests: `auto_and_delta_modes_emit_valid_files_round_127`,
  `dirty_rect_shrinks_anmf_payload_for_localised_change`,
  `auto_mode_picks_dirty_rect_on_localised_change`,
  `dirty_rect_canvas_coords_covers_only_the_changed_pixels`,
  `dirty_rect_is_none_on_identical_frames` (lib unit), and
  `auto_and_delta_modes_round_trip_byte_exact`,
  `delta_mode_three_frames_round_trip_byte_exact`,
  `auto_mode_picks_dirty_rect_on_small_localised_change`
  (`published_anim_api.rs`). 345 tests total.

* **Clean-room round 127 (2026-05-25).** Decoder-side §2.7.1.1
  **canvas compositing**. `decode_webp` / `decode_animation` now sizes
  a canvas from the §2.7.1 `VP8X` chunk, initialises it to the
  §2.7.1.1 `ANIM` `Background Color`, applies the previous frame's
  disposal method (`None` or `Background`) to its sub-rectangle, then
  draws the current frame at its `(x, y)` offset using its blending
  method: `Overwrite` copies the sub-rect pixels verbatim onto the
  canvas; `AlphaBlend` runs the §2.7.1.1 8-bit integer approximation
  of `blend.A = src.A + dst.A * (1 - src.A / 255)` /
  `blend.RGB = (src.RGB * src.A + dst.RGB * dst.A *
  (1 - src.A / 255)) / blend.A` (sRGB space, no gamma
  linearisation — matching the spec's stated 8-bit formula). Each
  returned `WebpFrame.rgba` is the full canvas snapshot after that
  frame is rendered, sized `canvas_w × canvas_h` (replacing the prior
  per-sub-rect-only convention). Frames whose declared rect overflows
  the canvas are rejected as `InvalidData`. The libwebp-encoded
  `animated-with-alpha.webp` fixture (all three ANMFs at offset (0,0)
  spanning the full 64×64 canvas) keeps decoding to the same per-frame
  RGBA buffers as before. New helpers: `lib::fill_canvas_rect`,
  `lib::blit_rect_overwrite`, `lib::blit_rect_alpha_blend`.

* **Clean-room round 127 (2026-05-25).** `AnimFrame::new` default
  `blend` switched from `BlendingMethod::AlphaBlend` to
  `BlendingMethod::Overwrite` so a full-canvas frame round-trips
  byte-for-byte through the new canvas compositor. Callers that need
  alpha-blending of a translucent sub-frame onto the existing canvas
  must build the struct literally and set `blend:
  BlendingMethod::AlphaBlend`. This is a behavioural change vs prior
  rounds (the existing `published_anim_api.rs` tests against varying-
  alpha frames updated to use the new semantics).

* **Clean-room round 124 (2026-05-25).** §2.5 `VP8 ` (lossy) **decode**
  path, routed through the `oxideav-vp8` sibling crate. Re-added the
  `oxideav-vp8 = "0.2"` dependency (vp8 0.2 now exposes a public
  `Vp8Error` at its crate root) with `default-features = false`, so it
  does not pull `oxideav-core` into the standalone build. New
  `vp8_decode` module routes a `WebpLossyChunk` payload to
  `oxideav_vp8::decode_vp8` (reconstructed, loop-filtered I420 key-frame)
  and converts it to interleaved RGBA via nearest-neighbour chroma
  up-sampling and the RFC 6386 §9.2 ITU-R BT.601 full-range YCbCr→RGB
  matrix. `decode_webp` / `decode_webp_image` now decode simple-lossy and
  `VP8X`-extended-lossy still images (with optional `ALPH`-over-`VP8 `
  alpha) instead of the previous `Unsupported(LossyVp8)` refusal. Added
  the `impl From<oxideav_vp8::DecodeError> for WebpError` adapter (a VP8
  inter-frame maps to `Unsupported`; every other decode failure to
  `InvalidData`) and the internal `Error::Vp8(DecodeError)` variant.
  Verified against the cwebp-encoded `lossy-1x1.webp` (simple) and
  `lossy-with-alpha-128x128.webp` (`VP8X` + `ALPH` + `VP8 `) fixtures;
  +13 tests (5 `vp8_decode` unit + rewired lossy-fixture/registry tests),
  339 total.

  *Deferred:* API-COMPAT.md specifies a
  `From<oxideav_vp8::Vp8Error> for WebpError` adapter against vp8's
  `Vp8Error` umbrella type. That type is on vp8 **master** (commit
  `d85d244`) but **not yet on crates.io** — it landed after the v0.2.0
  tag. The live decode path is wired against the published 0.2.0
  `DecodeError`; the `Vp8Error` adapter is a follow-up for once vp8
  publishes a release carrying it.

* **Clean-room round 121 (2026-05-25).** §5.2.1 / §5.2.3 **color-cache
  writer** in the VP8L encoder. `encode_argb_literals` now evaluates a
  256-entry color cache (`color_cache_code_bits = 8`) alongside the
  no-cache path and emits whichever is smaller; combined with the
  round-120 subtract-green chooser the encoder now picks the smallest of
  all four `(no-tx | subtract-green) × (no-cache | cache)` candidates.
  When the cache is enabled, the §3.8.3 `color-cache-info` header
  becomes `%b1 8` (1-bit flag + 4-bit `code_bits`), the GREEN alphabet
  grows to `256 + 24 + 256 = 536` symbols per §6.2.3, and a literal
  repeat is written as a single §5.2.3 cache code (`256 + 24 + index`)
  instead of four channel literals. New `EncoderColorCache` helper
  mirrors the decoder's `vp8l_decode::ColorCache` semantics bit-for-bit
  (hash formula `(0x1e35a7bd * argb) >> (32 - code_bits)`,
  zero-initialised entries, every emitted pixel re-inserted in stream
  order — both literals and every pixel covered by a §5.2.2
  backward-reference copy). A new `cacheify_tokens` 2nd-pass walks the
  LZ77 token stream and rewrites any `Literal(argb)` whose hashed slot
  already holds `argb` to a `Token::CacheRef { index }`. Cache state
  stays in sync with the decoder by inserting every covered pixel of a
  `Copy` token. New test-only `encode_argb_literals_color_cache`
  forces the cache path for the round-121 size-reduction comparison;
  production callers stay on the chooser. Headline: a 32×32
  pseudo-random small-palette (8 distinct ARGB colors) image compresses
  from 1131 B (no-cache LZ77) to 622 B (color-cache on), a ~45 % size
  reduction. Uncorrelated-noise images stay on the no-cache no-tx path
  (the chooser never regresses). Round-trip is bit-exact through
  `decode_lossless_image` on every existing fixture + the new
  color-cache round-trip + meta-prefix-header read-back tests. New
  tests: `encoder_color_cache_hash_matches_decoder_hash`,
  `encoder_color_cache_starts_zero_initialized`,
  `encoder_color_cache_insert_then_contains_round_trips`,
  `cacheify_tokens_collapses_repeat_literal_into_cache_ref`,
  `cacheify_tokens_copy_updates_cache_for_subsequent_literal`,
  `color_cache_path_round_trips_via_public_entry_points`,
  `color_cache_beats_no_cache_on_small_palette_image`,
  `color_cache_chooser_does_not_regress_on_uncorrelated_noise`,
  `color_cache_header_round_trips_through_meta_prefix_reader`. The
  crate still builds + tests under `--no-default-features` (the cache
  uses only the existing `oxideav-core`-free decode helpers).

* **Clean-room round 120 (2026-05-24).** §3.5.3 / §3.8.2 **subtract-green
  transform** forward path in the VP8L encoder. New `apply_subtract_green`
  helper subtracts the green channel from red and blue per pixel
  (`r := (r - g) & 0xff`, `b := (b - g) & 0xff`), the exact inverse of
  the decoder's existing `vp8l_transform::inverse_subtract_green`.
  `encode_argb_literals` now evaluates both the no-transform and the
  subtract-green paths and emits whichever is smaller — the §3.8.2
  transform header costs only three bits (`%b1 %b10`, transform type 2
  with no body), so on green-correlated natural-image-like content the
  per-channel red/blue entropy drops sharply for a near-free win;
  uncorrelated noise falls back to no-transform (the chooser never
  regresses). The literal-only and subtract-green-forced paths stay
  available as `encode_argb_literals_only` and
  `encode_argb_literals_subtract_green` for the round-119/120 size
  comparison tests. Headline: a 32×32 synthetic green-correlated image
  (red and blue track green plus small noise) compresses from 3243 B
  (no-transform) to 2211 B (subtract-green) — a ~32 % size reduction.
  Round-trip is bit-exact through `decode_lossless_image` because the
  decoder's §4 inverse pass undoes the encoded transform. New tests:
  `apply_subtract_green_is_inverse_of_inverse_subtract_green`,
  `apply_subtract_green_only_touches_red_and_blue`,
  `subtract_green_beats_no_transform_on_green_correlated_image`,
  `encode_argb_literals_chooses_smaller_path`,
  `subtract_green_path_round_trips_via_public_entry_points`,
  `encode_argb_literals_does_not_regress_on_uncorrelated_noise`.
  The crate still builds + tests under `--no-default-features` (the
  forward transform uses no `oxideav-core` surface).

* **Clean-room round 119 (2026-05-24).** §5.2.2 **LZ77 backward-reference
  matching** in the VP8L encoder. `encode_argb_literals` now runs a
  hash-chain matcher (`Lz77Matcher`) over the ARGB pixel buffer before
  the entropy stage: every repeated run of `>= MIN_MATCH` (3) pixels at
  scan-line distance `D` becomes a §5.2.2 length + distance backward
  reference instead of `length` separate ARGB literals. Length values
  flow through the GREEN alphabet's `256 + length_prefix` symbols;
  distances use prefix code #5 with the §3.6.2.2.1 scan-line form
  `distance_code = D + 120` (always valid per the spec's `> 120` branch
  — the §3.6.2.2.1 distance map is an optional decoder convenience the
  encoder declines to use). The new `value_to_prefix` helper is the
  exact inverse of the decoder's `read_lz77_value` prefix-value
  transform, round-tripped through the live decoder at a spread of
  values and at every length `1..=MAX_MATCH` (4096). The previous
  literal-only emit path stays available as `encode_argb_literals_only`
  for the size-reduction comparison test. Headline: a 64×64 image whose
  rows repeat an 8-color palette compresses from 4758 B (literal-only)
  to 163 B (LZ77), a ~97 % reduction; pixels with no exploitable
  repetition (xorshift noise) come out the same size. New tests:
  `value_to_prefix_small_values_have_no_extra_bits`,
  `value_to_prefix_round_trips_length_range`,
  `value_to_prefix_round_trips_through_decoder`,
  `round_trip_solid_color_uses_lz77_copy`,
  `round_trip_periodic_pattern_uses_overlapping_copy`,
  `lz77_beats_literal_only_on_repetitive_image`,
  `lz77_round_trips_incompressible_pixels`,
  `round_trip_splits_match_at_max_length`. The crate still builds under
  `--no-default-features` (the matcher uses only the existing
  `oxideav-core`-free decode helpers).

* **Clean-room round 118 (2026-05-24).** Re-exposed the
  **published-0.1.5 animation-encode API** for the VP8L-lossless path, on
  top of the round-115 VP8L encoder + the §2.7.1.1 `ANIM` / `ANMF` framing
  (see `API-COMPAT.md`). Standalone (no `oxideav-core` dep):
  * `build_animated_webp(frames) -> Result<Vec<u8>, WebpError>` and
    `build_animated_webp_with_options(frames, opts)` — assemble a
    multi-frame `.webp` (`RIFF`/`WEBP` + `VP8X(A[,L][,I][,E][,X])` +
    [`ICCP`] + `ANIM` + `ANMF…ANMF` + [`EXIF`] + [`XMP `]). The `VP8X`
    canvas is sized to cover every frame; each frame's pixels become a
    §2.6 `VP8L` chunk inside the `ANMF` Frame Data.
  * `AnimFrame { pixels, width, height, x, y, duration, blend, dispose,
    mode }` (flat RGBA `pixels`; even `x`/`y`; `AnimFrame::new` helper),
    `AnimFrameMode { Auto, Delta, Lossless }` (`Lossless` wired;
    `Auto`/`Delta` → `WebpError::Unsupported`, blocked on `oxideav-vp8`
    #1041), `AnimEncoderOptions { loop_count, background_rgba, metadata,
    delta }`, `DeltaConfig` (`max_components_override` /
    `auto_inner_threshold_bytes` / `msssim_downsample_kernel` builders),
    `DownsampleKernel { Box, Gaussian }`.
  * `decode_webp` now assembles an animated file into N `WebpFrame`s
    (per-frame `VP8L` decode + optional `ALPH` alpha override), populating
    `WebpImage::anim_background_rgba` / `anim_loop_count`.
  * Standalone test `tests/published_anim_api.rs` (runs under
    `--no-default-features`): 3-frame round trip, options + metadata,
    blend/dispose/offset carry, `Auto`/`Delta` `Unsupported`, and the
    `DeltaConfig` builder chain.

* **Clean-room round 117 (2026-05-24).** Re-exposed the
  **published-0.1.5 lossless-encode public names** on top of the round-115
  in-crate VP8L encoder (see `API-COMPAT.md`). All available standalone
  (no `oxideav-core` dep):
  * `encode_vp8l_argb(argb, width, height) -> Result<Vec<u8>, WebpError>`
    — a **bare** §2.6 / §3.4 `VP8L` bitstream (image-header + image
    stream), **no** RIFF wrapper. `argb` is `width * height` packed ARGB
    (`(a<<24)|(r<<16)|(g<<8)|b`); the §3.4 `alpha_is_used` header bit is
    auto-detected.
  * `encode_vp8l_argb_with(argb, width, height, has_alpha)` — the fixed
    (non-RDO) form: `has_alpha` sets the header bit explicitly.
  * `encode_vp8l_argb_with_metadata(w, h, &argb, has_alpha, &meta) ->
    Result<Vec<u8>, WebpError>` — a complete `.webp`. Emits the simple
    `VP8L` layout when opaque and metadata-free, else auto-promotes to the
    §2.7 extended `VP8X` layout (`VP8X` + [`ICCP`] + `VP8L` + [`EXIF`] +
    [`XMP `], chunks in §2.7 order, flag octet declaring exactly the
    present features). Round-trips through `decode_webp`; embedded metadata
    reads back via `extract_metadata`.
  * `WebpMetadata<'a> { icc/exif/xmp: Option<&'a [u8]> }` (borrowed encode
    input, `::default()` = embed nothing) and
    `WebpMetadataOwned { icc/exif/xmp: Option<Vec<u8>> }` (owned,
    registry-side; `as_borrowed()` + `From<WebpMetadataOwned> for
    WebpFileMetadata`).
  * `pub const CODEC_ID_VP8L = "webp_vp8l"`.
* **Registry `webp_vp8l` encoder (dual-API).** `register` now also
  installs a VP8L encoder codec under `CODEC_ID_VP8L` (alongside a decoder
  for symmetry). It accepts `Rgba` / `Rgb24` input (the `Rgb24` path
  streams as fully opaque, no 3→4 expansion) and emits a `.webp` per
  frame. Direct factories `registry::make_encoder(&params)`,
  `registry::make_encoder_with_metadata(&params, WebpMetadataOwned)`, and
  the `VideoFrame`-flavoured `registry::encode_vp8l_frame(...)` keep the
  registry path + direct factory dual-API convention.
* **New tests.** `tests/published_encode_api.rs` (standalone, runs under
  `--no-default-features`): bare-bitstream shape, simple/extended layout
  selection, metadata embed + read-back, forced-alpha round trip,
  dimension-mismatch rejection. Plus in-crate unit tests for the bare
  encode helpers (`vp8l_encode`) and the registry encoder
  (round-trip RGBA, Rgb24-as-opaque, VP8X-on-metadata, NeedMore/Eof).

* **Clean-room round 116 (2026-05-24).** First step of restoring the
  **published-0.1.5 public decode API shape** so downstream consumers
  compile again (see `API-COMPAT.md`). New published-shape decode types
  (all available standalone, no `oxideav-core` dep):
  * `WebpImage { frames: Vec<WebpFrame>, metadata: WebpFileMetadata,
    anim_background_rgba: Option<[u8; 4]>, anim_loop_count: Option<u16> }`.
  * `WebpFrame { rgba: Vec<u8>, width: u32, height: u32, duration_ms: u32 }`
    — `rgba.len() == width * height * 4`, tightly packed `[R, G, B, A]`,
    no stride padding (drops straight into `image::ImageBuffer::from_raw`).
  * `WebpFileMetadata { icc: Option<Vec<u8>>, exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>> }`.
  * `WebpError { InvalidData, Unsupported, Eof, NeedMore }`, with
    `From<Error>` mapping the rich internal error onto the coarse
    published shape.

### Changed

* **`decode_webp` restored to the published shape.** It now returns
  `Result<WebpImage, WebpError>` (was the rebuild's own unpublished
  `Result<Vec<u8>, Error>`). Built on the already-rebuilt §4–§6 VP8L
  decoder: a simple/extended-lossless file yields a single-frame
  `WebpImage`. VP8 lossy and animation paths are reported
  `WebpError::Unsupported` (never faked) until those decoders are
  rebuilt. The flat-`Vec<u8>` behaviour is preserved via
  `decode_webp(..).frames[0].rgba`; the low-level
  `decode_webp_image -> DecodedWebp` and `decode_lossless_image` helpers
  are unchanged and remain as additional API.
* New `extract_metadata(bytes) -> Result<WebpFileMetadata, WebpError>` —
  metadata-only walk (ICC / Exif / XMP), decodes no pixels.
* New standalone integration test `tests/published_decode_api.rs`
  (runs under `--no-default-features`): builds an in-memory RGBA buffer,
  encodes via the VP8L lossless encoder, decodes via `decode_webp`, and
  asserts the round-tripped `WebpFrame.rgba` is byte-exact with
  `len == w * h * 4` — proving the flat `image`-crate buffer shape.

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
