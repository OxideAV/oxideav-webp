# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(vp8-enc)* psy-RDO source analysis + per-frame target-size rate
  control. Each `send_frame` on the non-explicit factories
  ([`make_encoder`] / [`make_encoder_with_quality`] /
  [`make_encoder_with_qindex`]) now runs a one-pass MAD analyser
  over the source luma plane (`encoder_vp8::compute_psy_stats`)
  and uses the resulting `PsyStats { mean_activity,
  high_variance_fraction, mb_count }` to bias both the per-segment
  quant deltas and the per-frequency AC/DC quant deltas before
  invoking the underlying VP8 encoder. CSF-style activity masking:
  frames with `mean_activity ≥ 16` (typical natural-image
  textures) get one step coarser high-freq AC bins (saves bits the
  eye won't notice on busy content); frames with `mean_activity
  < 6` (sky photos / flat plates) get one step finer to suppress
  visible banding on the rare edges. Variance-segment 3 delta also
  modulates: widens on segment-3-heavy frames (most MBs textured →
  recover bits there), narrows on segment-3-empty frames (rare
  textured MB shouldn't get hammered). Modulation strength scales
  with qindex so the qindex=0 endpoint reproduces the pre-psy
  bitstream byte-for-byte. New
  [`encoder_vp8::make_encoder_with_target_size`] factory hits a
  caller-supplied byte budget within ±10 % via a 5-iteration
  bisection over qindex (worst-case 6× single-shot encode cost,
  converges in 2-3 iterations on natural-image content because the
  size-vs-qindex curve is monotone). Measured win on the AC-rich
  noisy 128×128 fixture at qindex 64: −1.1 % bytes and +0.01 dB
  PSNR vs the no-psy baseline; both bitstreams cross-decode
  cleanly through libwebp's `dwebp`. Explicit `*_and_freq_deltas`
  factories continue to pass freq-deltas through verbatim and
  bypass the psy modulation, preserving the
  `Vp8FreqDeltas::default()` byte-identical guarantee for tuning
  callers.
- *(test)* `vp8_lossy_psy_rdo` integration suite (7 tests):
  `psy_stats_distinguish_smooth_vs_noisy` (asserts the analyser
  ranks the smooth fixture at activity 3.0 and the noisy fixture
  at 18.4 on the same 128×128 frame), `psy_stats_sub_mb_frame_returns_default`
  (8×8 frames return `mb_count=0` rather than panic),
  `psy_modulation_changes_bitstream_and_keeps_quality_on_noisy_source`
  (psy-on vs psy-off bytes / PSNR comparison + dwebp cross-decode),
  `psy_modulation_smooth_source_does_not_blow_up_bytes` (bytes
  growth ≤ 25 % on smooth content),
  `target_size_rate_control_hits_within_tolerance` (6 KB / 12 KB
  / 20 KB targets all land within ±15 %),
  `target_size_rate_control_handles_unreachable_target` (50-byte
  target on a 128×128 noisy fixture lands at the smallest
  achievable size without panicking),
  `psy_modulation_collapses_to_baseline_at_qindex_zero`
  (high-quality byte-identical guarantee).

## [0.1.2](https://github.com/OxideAV/oxideav-webp/compare/v0.1.1...v0.1.2) - 2026-05-05

### Other

- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-webp/pull/502))

## [0.1.1](https://github.com/OxideAV/oxideav-webp/compare/v0.1.0...v0.1.1) - 2026-05-05

### Added

- *(vp8l-enc)* two-pass cost-modelled LZ77 + predictor-tile-bits RDO sweep

### Other

- *(changelog)* clean duplicate Unreleased entries after rebase
- release v0.1.0 ([#14](https://github.com/OxideAV/oxideav-webp/pull/14))

### Added

- *(vp8l-enc)* **two-pass cost-modelled LZ77** with cost-aware
  match selection. Pass 1 runs greedy first-match; the resulting
  histogram seeds a per-alphabet bit-cost model
  (`-log2(p) × 16` in 1/16-bit units, capped at 40 bits). Pass 2
  re-walks the LZ77 hash chain and at each position picks the
  candidate match whose bit cost per pixel under the model is
  lowest, plus a one-step lazy-lookahead defer when
  `literal-here + match-at-i+1` is cheaper than `match-here`. Both
  candidate streams score against the SAME model; the cheaper one
  wins. Adds 1-2 % byte savings on photographic content
  (1024×768 photo: -525 B; 64×64 cache stress: -3 B; 128×128
  natural: -14 B vs the K=16 baseline).
- *(vp8l-enc)* **predictor tile-bits RDO sweep** over
  `{8, 16, 32}` px tiles. Previously fixed at 16 px (libwebp
  default); the encoder now picks the smallest output across the
  three tile sizes per image. Wins on smooth photographic content
  where 32-px tiles' smaller predictor sub-image saves a few
  hundred bits, and on dense small-feature content (icons, line
  art) where 8-px tiles' finer per-tile mode accuracy beats the
  side-image overhead. Sweep is gated to predictor-on /
  non-palette trials so the trial budget grows by 3× only on the
  trials that benefit. New `predictor_tile_bits` field on
  `EncoderOptions` lets explicit callers pin a single value.

## [0.1.0](https://github.com/OxideAV/oxideav-webp/compare/v0.0.11...v0.1.0) - 2026-05-05

### Added

- *(vp8l-enc)* K=16 meta-Huffman + wider LZ77 + denser cache_bits sweep
- *(vp8-enc)* quality-driven per-frequency quant matrix + fix VP8L simple-Huffman bit leak

### Other

- promote to 0.1

## [0.0.11](https://github.com/OxideAV/oxideav-webp/compare/v0.0.10...v0.0.11) - 2026-05-04

### Added

- *(vp8-enc)* wire per-frequency AC/DC quantiser deltas via vp8 0.1.7
- *(vp8l-enc)* add K=8 meta-Huffman trial to per-tile grouping sweep
- *(vp8-enc)* re-wire per-segment LF deltas via vp8 0.1.6

### Fixed

- *(vp8-decode)* bit-exact YUV→RGB + fancy chroma upsample vs libwebp

### Other

- *(vp8l-decode)* bit-exact lossless corpus walker

### Added

- *(vp8-enc)* quality-driven per-frequency quantiser-matrix preset
  (closes the libwebp-parity gap noted in #465). The `quality` knob
  on `make_encoder_with_quality` (and `make_encoder_with_qindex`) now
  also drives the five per-frequency AC/DC qindex deltas (RFC 6386
  §6.6 + §9.6) in addition to the existing per-segment QP / LF
  deltas. The new `freq_deltas_for_qindex` curve keeps every delta at
  zero at qindex=0 (no shift on top of the already-finest step), then
  widens the high-frequency bins as quality drops: at qindex=127 the
  preset lands on `[y_dc=0, y2_dc=-2, y2_ac=+4, uv_dc=0, uv_ac=+4]`
  — coarser Y2/chroma AC for compression, finer Y2 DC to suppress
  visible block-mean banding, chroma DC held put to avoid colour
  shifts. Compression behaviour stays strictly byte-size monotone
  with quality on AC-rich content; bitstreams remain spec-compliant
  under libwebp's `dwebp`. The explicit
  `make_encoder_with_qindex_and_freq_deltas` /
  `make_encoder_with_quality_and_freq_deltas` factories pass user
  freq-deltas through *verbatim* (no preset added on top), so callers
  that have already done their own perceptual tuning aren't double-
  shifted; an explicit `Vp8FreqDeltas::default()` argument still
  reproduces the pre-#465 bitstream byte-for-byte.
- *(test)* `vp8_quality_driven_quant_matrix_q20_vs_q80_roundtrips`
  exercises the new preset end-to-end: encodes the AC-rich
  `build_noisy_pattern` at q=20 and q=80, asserts the bitstreams
  differ, both round-trip through `decode_webp`, q=20 produces a
  strictly smaller file than q=80 (≈4.5x growth on the test
  fixture), and both cross-decode through libwebp's `dwebp` when
  installed.
- *(test)* `vp8l_fuzz_simple_huffman_leak.rs` regression suite for
  the `vp8l_lossless_roundtrip` fuzz crash on a 1×1 black-opaque
  input (#458). Pre-fix the encoder's `try_emit_simple_huffman`
  wrote `simple=1, num_symbols-1` into the bitstream *before*
  branching on the symbol-index range, then returned `None` for any
  single-active-symbol alphabet whose only nonzero index was ≥ 256
  (typical for green-alphabet cache refs at indices ≥ 280). The
  caller's normal-tree fallback then wrote its own header on top of
  the leaked 2-bit prefix, desynchronising every subsequent tree
  read. Reproduced by predictor + colour-cache on a 1×1 black-opaque
  ARGB pixel: the residual collapses to `0x00000000`, lands at cache
  index 0 (matching the cache's zero-initialised state), and the
  only emitted symbol is `CacheRef{index: 0}` → green_freq[280] = 1.

### Fixed

- *(vp8l-enc)* `try_emit_simple_huffman` no longer leaks the
  `simple=1, num_symbols-1` prefix into the bitstream when bailing
  to the normal-tree path. Eligibility checks (`s >= 256` for the
  1-symbol case, `a >= 256 || b >= 256` for the 2-symbol case) now
  run *before* any `bw.write` call, so the function either commits
  its full header in one shot or returns `None` without touching the
  writer. Pre-fix, our own decoder rejected the resulting stream
  with `"VP8L: canonical Huffman length table self-collides"`; with
  the fix the `vp8l_lossless_roundtrip` fuzz target completes 350K+
  runs without a crash and the `oxideav_encode_webp_decode_lossless`
  cross-decode target completes 350K+ runs cleanly through libwebp.

- *(vp8-enc)* per-frequency AC/DC quantiser delta knob. Wires the
  `y_dc_delta` / `y2_dc_delta` / `y2_ac_delta` / `uv_dc_delta` /
  `uv_ac_delta` fields published in `oxideav-vp8` 0.1.7 (#417)
  through to the WebP single-frame lossy path via the new
  `encoder_vp8::Vp8FreqDeltas` struct plus
  `encoder_vp8::make_encoder_with_qindex_and_freq_deltas` /
  `make_encoder_with_quality_and_freq_deltas` factories. Defaults
  (all-zero) reproduce the pre-#417 bitstream byte-for-byte; non-zero
  values bias bits toward a specific frequency band (e.g. negative
  `y_dc_delta` for finer luma AC where banding shows up, positive
  `uv_ac_delta` to lighten chroma AC on screen-recording / line-art
  content). Each delta is clamped to `[-15, 15]` by the underlying
  encoder. Closes the deferral noted in `c8c1d69` and bumps
  `oxideav-vp8` minimum version to `0.1.7`.
- *(test)* `vp8_per_frequency_deltas_change_bitstream_and_round_trip`
  encodes the same YUV420P frame twice at the same qindex (zero
  deltas vs a non-trivial mix) and asserts the bitstreams differ,
  both round-trip through `decode_webp`, and both cross-decode
  through libwebp's `dwebp` binary when installed (silent skip
  otherwise — keeps CI green on hosts without libwebp).
- *(vp8l-enc)* extend the meta-Huffman per-tile grouping sweep from
  K = {1, 2, 4} to K = {1, 2, 4, 8}. Each `encode_image_stream` call
  speculatively emits the K=8 candidate (gated on ≥ 16384 px = 128×128
  pixels — below it the eight extra Huffman-tree headers dominate any
  per-cluster savings) alongside the existing K=2 / K=4 trials and
  keeps the byte-smallest. The k-means clustering already generalised
  to arbitrary K so this is a one-line widening of the K-loop plus the
  matching minimum-pixel-count gate. The decoder side already supported
  arbitrary K (the meta-image's group-id field is 16 bits wide). Pushes
  VP8L encoder closer to libwebp's per-tile entropy-image parity on
  fixtures with several visually distinct sub-regions. The K=8 candidate
  is now also probed inside every RDO trial (the 32-trial sweep already
  exercises every transform × cache combination, and the K=8 trial
  rides inside each one), so it picks the winner across the joint
  configuration space without extra outer iterations.
- *(test)* `meta_huffman_k8_round_trips_eight_strip_fixture` +
  `meta_huffman_k8_shrinks_or_matches_smaller_k_on_eight_strip` —
  exercises the K=8 path on a 128×128 fixture with eight horizontal
  strips of distinctly-different statistics; verifies round-trip
  through the in-crate decoder and that the encoded stream is at least
  half the raw RGBA size.
- *(vp8-enc)* re-wire per-segment loop-filter delta tuning. Now that
  `oxideav-vp8` 0.1.6 publishes `Vp8EncoderConfig::segment_lf_deltas`
  (#337), the WebP single-frame lossy config plumbs the
  quality-scaled deltas computed by `segment_lf_deltas_for_qindex`
  into the encoder. Variance-segment 0 (smooth) gets a *lighter*
  deblock so flat regions don't over-smooth; variance-segment 3
  (textured) gets a *heavier* deblock so the coarser per-segment QP's
  DCT block boundaries get masked. Closes the wiring previously
  deferred in `a46ecf2 fix(vp8-enc): drop segment_lf_deltas — not on
  published vp8 0.1.5`. Bumps `oxideav-vp8` minimum version to
  `0.1.6`.
- *(test)* `lossless_corpus` integration test walks the seven
  workspace `docs/image/webp/fixtures/lossless-*` fixtures and asserts
  bit-exact RGBA equality with each `expected.png` ground truth.
  Covers the trivial single-pixel case, opaque RGB, RGBA, photo-like
  content (subtract-green + predictor + LZ77 + Huffman), the
  colour-cache hit path, the colour-indexing/palette transform, and
  the cross-colour transform. All seven fixtures pass on the first
  CI run, which proves the VP8L decoder is bit-exact with libwebp on
  the workspace docs corpus.

### Fixed

- *(vp8-decode)* bit-exact YUV→RGB + chroma upsampling alignment with
  libwebp's reference path. The decoder's YUV→RGB conversion now uses
  libwebp's exact two-stage truncating fixed-point arithmetic
  (`MultHi(v, coeff) = (v*coeff)>>8` per term, then `>> 6` on the sum,
  matching `src/dsp/yuv.h::VP8YUVToR/G/B`) instead of folding the
  shifts into a single `(KY*y + KC*c + …) >> 14` — the previous
  algebraically-equivalent form lost a few low bits of precision per
  channel, biasing every output ~1 LSB high vs libwebp. Chroma
  upsampling now also matches libwebp's "fancy" 9/3/3/1 weighted
  bilinear interpolation (`src/dsp/upsampling.c::UPSAMPLE_FUNC`)
  instead of nearest-neighbour replication. Combined effect on the
  lossy_corpus integration test against libwebp-cwebp ground truth:
  q100 + near-lossless + 1×1 fixtures are now bit-exact (previously
  ~20% per-channel exact match), q75 + with-alpha 73-75% per-channel
  (PSNR 48-50 dB), q1 ~33% (PSNR 42 dB). Three previously
  ReportOnly fixtures promoted to BitExact tier.

## [0.0.10](https://github.com/OxideAV/oxideav-webp/compare/v0.0.9...v0.0.10) - 2026-05-04

### Added

- *(anim-enc)* per-frame mode selection (lossy vs lossless) ([#335](https://github.com/OxideAV/oxideav-webp/pull/335))
- *(vp8-enc)* per-segment QP + LF delta tuning by quality ([#334](https://github.com/OxideAV/oxideav-webp/pull/334))
- *(vp8l-enc)* K=4 meta-Huffman + near-lossless smoothing close #380
- *(vp8l-enc)* add meta-Huffman per-tile grouping (2 groups)
- *(vp8l-enc)* add meta-Huffman per-tile grouping (2 groups)
- *(vp8l-enc)* add near-lossless preprocessing knob
- gate oxideav-core behind default-on `registry` feature ([#358](https://github.com/OxideAV/oxideav-webp/pull/358))
- *(webp)* pick best ALPH filter mode per alpha plane
- *(vp8l-enc)* add colour-indexing (palette) transform
- *(vp8l-enc)* widen predictor mode pool to all 14 modes
- *(webp)* surface VP8X metadata (ICCP / EXIF / XMP) to callers

### Fixed

- *(vp8-enc)* drop segment_lf_deltas — not on published vp8 0.1.5
- *(vp8l-enc)* simple-Huffman + palette RDO bias close #379
- *(vp8l)* correct predictor mode 5 nesting per RFC 9649 §4.1

### Other

- cargo fmt for #334 / #335 helpers
- Revert "feat(vp8l-enc): add meta-Huffman per-tile grouping (2 groups)"
- end-to-end external roundtrip via libwebp (lossless VP8L)

## [0.0.9](https://github.com/OxideAV/oxideav-webp/compare/v0.0.8...v0.0.9) - 2026-05-03

### Fixed

- *(vp8l)* keep Huffman meta-tree Kraft-complete after length-limit

### Other

- allow style-only clippy lints in Kraft regression test

## [0.0.8](https://github.com/OxideAV/oxideav-webp/compare/v0.0.7...v0.0.8) - 2026-05-03

### Fixed

- *(vp8l)* swap red↔blue in cross-color coefficient packing
- *(webp)* use libwebp 14-bit BT.601 constants for VP8 YUV→RGB
- *(clippy)* regroup xorshift seed hex digits into 4-nibble groups
- *(clippy)* unnecessary cast + doc lazy continuation

### Other

- batch cache_add across LZ77 backref runs
- specialise apply_color_index per bits_per_pixel
- hoist tile lookup out of predictor / cross-color inner loops
- disable strip_transparent_color in roundtrip harnesses
- rustfmt — wrap PALETTE_WEBP include_bytes per fmt rules
- split pixel decode loop into meta / no-meta specialisations
- tighten select_argb / clamp_add_sub_argb predictor inner loops
- drop residual.to_vec() memcpy in apply_predictor
- SWAR-ify add_argb / avg2 / apply_subtract_green per-byte math
- HuffmanTree::decode 8-bit LUT fast-path (3-5× entropy stage)
- criterion harness for VP8L decode hot paths
- fix clippy doc-list-overindent + complex-type lints
- revert one-time panic + record observed numbers
- rustfmt + one-time summary dump (reverted next commit)
- wire lossy fixture corpus into integration test (lossy_corpus.rs)
- add daily fuzz workflow shim

## [0.0.7](https://github.com/OxideAV/oxideav-webp/compare/v0.0.6...v0.0.7) - 2026-05-03

### Other

- rustfmt — collapse short assert_eq! line in issue #8 test
- predictor TR at rightmost column uses leftmost pixel of current row
- drop trait imports redundant on Box<dyn …>
- silence clippy unnecessary_unwrap + unused ch
- rustfmt — wrap long arg lists
- accept Rgb24 (VP8L + VP8) and Yuva420P (VP8+ALPH) input
- rustfmt — wrap long expect_err line
- replace .err().expect() with .expect_err() for clippy::err_expect
- silence clippy::needless_range_loop in test helpers
- fix CI compile + rustfmt issues from new test files
- ~57 unit tests for Huffman parser + simple/normal/build edge cases

### Fixed

- VP8L predictor: top-right (TR) neighbour at the rightmost column now
  uses the leftmost pixel of the current row (column 0 of row y), as
  specified by RFC 9649 §4.1, instead of the LEFT neighbour. The wrong
  TR caused libwebp-encoded streams to decode with cascading per-row
  errors that surfaced as off-by-channel pixel mismatches. Closes #8.
  Both the decoder (`apply_predictor`) and the encoder
  (`predict_argb` mirror) were corrected; pure self-roundtrips were
  bit-exact under both definitions, so the bug only showed up against
  spec-conformant encoders like libwebp.

## [0.0.6](https://github.com/OxideAV/oxideav-webp/compare/v0.0.5...v0.0.6) - 2026-05-03

### Other

- count run codes toward max_symbol when use_length=1
- use struct-update syntax for EncoderOptions in strip-color test
- strip RGB from fully-transparent pixels by default
- add libwebp-style quality knob (0..=100) alongside qindex
- replace `webp` crate with libloading-based libwebp shim
- Add roundtrip harnesses that ensure compatibility with libwebp for lossy WebP. Does not assert the contents stayed the same, we cannot do that for lossy data.
- Add roundtrip harnesses that ensure compatibility with libwebp
- Add a roundtrip fuzzing harness to verify lossless encoding/decoding correctness
- drop unused TimeBase import in vp8_lossy_roundtrip
- replace never-match regex with semver_check = false
- encoder RDO sweep over transforms + colour-cache widths
- add animated WebP encode + fix ANMF bit order
- migrate to centralized OxideAV/.github reusable workflows
- adopt slim VideoFrame shape
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

### Added

- Animated WebP encode via `build_animated_webp` — emits a
  `RIFF/WEBP/VP8X + ANIM + ANMF...ANMF` file from a slice of
  `AnimFrame` entries, each carrying duration_ms + offset + blend +
  disposal flags. Per-frame data is encoded losslessly through the
  existing VP8L pipeline.
- VP8L encoder RDO loop. `encode_vp8l_argb` now probes 32 candidate
  configurations (predictor on/off × colour-transform on/off ×
  subtract-green on/off × colour-cache size in {off, 6, 8, 10}) and
  keeps the smallest encoded variant. Callers wanting a deterministic
  fixed configuration can still use `encode_vp8l_argb_with`.

### Fixed

- ANMF flags decoder swapped the disposal/blending bits relative to
  the WebP container spec. Bit 0 is now correctly read as the
  blending method (0 = blend, 1 = overwrite) and bit 1 as the
  disposal method (0 = none, 1 = dispose-to-background).

## [0.0.5](https://github.com/OxideAV/oxideav-webp/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.4](https://github.com/OxideAV/oxideav-webp/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- bump oxideav-vp8 dep to 0.1
- update README + lib doc for colour transform, VP8X, and ALPH
- encode RGBA through VP8 lossy with an ALPH sidecar
- emit VP8X extended header when RGBA frames carry alpha
- add VP8L colour transform for G-R/B decorrelation
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
- add subtract-green + predictor transforms and 256-entry color cache
