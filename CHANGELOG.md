# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
