# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
