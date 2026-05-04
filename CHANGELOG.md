# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Added

- *(anim-enc)* per-frame mode selection for animated WebP. New
  [`build_animated_webp_with_options`] entry point + `AnimEncoderOptions`
  knob bag with three policies: `Lossless` (the historic
  always-VP8L behaviour, kept for `build_animated_webp`), `Lossy`
  (always VP8 + ALPH), and `Auto` (the default — encode each frame
  both ways and pick whichever sub-chunk(s) lay out a smaller ANMF
  payload). Mirrors libwebp's `WebPAnimEncoderAdd` per-frame
  decision. The WebP container permits mixing VP8L and VP8+ALPH
  frames in a single animation and the in-crate decoder already
  handles both shapes. Closes #335.
- *(vp8-enc)* per-segment quantiser + loop-filter delta tuning driven
  by quality. The lossy WebP encoder now routes every keyframe through
  `make_encoder_with_config` with `enable_segments = true` and
  per-segment QP / LF deltas scaled with the frame qindex (RFC 6386
  §10 + §15.2). The variance classifier in `oxideav-vp8` lands smooth
  MBs in segment 0 and high-variance MBs in segment 3; we then spend
  more bits on the smooth segment (where banding is visible at high
  QP) and save bits on the textured segment (where DCT noise is
  masked). Delta magnitudes scale linearly with qindex so the tuning
  is most aggressive at low quality and collapses toward zero at
  high quality. Closes #334.
- *(vp8l-enc)* K=4 meta-Huffman per-tile grouping, in addition to the
  existing K=2 trial. The encoder now tries K=1 (single-group),
  K=2, and (above 4096 px) K=4 splits; whichever produces the
  shortest bitstream wins. Tile clustering uses k-means++ farthest-
  first seeding plus a 2-iteration k-means assignment pass. K=4 is
  a clear win on images with several distinct visual regions
  (typical photos with sky / foreground / detail). Closes #380.
- *(vp8l-enc)* near-lossless smoothing pass. Runs after the per-pixel
  bit-shift quantisation: walks the 3×3 neighbourhood of every
  interior pixel and snaps the centre to the local-majority ARGB
  value when ≥ 6 of the 9 neighbours agree AND the snap stays within
  the per-channel quantisation step of the ORIGINAL pre-quantisation
  pixel. Catches "boundary jitter" — adjacent pixels that straddle a
  quantisation bin and would otherwise leave one-pixel runs in the
  LZ77 stream. Drift envelope is unchanged from the bare per-pixel
  pass: `≤ step` per channel, alpha bit-exact. (#380)
- *(vp8l-enc)* simple-Huffman tree emission for ≤ 2-active-symbol
  alphabets (spec §3.7.2.1.1). Single-symbol alphabets now emit a
  4-12 bit header and zero per-symbol bits (decoder's `only_symbol`
  short-circuit returns without consuming bits); 2-symbol alphabets
  emit a 12-13 bit header and 1 bit per symbol. Big win on palette
  index streams, single-colour-channel images, and the palette
  delta-encoded sub-stream. (#379)
- New default-on `registry` feature. With `default-features = false`
  the crate compiles without `oxideav-core` (and pulls `oxideav-vp8`
  in with its `registry` feature also off) and exposes a free-standing
  decode/encode API: `decode_webp(buf) -> Result<WebpImage, WebpError>`,
  `encode_vp8l_argb` / `encode_vp8l_argb_with`, `build_animated_webp`,
  `extract_metadata`. The standalone `decode_webp` walks the parsed
  RIFF container directly without going through the framework's
  `Demuxer` / `Decoder` traits. `WebpImage` / `WebpFrame` /
  `WebpFileMetadata` already used std-primitive fields; `WebpError`
  is a new local enum (`InvalidData` / `Unsupported` / `Eof` /
  `NeedMore`) plus `From<oxideav_vp8::Vp8Error>` so the VP8 path
  composes cleanly. The default-feature path keeps the existing
  `Decoder` / `Encoder` / `Demuxer` trait implementations + the
  `register` helpers + `WebpDecoder` streaming type — every current
  consumer (`oxideav` umbrella, `oxideav-pipeline`, mp4 + mkv WebP
  extraction) keeps working unchanged. (#358)

### Changed

- *(vp8l-enc)* RDO now mildly prefers the colour-indexing (palette)
  transform when the fixture is palette-feasible and the palette
  bitstream is within 16 bytes of the non-palette winner. Matches
  libwebp behaviour: palette is preferred whenever feasible because
  the index image is faster to decode (no per-pixel predictor /
  colour-transform application). Past 16 bytes the non-palette path
  is meaningfully better and RDO picks it. Closes #379.

### Fixed

- *(clippy)* doc list item overindented in `apply_alph_filter`
  (mode-3 description) and doc lazy-continuation on em-dash line in
  `vp8l/encoder` module preamble. (#379)
- *(clippy)* unnecessary `as usize` / `as u32` casts in
  `tests/animated_disposal_blend.rs`, `tests/metadata_roundtrip.rs`,
  `tests/vp8l_near_lossless.rs`.

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
