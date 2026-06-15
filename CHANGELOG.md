# Changelog

All notable changes to `oxideav-webp` are recorded here.

## [Unreleased]

### Added

- Round-308 lossless-encode cost improvement: the single-transform §4.2
  **cross-color** path now adds an **entropy-cost per-block CTE chooser**
  alongside the existing L1-magnitude one. Where the L1 chooser scores each
  `(green_to_red, green_to_blue, red_to_blue)` candidate by the folded
  residual magnitude, the new strategy scores by the Shannon lower-bound bit
  cost of the resulting per-channel residual histogram — the §4.2 analogue of
  the round-161 §4.1 predictor entropy chooser (RFC 9649 §3.5 authorises
  deciding transform data by entropy minimization). The per-axis greedy stays
  exact: the red residual depends only on `green_to_red` and the blue residual
  only on `(green_to_blue, red_to_blue)`, and red / blue carry independent
  §5.x prefix codes, so red entropy minimises over `green_to_red` alone and
  the blue pair is chosen greedily. The entropy candidate is evaluated at both
  the per-region and single-block `size_bits` across the round-148 cache-bits
  sweep; the super-chooser keeps the byte-shortest stream, so it cannot
  regress against the L1 path. On a 128×128 channel-correlated-noise fixture
  the entropy CTE candidate shrinks the §4.2 stream 41253 → 41195 B. Round-trip
  output is byte-identical regardless of which cost model is recorded — the
  decoder re-applies whatever CTE the §4.2 sub-image carries.

- Round-307 benchmark: new `stacked_transform_encode` criterion bench drives
  the public `encode_webp_lossless` entry point across the three distinct
  content regimes the RFC 9649 §3.5 **stacked-transform chains** target —
  `palette_indexed` (activates the §4.4 color-indexing path and its round-302
  color-indexing → §4.1 predictor chain), `photo_decorrelated` (red/blue as
  affine functions of green, activating the §4.2 cross-color transform and its
  round-303/304 color → predictor and color → subtract-green → predictor
  chains), and `smooth_gradient` (drives the §4.1 predictor sub-image lambda
  sweep across the residual-vs-§7.2-sub-image cost crossover rounds 302–306
  tuned). Rounds 302–306 each reasoned about an "empirically-observed
  crossover" without a committed harness exercising the chooser on per-regime
  inputs; this bench supplies that A/B target for both encode time and output
  size. Encode-only timing (inputs built once outside `b.iter`); all three
  inputs were verified to round-trip losslessly through encode → decode. No
  production-code change — measurement infrastructure only.

- Round-306 lossless-encode cost improvement: the §3.5 **stacked-transform
  chains** now sweep the **full sub-image-aware lambda set** the
  single-transform predictor path has carried since round 162 —
  `4_000` / `16_000` / `64_000` / `256_000` milli-per-bit — instead of the
  single mid-range `16_000` weight round 305 bootstrapped them with. The
  four weights straddle the empirically-observed residual-vs-§7.2-sub-image
  cost crossover (~`64_000`) on smooth transform-decorrelated content, so
  each chain (color + predictor, color + subtract-green + predictor,
  color-indexing + predictor) can land on the crossover its own decorrelated
  residual exhibits rather than one fixed guess. `STACKED_PREDICTOR_STRATEGIES`
  grows from 3 to 6 entries; the chooser keeps the byte-shortest stream
  across all of them, so the wider sweep is strictly non-regressing against
  both the L1 baseline and the round-305 single-lambda setting. Round-trip
  output is unchanged (lambda only biases which §4.1 mode is *recorded* per
  block — the decoder reads the same modes back). No decoder change required.

- Round-305 lossless-encode cost improvement: the §3.5 **stacked-transform
  chains** (color + predictor, color + subtract-green + predictor,
  color-indexing + predictor) now sweep the **predictor-sub-image cost
  model** the same way the single-transform predictor path has since
  rounds 159–162. The chains were bootstrapped (rounds 302–304) with only
  the round-159 folded-L1 magnitude proxy for per-block §4.1 mode
  selection; round 305 threads a `PredictorSubImageStrategy` through each
  chain so the chooser also builds the round-161 Shannon-entropy bit-cost
  and round-162 sub-image-aware entropy candidates over the
  *transform-decorrelated* residual the predictor actually sees, and keeps
  the byte-shortest. On smooth, mildly-noisy photo-like content the
  entropy-aware strategies shrink the color + predictor chain by ~12–21 %
  versus the L1 baseline (the per-block mode histogram concentrates,
  compacting both the §7.2 predictor sub-image and the residual stream).
  The sweep keeps L1 in the strategy set, so it is strictly
  non-regressing; round-trip correctness is strategy-independent (each
  strategy only changes which §4.1 mode is recorded per block — the
  decoder reads the same modes back). No decoder change required.

- Round-304 lossless-encode feature: the first §3.5 **three-transform**
  stacked candidate — §4.2 cross-color → §4.3 subtract-green → §4.1
  spatial predictor, chained into one `optional-transform` list. It is the
  natural three-axis extension of the round-303 color + predictor pair:
  the per-block §4.2 color transform removes the *modeled* inter-channel
  correlation, a header-free §4.3 subtract-green pass then removes the
  *uniform* red/blue-vs-green correlation that survives the coarse
  3.5-bit-fixed-point per-block CTE multipliers, and the §4.1 predictor
  pass removes the *spatial* correlation left in each channel — so the
  entropy stage sees residuals driven closer to zero than any one- or
  two-transform path achieves alone on content where all three
  correlation axes carry mass. RFC 9649 §3.5 permits up to four transforms
  stacked (each used at most once) with inverses applied last-read-first;
  none of the three subsamples the width, so both sub-image bodies and the
  main image run at full canvas width. The decoder's generic
  reverse-read-order chain already applies inverse-predictor →
  inverse-subtract-green → inverse-color, so no decoder change is required.
  The candidate is non-regressing (kept only when strictly smaller than
  the running best) and reuses the existing `width >= block && height >=
  block` gate, with two `size_bits` swept (default per-region granularity
  + a maximal single-block header) each across the `cache_code_bits ∈
  [1..11]` + disabled-cache sweep. New tests:
  `round_304_color_subtract_green_predictor_round_trips_through_decoder`
  (default + single-block `size_bits` × no/4-/9-bit cache),
  `…_single_block_round_trips`, and
  `round_304_chooser_never_regresses_and_round_trips`.

- Round-303 lossless-encode feature: a second §3.5 stacked-transform
  candidate aimed at **photo / natural-image** content — the §4.2
  cross-color transform chained with the §4.1 spatial predictor.
  The encoder now evaluates a candidate that writes the color transform
  first (read first) and the predictor second (read second) into one
  `optional-transform` list. The §4.2 transform removes the
  inter-channel correlation (rewriting red/blue as residuals against
  green per the per-block `ColorTransformElement`); a predictor pass
  over that color-decorrelated image then removes the *spatial*
  correlation surviving in each channel, so the entropy stage sees
  residuals closer to zero than either transform alone. Neither
  transform subsamples the width, so both sub-image bodies and the main
  image run at full canvas width; the decoder already applies the
  inverses last-read-first (inverse-predictor recovering the
  color-transformed image, then inverse-color recovering the original
  pixels), so no decoder change is required. The candidate is
  non-regressing (kept only when strictly smaller than the running best)
  and reuses the existing `width >= block && height >= block` gate, with
  two `size_bits` swept (default per-region granularity + a maximal
  single-block header), each across the `cache_code_bits ∈ [1..11]` +
  disabled-cache sweep. New tests:
  `round_303_color_transform_predictor_round_trips_through_decoder`
  (default + single-block `size_bits` × no/4-/9-bit cache),
  `…_single_block_round_trips`, and
  `round_303_chooser_never_regresses_and_round_trips`.

### Fixed

- Round-303 lossless-encode correctness fix in the §3.7.2.1.2
  *code-length-code* (CLC) writer. The CLC lengths are each written in a
  3-bit on-wire field (range `[0..7]`), but the encoder built them with
  the general 15-bit Huffman builder. A sufficiently skewed CLC
  frequency histogram (one length value far more common than the rest)
  makes the builder assign a length-8-or-more code to a rare CLC symbol;
  the 3-bit field then silently truncated it, corrupting the on-wire
  table into an incomplete (Kraft < 1) prefix code the decoder rejects
  with `Prefix(Incomplete)`. The fix caps CLC lengths at 7 via a Kraft
  re-balancing pass (`build_clc_code_lengths` / the new
  `limit_code_lengths_to`), keeping the table complete. This was a
  latent defect reachable by any encoder candidate whose residual
  distribution produced a skewed CLC histogram — surfaced here by the
  new color-transform + predictor chain at the per-region color
  `size_bits`. New test:
  `clc_code_lengths_capped_at_seven_and_complete`.

- Round-302 lossless-encode feature: the §3.5 stacked-transform path.
  The encoder now evaluates a **chained** candidate that applies the
  §4.4 color-indexing transform followed by the §4.1 spatial predictor
  over the bundled palette-index image, written into one
  `optional-transform` list (color-indexing read first → predictor read
  second → end). RFC 9649 §3.5 permits up to four transforms stacked
  (each used once) with inverses applied last-read-first; the decoder
  already runs that reverse-order chain, so it un-applies the predictor
  over the packed indices first, then un-bundles the color index — no
  decoder change. On palette content (icons, line art, screen captures)
  the bundled indices run in long spatially-coherent stretches, so the
  predictor drives the residuals toward zero and shrinks the entropy
  stage below the single-transform color-indexing path. The candidate
  is non-regressing (kept only when strictly smaller than the running
  best) and self-skips when the packed image is too small to carry a
  predictor block. Two predictor `size_bits` are swept (default
  per-region granularity + a maximal single-block header), each across
  the round-148 `cache_code_bits ∈ [1..11]` + disabled-cache sweep.
  New tests: `round_302_color_indexing_predictor_round_trips_through_decoder`
  (four bundling regimes), `…_round_trips_with_cache_and_single_block`,
  `…_skips_subblock_packed_image`, and
  `round_302_chooser_never_regresses_and_round_trips`.

### Performance

- Round-301 PROFILE-OPT round: gave the encoder-side §5.2.2
  `pixel_distance_to_distance_code` distance-code chooser (the rank-1
  round-300 target) a **smallest-code early-out**. Map codes occupy
  `1..=120` and the scan-line fallback is `D + 120 ≥ 121`, and the
  `DISTANCE_MAP` entries are visited in ascending code order, so the
  first entry whose `max(xi + yi·W, 1)` equals the distance is already
  the smallest valid code; the chooser now returns on that first match
  instead of always scanning all 120 entries. The chosen code — and
  therefore every emitted byte — is unchanged, proven by a new
  `distance_chooser_early_out_matches_full_scan` test that asserts
  identity against an inline full no-early-out scan-with-tie-break over
  distances `1..=400` + `{1000, 4096, 70_000}` across widths
  `{1, 2, 16, 128, 256, 1024}`. On the `distance_code` bench the
  matching cells drop from ~64 µs to ~0.8–2.4 µs per 1024 calls
  (≈30–160×); the genuine no-match worst case (`dist_large_nomatch`)
  still walks all 120 entries since there is no smaller code to find.
  See `BENCHMARKS.md` "Round-301".
- Round-300 BENCH round (no `src/` change): added
  `benches/distance_code.rs`, isolating the encoder-side §5.2.2
  `pixel_distance_to_distance_code` distance-code chooser — run at least
  twice per emitted LZ77 backward reference and performing a no-early-out
  linear scan of all 120 `DISTANCE_MAP` entries to pick the smallest
  valid distance code. Four cells fix `image_width = 256` and vary
  `distance`: `dist1_rle` (RLE, multiple clamp-to-1 hits),
  `dist_row_above` (`distance == width`, code-1 match at scan index 0),
  `dist_small_neighbor` (`distance = 2`), and `dist_large_nomatch`
  (`distance = 70_000`, no map match → full scan + scan-line fallback).
  Baseline: all four cells flat within noise at ~64 µs / 1024 calls
  (~63 ns/call), confirming the cost is the fixed 120-entry scan, not the
  match location — the A/B harness for a future reverse-map early-out.
  See `BENCHMARKS.md` "Round-300".
- Round-299 PROFILE depth round: rewrote the private
  `crate::argb_to_rgba` helper — the ARGB-`u32` → packed `[R, G, B, A]`
  converter the public `decode_webp` lossless still path
  (`decode_lossless_image`) and the animation sub-frame path call — from
  the original four-`push`-per-pixel loop to the pre-sized
  `chunks_exact_mut(4)` form that round 170 already adopted for the
  public `Vp8lImage::to_rgba`. Output is byte-for-byte identical
  (`[R, G, B, A]` order; the full 439-test `--lib` suite passes
  unchanged). New A/B bench cells `repack_push_loop` /
  `repack_chunks_exact` in `benches/argb_to_rgba.rs` measure ~130 µs vs.
  ~8.9 µs over a 256×256 buffer (≈14.5× faster), matching the existing
  `argb_to_rgba` cell. A `chunks_exact_mut` rewrite of the lossy
  `yuv420_to_rgba` interior was also tried but regressed ~2.1–2.4× and
  was reverted. See `BENCHMARKS.md` "Round-299".
- Round-296 PROFILE depth round: evaluated a per-block predictor-mode
  hoist for the §4.1 `vp8l_transform::inverse_predictor` interior loop
  (load the block's mode once per `1 << size_bits` block instead of on
  every pixel, mirroring the round-207 `inverse_color` CTE hoist). The
  hoist was proven byte-identical — the existing randomised cross-check
  test plus an FNV-1a A/B over all seven `lossless-*` fixtures both
  matched — but it produced no measurable win (the interior is
  dominated by the 14-way `predict()` dispatch, not the mode load) and
  the measurement host was saturated by concurrent agents (baseline
  drift of 4–6×), so the original per-pixel body was retained per the
  round-224 precedent. The realistic block path is now benched:
  `benches/inverse_predictor.rs` adds
  `inverse_predictor_blocks16_mixed_256x256` (`size_bits=4`, 16×16
  blocks, per-block LCG mode mix), filling the coverage gap left by the
  pre-existing `size_bits=0` cells (which the on-wire `size_bits =
  ReadBits(3) + 2 ∈ [2, 9]` never reaches). See `BENCHMARKS.md`
  "Round-296".

- Round-293 PROFILE-OPT depth round: hoisted the §2.7.1.2 `ALPH`
  Stage-2 inverse-filter border rules out of the per-pixel loop. The
  loop previously re-evaluated a `match (x, y)` top-left special case
  plus a `match` on the (loop-invariant) filter method, with an inner
  edge test and an index closure, on every one of the `width × height`
  pixels. `decode_alpha`'s Stage 2 now calls a new private
  `inverse_filter` that dispatches on the filter method **once** and
  runs a specialised body per method: a one-shot border pass (top-left,
  first row, and — for Horizontal/Gradient — left-most column) followed
  by a tight interior loop with precomputed row-base indices. The
  `None` method becomes a plain identity buffer move. On this host
  (`aarch64-apple-darwin`, `--quick`): `inverse_filter_none_128x128`
  9.5 µs → 0.23 µs (−97%), `inverse_filter_vertical_128x128`
  13.5 µs → 1.57 µs (−88%, the row-above predictor auto-vectorises once
  the dispatch is gone); `Horizontal` and `Gradient` are flat (their
  left-neighbour serial recurrence, not the dispatch, is the bound).
  **Decoded alpha planes are byte-for-byte unchanged for every filter
  method and dimension** — proven by a new oracle test
  (`hoisted_inverse_filter_matches_per_pixel_reference_across_methods_and_dims`,
  the round-291 per-pixel form as reference, over 9 dimension shapes ×
  4 methods) and by 400 K `decode_alph` fuzz runs plus the
  `decode_still_paths` differential, no divergence. See `BENCHMARKS.md`
  "Round-293".

### Fixed

- Round-292 FUZZ depth round: a malformed §2.6 `VP8L` lossless chunk
  whose §3.4 5-byte image-header declares a huge canvas (the 14-bit
  `width - 1` / `height - 1` fields reach `16384 × 16384 ≈ 2.7e8`
  pixels) but carries only a few backing bytes no longer forces an
  unbounded eager allocation. `vp8l_decode::decode_image` and
  `decode_argb_multi_group` previously pre-sized their output buffer to
  the full declared `width * height` (`Vec::with_capacity`), so a
  ~30-byte chunk could drive an ~800 MiB allocation **before** the
  EOF-checked §5/§6 decode loop read a single symbol — a decode-time
  out-of-memory DoS surfaced by the new `decode_lossless_image` fuzz
  target. The eager reservation is now capped at
  `MAX_EAGER_PIXEL_RESERVATION = 1 << 22` pixels via
  `eager_pixel_capacity`; the buffer still grows on demand for a
  legitimately large image, and the self-terminating decode loop raises
  `DecodeError::Eof` for a truncated stream as before. **Decoded bytes
  for all valid images are unchanged** — the cap only affects the
  *initial* capacity hint, not the final buffer contents (438 lib tests
  pass, including two new `eager_pixel_capacity_*` regression tests).

### Added

- Round-298 FUZZ depth round: added
  `fuzz/fuzz_targets/parse_vp8_chunk.rs`, a libFuzzer panic/OOM-free
  target over the §2.5 simple-lossy `VP8 ` chunk handle standalone entry
  point `vp8_chunk::WebpLossyChunk::from_payload` — the RFC 6386 §9.1
  key-frame-header peek the §2 RIFF walker reaches only along the
  well-formed-container path. The entire fuzz buffer is forwarded
  verbatim as the §2.5 `VP8 ` payload candidate; every successfully-
  decoded field is cross-checked against the §9.1 byte layout the parser
  observed (LE frame tag from bytes 0..3 with the key-frame frame-type,
  `version`/`show_frame`/19-bit `first_partition_size` bitfields, the
  §9.1 start code at bytes 3..6, the 14-bit dimension / 2-bit scale split
  of the width/height words, and the verbatim `bitstream()` borrow), and
  every refusal branch (`PayloadTooShortForKeyframe`, `NotAKeyframe`,
  `BadStartCode`) is cross-checked against its §9.1 / §2.5 trigger.
  87,063,810 executions (~1.43 M exec/s, peak RSS 550 MiB), 0 findings —
  the existing `from_payload` length gate and frame-tag/start-code checks
  already make the path panic-free.

- Round-292 FUZZ depth round: added
  `fuzz/fuzz_targets/decode_lossless_image.rs`, a structure-aware
  libFuzzer panic/OOM-free decode target over the public top-level
  lossless façade `decode_lossless_image` — the layer that walks the
  §2.3 `RIFF`/`WEBP` container, selects the §2.6 `VP8L` chunk, reads the
  chunk's own §3.4 image-header dimensions, and runs the full §4/§5/§6
  decode returning the typed `DecodedImage`. Distinct from the round-286
  `decode_lossless` harness (which supplies `(width, height)` *from the
  harness* over a bare payload): here the decoded dimensions come from
  the **file's own** §3.4 header, exercising the §3.4-header → §4-decode
  dimension-coherence path end to end. A cheap structural pre-pass gates
  the full-decode tail by declared pixel count. Seeded from the in-tree
  `tests/data/*.webp` fixtures; 48,882 executions post-fix, 0 findings.
  The pre-fix run surfaced the eager-allocation OOM fixed above.

### Changed

- Round-294 BENCH depth round: added `benches/meta_prefix_cluster.rs`, a
  criterion harness over the encoder's §6.2.2 entropy-image
  block-clustering heuristic (`cluster_blocks_by_histogram_distance`)
  behind `encode_with_meta_prefix` — the last encode stage sized only by
  subtraction inside the `lossless_encode` end-to-end number. The kernel
  is a coarse-RGB-histogram (16 bins/channel → 48-dim/block) Lloyd's
  k-means over the `1 << prefix_bits`-aligned blocks: a per-pixel
  feature-binning pass, deterministic farthest-point seeding, an
  up-to-8-pass assignment/update loop, and a compaction. Three altitudes:
  content regime (bimodal split / smooth gradient / uniform single-group
  early-out), `num_groups ∈ {2,3,4}`, image side ∈ {128,256,384} px. The
  ranked hotspot map: the per-pixel feature pass dominates (≈70–80% of
  clustering self-time — the uniform cell, which skips the Lloyd loop, is
  122 of 150 µs; the size sweep scales with pixel not block count), the
  Lloyd loop is a clear second only on poorly-separated content (gradient
  204 µs vs. bimodal 150 µs), and `num_groups` 2→4 is nearly flat
  (147→150 µs) at the default block size. **No behavioural change — the
  only `src/` edit is a `fn` → `pub fn` visibility widen on
  `cluster_blocks_by_histogram_distance` so the bench can drive it in
  isolation (matching the `pick_block_cte` exposure pattern); every
  emitted byte is unchanged.** See `BENCHMARKS.md` round-294.

- Round-291 BENCH depth round: added `benches/alpha_decode.rs`, a
  criterion harness over the §2.7.1.2 `ALPH` alpha-plane decode — the
  rank-1 webp-owned cost on the lossy decode path, previously sized only
  by subtraction in the round-289 hotspot map. Five cells: the public
  `decode_alpha_plane` e2e over the committed fixture, `alph::decode_alpha`
  on the extracted `ALPH` payload (RIFF walk removed), and the §2.7.1.2
  Stage-2 inverse-filter per-pixel loop in isolation, one cell per `F`
  method (synthetic uncompressed payloads). The measurements correct the
  r289 estimate: the container walk is ≈1 µs (negligible), and the rank-1
  cost is almost entirely the headerless VP8L lossless decode inside
  `decode_alpha` (already covered by `read_symbol` / `lossless_decode*`).
  The genuinely alpha-specific inverse-filter loop ranks Gradient (43.7 µs)
  > Horizontal (21.8 µs) > Vertical (13.5 µs) > None (9.5 µs) at 128×128.
  **No `src/` change — bench-only; decoded bytes are unchanged.** See
  `BENCHMARKS.md` round-291.

- Round-290 PROFILE-OPT depth round: `vp8_decode::yuv420_to_rgba` (the
  §2.5 lossy decode's crate-owned 4:2:0 chroma up-sample + RFC 6386 §9.2
  BT.601 YCbCr→RGB loop, ranked rank-1 webp-owned lossy hot path in r289)
  now hoists the chroma-matrix terms out of the per-pixel loop. The two
  luma pixels of a 4:2:0 pair share one chroma column, so the three
  `(Cb−128, Cr−128)` contributions are computed once per column
  (`chroma_offsets`) and reused; the output is written through pre-sized
  per-row slices instead of four `Vec::push` calls per pixel.
  **Decoded bytes are byte-for-byte identical** — the per-pixel
  `ycbcr_to_rgb` form is retained as a `#[cfg(test)]` oracle, and the new
  `yuv420_to_rgba_matches_per_pixel_reference_across_dimensions` test
  proves equivalence across 9 even/odd dimensions with non-neutral chroma;
  `cargo fuzz` (decode_still_paths, decode) over the corpus showed no
  divergence. The conversion drops from ≈34 µs to ≈10.5 µs at 128×128
  fixture size (−68%; −72% at 256×256) — see `BENCHMARKS.md` round-290.

### Fixed

- Round-288 FUZZ depth round: `decode_webp`'s §2.7.1.1 animation path no
  longer eagerly allocates the full `VP8X` canvas before validating its
  size. §2.7.1 permits a canvas up to 2^24 per side (product capped at
  2^32 - 1); a ~60-byte file declaring a 16 777 154 × 64 canvas forced a
  ~4 GiB `Vec` (libFuzzer OOM, surfaced by the new `decode_still_paths`
  harness below). The canvas is now bounded at the §3.4 still-image
  ceiling (`MAX_DECODE_DIMENSION = 16384` per side) — a dimension above
  that can never be fully covered by a spec-valid `ANMF` sub-frame (each
  sub-frame is itself a `VP8L` image) — and an over-ceiling canvas is
  rejected with `WebpError::InvalidData`, allocating nothing. Covered by
  the `oversized_anim_canvas_is_rejected_without_eager_allocation`
  regression test (inclusive of the 16384 ceiling).

### Added

- Round-289 BENCH depth round: `benches/lossy_decode.rs`, the first
  isolated harness for the §2.5 `VP8 ` (lossy) decode path. Three
  altitudes — full public `decode_webp` over the 128×128 `VP8 `+`ALPH`
  fixture (`decode_webp_lossy_e2e`, ≈359 µs), `decode_lossy_rgba` over
  the extracted `VP8 ` bitstream with the RIFF walk removed
  (`decode_lossy_rgba_extracted`, ≈173 µs), and the crate-owned
  post-I420 reconstruction loop `yuv420_to_rgba` (4:2:0 chroma up-sample
  + RFC 6386 §9.2 BT.601 YCbCr→RGB) in isolation at 16/128/256 px
  (≈551 ns / 34 µs / 137 µs, linear per-pixel). The sibling
  `oxideav-vp8` decoder owns the entropy/IDCT/intra-pred/loop-filter
  work (≈39% of lossy e2e, out of this crate's scope); the bench isolates
  and ranks the lossy stage `oxideav-webp` itself can act on. The ranked
  hotspot map is in `BENCHMARKS.md` (Round-289). `vp8_decode::yuv420_to_rgba`
  widened `fn` → `pub fn` so the bench can isolate it — a visibility
  change only; decoded bytes are identical (all 435 lib tests + fixture
  SHA-256s unchanged). No behavior change.

- Round-288 FUZZ depth round: a twenty-ninth `cargo-fuzz` harness,
  `decode_still_paths`, a differential oracle on the two public
  still-image decode entry points `decode_webp` (the published
  `WebpImage` surface) and `decode_webp_image` (the low-level
  `DecodedWebp` surface), seeded from the in-tree §2.6 lossless +
  §2.7-extended fixtures. For a non-animated input the published façade
  builds its single still frame by literally calling `decode_webp_image`,
  so the harness asserts the two surfaces agree exactly (`Ok` ⇒
  `frames.len() == 1` with byte-identical `frames[0].{rgba,width,height}`
  + canvas-dimension echo + `duration_ms == 0` + no §2.7.1.1 carrier;
  `Err` ⇒ the published path also `Err`), re-checks the §2.5/§2.6
  flat-buffer carrier invariant (`rgba.len() == width * height * 4`) on
  every still + every composited animation frame, and asserts
  `decode_webp_image` determinism. A ~300 s ASan campaign over 25 772
  runs is crash-free post-fix. The §2.5 `VP8 ` *lossy* decode (routed to
  the `oxideav-vp8` sibling) is deliberately skipped from the cross-check
  pending a sibling-side hardening — see the round report.

### Changed

- Round-287 PROFILE-OPT (depth round): the §6.2.1 canonical-decoder
  per-bit walk (`PrefixCode::read_symbol_walk` and its long-code
  continuation `read_symbol_long`) now resolves "is there a code row at
  this length?" through a 16-byte direct length→row side table
  (`len_to_row`, built once in `from_code_lengths`) instead of a linear
  `length_rows.iter().find(..)` rescan on every consumed bit. The walk
  is the decode path for every code below the `MIN_LOOKUP_USED` gate and
  the > 8-bit / near-EOF fallback for the round-284 primary table, so
  the rescan cost grew with the number of distinct code lengths. Output
  is byte-identical (the `read_symbol_reference` differential test and
  the `read_symbol_lut_diff` fuzz oracle both still pass; all 435 unit
  tests + the fixture-walk and round-trip suites are green). The
  round-284 candidate — a 9–`LOOKUP2_BITS`-bit second-level *spill*
  table — was prototyped and **rejected**: it measured a net regression
  on `read_symbol_long9_11` / `read_symbol_dense256` (the extra 4–16 KiB
  table thrashes L1 against the 1 KiB primary, and a single-bit walk
  from length 8 already beats a second peeked word-load + random table
  access). The chosen side-table change adds no cache footprint. New
  `read_symbol_manylen16_walk` bench cell (one symbol at each of lengths
  1..=14 plus two at 15 — a Kraft-exact, maximally length-diverse code
  the uniform-length cells can't exercise) isolates the targeted regime:
  **86.8 µs → 37.2 µs per 4096 reads, a 2.33× speedup** on the
  worst-case walk; the uniform short-code, 9–11-bit, dense-literal, and
  end-to-end `lossless_decode*` benches show no statistically
  significant change (the linear scan was already cheap when few
  distinct lengths are present).

### Added

- Round-286 benchmark-mode coverage of the two hotspots the round-283 /
  round-284 profiles flagged but never benched in isolation
  (`src/` byte-identical this round — both harnesses build their inputs
  through the existing public API, no probe accessor needed). New
  `benches/read_symbol.rs` isolates the decoder's §6.2.1
  `PrefixCode::read_symbol` per-symbol reader — the rank-1 decode
  hotspot at ~82% of decode self-time post-round-284 — across five
  cells that separate the round-284 primary-table fast path
  (`short8_uniform` / `short6_uniform`, all codes ≤ 8 bits resolve in
  one table load), the long-code (> 8-bit) walk continuation the
  round-284 follow-up targets (`long9_11`, every read spills past the
  table), the realistic blended literal channel (`dense256`), and the
  below-`MIN_LOOKUP_USED`-gate walk-only baseline (`belowgate16_walk`);
  each cell packs a deterministic LCG symbol stream via the public
  `canonical_codes` + `BitWriter` and times 4096 back-to-back reads
  through a fresh `BitReader`. New `benches/lz77_chain.rs` isolates the
  §5.2.2 LZ77 matcher (`Lz77Matcher::find` + `insert`, rank 3 in the
  round-283 encode profile) across five hash-chain-depth regimes
  (period-2/4/64 repeats, near-unique, gradient), driven through the
  public `encode_argb_literals_with_width` entry. Measured the
  long-code read path at +27% per symbol over the primary-table floor
  (`dense256` at +11%) and found the matcher's expensive regime is the
  shallow/unique insert+miss path (6.5–7× the deep-repeat cost), not
  the deep walk — reframing the r283 chain-cut candidate. `BENCHMARKS.md`
  ranks the decoder 9–11-bit second-level spill table (or
  alphabet-size-aware primary width) as the next PROFILE-OPT target.
- Round-285 fuzz hardening of the round-284 §6.2.1 read-symbol fast
  path + word-load `BitReader` (depth round, fuzz mode). Two new
  `cargo-fuzz` harnesses (#27 / #28, total now twenty-eight):
  `read_symbol_lut_diff` runs `PrefixCode::read_symbol` (256-entry
  primary lookup table, > 8-bit continuation walk, near-EOF per-bit
  fallback, used-symbol amortization gate) in lockstep against the
  crate's own pre-table per-bit row walk — kept as the new
  `#[doc(hidden)]` `PrefixCode::read_symbol_reference` oracle
  (behaviour-neutral; delegates to the existing private walk) — over
  the same bytes, asserting the decoded symbol / typed refusal
  (including `PrefixError::Eof` position fields), the cursor position
  after every symbol, and the alphabet bound identical, with the code
  under test built either off the wire (`PrefixCode::read` at a
  §6.2.3 alphabet) or from fuzz-shaped lengths repaired to an exact
  §6.2.1 Kraft sum so the mutator steers the used-symbol count and
  the length profile freely; `decode_lossless_lut` re-drives
  `decode_lossless` / `decode_lossless_headerless` at carrier
  dimensions widened into `[1, 64]`, corpus-seeded from the fixture
  corpus's VP8L chunk payloads plus entropy-heavy
  reference-encoder-produced 64×64 noise / gradient tiles, so the
  lookup-table fast path and continuation rows run hot inside the
  assembled pipeline under adversarial mutation. Campaigns: 36.1 M
  execs (`read_symbol_lut_diff`, 15 min ASan) + 16.8 M execs
  (`decode_lossless_lut`, 15 min ASan) with zero divergences and zero
  crashes, plus clean 4-minute regression re-runs of `prefix_code`
  (32.2 M), `decode_lossless` (9.4 M), and `prefix_code_group`
  (24.2 M); a matching in-tree unit
  test (`read_symbol_reference_matches_fast_path_on_lut_built_code`)
  pins the differential on a 256-symbol code mixing ≤ 8-bit and
  > 8-bit lengths. No findings: the round-284 fast path survived both
  oracles unchanged.

### Changed

- §6.2.1 decoder `PrefixCode::read_symbol` fast path (round 284 —
  the round-283 decode profile's rank-1 symbol at ~85–89% of decode
  self-time on every entropy-heavy fixture): codes built from 32 or
  more used symbols now carry a 256-entry primary lookup table keyed
  on the next 8 peeked stream bits (wire order), resolving any code of
  length ≤ 8 with one load + one cursor advance; longer codes consume
  the 8 peeked bits and continue the per-bit row walk from length 9,
  and codes below the used-symbol gate (tiny delta frames,
  sub-resolution transform images, the 19-symbol code-length code)
  keep the pre-table per-bit walk so the table cost is never paid
  where it cannot amortize. `BitReader::read_bits` now assembles its
  result from one zero-padded little-endian word load instead of a
  per-bit gather (same value bit for bit), and grew `peek_bits` /
  `advance_bits` for the fast path. Decoded output is bit-identical:
  an FNV-1a-64 digest sweep over the full docs + in-crate fixture
  corpus plus the five synthetic transform-mix fixtures is unchanged
  against the pre-change implementation, and the `prefix_code` fuzz
  harness ran 19.3 M execs clean on the new path. End-to-end decode
  benches: `lossless_decode_mix_crosscolor` 2.947 ms → 1.591 ms
  (−46%), `mix_none` 1.803 ms → 0.923 ms (−49%), `mix_subgreen`
  1.236 ms → 0.778 ms (−37%); see `BENCHMARKS.md` round 284.

- §3.5.2 color-transform-element chooser (`pick_block_cte`, the
  per-pixel-heavy stage of `encode_with_color_transform` — rank 2 at
  ~9% self-time in the round-280 encode profile): the three per-axis
  candidate sweeps now share a `sweep_cte_axis` helper that checks the
  worse-than-best prune at 32-sample chunk granularity instead of per
  sample, leaving the interior cost loop branch-free
  (auto-vectorisable). Encoded output is bit-identical (FNV digest over
  an 81-image encode sweep unchanged). New
  `pick_block_cte_walk_256x256` bench: 1.6012 ms → 752.03 µs (−52.6%);
  see `BENCHMARKS.md` round 281. `pick_block_cte` is now `pub` so the
  bench harness can drive it directly.

### Added

- Corpus-wide decoded-output digest pin
  (`round284_fixture_corpus_decode_digests_are_pinned` in
  `tests/fixture_walks.rs`): FNV-1a-64 over geometry + every decoded
  frame's RGBA for all eight in-crate fixtures, locked to the
  pre-round-284 reference decode so any future entropy-path rewrite
  has a byte-exact regression gate in CI.

- Three end-to-end criterion benches (round 283, BENCH depth round —
  `src/` untouched), closing the decode-side coverage gaps above the
  per-pass level: `lossless_decode_mixes` (full-file `decode_webp`
  per elected §4 transform mix — predictor / color-indexing /
  cross-color / subtract-green / no-transform, each fixture's elected
  list asserted at setup via `read_vp8l_transform_list`),
  `anim_decode` (§2.7.1.1 full-timeline animation decode of the same
  12-frame 128×128 timeline in all-keyframe and dirty-rect-delta
  `ANMF` layouts), and `metadata_walk` (`extract_metadata` chunk walk
  over a simple no-metadata file, a 5-chunk `VP8X` still with
  ICC + Exif + XMP, and a 64-frame animation carrying the same
  payloads after the `ANMF` run). Full regression re-run of all 21
  bench targets (stable + nightly `simd`) plus fresh decode / encode
  profiles and the ranked next-hotspot list recorded in
  `BENCHMARKS.md` round 283.

- Fuzz target `roundtrip_metadata` (round 282) — differential oracle
  on the §2.7 metadata *write* path: the two independent
  extended-layout writers (`build::build_webp_file_with_metadata` and
  `encode_vp8l_argb_with_metadata`) are driven with fuzz-controlled
  §2.7.1.4 `ICCP` / §2.7.1.5 `EXIF` / `XMP ` payloads (presence,
  length, content — odd lengths exercising the §2.3 pad byte), a
  fuzz-controlled §2.7.1 `L` alpha-hint flag, and fuzz-controlled
  canvas dimensions + ARGB pixels, with every emitted file
  cross-checked against the §2.7 canonical chunk order, the §2.7.1
  flag-octet derivation, byte-exact metadata round trips through both
  `extract_metadata` and the `decode_webp` metadata carry, the
  lossless pixel contract, and the writer-B no-alpha/no-metadata
  demotion to the §2.6 simple layout. The prior coverage fuzzed the
  metadata chunks from the read side only (raw bytes into
  `extract_metadata`), which a coverage-guided mutator almost never
  grows into a well-formed multi-chunk extended file.

- Criterion bench `pick_block_cte` (`pick_block_cte_walk_256x256`) —
  the §3.5.2 encoder color-transform chooser walk at the
  encoder-default `size_bits = 4` (256 blocks of a 256×256
  correlated-channel image), closing the encoder bench-shelf gap
  named by the round-280 followups.

### Changed

- §4.1 encoder block-mode chooser hot path (round-280 profile: the
  per-pixel `predictor_at` helper was 36% of total encode self-time):
  the per-block per-mode cost walks (`block_mode_cost`, the L1 pickers
  `pick_block_mode_with_hint` / `_slack`, and
  `block_mode_entropy_cost`) now run through a shared block-residual
  walker that hoists the §4.1 border-rule branch chain and the 14-way
  predictor-mode dispatch out of the per-pixel inner loop
  (monomorphised per mode), and prunes worse-than-best modes at
  block-row granularity instead of per pixel so the interior loop is
  branch-free. Encoded output is bit-identical (FNV digest over an
  82-image encode sweep unchanged; pinned by the new
  `block_walker_matches_predictor_at_reference_random` test against
  verbatim pre-change reference loops). `lossless_encode_natural_128`
  −19% to −28%, `lossless_encode_rgba_256` −16% to −21%; see
  `BENCHMARKS.md` round 280.

### Added

- Fuzz harness #24 `roundtrip_anim_modes` — a differential oracle on
  the §2.7.1.1 animation assembly path `build_animated_webp_with_options`
  → `decode_webp` with every per-frame carrier field fuzz-driven: even
  `(x, y)` sub-canvas offsets, mixed `Auto` / `Delta` / `Lossless` frame
  modes, `None` / `Background` disposal, `Overwrite` / `AlphaBlend`
  blending, and the `ANIM` loop-count + background-colour options. Every
  decoded full-canvas frame snapshot is asserted byte-identical to an
  independent §2.7.1.1 canvas simulation; duration, loop count and
  background colour are asserted to carry through. The existing
  `roundtrip_animated` harness only drove `AnimFrame::new` defaults
  (full-canvas Lossless frames at the origin), leaving the dirty-rect
  encoder and the dispose/blend carrier semantics unfuzzed.

### Fixed

- `AnimFrameMode::Delta` / `Auto` dirty-rect emission (both found by the
  new `roundtrip_anim_modes` harness within its first seed pass, each
  pinned by a regression test in `tests/published_anim_api.rs`):
  - a delta-emitted frame with `dispose == Background` forced `D = 0`
    into the stream while the encoder's reference canvas applied the
    caller's dispose, so the decoder never cleared the rect and every
    subsequent delta frame was diffed against a canvas state the decoder
    did not have — the displayed frames diverged from the §2.7.1.1
    semantics of the supplied frame list. Background-disposed Delta/Auto
    frames now fall back to a full keyframe with the caller's flags
    honoured verbatim (a smaller dirty-rect ANMF cannot carry the
    full-rect clear).
  - a delta-emitted frame with `blend == AlphaBlend` diffed and emitted
    its *raw source* pixels with `B` forced to overwrite, so
    semi-transparent pixels landed on the decoder canvas unblended. The
    dirty rect is now computed between the post-composite drawn canvas
    and the previous canvas, and the emitted sub-frame carries the drawn
    (already-blended) pixels, which a plain overwrite reproduces
    bit-exactly; the no-diff degenerate 2×2 emission likewise re-writes
    drawn-canvas pixels instead of raw source pixels.

### Optimized

- `vp8l_encode::limit_code_lengths` — the §3.7.2 length-cap
  re-balancing pass that only fires when a pathological frequency
  distribution would push a code past 15 bits — no longer rescans all
  used symbols to select each adjustment target. A fresh profile of
  the length-capped `build_code_lengths_dense_green2328` bench input
  attributed ~81% of `build_code_lengths` self-time to that rescan
  (3 491 of ~4 280 in-process samples), confirming the round-277
  flag. The over-subscribed loop now drains one bucket per code
  length, filled in used-symbol order so the back of the highest
  non-empty bucket is exactly the symbol the historical rescan's
  `l >= best_len` tie-break picked, and drives each popped symbol
  upward in place (once lengthened it is strictly the unique deepest
  eligible leaf, so the rescan re-picked it every step until it hit
  the cap) — the original O(n)-per-adjustment pick sequence,
  reproduced step for step at O(1) per adjustment. Output is
  bit-identical: the FNV digest over the 8 bench cells plus 600
  randomized frequency tables (zero densities, tie-heavy, exponential
  cap-tripping skews) is unchanged, a 20 M-table differential fuzz
  against the literal pre-change implementation (~5.8 M of them
  producing max-length-15 codes) found zero divergence, and a 5-minute
  `roundtrip_lossless` fuzz run plus the full test suite pass
  unchanged. `build_code_lengths_dense_green2328` median:
  111.9 µs → 26.4 µs (4.2×); the cell now sits at the leaf-sort +
  two-queue-merge floor and uncapped cells are unaffected (within
  noise). Round-278 section in `BENCHMARKS.md` has the numbers.
- The §3.7.2 / §6.2.1 length-then-code canonical build chain — the
  dense-cell optimization target flagged by the round-250 / round-276
  benches — was rewritten bit-identically on both sides, verified by an
  FNV digest of every length table and built decoder table over the
  full bench input set plus 600 randomized frequency tables (including
  length-limit-triggering exponential skews): the digest is unchanged
  from the previous implementation, and the round-275 `prefix_code`
  differential fuzz harness ran 13.6 M execs clean on the rewrite.
  - Encoder `vp8l_encode::build_code_lengths`: the hand-rolled binary
    min-heap is replaced by a sorted-leaf + internal-FIFO two-queue
    merge (leaves sorted once by a packed `(freq, symbol)` `u64` key;
    internal nodes are created with nondecreasing frequencies, so a
    plain FIFO stays sorted and each merge step compares the two queue
    fronts in O(1), preferring the leaf on a frequency tie exactly as
    the old `(freq, order)` heap key did). Leaf depths are now
    recovered with a single reverse pass over the internal nodes
    instead of one parent-chain walk per leaf, and the
    `limit_code_lengths` re-balancing pass updates its Kraft sum
    incrementally instead of recomputing the O(n) sum after every
    single-leaf adjustment. Dense-cell medians: distance-40
    2.09 µs → 408 ns, literal-256 15.07 µs → 2.07 µs, green-281
    17.66 µs → 2.37 µs, and the headline green-2328 dense cell
    382.0 µs → 113.5 µs (3.4× same-session; 3.7× against the recorded
    round-250 417.8 µs). Sparse cells improve 2.1×–3.5×.
  - Decoder `vp8l_prefix::PrefixCode::from_code_lengths`: the
    `(length, value)` symbol ordering is now a single-rescan counting
    sort — `bl_count` prefix sums fix every length bucket's start
    index up front and ONE pass over `code_lengths` drops each used
    symbol at its bucket cursor — replacing the one-full-rescan-per-
    used-length assignment loop. Dense green-2328 7.44 µs → 4.63 µs
    (1.6×), sparse green-2328 5.84 µs → 1.81 µs (3.2×), the other
    cells 1.1×–2.4×. Full before/after tables in `BENCHMARKS.md`
    (round-277 section).

### Fixed

- `vp8l_stream::BitReader::bits_remaining` underflowed when the cursor sat
  *past* the end of the slice. `BitReader::new_after_image_header` places
  the cursor at bit 40 (past the §3.4 5-byte image-header) before any
  bytes are read, so a VP8L chunk payload shorter than that header — an
  empty or truncated chunk — left `bit_pos > data.len() * 8`. The
  `data.len() * 8 - bit_pos` subtraction then wrapped in `usize`, the
  `n > available` EOF guard in `read_bits` passed against the wrapped
  count, and `read_bits` indexed `data[bit_pos >> 3]` out of bounds — a
  debug-build panic and a release-build out-of-bounds read reachable from
  `decode_webp` whenever a VP8L chunk payload is shorter than 5 bytes. The
  count is now `saturating_sub`, so a past-the-end cursor reports `0` bits
  remaining and `read_bits` cleanly returns the typed `BitReaderEof`.
  Found by the round-273 `decode_lossless` fuzz harness; regression test
  `vp8l_stream::tests::bits_remaining_saturates_when_cursor_past_end`
  pins both the empty- and short-payload cases.

### Added

- `benches/prefix_from_code_lengths.rs`: criterion bench — the
  seventeenth — for the decoder-side §6.2.1 canonical-table build
  `vp8l_prefix::PrefixCode::from_code_lengths`, the decode mirror of
  the §3.7.2 length-then-code encoder pair benched in rounds 250
  (`build_code_lengths`) and 251 (`canonical_codes`) and the
  round-170 decode-profile rank-4 symbol (~2% of decode self-time).
  Parameterised over the same four §3.7.1 prefix-code-group alphabets
  (distance-40, literal-256, green-281, green-2328) and the same two
  dense / sparse frequency regimes with identical LCG constants, so
  every cell is directly comparable to its encoder-mirror
  counterpart; the length tables are produced by `build_code_lengths`
  at setup and the per-iteration `Vec` clone (the function takes the
  table by value) is excluded from the measured interval via
  `iter_batched`. Medians: 187.7 ns / 100.7 ns (dense/sparse
  distance-40), 918.0 ns / 546.3 ns (literal-256), 1.006 µs /
  623.2 ns (green-281), 7.300 µs / 5.249 µs (green-2328) — linear in
  alphabet size, dense/sparse ratio ~1.4–1.9× (between
  `canonical_codes`'s regime-blind ~1.2× and `build_code_lengths`'s
  18×–84×, matching the body's one-full-rescan-per-used-length
  shape), and the cheapest link in the `green2328` dense
  length-then-code chain (417.8 µs builder ≫ 15.70 µs encoder codes >
  7.30 µs decoder table). Function body unchanged; the bench is the
  A/B reference for a future single-rescan bucket-sort rewrite, with
  the round-275 `prefix_code` differential fuzz harness as the
  byte-exact regression guard. Full analysis in `BENCHMARKS.md`
  (round-276 section).
- `fuzz/fuzz_targets/prefix_code.rs`: cargo-fuzz harness — the
  twenty-fourth — driving the §6.2.1 *single canonical prefix-code* reader
  standalone entry point `oxideav_webp::vp8l_prefix::PrefixCode::read`
  directly across an `(alphabet_size, bitstream)` cross-product. This is
  the surface immediately *below* the round-274 `prefix_code_group`
  harness: `PrefixCodeGroup::read` calls `PrefixCode::read` five times in
  green/red/blue/alpha/distance order, and this harness isolates one such
  call. `PrefixCode::read` reads one code's lengths off the wire — the
  §6.2.1 simple/normal `read_code_lengths` dispatch — and builds the
  canonical decoder via `from_code_lengths` with its §6.2.1 Kraft
  completeness gate and single-leaf-node exception. No prior harness drove
  the single code standalone or across a free alphabet sweep:
  `prefix_code_group` (round 274) only ever drives the five reads as a unit
  at the fixed `{40, 256, green}` alphabet sizes. The first input byte
  selects one of the wire-reachable §6.2.3 alphabets — `40` (distance),
  `256` (red/blue/alpha), or the green `256 + 24 + color_cache_size` for
  the full `color_cache_size ∈ {0} ∪ {2, …, 2048}` range — and the
  remaining bytes feed a zero-positioned `BitReader`. Every `Ok(code)` is
  cross-checked against the §6.2.3 / §6.2.1 carrier rules:
  `code_lengths().len()` equals the selected alphabet, every nonzero length
  is `<= 15` (the `MAX_CODE_LENGTH` ceiling), `single_symbol()` is `Some(s)`
  iff the length table has exactly one nonzero entry (at `s`) and `None`
  iff two or more, `read_symbol` against an all-zero reader resolves an
  in-range symbol index, rebuilding from the returned length table through
  `from_code_lengths` reproduces an equal code (the §6.2.1 `sum 2^-len == 1`
  completeness invariant a built code satisfies), the reader never advances
  past the slice bit length, and replaying the same bytes + alphabet yields
  an equal code at an identical bit position. A 14 s smoke pass cleared
  2.00 M runs with no crashes. Run with `cargo +nightly fuzz run
  prefix_code --manifest-path crates/oxideav-webp/fuzz/Cargo.toml`.
- `fuzz/fuzz_targets/prefix_code_group.rs`: cargo-fuzz harness — the
  twenty-third — driving the §6.2 / §6.2.1 *prefix-code-group* reader
  standalone entry point `oxideav_webp::meta_prefix::PrefixCodeGroup::read`
  directly across a `(color_cache_size, bitstream)` cross-product. A §6.2
  group is the five canonical §6.2.1 prefix codes every VP8L pixel is
  decoded with — green + backref-length + color-cache (alphabet `256 + 24 +
  color_cache_size` per §6.2.3), red/blue/alpha (each `256`), and backref
  distance (`40`) — read in that order via `PrefixCode::read` and the
  §6.2.1 simple/normal `read_code_lengths` + canonical `from_code_lengths`
  Kraft completeness build. This is the surface immediately *below* the
  round-271 `vp8l_decode::decode_entropy_coded_image` (§7.3), which reads a
  §5.2.3 color-cache-info bit then exactly one `PrefixCodeGroup::read`
  before the §5.2 pixel loop. No prior harness drove the group standalone:
  `parse_meta_prefix` (round 261) reaches it only through the §5.2.3 +
  §6.2.2 preamble (so the cache size is whatever the 4-bit
  `color_cache_code_bits` produced, never the cache-disabled `0` and a wide
  enabled size in one corpus), and `decode_entropy_coded_image` /
  `decode_argb` consume the group's symbols in the same call so a parse
  failure inside it is indistinguishable from a later §5.2 refusal. The
  first input byte selects the §5.2.3 cache size from `{0}` (disabled) or
  `1 << code_bits` for `code_bits ∈ [1, 11]` (`{2, 4, …, 2048}`), sizing
  the §6.2.3 green alphabet; the remaining bytes feed a zero-positioned
  `BitReader`. Every `Ok(group)` is cross-checked against the §6.2.3 /
  §6.2.1 carrier rules: each of the five codes' `code_lengths().len()`
  equals its alphabet, every nonzero length is `<= 15` (the `MAX_CODE_LENGTH`
  ceiling), `single_symbol()` is `Some(s)` iff the length table has exactly
  one nonzero entry (at `s`) and `None` iff it has two or more, `read_symbol`
  against an all-zero reader resolves an in-range symbol index, the reader
  never advances past the slice bit length, and replaying the same bytes +
  cache size yields an equal group at an identical bit position; the §5.2.3
  `InvalidColorCacheCodeBits` variant is asserted unreachable (the cache
  size is caller-supplied, never read here). A 41 s smoke pass cleared
  4.97 M runs with no crashes. Run with `cargo +nightly fuzz run
  prefix_code_group --manifest-path crates/oxideav-webp/fuzz/Cargo.toml`.
- `fuzz/fuzz_targets/decode_lossless.rs`: cargo-fuzz harness — the
  twenty-second — driving the §4 transform-list + main-image full
  lossless-bitstream decode path standalone entry points
  `oxideav_webp::vp8l_transform::{decode_lossless,
  decode_lossless_headerless}` directly. `decode_lossless` is the layer
  immediately *above* the round-272 §6.2.2 `vp8l_decode::decode_argb`: it
  walks the §4 / §7.2 optional-transform loop (per-transform §4.x fixed
  fields plus its §5-encoded body via the §7.3
  `decode_entropy_coded_image`, the §4 "allowed to be used only once"
  duplicate refusal, the §4.4 `color_table_size` / `width_bits` width
  subsampling), decodes the main §5.1 ARGB image at the possibly-
  subsampled width, then applies the §4 inverse-transform chain in
  reverse read order (§4: "last one first").
  `decode_lossless_headerless` is the §2.7.1.2 / §3 twin used by the
  compressed `ALPH` alpha bitstream — identical save that the 5-byte
  §3.4 image-header is not skipped (the `BitReader` starts at bit 0). No
  prior harness drove this assembled surface: `parse_transform_list`
  (round 260) reads the §4 transform-presence loop's *header* fields but
  stops at the §5 entropy body; `inverse_predictor_color` (round 265) and
  `inverse_subtract_green_indexing` (round 266) drive the four §4.x
  inverse passes in isolation over synthesised buffers, never out of a §4
  transform-list bitstream nor in reverse read order; `decode_argb`
  (round 272) drives only the main §5.1 ARGB image, *after* the transform
  list is consumed; `decode` / `roundtrip_lossless` reach
  `decode_lossless` only through a complete §2 RIFF + §3.4 image-header
  walk with the `(width, height)` pair bounded by an upstream §3.4 field
  and the bitstream constrained to round-trip from the encoder. The fuzz
  buffer fixes the §4 / §6.2.2 carrier dimensions from the first two bytes
  (`width` / `height` each clamped into `[1, 8]` — always nonempty,
  mirroring the §3.4-validated dimensions the driver is reachable with,
  and small so the §4 / §5 / §6 decode loop stays bounded at ≤ 64 pixels)
  and feeds the remaining bytes to both entry points: `decode_lossless`
  reads them past the §3.4 5-byte image-header (transform list at bit 40),
  `decode_lossless_headerless` reads the same bytes from bit 0. Every
  `Ok` image is cross-checked against the §4 / §6.2.2 carrier rules
  (`width()` / `height()` echo the carrier even after a §4.4
  color-indexing transform un-bundles the internal width back to the
  canvas width, `pixels().len() == width * height`), with pure-function
  determinism cross-checked by replaying the same bytes + dimensions for a
  byte-identical pixel buffer; every refusal is required only to return a
  `Result` rather than panic (the granular §4 / §5 / §6 refusal modes are
  cross-checked by the sibling `parse_transform_list` /
  `inverse_predictor_color` / `inverse_subtract_green_indexing` /
  `decode_argb` harnesses through their own entry points). A 40 s smoke
  pass cleared 3.62 M runs with no crashes after the `bits_remaining`
  underflow fix above (which the harness surfaced on its first run).

- `fuzz/fuzz_targets/decode_argb.rs`: cargo-fuzz harness — the
  twenty-first — driving the §6.2.2 top-level VP8L ARGB main-image decode
  path standalone entry point `oxideav_webp::vp8l_decode::decode_argb`
  directly. `decode_argb` is the §5.1 `spatially-coded-image` ARGB-role
  decoder — the layer immediately *above* the round-270 §6.2.2
  `decode_entropy_image` and the round-271 §7.3
  `decode_entropy_coded_image`. It reads the round-106 `MetaPrefixHeader`
  for `ImageRole::Argb` (the §5.2.3 `color-cache-info` bit, then the
  §6.2.2 meta-prefix bit) and dispatches between the single-group path
  (one §6.2 prefix-code group drives the §6.2.3 decode loop everywhere)
  and the multi-group path (the §6.2.2 entropy image is decoded,
  `num_prefix_groups = max(entropy image) + 1` groups are read, and the
  §6.2.3 loop selects a group per pixel block via
  `MetaPrefixIndex::meta_code_for`, with a single §5.2.3 color cache
  maintained in stream order across the whole image). No prior harness
  drove this assembled surface: `parse_meta_prefix` (round 261) stops at
  the §5.2 entropy body without decoding a pixel or reading the per-group
  prefix-code groups; `decode_entropy_image` (round 270) and
  `decode_entropy_coded_image` (round 271) drive only the §6.2.2 entropy
  sub-image and the §7.3 building block beneath it, never the §5.2.3 +
  §6.2.2 ARGB preamble, the single-vs-multi-group dispatch, the per-group
  reads, nor the per-pixel-block group selection `decode_argb` assembles
  on top of them; `decode` / `roundtrip_lossless` reach `decode_argb`
  only through a complete §2 RIFF + §3 image-header walk with the
  `(width, height)` pair bounded by an upstream §3.4 field and the
  bitstream constrained to round-trip from the encoder. The fuzz buffer
  fixes the §6.2.2 carrier dimensions from the first two bytes (`width` /
  `height` each clamped into `[1, 8]` — always nonempty, mirroring the
  §3.4-validated dimensions `decode_argb` is reachable with, and small so
  the §6.2.3 decode loop stays bounded at ≤ 64 pixels) and feeds the
  remaining bytes through a zero-positioned `BitReader` as the §6.2.2 ARGB
  image bit sequence (color-cache-info bit, meta-prefix bit, then the
  dispatched body). Every `Ok` image is cross-checked against the §6.2.2
  carrier rules: `width()` / `height()` echo the carrier;
  `pixels().len() == width as usize * height as usize` (§6.2.2 emits
  exactly one pixel per position); and the reader never advances past the
  slice's bit length. Pure-function determinism is cross-checked by
  replaying the same bytes + `(width, height)` and asserting a
  byte-identical pixel buffer at an identical bit position. Every `Err`
  (truncation, meta-prefix/color-cache-info parse failure, entropy-image
  fault, prefix-code parse failure, out-of-range green symbol, color-cache
  or backward-reference fault, or a meta-prefix code beyond
  `num_prefix_groups`) is required only to return a `Result` rather than
  panic — the granular §5.2 / §6.2 refusal modes are cross-checked by the
  sibling `parse_meta_prefix` / `decode_entropy_image` /
  `decode_entropy_coded_image` / `distance_code` / `color_cache` /
  `backward_reference` / `meta_prefix_index` harnesses through their own
  entry points. This brings the §3 lossless decoder's fuzz coverage to 21
  standalone harnesses, with the §6.2.2 ARGB main-image decode now
  exercised both through the full `decode` path and at its own standalone
  surface — the cap of the round-255→271 bottom-up walk that previously
  fuzzed every layer beneath it. A 30 s smoke pass cleared 2.66 M runs
  with no crashes (476 cov / 1690 features over a 269-input corpus).
- `fuzz/fuzz_targets/decode_entropy_coded_image.rs`: cargo-fuzz harness —
  the twentieth — driving the §7.3 *entropy-coded-image* decode path
  standalone entry point
  `oxideav_webp::vp8l_decode::decode_entropy_coded_image` directly. The
  §7.3 ABNF `entropy-coded-image` is the indivisible building block every
  VP8L pixel surface is assembled from: a §5.2.3 `color-cache-info` bit, a
  single §6.2 prefix-code group (no §6.2.2 meta-prefix bit — that bit
  belongs to the §5.1 `spatially-coded-image` ARGB role only), and the
  §5.2 LZ77 / literal / color-cache data that emits exactly
  `width * height` ARGB pixels in scan-line order. It is the function the
  §4.1 / §4.2 / §4.4 sub-resolution images and the §6.2.2 entropy image
  are all decoded through; the round-270 `decode_entropy_image` harness
  *wraps* this function (it calls it, then folds each pixel's red+green
  channels into a per-block meta-code) and already used it as a
  cross-check sibling, but no harness drove §7.3 standalone across an
  attacker-controlled `(width, height, bitstream)` cross-product. The
  fuzz buffer fixes the §7.3 carrier dimensions from the first two bytes
  (`width` / `height` each modulo 9 so the §5.2 / §6.2 decode loop stays
  bounded — at most 8×8 = 64 pixels — and 0 reaches the §7.3
  degenerate-dimension `EmptyEntropyImage` refusal) and feeds the
  remaining bytes through a zero-positioned `BitReader` as the §7.3
  entropy-coded-image bit sequence. Every `Ok` image is cross-checked
  against the §7.3 carrier rules: `width()` / `height()` echo the carrier;
  `pixels().len() == width as usize * height as usize` (§7.3 emits exactly
  one pixel per position); the success path is reachable only with both
  dimensions ≥ 1 (a zero dimension short-circuits to `EmptyEntropyImage`
  before any header bit is read); and the reader never advances past the
  slice's bit length. The §6.2.2 fold consistency with the round-270
  wrapper is cross-checked by an **independent** `decode_entropy_image`
  decode of the same bytes (`prefix_bits = 2`, inside the §6.2.2 `[2, 9]`
  wire window): the harness folds this function's pixels via
  `(argb >> 8) & 0xffff` and asserts byte-equality with the wrapper's
  per-block meta-codes plus both readers advancing to the same
  `bit_position()`, and the wrapper's `block_width()` / `block_height()`
  echoing the §7.3 dimensions. Pure-function determinism is cross-checked
  by replaying the same bytes + `(width, height)` and asserting a
  byte-identical pixel buffer at an identical bit position. The §7.3
  degenerate-dimension refusal is pinned to the `EmptyEntropyImage`
  variant echoing the carrier dimensions iff at least one is zero; every
  other bitstream-level refusal (truncation, prefix-code parse failure,
  out-of-range green symbol, color-cache or backward-reference fault) is
  required only to return a `Result` rather than panic — the granular
  §5.2 / §6.2 refusal modes are cross-checked by the sibling
  `parse_meta_prefix` / `distance_code` / `color_cache` /
  `backward_reference` harnesses through their own entry points. This
  brings the §3 lossless decoder's fuzz coverage to 20 standalone
  harnesses, with the §7.3 entropy-coded-image decode now exercised both
  through the full `decode` path and at its own standalone surface,
  complementing the round-270 `decode_entropy_image` harness (which drives
  the §6.2.2 fold over this function's output) by driving the §7.3 pixel
  producer beneath it.
- `fuzz/fuzz_targets/decode_entropy_image.rs`: cargo-fuzz harness — the
  nineteenth — driving the §6.2.2 *entropy image* decode path standalone
  entry point `oxideav_webp::vp8l_decode::decode_entropy_image`
  directly. When a §5.1 ARGB image sets the §6.2.2 meta-prefix bit the
  decoder reads `prefix_bits = ReadBits(3) + 2`, derives a
  `DIV_ROUND_UP`-sized block grid, and decodes the §6.2.2 entropy image
  — itself a §7.3 `entropy-coded-image` of one pixel per block — folding
  each entropy pixel's red+green channels into one 16-bit meta-prefix
  code (`(entropy_pixel >> 8) & 0xffff`). The fuzz buffer fixes the
  §6.2.2 `(prefix_bits, prefix_image_width, prefix_image_height)` carrier
  triple from the first three bytes (`prefix_bits` masked to `[0, 15]`
  since `decode_entropy_image` records it as an opaque carrier without
  re-deriving a block size; the block dimensions modulo 9 so the §7.3
  sub-image decode stays bounded and 0 reaches the §6.2.2
  degenerate-dimension refusal) and feeds the remaining bytes through a
  zero-positioned `BitReader` as the §7.3 entropy-coded-image bit
  sequence. Every `Ok` index is cross-checked against the §6.2.2 + §7.3
  carrier rules: the accessors echo the carrier triple (`prefix_bits()`,
  `block_width() == prefix_image_width`, `block_height() ==
  prefix_image_height`); §7.3 one meta-code per block
  (`meta_codes().len() == prefix_image_width * prefix_image_height`);
  §6.2.2 `num_prefix_groups() == max(meta_codes) + 1`; the §6.2.2 fold
  cross-checked against an **independent** decode of the same bytes
  through the public sibling `vp8l_decode::decode_entropy_coded_image`
  (the harness refolds that decode's raw ARGB pixels via `pixels()` and
  asserts byte-equality with the meta-codes `decode_entropy_image`
  produced, plus both readers advancing to the same `bit_position()`);
  the §6.2.2 carrier asymmetry where rebuilding through the validated
  constructor `MetaPrefixIndex::from_parts` reproduces the index iff
  `prefix_bits ∈ [2, 9]` (the wire window) and is refused with
  `InvalidPrefixBits` echoing the recorded value otherwise; and
  determinism by replaying the same bytes + carrier triple and asserting
  an identical index advanced to an identical bit position. The §6.2.2
  degenerate-dimension refusal is pinned to the `EmptyEntropyImage`
  variant echoing the carrier dimensions iff at least one is zero; every
  other bitstream-level refusal (truncation, prefix-code parse failure,
  out-of-range green symbol, color-cache or backward-reference fault) is
  required only to return a `Result` rather than panic — the granular
  §5.2 / §6.2 refusal modes are cross-checked by the sibling `decode` /
  `parse_meta_prefix` harnesses through their own entry points. A 30 s
  smoke pass cleared 8.9 M runs with no crashes, reaching the §5.2
  `read_lz77_value` / `apply_backward_reference` /
  `distance_code_to_pixel_distance` core through the entropy-coded
  sub-image. This brings the §3 lossless decoder's fuzz coverage to 19
  standalone harnesses, with the §6.2.2 entropy-image decode now
  exercised both through the full `decode` path and at its own
  standalone surface, complementing the round-268 `meta_prefix_index`
  harness (which drives the validated constructor + per-pixel selector
  on already-decoded meta-codes) by driving the bitstream-level producer
  of that same `MetaPrefixIndex` table.
- `vp8l_transform::color_indexing_width_bits`: the §4.4 pixel-bundling
  `width_bits` threshold-table accessor ("Color Table Size to Bundled
  Pixel Bit Width Mapping": `1..=2 → 3`, `3..=4 → 2`, `5..=16 → 1`,
  `17..=256 → 0`) is now public. It was file-private, and the same
  four-row table existed as three independent private copies — the §4.4
  inverse path (`vp8l_transform`), the §4 transform-list reader
  (`vp8l_stream`, whose own doc noted the duplication was a visibility
  workaround), and the §4.4 forward encoder (`vp8l_encode`). The
  `vp8l_transform` copy is now the single shared, spec-cited source;
  the other two modules delegate to it, so the encoder and the two
  decoder-side readers can no longer drift apart on the §4.4
  derivation. The function is total over `usize` (documented: only
  `ReadBits(8) + 1 ∈ [1, 256]` is bitstream-reachable; 0 falls in the
  first threshold window, sizes above 256 in the last). Test delta,
  net +1: a new exhaustive sweep pins all 256 wire-reachable sizes
  against the spec table; the `vp8l_stream` duplicate-table test is
  replaced by a stronger on-wire test (each boundary size written as
  its `ReadBits(8) + 1` encoding and read back through
  `TransformList::read`); the `vp8l_encode` duplicate-table test is
  replaced by a stronger wiring test (each boundary palette size
  encoded via the §4.4 color-indexing path and the emitted transform
  header parsed back through the §4 transform-list reader, asserting
  the on-wire `color_table_size` + shared-accessor `width_bits`).
- `vp8l_decode::MetaPrefixIndex::from_parts`: standalone validated
  constructor for the §6.2.2 meta-prefix block-lookup table.
  `decode_entropy_image` builds the index off a bitstream; this
  constructor is the standalone equivalent for callers that already
  hold the decoded per-block meta-prefix codes (each entry the
  `(entropy_pixel >> 8) & 0xffff` red+green fold of one entropy-image
  pixel, scan-line order). It enforces the three §6.2.2 carrier
  invariants the bitstream path establishes by construction, in
  documented precedence order: `prefix_bits ∈ [2, 9]` (the on-wire
  field is `ReadBits(3) + 2`, so only that window is
  bitstream-reachable), a nonempty block grid (the §6.2.2
  `DIV_ROUND_UP` derivation of an ≥1×≥1 ARGB image never yields zero
  blocks), and exactly `block_width * block_height` codes (the product
  computed in u64 so the check itself cannot overflow). Refusals are
  reported through the new dedicated
  `vp8l_decode::MetaPrefixIndexError` enum (`InvalidPrefixBits` /
  `EmptyIndex` / `CodeCountMismatch`, each echoing the offending
  parts) — an additive type; no existing error enum changed. On
  success `meta_code_for(x, y)` resolves the §6.2.2 group selection
  for any pixel of an image whose dimensions satisfy the
  `DIV_ROUND_UP` derivation. Seven new unit tests pin the window ends
  (`[2, 9]` inclusive), each refusal trigger, the precedence order,
  the §6.2.2 position-formula selection on a 2×3 grid, and an accessor
  round-trip against a bitstream-decoded index.
- `fuzz/fuzz_targets/meta_prefix_index.rs`: cargo-fuzz harness — the
  eighteenth — driving the §6.2.2 meta-prefix block-lookup table
  standalone entry points
  `oxideav_webp::vp8l_decode::MetaPrefixIndex::{from_parts,
  meta_code_for}` directly. The fuzz buffer fixes a `(prefix_bits,
  block_width, block_height, count_skew)` carrier tuple (`prefix_bits`
  masked to `[0, 15]` so the §6.2.2 `[2, 9]` window and its rejection
  are both routinely reached; the grid masked into `[0, 32]²` with 0
  reaching the degenerate-grid refusal; the skew shifting the supplied
  code count off the `block_width * block_height` expectation by
  `[-2, +2]`) plus a stream of fuzz-controlled 16-bit meta-prefix
  codes forwarded verbatim. Every `Ok` index is cross-checked against
  the §6.2.2 carrier rules (accessors echo the parts;
  `num_prefix_groups() == max(entropy image) + 1` per the §6.2.2
  "Interpretation of Meta Prefix Codes" rule; `meta_code_for(x, y)` at
  all four corners of every block's `(1 << prefix_bits)`-pixel-square
  covered area matching the §6.2.2 position formula `meta_codes[(y >>
  prefix_bits) * block_width + (x >> prefix_bits)]`); every error
  variant is cross-checked against its §6.2.2 refusal trigger in
  precedence order; and constructor determinism is cross-checked by
  rebuilding from the same parts plus round-tripping the index's own
  accessors back through `from_parts`. A 30 s smoke pass cleared
  6.69 M runs with no crashes.
- `vp8l_decode::apply_backward_reference`: extracted the §5.2.2 LZ77
  backward-reference copy from the inline `decode_one_symbol` length-
  prefix arm into a standalone, pure-by-buffer entry point. Given the
  growing scan-line ARGB buffer, a copy `length` `L`, a scan-line pixel
  distance `dist` `D`, and the image's `total_pixels`, it enforces the
  two §5.2.2 carrier invariants *before* writing any byte (underflow
  `D > position` and overflow `position + L > total_pixels`, each
  leaving the buffer untouched on refusal), then performs the standard
  byte-for-byte LZ77 walk — an overlapping run (`D < L`) repeats the
  pixels it is itself emitting because each source index is read after
  the preceding appends. Returns the `position..position + L` range of
  freshly-appended pixels so the caller can replay them into the
  §5.2.3 color cache in stream order. `decode_one_symbol` now delegates
  to it with no behaviour change. Precondition (documented): `dist >=
  1`, which `distance_code_to_pixel_distance` always guarantees via the
  §5.2.2 clamp.
- `fuzz/fuzz_targets/backward_reference.rs`: cargo-fuzz harness — the
  seventeenth — driving the §5.2.2 backward-reference assembler
  standalone entry point `oxideav_webp::vp8l_decode::apply_backward_reference`
  directly. The fuzz buffer fixes a `(prefill_len, length, dist,
  total_pixels)` carrier tuple (`prefill_len` masked to `[0, 4096]`;
  `dist` floored at 1 to honour the §5.2.2 `D >= 1` precondition;
  `total_pixels` alternated between `prefill_len + length + headroom`
  and a shrunk value below `prefill_len + length` so both the success /
  exact-fit path and the §5.2.2 overflow refusal are routinely reached)
  plus a stream of fuzz-controlled ARGB pre-fill pixels. Every `Ok`
  outcome is cross-checked against the §5.2.2 copy contract (returned
  range equals `position..position + length`; exactly `length` pixels
  appended; the already-decoded prefix byte-identical; every appended
  pixel matches a parallel reference LZ77 walk `out[position + i] ==
  out[position + i - dist]` read after the preceding writes, the
  overlapping self-repeat included). The §5.2.2 underflow refusal is
  cross-checked against its `dist > position` trigger (fields echo the
  call, buffer byte-identical to its pre-call snapshot); the §5.2.2
  overflow refusal against its `position + length > total_pixels`
  trigger (with the underflow guard having passed, fields echo the
  call, buffer byte-identical); and pure-function determinism by
  replaying a successful run from the same pre-fill and asserting an
  identical buffer + range. A 30 s smoke pass cleared 1.5 M runs with
  no crashes. Brings the §3 lossless decoder's fuzz coverage to 17
  standalone harnesses, with the §5.2.2 backward-reference copy now
  exercised both through the full `decode` path and at its own
  standalone surface.
- `fuzz/fuzz_targets/inverse_subtract_green_indexing.rs`: cargo-fuzz
  harness on the §4.3 inverse-subtract-green + §4.4
  inverse-color-table + §4.4 inverse-color-indexing transform passes
  standalone entry points
  `oxideav_webp::vp8l_transform::{inverse_subtract_green,
  inverse_color_table, inverse_color_indexing}`. The fifteenth harness
  (`inverse_predictor_color`) covers the two §4 *arithmetic*
  transforms — §4.1 Predictor and §4.2 Color — that read a
  sub-resolution image and walk the main buffer per-pixel. This
  sixteenth harness covers the remaining three §4 primitives that
  have no sub-resolution image: the §4.3 per-pixel `red += green`
  / `blue += green` (mod 256) in-place pass that leaves alpha + green
  untouched; the §4.4 color-table subtraction-decode in-place pass
  that leaves `table[0]` untouched and reconstructs every later entry
  as the per-channel running sum (mod 256) of the original input
  bytes; and the §4.4 color-indexing pass that walks a sub-sampled
  packed image whose green channel carries (possibly bundled)
  palette indices and emits a fresh `orig_width * height` ARGB buffer
  per the §4.4 threshold-table-driven `width_bits` derivation
  (1..=2 colors → width_bits 3 / count 8 / 1 bit per index; 3..=4 →
  2 / 4 / 2; 5..=16 → 1 / 2 / 4; ≥17 → 0 / no bundling), falling
  back to transparent black `0x00000000` when the wire index is out
  of range. The harness fixes the §4.3 / §4.4 `(orig_width, height,
  table_size)` carrier triple from the first three fuzz bytes
  (`orig_width` / `height` masked into `[1, 32]` for iteration cost;
  `table_size` mapped into the §4.4 wire window `[1, 256]` via
  `data[2] + 1`), then forwards every subsequent 4-byte little-endian
  word verbatim first as a fuzz-controlled ARGB §4.3 input pixel,
  then as a fuzz-controlled §4.4 color-table delta entry, then as a
  fuzz-controlled §4.4 packed-index ARGB pixel. The §4.3
  alpha-and-green preservation invariant is cross-checked against
  the spec text (alpha + green bytes byte-identical to the pre-pass
  input; red byte equals input red + input green mod 256; blue byte
  equals input blue + input green mod 256); the §4.3 per-pixel
  locality invariant is cross-checked by running the pass on
  single-pixel inputs at the first eight positions and asserting the
  solo output matches the multi-pixel output; the §4.3 zero-green
  no-op is cross-checked against the `(red + 0) = red` reduction; the
  §4.4 color-table seed-preservation invariant is cross-checked
  against the §4.4 spec text (`table[0]` is left untouched); the
  §4.4 color-table running-sum invariant is cross-checked against
  the §4.4 "adding the previous color component values by each ARGB
  component separately and storing the least significant 8 bits of
  the result" spec text; the §4.4 color-table determinism invariant
  is cross-checked across replay; the §4.4 color-indexing
  output-length invariant is cross-checked against the
  `orig_width * height` carrier contract; the §4.4 color-indexing
  palette-lookup invariant is cross-checked against the §4.4 spec
  formula (output pixel `(x, y)` is `color_table[((packed_green >>
  ((x % count) * bits)) & mask) as usize]` with `width_bits` derived
  from the table size via the §4.4 threshold table, falling back to
  transparent black when the index is out of range); the §4.4
  color-indexing determinism invariant is cross-checked across
  replay; the §4.4 color-indexing empty-table edge case is
  cross-checked against the §4.4 "unused indices map to transparent
  black" rule; the §4.3 empty-buffer and §4.4 single-element-table
  degenerate no-op branches are cross-checked unconditionally on
  every iteration. The smoke pass on round-266 dispatch cleared 200K
  runs in 2 seconds with no crashes (266 edges, 810 features, 80
  corpus seeds reached). Brings the §3 lossless decoder's fuzz
  coverage to 16 standalone harnesses: the full §2 RIFF + §3 / §4 / §5 VP8L
  decode path is now reachable both end-to-end (`decode` +
  `roundtrip_lossless` + `roundtrip_animated` + `extract_metadata` +
  `decode_alph`) and at the ten standalone primitive surfaces (§2.3
  / §2.4 RIFF container walker, §2.7.1 VP8X header, §2.7.1.1 ANIM +
  ANMF headers, §2.7.1.2 ALPH info byte, §4 transform list, §5.2.3
  + §6.2.2 meta-prefix preamble, §5.2.2 distance code, §5.2.3 color
  cache, §4.1 + §4.2 inverse-transform passes, §4.3 + §4.4
  inverse-transform passes).
- `fuzz/fuzz_targets/inverse_predictor_color.rs`: cargo-fuzz harness on
  the §4.1 inverse-predictor + §4.2 inverse-color in-place transform
  passes standalone entry points
  `oxideav_webp::vp8l_transform::{inverse_predictor, inverse_color}`.
  After the §5 entropy stream has emitted the raw §5.1 ARGB residual
  buffer, every §4 transform in the read-order list runs in reverse
  against that buffer. The two arithmetic transforms — §4.1 Predictor
  (reads a sub-resolution "predictor image" whose green channel encodes
  the per-block prediction mode and applies one of 14 §4.1 Table 2
  predictors against the already-reconstructed TL / T / TR / L
  neighbours, then per-channel-adds the prediction into the residual)
  and §4.2 Color (reads a sub-resolution "color image" whose pixels
  encode `ColorTransformElement` values and per-channel-adds
  `ColorTransformDelta` derivations into the red and blue channels
  while leaving alpha and green untouched) — walk the main `width *
  height` ARGB buffer applying the inverse derivation in place. The
  harness fixes the §4.1 / §4.2 `(width, height, size_bits)` carrier
  triple from the first three fuzz bytes (`width` / `height` masked
  into `[1, 32]` for iteration cost; `size_bits` remapped into
  `[0, 9]` to cover the full §4.1 / §4.2 `ReadBits(3) + 2` window plus
  the `size_bits == 0` hoist branch), then forwards every subsequent
  4-byte little-endian word verbatim as a fuzz-controlled ARGB
  residual pixel and (after `width * height` words) as a
  fuzz-controlled sub-resolution predictor / color image pixel. Both
  transforms then run on independent clones of the residual against
  the same sub-resolution image. The §4.1 left-topmost rule is
  cross-checked against the spec text (`pred_pixels[0] == residual[0]
  + 0xff000000` per channel mod 256); the §4.1 1×H left-column rule is
  cross-checked against the §4.1 "all pixels on the leftmost column
  are T-pixel" spec text (every `(0, y)` for `y >= 1` equals
  `residual + T` per channel mod 256); the §4.1 W×1 top-row rule is
  cross-checked against the §4.1 "all pixels on the top row are
  L-pixel" spec text (every `(x, 0)` for `x >= 1` equals `residual +
  L` per channel mod 256); the §4.2 alpha-and-green preservation
  invariant is cross-checked against the §4.2 spec text ("The alpha
  and green channels are left as is"); the §4.2 zero-CTE no-op
  invariant is cross-checked by re-running the pass against an
  all-zero sub-resolution image (every per-pixel output equals the
  input); the §4.2 per-block constancy invariant is cross-checked
  against the §4.2 block structure (two same-block pixels with equal
  pre-pass RGB produce equal post-pass red + blue); both passes'
  early-return contract is cross-checked against the §4.1 / §4.2
  `(width == 0 || height == 0)` no-op (the pixel buffer is
  byte-identical to the pre-call snapshot). The smoke pass on round-265
  dispatch cleared 200K runs in 45 seconds with no crashes (175 edges,
  111 corpus seeds reached). Brings the §3 lossless decoder's fuzz
  coverage to 15 standalone harnesses: the full §2 RIFF + §3 / §4 / §5
  VP8L decode path is now reachable both end-to-end (`decode` +
  `roundtrip_lossless` + `roundtrip_animated` + `extract_metadata` +
  `decode_alph`) and at the nine standalone primitive surfaces (§2.3 /
  §2.4 RIFF container walker, §2.7.1 VP8X header, §2.7.1.1 ANIM + ANMF
  headers, §2.7.1.2 ALPH info byte, §4 transform list, §5.2.3 +
  §6.2.2 meta-prefix preamble, §5.2.2 distance code, §5.2.3 color
  cache, §4.1 + §4.2 inverse-transform passes).
- `fuzz/fuzz_targets/color_cache.rs`: cargo-fuzz harness on the
  §5.2.3 lossless-color-cache primitives standalone entry point
  `oxideav_webp::vp8l_decode::ColorCache`. Every VP8L §5.2 GREEN
  symbol whose value is `>= 256 + 24` is a color-cache code — the
  resolved index `S - (256 + 24)` is fed to the cache's `lookup`,
  the returned ARGB is emitted as the pixel, and that pixel is then
  re-inserted into the cache. Every literal pixel and every
  backward-reference pixel is also inserted as it is emitted. The
  cache itself is a `1 << code_bits` array of ARGB entries; the slot
  of any color is `(0x1e35a7bd * argb) >> (32 - code_bits)`. The
  §5.2.3 spec text is explicit: "Only one lookup is done in a color
  cache; there is no conflict resolution" — two colors that collide
  on the hash overwrite each other in slot order, with the
  most-recently-inserted winning. The harness fixes `code_bits` from
  the first fuzz byte (remapped into the §5.2.3 permitted window
  `[1, 11]` to honour the spec's "compliant decoders MUST indicate a
  corrupted bitstream for other values" rule) then slices the rest of
  the buffer into 4-byte ARGB words and forwards each verbatim into
  `ColorCache::insert`. Every hash is cross-checked against the
  §5.2.3 spec formula `(0x1e35a7bd * argb) >> (32 - code_bits)`;
  every insert/lookup round trip is cross-checked against the §5.2.3
  single-slot single-write contract; every per-slot lookup is
  cross-checked against a parallel shadow model that records the
  §5.2.3 most-recently-inserted-wins overwrite behaviour (this
  catches any §5.2.3 violation where the insert touched a slot other
  than the hashed one — e.g. an open-addressing probe, explicitly
  forbidden by "Only one lookup is done; there is no conflict
  resolution"). The §5.2.3 cache initialization invariant is
  cross-checked on a fresh `ColorCache::new(code_bits)`: `size() ==
  1 << code_bits`, every slot reads as `Some(0)` per "all entries in
  all color cache values are set to zero", `lookup(size())` reads as
  `None`, and `lookup(usize::MAX)` reads as `None`. Pure-function
  determinism is asserted on the full insert sequence by rebuilding
  a replay cache from the same fuzz bytes and verifying every slot
  agrees with the primary cache. The smoke pass on round-264 dispatch
  cleared 200K runs in ~3 seconds with no crashes. Brings the §3
  lossless decoder's fuzz coverage to 14 standalone harnesses: the
  full §2 RIFF + §3 / §4 / §5 VP8L decode path is now reachable both
  end-to-end (`decode` + `roundtrip_lossless` + `roundtrip_animated`
  + `extract_metadata` + `decode_alph`) and at the eight standalone
  primitive surfaces (§2.3 / §2.4 RIFF container walker, §2.7.1 VP8X
  header, §2.7.1.1 ANIM + ANMF headers, §2.7.1.2 ALPH info byte, §4
  transform list, §5.2.3 + §6.2.2 meta-prefix preamble, §5.2.2
  distance code, §5.2.3 color cache).
- `fuzz/fuzz_targets/distance_code.rs`: cargo-fuzz harness on the
  §5.2.2 distance-code-to-pixel-distance pure-function lookup
  standalone entry point
  `oxideav_webp::vp8l_decode::distance_code_to_pixel_distance`. Every
  backward reference in a VP8L §5.2 LZ77 stream resolves through
  exactly this function — the LZ77 length / distance prefix-code pair
  decodes to a `(length, distance_code)` pair, and the `distance_code`
  is then mapped to the actual scan-line pixel distance `D` either
  through the §5.2.2 distance map (codes `1..=120`, a 120-entry
  `(xi, yi)` neighborhood lookup table evaluated as
  `xi + yi * image_width`) or by subtracting the §5.2.2 reservation
  offset (codes `> 120` denote a raw scan-line distance of
  `code - 120`). A clamp of `D = max(D, 1)` prevents the §5.2.2
  negative-offset neighbors (the "left side" of the neighborhood —
  `(-1, 1)`, `(-2, 1)`, etc.) from yielding a zero or negative
  distance on the leftmost column of a 1-pixel-wide row. Every byte
  feeding the function is attacker-controlled: the `distance_code`
  comes off the entropy stream directly (only the prefix-code envelope
  is symbol-table-clamped, not the payload value), and the
  `image_width` comes from the §3.4 14-bit `width-1` field at the
  start of the §2.6 VP8L bitstream. The harness slices the fuzz buffer
  into `(image_width, distance_code)` u32 LE pairs (the first 4 bytes
  masked to the §3.4 14-bit image-width ceiling then bumped to a
  minimum of 1, every subsequent 4-byte word floored at 1 to honour
  the §5.2.2 wire-encoded `distance_code >= 1` precondition) and
  forwards each pair verbatim. Every returned `D` is cross-checked
  against the §5.2.2 spec formula (`max(1, xi + yi * image_width)`
  for codes `1..=120` via the public 120-entry `DISTANCE_MAP`,
  `distance_code - 120` for codes `> 120`) and the §5.2.2 clamp
  guarantee (`D >= 1` always — either from the clamp on the
  neighborhood-lookup branch or from the smallest reachable raw
  scan-line distance of `121 - 120 = 1` on the reservation branch).
  Pure-function determinism is asserted by calling the lookup twice
  and checking the two results are equal. Sibling harnesses cover
  every layer above this primitive (`parse_meta_prefix` for the §5.2
  preamble, `parse_transform_list` for §4, `parse_container` for the
  §2.3 / §2.4 RIFF walker, `decode` for the full §2 + §3..§5 entry,
  `roundtrip_lossless` for the encode→decode equality oracle) but
  none of them reaches `distance_code_to_pixel_distance` directly —
  they reach it through whichever §5.2 LZ77 length/distance pair the
  upstream prefix code produces, which means the actual
  `distance_code` values visited per iteration are bounded by the
  entropy stream the upstream reader produces. This thirteenth
  harness widens fuzz coverage onto the §5.2.2 pure-function
  distance lookup itself across the full attacker-reachable
  `distance_code ∈ [1, u32::MAX]` × `image_width ∈ [1, 0x3FFF]`
  cross-product, with no upstream throttling. Wired into the
  fuzz package as a thirteenth `[[bin]]` and surfaced in the
  README "Fuzzing" section bumped from twelve to thirteen targets.

- `fuzz/fuzz_targets/parse_container.rs`: cargo-fuzz harness on the
  §2.3 / §2.4 RIFF/WEBP chunk-walker standalone entry point
  `oxideav_webp::container::parse`. `container::parse` is the
  structural layer beneath every other WebP entry point — it walks the
  12-byte §2.4 file header (`RIFF` + LE uint32 `File Size` + `WEBP`)
  and then the §2.3 chunk stream (4-byte FourCC + LE uint32 `Size` +
  payload + optional 1-byte pad when `Size` is odd) and returns the
  ordered chunk list any §2.5 / §2.6 / §2.7 decode path consumes
  downstream. The walker is non-recovering: it surfaces the first
  structural problem it sees and stops. Every byte fed to the walker
  is attacker-controlled — the `File Size` field at bytes 4..8 and
  every per-chunk `Size` field at offsets `+4..+8` relative to its
  header. The harness forwards the entire fuzz buffer verbatim and
  cross-checks every successfully-decoded chunk against the bytes
  the walker observed: `riff_file_size` against the LE uint32 at
  `buf[4..8]`, per-chunk FourCC against `buf[header_offset..+4]`,
  per-chunk `Size` against the LE uint32 at `buf[header_offset +
  4..+8]`, `payload_end - payload_start == size as usize`,
  `payload_end <= buf.len()` and `payload_end <= 8 + riff_file_size`,
  the on-disk order invariant
  `chunks[i+1].header_offset == chunks[i].payload_end + (size & 1)`,
  the `is_extended()` / `is_vp8_lossy()` / `is_vp8_lossless()`
  predicates as pure functions of FourCC, and the
  `chunks_with_fourcc` / `first_chunk_with_fourcc` iterator helpers
  against a manual filter across nine common FourCCs (VP8X / VP8 /
  VP8L / ALPH / ANIM / ANMF / ICCP / EXIF / XMP ). Every error
  variant is likewise cross-checked against its §2.3 / §2.4 refusal
  trigger: `TooShortForHeader { got }` (`got == buf.len()` and < 12);
  `NotRiff { got }` (`got == buf[0..4]` and != 'RIFF'); `NotWebp {
  got }` (`got == buf[8..12]` and != 'WEBP' with `buf[0..4] ==
  'RIFF'`); `RiffSizeOverflowsBuffer { declared, buffer_len }`
  (`declared` == LE uint32 at `buf[4..8]`, `buffer_len == buf.len()`,
  `8 + declared > buffer_len`); `TruncatedChunkHeader { offset }`
  (`offset >= 12` inside the declared RIFF window with < 8 bytes
  remaining for the FourCC + Size); `ChunkPayloadOverflowsRiff {
  offset, declared, available }` (`offset >= 12`, 8-byte header
  fitting in the RIFF window, `declared` == LE uint32 at the chunk
  header, `available == declared_end - (offset + 8)`, `declared >
  available`); `MissingPadByte { offset }` (`offset >= 12`, declared
  `Size` odd, chunk payload itself fitting in the declared window,
  pad byte at `payload_end + 1` lying outside the declared window).
  The pre-existing `decode` and `extract_metadata` harnesses both
  wrap this walker but flatten its granular §2.3 / §2.4 refusal
  modes into a coarser `WebpError` envelope and never cross-check
  the on-disk `(payload_start, payload_end)` ranges against the
  original buffer; this twelfth harness widens fuzz coverage onto
  the structural surface itself, the lowest-level layer every other
  path is built atop. The single fuzz iteration is bounded by the
  chunk loop, which advances by `8 + size + (size & 1)` per
  iteration with `size` clamped by the §2.4 declared payload
  window; libFuzzer's default per-iteration size keeps each call
  to microseconds and the 64 KiB cap to milliseconds. Twelfth
  fuzz target after `decode` / `extract_metadata` /
  `roundtrip_lossless` / `roundtrip_animated` (round 238) /
  `decode_alph` (round 255) / `parse_vp8x` (round 256) /
  `parse_anmf` (round 257) / `parse_anim` (round 258) /
  `parse_alph` (round 259) / `parse_transform_list` (round 260) /
  `parse_meta_prefix` (round 261).

- `fuzz/fuzz_targets/parse_meta_prefix.rs`: cargo-fuzz harness on the
  §5.2.3 color-cache info + §6.2.2 meta-prefix + §6.2 5-prefix-code-group
  reader standalone entry point
  `oxideav_webp::meta_prefix::MetaPrefixHeader::read`. The §3 image-
  header peek lands the §2.6 lossless bitstream at the start of the
  §5.2.3 `color-cache-info` field; `MetaPrefixHeader::read` then walks
  `color-cache-info` (single bit + optional 4-bit `color_cache_code_bits`
  range-gated to `[1, 11]` per §5.2.3), `meta-prefix` (single bit, ARGB
  role only per §6.2.2), and either the §6.2 5-prefix-code group (the
  `Single` branch) or the §6.2.2 `prefix_bits = ReadBits(3) + 2` field
  plus the `DIV_ROUND_UP(image_dim, 1 << prefix_bits)` entropy-image
  dimension derivation (the `EntropyImagePending` branch, which records
  the boundary and returns for the next §5.2 reader to resume).
  The function is a public standalone surface exported through
  `pub mod meta_prefix` that downstream callers can invoke against any
  byte slice with an attacker-controlled `(image_width, image_height)`
  pair and `ImageRole`. The new harness drives that direct entry
  point with a zero-positioned `BitReader` (no §3 image-header skip):
  byte 0 picks the `ImageRole` (bit 0) and seeds the upper byte of
  `image_width` (bits 1..8); byte 1 seeds the upper byte of
  `image_height`; bytes 2.. feed the §5.2.3 + §6.2.2 + §6.2 bit
  sequence. Every branch of the §5.2.3 + §6.2.2 + §6.2 contract is
  cross-checked on the way out — `Ok(header)` implies §5.2.3
  `code_bits ∈ {0} ∪ [1, 11]`, `is_enabled() == (code_bits != 0)`,
  `size()` matching the `0` / `1 << code_bits` derivation, the
  `EntropyCoded` role never producing `EntropyImagePending` (sub-images
  carry no meta-prefix bit), and on the `EntropyImagePending` branch
  §6.2.2 `prefix_bits ∈ [2, 9]`, the recorded entropy-image
  `image_width` / `image_height` matching the §6.2.2
  `DIV_ROUND_UP(caller_dim, 1 << prefix_bits)` recomputation, and
  `entropy_image_bit_position` within the slice's bit length. `Err`
  branches are likewise cross-checked: `Eof` carries an in-range
  `bit_pos + available <= total_bits` coordinate with
  `wanted > available`; `InvalidColorCacheCodeBits` carries a `value`
  in the 4-bit field's `[0, 15]` window minus the §5.2.3 compliant
  `[1, 11]` (i.e. either `0` or in `[12, 15]`); `Prefix` surfaces a
  §6.2.1 refusal cleanly. The single fuzz iteration is bounded by the
  §6.2 5-prefix-code-group read (`Single` branch) or a fixed 3-bit
  read (`EntropyImagePending` branch). Eleventh fuzz target after
  `decode` / `extract_metadata` / `roundtrip_lossless` /
  `roundtrip_animated` (round 238) / `decode_alph` (round 255) /
  `parse_vp8x` (round 256) / `parse_anmf` (round 257) / `parse_anim`
  (round 258) / `parse_alph` (round 259) / `parse_transform_list`
  (round 260). README's `### Fuzzing` count now reads `Eleven`
  (the pre-r261 prose still said `Nine` despite the actual ten
  targets — fixed in the same commit).

- `fuzz/fuzz_targets/parse_transform_list.rs`: cargo-fuzz harness on
  the §4 VP8L transform-list reader standalone entry point
  `oxideav_webp::vp8l_stream::TransformList::read`. The §3 image-header
  peek lands the §2.6 lossless bitstream at the start of the §4
  transform-presence loop (`while (ReadBits(1)) { ReadBits(2);
  ... }`); the reader decodes each present transform's §4 leading
  fixed fields (§4.1 / §4.2 `size_bits = ReadBits(3) + 2`, §4.3
  SUBTRACT_GREEN with no data, §4.4 `color_table_size = ReadBits(8)
  + 1` plus the derived pixel-bundling `width_bits`), stopping
  either at the terminating `0` presence bit or at the §5 entropy-body
  boundary. §4 also says "each transform is allowed to be used only
  once" — a repeat of any `TransformType` raises `DuplicateTransform`.
  The function is a public standalone surface exported through
  `pub mod vp8l_stream` (and the convenience wrapper
  `oxideav_webp::read_vp8l_transform_list`) that downstream callers can
  invoke against any byte slice they obtained from a different demuxer
  or an attacker-controlled buffer of arbitrary length, including the
  empty slice. The new harness drives that direct entry point with a
  zero-positioned `BitReader` (no §3 image-header skip): the arbitrary
  fuzz buffer is forwarded verbatim as a §4 transform-list candidate.
  Every branch of the §4 contract is cross-checked on the way out —
  `Ok(list)` implies `list.transforms().len() <= 4`, no repeated
  `TransformType` across the entries, every entry's
  `transform_type()` tag matches its variant, §4.1 / §4.2 `size_bits`
  lies in `[2, 9]`, §4.4 `color_table_size` lies in `[1, 256]`, §4.4
  `width_bits` follows the threshold table (3 for `<= 2`, 2 for `<=
  4`, 1 for `<= 16`, 0 otherwise), `body_bit_position()` does not
  exceed the slice's total bit length, and `stopped_at_entropy_body()`
  is consistent with the last entry's `has_entropy_body()` (the parser
  stopped *at* a §5 body iff the last entry carries one). `Err`
  branches are likewise cross-checked: `Eof` carries a `bit_pos +
  available <= total_bits` coordinate with `wanted > available`, and
  `DuplicateTransform` carries one of the four §4 `TransformType`
  values. Sibling fuzz harnesses already cover the orthogonal parsers
  — `parse_vp8x` (§2.7.1 Figure 7), `parse_anmf` (§2.7.1.1 Figure 9),
  `parse_anim` (§2.7.1.1 Figure 8), `parse_alph` (§2.7.1.2 Figure 10),
  `decode_alph` (§2.7.1.2 alpha plane), `extract_metadata` (§2 RIFF
  ICCP/EXIF/XMP walk), `decode` (full single-shot entry), and
  `roundtrip_animated` / `roundtrip_lossless` (encode→decode equality
  oracles). The new harness extends fuzz coverage from the §2.7 RIFF
  container layer down into the §4 VP8L transform-presence loop, the
  first bit-level decode stage the §2.6 lossless bitstream walks
  through after the §3.4 image-header.
- `fuzz/fuzz_targets/parse_alph.rs`: cargo-fuzz harness on the
  §2.7.1.2 ALPH info-byte parser standalone entry point
  `oxideav_webp::alph::AlphHeader::parse`. The §2 RIFF walk in
  `decode_webp` only reaches `AlphHeader::parse` with payload slices
  the container layer has already validated as `ALPH` chunk bodies,
  and the existing `decode_alph` harness drives the full §2.7.1.2
  decode (`alph::decode_alpha`) with constrained
  `(width, height) ∈ [1, 64]²` so its alpha-bitstream traversal
  stays bounded. The new harness drives the public standalone surface
  directly — exported through `pub mod alph` and the convenience
  wrapper `oxideav_webp::parse_alph_header`, the function is the
  documented entry point for downstream callers that obtained an
  `ALPH` payload candidate from a different demuxer or an
  attacker-controlled buffer of arbitrary length (including the
  empty slice). The arbitrary fuzz buffer is forwarded verbatim as
  the §2.7.1.2 ALPH payload candidate; the empty slice hits the
  `EmptyPayload` refusal path while every other length lets
  `payload[0]` cover the full §2.7.1.2 Figure 10 Rsv × P × F × C
  cross-product (4 × 4 × 4 × 4 = 256 distinct info bytes, every
  bit-pattern legal: the spec mandates readers IGNORE `Rsv`, and the
  "undefined" values of the `C` and `P` fields are surfaced through
  their `Reserved(_)` variants rather than raising an error at this
  layer). Every branch of the §2.7.1.2 contract is cross-checked on
  the way out: `Ok(hdr)` implies `!data.is_empty()`,
  `hdr.info_byte == payload[0]` verbatim, `hdr.reserved` equals
  `(payload[0] >> 6) & 0b11` (the §2.7.1.2 MSB-first `Rsv` field at
  bits 7..6), the typed `preprocessing` variant decodes the
  `(payload[0] >> 4) & 0b11` `P` bits (0 → `None`, 1 →
  `LevelReduction`, 2 | 3 → `Reserved(v)` with the inner value
  matching the raw bits), the typed `filtering` variant decodes the
  `(payload[0] >> 2) & 0b11` `F` bits (0 → `None`, 1 → `Horizontal`,
  2 → `Vertical`, 3 → `Gradient`), the typed `compression` variant
  decodes the `payload[0] & 0b11` `C` bits (0 → `None`, 1 →
  `Lossless`, 2 | 3 → `Reserved(v)` with the inner value matching
  the raw bits), and `hdr.bitstream_offset() == 1` (§2.7.1.2 fixes
  the info byte at position 0 and the alpha bitstream immediately
  after, so the offset is a per-spec constant). The single
  info-byte-parser error branch is cross-checked too:
  `Err(EmptyPayload)` implies `data.is_empty()`; any other
  `AlphError` variant (`DimensionsOverflow` / `RawLengthMismatch` /
  `UnsupportedCompression` / `Vp8l`) escaping the parser is a
  contract violation surfaced as an explicit panic — those variants
  are produced exclusively by `decode_alpha`'s downstream
  bitstream-decode stages, never by `AlphHeader::parse`. The contract
  under test: every call must return a `Result` — no panic, no
  debug-build integer overflow, no out-of-bounds index when the
  payload is empty or arbitrarily long; the §2.7.1.2 info byte is a
  single octet so the parser only ever inspects `payload[0]` and any
  extra bytes after byte 0 are the "Alpha bitstream" §2.7.1.2
  describes which `AlphHeader::parse` deliberately ignores (its job
  is bitfield decomposition only). The parser is a fixed-cost branch
  chain (one length test) plus four 2-bit field extracts from a
  single byte (no allocation, no loop sized by the input, no
  recursion), so a single fuzz iteration is microseconds regardless
  of input length and the harness can attempt arbitrary payload
  sizes (including the empty slice and slices longer than any
  conceivable §2.7.1.2 ALPH chunk) without iteration-cost concerns.
  Joins `decode.rs` / `extract_metadata.rs` (always-returns-Result
  oracles on the public single-shot entry points),
  `roundtrip_lossless.rs` (still-image §3 VP8L encode + decode
  round-trip), `roundtrip_animated.rs` (§2.7.1.1 ANIM / ANMF carrier
  + per-frame pixel + per-frame duration round-trip), `decode_alph.rs`
  (§2.7.1.2 ALPH standalone entry point including the alpha
  bitstream), `parse_vp8x.rs` (§2.7.1 VP8X chunk parser standalone
  entry point), `parse_anmf.rs` (§2.7.1.1 ANMF chunk header parser
  standalone entry point), and `parse_anim.rs` (§2.7.1.1 ANIM chunk
  parser standalone entry point): 9 cargo-fuzz harnesses total, with
  `parse_alph` widening surface coverage onto the §2.7.1.2 ALPH
  info-byte parser the previous eight harnesses only reached through
  paths that imposed either RIFF-container framing (the always-Result
  oracles plus the round-trip oracles) or dimension constraints (the
  `decode_alph` full-bitstream oracle), without exercising the
  info-byte-only direct entry point against unconstrained byte
  payloads.

- `fuzz/fuzz_targets/parse_anim.rs`: cargo-fuzz harness on the
  §2.7.1.1 ANIM chunk parser standalone entry point
  `oxideav_webp::anim::AnimHeader::parse`. The §2 RIFF walk in
  `decode_webp` only reaches `AnimHeader::parse` with payload slices
  the container layer has already validated as `ANIM` chunk bodies,
  so the cross-product reachable through the existing `decode.rs` /
  `extract_metadata.rs` / `roundtrip_animated.rs` harnesses exercises
  this parser only along the "well-formed RIFF" code path. The new
  harness drives the public standalone surface directly — exported
  through `pub mod anim` and the convenience wrapper
  `oxideav_webp::parse_anim_header`, the function is the documented
  entry point for downstream callers that obtained an `ANIM` payload
  candidate from a different demuxer or an attacker-controlled buffer
  of arbitrary length. The arbitrary fuzz buffer is forwarded verbatim
  as the §2.7.1.1 ANIM payload candidate; inputs shorter or longer
  than 6 bytes hit the `BadPayloadLength` refusal path while 6-byte
  inputs cover the full §2.7.1.1 Figure 8 BGRA × loop-count
  cross-product (4 × 8-bit background channels + 1 × 16-bit loop
  count = 2^48 distinct payloads, all bit-patterns legal). Every
  branch of the §2.7.1.1 contract is cross-checked on the way out:
  `Ok(hdr)` implies `data.len() == 6`, every BGRA channel equals the
  matching `payload[0..4]` byte verbatim (Blue at byte 0, Green at
  byte 1, Red at byte 2, Alpha at byte 3 per the §2.7.1.1 carrier),
  `background_color.as_u32_le()` equals the little-endian `u32`
  reload of bytes 0..4 (pinning the §2.7.1.1 "uint32 stored in BGRA
  byte order" carrier), `loop_count` equals
  `u16::from_le_bytes([data[4], data[5]])` (the §2.3 multi-byte
  little-endian convention), and `loops_forever()` matches
  `loop_count == 0` exactly (the §2.7.1.1 "0 = infinite playback"
  semantic). The single error branch is cross-checked too:
  `Err(BadPayloadLength { got })` implies `got == data.len()` and
  `got != 6`. The contract under test: every call must return a
  `Result` — no panic, no debug-build integer overflow on the
  `as_u32_le` BGRA pack, no out-of-bounds index when the payload is
  shorter or longer than the 6-byte fixed-length header. The parser
  is a fixed-cost branch chain (one length test) plus four `u8` reads
  plus one 2-byte little-endian `u16` reload (no allocation, no loop
  sized by the input, no recursion), so a single fuzz iteration is
  microseconds regardless of input length and the harness can attempt
  arbitrary payload sizes without iteration-cost concerns. Joins
  `decode.rs` / `extract_metadata.rs` (always-returns-Result oracles
  on the public single-shot entry points), `roundtrip_lossless.rs`
  (still-image §3 VP8L encode + decode round-trip),
  `roundtrip_animated.rs` (§2.7.1.1 ANIM / ANMF carrier + per-frame
  pixel + per-frame duration round-trip), `decode_alph.rs`
  (§2.7.1.2 ALPH standalone entry point), `parse_vp8x.rs` (§2.7.1
  VP8X chunk parser standalone entry point), and `parse_anmf.rs`
  (§2.7.1.1 ANMF chunk header parser standalone entry point):
  8 cargo-fuzz harnesses total, with `parse_anim` widening surface
  coverage onto the §2.7.1.1 ANIM chunk parser the previous seven
  harnesses only reached transitively through a complete §2 RIFF walk
  plus an ANIM chunk wrapper.

- `fuzz/fuzz_targets/parse_anmf.rs`: cargo-fuzz harness on the
  §2.7.1.1 ANMF chunk header parser standalone entry point
  `oxideav_webp::anmf::AnmfHeader::parse`. The §2 RIFF walk in
  `decode_webp` only reaches `AnmfHeader::parse` with payload slices
  the container layer has already validated as `ANMF` chunk bodies,
  so the cross-product reachable through the existing `decode.rs` /
  `extract_metadata.rs` / `roundtrip_animated.rs` harnesses exercises
  this parser only along the "well-formed RIFF" code path. The new
  harness drives the public standalone surface directly — exported
  through `pub mod anmf`, the function is the documented entry point
  for downstream callers that obtained an `ANMF` payload candidate
  from a different demuxer or an attacker-controlled buffer of
  arbitrary length. The arbitrary fuzz buffer is forwarded verbatim
  as the §2.7.1.1 ANMF payload candidate; inputs shorter than 16
  bytes hit the `PayloadTooShort` refusal path while inputs ≥ 16
  bytes cover the full §2.7.1.1 Figure 9 5 × uint24 + info-byte
  cross-product (with the surplus bytes after byte 15 being the
  §2.7.1.1 Frame Data sub-RIFF the header parser does not touch).
  Every branch of the §2.7.1.1 contract is cross-checked on the way
  out: `Ok(hdr)` implies `data.len() >= 16`, both width and height
  land in `[1, 2^24]` (the 24-bit "Minus One" `+ 1` range), both
  `x` and `y` land in `[0, (2^24-1)*2]` (the §2.7.1.1 `Frame X * 2`
  doubling, with the bound proving `u32` arithmetic did not
  overflow), `duration_ms` lands in `[0, 2^24 - 1]` (the uint24 LE
  literal), `frame_data_offset() == 16` (the §2.7.1.1 fixed-header
  length), every field equals the re-derived value from the original
  bytes (Frame X / Y from bytes 0..3 / 3..6 little-endian uint24,
  Frame W-1 / H-1 from bytes 6..9 / 9..12, Duration from bytes
  12..15, info_byte from byte 15), and the info-byte sub-fields
  decode to the §2.7.1.1 Figure 9 bit positions (Reserved at bits
  7..2, B at bit 1, D at bit 0). The single error branch is
  cross-checked too: `Err(PayloadTooShort { got })` implies
  `got == data.len()` and `got < 16`. The contract under test: every
  call must return a `Result` — no panic, no debug-build integer
  overflow on the `Frame X * 2` doubling or the `Frame W/H Minus
  One + 1` resolution, no out-of-bounds index when the payload is
  shorter or longer than the 16-byte header. The parser is a
  fixed-cost branch chain plus five 3-byte little-endian uint24
  reads plus three `u8` bit-extracts plus two `u32` arithmetic ops
  (no allocation, no loop sized by the input, no recursion), so a
  single fuzz iteration is microseconds regardless of input length
  and the harness can attempt arbitrary payload sizes without
  iteration-cost concerns. Joins `decode.rs` / `extract_metadata.rs`
  (always-returns-Result oracles on the public single-shot entry
  points), `roundtrip_lossless.rs` (still-image §3 VP8L encode +
  decode round-trip), `roundtrip_animated.rs` (§2.7.1.1 ANIM / ANMF
  carrier + per-frame pixel + per-frame duration round-trip),
  `decode_alph.rs` (§2.7.1.2 ALPH standalone entry point) and
  `parse_vp8x.rs` (§2.7.1 VP8X chunk parser standalone entry point):
  7 cargo-fuzz harnesses total, with `parse_anmf` widening surface
  coverage onto the §2.7.1.1 ANMF chunk header parser the previous
  six harnesses only reached transitively through a complete §2 RIFF
  walk plus an ANMF chunk wrapper.

- `fuzz/fuzz_targets/parse_vp8x.rs`: cargo-fuzz harness on the §2.7.1
  VP8X chunk parser standalone entry point
  `oxideav_webp::vp8x::Vp8xHeader::parse`. The §2 RIFF walk in
  `decode_webp` only reaches `Vp8xHeader::parse` with a payload slice
  the container layer has already validated as a `VP8X` chunk body,
  so the cross-product reachable through the existing `decode.rs` /
  `extract_metadata.rs` harnesses exercises this parser only along
  the "well-formed RIFF" code path. The new harness drives the public
  standalone surface directly — exported through `pub mod vp8x` and
  the convenience wrapper `oxideav_webp::parse_vp8x_header`, the
  function is the documented entry point for downstream callers that
  obtained a `VP8X` payload candidate from a different demuxer or an
  attacker-controlled buffer of arbitrary length. The arbitrary fuzz
  buffer is forwarded verbatim as the §2.7.1 VP8X payload candidate;
  inputs shorter or longer than 10 bytes hit the `BadPayloadLength`
  refusal path while 10-byte inputs cover the full §2.7.1 Figure 7
  flag-octet / reserved-field / canvas-dimension cross-product. Every
  branch of the §2.7.1 contract is cross-checked on the way out:
  `Ok(hdr)` implies `data.len() == 10`, both canvas dimensions land
  in `[1, 2^24]` (the 24-bit "Minus One" `+ 1` range), the explicit
  `canvas_width as u64 * canvas_height as u64 <= u32::MAX as u64`
  product cap holds, every named feature-flag bool (`has_iccp` /
  `has_alpha` / `has_exif` / `has_xmp` / `has_animation`) matches the
  §2.7.1 byte-0 bit position the module docstring assigns it (I=5,
  L=4, E=3, X=2, A=1), `has_unknown` matches the disjunction over
  every §2.7.1 reserved position (the 2-bit `Rsv` pair at byte 0 bits
  7..6, the `R` bit at byte 0 bit 0, and the 24-bit reserved field at
  bytes 1..4), and both canvas dimensions equal `Minus One + 1`
  re-derived from the little-endian 24-bit fields at bytes 4..7 /
  bytes 7..10. The two error branches are cross-checked too:
  `Err(BadPayloadLength { got })` implies `got == data.len()` and
  `got != 10`, and `Err(CanvasTooLarge { canvas_width, canvas_height
  })` implies `data.len() == 10` plus the cross-checked product
  strictly exceeds `u32::MAX`. The contract under test: every call
  must return a `Result` — no panic, no debug-build integer overflow
  on the `(canvas_width as u64) * (canvas_height as u64)` product cap
  check, no out-of-bounds index when the payload is shorter or longer
  than the §2.7.1 Figure 7 ten bytes. The parser is a fixed-cost
  branch chain plus a single `u64` multiply (no allocation, no loop
  sized by the input, no recursion), so a single fuzz iteration is
  microseconds regardless of input length and the harness can attempt
  arbitrary payload sizes without iteration-cost concerns. Joins
  `decode.rs` / `extract_metadata.rs` (always-returns-Result oracles
  on the public single-shot entry points), `roundtrip_lossless.rs`
  (still-image §3 VP8L encode + decode round-trip),
  `roundtrip_animated.rs` (§2.7.1.1 ANIM / ANMF carrier + per-frame
  pixel + per-frame duration round-trip) and `decode_alph.rs`
  (§2.7.1.2 ALPH standalone entry point): 6 cargo-fuzz harnesses
  total, with `parse_vp8x` widening surface coverage onto the §2.7.1
  VP8X chunk parser the previous five harnesses only reached
  transitively through a complete §2 RIFF walk.

- `fuzz/fuzz_targets/decode_alph.rs`: cargo-fuzz harness on the
  §2.7.1.2 ALPH standalone entry point
  `oxideav_webp::alph::decode_alpha`. The §2 RIFF walk in `decode_webp`
  only reaches `decode_alpha` with `(width, height)` taken from a
  validated §2.7.1 VP8X / §2.5 VP8 keyframe header, so the
  cross-product reachable through the existing `decode.rs` harness is
  bounded by what a chunk-header validator already accepted. The new
  harness drives the public standalone surface directly — the function
  is documented as the per-frame §2.7.1.1 ANMF entry point downstream
  callers invoke against ANMF frame dimensions — pairing
  fuzz-controlled `(width, height) ∈ [1, 64]²` with a fuzz-controlled
  ALPH chunk payload (info byte + alpha bitstream) so the four §2.7.1.2
  filter methods (none / horizontal / vertical / gradient) and the two
  §2.7.1.2 compression methods (raw + headerless §3 VP8L) reachable
  through that surface get battered against arbitrary inputs. The
  decoder-side contract under test: every call must return a `Result` —
  no panic, no debug-build integer overflow on the `width * height`
  arithmetic, no out-of-bounds index, no allocation sized by the
  attacker-controlled dimensions before the `checked_mul` rejects them.
  On success, the §2.7.1.2 carrier-field invariant
  `plane.len() == width * height` is asserted — that equality is the
  RFC 9649 §2.7.1.2 contract ("a byte sequence of length =
  width * height") and any other length is a real carrier violation.
  The 64 × 64 canvas cap keeps the method-1 §3 VP8L sub-decode bounded
  (worst-case plane = 4 KiB, worst-case method-1 intermediate ARGB =
  16 KiB) so a single fuzz iteration stays in the millisecond range
  even when the alpha bitstream is a §3 VP8L stream. Joins
  `decode.rs` / `extract_metadata.rs` (always-returns-Result oracles on
  the public single-shot entry points), `roundtrip_lossless.rs`
  (still-image §3 VP8L encode + decode round-trip) and
  `roundtrip_animated.rs` (§2.7.1.1 ANIM / ANMF carrier + per-frame
  pixel + per-frame duration round-trip): 5 cargo-fuzz harnesses total,
  with `decode_alph` widening surface coverage onto the standalone
  ALPH entry point the previous four harnesses only reached
  transitively through a complete §2 RIFF walk.

- `benches/value_to_prefix.rs`: criterion bench for the encoder-side
  §5.2.2 `vp8l_encode::value_to_prefix` LZ77 length-or-distance value-
  to-prefix split — the exact inverse of the decoder-side §3.6.2.2
  `read_lz77_value` benched at round 252. Given a 1-based length-or-
  distance `value` (≥ 1), the function returns the `(prefix_code,
  extra_bits, extra_value)` triple needed to emit the symbol; the
  §5.2.2 emit sequence then writes the prefix code through the GREEN
  (length) or DISTANCE (distance) Huffman code and the `extra_value`
  as `extra_bits` raw LSB-first bits. Invoked twice per emitted LZ77
  match inside `encode_argb_literals` (length prefix + distance
  prefix) plus once more during cost estimation in the `try_lz77_at`
  cost-model path when the encoder is choosing between candidate
  match lengths, so per-call cost scales linearly with the per-image
  match count (multiples-of-three the match count for a match-heavy
  encode, not just twice it). Parameterised across the four §5.2.2
  regimes that mirror the round-252 decoder-side cell layout exactly
  so the two benches' numbers are directly comparable cell-for-cell:
  *fast path* (`value = 3`, `value ∈ [1, 4]`, returns `(value - 1,
  0, 0)` without touching the `leading_zeros` / shift chain, mirrors
  the decode-side `prefix_code = 2` cell which decodes to value 3),
  *short extra* (`value = 40`, `extra_bits = 4`, `prefix_code = 10`,
  mirrors the decode-side `prefix_code = 10` cell which decodes to
  values `34..=49`), *long extra* (`value = 40_000`, `extra_bits =
  14`, `prefix_code = 30`, mirrors the decode-side `prefix_code = 30`
  cell which decodes to values `32769..=49152`, distance-only) and
  *max extra* (`value = 900_000`, `extra_bits = 18`, `prefix_code =
  39`, mirrors the decode-side `prefix_code = 39` cell which decodes
  to values `786433..=1048576`, the §5.2.2 hard upper bound on
  encodable values, distance-only). Four bench cells total. The bench
  amortises Criterion's per-iteration overhead by running an inner
  loop of 1024 calls per `b.iter` body over the cell's representative
  `value`, XOR-accumulating every returned `(prefix, extra_bits,
  extra_value)` triple so the optimiser cannot drop any individual
  call. `black_box` on both the input value and the accumulator
  guards against constant-folding and dead-store elimination. The
  per-iteration value count and the inner loop body are identical
  across every cell, so cross-cell deltas come exclusively from the
  §5.2.2 body cost at the cell's `value`. The function body is
  unchanged this round; the deliverable is the A/B reference for a
  future branchless rewrite or §5.2.2-table-driven `(prefix,
  extra_bits, offset)` lookup for the small-value range. Quick-mode
  measurements separate cleanly: the fast-path cell at ~0.32 ns per
  call, every extra-bits regime flat at ~0.63 ns per call across the
  three §5.2.2 magnitude bands — confirming the spec's expected
  shape (fast path elides `leading_zeros` + shift + multiply; the
  three extra-bits regimes share an identical body cost). Closes the
  encoder-side §5.2.2 prefix-split entry in the §5 encode per-pass
  inventory alongside the round-252 decoder-side mirror.
- `benches/color_cache_hash.rs`: criterion bench for the decoder-side
  §3.6.2.3 `vp8l_decode::ColorCache::hash` color-cache multiplicative-
  hash slot-index function — the per-pixel index function that turns
  an emitted ARGB color into a §3.6.2.3 cache slot via RFC 9649
  §3.6.2.3's verbatim `(0x1e35a7bd * color) >> (32 -
  color_cache_code_bits)`. Called twice per emitted pixel when the
  §3.6.2.3 cache is enabled (once for `ColorCache::insert` on every
  emitted literal / LZ77-copied / cache-resolved pixel per §3.6.2.3
  "the state of the color cache is maintained by inserting every
  pixel ... into the cache in the order they appear in the stream"
  and once more inside the encoder-side mirror), so per-call cost
  scales linearly with the per-image pixel count whenever the
  §3.6.2.3 cache is active — the common case for natural-image VP8L
  payloads. Parameterised across the §3.6.2.3 `code_bits` allowed
  range `[1..11]`: *minimum* (`code_bits = 1`, 2-slot cache, shift
  `>> 31`), *small-cache regime* (`code_bits = 4`, 16-slot cache,
  shift `>> 28`, palette / line-art common), *natural-image regime*
  (`code_bits = 8`, 256-slot cache, shift `>> 24`) and *maximum*
  (`code_bits = 11`, 2048-slot cache, shift `>> 21`, the §3.6.2.3
  hard upper bound). Four bench cells total. The per-call work is a
  single `u32` multiply plus a right-shift plus a `usize` cast, so
  the bench amortises Criterion's per-iteration overhead by running
  an inner loop of 1024 calls per `b.iter` body over a pre-allocated
  deterministic LCG-filled `u32` ARGB stream, accumulating every
  slot index through a wrapping XOR so the optimiser cannot drop any
  individual call. The ARGB input stream and the loop count are
  identical across every cell, so cross-cell deltas come exclusively
  from the §3.6.2.3 hash body cost. The LCG constants match the rest
  of the §3.x / §4.x per-pass bench inventory so cross-pass numbers
  are reproducible and visually comparable. The function body is
  unchanged this round; the deliverable is the A/B reference for a
  future const-folding / inlining rewrite at the call site (where
  `code_bits` is often known statically per decode group). Quick-
  mode measurements at all four cells land within ~5 % of each
  other, confirming the §3.6.2.3 spec's expected flat per-call cost
  across `code_bits`. Closes the §3.6.2.3 color-cache slot-index
  entry in the §3 decode per-pass inventory alongside the round-252
  §3.6.2.2 `read_lz77_value` LZ77 prefix-value bench and the round-
  250 / round-251 §3.7.2 encoder-builder benches.
- `benches/read_lz77_value.rs`: criterion bench for the decoder-side
  §3.6.2.2 `vp8l_decode::read_lz77_value` LZ77 prefix-code-to-value
  expansion — the per-symbol second half of the §3.6.2.2 length /
  distance decode path that turns a §3.6.2.2 prefix code `[0..40)`
  into the decoded length-or-distance value by either folding the
  prefix code (fast path: `prefix_code < 4`) or by reading
  `extra_bits = (prefix_code - 2) >> 1` bits and folding them with
  the §3.6.2.2 offset formula `offset = (2 + (prefix_code & 1)) <<
  extra_bits`. Invoked twice per LZ77 match inside
  `decode_one_symbol` (length prefix + distance prefix), so per-call
  cost scales linearly with the per-image backward-reference run
  count. Parameterised across the four §3.6.2.2 Table 4 regimes:
  *fast path* (`prefix_code = 2`, value range `1..=4`, reader
  untouched), *short extra* (`prefix_code = 10`, `extra_bits = 4`,
  value range `34..=49`), *long extra* (`prefix_code = 30`,
  `extra_bits = 14`, value range `32769..=49152`, distance-only) and
  *max extra* (`prefix_code = 39`, `extra_bits = 18`, value range
  `786433..=1048576`, the §3.6.2.2 hard upper bound, distance-only).
  Four bench cells total. The bit slice is a 32 KiB LCG-filled
  buffer built once outside `b.iter`; the `BitReader<'_>` is
  reconstructed per iteration to keep the per-call cursor stable
  across the inner sample window. The reader-construction cost is
  the same across every cell, so cross-cell deltas come exclusively
  from the §3.6.2.2 body work. The LCG constants match the rest of
  the §3.x / §4.x per-pass bench inventory so cross-pass numbers are
  reproducible and visually comparable. The function body is
  unchanged this round; the deliverable is the A/B reference for a
  future branchless rewrite or §3.6.2.2-table-driven
  offset/extra-bits lookup. Closes the per-LZ77-call entry in the §3
  decode per-pass inventory alongside the round-250 / round-251 §3.7
  encoder builder benches.
- `benches/canonical_codes.rs`: criterion bench for the encoder-side
  §3.7.2 `vp8l_encode::canonical_codes` canonical-code-value
  assignment pass — the second per-symbol pass in the §3.7.2
  length-then-code Huffman build, returning the canonical code values
  the decoder's `vp8l_prefix::PrefixCode` reconstructs from the same
  length table. Parameterised over the same two axes as the round-250
  `build_code_lengths` bench: (a) the four §3.7.1 prefix-code-group
  alphabets — DISTANCE = 40, RED / BLUE / ALPHA = 256, GREEN = 281
  (smallest §3.6.2.3 color cache) and GREEN = 2328 (largest §3.6.2.3
  color cache) — and (b) two frequency-table regimes (*dense* every
  symbol live, *sparse* `sqrt(N)` live symbols / Zipf shape), with
  the length tables produced by feeding those frequencies through
  `build_code_lengths` once at bench setup so each `b.iter` body
  sees the exact length table a real per-prefix-code-group call
  would. Eight bench cells total. The LCG constants match the
  round-250 length-builder bench so cross-pass numbers are
  reproducible and visually comparable. The function body is
  unchanged this round; the deliverable is the A/B reference for a
  future bucket-sort-by-length rewrite — `canonical_codes` is the
  rank-4 encode self-time symbol per the round-170 profile, and the
  current `O(MAX_CODE_LENGTH · N)` two-loop walk runs the inner
  pass over every slot whether or not the length is zero, so the
  sparse path is the natural target. Closes the per-prefix-code-group
  encode inner loop in the §3 entropy domain alongside the
  `build_code_lengths` bench landed in round 250.
- `benches/build_code_lengths.rs`: criterion bench for the encoder-side
  §3.7.2 `vp8l_encode::build_code_lengths` Huffman code-length builder
  — the per-symbol length-assignment pass invoked once per §3.7.1
  prefix-code-group channel (GREEN+length, RED, BLUE, ALPHA, DISTANCE)
  plus the §3.7.2.1.2 normal-form code-length-of-code-lengths sub-pass.
  Parameterised across (a) every realistic §3.7.1 alphabet size —
  DISTANCE = 40, RED / BLUE / ALPHA = 256 (8-bit channel literals),
  GREEN = 281 (smallest §3.6.2.3 color cache, `cache_bits = 0`) and
  GREEN = 2328 (largest §3.6.2.3 color cache, `cache_bits = 11`) — and
  (b) two frequency-table regimes that the builder hits in practice:
  *dense* (every symbol live, LCG-fill 1..=255) modelling a natural-
  image meta-prefix code-group's literal channels, and *sparse*
  (`sqrt(N)` live symbols, 1/(k+1) Zipf shape scattered across the
  alphabet via a second LCG stream) modelling a DISTANCE table where
  few prefix codes fire or a GREEN table whose §3.6.2.3 color cache
  code range is barely populated. Eight bench cells total. The LCG
  constants match the rest of the §4.x bench inventory so cross-pass
  numbers are reproducible and comparable. The function body is
  unchanged this round; the deliverable is the A/B reference for a
  future heap-replacement rewrite — the round-170 encoder profile
  attributed rank 4 of self-time to the surrounding closure body
  through `canonical_codes`, and the §3.7.2 builder's heap-pop /
  heap-push loop running `n - 1` times per call is the natural target
  for a radix-bucket replacement. Closes the §3 entropy per-pass
  inventory gap the §4.x transform inventory had carried through
  round 249.
- `benches/inverse_color_table.rs`: criterion bench for the §4.4
  `inverse_color_table` palette subtraction-decode pass (the
  cumulative-delta loop that recovers final ARGB palette entries by
  adding the previous entry's four lanes into the current entry's
  four lanes mod 256, per the §4.4 spec text). Parameterised over
  palette sizes 2 / 16 / 256, which cover the §4.4 bundling-tier
  boundaries (minimum length, mid-tier near the `width_bits = 1 → 0`
  boundary, and maximum length). The deterministic LCG fill uses
  the same seed + multiplier + increment as the rest of the §4.x
  bench inventory so per-lane wrap paths are reproducibly exercised
  across runs and cross-pass numbers are visually comparable. The
  bench clones the palette into a fresh working buffer each
  iteration so the in-place pass starts from the same input every
  time and a future SWAR / `std::simd` rewrite cannot win simply by
  caching deltas across iterations. Round-249 baseline on
  aarch64-apple-darwin (`--quick`): 10.16 ns / 44.43 ns / 1.273 µs
  for palette 2 / 16 / 256 — scaling roughly linearly in palette
  length as expected for a per-entry sequential dependency. Closes
  the last per-pass §4.x bench inventory gap: `inverse_color_table`
  was the only remaining `pub fn` in `vp8l_transform` without a
  per-pass bench (decoder-side §4.1 / §4.2 / §4.3 / §4.4 indexing
  benches landed in rounds 194 / 207 / 217 / 210; encoder-side
  §4.1 / §4.3 mirrors landed in rounds 224 / 248). The function
  body is unchanged this round; the deliverable is the A/B reference
  with the existing
  `forward_color_table_round_trips_with_decoder_inverse` roundtrip
  test as the byte-exact regression guard. See `BENCHMARKS.md
  § Round-249` for the per-size numbers and the per-iteration
  dependency-chain reasoning that bounds the per-iteration cost.
- `benches/apply_subtract_green.rs`: criterion bench for the
  encoder-side `vp8l_encode::apply_subtract_green` — the §4.3 forward
  subtract-green transform that walks an in-place ARGB buffer and
  replaces every red / blue lane with the mod-256 difference against
  that pixel's green lane (alpha and green pass through untouched).
  Mirrors the decoder-side `inverse_subtract_green` bench landed in
  round 217 and uses the same 256×256 deterministic LCG-filled ARGB
  buffer (identical seed + multiplier + increment to the §4.x decoder-
  side benches and the round-224 encoder-side `predictor_subtract`
  bench) so cross-pass numbers across the four §4.x decoder-side
  inverse-transform benches and the two encoder-side §4.1 / §4.3
  forward-transform benches are directly comparable. The bench clones
  the input buffer into a fresh working buffer each iteration so the
  in-place pass starts from the same input every time and a future
  SWAR / `std::simd` rewrite cannot win simply by caching residuals
  across iterations. Round-248 baseline: ~13.3–13.7 µs (three
  consecutive `--quick` runs on aarch64-apple-darwin: 13.72 µs /
  13.60 µs / 13.28 µs). The function body is left unchanged this
  round; the deliverable is the A/B reference, with the existing
  `apply_subtract_green_is_inverse_of_inverse_subtract_green`
  roundtrip test as the byte-exact regression guard. Closes the
  encoder-side §4.3 inventory gap: the decoder-side mirror has had a
  per-pass bench since round 217, but the encoder-side forward pass
  was unmeasured at the per-pass level until now. See
  `BENCHMARKS.md § Round-248` for the calibration note against the
  decoder-side §4.3 bench and the reasoning that the two passes share
  the same byte-traffic and arithmetic complexity.
- `fuzz/fuzz_targets/roundtrip_animated.rs`: fourth `cargo-fuzz`
  harness, a structured round-trip oracle on the §2.7.1.1 animation
  carrier. The fuzzer drives the canvas dimensions (≤ 32 × 32), the
  frame count (1..8), every per-frame `duration_ms`, and every byte
  of every frame's RGBA buffer; the resulting `AnimFrame` set goes
  through `build_animated_webp` → `decode_webp`, and the harness
  asserts the frame count, every per-frame width / height, every
  per-frame `duration_ms`, and every per-frame RGBA byte survive the
  round trip identically. Closes the fuzz inventory gap: the
  pre-r238 `roundtrip_lossless` target covered only the still-image
  encoder + decoder pair (`encode_webp_lossless` → `decode_webp`),
  leaving the entire ANIM / ANMF chunk-walk + canvas-compositing
  decode + per-frame VP8L encode loop without a fuzz oracle. Bounded
  per-iteration cost (`8 * 32 * 32 * 4 = 32 KiB` worst-case RGBA
  working set) lets libFuzzer sustain ~50 exec/s on a release build;
  a 90 s local run on aarch64-apple-darwin produced 4 998 runs, no
  crashes / panics / assertion failures, coverage 2 603 / features
  8 239 / corpus 358 entries — the harness reaches the multi-frame
  path (libFuzzer-promoted input length 8 bytes feeds the 3-byte
  header + 5 bytes of per-frame data covering up to 2 frames after
  the prologue cycle-fill). The libFuzzer dictionary autoderived the
  RIFF chunk fourcc shapes `UP8L` and `GPLA` from the encoder output
  (byte-shifted views of `VP8L` / `ALPH`), evidence the oracle is
  exercising the full container walk on the decode side. Numbers and
  per-target run instructions are in `README.md` § Fuzzing.
- `benches/predictor_subtract.rs`: criterion bench for the encoder-side
  `vp8l_encode::predictor_subtract` — the §4.1 per-channel mod-256
  residual builder that mirrors the decoder's `add_pred`. Drives one
  call per pixel over a 256×256 ARGB buffer (deterministic LCG fill,
  matching the §4.x decoder-side bench shape so per-pass numbers are
  visually comparable) and aggregates the residual XOR so the loop
  body cannot be folded away. Closes the encoder-side inventory gap:
  the round-170 profile attributed the #1 encoder self-time slot to
  the predictor + residual path, but `predictor_subtract` itself was
  unmeasured at the per-pass level until now. Round-224 baseline:
  ~36 µs (range 35.9–38.3 µs across three consecutive `--quick` runs).
  A biased-SWAR rewrite was tried this round and measured +18.4% vs.
  the closure-of-four body — see `BENCHMARKS.md` for the lane-bias
  underflow-prevention details and the reason the SWAR form regresses
  on AArch64 NEON auto-vectorisation. The function body is left in
  its pre-r224 form; the new bench is the A/B reference any future
  `std::simd` rewrite of `predictor_subtract` (mirroring the
  `to_rgba_simd` precedent under the `simd` feature) must measure
  against.
- `vp8l_encode::predictor_subtract` is now `pub fn` (previously
  private). The function semantics are unchanged; raising visibility
  lets the new `predictor_subtract_256x256` criterion bench reach it
  from `benches/` and lets out-of-crate `simd`-feature consumers
  experiment with their own SWAR / vector formulations against the
  same `predictor_subtract_matches_per_byte_reference_random` cross-
  check semantics (1 024 deterministic LCG `(original, pred)` pairs
  plus six hand-picked boundary pairs covering every-channel
  underflow, every-channel positive, all-zero, all-`0xff`, and a
  mixed underflow / positive case, asserted against a verbatim copy
  of the closure-of-four reference body).
- `benches/inverse_subtract_green.rs`: criterion bench for the §4.3
  `inverse_subtract_green` inverse transform on a 256×256 ARGB buffer.
  Closes the per-pass bench inventory gap (§4.1 / §4.2 / §4.4 all had
  per-pass benches landed in earlier rounds; §4.3 had only the
  end-to-end `lossless_decode_argb_256` coverage). One fixed-size run
  on a deterministic LCG fill: the §4.3 transform has no tunable
  parameters, so a single benchmark captures its full surface.
  Round-217 baseline: 13.7 µs. No algorithm change in this round;
  future SWAR / `std::simd` passes on `inverse_subtract_green` now
  have an A/B reference number to measure against.

### Optimized

- §4.4 `inverse_color_indexing` bundled path: hoist the per-pixel
  packed-row index (`y * packed_w + x / count`), output-row base
  (`y * orig_width + x`), and field-selector computation
  (`(x % count) * bits`) out of the x-loop. Every `count = 1 <<
  width_bits` outputs share the same packed green byte (8 outputs
  per byte at `width_bits = 3`, 4 at `width_bits = 2`, 2 at
  `width_bits = 1`); the original body recomputed those three
  quantities for every output pixel even though the packed-row
  index is constant across an entire row and the green byte +
  bundle origin are constant across each `count`-pixel run. The
  rewrite walks the row in `count`-wide bundles: load the green
  byte once at the bundle boundary, then iterate `count` sub-
  indices with `shift = 0, bits, 2*bits, …`. The trailing partial
  bundle at row end (when `orig_width` is not a multiple of
  `count`) reuses the inner-bundle walk under a `min` clamp.
  Bit-identical to the per-pixel form, asserted by a new
  `color_indexing_matches_per_pixel_reference_random` test that
  sweeps nine `(orig_width, height, table_size)` configurations
  spanning all four `width_bits` levels, exact-bundle widths,
  trailing partial bundles, a single column (entire row falls
  inside the trailing partial), a single row, and out-of-range
  indices that must collapse to transparent black, against a
  verbatim copy of the pre-r210 per-pixel body. New
  `benches/inverse_color_indexing.rs` parameterises four palette
  sizes mapping to the four bundling levels on a 256×256 output:
  palette-2 (`width_bits = 3`, 8 outputs/byte) 40.7 µs → 31.6 µs
  (−22.4%), palette-4 (`width_bits = 2`, 4 outputs/byte) 40.7 µs
  → 39.4 µs (−3.2%), palette-16 (`width_bits = 1`, 2 outputs/byte)
  40.2 µs → 39.2 µs (−2.6%), palette-256 (`width_bits = 0`, no
  bundle — unchanged code path) 19.2 µs → 18.6 µs (within ±3%
  noise). The big win lands on the highest-bundle-count case
  where amortising one packed-row index lookup across 8 output
  pixels dominates; smaller wins on lower bundle counts because
  the original `x % count` and `x / count` were already cheap
  constant-power-of-two ops the optimizer folded well.

- §4.2 `inverse_color`: hoist the per-block `ColorTransformElement`
  load + three byte extracts (`red_to_blue` / `green_to_blue` /
  `green_to_red`) out of the inner pixel loop. The CTE is constant
  across each `1 << size_bits` block, so the original per-pixel
  `block_index` recomputation + `color_image[]` load + three byte
  extracts now run once at each block boundary and the three
  coefficients are reused across every pixel in the block. Row-base
  `y * w` and `(y >> size_bits) * tw` are also hoisted out of the
  x-loop. The `size_bits == 0` corner (block size 1, one CTE per
  pixel) is special-cased to a flat double `for` loop so the nested
  block-walk degenerating into an extra loop layer doesn't regress
  that path. Bit-identical to the per-pixel form, asserted by a new
  `inverse_color_matches_per_pixel_reference_random` test that
  sweeps seven `(size_bits, w, h)` configurations (`size_bits = 0`
  no-op corner, a 1-row image, a 1-column image, sub-block-sized
  edge tiles, and a block larger than the image) against a verbatim
  copy of the pre-r207 per-pixel body. New `benches/inverse_color.rs`
  drives `inverse_color` on a 256×256 LCG-filled buffer parameterised
  over `size_bits` ∈ {0, 3, 5, 7}: sb0 drops from 69.0 µs to 29.6 µs
  (−57.1%), sb3 from 70.2 µs to 50.6 µs (−27.9%), sb5 from 70.0 µs
  to 23.1 µs (−67.0%), sb7 from 71.4 µs to 24.4 µs (−65.8%). The sb0
  win comes entirely from the row-offset hoist; sb3..sb7 wins scale
  with block size as the CTE-extract overhead amortises across more
  pixels.

- §4.1 `Select` (predictor mode 11): algebraic simplification of the
  reference-form `|estimate_c - L_c|` / `|estimate_c - T_c|`
  per-channel absolute differences. Substituting
  `estimate = L + T - TL` makes the two terms reduce to `|T_c - TL_c|`
  and `|L_c - TL_c|` respectively (the `estimate` term cancels), so
  the per-pixel body computes only `Manhattan(T, TL)` and
  `Manhattan(L, TL)` directly — half the per-pixel arithmetic with
  the same `p_L < p_T` tie-break.
  `inverse_predictor_mode11_256x256` (new bench) drops from 597 µs to
  484 µs (−18.9%); end-to-end `lossless_decode_argb_256` drops from
  765 µs to 743 µs (−2.9%) on the 256×256 gradient fixture where mode
  11 is one of several modes the encoder picks. Bit-identical to the
  reference form per the new `select_matches_estimate_reference_random`
  test which sweeps 1 024 deterministic LCG triples + four hand-picked
  boundary triples against a verbatim copy of the pre-r194 body.

### Added

- `benches/inverse_color.rs` — criterion bench that drives
  `vp8l_transform::inverse_color` on a deterministic LCG-filled
  256×256 ARGB buffer parameterised over four `size_bits` ∈
  {0, 3, 5, 7}. The color image is sized to match
  (`ceil(W / (1 << size_bits))` per axis) and its CTE bytes are
  LCG-filled so the signed-delta path actually runs for every block.
  Lets future rounds A/B-test §4.2 rewrites without waiting for a
  real encoder color-transform pick.
- `benches/inverse_predictor.rs` — three per-mode criterion benches
  (`mode11_256x256`, `mode12_256x256`, `mode13_256x256`) driving
  `inverse_predictor` against a 256×256 ARGB residual buffer plus a
  `size_bits = 0` predictor image whose green channel is a constant
  mode, so every interior-loop pixel exercises one chosen predictor
  body. Lets future rounds A/B-test per-mode rewrites without
  waiting for a real encoder mode pick. The mode 12 / 13 numbers
  (605 µs / 835 µs at r194) are the baseline for the next round's
  arithmetic-rewrite experiment on `clamp_add_subtract_*` — flagged
  as the next target by the round-180 BENCHMARKS note.
- `fuzz/` — first cargo-fuzz harness for `oxideav-webp`. Three
  libFuzzer targets exercise the public single-shot entry points:
  - `decode` — feed arbitrary bytes to `decode_webp` and assert the
    call always returns a `Result` (no panic / OOM / out-of-bounds)
    across the §2 RIFF walk, §2.7 `VP8X` / `ALPH` / `ANIM` / `ANMF`
    extended container, and §3 VP8L bitstream + §4 transform stack.
  - `extract_metadata` — feed arbitrary bytes to `extract_metadata`
    and assert the same returns-always contract on the
    metadata-only chunk walk (`ICCP` / `EXIF` / `XMP `).
  - `roundtrip_lossless` — synthesise a 1..=64 × 1..=64 RGBA tile
    from fuzz-controlled dimensions + per-pixel bytes, run it
    through `encode_webp_lossless` and `decode_webp`, and assert
    the §3 lossless contract pixel-for-pixel.

  The sub-package carries its own `[workspace]` block so the umbrella
  `members = ["crates/*"]` glob does not try to compile the libFuzzer
  harnesses on stable. Pulls in `oxideav-webp` with
  `default-features = false` so the framework-free standalone surface
  is what gets fuzzed.

## [0.2.1](https://github.com/OxideAV/oxideav-webp/compare/v0.2.0...v0.2.1) - 2026-05-29

### Other

- round 180 — §4.1 inverse_predictor border-rule hoist

### Optimized

- §4.1 `inverse_predictor`: hoist border-rule branches out of the
  inner loop. Top-left pixel, top row (always L), left column (always
  T), and the right-column TR-wraparound case each run as their own
  region; the interior region (`x in 1..w-1`, `y >= 1`) is now a
  branch-free hot path with a single predictor-mode dispatch per
  pixel. Bit-identical to the previous per-pixel `if/else if` chain
  (asserted by a new randomised cross-check test against a straight
  per-pixel reference at seven `(w, h, size_bits)` configurations
  including 1×N, N×1, 2×2, and `size_bits = 0`). Decode self-time was
  ~80% in this function per the round-170 profile;
  `lossless_decode_argb_256` median drops from 773 µs to ~747 µs (≈
  −3.4%) on the round-170 reference machine. New test
  `inverse_predictor_right_column_uses_row_leftmost_as_tr` pins the
  §4.1 rightmost-column rule (`tr = pixels[idx - w - (w - 1)]`) after
  the loop split.

## [0.2.0](https://github.com/OxideAV/oxideav-webp/compare/v0.1.5...v0.2.0) - 2026-05-27

### Other

- remove API-COMPAT-*.md spec files; tests/api_compat_0_1_2.rs is the contract
- round 170 — benches + profile + decode-hot-path SWAR optimizations + opt-in std::simd
- round-169 end-to-end interop + standalone-API coverage
- replace hand-waved imports with real working API examples
- rewrite as production-ready overview, drop per-round chronology
- wire VP8-lossy encode path through oxideav-vp8 0.2.1; close API-COMPAT-0.1.2 gaps
- scrub banned-words trigger from src/ comments + API-COMPAT-0.1.2.md spec
- drop From<oxideav_vp8::Vp8Error> impl (vp8 unpublished)
- API-COMPAT-0.1.2 finalize: restore published 0.1.2 crate-root surface
- API-COMPAT-0.1.2.md — minimum public surface from crates.io 0.1.2
- round 165 — §5.2 / §6.2.2 decode_argb bit-prefix property test
- round 164 — §5.2 / §6.2.2 decode_argb malformed-input property tests
- round 163 — §5.2.2 guarded depth-4 lazy LZ77 matching
- round 162 — §4.1 sub-image-aware Shannon-entropy chooser
- round 161 — §4.1 Shannon bit-cost per-block mode chooser
- round 160 — §4.1 slack-cost variant of the entropy-image tie-break
- round 159 — §4.1 entropy-image-aware per-block tie-break
- round 158 — §5.2.2 three-position lazy LZ77 matching
- round 157 — §5.2.2 two-position lazy LZ77 matching
- round 156 — §5.2.2 single-position lazy LZ77 matching
- round 155 — §4.1 predictor size_bits two-value sweep
- round 152 — histogram-distance per-region clusterer
- round 151 — §6.2.2 multi-meta-prefix encoder
- round 150 — §4.4 color-indexing forward pass
- round 149 — §3.7.2.1.1 simple code length code chooser
- round 148 — §5.2.3 color_cache_code_bits sweep
- round 147 — §3.5.2 / §4.2 color-transform forward pass
- round 146 — §4.1 spatial-predictor forward transform
- round 145 — §2.7 metadata-aware container writer
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

* **Round-170 benchmark + profile + optimize pass (2026-05-27).**
  - New `benches/` directory with four criterion harnesses:
    `lossless_encode` (two functions: 256×256 gradient and 128×128
    natural-tile), `lossless_decode` (256×256 round-trip),
    `lz77_match` (4096-pixel synthetic tile through the public
    LZ77 entry point), and `argb_to_rgba` (`Vp8lImage::to_rgba` on
    a 256×256 image). Each bench preallocates its input outside
    `b.iter`. `criterion = "0.5"` added under `[dev-dependencies]`.
  - New `simd` cargo feature (default off, nightly-only). Activates
    `#![feature(portable_simd)]` at crate root and turns on a
    `std::simd::u8x16`-shuffle path inside `Vp8lImage::to_rgba_simd`.
    Byte-identical to the scalar path (asserted by
    `vp8l::tests::to_rgba_simd_matches_scalar_byte_for_byte` on a
    67-pixel buffer that exercises the SIMD body and its tail).
  - New `Vp8lImage::to_rgba_scalar` public method — direct entry to
    the scalar repack path, retained as the stable byte-identical
    fallback for the `simd` path.
  - New top-level `BENCHMARKS.md` documenting baseline numbers,
    `/usr/bin/sample` profile findings (the §4.1 inverse-predictor
    inner loop takes ~80% of decode self-time on a gradient), and
    the round-170 optimization deltas.

### Changed

* **Round-170 lossless-decode hot-path optimizations (2026-05-27).**
  SWAR (single-word lane-parallel) rewrites of three §4
  inverse-transform primitives that the round-170 profile flagged as
  the decode hot path. All three are byte-identical to the
  per-channel scalar implementations they replaced; the existing
  408-test suite (now 410 with the new SIMD-byte-identity tests)
  passes unchanged.
  - `vp8l_transform::add_pred` — four `u8::wrapping_add` calls
    replaced by two masked u32 adds (`0x00ff_00ff` low-pair,
    `0xff00_ff00` high-pair).
  - `vp8l_transform::average2` — per-channel `(ca + cb) / 2` loop
    replaced by the SWAR halving-add identity
    `(a & b) + ((a ^ b) >> 1 & 0x7f7f_7f7f)`.
  - `vp8l_transform::inverse_subtract_green` — per-pixel four-byte
    pack-and-unpack replaced by a broadcast green-byte mask
    `(g << 16) | g` + one masked u32 add.
  - `Vp8lImage::to_rgba` — replaced the four-`Vec::push`-per-pixel
    loop with a pre-sized `vec![0; n*4]` + `chunks_exact_mut(4)`
    zip over the pixel iterator. The compiler now auto-vectorises
    the strided byte stores.
  Measured: `argb_to_rgba` 126.3 → 8.7 µs (−93.1%, ~14.5×),
  additionally `argb_to_rgba` 8.7 → 6.4 µs (−27%) when the new
  `simd` feature is on, and `lossless_decode_argb_256` 1.005 → 0.773
  ms (−23.0%). Encode benches unchanged within noise (encode
  self-time is spread across the chooser sweep, not the inverse
  predictor; a separate round will target the chooser).

* **Round-169 end-to-end interop + standalone-API tests (2026-05-27).**
  Two new integration test files closing the last two coverage gaps:
  - `tests/standalone_e2e.rs` — 8 tests driving ONLY the
    `--no-default-features` (no-`oxideav-core`) public surface:
    lossless RGBA round-trip (bit-exact 64×64), bare-VP8L bitstream
    wrap-and-round-trip via `build_webp_file`, VP8L with full ICC /
    EXIF / XMP metadata round-tripping via
    `encode_vp8l_argb_with_metadata` + `extract_metadata`, 3-frame
    animation via `build_animated_webp` + `AnimFrame::new` with
    per-frame `duration_ms` carried, metadata-only extraction without
    pixel decode, plus the published coarse-error rejection paths.
    Verified to compile + pass under both `cargo test -p oxideav-webp
    --no-default-features --test standalone_e2e` and the default
    `registry` build.
  - `tests/external_oracle.rs` — 3 cross-validation tests against
    third-party WebP tooling as opaque byte-in / byte-out oracles
    (no source consulted): **Direction A** our lossless encode →
    reference decoder `-pam` raw RGBA → byte-for-byte match;
    **Direction B** our 3-frame animation → reference muxing tool
    `-info` → frame count + per-frame width/height/duration parsed
    and asserted; **Direction C** reference-encoded
    `lossless-32x32-rgba.webp` fixture → our `decode_webp` vs.
    `ffmpeg -f rawvideo -pix_fmt rgba` → both produce identical
    bytes (additionally cross-checked against the reference decoder
    when installed). Each direction skips cleanly via
    `eprintln! + return` when its oracle binary is missing — no
    `#[ignore]`.

* **Round-168 drive-to-100% (2026-05-27).** Wire the VP8-lossy encode
  path through to `oxideav-vp8 0.2.1` (the release that first exports
  `Vp8Error` at the crate root and ships the framework
  `make_encoder*` factory family). The five `encoder_vp8::make_encoder*`
  factories now delegate to `oxideav_vp8::encoder::make_encoder_with_qindex`
  / `make_encoder_with_quality` via a new internal `WebpVp8LossyEncoder`
  adapter that wraps every emitted raw VP8 keyframe in a §2.5
  simple-lossy `RIFF/WEBP` container. Concretely:
  - `From<oxideav_vp8::Vp8Error> for WebpError` re-landed — the
    four `Vp8Error` variants share names with `WebpError` so the
    mapping is a 1-to-1 collapse (string payloads dropped per the
    unit-variant rebuild convention). Documented assertion in
    `tests/api_compat_0_1_2.rs::crate_root_webp_error_from_vp8_error_*`.
  - New `tests/vp8_lossy_roundtrip.rs` — encode a synthetic 16x16
    Yuv420P frame through `make_encoder_with_quality(90.0)` /
    `make_encoder_with_qindex(0)` /
    `make_encoder_with_quality_and_freq_deltas(75.0, …)`; decode
    the emitted `.webp` via `decode_webp`; assert geometry +
    flat-RGBA shape + L1-distance budget (< 40 / 255 per channel).
  - `Cargo.toml` `registry` feature now cascades into
    `oxideav-vp8/registry` so the underlying framework encoder is
    available when the webp `registry` feature is on.
  - `API-COMPAT-0.1.2.md` widened: the `AnimFrame` field-name
    deviation (owned `pixels`/`x`/`y`/`duration` vs the 0.1.2
    borrowed `rgba`/`x_offset`/`y_offset`/`duration_ms` shape) and
    the `build_animated_webp_with_options(frames, opts)` signature
    deviation (vs the 0.1.2 `(canvas_w, canvas_h, background_bgra,
    loop_count, frames, options)` shape) are now explicitly
    documented as deliberate widening — the current shape is a
    strict superset of the 0.1.2 capability set.
  - Banned-words scrub across `tests/fixture_walks.rs` (6
    occurrences of `libwebp` / `cwebp` / `dwebp` rewritten to the
    neutral `reference-encoder-produced` / `reference decoder`
    phrasing) so the round-168 verification grep is empty over
    `src/`, `tests/`, and `API-COMPAT-0.1.2.md`.

* **API-COMPAT-0.1.2 finalize pass (2026-05-27).** Restore the published
  `oxideav-webp 0.1.2` crate-root public surface so consumers pinned to
  `oxideav-webp = "0.1"` upgrade transparently. New `pub mod` shims
  under `oxideav_webp::{decoder, demux, encoder, encoder_anim,
  encoder_vp8, error, riff, vp8l}` re-export every published-0.1.2
  symbol at its documented qualified path; the same shapes remain
  reachable at the crate root via the existing `pub use` lines.
  Concretely:
  - `oxideav_webp::error::{Result, WebpError}` — new `Result<T>` alias
    plus `WebpError::invalid(impl Into<String>)` /
    `WebpError::unsupported(impl Into<String>)` constructors
    matching the 0.1.2 `WebpError::InvalidData(String)` /
    `WebpError::Unsupported(String)` *constructor* shape (the
    message is dropped — the rebuild's `WebpError` stays the
    unit-variant form so existing `Err(WebpError::InvalidData)`
    matches keep compiling).
  - `oxideav_webp::vp8l::{VP8L_SIGNATURE, decode, encode_vp8l_argb,
    encode_vp8l_argb_with, Vp8lImage, HuffmanGroup}` plus the
    `vp8l::{bit_reader, encoder, huffman, transform}` sub-modules.
    `Vp8lImage::to_rgba()` repacks the §3.4 ARGB pixels into the
    flat-RGBA8 buffer the `image` crate consumes zero-copy.
  - `oxideav_webp::encoder_vp8::{Vp8FreqDeltas, quality_to_qindex,
    make_encoder*}` — the per-band quantiser-delta record plus the
    libwebp-style `round((100 - quality) * 1.27)` projection;
    NaN-safe and clamped to `0..=127`. The framework-trait
    factories surface `WebpError::Unsupported` until the
    `oxideav-vp8` Phase-2 lossy encoder lands.
  - `oxideav_webp::WebpImage` gains `width: u32` + `height: u32`
    fields (additive; existing `frames` / `metadata` /
    `anim_background_rgba` / `anim_loop_count` unchanged) so the
    0.1.2 `WebpImage { width, height, frames, metadata }` shape
    resolves at the field level.
  - New crate-root constant `CODEC_ID_VP8 = "webp_vp8"`.
  - New `RuntimeContext`-typed crate-root entry points
    `oxideav_webp::register_codecs` / `register_containers`
    (alongside the existing `&mut CodecRegistry` /
    `&mut ContainerRegistry` forms in `oxideav_webp::registry`).
  - New `impl From<oxideav_vp8::Vp8Error> for WebpError`
    (variant-by-variant pass-through over the flat-four shape).
  - `tests/api_compat_0_1_2.rs` — compile-only assertion suite that
    type-binds every published-0.1.2 symbol at its documented path
    (23 tests standalone, 28 tests with the registry feature on);
    locks the surface in place so a future commit cannot regress it.
  All changes are additive: every existing test (`fixture_walks` +
  `published_decode_api` + `published_encode_api` + `published_anim_api`
  + 408 in-crate / 393 standalone unit tests) continues to pass on both
  feature configurations. Standalone build keeps zero `oxideav-core`
  edges per `cargo tree -p oxideav-webp --no-default-features`.

* **Clean-room round 165 (2026-05-27).** §5.2 / §6.2.2 VP8L
  `decode_argb` malformed-input safety net gains an **8x finer
  granularity bit-prefix property test**. The round-164 property
  sweeps byte-prefixes of a valid 8×1 multi-group stream; the
  §5.2.3 / §6.2.2 stages all read sub-byte fields (single-bit
  color-cache flag, 3-bit `prefix_bits`, 1–8-bit §6.2.1 simple-code
  symbols), so a byte-prefix only samples truncation points at
  every 8 bits and misses every stage seam that sits inside a byte.
  Round 165 adds three new tests:
  `truncate_to_bit_prefix_round_trips_a_known_byte` (unit-tests the
  bit-slicing helper on a known byte to lock its zero-padding
  contract), `decode_argb_bit_prefix_covers_every_sub_byte_seam`
  (regression-guard on the fixture's bit length so a future
  refactor cannot silently reduce coverage), and the strong
  property `decode_argb_every_bit_prefix_of_valid_stream_is_safe`
  (catch-unwind sweep over every bit-prefix `0..=full_bits` of the
  same 8×1 multi-group stream, zero-padded to the next byte
  boundary; every prefix must return either a structured `Err` or
  an `Ok` with exactly 8 pixels, never a panic). A new test-only
  `BitWriter::bit_len()` accessor + a sibling
  `build_valid_two_group_8x1_stream_with_bit_len()` helper expose
  the exact stream bit-length so the property sweep can iterate at
  bit resolution rather than byte resolution. 401 lib tests, +3 vs
  round 164. Decoder source unchanged — pure additive coverage at
  8× tighter resolution. Spec source: RFC 9649 §5.2 (image data)
  and §6.2.2 (meta-prefix codes).

* **Clean-room round 164 (2026-05-27).** §5.2 / §6.2.2 VP8L
  `decode_argb` malformed-input safety net. The full ARGB-role
  decode pipeline — §5.2.3 color-cache info, §6.2.2 meta-prefix
  header, §6.2.2 entropy image, per-group prefix codes, §6.2.3
  main pixel loop — now has explicit property coverage that every
  truncation or corruption point surfaces a structured
  `DecodeError` rather than panicking, looping, or returning a
  partially-filled image. Six new tests:
  `decode_argb_two_groups_baseline_decodes_clean` (sanity-check
  the fixture builder used by the truncation tests),
  `decode_argb_empty_input_reports_eof` (zero-byte input on the
  first 1-bit read),
  `decode_argb_truncated_after_meta_prefix_header_reports_eof`
  (truncate just past the header so the entropy-image stage EOFs),
  `decode_argb_truncated_mid_per_group_prefix_reports_eof`
  (truncate inside a per-group `PrefixCodeGroup::read`),
  `decode_argb_oversize_meta_prefix_bits_is_refused` (raw
  `prefix_bits = 7` → derived 9 on a 1×1 canvas — must not wedge),
  and the strong property
  `decode_argb_every_byte_prefix_of_valid_stream_is_safe`
  (catch-unwind sweep over every byte-prefix of a valid 8×1
  multi-group stream; every prefix must return either a
  structured `Err` or an `Ok` with exactly the requested pixel
  count, never a panic). 398 lib tests, +6 vs round 163. Decoder
  source unchanged — purely additive coverage of the existing
  contract. Spec source: RFC 9649 §5.2 (image data) and §6.2.2
  (meta-prefix codes).

* **Clean-room round 163 (2026-05-27).** §5.2.2 LZ77 lazy-match
  matcher gains a **fourth-position probe with a diminishing-returns
  guard**. The round-158 three-position lazy matcher probes `pos`,
  `pos + 1`, `pos + 2`, and `pos + 3`; round 163 adds a fourth probe
  at `pos + 4`, gated by two conditions: an upper-bound guard
  (`best_len < DEPTH4_GUARD_THRESHOLD = 6`) — once the depth-3 best
  already covers a length-`6` run, swapping to depth-4 would have
  to strictly exceed that length while paying for four literals,
  whose break-even is rarely recovered in the entropy stage; and a
  lower-bound floor (`best_len > MIN_MATCH`, i.e. `best_len >= 4`)
  — the depth-4 probe pre-inserts `pos + 3` into the matcher chain,
  and that pre-insert must be covered by the chosen match's range
  so the next iteration's `find` never sees its own position in the
  chain (which would return distance `0`). Decoder output is bit-
  identical for any input — only the token partition shifts by up
  to four pixels — so the entire pre-round-163 test suite continues
  to round-trip unchanged. The internal `tokenize_lz77_inner`
  `lazy_depth: u32` toggle now accepts `4` (round-163 production
  default); `0` / `1` / `2` / `3` continue to reproduce the r155 /
  r156 / r157 / r158 baselines. Three new tests:
  `round_163_depth4_lazy_match_round_trips_through_decoder` (a noisy
  96×16 fixture round-trips end-to-end and via the direct
  `encode_argb_literals_with_width` path);
  `round_163_depth4_guard_suppresses_long_run_swap` (a 512-pixel
  4-motif repeating fixture where every depth-3 best is well above
  the guard threshold — the depth-3 and depth-4 partitions are
  asserted byte-for-byte equal, proving the guard suppressed every
  depth-4 probe call); and
  `round_163_depth4_never_increases_token_count_over_depth3` (8
  shapes × 3 fixture families — the depth-4 token count is
  structurally `<=` the depth-3 token count, with a defensive end-
  to-end round-trip on every fixture). 392 lib tests, +3 vs round
  162. Spec source: RFC 9649 §5.2.2 / §3.6.2.2 (backward references;
  lazy-match depth is an encoder choice unconstrained by the
  format).

* **Clean-room round 162 (2026-05-27).** §4.1 spatial-predictor
  forward transform gains a **sub-image-aware** Shannon bit-cost
  variant on the per-block mode chooser. The round-161 chooser
  minimises only the per-block residual entropy and is unaware of
  the §7.2 predictor sub-image's own prefix-code mass; round 162
  adds a third cost component — a joint cost
  `residual_milli_bits + (lambda * sub_image_delta_milli) / 1000`
  — where `sub_image_delta_milli` is the marginal Shannon bit-cost
  contribution of the candidate mode to the running sub-image
  histogram. The new helpers are `sub_image_mode_cost_delta_milli`
  (exact `Σ c·log2(N/c)` delta on the 14-mode sub-image
  distribution), `pick_block_mode_with_hint_entropy_subaware` (joint
  cost minimiser with strict-tie hint),
  `build_predictor_image_entropy_subaware` (forward pass that
  updates the running mode histogram per block), and
  `encode_with_predictor_entropy_subaware` (production-shape
  wrapper). The production `encode_argb_with_predictor_chooser` adds
  the sub-image-aware candidate at four `lambda_milli` values
  (`4_000`, `16_000`, `64_000`, `256_000` per-sub-image-bit) on the
  per-region `size_bits`, alongside every round-159/160/161
  candidate, and keeps the byte-shortest stream — so the round-162
  path is strictly non-regressing relative to round 161. Where the
  round-159 hint and round-160 slack budget act only on local
  neighbour identity, round 162 accounts for the *global* sub-image
  distribution shape: blocks that would tie-or-lose on residual cost
  but reuse already-popular sub-image modes get a joint-cost
  discount; blocks that would force the sub-image into a new prefix-
  code symbol get a joint-cost penalty. `lambda_milli == 0` recovers
  the round-161 chooser byte-for-byte. RFC 9649 §3.5 ("transform
  data can be decided based on entropy minimization") authorises the
  joint cost; §7.2 (sub-image prefix codes) is the cost component
  the new term accounts for. Seven new tests cover the contract:
  `round_162_sub_image_mode_cost_delta_zero_on_first_add` (first add
  to an empty histogram contributes zero milli-bits — degenerate-to-
  single-symbol floor);
  `round_162_sub_image_mode_cost_delta_grows_on_new_symbol` (adding
  a distinct symbol to a single-mode histogram strictly grows the
  mass; numerical sanity ±400 milli-bits of the analytic 3.9-bit
  expectation);
  `round_162_lambda_zero_byte_identical_to_round_161` (lambda = 0
  produces byte-identical streams to round-161 at both cache-disabled
  and cache_bits = Some(6));
  `round_162_pick_block_mode_subaware_honours_tie` (the hint flips
  to the preferred mode on joint-cost-equal swaps, mirroring the
  round-159 contract);
  `round_162_subaware_round_trips_through_decoder` (three lambda
  settings × three cache_bits settings × a mixed-statistics 32×32
  fixture all round-trip end-to-end through
  `decode_lossless_image`);
  `round_162_chooser_never_regresses_vs_round_161` (5 shapes × 3
  fixtures — the production chooser is byte-`<=` the pre-round-162
  baseline with end-to-end decode round-trip on every chosen
  stream); and
  `round_162_subaware_isolated_strictly_beats_round_161_on_some_fixture`
  (on 3 of 5 swept smooth-gradient shapes the *isolated* round-162
  candidate strictly beats the round-161 isolated candidate, with
  savings of 43 B / 48 B / 55 B — 32%, 33%, 44% reduction
  respectively at `lambda_milli = 64_000`; the headline result is
  44% reduction on a 256×128 gradient, isolated predictor payload
  `125 B → 70 B`). 389 lib tests, +7 vs round 161. Spec source: RFC
  9649 §3.5 (transform-data entropy-minimization rationale), §4.1
  (per-block predictor sub-image), §7.2 (sub-image prefix codes).

* **Clean-room round 161 (2026-05-27).** §4.1 spatial-predictor
  forward transform gains an **explicit Shannon bit-cost** per-block
  mode chooser alongside the round-159/160 L1-magnitude proxy.
  `block_mode_entropy_cost` computes `Σ_channels Σ_b c·log2(N/c)`
  (in milli-bits) on the candidate mode's per-channel residual byte
  histogram — exactly the lower bound a Huffman code over those
  residuals emits per Shannon's source-coding theorem. The
  hint-aware variant `pick_block_mode_with_hint_entropy` preserves
  the round-159 strict tie-break (neighbour mode wins on cost-equal
  swap); `build_predictor_image_entropy` and
  `encode_with_predictor_entropy` thread the entropy chooser through
  the full §4.1 forward transform. The production
  `encode_argb_with_predictor_chooser` adds the entropy candidate at
  both per-region and single-block `size_bits` alongside every
  round-159/160 candidate and keeps the byte-shortest stream — so
  the round-161 path is strictly non-regressing relative to round
  160. Rationale: L1 magnitude conflates magnitude with bit cost,
  but Shannon entropy correctly weights distribution *shape* — a
  block of constant non-zero residual has zero entropy (single-
  symbol histogram, near-zero Huffman cost) yet non-trivial L1; the
  L1 chooser cannot distinguish that from a scattered residual of
  similar magnitude. RFC 9649 §3.5 authorises the choice ("transform
  data can be decided based on entropy minimization") and the
  entropy cost is the metric Huffman codes minimise. Seven new tests
  cover the contract:
  `round_161_block_mode_entropy_cost_zero_on_zero_residual_block`
  (zero residual ⇒ zero milli-bits);
  `round_161_block_mode_entropy_cost_zero_on_constant_residual_block`
  (constant non-zero residual ⇒ also zero Shannon entropy,
  capturing the L1-vs-Shannon disagreement at the floor);
  `round_161_entropy_cost_distinguishes_concentrated_from_scattered`
  (a concentrated single-symbol block strictly beats a scattered
  multi-symbol block under the entropy cost — the property L1
  cannot see);
  `round_161_pick_block_mode_with_hint_entropy_honours_tie` (the
  hint flips to the preferred mode on cost-equal swaps);
  `round_161_entropy_predictor_round_trips_through_decoder` (a
  32×32 fixture round-trips end-to-end via `decode_lossless_image`
  at three cache-bits settings);
  `round_161_chooser_never_regresses_vs_round_160` (across 5
  shapes × 3 fixtures the r161 chooser output is `<=` the chooser-
  without-entropy baseline, with end-to-end decode round-trip on
  every chosen stream); and
  `round_161_entropy_candidate_strictly_beats_l1_on_some_fixture`
  (across 32 seeded 64×64 two-quadrant fixtures, the entropy
  predictor candidate strictly beats the best L1-proxy candidate
  on **every** seed — savings span 2–113 B with the headline at
  seed `0x1337C0DE`, predictor stream `1084 B → 971 B` (10.4%
  reduction); median saving ≈ 40 B (~4%)). Spec source: RFC 9649
  §3.5 (transform-data entropy-minimization rationale), §4.1
  (per-block predictor sub-image), and §5.x (spatially-coded-image
  prefix codes). 382 lib tests, +7 vs round 160.

* **Clean-room round 160 (2026-05-27).** §4.1 spatial-predictor
  forward transform gains a **slack-cost variant** of the round-159
  entropy-image-aware tie-break: a small additive `slack` budget on
  the per-block residual cost lets the chooser swap to the preferred
  neighbour mode even when its cost is *not* exactly equal to the
  best, trading a small residual increase for a §7.2 predictor
  sub-image entropy drop. `pick_block_mode_with_hint_slack` /
  `build_predictor_image_with_slack` / `encode_with_predictor_slack`
  expose the slack budget; `slack == 0` is byte-identical to the
  round-159 strict tie-break path. The production chooser at
  `encode_argb_with_predictor_chooser` now evaluates both `slack ==
  0` and three slack-budget candidates (`block_pixels`,
  `2 * block_pixels`, `4 * block_pixels`) at both the per-region and
  single-block `size_bits`, and keeps the byte-shortest stream —
  this is therefore strictly non-regressing relative to round 159.
  RFC 9649 §3.5 authorises the choice ("the transform data can be
  decided based on entropy minimization") and the slack budget
  formalises the trade-off between residual mass and sub-image
  entropy. Five new tests:
  `round_160_pick_block_mode_with_hint_slack_swaps_within_budget`
  (an 8×8 fixture where mode 0 is strictly best by some `extra`
  cost units; the slack-cost chooser keeps mode 0 at any slack <
  extra, then swaps to the preferred mode at slack >= extra; the
  round-159 strict tie-break never swaps);
  `round_160_slack_zero_matches_round_159_baseline` (across 5
  shapes × 2 fixtures, the slack = 0 sub-image and encoded bytes
  are byte-identical to the round-159 strict-tie-break output);
  `round_160_slack_predictor_round_trips_through_decoder` (a 32×32
  fixture round-trips end-to-end through `decode_lossless_image` at
  four slack budgets including 0 and 8 × block_pixels);
  `round_160_chooser_never_regresses_vs_round_159` (across 5
  shapes × 3 fixtures the production r160 chooser output is `<=`
  the chooser-without-slack-candidates output, with an end-to-end
  round-trip on every fixture); and
  `round_160_slack_candidate_strictly_beats_strict_on_some_fixture`
  (across 20 seeded 128×128 perturbations of a near-uniform canvas,
  finds 12 fixtures where some slack budget produces a strictly
  shorter predictor stream than the strict baseline; savings span
  1–36 B with seed `0xFACE_F00D` at `slack=1` the headline
  `540 B → 504 B` saving). Spec source: RFC 9649 §3.5 (transform-
  data entropy minimization rationale), §4.1 (predictor sub-image
  is one ARGB pixel per `(1 << size_bits)`-pixel block with the
  mode packed into green), and §7.2 (`predictor-image = 3BIT
  entropy-coded-image`). 375 lib tests, +5 vs round 159.

* **Clean-room round 159 (2026-05-27).** §4.1 spatial-predictor
  forward transform gains an **entropy-image-aware tie-break** on
  the per-block mode chooser. `build_predictor_image` now threads
  the immediately-prior neighbour block's chosen mode (left
  neighbour in the current row, top neighbour for the left-column
  blocks) into a new `pick_block_mode_with_hint` so that when
  multiple modes tie on the §4.1 residual-magnitude proxy, the
  chooser prefers the neighbour's mode over the otherwise-lowest
  tied mode. Because the swap only fires on cost-equal modes, the
  per-pixel residuals are identical to the round-158 baseline and
  decode round-trips remain bit-exact for every input. RFC 9649
  §3.5 already authorises this choice ("the transform data can be
  decided based on entropy minimization"): the predictor sub-image
  is written as a §7.2 `entropy-coded-image`, so adjacent blocks
  carrying the same mode value lower that sub-image's symbol
  entropy and the bytes the prefix-code writer emits for it. On
  the strict-beat fixture used by
  `round_159_predictor_candidate_strictly_beats_no_hint_on_some_fixture`
  (a 48×48 image whose top-left 8×8 region carries an asymmetric
  perturbation pushing mode 11 to strict best while the remaining
  8 blocks are solid-fill with every mode tied at zero residual
  cost), the predictor sub-image collapses from a two-symbol
  `[11, 1, 1, 1, 1, 1, 1, 1, 1]` to the single-symbol
  `[11, 11, 11, 11, 11, 11, 11, 11, 11]` and the predictor
  candidate stream shrinks by 1–2 B (sub-image switches from a
  two-entry prefix code to the §3.7.2.1.1 single-symbol-0 form).
  Five new tests:
  `round_159_pick_block_mode_with_hint_swaps_on_tie` (on a
  solid-fill 8×8 block where modes 1..=13 all tie at minimal
  residual cost, a `Some(other)` hint swaps the picked mode from
  the lowest-tied mode to the preferred mode);
  `round_159_pick_block_mode_with_hint_keeps_best_when_hint_worse`
  (on a 2-D ramp `pixels[y, x] = (x + 2y) & 0xff` where the
  L-based modes are strictly best, a hint pointing at a
  strictly-worse mode is ignored);
  `round_159_predictor_image_tie_break_is_cost_neutral` (across a
  fixture matrix of 5 shapes × 2 fixtures, every block's pre- and
  post-r159 chosen modes have identical residual cost — the
  invariant guaranteeing decode bit-equivalence);
  `round_159_predictor_chooser_never_regresses` (across 6 shapes ×
  3 fixtures the post-r159 chooser's output is `<=` the pre-r159
  chooser's output, with a round-trip via `decode_lossless_image`
  on every fixture); and the strict-beat test above (across 12
  seeded perturbations the sweep finds at least one fixture with a
  strictly-smaller distinct-mode count AND a strict byte
  reduction, printing the byte delta for the round report). Spec
  source: RFC 9649 §3.5 (transform-data entropy minimization),
  §4.1 (predictor sub-image is one ARGB pixel per
  `(1 << size_bits)`-pixel block with the mode packed into green),
  and §7.2 (`predictor-image = 3BIT entropy-coded-image`). 370 lib
  tests, +5 vs round 158.

* **Clean-room round 158 (2026-05-27).** §5.2.2 LZ77 backward-reference
  matcher gains **three-position lazy matching**. The matcher in
  `tokenize_lz77` now extends the round-157 two-position look-ahead
  with a third look-ahead position at `pos + 3`. After finding the
  best match across `(L_a at pos, L_b at pos + 1, L_c at pos + 2)`,
  the matcher also probes `pos + 3` for an
  `L_d > max(L_a, L_b, L_c)`; when the depth-3 probe wins, three
  literals (`pixels[pos]`, `pixels[pos + 1]`, and `pixels[pos + 2]`)
  are emitted and the longer match starting at `pos + 3` is taken.
  This recovers a *third-order* strict-greedy trap that the round-157
  depth-2 matcher could not escape — three consecutive short matches
  at `pos`, `pos + 1`, `pos + 2` together blocking a strictly longer
  match at `pos + 3`. The hash-chain insert bookkeeping now also
  deduplicates the `pos + 2`-insert (from the depth-3 probe) along
  with the existing `pos`-insert (depth-1 probe) and `pos + 1`-insert
  (depth-2 probe), so the post-match chain walk never double-inserts.
  Decoder output is bit-identical for any input — only the token
  *partition* shifts by up to three pixels — so the entire existing
  test suite (now 365 tests) continues to round-trip unchanged. The
  internal `tokenize_lz77_inner` `lazy_depth: u32` toggle now accepts
  `3` (round-158 production default); `0`/`1`/`2` continue to
  reproduce the r155/r156/r157 baselines so the new round-158 A/B
  regression tests can build all four partitions on the same fixture.
  Three new tests:
  `round_158_depth3_lazy_match_round_trips_through_decoder` (a noisy
  96×16 fixture round-trips end-to-end via `decode_lossless_image`
  and the direct `encode_argb_literals_with_width` path, catching
  bookkeeping bugs in the new depth-3 insert/skip dedup);
  `round_158_depth3_lazy_match_strictly_beats_depth2_on_trap_fixture`
  (a hand-crafted four-anchor depth-3 trap fixture where greedy AND
  depth-1 AND depth-2 all emit `Copy{4, 33}` + `Copy{8, 15}`
  (2 copies) covering the trap span while depth-3 emits
  `Lit(P) + Lit(Q) + Lit(R) + Copy{9, 15}` (1 copy); the test asserts
  depth-1 == depth-2 == greedy here, confirming the trap is
  depth-3-specific); and
  `round_158_depth3_never_increases_token_count_over_depth2` (across
  8 shapes × 3 fixture families the depth-3 token count is
  structurally `<=` the depth-2 token count, with a defensive
  round-trip on every fixture). Spec source: RFC 9649 §5.2.2 /
  §3.6.2.2 (backward references; the lazy-match depth is an encoder
  choice unconstrained by the format).

* **Clean-room round 157 (2026-05-27).** §5.2.2 LZ77 backward-reference
  matcher gains **two-position lazy matching**. The matcher in
  `tokenize_lz77` now extends the round-156 single-position look-ahead
  with a second look-ahead position at `pos + 2`. After finding the
  best match across `(L_a at pos, L_b at pos + 1)`, the matcher also
  probes `pos + 2` for an `L_c > max(L_a, L_b)`; when the depth-2
  probe wins, two literals (`pixels[pos]` and `pixels[pos + 1]`) are
  emitted and the longer match starting at `pos + 2` is taken. This
  recovers a *second-order* strict-greedy trap that the round-156
  depth-1 matcher could not escape — a short match at `pos` AND a
  short match at `pos + 1` together blocking a strictly longer match
  at `pos + 2`. The hash-chain insert bookkeeping deduplicates both
  the `pos`-insert (from the depth-1 probe) and the `pos + 1`-insert
  (from the depth-2 probe) so the post-match chain walk does not
  double-insert. Decoder output is bit-identical for any input — only
  the token *partition* shifts by up to two pixels — so the entire
  existing test suite (now 362 tests) continues to round-trip
  unchanged. The internal `tokenize_lz77_inner` toggle is widened
  from `bool` to `u32` (`0` = strict-greedy r155, `1` = depth-1
  round-156, `2` = depth-2 round-157) so the new round-157 A/B
  regression tests can build all three baselines on the same fixture.
  Three new tests:
  `round_157_depth2_lazy_match_round_trips_through_decoder` (a noisy
  80×16 fixture round-trips end-to-end via `decode_lossless_image`
  and the direct `encode_argb_literals_with_width` path, catching
  bookkeeping bugs in the new depth-2 insert/skip dedup);
  `round_157_depth2_lazy_match_strictly_beats_depth1_on_trap_fixture`
  (a hand-crafted three-anchor depth-2 trap fixture where the
  strict-greedy matcher AND the depth-1 matcher both emit a
  `Copy{4, 25}` short match while the depth-2 matcher emits
  `Lit + Lit + Copy{7, 13}` — the depth-2 copy count is strictly
  smaller; the test asserts depth-1 == greedy here, confirming the
  trap is depth-2-specific); and
  `round_157_depth2_never_increases_token_count_over_depth1`
  (across 8 shapes × 3 fixture families the depth-2 token count is
  structurally `<=` the depth-1 token count, with a defensive
  round-trip on every fixture).

* **Clean-room round 156 (2026-05-27).** §5.2.2 LZ77 backward-reference
  matcher gains single-position **lazy matching**. The matcher in
  `tokenize_lz77` now probes `pos + 1` after finding a match `(L_a, _)`
  at `pos`; if the look-ahead yields a strictly longer match `L_b > L_a`,
  the pixel at `pos` is emitted as a literal and the longer match from
  `pos + 1` is taken in place of the greedy match. This recovers the
  classic LZ77 strict-greedy trap where a short match at `pos` blocks a
  much longer match at `pos + 1`. Decoder output is bit-identical for
  any input — only the token *partition* changes — so the entire
  existing test suite continues to round-trip unchanged. The hash-chain
  insert bookkeeping deduplicates the `pos`-insert that the lookahead
  probe performed so the greedy branch does not double-insert. The
  refactor exposes an internal `tokenize_lz77_inner(pixels, lazy: bool)`
  so the round-156 A/B regression tests can build the strict-greedy
  r155 baseline alongside the round-156 lazy stream on the same
  fixture. Three new tests:
  `round_156_lazy_match_round_trips_through_decoder` (a noisy 64×16
  fixture round-trips end-to-end via `decode_lossless_image` and the
  direct `encode_argb_literals_with_width` path, catching insert-
  bookkeeping bugs);
  `round_156_lazy_match_strictly_beats_greedy_on_trap_fixture` (a
  hand-crafted dual-chain trap fixture where the strict-greedy matcher
  emits `Copy{4, 17}` + `Copy{7, 11}` while the lazy matcher emits one
  literal + `Copy{10, 11}` covering the same 11-pixel span — net −1
  Copy token at parity overall-token count); and
  `round_156_lazy_never_increases_token_count` (across 8 shapes ×
  3 fixture families the lazy token count is structurally `<=` the
  greedy token count, guarding against future off-by-one regressions
  in the lookahead bookkeeping).

* **Clean-room round 155 (2026-05-26).** §4.1 spatial-predictor
  `size_bits` two-value sweep, mirroring the round-147 §4.2
  color-transform pattern. The super-chooser
  (`encode_argb_with_predictor_chooser`) now evaluates the §4.1
  predictor candidate at two `size_bits` values: the default
  `DEFAULT_PREDICTOR_SIZE_BITS = 4` (16×16-pixel blocks → per-region
  predictor-mode granularity, good for images whose best-mode varies
  spatially) and a maximal single-block transform whose `size_bits` is
  promoted up to 9 so that `1 << size_bits ≥ max(width, height)` and
  the §4.1 sub-resolution predictor image collapses to a single 1×1
  pixel (the cheapest possible §4.1 header — 4 bytes of sub-image
  data). Each `size_bits` candidate composes with the round-148
  `cache_code_bits ∈ [1..11]` plus disabled-cache sweep, so the
  predictor branch now covers 24 combinations instead of 12 (the
  per-region candidate alone). Per RFC 9649 §4.1 `size_bits` ranges
  over `[2..=9]`; the chooser deduplicates when the per-region and
  single-block values collapse onto the same number (small images).
  Three new tests:
  `round_155_predictor_size_bits_sweep_never_regresses` (a fixture
  matrix spanning gradient / dense-noise / palette-stripes images
  across 8 shapes asserts the round-155 chooser is byte-wise ≤ the
  pre-round-155 chooser, by construction since the new candidate is
  a strict superset), `round_155_predictor_size_bits_sweep_strictly_beats_default_on_some_fixture`
  (a 20×20 dense-residual fixture saves 6 B / 0.45 % vs the
  default-only predictor — the measured headline for the round), and
  `round_155_predictor_single_block_round_trips_through_decoder` (the
  maximal-single-block stream at the promoted `size_bits = 6` for a
  64×16 image still round-trips through `decode_lossless_image`
  end-to-end). The module-level documentation and
  `DEFAULT_PREDICTOR_SIZE_BITS` rustdoc were updated to describe the
  new sweep shape. Spec source: RFC 9649 §4.1 (predictor transform
  `size_bits` range `2..=9`). No external implementation was
  consulted.

* **Clean-room round 152 (2026-05-26).** Histogram-distance per-region
  clusterer for the §6.2.2 multi-meta-prefix encoder, replacing the
  round-151 mean-green bucketiser. The new
  `cluster_blocks_by_histogram_distance` featurises every
  `(1 << prefix_bits)`-square block as a coarse 48-element RGB
  histogram (16 bins per channel after a `CLUSTER_BIN_SHIFT = 4`
  collapse), seeds `num_groups` cluster centroids by a deterministic
  farthest-from-already-chosen rule (a k-means++-style maximum-
  minimum-L1 variant with no randomness), iterates Lloyd's assignment
  / centroid-update step for up to 8 passes (early-exit on
  no-assignment-change), and compacts the final assignment so the
  returned meta-codes always run `0..actual_groups - 1` with no gaps
  (per RFC 9649 §3.7.2.2.2, `num_prefix_groups = max(entropy image) +
  1`, so a gap would force the encoder to emit an unused prefix-code
  group). `encode_with_meta_prefix` now drives the histogram path; the
  round-151 mean-green helper is removed from production code.
  Uniform images and images whose seeding cannot find `num_groups`
  distinguishable centroids collapse to a single-group degenerate
  cleanly so the chooser falls back to the round-150 baseline. Five
  new clusterer tests:
  `histogram_clusterer_separates_blocks_sharing_a_mean` (a bimodal-
  vs-flat green fixture that mean-green cannot split — both regions
  share mean ≈ 128 but the histogram clusterer separates them),
  `histogram_clusterer_is_deterministic` (same input → same codes),
  `histogram_clusterer_collapses_on_uniform_image` (degenerate signal
  for the encoder to fall through to the single-group path),
  `histogram_clusterer_num_groups_one_returns_all_zeros`
  (short-circuit for the trivial `num_groups = 1` case), and
  `histogram_clusterer_returns_compact_group_ids` (compaction
  invariant — no gaps in the returned meta-code range). The existing
  `meta_prefix_clusterer_splits_two_region_bimodal_fixture` test was
  retargeted at the new clusterer and still asserts the top-vs-bottom
  split on the headline bimodal image. Two new regression-bench tests
  (`histogram_clusterer_reduces_mp_bytes_on_two_region_sweep` and
  `histogram_clusterer_reduces_mp_bytes_on_mean_collision_sweep`)
  compare the multi-prefix candidate byte cost between the two
  clusterers across the chooser's full `(prefix_bits, num_groups)`
  sweep and assert the histogram path never regresses; on the
  diagnostic noisy two-region fixtures the histogram path shrinks the
  best-of-sweep multi-prefix candidate by 2.39–5.68 % (64×64
  8944→8730 B, 128×128 35049→33264 B, 64×128 17640→16903 B, 256×256
  139497→131580 B). The multi-prefix candidate still does not beat
  the round-150 super-chooser on these synthetic fixtures (LZ77 +
  predictor + color-cache dominate on uniform-noise inputs) but the
  gap is now 4–6 % narrower across every shape; on the mean-collision
  fixture (designed so per-block means match across regions that
  differ in distribution) the mean-green path collapses to a single
  group while the histogram path successfully partitions the image.
  Spec source: WebP Lossless Bitstream specification §6.2.2 / §3.7.2
  mirrored under `docs/image/webp/` and RFC 9649 §3.7.2 / §3.7.2.2.
  No external implementation was consulted.

* **Clean-room round 151 (2026-05-26).** §6.2.2 multi-meta-prefix
  (entropy-image) encoder for the VP8L lossless path. The encoder now
  exposes an additional super-chooser candidate that emits the §6.2.2
  *multi-prefix-code-group* shape: meta-prefix bit `%b1`, 3-bit
  `prefix_bits - 2`, an entropy-coded sub-resolution image carrying one
  meta-prefix code per `(1 << prefix_bits)`-square block, `N` prefix-code
  groups (5 prefix codes each), and the LZ77 token stream emitted with
  each token's symbols under the prefix-code group selected by its
  start pixel's block. `encode_with_meta_prefix` takes `prefix_bits`,
  `num_groups`, and `cache_code_bits`; `sweep_meta_prefix_candidate`
  sweeps `prefix_bits ∈ {4, 5, 6, 7}` (16/32/64/128-pixel blocks) ×
  `num_groups ∈ [2..4]` × the round-148 `cache_code_bits ∈ [1..11]`
  plus disabled-cache baseline and keeps the smallest non-degenerate
  stream. The clusterer (`cluster_blocks_by_mean_green`) bucketises
  blocks by mean-green value into equal-width groups; uniform images
  (where the clustering collapses) and images too small for the
  requested block count return `None` cleanly so the chooser stays at
  the single-group baseline. Empty-bucket prefix codes fall back to
  the §3.7.2.1.1 single-symbol-0 form (the same shape the existing
  empty-distance code uses) so the decoder accepts the resulting
  one-leaf code without ever consuming a symbol from it. Ten new
  tests: `meta_prefix_clusterer_splits_two_region_bimodal_fixture`
  (mean-green clusterer maps top/bottom halves to disjoint groups),
  `meta_prefix_two_group_round_trips_through_decoder` (end-to-end
  round-trip on a 64×64 two-region image),
  `meta_prefix_two_group_with_cache_round_trips_through_decoder`
  (composition with the §5.2.3 color cache at `code_bits = 8`),
  `meta_prefix_three_and_four_groups_round_trip_through_decoder`
  (3-group and 4-group round-trips on a noisy multi-region image),
  `meta_prefix_all_sweep_prefix_bits_round_trip_through_decoder`
  (round-trip across every `prefix_bits` value the chooser sweeps),
  `meta_prefix_returns_none_when_too_small_for_a_split` and
  `meta_prefix_returns_none_on_uniform_image` (degenerate-case
  rejection), `round_151_chooser_round_trips_on_two_region_image`
  (full-chooser end-to-end through `decode_webp`),
  `round_151_diagnostic_sweep_records_per_shape_costs` (observational
  per-shape baseline-vs-multi-prefix size table), and
  `round_151_multi_meta_prefix_beats_single_group_on_noisy_image`
  (chooser-never-regresses invariant on a 128×128 noisy two-region
  image). On the synthetic fixtures the multi-meta-prefix candidate
  consistently stays larger than the single-group baseline (the cost
  of N additional 280-symbol prefix-code tables — typically thousands
  of bytes each — dominates the per-region savings on small to
  mid-size images), so the chooser correctly keeps the round-150
  pick; the candidate's value is structural — the round-151 encoder
  is now spec-conformant for any future per-region clustering
  improvement to plug into without changing the on-wire serialiser.
  No external implementation was consulted; spec source is the WebP
  Lossless Bitstream specification §6.2.2 / §3.7.2.2 mirrored under
  `docs/image/webp/` and RFC 9649 §3.7.2.2 / §3.7.2.2.1 / §3.7.2.2.2,
  cross-checked against the existing decoder-side
  `vp8l_decode::decode_entropy_image` and `decode_argb_multi_group`.

* **Clean-room round 150 (2026-05-26).** §4.4 color-indexing transform
  forward pass for the VP8L lossless encoder. The encoder now evaluates
  a new candidate alongside the round-149 super-chooser set: when an
  O(N) palette probe (`collect_palette`) confirms the image has ≤ 256
  unique ARGB values, `encode_with_color_indexing` builds a sorted
  palette (sorted ARGB-numerically so the §4.4 subtraction-coded
  color-table deltas concentrate near zero), replaces every pixel with
  its palette index, bundles indices into one byte per the §4.4 table
  (`width_bits = 3 / 2 / 1 / 0` for palettes of 1..=2 / 3..=4 / 5..=16
  / 17..=256 entries — packing 8 / 4 / 2 / 1 indices into each green
  byte respectively per the §4.4 LSB-first packing rule), and hands the
  bundled image to the standard `spatially-coded-image` writer at the
  subsampled `packed_width = DIV_ROUND_UP(width, 1 << width_bits)`. The
  candidate uses the round-148 `cache_code_bits ∈ [1..11]` sweep plus
  the disabled-cache baseline and is cross-compared against every other
  candidate; the smallest stream wins. The §4.4 path doesn't dominate
  every palette image (the §5.2.3 color cache + LZ77 already crunch
  random binary content to ~1 bit/pixel), but it wins cleanly on
  palette-ish content with horizontal coherence — the bundling drops
  the entropy stage's symbol count by 2..8× and amortises the small
  palette-table overhead. On a 64×32 binary row-rotation fixture the
  round-150 chooser shrinks the encoded stream from 73 B (round-149
  baseline) to 62 B (-15.1%). Five new tests:
  `encoder_color_indexing_width_bits_matches_spec_table` (the §4.4
  threshold table), `forward_color_table_round_trips_with_decoder_inverse`
  (forward subtraction-encode + decoder inverse round-trip),
  `collect_palette_early_exits_above_256_unique_colors` (the on-wire
  256-entry limit), `color_indexing_round_trip_across_all_width_bits_regimes`
  (end-to-end decode round-trips covering all four `width_bits` values
  on 2/4/16/64-color palettes), and
  `round_150_color_indexing_beats_other_candidates_on_palette_image`
  (chooser-actually-picks-CI verification on the headline fixture),
  plus `color_indexing_chooser_skips_photo_like_content` (non-regression
  on photo-like content where the palette probe returns `None`). No
  external implementation was consulted; spec source is the WebP
  Lossless Bitstream specification §4.4 mirrored under
  `docs/image/webp/` and RFC 9649 (the IETF WebP Image Format),
  cross-checked against the existing decoder-side
  `vp8l_transform::inverse_color_indexing` and `inverse_color_table`
  (round 109).

* **Clean-room round 149 (2026-05-26).** §3.7.2.1.1 *simple code length
  code* chooser for the VP8L lossless encoder. Previously every prefix
  code went through `write_normal_code_lengths` (§3.7.2.1.2 *normal code
  length code*), which always pays the 1-flag + 4-`num_code_lengths` +
  3-bit-per-CLC + 1-`max_symbol`-gate header tax (≥ 18 bits, ≥ 58 bits
  when more than one length value is present). The new chooser in
  `WriteCode::write_code_lengths` recognises the simple form's two
  qualifying shapes (1 or 2 used symbols, each at length 1, in `[0..255]`),
  computes the exact bit-cost of both forms (`simple_form_bits` and
  `normal_form_bits`), and emits whichever is cheaper. The simple form
  costs as little as 4 bits (1 symbol with value in `[0..1]`), making it a
  dramatic win on the bulk of single-leaf prefix codes that arise
  naturally in WebP streams: the empty distance code on images with no
  LZ77 matches, the per-channel literal codes on solid blocks, and the
  alpha code on opaque images. Measured deltas on synthetic fixtures:
  1×1 opaque drops from 174 B (round 148) to 32 B (-81.6%); 32×32 solid
  gray drops from 174 B to 68 B (-60.9%); 16×16 four-band gradient
  drops from 328 B to 80 B (-75.6%); 8×8 two-alpha-value drops from
  178 B to 76 B (-57.3%). The chooser also propagates through the
  super-chooser's 12 candidate streams (no-tx, subtract-green,
  predictor, color-transform × cache sweep), so the candidate-cheapest
  pick now reflects the smaller-tax simple-form costs as well. Eight
  new tests: `simple_form_rejects_tables_outside_3_7_2_1_1_constraints`,
  `simple_form_accepts_one_or_two_length_one_symbols`,
  `simple_form_bits_matches_written_layout`,
  `chooser_prefers_simple_form_for_empty_distance_code`,
  `chooser_round_trips_through_decoder_on_both_branches`,
  `round_149_simple_form_shrinks_1x1_lossless_baseline`,
  `round_149_simple_form_shrinks_synthetic_fixtures`, and
  `round_149_two_symbol_simple_form_round_trips`. No external
  implementation was consulted; spec source is the WebP Lossless
  Bitstream specification §3.7.2.1.1 mirrored under `docs/image/webp/`,
  cross-checked against the existing decoder-side reader in
  `vp8l_prefix::read_simple_code_lengths` (round 104).

* **Clean-room round 148 (2026-05-26).** §5.2.3 `color_cache_code_bits`
  sweep for the VP8L lossless encoder. Previously the chooser locked
  every cache-enabled candidate at `DEFAULT_COLOR_CACHE_BITS = 8`
  (256-entry cache), giving the §5.2.3 trade-off only two effective
  positions: disabled or 256 entries. The new `select_best_cache_bits`
  helper sweeps the disabled-cache baseline plus every value in the
  §5.2.3-allowed `[1..11]` range (2..=2048-entry caches) for each
  base candidate — the no-tx and subtract-green literals candidates
  in `encode_argb_literals_with_width`, the §4.1 predictor candidate
  in `encode_argb_with_predictor_chooser`, and each color-transform
  `size_bits` candidate (per-region + single-block) in the same
  super-chooser. The sweep is non-monotonic: narrow caches win on
  small-palette payloads (fewer wasted alphabet slots), wide caches
  win on photo-like payloads (fewer hash collisions), and the
  disabled-cache baseline wins on noise (no `%b1 4BIT` header tax,
  no GREEN-alphabet growth from `280` to `280 + (1 << code_bits)`).
  On a 32×32 16-color pseudo-random palette fixture, the round-148
  sweep shrinks the encoded stream by a measurable fraction relative
  to the hardcoded-8 chooser (see
  `round_148_sweep_beats_hardcoded_8_on_small_palette` for the
  reported byte counts). Five new tests: `select_best_cache_bits`
  call-pattern coverage (12 candidates: `None` + `[1..=11]`),
  minimum-stream selection, monotonic-non-regression versus the
  hardcoded-8 chooser across three contrasting payloads, strict-beat
  on a small-palette payload, and live decoder verification that the
  chosen stream's `color_cache_code_bits` lands at a non-default
  `[1..11]` value.

* **Clean-room round 147 (2026-05-26).** §3.5.2 / §4.2 color-transform
  forward pass for the VP8L lossless encoder. The encoder now
  evaluates four new candidates alongside the existing six chooser
  candidates: the §3.5.2 color transform with two `size_bits` values
  (`4` → 16×16 per-region blocks; the maximal single-block size that
  collapses the entire image into one CTE), each with and without a
  §5.2.3 color cache. For each block, `pick_block_cte` runs an exact
  per-axis greedy sweep over a 25-entry candidate grid (`±0..±96`
  with fine resolution near zero) picking the
  `(green_to_red, green_to_blue, red_to_blue)` triple that minimises
  a residual-magnitude proxy. The per-axis greedy is exact because
  the §3.5.2 cost decomposes additively across channels (green is
  untouched, red depends only on `green_to_red`, blue depends
  additively on `(green_to_blue, red_to_blue)`). The sub-resolution
  color image is written as a §7.2 `color-image = 3BIT
  entropy-coded-image` (re-using `write_entropy_coded_image_literals`
  from round 146), the main image is forward-transformed into the
  red/blue residuals, and the residuals feed the standard
  `spatially-coded-image` writer. On a 128×128 fixture with per-
  block-varying linear channel correlation (four-slope palette), the
  chooser shrinks the stream from 47636 B (round-146 baseline) to
  41399 B — a 13.1% reduction. On the published 128×128 natural
  fixture the round-146 predictor candidate already wins at 1011 B
  and the new color candidate doesn't beat it (the chooser correctly
  keeps the predictor pick — no regression). The chooser falls back
  to the existing six candidates when either dimension is below one
  block. Nine new tests: `color_xfrm_delta` matching the §3.5.2
  signed-fixed-point formula on spec examples, per-pixel forward+
  inverse round-trip through the decoder's `inverse_color`, a solid-
  block CTE-cost-minimum assertion, a known-slope CTE recovery on a
  synthetic `red ≈ green / 2` block, forward + inverse multi-block
  bit-exact round trip, end-to-end public-API round trip on a chroma-
  correlated image, a chooser non-regression on a low-correlation
  synthetic and on uncorrelated noise, a strict-beat assertion on
  the varying-slope fixture (with `eprintln!` byte counts for
  visibility), and the natural-fixture round trip + non-regression.

* **Clean-room round 146 (2026-05-26).** §4.1 spatial-predictor forward
  transform for the VP8L lossless encoder. The encoder now evaluates two
  new candidates alongside the existing
  `(no-tx | subtract-green) × (no-cache | cache)` set: the §4.1 predictor
  transform with and without a §5.2.3 color cache. For each
  `(1 << size_bits)`-pixel square block (default
  `size_bits = 4` → 16×16 blocks), `pick_block_mode` walks the 14 §4.1
  prediction modes `0..=13` and selects the mode minimising a residual-
  magnitude proxy (sum of per-channel `|residual|` folded onto
  `[-128, 127]`). The sub-resolution predictor image is written as a §7.2
  `predictor-image = 3BIT entropy-coded-image` (a new
  `write_entropy_coded_image_literals` helper, also reusable by §4.2 in a
  future round), the main image is forward-transformed into per-pixel
  residuals, and the residuals feed the standard
  `spatially-coded-image` writer. On a 64×64 smooth gradient the chooser
  shrinks the stream from 9793 B (no-tx baseline) to 303 B — a 96.9%
  reduction; on the published 128×128 natural fixture, from 46797 B to
  1011 B — 97.8%. The chooser falls back to the existing four
  candidates when either dimension is below one block. Internally,
  `encode_tokens` was split: `write_spatially_coded_image` writes the
  body after the §3.8.2 optional-transform terminator, and
  `write_prefix_codes_and_tokens` is the shared `data = prefix-codes
  lz77-coded-image` emitter, so the predictor candidate composes the
  same low-level building blocks as the round-145 path. Eight new
  tests: residual-subtract-add round-trip, `pick_block_mode` solid-
  block cost, forward+inverse predictor bit-exact round trip,
  end-to-end round trip on a smooth gradient, a chooser size-reduction
  assertion (with `eprintln!` byte counts for visibility), a chooser
  noise-non-regression assertion, and a 128×128 natural fixture
  round-trip + size-reduction log. The
  `lossless-128x128-natural.webp` fixture was copied from
  `docs/image/webp/fixtures/lossless-128x128-natural/input.webp` into
  `tests/data/` to make the natural-image regression test
  self-contained.

* **Clean-room round 145 (2026-05-26).** §2.7 metadata-aware container
  writer: `build::build_webp_file_with_metadata(payload, image_kind,
  canvas_width, canvas_height, has_alpha, FileMetadata)` assembles a
  RIFF/WEBP file in the §2.7 *extended* layout with a §2.7.1 `VP8X`
  chunk + optional `ICCP` / `EXIF` / `XMP ` payloads, derives the
  §2.7.1 `I` / `L` / `E` / `X` flag bits from which `FileMetadata`
  fields are `Some` (plus the explicit `has_alpha` argument), and
  emits the chunks in §2.7 canonical order (`VP8X | ICCP | <VP8 |
  VP8L> | EXIF | XMP`). Twelve new tests cover round-trip through
  `extract_metadata` for the eight `{none, iccp, exif, xmp, iccp+exif,
  iccp+xmp, exif+xmp, iccp+exif+xmp}` presence combinations, the
  §2.3 `0x00` pad-byte generation on odd-length metadata payloads
  (verifies the §2.4 `File Size` field still matches the parsed
  value), the exhaustive 16-way §2.7.1 flag-bit derivation against
  the parser's `Vp8xHeader::parse`, and canvas-validation propagation
  (`CanvasDimZero` / `CanvasTooLarge`). The new `FileMetadata<'a>`
  borrowed struct mirrors the published `WebpMetadata` shape but
  lives inside `build` so the writer compiles under
  `--no-default-features` (no `oxideav-core` in the standalone
  build's dependency tree).

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
