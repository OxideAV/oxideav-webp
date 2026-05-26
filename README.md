# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM + ANMF).

## Status — 2026-05-27 (clean-room round 159)

**Round 159 gives the VP8L §4.1 spatial-predictor forward transform
an entropy-image-aware tie-break on its per-block mode chooser.**
`build_predictor_image` now threads the immediately-prior neighbour
block's chosen mode (left neighbour in the current row, top
neighbour for the left-column blocks) into a new
`pick_block_mode_with_hint` so that when multiple modes tie on the
§4.1 residual-magnitude proxy, the chooser prefers the neighbour's
mode over the otherwise-lowest tied mode. The swap fires *only* on
cost-equal modes, so the per-pixel residuals are identical to the
round-158 baseline and decode round-trips remain bit-exact for every
input.

RFC 9649 §3.5 already authorises this choice ("the transform data
can be decided based on entropy minimization"). The predictor
sub-image is written as a §7.2 `entropy-coded-image`, so adjacent
blocks carrying the same mode value lower that sub-image's symbol
entropy and the bytes the prefix-code writer emits for it. The
headline structural win is captured by the strict-beat sweep used
by
`round_159_predictor_candidate_strictly_beats_no_hint_on_some_fixture`:
a 48×48 image whose top-left 8×8 region carries an asymmetric
perturbation pushing mode 11 to strict best, while the remaining 8
blocks are filled with a constant ARGB value so every mode 1..=13
ties at zero residual cost. The predictor sub-image collapses
from a two-symbol `[11, 1, 1, 1, 1, 1, 1, 1, 1]` to the
single-symbol `[11, 11, 11, 11, 11, 11, 11, 11, 11]` and the §4.1
predictor-candidate stream shrinks by 1–2 B: the sub-image
switches from a two-entry prefix code to the §3.7.2.1.1
single-symbol-0 form. The savings scale with the number of
otherwise-tied blocks that pick up the propagated hint.

Five new tests cover the contract:
`round_159_pick_block_mode_with_hint_swaps_on_tie` (on a solid-fill
8×8 block where modes 1..=13 all tie at minimal residual cost, a
`Some(other)` hint swaps the picked mode from the lowest-tied mode
to the preferred mode);
`round_159_pick_block_mode_with_hint_keeps_best_when_hint_worse`
(on a 2-D ramp `pixels[y, x] = (x + 2y) & 0xff` where the L-based
modes are strictly best, a hint pointing at a strictly-worse mode
is ignored);
`round_159_predictor_image_tie_break_is_cost_neutral` (across a
fixture matrix of 5 shapes × 2 fixtures, every block's pre- and
post-r159 chosen modes have identical residual cost — the
invariant guaranteeing decode bit-equivalence);
`round_159_predictor_chooser_never_regresses` (across 6 shapes × 3
fixtures the post-r159 chooser's output is `<=` the pre-r159
chooser's output, with a round-trip via `decode_lossless_image` on
every fixture); and the strict-beat sweep above (across 12 seeded
perturbations the test finds at least one fixture with a strictly-
smaller distinct-mode count AND a strict byte reduction, printing
the byte delta for the round report). Spec source: RFC 9649 §3.5
(transform-data entropy minimization), §4.1 (predictor sub-image
is one ARGB pixel per `(1 << size_bits)`-pixel block with the mode
packed into the green channel), and §7.2 (`predictor-image = 3BIT
entropy-coded-image`).

Still lacks (post-round-159): the per-block mode-cost proxy is now
neighbour-aware but still uses a folded-residual L1 magnitude (no
explicit Huffman bit-cost model on the residual stream); the
tie-break is strict (no slack-cost variant that would trade a
small residual delta for a larger sub-image entropy drop); deeper
LZ77 lazy depths beyond round-158's three-position remain future
work; and VP8 lossy encode is still entirely absent.

## Status — 2026-05-27 (clean-room round 158)

**Round 158 widened the VP8L §5.2.2 LZ77 backward-reference matcher to
three-position lazy matching.** The matcher in `tokenize_lz77` now
extends the round-157 two-position look-ahead with a third
look-ahead position at `pos + 3`. After finding the best match across
`(L_a at pos, L_b at pos + 1, L_c at pos + 2)`, the matcher also
probes `pos + 3` for an `L_d > max(L_a, L_b, L_c)`; when the depth-3
probe wins, three literals (`pixels[pos]`, `pixels[pos + 1]`, and
`pixels[pos + 2]`) are emitted and the longer match starting at
`pos + 3` is taken in place of the depth-2 choice. This recovers a
*third-order* strict-greedy trap that the round-157 depth-2 matcher
could not escape — three consecutive short matches at `pos`,
`pos + 1`, `pos + 2` together blocking a strictly longer match at
`pos + 3`.

Decoder output is bit-identical for any input — only the token
*partition* changes — so the entire existing test suite (now 365
unit tests) continues to round-trip unchanged. The hash-chain insert
bookkeeping now also deduplicates the `pos + 2`-insert that the
depth-3 probe performs, along with the existing `pos`-insert
(depth-1 probe) and `pos + 1`-insert (depth-2 probe), so the
post-match chain walk never double-inserts. The internal
`tokenize_lz77_inner` `lazy_depth: u32` toggle now accepts `3`:
`0` = strict-greedy r155 baseline, `1` = depth-1 round-156, `2` =
depth-2 round-157, `3` = depth-3 round-158 (now the production
default). Round-158 A/B regression tests can build all four
baselines on the same fixture.

The headline structural win is captured by the hand-crafted
four-anchor depth-3 trap fixture: where strict greedy AND the
round-156 depth-1 matcher AND the round-157 depth-2 matcher all emit

    Copy{length=4, dist=33}, Copy{length=8, dist=15}, ...   (2 copies)

covering the trap span (33..45), the round-158 depth-3 matcher emits

    Literal(P), Literal(Q), Literal(R), Copy{length=9, dist=15}, ...
                                                              (1 copy)

over the same span. Net effect: one fewer Copy token at +1 overall
token cost (the extra literal). The longer-match-with-shorter-distance
partition compresses through the §3.7 prefix codes at lower bit cost
on inputs where the trap pattern recurs (three-pixel-run-difference
regions interrupting an underlying long repeat — exactly the case
round-157 could not escape).

Three new tests cover the contract:
`round_158_depth3_lazy_match_round_trips_through_decoder` (a noisy
96×16 fixture round-trips end-to-end via `decode_lossless_image` and
the direct `encode_argb_literals_with_width` path, catching
bookkeeping bugs in the new depth-3 insert/skip dedup);
`round_158_depth3_lazy_match_strictly_beats_depth2_on_trap_fixture`
(the depth-3 trap asserts depth-3 emits strictly fewer Copy tokens
than ALL of the strict-greedy, depth-1, and depth-2 baselines, with
`lazy3_copies = 1 < lazy2_copies = 2 = lazy1_copies = 2 =
greedy_copies = 2`); and
`round_158_depth3_never_increases_token_count_over_depth2` (across
8 shapes × 3 fixture families the depth-3 token count is structurally
`<=` the depth-2 token count, with a defensive round-trip on every
fixture). Spec source: RFC 9649 §5.2.2 / §3.6.2.2 (backward
references; the lazy-match depth is an encoder choice unconstrained
by the format).

Still lacks (post-round-158): the §4.1 per-block mode-cost proxy still
uses a folded-residual magnitude rather than accounting for the
entropy-image bit cost (an r155 follow-up), the LZ77 lazy depth is
now fixed at three-position (deeper look-aheads — depth-4+, or
optimal-parsing — remain future work), and VP8 lossy encode is still
entirely absent.

## Status — 2026-05-27 (clean-room round 157)

**Round 157 widened the VP8L §5.2.2 LZ77 backward-reference matcher to
two-position lazy matching.** The matcher in `tokenize_lz77` now
extends the round-156 single-position look-ahead with a second
look-ahead position at `pos + 2`. After finding the best match across
`(L_a at pos, L_b at pos + 1)`, the matcher also probes `pos + 2` for
an `L_c > max(L_a, L_b)`; when the depth-2 probe wins, two literals
(`pixels[pos]` and `pixels[pos + 1]`) are emitted and the longer
match starting at `pos + 2` is taken in place of the depth-1 choice.
This recovers a *second-order* strict-greedy trap that the round-156
depth-1 matcher could not escape — a short match at `pos` AND a
short match at `pos + 1` together blocking a strictly longer match at
`pos + 2`.

Decoder output is bit-identical for any input — only the token
*partition* changes — so the entire existing test suite (now 362
unit tests) continues to round-trip unchanged. The hash-chain insert
bookkeeping deduplicates both the `pos`-insert that the depth-1 probe
performs and the `pos + 1`-insert that the depth-2 probe performs, so
the post-match chain walk never double-inserts. The internal
`tokenize_lz77_inner` toggle is widened from `bool` to `u32`:
`0` = strict-greedy r155 baseline, `1` = depth-1 round-156, `2` =
depth-2 round-157 (now the production default). Round-157 A/B
regression tests can build all three baselines on the same fixture.

The headline structural win is captured by the hand-crafted
three-anchor depth-2 trap fixture: where strict greedy AND the
round-156 depth-1 matcher both emit

    Copy{length=4, dist=25}, Copy{length=5, dist=13}, ...   (2 copies)

across the trap span, the round-157 depth-2 matcher emits

    Literal(P), Literal(Q), Copy{length=7, dist=13}, ...   (1 copy)

over the same span. Net effect: one fewer Copy token at +1 overall
token cost (the extra literal). The longer-match-with-shorter-distance
partition compresses through the §3.7 prefix codes at lower bit cost
on inputs where the trap pattern recurs (natural-image regions where
two-pixel run differences interrupt a longer underlying repeat —
exactly the case round-156 could not escape).

Three new tests cover the contract:
`round_157_depth2_lazy_match_round_trips_through_decoder` (a noisy
80×16 fixture round-trips end-to-end via `decode_lossless_image` and
the direct `encode_argb_literals_with_width` path, catching
bookkeeping bugs in the new depth-2 insert/skip dedup);
`round_157_depth2_lazy_match_strictly_beats_depth1_on_trap_fixture`
(the depth-2 trap asserts depth-2 emits strictly fewer Copy tokens
than BOTH the strict-greedy and depth-1 baselines, with `lazy2_copies
= 1 < lazy1_copies = 2 = greedy_copies = 2`); and
`round_157_depth2_never_increases_token_count_over_depth1` (across
8 shapes × 3 fixture families the depth-2 token count is structurally
`<=` the depth-1 token count, with a defensive round-trip on every
fixture). Spec source: RFC 9649 §5.2.2 / §3.6.2.2 (backward
references; the lazy-match depth is an encoder choice unconstrained
by the format).

## Status — 2026-05-27 (clean-room round 156)

**Round 156 added single-position lazy matching to the VP8L §5.2.2
LZ77 backward-reference matcher.** The matcher in `tokenize_lz77` now
probes `pos + 1` after finding a match `(L_a, _)` at `pos`; if the
look-ahead yields a strictly longer match `L_b > L_a`, the pixel at
`pos` is emitted as a literal and the longer match from `pos + 1` is
taken in place of the greedy match. This recovers the classic LZ77
strict-greedy trap where a short match at `pos` blocks a much longer
match at `pos + 1`.

Decoder output is bit-identical for any input — only the token
*partition* changes — so the entire existing test suite continues to
round-trip unchanged. The hash-chain insert bookkeeping deduplicates
the `pos`-insert that the lookahead probe performs so the greedy
branch does not double-insert. The refactor exposes an internal
`tokenize_lz77_inner(pixels, lazy: bool)` so round-156 A/B regression
tests can build the strict-greedy r155 baseline alongside the
round-156 lazy stream on the same fixture.

The headline structural win is captured by the hand-crafted dual-chain
trap fixture: where strict greedy emits

    Copy{length=4, dist=17}, Copy{length=7, dist=11}    (3 copies)

across an 11-pixel span, lazy emits

    Literal(A), Copy{length=10, dist=11}                (2 copies)

over the same span. Net effect: one fewer Copy token at parity overall
token count, and the longer-match-with-shorter-distance partition
should encode through the §3.7 prefix codes at lower bit cost on
inputs where the trap pattern recurs (natural-image regions where
inter-row repeats are interrupted by a single-pixel difference, exactly
the case strict greedy can't escape).

Three new tests cover the contract:
`round_156_lazy_match_round_trips_through_decoder` (a noisy 64×16
fixture round-trips end-to-end via `decode_lossless_image` and the
direct `encode_argb_literals_with_width` path, catching insert-
bookkeeping bugs),
`round_156_lazy_match_strictly_beats_greedy_on_trap_fixture` (the
hand-crafted dual-chain trap asserts lazy emits strictly fewer Copy
tokens — `lazy_copies = 2 < greedy_copies = 3`), and
`round_156_lazy_never_increases_token_count` (across 8 shapes ×
3 fixture families the lazy token count is structurally `<=` the
greedy token count, guarding against future off-by-one regressions
in the lookahead bookkeeping). Spec source: RFC 9649 §5.2.2 / §3.6.2.2
(backward references; the lazy-match strategy is an encoder choice
unconstrained by the format).

Still lacks (post-round-156): the §4.1 per-block mode-cost proxy still
uses a folded-residual magnitude rather than accounting for the
entropy-image bit cost (an r155 follow-up), the LZ77 lazy depth is
fixed at single-position (no two-position lookahead yet), and VP8
lossy encode is still entirely absent.

## Status — 2026-05-26 (clean-room round 155)

**Round 155 added a `size_bits` two-value sweep to the VP8L §4.1
spatial-predictor candidate**, mirroring the round-147 §4.2
color-transform shape that has been in place for ~8 rounds.

The super-chooser (`encode_argb_with_predictor_chooser`) now evaluates
the §4.1 predictor candidate at two `size_bits` values:

* the default `DEFAULT_PREDICTOR_SIZE_BITS = 4` (16×16-pixel blocks →
  per-region predictor-mode granularity, good for images whose
  best-mode varies spatially); and
* a maximal single-block transform whose `size_bits` is promoted up
  to 9 so that `1 << size_bits ≥ max(width, height)` and the §4.1
  sub-resolution predictor image collapses to a 1×1 pixel — the
  cheapest possible §4.1 header. Single-block is best on images
  whose best predictor mode agrees everywhere; per-region wins on
  images whose best-mode varies spatially.

Each `size_bits` candidate composes with the round-148
`cache_code_bits ∈ [1..11]` plus disabled-cache sweep, so the §4.1
predictor branch now covers 24 combinations vs the pre-round-155 12.
Per RFC 9649 §4.1 `size_bits` ranges over `[2..=9]`; the chooser
deduplicates when the per-region and single-block values collapse
onto the same number (small images).

Measured savings (best of the round-155 chooser vs best of the
pre-round-155 chooser):

| fixture | shape | pre-r155 | post-r155 | delta |
|---|---|---|---|---|
| dense xorshift noise | 20×20 | 1329 B | 1323 B | −6 B (−0.45 %) |

The savings are modest because the §4.2 color transform already
covers most of the high-correlation cases where a single global
transform wins; the §4.1 single-block predictor mostly recovers
small wins on dense-residual images where the per-region predictor
sub-image's overhead outweighs the savings from a finer mode-image.

Three new tests cover the contract:
`round_155_predictor_size_bits_sweep_never_regresses` (a fixture
matrix spanning gradient / dense-noise / palette-stripes images
across 8 shapes asserts the round-155 chooser is byte-wise ≤ the
pre-round-155 chooser — the new candidate is a strict superset so
this is a structural guarantee),
`round_155_predictor_size_bits_sweep_strictly_beats_default_on_some_fixture`
(20×20 dense-residual headline savings), and
`round_155_predictor_single_block_round_trips_through_decoder` (the
maximal-single-block stream at the promoted `size_bits = 6` for a
64×16 image still round-trips through `decode_lossless_image`
end-to-end). Spec source: RFC 9649 §4.1 (predictor transform
`size_bits` range `2..=9`).

Still lacks (post-round-155): the savings on the diagnostic noisy
fixtures remain modest because the §4.2 color transform already
captures most of the global-statistic wins on those images; the next
bounded predictor step is a per-block mode-cost proxy that accounts
for the entropy-image bit cost (instead of just folded-residual
magnitude), and VP8 lossy encode is still entirely absent.

## Status — 2026-05-26 (clean-room round 152)

**Round 152 swapped the round-151 mean-green bucketiser inside the
§6.2.2 multi-meta-prefix encoder for a histogram-distance per-region
clusterer.** Each `(1 << prefix_bits)`-square block is now featurised
as a coarse 48-element RGB histogram (16 bins per channel after a
`CLUSTER_BIN_SHIFT = 4` collapse of the 256-value channel range), and
`num_groups` cluster centroids are seeded by a deterministic
farthest-from-already-chosen rule (k-means++-style, no randomness)
then refined by up to 8 Lloyd's iteration passes (early-exit on
no-assignment-change). The final assignment is compacted so the
returned meta-codes always run `0..actual_groups - 1` without gaps
(per RFC 9649 §3.7.2.2.2 `num_prefix_groups = max(entropy image) + 1`,
so a gap would inflate the encoder's per-group prefix-code-table cost
for nothing). Uniform images and inputs whose seeding cannot find
`num_groups` distinguishable centroids collapse to a single-group
degenerate cleanly so the chooser stays at the round-150 baseline.

Measured impact on the diagnostic two-region noisy sweep
(multi-meta-prefix candidate byte cost; best non-degenerate
`(prefix_bits, num_groups)` from the round-151 chooser sweep is
selected for each clusterer):

| shape | mean-green | histogram | delta |
|---|---|---|---|
| 64×64 | 8944 B | 8730 B | −214 B (−2.39%) |
| 128×128 | 35049 B | 33264 B | −1785 B (−5.09%) |
| 64×128 | 17640 B | 16903 B | −737 B (−4.18%) |
| 256×256 | 139497 B | 131580 B | −7917 B (−5.68%) |

On a four-region "mean-collision" fixture (designed so per-block
green means agree across regions whose full per-channel distributions
diverge — exactly the failure mode the histogram path is meant to
fix) the mean-green clusterer collapses to a single group at every
swept configuration, while the histogram clusterer finds a
non-degenerate partition.

The multi-prefix candidate still does not beat the round-150
super-chooser on these synthetic fixtures (the LZ77 + predictor +
color-cache stack is very strong on uniform-noise inputs), but the
4–6 % narrower gap puts the candidate within striking distance on
larger natural images where per-region adaptation has more to capture
than synthetic uniform-noise PRNGs can simulate.

Five new clusterer tests cover the histogram path:
`histogram_clusterer_separates_blocks_sharing_a_mean` (a bimodal-vs-
flat fixture mean-green could not split — both regions share mean ≈
128 but the histogram clusterer separates them),
`histogram_clusterer_is_deterministic` (same input → same codes),
`histogram_clusterer_collapses_on_uniform_image`,
`histogram_clusterer_num_groups_one_returns_all_zeros` (short-circuit
for the trivial single-group case), and
`histogram_clusterer_returns_compact_group_ids` (no gaps in the
returned meta-code range). The existing
`meta_prefix_clusterer_splits_two_region_bimodal_fixture` test was
retargeted at the new clusterer and still asserts the top-vs-bottom
split on the headline bimodal image. Two regression-bench tests
(`histogram_clusterer_reduces_mp_bytes_on_two_region_sweep` and
`histogram_clusterer_reduces_mp_bytes_on_mean_collision_sweep`)
sweep `(prefix_bits, num_groups)` across both clusterers and assert
the histogram path never regresses.

Still lacks (post-round-152): the histogram-distance clusterer brings
the multi-meta-prefix candidate within 4–6 % of winning on the
diagnostic noisy fixtures but does not flip the chooser on them; the
remaining gap is dominated by the cost of N additional 280-symbol
prefix-code tables (a future round can stack the §4.1 / §4.2
transforms atop the multi-prefix path to amortise that overhead), and
VP8 lossy encode is still entirely absent.

## Status — 2026-05-26 (clean-room round 151)

**Round 151 added the §6.2.2 multi-meta-prefix (entropy-image) encoder
to the VP8L encoder.** The encoder now exposes a new super-chooser
candidate that emits the §6.2.2 *multi-prefix-code-group* shape:
meta-prefix bit `%b1`, 3-bit `prefix_bits - 2`, an entropy-coded
sub-resolution image carrying one meta-prefix code per
`(1 << prefix_bits)`-square block, `N` prefix-code groups (5 prefix
codes each), and the LZ77 token stream emitted with each token's
symbols under the prefix-code group selected by its start pixel's
block. `encode_with_meta_prefix` takes `prefix_bits`, `num_groups`,
and `cache_code_bits`; `sweep_meta_prefix_candidate` sweeps
`prefix_bits ∈ {4, 5, 6, 7}` (16/32/64/128-pixel blocks) ×
`num_groups ∈ [2..4]` × the round-148 `cache_code_bits ∈ [1..11]`
plus disabled-cache baseline and keeps the smallest non-degenerate
stream.

Wire format (§3.8.3 / §7.3 grammar, ARGB role):

```
spatially-coded-image =
  color-cache-info     ; %b0 or (%b1 4BIT)
  meta-prefix          ; %b1 entropy-image  (this candidate)
  data
meta-prefix-multi =
  %b1                  ; multi-group flag
  3BIT                 ; prefix_bits - 2 (in [0..7] → prefix_bits in [2..9])
  entropy-coded-image  ; sub-resolution entropy image (red+green = meta-code)
data =
  prefix-codes         ; num_prefix_groups × 5 prefix codes
  lz77-coded-image     ; per-token group selected by start pixel's block
```

Round-trip integrity covered across `prefix_bits ∈ {4, 5, 6, 7}` × 2/3/4
groups by the new `meta_prefix_*_round_trip_through_decoder` test set
(every combination decodes back to the exact input pixels via
`vp8l_transform::decode_lossless` and the top-level `decode_webp`).
The clusterer (`cluster_blocks_by_mean_green`) bucketises blocks by
mean-green value into equal-width groups; uniform images and images
too small for the requested block count return `None` so the chooser
stays at the single-group baseline. On the synthetic two-region
fixtures the multi-meta-prefix candidate stays larger than the
single-group baseline (the cost of N additional 280-symbol prefix-code
tables — typically thousands of bytes each — dominates the per-region
savings on small to mid-size images), so the chooser correctly keeps
the round-150 pick; the candidate's value is structural — the
round-151 encoder is now spec-conformant for any future per-region
clustering improvement to plug into without changing the on-wire
serialiser.

Still lacks: smarter per-region clustering (the round-151 mean-green
bucketer is a stand-in; a histogram-distance partitioner would shrink
the per-group code overhead enough to flip the chooser on natural
images), and VP8 lossy encode.

## Status — 2026-05-26 (clean-room round 150)

**Round 150 added the §4.4 color-indexing transform forward pass to the
VP8L encoder.** The encoder now evaluates a new candidate alongside the
round-149 super-chooser set: when an O(N) palette probe confirms the
image has ≤ 256 unique ARGB values, `encode_with_color_indexing` builds
a sorted palette (sorted ARGB-numerically so the §4.4 subtraction-coded
color-table deltas concentrate near zero), replaces every pixel with
its palette index, bundles indices into one byte per the §4.4 table
(`width_bits = 3 / 2 / 1 / 0` for palettes of 1..=2 / 3..=4 / 5..=16 /
17..=256 entries — packing 8 / 4 / 2 / 1 indices into each green byte
respectively), and feeds the bundled-width image to the standard
`spatially-coded-image` writer. The candidate uses the round-148
`cache_code_bits ∈ [1..11]` sweep plus the disabled-cache baseline and
is cross-compared against every other candidate; the smallest stream
wins.

Wire format (§3.8.2 / §7.2 grammar):

```
optional-transform =
  %b1                  ; transform present
  %b11                 ; type ColorIndexing = 3
  8BIT                 ; color_table_size - 1
  entropy-coded-image  ; subtraction-encoded palette (height = 1)
  %b0                  ; end of optional-transform list
spatially-coded-image  ; packed indices at packed_width
```

Headline measurement (encoder-stream byte count, identical pixel
content, round-149 baseline vs round-150 color-indexing-aware chooser):

| Fixture                              | round-149 baseline | round-150 chooser | Δ       |
|--------------------------------------|--------------------:|-------------------:|--------:|
| 64×32 binary row-rotation (2-color)  | 73 B               | 62 B              | -15.1 % |

The §4.4 path doesn't dominate every palette image (the §5.2.3 color
cache + LZ77 already crunch random binary content to ~1 bit/pixel), but
it wins cleanly on palette-ish content with horizontal coherence — the
bundling drops the entropy stage's symbol count by 2..8× and amortises
the small palette-table overhead. Round-trips through
[`decode_webp`](src/lib.rs) and `vp8l_transform::decode_lossless` remain
bit-exact across all four `width_bits` regimes (2-color, 4-color,
16-color, 64-color round-trips covered by the new
`color_indexing_round_trip_across_all_width_bits_regimes` test).

Still lacks: §6.2.2 multi-meta-prefix (entropy image) encoding (round
151 lands the encoder; per-region clustering refinement remains), and
VP8 lossy encode.

## Status — 2026-05-26 (clean-room round 149)

**Round 149 added the §3.7.2.1.1 simple code length code chooser to the
VP8L encoder.** Previously every prefix code went through
`write_normal_code_lengths` (§3.7.2.1.2 *normal code length code*) — the
general form that always pays the 1-flag + 4-`num_code_lengths` +
3-bit-per-CLC + 1-`max_symbol`-gate header tax (≥ 18 bits, ≥ 58 bits
when the literal length table has more than one length value). The new
chooser in `WriteCode::write_code_lengths` recognises the simple form's
two qualifying shapes (1 or 2 used symbols, each at length 1, in
`[0..255]`), computes the exact bit-cost of both forms
(`simple_form_bits` and `normal_form_bits`), and emits whichever is
cheaper. The simple form costs as little as 4 bits (1 symbol with value
in `[0..1]`), making it a dramatic win on the bulk of single-leaf
prefix codes that arise naturally in WebP streams: the empty distance
code on images with no LZ77 matches, the per-channel literal codes on
solid blocks, and the alpha code on opaque images.

Headline measurements (encoded file byte count, identical pixel
content, round-148 chooser vs round-149 chooser):

| Fixture                                  | round 148 | round 149 | Δ          |
|------------------------------------------|----------:|----------:|-----------:|
| 1×1 opaque                               | 174 B     | 32 B      | -81.6 %    |
| 32×32 solid gray                         | 174 B     | 68 B      | -60.9 %    |
| 16×16 four-band gradient                 | 328 B     | 80 B      | -75.6 %    |
| 8×8 two-alpha-value                      | 178 B     | 76 B      | -57.3 %    |

The chooser also propagates through the super-chooser's 12 candidate
streams (no-tx, subtract-green, predictor, color-transform × cache
sweep), so the candidate-cheapest pick now reflects the smaller-tax
simple-form costs as well. Round-trips through
[`decode_webp`](src/lib.rs) remain bit-exact — the decoder's
`vp8l_prefix::read_simple_code_lengths` (round 104) already handles
both forms.

Still lacks: §3.8.2 color-indexing encoding (predictor + color +
subtract-green are wired; round 150 fills this gap), and VP8 lossy
encode.

## Status — 2026-05-26 (clean-room round 148)

**Round 148 added the §5.2.3 `color_cache_code_bits` sweep to the VP8L
encoder chooser.** Previously every cache-enabled candidate locked at
`DEFAULT_COLOR_CACHE_BITS = 8` (a 256-entry cache), giving the §5.2.3
trade-off only two effective positions: disabled or 256 entries. The
new `select_best_cache_bits` helper sweeps the disabled-cache baseline
plus every value in the §5.2.3-allowed `[1..11]` range (2..=2048-entry
caches) for each base candidate: the no-tx and subtract-green literals
candidates in `encode_argb_literals_with_width`, the §4.1 predictor
candidate, and each color-transform `size_bits` candidate (per-region
+ single-block) in `encode_argb_with_predictor_chooser`. The sweep is
non-monotonic — narrow caches win on small-palette payloads (fewer
wasted alphabet slots), wide caches win on photo-like payloads (fewer
hash collisions), and the disabled-cache baseline wins on noise (no
`%b1 4BIT` header tax, no GREEN-alphabet growth from `280` to
`280 + (1 << code_bits)`).

Headline measurements (encoder-stream byte count, identical pixel
content, round-121 hardcoded-8 chooser vs round-148 sweep):

| Fixture                                  | hardcoded-8 | round-148 sweep | Δ          |
|------------------------------------------|------------:|----------------:|-----------:|
| 32×32 16-color pseudo-random palette     | 740 B       | 716 B           | -3.2 %     |

Sample of `code_bits` choices the sweep makes on contrasting payloads
(none of these collapses to the round-121 hardcoded `8`):

| Fixture                                  | sweep picks |
|------------------------------------------|------------:|
| 32×32 4-color pseudo-random palette      | `code_bits = 3` (8-entry cache)   |
| 64×64 32-color pseudo-random palette     | `code_bits = 9` (512-entry cache) |
| 64×64 256-color pseudo-random palette    | `code_bits = 9` (512-entry cache) |

Bit-exact round trips hold through [`decode_webp`](src/lib.rs) /
[`decode_lossless_image`](src/lib.rs) — the sweep only changes the
chosen `color_cache_code_bits` field in the §3.8.3 `color-cache-info`
header; the decoder reads the same `[1..11]` range it already accepted.

Still lacks: §3.8.2 color-indexing encoding (predictor + color +
subtract-green are wired), and VP8 lossy encode.

## Status — 2026-05-26 (clean-room round 147)

**Round 147 added the §3.5.2 / §4.2 color-transform forward pass to the
VP8L encoder.** The encoder now evaluates four new candidates alongside
the round-146 set: the §3.5.2 color transform with two `size_bits`
values (`4` → 16×16 per-region blocks, and the maximal single-block
size that collapses the entire image into one CTE), each with and
without a §5.2.3 color cache. For each block, `pick_block_cte` runs an
exact per-axis greedy sweep over a 25-entry candidate grid
(`±0..±96` with fine resolution near zero) picking the
`(green_to_red, green_to_blue, red_to_blue)` triple that minimises a
residual-magnitude proxy. The per-axis greedy is exact because the
§3.5.2 cost decomposes additively across channels (green is untouched,
red depends only on `green_to_red`, blue depends additively on
`(green_to_blue, red_to_blue)`). The sub-resolution color image is
written as a §7.2 `color-image = 3BIT entropy-coded-image` (re-using
the round-146 `write_entropy_coded_image_literals` helper), the main
image is forward-transformed into per-pixel red/blue residuals, and
the residuals feed the existing `spatially-coded-image` writer. The
chooser falls back to the round-146 six candidates when either
dimension is below one block.

Headline measurements (encoder-stream byte count, identical pixel
content, round-146 baseline vs round-147 color-transform-aware chooser):

| Fixture                                        | round-146 baseline | round-147 chooser | Δ        |
|------------------------------------------------|--------------------:|-------------------:|---------:|
| 128×128 varying-slope channel-correlated noise | 47 636 B            | 41 399 B           | -13.1 %  |
| 128×128 natural fixture (published)            | 1 011 B             | 1 011 B            | 0.0 %    |

On natural images the round-146 predictor candidate already wins
outright at 1011 B (-97.8 % vs the no-tx baseline of 46 797 B) and
the new color candidate doesn't beat it — the chooser correctly
keeps the predictor pick, no regression. The chooser is also non-
regressing on uncorrelated noise (neither the predictor nor the
color transform produces a smaller stream than literals when no
neighbour or channel-correlation helps). Bit-exact round trips hold
through [`decode_webp`](src/lib.rs) / [`decode_lossless_image`](src/lib.rs)
for every fixture above.

Still lacks: §3.8.2 color-indexing encoding (predictor + color +
subtract-green are wired), and VP8 lossy encode.

## Status — 2026-05-26 (clean-room round 146)

**Round 146 added the §4.1 spatial-predictor forward transform to the VP8L
encoder.** The encoder now evaluates two new candidates alongside the
existing `(no-tx | subtract-green) × (no-cache | cache)` set: the §4.1
predictor transform with and without a §5.2.3 color cache. For each
`(1 << size_bits)`-pixel square block (default `size_bits = 4` →
16×16 blocks), `pick_block_mode` walks all 14 §4.1 prediction modes
`0..=13` (`Black`, `L`, `T`, `TR`, `TL`, four `Average2(..)`
combinations, `Select(L, T, TL)`, two `ClampAddSubtract..(..)` forms)
and picks the mode minimising a per-channel `|residual|` proxy folded
onto `[-128, 127]`. The chosen modes are packed into a sub-resolution
predictor image (one ARGB pixel per block, mode in the green channel)
that is written as a §7.2 `predictor-image = 3BIT entropy-coded-image`;
the main image is then forward-transformed into per-pixel residuals
which feed the existing `spatially-coded-image` writer. The chooser
falls back to the four pre-round-146 candidates when either dimension
is below one block.

Headline measurements (encoder-stream byte count, identical pixel
content, no-tx baseline vs round-146 predictor-aware chooser):

| Fixture                                    | no-tx baseline | predictor chooser | Δ        |
|--------------------------------------------|----------------:|-------------------:|---------:|
| 64×64 smooth gradient                      | 9 793 B         | 303 B              | -96.9 %  |
| 128×128 natural fixture (published)        | 46 797 B        | 1 011 B            | -97.8 %  |

The chooser is non-regressing on uncorrelated noise (the predictor
candidate never produces a smaller stream than literals when no
neighbour predicts the next pixel), and bit-exact round trips hold
through [`decode_webp`](src/lib.rs) / [`decode_lossless_image`](src/lib.rs)
for every fixture in the table above. Internally `encode_tokens` was
split into `write_spatially_coded_image` (the body after the §3.8.2
optional-transform terminator) and `write_prefix_codes_and_tokens`
(the shared `data = prefix-codes lz77-coded-image` emitter), and the
new `write_entropy_coded_image_literals` helper writes a §7.3
`entropy-coded-image` for the predictor sub-image — these building
blocks will also serve the §4.2 color transform encoder in a future
round.

Still lacks: §3.8.2 color-transform / color-indexing encoding
(predictor + subtract-green are wired); color transform later landed
in round 147 — see the round-147 status block at the top.

## Status — 2026-05-26 (clean-room round 145)

**Round 145 added the §2.7 metadata-aware container writer
`build::build_webp_file_with_metadata`.** Given a `VP8 ` / `VP8L`
payload, canvas dimensions, an `has_alpha` flag, and a
`build::FileMetadata { iccp, exif, xmp }` triple of optional borrowed
byte slices, the writer assembles a §2.7 extended-layout `.webp`:

```text
RIFF | <File Size LE u32> | WEBP | VP8X | ICCP? | <VP8 | VP8L> | EXIF? | XMP ?
```

with the §2.7.1 `I` / `L` / `E` / `X` flag bits derived from which
`FileMetadata` fields are `Some(..)` (plus the explicit `has_alpha`)
and the chunks emitted in §2.7 canonical order. Twelve new tests
cover the eight `{none, iccp, exif, xmp, iccp+exif, iccp+xmp,
exif+xmp, iccp+exif+xmp}` presence combinations as a round trip
through the existing `extract_metadata` reader, plus the §2.3
`0x00` pad-byte path on odd-length metadata payloads (a 16-way
exhaustive sweep through the parser's `Vp8xHeader::parse` confirms
the flag-bit derivation). The borrowed `FileMetadata<'a>` struct
mirrors the published `WebpMetadata` shape but lives inside `build`
so the writer still compiles under `--no-default-features`.

## Status — 2026-05-25 (clean-room round 130)

**Round 130 added a §5.2.2 width-aware distance-code chooser to the VP8L
encoder.** Each backward reference now picks the *smaller* of two valid
distance-code forms:

* The round-119 *scan-line* code `distance_code = D + 120`.
* Any §5.2.2 *distance-map* code `c ∈ 1..=120` whose `(xi, yi) =
  DISTANCE_MAP[c-1]` reconstructs to `D` for the image width
  (`xi + yi * W`, clamped to 1).

The decoder's [`distance_code_to_pixel_distance`](src/vp8l_decode.rs) is
identical for both forms, so the round trip is byte-exact regardless of
which code the encoder emitted. Picking the smaller raw code feeds
[`value_to_prefix`](src/vp8l_encode.rs) through low-prefix slots
(codes `1..=4` use 0 extra bits; code `5` uses 1 extra bit; …) instead
of the high-prefix slots that `D + 120` for typical row distances would
fall into — a row-distance match on a 256-wide image goes from prefix 16
(8-ish bits Huffman + 7 extra) to prefix 0 (1–4 bits Huffman + 0 extra),
shrinking the per-match cost by ~7 bits. The new public helper is
[`pixel_distance_to_distance_code`](src/vp8l_encode.rs); the new
internal entry point is `encode_argb_literals_with_width(pixels,
image_width)`, wired into `encode_vp8l_payload` so every `.webp`
produced via [`encode_webp_lossless`](src/lib.rs) /
[`encode_vp8l_argb`](src/lib.rs) / the animation encoders threads the
actual width into the chooser. The width-less
[`encode_argb_literals`](src/vp8l_encode.rs) is retained for callers
that exercise the entropy stage without spatial structure; it defaults
to width = 1, which disables the optimisation (no distance-map entry
reconstructs typical distances at a single-pixel-wide row).

Headline measurements (encoder-stream byte count, identical pixel
content, width-1 baseline vs round-130 width-aware path):

| Fixture                               | width=1 (round-119) | width-aware (round-130) | Δ      |
|---------------------------------------|---------------------:|-------------------------:|--------|
| 256×256 row-repeating                 | 972 B                | 958 B                    | -1.4 % |
| 128×128 row-correlated                | 522 B                | 519 B                    | -0.6 % |
| 64×64 row-shifted (per-row `(y%4)`)   | 328 B                | 326 B                    | -0.6 % |
| 64×64 photo-like (ramp + small noise) | 3 923 B              | 3 919 B                  | -0.1 % |

Real-world photographic content with many short backward references at
near-row distances compounds the per-emission saving via the §5.2.2
distance-prefix Huffman tree (more frequent low-prefix codes amortise
the table overhead). Round trip is bit-exact across every fixture above
through [`decode_webp`](src/lib.rs) / [`decode_lossless_image`](src/lib.rs).
(Round 146 has since added §4.1 predictor encoding — see the round-146
status block at the top.)

## Status — 2026-05-25 (clean-room round 127)

**Round 127 implemented `AnimFrameMode::Auto` / `::Delta` (lossless
dirty-rect deltas) and §2.7.1.1 canvas compositing on the decode side.**

The animation encoder now offers three modes:

* `Lossless` — full-canvas VP8L keyframe per frame, as before.
* `Delta` (new) — emits only the **dirty rectangle** of each frame
  (bounding box of pixels differing from the previous canvas) as a §2.6
  `VP8L` sub-frame placed at its `(x, y)` offset with `B = 1`
  (overwrite) / `D = 0` (no dispose). The first frame and any frame
  whose dirty rect spans the whole canvas fall back to a full keyframe.
* `Auto` (new) — evaluates both the full-canvas keyframe and the
  dirty-rect sub-frame and emits whichever produces a smaller bitstream.

Headline: a 128×128 frame pair where only an 8×8 block changes
compresses from 87 476 B (all-Lossless) to 43 986 B (Delta or Auto) —
**~50 % size reduction** with a byte-exact round trip. Both modes are
**lossless**; the original lossy-keyframe-vs-inter-frame-delta `Auto`
semantics will return once `oxideav-vp8` ships a real lossy encoder.

Decoder-side, [`decode_webp`](src/lib.rs) now **composites** each `ANMF`
sub-frame onto a shared canvas per RFC 9649 §2.7.1.1, honouring the
per-frame `B` (blending: alpha-blend vs overwrite) and `D` (disposal:
none vs background) bits — including the §2.7.1.1 alpha-blending
formula `blend.A = src.A + dst.A * (1 - src.A / 255)` (8-bit integer
approximation, sRGB space). Returned `WebpFrame.rgba` is now a
full-canvas snapshot per frame (a playback-ready buffer), with
`width` / `height` set to the §2.7.1 `VP8X` canvas dimensions instead
of the per-frame sub-rect dims. `AnimFrame::new`'s default `blend` is
now `Overwrite` so single-frame round-trips remain byte-exact.

Still framing-only: VP8 / VP8L bitstream *encode* (the lossy keyframe
path) and `Auto`/`Delta` evaluation against a lossy candidate.

## Status — 2026-05-25 (clean-room round 124)

**Round 124 wired the §2.5 `VP8 ` (lossy) decode path through the
`oxideav-vp8` sibling crate.** The `VP8 ` chunk payload is routed to
[`oxideav_vp8::decode_vp8`] (now that vp8 0.2 exposes a public `Vp8Error`
at its crate root), which reconstructs the loop-filtered I420 key-frame;
[`vp8_decode::decode_lossy_rgba`](src/vp8_decode.rs) then converts it to
interleaved RGBA with nearest-neighbour chroma up-sampling and the RFC
6386 §9.2 ITU-R BT.601 full-range YCbCr→RGB matrix. `decode_webp` /
`decode_webp_image` now decode both simple-lossy (`VP8 `) and
extended-lossy (`VP8X` + `VP8 `, with optional `ALPH`-over-`VP8 ` alpha)
still images — the previous clean `Unsupported(LossyVp8)` refusal is
gone. `oxideav-vp8` is pulled in with `default-features = false`, so it
does **not** drag `oxideav-core` into the standalone build; the lossy
decode and the `impl From<oxideav_vp8::DecodeError> for WebpError`
adapter are part of the standalone surface. (The API-COMPAT.md
`From<oxideav_vp8::Vp8Error>` adapter is deferred — vp8's `Vp8Error`
umbrella is on vp8 master but not yet on crates.io.) Still framing-only:
VP8 / VP8L bitstream *encode* and animation `ANMF` frames carrying
`VP8 ` lossy sub-chunks.

**Round 121 added the §5.2.1 / §5.2.3 color-cache writer to the VP8L
encoder.** [`encode_argb_literals`](src/vp8l_encode.rs) now evaluates a
256-entry color cache (`color_cache_code_bits = 8`) alongside the no-cache
path and emits whichever is smaller; combined with the round-120
subtract-green chooser the encoder now picks the smallest of all four
`(no-tx | subtract-green) × (no-cache | cache)` candidates. The cache
state is maintained in stream order per §5.2.3 — every emitted ARGB
literal **and** every pixel covered by a §5.2.2 backward-reference copy
is re-inserted at its hashed slot
(`(0x1e35a7bd * argb) >> (32 - code_bits)`), matching the decoder's
[`ColorCache`](src/vp8l_decode.rs) bit-for-bit. When the cache is on, the
§3.8.3 `color-cache-info` header becomes `%b1 8` (1-bit flag + 4-bit
`code_bits`), the GREEN alphabet grows to `256 + 24 + 256 = 536` symbols,
and a literal repeat is written as a single §5.2.3 cache code
(`256 + 24 + index`) instead of four channel literals. Headline: a 32×32
pseudo-random small-palette (8 distinct ARGB colors) image shrinks from
1131 B (no-cache LZ77 + subtract-green chooser) to 622 B (color-cache on),
a ~45 % size reduction. Round trip is bit-exact through
[`decode_lossless_image`](src/lib.rs) on every fixture. The chooser never
regresses (uncorrelated noise stays on the no-cache no-tx path). A new
[`encode_argb_literals_color_cache`](src/vp8l_encode.rs) test-only entry
forces the cache path for the round-121 size-reduction comparison. Lacks
§3.8.2 predictor / color / color-indexing transform encoding.

## Status — 2026-05-24 (clean-room round 120)

**Round 120 added the §3.5.3 / §3.8.2 subtract-green forward transform.**
New [`apply_subtract_green`](src/vp8l_encode.rs) subtracts the green
channel from red and blue per pixel — the exact inverse of the decoder's
existing [`inverse_subtract_green`](src/vp8l_transform.rs).
[`encode_argb_literals`](src/vp8l_encode.rs) now evaluates both the
no-transform and the subtract-green paths per image and emits whichever
is smaller. The §3.8.2 transform header costs three bits (`%b1 %b10`,
transform type 2, no body), so on green-correlated natural-image content
the per-channel red/blue entropy drops sharply; uncorrelated noise
falls back to no-transform (the chooser never regresses). Headline: a
32×32 synthetic green-correlated image compresses from 3243 B (no-tx)
to 2211 B (subtract-green) — a ~32 % size reduction. Round trip is
bit-exact through [`decode_lossless_image`](src/lib.rs). The literal-only
and force-subtract-green paths stay available as
[`encode_argb_literals_only`](src/vp8l_encode.rs) and
[`encode_argb_literals_subtract_green`](src/vp8l_encode.rs) for the
size-comparison tests. Lacks §3.8.2 predictor / color / color-indexing
transform encoding and §5.2.3 color-cache compression. *(Round 121
landed the color-cache writer — see the section above.)*

## Status — 2026-05-24 (clean-room round 118)

**The published-0.1.5 animation-encode API (VP8L path) is restored (round 118).**
On top of the round-115 VP8L encoder and the §2.7.1.1 `ANIM` / `ANMF` framing,
[`build_animated_webp`](src/lib.rs) /
[`build_animated_webp_with_options`](src/lib.rs)`(frames, opts)` assemble a
multi-frame `.webp` (`RIFF`/`WEBP` + `VP8X(A)` + `[ICCP]` + `ANIM` +
`ANMF…ANMF` + `[EXIF]` + `[XMP ]`). Each [`AnimFrame`](src/anim_encode.rs)
(flat RGBA `pixels` + `width`/`height` + even `x`/`y` offset + `duration` +
`blend` + `dispose` + `mode`) is encoded to a §2.6 `VP8L` chunk and wrapped in
the `ANMF` Frame Data; the `VP8X` canvas is sized to cover every frame and the
`A`/`L`/`I`/`E`/`X` flags declare exactly the features present.
[`AnimEncoderOptions`](src/anim_encode.rs) carries `loop_count`,
`background_rgba`, borrowed `metadata`, and a [`DeltaConfig`](src/anim_encode.rs)
(`max_components_override` / `auto_inner_threshold_bytes` /
`msssim_downsample_kernel` builders, [`DownsampleKernel`](src/anim_encode.rs)).
[`AnimFrameMode`](src/anim_encode.rs)`::Lossless` is fully wired; `Auto` /
`Delta` return `WebpError::Unsupported` (VP8 lossy + delta blocked on
`oxideav-vp8`, #1041). [`decode_webp`](src/lib.rs) now assembles animated files
into N `WebpFrame`s (per-frame VP8L decode + optional `ALPH` override), with
`anim_background_rgba` / `anim_loop_count` populated. A standalone test
(`tests/published_anim_api.rs`, runs under `--no-default-features`) covers a
3-frame round trip, options/metadata, blend/dispose/offset, the Auto/Delta
`Unsupported` path, and the `DeltaConfig` builders. **Not yet restored:** the
VP8 lossy `encode_vp8_lossy_*` entry points and the `Auto`/`Delta` animation
modes (lossy path blocked on `oxideav-vp8`).

## Status — 2026-05-24 (clean-room round 117)

**The published-0.1.5 lossless-encode public names are restored (round 117).**
On top of the round-115 in-crate VP8L encoder, the published encode surface
is re-exposed: [`encode_vp8l_argb`](src/lib.rs) /
[`encode_vp8l_argb_with`](src/lib.rs) emit a **bare** §2.6 / §3.4 `VP8L`
bitstream (image-header + image stream, **no** RIFF wrapper) from packed
`width * height` ARGB; the first auto-detects the §3.4 `alpha_is_used` bit,
the second sets it explicitly (the fixed/non-RDO form).
[`encode_vp8l_argb_with_metadata`](src/lib.rs)`(w, h, &argb, has_alpha, &meta)`
emits a complete `.webp`, staying on the simple `VP8L` layout when opaque and
metadata-free, else auto-promoting to the §2.7 extended `VP8X` layout
(`VP8X` + `ICCP` + `VP8L` + `EXIF` + `XMP ` in §2.7 order, flag octet
declaring exactly the present features). [`WebpMetadata`](src/lib.rs)
(borrowed, `::default()`) and [`WebpMetadataOwned`](src/lib.rs) (owned) carry
the ICC / Exif / XMP payloads; the embedded metadata reads back via
[`extract_metadata`](src/lib.rs). The registry gains a `webp_vp8l` encoder
codec ([`CODEC_ID_VP8L`](src/lib.rs)) accepting `Rgba` / `Rgb24` input
(`Rgb24` streamed as opaque), with the dual-API direct factories
[`registry::make_encoder`](src/registry.rs) /
`make_encoder_with_metadata` / `encode_vp8l_frame`. A standalone test
(`tests/published_encode_api.rs`, runs under `--no-default-features`) covers
bare-bitstream shape, layout selection, and metadata round-trip. **Not yet
restored:** the VP8 lossy `encode_vp8_lossy_*` entry points and animation
encode (the lossy path is blocked on `oxideav-vp8`'s `Vp8Error` symbol).

**The published-0.1.5 decode API shape is being restored (round 116).**
The orphan rebuild had invented its own `decode_webp -> Result<Vec<u8>,
Error>` surface; downstream consumers depend on the *published* shape
recorded in [`API-COMPAT.md`](API-COMPAT.md). This round restores it:
[`decode_webp`](src/lib.rs) now returns `Result<WebpImage, WebpError>`,
where [`WebpImage`](src/lib.rs) carries `frames: Vec<WebpFrame>`,
`metadata: WebpFileMetadata`, `anim_background_rgba: Option<[u8; 4]>`, and
`anim_loop_count: Option<u16>`. Each [`WebpFrame`](src/lib.rs) holds a
flat `rgba: Vec<u8>` (`len == width * height * 4`, tightly packed
`[R, G, B, A]`, no stride padding — wraps zero-copy as
`image::ImageBuffer::from_raw`) plus `width`, `height`, `duration_ms`.
[`WebpFileMetadata`](src/lib.rs) exposes `icc` / `exif` / `xmp`, and
[`extract_metadata`](src/lib.rs) reads them without decoding pixels.
Errors collapse to the published [`WebpError`](src/lib.rs)
(`InvalidData` / `Unsupported` / `Eof` / `NeedMore`). The path is built
on the already-rebuilt §4–§6 VP8L decoder: a simple/extended-lossless
file yields a single-frame `WebpImage`; VP8 lossy and animation report
`WebpError::Unsupported` (never faked) until those decoders are rebuilt.
The rebuild's low-level [`decode_webp_image`](src/lib.rs) →
`DecodedWebp` and [`decode_lossless_image`](src/lib.rs) helpers are
unchanged and remain as additional API. A standalone test
(`tests/published_decode_api.rs`, runs under `--no-default-features`)
encodes an in-memory RGBA buffer, decodes via `decode_webp`, and asserts
the round-tripped `WebpFrame.rgba` is byte-exact. **Not yet restored:**
the published VP8 lossy / animation encode entry points and the
`From<oxideav_vp8::Vp8Error>` conversion (blocked — `oxideav-vp8` has not
restored a `Vp8Error` symbol).

**A VP8L lossless encoder landed in round 115.**
[`encode_webp_lossless`](src/lib.rs) takes an interleaved 8-bit RGBA
buffer (`[R, G, B, A]` scan order — the `DecodedWebp::rgba` layout) plus
dimensions and emits a complete `RIFF/WEBP` file carrying a §2.6
simple-lossless `VP8L` chunk. The encoded file decodes back to the exact
input bytes through [`decode_webp`](src/lib.rs) — a pixel-exact round
trip, validated against the `lossless-1x1`, `lossless-32x32-rgba`, and
`lossless-color-indexing-paletted` fixtures (decoded by the independent
decode path, re-encoded, re-decoded, compared byte-for-byte). The encoder
([`vp8l_encode`](src/vp8l_encode.rs)) takes a simple spec-conformant
path: no §3.8.2 transform (pass-through), no §3.8.3 color cache, a single
§3.7.2.2 meta-prefix code. It builds the §3.7.2 canonical prefix codes
per-image from channel frequencies (length-limited ≤ 15-bit Huffman →
`(length, value)`-ordered canonical codes, the exact assignment the
round-104 reader consumes) and writes their lengths with the §3.7.2.1.2
normal code length code.

**Round 119 added §5.2.2 LZ77 backward-reference matching.**
[`encode_argb_literals`](src/vp8l_encode.rs) now runs a hash-chain
matcher ([`Lz77Matcher`](src/vp8l_encode.rs)) before the entropy stage:
4-pixel hashes bucket every position into `2^14` chains (capped at
64 walks per match), repeated pixel runs of `>= 3` pixels (up to the
spec's 4096-pixel limit) become §5.2.2 length + distance backward
references. Length values feed the GREEN alphabet's `256 + length_prefix`
symbols (the §5.2.2 prefix-coding table). Distances are emitted via the
scan-line encoding `distance_code = D + 120` (the spec's §3.6.2.2.1
distance map is an optional decoder convenience for near-pixel offsets;
the `> 120` form is always valid and the in-crate
[`distance_code_to_pixel_distance`](src/vp8l_decode.rs) reconstructs `D`
exactly). The new [`value_to_prefix`](src/vp8l_encode.rs) helper is the
exact inverse of the decoder's [`read_lz77_value`](src/vp8l_decode.rs)
prefix-value transform, round-tripped at every value in `1..=4096`.
A 64×64 image whose rows repeat an 8-color palette compresses from
4758 B (literal-only) to 163 B (LZ77) — a ~97% size reduction; pixels
without exploitable repetition (xorshift noise) come out the same size,
and a solid color costs ~2 bytes extra to encode the length symbol.
The literal-only baseline is retained as
[`encode_argb_literals_only`](src/vp8l_encode.rs) for the size-reduction
comparison test. Output remains spec-valid and round-trip-exact on every
covered case. Lacks §3.8.2 transform encoding (predictor /
subtract-green / color / color-indexing) and §5.2.3 color-cache
compression.

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
frame. As of round 124 a §2.5 `VP8 ` lossy file decodes through
`oxideav-vp8`; an animation / header-only file with no `VP8L`/`VP8 `
image-data chunk is still a clean `oxideav_core::Error::Unsupported`.
The [`extract_lossy_chunk`](src/lib.rs) routing API remains for callers
that want the raw VP8 bitstream slice instead. The
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
[`decode_webp`](src/lib.rs) is the published flat-RGBA entry point (round
116) — it returns a `WebpImage` whose single `WebpFrame.rgba` is the same
flat buffer. As of round 124 a §2.5 `VP8 ` lossy file decodes too,
through the `oxideav-vp8` sibling crate (see the status section above).

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
  present. As of round 124 a `VP8 ` lossy file is decoded via the
  `oxideav-vp8` sibling crate ([`vp8_decode::decode_lossy_rgba`](src/vp8_decode.rs));
  [`extract_lossy_chunk`](src/lib.rs) still exposes the raw `VP8 `
  payload for callers that want to route it elsewhere.
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
| VP8 lossy bitstream decode    | done (r124) — routed through `oxideav-vp8` → BT.601 RGBA |
| VP8 / VP8L bitstream encode   | not yet — payload is opaque input to the builder    |

Test count: **339** (273 unit + 66 integration against the
`docs/image/webp/fixtures/` corpus). Round 124 adds 5 unit tests in
`vp8_decode::tests` (the BT.601 YCbCr→RGB matrix at neutral / primary /
clamped chroma, plus the I420→RGBA flat-buffer + odd-dimension chroma
up-sampling) and rewires the lossy-fixture tests: the round-111
`decode_webp` lossy refusals become round-124 decode-and-assert-dims
checks against the cwebp-encoded `lossy-1x1.webp` (simple) and
`lossy-with-alpha-128x128.webp` (`VP8X` + `ALPH` + `VP8 `, asserting the
alpha plane introduced transparency), the low-level
`decode_webp_image` lossy path, and the registered-decoder lossy decode.
Round 112 adds 10 unit tests inside
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
