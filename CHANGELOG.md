# Changelog

All notable changes to `oxideav-webp` are recorded here.

## [Unreleased]

### Added

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
