# `oxideav-webp` benchmarks

Round-170 (2026-05-27) introduced the first criterion-driven bench
harness for `oxideav-webp` plus the first round of measured
optimizations on the VP8L (lossless) decode + repack hot paths.
This file records the baseline numbers, the profile findings that
drove the optimization choices, and the post-optimization deltas.

All numbers were captured on an `aarch64-apple-darwin` machine (macOS
26.1, M-series CPU) with:

```text
CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
  cargo bench --manifest-path crates/oxideav-webp/Cargo.toml \
    --bench <name> -- --quick
```

`--quick` runs criterion's reduced-sample mode for sub-minute runs;
the medians are still stable to a few percent.

## Bench inventory

| Bench file | Function name | What it measures |
|---|---|---|
| `benches/lossless_encode.rs` | `lossless_encode_rgba_256` | Full RIFF/WEBP encode of a 256×256 RGBA gradient |
| `benches/lossless_encode.rs` | `lossless_encode_natural_128` | Full RIFF/WEBP encode of a 128×128 tile from the in-tree natural-image fixture |
| `benches/lossless_decode.rs` | `lossless_decode_argb_256` | Full RIFF/WEBP decode of the encoded 256×256 gradient |
| `benches/lz77_match.rs`      | `vp8l_lz77_match` | §5.2.2 hash-chain LZ77 matcher over a 4096-pixel synthetic tile |
| `benches/argb_to_rgba.rs`    | `argb_to_rgba` | `Vp8lImage::to_rgba` repack on a 256×256 image |
| `benches/inverse_predictor.rs` | `inverse_predictor_modeN_256x256` | §4.1 inverse predictor on a 256×256 buffer, mode-pinned (N ∈ {11, 12, 13}) |
| `benches/inverse_color.rs` | `inverse_color_256x256_sbN` | §4.2 inverse color transform on a 256×256 buffer, parameterised over `size_bits` (N ∈ {0, 3, 5, 7}) |
| `benches/inverse_color_indexing.rs` | `inverse_color_indexing_256x256_paletteN` | §4.4 inverse color-indexing transform on a 256×256 output buffer, parameterised over palette size (N ∈ {2, 4, 16, 256}) which selects all four `width_bits` bundling levels |
| `benches/inverse_subtract_green.rs` | `inverse_subtract_green_256x256` | §4.3 subtract-green inverse transform on a 256×256 ARGB buffer (deterministic LCG fill) |
| `benches/predictor_subtract.rs` | `predictor_subtract_256x256` | Encoder-side §4.1 per-channel mod-256 residual builder (mirror of decoder's `add_pred`) over a 256×256 ARGB buffer (deterministic LCG fill) |
| `benches/apply_subtract_green.rs` | `apply_subtract_green_256x256` | Encoder-side §4.3 forward subtract-green transform (mirror of decoder's `inverse_subtract_green`) over a 256×256 ARGB buffer (deterministic LCG fill) |
| `benches/inverse_color_table.rs` | `inverse_color_table_paletteN` | §4.4 palette subtraction-decode (cumulative-delta) pass over a `N`-entry palette, parameterised over `N ∈ {2, 16, 256}` to cover the bundling-tier boundaries (smallest, mid-tier, max palette length) |
| `benches/build_code_lengths.rs` | `build_code_lengths_{dense,sparse}_{distance40,literal256,green281,green2328}` | Encoder-side §3.7.2 Huffman code-length builder over a single §3.7.1 prefix-code-group alphabet, parameterised over (a) alphabet size — DISTANCE = 40, RED/BLUE/ALPHA = 256, GREEN at smallest color-cache (281) and largest color-cache (2328) — and (b) frequency-table regime: *dense* (every symbol live, LCG-fill 1..=255) or *sparse* (`sqrt(N)` live symbols, 1/(k+1) Zipf shape). Eight cells total |
| `benches/prefix_from_code_lengths.rs` | `prefix_from_code_lengths_{dense,sparse}_{distance40,literal256,green281,green2328}` | Decoder-side §6.2.1 canonical-table build (`PrefixCode::from_code_lengths`) over the same four §3.7.1 prefix-code-group alphabets and the same two dense / sparse frequency regimes as `build_code_lengths` / `canonical_codes`, with length tables produced by `build_code_lengths` at setup. The per-iteration `Vec` clone (the function takes the table by value) is excluded via `iter_batched` setup. Eight cells total |
| `benches/canonical_codes.rs` | `canonical_codes_{dense,sparse}_{distance40,literal256,green281,green2328}` | Encoder-side §3.7.2 canonical-code-value assignment over the same four §3.7.1 prefix-code-group alphabets and the same two dense / sparse frequency regimes as `build_code_lengths`, sampling only the per-symbol code-value pass (the `build_code_lengths` call that produces the length table is outside the measured interval). Eight cells total |

## Round-170 baseline (pre-optimization)

| Bench | Median |
|---|---|
| `argb_to_rgba` | 126.29 µs |
| `vp8l_lz77_match` | 819.06 µs |
| `lossless_decode_argb_256` | 1.005 ms |
| `lossless_encode_natural_128` | 169.88 ms |
| `lossless_encode_rgba_256` | 1.541 s |

## Profile findings

A `/usr/bin/sample` trace of a driver that loops the public
`decode_webp` entry point (a 64×64 gradient encoded once, decoded
50 000 times) attributed self-time as follows. Numbers are sample
counts out of 2 700 over 4 wall-clock seconds of sampling.

| Rank | Function | Self-time (samples) | % of decode |
|---:|---|---:|---:|
| 1 | `vp8l_transform::inverse_predictor` | 2 163 | ~80% |
| 2 | `meta_prefix::PrefixCodeGroup::read` (in §6.2.2 header) | 124 | ~5% |
| 3 | `vp8l_decode::decode_image` (per-symbol §6.2.3 dispatch) | 125 | ~5% |
| 4 | `vp8l_prefix::PrefixCode::from_code_lengths` | ~50 | ~2% |
| 5 | per-pixel `to_rgba` channel shuffle (in `decode_webp_image`) | ~30 | ~1% |

(The remaining ~7% is RIFF walk + scratch allocations.)

A second sample of a driver that loops `encode_webp_lossless` on the
natural 128×128 tile attributed the encode-side hot spots as:

| Rank | Function | Self-time (samples) |
|---:|---|---:|
| 1 | `vp8l_encode::predictor_at` (forward predictor) | 59 |
| 2 | `vp8l_encode::Lz77Matcher::find` (chain walker) | 56 |
| 3 | `encode_argb_with_predictor_chooser` closure body | 50 |
| 4 | `vp8l_encode::canonical_codes` | 40 |
| 5 | `vp8l_encode::Lz77Matcher::insert` | 40 |

Encode self-time is spread across many candidates (the chooser sweeps
~30 encoding variants per call), so encode wins from a single
hot-loop SWAR rewrite are bounded — encode optimization will need
its own round.

## Optimizations landed (round 170)

The round-170 commit targets the decode hot path identified above:

1. **`vp8l_transform::add_pred` (SWAR).** The §4.1 per-channel
   `r + p` add is now expressed as two masked u32 adds
   (`0x00ff_00ff` low-pair, `0xff00_ff00` high-pair) instead of four
   sequential `u8::wrapping_add` calls + a `pack_argb`. The new
   identity is bit-identical to the original (the spec says add per
   channel mod 256; SWAR with disjoint lane masks gives exactly that).
2. **`vp8l_transform::average2` (SWAR halving-add).** The §4.1
   `Average2(a, b) = (a + b) / 2` per channel is now the standard
   halving-add identity `(a & b) + ((a ^ b) >> 1 & 0x7f7f7f7f)`.
   Bit-identical to the old per-channel `(ca + cb) / 2`.
3. **`vp8l_transform::inverse_subtract_green` (SWAR).** §4.3 add of
   green into red + blue is now one mask + one masked u32 add
   instead of four `u8` operations + a pack. Bit-identical.
4. **`Vp8lImage::to_rgba` (scalar rewrite).** Replaced four
   `Vec::push` per pixel with one pre-sized `vec![0; n*4]` + a
   `chunks_exact_mut(4)` zip over the pixel iterator. Removes the
   per-byte capacity-grow + bounds check; the compiler now
   auto-vectorises the strided byte stores.
5. **`Vp8lImage::to_rgba_simd` (NEW; nightly-only, `simd` feature).**
   `std::simd::u8x16` shuffle that processes 4 ARGB pixels per
   iteration in a single 16-byte load + shuffle + store. Reads
   directly from the `&[u32]` reinterpreted as `&[u8]` (no per-pixel
   `to_le_bytes()` round trip). Byte-identical to the scalar path,
   asserted by `vp8l::tests::to_rgba_simd_matches_scalar_byte_for_byte`.

## Round-170 post-optimization numbers

| Bench | Baseline | Round-170 | Δ |
|---|---:|---:|---:|
| `argb_to_rgba` (scalar) | 126.29 µs | 8.71 µs | **−93.1%** (~14.5×) |
| `argb_to_rgba` (simd) | — | 6.40 µs | **−94.9%** vs. baseline / −27% vs. round-170 scalar |
| `lossless_decode_argb_256` | 1.005 ms | 773 µs | **−23.0%** |
| `vp8l_lz77_match` | 819.06 µs | 812.17 µs | −0.8% (noise) |
| `lossless_encode_natural_128` | 169.88 ms | 176.74 ms | +4% (noise; encoder hot loop is the chooser) |
| `lossless_encode_rgba_256` | 1.541 s | 1.590 s | +3% (noise; same reason) |

The two flagged targets (decode + repack) cleared the 10% bar
comfortably; the LZ77 + encode benches were not the round-170
focus (LZ77 is already a tight hash-chain loop; encode self-time is
spread across the chooser).

## Round-180 (2026-05-29) — predictor border-rule hoist

The round-170 profile attributed ~80% of decode self-time to
`vp8l_transform::inverse_predictor`. The SWAR rewrites of
`add_pred` / `average2` / `inverse_subtract_green` cut the
arithmetic per pixel; the round-180 change cuts the *control flow*
per pixel. Each iteration of the original inner loop ran a
four-arm chain (`x == 0 && y == 0` → `y == 0` → `x == 0` → else)
plus a nested `x == w - 1` check inside the else branch — five
predictable but per-pixel branches. We hoist all of them out so
the predicate is implicit in which loop the iteration is part of:

* `(0, 0)` — one statement, runs once.
* Top row (`y == 0`, `x in 1..w`) — its own loop, always predicts L.
* Left column (`x == 0`, `y in 1..h`) — its own loop, always
  predicts T.
* Interior + right column (`y in 1..h`, `x in 1..w`) — split into a
  branch-free `x in 1..w-1` loop and a one-statement right-column
  case (the §4.1 wraparound `tr = pixels[idx - w - (w - 1)]`).

| Bench | Round-170 | Round-180 | Δ vs. round-170 | Δ vs. baseline |
|---|---:|---:|---:|---:|
| `lossless_decode_argb_256` | 773 µs | **747 µs** | −3.4% | −25.7% |
| `argb_to_rgba` (scalar) | 8.71 µs | 8.56 µs | within noise | −93.2% |

Decode-side win is modest (~3.4%) because (a) for the 256×256
gradient fixture the predictor self-time is amortised against the
prefix-code read in the §6.2 entropy loop, and (b) the branch
predictor already does a good job on the original pattern (`(0, 0)`
is taken exactly once, `y == 0` for one whole row, `x == 0` for one
column per row). The structural win — branch-free interior — is the
foundation for a future round-N SWAR pass over the predictor body
itself (modes 5–13 still do per-channel `u8` arithmetic in
`select` / `clamp_add_subtract_*`); the round-180 split makes those
inner-loop calls auto-vectorisable without re-introducing the border
branches.

Bit-identical to the prior implementation per the new
`inverse_predictor_matches_unsplit_reference_random` test, which
runs seven `(width, height, size_bits)` shapes against an in-test
copy of the original per-pixel reference (including `1×N`, `N×1`,
`2×2`, and `size_bits = 0`, the four boundary regimes where the
split is most subtle).

## Round-194 (2026-05-31) — `Select` algebraic simplification + per-mode bench

The round-180 BENCHMARKS note flagged the `select` /
`clamp_add_subtract_*` per-channel arithmetic inside `predict` as
the next mode-by-mode optimization target. Round 194 lands two
things.

### 1. New per-mode bench harness — `benches/inverse_predictor.rs`

The existing `lossless_decode_argb_256` bench feeds a 256×256
gradient that the encoder's mode chooser resolves with a mix of
modes 0–10, so the heavier arithmetic modes 11 / 12 / 13 are
under-exercised. The new `inverse_predictor` bench builds a
256×256 ARGB residual buffer plus a `size_bits = 0` predictor
image whose green channel is the **same constant mode for every
block**, so every interior-loop pixel exercises one chosen
predictor body. Three runs ship: `mode11`, `mode12`, `mode13`.
This lets future rounds A/B-test per-mode rewrites without
waiting for a real encoder mode pick.

### 2. `Select` (mode 11) algebraic simplification

`Select(L, T, TL)` in the §4.1 reference form computes a 4-channel
`estimate = L + T - TL`, then takes 8 per-channel absolute
differences (`|estimate_c - L_c|` and `|estimate_c - T_c|`, each
summed across the four channels to form `p_L` / `p_T`).
Substituting the definition of `estimate`:

* `estimate_c - L_c = (L_c + T_c - TL_c) - L_c = T_c - TL_c`
* `estimate_c - T_c = (L_c + T_c - TL_c) - T_c = L_c - TL_c`

so the four `estimate_c` additions and the eight
`estimate_c - {L,T}_c` subtractions cancel down to just the two
Manhattan distances. The simplified body computes only
`Manhattan(T, TL)` and `Manhattan(L, TL)` directly — half the
per-pixel arithmetic with the same comparison and the same
tie-break. Bit-identical to the reference form, asserted by
`select_matches_estimate_reference_random` which sweeps 1 024
deterministic LCG `(l, t, tl)` triples + four hand-picked boundary
triples (`tl == l`, `tl == t`, all-equal, and a maximally-
separated triple) against a verbatim copy of the pre-r194
estimate-based body.

| Bench | Pre-r194 | Round-194 | Δ |
|---|---:|---:|---:|
| `inverse_predictor_mode11_256x256` (new) | 597 µs | **484 µs** | **−18.9%** |
| `lossless_decode_argb_256` | 765 µs | **743 µs** | −2.9% |
| `inverse_predictor_mode12_256x256` (new, unchanged impl) | — | 605 µs | n/a (future-round target) |
| `inverse_predictor_mode13_256x256` (new, unchanged impl) | — | 835 µs | n/a (future-round target) |

The mode-11 microbench captures the full ~19% win on a buffer
that's exclusively mode-11; the end-to-end decode bench moves
−2.9% because the 256×256 gradient fixture's mode-11 share is
modest. The mode 12 / 13 micro-numbers are the round-194 baseline
for any future SWAR / lane-parallel experiment on
`clamp_add_subtract_*` — note that a naïve `to_le_bytes()` +
4-iteration `i16` loop **regressed** mode 12 to ~1 250 µs in
exploratory work this round (LLVM already auto-vectorises the
closure-of-four `i32` body well on AArch64), so the next attempt
will need a true SWAR formulation rather than a byte-loop rewrite.

## Round-207 (2026-06-02) — `inverse_color` per-block CTE hoist

The §4.2 inverse color transform reads a `ColorTransformElement`
(three signed-byte coefficients: `green_to_red`, `green_to_blue`,
`red_to_blue`) from a sub-resolution color image and applies them
to every pixel. The CTE is **constant inside each `1 << size_bits`
block**, so the original per-pixel `block_index` recomputation +
`color_image[]` load + three byte extracts are recomputed for every
pixel even though their value is identical across the entire block.

Round-207 hoists that work out: the inner loop is replaced by a
nested block-walk that loads the CTE once at each block boundary
and then iterates the pixels inside the block with the three
coefficients held in registers. Row-base offsets (`row_off = y * w`
and `block_row = (y >> size_bits) * tw`) are also hoisted out of
the x-loop.

The `size_bits == 0` corner (block size 1, one CTE per pixel) is
special-cased to a flat double `for` loop, because the nested
block-walk degenerates into an extra loop layer the optimizer
can't always flatten and would otherwise regress that case.

Bit-identical to the pre-r207 per-pixel form, asserted by
`inverse_color_matches_per_pixel_reference_random` which sweeps
seven `(size_bits, w, h)` configurations (including the
`size_bits = 0` no-op corner, a 1-row image, a 1-column image,
sub-block-sized edge tiles, and a block larger than the image)
against a verbatim copy of the pre-r207 per-pixel body.

### New bench: `inverse_color`

`benches/inverse_color.rs` drives `inverse_color` on a 256×256 ARGB
buffer with a deterministic LCG fill, parameterised over four
`size_bits` ∈ {0, 3, 5, 7}. The color image is sized to match
(`ceil(W / (1 << size_bits))` × `ceil(H / (1 << size_bits))`) and
its CTE bytes are LCG-filled so the signed-delta path actually
runs for every block.

| Bench | Pre-r207 | Round-207 | Δ |
|---|---:|---:|---:|
| `inverse_color_256x256_sb0` | 69.0 µs | **29.6 µs** | **−57.1%** |
| `inverse_color_256x256_sb3` | 70.2 µs | **50.6 µs** | **−27.9%** |
| `inverse_color_256x256_sb5` | 70.0 µs | **23.1 µs** | **−67.0%** |
| `inverse_color_256x256_sb7` | 71.4 µs | **24.4 µs** | **−65.8%** |

The sb0 win comes entirely from hoisting `row_off` and `block_row`
out of the x-loop (the per-pixel work itself is unchanged). For
sb3..sb7 the win scales with block size: amortising one CTE load +
three byte extracts across 64 / 1024 / 16384 pixels per block. sb5
and sb7 plateau at ~24 µs — at that point the per-pixel
`inverse_color_pixel` arithmetic (three signed multiplies + three
arithmetic shifts + the new-red feedback into new-blue) dominates,
not the block-level overhead.

The end-to-end `lossless_decode_argb_256` bench is unchanged
(~643 µs) because the 256×256 LCG-encoded gradient fixture happens
not to elect the color transform on its mode chooser; the win
shows up on natural images and on any fixture whose encoder
selected the §4.2 transform.

## Round-210 (2026-06-02) — `inverse_color_indexing` per-bundle hoist

The §4.4 inverse color-indexing transform replaces each output pixel
with `color_table[green(packed_pixel)]`. For palettes of ≤16 colors
the packed buffer is **bundled**: `count = 1 << width_bits` output
pixels share one packed green byte, with each index occupying `bits
= 8 / count` bits in that byte (LSB-first per §4.4).

The pre-round-210 inner loop recomputed three quantities for every
output pixel even though they are constant across larger units:

* `y * packed_w + x / count` — the packed-row index. Constant across
  each `count`-pixel run; the row base `y * packed_w` is constant
  across an entire row.
* `y * orig_width + x` — the output-row index. The row base
  `y * orig_width` is constant across an entire row.
* `(x % count) * bits` — the field-selector shift. Cycles through
  the same `count` values per bundle (`0, bits, 2*bits, …`).

Round-210 hoists the two row bases out of the x loop and walks the
row as a sequence of `count`-wide bundles: load the packed green
byte once at the bundle boundary, then iterate `count` sub-indices
with a stepping `shift` variable that increments by `bits` each
iteration. The trailing partial bundle at row end (when `orig_width`
is not a multiple of `count`) reuses the inner-bundle walk under a
`min` clamp.

The `width_bits = 0` (no-bundle) path was already tight (a
`zip`-based slice walk with no row arithmetic) and is left
unchanged.

Bit-identical to the per-pixel form, asserted by
`color_indexing_matches_per_pixel_reference_random` which sweeps
nine `(orig_width, height, table_size)` configurations spanning all
four `width_bits` levels, exact-bundle widths, trailing partial
bundles, a single column (entire row falls inside the trailing
partial), a single row, and out-of-range indices that must collapse
to transparent black — against a verbatim copy of the pre-r210
per-pixel body.

### New bench: `inverse_color_indexing`

`benches/inverse_color_indexing.rs` drives `inverse_color_indexing`
on a 256×256 output buffer with a deterministic LCG fill,
parameterised over four palette sizes that select all four
bundling levels:

| Palette size | `width_bits` | `count` (outputs/byte) | `bits` (per index) |
|---:|---:|---:|---:|
| 2 | 3 | 8 | 1 |
| 4 | 2 | 4 | 2 |
| 16 | 1 | 2 | 4 |
| 256 | 0 | 1 (no bundle) | 8 |

| Bench | Pre-r210 | Round-210 | Δ |
|---|---:|---:|---:|
| `inverse_color_indexing_256x256_palette2` | 40.7 µs | **31.6 µs** | **−22.4%** |
| `inverse_color_indexing_256x256_palette4` | 40.7 µs | 39.4 µs | −3.2% |
| `inverse_color_indexing_256x256_palette16` | 40.2 µs | 39.2 µs | −2.6% |
| `inverse_color_indexing_256x256_palette256` | 19.2 µs | 18.6 µs | −2.7% (unchanged path; noise) |

The big win lands on the highest-bundle-count case (palette-2,
`count = 8`) where amortising the packed-row index lookup across
8 output pixels dominates. The lower-bundle cases see only the
row-offset hoist contribution; `(x % 4) * 2` and `(x % 2) * 4` were
already cheap constant-power-of-two operations the optimizer
already folded well. The unbundled path was untouched and the
~2.7% movement on palette-256 is within criterion-`--quick`
sampling noise on the M-series host.

The end-to-end `lossless_decode_argb_256` bench is unchanged because
the 256×256 LCG gradient fixture happens not to elect the §4.4
transform on the encoder's mode chooser; the win shows up on
small-palette images (icons, logos, screenshots with limited color
counts) whose encoder selected color-indexing with `width_bits = 3`.

## Round-217 (2026-06-03) — `inverse_subtract_green` bench coverage

The §4.x transform inventory now exposes a per-pass bench for the only
one that was still un-measured: §4.3 `inverse_subtract_green`. The
round-170 SWAR rewrite collapsed the per-pixel `r += g; b += g` to one
masked add (broadcast `(g << 16) | g` into `(p & 0x00ff_00ff)`, mask
back, OR with `p & 0xff00_ff00`), but no bench was committed at the
time, so subsequent rounds couldn't A/B-test further changes against a
fixed baseline. This round closes that inventory gap.

`benches/inverse_subtract_green.rs` drives `inverse_subtract_green`
on a 256×256 ARGB buffer with the same deterministic LCG fill shape
used by `inverse_color` and `inverse_color_indexing` so the per-pass
numbers are comparable. The §4.3 transform has no tunable parameters
(unlike `inverse_predictor`'s `mode` or `inverse_color`'s `size_bits`),
so one fixed-size run captures its full surface.

| Bench | Round-217 (median) |
|---|---:|
| `inverse_subtract_green_256x256` | **13.7 µs** |

For the comparable 256×256 surfaces, that sits between the §4.4
`inverse_color_indexing_256x256_palette256` (18.6 µs, the simplest
unbundled path) and the §4.2 `inverse_color_256x256_sb5/sb7` numbers
(~23 µs). The round-170 SWAR mask + OR pattern is already as tight as
the spec allows for the spec-prescribed per-pixel work (extract green,
broadcast into r + b lanes, masked add); future optimization rounds
that touch `inverse_subtract_green` (e.g. a `std::simd` lane-parallel
pass for the `simd` feature, mirroring the `to_rgba_simd` precedent)
now have a documented baseline to A/B against. No algorithm change
landed this round.

## Round-224 (2026-06-04) — `predictor_subtract` bench + SWAR experiment

The §4.x inverse-transform inventory has had per-pass benches for every
decoder transform since round 217 closed `inverse_subtract_green`. The
**encoder** side still had only the end-to-end `lossless_encode_*`
benches at the public-API level — the round-170 profile attributed the
#1 encoder self-time slot to the predictor + residual path, but the
residual builder `predictor_subtract` itself was unmeasured at the
per-pass level. Round 224 closes that inventory gap.

### New bench: `predictor_subtract`

`benches/predictor_subtract.rs` drives `predictor_subtract` once per
pixel over a 256×256 ARGB buffer with the same deterministic LCG fill
shape used by the §4.x decoder benches so the per-pass numbers are
visually comparable. The bench accumulates the per-pixel residual into
a `u32` XOR so the loop body cannot be folded away by the optimizer.

`predictor_subtract` is now `pub fn` (previously private) to make it
reachable from `benches/`. Its semantics — the per-channel mod-256
inverse of the decoder's `add_pred` — are unchanged from prior rounds
and are pinned bit-identically against the closure-of-four reference
body by the new
`predictor_subtract_matches_per_byte_reference_random` test (1 024
deterministic LCG `(original, pred)` pairs plus six hand-picked
boundary pairs covering every-channel underflow, every-channel
positive, all-zero, all-`0xff`, and a mixed underflow / positive case).

| Bench | Round-224 (median) |
|---|---:|
| `predictor_subtract_256x256` | **~36 µs** |

(Range across three consecutive `--quick` runs on the same host:
35.9–38.3 µs; criterion `--quick` is sensitive to system load on the
M-series machine.)

For the comparable 256×256 surfaces, that sits between the §4.3
`inverse_subtract_green_256x256` (13.7 µs, the cheapest per-pixel
work — a single masked add) and the §4.4
`inverse_color_indexing_256x256_palette256` (18.6 µs, the cheapest
unbundled path); twice the cost of either is expected because
`predictor_subtract` does four sequential per-channel mod-256
subtracts plus a four-byte reassembly per pixel.

### SWAR-mirror experiment (regressed, body retained)

The decoder-side `add_pred` was rewritten in round 170 as a two-pair
SWAR (`(x & 0x00ff_00ff).wrapping_add(...)` / `(x & 0xff00_ff00)
.wrapping_add(...)`) because addition does not propagate carry across
the zero "guard" bytes when the summand has its high bit masked out.
Round 224 attempted the symmetric subtraction rewrite. Subtraction is
asymmetric: a borrow at the low byte of a lane DOES propagate through
the zero guard byte and corrupt the adjacent lane, so the mirror
rewrite biases the minuend with a `0x0100` guard per lane
(`(orig & 0x00ff_00ff) | 0x0100_0100`) to suppress underflow before
the subtract, with a final `& 0x00ff_00ff` mask to clear the guard.
The high pair is brought into the same low-of-pair layout with a
`>> 8` and re-positioned with `<< 8` after masking.

| Form | Median |
|---|---:|
| Closure-of-four (kept) | 34.1 µs |
| Biased-SWAR (tried, reverted) | 40.5 µs (**+18.4%**) |

AArch64 NEON auto-vectorisation across the four sequential per-byte
`wrapping_sub` calls in the closure body is tighter than the explicit
biased-SWAR pattern at this call site — the lane-bias `| 0x0100_0100`
and the final mask `& 0x00ff_00ff` are extra micro-ops that don't
amortise across a single-pixel call. Same shape as the round-194
BENCHMARKS footnote that recorded a regression for a
`clamp_add_subtract_*` (mode 12) per-channel `to_le_bytes()` + `i16`
byte-loop attempt — the closure-of-four `i32` body remains the right
starting point on this target.

The function body is left in its pre-r224 form. The randomised cross-
check test stays in place as a regression guard so any future
`std::simd` rewrite of `predictor_subtract` (the next plausible
attempt, mirroring the `to_rgba_simd` precedent under the `simd`
feature where the 16-byte vector load amortises the lane-bias cost
across four pixels per iteration) can re-use this test and this bench
as the A/B reference.

## Round-248 (2026-06-07) — `apply_subtract_green` bench

Closes the encoder-side §4.3 inventory gap. The decoder-side mirror
(`vp8l_transform::inverse_subtract_green`) has had its own
per-pass criterion bench since round 217; the encoder-side forward
pass `vp8l_encode::apply_subtract_green` was unmeasured at the
per-pass level until this round.

### New bench: `apply_subtract_green`

`benches/apply_subtract_green.rs` builds a 256×256 deterministic
LCG-filled ARGB buffer (identical seed + multiplier + increment to
the §4.x decoder-side benches and the round-224 encoder-side
`predictor_subtract` bench) and runs `apply_subtract_green` once per
iteration over a fresh clone of that buffer. The clone is intentional:
`apply_subtract_green` mutates in place, so the bench must start from
the same input each iteration so a future SWAR / `std::simd` rewrite
cannot win by caching residuals across iterations.

`apply_subtract_green` was already `pub fn` (since round-170's encode
hot-path coverage). No visibility change was required this round.

### Round-248 measurement

| Bench | Median |
|---|---|
| `apply_subtract_green_256x256` | **~13.3–13.7 µs** (three consecutive `--quick` runs: 13.72 µs / 13.60 µs / 13.28 µs) |

Calibration against the matching §4.3 decoder pass: the round-217
`inverse_subtract_green_256x256` bench measures the same shape of work
(read one ARGB lane, write one ARGB lane, two per-pixel mod-256 adds
on R and B, A and G untouched) over the same 256×256 deterministic
LCG-filled buffer. The two passes share the same byte-traffic and
arithmetic complexity, so reading similar numbers is the expected
shape — the encoder-side forward pass and the decoder-side inverse
pass are symmetric.

The function body is left unchanged; this round's deliverable is the
A/B reference, not an optimization. Any future SWAR / `std::simd`
rewrite of the forward §4.3 pass (mirroring the round-224 SWAR
experiment for `predictor_subtract` and the `to_rgba_simd` precedent
under the `simd` feature) can use this bench as the A/B reference,
with the existing `apply_subtract_green_is_inverse_of_inverse_subtract_green`
roundtrip test as the byte-exact regression guard.

## Round-249: §4.4 palette subtraction-decode bench

`benches/inverse_color_table.rs` adds a criterion harness for
`vp8l_transform::inverse_color_table` — the §4.4 palette
subtraction-decode pass. Per the §4.4 spec text, the color table is
stored subtraction-coded, and every final color is recovered by
adding the previous color's ARGB components into the current entry's
ARGB components mod 256. The implementation walks the palette in
place and, for `i` in `1..len`, performs four byte-wise wrapping
adds (one per A / R / G / B lane) against the previous entry.

This was the last `pub fn` in `vp8l_transform` that had no per-pass
bench. The rest of the §4.x inventory has been covered since round
217 (`inverse_subtract_green`) / round 207 (`inverse_color`) /
round 210 (`inverse_color_indexing`) / round 194
(`inverse_predictor` per-mode) on the decoder side, and rounds
224 / 248 added the encoder-side mirrors for §4.1 / §4.3.
`inverse_color_table` closes the decoder-side §4.4 sub-step gap.

The bench parameterises three palette sizes:

* `palette2` — `width_bits = 3` tier, minimum length (one pass
  iteration). The pass is dominated by call overhead at this size.
* `palette16` — boundary between the `width_bits = 1` (1..16) and
  `width_bits = 0` (17..256) bundling tiers.
* `palette256` — `width_bits = 0` tier, the §4.4 maximum length.

The §4.4 subtraction-decode itself walks every palette entry the
same way regardless of which bundling tier the indexing pass will
later use — the per-tier widths above are sampled here only as a
representative cross-section of palette lengths.

The palette is filled with a deterministic LCG (same constants as
the rest of the §4.x bench inventory) so per-lane wrap paths are
exercised across runs. The bench clones the palette into a fresh
working buffer each iteration so the in-place pass starts from the
same input every time and a future SWAR / `std::simd` rewrite
cannot win simply by caching deltas across iterations.

`inverse_color_table` was already `pub fn` (since round-170's
decoder hot-path coverage). No visibility change was required this
round.

### Round-249 measurement

| Bench | Median |
|---|---|
| `inverse_color_table_palette2`   | **10.16 ns** |
| `inverse_color_table_palette16`  | **44.43 ns** |
| `inverse_color_table_palette256` | **1.273 µs** |

The numbers scale roughly linearly in palette length, as expected
for a per-entry sequential dependency: each iteration adds the
previous entry's four lanes into the current entry's four lanes,
so the body is bounded by the latency of those wrapping-add
dependency chains. A future byte-wise SIMD rewrite that processed
four lanes in parallel within a single 32-bit word (or even a
cross-iteration look-ahead with a per-iteration carry) would have
this bench available as an A/B reference.

The function body is left unchanged; this round's deliverable is
the A/B reference, not an optimization. The existing test coverage
in `vp8l_transform::tests` (`inverse_color_table` round-trips with
`vp8l_encode::forward_color_table` per
`forward_color_table_round_trips_with_decoder_inverse`) is the
byte-exact regression guard any future rewrite must hold.

## Round-250: §3.7.2 Huffman code-length builder bench

`benches/build_code_lengths.rs` adds a criterion harness for
`vp8l_encode::build_code_lengths` — the per-symbol Huffman
length-assignment pass invoked once per §3.7.1 prefix-code-group
channel (GREEN + length, RED, BLUE, ALPHA, DISTANCE) plus the
§3.7.2.1.2 normal-form *code-length-of-code-lengths* sub-pass. The
implementation is a textbook min-heap Huffman build followed by an
optional `MAX_CODE_LENGTH`-cap rebalancing pass; on every call it
allocates the parent / node-frequency vectors fresh and walks the
heap-pop / heap-push loop `used.len() - 1` times.

This is the first per-pass bench from the §3 entropy domain. The
§4.x transform inventory has been complete since round 249; the
§3.7.2 builder was the natural next pick because the round-170
encoder profile attributed rank 4 of self-time to the surrounding
closure body through `canonical_codes`, and `build_code_lengths`
runs ahead of `canonical_codes` on the same length tables.

The bench parameterises two axes:

* **Alphabet size**: the four §3.7.1 alphabets that occur:
  * `distance40`  — DISTANCE alphabet (§3.6.2.2).
  * `literal256`  — 8-bit channel literal alphabet (RED / BLUE / ALPHA).
  * `green281`    — GREEN with the smallest §3.6.2.3 color cache
    (`cache_bits = 0` ⇒ `cache_size = 1` ⇒ `256 + 24 + 1`).
  * `green2328`   — GREEN with the largest §3.6.2.3 color cache
    (`cache_bits = 11` ⇒ `cache_size = 2048` ⇒ `256 + 24 + 2048`).
* **Frequency regime**: the two distribution shapes the builder
  hits in practice:
  * `dense`  — every symbol live, LCG-fill `1..=255`. Models a
    natural-image meta prefix code group's literal channels.
  * `sparse` — `sqrt(N)` live symbols, frequencies shaped `1/(k+1)`
    Zipf-style and scattered across the alphabet via a second LCG
    stream. Models a DISTANCE table where few prefix codes fire or
    a GREEN table whose §3.6.2.3 color cache code range is barely
    populated.

The LCG constants match the rest of the §4.x bench inventory so
cross-pass numbers are reproducible and cross-bench comparable.
Each `b.iter` call hands the same `&[u32]` frequency slice to
`build_code_lengths`; the function's own per-call allocations
(parent / node-frequency vectors, the heap, the returned
`Vec<u8>`) are inside the measured interval. No fixture lookup or
shared mutable state crosses iterations.

### Round-250 measurement

| Bench | Median |
|---|---|
| `build_code_lengths_dense_distance40`   | **2.15 µs** |
| `build_code_lengths_sparse_distance40`  | **272.6 ns** |
| `build_code_lengths_dense_literal256`   | **15.33 µs** |
| `build_code_lengths_sparse_literal256`  | **833.0 ns** |
| `build_code_lengths_dense_green281`     | **17.50 µs** |
| `build_code_lengths_sparse_green281`    | **867.1 ns** |
| `build_code_lengths_dense_green2328`    | **417.8 µs** |
| `build_code_lengths_sparse_green2328`   | **4.96 µs** |

Observations:

* The dense-regime cost scales roughly `N · log N` from
  `distance40` to `literal256` to `green2328`, consistent with the
  textbook `O(N log N)` heap-build + heap-pop loop. The sparse-
  regime cost scales with the active-symbol count
  (`sqrt(N) · log sqrt(N)`), which is why `sparse_green2328` at
  ~`sqrt(2328) ≈ 48` live symbols stays well under
  `dense_literal256`'s ~256 live symbols.
* The dense / sparse ratio inside each alphabet (e.g. 18× for
  `literal256`, 20× for `green281`, 84× for `green2328`) shows the
  active-symbol count, not the nominal alphabet size, dominates
  the per-call cost. A future radix-bucket replacement for the
  heap would target the dense path (large active-symbol count) most
  effectively.
* `dense_green2328` at 417 µs is by far the heaviest bench in the
  per-pass inventory and the single most attractive future
  optimization target — a 5× speedup there would visibly move
  `lossless_encode_natural_128` end-to-end.

The function body is left unchanged this round; the deliverable is
the A/B reference. The existing `vp8l_encode::tests` coverage —
`code_lengths_kraft_sum_is_one`,
`code_lengths_single_symbol_is_length_one`,
`code_lengths_two_symbols_length_one_each` — is the byte-exact
regression guard any future rewrite must hold.

## Round-251: §3.7.2 canonical-code-value bench

`benches/canonical_codes.rs` adds a criterion harness for
`vp8l_encode::canonical_codes` — the second per-symbol pass in the
§3.7.2 length-then-code Huffman build. Given the per-symbol code
lengths produced by `build_code_lengths` (sampled by the round-250
bench), `canonical_codes` returns the canonical code values that
the decoder's `vp8l_prefix::PrefixCode` reconstructs from those same
lengths (symbols ordered by `(length, value)`, codes assigned
sequentially, read most-significant-bit-first within a code).

`canonical_codes` is the encode-profile rank-4 self-time symbol
attributed by the round-170 trace (40 / 2 700 samples on the
`encode_webp_lossless` driver). With `build_code_lengths` covered
by round 250, this pass closes the per-prefix-code-group encode
inner loop in the §3 entropy domain: every per-prefix-code-group
call now has a stand-alone bench on the same four alphabets and
the same two frequency regimes.

The implementation walks `1..=MAX_CODE_LENGTH` outer and the full
`lengths` slice inner — an explicit `O(MAX_CODE_LENGTH · N)` pass
that, unlike the §3.7.2 length builder, ignores the active-symbol
count (the inner loop runs over every slot regardless of whether the
length is zero). So the dense / sparse ratio inside each alphabet is
expected to be much *smaller* than for `build_code_lengths`, and the
four alphabet sizes are the dominant axis. The bench samples both
regimes anyway, so a future single-pass bucket-sort-by-length rewrite
that skips zero-length symbols would show up as a sparse-side
speedup.

The bench parameterises the same two axes as
`benches/build_code_lengths.rs`:

* **Alphabet size**: the four §3.7.1 alphabets that occur:
  * `distance40`  — DISTANCE alphabet (§3.6.2.2).
  * `literal256`  — 8-bit channel literal alphabet (RED / BLUE / ALPHA).
  * `green281`    — GREEN with the smallest §3.6.2.3 color cache
    (`cache_bits = 0` ⇒ `cache_size = 1` ⇒ `256 + 24 + 1`).
  * `green2328`   — GREEN with the largest §3.6.2.3 color cache
    (`cache_bits = 11` ⇒ `cache_size = 2048` ⇒ `256 + 24 + 2048`).
* **Frequency regime**: the same two distribution shapes used in the
  round-250 length bench. The length tables are produced by feeding
  those dense / sparse frequency tables through `build_code_lengths`
  once at bench setup, so each `b.iter` body sees the *exact* length
  table a real per-prefix-code-group call would. The
  `build_code_lengths` call itself is outside the measured interval.

The LCG constants match the rest of the per-pass bench inventory so
the length tables are reproducible across runs and hosts and
cross-pass comparable to the round-250 numbers. The function's own
per-call allocations (`bl_count`, `next_code`, the returned
`Vec<u32>`) are inside the measured interval.

### Round-251 measurement

| Bench | Median |
|---|---|
| `canonical_codes_dense_distance40`   | **251 ns** |
| `canonical_codes_sparse_distance40`  | **204 ns** |
| `canonical_codes_dense_literal256`   | **1.80 µs** |
| `canonical_codes_sparse_literal256`  | **1.36 µs** |
| `canonical_codes_dense_green281`     | **1.83 µs** |
| `canonical_codes_sparse_green281`    | **1.40 µs** |
| `canonical_codes_dense_green2328`    | **15.70 µs** |
| `canonical_codes_sparse_green2328`   | **9.98 µs** |

Observations:

* Cost scales linearly with the alphabet size on both regimes
  (40 → 256 → 281 → 2328 ≈ 1× → 7× → 7× → 63× on the dense path),
  consistent with the explicit `O(MAX_CODE_LENGTH · N)` two-loop
  shape that runs `15 · N` index checks per call.
* The dense / sparse ratio inside each alphabet is roughly 1.2× —
  *much* smaller than `build_code_lengths`'s 18×–84× ratio. As
  expected: the inner loop runs over every slot whether or not the
  length is zero, so the active-symbol count enters only via a
  small constant-factor branch-prediction effect.
* Compared to round 250's `build_code_lengths` numbers,
  `canonical_codes` is ~9× cheaper on the dense path
  (`green2328`: 417.8 µs vs 15.70 µs) and ~2× cheaper on the sparse
  path (`green2328`: 4.96 µs vs 9.98 µs). The crossover happens
  because the sparse path is cheap for the length builder (small
  heap) but unchanged for `canonical_codes` (the outer loop still
  runs 15 times over the whole alphabet).
* A bucket-sort-by-length rewrite would target the sparse path
  most effectively (currently dominated by the redundant
  zero-length scans) and would, in aggregate, more than pay for
  itself across the per-prefix-code-group call mix where most
  channels see a mostly-active alphabet but DISTANCE and the
  color-cache range of GREEN both run sparse.

The function body is left unchanged this round; the deliverable is
the A/B reference. The existing `vp8l_encode::tests` round-trip
coverage — `chooser_round_trips_through_decoder_on_both_branches`,
`round_trip_solid_color_uses_single_leaf_codes`,
`round_trip_larger_random_like`, and the `round_trip_*` family that
re-decodes every encoder bit against `decode_webp` — is the
byte-exact regression guard any future rewrite must hold.

## Round-276 (2026-06-11): §6.2.1 decoder canonical-table build bench

`benches/prefix_from_code_lengths.rs` adds a criterion harness for
`vp8l_prefix::PrefixCode::from_code_lengths` — the decoder-side
§6.2.1 canonical-table build. It is the decode mirror of the
§3.7.2 length-then-code encoder pair sampled in rounds 250 / 251:
given the per-symbol code lengths recovered from the bitstream, it
counts symbols per length, validates the §6.2.1 Kraft completeness
rule (`sum 2^-len == 1`, single-leaf-node exception), assigns
canonical code values in `(length, value)` order, and materialises
the per-length decode rows `read_symbol` walks.

`from_code_lengths` is the round-170 decode-profile rank-4 symbol
(`vp8l_prefix::PrefixCode::from_code_lengths`, ~50 / 2 700 samples,
~2% of decode self-time). Like its encoder mirrors it runs once per
prefix code per §6.2 prefix code group (five codes per group) plus
once for the inner code-length code of every normal-form §6.2.1
read, so its self-time scales with the per-image meta-prefix
code-group count. It was the last pass in the §3.7.2 / §6.2.1
length-then-code chain with no per-pass bench.

The bench parameterises the same two axes as rounds 250 / 251 — the
four §3.7.1 alphabets (`distance40` / `literal256` / `green281` /
`green2328`) and the two dense / sparse frequency regimes — with the
same LCG constants, so each cell is directly comparable to its
`build_code_lengths` / `canonical_codes` counterpart: the length
tables fed to `from_code_lengths` here are byte-for-byte the tables
those benches build / consume. `build_code_lengths` runs once per
cell at setup; the per-iteration `Vec<u8>` clone (the function takes
the length table by value) lives in `iter_batched` setup, outside
the measured interval — only the canonical-table build itself is
sampled.

### Round-276 measurement

| Bench | Median |
|---|---:|
| `prefix_from_code_lengths_dense_distance40`   | **187.7 ns** |
| `prefix_from_code_lengths_sparse_distance40`  | **100.7 ns** |
| `prefix_from_code_lengths_dense_literal256`   | **918.0 ns** |
| `prefix_from_code_lengths_sparse_literal256`  | **546.3 ns** |
| `prefix_from_code_lengths_dense_green281`     | **1.006 µs** |
| `prefix_from_code_lengths_sparse_green281`    | **623.2 ns** |
| `prefix_from_code_lengths_dense_green2328`    | **7.300 µs** |
| `prefix_from_code_lengths_sparse_green2328`   | **5.249 µs** |

(A second consecutive `--quick` run reproduced every cell within
~5%; medians above are from the first run.)

Observations:

* Cost scales linearly with alphabet size on both regimes
  (40 → 256 → 281 → 2328 ≈ 1× → 4.9× → 5.4× → 39× dense), as
  expected for a body whose assignment loop rescans the full
  `code_lengths` slice once per *used* length: the used-length
  count saturates well below `MAX_CODE_LENGTH`, leaving `N` as the
  dominant axis.
* The dense / sparse ratio sits at ~1.4–1.9× per alphabet — between
  `canonical_codes`'s ~1.2× (fixed 15-pass rescan, regime-blind)
  and `build_code_lengths`'s 18×–84× (heap scales with the
  active-symbol count). That matches the per-used-length rescan
  shape: sparse tables use fewer distinct lengths (fewer rescans)
  but each rescan still walks all `N` slots.
* Cell-for-cell against the encoder mirrors at `green2328` dense:
  `build_code_lengths` 417.8 µs ≫ `canonical_codes` 15.70 µs >
  `from_code_lengths` 7.30 µs. The decoder build is the cheapest
  link in the length-then-code chain, consistent with the §6.2.1
  spec shape (the decoder never sees frequencies — only lengths).
* A future single-rescan bucket-sort rewrite (one pass appending
  each used symbol to its length bucket, instead of one full rescan
  per used length) would target the dense cells first; this bench
  is its A/B reference.

The function body is left unchanged this round; the deliverable is
the A/B reference. The existing `vp8l_prefix::tests` coverage plus
the round-275 `fuzz/fuzz_targets/prefix_code.rs` differential
harness (rebuild-from-`code_lengths()` reproduces every decode) are
the byte-exact regression guards any future rewrite must hold.

## Round-277 (2026-06-11): length-then-code dense-cell rewrite

The dense-cell rewrite the round-250 / round-276 observations
flagged landed this round, on both sides of the §3.7.2 / §6.2.1
length-then-code chain. Output is **bit-identical** — verified by an
FNV digest over every length table and built decoder table for the
full 8-cell bench input set plus 600 randomized frequency tables
(varied alphabet sizes, zero densities, tie-heavy and exponential
skews that trip the length-limit pass): the digest matches the
previous implementation exactly. The round-275 `prefix_code`
differential fuzz harness ran 13.6 M execs clean on the rewrite
(plus 5.0 M on `prefix_code_group`), and the full test suite (11
test binaries) passes unchanged.

What changed:

* **Encoder `build_code_lengths`** — the hand-rolled binary min-heap
  is gone. The leaves are sorted once by a packed `(freq, symbol)`
  `u64` key; the merge loop then exploits the two-queue property
  (each internal node is created with a frequency no smaller than
  any earlier one, so a plain FIFO of internal nodes stays sorted
  for free) and takes the two smallest nodes per step by comparing
  the two queue fronts in O(1), preferring the leaf on a frequency
  tie exactly as the old `(freq, order)` heap key did — merge for
  merge the same pop sequence, hence identical trees. Leaf depths
  are recovered with one reverse pass over the internal nodes
  (a node's parent always has a larger index) instead of one
  parent-chain walk per leaf, and `limit_code_lengths` updates its
  Kraft sum incrementally (`±2^(MAX-len)` exact-integer steps)
  instead of recomputing the O(n) sum after every adjustment.
* **Decoder `PrefixCode::from_code_lengths`** — the `(length,
  value)` ordering is now the single-rescan counting sort the
  round-276 observations sketched: `bl_count` prefix sums fix every
  length bucket's start index, and ONE pass over `code_lengths`
  drops each used symbol at its bucket cursor (symbols are visited
  in ascending value order, so each bucket stays value-sorted),
  replacing the one-full-rescan-per-used-length assignment loop.

### Round-277 measurement

Before columns are a same-session re-run of the round-250 /
round-276 benches on the pre-rewrite code (`--quick`, same host);
the recorded round-250 `dense_green2328` median was 417.8 µs.

`build_code_lengths`:

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `build_code_lengths_dense_distance40`   | 2.093 µs | **408.4 ns** | 5.1× |
| `build_code_lengths_sparse_distance40`  | 252.5 ns | **118.4 ns** | 2.1× |
| `build_code_lengths_dense_literal256`   | 15.07 µs | **2.072 µs** | 7.3× |
| `build_code_lengths_sparse_literal256`  | 839.8 ns | **286.0 ns** | 2.9× |
| `build_code_lengths_dense_green281`     | 17.66 µs | **2.372 µs** | 7.4× |
| `build_code_lengths_sparse_green281`    | 867.9 ns | **302.9 ns** | 2.9× |
| `build_code_lengths_dense_green2328`    | 382.0 µs | **113.5 µs** | 3.4× |
| `build_code_lengths_sparse_green2328`   | 4.875 µs | **1.404 µs** | 3.5× |

`prefix_from_code_lengths`:

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `prefix_from_code_lengths_dense_distance40`   | 194.7 ns | **178.0 ns** | 1.1× |
| `prefix_from_code_lengths_sparse_distance40`  | 106.7 ns | **73.8 ns**  | 1.4× |
| `prefix_from_code_lengths_dense_literal256`   | 1.011 µs | **582.0 ns** | 1.7× |
| `prefix_from_code_lengths_sparse_literal256`  | 588.8 ns | **255.6 ns** | 2.3× |
| `prefix_from_code_lengths_dense_green281`     | 1.025 µs | **634.6 ns** | 1.6× |
| `prefix_from_code_lengths_sparse_green281`    | 645.7 ns | **269.3 ns** | 2.4× |
| `prefix_from_code_lengths_dense_green2328`    | 7.435 µs | **4.629 µs** | 1.6× |
| `prefix_from_code_lengths_sparse_green2328`   | 5.840 µs | **1.806 µs** | 3.2× |

Observations:

* The builder's mid-size dense cells gain the most (7.3–7.4× at
  256 / 281): the heap's cache-hostile sift swaps are replaced by
  one branch-predictable `u64` sort plus a linear merge.
* `dense_green2328` lands at 113.5 µs rather than the ~25 µs a pure
  `N log N` extrapolation from `green281` would predict because this
  input genuinely tripped the §3.7.2 length cap all along (2 328
  leaves with 1..=255 frequencies push the deepest leaves past 15
  bits): the remaining time is dominated by `limit_code_lengths`'s
  per-adjustment O(n) *target-selection* rescan, which the
  incremental-Kraft change already halved but did not remove. That
  rescan is the next optimization target if the cell is flagged
  again — a bucket-by-length selection structure could make each
  adjustment O(1) — but it is correctness-insurance code that only
  fires on capped inputs, so it was left structurally untouched this
  round.
* The decoder's sparse cells gain more than its dense cells (3.2× vs
  1.6× at green-2328): the old loop's per-used-length rescan cost
  scaled with `used_lengths · N`, but its *useful* work scaled with
  the used-symbol count, so the sparse tables were proportionally
  the most rescan-bound. Dense cells now sit at the two-pass
  (count/validate + place) floor.
* End-to-end encode impact is bounded: at ~170 ms for
  `lossless_encode_natural_128` the builder is a small slice of
  total encode time; the per-pass wins matter most for many-group
  images (high meta-prefix granularity) where the chain runs
  hundreds of times.

## Round-278 (2026-06-11): `limit_code_lengths` O(1)-per-adjustment target selection

The round-277 observations flagged `limit_code_lengths`'s
per-adjustment O(n) *target-selection* rescan as the next target if
the length-capped `dense_green2328` cell were flagged again. This
round profiled it fresh and the flag held, so the rescan is gone.

### Fresh profile (pre-change)

A release driver looping `build_code_lengths` on the
`dense_green2328` bench input (the one capped cell in the inventory:
2 328 leaves with 1..=255 frequencies push the deepest leaves past
15 bits) was sampled for 5 s with `limit_code_lengths` temporarily
`#[inline(never)]` for attribution:

| Top-of-stack symbol | Samples | Share of in-process |
|---|---:|---:|
| `limit_code_lengths` | 3 491 | ~81% |
| leaf sort (`quicksort` + `small_sort` + pivot) | ~435 | ~10% |
| `build_code_lengths` body (merge + depth recovery) | 205 | ~5% |

So on capped inputs the §3.7.2 cap pass — correctness-insurance code
— dominated the entire builder. Hotspot confirmed; optimization
warranted.

### What changed

The over-subscribed loop's per-step rescan walked all used symbols
and kept the LAST `used`-order symbol among those sharing the
largest current length below the cap (the `l >= best_len` comparison
updates on ties). Two structural facts make that selection
reproducible without the rescan:

1. One bucket per code length, filled in a single pass over `used`,
   holds each bucket's symbols in `used` order — the back of the
   highest non-empty bucket IS the rescan's pick.
2. Once a pick is lengthened from `l` to `l + 1 < 15` it is strictly
   the unique deepest eligible leaf, so the rescan re-picked the same
   symbol every subsequent step until it reached the cap (leaving the
   eligible set) or the Kraft sum reached 1. Driving the popped
   symbol upward in place therefore replays the original step
   sequence exactly, and no eligible bucket ever gains a member while
   the pass is running.

Each adjustment is now O(1) (bucket pop + in-place drive) instead of
O(n); the Kraft bookkeeping keeps the round-277 incremental
`±2^(15-len)` exact-integer updates. The under-subscribed loop is
untouched: `build_code_lengths`'s post-clamp Kraft sum is always
strictly over-subscribed (clamping only shortens lengths), so that
loop only runs in the defensive overshoot case, and the rewritten
over-subscribed pass hands it a state identical to before because it
replays the same adjustment sequence.

### Bit-identical proof

* FNV-1a digest over every emitted length table for the 8 bench
  cells plus 600 randomized frequency tables (varied alphabets
  2..=2328, zero densities, tie-heavy all-equal tables, Fibonacci
  and power-of-two cap-tripping skews): `0x0e7252a02fbaa388` before
  and after — unchanged.
* Differential fuzz against the *literal pre-change implementation*
  (the crate at the previous commit built side-by-side as a renamed
  package): 20 M randomized tables, full length-table equality
  asserted per input, ~5.8 M of them producing max-length-15 codes
  (cap-pass candidates). Zero divergence in 2 m 36 s.
* 5-minute `roundtrip_lossless` fuzz run (encode → decode
  pixel-exact oracle; 3 405 full round trips, 331 new corpus units)
  clean; full test suite (433 lib tests + all integration binaries)
  passes unchanged.

### Round-278 measurement

Before columns re-measured same-session on the pre-change code
(`--quick`, same host as round 277):

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `build_code_lengths_dense_green2328`    | 111.9 µs | **26.4 µs** | 4.2× |
| `build_code_lengths_sparse_green2328`   | 1.335 µs | **1.342 µs** | — |
| `build_code_lengths_dense_distance40`   | 408.9 ns | **414.0 ns** | — |
| `build_code_lengths_dense_literal256`   | 2.135 µs | **2.093 µs** | — |
| `build_code_lengths_dense_green281`     | 2.315 µs | **2.268 µs** | — |

Observations:

* `dense_green2328` lands at 26.4 µs — right on the ~25 µs pure
  `N log N` extrapolation from `green281` that round 277 predicted
  for a rescan-free cap pass. A post-change re-sample of the same
  driver shows the leaf sort (~1 691 samples) and the merge/depth
  body (1 668) now co-dominant; `limit_code_lengths` no longer
  registers as a separate cost center.
* Every uncapped cell is within run-to-run noise of its round-277
  median, as expected: the bucket structure is only built when the
  post-clamp Kraft sum exceeds 1, so inputs that never trip the
  §3.7.2 cap don't pay for it.
* This closes the length-then-code chain's flagged-target backlog:
  builder (r277), decoder table (r277), cap pass (r278). The chain's
  remaining cost on capped inputs is the one-time leaf sort, which is
  shared with the uncapped path and already at the comparison-sort
  floor for the alphabet sizes §3.7.1 allows.

## Reproducing

```bash
# Stable (default-features, scalar everywhere):
CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
  cargo bench --manifest-path crates/oxideav-webp/Cargo.toml \
  --bench argb_to_rgba -- --quick

# Nightly (`simd` feature, enables `to_rgba_simd`):
CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
  RUSTC=$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin/rustc \
  $HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin/cargo bench \
  --features simd --bench argb_to_rgba -- --quick
```

To re-profile (macOS):

```bash
# Build a debug-info release driver that loops the hot path:
cargo build --release  # in a scratch crate that depends on oxideav-webp
./target/release/prof_decode &
sample $! 4 -f /tmp/prof_decode.sample
```

The driver source lives in this repo's history; a fresh copy is a
five-line `main()` that calls `decode_webp` in a tight loop.
