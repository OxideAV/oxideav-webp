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
| `benches/read_symbol.rs` | `read_symbol_{short8_uniform,short6_uniform,long9_11,dense256,belowgate16_walk}` | Decoder-side §6.2.1 `PrefixCode::read_symbol` per-symbol reader — the rank-1 decode hotspot (~82% of decode self-time post-round-284). Five cells isolate the round-284 primary-table fast path (`short8` / `short6` all-table-hit), the long-code (> 8-bit) walk continuation the round-284 follow-up targets (`long9_11`, every read spills to the per-bit walk), the realistic blended literal channel (`dense256`), and the below-gate walk-only baseline (`belowgate16_walk`, no table built). Each cell packs a deterministic LCG symbol stream via the public `canonical_codes` + `BitWriter` and times 4096 back-to-back reads through a fresh `BitReader` |
| `benches/lz77_chain.rs` | `lz77_chain_{deep_period2,deep_period4,medium_period64,shallow_unique,natural_gradient}` | §5.2.2 LZ77 hash-chain matcher (`Lz77Matcher::find` + `insert`, rank 3 in the round-283 encode profile) chain-*depth* scenarios — five 8192-pixel tiles that vary hash-chain depth from maximal (period-2/4 repeats), through moderate (period-64), to shallow (near-unique LCG / natural gradient). Driven through the public `encode_argb_literals_with_width` entry like `lz77_match`, but with depth-controlled inputs so a future chain-walk cut has A/B numbers per regime instead of one blended figure |
| `benches/lossy_decode.rs` | `decode_webp_lossy_e2e`, `decode_lossy_rgba_extracted`, `yuv420_to_rgba_{16x16,128x128,256x256}` | §2.5 `VP8 ` (lossy) decode path **this crate owns** at three altitudes: full public `decode_webp` over the committed 128×128 `VP8 `+`ALPH` fixture; `decode_lossy_rgba` over the extracted `VP8 ` bitstream (RIFF walk removed); and the crate-owned post-I420 reconstruction loop `yuv420_to_rgba` (4:2:0 nearest-neighbour chroma up-sample + RFC 6386 §9.2 BT.601 full-range YCbCr→RGB matrix) in isolation at three sizes. The sibling `oxideav-vp8` decoder owns the entropy/IDCT/prediction/loop-filter work; this bench isolates and ranks the lossy stage the webp crate can act on |
| `benches/alpha_decode.rs` | `decode_alpha_plane_e2e`, `decode_alpha_lossless_extracted`, `inverse_filter_{none,horizontal,vertical,gradient}_128x128` | §2.7.1.2 `ALPH` alpha-plane decode — the rank-1 webp-owned cost on the lossy path (≈52% of lossy e2e in the round-289 map, previously sized only by subtraction). Public `decode_alpha_plane` e2e over the committed fixture; `alph::decode_alpha` on the extracted `ALPH` payload (RIFF walk removed); and the §2.7.1.2 Stage-2 inverse-filter per-pixel loop in isolation, one cell per `F` method via synthetic uncompressed payloads. Splits the r289 rank-1 bucket: container walk ≈1 µs, the rest is the headerless VP8L lossless decode (already covered by `read_symbol` / `lossless_decode*`) |

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

## Round-280 (2026-06-12): §4.1 encoder chooser block-walk despecialisation

### Fresh profile (pre-change)

An 8-second `/usr/bin/sample` of a release driver looping
`encode_webp_lossless` on the natural 128×128 tile (the
`lossless_encode_natural_128` bench input) attributed top-of-stack
self-time as:

| Rank | Symbol | Samples | Share |
|---:|---|---:|---:|
| 1 | `vp8l_encode::predictor_at` | 2 197 | ~36% |
| 2 | `encode_argb_with_predictor_chooser` closure | 1 027 | ~17% |
| 3 | `encode_with_color_transform` | 547 | ~9% |
| 4 | `encode_with_predictor_slack` | 425 | ~7% |
| 5 | `encode_with_predictor_entropy` | 390 | ~6% |
| 6 | `Lz77Matcher::find` | 303 | ~5% |

`predictor_at` was the round-170 encode-profile rank-1 too and had
never been optimized. Every §4.1 block-mode chooser pass
(`pick_block_mode_with_hint`, `pick_block_mode_with_hint_slack`,
`block_mode_entropy_cost`, plus the `block_mode_cost` tie-break) ran
it once per pixel **per candidate mode** (14 modes per block, per
`size_bits` candidate, per cost model), and each call re-ran the
§4.1 border-rule branch chain (`x == 0 && y == 0` → `y == 0` →
`x == 0` → `x == w - 1`) plus a 14-way `match mode` dispatch on a
runtime `mode`, with no inlining into the cost loop.

### What changed

The four chooser cost paths now run through one shared
block-residual walker (`walk_block_residuals` +
`for_each_block_residual`):

* **Border rules hoisted** out of the per-pixel loop into per-region
  loops (top row / left column / interior / right-column TR
  wraparound) — the same split the round-180 decoder
  `inverse_predictor` rewrite proved out.
* **Mode dispatch hoisted**: the 14-way `match mode` runs once per
  block walk; each arm monomorphises the walker over the §4.1
  predictor closure for that mode, so the interior loop inlines the
  predictor body with no per-pixel dispatch.
* **Row-granular pruning**: the L1 pickers' per-pixel
  `cost >= best_cost` early-out is now checked at block-row
  boundaries (`MagnitudeCostSink::row_end`), leaving the interior
  pixel loop branch-free (auto-vectorisable). Pick-identical: a
  pruned partial sum is only ever compared `>= cap`, and per-pixel
  contributions are non-negative, so any prune implies the full sum
  also compares `>= cap`; full (uncapped) sums are unchanged, so
  every argmin and every tie-break resolves exactly as before. Cost
  is at most one extra block row of work on a pruned mode.

`predictor_at` itself is unchanged (still used by
`apply_forward_predictor`, whose mode varies per block run, and by
the chooser tests).

### Bit-identical proof

* FNV-1a digest over the full encoded output of an 82-image sweep —
  the two bench inputs (256×256 gradient, natural 128×128 tile) plus
  16 shapes from 1×1 to 33×129 × five fill regimes each (uniform
  random, smooth gradient, 5-color palette, solid, per-row random
  walk with varying alpha): `0fb035b5e0f085a7` (90 640 encoded
  bytes) before and after — unchanged.
* New `block_walker_matches_predictor_at_reference_random` test pins
  `block_mode_cost`, the `block_mode_entropy_cost` histograms, and
  both hinted pickers (every hint × slack ∈ {0, 1, 7, 64}) against
  verbatim copies of the pre-round-280 per-pixel `predictor_at`
  loops across 13 block/image geometries covering every walker
  border regime, for modes `0..=13` plus an out-of-range mode.
* 3-minute `roundtrip_lossless` fuzz run (encode → decode
  pixel-exact oracle, 1 799 full round trips) clean; full test suite
  (434 lib tests + all integration binaries) passes unchanged.

### Round-280 measurement

Before is the same-session pre-change `--quick` baseline; two
consecutive post-change runs are shown (the machine-load spread on
this host straddles them):

| Bench | Before | After (run 1) | After (run 2) | Δ |
|---|---:|---:|---:|---:|
| `lossless_encode_natural_128` | 170.65 ms | 138.10 ms | **123.55 ms** | **−19% to −28%** |
| `lossless_encode_rgba_256` | 1.5116 s | 1.2684 s | **1.1947 s** | **−16% to −21%** |

A post-change re-sample of the same driver shows the monomorphised
walker instantiations co-leading with the remaining per-pass bodies;
the border/dispatch overhead no longer registers, and the remaining
chooser cost is the inherent 14-mode × per-pixel arithmetic itself.
Next plausible targets from the post-change profile:
`encode_with_color_transform` (its `pick_block_cte` per-pixel walk
is the analogous §4.2 chooser, now rank-2) and the
`Lz77Matcher::find` chain walk.

## Round-281 (2026-06-12): §3.5.2 CTE chooser bench + chunk-granular prune

### New bench: `pick_block_cte_walk_256x256`

Round 280's post-change profile named `encode_with_color_transform`
(rank 2, ~9% self-time) as the next target, with its `pick_block_cte`
walk being the per-pixel-heavy stage — the §3.5.2 analogue of the
§4.1 predictor chooser round 280 despecialised. No bench covered it
(the bench shelf had every §4.x *decoder* transform plus the encoder
predictor-residual and LZ77 paths, but nothing on the color-transform
chooser), so this round adds `benches/pick_block_cte.rs`.

The scenario drives the exact walk `build_color_image` performs at
the encoder-default `size_bits = 4`: 256 `pick_block_cte` calls (a
16×16 grid of 16×16-pixel blocks over a 256×256 ARGB image), each a
per-axis greedy sweep of the 25-entry candidate table (75 cost
evaluations per block). The input is a deterministic LCG image with
genuinely correlated channels (red ≈ green/2, blue ≈ green/3 + red/4,
plus ±8 noise) so the `cost >= best` early-out prunes the way it does
on natural content — neither the all-prune degenerate (solid) nor the
never-prune one (uniform random). `pick_block_cte` was made `pub` for
the harness, same shelf as `predictor_subtract` /
`apply_subtract_green`.

### Baseline (pre-change)

| Bench | Time (full run) | Per block |
|---|---:|---:|
| `pick_block_cte_walk_256x256` | **1.5964–1.6070 ms** (median 1.6012 ms) | ~6.3 µs |

(`--quick` same session: 1.5886 ms — consistent.)

### Measurement-driven change: chunk-granular prune

The baseline's inner cost loops carried a per-sample
`if cost >= best { break }` — a loop-carried data-dependent exit that
blocks auto-vectorisation, exactly the shape the round-280 walker
moved to block-row granularity. The three per-axis sweeps now share
one `sweep_cte_axis` helper that accumulates a branch-free 32-sample
chunk (`CTE_PRUNE_CHUNK`) into a `u32` partial and checks the prune
at chunk boundaries only.

Pick-identical by the round-280 argument: per-sample contributions
are non-negative (`channel_magnitude <= 128`), so a partial sum
reaching `>= best` implies the full sum also compares `>= best`; a
candidate that now completes instead of pruning yields its exact full
sum, still `>= best`, and still loses. Completed sums and the
strict-`<` earliest-wins tie-break are unchanged. Worst case is one
extra 32-sample chunk per pruned candidate.

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `pick_block_cte_walk_256x256` | 1.6012 ms | **752.03 µs** | **−52.6%** |

End-to-end (`--quick`, post-change): `lossless_encode_natural_128`
119.88 ms, `lossless_encode_rgba_256` 1.2323 s — at the fast edge of
the round-280 post-change spread (123.55–138.10 ms / 1.1947–1.2684 s),
as expected for a pass that is one slice of a rank-2 ~9% profile
entry.

### Bit-identical proof

* FNV-1a digest over the full encoded output of an 81-image
  `encode_webp_lossless` sweep — 16 shapes from 1×1 to 128×128 × five
  fill regimes (uniform random, gradient, 5-color palette, solid,
  per-pixel random walk) + the 256×256 gradient bench shape:
  `111b83e9ec73d760` (253 568 encoded bytes) before and after —
  unchanged.
* Full crate test suite passes unchanged (434 lib tests + all 10
  integration binaries), including the two `pick_block_cte` pin
  tests (solid-block minimum, known-slope recovery).
* 3-minute `roundtrip_lossless` fuzz run (encode → decode pixel-exact
  oracle, 1 574 runs) clean.

### Followups

* `Lz77Matcher::find` (rank 6 in the round-280 profile) — the
  hash-chain walk is only benched indirectly through
  `lz77_match`'s public-entry drive; a chain-depth-targeted scenario
  would isolate it.
* `pick_block_cte` still allocates its `samples` gather `Vec` per
  block (256 allocations per 256×256 image at `size_bits = 4`);
  fold-out via a caller-owned scratch buffer is the next obvious
  micro-cut if the §3.5.2 pass shows up again post-chunking.

## Round-283 (2026-06-12): end-to-end decode coverage + full regression refresh

BENCH-mode depth round — `src/` untouched. Three deliverables: three
new end-to-end bench harnesses covering decode paths that only had
per-pass coverage, a full regression re-run of all 21 bench targets
on this host (stable + nightly `simd`) reflecting the state after the
r277–r281 optimizations, and fresh decode / encode profiles ranking
the next optimization candidates.

### New bench: `lossless_decode_mixes`

`benches/lossless_decode_mixes.rs` measures the public `decode_webp`
entry point per **elected §4 transform mix**. The long-standing
`lossless_decode` bench drives one gradient fixture whose encoder
elects the §4.1 predictor path, so the end-to-end cost of the §4.2 /
§4.3 / §4.4 inverse transforms and of the transform-free path was
never visible above the per-pass `inverse_*` microbenches. Five
256×256 fixtures steer the encoder's chooser onto each mix, and the
elected transform list is **asserted at setup** via
`read_vp8l_transform_list` so a future chooser change that re-routes
a cell fails loudly instead of silently mislabeling the measurement:

| Cell | Content | Elected (asserted) | Encoded size | Median |
|---|---|---|---:|---:|
| `predictor` | smooth gradient | §4.1 `Predictor(size_bits=4)` | 108 B | **254.7 µs** |
| `colorindex` | 4-color 8×8 blocks | §4.4 `ColorIndexing(4)` | 218 B | **172.3 µs** |
| `crosscolor` | random G, R≈G/2, B≈G/3+R/4 | §4.2 `Color(size_bits=8)` | 154 376 B | **3.033 ms** |
| `subgreen` | random G, R≈G, B≈G | §4.3 `SubtractGreen` | 98 442 B | **1.254 ms** |
| `none` | uniform random noise | (empty) | 196 686 B | **1.825 ms** |

Observations:

* The spread is dominated by the **§6 entropy decode**, not the §4
  inverse pass: the per-pass benches put every 256×256 inverse
  transform at 13–50 µs (predictor modes 475–830 µs), yet the cells
  span 0.17–3.03 ms tracking encoded size (i.e. symbol count /
  prefix-code mix), not transform identity.
* `crosscolor` is the heaviest decode in the whole inventory: ~1.7×
  the `none` cell on a *smaller* payload. Its literals decode through
  four separate prefix codes per pixel (G, R, B + the §4.2 color
  image's own prefix-code group), making it the best end-to-end probe
  for any future `read_symbol` fast-path work.
* `colorindex` beats every other cell despite running an extra §4.4
  pass — the bundled 4-color image decodes 4 packed indices per green
  symbol, cutting the per-pixel symbol count below 1.

### New bench: `anim_decode`

`benches/anim_decode.rs` is the first animated-path bench: the
`ANIM`/`ANMF` chunk walk, per-frame §2.6 VP8L decode, and the
§2.7.1.1 canvas compositor (blend / dispose / sub-frame placement)
previously had no coverage at any level. One 12-frame 128×128
timeline (a 32×32 square moving over a gradient) is assembled twice
via `anim_encode::build_animated_webp` — once all-keyframes
(`AnimFrameMode::Lossless`), once dirty-rect deltas
(`AnimFrameMode::Delta`) — and setup asserts both layouts decode to
identical final-frame pixels:

| Cell | File size | Median | Per frame |
|---|---:|---:|---:|
| `anim_decode_keyframes_12x128` | 12 full-canvas frames | **2.181 ms** | ~182 µs |
| `anim_decode_delta_12x128` | 1 keyframe + 11 dirty rects | **372.1 µs** | ~31 µs |

The keyframe cell is 12 × the single-frame decode cost plus the
compositor (a 128×128 frame decodes in ~170 µs standalone — the
~12 µs/frame delta is the full-canvas overwrite composite + canvas
clone per emitted frame). The delta cell shows the §2.7.1.1 sub-frame
path working as intended: ~5.9× cheaper for the same visual timeline.

### New bench: `metadata_walk`

`benches/metadata_walk.rs` measures `extract_metadata` — the
published §2.7 demux surface (full RIFF chunk walk + `ICCP` / `EXIF`
/ `XMP ` payload lift). Cells split the chunk-walk cost from the
payload-copy cost:

| Cell | Layout | Median |
|---|---|---:|
| `metadata_walk_simple_nometa` | 1 chunk, no metadata | **18.86 ns** |
| `metadata_walk_vp8x_full` | 5 chunks, ICC 3 KiB + Exif 1 KiB + XMP 2 KiB | **185.5 ns** |
| `metadata_walk_anim64_full` | ~68 chunks (64 `ANMF`), same payloads | **421.0 ns** |

The walk is comfortably non-hot: ~3.5 ns per chunk crossed (the
anim64 − vp8x spread over ~63 extra chunks) plus ~25 ns/KiB of
payload copy. No optimization warranted; the cells exist to keep the
demux surface honest as the container code evolves.

### Full regression table (stable, `--quick`, this host)

All 21 bench targets re-run in one session on the same machine
(`aarch64-apple-darwin`, M4). Reference is each cell's most recent
recorded median (round noted); Δ beyond ±5% is called out.

| Bench cell | Last recorded | Round-283 | Note |
|---|---:|---:|---|
| `lossless_encode_rgba_256` | 1.1947–1.2684 s (r280/281) | **1.247 s** | mid-spread |
| `lossless_encode_natural_128` | 119.88–138.10 ms (r280/281) | **123.96 ms** | mid-spread |
| `lossless_decode_argb_256` | ~643 µs (r207) | **655.5 µs** | within noise |
| `vp8l_lz77_match` | 812.17 µs (r170) | **746.6 µs** | −8% (drift since r170; chain untouched) |
| `argb_to_rgba` (scalar) | 8.56 µs (r180) | **8.69 µs** | within noise |
| `inverse_predictor_mode11/12/13` | 484 / 605 / 835 µs (r194) | **476 / 603 / 831 µs** | unchanged |
| `inverse_color_sb0/3/5/7` | 29.6 / 50.6 / 23.1 / 24.4 µs (r207) | **30.2 / 49.7 / 22.5 / 23.8 µs** | unchanged |
| `inverse_color_indexing_p2/4/16/256` | 31.6 / 39.4 / 39.2 / 18.6 µs (r210) | **31.8 / 40.6 / 40.0 / 18.6 µs** | unchanged |
| `inverse_subtract_green_256x256` | 13.7 µs (r217) | **13.43 µs** | unchanged |
| `predictor_subtract_256x256` | ~36 µs (r224) | **34.32 µs** | unchanged |
| `apply_subtract_green_256x256` | ~13.3–13.7 µs (r248) | **13.47 µs** | unchanged |
| `inverse_color_table_p2/16/256` | 10.2 ns / 44.4 ns / 1.273 µs (r249) | **10.1 ns / 46.1 ns / 1.360 µs** | unchanged |
| `build_code_lengths` dense d40/l256/g281/g2328 | 414 ns / 2.09 / 2.27 / 26.4 µs (r278) | **405 ns / 2.14 / 2.30 / 26.8 µs** | unchanged |
| `build_code_lengths` sparse d40/l256/g281/g2328 | 118 / 286 / 303 ns / 1.34 µs (r277/278) | **122 / 276 / 294 ns / 1.32 µs** | unchanged |
| `canonical_codes` dense d40/l256/g281/g2328 | 251 ns / 1.80 / 1.83 / 15.7 µs (r251) | **274 ns / 1.82 / 1.95 / 16.4 µs** | unchanged |
| `canonical_codes` sparse d40/l256/g281/g2328 | 204 ns / 1.36 / 1.40 / 9.98 µs (r251) | **203 ns / 1.49 / 1.46 / 10.57 µs** | within noise |
| `prefix_from_code_lengths` dense d40/l256/g281/g2328 | 178 / 582 / 635 ns / 4.63 µs (r277) | **188 / 625 / 661 ns / 4.87 µs** | within noise |
| `prefix_from_code_lengths` sparse d40/l256/g281/g2328 | 74 / 256 / 269 ns / 1.81 µs (r277) | **74 / 278 / 284 ns / 1.82 µs** | within noise |
| `read_lz77_value` fast/short/long/max | 0.52 / 1.83 / 5.35 / 7.0 ns (r252) | **0.52 / 1.83 / 5.35 / 6.99 ns** | unchanged |
| `color_cache_hash` bits 1/4/8/11 | ~443 ns (r253) | **443–447 ns** | unchanged |
| `value_to_prefix` fast/short/long/max | ~338 / 645 / 644 / 637 ns (r254) | **339 / 645 / 644 / 637 ns** | unchanged |
| `pick_block_cte_walk_256x256` | 752.03 µs (r281) | **768.4 µs** | within noise |

Every previously-optimized cell holds its post-optimization level —
no regressions since the r277–r281 work landed.

### Nightly `simd` feature pass

The `simd` feature only swaps the `Vp8lImage::to_rgba` repack, so the
nightly re-run covers the repack bench plus the end-to-end decode
benches that flow through it:

| Bench | Stable scalar | Nightly `simd` | Δ |
|---|---:|---:|---:|
| `argb_to_rgba` | 8.69 µs | **6.57 µs** | −24% (matches the r170 6.40 µs recording) |
| `lossless_decode_argb_256` | 655.5 µs | 652.3 µs | within noise |
| `lossless_decode_mix_*` (5 cells) | 172 µs – 3.03 ms | 183 µs – 3.10 ms | within nightly-codegen noise |
| `anim_decode_keyframes_12x128` | 2.181 ms | 2.138 ms | within noise |
| `anim_decode_delta_12x128` | 372.1 µs | 374.9 µs | within noise |

The repack win is real but its end-to-end share is now ~1–4%, so the
full-decode cells don't move outside noise — consistent with the
profile below.

### Fresh profiles (post-r281) and ranked next candidates

**Decode** — an 8 s `/usr/bin/sample` of a release driver looping
`decode_webp` on the `crosscolor` mix fixture (the heaviest decode
cell), and a 6 s sample on the `none` (noise) fixture:

| Top-of-stack symbol | crosscolor | noise |
|---|---:|---:|
| `vp8l_prefix::PrefixCode::read_symbol` | 6 031 (**~89%**) | 4 309 (**~85%**) |
| `vp8l_decode::decode_one_symbol` | 397 (~6%) | 428 (~8%) |
| `argb_to_rgba` repack | 288 (~4%) | 320 (~6%) |
| `vp8l_transform::inverse_color` | 42 (<1%) | — |
| `vp8l_prefix` table builds (`read_code_lengths` + `from_code_lengths`) | 41 (<1%) | — |

**Encode** — an 8 s sample of a release driver looping
`encode_webp_lossless` on the natural 128×128 tile:

| Rank | Symbol | Samples | Share |
|---:|---|---:|---:|
| 1 | `for_each_block_residual` (two monomorphised walker instantiations) | 2 868 | ~38% |
| 2 | `encode_argb_with_predictor_chooser` closure | 856 | ~11% |
| 3 | `Lz77Matcher::find` + `insert` | 732 | ~10% |
| 4 | `apply_forward_predictor` | 367 | ~5% |
| 5 | `pick_block_cte` | 330 | ~4% |
| 6 | `encode_vp8l_payload` + `encode_with_predictor_entropy` | 419 | ~6% |
| 7 | `canonical_codes` + `tokenize_lz77` | 321 | ~4% |

Ranked next-round optimization candidates:

1. **Decoder `PrefixCode::read_symbol` k-bit primary lookup table.**
   At ~85–89% of decode self-time on every entropy-heavy fixture this
   is the single largest remaining target in the crate. The current
   read walks the canonical decode rows length by length; a
   `(1 << k)`-entry primary table indexed by the next `k` peeked bits
   (each entry carrying `(symbol, code_length)` for codes ≤ k bits,
   with a spill path for longer codes) turns the common case into one
   load + one bit-advance. Must hold the round-275 `prefix_code`
   differential fuzz harness and the §6.2.1 Kraft validation
   semantics bit-for-bit. A/B references: all five
   `lossless_decode_mix_*` cells (best probe: `crosscolor` at
   3.03 ms) plus `lossless_decode_argb_256` and both `anim_decode`
   cells.
2. **Encoder chooser residual-walk arithmetic
   (`for_each_block_residual` + chooser closure, ~49% combined).**
   The r280 despecialisation removed the border/dispatch overhead;
   what remains is the inherent 14-modes × per-pixel cost
   arithmetic. Two shapes worth trying, in order: (a) SWAR the
   per-pixel residual-magnitude accumulation inside the monomorphised
   interior loops (the per-channel magnitude sums are independent
   byte lanes — same identity family as the r170 `add_pred` rewrite);
   (b) a per-block mode pre-filter that evaluates the cheap
   single-source modes first and seeds the existing prune cap with
   their best — pick-identical because it only tightens the cap
   earlier. A/B references: `lossless_encode_natural_128` /
   `lossless_encode_rgba_256`.
3. **`Lz77Matcher::find` chain walk (~10% combined with `insert`).**
   Flagged by the r281 followups and still unbenched in isolation —
   the public-entry `lz77_match` bench amortises it against
   tokenisation. First step is a chain-depth-targeted bench scenario
   (repetitive content forcing deep hash chains), then a
   pick-identical walk cut. Bench-first, same as `pick_block_cte` in
   r281.

The §2.7 demux walk (`metadata_walk`, ns-scale) and the §4.x inverse
transforms (all ≤ 50 µs per 256×256 pass, < 1% of decode self-time)
need no further dedicated rounds at current workloads.

## Round-284 (2026-06-12): §6.2.1 `read_symbol` primary lookup table

PROFILE-OPT depth round on the round-283 rank-1 candidate: the
decoder's `PrefixCode::read_symbol` at ~85–89% of decode self-time on
every entropy-heavy fixture. A fresh pre-change 6 s
`/usr/bin/sample` of a release driver looping `decode_webp` on the
`crosscolor` mix fixture reproduced the r283 attribution (2 300
samples on `read_symbol`, ~90% of decode-side self-time), so the
flagged optimization was warranted.

### What changed

* **`PrefixCode` primary lookup table.** Codes built from ≥ 32 used
  symbols now carry a 256-entry table indexed by the next 8 stream
  bits *in wire order* (first bit read = bit 0, matching the LSB-first
  §2 `ReadBits` contract; each entry's index set is the canonical code
  value bit-reversed across its length, stamped over the free high
  bits). An entry packs `(code_length << 16) | symbol`; the §6.2.1
  Kraft completeness gate guarantees stamps never collide. The old
  walk read one bit at a time and linearly re-scanned the per-length
  decode rows at *every* accumulated length; the new fast path is one
  peek + one load + one cursor advance for any code ≤ 8 bits.
* **Long codes (> 8 bits)** consume the 8 peeked bits (their
  accumulated MSB-first value is the bit-reversal of the peeked
  wire-order value) and continue the per-bit row walk from length 9 —
  decisions, bit consumption, and error positions identical to the
  pre-table loop.
* **Used-symbol amortization gate (`MIN_LOOKUP_USED = 32`).** The
  table is an investment (allocation + zero fill + 16 cold cache
  lines): tiny codes — animation delta frames, sub-resolution
  transform images, the 19-symbol code-length code — would build it,
  touch it a handful of times, and throw it away. A first ungated cut
  regressed `anim_decode_delta_12x128` by ~9% for exactly that reason
  (the post-change profile showed the header/table region absorbing
  the loss); below the gate the pre-table per-bit walk runs unchanged,
  and a code with `used < 32` has codes ≤ 31 bits short by the Kraft
  equality anyway. Near-EOF reads where the table match could have
  leaned on the zero padding also replay the per-bit walk, so
  `PrefixError::Eof` carries the exact pre-table position fields.
* **`BitReader::read_bits` word-load rewrite.** The §2 `ReadBits(n)`
  primitive now assembles its result from one zero-padded
  little-endian `u64` load + shift + mask instead of an `n`-iteration
  per-bit gather — bit-for-bit the same value (stream bit `i` lands at
  result bit `i`). New `peek_bits` / `advance_bits` carry the fast
  path.

### Bit-identical proof

* FNV-1a-64 digest sweep over the decoded output (geometry + every
  frame's RGBA) of the **full fixture corpus** — all 18
  `docs/image/webp/fixtures/*/input.webp` (lossless, lossy, animated,
  metadata, color-cache stress, cross-color active), all 8 in-crate
  `tests/data/*.webp`, the decoded alpha plane, and the five synthetic
  256×256 transform-mix fixtures (plus the encoded bytes of those five
  — the encoder is untouched): every digest identical between the
  pre-change tree (built from the prior commit in a side worktree) and
  this round. The alpha-plane digest also re-matches the long-standing
  `0x42e1_6029_2eb0_d472` validator pin.
* New CI pin `round284_fixture_corpus_decode_digests_are_pinned`
  locks all eight in-crate fixture decode digests permanently.
* Fuzz: `prefix_code` 19.3 M execs / 151 s, `prefix_code_group` 8.9 M
  / 91 s, `decode_lossless` 4.3 M / 91 s, `roundtrip_lossless`
  (encode → decode pixel-exact oracle) 1 302 full round trips / 181 s
  — all clean. Full test suite passes unchanged (434 lib tests + all
  integration binaries, +1 new pin test).

### Round-284 measurement

Before columns are same-session interleaved re-runs of the prior
commit in a side worktree (`--quick`, same host, alternating with the
post-change runs to cancel machine-load drift):

| Bench | Before | After | Δ |
|---|---:|---:|---:|
| `lossless_decode_mix_crosscolor_256x256` | 2.947 ms | **1.591 ms** | **−46%** |
| `lossless_decode_mix_none_256x256` | 1.803 ms | **0.923 ms** | **−49%** |
| `lossless_decode_mix_subgreen_256x256` | 1.236 ms | **0.778 ms** | **−37%** |
| `lossless_decode_mix_predictor_256x256` | 253.9 µs | 254.2 µs | within noise |
| `lossless_decode_mix_colorindex_256x256` | 175.0 µs | 176.2 µs | within noise |
| `lossless_decode_argb_256` | 647.8 µs | 650.3 µs | within noise |
| `anim_decode_keyframes_12x128` | 2.167–2.193 ms | 2.070–2.131 ms | −2 to −4% |
| `anim_decode_delta_12x128` | 374.0–381.9 µs | 383.8 µs | within noise (was +9% before the gate) |

`prefix_from_code_lengths` absorbs the table build on gated-in cells
(dense d40 / l256 / g281 / g2328: 183 / 638 / 630 ns / 4.66 µs →
275 / 755 / 766 ns / 4.64 µs — the g2328 build cost is amortized
inside an already-larger body) while gated-out sparse cells stay at
baseline (78 / 265 / 278 ns; sparse g2328 sits at 48 used symbols,
above the gate: 1.84 → 1.99 µs). Each +90–140 ns build pays for
itself within tens of decoded symbols on the streams that elect it.

The three big movers are exactly the cells the r283 observations
predicted: entropy-dominated streams whose literals decode through
dense 8–9-bit prefix codes. The predictor / colorindex / gradient
cells are bounded by `inverse_predictor`, `argb_to_rgba`, and the
§4.4 bundling (sub-1 symbol per pixel), so the entropy win does not
register there.

### Post-change profile and next ranked hotspot

A 6 s re-sample of the same `crosscolor` driver post-change:

| Top-of-stack symbol | Samples | Share of decode-side |
|---|---:|---:|
| `vp8l_prefix::PrefixCode::read_symbol` | 4 215 | ~82% |
| `argb_to_rgba` repack | 364 | ~7% |
| `vp8l_decode::decode_one_symbol` | 361 | ~7% |
| `BitReader::read_bits` | 95 | ~2% |
| `vp8l_transform::inverse_color` | 64 | ~1% |

`read_symbol` remains rank 1 at ~82% of a decode that is now ~1.9×
faster — its absolute cost roughly halved, and what remains is the
inherent four-prefix-reads-per-pixel call mix plus the long-code
(> 8-bit) continuation that dense 256+-symbol alphabets still route
~half their reads through. **Next ranked candidate:** widen the fast
path's coverage of 9–11-bit codes — either a second-level spill table
(the classic two-level layout: the primary entry for a long-code
prefix points at a sub-table indexed by the next `max_len − 8` bits)
or an alphabet-size-aware primary width. Both must re-prove the digest
sweep and re-clear the `prefix_code` differential harness; the
`crosscolor` / `none` / `subgreen` cells are the A/B references.
After that, the encoder-side candidates from r283 (chooser residual
walk SWAR, `Lz77Matcher::find` chain bench) are unchanged.

## Round-286 (2026-06-13): isolate the rank-1 decode + rank-3 encode hotspots

DEPTH round, **BENCHMARK** mode. The round-284 PROFILE-OPT landed the
decoder's §6.2.1 primary lookup table and flagged its own next target —
the long-code (> 8-bit) continuation that dense 256+-symbol alphabets
still route ~half their reads through, to be addressed by a
second-level spill table or an alphabet-size-aware primary width. The
round-283 encode profile flagged `Lz77Matcher::find` (rank 3, ~10%
combined with `insert`) as "still unbenched in isolation." Both
hotspots were measured only *through* end-to-end benches that blend
them with surrounding work, so neither flagged candidate had a clean
A/B reference. This round adds two harnesses that isolate them, then
ranks the next PROFILE-OPT target.

src/ is **byte-identical** this round — both benches construct their
inputs through the existing public API (`canonical_codes` + `BitWriter`
for the symbol streams; `encode_argb_literals_with_width` for the
matcher), so no `#[doc(hidden)]` probe was needed.

### New bench: `read_symbol` (decoder §6.2.1 rank-1 hotspot)

`PrefixCode::read_symbol` is ~82% of decode self-time post-round-284 —
the single largest symbol in the crate — yet the two existing
prefix-code benches measure only the *table build*
(`prefix_from_code_lengths`) and the §3.6.2.2 *value expansion*
(`read_lz77_value`); the symbol-read hot loop itself had no isolated
harness. The five cells separate the two paths the round-284 table
created. Each times 4096 back-to-back reads over a deterministic
LCG-packed stream (`--quick` medians, same `aarch64-apple-darwin`
host):

| Cell | Median (4096 reads) | Per symbol | Path |
|---|---:|---:|---|
| `read_symbol_short8_uniform` | 15.99 µs | ~3.90 ns | pure primary-table hit (all 8-bit codes) |
| `read_symbol_dense256` | 17.73 µs | ~4.33 ns | table hit + long-code tail (lengths 7–13, peak 8) |
| `read_symbol_long9_11` | 20.31 µs | ~4.96 ns | **every read spills to the > 8-bit walk** |
| `read_symbol_short6_uniform` | 21.20 µs | ~5.18 ns | table hit, short codes packed densely |
| `read_symbol_belowgate16_walk` | 20.93 µs | ~5.11 ns | no table (below `MIN_LOOKUP_USED`), per-bit walk |

The signal the round-284 follow-up predicted is isolated cleanly:
`long9_11` (every read overshoots the 8-bit table and continues the
per-bit walk) costs **+27%** per symbol over the pure-table-hit
`short8_uniform` lower bound, and `dense256` sits between them at +11%
— exactly the long-code overhead a second-level spill table would
remove. `short8` is the floor the spill-table change must not regress;
`belowgate16_walk` and `short6` bound the gated-out walk path it must
also leave unchanged.

### New bench: `lz77_chain` (encoder §5.2.2 rank-3 hotspot)

Five 8192-pixel tiles vary hash-chain depth, driven through the public
`encode_argb_literals_with_width` entry (the matcher is package-private,
as in `lz77_match`):

| Cell | Median | Chain regime |
|---|---:|---|
| `lz77_chain_deep_period4` | 1.026 ms | maximal depth, 4 buckets — long match found immediately |
| `lz77_chain_deep_period2` | 1.041 ms | maximal depth, 2 buckets |
| `lz77_chain_medium_period64` | 1.157 ms | moderate depth, one-row period |
| `lz77_chain_shallow_unique` | 6.81 ms | near-unique pixels, insert + miss dominate |
| `lz77_chain_natural_gradient` | 7.38 ms | realistic gradient + noise, near-miss walks + literals |

The depth axis reframes the r283 candidate. The deepest-chain cells are
the **cheapest** (~1 ms): when content repeats with a short period the
matcher finds a long match on its first candidate and skips the run, so
total `find` calls collapse. The expensive regime is the opposite —
`shallow_unique` / `natural_gradient` at **6.5–7×** the cost — where
almost nothing matches and the per-position insert + short near-miss
walk + literal-entropy coding runs on every pixel. A chain-walk
depth-cap (the r283-flagged shape) would therefore barely move the
expensive cells; the headroom on the matcher's hot regime is in the
insert + miss-reject path and the literal coding it feeds, not the deep
walk. This is a bench-first finding for the next encoder round.

### Ranked next PROFILE-OPT target

| Rank | Target | Evidence | A/B references |
|---:|---|---|---|
| **1** | **Decoder `read_symbol` 9–11-bit long-code coverage** (second-level spill table or alphabet-size-aware primary width) | `read_symbol_long9_11` +27% / `dense256` +11% over the `short8` floor; `read_symbol` is rank 1 at ~82% of decode self-time and dense 256+-symbol alphabets route ~half their reads through this path | `read_symbol_{long9_11,dense256,short8_uniform}` (must-not-regress: `short8_uniform`, `belowgate16_walk`); decode cells `crosscolor` 1.68 ms / `none` 0.94 ms / `subgreen` 0.88 ms |
| 2 | Encoder chooser residual walk SWAR (`for_each_block_residual` + chooser closure, ~49% combined in r283) | unchanged from r283; the inherent 14-modes × per-pixel arithmetic after the r280 despecialisation | `lossless_encode_natural_128` / `lossless_encode_rgba_256`, `pick_block_cte` |
| 3 | Encoder LZ77 **insert + miss-reject** path (not the deep-chain walk) | `lz77_chain` shows the shallow/unique regime at 6.5–7× the deep-repeat cost — the cost is in per-position insert + literal coding, not chain depth | `lz77_chain_{shallow_unique,natural_gradient}` vs `_deep_period2/4` |

Target **1 is the recommended next PROFILE-OPT round** — highest share,
cleanest A/B isolation, and the candidate the round-284 profile already
named. The byte-identity bar for any such change is the round-284
FNV-1a digest sweep over the full fixture corpus plus the
`round284_fixture_corpus_decode_digests_are_pinned` CI pin, and the
`prefix_code` differential fuzz harness.

### Round-286 verification

* **Byte identity:** `src/` is untouched this round (`git diff --stat`
  shows only `Cargo.toml` + the two new bench files + this file), so
  decode output is unchanged by construction. The full suite —
  435 lib tests (was 434; +1 is the round-284 corpus pin already on
  master) + all integration binaries — passes unchanged.
* **References re-captured:** `lossless_decode_mix_{crosscolor,none,subgreen}_256x256`
  at 1.68 / 0.94 / 0.88 ms reproduce the round-284 post-change
  baselines within machine drift.
* `cargo fmt --check` + `cargo clippy --all-targets --no-deps -D warnings`
  clean.

## Round 287 — decoder per-bit walk: direct length→row side table

PROFILE-OPT depth round acting on the round-286 rank-1 candidate. The
§6.2.1 canonical-decoder per-bit walk (`read_symbol_walk` + its
long-code continuation `read_symbol_long`) tested, **at every consumed
bit**, whether a code row exists at the current length via a linear
`length_rows.iter().find(|r| r.length == len)` rescan. The cost of that
rescan scales with the number of distinct code lengths in the code, so
it is invisible on uniform-length codes (every `belowgate` / `short`
cell) but real on the length-diverse codes that header-dominated and
adversarial streams produce. The change records a 16-byte
`len_to_row: Vec<u8>` (length → row index, `NO_ROW = u8::MAX` sentinel)
once in `from_code_lengths` and turns the per-bit test into one indexed
load. No decode table widening, **no added cache footprint** (the side
table shares a line or two with the already-hot `PrefixCode` header).

### A/B (this machine, criterion `--baseline`)

| Cell | HEAD (linear `find`) | r287 (`len_to_row`) | Δ |
|---|---:|---:|---|
| `read_symbol_manylen16_walk` (1..=14 + 2×15, all 15 lengths) | 86.76 µs | 37.18 µs | **−57% (2.33×)** |
| `read_symbol_short8_uniform` | — | — | no sig. change (p≈1) |
| `read_symbol_short6_uniform` | — | — | no sig. change |
| `read_symbol_long9_11` | — | — | no sig. change |
| `read_symbol_dense256` | — | — | no sig. change |
| `read_symbol_belowgate16_walk` | — | — | no sig. change |
| `lossless_decode_*` (all 6 end-to-end cells) | — | — | no sig. change |

The new `manylen16_walk` cell is a Kraft-exact, maximally
length-diverse code (one symbol at each of lengths 1..=14 plus two at
15) — the regime the uniform-length `belowgate` cell cannot exercise.
It is the only cell where the linear scan was hot, and it shows the full
2.33× win; every other cell is statistically flat because few distinct
lengths meant the scan was already cheap.

### Rejected candidate — second-level spill table

The round-284/286 named candidate (a 9–`LOOKUP2_BITS`-bit second-level
spill table for codes overshooting the 8-bit primary) was **prototyped
and measured a net regression**: at `LOOKUP2_BITS` = 12 (16 KiB) and 10
(4 KiB) `read_symbol_long9_11` rose to ~21.7–22.5 µs (from ~18.9 µs
baseline) and `dense256` to ~20.7–21.3 µs (from ~17.7 µs). The second
peeked word-load plus a random access into a 4–16 KiB table that
thrashes L1 against the 1 KiB primary costs more than the 1–3 extra
per-bit iterations the walk-from-8 actually does for a 9–11-bit code.
The conclusion: the long-code path's cost was the **per-bit row lookup**
(now O(1)), not the bit-by-bit consumption — so a bigger table was the
wrong lever.

### Round-287 verification

* **Byte identity:** the `read_symbol_reference` differential unit test
  (fast path vs pre-table per-bit walk, lockstep cursor + EOF-field
  comparison) and the `read_symbol_lut_diff` fuzz oracle both still
  pass; all 435 lib tests + every integration binary (fixture walks,
  round-trips, published-API, standalone e2e) are green.
* `src/` diff is confined to `src/vp8l_prefix.rs`; the only other change
  is the new `manylen` bench cell in `benches/read_symbol.rs`.
* `cargo fmt --check` + `cargo clippy --all-targets --no-deps -D warnings`
  clean.

### Ranked next PROFILE-OPT target (post-r287)

| Rank | Target | Evidence | A/B references |
|---:|---|---|---|
| 1 | Encoder chooser residual walk SWAR (`for_each_block_residual` + chooser closure, ~49% combined in r283) | the inherent 14-modes × per-pixel arithmetic after the r280 despecialisation | `lossless_encode_natural_128` / `lossless_encode_rgba_256`, `pick_block_cte` |
| 2 | Encoder LZ77 **insert + miss-reject** path (not the deep-chain walk) | `lz77_chain` shows the shallow/unique regime at 6.5–7× the deep-repeat cost | `lz77_chain_{shallow_unique,natural_gradient}` vs `_deep_period2/4` |
| — | ~~Decoder `read_symbol` 9–11-bit spill table~~ | **resolved r287**: spill table rejected (L1-thrash regression); the per-bit row lookup it would have shortcut is now O(1) via `len_to_row` | — |

## Round-289 (2026-06-13): §2.5 `VP8 ` lossy decode bench + hotspot map

Every prior bench round profiled the **lossless** (VP8L) decode + encode
paths. The §2.5 `VP8 ` lossy decode path had **no isolated harness** at
all, even though `decode_webp` routes lossy files through it. This round
adds `benches/lossy_decode.rs` and ranks the lossy decode hot path — with
no behavior change (all 435 lib tests + integration binaries still
green; decoded bytes identical).

### Scope: what the webp crate owns on the lossy path

The §2.5 lossy bitstream's entropy decode, inverse DCT (the "VP8 lossy
IDCT"), intra prediction, and loop filter are **owned by the sibling
`oxideav-vp8` decoder crate**, which `oxideav-webp` calls via
`decode_vp8`. They are therefore out of this crate's editable scope — an
IDCT bench belongs in `oxideav-vp8`. What `oxideav-webp` itself runs on
the lossy path is the stage *after* the reconstructed I420 key-frame
comes back: `vp8_decode::decode_lossy_rgba` → `vp8_decode::yuv420_to_rgba`
(4:2:0 nearest-neighbour chroma up-sample + the RFC 6386 §9.2 / RFC 9649
§10 BT.601 full-range YCbCr→RGB matrix, evaluated once per output pixel).
`yuv420_to_rgba` was widened from `fn` to `pub fn` (a visibility change
only — the same function `decode_lossy_rgba` already called; no emitted
byte changes) so the bench can isolate it from the sibling decode.

### New bench: `lossy_decode`

Three altitudes, on this host (`--quick`):

| Cell | Median | What it isolates |
|---|---:|---|
| `decode_webp_lossy_e2e` | 359.2 µs | full public `decode_webp`: RIFF walk + `VP8 ` decode + `ALPH` layering + YCbCr→RGB, over the 128×128 fixture |
| `decode_lossy_rgba_extracted` | 173.4 µs | `decode_lossy_rgba` on the extracted `VP8 ` bitstream — sibling decode + conversion, **no** RIFF/`ALPH` |
| `yuv420_to_rgba_16x16` | 551 ns | crate-owned conversion loop, 256 px |
| `yuv420_to_rgba_128x128` | 34.0 µs | crate-owned conversion loop at fixture size |
| `yuv420_to_rgba_256x256` | 136.6 µs | crate-owned conversion loop, 65 536 px |

### Ranked lossy-decode hotspot map

Decomposing the e2e cost by subtraction (all on this host, `--quick`):

| Rank | Stage | Owner | Cost (per 128×128 lossy frame) | Share of e2e |
|---:|---|---|---:|---:|
| 1 | Container walk + `ALPH` plane layering | webp (this crate, already split across `metadata_walk` / decode) | ≈ 359.2 − 173.4 = **185.8 µs** | ≈ 52% |
| 2 | Sibling VP8 decode (entropy + IDCT + intra-pred + loop filter) | **`oxideav-vp8`** (out of scope) | ≈ 173.4 − 34.0 = **139.4 µs** | ≈ 39% |
| 3 | `yuv420_to_rgba` YCbCr→RGB + chroma up-sample | webp (this crate) | **34.0 µs** | ≈ 9% |

The conversion scales linearly with pixel count (551 ns → 34.0 µs →
136.6 µs across 256 → 16 384 → 65 536 px, ~4× per area-doubling),
confirming it is purely per-pixel-bound: a Q16 fixed-point matrix + clamp
+ a chroma index per pixel, no branch on content. It is the **only fully
webp-owned lossy hot loop** and the cleanest A/B target for a future SIMD
or matrix-fusion pass — the `argb_to_rgba` lossless repack got exactly
that treatment (scalar fallback + nightly `simd`), and `yuv420_to_rgba`
is the lossy analogue. Rank 1 (container + `ALPH`) is dominated by the
extended-layout `ALPH` plane decode, which has its own surface; rank 2 is
the sibling's to optimize.

### Ranked next PROFILE-OPT target (post-r289, lossy path)

| Rank | Target | Evidence | A/B references |
|---:|---|---|---|
| 1 | `yuv420_to_rgba` SIMD / matrix-fusion (lossy analogue of the `argb_to_rgba` SIMD pass) | linear per-pixel scaling, ~9% of lossy e2e and 100%-webp-owned | `yuv420_to_rgba_{16x16,128x128,256x256}` |
| — | VP8 lossy IDCT / entropy decode (~39% of lossy e2e) | **out of scope** — owned by sibling `oxideav-vp8`; an IDCT bench belongs in that crate | — |

### Round-289 verification

* **Byte identity:** all 435 lib tests + integration binaries
  (`fixture_walks`, `vp8_lossy_roundtrip`, published-API, standalone e2e)
  pass; the `lossy-with-alpha-128x128` fixture's expected SHA-256 still
  matches (decoded bytes unchanged). The only `src/` change is the
  `fn` → `pub fn` visibility widening of `yuv420_to_rgba`.
* `cargo fmt --check` + `cargo clippy --all-targets --no-deps -D warnings`
  clean.

## Round-291 (2026-06-14): §2.7.1.2 `ALPH` alpha-plane decode bench + refined lossy hotspot map

BENCH depth round. The round-289 `lossy_decode` hotspot map attributed
≈52% of the lossy end-to-end cost (≈185.8 µs of the 359.2 µs e2e fixture
decode) to "container walk + `ALPH` plane layering" — but that figure was
obtained purely by **subtraction** (`decode_webp_lossy_e2e` −
`decode_lossy_rgba_extracted`), with no direct instrument on the `ALPH`
decode at all. This round adds `benches/alpha_decode.rs`, which times the
rank-1 webp-owned alpha stage directly and decomposes it. **No behavior
change** — no `src/` edit; all 436 lib tests still green; decoded bytes
identical (the only diff is the new bench file + its `Cargo.toml`
registration).

### What the `ALPH` decode owns (§2.7.1.2)

`alph::decode_alpha` runs two stages:

1. **De-compression** (`C` field). Method 0 (`None`) copies the raw
   bytes; method 1 (`Lossless`) decodes a headerless §3 VP8L image stream
   and lifts the alpha out of each pixel's green channel. The committed
   fixture's `ALPH` is `Lossless`/`None`-filter, decoding to a 16 384-byte
   (128×128) plane.
2. **Inverse filtering** (`F` field). A per-pixel predictor
   (none / horizontal / vertical / gradient `clip(A+B−C)`) added mod-256
   over the *reconstructed* plane, with §2.7.1.2 left-most / top-most edge
   cases — the fully crate-owned per-pixel loop.

### New bench: `alpha_decode`

Five cells, on this host (`aarch64-apple-darwin`, `--quick`, median):

| Cell | Median | What it isolates |
|---|---:|---|
| `decode_alpha_plane_e2e` | 171.8 µs | public `decode_alpha_plane`: RIFF walk + dimension resolve + the fixture's real `Lossless` `ALPH` decode |
| `decode_alpha_lossless_extracted` | 170.9 µs | `alph::decode_alpha` on the extracted `ALPH` payload — RIFF walk removed |
| `inverse_filter_none_128x128` | 9.5 µs | §2.7.1.2 Stage-2 loop, `F = None` (memcpy + predictor-0), synthetic uncompressed payload |
| `inverse_filter_horizontal_128x128` | 21.8 µs | Stage-2 loop, `F = Horizontal` (left neighbour) |
| `inverse_filter_vertical_128x128` | 13.5 µs | Stage-2 loop, `F = Vertical` (above neighbour) |
| `inverse_filter_gradient_128x128` | 43.7 µs | Stage-2 loop, `F = Gradient` (`clip(A+B−C)`, deepest branch) |

### Refined lossy-decode hotspot map (corrects the r289 subtraction estimate)

The two direct measurements above let us split the r289 rank-1 bucket:

| Sub-stage of the r289 "container + `ALPH`" bucket | Direct cost (128×128 fixture) | Evidence |
|---|---:|---|
| RIFF walk + dimension resolution | ≈ 171.8 − 170.9 = **≈ 1 µs** | e2e − extracted |
| `ALPH` decode (headerless VP8L de-compress + green lift + inverse filter) | **≈ 170.9 µs** | `decode_alpha_lossless_extracted` |

The container walk the r289 map lumped into rank 1 is **negligible**
(~1 µs); the rank-1 cost is **almost entirely the headerless VP8L
lossless decode** inside `decode_alpha` (the fixture's `ALPH` is
`Lossless`-compressed). That decode is the same §3/§6.2.1 machinery the
`lossless_decode*` / `read_symbol` / `prefix_from_code_lengths` benches
already cover — the rank-1 lossy webp-owned cost is therefore *not* a new,
unbenched loop but the **VP8L lossless decoder applied to the alpha
plane**, which the r284/r287 `read_symbol` optimizations already speed up.

The §2.7.1.2 **inverse-filter** loop (Stage 2) is the one genuinely
alpha-specific crate-owned loop. Its cost ranks by filter method:

| Rank | `F` method | Median (128×128) | Relative to `None` |
|---:|---|---:|---:|
| 1 | Gradient (`clip(A+B−C)`) | 43.7 µs | 4.6× |
| 2 | Horizontal (left) | 21.8 µs | 2.3× |
| 3 | Vertical (above) | 13.5 µs | 1.4× |
| 4 | None (predictor 0) | 9.5 µs | 1.0× (memcpy floor) |

The `None` cell is the de-compression-copy + allocation floor; the deltas
above it are the per-pixel predictor walk. Gradient's 4.6× over the floor
is the per-pixel `(x, y)` match dispatch plus three neighbour loads and
the `clip(A+B−C)` arithmetic evaluated in the innermost loop body.

### Ranked next PROFILE-OPT target (post-r291, lossy/alpha path)

| Rank | Target | Evidence | A/B references |
|---:|---|---|---|
| 1 | §2.7.1.2 inverse-filter loop: hoist the per-pixel `match (x, y)` + per-method dispatch out of the inner loop (split first-row / first-column / interior, one specialised loop per `F`) — the lossless `inverse_predictor` got exactly this border-rule hoist in r180 | gradient at 4.6× the memcpy floor; the dispatch + edge-case match is re-evaluated every pixel | `inverse_filter_{none,horizontal,vertical,gradient}_128x128` |
| — | `ALPH` `Lossless` de-compression (≈ the whole `decode_alpha_lossless_extracted` cost) | **already covered** — it is the VP8L lossless decoder (`read_symbol` / `lossless_decode*`), optimized in r284/r287; not alpha-specific | `decode_alpha_lossless_extracted`, `read_symbol`, `lossless_decode_mixes` |

### Round-291 verification

* **Byte identity:** no `src/` change — bench-only round (new
  `benches/alpha_decode.rs` + its `[[bench]]` stanza). All 436 lib tests
  pass; the `alpha_decode` bench's per-cell sanity assertions confirm the
  fixture's alpha plane and the synthetic uncompressed payloads decode to
  the expected `128 × 128` plane before timing.
* `cargo fmt --check` + `cargo clippy -p oxideav-webp --all-targets
  --no-deps -D warnings` clean.

## Round-290 (2026-06-13): `yuv420_to_rgba` chroma-pair hoist (rank-1 lossy opt)

PROFILE-OPT depth round acting on the round-289 rank-1 candidate:
`yuv420_to_rgba`, the only fully webp-owned lossy decode hot loop (4:2:0
nearest-neighbour chroma up-sample + RFC 6386 §9.2 / RFC 9649 §10 BT.601
full-range YCbCr→RGB matrix). **No output change** — decoded bytes are
byte-for-byte identical.

### The optimization

Two structural wins, both byte-preserving:

1. **Chroma-pair hoist.** §9.2 nearest-neighbour 4:2:0 maps the two luma
   pixels `(2k, 2k+1)` of a row to the same chroma column `k`. The
   matrix's three chroma contributions depend only on `(Cb−128, Cr−128)`,
   so they are now computed **once per chroma column** (`chroma_offsets`)
   and reused across both pixels of the pair — only the luma offset
   `Y << 16` differs per pixel. The previous loop re-evaluated all three
   chroma multiplies for every output pixel, i.e. twice per chroma sample.
   The per-pixel arithmetic (`(Y << 16) + offset + HALF >> 16`, clamp) is
   unchanged; `ycbcr_to_rgb` was refactored to call the same
   `chroma_offsets` so the two forms share one coefficient source and are
   provably identical. A new oracle test
   (`yuv420_to_rgba_matches_per_pixel_reference_across_dimensions`) checks
   the hoisted loop against the per-pixel `ycbcr_to_rgb` form across 9
   even/odd dimension combinations with non-neutral chroma.
2. **Pre-sized output + row slices.** The output is a single `vec![0; …]`
   written through per-row sub-slices instead of four `Vec::push` calls
   per pixel; the row-slice bounds let the optimiser drop per-pixel
   capacity checks and reallocation branches.

### Before/after (this host, `--quick`, median)

| Cell | r289 (before) | r290 (after) | Δ |
|---|---:|---:|---:|
| `yuv420_to_rgba_16x16` | 530 ns | 346 ns | **−35%** |
| `yuv420_to_rgba_128x128` | 33.3 µs | 10.5 µs | **−68%** |
| `yuv420_to_rgba_256x256` | 130.4 µs | 36.5 µs | **−72%** |

At fixture size (128×128) the crate-owned conversion drops from ~34 µs to
~10.5 µs — roughly a 23.5 µs cut on every lossy still that path decodes.

### Round-290 verification

* **Byte identity:** all 436 lib tests (435 prior + the new oracle test) +
  integration binaries (`fixture_walks`, `vp8_lossy_roundtrip`,
  published-API, standalone e2e) pass; the new
  `yuv420_to_rgba_matches_per_pixel_reference_across_dimensions` test
  proves the hoisted loop equals the per-pixel reference byte-for-byte.
  `cargo fuzz run decode_still_paths` (3170 runs) and `decode` (9.6M runs)
  over the committed corpus produced no crash or divergence.
* `cargo fmt --check` + `cargo clippy --all-targets --no-deps -D warnings`
  clean. The only `src/` change is `vp8_decode.rs` (`yuv420_to_rgba`
  rewrite + `chroma_offsets` helper + `ycbcr_to_rgb` refactored onto it,
  now `#[cfg(test)]` as the oracle).

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
