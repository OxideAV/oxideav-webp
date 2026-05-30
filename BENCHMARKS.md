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
