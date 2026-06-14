# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH + ANIM +
ANMF). Decoder and encoder both at production status as of 2026-05-27.

* Full **decode** of every container variant: simple-lossy (VP8),
  simple-lossless (VP8L), extended (`VP8X`) with `ALPH` alpha plane,
  ICCP / EXIF / XMP metadata, and animated WebP (`ANIM` + `ANMF`).
* **Encode** of complete `.webp` files in both lossless (VP8L) and
  lossy (VP8) modes, plus complete animated `.webp` files.
* Decoded pixels land in a tightly-packed `Vec<u8>` of `width * height
  * 4` RGBA bytes — drops directly into [`image`](https://crates.io/crates/image)'s
  `ImageBuffer::from_raw` with zero copy.
* The full crates.io `0.1.2` public surface is reachable, both with
  the default `registry` build and under `--no-default-features`.
  [`tests/api_compat_0_1_2.rs`](./tests/api_compat_0_1_2.rs) is the
  29-test compile-only assertion suite that pins every published
  symbol in place.

## Install

```toml
# Standalone — flat RGBA in / flat RGBA out, no framework dep:
[dependencies]
oxideav-webp = { version = "0.1", default-features = false }

# With the OxideAV runtime:
[dependencies]
oxideav-webp = "0.1"
```

| Feature | Default | What it does |
|---|---|---|
| `registry` | ✅ on | Pulls `oxideav-core` plus the framework-trait factories. Cascades into `oxideav-vp8/registry` so the VP8-lossy encode delegation can reach the sibling crate's factories. With this off, **lossless encode/decode + animation + metadata extraction all still work**; only the VP8-lossy *encode* requires `registry`. |
| `simd` | off (nightly only) | Opt-in `std::simd` acceleration of the hottest pixel-repack loop (`Vp8lImage::to_rgba`). Requires a nightly rustc because it activates `#![feature(portable_simd)]`. Byte-identical to the scalar path (asserted by `vp8l::tests::to_rgba_simd_matches_scalar_byte_for_byte`); see [`BENCHMARKS.md`](./BENCHMARKS.md) for the round-170 before/after numbers. |

### Benchmarks

The crate ships twenty-eight criterion benches under `benches/`,
grouped by domain:

* **End-to-end** — `lossless_decode`, `lossless_encode`,
  `lossless_decode_mixes` (round 283: full-file decode per elected
  §4 transform mix — predictor / color-indexing / cross-color /
  subtract-green / no-transform, the elected list asserted at
  setup), `anim_decode` (round 283: §2.7.1.1 full-timeline animation
  decode, all-keyframe vs dirty-rect-delta `ANMF` layouts),
  `metadata_walk` (round 283: `extract_metadata` chunk walk at three
  chunk-count / payload tiers), and `lossy_decode` (round 289: the
  §2.5 `VP8 ` lossy path at three altitudes — full public
  `decode_webp`, `decode_lossy_rgba` on the extracted bitstream, and
  the crate-owned `yuv420_to_rgba` YCbCr→RGB conversion loop in
  isolation; the sibling `oxideav-vp8` decoder owns the
  entropy/IDCT/loop-filter work), and `alpha_decode` (round 291: the
  §2.7.1.2 `ALPH` alpha-plane decode — the rank-1 webp-owned lossy
  cost — at three altitudes: public `decode_alpha_plane` e2e,
  `alph::decode_alpha` on the extracted payload, and the Stage-2
  inverse-filter per-pixel loop in isolation, one cell per `F` method).
* **Decoder §4.x inverse transforms** — `inverse_predictor`
  (per-mode), `inverse_color` (per `size_bits`),
  `inverse_color_indexing` (per palette tier),
  `inverse_subtract_green`, `inverse_color_table`, plus the
  `argb_to_rgba` repack.
* **Encoder forward passes** — `predictor_subtract`,
  `apply_subtract_green`, `lz77_match`, `lz77_chain` (round 286: the
  §5.2.2 matcher across five hash-chain-depth regimes — period-2/4/64
  repeats, near-unique, gradient), `pick_block_cte` (the §3.5.2
  chooser walk), `meta_prefix_cluster` (round 294: the §6.2.2
  entropy-image block-clustering heuristic behind
  `encode_with_meta_prefix` — coarse-RGB-histogram Lloyd's k-means
  across content regime / `num_groups` / image-size sweeps), the
  §5.2.2 `value_to_prefix` split, and `distance_code` (round 300: the
  §5.2.2 `pixel_distance_to_distance_code` chooser run twice per match
  — a 120-entry `DISTANCE_MAP` scan picking the smallest distance code
  — across RLE / row-above / close-neighbour / no-match regimes). Round
  301 gave the chooser a smallest-code early-out: because map codes
  occupy `1..=120` and the scan-line fallback is `D + 120 ≥ 121`, the
  *first* entry (in ascending code order) whose `max(xi + yi·W, 1)`
  equals the distance is already the smallest valid code, so the scan
  returns on first match instead of running all 120 entries. The chosen
  code — and therefore every emitted byte — is unchanged (proven by an
  equivalence test against the full no-early-out scan over distances
  1..=400 + a large tail across six widths); the matching regimes drop
  from ~64 µs/cell to ~0.8–2.4 µs (≈30–160× on the inner-loop probe),
  while the genuine no-match worst case (`dist_large_nomatch`) still
  scans all 120 entries as before.
* **Entropy / prefix-code chain** — `build_code_lengths` and
  `canonical_codes` (encoder §3.7.2) and `prefix_from_code_lengths`
  (decoder §6.2.1), each over the four §3.7.1 alphabets
  (distance-40 / literal-256 / green-281 / green-2328) in dense and
  sparse frequency regimes, plus `read_symbol` (round 286: the
  §6.2.1 per-symbol reader — the rank-1 decode hotspot — across the
  primary-table fast path vs the > 8-bit walk continuation), the
  per-call `read_lz77_value` (§3.6.2.2 Table 4 regimes) and
  `color_cache_hash` (§3.6.2.3 `code_bits` 1 / 4 / 8 / 11) decoder
  benches, plus `backward_reference` (round 297: the §5.2.2 decoder
  LZ77 copy-back `apply_backward_reference` — the run replay that
  mirrors the `lz77_match` / `lz77_chain` encoder matchers — across
  non-overlap / partial-overlap / `dist == 1` RLE / many-short-run
  regimes).

Rounds 277 / 278 rewrote the §3.7.2 / §6.2.1 length-then-code chain
bit-identically (sorted-leaf two-queue merge + single-rescan counting
sort + O(1)-per-adjustment length cap; capped dense `green2328`
417.8 µs → 26.4 µs end to end), rounds 280 / 281 hoisted the
§4.1 / §3.5.2 encoder chooser walks out of their per-pixel loops
(`lossless_encode_natural_128` ~170 ms → ~120 ms), and round 284 gave
the §6.2.1 `read_symbol` decoder a 256-entry peeked-bits primary
lookup table (entropy-heavy full-file decodes −37% to −49%,
bit-identical across the full fixture corpus and pinned in CI by a
corpus-wide decode digest test). Round 286 (benchmark mode, `src/`
byte-identical) added the `read_symbol` and `lz77_chain` harnesses
that isolate the rank-1 decode and rank-3 encode hotspots, measured
the long-code (> 8-bit) read path at +27% per symbol over the
primary-table floor, and ranked the decoder 9–11-bit spill table as
the next PROFILE-OPT target. Round 287 acted on that candidate:
the per-bit §6.2.1 walk now resolves "is there a code row at this
length?" through a 16-byte direct length→row side table instead of a
linear rescan per bit — a 2.33× speedup on the worst-case
many-distinct-length walk (`read_symbol_manylen16_walk` 86.8 → 37.2 µs),
byte-identical, with no added cache footprint; the spill table itself
was prototyped and rejected as an L1-thrashing regression. Round 289
(benchmark mode, decoded bytes identical) added the `lossy_decode`
harness — the first coverage of the §2.5 `VP8 ` lossy path — and a
ranked lossy-decode hotspot map: of a 128×128 lossy frame's ≈359 µs
end-to-end decode, the container walk + `ALPH` layering is ≈52%, the
sibling `oxideav-vp8` decode (entropy + IDCT + intra-pred + loop
filter, out of this crate's scope) ≈39%, and the crate-owned
`yuv420_to_rgba` YCbCr→RGB conversion ≈9% — the latter purely
per-pixel-bound and the cleanest A/B target for a future SIMD pass (the
lossy analogue of the `argb_to_rgba` SIMD treatment). Round 290 acted on
that candidate: `yuv420_to_rgba` now hoists the §9.2 chroma-matrix terms
out of the per-pixel loop — the two luma pixels of a 4:2:0 pair share one
chroma column, so the three `(Cb−128, Cr−128)` contributions are computed
once per column and reused, and the output is written through pre-sized
per-row slices instead of per-pixel `Vec::push`. The conversion drops from
≈34 µs to ≈10.5 µs at fixture size (−68%; −72% at 256×256), byte-for-byte
identical — proven by a per-pixel oracle test across 9 even/odd dimensions
and by `cargo fuzz` (decode_still_paths + decode, no divergence). Round 291
(benchmark mode, no `src/` change) added the `alpha_decode` harness over
the §2.7.1.2 `ALPH` decode — the rank-1 webp-owned lossy cost the round-289
map had sized only by subtraction — and refined that map: a direct
measurement shows the container walk is ≈1 µs (negligible) and the rank-1
cost is almost entirely the headerless VP8L lossless decode inside
`decode_alpha` (already covered by `read_symbol` / `lossless_decode*`),
while the genuinely alpha-specific §2.7.1.2 inverse-filter loop ranks
Gradient (43.7 µs) > Horizontal (21.8 µs) > Vertical (13.5 µs) > None
(9.5 µs) at 128×128 — flagging a per-method border-rule hoist (the r180
`inverse_predictor` treatment) as the next PROFILE-OPT target. Round 293
acted on that candidate: the §2.7.1.2 Stage-2 inverse filter now dispatches
on `F` once and splits each method into a one-shot border pass (top-left /
first-row / first-column) plus a tight interior loop, instead of
re-evaluating a `match (x, y)` + `match filtering` on every pixel. `None`
becomes a plain identity move (no per-pixel work) and drops 9.5 µs → 0.23 µs
(−97%); `Vertical` — whose predictor reads the row above, so the interior
loop vectorises — drops 13.5 µs → 1.57 µs (−88%); `Horizontal` / `Gradient`
are flat (their left-neighbour serial dependency, not the dispatch, was the
bound). Byte-for-byte identical — proven by a new per-pixel oracle test
across 9 dimension/method combinations and by 400 K `decode_alph` fuzz runs
with no divergence. Round 294
(benchmark mode, no behavioural change — one `fn` → `pub fn` visibility
widen on `cluster_blocks_by_histogram_distance`, matching the
`pick_block_cte` exposure pattern) added the `meta_prefix_cluster`
harness over the encoder's §6.2.2 entropy-image block-clustering
heuristic — the last encode stage sized only by subtraction inside the
`lossless_encode` e2e number — and ranked it: the per-pixel
feature-binning pass dominates (≈70–80% of clustering self-time, isolated
by the uniform-content cell that skips the Lloyd loop; the kernel is
pixel-bound not block-bound), the Lloyd assignment/update loop is a clear
second only on poorly-separated content (gradient +36% over a clean
bimodal split), and `num_groups` 2→4 is nearly free at the default block
size — flagging a feature-pass scattered-write reduction as the next
PROFILE-OPT target. Round 296 returned to the rank-1 lossless-decode
hotspot `inverse_predictor`: its interior loads the per-pixel predictor
mode every pixel though the mode is constant across each `1 <<
size_bits` block (`size_bits = ReadBits(3) + 2 ∈ [2, 9]`, so blocks are
always multi-pixel). A per-block mode hoist (mirroring the round-207
`inverse_color` CTE hoist) was implemented and proven byte-identical —
the existing cross-check test plus an FNV-1a A/B over all seven
`lossless-*` fixtures both matched — but yielded no measurable win (the
interior is dominated by the 14-way `predict()` dispatch, not the mode
load) and the host was saturated during measurement, so the original
body was retained per the round-224 precedent. The realistic block path
is now benched (`inverse_predictor_blocks16_mixed_256x256`, `size_bits
= 4`), filling the gap left by the pre-existing `size_bits = 0` cells.
Numbers,
profile findings, the full
round-283 regression re-run (stable + nightly `simd`), and the
optimization log live in [`BENCHMARKS.md`](./BENCHMARKS.md). Run
with:

```text
CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
  cargo bench --manifest-path crates/oxideav-webp/Cargo.toml \
    --bench <name> -- --quick
```

### Fuzzing

Thirty-two [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
targets live under [`fuzz/fuzz_targets/`](./fuzz/fuzz_targets):
`decode` and `extract_metadata` feed arbitrary bytes through the two
public single-shot entry points; `roundtrip_lossless` synthesises a
≤64 × 64 RGBA tile from fuzz-controlled bytes and asserts the §3
lossless contract pixel-for-pixel across `encode_webp_lossless` →
`decode_webp`; `roundtrip_animated` (round 238) widens the same
contract across the §2.7.1.1 animation carrier — a fuzz-controlled
1..8-frame animation (canvas ≤ 32 × 32) goes through
`build_animated_webp` → `decode_webp` and the frame count + per-frame
width/height + per-frame `duration_ms` + per-frame RGBA bytes are
asserted byte-identical; `decode_alph` (round 255) drives the
§2.7.1.2 ALPH standalone entry point `alph::decode_alpha` directly
across the four filter methods (none / horizontal / vertical /
gradient) and the two compression methods (raw + headerless §3 VP8L)
with `plane.len() == width * height` asserted on success;
`parse_vp8x` (round 256) drives the §2.7.1 VP8X chunk parser
standalone entry point `vp8x::Vp8xHeader::parse` directly across the
full §2.7.1 Figure 7 flag-octet / reserved-field / canvas-dimension
cross-product with every successfully-decoded field cross-checked
against the input bytes the parser observed and every error branch
cross-checked against the §2.7.1 refusal triggers; `parse_anmf`
(round 257) drives the §2.7.1.1 ANMF chunk header parser standalone
entry point `anmf::AnmfHeader::parse` directly across the full
§2.7.1.1 Figure 9 5 × uint24 + info-byte cross-product (Frame X * 2
doubling, Frame W/H Minus One + 1 resolution, uint24 LE duration,
info-byte Reserved / B / D extraction at bits 7..2 / 1 / 0) with
every successfully-decoded field cross-checked against the input
bytes the parser observed and the `PayloadTooShort` branch
cross-checked against the §2.7.1.1 16-byte minimum; `parse_anim`
(round 258) drives the §2.7.1.1 ANIM chunk parser standalone entry
point `anim::AnimHeader::parse` directly across the full §2.7.1.1
Figure 8 BGRA × loop-count cross-product (BGRA byte-order
background, `as_u32_le()` matching the LE u32 reload, LE u16 loop
count, `loops_forever()` predicate) with the `BadPayloadLength`
branch cross-checked against the §2.7.1.1 fixed 6-byte length;
`parse_alph` (round 259) drives the §2.7.1.2 ALPH info-byte parser
standalone entry point `alph::AlphHeader::parse` directly across the
full §2.7.1.2 Figure 10 `Rsv|P|F|C` 2-bit-field cross-product
(MSB-first bit decomposition at bits 7..6 / 5..4 / 3..2 / 1..0,
typed-variant mapping for the `C` / `F` / `P` enums including the
`Reserved(_)` variants on undefined 2 / 3, fixed `bitstream_offset
== 1`) with the `EmptyPayload` branch cross-checked against the
§2.7.1.2 requirement that the payload carry at minimum the one info
byte; `parse_transform_list` (round 260) drives the §4 VP8L
transform-list reader standalone entry point
`vp8l_stream::TransformList::read` directly across the full §4
transform-presence loop (per-type fixed fields, duplicate-detection
refusal, deferred §5 entropy-body boundary) with `Ok(list)` cross-checked
against `transforms().len() <= 4`, no repeated `TransformType` across
entries, §4.1 / §4.2 `size_bits ∈ [2, 9]`, §4.4 `color_table_size ∈
[1, 256]` plus the threshold-table `width_bits` derivation, the
`body_bit_position()` within the slice's bit length, and the
`stopped_at_entropy_body()` flag consistent with the last entry's
`has_entropy_body()`; `parse_meta_prefix` (round 261) drives the
§5.2.3 color-cache info + §6.2.2 meta-prefix + §6.2
5-prefix-code-group reader standalone entry point
`meta_prefix::MetaPrefixHeader::read` directly across the full
§5.2.3 + §6.2.2 preamble cross-product (color-cache enable bit +
4-bit `color_cache_code_bits` range gate, §6.2.2 `ImageRole`
dispatch, `EntropyImagePending` `prefix_bits = ReadBits(3) + 2`
range, and the §6.2.2 `DIV_ROUND_UP(image_dim, 1 << prefix_bits)`
entropy-image dimension derivation) with `Ok(header)` cross-checked
against the §5.2.3 `code_bits ∈ {0} ∪ [1, 11]` range, the
`is_enabled()` / `size()` derivations, the `EntropyCoded` role
never reaching `EntropyImagePending` (the meta-prefix bit is
absent for sub-images), the `EntropyImagePending` branch's
`prefix_bits ∈ [2, 9]`, the recomputed entropy-image
width/height matching the recorded values, and the
`entropy_image_bit_position` within the slice's bit length;
`Err(InvalidColorCacheCodeBits)` cross-checked against the
`value ∈ {0} ∪ [12, 15]` rejection-window;
`parse_container` (round 262) drives the §2.3 / §2.4 RIFF/WEBP
chunk-walker standalone entry point `container::parse` directly
with every byte of the fuzz buffer attacker-controlled (including
the §2.4 `File Size` field at bytes 4..8 and every per-chunk `Size`
field at offsets `+4..+8` relative to its header) with `Ok(container)`
cross-checked against the §2.3 + §2.4 carrier rules (`riff_file_size`
== LE uint32 at `buf[4..8]`, every recorded `WebpChunk` cross-checked
byte-for-byte against the buffer it points into — FourCC at
`buf[header_offset..+4]`, LE uint32 `Size` at
`buf[header_offset + 4..+8]`, `payload_end - payload_start ==
size as usize`, `payload_end` inside both the buffer length and the
§2.4 declared RIFF window, on-disk order with
`chunks[i+1].header_offset == chunks[i].payload_end + (size & 1)`, the
`is_extended()` / `is_vp8_lossy()` / `is_vp8_lossless()` predicates
pure functions of FourCC, the `chunks_with_fourcc` /
`first_chunk_with_fourcc` helpers matching a manual filter) and every
error variant cross-checked against the §2.3 / §2.4 refusal trigger
(TooShortForHeader.got == buf.len() < 12; NotRiff.got == buf[0..4]
!= 'RIFF'; NotWebp.got == buf[8..12] != 'WEBP' with buf[0..4] ==
'RIFF'; RiffSizeOverflowsBuffer.declared == LE uint32 at buf[4..8]
with 8 + declared > buffer_len; TruncatedChunkHeader.offset >= 12
inside declared window with < 8 bytes remaining;
ChunkPayloadOverflowsRiff.offset >= 12 with 8-byte header fitting,
declared == LE uint32 at chunk header, available == declared_end
- (offset + 8), declared > available; MissingPadByte.offset >= 12
with declared Size odd, payload itself fitting, and pad byte at
payload_end + 1 outside declared window); `distance_code` (round 263)
drives the §5.2.2 distance-code-to-pixel-distance pure-function
lookup standalone entry point
`vp8l_decode::distance_code_to_pixel_distance` directly across the
full attacker-reachable `(distance_code, image_width)` cross-product
(a series of `(image_width, distance_code)` u32 LE pairs sliced
out of the fuzz buffer, with the §3.4 14-bit image-width ceiling
applied and the §5.2.2 `distance_code >= 1` precondition honoured)
with every returned `D` cross-checked against the §5.2.2 spec
formula (`max(1, xi + yi * image_width)` for codes `1..=120` via
the 120-entry `DISTANCE_MAP`, `distance_code - 120` for codes
`> 120`) and the §5.2.2 clamp guarantee (`D >= 1` always — either
from the clamp on the neighborhood-lookup branch or from the
smallest reachable raw scan-line distance of `121 - 120 = 1`),
plus pure-function determinism asserted via a double-call equality
check; `color_cache` (round 264) drives the §5.2.3
lossless-color-cache primitives standalone entry point
`vp8l_decode::ColorCache` directly across the full attacker-reachable
`code_bits ∈ [1, 11]` × `argb ∈ [0, u32::MAX]` cross-product (the
first fuzz byte fixes the §5.2.3 `code_bits` remapped into the
permitted window per the §5.2.3 "compliant decoders MUST indicate a
corrupted bitstream for other values" rule, every subsequent 4-byte
word is forwarded verbatim as a fuzz-controlled ARGB color into
`ColorCache::insert`) with every hash cross-checked against the
§5.2.3 spec formula `(0x1e35a7bd * argb) >> (32 - code_bits)`, every
insert/lookup round trip cross-checked against the §5.2.3 single-slot
single-write spec text ("Only one lookup is done in a color cache;
there is no conflict resolution"), every per-slot lookup cross-checked
against a parallel shadow model that records the §5.2.3
most-recently-inserted-wins overwrite behaviour, the §5.2.3 cache
initialization invariant cross-checked on a fresh cache (`size() ==
1 << code_bits`, every slot reads as `Some(0)`, `lookup(size())`
reads as `None`), and pure-function determinism asserted on the
insert sequence by rebuilding a replay cache from the same fuzz
bytes and verifying every slot agrees with the primary cache;
`inverse_predictor_color` (round 265) drives the §4.1 inverse-predictor
+ §4.2 inverse-color in-place transform passes standalone entry
points `vp8l_transform::inverse_predictor` +
`vp8l_transform::inverse_color` directly across the full
attacker-reachable `(width, height, size_bits, residual_pixels,
sub_resolution_image)` cross-product (the first three fuzz bytes
fix the §4.1 / §4.2 `(width, height, size_bits)` carrier triple with
`width` / `height` masked into `[1, 32]` for iteration cost and
`size_bits` remapped into `[0, 9]` to cover the full §4.1 / §4.2
`ReadBits(3) + 2` window plus the `size_bits == 0` hoist branch;
every subsequent 4-byte little-endian word is forwarded verbatim as
a fuzz-controlled ARGB residual pixel and, after `width * height`
words, as a fuzz-controlled sub-resolution predictor / color image
pixel) with the §4.1 left-topmost rule cross-checked against the
spec text (`pred_pixels[0] == residual[0] + 0xff000000` per channel
mod 256), the §4.1 single-column left-column rule cross-checked
against the §4.1 "all pixels on the leftmost column are T-pixel"
spec text (every `(0, y)` for `y >= 1` equals `residual + T` per
channel mod 256), the §4.1 single-row top-row rule cross-checked
against the §4.1 "all pixels on the top row are L-pixel" spec text
(every `(x, 0)` for `x >= 1` equals `residual + L` per channel mod
256), the §4.2 alpha-and-green preservation invariant cross-checked
against the §4.2 spec text ("The alpha and green channels are left
as is"), the §4.2 zero-CTE no-op invariant cross-checked by
re-running the pass against an all-zero sub-resolution image (every
per-pixel output equals the input), the §4.2 per-block constancy
invariant cross-checked against the §4.2 block structure (two
same-block pixels with equal pre-pass RGB produce equal post-pass
red + blue), and both passes' early-return contract cross-checked
against the §4.1 / §4.2 `(width == 0 || height == 0)` no-op (the
pixel buffer is byte-identical to the pre-call snapshot);
`inverse_subtract_green_indexing` (round 266) drives the §4.3
inverse-subtract-green + §4.4 inverse-color-table + §4.4
inverse-color-indexing transform passes standalone entry points
`vp8l_transform::{inverse_subtract_green, inverse_color_table,
inverse_color_indexing}` directly across their full
attacker-reachable input cross-products (the first three fuzz bytes
fix the §4.3 / §4.4 `(orig_width, height, table_size)` carrier triple
with `orig_width` / `height` masked into `[1, 32]` for iteration cost
and `table_size` mapped into the §4.4 wire window `[1, 256]`; every
subsequent 4-byte little-endian word is forwarded verbatim first as a
fuzz-controlled ARGB §4.3 input pixel, then as a fuzz-controlled §4.4
color-table delta entry, then as a fuzz-controlled §4.4 packed-index
ARGB pixel) with the §4.3 alpha-and-green preservation invariant
cross-checked against the spec text (every pixel's red byte equals
input red + input green mod 256, every pixel's blue byte equals input
blue + input green mod 256, alpha + green bytes byte-identical), the
§4.3 per-pixel locality invariant cross-checked by running the pass
on single-pixel inputs at the first eight positions and asserting the
solo output matches the multi-pixel output, the §4.3 zero-green-byte
no-op cross-checked against the `(red + 0) = red` reduction, the §4.4
color-table seed preservation cross-checked against the spec text
(`table[0]` is left untouched), the §4.4 color-table running-sum
invariant cross-checked against the §4.4 "adding the previous color
component values by each ARGB component separately and storing the
least significant 8 bits of the result" spec text (every `i >= 1`
entry is the per-channel running sum mod 256 of the original input
bytes), the §4.4 color-indexing output-length cross-checked against
the `orig_width * height` carrier contract, the §4.4 color-indexing
palette-lookup cross-checked against the §4.4 spec formula (output
pixel `(x, y)` is `color_table[((packed_green >> ((x % count) *
bits)) & mask) as usize]` with `width_bits` derived from the table
size via the §4.4 threshold table, falling back to transparent black
`0x00000000` when the index is out of range), and the §4.4
color-indexing empty-table edge case cross-checked against the §4.4
"unused indices map to transparent black" rule; the §4.3 empty-buffer
and §4.4 single-element-table degenerate no-op branches are
cross-checked unconditionally on every iteration. `backward_reference`
(round 267) drives the §5.2.2 backward-reference assembler standalone
entry point `vp8l_decode::apply_backward_reference` directly: the fuzz
buffer fixes a `(prefill_len, length, dist, total_pixels)` carrier
tuple (`prefill_len` masked to `[0, 4096]`; `dist` floored at 1 to
honour the §5.2.2 `D >= 1` precondition the
`distance_code_to_pixel_distance` clamp guarantees; `total_pixels`
alternated between `prefill_len + length + headroom` and a shrunk
value below `prefill_len + length` so both the success / exact-fit
path and the §5.2.2 overflow refusal are routinely reached) plus a
stream of fuzz-controlled ARGB pre-fill pixels, with every `Ok`
outcome cross-checked against the §5.2.2 copy contract (returned range
equals `position..position + length`, exactly `length` pixels
appended, the already-decoded prefix byte-identical, every appended
pixel matching a parallel reference LZ77 walk `out[position + i] ==
out[position + i - dist]` read after the preceding writes — the
overlapping `dist < length` self-repeat included), the §5.2.2
underflow refusal cross-checked against its `dist > position` trigger
(fields echo the call, buffer byte-identical to its pre-call
snapshot), the §5.2.2 overflow refusal cross-checked against its
`position + length > total_pixels` trigger (with the underflow guard
having passed), and pure-function determinism cross-checked by
replaying a successful run from the same pre-fill;
`meta_prefix_index` (round 268) drives the §6.2.2 meta-prefix
block-lookup table standalone entry points
`vp8l_decode::MetaPrefixIndex::{from_parts, meta_code_for}` directly
across the full `(prefix_bits, block_width, block_height,
meta_codes)` cross-product (the first fuzz byte fixes `prefix_bits`
masked to `[0, 15]` so the §6.2.2 `ReadBits(3) + 2` window `[2, 9]`
and its rejection are both routinely reached; the next two bytes fix
the block grid in `[0, 32]²` with 0 reaching the degenerate-grid
refusal; a skew byte shifts the supplied code count off the
`block_width * block_height` expectation by `[-2, +2]`; every
remaining 2-byte LE word is forwarded verbatim as a meta-prefix code)
with every `Ok` index cross-checked against the §6.2.2 carrier rules
(accessors echo the parts, `num_prefix_groups() == max(entropy image)
+ 1`, and `meta_code_for(x, y)` at all four corners of every block's
`(1 << prefix_bits)`-pixel-square covered area matching the §6.2.2
position formula `meta_codes[(y >> prefix_bits) * block_width + (x >>
prefix_bits)]`), every error variant cross-checked against its §6.2.2
refusal trigger in precedence order (`InvalidPrefixBits` ⇔
`prefix_bits ∉ [2, 9]`; `EmptyIndex` ⇔ zero-block grid with the
prefix-bits gate passed; `CodeCountMismatch` ⇔ count off the
expectation with both earlier gates passed, `expected` / `got`
echoing the call), and constructor determinism cross-checked by
rebuilding from the same parts plus round-tripping the index's own
accessors back through `from_parts`; `decode_entropy_image`
(round 270) drives the §6.2.2 *entropy image* decode path standalone
entry point `vp8l_decode::decode_entropy_image` directly across the
`(prefix_bits, prefix_image_width, prefix_image_height, bitstream)`
cross-product (the first three fuzz bytes fix the §6.2.2 carrier triple
— `prefix_bits` masked to `[0, 15]` since the function records it as an
opaque carrier without re-deriving a block size, the block dimensions
modulo 9 so the §7.3 sub-image decode stays bounded and 0 reaches the
§6.2.2 degenerate-dimension refusal; the remaining bytes feed a
zero-positioned `BitReader` the §7.3 `entropy-coded-image` bit
sequence) with every `Ok` index cross-checked against the §6.2.2 +
§7.3 carrier rules (accessors echo the carrier triple; §7.3 one
meta-code per block `meta_codes().len() == prefix_image_width *
prefix_image_height`; §6.2.2 `num_prefix_groups() == max(meta_codes) +
1`; the §6.2.2 fold `meta_prefix_code == (entropy_pixel >> 8) & 0xffff`
cross-checked against an independent decode of the same bytes through
the public sibling `decode_entropy_coded_image` — the harness refolds
that decode's raw ARGB pixels and asserts byte-equality with the
meta-codes plus both readers advancing to the same bit position; the
§6.2.2 carrier asymmetry where `from_parts` reproduces the index iff
`prefix_bits ∈ [2, 9]` and refuses with `InvalidPrefixBits` otherwise;
and determinism by replaying the same bytes + carrier triple), the
§6.2.2 degenerate-dimension refusal pinned to the `EmptyEntropyImage`
variant echoing the carrier dimensions iff at least one is zero, and
every other bitstream-level refusal required only to return a `Result`
rather than panic. A 30 s smoke pass cleared 8.9 M runs with no
crashes (reaching the §5.2 `read_lz77_value` / `apply_backward_reference`
/ `distance_code_to_pixel_distance` core through the entropy-coded
sub-image); `decode_entropy_coded_image` (round 271) drives the §7.3
*entropy-coded-image* decode path standalone entry point
`vp8l_decode::decode_entropy_coded_image` directly — the §7.3 ABNF
building block beneath the round-270 §6.2.2 entropy image (which wraps
it and folds its pixels) and the §4.1 / §4.2 / §4.4 sub-resolution
images — across the `(width, height, bitstream)` cross-product (the
first two fuzz bytes fix the §7.3 carrier dimensions each modulo 9 so
the §5.2 / §6.2 decode loop stays bounded and 0 reaches the §7.3
degenerate-dimension `EmptyEntropyImage` refusal; the remaining bytes
feed a zero-positioned `BitReader` the §5.2.3 color-cache-info bit +
one §6.2 prefix-code group + §5.2 LZ77 / color-cache data) with every
`Ok` image cross-checked against the §7.3 carrier rules (`width()` /
`height()` echo the carrier, `pixels().len() == width * height`, the
success path reachable only with both dimensions ≥ 1, the reader never
advancing past the slice's bit length) and against the §6.2.2 wrapper
(an independent `decode_entropy_image` over the same bytes reproduces
the `(pixel >> 8) & 0xffff` per-pixel fold as its per-block meta-codes
and advances the reader to the same bit position), plus pure-function
determinism cross-checked by replaying the same bytes + dimensions for
a byte-identical pixel buffer at an identical bit position; the §7.3
degenerate-dimension refusal pinned to the `EmptyEntropyImage` variant
echoing the carrier dimensions iff at least one is zero and every other
refusal required only to return a `Result` rather than panic;
`decode_argb` (round 272) drives the §6.2.2 top-level VP8L ARGB
main-image decode path standalone entry point
`vp8l_decode::decode_argb` directly — the §5.1 `spatially-coded-image`
ARGB-role decoder one layer above the round-270 / round-271 entropy-image
harnesses, reading the §5.2.3 `color-cache-info` bit + §6.2.2
meta-prefix bit, dispatching the single-group (one §6.2 prefix-code
group everywhere) vs multi-group (§6.2.2 entropy image →
`num_prefix_groups = max + 1` groups → per-pixel-block group selection
via `meta_code_for`, single §5.2.3 color cache in stream order) paths,
and running the §6.2.3 decode loop — across the `(width, height,
bitstream)` cross-product (the first two fuzz bytes fix the carrier
dimensions each clamped into `[1, 8]` so the success contract holds —
mirroring the §3.4-validated dimensions `decode_argb` is reachable with
— and the image stays ≤ 64 pixels; the remaining bytes feed a
zero-positioned `BitReader` the §6.2.2 ARGB image bit sequence) with
every `Ok` image cross-checked against the §6.2.2 carrier rules
(`width()` / `height()` echo the carrier, `pixels().len() == width *
height`, the reader never advancing past the slice's bit length), plus
pure-function determinism cross-checked by replaying the same bytes +
dimensions for a byte-identical pixel buffer at an identical bit
position, and every refusal (truncation, meta-prefix/color-cache-info
parse failure, entropy-image fault, prefix-code parse failure,
out-of-range green symbol, color-cache or backward-reference fault, or a
meta-prefix code beyond `num_prefix_groups`) required only to return a
`Result` rather than panic. A 30 s smoke pass cleared 2.66 M runs with
no crashes (476 cov / 1690 features over a 269-input corpus);
`decode_lossless` (round 273) drives the §4 transform-list + main-image
full lossless-bitstream decode path standalone entry points
`vp8l_transform::{decode_lossless, decode_lossless_headerless}` directly
— the layer immediately above the round-272 §6.2.2 `decode_argb`: it
walks the §4 / §7.2 optional-transform loop (per-transform §4.x fixed
fields + §5-encoded body via the §7.3 `decode_entropy_coded_image`, the
§4 once-each duplicate refusal, the §4.4 `color_table_size` /
`width_bits` width subsampling), decodes the main §5.1 ARGB image at the
subsampled width, then applies the §4 inverse-transform chain in reverse
read order ("last one first"); `decode_lossless_headerless` is the
§2.7.1.2 / §3 `ALPH` twin reading the same bytes from bit 0 (no §3.4
5-byte image-header skip). The first two fuzz bytes fix the
`(width, height)` carrier each clamped into `[1, 8]` (so the success
contract holds and the decode stays ≤ 64 pixels); the remaining bytes are
the VP8L chunk-payload bits, with every `Ok` image cross-checked against
the §4 / §6.2.2 carrier rules (`width()` / `height()` echo the carrier
even after a §4.4 color-indexing transform un-bundles the internal width
back to the canvas width, `pixels().len() == width * height`) plus
replay determinism, and every refusal required only to return a `Result`
rather than panic. This harness surfaced (on its first run) and the round
fixed a `BitReader::bits_remaining` `usize` underflow that let a
sub-5-byte VP8L chunk payload index out of bounds past the §3.4
image-header skip — now `saturating_sub`. A 40 s smoke pass cleared
3.62 M runs with no crashes after the fix; `prefix_code_group`
(round 274) drives the §6.2 / §6.2.1 *prefix-code-group* reader
standalone entry point `meta_prefix::PrefixCodeGroup::read` directly —
the surface immediately below the round-271 §7.3
`decode_entropy_coded_image`, which reads a §5.2.3 color-cache-info bit
then exactly one `PrefixCodeGroup::read` before the §5.2 pixel loop. A
§6.2 group is the five canonical §6.2.1 prefix codes every VP8L pixel is
decoded with: green + backref-length + color-cache (alphabet `256 + 24 +
color_cache_size` per §6.2.3), red/blue/alpha (each `256`), and backref
distance (`40`), each read via `PrefixCode::read` then the §6.2.1
simple/normal `read_code_lengths` dispatch and the §6.2.1 canonical
`from_code_lengths` Kraft completeness build. The first fuzz byte selects
the §5.2.3 cache size from `{0}` (disabled) or `1 << code_bits` for
`code_bits ∈ [1, 11]` (`{2, 4, …, 2048}`), sizing the §6.2.3 green
alphabet; the remaining bytes feed a zero-positioned `BitReader`. Every
`Ok(group)` is cross-checked against the §6.2.3 / §6.2.1 carrier rules:
each of the five codes' `code_lengths().len()` equals its alphabet, every
nonzero length is `<= 15` (the `MAX_CODE_LENGTH` ceiling), `single_symbol()`
is `Some(s)` iff the length table has exactly one nonzero entry (at `s`)
and `None` iff two or more, `read_symbol` against an all-zero reader
resolves an in-range symbol index, the reader never advances past the
slice bit length, and replaying the same bytes + cache size yields an
equal group at an identical bit position; the §5.2.3
`InvalidColorCacheCodeBits` variant is asserted unreachable (the cache
size is caller-supplied, never read here). A 41 s smoke pass cleared
4.97 M runs with no crashes. `prefix_code` (round 275) drops one layer
further to the §6.2.1 *single canonical prefix-code* reader standalone
entry point `vp8l_prefix::PrefixCode::read` directly — the surface
`PrefixCodeGroup::read` calls five times in green/red/blue/alpha/distance
order. It reads one code's lengths off the wire (the §6.2.1 simple/normal
`read_code_lengths` dispatch) and builds the canonical decoder via
`from_code_lengths` with its Kraft completeness gate and single-leaf
exception, isolated across an attacker-controlled `(alphabet_size,
bitstream)` cross-product where the first fuzz byte selects one of the
wire-reachable §6.2.3 alphabets — `40` (distance), `256` (red/blue/alpha),
or the green `256 + 24 + color_cache_size` for the full
`color_cache_size ∈ {0} ∪ {2, …, 2048}` range — and the remaining bytes
feed a zero-positioned `BitReader`. Every `Ok(code)` is cross-checked
against the §6.2.3 / §6.2.1 carrier rules: `code_lengths().len()` equals
the selected alphabet, every nonzero length is `<= 15`, `single_symbol()`
is `Some(s)` iff exactly one nonzero entry (at `s`) and `None` iff two or
more, `read_symbol` against an all-zero reader resolves an in-range symbol
index, rebuilding from the returned length table through
`from_code_lengths` reproduces an equal code (the §6.2.1 `sum 2^-len == 1`
completeness invariant), the reader never advances past the slice bit
length, and replaying the same bytes + alphabet yields an equal code at an
identical bit position. A 14 s smoke pass cleared 2.00 M runs with no
crashes. `roundtrip_anim_modes` (round 279) is a differential oracle on
the §2.7.1.1 animation *assembly* path
`build_animated_webp_with_options` → `decode_webp` with every per-frame
carrier field fuzz-driven — even `(x, y)` sub-canvas offsets, mixed
`Auto` / `Delta` / `Lossless` frame modes (the dirty-rect sub-frame
encoder), `None` / `Background` disposal, `Overwrite` / `AlphaBlend`
blending, and the `ANIM` loop-count + background-colour options — with
every decoded full-canvas frame snapshot asserted byte-identical to an
independent §2.7.1.1 canvas simulation and duration / loop count /
background colour asserted to carry through. `roundtrip_metadata`
(round 282) is a differential oracle on the §2.7 metadata *write* path:
the two independent extended-layout writers
`build::build_webp_file_with_metadata` and
`encode_vp8l_argb_with_metadata` are driven with fuzz-controlled
§2.7.1.4 `ICCP` / §2.7.1.5 `EXIF` / `XMP ` payloads (presence, length
0..=255 — odd lengths exercising the §2.3 pad byte — and content), a
fuzz-controlled §2.7.1 `L` alpha-hint flag, and fuzz-controlled canvas
dimensions + ARGB pixels; every emitted file is cross-checked against
the §2.7 documented contract — the §2.3/§2.4 walker parses it, the
chunk sequence is the canonical `VP8X, ICCP?, VP8L, EXIF?, XMP?` order
(§2.7.1.4: the color profile "MUST appear before the image data"),
each metadata chunk's `Size` + payload bytes round-trip verbatim, the
§2.7.1 flag octet declares exactly the supplied features with the
canvas dimensions echoed, `extract_metadata` and the `decode_webp`
metadata carry agree byte-for-byte, and the lossless pixels survive
the metadata-bearing layout exactly — with the writer-B
no-alpha/no-metadata demotion to the §2.6 simple single-`VP8L` layout
pinned and both writers' metadata walks asserted identical to each
other. A 12-minute ASan pass cleared 30,543 runs with no crashes and
no assertion failures (3780 cov / 9031 features over a 790-input
corpus). `read_symbol_lut_diff` (round 285) is a differential oracle
on the round-284 §6.2.1 read-symbol fast path: `PrefixCode::read_symbol`
(the 256-entry primary lookup table keyed on the next 8 peeked
wire-order bits, the > 8-bit continuation walk resuming at length 9,
the near-EOF per-bit fallback, and the `MIN_LOOKUP_USED` used-symbol
amortization gate) is run in lockstep against the crate's own
pre-table per-bit row walk, kept as the `#[doc(hidden)]`
`PrefixCode::read_symbol_reference` oracle, over the same bytes — with
the decoded symbol (or typed refusal, including the `PrefixError::Eof`
`bit_pos` / `wanted` / `available` fields), the cursor bit position
after *every* symbol, and the alphabet bound asserted identical. The
code under test is built two ways: *wire mode* reads it off the fuzz
bytes through `PrefixCode::read` at a fuzz-selected §6.2.3 alphabet
(`40` / `256` / `256 + 24 + cache_size`) with the rest of the same
stream as the symbol soup (the on-disk §5 entropy-body layout), and
*table mode* synthesises the per-symbol lengths from fuzz bytes
repaired to an exact §6.2.1 Kraft sum (greedy front fill +
binary-decomposition tail fill), so the mutator steers the used-symbol
count across the table-build gate and the length profile across the
≤ 8-bit fast path, the > 8-bit continuation rows, and the 15-bit
ceiling at will; the §6.2.1 single-leaf-node tree (consumes no bits)
is compared once instead of looped. A 15-minute ASan campaign cleared
36.1 M runs with no divergence. `decode_lossless_lut` (round 285)
re-drives the §4 transform-list + main-image lossless decode entry
points `vp8l_transform::{decode_lossless, decode_lossless_headerless}`
at carrier dimensions widened into `[1, 64]` (≤ 4096 pixels — the
round-273 `decode_lossless` sibling clamps at `[1, 8]`, so its
accepted streams read only a handful of symbols per prefix code), with
the corpus seeded from the VP8L chunk payloads of the committed
fixture corpus plus entropy-heavy reference-encoder-produced streams
(64×64 noise / gradient / plasma tiles) whose §6.2 groups carry
100+-symbol codes with 9..15-bit tails — so the round-284 lookup-table
fast path, its continuation walk, and the word-load
`BitReader::read_bits` / `peek_bits` / `advance_bits` run hot inside
the assembled pipeline (transform sub-images, color cache, LZ77,
inverse-transform chain) under adversarial mutation at every cursor
phase; the round-273 carrier-echo / pixel-count / replay-determinism
contract is asserted unchanged. A 15-minute ASan campaign cleared
16.8 M runs with no crashes; same-session 4-minute regression re-runs
of `prefix_code` (32.2 M), `decode_lossless` (9.4 M), and
`prefix_code_group` (24.2 M) also ran clean. `decode_still_paths`
(round 288) is a differential oracle on the two public still-image
decode entry points `decode_webp` (the published `WebpImage` surface)
and `decode_webp_image` (the low-level `DecodedWebp` surface), seeded
from the in-tree §2.6 lossless + §2.7-extended fixtures. For a
non-animated input the published façade builds its single still frame by
literally calling `decode_webp_image`, so the harness asserts the two
surfaces agree exactly (`Ok` ⇒ `frames.len() == 1` with byte-identical
`frames[0].{rgba, width, height}` + canvas-dimension echo +
`duration_ms == 0` + no §2.7.1.1 carrier; `Err` ⇒ the published path
also `Err`), re-checks the §2.5/§2.6 flat-buffer carrier invariant
(`rgba.len() == width * height * 4`, non-empty iff both dimensions
nonzero) on every decoded still and every composited animation frame,
and asserts `decode_webp_image` replay determinism. The harness surfaced
a libFuzzer OOM — a ~60-byte file declaring a §2.7.1 16 777 154 × 64
animation canvas forced a ~4 GiB eager `Vec`; the round fixed it by
bounding the animation canvas at the §3.4 still-image ceiling
(`MAX_DECODE_DIMENSION = 16384` per side, rejected with `InvalidData`
before allocating). A ~300 s ASan campaign over 25 772 runs is now
crash-free. The §2.5 `VP8 ` *lossy* decode (routed to the `oxideav-vp8`
sibling, which currently panics on some malformed bitstreams at its
inverse-DCT stage) is deliberately skipped from the cross-check pending a
sibling-side hardening. `decode_lossless_image` (round 292) drives the
public top-level lossless façade `decode_lossless_image` — the layer that
walks the §2.3 `RIFF`/`WEBP` container, selects the §2.6 `VP8L` chunk,
reads the chunk's own §3.4 image-header dimensions, and runs the full
§4/§5/§6 decode to a typed `DecodedImage`. Unlike the round-273
`decode_lossless` harness (dimensions supplied *by the harness* over a
bare payload), the decoded dimensions here come from the **file's own**
§3.4 14-bit fields, exercising the §3.4-header → §4-decode
dimension-coherence path end to end; on every `Ok(Some(image))` it
asserts `image.{width,height}` echo the §3.4-resolved chunk dimensions,
the §6.2.2 `width * height` pixel count, a non-empty buffer, and replay
determinism. A cheap structural pre-pass gates the full-decode tail by
declared pixel count so an adversarial `16384 × 16384` header can't blow
the per-iteration budget. The harness surfaced a second libFuzzer OOM —
distinct from the r288 animation-canvas finding: a ~30-byte `VP8L` chunk
declaring a §3.4 `16360 × 12284` still forced `vp8l_decode::decode_image`
to eager-reserve ~800 MiB *before* the EOF-checked §5/§6 loop ran. The
round fixed it by capping the eager `Vec::with_capacity` at
`MAX_EAGER_PIXEL_RESERVATION = 1 << 22` pixels (`eager_pixel_capacity`);
the buffer still grows on demand for a legitimately large image and the
self-terminating loop raises `DecodeError::Eof` on a truncated stream, so
decoded bytes for all valid images are unchanged. A ~120 s ASan campaign
over 48 882 runs is now crash-free, peak RSS 1.2 GiB.
`decode_alpha_plane` (round 295) drives the public *file-level
still-image alpha* entry point `decode_alpha_plane` — the layer that
walks the §2.3 `RIFF`/`WEBP` container, selects the §2.7.1.2 `ALPH`
chunk (`Ok(None)` when absent), resolves the plane dimensions *from the
file itself* (the §2.7.1 `VP8X` 24-bit canvas Width/Height, else the
§2.5 `VP8 ` keyframe header), and decodes the alpha bitstream through
both §2.7.1.2 compression methods (raw + headerless §3 lossless) and all
four filter methods. Unlike `decode_alph` (dimensions supplied *by the
harness* over a bare chunk-payload slice), the dimensions here come from
the file's own §2.7.1 / §2.5 header, exercising the dimension-source →
§2.7.1.2 alpha-decode coherence path end to end. A structural pre-pass
reads the §2.7.1 `VP8X` canvas and gates the decode tail by declared
pixel count so an adversarial canvas can't blow the per-iteration budget.
On every `Ok(Some(plane))` the §2.7.1.2 carrier invariant and replay
determinism are cross-checked. A ~90 s ASan campaign over **23 926 275
runs** (~263 K exec/s, peak RSS 541 MiB) is crash-free — no panic, OOM,
or overflow surfaced; the existing `decode_alpha` `checked_mul` and the
headerless lossless eager-reservation cap already defend this path.
`parse_vp8_chunk` (round 298) drives the §2.5 simple-lossy `VP8 ` chunk
handle standalone entry point `vp8_chunk::WebpLossyChunk::from_payload` —
the keyframe-header peek the §2 RIFF walker reaches only along the
well-formed-container path — over an attacker-controlled byte slice of
arbitrary length. Every successfully-decoded field is cross-checked
against the RFC 6386 §9.1 key-frame header byte layout the parser
observed (the little-endian frame tag from bytes 0..3 with the key-frame
frame-type, `version` at bits 1..3, `show_frame` at bit 4, the 19-bit
`first_partition_size` at bits 5..23, the §9.1 start code `0x9D 0x01 0x2A`
at bytes 3..6, the 14-bit `width` / 2-bit `horizontal_scale` split of the
width word at bytes 6..8, the same split of the height word at bytes
8..10, and `bitstream()` echoing the input verbatim); every refusal
branch is cross-checked against its §9.1 / §2.5 trigger
(`PayloadTooShortForKeyframe` below the 10-byte minimum, `NotAKeyframe`
on an interframe frame-type bit §2.5 forbids, `BadStartCode` echoing
bytes 3..6 verbatim). A ~60 s ASan campaign over **87 063 810 runs**
(~1.43 M exec/s, peak RSS 550 MiB) is crash-free — no panic, OOM, or
overflow surfaced. Run any one with (nightly + `cargo-fuzz` installed):

```text
cargo +nightly fuzz run decode               --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run extract_metadata     --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run roundtrip_lossless   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run roundtrip_animated   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_alph          --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_vp8x           --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_anmf           --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_anim           --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_alph           --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_transform_list --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_meta_prefix    --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_container      --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run distance_code        --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run color_cache          --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run inverse_predictor_color --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run inverse_subtract_green_indexing --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run backward_reference   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run meta_prefix_index    --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_entropy_image --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_entropy_coded_image --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_argb          --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_lossless      --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run prefix_code_group    --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run prefix_code          --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run roundtrip_anim_modes --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run roundtrip_metadata   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run read_symbol_lut_diff --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_lossless_lut  --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_still_paths   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_lossless_image --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run decode_alpha_plane   --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
cargo +nightly fuzz run parse_vp8_chunk      --manifest-path crates/oxideav-webp/fuzz/Cargo.toml
```

## Standalone use (no `oxideav-core`)

### Decode any `.webp` file

```rust
use oxideav_webp::{decode_webp, WebpImage};

let webp_bytes: &[u8] = /* file bytes from disk, HTTP, … */;
let image: WebpImage = decode_webp(webp_bytes)?;

println!("{} × {}, {} frame(s)", image.width, image.height, image.frames.len());
for frame in &image.frames {
    // frame.rgba is a tight Vec<u8> of width*height*4 RGBA bytes,
    // row-major, no per-row padding — drops into `image::ImageBuffer`:
    //
    //   let img = image::RgbaImage::from_raw(frame.width, frame.height,
    //                                        frame.rgba.clone()).unwrap();
    //
    println!("  frame: {}×{}, {} ms", frame.width, frame.height, frame.duration_ms);
}

// ICC / EXIF / XMP are on image.metadata.{icc, exif, xmp} (each Option<Vec<u8>>).
```

### Read metadata only (no pixel decode)

```rust
use oxideav_webp::extract_metadata;

let meta = extract_metadata(webp_bytes)?;
if let Some(icc) = meta.icc.as_deref()  { /* color-management profile */ }
if let Some(exif) = meta.exif.as_deref() { /* EXIF blob */ }
if let Some(xmp) = meta.xmp.as_deref()   { /* XMP UTF-8 XML */ }
```

### Encode a lossless `.webp` from RGBA bytes

The lossless encoder is a byte-cost **super-chooser**: it builds the §3
no-transform / subtract-green baseline plus every §4 single-transform and
§3.5 stacked-transform candidate — sweeping `size_bits`, the §5.2.3 color
cache, and the §6.2.2 meta-prefix grouping — and emits the byte-shortest
stream, so adding a candidate can never enlarge the output. Round 305
brought the three §3.5 stacked chains (color + predictor, color +
subtract-green + predictor, color-indexing + predictor) up to the same
per-block §4.1 mode-selection cost models the single-transform predictor
path has carried since rounds 159–162: each chain now sweeps the
folded-L1 magnitude proxy, the round-161 Shannon-entropy bit cost, and the
round-162 sub-image-aware entropy cost over the *transform-decorrelated*
residual the predictor actually models, keeping the smallest. Round 306
widened the sub-image-aware setting on the stacked chains from the single
mid-range weight round 305 bootstrapped to the **full lambda sweep** the
single-transform path uses — `4 000` / `16 000` / `64 000` / `256 000`
milli-per-bit — so each chain lands on the residual-vs-sub-image cost
crossover its own decorrelated residual exhibits rather than one fixed
guess. On smooth,
mildly-noisy photo-like content the entropy-aware models shrink the color +
predictor chain ~12–21 % versus the L1 proxy (the per-block mode histogram
concentrates, compacting both the §7.2 predictor sub-image and the residual
stream). Round-trip output is byte-identical regardless of which cost model
is chosen — the cost model only changes which §4.1 mode is *recorded*, and
the decoder reads the same modes back.

The shortest path — flat RGBA in, complete `.webp` file out:

```rust
use oxideav_webp::encode_webp_lossless;

let rgba: Vec<u8> = /* width*height*4 RGBA bytes */;
let webp_bytes: Vec<u8> = encode_webp_lossless(&rgba, width, height)?;
// Write to disk:
std::fs::write("out.webp", &webp_bytes)?;
```

### Encode lossless with metadata (ICC / EXIF / XMP)

```rust
use oxideav_webp::{encode_vp8l_argb_with_metadata, WebpMetadata};

// VP8L works in ARGB, one u32/pixel.
let argb: Vec<u32> = /* width*height ARGB pixels */;

let meta = WebpMetadata {
    icc:  Some(&my_icc_profile),
    exif: Some(&my_exif_blob),
    xmp:  Some(&my_xmp_xml),
};
let webp_bytes = encode_vp8l_argb_with_metadata(
    width, height, &argb, /* has_alpha = */ true, &meta,
)?;
```

If `has_alpha` is `true` or any metadata field is set, the output
auto-promotes to the extended `VP8X` layout; otherwise it's the
simple lossless layout.

### Bare VP8L bitstream (no RIFF wrap)

For consumers that wrap the bitstream themselves:

```rust
use oxideav_webp::vp8l::encode_vp8l_argb;
let vp8l: Vec<u8> = encode_vp8l_argb(&argb, width, height)?;
```

### Build an animated `.webp`

```rust
use oxideav_webp::{build_animated_webp, build_animated_webp_with_options,
                   AnimFrame, AnimEncoderOptions};

// Each AnimFrame is a tile (width × height RGBA) at (x, y) on the
// canvas, with a duration in milliseconds.
let frames = vec![
    AnimFrame::new(/* w */ 64, /* h */ 64, /* rgba */ frame0_rgba, /* duration_ms */ 100),
    AnimFrame::new(64, 64, frame1_rgba, 100),
    AnimFrame::new(64, 64, frame2_rgba, 100),
];

// Defaults: per-frame Auto mode (picks byte-smallest of Lossless / Delta).
let webp = build_animated_webp(&frames)?;

// Or with options (loop count, background colour, file-level metadata):
let opts = AnimEncoderOptions {
    loop_count: 0,                      // 0 = infinite
    background_rgba: [0xff, 0xff, 0xff, 0xff],
    ..Default::default()
};
let webp = build_animated_webp_with_options(&frames, &opts)?;
```

## With the OxideAV runtime (`registry` feature on)

```rust
use oxideav_core::RuntimeContext;
use oxideav_webp::{CODEC_ID_VP8, CODEC_ID_VP8L};   // "webp_vp8" / "webp_vp8l"

let mut ctx = RuntimeContext::new();
oxideav_webp::register(&mut ctx);
// ctx now exposes the "webp" container plus "webp_vp8" + "webp_vp8l" codecs.
```

This is the only way to reach the **VP8-lossy encoder** — it delegates
to the `oxideav-vp8` sibling crate's framework factory family:

```rust
use oxideav_webp::encoder_vp8::{make_encoder_with_quality, make_encoder_with_qindex};

// Returns Box<dyn oxideav_core::Encoder>; emits RIFF/WEBP-wrapped output.
let enc = make_encoder_with_quality(&params, 75.0)?;
let enc = make_encoder_with_qindex(&params, 32)?;
```

(Lossless encode + decode + animation + metadata extraction all work
without `registry`; only the VP8 *lossy* encode path needs it.)

## Clean-room sources

Implementation is derived entirely from the public format specs:

* **RFC 9649** — WebP Image Format
  (`docs/image/webp/rfc9649-webp.txt`, also `rfc9649-webp.pdf`).
* **WebP Lossless Bitstream Specification** — the LZ77 + prefix-coded
  literals + color cache + spatial / color / color-indexing transforms
  (also reproduced in RFC 9649 §3).
* **RFC 6386** — VP8 Data Format and Decoding Guide
  (`docs/video/vp8/rfc6386-vp8-bitstream.txt`) for the VP8 lossy
  framing routed through the `oxideav-vp8` sibling.

The 18-fixture corpus at `docs/image/webp/fixtures/` is consumed as
opaque byte streams; end-to-end fixture tests validate against the
ARGB pixels of each fixture's committed `expected.png`. No third-party
codec library source is consulted.

## License

MIT. See [`LICENSE`](./LICENSE).
