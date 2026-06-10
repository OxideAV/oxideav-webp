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

The crate ships sixteen criterion benches under `benches/`: the
original four end-to-end / hot-loop targets (`lossless_decode`,
`lossless_encode`, `lz77_match`, `argb_to_rgba`); the four §4.x
decoder-side inverse-transform per-pass benches added across rounds
194 / 207 / 210 / 217 (`inverse_predictor`, `inverse_color`,
`inverse_color_indexing`, `inverse_subtract_green`); the round-249
§4.4 palette subtraction-decode bench (`inverse_color_table`); the
two encoder-side §4.1 / §4.3 forward-transform per-pass benches
added in rounds 224 / 248 (`predictor_subtract`,
`apply_subtract_green`); the two encoder-side §3.7.2 per-pass
benches added in rounds 250 / 251 (`build_code_lengths`,
`canonical_codes`) covering each §3.7.1 prefix-code-group alphabet
(distance-40, literal-256, green-281, green-2328) in both dense and
sparse frequency regimes; the round-252 decoder-side §3.6.2.2 LZ77
prefix-code-to-value per-call bench (`read_lz77_value`) covering
the fast-path / short-extra / long-extra / max-extra regimes from
§3.6.2.2 Table 4; the round-253 decoder-side §3.6.2.3 color-
cache hash slot-index per-call bench (`color_cache_hash`) covering
the §3.6.2.3 `code_bits` allowed range `[1..11]` at four
representative points (1 / 4 / 8 / 11); and the round-254 encoder-
side §5.2.2 LZ77 value-to-prefix-split per-call bench
(`value_to_prefix`, mirror of the round-252 decoder-side cell
layout) covering the same fast-path / short-extra / long-extra /
max-extra regimes at value-side samples (3 / 40 / 40_000 / 900_000).
Numbers, profile findings, and the
optimization log live in [`BENCHMARKS.md`](./BENCHMARKS.md). Run
with:

```text
CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
  cargo bench --manifest-path crates/oxideav-webp/Cargo.toml \
    --bench <name> -- --quick
```

### Fuzzing

Seventeen [`cargo-fuzz`](https://rust-fuzz.github.io/book/cargo-fuzz.html)
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
replaying a successful run from the same pre-fill. Run any
one with (nightly + `cargo-fuzz` installed):

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
