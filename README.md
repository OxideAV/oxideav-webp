# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH).

## Status — 2026-05-21 (clean-room round 2)

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
  has_exif, has_xmp, has_animation, has_unknown }`. The flag-byte
  bit mapping is anchored against six libwebp-encoded fixtures whose
  `trace.txt` files independently report the same decoded flags +
  canvas dimensions. The §2.7.1 product cap (`width * height ≤
  2^32 - 1`) is enforced; reserved bits are surfaced via `has_unknown`
  but do **not** trigger refusal (§2.7.1 says readers MUST ignore them).
* **Pixel decode (VP8 / VP8L / ALPH):** not implemented yet —
  [`decode_webp`](src/lib.rs) returns [`Error::NotImplemented`].
* **Registry hook:** [`register`](src/lib.rs) is a no-op; round 2
  ships no decoder/encoder to the runtime context.

## What round 2 lands

| Item                          | Status                                              |
| ----------------------------- | --------------------------------------------------- |
| §2.4 WebP file header check   | done (`RIFF` + `File Size` + `WEBP`)                |
| §2.3 chunk walker             | done (FourCC, Size, payload range, odd-size pad)    |
| §2.5 simple lossy             | structural pass — `VP8 ` chunk surfaced             |
| §2.6 simple lossless          | structural pass — `VP8L` chunk surfaced             |
| §2.7 extended (VP8X et al.)   | structural pass — every documented FourCC surfaced  |
| §2.7.1 VP8X flag-byte parse   | done (`I`/`L`/`E`/`X`/`A` + `Rsv`/`R` ignored)      |
| §2.7.1 canvas dim decode      | done (24-bit LE × 2, 1-based)                       |
| §2.7.1 canvas product cap     | done (rejects `w*h > 2^32 - 1`)                     |
| §2.7.1.1 `ANIM` / `ANMF`      | surfaced as opaque chunks                           |
| §2.7.1.2 `ALPH`               | surfaced as opaque chunk                            |
| §2.7.1.4 `ICCP`               | surfaced as opaque chunk                            |
| §2.7.1.5 `EXIF` / `XMP `      | surfaced as opaque chunks                           |
| §2.7.1.6 unknown chunks       | surfaced (no special handling required by §2.7.1.6) |
| VP8 / VP8L / ALPH **decode**  | not yet — `Error::NotImplemented`                   |

Test count: **27** (23 unit + 4 integration against the
`docs/image/webp/fixtures/` corpus).

## Clean-room sources

Rounds 1 + 2 were implemented entirely against:

* **RFC 9649** — WebP Image Format (`docs/image/webp/rfc9649-webp.txt`,
  also available as `rfc9649-webp.pdf`). Round 2 cites §2.7.1
  (extended file header) Figure 7 + Reserved/Canvas field
  definitions.
* The 18-fixture corpus at `docs/image/webp/fixtures/` — consumed
  as opaque byte streams. Round 2 additionally cross-checks the
  VP8X bit-position decode against the fixtures' `trace.txt`
  files (libwebp's instrumented decoder output, treated as a
  black-box golden record).

No external library source — libwebp, libvpx, image-rs, webp-rs,
etc. — was consulted. `cwebp` / `dwebp` would be permissible as
black-box validators; rounds 1 + 2 did not invoke them directly
(round 2 reads only the `trace.txt` outputs already committed to
`docs/`).

## License

MIT. See `LICENSE`.
