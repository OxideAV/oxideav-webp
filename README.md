# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH).

## Status — 2026-05-20 (clean-room round 1)

* **Container walker:** RFC 9649 §2.3–§2.7 RIFF/WEBP chunk walk.
  Surfaces the chunk list (`VP8 ` / `VP8L` / `VP8X` / `ALPH` /
  `ANIM` / `ANMF` / `ICCP` / `EXIF` / `XMP ` plus unknowns) with
  per-chunk FourCC, size, and absolute payload range. Validates the
  §2.4 file header, the §2.4 `File Size` against buffer length, every
  chunk's `Size` against the remaining RIFF payload, and the §2.3
  odd-size pad byte. Order-on-disk is preserved so §2.7 ordering
  rules can be applied by callers.
* **Pixel decode (VP8 / VP8L / ALPH):** not implemented yet —
  [`decode_webp`](src/lib.rs) returns [`Error::NotImplemented`].
* **Registry hook:** [`register`](src/lib.rs) is a no-op; round 1
  ships no decoder/encoder to the runtime context.

## What round 1 lands

| Item                         | Status                                              |
| ---------------------------- | --------------------------------------------------- |
| §2.4 WebP file header check  | done (`RIFF` + `File Size` + `WEBP`)                |
| §2.3 chunk walker            | done (FourCC, Size, payload range, odd-size pad)    |
| §2.5 simple lossy            | structural pass — `VP8 ` chunk surfaced             |
| §2.6 simple lossless         | structural pass — `VP8L` chunk surfaced             |
| §2.7 extended (VP8X et al.)  | structural pass — every documented FourCC surfaced  |
| §2.7.1.1 `ANIM` / `ANMF`     | surfaced as opaque chunks                           |
| §2.7.1.2 `ALPH`              | surfaced as opaque chunk                            |
| §2.7.1.4 `ICCP`              | surfaced as opaque chunk                            |
| §2.7.1.5 `EXIF` / `XMP `     | surfaced as opaque chunks                           |
| §2.7.1.6 unknown chunks      | surfaced (no special handling required by §2.7.1.6) |
| VP8 / VP8L / VP8X **decode** | not yet — `Error::NotImplemented`                   |

Test count: **11** (8 unit + 3 integration against the
`docs/image/webp/fixtures/` corpus).

## Clean-room sources

Round 1 was implemented entirely against:

* **RFC 9649** — WebP Image Format (`docs/image/webp/rfc9649-webp.txt`,
  also available as `rfc9649-webp.pdf`).
* The 18-fixture corpus at `docs/image/webp/fixtures/` — consumed
  as opaque byte streams to validate the walker's structural pass
  on real WebP files (no `expected.png` comparison yet).

No external library source — libwebp, libvpx, image-rs, webp-rs,
etc. — was consulted. `cwebp` / `dwebp` would be permissible as
black-box validators; round 1 did not invoke them.

## License

MIT. See `LICENSE`.
