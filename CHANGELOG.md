# Changelog

All notable changes to `oxideav-webp` are recorded here.

## [Unreleased]

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

### Notes

Pixel decode (VP8 / VP8L / ALPH / VP8X field parsing) is not yet
implemented; `decode_webp` still returns `Error::NotImplemented`.
Subsequent rounds will decode each layer against the
RFC-9649-referenced specifications and the fixture corpus.

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Per the workspace's Implementer-Round
  procedure, such audit failures are unrecoverable via incremental
  cleanup and require an orphan rebuild.

  No `old` branch is retained; long-standing audit failures forfeit
  the archive per workspace policy.
