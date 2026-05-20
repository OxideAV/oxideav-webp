# oxideav-webp

Pure-Rust WebP image codec (RIFF + VP8 + VP8L + VP8X + ALPH).

## Status — 2026-05-20

**Orphan-rebuild scaffold.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core modules could not be defended against the "no external library
source as reference" rule that governs every crate in this workspace.

Per workspace policy, the only acceptable response is a full
clean-room re-implementation against the WebP standards documents and
black-box validator binaries. That work has not yet been scheduled.

Every public entry point currently returns `Error::NotImplemented`.

## Planned clean-room sources

The clean-room rebuild will consult only:

* RFC 9649 — WebP Image Format.
* RFC 6386 — VP8 Data Format and Decoding Guide (the lossy path).
* Black-box invocations of `cwebp` / `dwebp` (the binaries — not their
  source) as opaque validators.

No external library source — libwebp, libvpx, etc. — is permitted as
a reference under the workspace clean-room policy.

## License

MIT. See `LICENSE`.
