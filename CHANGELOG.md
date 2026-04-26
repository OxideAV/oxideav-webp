# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.6](https://github.com/OxideAV/oxideav-webp/compare/v0.0.5...v0.0.6) - 2026-04-26

### Other

- adopt slim VideoFrame shape
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

## [0.0.5](https://github.com/OxideAV/oxideav-webp/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.4](https://github.com/OxideAV/oxideav-webp/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- bump oxideav-vp8 dep to 0.1
- update README + lib doc for colour transform, VP8X, and ALPH
- encode RGBA through VP8 lossy with an ALPH sidecar
- emit VP8X extended header when RGBA frames carry alpha
- add VP8L colour transform for G-R/B decorrelation
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
- add subtract-green + predictor transforms and 256-entry color cache
