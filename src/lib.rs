//! Pure-Rust WebP image decoder.
//!
//! WebP is Google's image format. This crate handles every major flavour:
//!
//! * **Simple file format, lossy** — `RIFF WEBP` + `VP8 ` chunk holding a
//!   single VP8 keyframe. The bitstream is fed into `oxideav-vp8`; the
//!   resulting YUV 4:2:0 frame is converted to RGBA for output (consumers of
//!   still images universally expect RGB/RGBA).
//! * **Simple file format, lossless** — `RIFF WEBP` + `VP8L` chunk. The
//!   lossless bitstream (Huffman + LZ77 + color-cache + four transforms) is
//!   decoded from scratch in [`vp8l`] per the WebP Lossless specification.
//!   Output is native RGBA.
//! * **Extended file format** — `RIFF WEBP` + `VP8X` header + optional
//!   `ICCP` / `EXIF` / `XMP ` / `ANIM` / `ANMF` / `ALPH`. We decode the VP8X
//!   flags, stitch `ALPH` onto a `VP8 ` luma path (filtered raw or
//!   VP8L-compressed), and iterate `ANMF` chunks for animation. Unknown
//!   auxiliary chunks are skipped gracefully.
//! * **Animated WebP** — each `ANMF` sub-chunk emits one `VideoFrame` with a
//!   matching PTS/duration expressed in milliseconds. Frame disposal and
//!   blending modes are honoured against an internal RGBA canvas.
//!
//! VP8L lossless encoding is supported through [`encoder::make_encoder`]
//! — length-limited Huffman + 4 KB-window LZ77 + subtract-green,
//! colour (G↔R/B), and tile-based predictor transforms + a 256-entry
//! colour cache. The encoder output is a full RIFF `.webp` file (with
//! the extended `VP8X` header whenever the frame carries alpha or
//! metadata).
//!
//! The VP8 lossy path also emits a full RIFF file. `Yuv420P` and
//! `Rgb24` frames use the simple `VP8 ` layout (Rgb24 streams the
//! RGB→YUV conversion without a Rgba alloc — issue #7); `Rgba` and
//! `Yuva420P` frames split the alpha plane into an `ALPH` sidecar
//! chunk and emit the extended `VP8X` layout. `Yuva420P` skips the
//! YUV→RGB→YUV roundtrip the `Rgba` path goes through. The palette
//! (colour-indexing) transform and meta-Huffman grouping are still
//! scoped non-goals on the lossless side.

// Many helpers live in registry-gated trait-impl land — when built
// without `registry`, the unconditional API doesn't exercise them and
// the dead-code warning would fire on every wrapper. Suppress
// crate-wide rather than gating each individually.
#![cfg_attr(not(feature = "registry"), allow(dead_code))]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_div_ceil)]

pub mod decoder;
pub mod demux;
pub mod encoder;
pub mod encoder_anim;
pub mod encoder_vp8;
pub mod error;
#[cfg(feature = "registry")]
pub mod registry;
pub mod riff;
pub mod vp8l;

/// Codec id string for the VP8L lossless still-image bitstream. Registered
/// so the codec registry reports it alongside other image codecs.
pub const CODEC_ID_VP8L: &str = "webp_vp8l";

/// Codec id string for the VP8 lossy WebP still-image path. The encoder
/// registered under this id takes a YUV420P frame and emits a full
/// RIFF/WEBP `.webp` file wrapping a single VP8 keyframe. Paired with
/// (and semantically aligned to) the decoder's existing handling of the
/// `VP8 ` chunk inside a WebP container.
pub const CODEC_ID_VP8: &str = "webp_vp8";

// Public unconditional API — works whether or not `registry` is enabled.
pub use decoder::{decode_webp, WebpFrame, WebpImage};
pub use demux::{extract_metadata, WebpFileMetadata};
pub use encoder_anim::{build_animated_webp, AnimFrame};
pub use error::{Result, WebpError};
pub use vp8l::{encode_vp8l_argb, encode_vp8l_argb_with, EncoderOptions};

// Public registry-gated API — keeps the framework integration surface
// (Decoder/Encoder/Demuxer trait impls, `register*` helpers,
// `WebpDecoder` streaming type) behind the default-on `registry`
// feature so image-library callers can build the crate without
// dragging in `oxideav-core`.
#[cfg(feature = "registry")]
pub use decoder::WebpDecoder;
#[cfg(feature = "registry")]
pub use registry::{register, register_codecs, register_containers};
