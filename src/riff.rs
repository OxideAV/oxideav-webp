//! Published-API `oxideav_webp::riff` module — re-export shim over the
//! in-crate RIFF/WEBP container walker / builder.
//!
//! Per the published 0.1.2 surface, the crate must expose a `riff` public
//! module path. The implementation lives in [`crate::container`] (the
//! walker) and [`crate::build`] (the builder); this module re-exports
//! both surfaces under the published shorter name so consumers pinned to
//! the 0.1.2 shape can `use oxideav_webp::riff::*` to reach the same
//! types.

pub use crate::build::{
    build_chunk, build_vp8x_chunk, build_webp_file, BuildError, ImageKind, Vp8xFlags,
};
pub use crate::container::{fourcc, parse, ContainerError, FourCc, WebpChunk, WebpContainer};
