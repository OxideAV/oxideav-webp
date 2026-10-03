//! Pre-contract `oxideav_webp::demux` module path — metadata extraction
//! under its historical qualified name.
//!
//! New code uses [`crate::info`] (presence flags) or
//! [`crate::decode`]`(..).metadata` (payloads).

pub use crate::{info, read_metadata, ImageInfo, Metadata};

#[allow(deprecated)]
pub use crate::{extract_metadata, WebpFileMetadata};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;
