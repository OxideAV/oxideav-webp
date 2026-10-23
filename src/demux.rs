//! Pre-contract `oxideav_webp::demux` module path — metadata extraction
//! under its historical qualified name.
//!
//! New code uses [`crate::info`] (presence flags) or
//! [`crate::decode`]`(..).metadata` (payloads).

pub use crate::{ImageInfo, Metadata, info, read_metadata};

#[allow(deprecated)]
pub use crate::{WebpFileMetadata, extract_metadata};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;
