//! Published-API `oxideav_webp::demux` module — metadata extraction
//! grouped under its qualified path.
//!
//! Per the published 0.1.2 surface, consumers may reach
//! [`extract_metadata`] either at the crate root or here. Both call
//! sites share the same implementation in [`crate::extract_metadata`].

pub use crate::{extract_metadata, WebpFileMetadata};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;
