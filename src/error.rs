//! Published-API `oxideav_webp::error` module — `Result` type alias plus
//! the [`WebpError`] re-export.
//!
//! Per the published 0.1.2 surface, the crate-root public surface exposes both
//! `oxideav_webp::Result` (the type alias) and `oxideav_webp::WebpError`
//! (the published error enum), and groups them under
//! `oxideav_webp::error::{Result, WebpError}` for consumers that prefer
//! the qualified path. Both forms point at the **same** [`WebpError`]
//! defined in the crate root, so a value flows through either path with
//! no conversion.

pub use crate::WebpError;

/// Result alias for the published-API decode / encode entry points.
///
/// Equivalent to `core::result::Result<T, oxideav_webp::WebpError>`. This
/// is the type the rustdoc for `oxideav-webp 0.1.2`'s `decode_webp` /
/// `extract_metadata` / `encode_vp8l_argb*` / animation-encode entry
/// points carries.
pub type Result<T> = core::result::Result<T, WebpError>;
