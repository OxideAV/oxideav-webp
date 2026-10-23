//! Pre-contract `oxideav_webp::encoder` module path — the framework
//! encoder factory under its historical qualified name.
//!
//! New code uses the crate root ([`crate::encode`], [`crate::encode_rgba8`],
//! [`crate::EncodeOptions`]).

pub use crate::{EncodeOptions, encode, encode_rgb8, encode_rgba8, encode_to};

#[cfg(feature = "registry")]
pub use crate::registry::make_encoder;
#[cfg(feature = "registry")]
#[doc(hidden)]
pub use crate::registry::{WebpVp8lEncoder, make_encoder_with_metadata};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;
