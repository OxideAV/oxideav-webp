//! Published-API `oxideav_webp::encoder` module — RIFF-wrapped output
//! factories.
//!
//! Per `API-COMPAT-0.1.2.md` this module exposes a single
//! [`make_encoder`] factory that returns a `Box<dyn Encoder>` registered
//! under the umbrella `"webp"` codec id. The factory is the framework
//! entry point; the registry side wires it under [`crate::registry::register_codecs`].
//!
//! The factory is gated behind the default-on `registry` Cargo feature
//! because it depends on the `oxideav_core::Encoder` trait.

#[cfg(feature = "registry")]
pub use crate::registry::{make_encoder, make_encoder_with_metadata, WebpVp8lEncoder};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;
