//! Published-API `oxideav_webp::encoder_anim` module — animation
//! encoder surface grouped under its qualified path.
//!
//! Per `API-COMPAT-0.1.2.md` consumers may reach
//! [`build_animated_webp`] / [`build_animated_webp_with_options`] /
//! [`AnimFrame`] / [`AnimFrameMode`] / [`AnimEncoderOptions`] either at
//! the crate root or via this module. Both paths share the same
//! implementation in [`crate::anim_encode`].

pub use crate::anim_encode::{
    build_animated_webp, build_animated_webp_with_options, AnimEncoderOptions, AnimFrame,
    AnimFrameMode, DeltaConfig, DownsampleKernel,
};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;
