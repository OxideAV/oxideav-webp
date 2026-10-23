//! Pre-contract `oxideav_webp::encoder_anim` module path — the animation
//! encoder surface under its historical qualified name.
//!
//! New code uses [`crate::encode_animation`] with [`crate::EncodeOptions`].

pub use crate::{
    AnimFrame, AnimFrameMode, DeltaConfig, DownsampleKernel, EncodeOptions, Frame,
    encode_animation, encode_animation_frames,
};

#[allow(deprecated)]
pub use crate::anim_encode::{
    AnimEncoderOptions, build_animated_webp, build_animated_webp_with_options,
};

/// Result alias for this module's entry points — `Result<T, WebpError>`.
pub use crate::error::Result;
