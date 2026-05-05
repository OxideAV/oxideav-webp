//! `oxideav-core` integration layer for `oxideav-webp`.
//!
//! Gated behind the default-on `registry` feature so image-library
//! consumers can depend on `oxideav-webp` with `default-features = false`
//! and skip the `oxideav-core` dependency entirely.
//!
//! The module exposes:
//! * [`register`] / [`register_codecs`] / [`register_containers`] — the
//!   `CodecRegistry` / `ContainerRegistry` entry points the umbrella
//!   `oxideav` crate calls during framework initialisation.
//! * The `From<WebpError> for oxideav_core::Error` conversion that lets
//!   the trait-side `Decoder` / `Encoder` / `Demuxer` impls (still
//!   living in `decoder.rs` / `encoder.rs` / `encoder_vp8.rs` /
//!   `demux.rs`) bubble bitstream errors up through the framework
//!   error type.

use oxideav_core::ContainerRegistry;
use oxideav_core::RuntimeContext;
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, Encoder};

use crate::error::WebpError;
use crate::{CODEC_ID_VP8, CODEC_ID_VP8L};

/// Convert a [`WebpError`] into the framework-shared
/// `oxideav_core::Error` so trait impls in this crate can use `?` on
/// errors returned by the framework-free decode/encode functions.
impl From<WebpError> for oxideav_core::Error {
    fn from(e: WebpError) -> Self {
        match e {
            WebpError::InvalidData(s) => oxideav_core::Error::InvalidData(s),
            WebpError::Unsupported(s) => oxideav_core::Error::Unsupported(s),
            WebpError::Eof => oxideav_core::Error::Eof,
            WebpError::NeedMore => oxideav_core::Error::NeedMore,
        }
    }
}

/// Register every codec implementation this crate provides.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("webp_vp8l_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(16384, 16384);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_VP8L))
            .capabilities(caps)
            .decoder(make_vp8l_decoder)
            .encoder(make_vp8l_encoder),
    );

    // VP8 lossy — encoder only for now. The decode side of a `.webp`
    // file goes through the WebP container demuxer, which already
    // dispatches VP8 chunks into `oxideav-vp8`.
    let vp8_caps = CodecCapabilities::video("webp_vp8_sw_enc")
        .with_intra_only(true)
        .with_lossy(true)
        .with_max_size(16383, 16383);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_VP8))
            .capabilities(vp8_caps)
            .encoder(make_vp8_encoder),
    );
}

/// Register the WebP container demuxer + the `.webp` extension + its probe.
pub fn register_containers(reg: &mut ContainerRegistry) {
    crate::demux::register(reg);
}

/// Unified registration entry point — installs WebP codecs into the
/// codec sub-registry and the WebP container into the container
/// sub-registry of the supplied [`RuntimeContext`].
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

fn make_vp8l_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    crate::decoder::make_vp8l_decoder(params)
}

fn make_vp8l_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    crate::encoder::make_encoder(params)
}

fn make_vp8_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    crate::encoder_vp8::make_encoder(params)
}

#[cfg(test)]
mod register_tests {
    use super::*;

    #[test]
    fn register_via_runtime_context_installs_both_sides() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id_vp8l = CodecId::new(CODEC_ID_VP8L);
        let id_vp8 = CodecId::new(CODEC_ID_VP8);
        assert!(
            ctx.codecs.has_decoder(&id_vp8l),
            "VP8L decoder factory not installed via RuntimeContext"
        );
        assert!(
            ctx.codecs.has_encoder(&id_vp8),
            "VP8 encoder factory not installed via RuntimeContext"
        );
        assert_eq!(
            ctx.containers.container_for_extension("webp"),
            Some("webp"),
            "WebP container extension not installed via RuntimeContext"
        );
    }
}
