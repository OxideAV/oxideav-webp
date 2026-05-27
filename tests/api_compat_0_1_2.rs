//! Compile-only API compatibility assertions for the published
//! `oxideav-webp 0.1.2` crates.io shape.
//!
//! Every binding in this file is a compile-time assertion: a `let _:
//! <type> = …;` that fails to type-check if the documented symbol moves
//! or changes shape. Run-time assertions are minimal (a handful of
//! sanity checks on the standalone-friendly constants); the bulk of the
//! contract enforcement is the type-bindings themselves.
//!
//! Pins the
//! strict minimum public surface every release on the `0.1` line must
//! keep exposing so a consumer pinned to `oxideav-webp = "0.1"` upgrades
//! transparently.
//!
//! Both feature configurations of this file must compile:
//!
//! * `cargo build -p oxideav-webp --no-default-features` — every
//!   binding under the unconditional crate-root re-exports must
//!   resolve (decoder/encoder factories, animation builder, VP8L bare
//!   bitstream encoder, error / metadata types, codec id constants).
//! * `cargo build -p oxideav-webp` (default registry feature on) —
//!   the cfg-gated registry surface (`WebpDecoder`, `register*`,
//!   `encoder::make_encoder`, `encoder_vp8::make_encoder*`) must also
//!   resolve.

// The whole point of this file is to spell out the exact fn-pointer
// shapes the published rustdoc carried. Replacing them with `type`
// aliases obscures the contract — keep the literal signatures.
#![allow(clippy::type_complexity)]

// ============================================================================
// Crate-root re-exports — unconditional surface (no registry feature needed).
// ============================================================================

#[test]
fn crate_root_decode_webp_signature() {
    // `pub fn decode_webp(buf: &[u8]) -> Result<WebpImage, WebpError>`
    let _: fn(&[u8]) -> Result<oxideav_webp::WebpImage, oxideav_webp::WebpError> =
        oxideav_webp::decode_webp;
}

#[test]
fn crate_root_extract_metadata_signature() {
    // `pub fn extract_metadata(buf: &[u8]) -> Result<WebpFileMetadata, WebpError>`
    let _: fn(&[u8]) -> Result<oxideav_webp::WebpFileMetadata, oxideav_webp::WebpError> =
        oxideav_webp::extract_metadata;
}

#[test]
fn crate_root_encode_vp8l_argb_signature() {
    // `pub fn encode_vp8l_argb(argb: &[u32], w: u32, h: u32) -> Result<Vec<u8>, WebpError>`
    let _: fn(&[u32], u32, u32) -> Result<Vec<u8>, oxideav_webp::WebpError> =
        oxideav_webp::encode_vp8l_argb;
}

#[test]
fn crate_root_codec_id_constants() {
    // Both constants must be `pub const &'static str` with exactly the
    // documented values.
    let _: &str = oxideav_webp::CODEC_ID_VP8;
    let _: &str = oxideav_webp::CODEC_ID_VP8L;
    assert_eq!(oxideav_webp::CODEC_ID_VP8, "webp_vp8");
    assert_eq!(oxideav_webp::CODEC_ID_VP8L, "webp_vp8l");
}

#[test]
fn crate_root_webp_error_constructors() {
    // The `invalid` / `unsupported` constructors take `impl Into<String>`.
    let _: oxideav_webp::WebpError = oxideav_webp::WebpError::invalid("msg");
    let _: oxideav_webp::WebpError = oxideav_webp::WebpError::unsupported(String::from("msg"));
    // Unit variants the published 0.1.2 surface enumerated.
    let _: oxideav_webp::WebpError = oxideav_webp::WebpError::Eof;
    let _: oxideav_webp::WebpError = oxideav_webp::WebpError::NeedMore;
}

#[test]
fn crate_root_webp_image_fields() {
    // Field-shape assertions: every documented field must exist with the
    // documented type. Construction-by-value also enforces this.
    let img = oxideav_webp::WebpImage {
        width: 0,
        height: 0,
        frames: Vec::new(),
        metadata: oxideav_webp::WebpFileMetadata::default(),
        anim_background_rgba: None,
        anim_loop_count: None,
    };
    let _: u32 = img.width;
    let _: u32 = img.height;
    let _: Vec<oxideav_webp::WebpFrame> = img.frames;
    let _: oxideav_webp::WebpFileMetadata = img.metadata;
}

#[test]
fn crate_root_webp_frame_fields() {
    let frame = oxideav_webp::WebpFrame {
        rgba: Vec::new(),
        width: 0,
        height: 0,
        duration_ms: 0,
    };
    let _: u32 = frame.width;
    let _: u32 = frame.height;
    let _: u32 = frame.duration_ms;
    let _: Vec<u8> = frame.rgba;
}

#[test]
fn crate_root_webp_file_metadata_default_and_fields() {
    // `pub struct WebpFileMetadata { icc, exif, xmp: Option<Vec<u8>> }`
    // with `Default` derived.
    let m: oxideav_webp::WebpFileMetadata = oxideav_webp::WebpFileMetadata::default();
    let _: Option<Vec<u8>> = m.icc;
    let _: Option<Vec<u8>> = m.exif;
    let _: Option<Vec<u8>> = m.xmp;
}

#[test]
fn crate_root_animation_re_exports() {
    // `build_animated_webp` / `_with_options` / `AnimFrame` / `AnimFrameMode`
    // / `AnimEncoderOptions` are reachable at the crate root and at
    // `oxideav_webp::encoder_anim::*` (qualified-path form).
    let _: fn(&[oxideav_webp::AnimFrame]) -> Result<Vec<u8>, oxideav_webp::WebpError> =
        oxideav_webp::build_animated_webp;
    // Default-constructible options.
    let _opts: oxideav_webp::AnimEncoderOptions<'_> = oxideav_webp::AnimEncoderOptions::default();
    // `AnimFrameMode` variants the 0.1.2 surface enumerated.
    let _: oxideav_webp::AnimFrameMode = oxideav_webp::AnimFrameMode::Auto;
    let _: oxideav_webp::AnimFrameMode = oxideav_webp::AnimFrameMode::Lossless;
}

// ============================================================================
// `oxideav_webp::error` qualified-path forms.
// ============================================================================

#[test]
fn error_module_result_alias() {
    // `pub type Result<T> = core::result::Result<T, WebpError>`.
    let _: oxideav_webp::error::Result<u32> = Ok(42);
    let _: oxideav_webp::error::Result<()> = Err(oxideav_webp::WebpError::InvalidData);
    // The error type re-exported.
    let _: oxideav_webp::error::WebpError = oxideav_webp::WebpError::Eof;
}

// ============================================================================
// `oxideav_webp::decoder` qualified-path forms.
// ============================================================================

#[test]
fn decoder_module_re_exports() {
    let _: fn(&[u8]) -> Result<oxideav_webp::decoder::WebpImage, oxideav_webp::WebpError> =
        oxideav_webp::decoder::decode_webp;
    // `WebpFrame` / `WebpImage` re-exported from the module.
    let _: oxideav_webp::decoder::WebpImage = oxideav_webp::WebpImage {
        width: 0,
        height: 0,
        frames: Vec::new(),
        metadata: Default::default(),
        anim_background_rgba: None,
        anim_loop_count: None,
    };
    let _: oxideav_webp::decoder::WebpFrame = oxideav_webp::WebpFrame {
        rgba: Vec::new(),
        width: 0,
        height: 0,
        duration_ms: 0,
    };
}

// ============================================================================
// `oxideav_webp::demux` qualified-path forms.
// ============================================================================

#[test]
fn demux_module_re_exports() {
    let _: fn(&[u8]) -> Result<oxideav_webp::demux::WebpFileMetadata, oxideav_webp::WebpError> =
        oxideav_webp::demux::extract_metadata;
}

// ============================================================================
// `oxideav_webp::encoder_anim` qualified-path forms.
// ============================================================================

#[test]
fn encoder_anim_module_re_exports() {
    // `build_animated_webp` / `_with_options` accessible under the module.
    let _: fn(
        &[oxideav_webp::encoder_anim::AnimFrame],
    ) -> Result<Vec<u8>, oxideav_webp::WebpError> = oxideav_webp::encoder_anim::build_animated_webp;
    let _: oxideav_webp::encoder_anim::AnimEncoderOptions<'_> =
        oxideav_webp::encoder_anim::AnimEncoderOptions::default();
}

// ============================================================================
// `oxideav_webp::vp8l` — bare-bitstream surface (standalone-friendly).
// ============================================================================

#[test]
fn vp8l_module_signature_byte() {
    let _: u8 = oxideav_webp::vp8l::VP8L_SIGNATURE;
    assert_eq!(oxideav_webp::vp8l::VP8L_SIGNATURE, 0x2F);
}

#[test]
fn vp8l_module_encode_decode_signatures() {
    let _: fn(&[u32], u32, u32) -> Result<Vec<u8>, oxideav_webp::WebpError> =
        oxideav_webp::vp8l::encode_vp8l_argb;
    let _: fn(&[u8]) -> Result<oxideav_webp::vp8l::Vp8lImage, oxideav_webp::WebpError> =
        oxideav_webp::vp8l::decode;
}

#[test]
fn vp8l_image_fields_and_to_rgba() {
    let img = oxideav_webp::vp8l::Vp8lImage {
        width: 1,
        height: 1,
        pixels: vec![0xff_aa_bb_ccu32],
        has_alpha: false,
    };
    let _: u32 = img.width;
    let _: u32 = img.height;
    let _: &[u32] = &img.pixels;
    let _: bool = img.has_alpha;
    let rgba: Vec<u8> = img.to_rgba();
    assert_eq!(rgba, vec![0xaa, 0xbb, 0xcc, 0xff]);
}

#[test]
fn vp8l_huffman_group_handle_exists() {
    let _: oxideav_webp::vp8l::HuffmanGroup = oxideav_webp::vp8l::HuffmanGroup::new();
}

#[test]
fn vp8l_submodule_re_exports() {
    // Each sub-module must exist as a documented path. Bind one identifier
    // from each to anchor the path resolution at compile time.
    let _: oxideav_webp::vp8l::bit_reader::BitReader<'_> =
        oxideav_webp::vp8l::bit_reader::BitReader::new(&[][..]);
    let _: oxideav_webp::vp8l::huffman::HuffmanGroup =
        oxideav_webp::vp8l::huffman::HuffmanGroup::new();
    // `transform` module path — bind to a function pointer to confirm the
    // identifier resolves.
    let _: fn(&mut [u32]) = oxideav_webp::vp8l::transform::inverse_subtract_green;
    let _: fn(&[u32], u32, u32) -> Result<Vec<u8>, oxideav_webp::WebpError> =
        oxideav_webp::vp8l::encoder::encode_vp8l_argb;
}

// ============================================================================
// `oxideav_webp::encoder_vp8` — VP8 lossy quality knobs (standalone-friendly).
// ============================================================================

#[test]
fn encoder_vp8_quality_to_qindex_signature() {
    let _: fn(f32) -> u8 = oxideav_webp::encoder_vp8::quality_to_qindex;
}

#[test]
fn encoder_vp8_freq_deltas_field_shape() {
    let d = oxideav_webp::encoder_vp8::Vp8FreqDeltas {
        y_dc_delta: 0,
        y2_dc_delta: 0,
        y2_ac_delta: 0,
        uv_dc_delta: 0,
        uv_ac_delta: 0,
    };
    let _: i8 = d.y_dc_delta;
    let _: i8 = d.y2_dc_delta;
    let _: i8 = d.y2_ac_delta;
    let _: i8 = d.uv_dc_delta;
    let _: i8 = d.uv_ac_delta;
    // Default-constructible.
    let _: oxideav_webp::encoder_vp8::Vp8FreqDeltas =
        oxideav_webp::encoder_vp8::Vp8FreqDeltas::default();
}

// ============================================================================
// `oxideav_webp::riff` — RIFF/WEBP container.
// ============================================================================

#[test]
fn riff_module_parse_signature() {
    let _: fn(
        &[u8],
    )
        -> Result<oxideav_webp::riff::WebpContainer, oxideav_webp::riff::ContainerError> =
        oxideav_webp::riff::parse;
}

#[test]
fn riff_module_build_signature() {
    let _: fn(
        &[u8],
        oxideav_webp::riff::ImageKind,
        u32,
        u32,
    ) -> Result<Vec<u8>, oxideav_webp::riff::BuildError> = oxideav_webp::riff::build_webp_file;
}

// ============================================================================
// Registry-gated surface — only checked when the default `registry` feature
// is on (the contract MAY hide these in the standalone build).
// ============================================================================

#[cfg(feature = "registry")]
#[test]
fn registry_register_signatures() {
    use oxideav_core::RuntimeContext;
    let _: fn(&mut RuntimeContext) = oxideav_webp::register;
    let _: fn(&mut RuntimeContext) = oxideav_webp::register_codecs;
    let _: fn(&mut RuntimeContext) = oxideav_webp::register_containers;
}

#[cfg(feature = "registry")]
#[test]
fn registry_webp_decoder_constructible() {
    use oxideav_core::{CodecId, CodecParameters};
    let params = CodecParameters::video(CodecId::new("webp"));
    let _dec: oxideav_webp::WebpDecoder = oxideav_webp::WebpDecoder::new(params);
}

// Type aliases to keep the registry-side factory bindings under
// clippy's `type_complexity` threshold while preserving an exact
// signature check.
#[cfg(feature = "registry")]
type EncFactory = fn(
    &oxideav_core::CodecParameters,
) -> Result<Box<dyn oxideav_core::Encoder>, oxideav_core::Error>;
#[cfg(feature = "registry")]
type EncFactoryF32 = fn(
    &oxideav_core::CodecParameters,
    f32,
) -> Result<Box<dyn oxideav_core::Encoder>, oxideav_core::Error>;
#[cfg(feature = "registry")]
type EncFactoryU8 = fn(
    &oxideav_core::CodecParameters,
    u8,
) -> Result<Box<dyn oxideav_core::Encoder>, oxideav_core::Error>;
#[cfg(feature = "registry")]
type EncFactoryU8Deltas = fn(
    &oxideav_core::CodecParameters,
    u8,
    oxideav_webp::encoder_vp8::Vp8FreqDeltas,
) -> Result<Box<dyn oxideav_core::Encoder>, oxideav_core::Error>;
#[cfg(feature = "registry")]
type EncFactoryF32Deltas = fn(
    &oxideav_core::CodecParameters,
    f32,
    oxideav_webp::encoder_vp8::Vp8FreqDeltas,
) -> Result<Box<dyn oxideav_core::Encoder>, oxideav_core::Error>;

#[cfg(feature = "registry")]
#[test]
fn registry_encoder_module_make_encoder_signature() {
    let _: EncFactory = oxideav_webp::encoder::make_encoder;
}

#[cfg(feature = "registry")]
#[test]
fn registry_encoder_vp8_factory_signatures() {
    let _: EncFactory = oxideav_webp::encoder_vp8::make_encoder;
    let _: EncFactoryF32 = oxideav_webp::encoder_vp8::make_encoder_with_quality;
    let _: EncFactoryU8 = oxideav_webp::encoder_vp8::make_encoder_with_qindex;
    let _: EncFactoryU8Deltas = oxideav_webp::encoder_vp8::make_encoder_with_qindex_and_freq_deltas;
    let _: EncFactoryF32Deltas =
        oxideav_webp::encoder_vp8::make_encoder_with_quality_and_freq_deltas;
}

#[cfg(feature = "registry")]
#[test]
fn registry_decoder_make_vp8l_decoder_signature() {
    let _: fn(u32, u32) -> oxideav_webp::WebpDecoder = oxideav_webp::decoder::make_vp8l_decoder;
}

// ============================================================================
// `From<oxideav_vp8::Vp8Error> for WebpError` — round-168 wiring against
// `oxideav-vp8 0.2.1` (the release that first exports `Vp8Error` at the
// crate root). The four variants share names with `WebpError` so the
// mapping is a straight 1-to-1 collapse; per-variant assertions below
// pin the contract.
// ============================================================================

#[test]
fn crate_root_webp_error_from_vp8_error_signature() {
    // Compile-time signature assertion — the adapter is a function from
    // `oxideav_vp8::Vp8Error` to `oxideav_webp::WebpError`.
    let _: fn(oxideav_vp8::Vp8Error) -> oxideav_webp::WebpError = oxideav_webp::WebpError::from;
}

#[test]
fn crate_root_webp_error_from_vp8_error_variant_mapping() {
    use oxideav_vp8::Vp8Error;
    use oxideav_webp::WebpError;
    assert_eq!(
        WebpError::from(Vp8Error::InvalidData("truncated".into())),
        WebpError::InvalidData
    );
    assert_eq!(
        WebpError::from(Vp8Error::Unsupported("interframe".into())),
        WebpError::Unsupported
    );
    assert_eq!(WebpError::from(Vp8Error::Eof), WebpError::Eof);
    assert_eq!(WebpError::from(Vp8Error::NeedMore), WebpError::NeedMore);
}
