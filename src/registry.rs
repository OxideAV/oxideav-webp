//! `oxideav-core` integration — `Decoder` trait impl, `Frame` / `Error`
//! conversions, and the [`register`] entry point.
//!
//! Gated behind the default-on `registry` Cargo feature so consumers
//! that just want the standalone walker / builder / decoder surface can
//! depend on `oxideav-webp` with `default-features = false` and skip
//! the `oxideav-core` dependency entirely.
//!
//! The module exposes:
//!
//! * [`register`] / [`register_codecs`] / [`register_containers`] — the
//!   `CodecRegistry` / `ContainerRegistry` entry points the umbrella
//!   `oxideav` crate calls during framework initialisation.
//! * [`WebpDecoder`] — the `Decoder` trait impl that wraps the
//!   framework-free [`crate::decode_webp_image`] entry point.
//! * [`decode_webp_to_frame`] — a `VideoFrame`-flavoured wrapper around
//!   [`crate::decode_webp_image`] preserved for callers that prefer the
//!   direct conversion to the framework's frame type.
//! * The `From<Error> for oxideav_core::Error` conversion that lets the
//!   trait impl use `?` on errors returned by the framework-free decode
//!   path.
//!
//! Per round 112, the registered decoder covers:
//!
//! * **§2.6 / §3.4 `VP8L` lossless** (simple or `VP8X`-extended) —
//!   decoded all the way to interleaved 8-bit RGBA, surfaced as a single
//!   planar [`VideoFrame`] with stride `width * 4` and
//!   [`PixelFormat::Rgba`].
//! * **§2.7.1.2 `ALPH`-over-`VP8L` alpha override** — the alpha plane
//!   from an accompanying `ALPH` chunk overrides the per-pixel alpha
//!   from the `VP8L` bitstream itself.
//! * **§2.5 `VP8 ` lossy** — a clean `Error::Unsupported`. WebP's lossy
//!   path is a VP8 bitstream; `oxideav-webp` deliberately does **not**
//!   take a runtime dependency on `oxideav-vp8`. Callers that need lossy
//!   should route the chunk via [`crate::extract_lossy_chunk`] to a
//!   downstream VP8 decoder.
//! * **Animations / header-only files** (no `VP8L`/`VP8 ` image-data
//!   chunk) — a clean `Error::Unsupported`.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag,
    ContainerRegistry, Decoder, Error as CoreError, Frame, MediaType, Packet, PixelFormat,
    RuntimeContext, VideoFrame, VideoPlane,
};

use crate::{decode_webp_image, DecodedWebp, Error, UnsupportedKind};

/// Stable on-wire identifier this crate registers under in the codec
/// registry. The single canonical value the framework uses to look up
/// a WebP decoder is `"webp"`.
pub const CODEC_ID_STR: &str = "webp";

/// Bridge crate-local errors to the framework-wide `oxideav_core::Error`
/// so trait impls can use `?` on the framework-free decode path.
///
/// `Error::Unsupported(LossyVp8)` and `Error::Unsupported(NoImageData)`
/// both map to `oxideav_core::Error::Unsupported(...)`. Every other
/// variant carries diagnostic text already built by the originating
/// sub-module's `Display` impl — it's surfaced verbatim via
/// `Error::InvalidData(...)`.
impl From<Error> for CoreError {
    fn from(e: Error) -> Self {
        match e {
            Error::Unsupported(kind) => CoreError::Unsupported(match kind {
                UnsupportedKind::LossyVp8 => {
                    "oxideav-webp: VP8 lossy bitstream (route to a VP8 decoder)".to_string()
                }
                UnsupportedKind::NoImageData => {
                    "oxideav-webp: no VP8L/VP8 image-data chunk (animation or header-only)"
                        .to_string()
                }
            }),
            Error::NotImplemented => {
                CoreError::Unsupported("oxideav-webp: code path not implemented yet".to_string())
            }
            other => CoreError::InvalidData(other.to_string()),
        }
    }
}

// ───────────────────────── Frame conversion ─────────────────────────

/// Convert a fully-decoded [`DecodedWebp`] into a single-planar
/// [`VideoFrame`] carrying interleaved 8-bit RGBA.
///
/// Stride is exactly `width * 4` bytes (no row padding). Stream-level
/// width / height / pixel format live on [`CodecParameters`], not the
/// frame — consumers read them from [`WebpDecoder::params`] (which is
/// updated to the decoded dimensions after the first frame).
fn decoded_webp_to_video_frame(img: DecodedWebp, pts: Option<i64>) -> VideoFrame {
    let stride = (img.width as usize).saturating_mul(4);
    VideoFrame {
        pts,
        planes: vec![VideoPlane {
            stride,
            data: img.rgba,
        }],
    }
}

/// `VideoFrame`-flavoured wrapper around [`decode_webp_image`].
///
/// Preserved for callers that already build [`VideoFrame`]s directly
/// without going through the [`Decoder`] trait (e.g. container demuxers
/// that want to drop a still WebP picture into a video stream).
pub fn decode_webp_to_frame(bytes: &[u8], pts: Option<i64>) -> oxideav_core::Result<VideoFrame> {
    let img = decode_webp_image(bytes)?;
    Ok(decoded_webp_to_video_frame(img, pts))
}

// ───────────────────────── Decoder + factory ─────────────────────────

/// Factory for the `Decoder` trait impl — installed in the codec
/// registry and called by the framework when a `webp` packet stream
/// needs decoding.
pub fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    Ok(Box::new(WebpDecoder::new(params.clone())))
}

/// WebP [`Decoder`] trait impl.
///
/// Each `send_packet` carries one complete `RIFF/WEBP` file (still
/// image — animations not yet implemented). The matching `receive_frame`
/// returns a [`Frame::Video`] holding interleaved 8-bit RGBA pixels.
///
/// The decoder caches the most recently observed image dimensions on
/// its [`CodecParameters`] — see [`Self::params`] — so consumers can
/// pull width / height / pixel format after the first
/// `receive_frame`. Before a packet has been decoded the values come
/// from whatever `CodecParameters` the factory was constructed with
/// (typically empty width / height for a still-image codec).
#[derive(Debug)]
pub struct WebpDecoder {
    /// Output stream parameters. `pixel_format` is fixed to
    /// [`PixelFormat::Rgba`] at construction (every supported WebP
    /// image kind decodes to RGBA today); `width` / `height` are
    /// refreshed after each successful decode.
    params: CodecParameters,
    /// Most-recently received packet, consumed by the next
    /// `receive_frame` call. The contract matches the framework's
    /// one-packet-in / one-frame-out pattern for image codecs (see e.g.
    /// the PNG / ICER impls).
    pending: Option<Packet>,
    /// `true` after `flush()` — the next `receive_frame` with an empty
    /// pending slot returns `Eof` instead of `NeedMore`.
    eof: bool,
}

impl WebpDecoder {
    /// Build a decoder whose output [`CodecParameters`] start from
    /// `params`. The factory always passes the caller-supplied
    /// `CodecParameters` here; the dimensions and pixel format are
    /// re-derived from each successfully decoded frame so the field is
    /// authoritative after the first `receive_frame`.
    pub fn new(params: CodecParameters) -> Self {
        let mut p = params;
        p.media_type = MediaType::Video;
        p.codec_id = CodecId::new(CODEC_ID_STR);
        p.pixel_format = Some(PixelFormat::Rgba);
        Self {
            params: p,
            pending: None,
            eof: false,
        }
    }

    /// Reference to the decoder's [`CodecParameters`]. After the first
    /// successful `receive_frame` the `width`, `height`, and
    /// `pixel_format` fields reflect the decoded image; before that
    /// they hold the factory-supplied values.
    pub fn params(&self) -> &CodecParameters {
        &self.params
    }
}

impl Decoder for WebpDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        if self.pending.is_some() {
            return Err(CoreError::other(
                "oxideav-webp decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> oxideav_core::Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(CoreError::Eof)
            } else {
                Err(CoreError::NeedMore)
            };
        };
        let img = decode_webp_image(&pkt.data)?;
        // Surface the decoded geometry on the decoder's params so
        // downstream consumers (filter graphs, sinks, the
        // `oxideav probe` command) can read width / height after the
        // first frame.
        self.params.width = Some(img.width);
        self.params.height = Some(img.height);
        self.params.pixel_format = Some(PixelFormat::Rgba);
        let vf = decoded_webp_to_video_frame(img, pkt.pts);
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ───────────────────────── Registration ─────────────────────────

/// Register the WebP decoder factory into a [`CodecRegistry`].
///
/// One [`CodecInfo`] is emitted under `CodecId("webp")` with the
/// [`PixelFormat::Rgba`] output declared on its capabilities. The codec
/// claims the `WEBP` FourCC on the off-chance a generic container
/// wraps a WebP still-image payload under that tag; everyday WebP
/// files live inside `RIFF/WEBP` and are routed via the file-extension
/// hook installed by [`register_containers`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("webp_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(16384, 16384)
        .with_pixel_formats(vec![PixelFormat::Rgba]);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .tag(CodecTag::fourcc(b"WEBP")),
    );
}

/// Register the `.webp` file extension so a `RuntimeContext` can map a
/// filename hint back to the WebP codec id.
///
/// WebP is its own container (`RIFF/WEBP`); this crate handles the
/// container walking via [`crate::parse_container`] directly rather
/// than via a separate `Demuxer` registration, so only the extension
/// hook is installed here.
pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_extension("webp", CODEC_ID_STR);
}

/// Unified registration entry point: install both the WebP decoder
/// factory and the `.webp` extension hint into the supplied
/// [`RuntimeContext`].
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    const LOSSLESS_1X1: &[u8] = include_bytes!("../tests/data/lossless-1x1.webp");
    const LOSSY_1X1: &[u8] = include_bytes!("../tests/data/lossy-1x1.webp");

    #[test]
    fn register_via_runtime_context_installs_decoder_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_ID_STR);
        assert!(
            ctx.codecs.has_decoder(&id),
            "webp decoder factory not installed via RuntimeContext"
        );
        // Encoder side stays unwired in round 112.
        assert!(!ctx.codecs.has_encoder(&id));
        // .webp file-extension hint is wired via the same call.
        assert_eq!(ctx.containers.container_for_extension("webp"), Some("webp"));
        assert_eq!(ctx.containers.container_for_extension("WEBP"), Some("webp"));
    }

    #[test]
    fn register_via_runtime_context_resolves_webp_fourcc_tag() {
        // FourCC tag claim — confirms the codec can be looked up
        // through `CodecRegistry::resolve_tag_ref` by an upstream
        // demuxer that surfaces a `WEBP` tag.
        use oxideav_core::ProbeContext;
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let tag = CodecTag::fourcc(b"WEBP");
        let id = ctx
            .codecs
            .resolve_tag_ref(&ProbeContext::new(&tag))
            .expect("WEBP fourcc resolves to a registered codec");
        assert_eq!(id.as_str(), CODEC_ID_STR);
    }

    #[test]
    fn first_decoder_returns_a_webp_decoder() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("webp decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }

    #[test]
    fn end_to_end_lossless_decode_via_runtime_context() {
        // The user-facing dispatch path: build the context, look up the
        // factory by codec id, push one packet, read one frame.
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("webp decoder factory");

        let pkt = Packet::new(0, TimeBase::new(1, 1000), LOSSLESS_1X1.to_vec());
        dec.send_packet(&pkt).expect("send_packet accepts file");
        let frame = dec.receive_frame().expect("receive_frame yields a frame");
        let v = match frame {
            Frame::Video(v) => v,
            other => panic!("expected Frame::Video, got {other:?}"),
        };
        assert_eq!(v.planes.len(), 1, "RGBA is a single interleaved plane");
        assert_eq!(v.planes[0].stride, 4, "1px-wide × 4 bytes/pixel");
        assert_eq!(v.planes[0].data.len(), 4, "1×1 image × 4 bytes/pixel");
        // lossless-1x1 fixture: ARGB 0xFFB43C5A → RGBA bytes 0xB4 0x3C
        // 0x5A 0xFF (R=180, G=60, B=90, A=255).
        assert_eq!(v.planes[0].data, [0xB4, 0x3C, 0x5A, 0xFF]);

        // After a successful decode we should be back to NeedMore.
        let again = dec.receive_frame();
        assert!(matches!(again, Err(CoreError::NeedMore)));
    }

    #[test]
    fn vp8_lossy_packet_returns_unsupported_via_registered_decoder() {
        // The §2.5 `VP8 ` lossy path is recognised-but-unsupported by
        // the registered decoder — the file walks cleanly but no frame
        // is produced. This isolates the lossy refusal as a clean
        // `Error::Unsupported`, distinct from a corrupted-bitstream
        // `InvalidData`.
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("webp decoder factory");
        let pkt = Packet::new(0, TimeBase::new(1, 1000), LOSSY_1X1.to_vec());
        dec.send_packet(&pkt).expect("send_packet accepts file");
        let err = dec
            .receive_frame()
            .expect_err("VP8 lossy must surface as Error::Unsupported");
        assert!(
            matches!(err, CoreError::Unsupported(_)),
            "expected Error::Unsupported, got {err:?}"
        );
    }

    #[test]
    fn decoder_params_carry_dims_and_pixel_format_after_first_frame() {
        // Pre-decode: pixel format is set at construction; width and
        // height are empty (the still-image codec doesn't know them
        // before the first packet has been parsed).
        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        assert_eq!(dec.params().pixel_format, Some(PixelFormat::Rgba));
        assert_eq!(dec.params().width, None);
        assert_eq!(dec.params().height, None);

        // Push the 1×1 fixture and pump the loop. Params should now
        // reflect the decoded dimensions.
        let pkt = Packet::new(0, TimeBase::new(1, 1000), LOSSLESS_1X1.to_vec());
        dec.send_packet(&pkt).unwrap();
        let _ = dec.receive_frame().expect("decodes");
        assert_eq!(dec.params().width, Some(1));
        assert_eq!(dec.params().height, Some(1));
        assert_eq!(dec.params().pixel_format, Some(PixelFormat::Rgba));
        // Codec id is forced to "webp" by the constructor regardless of
        // what the factory params said.
        assert_eq!(dec.params().codec_id.as_str(), CODEC_ID_STR);
        assert_eq!(dec.params().media_type, MediaType::Video);
    }

    #[test]
    fn double_send_packet_without_receive_is_rejected() {
        // Image-codec contract: one packet → one frame. Two consecutive
        // send_packet calls without a receive_frame between them is a
        // caller bug and surfaces as Error::Other.
        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        let pkt = Packet::new(0, TimeBase::new(1, 1000), LOSSLESS_1X1.to_vec());
        dec.send_packet(&pkt).unwrap();
        let err = dec
            .send_packet(&pkt)
            .expect_err("second send_packet without receive_frame must fail");
        // The framework's `Error::other(...)` constructor lands in the
        // catch-all `Other` variant — we don't assert the variant
        // directly because it isn't part of the public ABI.
        let s = err.to_string();
        assert!(
            s.contains("receive_frame"),
            "error message should mention receive_frame: {s}"
        );
    }

    #[test]
    fn flush_then_receive_with_no_pending_returns_eof() {
        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        dec.flush().unwrap();
        let err = dec
            .receive_frame()
            .expect_err("post-flush, no pending packet → Eof");
        assert!(matches!(err, CoreError::Eof));
    }

    #[test]
    fn decode_webp_to_frame_returns_rgba_video_frame() {
        // The framework-free helper, exercised directly (without the
        // Decoder/CodecRegistry plumbing).
        let frame = decode_webp_to_frame(LOSSLESS_1X1, Some(123)).expect("decodes");
        assert_eq!(frame.pts, Some(123));
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].stride, 4);
        assert_eq!(frame.planes[0].data, [0xB4, 0x3C, 0x5A, 0xFF]);
    }

    #[test]
    fn unsupported_error_conversion_maps_to_core_unsupported() {
        // Lossy and NoImageData both flow through Error::Unsupported.
        let lossy: CoreError = Error::Unsupported(UnsupportedKind::LossyVp8).into();
        assert!(matches!(lossy, CoreError::Unsupported(_)));
        let none: CoreError = Error::Unsupported(UnsupportedKind::NoImageData).into();
        assert!(matches!(none, CoreError::Unsupported(_)));
    }
}
