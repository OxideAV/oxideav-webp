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

use std::collections::VecDeque;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag,
    ContainerRegistry, Decoder, Encoder, Error as CoreError, Frame, MediaType, Packet, PixelFormat,
    RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};

use crate::{
    decode_webp_image, encode_vp8l_argb_with_metadata, DecodedWebp, Error, UnsupportedKind,
    WebpMetadata, WebpMetadataOwned, CODEC_ID_VP8L,
};

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

// ───────────────────────── Encoder + factory ─────────────────────────

/// Repack one interleaved-pixel [`VideoFrame`] plane into scan-line ARGB
/// (`(a << 24) | (r << 16) | (g << 8) | b`), the layout the VP8L encoder
/// consumes.
///
/// Accepts the two input pixel formats the published `webp_vp8l` codec
/// declares: [`PixelFormat::Rgba`] (4 B/px) and [`PixelFormat::Rgb24`]
/// (3 B/px, treated as fully opaque, streamed without a 3→4 expansion
/// alloc). Both read the plane's `stride` so a padded source row is handled.
fn video_frame_to_argb(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
) -> oxideav_core::Result<(Vec<u32>, bool)> {
    let plane = frame
        .planes
        .first()
        .ok_or_else(|| CoreError::invalid("webp_vp8l encoder: frame has no planes"))?;
    let w = width as usize;
    let h = height as usize;
    let stride = plane.stride;
    let mut pixels = Vec::with_capacity(w * h);
    let mut alpha_is_used = false;
    match pix {
        PixelFormat::Rgba => {
            for y in 0..h {
                let row = &plane.data[y * stride..];
                for x in 0..w {
                    let p = &row[x * 4..x * 4 + 4];
                    let (r, g, b, a) = (p[0] as u32, p[1] as u32, p[2] as u32, p[3] as u32);
                    if a != 0xff {
                        alpha_is_used = true;
                    }
                    pixels.push((a << 24) | (r << 16) | (g << 8) | b);
                }
            }
        }
        PixelFormat::Rgb24 => {
            for y in 0..h {
                let row = &plane.data[y * stride..];
                for x in 0..w {
                    let p = &row[x * 3..x * 3 + 3];
                    let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
                    pixels.push((0xff << 24) | (r << 16) | (g << 8) | b);
                }
            }
        }
        other => {
            return Err(CoreError::invalid(format!(
                "webp_vp8l encoder: unsupported input pixel format {other:?} (want Rgba or Rgb24)"
            )));
        }
    }
    Ok((pixels, alpha_is_used))
}

/// Factory for the VP8L `Encoder` trait impl — installed in the codec
/// registry under [`CODEC_ID_VP8L`] and called by the framework when a
/// `webp_vp8l` encode is requested.
///
/// Reads `width` / `height` / `pixel_format` from `params`; the encoder
/// accepts [`PixelFormat::Rgba`] or [`PixelFormat::Rgb24`] input and always
/// emits a §2.6 `VP8L` lossless `.webp`. ICC / Exif / XMP metadata is
/// carried as a [`WebpMetadataOwned`] derived from `params.extradata` is
/// **not** assumed here — the framework path embeds no metadata; the direct
/// factory [`make_encoder_with_metadata`] takes it explicitly.
pub fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    make_encoder_with_metadata(params, WebpMetadataOwned::default())
}

/// Direct factory: build a VP8L encoder embedding the supplied file-level
/// metadata (ICC / Exif / XMP) into every encoded `.webp`.
///
/// The dual-API counterpart of [`make_encoder`] — the registry path uses the
/// no-metadata form, while a direct caller that wants to embed an ICC
/// profile / Exif / XMP block constructs the encoder through this factory.
pub fn make_encoder_with_metadata(
    params: &CodecParameters,
    metadata: WebpMetadataOwned,
) -> oxideav_core::Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| CoreError::invalid("webp_vp8l encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| CoreError::invalid("webp_vp8l encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Rgba);
    if !matches!(pix, PixelFormat::Rgba | PixelFormat::Rgb24) {
        return Err(CoreError::invalid(format!(
            "webp_vp8l encoder: unsupported input pixel format {pix:?} (want Rgba or Rgb24)"
        )));
    }

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(CODEC_ID_VP8L);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);

    Ok(Box::new(WebpVp8lEncoder {
        output_params,
        width,
        height,
        pix,
        metadata,
        pending_out: VecDeque::new(),
        eof: false,
    }))
}

/// WebP VP8L (lossless) [`Encoder`] trait impl.
///
/// One frame in → one `.webp` packet out. Each `send_frame` carries an
/// interleaved RGBA / RGB24 picture; the matching `receive_packet` (after
/// the frame, or on flush) emits a complete §2.6 / §2.7 `.webp` file. The
/// output auto-promotes to the extended `VP8X` layout when the frame carries
/// alpha or the encoder was constructed with non-empty metadata.
#[derive(Debug)]
pub struct WebpVp8lEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    pix: PixelFormat,
    metadata: WebpMetadataOwned,
    pending_out: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for WebpVp8lEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> oxideav_core::Result<()> {
        let Frame::Video(v) = frame else {
            return Err(CoreError::invalid("webp_vp8l encoder: video frames only"));
        };
        let (argb, frame_alpha) = video_frame_to_argb(v, self.width, self.height, self.pix)?;
        let has_alpha = frame_alpha;
        let meta = self.metadata.as_borrowed();
        let bytes =
            encode_vp8l_argb_with_metadata(self.width, self.height, &argb, has_alpha, &meta)
                .map_err(|e| CoreError::InvalidData(e.to_string()))?;
        let mut pkt = Packet::new(0, TimeBase::new(1, 1000), bytes);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = true;
        self.pending_out.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> oxideav_core::Result<Packet> {
        if let Some(p) = self.pending_out.pop_front() {
            return Ok(p);
        }
        if self.eof {
            Err(CoreError::Eof)
        } else {
            Err(CoreError::NeedMore)
        }
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// `Vec<u8>`-flavoured wrapper around [`encode_vp8l_argb_with_metadata`] —
/// repacks a [`VideoFrame`] (Rgba / Rgb24) into ARGB and encodes a `.webp`.
///
/// Preserved alongside the [`Encoder`] trait impl for callers that already
/// build [`VideoFrame`]s directly and want a one-shot `.webp` without the
/// trait plumbing (the dual-API direct path).
pub fn encode_vp8l_frame(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    metadata: &WebpMetadata<'_>,
) -> oxideav_core::Result<Vec<u8>> {
    let (argb, alpha_is_used) = video_frame_to_argb(frame, width, height, pix)?;
    encode_vp8l_argb_with_metadata(width, height, &argb, alpha_is_used, metadata)
        .map_err(|e| CoreError::InvalidData(e.to_string()))
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

    // VP8L lossless encoder codec. Accepts Rgba / Rgb24 input and emits a
    // §2.6 / §2.7 VP8L `.webp`; also exposes the decoder under the same id
    // so a `webp_vp8l` stream round-trips through one codec entry.
    let vp8l_caps = CodecCapabilities::video("webp_vp8l_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(16384, 16384)
        .with_pixel_formats(vec![PixelFormat::Rgba, PixelFormat::Rgb24]);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_VP8L))
            .capabilities(vp8l_caps)
            .decoder(make_decoder)
            .encoder(make_encoder),
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

    // ───────────────────── VP8L encoder ─────────────────────

    /// Build a small Rgba [`Frame`] (no stride padding).
    fn rgba_frame(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> Frame {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&fill(x, y));
            }
        }
        Frame::Video(VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: (width * 4) as usize,
                data,
            }],
        })
    }

    fn vp8l_params(width: u32, height: u32, pix: PixelFormat) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8L));
        p.width = Some(width);
        p.height = Some(height);
        p.pixel_format = Some(pix);
        p
    }

    #[test]
    fn register_installs_vp8l_encoder_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_ID_VP8L);
        assert!(
            ctx.codecs.has_encoder(&id),
            "webp_vp8l encoder factory not installed"
        );
        assert!(
            ctx.codecs.has_decoder(&id),
            "webp_vp8l decoder factory not installed"
        );
    }

    #[test]
    fn vp8l_encoder_round_trips_rgba_through_registry() {
        // Encode an RGBA frame through the registered encoder, decode the
        // resulting `.webp`, and assert the pixels survive exactly.
        let (w, h) = (4u32, 3u32);
        let frame = rgba_frame(w, h, |x, y| {
            [(x * 40) as u8, (y * 60) as u8, ((x + y) * 25) as u8, 0xff]
        });

        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut enc = ctx
            .codecs
            .first_encoder(&vp8l_params(w, h, PixelFormat::Rgba))
            .expect("webp_vp8l encoder factory");
        enc.send_frame(&frame).expect("send_frame");
        let pkt = enc.receive_packet().expect("one packet out");

        let img = crate::decode_webp(&pkt.data).expect("decode our own webp");
        assert_eq!(img.frames.len(), 1);
        assert_eq!(img.frames[0].width, w);
        assert_eq!(img.frames[0].height, h);
        // Re-derive expected RGBA.
        let Frame::Video(v) = &frame else {
            unreachable!()
        };
        assert_eq!(img.frames[0].rgba, v.planes[0].data);
    }

    #[test]
    fn vp8l_encoder_streams_rgb24_as_opaque() {
        // An Rgb24 frame is treated as fully opaque (alpha 0xff) and emits
        // the simple (non-VP8X) layout.
        let (w, h) = (3u32, 2u32);
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[(x * 50) as u8, (y * 70) as u8, 0x33]);
            }
        }
        let frame = Frame::Video(VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: (w * 3) as usize,
                data,
            }],
        });

        let mut enc =
            make_encoder(&vp8l_params(w, h, PixelFormat::Rgb24)).expect("make_encoder rgb24");
        enc.send_frame(&frame).unwrap();
        let pkt = enc.receive_packet().unwrap();

        // Simple lossless layout: no VP8X chunk.
        let c = crate::parse_container(&pkt.data).unwrap();
        assert!(c
            .first_chunk_with_fourcc(crate::container::fourcc::VP8X)
            .is_none());
        let img = crate::decode_webp(&pkt.data).unwrap();
        // Every pixel opaque, RGB preserved.
        for px in img.frames[0].rgba.chunks_exact(4) {
            assert_eq!(px[3], 0xff);
        }
    }

    #[test]
    fn vp8l_encoder_with_metadata_promotes_to_vp8x() {
        let (w, h) = (2u32, 2u32);
        let frame = rgba_frame(w, h, |x, _| [(x * 100) as u8, 0x10, 0x20, 0x80]);

        let meta = WebpMetadataOwned {
            icc: Some(b"icc-profile".to_vec()),
            exif: Some(b"Exif\x00\x00II".to_vec()),
            xmp: None,
        };
        let mut enc = make_encoder_with_metadata(&vp8l_params(w, h, PixelFormat::Rgba), meta)
            .expect("make_encoder_with_metadata");
        enc.send_frame(&frame).unwrap();
        let pkt = enc.receive_packet().unwrap();

        // Extended layout: VP8X present, metadata round-trips.
        let c = crate::parse_container(&pkt.data).unwrap();
        assert!(c
            .first_chunk_with_fourcc(crate::container::fourcc::VP8X)
            .is_some());
        let read = crate::extract_metadata(&pkt.data).unwrap();
        assert_eq!(read.icc.as_deref(), Some(&b"icc-profile"[..]));
        assert_eq!(read.exif.as_deref(), Some(&b"Exif\x00\x00II"[..]));
        assert_eq!(read.xmp, None);

        // Pixels still round-trip through the alpha-bearing image.
        let img = crate::decode_webp(&pkt.data).unwrap();
        let Frame::Video(v) = &frame else {
            unreachable!()
        };
        assert_eq!(img.frames[0].rgba, v.planes[0].data);
    }

    #[test]
    fn vp8l_encoder_receive_before_send_is_need_more() {
        let mut enc = make_encoder(&vp8l_params(1, 1, PixelFormat::Rgba)).unwrap();
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));
        enc.flush().unwrap();
        assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));
    }
}
