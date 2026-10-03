//! `oxideav-core` integration — the framework [`Decoder`] / [`Encoder`]
//! adapters, the [`WebpImage`] ⇄ [`VideoFrame`] conversions, and the
//! [`register`] entry points.
//!
//! Gated behind the default-on `registry` Cargo feature. Everything here
//! is a thin adapter over the standalone contract functions
//! ([`crate::decode_with`], [`crate::encode`]): one implementation, two
//! entry styles.
//!
//! * The `"webp"` decoder emits each still in its **native** layout —
//!   [`PixelFormat::Rgba`] for lossless, [`PixelFormat::Yuv420P`] /
//!   [`PixelFormat::Yuva420P`] (limited-range BT.601, carried as the
//!   frame's colour signal) for lossy — exactly what [`crate::decode`]
//!   returns. An animated file decodes to its first composited frame.
//! * The `"webp_vp8l"` encoder accepts `Rgba` / `Rgb24` frames and writes
//!   a lossless `.webp`; the `"webp_vp8"` encoder (see
//!   [`crate::encoder_vp8`]) writes the lossy path.

use std::collections::VecDeque;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, ColorSignal,
    ContainerRegistry, Decoder, Encoder, Error as CoreError, Frame, MediaType, Packet, PixelFormat,
    RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};

use crate::{
    ColorInfo, ColorRange, DecodeOptions, EncodeOptions, Metadata, Plane, WebpError, WebpImage,
    WebpMetadataOwned, WebpPixelFormat, CODEC_ID_VP8L,
};

/// Stable on-wire identifier this crate registers under in the codec
/// registry: `"webp"`.
#[doc(hidden)]
pub const CODEC_ID_STR: &str = "webp";

/// Bridge the crate error to the framework-wide `oxideav_core::Error`.
impl From<WebpError> for CoreError {
    fn from(e: WebpError) -> Self {
        match e {
            WebpError::Unsupported(m) => CoreError::Unsupported(format!("oxideav-webp: {m}")),
            WebpError::Eof => CoreError::Eof,
            WebpError::NeedMore => CoreError::NeedMore,
            WebpError::Io(io) => CoreError::Io(io),
            other => CoreError::InvalidData(other.to_string()),
        }
    }
}

// ───────────────────────── pixel-format mapping ──────────────────────────

impl From<WebpPixelFormat> for PixelFormat {
    fn from(f: WebpPixelFormat) -> Self {
        match f {
            WebpPixelFormat::Rgb24 => PixelFormat::Rgb24,
            WebpPixelFormat::Rgba => PixelFormat::Rgba,
            WebpPixelFormat::Yuv420P => PixelFormat::Yuv420P,
            WebpPixelFormat::Yuva420P => PixelFormat::Yuva420P,
        }
    }
}

impl TryFrom<PixelFormat> for WebpPixelFormat {
    type Error = WebpError;

    fn try_from(f: PixelFormat) -> Result<Self, WebpError> {
        match f {
            PixelFormat::Rgb24 => Ok(WebpPixelFormat::Rgb24),
            PixelFormat::Rgba => Ok(WebpPixelFormat::Rgba),
            PixelFormat::Yuv420P => Ok(WebpPixelFormat::Yuv420P),
            PixelFormat::Yuva420P => Ok(WebpPixelFormat::Yuva420P),
            other => Err(WebpError::unsupported(format!(
                "pixel format {other:?} has no WebP layout (want Rgba, Rgb24, Yuv420P or Yuva420P)"
            ))),
        }
    }
}

impl From<ColorInfo> for ColorSignal {
    fn from(c: ColorInfo) -> Self {
        let sig = ColorSignal::from_code_points(c.primaries, c.transfer, c.matrix, false);
        let range = match c.range {
            ColorRange::Limited => oxideav_core::ColorRange::Limited,
            ColorRange::Full => oxideav_core::ColorRange::Full,
            _ => oxideav_core::ColorRange::Unspecified,
        };
        sig.with_range(range)
    }
}

impl From<ColorSignal> for ColorInfo {
    fn from(s: ColorSignal) -> Self {
        let range = match s.range {
            oxideav_core::ColorRange::Limited => ColorRange::Limited,
            oxideav_core::ColorRange::Full => ColorRange::Full,
            _ => ColorRange::Unspecified,
        };
        ColorInfo::new(
            range,
            s.primaries.code_point(),
            s.transfer.code_point(),
            s.matrix.code_point(),
        )
    }
}

// ───────────────────────── frame conversion ──────────────────────────────

/// A [`WebpImage`] becomes a [`VideoFrame`] with one [`VideoPlane`] per
/// image plane (strides as reported) and the colour description attached
/// as the frame's colour signal. Width / height / pixel format travel on
/// [`CodecParameters`], not the frame.
impl From<WebpImage> for VideoFrame {
    fn from(img: WebpImage) -> Self {
        let color: ColorSignal = img.color.into();
        let frame = VideoFrame {
            pts: None,
            planes: img
                .planes
                .into_iter()
                .map(|p| VideoPlane {
                    stride: p.stride,
                    data: p.data,
                })
                .collect(),
        };
        frame.with_color_signal(color)
    }
}

impl WebpImage {
    /// Rebuild a [`WebpImage`] from a framework frame plus the stream
    /// geometry the frame does not carry. Side-channel planes (palette,
    /// colour signal, …) are skipped; a colour signal on the frame
    /// overrides the layout default.
    pub fn from_video_frame(
        frame: &VideoFrame,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Self, WebpError> {
        let format = WebpPixelFormat::try_from(format)?;
        let planes: Vec<Plane> = frame
            .image_planes()
            .iter()
            .map(|p| Plane::new(p.stride, p.data.clone()))
            .collect();
        let mut img = WebpImage::new(width, height, format, planes);
        if let Some(sig) = frame.color_signal() {
            if !sig.is_unspecified() {
                img.color = sig.into();
            }
        }
        img.check_geometry()?;
        Ok(img)
    }
}

/// Decode a still `.webp` straight to a [`VideoFrame`] in its native
/// layout; the returned [`CodecParameters`] carry the geometry.
#[doc(hidden)]
pub fn decode_webp_to_frame(
    bytes: &[u8],
    pts: Option<i64>,
) -> oxideav_core::Result<(VideoFrame, CodecParameters)> {
    let img = crate::decode_with(bytes, &DecodeOptions::default())?;
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(img.width);
    params.height = Some(img.height);
    params.pixel_format = Some(img.format.into());
    let mut frame: VideoFrame = img.into();
    frame.pts = pts;
    Ok((frame, params))
}

// ───────────────────────── Decoder + factory ─────────────────────────────

/// Factory for the `Decoder` trait impl — installed in the codec registry.
pub fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    Ok(Box::new(WebpDecoder::new(params.clone())))
}

/// WebP [`Decoder`] trait impl: one complete `RIFF/WEBP` file per packet,
/// one native-layout [`Frame::Video`] per `receive_frame`.
///
/// The decoder refreshes `width` / `height` / `pixel_format` on its
/// [`CodecParameters`] after every decode — see [`Self::params`].
#[derive(Debug)]
pub struct WebpDecoder {
    params: CodecParameters,
    opts: DecodeOptions,
    pending: Option<Packet>,
    eof: bool,
}

impl WebpDecoder {
    /// Build a decoder whose output [`CodecParameters`] start from
    /// `params`; `pixel_format` is refined after the first frame.
    pub fn new(params: CodecParameters) -> Self {
        Self::with_options(params, DecodeOptions::default())
    }

    /// [`Self::new`] with explicit decode limits.
    pub fn with_options(params: CodecParameters, opts: DecodeOptions) -> Self {
        let mut p = params;
        p.media_type = MediaType::Video;
        p.codec_id = CodecId::new(CODEC_ID_STR);
        if p.pixel_format.is_none() {
            p.pixel_format = Some(PixelFormat::Rgba);
        }
        Self {
            params: p,
            opts,
            pending: None,
            eof: false,
        }
    }

    /// The decoder's [`CodecParameters`]; authoritative after the first
    /// successful `receive_frame`.
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
        let img = crate::decode_with(&pkt.data, &self.opts)?;
        self.params.width = Some(img.width);
        self.params.height = Some(img.height);
        self.params.pixel_format = Some(img.format.into());
        let mut vf: VideoFrame = img.into();
        vf.pts = pkt.pts;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ───────────────────────── Encoder + factory ─────────────────────────────

/// Factory for the VP8L `Encoder` trait impl — installed in the codec
/// registry under [`CODEC_ID_VP8L`]. Accepts `Rgba` / `Rgb24` input and
/// always emits a lossless `.webp`; the framework path embeds no
/// metadata (see [`make_encoder_with_metadata`]).
pub fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    make_encoder_with_metadata(params, WebpMetadataOwned::default())
}

/// Direct factory: a VP8L encoder embedding `metadata` (ICC / Exif / XMP)
/// into every encoded `.webp`.
#[doc(hidden)]
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
        metadata: metadata.into(),
        pending_out: VecDeque::new(),
        eof: false,
    }))
}

/// WebP VP8L (lossless) [`Encoder`] trait impl: one frame in → one
/// `.webp` packet out, auto-promoting to the `VP8X` layout when the frame
/// carries alpha or the encoder holds metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct WebpVp8lEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    pix: PixelFormat,
    metadata: Metadata,
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
        let img = WebpImage::from_video_frame(v, self.width, self.height, self.pix)?
            .with_metadata(self.metadata.clone());
        let bytes = crate::encode(&img, &EncodeOptions::default())?;
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

/// One-shot lossless encode of a [`VideoFrame`] (`Rgba` / `Rgb24`) to a
/// `.webp` with `metadata` embedded.
#[doc(hidden)]
pub fn encode_vp8l_frame(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    metadata: &crate::WebpMetadata<'_>,
) -> oxideav_core::Result<Vec<u8>> {
    let img = WebpImage::from_video_frame(frame, width, height, pix)?.with_metadata(Metadata {
        icc: metadata.icc.map(<[u8]>::to_vec),
        exif: metadata.exif.map(<[u8]>::to_vec),
        xmp: metadata.xmp.map(<[u8]>::to_vec),
        gamma: None,
    });
    Ok(crate::encode(&img, &EncodeOptions::default())?)
}

// ───────────────────────── Registration ──────────────────────────────────

/// Register the WebP codecs into a [`CodecRegistry`]: the `"webp"`
/// decoder (claiming the `WEBP` FourCC), the `"webp_vp8l"` lossless
/// encoder + decoder, and the `"webp_vp8"` lossy encoder + decoder.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("webp_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(crate::MAX_DIMENSION, crate::MAX_DIMENSION)
        .with_pixel_formats(vec![
            PixelFormat::Rgba,
            PixelFormat::Yuv420P,
            PixelFormat::Yuva420P,
        ]);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .tag(CodecTag::fourcc(b"WEBP")),
    );

    let vp8l_caps = CodecCapabilities::video("webp_vp8l_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(crate::MAX_DIMENSION, crate::MAX_DIMENSION)
        .with_pixel_formats(vec![PixelFormat::Rgba, PixelFormat::Rgb24]);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_VP8L))
            .capabilities(vp8l_caps)
            .decoder(make_decoder)
            .encoder(make_encoder),
    );

    let vp8_caps = CodecCapabilities::video("webp_vp8_sw")
        .with_intra_only(true)
        .with_max_size(crate::MAX_DIMENSION - 1, crate::MAX_DIMENSION - 1)
        .with_pixel_formats(vec![PixelFormat::Yuv420P]);
    reg.register(
        CodecInfo::new(CodecId::new(crate::CODEC_ID_VP8))
            .capabilities(vp8_caps)
            .decoder(make_decoder)
            .encoder(crate::encoder_vp8::make_encoder),
    );
}

/// Register the `.webp` file extension so a `RuntimeContext` can map a
/// filename hint back to the WebP codec id.
pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_extension("webp", CODEC_ID_STR);
}

/// Unified registration: codecs + container hooks.
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
    const LOSSY_ALPHA: &[u8] = include_bytes!("../tests/data/lossy-with-alpha-128x128.webp");

    #[test]
    fn register_via_runtime_context_installs_decoder_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_ID_STR);
        assert!(ctx.codecs.has_decoder(&id));
        assert!(!ctx.codecs.has_encoder(&id));
        assert!(ctx.codecs.has_encoder(&CodecId::new(CODEC_ID_VP8L)));
        assert!(ctx.codecs.has_encoder(&CodecId::new(crate::CODEC_ID_VP8)));
        assert_eq!(ctx.containers.container_for_extension("webp"), Some("webp"));
        assert_eq!(ctx.containers.container_for_extension("WEBP"), Some("webp"));
    }

    #[test]
    fn register_via_runtime_context_resolves_webp_fourcc_tag() {
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
    fn fleet_signature_register_codecs_takes_a_codec_registry() {
        let mut ctx = RuntimeContext::new();
        crate::register_codecs(&mut ctx.codecs);
        assert!(ctx.codecs.has_decoder(&CodecId::new(CODEC_ID_STR)));
        crate::register_containers(&mut ctx.containers);
        assert_eq!(ctx.containers.container_for_extension("webp"), Some("webp"));
    }

    #[test]
    fn end_to_end_lossless_decode_via_runtime_context() {
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
        let Frame::Video(v) = frame else {
            panic!("expected Frame::Video")
        };
        assert_eq!(
            v.image_planes().len(),
            1,
            "RGBA is a single interleaved plane"
        );
        assert_eq!(v.planes[0].stride, 4);
        assert_eq!(v.planes[0].data, [0xB4, 0x3C, 0x5A, 0xFF]);
        assert_eq!(v.color_signal(), Some(ColorSignal::srgb()));
        assert!(matches!(dec.receive_frame(), Err(CoreError::NeedMore)));
    }

    #[test]
    fn vp8_lossy_packet_decodes_to_native_yuv420p() {
        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        let pkt = Packet::new(0, TimeBase::new(1, 1000), LOSSY_1X1.to_vec());
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().expect("VP8 lossy decodes") else {
            panic!("expected Frame::Video")
        };
        assert_eq!(dec.params().pixel_format, Some(PixelFormat::Yuv420P));
        assert_eq!(v.image_planes().len(), 3);
        assert_eq!(v.planes[0].data, [101]);
        let sig = v.color_signal().expect("colour signal attached");
        assert_eq!(sig.range, oxideav_core::ColorRange::Limited);
        assert_eq!(sig.matrix.code_point(), 6);

        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        dec.send_packet(&Packet::new(
            0,
            TimeBase::new(1, 1000),
            LOSSY_ALPHA.to_vec(),
        ))
        .unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!()
        };
        assert_eq!(dec.params().pixel_format, Some(PixelFormat::Yuva420P));
        assert_eq!(v.image_planes().len(), 4);
    }

    #[test]
    fn decoder_params_carry_dims_and_pixel_format_after_first_frame() {
        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        assert_eq!(dec.params().pixel_format, Some(PixelFormat::Rgba));
        assert_eq!(dec.params().width, None);
        let pkt = Packet::new(0, TimeBase::new(1, 1000), LOSSLESS_1X1.to_vec());
        dec.send_packet(&pkt).unwrap();
        let _ = dec.receive_frame().expect("decodes");
        assert_eq!(dec.params().width, Some(1));
        assert_eq!(dec.params().height, Some(1));
        assert_eq!(dec.params().pixel_format, Some(PixelFormat::Rgba));
        assert_eq!(dec.params().codec_id.as_str(), CODEC_ID_STR);
        assert_eq!(dec.params().media_type, MediaType::Video);
    }

    #[test]
    fn double_send_packet_without_receive_is_rejected() {
        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        let pkt = Packet::new(0, TimeBase::new(1, 1000), LOSSLESS_1X1.to_vec());
        dec.send_packet(&pkt).unwrap();
        let err = dec.send_packet(&pkt).expect_err("second send must fail");
        assert!(err.to_string().contains("receive_frame"));
    }

    #[test]
    fn flush_then_receive_with_no_pending_returns_eof() {
        let mut dec = WebpDecoder::new(CodecParameters::video(CodecId::new(CODEC_ID_STR)));
        dec.flush().unwrap();
        assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
    }

    #[test]
    fn decode_limits_surface_as_invalid_data() {
        let opts = DecodeOptions::default().with_max_width(Some(16));
        let mut dec =
            WebpDecoder::with_options(CodecParameters::video(CodecId::new(CODEC_ID_STR)), opts);
        dec.send_packet(&Packet::new(
            0,
            TimeBase::new(1, 1000),
            LOSSY_ALPHA.to_vec(),
        ))
        .unwrap();
        assert!(matches!(
            dec.receive_frame(),
            Err(CoreError::InvalidData(_))
        ));
    }

    #[test]
    fn decode_webp_to_frame_returns_native_video_frame() {
        let (frame, params) = decode_webp_to_frame(LOSSLESS_1X1, Some(123)).expect("decodes");
        assert_eq!(frame.pts, Some(123));
        assert_eq!(frame.planes[0].data, [0xB4, 0x3C, 0x5A, 0xFF]);
        assert_eq!(params.pixel_format, Some(PixelFormat::Rgba));
        assert_eq!((params.width, params.height), (Some(1), Some(1)));
    }

    #[test]
    fn error_conversion_maps_variants() {
        let u: CoreError = WebpError::unsupported("x").into();
        assert!(matches!(u, CoreError::Unsupported(_)));
        let i: CoreError = WebpError::invalid("x").into();
        assert!(matches!(i, CoreError::InvalidData(_)));
        let l: CoreError = WebpError::limit("x").into();
        assert!(matches!(l, CoreError::InvalidData(_)));
        assert!(matches!(CoreError::from(WebpError::Eof), CoreError::Eof));
    }

    #[test]
    fn image_video_frame_round_trip_keeps_planes_and_colour() {
        let img = crate::decode(LOSSY_ALPHA).unwrap();
        let fmt: PixelFormat = img.format.into();
        assert_eq!(fmt, PixelFormat::Yuva420P);
        let vf: VideoFrame = img.clone().into();
        assert_eq!(vf.image_planes().len(), 4);
        let back = WebpImage::from_video_frame(&vf, img.width, img.height, fmt).unwrap();
        assert_eq!(back.planes, img.planes);
        assert_eq!(back.color, img.color);
        assert!(WebpImage::from_video_frame(&vf, 1, 1, PixelFormat::Gray8).is_err());
    }

    // ───────────────────── VP8L encoder ─────────────────────

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
    fn vp8l_encoder_round_trips_rgba_through_registry() {
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

        let img = crate::decode_rgba8(&pkt.data).expect("decode our own webp");
        assert_eq!((img.width, img.height), (w, h));
        let Frame::Video(v) = &frame else {
            unreachable!()
        };
        assert_eq!(img.data, v.planes[0].data);
    }

    #[test]
    fn vp8l_encoder_streams_rgb24_as_opaque() {
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
                data: data.clone(),
            }],
        });
        let mut enc =
            make_encoder(&vp8l_params(w, h, PixelFormat::Rgb24)).expect("make_encoder rgb24");
        enc.send_frame(&frame).unwrap();
        let pkt = enc.receive_packet().unwrap();
        let c = crate::parse_container(&pkt.data).unwrap();
        assert!(c
            .first_chunk_with_fourcc(crate::container::fourcc::VP8X)
            .is_none());
        assert_eq!(crate::decode_rgb8(&pkt.data).unwrap().data, data);
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

        let img = crate::decode(&pkt.data).unwrap();
        assert_eq!(img.metadata.icc.as_deref(), Some(&b"icc-profile"[..]));
        assert_eq!(img.metadata.exif.as_deref(), Some(&b"Exif\x00\x00II"[..]));
        assert_eq!(img.metadata.xmp, None);
        let Frame::Video(v) = &frame else {
            unreachable!()
        };
        assert_eq!(img.as_bytes().unwrap(), &v.planes[0].data[..]);
    }

    #[test]
    fn vp8l_encoder_receive_before_send_is_need_more() {
        let mut enc = make_encoder(&vp8l_params(1, 1, PixelFormat::Rgba)).unwrap();
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));
        enc.flush().unwrap();
        assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));
    }
}
