//! `oxideav_codec::Encoder` adapter around the bare VP8L bitstream encoder
//! in [`crate::vp8l::encoder`].
//!
//! The encoder accepts a single `Rgba` video frame per `send_frame` and
//! emits a RIFF-wrapped `.webp` file on `receive_packet`. Frames whose
//! alpha is uniformly opaque go into a simple `RIFF/WEBP/VP8L` layout;
//! frames that carry alpha information get an extended `RIFF/WEBP/VP8X +
//! VP8L` layout so the VP8X header can advertise the alpha flag and the
//! canvas size — required for any WebP reader that honours the extended
//! format spec.
//!
//! Callers that want to stay on the bare-bitstream path (decodable
//! directly by [`crate::vp8l::decode`]) should call
//! [`crate::vp8l::encode_vp8l_argb`] themselves — that entry point
//! remains unchanged.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame,
};

use crate::riff::{build_webp_file, ImageKind, WebpMetadata};
use crate::vp8l::encode_vp8l_argb;
use crate::CODEC_ID_VP8L;

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("VP8L encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("VP8L encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Rgba);
    if pix != PixelFormat::Rgba {
        return Err(Error::unsupported(format!(
            "VP8L encoder: pixel format {pix:?} not supported — feed Rgba"
        )));
    }

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(CODEC_ID_VP8L);
    output_params.pixel_format = Some(PixelFormat::Rgba);
    output_params.width = Some(width);
    output_params.height = Some(height);

    let time_base = TimeBase::new(1, 1000);

    Ok(Box::new(Vp8lEncoder {
        output_params,
        width,
        height,
        time_base,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct Vp8lEncoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for Vp8lEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("VP8L encoder: video frames only")),
        };
        if v.width != self.width || v.height != self.height {
            return Err(Error::invalid(
                "VP8L encoder: frame dimensions must match encoder config",
            ));
        }
        if v.format != PixelFormat::Rgba {
            return Err(Error::invalid(format!(
                "VP8L encoder: frame format {:?} must be Rgba",
                v.format
            )));
        }
        let bytes = encode_frame(v)?;
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = v.pts;
        pkt.dts = pkt.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.pending.pop_front() {
            return Ok(p);
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Pack an Rgba `VideoFrame` into ARGB u32 pixels and run the VP8L encoder.
/// Returns a full `.webp` file — simple-layout when the frame is fully
/// opaque, extended (VP8X + VP8L) when alpha carries data.
fn encode_frame(v: &VideoFrame) -> Result<Vec<u8>> {
    let w = v.width as usize;
    let h = v.height as usize;
    if v.planes.is_empty() {
        return Err(Error::invalid("VP8L encoder: frame has no planes"));
    }
    let plane = &v.planes[0];
    if plane.stride < w * 4 {
        return Err(Error::invalid("VP8L encoder: RGBA stride too small"));
    }
    let mut pixels = Vec::with_capacity(w * h);
    let mut has_alpha = false;
    for y in 0..h {
        let row = &plane.data[y * plane.stride..y * plane.stride + w * 4];
        for x in 0..w {
            let r = row[x * 4] as u32;
            let g = row[x * 4 + 1] as u32;
            let b = row[x * 4 + 2] as u32;
            let a = row[x * 4 + 3] as u32;
            if a != 0xff {
                has_alpha = true;
            }
            pixels.push((a << 24) | (r << 16) | (g << 8) | b);
        }
    }
    let bitstream = encode_vp8l_argb(v.width, v.height, &pixels, has_alpha)?;
    // When the frame carries alpha we *must* emit the extended layout
    // (VP8X) per the RIFF container spec — readers that parse only the
    // simple form would otherwise miss the alpha flag. A fully-opaque
    // frame takes the cheaper simple layout.
    let meta = WebpMetadata::default();
    if has_alpha {
        // Force VP8X path by going through `build_extended` via the
        // metadata helper: we re-use `icc`/`exif`/`xmp` == None but
        // still want VP8X. The `riff` module switches to extended when
        // we pass a sentinel ALPH or any metadata; for VP8L-with-alpha
        // we need a third trigger — the alpha flag itself. Expose that
        // through a dedicated helper.
        Ok(crate::riff::build_vp8l_with_alpha(
            &bitstream, v.width, v.height, &meta,
        ))
    } else {
        Ok(build_webp_file(
            ImageKind::Vp8lLossless,
            &bitstream,
            v.width,
            v.height,
            None,
            &meta,
        ))
    }
}
