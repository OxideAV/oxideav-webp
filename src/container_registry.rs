//! The framework `webp` container — demuxer, muxer, content probe —
//! installed by [`crate::register_containers`] next to the codecs.
//!
//! A WebP file is its own container (RFC 9649 §2.4 `RIFF` / `WEBP`), so
//! the demuxer publishes the stream layout the registered decoder will
//! emit and cuts an animation into one packet per `ANMF` frame. Every
//! packet this module produces or accepts is a complete standalone
//! `.webp` file, so Layer 1 [`crate::decode`] reads any packet on its
//! own and the `webp_vp8l` / `webp_vp8` encoders' output (one still per
//! frame) muxes without translation.
//!
//! # Stream layout
//!
//! One video stream, time base [`TIME_BASE`] = 1/1000 s (the §2.7.1.1
//! `Frame Duration` unit). `codec_id` is `webp_vp8l` for a `VP8L`
//! bitstream and `webp_vp8` for `VP8 ` (an animation: its first frame's;
//! both decoders accept either kind, see [`crate::make_decoder`]).
//!
//! * **Still:** one packet holding the whole file; `width` / `height` /
//!   `pixel_format` / `color_signal` are what [`crate::info`] reports —
//!   `Rgba` (sRGB) for lossless, `Yuv420P` / `Yuva420P` (with `ALPH`)
//!   with the RFC 9649 §2.5 limited-range BT.601 colour signal for lossy.
//! * **Animation** (`VP8X` `A` flag / `ANIM`): one packet per `ANMF`
//!   chunk in file order, `pixel_format = Rgba` (the composited canvas,
//!   as [`crate::decode_all`] returns; `width` / `height` = the `VP8X`
//!   canvas), `pts` cumulative from `0`, `duration` = the frame's
//!   `Frame Duration` in ms. Packet `i` is a one-frame animated file:
//!   `RIFF` / `WEBP`, `VP8X` (animation flag, the canvas), `ANIM`
//!   (background, loop count) and the original `ANMF` chunk verbatim;
//!   packet `0` also carries the file's `ICCP` / `EXIF` / `XMP ` chunks
//!   (with the matching `VP8X` flags). Only the first packet is a
//!   keyframe: later frames compose over the previous canvas.
//!
//! [`Demuxer::metadata`] carries `("loop_count", n)` and
//! `("background_color", "#RRGGBBAA")` from the `ANIM` chunk.
//!
//! # `extradata`
//!
//! The demuxer tells the decoder which packetisation a stream uses:
//!
//! ```text
//! byte 0   EXTRADATA_VERSION (1)
//! byte 1   EXTRADATA_STILL (0) | EXTRADATA_ANIMATION (1)
//! ```
//!
//! [`is_animation_stream`] reads it. A decoder built from parameters
//! without this record keeps the whole-file behaviour: every packet is a
//! complete file, a still comes out native and an animated file decodes
//! to its first composited frame (as [`crate::decode`] does).
//!
//! # Muxer
//!
//! [`open_muxer`] takes one video stream whose `codec_id` is `webp`,
//! `webp_vp8l` or `webp_vp8`. A single packet is written verbatim. Two or
//! more packets are merged into one animated file at the chunk level —
//! no pixel is re-encoded: a packet that is itself a one-frame animation
//! (the demuxer's output) contributes its `ANMF` chunk as is; a still
//! packet (either encoder's output) becomes an `ANMF` frame at the
//! canvas origin with blending off (`B = 1`) and no disposal, its frame
//! data the packet's `ALPH` (if any) and `VP8 ` / `VP8L` chunks. The
//! canvas is the first packet's (`VP8X` canvas, else its bitstream
//! size); every frame must fit. Each packet's `duration` — rescaled from
//! its `time_base` to milliseconds — becomes its frame's `Frame
//! Duration` (a packet without one keeps its `ANMF` value, or `0`). The
//! `ANIM` chunk comes from the first packet's when it has one, else from
//! the stream's `loop_count` / `background_color` options (`0` = loop
//! forever and transparent black by default); `ICCP` / `EXIF` / `XMP `
//! come from the first packet. Because the merge is chunk-level, lossy
//! (`webp_vp8`) and mixed frames mux fine even though Layer 1
//! [`crate::encode_animation`] re-encodes from RGBA and refuses a
//! `quality`. `decode_all(mux(frames)) == frames` for full-canvas `Rgba`
//! frames (lossless packets).
//!
//! Whole module gated behind the `registry` feature.

use std::io::{Read, SeekFrom, Write};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, ColorSignal, ContainerRegistry, Demuxer,
    Error as CoreError, MediaType, Muxer, Packet, ProbeData, ProbeScore, ReadSeek,
    Result as CoreResult, Rounding, StreamInfo, TimeBase, WriteSeek, MAX_PROBE_SCORE,
    PROBE_SCORE_EXTENSION,
};

use crate::anim_encode::{build_anim_payload, build_anmf_header_then_data};
use crate::anmf::{AnmfHeader, BlendingMethod, DisposalMethod};
use crate::build::{build_chunk, build_vp8x_chunk, Vp8xFlags};
use crate::container::{fourcc, FourCc, WebpChunk, WebpContainer};
use crate::{WebpError, CODEC_ID_VP8, CODEC_ID_VP8L};

/// The §2.7.1.1 `Frame Duration` unit: milliseconds. Every packet and
/// the stream use this time base.
pub const TIME_BASE: TimeBase = TimeBase::new(1, 1000);

/// `extradata[0]`: layout version of the record this module writes.
pub const EXTRADATA_VERSION: u8 = 1;
/// `extradata[1]` for a still (one packet, native layout).
pub const EXTRADATA_STILL: u8 = 0;
/// `extradata[1]` for an animation (one packet per `ANMF`, `Rgba`).
pub const EXTRADATA_ANIMATION: u8 = 1;

/// Container name registered for WebP (demuxer, muxer, probe, extension).
pub const CONTAINER_NAME: &str = "webp";

/// Register the WebP container: demuxer + muxer + `.webp` extension +
/// probe.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_NAME, open_demuxer);
    reg.register_muxer(CONTAINER_NAME, open_muxer);
    reg.register_extension("webp", CONTAINER_NAME);
    reg.register_probe(CONTAINER_NAME, probe);
}

/// Content probe: the §2.4 `RIFF` … `WEBP` header is unambiguous; the
/// `.webp` extension alone scores [`PROBE_SCORE_EXTENSION`].
pub fn probe(data: &ProbeData) -> ProbeScore {
    if crate::probe(data.buf) {
        return MAX_PROBE_SCORE;
    }
    if data.ext == Some("webp") {
        PROBE_SCORE_EXTENSION
    } else {
        0
    }
}

/// `true` when `params.extradata` carries this module's record and marks
/// the stream as an animation (one `ANMF` frame per packet, composited
/// across packets).
pub fn is_animation_stream(params: &CodecParameters) -> bool {
    matches!(
        params.extradata.as_slice(),
        [EXTRADATA_VERSION, EXTRADATA_ANIMATION, ..]
    )
}

/// `true` when the codec id is one this container carries.
pub fn is_webp_codec(id: &CodecId) -> bool {
    matches!(
        id.as_str(),
        crate::registry::CODEC_ID_STR | CODEC_ID_VP8L | CODEC_ID_VP8
    )
}

fn invalid(e: impl std::fmt::Display) -> CoreError {
    CoreError::invalid(format!("webp container: {e}"))
}

/// A crate-level parse error as the framework error (through the
/// crate's own `WebpError` mapping).
fn w<E: Into<WebpError>>(e: E) -> CoreError {
    CoreError::from(e.into())
}

/// The §2.3 walk as a core error.
fn parse_container(buf: &[u8]) -> CoreResult<WebpContainer> {
    crate::container::parse(buf).map_err(w)
}

/// The whole on-disk chunk (`FourCC` + `Size` + payload + pad byte).
fn raw_chunk<'a>(buf: &'a [u8], c: &WebpChunk) -> &'a [u8] {
    let start = c.payload_start - 8;
    let end = (c.payload_end + (c.size as usize & 1)).min(buf.len());
    &buf[start..end]
}

// ---- Demuxer ------------------------------------------------------------

/// Open a WebP file as a one-stream container (see the module docs for
/// the still / animation packetisation).
pub fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> CoreResult<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    drop(input);

    if !crate::probe(&buf) {
        return Err(invalid("bad magic (expected RIFF … WEBP)"));
    }
    let c = parse_container(&buf)?;
    // Header only: dimensions, native layout, colour, animation flags.
    let header = crate::info(&buf)?;
    let codec = CodecId::new(if header.is_lossy {
        CODEC_ID_VP8
    } else {
        CODEC_ID_VP8L
    });
    let mut params = CodecParameters::video(codec);
    params.width = Some(header.width);
    params.height = Some(header.height);
    params.pixel_format = Some(header.format.into());
    params.color_signal = ColorSignal::from(header.color);

    let mut metadata: Vec<(String, String)> = Vec::new();
    let mut packets = Vec::new();
    if header.is_animated {
        params.extradata = vec![EXTRADATA_VERSION, EXTRADATA_ANIMATION];
        let opts = crate::DecodeOptions::default();
        let (hdr, anim) = crate::api::animation_headers(&buf, &c, &opts)?;
        let bg = anim.background_color;
        metadata.push(("loop_count".into(), anim.loop_count.to_string()));
        metadata.push((
            "background_color".into(),
            format!(
                "#{:02X}{:02X}{:02X}{:02X}",
                bg.red, bg.green, bg.blue, bg.alpha
            ),
        ));
        let anim_chunk = build_chunk(
            fourcc::ANIM,
            &build_anim_payload(anim.loop_count, [bg.red, bg.green, bg.blue, bg.alpha]),
        )
        .map_err(invalid)?;
        let meta_chunk = |tag: FourCc| c.first_chunk_with_fourcc(tag).map(|ch| raw_chunk(&buf, ch));
        let (iccp, exif, xmp) = (
            meta_chunk(fourcc::ICCP),
            meta_chunk(fourcc::EXIF),
            meta_chunk(fourcc::XMP),
        );
        let mut pts: i64 = 0;
        for (i, anmf) in c.chunks_with_fourcc(fourcc::ANMF).enumerate() {
            let frame = AnmfHeader::parse(anmf.payload(&buf)).map_err(w)?;
            let first = i == 0;
            let flags = Vp8xFlags {
                has_iccp: first && iccp.is_some(),
                has_alpha: hdr.has_alpha,
                has_exif: first && exif.is_some(),
                has_xmp: first && xmp.is_some(),
                has_animation: true,
            };
            let vp8x = build_chunk(
                fourcc::VP8X,
                &build_vp8x_chunk(hdr.canvas_width, hdr.canvas_height, flags).map_err(invalid)?,
            )
            .map_err(invalid)?;
            // §2.7 chunk order: VP8X, ICCP, ANIM, ANMF, EXIF, XMP.
            let mut body = Vec::new();
            body.extend_from_slice(&vp8x);
            if first {
                if let Some(ch) = iccp {
                    body.extend_from_slice(ch);
                }
            }
            body.extend_from_slice(&anim_chunk);
            body.extend_from_slice(raw_chunk(&buf, anmf));
            if first {
                if let Some(ch) = exif {
                    body.extend_from_slice(ch);
                }
                if let Some(ch) = xmp {
                    body.extend_from_slice(ch);
                }
            }
            let data = crate::api::frame_riff(body)?;
            let mut pkt = Packet::new(0, TIME_BASE, data);
            pkt.pts = Some(pts);
            pkt.dts = Some(pts);
            pkt.duration = Some(i64::from(frame.duration_ms));
            pkt.flags.keyframe = first;
            pts += i64::from(frame.duration_ms);
            packets.push(pkt);
        }
        if packets.is_empty() {
            return Err(invalid("animation has no ANMF frames"));
        }
    } else {
        params.extradata = vec![EXTRADATA_VERSION, EXTRADATA_STILL];
        let mut pkt = Packet::new(0, TIME_BASE, buf);
        pkt.pts = Some(0);
        pkt.dts = Some(0);
        pkt.flags.keyframe = true;
        packets.push(pkt);
    }

    let total: i64 = packets.iter().filter_map(|p| p.duration).sum();
    let stream = StreamInfo {
        index: 0,
        time_base: TIME_BASE,
        duration: if header.is_animated {
            Some(total)
        } else {
            None
        },
        start_time: Some(0),
        params,
    };
    Ok(Box::new(WebpDemuxer {
        stream,
        packets,
        pos: 0,
        metadata,
    }))
}

struct WebpDemuxer {
    stream: StreamInfo,
    packets: Vec<Packet>,
    pos: usize,
    metadata: Vec<(String, String)>,
}

impl Demuxer for WebpDemuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }

    fn streams(&self) -> &[StreamInfo] {
        std::slice::from_ref(&self.stream)
    }

    fn next_packet(&mut self) -> CoreResult<Packet> {
        let Some(pkt) = self.packets.get(self.pos) else {
            return Err(CoreError::Eof);
        };
        self.pos += 1;
        Ok(pkt.clone())
    }

    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    fn duration_micros(&self) -> Option<i64> {
        self.stream.duration.map(|d| d.saturating_mul(1_000))
    }
}

// ---- Muxer --------------------------------------------------------------

/// Open a WebP muxer for exactly one `webp` / `webp_vp8l` / `webp_vp8`
/// video stream (see the module docs for how several packets become one
/// animated file).
pub fn open_muxer(
    output: Box<dyn WriteSeek>,
    streams: &[StreamInfo],
) -> CoreResult<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(CoreError::invalid(
            "webp muxer: exactly one video stream expected",
        ));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Video {
        return Err(CoreError::invalid("webp muxer: stream must be video"));
    }
    if !is_webp_codec(&s.params.codec_id) {
        return Err(CoreError::invalid(format!(
            "webp muxer: codec_id must be webp, webp_vp8l or webp_vp8 (got {})",
            s.params.codec_id
        )));
    }
    let loop_count = match s.params.options.get("loop_count") {
        Some(v) => v.trim().parse::<u16>().map_err(|_| {
            CoreError::invalid(format!(
                "webp muxer: option `loop_count` got {v:?}; expected 0..=65535"
            ))
        })?,
        None => 0,
    };
    let background_rgba = match s.params.options.get("background_color") {
        Some(v) => parse_rgba_hex(v).ok_or_else(|| {
            CoreError::invalid(format!(
                "webp muxer: option `background_color` got {v:?}; expected #RRGGBBAA"
            ))
        })?,
        None => [0, 0, 0, 0],
    };
    Ok(Box::new(WebpMuxer {
        output,
        loop_count,
        background_rgba,
        packets: Vec::new(),
        header_written: false,
        trailer_written: false,
    }))
}

/// `#RRGGBBAA` (or `#RRGGBB`, alpha 255) → `[R, G, B, A]`.
fn parse_rgba_hex(s: &str) -> Option<[u8; 4]> {
    let hex = s.trim().strip_prefix('#')?;
    let byte = |i: usize| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok();
    match hex.len() {
        6 => Some([byte(0)?, byte(2)?, byte(4)?, 255]),
        8 => Some([byte(0)?, byte(2)?, byte(4)?, byte(6)?]),
        _ => None,
    }
}

struct WebpMuxer {
    output: Box<dyn WriteSeek>,
    loop_count: u16,
    background_rgba: [u8; 4],
    packets: Vec<Packet>,
    header_written: bool,
    trailer_written: bool,
}

impl Muxer for WebpMuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }

    fn write_header(&mut self) -> CoreResult<()> {
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if !self.header_written {
            return Err(CoreError::other("webp muxer: write_header not called"));
        }
        if !crate::probe(&packet.data) {
            return Err(CoreError::invalid(
                "webp muxer: packet is not a RIFF/WEBP file",
            ));
        }
        self.packets.push(packet.clone());
        Ok(())
    }

    fn write_trailer(&mut self) -> CoreResult<()> {
        if self.trailer_written {
            return Ok(());
        }
        match self.packets.len() {
            0 => return Err(CoreError::invalid("webp muxer: no packets written")),
            1 => self.output.write_all(&self.packets[0].data)?,
            _ => {
                let merged = merge_packets(&self.packets, self.loop_count, self.background_rgba)?;
                self.output.write_all(&merged)?;
            }
        }
        self.output.flush()?;
        self.trailer_written = true;
        Ok(())
    }
}

/// `duration` in `tb` as a §2.7.1.1 `Frame Duration` (ms, 24-bit).
fn duration_to_ms(duration: i64, tb: TimeBase) -> u32 {
    let ms = tb.rescale_rnd(duration, TIME_BASE, Rounding::NearestAway);
    u32::try_from(ms.max(0))
        .unwrap_or(u32::MAX)
        .min(0x00FF_FFFF)
}

/// One parsed input packet of the merge.
struct Part<'a> {
    bytes: &'a [u8],
    container: WebpContainer,
    header: crate::ImageInfo,
}

/// Merge N complete `.webp` files (one per packet) into one animated
/// file — see the module docs.
fn merge_packets(
    packets: &[Packet],
    loop_count: u16,
    background_rgba: [u8; 4],
) -> CoreResult<Vec<u8>> {
    let mut parts = Vec::with_capacity(packets.len());
    for (i, p) in packets.iter().enumerate() {
        let container = parse_container(&p.data).map_err(|e| {
            CoreError::invalid(format!("webp muxer: packet {i} does not parse: {e}"))
        })?;
        let header = crate::info(&p.data)
            .map_err(|e| CoreError::invalid(format!("webp muxer: packet {i}: {e}")))?;
        parts.push(Part {
            bytes: &p.data,
            container,
            header,
        });
    }
    let first = &parts[0];
    // The canvas: the first packet's VP8X canvas, else its bitstream.
    let (canvas_w, canvas_h) = (first.header.width, first.header.height);
    let any_alpha = parts.iter().any(|p| p.header.has_alpha);

    // ANIM: the first packet's when it is an animation, else the options.
    let anim_payload = match first.container.first_chunk_with_fourcc(fourcc::ANIM) {
        Some(ch) => ch.payload(first.bytes).to_vec(),
        None => build_anim_payload(loop_count, background_rgba),
    };
    let meta_chunk = |tag: FourCc| {
        first
            .container
            .first_chunk_with_fourcc(tag)
            .map(|ch| raw_chunk(first.bytes, ch))
    };
    let (iccp, exif, xmp) = (
        meta_chunk(fourcc::ICCP),
        meta_chunk(fourcc::EXIF),
        meta_chunk(fourcc::XMP),
    );

    let flags = Vp8xFlags {
        has_iccp: iccp.is_some(),
        has_alpha: any_alpha,
        has_exif: exif.is_some(),
        has_xmp: xmp.is_some(),
        has_animation: true,
    };
    let mut body = Vec::new();
    body.extend_from_slice(
        &build_chunk(
            fourcc::VP8X,
            &build_vp8x_chunk(canvas_w, canvas_h, flags).map_err(invalid)?,
        )
        .map_err(invalid)?,
    );
    if let Some(ch) = iccp {
        body.extend_from_slice(ch);
    }
    body.extend_from_slice(&build_chunk(fourcc::ANIM, &anim_payload).map_err(invalid)?);

    for (i, (part, pkt)) in parts.iter().zip(packets).enumerate() {
        let duration = pkt.duration.map(|d| duration_to_ms(d, pkt.time_base));
        let anmfs: Vec<&WebpChunk> = part.container.chunks_with_fourcc(fourcc::ANMF).collect();
        if anmfs.is_empty() {
            // A still: one ANMF at the origin, overwrite, no disposal.
            let (w, h) = (part.header.width, part.header.height);
            if w > canvas_w || h > canvas_h {
                return Err(CoreError::invalid(format!(
                    "webp muxer: packet {i} is {w}x{h}, larger than the {canvas_w}x{canvas_h} canvas"
                )));
            }
            let mut frame_data = Vec::new();
            if let Some(alph) = part.container.first_chunk_with_fourcc(fourcc::ALPH) {
                frame_data.extend_from_slice(raw_chunk(part.bytes, alph));
            }
            let bitstream = part
                .container
                .first_chunk_with_fourcc(fourcc::VP8L)
                .or_else(|| part.container.first_chunk_with_fourcc(fourcc::VP8))
                .ok_or_else(|| {
                    CoreError::invalid(format!(
                        "webp muxer: packet {i} has no VP8L / VP8 bitstream chunk"
                    ))
                })?;
            frame_data.extend_from_slice(raw_chunk(part.bytes, bitstream));
            let payload = build_anmf_header_then_data(
                0,
                0,
                w,
                h,
                duration.unwrap_or(0),
                BlendingMethod::Overwrite,
                DisposalMethod::None,
                &frame_data,
            );
            body.extend_from_slice(&build_chunk(fourcc::ANMF, &payload).map_err(invalid)?);
            continue;
        }
        // An animation packet: its frames as they are, durations patched.
        for anmf in anmfs {
            let payload = anmf.payload(part.bytes);
            let hdr = AnmfHeader::parse(payload).map_err(w)?;
            if hdr.x + hdr.width > canvas_w || hdr.y + hdr.height > canvas_h {
                return Err(CoreError::invalid(format!(
                    "webp muxer: packet {i} frame {}x{} at ({}, {}) overflows the {canvas_w}x{canvas_h} canvas",
                    hdr.width, hdr.height, hdr.x, hdr.y
                )));
            }
            match duration {
                Some(ms) if ms != hdr.duration_ms => {
                    let mut patched = payload.to_vec();
                    patched[12..15].copy_from_slice(&ms.to_le_bytes()[..3]);
                    body.extend_from_slice(&build_chunk(fourcc::ANMF, &patched).map_err(invalid)?);
                }
                _ => body.extend_from_slice(raw_chunk(part.bytes, anmf)),
            }
        }
    }
    if let Some(ch) = exif {
        body.extend_from_slice(ch);
    }
    if let Some(ch) = xmp {
        body.extend_from_slice(ch);
    }
    Ok(crate::api::frame_riff(body)?)
}
