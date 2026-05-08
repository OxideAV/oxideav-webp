//! WebP container demuxer.
//!
//! WebP is a RIFF file whose top-level form-type is `WEBP`. Inside the
//! form we find one of three layouts:
//!
//! ```text
//! RIFF <size> WEBP VP8  <size> <VP8 keyframe bytes>       — simple lossy
//! RIFF <size> WEBP VP8L <size> <VP8L bytes>               — simple lossless
//! RIFF <size> WEBP VP8X <size> <flags+size hdr>
//!                 [ICCP|ANIM|ALPH|VP8 |VP8L|ANMF|EXIF|XMP ]* — extended
//! ```
//!
//! Chunks are even-padded the same way as other RIFF formats: if the payload
//! length is odd, one zero byte follows. All multi-byte integers are
//! little-endian.
//!
//! The demuxer emits each still frame (or each animation frame from an
//! `ANMF` chunk) as a single `Packet` on stream 0, with `codec_id = "webp"`
//! (a synthetic codec id local to this crate — the decoder handles all
//! three flavours transparently).

#[cfg(feature = "registry")]
use std::io::{Read, SeekFrom};

#[cfg(feature = "registry")]
use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, MediaType, Packet, PixelFormat, StreamInfo, TimeBase,
};
#[cfg(feature = "registry")]
use oxideav_core::{ContainerRegistry, Demuxer, ProbeData, ReadSeek};

use crate::error::{Result, WebpError as Error};

/// Codec id we attach to every packet emitted by this demuxer. The decoder
/// registered under the same id dispatches to the VP8, VP8L, or extended
/// path based on the chunk layout.
pub const WEBP_CODEC_ID: &str = "webp";

#[cfg(feature = "registry")]
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("webp", open);
    reg.register_extension("webp", "webp");
    reg.register_probe("webp", probe);
}

#[cfg(feature = "registry")]
fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() < 12 {
        return 0;
    }
    if &p.buf[0..4] != b"RIFF" {
        return 0;
    }
    if &p.buf[8..12] != b"WEBP" {
        return 0;
    }
    // `VeryHigh` — the RIFF magic + WEBP form-type is unambiguous.
    100
}

/// Public wrapper over `open` so the decoder-side convenience API can
/// instantiate a demuxer without duplicating the boxing dance.
#[cfg(feature = "registry")]
pub fn open_boxed(input: Box<dyn ReadSeek>) -> oxideav_core::Result<Box<dyn Demuxer>> {
    open(input, &oxideav_core::NullCodecResolver)
}

#[cfg(feature = "registry")]
fn open(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> oxideav_core::Result<Box<dyn Demuxer>> {
    // Read the whole file into memory. WebP stills are inherently small
    // (max 16384x16384 lossless / 16383x16383 VP8) and a full-buffer pass
    // simplifies chunk iteration + random access over the `ANMF` loop.
    let mut buf = Vec::new();
    input.seek(SeekFrom::Start(0))?;
    input.read_to_end(&mut buf)?;
    drop(input);

    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WEBP" {
        return Err(oxideav_core::Error::invalid("WebP: bad RIFF/WEBP magic"));
    }
    let riff_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    // `riff_size` excludes the "RIFF" FourCC + the 4-byte size field, so
    // the total file is 8 + riff_size bytes. Clamp to the actual buffer to
    // survive files whose size field lies (we'd rather keep decoding).
    let end = (8 + riff_size).min(buf.len());
    let body = &buf[12..end];

    let parsed = parse_webp_body(body)?;
    // Width/height default to the dimensions declared by the first image
    // chunk (VP8/VP8L) or the VP8X canvas, whichever we saw first.
    let (w, h) = parsed.canvas;

    let mut params = CodecParameters::video(CodecId::new(WEBP_CODEC_ID));
    params.media_type = MediaType::Video;
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Rgba);

    // Time base: milliseconds. Animation chunk durations are already in ms.
    let time_base = TimeBase::new(1, 1000);
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(parsed.total_duration_ms as i64),
        start_time: Some(0),
        params,
    };

    Ok(Box::new(WebpDemuxer {
        stream,
        packets: parsed.into_packets(time_base),
        pos: 0,
    }))
}

/// Result of parsing a `WEBP` form body.
#[derive(Debug)]
pub(crate) struct ParsedContainer {
    /// (width, height) — final rendered canvas size.
    pub canvas: (u32, u32),
    /// Frames in presentation order. Each frame has an already-extracted
    /// payload chunk (VP8 or VP8L) plus optional ALPH and offset/disposal
    /// info for animations. Static files produce exactly one entry.
    pub frames: Vec<ParsedFrame>,
    pub total_duration_ms: u32,
    /// Optional auxiliary metadata chunks (ICCP / EXIF / XMP). Empty for
    /// simple-layout files that omitted them; populated whenever the
    /// container carries the matching FourCC.
    pub metadata: WebpFileMetadata,
    /// Animated-WebP background colour as parsed from the `ANIM` chunk
    /// (RFC 9649 §2.5). Stored exactly as the spec lays it down — `[B,
    /// G, R, A]` byte order — so callers that want to surface the raw
    /// chunk see the original bytes. `None` for non-animated files (no
    /// `ANIM` chunk).
    ///
    /// The decoder converts these to RGBA when initialising the canvas
    /// and when filling dispose-to-background regions; see
    /// [`bgra_to_rgba`] for the conversion.
    pub anim_background_bgra: Option<[u8; 4]>,
    /// `loop_count` from the ANIM chunk (`0` = infinite). `None` for
    /// non-animated files.
    pub anim_loop_count: Option<u16>,
}

/// Convert a 4-byte ANIM-chunk background colour from the spec's `[B,
/// G, R, A]` byte order to row-major RGBA. The decoder uses this to
/// initialise the rendering canvas + fill dispose-to-background
/// regions per RFC 9649 §2.5. Centralised so both the standalone
/// [`crate::decode_webp`] path and the registry-side `WebpDecoder`
/// agree on the byte order.
pub(crate) fn bgra_to_rgba(bgra: [u8; 4]) -> [u8; 4] {
    [bgra[2], bgra[1], bgra[0], bgra[3]]
}

/// Auxiliary `.webp` file-level metadata chunks. Values are the raw
/// chunk payloads — ICC profile bytes, EXIF binary blob, XMP UTF-8
/// XML — exactly as written by the encoder. Per the WebP container
/// spec §3 these are file-global (not per-frame) and may appear in
/// any order after the `VP8X` header.
#[derive(Debug, Clone, Default)]
pub struct WebpFileMetadata {
    /// Raw `ICCP` chunk payload (an ICC colour profile). `None` when the
    /// file omitted the chunk.
    pub icc: Option<Vec<u8>>,
    /// Raw `EXIF` chunk payload. `None` when absent.
    pub exif: Option<Vec<u8>>,
    /// Raw `XMP ` chunk payload (UTF-8 XML, typically). `None` when absent.
    pub xmp: Option<Vec<u8>>,
}

impl WebpFileMetadata {
    /// True if any of the three metadata fields is populated. Useful for
    /// `decode_webp` callers that want to skip work when the file is
    /// metadata-free.
    pub fn any(&self) -> bool {
        self.icc.is_some() || self.exif.is_some() || self.xmp.is_some()
    }
}

/// Parse only the auxiliary metadata chunks (`ICCP`, `EXIF`, `XMP `)
/// out of a complete `.webp` file. Useful for callers that want to
/// inspect the colour profile or sidecar metadata without decoding any
/// pixels.
///
/// Returns `Ok(default)` (all fields `None`) for simple-layout files
/// that don't carry a `VP8X` header — those layouts can't carry
/// metadata by definition. Returns an error only for malformed
/// containers (bad RIFF header, truncated chunks); unknown auxiliary
/// chunks are skipped silently per the WebP container spec.
pub fn extract_metadata(buf: &[u8]) -> Result<WebpFileMetadata> {
    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WEBP" {
        return Err(Error::invalid("WebP: bad RIFF/WEBP magic"));
    }
    let riff_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let end = (8 + riff_size).min(buf.len());
    let body = &buf[12..end];
    let mut chunks = RiffChunks::new(body);
    // Peek the first chunk: only VP8X-extended layouts can carry metadata.
    let Some(first) = chunks.next().transpose()? else {
        return Ok(WebpFileMetadata::default());
    };
    if &first.id != b"VP8X" {
        // Simple lossy/lossless layout — no metadata possible.
        return Ok(WebpFileMetadata::default());
    }
    let mut meta = WebpFileMetadata::default();
    while let Some(c) = chunks.next().transpose()? {
        match &c.id {
            b"ICCP" => meta.icc = Some(c.data.to_vec()),
            b"EXIF" => meta.exif = Some(c.data.to_vec()),
            b"XMP " => meta.xmp = Some(c.data.to_vec()),
            _ => {}
        }
    }
    Ok(meta)
}

#[derive(Debug)]
pub(crate) struct ParsedFrame {
    /// Raw payload of the image chunk — VP8 keyframe or VP8L bitstream.
    pub image: ImagePayload,
    /// Optional ALPH chunk (extended-format alpha plane).
    pub alph: Option<AlphChunk>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub width: u32,
    pub height: u32,
    pub duration_ms: u32,
    /// True → dispose to background colour before rendering the next frame.
    pub dispose_to_background: bool,
    /// True → blend with the canvas (false = overwrite).
    pub blend_with_previous: bool,
}

#[derive(Debug)]
pub(crate) enum ImagePayload {
    Vp8(Vec<u8>),
    Vp8l(Vec<u8>),
}

#[derive(Debug)]
pub(crate) struct AlphChunk {
    pub pre_processing: u8,
    pub filtering: u8,
    pub compression: u8,
    pub data: Vec<u8>,
}

/// Memory-tight counterpart to [`ParsedContainer`] used by
/// [`crate::WebpAnimDecoder`]. Per-frame VP8/VP8L bitstreams + ALPH
/// payloads are recorded as `(start_offset, length)` ranges into the
/// buffer the decoder owns, instead of being copied into per-frame
/// `Vec<u8>`s up front. For a 1000-frame animation that's ~1000 saved
/// allocations + the file size's worth of memcpy traffic.
///
/// The decoder keeps the body buffer alive for the lifetime of the
/// `WebpAnimDecoder` (it copies the caller's bytes once at
/// construction), so the offsets stay valid until the decoder is
/// dropped. This is the WebP analogue to libwebp's `WebPAnimDecoder`
/// holding the file blob and resolving frame ranges lazily.
#[derive(Debug)]
pub(crate) struct LazyParsedContainer {
    pub canvas: (u32, u32),
    pub frames: Vec<LazyParsedFrame>,
    /// Sum of all `duration_ms` fields across `frames` — kept for
    /// parity with [`ParsedContainer`] so the lazy path can produce
    /// the same `StreamInfo::duration` whenever a future hookup wants
    /// it. Currently unused by the streaming `WebpAnimDecoder`, which
    /// surfaces duration on a per-frame basis.
    #[allow(dead_code)]
    pub total_duration_ms: u32,
    pub metadata: WebpFileMetadata,
    pub anim_background_bgra: Option<[u8; 4]>,
    pub anim_loop_count: Option<u16>,
}

/// Per-frame entry in [`LazyParsedContainer`]. `image` and `alph` carry
/// only the kind tag + a `(offset, length)` pair into the body buffer
/// — the actual VP8/VP8L/ALPH bytes stay in the buffer until the
/// decoder slices them on demand.
#[derive(Debug, Clone)]
pub(crate) struct LazyParsedFrame {
    pub image: LazyImageRef,
    pub alph: Option<LazyAlphRef>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub width: u32,
    pub height: u32,
    pub duration_ms: u32,
    pub dispose_to_background: bool,
    pub blend_with_previous: bool,
}

/// Image-payload reference into the owning buffer. Same semantics as
/// [`ImagePayload`] but byte-range instead of owned `Vec`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LazyImageRef {
    Vp8 { offset: usize, len: usize },
    Vp8l { offset: usize, len: usize },
}

/// ALPH-chunk reference into the owning buffer — the 3 header nibbles
/// (`pre_processing` / `filtering` / `compression`) are still parsed
/// eagerly because the VP8X parser uses them, but the (potentially
/// large) compressed alpha plane stays in the body buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LazyAlphRef {
    /// Pre-processing nibble of the ALPH header byte (RFC 9649 §3.4).
    /// Currently unused by the decoder (libwebp's encoder doesn't emit
    /// any pre-processing other than 0); kept for parity with
    /// [`AlphChunk`] so a future implementation can pick it up without
    /// re-walking the chunks.
    #[allow(dead_code)]
    pub pre_processing: u8,
    pub filtering: u8,
    pub compression: u8,
    pub data_offset: usize,
    pub data_len: usize,
}

#[cfg(feature = "registry")]
impl ParsedContainer {
    fn into_packets(self, tb: TimeBase) -> Vec<Packet> {
        let mut pkts = Vec::with_capacity(self.frames.len());
        let mut pts: i64 = 0;
        let canvas = self.canvas;
        let bg = self.anim_background_bgra;
        for (i, f) in self.frames.into_iter().enumerate() {
            let duration = f.duration_ms;
            let data = encode_frame_payload(&f, canvas, bg);
            let mut pkt = Packet::new(0, tb, data);
            pkt.pts = Some(pts);
            pkt.dts = Some(pts);
            pkt.duration = Some(duration.max(1) as i64);
            pkt.flags.keyframe = i == 0;
            pts += duration.max(1) as i64;
            pkts.push(pkt);
        }
        pkts
    }
}

/// Serialise a parsed frame into a self-contained payload the decoder can
/// consume without touching the original file. The layout is a tiny custom
/// TLV — a 32-byte header followed by the VP8/VP8L bitstream and an
/// optional ALPH payload, then the optional ANIM background colour.
///
/// This is local to the crate; it never escapes into `Packet::data`'s
/// public consumers because WebP packets only travel from `WebpDemuxer` to
/// `WebpDecoder` in the same process.
///
/// `anim_background_bgra` is the parsed ANIM chunk background (RFC 9649
/// §2.5), `[B, G, R, A]` byte order. `None` for non-animated files; the
/// decoder uses it for canvas init + dispose-to-background fills.
pub(crate) fn encode_frame_payload(
    f: &ParsedFrame,
    canvas: (u32, u32),
    anim_background_bgra: Option<[u8; 4]>,
) -> Vec<u8> {
    let img_bytes = match &f.image {
        ImagePayload::Vp8(v) | ImagePayload::Vp8l(v) => v,
    };
    let mut out = Vec::with_capacity(
        64 + img_bytes.len() + f.alph.as_ref().map(|a| a.data.len() + 16).unwrap_or(0) + 8,
    );
    // Magic "OWEB" + version byte. v2 adds the trailing
    // ANIM-background block (1 flag byte + 4 BGRA bytes when present);
    // a v1 reader on a v2 payload would error out — same in-process
    // crate so version bumps are coordinated.
    out.extend_from_slice(b"OWEB");
    out.push(2);
    // Flags: bit0 = has_alph, bit1 = is_vp8l, bit2 = dispose,
    // bit3 = blend_with_previous, bit4 = anim_bg_present (v2+).
    let mut flags = 0u8;
    if f.alph.is_some() {
        flags |= 0x01;
    }
    if matches!(f.image, ImagePayload::Vp8l(_)) {
        flags |= 0x02;
    }
    if f.dispose_to_background {
        flags |= 0x04;
    }
    if f.blend_with_previous {
        flags |= 0x08;
    }
    if anim_background_bgra.is_some() {
        flags |= 0x10;
    }
    out.push(flags);
    // Canvas + frame bbox, 6 x u32.
    for v in [
        canvas.0, canvas.1, f.x_offset, f.y_offset, f.width, f.height,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    // Duration.
    out.extend_from_slice(&f.duration_ms.to_le_bytes());
    // Image chunk length + data.
    out.extend_from_slice(&(img_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(img_bytes);
    // ALPH chunk (optional): pre/filter/comp bytes + length + data.
    if let Some(a) = &f.alph {
        out.push(a.pre_processing);
        out.push(a.filtering);
        out.push(a.compression);
        out.extend_from_slice(&(a.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&a.data);
    }
    // v2: optional ANIM background colour at the very end (4 bytes
    // `[B, G, R, A]`). Gated on the `anim_bg_present` flag bit so
    // non-animated files don't pay the trailing 4 bytes.
    if let Some(bgra) = anim_background_bgra {
        out.extend_from_slice(&bgra);
    }
    out
}

/// Counterpart to `encode_frame_payload`. Used by the decoder.
pub(crate) fn decode_frame_payload(buf: &[u8]) -> Result<DecodedPayload<'_>> {
    if buf.len() < 4 + 1 + 1 + 6 * 4 + 4 + 4 {
        return Err(Error::invalid("WebP: frame payload too short"));
    }
    if &buf[0..4] != b"OWEB" {
        return Err(Error::invalid("WebP: bad frame payload magic"));
    }
    let version = buf[4];
    if version != 1 && version != 2 {
        return Err(Error::invalid("WebP: unknown frame payload version"));
    }
    let flags = buf[5];
    let mut p = 6usize;
    let read_u32 = |p: &mut usize, buf: &[u8]| -> u32 {
        let v = u32::from_le_bytes([buf[*p], buf[*p + 1], buf[*p + 2], buf[*p + 3]]);
        *p += 4;
        v
    };
    let canvas_w = read_u32(&mut p, buf);
    let canvas_h = read_u32(&mut p, buf);
    let x_off = read_u32(&mut p, buf);
    let y_off = read_u32(&mut p, buf);
    let frame_w = read_u32(&mut p, buf);
    let frame_h = read_u32(&mut p, buf);
    let duration_ms = read_u32(&mut p, buf);
    let img_len = read_u32(&mut p, buf) as usize;
    if p + img_len > buf.len() {
        return Err(Error::invalid("WebP: image chunk extends past payload"));
    }
    let image = &buf[p..p + img_len];
    p += img_len;
    let alph = if flags & 0x01 != 0 {
        if p + 3 + 4 > buf.len() {
            return Err(Error::invalid("WebP: truncated ALPH header"));
        }
        let pre = buf[p];
        let filt = buf[p + 1];
        let comp = buf[p + 2];
        p += 3;
        let alen = read_u32(&mut p, buf) as usize;
        if p + alen > buf.len() {
            return Err(Error::invalid("WebP: ALPH data extends past payload"));
        }
        let a = &buf[p..p + alen];
        p += alen;
        Some(DecodedAlph {
            pre_processing: pre,
            filtering: filt,
            compression: comp,
            data: a,
        })
    } else {
        None
    };
    // v2 trailing ANIM-background block. Bit 4 of `flags` gates it; v1
    // payloads always have the bit clear.
    let anim_background_bgra = if version >= 2 && (flags & 0x10) != 0 {
        if p + 4 > buf.len() {
            return Err(Error::invalid("WebP: truncated ANIM bg in payload"));
        }
        Some([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]])
    } else {
        None
    };
    Ok(DecodedPayload {
        is_vp8l: flags & 0x02 != 0,
        dispose_to_background: flags & 0x04 != 0,
        blend_with_previous: flags & 0x08 != 0,
        canvas: (canvas_w, canvas_h),
        x_offset: x_off,
        y_offset: y_off,
        width: frame_w,
        height: frame_h,
        duration_ms,
        image,
        alph,
        anim_background_bgra,
    })
}

pub(crate) struct DecodedPayload<'a> {
    pub is_vp8l: bool,
    pub dispose_to_background: bool,
    pub blend_with_previous: bool,
    pub canvas: (u32, u32),
    pub x_offset: u32,
    pub y_offset: u32,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub duration_ms: u32,
    pub image: &'a [u8],
    pub alph: Option<DecodedAlph<'a>>,
    /// ANIM background colour as parsed from the file's `ANIM` chunk
    /// (`[B, G, R, A]` byte order; `None` for non-animated files).
    /// The decoder converts this to RGBA when initialising the canvas
    /// + filling dispose-to-background regions.
    pub anim_background_bgra: Option<[u8; 4]>,
}

pub(crate) struct DecodedAlph<'a> {
    #[allow(dead_code)]
    pub pre_processing: u8,
    pub filtering: u8,
    pub compression: u8,
    pub data: &'a [u8],
}

pub(crate) fn parse_webp_body(body: &[u8]) -> Result<ParsedContainer> {
    let mut chunks = RiffChunks::new(body);
    // Peek the first chunk to distinguish simple vs extended layout.
    let first = chunks
        .next()
        .transpose()?
        .ok_or_else(|| Error::invalid("WebP: empty RIFF body"))?;

    match &first.id {
        b"VP8 " => {
            // Simple lossy still.
            let (w, h) = parse_vp8_keyframe_dims(first.data)?;
            let frame = ParsedFrame {
                image: ImagePayload::Vp8(first.data.to_vec()),
                alph: None,
                x_offset: 0,
                y_offset: 0,
                width: w,
                height: h,
                duration_ms: 0,
                dispose_to_background: false,
                blend_with_previous: false,
            };
            Ok(ParsedContainer {
                canvas: (w, h),
                frames: vec![frame],
                total_duration_ms: 0,
                metadata: WebpFileMetadata::default(),
                anim_background_bgra: None,
                anim_loop_count: None,
            })
        }
        b"VP8L" => {
            let (w, h) = parse_vp8l_dims(first.data)?;
            let frame = ParsedFrame {
                image: ImagePayload::Vp8l(first.data.to_vec()),
                alph: None,
                x_offset: 0,
                y_offset: 0,
                width: w,
                height: h,
                duration_ms: 0,
                dispose_to_background: false,
                blend_with_previous: false,
            };
            Ok(ParsedContainer {
                canvas: (w, h),
                frames: vec![frame],
                total_duration_ms: 0,
                metadata: WebpFileMetadata::default(),
                anim_background_bgra: None,
                anim_loop_count: None,
            })
        }
        b"VP8X" => parse_extended(first.data, &mut chunks),
        other => Err(Error::invalid(format!(
            "WebP: unexpected first chunk {:?}",
            std::str::from_utf8(other).unwrap_or("???")
        ))),
    }
}

fn parse_extended(vp8x: &[u8], chunks: &mut RiffChunks<'_>) -> Result<ParsedContainer> {
    if vp8x.len() < 10 {
        return Err(Error::invalid("WebP: VP8X chunk too short"));
    }
    // VP8X layout: 1 byte flags, 3 bytes reserved, 3 bytes canvas_w-1, 3 bytes canvas_h-1.
    let flags = vp8x[0];
    let has_anim = flags & 0x02 != 0;
    let canvas_w = (u32::from_le_bytes([vp8x[4], vp8x[5], vp8x[6], 0]) & 0x00FF_FFFF) + 1;
    let canvas_h = (u32::from_le_bytes([vp8x[7], vp8x[8], vp8x[9], 0]) & 0x00FF_FFFF) + 1;

    let mut frames: Vec<ParsedFrame> = Vec::new();
    // Static extended WebP state — we accumulate the VP8/VP8L chunk and
    // optional ALPH, and emit one frame when we've seen an image.
    let mut pending_alph: Option<AlphChunk> = None;
    let mut pending_image: Option<ImagePayload> = None;
    let mut metadata = WebpFileMetadata::default();
    // ANIM chunk state (RFC 9649 §2.5). Only meaningful when the VP8X
    // ANIM flag is set; we still capture it on flag-mismatch files
    // (the spec says "MUST be ignored" on flag-clear, but a strict
    // ignore would also drop a benign chunk on a malformed file —
    // gating on `has_anim` below preserves the spec's "ignore" while
    // still reading the chunk so a future loose-mode could surface it).
    let mut anim_background_bgra: Option<[u8; 4]> = None;
    let mut anim_loop_count: Option<u16> = None;

    let mut total_duration = 0u32;

    while let Some(c) = chunks.next().transpose()? {
        match &c.id {
            b"VP8 " => {
                pending_image = Some(ImagePayload::Vp8(c.data.to_vec()));
            }
            b"VP8L" => {
                pending_image = Some(ImagePayload::Vp8l(c.data.to_vec()));
            }
            b"ALPH" => {
                if c.data.is_empty() {
                    return Err(Error::invalid("WebP: ALPH chunk empty"));
                }
                let hdr = c.data[0];
                let pre = (hdr >> 4) & 0x3;
                let filt = (hdr >> 2) & 0x3;
                let comp = hdr & 0x3;
                pending_alph = Some(AlphChunk {
                    pre_processing: pre,
                    filtering: filt,
                    compression: comp,
                    data: c.data[1..].to_vec(),
                });
            }
            b"ANMF" => {
                let anmf = parse_anmf(c.data)?;
                let f = anmf.into_frame();
                total_duration = total_duration.saturating_add(f.duration_ms);
                frames.push(f);
            }
            // Auxiliary metadata chunks — captured into `metadata` so
            // callers can read them off `WebpImage::metadata` after
            // decode. Unknown auxiliary FourCCs (and `ANIM` whose flags
            // are already represented in the per-frame state) are
            // skipped silently per the WebP container spec.
            b"ICCP" => metadata.icc = Some(c.data.to_vec()),
            b"EXIF" => metadata.exif = Some(c.data.to_vec()),
            b"XMP " => metadata.xmp = Some(c.data.to_vec()),
            // ANIM chunk: 4 bytes [B, G, R, A] background colour +
            // 2 bytes little-endian loop count (RFC 9649 §2.5,
            // "ANIM Chunk"). Per spec: "This chunk MUST appear if
            // the Animation flag in the VP8X Chunk is set. If the
            // Animation flag is not set and this chunk is present,
            // it MUST be ignored." We capture it unconditionally
            // and gate exposure on `has_anim` after the loop so a
            // file with the flag clear still parses cleanly.
            b"ANIM" if c.data.len() >= 6 => {
                anim_background_bgra = Some([c.data[0], c.data[1], c.data[2], c.data[3]]);
                anim_loop_count = Some(u16::from_le_bytes([c.data[4], c.data[5]]));
            }
            b"ANIM" => {
                // Truncated ANIM payload — leave fields at None.
            }
            _ => {
                // Unknown chunk — skip silently per the spec.
            }
        }
    }
    // Per RFC 9649 §2.5, the ANIM chunk is meaningful only when the
    // VP8X ANIM flag is set. Drop it when the flag is clear so callers
    // see `None` — matches "MUST be ignored" wording.
    if !has_anim {
        anim_background_bgra = None;
        anim_loop_count = None;
    }

    if !has_anim {
        let image = pending_image
            .ok_or_else(|| Error::invalid("WebP: extended file has no image chunk"))?;
        let (w, h) = match &image {
            ImagePayload::Vp8(v) => parse_vp8_keyframe_dims(v).unwrap_or((canvas_w, canvas_h)),
            ImagePayload::Vp8l(v) => parse_vp8l_dims(v).unwrap_or((canvas_w, canvas_h)),
        };
        let frame = ParsedFrame {
            image,
            alph: pending_alph.take(),
            x_offset: 0,
            y_offset: 0,
            width: w,
            height: h,
            duration_ms: 0,
            dispose_to_background: false,
            blend_with_previous: false,
        };
        frames.push(frame);
    }

    Ok(ParsedContainer {
        canvas: (canvas_w, canvas_h),
        frames,
        total_duration_ms: total_duration,
        metadata,
        anim_background_bgra,
        anim_loop_count,
    })
}

struct AnmfBundle {
    x_offset: u32,
    y_offset: u32,
    width: u32,
    height: u32,
    duration_ms: u32,
    dispose_to_background: bool,
    blend_with_previous: bool,
    image: ImagePayload,
    alph: Option<AlphChunk>,
}

impl AnmfBundle {
    fn into_frame(self) -> ParsedFrame {
        ParsedFrame {
            image: self.image,
            alph: self.alph,
            x_offset: self.x_offset,
            y_offset: self.y_offset,
            width: self.width,
            height: self.height,
            duration_ms: self.duration_ms,
            dispose_to_background: self.dispose_to_background,
            blend_with_previous: self.blend_with_previous,
        }
    }
}

fn parse_anmf(data: &[u8]) -> Result<AnmfBundle> {
    // ANMF: 3 bytes X/2, 3 bytes Y/2, 3 bytes w-1, 3 bytes h-1, 3 bytes duration,
    //       1 byte flags (bit0 = blending=overwrite, bit1 = dispose-to-bg).
    //       Then nested sub-chunks (ALPH? + VP8/VP8L).
    if data.len() < 16 {
        return Err(Error::invalid("WebP: ANMF header too short"));
    }
    let x_off = u32::from_le_bytes([data[0], data[1], data[2], 0]) & 0x00FF_FFFF;
    let y_off = u32::from_le_bytes([data[3], data[4], data[5], 0]) & 0x00FF_FFFF;
    let w = (u32::from_le_bytes([data[6], data[7], data[8], 0]) & 0x00FF_FFFF) + 1;
    let h = (u32::from_le_bytes([data[9], data[10], data[11], 0]) & 0x00FF_FFFF) + 1;
    let dur = u32::from_le_bytes([data[12], data[13], data[14], 0]) & 0x00FF_FFFF;
    let flags = data[15];
    // Spec (WebP container, ANMF flags byte):
    //   bit 0 = blending_method (0 = alpha-blend with canvas,
    //                            1 = no blend / overwrite)
    //   bit 1 = disposal_method (0 = do nothing,
    //                            1 = dispose to background after rendering)
    // The flag we keep internally is the *positive* sense — true means
    // "blend onto the canvas using the source alpha". So invert bit 0.
    let blend_with_previous = flags & 0x01 == 0;
    let dispose_to_background = flags & 0x02 != 0;

    let mut chunks = RiffChunks::new(&data[16..]);
    let mut image: Option<ImagePayload> = None;
    let mut alph: Option<AlphChunk> = None;
    while let Some(c) = chunks.next().transpose()? {
        match &c.id {
            b"VP8 " => image = Some(ImagePayload::Vp8(c.data.to_vec())),
            b"VP8L" => image = Some(ImagePayload::Vp8l(c.data.to_vec())),
            b"ALPH" if !c.data.is_empty() => {
                let hdr = c.data[0];
                alph = Some(AlphChunk {
                    pre_processing: (hdr >> 4) & 0x3,
                    filtering: (hdr >> 2) & 0x3,
                    compression: hdr & 0x3,
                    data: c.data[1..].to_vec(),
                });
            }
            _ => {}
        }
    }
    let image = image.ok_or_else(|| Error::invalid("WebP: ANMF has no image chunk"))?;
    Ok(AnmfBundle {
        x_offset: x_off * 2, // spec: multiples of 2 → stored /2
        y_offset: y_off * 2,
        width: w,
        height: h,
        duration_ms: dur,
        dispose_to_background,
        blend_with_previous,
        image,
        alph,
    })
}

fn parse_vp8_keyframe_dims(vp8: &[u8]) -> Result<(u32, u32)> {
    // Bare VP8 keyframe tag: 3 byte frame tag + 3 byte start code + 4 byte hdr.
    if vp8.len() < 10 {
        return Err(Error::invalid("WebP: VP8 chunk too short"));
    }
    if vp8[3] != 0x9d || vp8[4] != 0x01 || vp8[5] != 0x2a {
        return Err(Error::invalid("WebP: missing VP8 keyframe start code"));
    }
    let w = u16::from_le_bytes([vp8[6], vp8[7]]) as u32 & 0x3FFF;
    let h = u16::from_le_bytes([vp8[8], vp8[9]]) as u32 & 0x3FFF;
    Ok((w, h))
}

fn parse_vp8l_dims(vp8l: &[u8]) -> Result<(u32, u32)> {
    // VP8L: signature byte 0x2f then 14 bit width-1, 14 bit height-1, ...
    if vp8l.len() < 5 {
        return Err(Error::invalid("WebP: VP8L chunk too short"));
    }
    if vp8l[0] != 0x2f {
        return Err(Error::invalid("WebP: bad VP8L signature"));
    }
    let bits = u32::from_le_bytes([vp8l[1], vp8l[2], vp8l[3], vp8l[4]]);
    let w = (bits & 0x3FFF) + 1;
    let h = ((bits >> 14) & 0x3FFF) + 1;
    Ok((w, h))
}

/// Memory-tight counterpart to [`parse_webp_body`]. Walks the RIFF
/// chunk tree once and returns a [`LazyParsedContainer`] whose frames
/// reference VP8/VP8L/ALPH bytes via `(offset, len)` ranges into
/// `body` instead of cloning each chunk into an owned `Vec<u8>`.
///
/// Returned offsets are relative to `body` (i.e. the slice the caller
/// passed in), not the full RIFF file. Consumers that own the file
/// buffer (e.g. [`crate::WebpAnimDecoder`]) need to remember the
/// `body` start offset (it's always the byte after the 12-byte
/// `RIFF/<size>/WEBP` preamble) when slicing.
///
/// This function is lossless w.r.t. [`parse_webp_body`] — the same set
/// of frames in the same order, the same metadata, the same ANIM bg /
/// loop-count, the same total duration. Only the per-frame storage
/// shape changes from owned to borrowed-via-offset.
pub(crate) fn parse_webp_body_lazy(body: &[u8]) -> Result<LazyParsedContainer> {
    let mut chunks = RiffChunks::new(body);
    let first = chunks
        .next()
        .transpose()?
        .ok_or_else(|| Error::invalid("WebP: empty RIFF body"))?;

    match &first.id {
        b"VP8 " => {
            let (w, h) = parse_vp8_keyframe_dims(first.data)?;
            let frame = LazyParsedFrame {
                image: LazyImageRef::Vp8 {
                    offset: first.payload_offset,
                    len: first.data.len(),
                },
                alph: None,
                x_offset: 0,
                y_offset: 0,
                width: w,
                height: h,
                duration_ms: 0,
                dispose_to_background: false,
                blend_with_previous: false,
            };
            Ok(LazyParsedContainer {
                canvas: (w, h),
                frames: vec![frame],
                total_duration_ms: 0,
                metadata: WebpFileMetadata::default(),
                anim_background_bgra: None,
                anim_loop_count: None,
            })
        }
        b"VP8L" => {
            let (w, h) = parse_vp8l_dims(first.data)?;
            let frame = LazyParsedFrame {
                image: LazyImageRef::Vp8l {
                    offset: first.payload_offset,
                    len: first.data.len(),
                },
                alph: None,
                x_offset: 0,
                y_offset: 0,
                width: w,
                height: h,
                duration_ms: 0,
                dispose_to_background: false,
                blend_with_previous: false,
            };
            Ok(LazyParsedContainer {
                canvas: (w, h),
                frames: vec![frame],
                total_duration_ms: 0,
                metadata: WebpFileMetadata::default(),
                anim_background_bgra: None,
                anim_loop_count: None,
            })
        }
        b"VP8X" => parse_extended_lazy(first.data, &mut chunks),
        other => Err(Error::invalid(format!(
            "WebP: unexpected first chunk {:?}",
            std::str::from_utf8(other).unwrap_or("???")
        ))),
    }
}

fn parse_extended_lazy(vp8x: &[u8], chunks: &mut RiffChunks<'_>) -> Result<LazyParsedContainer> {
    if vp8x.len() < 10 {
        return Err(Error::invalid("WebP: VP8X chunk too short"));
    }
    let flags = vp8x[0];
    let has_anim = flags & 0x02 != 0;
    let canvas_w = (u32::from_le_bytes([vp8x[4], vp8x[5], vp8x[6], 0]) & 0x00FF_FFFF) + 1;
    let canvas_h = (u32::from_le_bytes([vp8x[7], vp8x[8], vp8x[9], 0]) & 0x00FF_FFFF) + 1;

    let mut frames: Vec<LazyParsedFrame> = Vec::new();
    let mut pending_alph: Option<LazyAlphRef> = None;
    let mut pending_image: Option<LazyImageRef> = None;
    let mut metadata = WebpFileMetadata::default();
    let mut anim_background_bgra: Option<[u8; 4]> = None;
    let mut anim_loop_count: Option<u16> = None;
    let mut total_duration = 0u32;

    while let Some(c) = chunks.next().transpose()? {
        match &c.id {
            b"VP8 " => {
                pending_image = Some(LazyImageRef::Vp8 {
                    offset: c.payload_offset,
                    len: c.data.len(),
                });
            }
            b"VP8L" => {
                pending_image = Some(LazyImageRef::Vp8l {
                    offset: c.payload_offset,
                    len: c.data.len(),
                });
            }
            b"ALPH" => {
                if c.data.is_empty() {
                    return Err(Error::invalid("WebP: ALPH chunk empty"));
                }
                let hdr = c.data[0];
                pending_alph = Some(LazyAlphRef {
                    pre_processing: (hdr >> 4) & 0x3,
                    filtering: (hdr >> 2) & 0x3,
                    compression: hdr & 0x3,
                    // Skip the 1-byte ALPH header — `data_offset`
                    // points at the compressed alpha plane proper.
                    data_offset: c.payload_offset + 1,
                    data_len: c.data.len() - 1,
                });
            }
            b"ANMF" => {
                let f = parse_anmf_lazy(c.data, c.payload_offset)?;
                total_duration = total_duration.saturating_add(f.duration_ms);
                frames.push(f);
            }
            // The auxiliary metadata chunks are typically small (a few
            // hundred bytes for ICCP, ~tens of KB for EXIF/XMP); we
            // continue cloning them eagerly because consumers of
            // `WebpAnimInfo` expect owned `Vec<u8>` and the savings
            // wouldn't change the asymptotic memory shape (animations
            // dwarf metadata).
            b"ICCP" => metadata.icc = Some(c.data.to_vec()),
            b"EXIF" => metadata.exif = Some(c.data.to_vec()),
            b"XMP " => metadata.xmp = Some(c.data.to_vec()),
            b"ANIM" if c.data.len() >= 6 => {
                anim_background_bgra = Some([c.data[0], c.data[1], c.data[2], c.data[3]]);
                anim_loop_count = Some(u16::from_le_bytes([c.data[4], c.data[5]]));
            }
            b"ANIM" => {}
            _ => {}
        }
    }
    if !has_anim {
        anim_background_bgra = None;
        anim_loop_count = None;
    }
    if !has_anim {
        let image = pending_image
            .ok_or_else(|| Error::invalid("WebP: extended file has no image chunk"))?;
        // Compute width/height from the image bitstream, falling back
        // to the canvas dims if the parse fails (matches the eager
        // path's tolerance).
        let body_full_view = chunks.body;
        let (w, h) = match image {
            LazyImageRef::Vp8 { offset, len } => {
                parse_vp8_keyframe_dims(&body_full_view[offset..offset + len])
                    .unwrap_or((canvas_w, canvas_h))
            }
            LazyImageRef::Vp8l { offset, len } => {
                parse_vp8l_dims(&body_full_view[offset..offset + len])
                    .unwrap_or((canvas_w, canvas_h))
            }
        };
        frames.push(LazyParsedFrame {
            image,
            alph: pending_alph.take(),
            x_offset: 0,
            y_offset: 0,
            width: w,
            height: h,
            duration_ms: 0,
            dispose_to_background: false,
            blend_with_previous: false,
        });
    }
    Ok(LazyParsedContainer {
        canvas: (canvas_w, canvas_h),
        frames,
        total_duration_ms: total_duration,
        metadata,
        anim_background_bgra,
        anim_loop_count,
    })
}

fn parse_anmf_lazy(data: &[u8], anmf_payload_offset: usize) -> Result<LazyParsedFrame> {
    if data.len() < 16 {
        return Err(Error::invalid("WebP: ANMF header too short"));
    }
    let x_off = u32::from_le_bytes([data[0], data[1], data[2], 0]) & 0x00FF_FFFF;
    let y_off = u32::from_le_bytes([data[3], data[4], data[5], 0]) & 0x00FF_FFFF;
    let w = (u32::from_le_bytes([data[6], data[7], data[8], 0]) & 0x00FF_FFFF) + 1;
    let h = (u32::from_le_bytes([data[9], data[10], data[11], 0]) & 0x00FF_FFFF) + 1;
    let dur = u32::from_le_bytes([data[12], data[13], data[14], 0]) & 0x00FF_FFFF;
    let flags = data[15];
    let blend_with_previous = flags & 0x01 == 0;
    let dispose_to_background = flags & 0x02 != 0;

    // The nested-chunk iterator walks `data[16..]`; offsets it produces
    // are relative to that sub-slice. We add `anmf_payload_offset + 16`
    // to translate them back to body-buffer offsets.
    let mut chunks = RiffChunks::new(&data[16..]);
    let mut image: Option<LazyImageRef> = None;
    let mut alph: Option<LazyAlphRef> = None;
    while let Some(c) = chunks.next().transpose()? {
        let abs_offset = anmf_payload_offset + 16 + c.payload_offset;
        match &c.id {
            b"VP8 " => {
                image = Some(LazyImageRef::Vp8 {
                    offset: abs_offset,
                    len: c.data.len(),
                });
            }
            b"VP8L" => {
                image = Some(LazyImageRef::Vp8l {
                    offset: abs_offset,
                    len: c.data.len(),
                });
            }
            b"ALPH" if !c.data.is_empty() => {
                let hdr = c.data[0];
                alph = Some(LazyAlphRef {
                    pre_processing: (hdr >> 4) & 0x3,
                    filtering: (hdr >> 2) & 0x3,
                    compression: hdr & 0x3,
                    data_offset: abs_offset + 1,
                    data_len: c.data.len() - 1,
                });
            }
            _ => {}
        }
    }
    let image = image.ok_or_else(|| Error::invalid("WebP: ANMF has no image chunk"))?;
    Ok(LazyParsedFrame {
        image,
        alph,
        x_offset: x_off * 2,
        y_offset: y_off * 2,
        width: w,
        height: h,
        duration_ms: dur,
        dispose_to_background,
        blend_with_previous,
    })
}

/// Iterator over RIFF chunks inside a body. Borrows the body slice.
struct RiffChunks<'a> {
    body: &'a [u8],
    pos: usize,
}

impl<'a> RiffChunks<'a> {
    fn new(body: &'a [u8]) -> Self {
        Self { body, pos: 0 }
    }
}

struct ChunkRef<'a> {
    id: [u8; 4],
    data: &'a [u8],
    /// Offset of `data[0]` inside the iterator's `body` slice. Used by
    /// the lazy demuxer to record `(offset, len)` ranges into the
    /// owning buffer instead of cloning chunks.
    payload_offset: usize,
}

impl<'a> Iterator for RiffChunks<'a> {
    type Item = Result<ChunkRef<'a>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 8 > self.body.len() {
            // Dangling trailing bytes <8 bytes long — treat as clean EOF
            // to survive tolerant muxers.
            return None;
        }
        let id = [
            self.body[self.pos],
            self.body[self.pos + 1],
            self.body[self.pos + 2],
            self.body[self.pos + 3],
        ];
        let size = u32::from_le_bytes([
            self.body[self.pos + 4],
            self.body[self.pos + 5],
            self.body[self.pos + 6],
            self.body[self.pos + 7],
        ]) as usize;
        let payload_start = self.pos + 8;
        let payload_end = payload_start.saturating_add(size);
        if payload_end > self.body.len() {
            return Some(Err(Error::invalid("WebP: chunk extends past RIFF body")));
        }
        let data = &self.body[payload_start..payload_end];
        let payload_offset = payload_start;
        let padded = (size + (size & 1)).min(self.body.len().saturating_sub(payload_start));
        self.pos = payload_start + padded;
        Some(Ok(ChunkRef {
            id,
            data,
            payload_offset,
        }))
    }
}

#[cfg(feature = "registry")]
struct WebpDemuxer {
    stream: StreamInfo,
    packets: Vec<Packet>,
    pos: usize,
}

#[cfg(feature = "registry")]
impl Demuxer for WebpDemuxer {
    fn format_name(&self) -> &str {
        "webp"
    }

    fn streams(&self) -> &[StreamInfo] {
        std::slice::from_ref(&self.stream)
    }

    fn next_packet(&mut self) -> oxideav_core::Result<Packet> {
        if self.pos >= self.packets.len() {
            return Err(oxideav_core::Error::Eof);
        }
        let pkt = self.packets[self.pos].clone();
        self.pos += 1;
        Ok(pkt)
    }

    fn duration_micros(&self) -> Option<i64> {
        self.stream.duration.map(|d| d * 1000)
    }
}

#[cfg(all(test, feature = "registry"))]
mod tests {
    use super::*;

    #[test]
    fn probe_recognises_webp() {
        let mut buf = vec![0u8; 16];
        buf[..4].copy_from_slice(b"RIFF");
        buf[8..12].copy_from_slice(b"WEBP");
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(probe(&p), 100);
    }

    #[test]
    fn probe_rejects_non_webp_riff() {
        let mut buf = vec![0u8; 16];
        buf[..4].copy_from_slice(b"RIFF");
        buf[8..12].copy_from_slice(b"AVI ");
        let p = ProbeData {
            buf: &buf,
            ext: None,
        };
        assert_eq!(probe(&p), 0);
    }
}
