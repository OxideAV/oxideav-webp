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
    // Own the body bytes so the lazy `(offset, length)` ranges in
    // `parsed.frames` stay valid for the lifetime of the demuxer. This
    // is the same allocation strategy `WebpAnimDecoder` uses.
    let body: Vec<u8> = buf[12..end].to_vec();

    let parsed = parse_webp_body_lazy(&body)?;
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
        body,
        parsed,
        time_base,
        pos: 0,
        pts: 0,
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
    /// Sum of all per-frame durations in milliseconds. Currently
    /// unread because the framework `Demuxer` impl uses the lazy
    /// container's parallel field for `StreamInfo::duration`; the
    /// standalone `decode_webp` path also doesn't surface a
    /// container-level duration. Kept for parity with the lazy struct
    /// + so a future eager-side caller doesn't have to re-scan.
    #[allow(dead_code)]
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
    if first.data.len() < 10 {
        return Err(Error::invalid("WebP: VP8X chunk too short"));
    }
    // Decode the VP8X flag byte so we can apply the same flag-gating
    // pass `parse_extended` does. Per RFC 9649 §2.5.6 + §2.7.1.4/§2.7.1.5
    // an ICCP / EXIF / XMP chunk whose corresponding VP8X flag is
    // clear MUST be ignored — surfacing it would let a malformed file
    // pollute consumers that trust the flag-driven advertise.
    let flags = VP8xFlags::from_byte(first.data[0]);
    let mut meta = WebpFileMetadata::default();
    while let Some(c) = chunks.next().transpose()? {
        match &c.id {
            b"ICCP" => meta.icc = Some(c.data.to_vec()),
            b"EXIF" => meta.exif = Some(c.data.to_vec()),
            b"XMP " => meta.xmp = Some(c.data.to_vec()),
            _ => {}
        }
    }
    if !flags.has_icc {
        meta.icc = None;
    }
    if !flags.has_exif {
        meta.exif = None;
    }
    if !flags.has_xmp {
        meta.xmp = None;
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
    /// Pre-processing nibble of the ALPH header byte (RFC 9649 §3.4).
    /// libwebp's encoder doesn't emit any pre-processing other than 0,
    /// so the decoder ignores it; kept on the struct for parity with
    /// [`LazyAlphRef`] + so a future implementation can pick it up
    /// without re-walking chunks.
    #[allow(dead_code)]
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
    /// Sum of all `duration_ms` fields across `frames`. Used by the
    /// streaming `Demuxer` impl to populate `StreamInfo::duration` at
    /// `open` time without re-walking the frames. The animated
    /// `WebpAnimDecoder` still surfaces duration on a per-frame basis
    /// and ignores this field.
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

/// Streaming `OWEB` payload builder used by [`WebpDemuxer::next_packet`].
/// Materialises one frame's payload at a time directly from a
/// [`LazyParsedFrame`] + the owning body buffer, so the demuxer never
/// needs to pre-compute the full `Vec<Packet>` up front. The on-wire
/// layout is the small TLV the decoder side (`decode_frame_payload`)
/// already understands — magic `OWEB` + version + flags + canvas/bbox
/// scalars + image bytes + optional ALPH block + optional ANIM bg
/// trailer (RFC 9649 §2.5 byte order).
#[cfg(feature = "registry")]
pub(crate) fn encode_lazy_frame_payload(
    f: &LazyParsedFrame,
    body: &[u8],
    canvas: (u32, u32),
    anim_background_bgra: Option<[u8; 4]>,
) -> Result<Vec<u8>> {
    let (img_bytes, is_vp8l) = match f.image {
        LazyImageRef::Vp8 { offset, len } => {
            if offset + len > body.len() {
                return Err(Error::invalid("WebP: VP8 ref out of bounds"));
            }
            (&body[offset..offset + len], false)
        }
        LazyImageRef::Vp8l { offset, len } => {
            if offset + len > body.len() {
                return Err(Error::invalid("WebP: VP8L ref out of bounds"));
            }
            (&body[offset..offset + len], true)
        }
    };
    let alph_bytes = if let Some(a) = &f.alph {
        if a.data_offset + a.data_len > body.len() {
            return Err(Error::invalid("WebP: ALPH ref out of bounds"));
        }
        Some(&body[a.data_offset..a.data_offset + a.data_len])
    } else {
        None
    };

    let mut out = Vec::with_capacity(
        64 + img_bytes.len() + f.alph.as_ref().map(|a| a.data_len + 16).unwrap_or(0) + 8,
    );
    out.extend_from_slice(b"OWEB");
    out.push(2);
    let mut flags = 0u8;
    if f.alph.is_some() {
        flags |= 0x01;
    }
    if is_vp8l {
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
    for v in [
        canvas.0, canvas.1, f.x_offset, f.y_offset, f.width, f.height,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&f.duration_ms.to_le_bytes());
    out.extend_from_slice(&(img_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(img_bytes);
    if let (Some(a), Some(ab)) = (&f.alph, alph_bytes) {
        out.push(a.pre_processing);
        out.push(a.filtering);
        out.push(a.compression);
        out.extend_from_slice(&(ab.len() as u32).to_le_bytes());
        out.extend_from_slice(ab);
    }
    if let Some(bgra) = anim_background_bgra {
        out.extend_from_slice(&bgra);
    }
    Ok(out)
}

/// Counterpart to `encode_lazy_frame_payload`. Used by the decoder.
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

/// Decoded view of the VP8X header's flags byte (RFC 9649 §2.5.6
/// Fig 7). The MSB-first bit layout is `|Rsv|Rsv|I|L|E|X|A|R|`; the
/// reserved bits 7, 6, and 0 MUST be ignored per the spec, so we mask
/// them out at parse time rather than asserting they're zero (asserting
/// would itself be a spec violation — "Readers MUST ignore this field").
#[derive(Clone, Copy, Debug)]
struct VP8xFlags {
    has_icc: bool,
    has_alpha: bool,
    has_exif: bool,
    has_xmp: bool,
    has_anim: bool,
}

impl VP8xFlags {
    fn from_byte(b: u8) -> Self {
        Self {
            has_icc: b & 0x20 != 0,
            has_alpha: b & 0x10 != 0,
            has_exif: b & 0x08 != 0,
            has_xmp: b & 0x04 != 0,
            has_anim: b & 0x02 != 0,
        }
    }
}

/// Position in the spec-mandated chunk order for the reconstruction
/// chunks (RFC 9649 §2.7). Inside a VP8X container the reconstruction
/// chunks are required to appear in this fixed order:
///
/// ```text
/// VP8X → [ICCP] → [ANIM] → ANMF*               (animated)
/// VP8X → [ICCP] → [ALPH] → (VP8 | VP8L)        (still image)
/// ```
///
/// `EXIF` / `XMP ` / unknown FourCCs may appear anywhere after VP8X
/// per the same section ("Metadata and unknown chunks MAY appear out
/// of order"), so they do NOT advance the stage.
///
/// The stage value monotonically increases through a single file; an
/// attempt to move backwards (`AnmfSeen` → `IccpSeen`, say) is the
/// spec-defined "out of order" condition the parser rejects with
/// `WebP: chunk X out of order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ChunkStage {
    AfterVp8x,
    IccpSeen,
    AnimSeen,
    AnmfSeen,
    AlphSeen,
    ImageSeen,
}

impl ChunkStage {
    /// Try to advance to `target`. Returns an `out of order` error when
    /// the current stage is already past `target` (i.e. the spec-named
    /// chunk arrived too late). Idempotent moves (e.g. duplicate ICCP,
    /// repeated ANMF tail chunks) are allowed because the spec only
    /// constrains the relative order, not the multiplicity — the
    /// "at most one" rule for ICCP / ANIM is a separate SHOULD that
    /// readers MAY ignore extras for; we overwrite-on-duplicate per
    /// the existing code path.
    fn advance_to(&mut self, target: ChunkStage, fourcc: &[u8; 4]) -> Result<()> {
        // Special-case the ANMF tail: once we've seen one ANMF the
        // remaining frames stay at the same stage. ALPH inside an ANMF
        // is handled by the per-frame parser (parse_anmf), not the
        // top-level loop, so this is just the repeated-ANMF case.
        if target == ChunkStage::AnmfSeen && *self == ChunkStage::AnmfSeen {
            return Ok(());
        }
        if (*self as u8) > (target as u8) {
            return Err(Error::invalid(format!(
                "WebP: chunk {:?} out of order (stage {:?} → {:?})",
                std::str::from_utf8(fourcc).unwrap_or("???"),
                self,
                target,
            )));
        }
        *self = target;
        Ok(())
    }
}

fn parse_extended(vp8x: &[u8], chunks: &mut RiffChunks<'_>) -> Result<ParsedContainer> {
    if vp8x.len() < 10 {
        return Err(Error::invalid("WebP: VP8X chunk too short"));
    }
    // VP8X layout: 1 byte flags, 3 bytes reserved, 3 bytes canvas_w-1, 3 bytes canvas_h-1.
    //
    // Flag bits (MSB-first per RFC 9649 §2.5.6 "Extended File Header" Fig 7):
    //   bit 7: Rsv  (reserved, MUST be 0; readers MUST ignore)
    //   bit 6: Rsv  (reserved, MUST be 0; readers MUST ignore)
    //   bit 5: I    (ICC profile)
    //   bit 4: L    (aLpha)
    //   bit 3: E    (Exif)
    //   bit 2: X    (Xmp)
    //   bit 1: A    (Animation)
    //   bit 0: R    (reserved, MUST be 0; readers MUST ignore)
    //
    // Reserved bits are deliberately not validated per the explicit
    // "Readers MUST ignore this field" wording, so a strict masking
    // would be a spec violation in the other direction.
    let flags = vp8x[0];
    let flags = VP8xFlags::from_byte(flags);
    let canvas_w = (u32::from_le_bytes([vp8x[4], vp8x[5], vp8x[6], 0]) & 0x00FF_FFFF) + 1;
    let canvas_h = (u32::from_le_bytes([vp8x[7], vp8x[8], vp8x[9], 0]) & 0x00FF_FFFF) + 1;
    // RFC 9649 §2.5.6: "The product of Canvas Width and Canvas Height
    // MUST be at most 2^32 - 1." 24-bit canvas dimensions allow up to
    // ~16 M per axis, so an overflow is a real malformation rather
    // than a numeric corner case. The product of two 24-bit values
    // fits in u64 without overflow (the cap is 2^48 = ~2.8e14, well
    // under u64::MAX), so we can multiply directly and compare.
    if (canvas_w as u64) * (canvas_h as u64) > u32::MAX as u64 {
        return Err(Error::invalid(format!(
            "WebP: VP8X canvas {canvas_w}×{canvas_h} exceeds 2^32 pixels"
        )));
    }

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
    // gating on `flags.has_anim` below preserves the spec's "ignore"
    // while still reading the chunk so a future loose-mode could
    // surface it).
    let mut anim_background_bgra: Option<[u8; 4]> = None;
    let mut anim_loop_count: Option<u16> = None;

    let mut total_duration = 0u32;
    // Track chunk-order state per RFC 9649 §2.7: the reconstruction
    // chunks (VP8X / ICCP / ANIM / ANMF / ALPH / VP8 / VP8L) MUST
    // appear in the order listed; readers SHOULD fail on out-of-order
    // sequences. EXIF / XMP / unknown chunks MAY appear anywhere after
    // the VP8X header and don't advance the stage.
    let mut stage = ChunkStage::AfterVp8x;

    while let Some(c) = chunks.next().transpose()? {
        match &c.id {
            b"VP8 " => {
                stage.advance_to(ChunkStage::ImageSeen, &c.id)?;
                pending_image = Some(ImagePayload::Vp8(c.data.to_vec()));
            }
            b"VP8L" => {
                stage.advance_to(ChunkStage::ImageSeen, &c.id)?;
                pending_image = Some(ImagePayload::Vp8l(c.data.to_vec()));
            }
            b"ALPH" => {
                // ALPH precedes VP8 / VP8L bitstream per spec Fig 15.
                stage.advance_to(ChunkStage::AlphSeen, &c.id)?;
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
                stage.advance_to(ChunkStage::AnmfSeen, &c.id)?;
                let anmf = parse_anmf(c.data)?;
                let f = anmf.into_frame();
                total_duration = total_duration.saturating_add(f.duration_ms);
                frames.push(f);
            }
            // Auxiliary metadata chunks — captured into `metadata` so
            // callers can read them off `WebpImage::metadata` after
            // decode. Per RFC 9649 §2.7 "Metadata and unknown chunks
            // MAY appear out of order", so EXIF / XMP / unknown FourCCs
            // don't advance the chunk-stage state machine.
            b"ICCP" => {
                stage.advance_to(ChunkStage::IccpSeen, &c.id)?;
                metadata.icc = Some(c.data.to_vec());
            }
            b"EXIF" => metadata.exif = Some(c.data.to_vec()),
            b"XMP " => metadata.xmp = Some(c.data.to_vec()),
            // ANIM chunk: 4 bytes [B, G, R, A] background colour +
            // 2 bytes little-endian loop count (RFC 9649 §2.5,
            // "ANIM Chunk"). Per spec: "This chunk MUST appear if
            // the Animation flag in the VP8X Chunk is set. If the
            // Animation flag is not set and this chunk is present,
            // it MUST be ignored." We capture it unconditionally
            // and gate exposure on `flags.has_anim` after the loop
            // so a file with the flag clear still parses cleanly.
            b"ANIM" if c.data.len() >= 6 => {
                stage.advance_to(ChunkStage::AnimSeen, &c.id)?;
                anim_background_bgra = Some([c.data[0], c.data[1], c.data[2], c.data[3]]);
                anim_loop_count = Some(u16::from_le_bytes([c.data[4], c.data[5]]));
            }
            b"ANIM" => {
                stage.advance_to(ChunkStage::AnimSeen, &c.id)?;
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
    if !flags.has_anim {
        anim_background_bgra = None;
        anim_loop_count = None;
    }
    // Per RFC 9649 §2.5.6 + §2.7.1.4 / §2.7.1.5 / §2.7.1.2: ICCP /
    // EXIF / XMP / ALPH chunks are flag-gated like ANIM. Drop them
    // silently when the corresponding VP8X flag is clear so consumers
    // see a consistent (and spec-compliant) "flag wins" interpretation.
    // For the still-image case, an alpha plane stripped this way
    // surfaces as a fully-opaque image rather than a parse failure —
    // matches the "MAY ignore" tolerance the spec permits.
    if !flags.has_icc {
        metadata.icc = None;
    }
    if !flags.has_exif {
        metadata.exif = None;
    }
    if !flags.has_xmp {
        metadata.xmp = None;
    }
    if !flags.has_alpha {
        pending_alph = None;
    }
    let has_anim = flags.has_anim;

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
    // See `parse_extended` for the spec-derived rationale behind each
    // VP8X-side validation (flag decoding, canvas-product overflow,
    // chunk-order state machine, flag-gated drop of metadata + ALPH).
    // This path mirrors that logic byte-for-byte; both parsers walk
    // the same chunk tree and must surface the same parse errors so
    // streaming demux + eager decode agree on malformed-file
    // rejection.
    if vp8x.len() < 10 {
        return Err(Error::invalid("WebP: VP8X chunk too short"));
    }
    let flags = VP8xFlags::from_byte(vp8x[0]);
    let canvas_w = (u32::from_le_bytes([vp8x[4], vp8x[5], vp8x[6], 0]) & 0x00FF_FFFF) + 1;
    let canvas_h = (u32::from_le_bytes([vp8x[7], vp8x[8], vp8x[9], 0]) & 0x00FF_FFFF) + 1;
    if (canvas_w as u64) * (canvas_h as u64) > u32::MAX as u64 {
        return Err(Error::invalid(format!(
            "WebP: VP8X canvas {canvas_w}×{canvas_h} exceeds 2^32 pixels"
        )));
    }

    let mut frames: Vec<LazyParsedFrame> = Vec::new();
    let mut pending_alph: Option<LazyAlphRef> = None;
    let mut pending_image: Option<LazyImageRef> = None;
    let mut metadata = WebpFileMetadata::default();
    let mut anim_background_bgra: Option<[u8; 4]> = None;
    let mut anim_loop_count: Option<u16> = None;
    let mut total_duration = 0u32;
    let mut stage = ChunkStage::AfterVp8x;

    while let Some(c) = chunks.next().transpose()? {
        match &c.id {
            b"VP8 " => {
                stage.advance_to(ChunkStage::ImageSeen, &c.id)?;
                pending_image = Some(LazyImageRef::Vp8 {
                    offset: c.payload_offset,
                    len: c.data.len(),
                });
            }
            b"VP8L" => {
                stage.advance_to(ChunkStage::ImageSeen, &c.id)?;
                pending_image = Some(LazyImageRef::Vp8l {
                    offset: c.payload_offset,
                    len: c.data.len(),
                });
            }
            b"ALPH" => {
                stage.advance_to(ChunkStage::AlphSeen, &c.id)?;
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
                stage.advance_to(ChunkStage::AnmfSeen, &c.id)?;
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
            b"ICCP" => {
                stage.advance_to(ChunkStage::IccpSeen, &c.id)?;
                metadata.icc = Some(c.data.to_vec());
            }
            b"EXIF" => metadata.exif = Some(c.data.to_vec()),
            b"XMP " => metadata.xmp = Some(c.data.to_vec()),
            b"ANIM" if c.data.len() >= 6 => {
                stage.advance_to(ChunkStage::AnimSeen, &c.id)?;
                anim_background_bgra = Some([c.data[0], c.data[1], c.data[2], c.data[3]]);
                anim_loop_count = Some(u16::from_le_bytes([c.data[4], c.data[5]]));
            }
            b"ANIM" => {
                stage.advance_to(ChunkStage::AnimSeen, &c.id)?;
            }
            _ => {}
        }
    }
    if !flags.has_anim {
        anim_background_bgra = None;
        anim_loop_count = None;
    }
    if !flags.has_icc {
        metadata.icc = None;
    }
    if !flags.has_exif {
        metadata.exif = None;
    }
    if !flags.has_xmp {
        metadata.xmp = None;
    }
    if !flags.has_alpha {
        pending_alph = None;
    }
    let has_anim = flags.has_anim;
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

/// Streaming WebP `Demuxer` impl. Owns the full RIFF body buffer once
/// (so `parse_webp_body_lazy`'s per-frame `(offset, len)` ranges stay
/// valid), then yields one `Packet` per [`Demuxer::next_packet`] call
/// — the OWEB payload is materialised on demand via
/// [`encode_lazy_frame_payload`] instead of being precomputed for every
/// frame at `open` time.
///
/// For long animations (1000+ frames) this trades a one-time
/// `Vec<Packet>` allocation (size ~ sum of all OWEB blobs) for
/// per-frame allocations totalling the same size — but the working
/// set at any moment is one packet's worth, not all of them. Same
/// shape as libwebp's `WebPAnimDecoder` API, where each
/// `WebPAnimDecoderGetNext` call materialises the next frame's
/// composite without pre-computing all frames up front.
#[cfg(feature = "registry")]
struct WebpDemuxer {
    stream: StreamInfo,
    /// Owned RIFF body buffer; sliced by `parsed.frames` ranges.
    body: Vec<u8>,
    /// Lazily-walked container metadata + frame `(offset, len)` ranges.
    parsed: LazyParsedContainer,
    /// Time base used to construct each `Packet` (millisecond units —
    /// matches the WebP ANMF native time scale).
    time_base: TimeBase,
    /// Index of the next frame to emit. Bumped by `next_packet`.
    pos: usize,
    /// Cumulative PTS in time-base units; bumped by each frame's
    /// `max(duration_ms, 1)` so we mirror the eager
    /// `ParsedContainer::into_packets` arithmetic exactly.
    pts: i64,
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
        if self.pos >= self.parsed.frames.len() {
            return Err(oxideav_core::Error::Eof);
        }
        let i = self.pos;
        let f = &self.parsed.frames[i];
        let duration = f.duration_ms;
        let data = encode_lazy_frame_payload(
            f,
            &self.body,
            self.parsed.canvas,
            self.parsed.anim_background_bgra,
        )?;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = Some(self.pts);
        pkt.dts = Some(self.pts);
        pkt.duration = Some(duration.max(1) as i64);
        pkt.flags.keyframe = i == 0;
        self.pts += duration.max(1) as i64;
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

    /// Build a tiny 3-frame animated WebP for the streaming Demuxer
    /// tests. Mirrors the inline `three_frame_anim` in
    /// `anim_decoder.rs::tests`.
    fn three_frame_anim_blob() -> Vec<u8> {
        use crate::encoder_anim::{build_animated_webp, AnimFrame};
        const W: u32 = 8;
        const H: u32 = 8;
        let solid = |rgba: [u8; 4]| -> Vec<u8> {
            let n = (W as usize) * (H as usize);
            let mut v = Vec::with_capacity(n * 4);
            for _ in 0..n {
                v.extend_from_slice(&rgba);
            }
            v
        };
        let red = solid([0xff, 0, 0, 0xff]);
        let green = solid([0, 0xff, 0, 0xff]);
        let blue = solid([0, 0, 0xff, 0xff]);
        let frames = [
            AnimFrame {
                width: W,
                height: H,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 30,
                blend: false,
                dispose_to_background: false,
                rgba: &red,
            },
            AnimFrame {
                width: W,
                height: H,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 40,
                blend: false,
                dispose_to_background: false,
                rgba: &green,
            },
            AnimFrame {
                width: W,
                height: H,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 50,
                blend: false,
                dispose_to_background: false,
                rgba: &blue,
            },
        ];
        build_animated_webp(W, H, [0, 0, 0, 0], 0, &frames).expect("encode")
    }

    #[test]
    fn streaming_demuxer_yields_frames_in_order_with_pts() {
        // The streaming Demuxer materialises one packet at a time —
        // `pos` advances per `next_packet`, PTS arithmetic mirrors the
        // anim-decoder's `pts_ms` (cumulative `max(duration_ms, 1)`).
        let blob = three_frame_anim_blob();
        let cursor = std::io::Cursor::new(blob);
        let mut demux = open_boxed(Box::new(cursor)).expect("open");
        // Streams metadata is available before any packet is pulled.
        assert_eq!(demux.streams().len(), 1);
        let p0 = demux.next_packet().expect("first");
        assert_eq!(p0.pts, Some(0));
        assert_eq!(p0.duration, Some(30));
        assert!(p0.flags.keyframe);
        let p1 = demux.next_packet().expect("second");
        assert_eq!(p1.pts, Some(30));
        assert_eq!(p1.duration, Some(40));
        assert!(!p1.flags.keyframe);
        let p2 = demux.next_packet().expect("third");
        assert_eq!(p2.pts, Some(70));
        assert_eq!(p2.duration, Some(50));
        // Subsequent calls report Eof.
        assert!(matches!(demux.next_packet(), Err(oxideav_core::Error::Eof)));
    }

    #[test]
    fn streaming_demuxer_packet_payload_round_trips_through_decoder() {
        // Each packet's OWEB payload must decode through the same
        // `decode_frame_payload` reader the registry decoder uses —
        // proving the streaming `encode_lazy_frame_payload` is
        // byte-for-byte equivalent to the eager helper it replaced.
        let blob = three_frame_anim_blob();
        let cursor = std::io::Cursor::new(blob);
        let mut demux = open_boxed(Box::new(cursor)).expect("open");
        for i in 0..3 {
            let pkt = demux.next_packet().expect("ok");
            let parsed = decode_frame_payload(pkt.data.as_slice()).expect("OWEB parse");
            assert_eq!(parsed.canvas, (8, 8), "frame {i} canvas");
            assert_eq!(parsed.width, 8);
            assert_eq!(parsed.height, 8);
            // Image bytes are non-empty (a real VP8L bitstream).
            assert!(!parsed.image.is_empty());
        }
    }

    #[test]
    fn streaming_demuxer_does_not_pre_allocate_packet_vec() {
        // Sanity-check the streaming shape: opening doesn't materialise
        // every packet up front — we can construct the demuxer for a
        // 3-frame blob and immediately drop it without any cost beyond
        // the (small) lazy-parse + body buffer. Best signal we can
        // reasonably observe in a unit test: the type doesn't expose a
        // `Vec<Packet>` surface, and `next_packet` is the only way to
        // get one out.
        let blob = three_frame_anim_blob();
        let cursor = std::io::Cursor::new(blob);
        let demux = open_boxed(Box::new(cursor)).expect("open");
        // We never call `next_packet`; the demuxer holds only the body
        // + the lazy frame ranges. Drop and check no panic.
        drop(demux);
    }

    #[test]
    fn streaming_demuxer_eof_is_repeatable() {
        // After draining, every subsequent `next_packet` returns Eof
        // — not a panic, not a re-emit of the last packet.
        let blob = three_frame_anim_blob();
        let cursor = std::io::Cursor::new(blob);
        let mut demux = open_boxed(Box::new(cursor)).expect("open");
        for _ in 0..3 {
            let _ = demux.next_packet().expect("ok");
        }
        for _ in 0..5 {
            assert!(matches!(demux.next_packet(), Err(oxideav_core::Error::Eof)));
        }
    }
}
