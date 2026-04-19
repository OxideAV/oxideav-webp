//! RIFF/WEBP container writer used by the VP8 lossy + VP8L lossless
//! encoders.
//!
//! The WebP on-disk container is a RIFF file whose form type is `WEBP`.
//! Three file layouts exist (see the demuxer's module docs for the parse
//! side):
//!
//! * **Simple lossy**: `RIFF ... WEBP VP8  ... <vp8-keyframe>`.
//! * **Simple lossless**: `RIFF ... WEBP VP8L ... <vp8l-bitstream>`.
//! * **Extended**: `RIFF ... WEBP VP8X <hdr> [ICCP|ANIM|ALPH|VP8 |VP8L|ANMF|EXIF|XMP ]*`.
//!
//! The extended layout is mandatory whenever any of (alpha sidecar,
//! ICC, EXIF, XMP, animation) is attached — the VP8X header announces
//! both the canvas size and which auxiliary chunks will follow.
//!
//! All helpers here produce a complete on-disk WebP file. Callers that
//! want a bare bitstream (e.g. the `encode_vp8l_argb` low-level entry
//! point) should skip this module entirely.

/// Optional auxiliary metadata attached alongside an image chunk. Any
/// field being `Some` triggers the extended (`VP8X`) container layout.
#[derive(Default, Clone)]
pub struct WebpMetadata<'a> {
    /// Raw ICC profile bytes, written into an `ICCP` chunk.
    pub icc: Option<&'a [u8]>,
    /// Raw EXIF payload, written into an `EXIF` chunk.
    pub exif: Option<&'a [u8]>,
    /// Raw XMP payload, written into an `XMP ` chunk (trailing space
    /// is part of the FourCC).
    pub xmp: Option<&'a [u8]>,
}

impl WebpMetadata<'_> {
    /// True if any of the metadata fields is populated.
    pub fn any(&self) -> bool {
        self.icc.is_some() || self.exif.is_some() || self.xmp.is_some()
    }
}

/// The image chunk identity we're wrapping.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    /// VP8 keyframe bytes (FourCC `VP8 ` — note the trailing space).
    Vp8Lossy,
    /// VP8L bitstream bytes (FourCC `VP8L`).
    Vp8lLossless,
}

impl ImageKind {
    fn fourcc(self) -> &'static [u8; 4] {
        match self {
            ImageKind::Vp8Lossy => b"VP8 ",
            ImageKind::Vp8lLossless => b"VP8L",
        }
    }
}

/// Write an image chunk (VP8 or VP8L) optionally accompanied by an
/// `ALPH` sidecar and/or metadata chunks. Returns a complete `.webp`
/// file.
///
/// `canvas_w` / `canvas_h` are 1-based pixel counts (1..=16384 for
/// lossless, 1..=16383 for lossy — the caller is expected to already
/// clamp).
///
/// Layout decisions:
/// * If `alph` is `None` AND `meta.any()` is false, a simple-file
///   layout is emitted: `RIFF ... WEBP <VP8|VP8L> ... payload`.
/// * Otherwise we emit the extended layout: `RIFF ... WEBP VP8X ...
///   [ICCP] [ALPH] <VP8|VP8L> [EXIF] [XMP ]`. The chunk order matches
///   the recommendation in the RIFF container spec.
pub fn build_webp_file(
    kind: ImageKind,
    image_bytes: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    alph: Option<&AlphChunkBytes>,
    meta: &WebpMetadata<'_>,
) -> Vec<u8> {
    let needs_extended = alph.is_some() || meta.any();
    if !needs_extended {
        return build_simple(kind, image_bytes);
    }
    build_extended(kind, image_bytes, canvas_w, canvas_h, alph, meta, false)
}

/// Emit an extended-layout `.webp` file for a VP8L bitstream whose own
/// payload already carries an alpha channel. The VP8X header gets its
/// ALPHA flag set (no separate `ALPH` sidecar — the VP8L chunk is
/// self-contained for alpha), and `meta` can attach any combination of
/// ICC/EXIF/XMP.
///
/// This is the RGBA-lossless entry point used by the `Vp8lEncoder`
/// adapter. It deliberately does NOT expose VP8 lossy's pattern —
/// VP8-lossy + RGBA goes through the ALPH sidecar path in
/// [`build_webp_file`] instead.
pub fn build_vp8l_with_alpha(
    image_bytes: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    meta: &WebpMetadata<'_>,
) -> Vec<u8> {
    build_extended(
        ImageKind::Vp8lLossless,
        image_bytes,
        canvas_w,
        canvas_h,
        None,
        meta,
        true,
    )
}

/// Pre-assembled ALPH chunk payload (header byte + compressed plane).
/// See the VP8X / ALPH spec §5.2.3 for the header byte layout.
pub struct AlphChunkBytes {
    /// `(reserved<<6) | (pre_processing<<4) | (filtering<<2) | compression`
    pub header_byte: u8,
    /// Raw alpha plane or VP8L-compressed alpha bitstream. No signature
    /// bytes — the demuxer assumes the payload starts immediately after
    /// the 1-byte header.
    pub payload: Vec<u8>,
}

fn build_simple(kind: ImageKind, image_bytes: &[u8]) -> Vec<u8> {
    let chunk_len = image_bytes.len() as u32;
    let pad = (chunk_len & 1) as usize;
    // riff body = "WEBP" (4) + chunk hdr (8) + payload + pad.
    let riff_size = 4 + 8 + chunk_len as usize + pad;
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(kind.fourcc());
    out.extend_from_slice(&chunk_len.to_le_bytes());
    out.extend_from_slice(image_bytes);
    if pad == 1 {
        out.push(0);
    }
    out
}

fn build_extended(
    kind: ImageKind,
    image_bytes: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    alph: Option<&AlphChunkBytes>,
    meta: &WebpMetadata<'_>,
    force_alpha_flag: bool,
) -> Vec<u8> {
    // First compose the body (everything after "WEBP"), then wrap it in
    // the top-level RIFF envelope.
    let mut body: Vec<u8> = Vec::with_capacity(
        8 + 10
            + image_bytes.len()
            + alph.map(|a| a.payload.len() + 16).unwrap_or(0)
            + meta.icc.map(|v| v.len() + 16).unwrap_or(0)
            + meta.exif.map(|v| v.len() + 16).unwrap_or(0)
            + meta.xmp.map(|v| v.len() + 16).unwrap_or(0),
    );

    // VP8X chunk header + payload.
    let mut flags: u8 = 0;
    if meta.icc.is_some() {
        flags |= 0x20; // ICC
    }
    if meta.exif.is_some() {
        flags |= 0x08; // EXIF
    }
    if meta.xmp.is_some() {
        flags |= 0x04; // XMP
    }
    if alph.is_some() || force_alpha_flag {
        flags |= 0x10; // ALPHA
    }
    write_chunk(&mut body, b"VP8X", &vp8x_payload(flags, canvas_w, canvas_h));

    // ICCP must come immediately after VP8X.
    if let Some(icc) = meta.icc {
        write_chunk(&mut body, b"ICCP", icc);
    }
    // ALPH precedes the VP8 chunk (only meaningful with VP8 lossy, but
    // we don't enforce that here — the demuxer ignores an ALPH that
    // rides with a VP8L chunk, which is the correct behaviour).
    if let Some(a) = alph {
        let mut alph_data = Vec::with_capacity(1 + a.payload.len());
        alph_data.push(a.header_byte);
        alph_data.extend_from_slice(&a.payload);
        write_chunk(&mut body, b"ALPH", &alph_data);
    }
    write_chunk(&mut body, kind.fourcc(), image_bytes);
    if let Some(exif) = meta.exif {
        write_chunk(&mut body, b"EXIF", exif);
    }
    if let Some(xmp) = meta.xmp {
        write_chunk(&mut body, b"XMP ", xmp);
    }

    // Wrap in the RIFF/WEBP envelope.
    let riff_size = 4 + body.len();
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(&body);
    out
}

/// Build the 10-byte VP8X payload (flags + reserved + canvas size).
///
/// Spec (RIFF container, VP8X chunk):
/// * Byte 0: flags (bit4=ALPHA, bit3=EXIF, bit2=XMP, bit5=ICC,
///   bit1=ANIM; bits 0/6/7 reserved).
/// * Bytes 1..4: reserved (zero).
/// * Bytes 4..7: canvas_width_minus_1 (24-bit LE).
/// * Bytes 7..10: canvas_height_minus_1 (24-bit LE).
fn vp8x_payload(flags: u8, canvas_w: u32, canvas_h: u32) -> [u8; 10] {
    let mut out = [0u8; 10];
    out[0] = flags;
    let w_minus_1 = canvas_w.saturating_sub(1) & 0x00FF_FFFF;
    let h_minus_1 = canvas_h.saturating_sub(1) & 0x00FF_FFFF;
    out[4] = (w_minus_1 & 0xff) as u8;
    out[5] = ((w_minus_1 >> 8) & 0xff) as u8;
    out[6] = ((w_minus_1 >> 16) & 0xff) as u8;
    out[7] = (h_minus_1 & 0xff) as u8;
    out[8] = ((h_minus_1 >> 8) & 0xff) as u8;
    out[9] = ((h_minus_1 >> 16) & 0xff) as u8;
    out
}

/// Append a single RIFF chunk — 4-byte FourCC, 4-byte LE size, payload,
/// and the 0 pad byte if the payload is odd-sized.
fn write_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_layout_vp8l_no_extras() {
        let payload = vec![0x2fu8; 10];
        let out = build_webp_file(
            ImageKind::Vp8lLossless,
            &payload,
            32,
            32,
            None,
            &WebpMetadata::default(),
        );
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        assert_eq!(&out[12..16], b"VP8L");
    }

    #[test]
    fn extended_layout_emits_vp8x_first() {
        let payload = vec![0x2fu8; 10];
        let meta = WebpMetadata {
            icc: Some(b"fake-icc"),
            ..Default::default()
        };
        let out = build_webp_file(ImageKind::Vp8lLossless, &payload, 64, 48, None, &meta);
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        assert_eq!(&out[12..16], b"VP8X");
        // 10-byte VP8X payload (ICC flag set: bit 5 = 0x20).
        assert_eq!(out[20], 0x20);
        // Canvas size encoded as w-1 (24-bit) / h-1 (24-bit).
        let w_minus_1 = u32::from_le_bytes([out[24], out[25], out[26], 0]) & 0x00FF_FFFF;
        let h_minus_1 = u32::from_le_bytes([out[27], out[28], out[29], 0]) & 0x00FF_FFFF;
        assert_eq!(w_minus_1, 63);
        assert_eq!(h_minus_1, 47);
    }

    #[test]
    fn extended_layout_with_alph_flags_both_bits() {
        let payload = vec![0x11u8; 5];
        let alph = AlphChunkBytes {
            header_byte: 0,
            payload: vec![0xffu8; 16],
        };
        let meta = WebpMetadata {
            exif: Some(b"exif!"),
            ..Default::default()
        };
        let out = build_webp_file(ImageKind::Vp8Lossy, &payload, 16, 16, Some(&alph), &meta);
        // VP8X byte0: ALPHA (0x10) + EXIF (0x08) = 0x18.
        assert_eq!(out[20], 0x18);
    }
}
