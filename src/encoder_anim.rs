//! Animated WebP encoder.
//!
//! Builds a `RIFF/WEBP/VP8X + ANIM + ANMF...ANMF` file from a sequence of
//! per-frame RGBA tiles. Each frame is encoded losslessly through the
//! in-crate VP8L pipeline ([`crate::vp8l::encode_vp8l_argb`]) and wrapped
//! inside an `ANMF` chunk so the whole file is a single self-contained
//! `.webp` animation that any reader speaking the WebP container spec
//! (Google `developers.google.com/speed/webp/docs/riff_container`) can
//! play.
//!
//! Decode of animated WebP already works — this module simply closes the
//! loop on the encode side. Mixed lossy + lossless animations are not
//! produced here yet; the lossy path requires a separate per-frame
//! orchestration of the VP8 + ALPH split that wasn't worth the extra
//! surface area for the first cut. The decoder accepts both shapes either
//! way.
//!
//! # Container layout
//!
//! ```text
//! RIFF <size> WEBP
//!   VP8X <10>   — flags (ANIM bit set), canvas_w-1, canvas_h-1
//!   ANIM <6>    — 4 BGRA bytes background, 2-byte loop count (0=infinite)
//!   ANMF <n>    — per-frame envelope (header + nested VP8L chunk)
//!   ANMF <n>    — ...
//! ```
//!
//! Per `ANMF` header (16 bytes before nested chunks):
//!
//! ```text
//!   3 bytes  X offset / 2          (must be even)
//!   3 bytes  Y offset / 2          (must be even)
//!   3 bytes  frame_w - 1
//!   3 bytes  frame_h - 1
//!   3 bytes  duration_ms
//!   1 byte   bit0 = blending (0=blend, 1=overwrite)
//!            bit1 = disposal (0=none,  1=dispose-to-background)
//! ```
//!
//! The nested chunk is a single `VP8L` (lossless) sub-chunk produced by
//! the existing per-frame VP8L encoder.

use oxideav_core::{Error, Result};

use crate::vp8l::encode_vp8l_argb;

/// One frame of an animation: an RGBA tile sized `width × height` rendered
/// at `(x_offset, y_offset)` on the canvas, displayed for `duration_ms`
/// before the next frame is composited.
///
/// `x_offset` and `y_offset` are stored on disk as half their value (the
/// spec mandates even offsets), so we silently round odd values down to
/// the next even number.
#[derive(Clone)]
pub struct AnimFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub x_offset: u32,
    pub y_offset: u32,
    pub duration_ms: u32,
    /// `true` → blend the frame's alpha onto the canvas. `false` → the
    /// frame overwrites the destination pixels (alpha included).
    pub blend: bool,
    /// `true` → after rendering, clear the frame's bbox to the background
    /// colour before drawing the next frame.
    pub dispose_to_background: bool,
    /// Row-major RGBA bytes for this tile — `width * height * 4` long.
    pub rgba: &'a [u8],
}

/// Build a complete animated `.webp` file from a slice of frames + a
/// canvas size. Loop count = 0 means infinite playback (the WebP default).
/// Background is BGRA; the spec writes B, G, R, A in that order — we
/// accept it the same way.
pub fn build_animated_webp(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
) -> Result<Vec<u8>> {
    if canvas_w == 0 || canvas_h == 0 {
        return Err(Error::invalid("animated WebP: zero canvas size"));
    }
    if canvas_w > 16384 || canvas_h > 16384 {
        return Err(Error::invalid("animated WebP: canvas exceeds 16384 px"));
    }
    if frames.is_empty() {
        return Err(Error::invalid("animated WebP: needs at least one frame"));
    }

    // Pre-encode every frame's nested VP8L chunk first. Doing it up
    // front lets us measure each chunk and lay out the RIFF body in a
    // single pass without a second iteration.
    let mut anmf_payloads: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for f in frames {
        if f.width == 0 || f.height == 0 {
            return Err(Error::invalid("animated WebP: zero frame size"));
        }
        if f.x_offset
            .checked_add(f.width)
            .map(|r| r > canvas_w)
            .unwrap_or(true)
            || f.y_offset
                .checked_add(f.height)
                .map(|r| r > canvas_h)
                .unwrap_or(true)
        {
            return Err(Error::invalid(
                "animated WebP: frame bbox extends past canvas",
            ));
        }
        if f.rgba.len() != (f.width as usize) * (f.height as usize) * 4 {
            return Err(Error::invalid(
                "animated WebP: frame rgba length mismatch frame_w*frame_h*4",
            ));
        }
        if f.duration_ms > 0x00FF_FFFF {
            return Err(Error::invalid(
                "animated WebP: duration_ms exceeds 24-bit field",
            ));
        }

        // Encode frame to a bare VP8L bitstream. Per-frame `has_alpha`
        // detection — same convention as the still-image encoder.
        let mut pixels = Vec::with_capacity((f.width as usize) * (f.height as usize));
        let mut has_alpha = false;
        for px in f.rgba.chunks_exact(4) {
            let r = px[0] as u32;
            let g = px[1] as u32;
            let b = px[2] as u32;
            let a = px[3] as u32;
            if a != 0xff {
                has_alpha = true;
            }
            pixels.push((a << 24) | (r << 16) | (g << 8) | b);
        }
        let vp8l_bytes = encode_vp8l_argb(f.width, f.height, &pixels, has_alpha)?;

        // Build the ANMF payload (16-byte header + nested VP8L chunk).
        let mut payload = Vec::with_capacity(16 + 8 + vp8l_bytes.len());
        // Even offsets; the spec stores them divided by 2.
        write_u24_le(&mut payload, (f.x_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.y_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.width - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.height - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, f.duration_ms & 0x00FF_FFFF);
        // bit 0: blending — 0 = use alpha blending, 1 = overwrite.
        // bit 1: disposal — 0 = none, 1 = dispose-to-background.
        let mut flags: u8 = 0;
        if !f.blend {
            flags |= 0x01;
        }
        if f.dispose_to_background {
            flags |= 0x02;
        }
        payload.push(flags);

        // Nested VP8L chunk inside the ANMF body.
        write_chunk(&mut payload, b"VP8L", &vp8l_bytes);
        anmf_payloads.push(payload);
    }

    // Assemble the body that lives between "WEBP" and the end of the
    // RIFF envelope: VP8X header + ANIM + N x ANMF.
    let mut body: Vec<u8> = Vec::new();

    // VP8X chunk: ALPHA flag (0x10) + ANIM flag (0x02). We always set the
    // ALPHA flag for animations — a per-frame VP8L chunk can carry alpha
    // and the decoder only respects ALPHA at the canvas-header level.
    let vp8x = vp8x_payload(0x12, canvas_w, canvas_h);
    write_chunk(&mut body, b"VP8X", &vp8x);

    // ANIM chunk: 4 bytes BGRA + 2 bytes loop count.
    let mut anim = [0u8; 6];
    anim[0] = background_bgra[0];
    anim[1] = background_bgra[1];
    anim[2] = background_bgra[2];
    anim[3] = background_bgra[3];
    anim[4] = (loop_count & 0xff) as u8;
    anim[5] = ((loop_count >> 8) & 0xff) as u8;
    write_chunk(&mut body, b"ANIM", &anim);

    // ANMF chunks.
    for payload in &anmf_payloads {
        write_chunk(&mut body, b"ANMF", payload);
    }

    // RIFF envelope.
    let riff_size = 4 + body.len();
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(&body);
    Ok(out)
}

/// VP8X payload: 1 byte flags, 3 bytes reserved, 3 bytes canvas_w-1,
/// 3 bytes canvas_h-1.
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

fn write_u24_le(out: &mut Vec<u8>, v: u32) {
    out.push((v & 0xff) as u8);
    out.push(((v >> 8) & 0xff) as u8);
    out.push(((v >> 16) & 0xff) as u8);
}

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

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w as usize) * (h as usize) * 4);
        for _ in 0..(w * h) {
            v.extend_from_slice(&rgba);
        }
        v
    }

    #[test]
    fn build_animated_emits_vp8x_anim_anmf_in_order() {
        let f0 = solid_frame(8, 8, [0xff, 0, 0, 0xff]);
        let f1 = solid_frame(8, 8, [0, 0xff, 0, 0xff]);
        let frames = [
            AnimFrame {
                width: 8,
                height: 8,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 100,
                blend: false,
                dispose_to_background: false,
                rgba: &f0,
            },
            AnimFrame {
                width: 8,
                height: 8,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 200,
                blend: false,
                dispose_to_background: false,
                rgba: &f1,
            },
        ];
        let out = build_animated_webp(8, 8, [0; 4], 0, &frames).expect("build");
        // RIFF / WEBP magic at the front.
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        // VP8X first, with ANIM bit (0x02) set.
        assert_eq!(&out[12..16], b"VP8X");
        assert_ne!(out[20] & 0x02, 0, "ANIM flag must be set in VP8X");
        // ANIM next.
        let vp8x_chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]) as usize;
        let anim_off = 12 + 8 + vp8x_chunk_len + (vp8x_chunk_len & 1);
        assert_eq!(&out[anim_off..anim_off + 4], b"ANIM");
        // First ANMF after ANIM.
        let anim_chunk_len = u32::from_le_bytes([
            out[anim_off + 4],
            out[anim_off + 5],
            out[anim_off + 6],
            out[anim_off + 7],
        ]) as usize;
        let anmf0_off = anim_off + 8 + anim_chunk_len + (anim_chunk_len & 1);
        assert_eq!(&out[anmf0_off..anmf0_off + 4], b"ANMF");
    }

    #[test]
    fn rejects_oversized_frame_bbox() {
        let f = solid_frame(8, 8, [0; 4]);
        let frames = [AnimFrame {
            width: 8,
            height: 8,
            x_offset: 4,
            y_offset: 4,
            duration_ms: 0,
            blend: false,
            dispose_to_background: false,
            rgba: &f,
        }];
        // 8x8 frame at (4,4) on a 8x8 canvas — extends past edge.
        let r = build_animated_webp(8, 8, [0; 4], 0, &frames);
        assert!(r.is_err(), "expected oversized-bbox to be rejected");
    }

    #[test]
    fn loop_count_and_background_round_trip_on_disk() {
        let f = solid_frame(4, 4, [0; 4]);
        let frames = [AnimFrame {
            width: 4,
            height: 4,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 1,
            blend: false,
            dispose_to_background: false,
            rgba: &f,
        }];
        let out = build_animated_webp(4, 4, [0x12, 0x34, 0x56, 0x78], 7, &frames).expect("build");
        let vp8x_chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]) as usize;
        let anim_off = 12 + 8 + vp8x_chunk_len + (vp8x_chunk_len & 1);
        // ANIM payload starts at anim_off + 8.
        let anim_payload = &out[anim_off + 8..anim_off + 8 + 6];
        assert_eq!(&anim_payload[0..4], &[0x12, 0x34, 0x56, 0x78]);
        let lc = u16::from_le_bytes([anim_payload[4], anim_payload[5]]);
        assert_eq!(lc, 7);
    }
}
