#![no_main]

//! Structure-aware hostility harness for the §2.7.1.1 / §2.7.2 animation
//! **compositor** — `decode_webp` → `decode_animation` — fed *raw*
//! attacker-assembled containers rather than encoder output.
//!
//! ## Why this harness
//!
//! Every existing animation harness (`roundtrip_animated`,
//! `roundtrip_anim_modes`) drives the compositor with files the in-crate
//! *encoder* produced: full-canvas or exact dirty-rect frames whose
//! `ANMF` `Frame X/Y/W/H` fields always agree with the inner bitstream
//! dimensions, whose rectangles always fit the canvas, and whose `VP8X`
//! canvas is always sized to cover every frame. That leaves the
//! compositor's *defensive* surface — the branches that reject or clamp
//! a **malformed** animation — unfuzzed through the public entry point:
//!
//!   * a §2.7.1.1 `ANMF` rectangle whose `(Frame X * 2 + frame width)`
//!     or `(Frame Y * 2 + frame height)` overflows the §2.7.1 `VP8X`
//!     canvas (`right > canvas_w` / `bottom > canvas_h` — the
//!     `WebpError::InvalidData` refusal), including the `checked_add`
//!     overflow guard on the offset+dimension sum;
//!   * a `VP8X` canvas whose declared dimensions *disagree* with what
//!     the frames need — larger (unwritten canvas regions keep the
//!     `ANIM` background) or smaller (frame rejected);
//!   * an `ANIM` chunk with **zero** `ANMF` frames (the
//!     `frames.is_empty()` refusal);
//!   * a missing `VP8X` alongside an `ANIM` (the `ok_or` refusal);
//!   * every §2.7.1.1 dispose (`None` / `Background`) × blend
//!     (`Overwrite` / `AlphaBlend`) combination applied at an
//!     attacker-chosen sub-rectangle, so the `fill_canvas_rect` /
//!     `blit_rect_overwrite` / `blit_rect_alpha_blend` inner loops run
//!     over adversarial placements;
//!   * an optional §2.7.1.2 `ALPH` sub-chunk carrying **raw fuzz bytes**
//!     layered over the frame — exercising `alph::decode_alpha` on
//!     hostile input and the `plane.len()` length-guard that gates the
//!     overlay;
//!   * `zero`-duration frames and arbitrary 24-bit durations.
//!
//! The inner per-frame bitstream is always a *valid* §2.6 `VP8L` stream
//! (built with `encode_vp8l_argb` from a fuzz-drawn pixel pool) so the
//! mutator spends its budget on the container/compositor framing rather
//! than re-deriving a decodable entropy stream — but the `ANMF` header
//! fields wrapping it are fully attacker-controlled and free to
//! contradict the bitstream's own §3.4 dimensions.
//!
//! ## Contract under test
//!
//! `decode_webp` must **always return a `Result`** on any assembled
//! byte string — no panic, no debug-build integer overflow, no
//! out-of-bounds canvas index, no allocation sized by an unbounded
//! header field. On `Ok`, the §2.7.1.1 flat-canvas carrier invariant
//! holds on every composited frame: `rgba.len() == width * height * 4`,
//! every frame carries the canvas dimensions, and the decode is
//! deterministic over the same bytes. This harness asserts only those
//! universal invariants — it is a hostility/DoS harness, not a
//! pixel-equality oracle (that role is covered by `roundtrip_anim_modes`
//! over well-formed input).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::container::fourcc;
use oxideav_webp::{decode_webp, encode_vp8l_argb};

/// Canvas / frame dimension ceiling — keeps every assembled animation's
/// working set tiny so iterations stay millisecond-scale.
const MAX_DIM: u32 = 48;
/// Frame-count ceiling.
const MAX_FRAMES: usize = 5;

/// A little-cursor byte reader over the fuzz buffer.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }
    fn u8(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }
    fn u16(&mut self) -> u16 {
        u16::from(self.u8()) | (u16::from(self.u8()) << 8)
    }
    /// Take up to `n` raw bytes (fewer if the buffer is exhausted).
    fn take(&mut self, n: usize) -> &'a [u8] {
        let start = self.pos.min(self.data.len());
        let end = (start + n).min(self.data.len());
        self.pos = end;
        &self.data[start..end]
    }
}

/// Little-endian uint24 encoder for the §2.7.1 / §2.7.1.1 24-bit fields.
fn u24_le(v: u32) -> [u8; 3] {
    [
        (v & 0xFF) as u8,
        ((v >> 8) & 0xFF) as u8,
        ((v >> 16) & 0xFF) as u8,
    ]
}

/// Append a §2.3 RIFF chunk (`FourCC` + LE uint32 `Size` + payload +
/// optional pad byte when `Size` is odd) to `out`.
fn push_chunk(out: &mut Vec<u8>, cc: [u8; 4], payload: &[u8]) {
    out.extend_from_slice(&cc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        out.push(0);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let mut r = Reader::new(data);

    // §2.7.1 VP8X canvas: attacker-chosen, may under- or over-cover the
    // frames below. `+ 1` keeps it >= 1; capped at MAX_DIM so the canvas
    // buffer stays tiny.
    let canvas_w = (u32::from(r.u16()) % MAX_DIM) + 1;
    let canvas_h = (u32::from(r.u16()) % MAX_DIM) + 1;

    // §2.7.1.1 ANIM: global background colour + loop count.
    let bg = [r.u8(), r.u8(), r.u8(), r.u8()];
    let loop_count = r.u16();

    let control = r.u8();
    let frame_count = (usize::from(r.u8()) % (MAX_FRAMES + 1)).min(MAX_FRAMES);
    // `omit_vp8x` (bit 0) exercises the missing-VP8X-with-ANIM refusal.
    let omit_vp8x = control & 0b0000_0001 != 0;

    // Assemble the §2.7.1 extended body: VP8X (canvas) + ANIM (carrier)
    // + a run of ANMF frames.
    let mut body: Vec<u8> = Vec::new();

    if !omit_vp8x {
        let mut vp8x = Vec::with_capacity(10);
        vp8x.push(0b0000_0010); // §2.7.1 A (animation) flag set.
        vp8x.extend_from_slice(&[0, 0, 0]); // reserved 24-bit field.
        vp8x.extend_from_slice(&u24_le(canvas_w - 1));
        vp8x.extend_from_slice(&u24_le(canvas_h - 1));
        push_chunk(&mut body, fourcc::VP8X, &vp8x);
    }

    let mut anim = Vec::with_capacity(6);
    anim.extend_from_slice(&bg); // Background Color (BGRA on disk).
    anim.extend_from_slice(&loop_count.to_le_bytes());
    push_chunk(&mut body, fourcc::ANIM, &anim);

    for _ in 0..frame_count {
        // §2.7.1.1 ANMF header fields — every one attacker-controlled.
        // Frame X/Y are stored halved; the decoder doubles them, so the
        // placement can land the rect anywhere in (and past) the canvas.
        let frame_x = u32::from(r.u8()) % MAX_DIM;
        let frame_y = u32::from(r.u8()) % MAX_DIM;
        // Inner bitstream dimensions — kept small and valid.
        let sub_w = (u32::from(r.u8()) % MAX_DIM) + 1;
        let sub_h = (u32::from(r.u8()) % MAX_DIM) + 1;
        // ANMF Frame W/H are set INDEPENDENTLY of the inner bitstream
        // (the decoder derives frame extent from the bitstream, so this
        // deliberately contradicts it).
        let anmf_w = (u32::from(r.u8()) % MAX_DIM) + 1;
        let anmf_h = (u32::from(r.u8()) % MAX_DIM) + 1;
        let duration = u32::from(r.u16()); // includes zero-duration.
        let info = r.u8(); // Reserved | B (blend) | D (dispose).
        let add_alph = info & 0b0100_0000 != 0;

        // Valid inner §2.6 VP8L bitstream from a fuzz-drawn pixel pool.
        let pixel_count = (sub_w * sub_h) as usize;
        let mut argb = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            let seed = r.data.get(r.pos + (i & 0x7)).copied().unwrap_or(0);
            argb.push(0xFF00_0000 | (u32::from(seed) * 0x0001_0101));
        }
        let Ok(vp8l) = encode_vp8l_argb(&argb, sub_w, sub_h) else {
            continue;
        };

        // Frame Data sub-RIFF: optional ALPH (raw fuzz bytes) + VP8L.
        let mut frame_data: Vec<u8> = Vec::new();
        if add_alph {
            let alph_len = usize::from(r.u8()) % 24;
            let alph_payload = r.take(alph_len).to_vec();
            push_chunk(&mut frame_data, fourcc::ALPH, &alph_payload);
        }
        push_chunk(&mut frame_data, fourcc::VP8L, &vp8l);

        // 16-byte §2.7.1.1 ANMF header + Frame Data.
        let mut anmf: Vec<u8> = Vec::with_capacity(16 + frame_data.len());
        anmf.extend_from_slice(&u24_le(frame_x));
        anmf.extend_from_slice(&u24_le(frame_y));
        anmf.extend_from_slice(&u24_le(anmf_w - 1));
        anmf.extend_from_slice(&u24_le(anmf_h - 1));
        anmf.extend_from_slice(&u24_le(duration));
        anmf.push(info);
        anmf.extend_from_slice(&frame_data);
        push_chunk(&mut body, fourcc::ANMF, &anmf);
    }

    // Wrap the body in the §2.4 RIFF/WEBP file header.
    let mut file: Vec<u8> = Vec::with_capacity(12 + body.len());
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&((4 + body.len()) as u32).to_le_bytes());
    file.extend_from_slice(b"WEBP");
    file.extend_from_slice(&body);

    // The Result contract is unconditional: decode must never panic.
    let Ok(img) = decode_webp(&file) else {
        return;
    };

    // §2.7.1.1 flat-canvas carrier invariant on every composited frame.
    for (i, frame) in img.frames.iter().enumerate() {
        let expected = (frame.width as usize)
            .checked_mul(frame.height as usize)
            .and_then(|n| n.checked_mul(4))
            .expect("frame dimensions must not overflow usize");
        assert_eq!(
            frame.rgba.len(),
            expected,
            "compose_animation frame {i}: RGBA buffer must be width * height * 4",
        );
        assert_eq!(
            (frame.width, frame.height),
            (img.width, img.height),
            "compose_animation frame {i}: every composited frame is a full-canvas snapshot",
        );
    }

    // Determinism: a second decode of the same bytes reproduces the
    // frame count + canvas + pixels exactly.
    let replay = decode_webp(&file).expect("a decodable animation must replay successfully");
    assert_eq!(
        replay.frames.len(),
        img.frames.len(),
        "deterministic frame count"
    );
    assert_eq!(
        (replay.width, replay.height),
        (img.width, img.height),
        "deterministic canvas dimensions",
    );
    for (i, (a, b)) in replay.frames.iter().zip(img.frames.iter()).enumerate() {
        assert_eq!(
            a.rgba, b.rgba,
            "compose_animation frame {i}: deterministic pixels"
        );
        assert_eq!(
            a.duration_ms, b.duration_ms,
            "compose_animation frame {i}: deterministic duration"
        );
    }
});
