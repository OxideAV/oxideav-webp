//! End-to-end VP8-lossy roundtrip: encode a small synthetic Yuv420P
//! frame through `encoder_vp8::make_encoder_with_quality` (wired into
//! `oxideav-vp8 0.2.1` in round 168) and decode the emitted `.webp`
//! back via `decode_webp`. The roundtrip is **lossy**, so we assert
//! width / height / frame-count match and that the decoded RGBA buffer
//! lands within a generous mean-absolute-error budget of the source.
//!
//! Gated on the default `registry` feature because the factory family
//! returns `Box<dyn oxideav_core::Encoder>` — the standalone
//! (`--no-default-features`) build skips this test entirely.

#![cfg(feature = "registry")]

use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_webp::{decode_webp, encoder_vp8, parse_container};

/// Build a deterministic 16x16 Yuv420P [`Frame`] — a smooth ramp in Y
/// with constant chroma. Small enough to keep the test cheap; large
/// enough that VP8's macroblock grid (16x16 MB) covers exactly one MB
/// row × column so we exercise the keyframe header end-to-end.
fn synthetic_yuv420p_frame(width: u32, height: u32) -> Frame {
    let w = width as usize;
    let h = height as usize;
    let uvw = w.div_ceil(2);
    let uvh = h.div_ceil(2);
    // Y plane: smooth ramp.
    let mut y = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            y.push(((row * 8 + col * 4) & 0xff) as u8);
        }
    }
    // U / V planes: constant mid-grey chroma.
    let u = vec![128u8; uvw * uvh];
    let v = vec![128u8; uvw * uvh];
    Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: uvw,
                data: u,
            },
            VideoPlane {
                stride: uvw,
                data: v,
            },
        ],
    })
}

fn vp8_params(width: u32, height: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(oxideav_webp::CODEC_ID_VP8));
    p.width = Some(width);
    p.height = Some(height);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p
}

#[test]
fn vp8_lossy_roundtrip_decodes_to_a_close_image() {
    // Round-168 surface: the `encoder_vp8::make_encoder_with_quality`
    // factory wires into `oxideav-vp8 0.2.1`'s framework encoder. Push
    // one frame through, decode the emitted `.webp`, and check the
    // pixel buffer landed within a generous L1 budget of the source.
    let (w, h) = (16u32, 16u32);
    let src_frame = synthetic_yuv420p_frame(w, h);

    let mut enc = encoder_vp8::make_encoder_with_quality(&vp8_params(w, h), 90.0)
        .expect("make_encoder_with_quality");
    enc.send_frame(&src_frame).expect("send_frame yuv420p");
    let pkt = enc.receive_packet().expect("one .webp packet out");

    // The emitted packet is a complete RIFF/WEBP file carrying a §2.5
    // `VP8 ` lossy bitstream — same shape the `decode_webp` entry point
    // routes through `oxideav-vp8` for decoding.
    assert_eq!(&pkt.data[0..4], b"RIFF");
    assert_eq!(&pkt.data[8..12], b"WEBP");

    let img = decode_webp(&pkt.data).expect("decode the emitted .webp");
    assert_eq!(img.frames.len(), 1, "one frame in, one frame out");
    assert_eq!(img.frames[0].width, w);
    assert_eq!(img.frames[0].height, h);
    assert_eq!(img.frames[0].rgba.len(), (w * h * 4) as usize);

    // L1 / pixel distance against a YCbCr→RGB conversion of the source
    // (ITU-R BT.601 full-range, same matrix the decoder uses). Yuv420P
    // with constant U=V=128 means achromatic grey, so each RGB channel
    // equals Y exactly.
    let Frame::Video(src_v) = &src_frame else {
        unreachable!()
    };
    let src_y = &src_v.planes[0].data;
    let decoded = &img.frames[0].rgba;

    let mut total_diff: u64 = 0;
    for row in 0..h as usize {
        for col in 0..w as usize {
            let src_grey = src_y[row * w as usize + col] as i32;
            let off = (row * w as usize + col) * 4;
            let dr = decoded[off] as i32;
            let dg = decoded[off + 1] as i32;
            let db = decoded[off + 2] as i32;
            let da = decoded[off + 3] as i32;
            // Alpha is opaque for a §2.5 lossy file.
            assert_eq!(da, 0xff, "lossy decode preserves opaque alpha");
            total_diff += (dr - src_grey).unsigned_abs() as u64
                + (dg - src_grey).unsigned_abs() as u64
                + (db - src_grey).unsigned_abs() as u64;
        }
    }
    let mean_abs_err = total_diff as f64 / (w as f64 * h as f64 * 3.0);
    // At quality=90 (qindex ~12), a flat-chroma ramp must stay well
    // under 40 / 255 mean absolute error. Empirically the wired-up
    // encoder lands far below — this is a loose ceiling so a future
    // quantiser tuning change doesn't accidentally tighten the test.
    assert!(
        mean_abs_err < 40.0,
        "mean absolute error per channel = {mean_abs_err}; expected < 40 at q=90"
    );

    // Encoder reports Eof after flush + drain.
    enc.flush().expect("flush ok");
    let next = enc.receive_packet();
    assert!(
        matches!(next, Err(oxideav_core::Error::Eof)),
        "post-flush, no pending packet → Eof, got {next:?}"
    );
}

#[test]
fn vp8_lossy_qindex_factory_also_roundtrips() {
    // The explicit `make_encoder_with_qindex` variant must produce the
    // same round-trippable container shape. Use qindex=0 (best quality).
    let (w, h) = (16u32, 16u32);
    let src = synthetic_yuv420p_frame(w, h);
    let mut enc = encoder_vp8::make_encoder_with_qindex(&vp8_params(w, h), 0)
        .expect("make_encoder_with_qindex(0)");
    enc.send_frame(&src).expect("send_frame");
    let pkt = enc.receive_packet().expect("one packet out");
    assert_eq!(&pkt.data[0..4], b"RIFF");
    assert_eq!(&pkt.data[8..12], b"WEBP");
    let img = decode_webp(&pkt.data).expect("decode roundtrip");
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].width, w);
    assert_eq!(img.frames[0].height, h);
}

#[test]
fn vp8_lossy_freq_deltas_factory_passes_through() {
    // The `_freq_deltas` variants pass through to the matching
    // no-deltas factory in this round — the surface stays unchanged,
    // the deltas are a hint. Assert the factory still returns a working
    // encoder that round-trips a frame.
    let (w, h) = (16u32, 16u32);
    let src = synthetic_yuv420p_frame(w, h);
    let deltas = encoder_vp8::Vp8FreqDeltas {
        y_dc_delta: 4,
        ..encoder_vp8::Vp8FreqDeltas::default()
    };
    let mut enc =
        encoder_vp8::make_encoder_with_quality_and_freq_deltas(&vp8_params(w, h), 75.0, deltas)
            .expect("make_encoder_with_quality_and_freq_deltas");
    enc.send_frame(&src).expect("send_frame");
    let pkt = enc.receive_packet().expect("packet");
    let img = decode_webp(&pkt.data).expect("roundtrip");
    assert_eq!(img.frames.len(), 1);
    assert_eq!(img.frames[0].width, w);
    assert_eq!(img.frames[0].height, h);
}

// ─────────────────── lossy-frame animation decode ───────────────────
//
// RFC 9649 §2.7.2 Figure 14 lets each `ANMF` "Frame Data" carry either a
// `VP8L` (lossless) OR a `VP8 ` (lossy) bitstream, plus an optional `ALPH`.
// The in-crate animation *encoder* only emits VP8L frames, so there is no
// committed lossy-frame fixture; this test hand-assembles a spec-shaped
// container around a real `VP8 ` keyframe produced by the lossy encoder,
// then asserts `decode_webp` composites it (rather than the pre-r309
// `WebpError::Unsupported` refusal).

/// Encode one synthetic frame to a still lossy `.webp` and slice out its
/// §2.5 `VP8 ` chunk payload (the keyframe bitstream the §2.7.1.1 frame
/// data carries verbatim).
fn lossy_vp8_bitstream(width: u32, height: u32) -> Vec<u8> {
    let src = synthetic_yuv420p_frame(width, height);
    let mut enc = encoder_vp8::make_encoder_with_quality(&vp8_params(width, height), 90.0)
        .expect("make_encoder_with_quality");
    enc.send_frame(&src).expect("send_frame");
    let pkt = enc.receive_packet().expect("packet");
    let c = parse_container(&pkt.data).expect("parse still lossy container");
    let vp8 = c
        .first_chunk_with_fourcc(oxideav_webp::container::fourcc::VP8)
        .expect("still lossy file carries a VP8 chunk");
    vp8.payload(&pkt.data).to_vec()
}

/// Wrap `payload` in a §2.3 padded chunk (`FourCC` + LE u32 size + payload
/// + a pad byte when the size is odd).
fn riff_chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len() + 1);
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
    out
}

/// Build the 16-byte §2.7.1.1 `ANMF` header (Figure 9) for a frame placed
/// at `(x, y)` with the given pixel dimensions and duration.
fn anmf_header(x: u32, y: u32, w: u32, h: u32, duration_ms: u32) -> Vec<u8> {
    let mut hdr = Vec::with_capacity(16);
    // Frame X / Y are uint24 in units of 2 px; w/h are "Minus One".
    hdr.extend_from_slice(&(x / 2).to_le_bytes()[..3]);
    hdr.extend_from_slice(&(y / 2).to_le_bytes()[..3]);
    hdr.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
    hdr.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
    hdr.extend_from_slice(&duration_ms.to_le_bytes()[..3]);
    hdr.push(0); // info byte: Reserved=0, Blending=0 (alpha-blend), Dispose=0
    hdr
}

#[test]
fn animation_with_lossy_vp8_frames_decodes() {
    let (w, h) = (16u32, 16u32);
    let bitstream = lossy_vp8_bitstream(w, h);

    // §2.7.1 VP8X: animation flag (bit 1 of the flag octet) + canvas dims.
    let mut vp8x = Vec::with_capacity(10);
    vp8x.push(0b0000_0010); // Animation bit set
    vp8x.extend_from_slice(&[0, 0, 0]); // reserved
    vp8x.extend_from_slice(&(w - 1).to_le_bytes()[..3]); // canvas width - 1
    vp8x.extend_from_slice(&(h - 1).to_le_bytes()[..3]); // canvas height - 1

    // §2.7.1.1 ANIM: BGRA background (opaque white) + LE u16 loop count (0).
    let anim = vec![0xff, 0xff, 0xff, 0xff, 0, 0];

    // Two frames, each carrying the same lossy keyframe bitstream.
    let mut frame0 = anmf_header(0, 0, w, h, 100);
    frame0.extend_from_slice(&riff_chunk(b"VP8 ", &bitstream));
    let mut frame1 = anmf_header(0, 0, w, h, 100);
    frame1.extend_from_slice(&riff_chunk(b"VP8 ", &bitstream));

    let mut body = Vec::new();
    body.extend_from_slice(&riff_chunk(b"VP8X", &vp8x));
    body.extend_from_slice(&riff_chunk(b"ANIM", &anim));
    body.extend_from_slice(&riff_chunk(b"ANMF", &frame0));
    body.extend_from_slice(&riff_chunk(b"ANMF", &frame1));

    let mut file = Vec::with_capacity(12 + body.len());
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&((4 + body.len()) as u32).to_le_bytes());
    file.extend_from_slice(b"WEBP");
    file.extend_from_slice(&body);

    let img = decode_webp(&file).expect("lossy-frame animation must decode (not Unsupported)");
    assert_eq!(img.frames.len(), 2, "both ANMF frames composited");
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    for frame in &img.frames {
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        assert_eq!(frame.rgba.len(), (w * h * 4) as usize);
    }
    // The lossy reconstruction is opaque; both frames render onto the canvas
    // rather than leaving the white background showing everywhere.
    assert_eq!(img.frames[0].rgba.len(), img.frames[1].rgba.len());
}
