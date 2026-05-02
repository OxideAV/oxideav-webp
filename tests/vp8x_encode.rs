//! End-to-end tests for the RIFF / VP8X encoder output.
//!
//! Two behaviours are asserted here:
//!
//! 1. The VP8L encoder adapter emits a RIFF-wrapped `.webp` file (not a
//!    bare bitstream) and uses the extended `VP8X` layout whenever the
//!    source RGBA frame carries transparent pixels. The test picks
//!    `offset 12 == "VP8X"` as the canonical marker, matching the
//!    byte layout documented in `src/riff.rs`.
//! 2. A fully-opaque RGBA frame stays on the cheaper simple layout
//!    (`VP8L` chunk directly after `WEBP`) — we don't pay the VP8X
//!    overhead when it's not needed.
//!
//! Both paths also round-trip through `decode_webp` to make sure the
//! emitted file is parseable by our own decoder.

use oxideav_core::ContainerRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, VideoFrame, VideoPlane};
use oxideav_core::{CodecRegistry, Encoder};
use oxideav_webp::{decode_webp, CODEC_ID_VP8L};

const W: u32 = 32;
const H: u32 = 32;

fn make_rgba_frame(with_alpha: bool) -> VideoFrame {
    let mut data = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            data[i] = (x * 8) as u8;
            data[i + 1] = (y * 8) as u8;
            data[i + 2] = ((x + y) * 4) as u8;
            data[i + 3] = if with_alpha {
                // Produce a varying alpha plane so `has_alpha` detection
                // in the encoder kicks in (any pixel with a != 0xff
                // qualifies).
                ((x + y) * 4) as u8
            } else {
                0xff
            };
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (W as usize) * 4,
            data,
        }],
    }
}

fn make_vp8l_encoder() -> Box<dyn Encoder> {
    let mut codecs = CodecRegistry::new();
    let mut containers = ContainerRegistry::new();
    oxideav_webp::register(&mut codecs, &mut containers);
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_VP8L));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Rgba);
    codecs.make_encoder(&params).expect("make_encoder")
}

#[test]
fn vp8l_encoder_rgba_with_alpha_emits_vp8x_chunk() {
    let frame = make_rgba_frame(true);
    let mut enc = make_vp8l_encoder();
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive");
    let out = pkt.data;

    // Strict byte layout: RIFF + size + WEBP + VP8X fourcc at offset 12.
    assert!(
        out.len() >= 30,
        "output too short for VP8X layout: {} bytes",
        out.len()
    );
    assert_eq!(&out[0..4], b"RIFF", "missing RIFF magic at bytes 0..4");
    assert_eq!(
        &out[8..12],
        b"WEBP",
        "missing WEBP form type at bytes 8..12"
    );
    assert_eq!(
        &out[12..16],
        b"VP8X",
        "expected VP8X fourcc at offset 12 for an alpha-carrying RGBA image, \
         got {:?}",
        &out[12..16]
    );
    // VP8X flags byte (at offset 20) must have the ALPHA bit (0x10).
    assert!(
        out[20] & 0x10 != 0,
        "VP8X ALPHA flag not set in emitted chunk (flags byte = {:#x})",
        out[20]
    );

    // Round-trip through the public decoder — the emitted `.webp` file
    // must be parseable end-to-end.
    let img = decode_webp(&out).expect("decode roundtrip");
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
    assert_eq!(img.frames.len(), 1);
    let decoded = &img.frames[0].rgba;
    assert_eq!(decoded.len(), (W * H * 4) as usize);
    // Lossless: pixels must match byte-for-byte.
    assert_eq!(decoded, &frame.planes[0].data);
}

#[test]
fn vp8l_encoder_opaque_rgba_stays_on_simple_layout() {
    let frame = make_rgba_frame(false);
    let mut enc = make_vp8l_encoder();
    enc.send_frame(&Frame::Video(frame)).expect("send");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive");
    let out = pkt.data;

    assert_eq!(&out[0..4], b"RIFF");
    assert_eq!(&out[8..12], b"WEBP");
    assert_eq!(
        &out[12..16],
        b"VP8L",
        "fully-opaque RGBA frame should land on the simple VP8L layout, \
         got {:?}",
        &out[12..16]
    );

    // Also confirm `decode_webp` can parse the simple-layout output.
    let img = decode_webp(&out).expect("decode roundtrip");
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
    assert_eq!(img.frames.len(), 1);
}
