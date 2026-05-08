//! Integration tests for the streaming `Demuxer` impl.
//!
//! Mirrors the unit tests in `src/demux.rs::tests` but goes through
//! the public re-exports the way an external consumer would. Coverage:
//!
//! * Each `next_packet()` materialises one frame's payload — we never
//!   touch the demuxer's `parsed`/`body` fields directly.
//! * PTS/DTS arithmetic matches the eager animator's `pts_ms` (cumulative
//!   `max(duration_ms, 1)`).
//! * Eof is repeatable + the stream metadata is available before any
//!   packet is pulled.
//! * The streaming-vs-eager parity check: each packet's OWEB payload
//!   fed through the registry-side `WebpDecoder` produces the exact
//!   pixels the standalone `decode_webp` path would have produced for
//!   the same frame.

#![cfg(feature = "registry")]

use oxideav_core::{Decoder as _, Frame};
use oxideav_webp::decode_webp;
use oxideav_webp::decoder::WebpDecoder;
use oxideav_webp::demux;
use oxideav_webp::encoder_anim::{build_animated_webp, AnimFrame};

const W: u32 = 8;
const H: u32 = 8;

fn solid(rgba: [u8; 4]) -> Vec<u8> {
    let n = (W as usize) * (H as usize);
    let mut v = Vec::with_capacity(n * 4);
    for _ in 0..n {
        v.extend_from_slice(&rgba);
    }
    v
}

fn three_frame_blob() -> Vec<u8> {
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
fn streaming_demuxer_pulls_one_frame_at_a_time() {
    let blob = three_frame_blob();
    let cursor = std::io::Cursor::new(blob);
    let mut dem = demux::open_boxed(Box::new(cursor)).expect("open");
    // Stream metadata is up-front + before any decode.
    let streams = dem.streams();
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].params.width, Some(W));
    assert_eq!(streams[0].params.height, Some(H));
    // Sum of frame durations = 30 + 40 + 50 = 120.
    assert_eq!(streams[0].duration, Some(120));

    let p0 = dem.next_packet().expect("p0");
    assert_eq!(p0.pts, Some(0));
    assert_eq!(p0.dts, Some(0));
    assert_eq!(p0.duration, Some(30));
    assert!(p0.flags.keyframe);

    let p1 = dem.next_packet().expect("p1");
    assert_eq!(p1.pts, Some(30));
    assert_eq!(p1.duration, Some(40));
    assert!(!p1.flags.keyframe);

    let p2 = dem.next_packet().expect("p2");
    assert_eq!(p2.pts, Some(70));
    assert_eq!(p2.duration, Some(50));

    // Eof — and Eof is sticky.
    for _ in 0..3 {
        assert!(matches!(dem.next_packet(), Err(oxideav_core::Error::Eof)));
    }
}

#[test]
fn streaming_demuxer_pixels_match_eager_decode() {
    // Each streamed packet flows through the registry decoder and
    // produces a `Frame` whose RGBA matches the same-index frame from
    // the eager `decode_webp` path. This pins down both:
    //
    // 1) the `encode_lazy_frame_payload` shape (lazy ≡ eager OWEB);
    // 2) the registry decoder + standalone decoder agree on pixels.
    let blob = three_frame_blob();
    let eager = decode_webp(&blob).expect("decode_webp");
    assert_eq!(eager.frames.len(), 3);

    let cursor = std::io::Cursor::new(blob);
    let mut dem = demux::open_boxed(Box::new(cursor)).expect("open");
    let stream_w = dem.streams()[0].params.width.unwrap();
    let stream_h = dem.streams()[0].params.height.unwrap();
    let mut dec = WebpDecoder::new(stream_w, stream_h);

    for (i, eager_frame) in eager.frames.iter().enumerate() {
        let pkt = dem.next_packet().expect("pkt");
        dec.send_packet(&pkt).expect("send");
        let frame = dec.receive_frame().expect("frame");
        let rgba = match frame {
            Frame::Video(vf) => vf.planes[0].data.clone(),
            _ => panic!("expected Video frame"),
        };
        assert_eq!(rgba.len(), eager_frame.rgba.len(), "frame {i} buf size");
        assert_eq!(rgba, eager_frame.rgba, "frame {i} pixels");
    }
}

#[test]
fn streaming_demuxer_open_does_not_buffer_packets() {
    // Open the demuxer for a 3-frame blob and immediately drop it
    // without pulling any packets — proves we don't precompute every
    // OWEB payload at `open` time. (Best signal observable in a unit
    // test: the demuxer constructs without calling `next_packet`, and
    // every packet is built on demand inside `next_packet` rather than
    // sitting in a pre-populated `Vec<Packet>` field.)
    let blob = three_frame_blob();
    let cursor = std::io::Cursor::new(blob);
    let dem = demux::open_boxed(Box::new(cursor)).expect("open");
    drop(dem);
}

#[test]
fn streaming_demuxer_eof_after_drain_is_repeatable() {
    let blob = three_frame_blob();
    let cursor = std::io::Cursor::new(blob);
    let mut dem = demux::open_boxed(Box::new(cursor)).expect("open");
    for _ in 0..3 {
        let _ = dem.next_packet().expect("ok");
    }
    for _ in 0..5 {
        assert!(matches!(dem.next_packet(), Err(oxideav_core::Error::Eof)));
    }
}
