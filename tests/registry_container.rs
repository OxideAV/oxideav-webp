//! The WebP container through the framework registry (round 472):
//! probe → `open_demuxer` → `first_decoder`, and `first_encoder` →
//! `open_muxer`, pinned byte-for-byte against Layer 1 `decode` /
//! `decode_all` on every native layout the fixtures cover.

#![cfg(feature = "registry")]

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use oxideav_core::{
    CodecId, CodecParameters, ColorSignal, Error as CoreError, Frame as CoreFrame, Packet,
    PixelFormat, RuntimeContext, StreamInfo, TimeBase, VideoFrame,
};
use oxideav_webp::container_registry::{self, is_animation_stream, TIME_BASE};
use oxideav_webp::{decode, decode_all, info, WebpImage, CODEC_ID_VP8, CODEC_ID_VP8L};

const LOSSLESS_1X1: &[u8] = include_bytes!("data/lossless-1x1.webp");
const LOSSLESS_RGBA: &[u8] = include_bytes!("data/lossless-32x32-rgba.webp");
const LOSSLESS_PALETTED: &[u8] = include_bytes!("data/lossless-color-indexing-paletted.webp");
const LOSSY_1X1: &[u8] = include_bytes!("data/lossy-1x1.webp");
const LOSSY_ALPHA: &[u8] = include_bytes!("data/lossy-with-alpha-128x128.webp");
const WITH_ICC: &[u8] = include_bytes!("data/extended-with-icc-profile.webp");
const ANIM_RGB: &[u8] = include_bytes!("data/animated-3-frames-rgb.webp");
const ANIM_ALPHA: &[u8] = include_bytes!("data/animated-with-alpha.webp");

// ---- helpers ---------------------------------------------------------------

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_webp::register(&mut ctx);
    ctx
}

fn open(ctx: &RuntimeContext, bytes: &[u8]) -> Box<dyn oxideav_core::Demuxer> {
    let reader: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(bytes.to_vec()));
    container_registry::open_demuxer(reader, &ctx.codecs).expect("open_demuxer")
}

/// Everything the registry path yields for `bytes`, pumped as the
/// gateway does: send, drain to NeedMore, flush at Eof, drain to Eof.
fn pump(ctx: &RuntimeContext, bytes: &[u8]) -> (StreamInfo, Vec<Packet>, Vec<VideoFrame>) {
    let mut demux = open(ctx, bytes);
    assert_eq!(demux.streams().len(), 1);
    let stream = demux.streams()[0].clone();
    let mut dec = ctx
        .codecs
        .first_decoder(&stream.params)
        .expect("first_decoder");
    let mut packets = Vec::new();
    let mut frames = Vec::new();
    loop {
        match demux.next_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).expect("send_packet");
                packets.push(pkt);
                loop {
                    match dec.receive_frame() {
                        Ok(CoreFrame::Video(v)) => frames.push(v),
                        Ok(_) => panic!("non-video frame"),
                        Err(CoreError::NeedMore) => break,
                        Err(e) => panic!("receive_frame: {e}"),
                    }
                }
            }
            Err(CoreError::Eof) => break,
            Err(e) => panic!("next_packet: {e}"),
        }
    }
    dec.flush().unwrap();
    loop {
        match dec.receive_frame() {
            Ok(CoreFrame::Video(v)) => frames.push(v),
            Ok(_) => panic!("non-video frame"),
            Err(CoreError::Eof) => break,
            Err(e) => panic!("receive_frame after flush: {e}"),
        }
    }
    (stream, packets, frames)
}

#[derive(Clone, Default)]
struct SharedBuf(Arc<Mutex<Cursor<Vec<u8>>>>);

impl SharedBuf {
    fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap().get_ref().clone()
    }
}

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

impl Seek for SharedBuf {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}

fn mux(stream: &StreamInfo, packets: &[Packet]) -> Vec<u8> {
    let out = SharedBuf::default();
    let sink: Box<dyn oxideav_core::WriteSeek> = Box::new(out.clone());
    let mut mux =
        container_registry::open_muxer(sink, std::slice::from_ref(stream)).expect("open_muxer");
    mux.write_header().unwrap();
    for p in packets {
        mux.write_packet(p).unwrap();
    }
    mux.write_trailer().unwrap();
    out.bytes()
}

fn planes_of(v: &VideoFrame) -> Vec<(usize, Vec<u8>)> {
    v.image_planes()
        .iter()
        .map(|p| (p.stride, p.data.clone()))
        .collect()
}

fn planes_of_image(img: &WebpImage) -> Vec<(usize, Vec<u8>)> {
    img.planes
        .iter()
        .map(|p| (p.stride, p.data.clone()))
        .collect()
}

/// Encode `frames` (Rgba, same geometry) with the registered `webp_vp8l`
/// encoder, stamping per-picture timing the way the gateway does.
fn vp8l_packets(
    ctx: &RuntimeContext,
    frames: &[WebpImage],
    tb: TimeBase,
    durations: &[i64],
    options: &[(&str, &str)],
) -> (StreamInfo, Vec<Packet>) {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_VP8L));
    params.width = Some(frames[0].width);
    params.height = Some(frames[0].height);
    params.pixel_format = Some(frames[0].format.into());
    for (k, v) in options {
        params.options.insert(*k, *v);
    }
    let mut enc = ctx.codecs.first_encoder(&params).expect("first_encoder");
    let mut packets = Vec::new();
    let mut pts = 0i64;
    for (img, dur) in frames.iter().zip(durations) {
        let mut vf: VideoFrame = img.clone().into();
        vf.pts = Some(pts);
        enc.send_frame(&CoreFrame::Video(vf)).unwrap();
        let mut pkt = enc.receive_packet().unwrap();
        pkt.time_base = tb;
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(*dur);
        pts += dur;
        packets.push(pkt);
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));
    }
    enc.flush().unwrap();
    assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));
    let stream = StreamInfo {
        index: 0,
        time_base: tb,
        duration: None,
        start_time: Some(0),
        params: enc.output_params().clone(),
    };
    (stream, packets)
}

fn solid_rgba(w: u32, h: u32, px: [u8; 4]) -> WebpImage {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        data.extend_from_slice(&px);
    }
    WebpImage::from_rgba8(w, h, data).unwrap()
}

// ---- acceptance 1: probe ------------------------------------------------

#[test]
fn probe_names_webp_from_magic_alone_and_with_hint_and_rejects_foreign_files() {
    let ctx = ctx();
    for bytes in [LOSSLESS_1X1, LOSSY_1X1, ANIM_RGB] {
        for hint in [None, Some("webp")] {
            let mut cur = Cursor::new(bytes.to_vec());
            let name = ctx
                .containers
                .probe_input(&mut cur as &mut dyn oxideav_core::ReadSeek, hint)
                .unwrap();
            assert_eq!(name, "webp");
        }
    }
    for foreign in [
        b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec(),
        b"RIFF\x24\0\0\0WAVEfmt ".to_vec(), // a RIFF that is not WEBP
        b"GIF89a\x01\0\x01\0\0\0\0;".to_vec(),
    ] {
        let mut cur = Cursor::new(foreign.clone());
        assert!(ctx
            .containers
            .probe_input(&mut cur as &mut dyn oxideav_core::ReadSeek, None)
            .is_err());
        let reader: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(foreign));
        assert!(container_registry::open_demuxer(reader, &ctx.codecs).is_err());
    }
}

// ---- acceptance 2 + 3: stills, native layout, byte-exact vs Layer 1 -----

#[test]
fn still_layouts_match_layer1_decode_byte_for_byte() {
    let ctx = ctx();
    let fixtures: [(&str, &[u8]); 5] = [
        ("lossless rgba", LOSSLESS_RGBA),
        ("lossless paletted", LOSSLESS_PALETTED),
        ("VP8X + icc", WITH_ICC),
        ("lossy 4:2:0", LOSSY_1X1),
        ("lossy + ALPH", LOSSY_ALPHA),
    ];
    let mut seen: Vec<PixelFormat> = Vec::new();
    for (name, bytes) in fixtures {
        let header = info(bytes).unwrap();
        let expect = decode(bytes).unwrap();
        assert!(!header.is_animated, "{name}");
        let want_fmt: PixelFormat = header.format.into();
        let want_codec = if header.is_lossy {
            CODEC_ID_VP8
        } else {
            CODEC_ID_VP8L
        };
        let (stream, packets, frames) = pump(&ctx, bytes);

        let p = &stream.params;
        assert_eq!(p.codec_id.as_str(), want_codec, "{name}");
        assert_eq!(p.width, Some(header.width), "{name}");
        assert_eq!(p.height, Some(header.height), "{name}");
        assert_eq!(p.pixel_format, Some(want_fmt), "{name}");
        assert_eq!(p.pixel_format, Some(expect.format.into()), "{name}");
        // The format-defined colour: BT.601 limited for lossy (RFC 9649
        // §2.5), sRGB for lossless — the same signal Layer 1 reports.
        assert_eq!(p.color_signal, ColorSignal::from(header.color), "{name}");
        if matches!(want_fmt, PixelFormat::Yuv420P | PixelFormat::Yuva420P) {
            assert_eq!(p.color_signal.range, oxideav_core::ColorRange::Limited);
            assert_eq!(p.color_signal.matrix.code_point(), 6);
        } else {
            assert_eq!(p.color_signal, ColorSignal::srgb());
        }
        assert_eq!(stream.time_base, TIME_BASE);
        assert!(!is_animation_stream(p), "{name}");

        assert_eq!(packets.len(), 1, "{name}");
        assert_eq!(packets[0].data, bytes, "{name}: the whole file");
        assert_eq!(packets[0].pts, Some(0));
        assert!(packets[0].flags.keyframe);

        assert_eq!(frames.len(), 1, "{name}");
        let v = &frames[0];
        assert_eq!(planes_of(v), planes_of_image(&expect), "{name}: planes");
        assert_eq!(v.color_signal(), Some(p.color_signal), "{name}: colour");
        assert_eq!(v.pts, Some(0));
        let back = WebpImage::from_video_frame(v, p).unwrap();
        assert_eq!(back.planes, expect.planes, "{name}");
        assert_eq!(back.format, expect.format, "{name}");
        seen.push(want_fmt);
    }
    // The matrix covered all three native layouts.
    for f in [
        PixelFormat::Rgba,
        PixelFormat::Yuv420P,
        PixelFormat::Yuva420P,
    ] {
        assert!(seen.contains(&f), "{f:?} layout pinned");
    }
}

// ---- acceptance 4: animations = one packet per ANMF, ms ticks ------------

#[test]
fn animation_packets_carry_delays_and_compose_like_decode_all() {
    let ctx = ctx();
    for (name, bytes) in [("rgb", ANIM_RGB), ("alpha", ANIM_ALPHA)] {
        let header = info(bytes).unwrap();
        let expect = decode_all(bytes).unwrap();
        assert!(header.is_animated && expect.len() >= 2, "{name}");
        let (stream, packets, frames) = pump(&ctx, bytes);

        let p = &stream.params;
        assert_eq!(
            (p.width, p.height),
            (Some(header.width), Some(header.height))
        );
        assert_eq!(p.pixel_format, Some(PixelFormat::Rgba), "{name}");
        assert_eq!(p.color_signal, ColorSignal::srgb(), "{name}");
        assert_eq!(
            p.codec_id.as_str(),
            if header.is_lossy {
                CODEC_ID_VP8
            } else {
                CODEC_ID_VP8L
            },
            "{name}"
        );
        assert!(is_animation_stream(p), "{name}");
        assert_eq!(stream.time_base, TimeBase::new(1, 1000));

        // One packet per ANMF; pts cumulative; duration = the frame delay.
        assert_eq!(packets.len(), expect.len(), "{name}");
        let mut pts = 0i64;
        for (i, (pkt, want)) in packets.iter().zip(&expect).enumerate() {
            let ms = want.delay.unwrap().as_millis() as i64;
            assert_eq!(pkt.pts, Some(pts), "{name} packet {i}");
            assert_eq!(pkt.duration, Some(ms), "{name} packet {i}");
            assert_eq!(pkt.flags.keyframe, i == 0, "{name} packet {i}");
            pts += ms;
            // Every packet is a standalone one-frame animated file.
            assert!(oxideav_webp::probe(&pkt.data));
            let one = info(&pkt.data).unwrap();
            assert!(one.is_animated, "{name} packet {i}");
            assert_eq!(one.frames, 1, "{name} packet {i}");
            assert_eq!((one.width, one.height), (header.width, header.height));
            assert_eq!(one.loop_count, header.loop_count, "{name} packet {i}");
            assert!(
                decode(&pkt.data).is_ok(),
                "{name} packet {i} decodes standalone"
            );
        }
        assert_eq!(stream.duration, Some(pts), "{name}");

        // Frames: decode_all's composited canvases, byte for byte.
        assert_eq!(frames.len(), expect.len(), "{name}");
        for (i, (v, want)) in frames.iter().zip(&expect).enumerate() {
            assert_eq!(v.planes[0].stride, header.width as usize * 4);
            assert_eq!(
                v.planes[0].data,
                want.image.as_bytes().unwrap(),
                "{name} frame {i}"
            );
            assert_eq!(v.pts, packets[i].pts, "{name} frame {i}");
            assert_eq!(v.color_signal(), Some(ColorSignal::srgb()));
        }

        // Metadata: loop count + background colour.
        let demux = open(&ctx, bytes);
        let meta = demux.metadata();
        assert_eq!(
            meta.iter()
                .find(|(k, _)| k == "loop_count")
                .map(|(_, v)| v.as_str()),
            Some(header.loop_count.unwrap().to_string().as_str()),
            "{name}"
        );
        let bg = header.background_rgba.unwrap();
        assert_eq!(
            meta.iter()
                .find(|(k, _)| k == "background_color")
                .map(|(_, v)| v.as_str()),
            Some(format!("#{:02X}{:02X}{:02X}{:02X}", bg[0], bg[1], bg[2], bg[3]).as_str()),
            "{name}"
        );
        assert_eq!(demux.duration_micros(), Some(pts * 1000));
    }
}

#[test]
fn whole_file_decoder_is_unchanged_without_the_record() {
    // An animated file as one packet without the container's extradata
    // keeps today's behaviour: the first composited frame.
    let params = CodecParameters::video(CodecId::new("webp"));
    let mut dec = oxideav_webp::make_decoder(&params).unwrap();
    dec.send_packet(&Packet::new(0, TIME_BASE, ANIM_RGB.to_vec()))
        .unwrap();
    let CoreFrame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("non-video");
    };
    let first = decode(ANIM_RGB).unwrap();
    assert_eq!(v.planes[0].data, first.as_bytes().unwrap());
    assert!(matches!(dec.receive_frame(), Err(CoreError::NeedMore)));
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
}

// ---- acceptance 5: muxer ---------------------------------------------------

#[test]
fn muxer_writes_a_still_layer1_reads_back_identically() {
    let ctx = ctx();
    let img = decode(LOSSLESS_RGBA).unwrap();
    let (stream, packets) = vp8l_packets(
        &ctx,
        std::slice::from_ref(&img),
        TimeBase::new(1, 1000),
        &[0],
        &[],
    );
    let file = mux(&stream, &packets);
    assert_eq!(file, packets[0].data, "one packet is written verbatim");
    assert_eq!(decode(&file).unwrap().planes, img.planes);
    let (s, _, frames) = pump(&ctx, &file);
    assert_eq!(s.params.codec_id.as_str(), CODEC_ID_VP8L);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].planes[0].data, img.as_bytes().unwrap());
}

#[test]
fn muxer_merges_lossless_packets_into_an_animation_with_delays_and_loop() {
    let ctx = ctx();
    let frames = vec![
        solid_rgba(4, 2, [255, 0, 0, 255]),
        solid_rgba(4, 2, [0, 255, 0, 128]),
        solid_rgba(4, 2, [0, 0, 255, 0]),
    ];
    // Centiseconds in, milliseconds out: 12 → 120 ms, 7 → 70 ms, 100 → 1 s.
    let (stream, packets) = vp8l_packets(
        &ctx,
        &frames,
        TimeBase::new(1, 100),
        &[12, 7, 100],
        &[("loop_count", "5"), ("background_color", "#10203040")],
    );
    let file = mux(&stream, &packets);
    let header = info(&file).unwrap();
    assert!(header.is_animated);
    assert_eq!(header.frames, 3);
    assert_eq!(header.loop_count, Some(5));
    assert_eq!(header.background_rgba, Some([0x10, 0x20, 0x30, 0x40]));
    assert!(header.has_alpha);
    let got = decode_all(&file).unwrap();
    assert_eq!(got.len(), 3);
    for (i, (g, want)) in got.iter().zip(&frames).enumerate() {
        assert_eq!(g.image.as_bytes(), want.as_bytes(), "frame {i}");
    }
    assert_eq!(
        got.iter().map(|f| f.delay).collect::<Vec<_>>(),
        vec![
            Some(Duration::from_millis(120)),
            Some(Duration::from_millis(70)),
            Some(Duration::from_millis(1000)),
        ]
    );
    // The registry round trip: demux(mux(frames)) == frames.
    let (s, pk, vfs) = pump(&ctx, &file);
    assert!(is_animation_stream(&s.params));
    assert_eq!(
        pk.iter().map(|p| (p.pts, p.duration)).collect::<Vec<_>>(),
        vec![
            (Some(0), Some(120)),
            (Some(120), Some(70)),
            (Some(190), Some(1000))
        ]
    );
    for (i, (v, want)) in vfs.iter().zip(&frames).enumerate() {
        assert_eq!(
            v.planes[0].data,
            want.as_bytes().unwrap(),
            "registry frame {i}"
        );
    }
    // Defaults: loop forever, transparent black.
    let (stream, packets) = vp8l_packets(&ctx, &frames, TIME_BASE, &[1, 2, 3], &[]);
    let header = info(&mux(&stream, &packets)).unwrap();
    assert_eq!(header.loop_count, Some(0));
    assert_eq!(header.background_rgba, Some([0, 0, 0, 0]));
}

#[test]
fn muxer_accepts_lossy_packets_at_the_chunk_level() {
    // Layer 1 `encode_animation` refuses a quality (it re-encodes from
    // RGBA); the muxer only re-frames existing bitstreams, so a `webp_vp8`
    // packet becomes a lossy ANMF frame as RFC 9649 §2.7.1.1 allows.
    let ctx = ctx();
    let lossy = decode(LOSSY_1X1).unwrap();
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
    params.width = Some(1);
    params.height = Some(1);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    let mut enc = ctx.codecs.first_encoder(&params).expect("webp_vp8 encoder");
    let mut vf: VideoFrame = lossy.clone().into();
    vf.pts = Some(0);
    enc.send_frame(&CoreFrame::Video(vf)).unwrap();
    let mut lossy_pkt = enc.receive_packet().unwrap();
    lossy_pkt.time_base = TIME_BASE;
    lossy_pkt.pts = Some(0);
    lossy_pkt.duration = Some(40);
    assert_eq!(
        info(&lossy_pkt.data).unwrap().format,
        oxideav_webp::PixelFormat::Yuv420P
    );

    let lossless = decode(LOSSLESS_1X1).unwrap();
    let (_, mut ll) = vp8l_packets(&ctx, std::slice::from_ref(&lossless), TIME_BASE, &[60], &[]);
    let mut lossless_pkt = ll.remove(0);
    lossless_pkt.pts = Some(40);

    let stream = StreamInfo {
        index: 0,
        time_base: TIME_BASE,
        duration: None,
        start_time: Some(0),
        params: enc.output_params().clone(),
    };
    let file = mux(&stream, &[lossy_pkt.clone(), lossless_pkt]);
    let header = info(&file).unwrap();
    assert!(header.is_animated && header.is_lossy);
    let got = decode_all(&file).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(
        got[0].image.to_rgba8(),
        decode(&lossy_pkt.data).unwrap().to_rgba8()
    );
    assert_eq!(got[1].image.to_rgba8(), lossless.to_rgba8());
    assert_eq!(
        got.iter().map(|f| f.delay).collect::<Vec<_>>(),
        vec![
            Some(Duration::from_millis(40)),
            Some(Duration::from_millis(60))
        ]
    );
    let (s, _, vfs) = pump(&ctx, &file);
    assert_eq!(s.params.codec_id.as_str(), CODEC_ID_VP8);
    assert_eq!(vfs.len(), 2);
}

#[test]
fn demux_then_mux_then_demux_preserves_frames_timing_metadata() {
    let ctx = ctx();
    for (name, bytes) in [("rgb", ANIM_RGB), ("alpha", ANIM_ALPHA)] {
        let want = decode_all(bytes).unwrap();
        let header = info(bytes).unwrap();
        let (stream, packets, _) = pump(&ctx, bytes);
        let remuxed = mux(&stream, &packets);

        let got = decode_all(&remuxed).unwrap();
        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.image.planes, w.image.planes, "{name} frame {i}");
            assert_eq!(g.delay, w.delay, "{name} frame {i}");
        }
        let h2 = info(&remuxed).unwrap();
        assert_eq!(h2.loop_count, header.loop_count, "{name}");
        assert_eq!(h2.background_rgba, header.background_rgba, "{name}");
        assert_eq!(h2.has_alpha, header.has_alpha, "{name}");
        assert_eq!((h2.width, h2.height), (header.width, header.height));

        let (_, packets2, frames2) = pump(&ctx, &remuxed);
        assert_eq!(
            packets2
                .iter()
                .map(|p| (p.pts, p.duration))
                .collect::<Vec<_>>(),
            packets
                .iter()
                .map(|p| (p.pts, p.duration))
                .collect::<Vec<_>>(),
            "{name}"
        );
        for (i, (v, w)) in frames2.iter().zip(&want).enumerate() {
            assert_eq!(
                v.planes[0].data,
                w.image.as_bytes().unwrap(),
                "{name} frame {i}"
            );
        }
    }
    // The metadata chunks of a still survive the demux → mux hop.
    let (stream, packets, _) = pump(&ctx, WITH_ICC);
    let remuxed = mux(&stream, &packets);
    assert_eq!(remuxed, WITH_ICC);
}

#[test]
fn muxer_rejects_wrong_streams_and_non_webp_packets() {
    let mut params = CodecParameters::video(CodecId::new("png"));
    params.width = Some(1);
    params.height = Some(1);
    let stream = StreamInfo {
        index: 0,
        time_base: TIME_BASE,
        duration: None,
        start_time: Some(0),
        params,
    };
    let sink: Box<dyn oxideav_core::WriteSeek> = Box::new(Cursor::new(Vec::new()));
    assert!(container_registry::open_muxer(sink, std::slice::from_ref(&stream)).is_err());

    for id in ["webp", CODEC_ID_VP8L, CODEC_ID_VP8] {
        let mut s = stream.clone();
        s.params.codec_id = CodecId::new(id);
        let sink: Box<dyn oxideav_core::WriteSeek> = Box::new(Cursor::new(Vec::new()));
        let mut m = container_registry::open_muxer(sink, std::slice::from_ref(&s)).unwrap();
        assert!(m
            .write_packet(&Packet::new(0, TIME_BASE, LOSSLESS_1X1.to_vec()))
            .is_err());
        m.write_header().unwrap();
        assert!(m
            .write_packet(&Packet::new(0, TIME_BASE, b"not a webp".to_vec()))
            .is_err());
        assert!(m.write_trailer().is_err(), "no packets");
    }
    // A bad option is refused when the muxer opens.
    let mut s = stream.clone();
    s.params.codec_id = CodecId::new(CODEC_ID_VP8L);
    s.params.options.insert("loop_count", "many");
    let sink: Box<dyn oxideav_core::WriteSeek> = Box::new(Cursor::new(Vec::new()));
    assert!(container_registry::open_muxer(sink, std::slice::from_ref(&s)).is_err());
}

// ---- acceptance 6: register installs codecs AND container ------------------

#[test]
fn register_installs_codecs_and_container() {
    let mut ctx = RuntimeContext::new();
    oxideav_webp::__oxideav_entry(&mut ctx);
    assert!(ctx.codecs.has_decoder(&CodecId::new("webp")));
    assert!(ctx.codecs.has_encoder(&CodecId::new(CODEC_ID_VP8L)));
    assert!(ctx.codecs.has_encoder(&CodecId::new(CODEC_ID_VP8)));
    assert!(ctx.containers.demuxer_names().any(|n| n == "webp"));
    assert!(ctx.containers.muxer_names().any(|n| n == "webp"));
    assert_eq!(ctx.containers.container_for_extension("webp"), Some("webp"));
    let reader: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(LOSSY_1X1.to_vec()));
    let d = ctx
        .containers
        .open_demuxer("webp", reader, &ctx.codecs)
        .unwrap();
    assert_eq!(d.format_name(), "webp");
}

// ---- acceptance 7: hostile input never panics ------------------------------

#[test]
fn hostile_inputs_fail_cleanly() {
    let ctx = ctx();
    let mut cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        b"RIFF".to_vec(),
        b"RIFF\x04\0\0\0WEBP".to_vec(),
        ANIM_RGB[..20].to_vec(),
        ANIM_RGB[..ANIM_RGB.len() / 2].to_vec(),
        LOSSY_ALPHA[..LOSSY_ALPHA.len() - 3].to_vec(),
    ];
    // Absurd VP8X canvas (16384 × 16384, animation flag) with no frames,
    // then with a 1×1 ANMF: nothing is allocated eagerly.
    let mut absurd = b"RIFF\0\0\0\0WEBPVP8X\x0a\0\0\0\x02\0\0\0\xff\x3f\0\xff\x3f\0".to_vec();
    absurd.extend_from_slice(b"ANIM\x06\0\0\0\0\0\0\0\0\0");
    let size = (absurd.len() - 8) as u32;
    absurd[4..8].copy_from_slice(&size.to_le_bytes());
    cases.push(absurd.clone());
    let mut huge_frame = absurd.clone();
    huge_frame.extend_from_slice(b"ANMF\x12\0\0\0\0\0\0\0\0\0\xff\x3f\0\xff\x3f\0\x0a\0\0\x02");
    huge_frame.extend_from_slice(b"VP8L\x00\0\0\0");
    let size = (huge_frame.len() - 8) as u32;
    huge_frame[4..8].copy_from_slice(&size.to_le_bytes());
    cases.push(huge_frame);
    for bytes in &cases {
        let reader: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(bytes.clone()));
        if let Ok(mut d) = container_registry::open_demuxer(reader, &ctx.codecs) {
            let params = d.streams()[0].params.clone();
            let mut dec = ctx.codecs.first_decoder(&params).unwrap();
            while let Ok(p) = d.next_packet() {
                let _ = dec.send_packet(&p);
                while dec.receive_frame().is_ok() {}
            }
        }
    }
    // Zero-length / foreign packets into both decoder modes.
    let still = CodecParameters::video(CodecId::new(CODEC_ID_VP8L));
    let mut anim = still.clone();
    anim.extradata = vec![
        container_registry::EXTRADATA_VERSION,
        container_registry::EXTRADATA_ANIMATION,
    ];
    for params in [&still, &anim] {
        let mut dec = oxideav_webp::make_decoder(params).unwrap();
        let empty = dec.send_packet(&Packet::new(0, TIME_BASE, Vec::new()));
        // The whole-file decoder defers to receive_frame; either way an
        // error comes out and nothing panics.
        if empty.is_ok() {
            assert!(dec.receive_frame().is_err());
        }
        let _ = dec.send_packet(&Packet::new(0, TIME_BASE, LOSSLESS_1X1[..10].to_vec()));
        let _ = dec.receive_frame();
        dec.reset().unwrap();
        assert!(matches!(dec.receive_frame(), Err(CoreError::NeedMore)));
    }
    // Animation mode: a still packet (no ANIM) and a packet with another
    // canvas are refused; the canvas survives; reset forgets it.
    let (stream, packets, _) = pump(&ctx, ANIM_RGB);
    let want = decode_all(ANIM_RGB).unwrap();
    let mut dec = ctx.codecs.first_decoder(&stream.params).unwrap();
    dec.send_packet(&packets[0]).unwrap();
    assert!(dec.receive_frame().is_ok());
    assert!(dec
        .send_packet(&Packet::new(0, TIME_BASE, LOSSLESS_1X1.to_vec()))
        .is_err());
    // A one-frame animated packet on a 6×4 canvas (built through the
    // muxer) does not belong to this stream's canvas.
    let hdr = info(ANIM_RGB).unwrap();
    assert_ne!((hdr.width, hdr.height), (6, 4));
    let small = [
        solid_rgba(6, 4, [1, 2, 3, 255]),
        solid_rgba(6, 4, [4, 5, 6, 255]),
    ];
    let (s6, p6) = vp8l_packets(&ctx, &small, TIME_BASE, &[10, 10], &[]);
    let (_, other, _) = pump(&ctx, &mux(&s6, &p6));
    assert!(dec.send_packet(&other[0]).is_err(), "different canvas");
    dec.send_packet(&packets[1]).unwrap();
    let CoreFrame::Video(v) = dec.receive_frame().unwrap() else {
        panic!()
    };
    assert_eq!(v.planes[0].data, want[1].image.as_bytes().unwrap());
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Err(CoreError::Eof)));
    dec.reset().unwrap();
    dec.send_packet(&packets[0]).unwrap();
    let CoreFrame::Video(v) = dec.receive_frame().unwrap() else {
        panic!()
    };
    assert_eq!(v.planes[0].data, want[0].image.as_bytes().unwrap());
}
