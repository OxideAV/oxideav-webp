//! The registered encoders' typed options: `effort` / `threads` on
//! `webp_vp8l`, plus `qindex` / `quality` on `webp_vp8`, through the
//! options bag, the typed factories and the execution context.

#![cfg(feature = "registry")]

mod common;

use oxideav_core::{
    CodecId, CodecParameters, Encoder, ExecutionContext, Frame, PixelFormat, RuntimeContext,
    VideoFrame, VideoPlane,
};
use oxideav_webp::encoder_vp8::{self, Vp8LossyEncoder, WebpVp8EncoderOptions};
use oxideav_webp::{CODEC_ID_VP8, CODEC_ID_VP8L, EncodeOptions, WebpVp8lEncoderOptions};

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_webp::register(&mut ctx);
    ctx
}

fn params(codec: &str, w: u32, h: u32, format: PixelFormat) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(codec));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(format);
    params
}

/// Limited-range 4:2:0 planes of the photo-like test picture.
fn yuv_planes(w: u32, h: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let rgba = common::photo_rgba(w, h);
    let image = oxideav_webp::decode(
        &oxideav_webp::encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_quality(100.0))
            .expect("encode"),
    )
    .expect("decode");
    let mut planes = image.planes.into_iter().map(|p| p.data);
    (
        planes.next().expect("Y"),
        planes.next().expect("U"),
        planes.next().expect("V"),
    )
}

fn yuv_frame(w: u32, (y, u, v): &(Vec<u8>, Vec<u8>, Vec<u8>)) -> Frame {
    let (ys, cs) = (w as usize, (w as usize).div_ceil(2));
    Frame::Video(VideoFrame {
        pts: Some(7),
        planes: vec![
            VideoPlane {
                stride: ys,
                data: y.clone(),
            },
            VideoPlane {
                stride: cs,
                data: u.clone(),
            },
            VideoPlane {
                stride: cs,
                data: v.clone(),
            },
        ],
    })
}

fn one_packet(mut encoder: Box<dyn Encoder>, frame: &Frame) -> Vec<u8> {
    encoder.send_frame(frame).expect("send_frame");
    let packet = encoder.receive_packet().expect("packet");
    assert_eq!(packet.pts, Some(7));
    assert!(packet.flags.keyframe);
    packet.data
}

#[test]
fn schemas_are_published() {
    let ctx = ctx();
    for (codec, keys) in [
        (
            CODEC_ID_VP8,
            &["effort", "threads", "qindex", "quality", "alpha_effort"][..],
        ),
        (CODEC_ID_VP8L, &["effort", "threads"][..]),
    ] {
        let schema = ctx
            .codecs
            .encoder_options_schema(&CodecId::new(codec))
            .unwrap_or_else(|| panic!("{codec}: no encoder options schema"));
        for key in keys {
            assert!(
                schema.iter().any(|f| f.name == *key),
                "{codec}: schema lacks '{key}'"
            );
        }
    }
}

#[test]
fn lossy_options_bag_matches_the_direct_encoder() {
    let ctx = ctx();
    let (w, h) = (176u32, 120u32);
    let planes = yuv_planes(w, h);
    let frame = yuv_frame(w, &planes);
    let (ys, cs) = (w as usize, (w as usize).div_ceil(2));

    // An empty bag is `make_encoder`: quantiser index 32, effort 4.
    let base = params(CODEC_ID_VP8, w, h, PixelFormat::Yuv420P);
    let default = one_packet(ctx.codecs.first_encoder(&base).expect("encoder"), &frame);
    assert_eq!(
        default,
        one_packet(encoder_vp8::make_encoder(&base).expect("encoder"), &frame)
    );
    assert_eq!(
        default,
        Vp8LossyEncoder::with_qindex(32)
            .encode_yuv420(w, h, &planes.0, ys, &planes.1, &planes.2, cs)
            .expect("direct")
    );

    // `effort` and `qindex` reach the encoder; every level is its own
    // bitstream up to the exhaustive search.
    let mut seen = Vec::new();
    for effort in 0..=6u8 {
        let mut with_options = base.clone();
        with_options.options = with_options
            .options
            .set("qindex", "25")
            .set("effort", effort.to_string());
        let file = one_packet(
            ctx.codecs.first_encoder(&with_options).expect("encoder"),
            &frame,
        );
        let direct = Vp8LossyEncoder::with_qindex(25)
            .with_effort(effort)
            .encode_yuv420(w, h, &planes.0, ys, &planes.1, &planes.2, cs)
            .expect("direct");
        assert_eq!(file, direct, "effort {effort}: registry and direct bytes");
        let typed = one_packet(
            encoder_vp8::make_encoder_with_options(
                &base,
                &WebpVp8EncoderOptions::new()
                    .with_qindex(25)
                    .with_effort(effort),
            )
            .expect("encoder"),
            &frame,
        );
        assert_eq!(file, typed, "effort {effort}: bag and typed options");
        let decoded = oxideav_webp::decode(&file).expect("decode");
        assert_eq!((decoded.width, decoded.height), (w, h));
        seen.push(file);
    }
    assert_eq!(seen[3], seen[4], "efforts 3 and 4 are one encode");
    for effort in [0usize, 1, 2, 5, 6] {
        assert_ne!(seen[effort], seen[4], "effort {effort} differs from 4");
    }

    // `quality` is the Layer 1 quality: the bytes of `encode` on the
    // same planes.
    let mut with_quality = base.clone();
    with_quality.options = with_quality.options.set("quality", "80");
    let file = one_packet(
        ctx.codecs.first_encoder(&with_quality).expect("encoder"),
        &frame,
    );
    let image = oxideav_webp::WebpImage::from_yuv420(
        w,
        h,
        planes.0.clone(),
        planes.1.clone(),
        planes.2.clone(),
    )
    .expect("image");
    let layer1 =
        oxideav_webp::encode(&image, &EncodeOptions::default().with_quality(80.0)).expect("encode");
    assert_eq!(file, layer1);
}

#[test]
fn lossy_thread_budget_never_changes_the_packet() {
    let ctx = ctx();
    let (w, h) = (320u32, 208u32);
    let planes = yuv_planes(w, h);
    let frame = yuv_frame(w, &planes);
    for effort in [0u8, 2, 4, 5] {
        let mut base = params(CODEC_ID_VP8, w, h, PixelFormat::Yuv420P);
        base.options = base
            .options
            .set("quality", "75")
            .set("effort", effort.to_string());
        let serial = one_packet(ctx.codecs.first_encoder(&base).expect("encoder"), &frame);
        // The execution context grants workers …
        let mut granted = ctx.codecs.first_encoder(&base).expect("encoder");
        granted.set_execution_context(&ExecutionContext::with_threads(6));
        assert!(
            one_packet(granted, &frame) == serial,
            "effort {effort}: context"
        );
        // … and an explicit `threads` option names its own budget.
        let mut explicit = base.clone();
        explicit.options = explicit.options.set("threads", "3");
        let mut encoder = ctx.codecs.first_encoder(&explicit).expect("encoder");
        encoder.set_execution_context(&ExecutionContext::serial());
        assert!(
            one_packet(encoder, &frame) == serial,
            "effort {effort}: option"
        );
    }
}

#[test]
fn lossless_options_bag_matches_layer_one() {
    let ctx = ctx();
    let (w, h) = (96u32, 72u32);
    let rgba = common::photo_rgba(w, h);
    let frame = Frame::Video(VideoFrame {
        pts: Some(7),
        planes: vec![VideoPlane {
            stride: w as usize * 4,
            data: rgba.clone(),
        }],
    });
    let base = params(CODEC_ID_VP8L, w, h, PixelFormat::Rgba);
    let mut sizes = Vec::new();
    for effort in [0u8, 2, 4, 6] {
        let layer1 =
            oxideav_webp::encode_rgba8(w, h, &rgba, &EncodeOptions::default().with_effort(effort))
                .expect("encode");
        let mut with_options = base.clone();
        with_options.options = with_options.options.set("effort", effort.to_string());
        let file = one_packet(
            ctx.codecs.first_encoder(&with_options).expect("encoder"),
            &frame,
        );
        assert!(
            file == layer1,
            "effort {effort}: registry and Layer 1 bytes"
        );
        let mut threaded = with_options.clone();
        threaded.options = threaded.options.set("threads", "4");
        let file = one_packet(
            ctx.codecs.first_encoder(&threaded).expect("encoder"),
            &frame,
        );
        assert!(file == layer1, "effort {effort}: 4 threads");
        let typed = one_packet(
            oxideav_webp::registry::make_encoder_with_options(
                &base,
                &WebpVp8lEncoderOptions::new()
                    .with_effort(effort)
                    .with_threads(2),
            )
            .expect("encoder"),
            &frame,
        );
        assert!(typed == layer1, "effort {effort}: typed options");
        assert_eq!(
            oxideav_webp::decode(&file).expect("decode").to_rgba8(),
            rgba,
            "effort {effort}: exactly lossless"
        );
        sizes.push(file.len());
    }
    assert!(
        sizes[0] >= sizes[3],
        "effort 0 is not smaller than effort 6"
    );
}

#[test]
fn bad_options_are_refused_and_muxer_keys_pass() {
    let ctx = ctx();
    for (codec, format) in [
        (CODEC_ID_VP8, PixelFormat::Yuv420P),
        (CODEC_ID_VP8L, PixelFormat::Rgba),
    ] {
        let base = params(codec, 16, 16, format);
        for (key, value) in [
            ("effort", "7"),
            ("effort", "fast"),
            ("threads", "-1"),
            ("no_such_option", "1"),
        ] {
            let mut bad = base.clone();
            bad.options = bad.options.set(key, value);
            assert!(
                ctx.codecs.first_encoder(&bad).is_err(),
                "{codec}: {key}={value} accepted"
            );
        }
        // The muxer reads these from the same stream parameters.
        let mut shared = base.clone();
        shared.options = shared
            .options
            .set("loop_count", "3")
            .set("background_color", "#00000000");
        assert!(
            ctx.codecs.first_encoder(&shared).is_ok(),
            "{codec}: muxer keys"
        );
    }
    let mut bad = params(CODEC_ID_VP8, 16, 16, PixelFormat::Yuv420P);
    bad.options = bad.options.set("qindex", "128");
    assert!(
        ctx.codecs.first_encoder(&bad).is_err(),
        "qindex 128 accepted"
    );
}

/// A stream-parameters event re-sizes the lossy encoder: every packet is
/// a key frame that restates its size.
#[test]
fn lossy_encoder_follows_a_size_change() {
    let ctx = ctx();
    let small = yuv_planes(32, 24);
    let large = yuv_planes(64, 40);
    let mut encoder = ctx
        .codecs
        .first_encoder(&params(CODEC_ID_VP8, 32, 24, PixelFormat::Yuv420P))
        .expect("encoder");
    encoder.send_frame(&yuv_frame(32, &small)).expect("send");
    let first = encoder.receive_packet().expect("packet");
    let header = oxideav_webp::info(&first.data).expect("info");
    assert_eq!((header.width, header.height), (32, 24));

    let resized = params(CODEC_ID_VP8, 64, 40, PixelFormat::Yuv420P);
    encoder
        .send_frame(&Frame::StreamParams(Box::new(resized)))
        .expect("stream params");
    assert_eq!(encoder.output_params().width, Some(64));
    assert_eq!(encoder.output_params().height, Some(40));
    encoder.send_frame(&yuv_frame(64, &large)).expect("send");
    let second = encoder.receive_packet().expect("packet");
    let header = oxideav_webp::info(&second.data).expect("info");
    assert_eq!((header.width, header.height), (64, 40));

    // A layout the encoder cannot take is refused, and the old size
    // stays.
    let gray = params(CODEC_ID_VP8, 64, 40, PixelFormat::Gray8);
    assert!(
        encoder
            .send_frame(&Frame::StreamParams(Box::new(gray)))
            .is_err()
    );
    // A frame smaller than the configured size is an error, not a panic.
    assert!(encoder.send_frame(&yuv_frame(32, &small)).is_err());
}

/// The lossy encoder takes packed RGB(A) and 4:2:0 planes with an alpha
/// plane too, and writes what Layer 1 writes for the same pixels.
#[test]
fn lossy_encoder_takes_rgb_and_alpha_layouts() {
    let ctx = ctx();
    let (w, h) = (72u32, 56u32);
    let mut rgba = common::photo_rgba(w, h);
    for (i, px) in rgba.chunks_mut(4).enumerate() {
        px[3] = if i % 72 < 30 { 255 } else { (i % 251) as u8 };
    }
    let rgb: Vec<u8> = rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let layer1 = EncodeOptions::default().with_quality(70.0).with_effort(2);
    let with_options = |format| {
        let mut p = params(CODEC_ID_VP8, w, h, format);
        p.options = p.options.set("quality", "70").set("effort", "2");
        p
    };
    let packed = |stride: usize, row: usize, data: &[u8]| {
        // A padded plane: `stride` bytes per row, `row` of them used.
        let mut out = vec![0x5au8; stride * h as usize];
        for y in 0..h as usize {
            out[y * stride..y * stride + row].copy_from_slice(&data[y * row..(y + 1) * row]);
        }
        Frame::Video(VideoFrame {
            pts: Some(7),
            planes: vec![VideoPlane { stride, data: out }],
        })
    };

    // RGBA with alpha: VP8X + ALPH + VP8, alpha exact.
    let file = one_packet(
        ctx.codecs
            .first_encoder(&with_options(PixelFormat::Rgba))
            .expect("encoder"),
        &packed(w as usize * 4 + 8, w as usize * 4, &rgba),
    );
    assert!(file == oxideav_webp::encode_rgba8(w, h, &rgba, &layer1).expect("encode"));
    let decoded = oxideav_webp::decode(&file).expect("decode");
    assert_eq!(decoded.format, oxideav_webp::PixelFormat::Yuva420P);
    let alpha: Vec<u8> = rgba.chunks(4).map(|p| p[3]).collect();
    assert_eq!(decoded.planes[3].data, alpha);

    // `alpha_effort` sets the alpha plane's coder alone.
    let mut tuned = with_options(PixelFormat::Rgba);
    tuned.options = tuned.options.set("effort", "4").set("alpha_effort", "1");
    let file = one_packet(
        ctx.codecs.first_encoder(&tuned).expect("encoder"),
        &packed(w as usize * 4, w as usize * 4, &rgba),
    );
    let layer1_tuned = EncodeOptions::default()
        .with_quality(70.0)
        .with_alpha_effort(1);
    assert!(file == oxideav_webp::encode_rgba8(w, h, &rgba, &layer1_tuned).expect("encode"));
    let mut bad = with_options(PixelFormat::Rgba);
    bad.options = bad.options.set("alpha_effort", "7");
    assert!(ctx.codecs.first_encoder(&bad).is_err(), "alpha_effort 7");

    // The registered decoder gives the same frame whatever thread budget
    // the framework announces.
    let frame_with = |threads: Option<usize>| {
        let mut decoder =
            oxideav_webp::make_decoder(&params(CODEC_ID_VP8, w, h, PixelFormat::Rgba))
                .expect("decoder");
        if let Some(threads) = threads {
            decoder.set_execution_context(&ExecutionContext::with_threads(threads));
        }
        let packet =
            oxideav_core::Packet::new(0, oxideav_core::time::TimeBase::new(1, 1000), file.clone());
        decoder.send_packet(&packet).expect("send_packet");
        match decoder.receive_frame().expect("frame") {
            Frame::Video(video) => video.planes.into_iter().map(|p| p.data).collect::<Vec<_>>(),
            other => panic!("not a video frame: {other:?}"),
        }
    };
    assert!(frame_with(None) == frame_with(Some(4)));

    // RGB: the simple lossy layout.
    let file = one_packet(
        ctx.codecs
            .first_encoder(&with_options(PixelFormat::Rgb24))
            .expect("encoder"),
        &packed(w as usize * 3 + 1, w as usize * 3, &rgb),
    );
    assert!(file == oxideav_webp::encode_rgb8(w, h, &rgb, &layer1).expect("encode"));

    // 4:2:0 planes with an alpha plane: what `encode` writes for the
    // same image; the planes of the decode above serve as input.
    let frame = Frame::Video(VideoFrame {
        pts: Some(7),
        planes: decoded
            .planes
            .iter()
            .map(|p| VideoPlane {
                stride: p.stride,
                data: p.data.clone(),
            })
            .collect(),
    });
    let file = one_packet(
        ctx.codecs
            .first_encoder(&with_options(PixelFormat::Yuva420P))
            .expect("encoder"),
        &frame,
    );
    assert!(file == oxideav_webp::encode(&decoded, &layer1).expect("encode"));
    assert_eq!(
        oxideav_webp::decode(&file).expect("decode").planes[3].data,
        alpha
    );

    // A frame with too few planes for its layout is an error.
    let mut encoder = ctx
        .codecs
        .first_encoder(&with_options(PixelFormat::Yuva420P))
        .expect("encoder");
    assert!(
        encoder
            .send_frame(&yuv_frame(w, &yuv_planes(w, h)))
            .is_err()
    );
    // A layout outside the list is refused at construction.
    assert!(
        ctx.codecs
            .first_encoder(&params(CODEC_ID_VP8, w, h, PixelFormat::Gray8))
            .is_err()
    );
}
