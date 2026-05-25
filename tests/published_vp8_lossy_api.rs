//! Round-131 published-API surface tests for the **VP8 lossy encode**
//! path — the published-0.1.5 names per `API-COMPAT.md`, currently
//! implemented as API-shape stubs that return `WebpError::Unsupported`
//! at call time.
//!
//! What we assert:
//!
//! 1. Every published entry point exists with the documented signature.
//! 2. Length / dimension validation runs **before** the
//!    Unsupported gate, so a caller bug (wrong buffer length, wrong
//!    plane stride, qindex out of range) surfaces as the published
//!    `WebpError::InvalidData` rather than masquerading as Unsupported.
//! 3. No call ever returns `Ok(bytes)` — the round-131 directive
//!    explicitly forbids producing garbage VP8 bytes from
//!    `oxideav-vp8`'s constant-grey-frame `encode_silent_keyframe`.
//!    The day `oxideav-vp8` lands the §13/§14 pixel-driven encode round
//!    these tests start failing — that failure is the cue to flip the
//!    bodies of the entry points from stubs to real wiring (a
//!    deliberate change-detector).
//! 4. `quality_to_qindex` is a stable mapping callers can pre-compute
//!    against without instantiating an encoder.
//!
//! Standalone (`--no-default-features`) builds get the free-function
//! coverage; the registry block exercises the `webp_vp8` codec id +
//! `WebpVp8LossyEncoder` trait impl.

use oxideav_webp::{
    encode_vp8_lossy_rgb24, encode_vp8_lossy_rgba, encode_vp8_lossy_yuv420p,
    encode_vp8_lossy_yuva420p,
    encoder_vp8::{
        compute_psy_stats, freq_deltas_for_qindex, quality_to_qindex, Vp8FreqDeltas, Vp8PsyStats,
        DEFAULT_QUALITY, QINDEX_MAX, QUALITY_MAX, QUALITY_MIN,
    },
    WebpError, WebpMetadata,
};

#[test]
fn quality_to_qindex_endpoints() {
    // libwebp convention: 100 = best (qindex 0), 0 = worst (qindex 127).
    assert_eq!(quality_to_qindex(QUALITY_MAX), 0);
    assert_eq!(quality_to_qindex(QUALITY_MIN), 127);
    // Default sits roughly in the middle of the band.
    let mid = quality_to_qindex(DEFAULT_QUALITY);
    assert!(
        mid > 10 && mid < 80,
        "default qindex {mid} should be mid-band"
    );
}

#[test]
fn quality_to_qindex_clamps_out_of_range() {
    assert_eq!(quality_to_qindex(-1.0), QINDEX_MAX);
    assert_eq!(quality_to_qindex(101.0), 0);
    assert_eq!(quality_to_qindex(f32::INFINITY), 0);
    assert_eq!(quality_to_qindex(f32::NEG_INFINITY), QINDEX_MAX);
    // NaN folds to the DEFAULT_QUALITY mapping, not a random value.
    assert_eq!(
        quality_to_qindex(f32::NAN),
        quality_to_qindex(DEFAULT_QUALITY)
    );
}

#[test]
fn freq_deltas_default_and_zero_match() {
    let d: Vp8FreqDeltas = Default::default();
    assert_eq!(d, Vp8FreqDeltas::zero());
    // freq_deltas_for_qindex returns the all-zero stub today (per the
    // encoder_vp8 module-level note); the round wiring the real pixel
    // encoder will flip this.
    assert_eq!(freq_deltas_for_qindex(0), Vp8FreqDeltas::zero());
    assert_eq!(freq_deltas_for_qindex(QINDEX_MAX), Vp8FreqDeltas::zero());
}

#[test]
fn compute_psy_stats_validates_length_and_returns_stub() {
    let stats = compute_psy_stats(4, 4, &[0u8; 4 * 4 * 4]).expect("4x4 rgba ok");
    assert_eq!(stats, Vp8PsyStats::default());

    // Wrong buffer length surfaces a caller error, not an Unsupported.
    assert_eq!(
        compute_psy_stats(4, 4, &[0u8; 4 * 4 * 4 - 1]),
        Err(WebpError::InvalidData)
    );
}

#[test]
fn encode_vp8_lossy_rgba_validates_then_unsupported() {
    let (w, h) = (8u32, 4u32);
    // Wrong length first — must be InvalidData, not Unsupported.
    let bad = encode_vp8_lossy_rgba(
        w,
        h,
        &vec![0u8; (w * h * 4 - 1) as usize],
        DEFAULT_QUALITY,
        &WebpMetadata::default(),
    );
    assert_eq!(bad, Err(WebpError::InvalidData));

    // Correct length — stub returns Unsupported, never garbage bytes.
    let rgba = vec![0u8; (w * h * 4) as usize];
    let out = encode_vp8_lossy_rgba(w, h, &rgba, DEFAULT_QUALITY, &WebpMetadata::default());
    assert_eq!(
        out,
        Err(WebpError::Unsupported),
        "stub must not emit bytes — see encoder_vp8 module note"
    );
}

#[test]
fn encode_vp8_lossy_rgb24_validates_then_unsupported() {
    let (w, h) = (3u32, 2u32);
    assert_eq!(
        encode_vp8_lossy_rgb24(
            w,
            h,
            &vec![0u8; (w * h * 3 + 1) as usize],
            DEFAULT_QUALITY,
            &WebpMetadata::default()
        ),
        Err(WebpError::InvalidData)
    );
    let rgb = vec![0u8; (w * h * 3) as usize];
    assert_eq!(
        encode_vp8_lossy_rgb24(w, h, &rgb, DEFAULT_QUALITY, &WebpMetadata::default()),
        Err(WebpError::Unsupported)
    );
}

#[test]
fn encode_vp8_lossy_yuv420p_validates_then_unsupported() {
    let (w, h) = (4u32, 4u32);
    let y_len = (w as usize) * (h as usize);
    let c_len = w.div_ceil(2) as usize * h.div_ceil(2) as usize;
    // Mismatched chroma plane size first.
    assert_eq!(
        encode_vp8_lossy_yuv420p(
            w,
            h,
            &vec![128u8; y_len],
            &vec![128u8; c_len - 1],
            &vec![128u8; c_len],
            &WebpMetadata::default()
        ),
        Err(WebpError::InvalidData)
    );
    // Correctly-sized planes — stub returns Unsupported.
    let y = vec![128u8; y_len];
    let u = vec![128u8; c_len];
    let v = vec![128u8; c_len];
    assert_eq!(
        encode_vp8_lossy_yuv420p(w, h, &y, &u, &v, &WebpMetadata::default()),
        Err(WebpError::Unsupported)
    );
}

#[test]
fn encode_vp8_lossy_yuva420p_validates_then_unsupported() {
    let (w, h) = (4u32, 4u32);
    let y_len = (w as usize) * (h as usize);
    let c_len = w.div_ceil(2) as usize * h.div_ceil(2) as usize;
    let y = vec![128u8; y_len];
    let u = vec![128u8; c_len];
    let v = vec![128u8; c_len];
    // Wrong alpha plane size first.
    assert_eq!(
        encode_vp8_lossy_yuva420p(
            w,
            h,
            &y,
            &u,
            &v,
            &vec![255u8; y_len - 2],
            &WebpMetadata::default()
        ),
        Err(WebpError::InvalidData)
    );
    // Correctly-sized — stub returns Unsupported.
    let a = vec![255u8; y_len];
    assert_eq!(
        encode_vp8_lossy_yuva420p(w, h, &y, &u, &v, &a, &WebpMetadata::default()),
        Err(WebpError::Unsupported)
    );
}

#[test]
fn encode_vp8_lossy_rgba_with_alpha_still_unsupported() {
    // Even a non-opaque image — which API-COMPAT.md says should
    // auto-promote to extended-lossy + ALPH — must stay Unsupported
    // until real pixel encode lands. We mainly want to be sure no
    // garbage bytes ever return.
    let (w, h) = (2u32, 2u32);
    let mut rgba = Vec::new();
    for i in 0..(w * h) as usize {
        rgba.extend_from_slice(&[i as u8 * 50, 100, 200, 0x80]);
    }
    assert_eq!(
        encode_vp8_lossy_rgba(w, h, &rgba, DEFAULT_QUALITY, &WebpMetadata::default()),
        Err(WebpError::Unsupported)
    );
}

// ─────────────────────── registry-side coverage ───────────────────────

#[cfg(feature = "registry")]
mod registry_side {
    use oxideav_core::{
        CodecId, CodecParameters, Error as CoreError, Frame, PixelFormat, RuntimeContext,
        VideoFrame, VideoPlane,
    };
    use oxideav_webp::{
        encoder_vp8::{
            self, make_encoder_with_qindex, make_encoder_with_quality,
            make_encoder_with_target_size,
        },
        register, WebpMetadataOwned, CODEC_ID_VP8,
    };

    fn vp8_params(width: u32, height: u32, pix: PixelFormat) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
        p.width = Some(width);
        p.height = Some(height);
        p.pixel_format = Some(pix);
        p
    }

    fn rgba_frame(w: u32, h: u32) -> Frame {
        let data = vec![0x42u8; (w * h * 4) as usize];
        Frame::Video(VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: (w * 4) as usize,
                data,
            }],
        })
    }

    #[test]
    fn webp_vp8_codec_id_is_registered() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_ID_VP8);
        assert!(
            ctx.codecs.has_encoder(&id),
            "webp_vp8 encoder factory not installed"
        );
        assert!(
            ctx.codecs.has_decoder(&id),
            "webp_vp8 decoder factory not installed"
        );
    }

    #[test]
    fn webp_vp8_factory_builds_encoder_with_qindex() {
        let params = vp8_params(16, 16, PixelFormat::Rgba);
        let enc = make_encoder_with_qindex(&params, 32).expect("ctor");
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_VP8);
    }

    #[test]
    fn webp_vp8_factory_rejects_qindex_out_of_range() {
        let params = vp8_params(16, 16, PixelFormat::Rgba);
        let res = make_encoder_with_qindex(&params, 128);
        let Err(err) = res else {
            panic!("qindex 128 > 127 should have rejected");
        };
        assert!(matches!(err, CoreError::InvalidData(_)));
    }

    #[test]
    fn webp_vp8_factory_rejects_missing_dimensions() {
        let mut p = CodecParameters::video(CodecId::new(CODEC_ID_VP8));
        p.pixel_format = Some(PixelFormat::Rgba);
        let res = make_encoder_with_qindex(&p, 32);
        let Err(err) = res else {
            panic!("missing width should have rejected");
        };
        assert!(matches!(err, CoreError::InvalidData(_)));
    }

    #[test]
    fn webp_vp8_factory_accepts_quality_path() {
        let params = vp8_params(16, 16, PixelFormat::Rgba);
        let enc = make_encoder_with_quality(&params, encoder_vp8::DEFAULT_QUALITY).expect("ctor");
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_VP8);
    }

    #[test]
    fn webp_vp8_factory_accepts_target_size_path() {
        let params = vp8_params(16, 16, PixelFormat::Rgba);
        let _enc = make_encoder_with_target_size(&params, 1024).expect("ctor");
    }

    #[test]
    fn webp_vp8_send_frame_returns_unsupported_not_garbage() {
        // The crux: the encoder builds, the frame validates against
        // dimensions, but `send_frame` returns `Error::Unsupported`
        // rather than a constant-grey VP8 stream wrapped in WebP RIFF.
        let params = vp8_params(16, 16, PixelFormat::Rgba);
        let mut enc = make_encoder_with_qindex(&params, 32).expect("ctor");
        let frame = rgba_frame(16, 16);
        let err = enc
            .send_frame(&frame)
            .expect_err("stub must not pretend to encode");
        let msg = err.to_string();
        assert!(
            matches!(err, CoreError::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
        assert!(
            msg.contains("oxideav-vp8") || msg.contains("encoder_vp8"),
            "Unsupported message should point at the gap: {msg}"
        );
        // receive_packet has nothing to give us either.
        let rx = enc.receive_packet();
        assert!(matches!(rx, Err(CoreError::NeedMore)));
    }

    #[test]
    fn webp_vp8_rejects_wrong_plane_stride() {
        // The encoder validates plane stride up-front so caller bugs
        // surface as InvalidData rather than (silently) Unsupported.
        let params = vp8_params(16, 16, PixelFormat::Rgba);
        let mut enc = make_encoder_with_qindex(&params, 32).expect("ctor");
        // Stride too small for the declared width.
        let bad = Frame::Video(VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: 16, // would need 16*4=64
                data: vec![0u8; 16 * 16 * 4],
            }],
        });
        let err = enc.send_frame(&bad).expect_err("stride mismatch");
        assert!(matches!(err, CoreError::InvalidData(_)));
    }

    #[test]
    fn webp_vp8_with_metadata_round_trips_through_accessor() {
        // The encoder retains the configured metadata so a caller can
        // round-trip its `WebpMetadataOwned` through the factory.
        let params = vp8_params(8, 8, PixelFormat::Rgba);
        let meta = WebpMetadataOwned {
            icc: Some(b"icc".to_vec()),
            exif: None,
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
        };
        let enc = encoder_vp8::make_encoder_with_quality_and_metadata(
            &params,
            encoder_vp8::DEFAULT_QUALITY,
            meta.clone(),
        )
        .expect("ctor");
        // Downcast via the published trait to read back the qindex /
        // freq_deltas / metadata accessors on `WebpVp8LossyEncoder`.
        // For now we just assert the codec id stays stable.
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_VP8);
    }
}
