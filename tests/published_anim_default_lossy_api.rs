//! Round-143 integration tests for the **animation-wide lossy-quality
//! default** — the new
//! [`oxideav_webp::AnimEncoderOptions::default_lossy_quality`] field and
//! its symmetric chainable builder
//! [`oxideav_webp::AnimEncoderOptions::with_default_lossy_quality`].
//!
//! These tests pin the **API-shape stub** contract for round 143:
//!
//! * The field is exposed on the published-0.1.5 surface and round-trips
//!   through its builder.
//! * It is fully symmetric to
//!   [`oxideav_webp::AnimEncoderOptions::default_near_lossless_quality`]
//!   (separate storage, separate builder, no cross-talk).
//! * It is a **no-op on the current encoder**: the existing
//!   `AnimFrameMode::Lossless` / `Delta` / `Auto` emission paths are
//!   lossless-only and ignore the value entirely. Bytes produced with
//!   any `Some(q)` are byte-exact-equal to the default `None` bytes.
//!
//! The lossy emission body itself is blocked on the `oxideav-vp8`
//! per-MB driver (workspace task #1041). Once that lands, the
//! lossless-only no-op contract here will become a behavioural test
//! (the lossy path will be invoked when a frame opts in), and these
//! tests should be revisited accordingly.

use oxideav_webp::{
    build_animated_webp, build_animated_webp_with_options, AnimEncoderOptions, AnimFrame,
    AnimFrameMode,
};

/// Deterministic xorshift32-driven noisy RGBA so the lossless encoder
/// has non-trivial entropy to compress.
fn make_noisy_rgba(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    let mut s: u32 = 0x9e37_79b9 ^ seed.wrapping_mul(0x85eb_ca6b);
    for _ in 0..(width * height) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        buf.push((s & 0xff) as u8);
        buf.push(((s >> 8) & 0xff) as u8);
        buf.push(((s >> 16) & 0xff) as u8);
        buf.push(0xff);
    }
    buf
}

// ───────── Public-API shape ─────────

#[test]
fn default_lossy_quality_field_is_none_on_default_options() {
    let opts = AnimEncoderOptions::default();
    assert_eq!(opts.default_lossy_quality, None);
}

#[test]
fn with_default_lossy_quality_builder_round_trips_through_published_surface() {
    let opts = AnimEncoderOptions::default().with_default_lossy_quality(Some(80));
    assert_eq!(opts.default_lossy_quality, Some(80));

    let opts2 = AnimEncoderOptions::default().with_default_lossy_quality(None);
    assert_eq!(opts2.default_lossy_quality, None);

    // Builder consumes self and returns Self — chains with the existing
    // near-lossless default builder without losing either value.
    let opts3 = AnimEncoderOptions::default()
        .with_default_lossy_quality(Some(60))
        .with_default_near_lossless_quality(Some(40));
    assert_eq!(opts3.default_lossy_quality, Some(60));
    assert_eq!(opts3.default_near_lossless_quality, Some(40));
}

// ───────── Symmetric independence with the near-lossless default ─────────

#[test]
fn default_lossy_and_near_lossless_defaults_are_independent_fields() {
    let opts = AnimEncoderOptions {
        default_near_lossless_quality: Some(70),
        default_lossy_quality: Some(30),
        ..AnimEncoderOptions::default()
    };
    assert_eq!(opts.default_near_lossless_quality, Some(70));
    assert_eq!(opts.default_lossy_quality, Some(30));

    // The two builders compose in either order and the resulting values
    // are equal to direct construction.
    let opts_chain_a = AnimEncoderOptions::default()
        .with_default_near_lossless_quality(Some(70))
        .with_default_lossy_quality(Some(30));
    let opts_chain_b = AnimEncoderOptions::default()
        .with_default_lossy_quality(Some(30))
        .with_default_near_lossless_quality(Some(70));
    assert_eq!(
        opts_chain_a.default_near_lossless_quality,
        opts_chain_b.default_near_lossless_quality
    );
    assert_eq!(
        opts_chain_a.default_lossy_quality,
        opts_chain_b.default_lossy_quality
    );
    assert_eq!(opts_chain_a.default_lossy_quality, Some(30));
    assert_eq!(opts_chain_a.default_near_lossless_quality, Some(70));
}

// ───────── No-op contract on the current encoder ─────────

#[test]
fn default_lossy_quality_is_byte_exact_no_op_on_lossless_keyframe_path() {
    // The Lossless mode is what `AnimFrame::new` defaults to. Whatever
    // is set on `default_lossy_quality` must not perturb the emitted
    // VP8L bytes — the API-shape stub guarantee.
    let (w, h) = (32u32, 32u32);
    let frames = vec![
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 0), 100),
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 1), 110),
    ];
    let baseline = build_animated_webp(&frames).expect("baseline");
    for q in [Some(0u8), Some(50), Some(75), Some(100), Some(255), None] {
        let opts = AnimEncoderOptions::default().with_default_lossy_quality(q);
        let bytes = build_animated_webp_with_options(&frames, &opts)
            .expect("lossy-default build on lossless frames");
        assert_eq!(
            bytes, baseline,
            "default_lossy_quality = {q:?} must be a no-op on the Lossless emission path"
        );
    }
}

#[test]
fn default_lossy_quality_is_byte_exact_no_op_on_delta_dirty_rect_path() {
    // Same contract for the Delta / Auto dirty-rect path — the lossy
    // default does not flow into the lossless dirty-rect bytes today.
    let (w, h) = (32u32, 32u32);
    let f0_px = make_noisy_rgba(w, h, 0);
    let mut f1_px = f0_px.clone();
    for row in 8..24 {
        for col in 8..24 {
            let off = (row * w as usize + col) * 4;
            f1_px[off] ^= 0xff;
            f1_px[off + 1] ^= 0xff;
            f1_px[off + 2] ^= 0xff;
        }
    }
    let mut f1 = AnimFrame::new(w, h, f1_px, 80);
    f1.mode = AnimFrameMode::Delta;
    let frames = vec![AnimFrame::new(w, h, f0_px, 80), f1];

    let baseline = build_animated_webp(&frames).expect("baseline Delta");
    for q in [Some(10u8), Some(50), Some(100), None] {
        let opts = AnimEncoderOptions::default().with_default_lossy_quality(q);
        let bytes = build_animated_webp_with_options(&frames, &opts)
            .expect("lossy-default build on delta frames");
        assert_eq!(
            bytes, baseline,
            "default_lossy_quality = {q:?} must be a no-op on the Delta emission path"
        );
    }
}

#[test]
fn default_lossy_quality_does_not_perturb_near_lossless_path() {
    // Combined: when the near-lossless default *does* take effect, the
    // lossy default still adds nothing — the two channels stay
    // independent, and the bytes equal the corresponding "near-lossless
    // only" file byte-for-byte.
    let (w, h) = (32u32, 32u32);
    let frames = vec![
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 0), 100),
        AnimFrame::new(w, h, make_noisy_rgba(w, h, 1), 110),
    ];
    let near_only = AnimEncoderOptions::default().with_default_near_lossless_quality(Some(60));
    let near_plus_lossy = AnimEncoderOptions::default()
        .with_default_near_lossless_quality(Some(60))
        .with_default_lossy_quality(Some(75));

    let bytes_near = build_animated_webp_with_options(&frames, &near_only).expect("near-only");
    let bytes_both = build_animated_webp_with_options(&frames, &near_plus_lossy).expect("both");
    assert_eq!(
        bytes_near, bytes_both,
        "lossy default must be a no-op when overlaid on near-lossless = Some(60)"
    );
}
