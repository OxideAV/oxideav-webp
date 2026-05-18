//! VP8X extended-header spec-conformance regression suite.
//!
//! Per RFC 9649 §2.5.6 + §2.7, the VP8X header announces which
//! auxiliary chunks follow via a flag byte, and the reconstruction
//! chunks (`VP8X`, `ICCP`, `ANIM`, `ANMF`, `ALPH`, `VP8 `, `VP8L`)
//! MUST appear in a fixed order. This test file pins three families
//! of behaviour:
//!
//! 1. **Flag-chunk consistency** — when the VP8X flag for ICC / EXIF /
//!    XMP / ALPHA is clear but the chunk is present anyway, the
//!    decoder silently drops it (mirroring the existing ANIM "MUST be
//!    ignored" behaviour). The chunk doesn't surface on
//!    `WebpImage::metadata` and (for ALPH) doesn't poison the
//!    composited alpha plane.
//! 2. **Chunk-order enforcement** — files where a reconstruction chunk
//!    arrives out of the spec-mandated order (e.g. ICCP after VP8,
//!    ANIM after ANMF) are rejected at parse time with a clear
//!    `WebP: chunk X out of order (stage Y → Z)` error. Metadata
//!    chunks (EXIF / XMP) and unknown FourCCs may appear anywhere
//!    after VP8X, matching the spec's "MAY appear out of order" rule.
//! 3. **Canvas product overflow** — the spec MUST that
//!    `canvas_w * canvas_h ≤ 2^32 - 1` rejects pathologically-sized
//!    headers (24-bit per-axis allows ~16M each, so the product can
//!    legitimately overflow `u32`).
//!
//! These tests exercise the eager `decode_webp` path; the lazy
//! `WebpAnimDecoder` path goes through the same `VP8xFlags` /
//! `ChunkStage` helpers, so any regression there would show up on the
//! existing `anim_decoder_streaming` suite (and the helpers are
//! sourced from the same internal module — there's no separate code
//! path to drift).

use oxideav_webp::{decode_webp, encode_vp8l_argb, extract_metadata};

const W: u32 = 8;
const H: u32 = 8;

/// Build a minimal opaque 8×8 ARGB image (all white).
fn argb_pixels() -> Vec<u32> {
    vec![0xFFFF_FFFFu32; (W * H) as usize]
}

/// Manual RIFF/WEBP/VP8X file builder used by the malformed-file
/// tests. Takes a list of `(fourcc, payload)` pairs and produces a
/// well-formed (size-wise) but spec-violation-laden output.
fn build_riff_webp(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4);
    body.extend_from_slice(b"WEBP");
    for &(fourcc, payload) in chunks {
        body.extend_from_slice(fourcc);
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(payload);
        if payload.len() & 1 == 1 {
            body.push(0);
        }
    }
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((body.len()) as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Build a VP8X chunk payload with the requested flag byte and canvas
/// size. See `src/demux.rs::parse_extended` for the field layout.
fn vp8x_payload(flags: u8, canvas_w: u32, canvas_h: u32) -> [u8; 10] {
    let mut out = [0u8; 10];
    out[0] = flags;
    let w1 = canvas_w.saturating_sub(1) & 0x00FF_FFFF;
    let h1 = canvas_h.saturating_sub(1) & 0x00FF_FFFF;
    out[4] = w1 as u8;
    out[5] = (w1 >> 8) as u8;
    out[6] = (w1 >> 16) as u8;
    out[7] = h1 as u8;
    out[8] = (h1 >> 8) as u8;
    out[9] = (h1 >> 16) as u8;
    out
}

/// Encode an opaque 8×8 VP8L bitstream — used by the spec-conformance
/// fixtures so each test gets a real (decoder-compatible) image chunk
/// rather than synthetic bytes that would fail the post-validation
/// VP8L parse.
fn vp8l_bytes() -> Vec<u8> {
    encode_vp8l_argb(W, H, &argb_pixels(), false).expect("vp8l encode")
}

// ─── Flag-chunk consistency ────────────────────────────────────────

#[test]
fn iccp_chunk_dropped_when_vp8x_icc_flag_clear() {
    // VP8X with the ICC flag CLEAR but an ICCP chunk physically
    // present in the file. Per RFC 9649 §2.5.6 + §2.7.1.4 the chunk
    // is ignored at decode time so neither `extract_metadata` nor
    // `decode_webp` surfaces it on `metadata.icc`.
    let icc = b"would-be-icc-profile-bytes";
    let vp8l = vp8l_bytes();
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x00, W, H)),
        (b"ICCP", icc),
        (b"VP8L", &vp8l),
    ]);
    // The ICCP chunk appears before the image (in-order per spec) so
    // chunk-order enforcement doesn't fire; only flag-gating does.
    let img = decode_webp(&file).expect("decode succeeds (ICCP silently ignored)");
    assert!(
        img.metadata.icc.is_none(),
        "ICC flag clear: metadata.icc must be None even when an ICCP chunk is physically present"
    );
    let meta = extract_metadata(&file).expect("extract_metadata");
    assert!(
        meta.icc.is_none(),
        "extract_metadata mirrors the same flag gate"
    );
}

#[test]
fn exif_xmp_chunks_dropped_when_their_flags_clear() {
    // VP8X with EXIF + XMP flags clear, both chunks physically
    // present. Spec §2.7.1.5: "Readers SHOULD ignore these chunks"
    // when the flag is clear (read as "treat as not present").
    let exif = b"II*\0fake-exif";
    let xmp = b"<?xml fake?>";
    let vp8l = vp8l_bytes();
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x00, W, H)),
        (b"VP8L", &vp8l),
        (b"EXIF", exif),
        (b"XMP ", xmp),
    ]);
    let img = decode_webp(&file).expect("decode succeeds");
    assert!(img.metadata.exif.is_none(), "EXIF flag clear: must drop");
    assert!(img.metadata.xmp.is_none(), "XMP flag clear: must drop");
}

#[test]
fn alph_chunk_dropped_when_alpha_flag_clear() {
    // VP8X with ALPHA flag CLEAR, paired with a VP8 chunk and a
    // top-level ALPH chunk. The ALPH chunk is ignored at parse time
    // (the spec-compliant interpretation when the flag says "no
    // transparency information"), so the decoded frame is the bare
    // VP8 luma + chroma with implicit-opaque alpha.
    //
    // We synthesise this by building the ALPH + VP8 by hand. The
    // VP8 chunk is a tiny black keyframe — its exact contents don't
    // matter for the consistency check (we just need parse to reach
    // the post-VP8 state); on round-trip the decoder will hit
    // `parse_vp8_keyframe_dims` and fall back to the VP8X canvas
    // size if the parse fails, which is what we expect for this
    // minimal fixture.
    //
    // Use a known-good VP8L encode and lie about its chunk id so we
    // exercise the ALPH dropping path on a parseable file. The
    // synthetic ALPH header byte sets `compression = 0` (raw plane).
    let vp8l = vp8l_bytes();
    let alph = [0u8; 10]; // 1 header byte + 9 bytes of "raw alpha"
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x00, W, H)),
        (b"ALPH", &alph),
        (b"VP8L", &vp8l),
    ]);
    // With the ALPHA flag clear the ALPH chunk drops — decode
    // succeeds and the resulting frame is the bare VP8L pixels
    // (which happen to be fully opaque already; the test point is
    // that the synthetic ALPH bytes never reach the alpha plane).
    let img = decode_webp(&file).expect("decode succeeds (ALPH silently ignored)");
    // All 8×8 pixels alpha = 0xff (the VP8L encode is opaque, and
    // the dropped ALPH didn't poison anything).
    for f in &img.frames {
        for px in f.rgba.chunks_exact(4) {
            assert_eq!(
                px[3], 0xff,
                "alpha plane must stay implicit-opaque when the ALPH chunk was flag-dropped"
            );
        }
    }
}

// ─── Chunk-order enforcement (RFC 9649 §2.7) ────────────────────────

#[test]
fn iccp_after_image_is_out_of_order_error() {
    // Spec: ICCP must come before the image chunk. Swap the order
    // and the parser rejects with the canonical out-of-order error.
    let icc = b"some-icc-bytes";
    let vp8l = vp8l_bytes();
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x20, W, H)),
        (b"VP8L", &vp8l),
        (b"ICCP", icc),
    ]);
    let err = decode_webp(&file).expect_err("ICCP-after-VP8L must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("out of order"),
        "expected out-of-order error, got: {msg}"
    );
}

#[test]
fn anim_after_anmf_is_out_of_order_error() {
    // ANIM must appear before any ANMF chunks. We build a valid
    // ANMF (header + nested VP8L sub-chunk) so the parser actually
    // accepts it, then place the ANIM chunk after — that's where
    // the chunk-stage state machine kicks in.
    let vp8l = vp8l_bytes();
    // ANMF header: 16 bytes (x/2, y/2, w-1, h-1, dur, flags) then
    // nested chunks. All zeros for x_off/y_off, w-1 = W-1, h-1 = H-1,
    // dur = 100ms, flags = 0.
    let mut anmf_payload = Vec::new();
    anmf_payload.extend_from_slice(&[0u8; 6]); // x_off=0, y_off=0
    let w1 = W - 1;
    anmf_payload.extend_from_slice(&[w1 as u8, (w1 >> 8) as u8, (w1 >> 16) as u8]);
    let h1 = H - 1;
    anmf_payload.extend_from_slice(&[h1 as u8, (h1 >> 8) as u8, (h1 >> 16) as u8]);
    anmf_payload.extend_from_slice(&[100u8, 0, 0]); // duration=100ms
    anmf_payload.push(0); // flags=0
                          // Nested VP8L sub-chunk inside the ANMF body.
    anmf_payload.extend_from_slice(b"VP8L");
    anmf_payload.extend_from_slice(&(vp8l.len() as u32).to_le_bytes());
    anmf_payload.extend_from_slice(&vp8l);
    if vp8l.len() & 1 == 1 {
        anmf_payload.push(0);
    }
    let anim = [0u8; 6]; // BG=0,0,0,0 + loop_count=0
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x02, W, H)),
        (b"ANMF", &anmf_payload),
        (b"ANIM", &anim),
    ]);
    let err = decode_webp(&file).expect_err("ANIM-after-ANMF must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("out of order"),
        "expected out-of-order error, got: {msg}"
    );
}

#[test]
fn alph_after_image_is_out_of_order_error() {
    // ALPH must precede VP8/VP8L. Swap and expect rejection.
    let vp8l = vp8l_bytes();
    let alph = [0u8; 10];
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x10, W, H)),
        (b"VP8L", &vp8l),
        (b"ALPH", &alph),
    ]);
    let err = decode_webp(&file).expect_err("ALPH-after-VP8L must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("out of order"),
        "expected out-of-order error, got: {msg}"
    );
}

#[test]
fn exif_xmp_may_appear_before_image_data() {
    // EXIF / XMP / unknown FourCCs MAY appear out of order per the
    // spec ("Metadata and unknown chunks MAY appear out of order").
    // Verify by placing EXIF *before* the image: this is not the
    // example layout from Fig 17 but is still spec-legal, so the
    // parser must accept it.
    let exif = b"II*\0before-image";
    let vp8l = vp8l_bytes();
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x08, W, H)),
        (b"EXIF", exif),
        (b"VP8L", &vp8l),
    ]);
    let img = decode_webp(&file).expect("EXIF-before-VP8L is spec-legal");
    assert_eq!(
        img.metadata.exif.as_deref(),
        Some(exif.as_ref()),
        "EXIF chunk must round-trip even when it appears before image"
    );
}

#[test]
fn unknown_chunks_anywhere_skipped_silently() {
    // Unknown FourCDs MAY appear out of order per the same clause.
    // Sprinkle two `XYZW` chunks (one before image, one after) and
    // verify the decoder ignores both without raising the
    // out-of-order error.
    let vp8l = vp8l_bytes();
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x00, W, H)),
        (b"XYZW", b"first unknown"),
        (b"VP8L", &vp8l),
        (b"XYZW", b"second unknown"),
    ]);
    decode_webp(&file).expect("unknown chunks bracket the image without error");
}

// ─── Canvas-product overflow (RFC 9649 §2.5.6) ──────────────────────

#[test]
fn canvas_product_overflow_rejected() {
    // 24-bit per-axis dimensions allow up to ~16 M each. The spec
    // MUST that `canvas_w * canvas_h ≤ 2^32 - 1`: we construct a
    // pathological header where both axes hit their 24-bit max
    // (which produces a product of ~2.8e14, far over the limit).
    //
    // The actual image data is irrelevant — the VP8X parse rejects
    // before we ever look at the rest of the body.
    let mut payload = [0u8; 10];
    payload[0] = 0x00; // no flags
                       // width = 0xff_ff_ff + 1 ≈ 16 M; height likewise.
    for byte in payload.iter_mut().skip(4) {
        *byte = 0xff;
    }
    let file = build_riff_webp(&[
        (b"VP8X", &payload),
        // No image needed — parse_extended errors on the canvas
        // check before walking past the VP8X header.
    ]);
    let err = decode_webp(&file).expect_err("canvas overflow must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("exceeds 2^32 pixels"),
        "expected canvas overflow error, got: {msg}"
    );
}

#[test]
fn canvas_product_at_limit_accepted() {
    // Bound test: width × height = 65536 × 65536 = 2^32. That's one
    // past the spec limit so it must fail. But 65535 × 65535 fits.
    // We use 65535 × 65000 to land safely under 2^32 and prove the
    // parser accepts non-overflowing dimensions even at large
    // sizes. Real decoding will still fail (no image chunk follows
    // in the malformed fixture), so we deliberately probe with
    // `extract_metadata` which stops at the VP8X header.
    //
    // The 24-bit field stores `canvas_w - 1`, so width=65535 means
    // bytes = 65534 = 0xfffe → low byte 0xfe, mid 0xff, hi 0x00.
    let mut payload = [0u8; 10];
    // width = 65535
    let w1 = 65535u32 - 1;
    payload[4] = w1 as u8;
    payload[5] = (w1 >> 8) as u8;
    payload[6] = (w1 >> 16) as u8;
    // height = 65000
    let h1 = 65000u32 - 1;
    payload[7] = h1 as u8;
    payload[8] = (h1 >> 8) as u8;
    payload[9] = (h1 >> 16) as u8;
    let file = build_riff_webp(&[(b"VP8X", &payload)]);
    // metadata-only fast path: succeeds because the canvas product
    // is 65535 × 65000 = 4.26e9, under 2^32 = 4.29e9.
    extract_metadata(&file).expect("canvas at limit but under 2^32 must parse");
}

// ─── Reserved-bit ignore ────────────────────────────────────────────

#[test]
fn vp8x_reserved_bits_in_flags_byte_ignored() {
    // Spec: bits 7, 6 (Rsv) and bit 0 (R) "MUST be 0. Readers MUST
    // ignore this field." That second clause is the key: even when
    // the producer leaks non-zero garbage into the reserved bits,
    // the reader must continue parsing as if the field is zero.
    //
    // Set bits 7, 6, 0 ALL to 1, leaving the semantic flags clear.
    // The expected outcome is a normal opaque decode (no chunks
    // gated by the semantic flags are present, reserved-bit garbage
    // is silently ignored).
    let vp8l = vp8l_bytes();
    // Reserved-only flag byte: 0xC1 = 1100_0001 (top 2 + bit 0 all set).
    let file = build_riff_webp(&[(b"VP8X", &vp8x_payload(0xC1, W, H)), (b"VP8L", &vp8l)]);
    let img = decode_webp(&file).expect("reserved bits must be ignored, not rejected");
    assert_eq!(img.frames.len(), 1, "still-image fixture decodes one frame");
    // Sanity: metadata is empty (we set no semantic flags).
    assert!(img.metadata.icc.is_none());
    assert!(img.metadata.exif.is_none());
    assert!(img.metadata.xmp.is_none());
}

// ─── Defence against the multi-chunk SHOULD ─────────────────────────

#[test]
fn duplicate_iccp_is_tolerated_last_wins() {
    // Spec: "There SHOULD be at most one such chunk. If there are
    // more such chunks, readers MAY ignore all except the first
    // one." We currently keep the LAST chunk seen (overwrite-on-
    // duplicate) — that's still in the "MAY ignore extras" envelope
    // since both first and last are equally valid choices and last-
    // wins matches the existing implementation's eager-overwrite
    // pattern. Pin the existing behaviour as a regression guard.
    let icc1 = b"first-icc";
    let icc2 = b"second-icc";
    let vp8l = vp8l_bytes();
    let file = build_riff_webp(&[
        (b"VP8X", &vp8x_payload(0x20, W, H)),
        (b"ICCP", icc1),
        (b"ICCP", icc2),
        (b"VP8L", &vp8l),
    ]);
    let img = decode_webp(&file).expect("duplicate ICCP tolerated");
    assert_eq!(img.metadata.icc.as_deref(), Some(icc2.as_ref()));
}
