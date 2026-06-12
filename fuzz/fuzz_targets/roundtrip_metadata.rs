#![no_main]

//! Differential oracle on the §2.7 metadata *write* path: the two
//! independent extended-layout writers
//! (`build::build_webp_file_with_metadata` and
//! `encode_vp8l_argb_with_metadata`) against the §2.3/§2.4 container
//! walker, the §2.7.1 `VP8X` flag parser, the metadata-only
//! `extract_metadata` walk, and the full `decode_webp` pixel +
//! metadata decode.
//!
//! Every prior harness fuzzes the metadata chunks from the *read*
//! side only (`extract_metadata` over raw bytes — which the
//! coverage-guided mutator almost never grows into a well-formed
//! `VP8X` + `ICCP` + bitstream + `EXIF` + `XMP ` file with five
//! self-consistent `Size` fields). This harness drives the writers
//! with fuzz-controlled §2.7.1.4 `ICCP` / §2.7.1.5 `EXIF` / `XMP `
//! payloads (presence, length 0..=255 — odd lengths exercising the
//! §2.3 pad byte — and content), a fuzz-controlled §2.7.1 `L`
//! (alpha-hint) flag, and fuzz-controlled canvas dimensions + ARGB
//! pixels, then cross-checks the §2.7 documented contract:
//!
//! * the emitted file parses through the §2.3/§2.4 chunk walker;
//! * the on-disk chunk sequence is the §2.7 canonical order
//!   `VP8X, ICCP?, VP8L, EXIF?, XMP?` (§2.7.1.4: the color profile
//!   "MUST appear before the image data");
//! * each metadata chunk's `Size` equals the payload length given to
//!   the writer and the payload bytes round-trip byte-for-byte;
//! * the §2.7.1 `VP8X` flag octet declares exactly the features
//!   present (`I`/`E`/`X` from which payloads were supplied, `L`
//!   verbatim, `A` clear) and echoes the canvas dimensions;
//! * `extract_metadata` lifts each payload byte-identically;
//! * `decode_webp` succeeds, reproduces the lossless pixels exactly,
//!   and carries the same metadata on `WebpImage::metadata`;
//! * both writers' outputs agree with each other on every one of the
//!   above (and `encode_vp8l_argb_with_metadata` demotes to the
//!   simple non-`VP8X` layout exactly when there is no alpha hint
//!   and no metadata).
//!
//! A crash is a panic in a writer or parser; an assertion failure is
//! a real §2.7 contract violation in the chunk framing, the pad-byte
//! emission, the `VP8X` flag derivation, or the metadata walk.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::build::{self, FileMetadata, ImageKind};
use oxideav_webp::container::{self, fourcc};
use oxideav_webp::{
    decode_webp, encode_vp8l_argb_with, encode_vp8l_argb_with_metadata, parse_vp8x_header,
    WebpMetadata,
};

fuzz_target!(|data: &[u8]| {
    // Layout: [0] width, [1] height, [2] presence/alpha flags,
    // [3..6] per-chunk metadata lengths, [6..] cycled byte pool.
    if data.len() < 6 {
        return;
    }

    // Small canvas so each iteration encodes in microseconds.
    let width = u32::from(data[0] % 16) + 1;
    let height = u32::from(data[1] % 16) + 1;
    let flags = data[2];
    let has_iccp = flags & 0x01 != 0;
    let has_exif = flags & 0x02 != 0;
    let has_xmp = flags & 0x04 != 0;
    let has_alpha = flags & 0x08 != 0;

    // Metadata payload lengths in [0, 255] — zero-length chunks and
    // odd lengths (the §2.3 pad-byte path) are both reachable.
    let iccp_len = usize::from(data[3]);
    let exif_len = usize::from(data[4]);
    let xmp_len = usize::from(data[5]);

    // Byte pool: cycle through the remaining input (zero-padded when
    // empty) so every byte position stays fuzzer-controlled even from
    // a short seed.
    let pool = &data[6..];
    let take = |offset: usize, len: usize| -> Vec<u8> {
        (0..len)
            .map(|i| {
                if pool.is_empty() {
                    0
                } else {
                    pool[(offset + i) % pool.len()]
                }
            })
            .collect()
    };

    // Fuzz-controlled ARGB pixels: (a << 24) | (r << 16) | (g << 8) | b.
    let n_pixels = (width as usize) * (height as usize);
    let pixel_bytes = take(0, n_pixels * 4);
    let argb: Vec<u32> = pixel_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // The RGBA scan-order bytes decode_webp must reproduce.
    let expected_rgba: Vec<u8> = argb
        .iter()
        .flat_map(|&p| {
            [
                (p >> 16) as u8, // R
                (p >> 8) as u8,  // G
                p as u8,         // B
                (p >> 24) as u8, // A
            ]
        })
        .collect();

    // Fuzz-controlled metadata payloads, offset into the pool past the
    // pixel bytes so the fuzzer can steer each independently.
    let iccp = has_iccp.then(|| take(n_pixels * 4, iccp_len));
    let exif = has_exif.then(|| take(n_pixels * 4 + iccp_len, exif_len));
    let xmp = has_xmp.then(|| take(n_pixels * 4 + iccp_len + exif_len, xmp_len));

    // Bare §2.6 VP8L bitstream with the §3.4 alpha hint set verbatim
    // (the hint is informative only — pixels round-trip regardless).
    let vp8l_payload = encode_vp8l_argb_with(&argb, width, height, has_alpha)
        .expect("in-bounds dimensions + exactly-sized ARGB buffer must encode");

    // ── Writer A: the metadata-aware container writer. ──────────────
    let file_a = build::build_webp_file_with_metadata(
        &vp8l_payload,
        ImageKind::Lossless,
        width,
        height,
        has_alpha,
        FileMetadata {
            iccp: iccp.as_deref(),
            exif: exif.as_deref(),
            xmp: xmp.as_deref(),
        },
    )
    .expect("small canvas + small payloads must build");

    check_extended_file(
        &file_a,
        width,
        height,
        has_alpha,
        iccp.as_deref(),
        exif.as_deref(),
        xmp.as_deref(),
        &expected_rgba,
    );

    // ── Writer B: the published-shape one-call encoder. ─────────────
    let file_b = encode_vp8l_argb_with_metadata(
        width,
        height,
        &argb,
        has_alpha,
        &WebpMetadata {
            icc: iccp.as_deref(),
            exif: exif.as_deref(),
            xmp: xmp.as_deref(),
        },
    )
    .expect("small canvas + small payloads must encode");

    if !has_alpha && iccp.is_none() && exif.is_none() && xmp.is_none() {
        // §2.6 simple layout: a single VP8L chunk, no VP8X.
        let c = container::parse(&file_b).expect("writer B simple output must parse");
        let seq: Vec<_> = c.chunks.iter().map(|ch| ch.fourcc).collect();
        assert_eq!(
            seq,
            vec![fourcc::VP8L],
            "no metadata + no alpha hint demotes to the simple layout"
        );
    } else {
        check_extended_file(
            &file_b,
            width,
            height,
            has_alpha,
            iccp.as_deref(),
            exif.as_deref(),
            xmp.as_deref(),
            &expected_rgba,
        );
    }

    // Both writers agree on what a metadata walk recovers.
    let meta_a = oxideav_webp::extract_metadata(&file_a).expect("writer A metadata walk");
    let meta_b = oxideav_webp::extract_metadata(&file_b).expect("writer B metadata walk");
    assert_eq!(meta_a, meta_b, "both writers carry identical metadata");
});

/// Cross-check one §2.7 extended-layout file against every read-side
/// contract: chunk order, payload round trip, `VP8X` flags, metadata
/// walk, and full pixel decode.
#[allow(clippy::too_many_arguments)]
fn check_extended_file(
    file: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    iccp: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    expected_rgba: &[u8],
) {
    let c = container::parse(file).expect("writer output must parse through the §2.3/§2.4 walker");

    // §2.7 canonical chunk order: VP8X, ICCP?, VP8L, EXIF?, XMP?
    // (§2.7.1.4: ICCP "MUST appear before the image data").
    let mut expected_seq = vec![fourcc::VP8X];
    if iccp.is_some() {
        expected_seq.push(fourcc::ICCP);
    }
    expected_seq.push(fourcc::VP8L);
    if exif.is_some() {
        expected_seq.push(fourcc::EXIF);
    }
    if xmp.is_some() {
        expected_seq.push(fourcc::XMP);
    }
    let seq: Vec<_> = c.chunks.iter().map(|ch| ch.fourcc).collect();
    assert_eq!(seq, expected_seq, "§2.7 canonical chunk order");

    // Each metadata chunk's Size + payload bytes round-trip verbatim.
    for (tag, payload) in [
        (fourcc::ICCP, iccp),
        (fourcc::EXIF, exif),
        (fourcc::XMP, xmp),
    ] {
        if let Some(expected) = payload {
            let chunk = c
                .first_chunk_with_fourcc(tag)
                .expect("declared metadata chunk is present");
            assert_eq!(
                chunk.size as usize,
                expected.len(),
                "chunk Size equals the payload length handed to the writer"
            );
            assert_eq!(
                chunk.payload(file),
                expected,
                "metadata payload bytes round-trip through the walker"
            );
        }
    }

    // §2.7.1 VP8X: flag octet + canvas dimensions echo the write call.
    let vp8x_chunk = c.first_chunk_with_fourcc(fourcc::VP8X).expect("VP8X chunk");
    let vp8x = parse_vp8x_header(vp8x_chunk.payload(file)).expect("VP8X payload parses");
    assert_eq!(vp8x.has_iccp, iccp.is_some(), "§2.7.1 I flag");
    assert_eq!(vp8x.has_exif, exif.is_some(), "§2.7.1 E flag");
    assert_eq!(vp8x.has_xmp, xmp.is_some(), "§2.7.1 X flag");
    assert_eq!(vp8x.has_alpha, has_alpha, "§2.7.1 L flag carried verbatim");
    assert!(!vp8x.has_animation, "still image never sets the A flag");
    assert_eq!(vp8x.canvas_width, width, "VP8X canvas width");
    assert_eq!(vp8x.canvas_height, height, "VP8X canvas height");

    // Metadata-only walk lifts each payload byte-identically.
    let meta = oxideav_webp::extract_metadata(file).expect("metadata walk succeeds");
    assert_eq!(
        meta.icc.as_deref(),
        iccp,
        "ICCP payload via extract_metadata"
    );
    assert_eq!(
        meta.exif.as_deref(),
        exif,
        "EXIF payload via extract_metadata"
    );
    assert_eq!(meta.xmp.as_deref(), xmp, "XMP payload via extract_metadata");

    // Full decode: lossless pixels + the same metadata on the image.
    let image = decode_webp(file).expect("extended lossless file decodes");
    assert_eq!(image.width, width, "decoded canvas width");
    assert_eq!(image.height, height, "decoded canvas height");
    assert_eq!(image.frames.len(), 1, "still image carries one frame");
    assert_eq!(
        image.frames[0].rgba, expected_rgba,
        "lossless pixels survive the metadata-bearing layout"
    );
    assert_eq!(image.metadata.icc.as_deref(), iccp, "decode-path ICCP");
    assert_eq!(image.metadata.exif.as_deref(), exif, "decode-path EXIF");
    assert_eq!(image.metadata.xmp.as_deref(), xmp, "decode-path XMP");
    assert!(image.anim_background_rgba.is_none(), "no ANIM background");
    assert!(image.anim_loop_count.is_none(), "no ANIM loop count");
}
