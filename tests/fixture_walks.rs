//! Integration tests that drive [`oxideav_webp::parse_container`]
//! against a small in-crate copy of the WebP fixture corpus.
//!
//! The five `tests/data/*.webp` files are byte-for-byte copies of
//! `docs/image/webp/fixtures/{lossy-1x1, lossless-1x1,
//! extended-with-exif, lossy-with-alpha-128x128,
//! animated-with-alpha}/input.webp`. They live inside the crate so
//! that CI (which checks out only this repository, not the umbrella
//! workspace) can reach them via `include_bytes!`.
//!
//! Fixtures here are *opaque* inputs to the structural walker plus
//! the `VP8X` / `ALPH` / `ANIM` header-field parsers. Pixel-level
//! decode is not yet implemented and is explicitly out of scope for
//! this test file.

use oxideav_webp::alph::{AlphCompression, AlphFiltering, AlphPreprocessing};
use oxideav_webp::anmf::{BlendingMethod, DisposalMethod};
use oxideav_webp::build::{ImageKind, Vp8xFlags};
use oxideav_webp::container::fourcc;
use oxideav_webp::vp8_chunk::{WebpLossyChunk, VP8_KEYFRAME_HEADER_LEN};
use oxideav_webp::{
    build_vp8x_chunk, build_webp_file, extract_lossy_chunk, parse_alph_header, parse_anim_header,
    parse_anmf_header, parse_container, parse_vp8x_header,
};

const LOSSY_1X1: &[u8] = include_bytes!("data/lossy-1x1.webp");
const LOSSLESS_1X1: &[u8] = include_bytes!("data/lossless-1x1.webp");
const EXTENDED_WITH_EXIF: &[u8] = include_bytes!("data/extended-with-exif.webp");
const LOSSY_WITH_ALPHA: &[u8] = include_bytes!("data/lossy-with-alpha-128x128.webp");
const ANIMATED_WITH_ALPHA: &[u8] = include_bytes!("data/animated-with-alpha.webp");

#[test]
fn fixture_lossy_1x1_has_a_single_vp8_chunk() {
    // RFC 9649 §2.5 — simple lossy: one 'VP8 ' chunk after the
    // 12-byte WebP file header.
    let c = parse_container(LOSSY_1X1).expect("lossy-1x1 fixture parses");
    assert_eq!(c.chunks.len(), 1);
    assert_eq!(c.chunks[0].fourcc, fourcc::VP8);
    assert!(!c.is_extended());
}

#[test]
fn fixture_lossless_1x1_has_a_single_vp8l_chunk() {
    // RFC 9649 §2.6 — simple lossless: one 'VP8L' chunk after the
    // 12-byte WebP file header.
    let c = parse_container(LOSSLESS_1X1).expect("lossless-1x1 fixture parses");
    assert_eq!(c.chunks.len(), 1);
    assert_eq!(c.chunks[0].fourcc, fourcc::VP8L);
    assert!(!c.is_extended());
}

#[test]
fn fixture_extended_with_exif_walks_vp8x_vp8_exif() {
    // RFC 9649 §2.7 — extended layout: 'VP8X' first, then the
    // image data ('VP8 ' here), then auxiliary chunks ('EXIF').
    let c = parse_container(EXTENDED_WITH_EXIF).expect("extended-with-exif fixture parses");
    let order: Vec<_> = c.chunks.iter().map(|c| c.fourcc).collect();
    assert_eq!(&order[0], &fourcc::VP8X);
    assert!(c.is_extended());
    assert!(c.first_chunk_with_fourcc(fourcc::VP8).is_some());
    assert!(c.first_chunk_with_fourcc(fourcc::EXIF).is_some());
}

#[test]
fn fixture_extended_with_exif_vp8x_payload_decodes_to_128x128_exif_only() {
    // Round-2 surface: walker → typed VP8X. Anchors the bit-position
    // decode against a real libwebp-produced VP8X chunk. The fixture's
    // own `trace.txt` reports the same flags / dimensions.
    let c = parse_container(EXTENDED_WITH_EXIF).expect("extended-with-exif fixture parses");
    let vp8x = c
        .first_chunk_with_fourcc(fourcc::VP8X)
        .expect("VP8X chunk present");
    let h = parse_vp8x_header(vp8x.payload(EXTENDED_WITH_EXIF))
        .expect("VP8X payload parses per §2.7.1");
    assert_eq!(h.canvas_width, 128);
    assert_eq!(h.canvas_height, 128);
    assert!(h.has_exif);
    assert!(!h.has_iccp);
    assert!(!h.has_xmp);
    assert!(!h.has_alpha);
    assert!(!h.has_animation);
    // Producer set every §2.7.1 reserved bit to 0 — has_unknown stays
    // clear, matching the trace report of `flags=0x00000008`.
    assert!(!h.has_unknown);
}

#[test]
fn fixture_lossy_with_alpha_alph_info_byte_decodes_to_lossless_no_filter_no_pre() {
    // Round-3 surface: walker → typed ALPH info byte. Cross-checks
    // the bit-position decode against `lossy-with-alpha-128x128/trace.txt`
    //   ALPH method=1 filter=0 pre_processing=0 header_byte=0x01
    let c = parse_container(LOSSY_WITH_ALPHA).expect("lossy-with-alpha fixture parses");
    let alph = c
        .first_chunk_with_fourcc(fourcc::ALPH)
        .expect("ALPH chunk present");
    let h = parse_alph_header(alph.payload(LOSSY_WITH_ALPHA))
        .expect("ALPH info byte parses per §2.7.1.2");
    assert_eq!(h.info_byte, 0x01);
    assert_eq!(h.compression, AlphCompression::Lossless);
    assert_eq!(h.filtering, AlphFiltering::None);
    assert_eq!(h.preprocessing, AlphPreprocessing::None);
    assert_eq!(h.reserved, 0);
    // The remaining `Chunk Size - 1` bytes are the alpha bitstream
    // proper (out of scope for round 3).
    let bitstream = &alph.payload(LOSSY_WITH_ALPHA)[h.bitstream_offset()..];
    assert_eq!(bitstream.len(), (alph.size as usize) - 1);
}

#[test]
fn fixture_animated_with_alpha_all_three_anmf_headers_decode_to_trace_values() {
    // Round-4 surface: walker → typed ANMF. Cross-checks the 16-byte
    // per-frame header decode against `animated-with-alpha/trace.txt`,
    // which reports for every one of the three frames:
    //   ANMF_FRAME x_offset=0 y_offset=0 width=64 height=64
    //              duration=100 dispose=0 blend=1 flags_byte=0x02
    let c = parse_container(ANIMATED_WITH_ALPHA).expect("animated-with-alpha fixture parses");
    let anmfs: Vec<_> = c
        .chunks
        .iter()
        .filter(|ch| ch.fourcc == fourcc::ANMF)
        .collect();
    assert_eq!(anmfs.len(), 3, "trace.txt reports three ANMF chunks");
    for (i, anmf_chunk) in anmfs.iter().enumerate() {
        let payload = anmf_chunk.payload(ANIMATED_WITH_ALPHA);
        let h = parse_anmf_header(payload)
            .unwrap_or_else(|e| panic!("ANMF #{i} payload parses per §2.7.1.1: {e:?}"));
        assert_eq!(h.x, 0, "frame {i} x_offset");
        assert_eq!(h.y, 0, "frame {i} y_offset");
        assert_eq!(h.width, 64, "frame {i} width");
        assert_eq!(h.height, 64, "frame {i} height");
        assert_eq!(h.duration_ms, 100, "frame {i} duration");
        assert_eq!(h.blend, BlendingMethod::Overwrite, "frame {i} blend");
        assert_eq!(h.dispose, DisposalMethod::None, "frame {i} dispose");
        assert_eq!(h.info_byte, 0x02, "frame {i} flags byte");
        assert_eq!(h.reserved, 0, "frame {i} reserved");
        // The Frame Data sub-RIFF starts immediately after the
        // 16-byte header — `Chunk Size - 16` bytes long per §2.7.1.1.
        let frame_data = &payload[h.frame_data_offset()..];
        assert_eq!(
            frame_data.len(),
            (anmf_chunk.size as usize) - 16,
            "frame {i} frame_data length matches §2.7.1.1 `Chunk Size - 16`"
        );
    }
}

#[test]
fn fixture_animated_with_alpha_anim_payload_decodes_to_white_opaque_infinite() {
    // Round-3 surface: walker → typed ANIM. Cross-checks the BGRA
    // byte order + loop-count u16 LE decode against
    // `animated-with-alpha/trace.txt`
    //   ANIM bgcolor=0xffffffff loop_count=0
    let c = parse_container(ANIMATED_WITH_ALPHA).expect("animated-with-alpha fixture parses");
    let anim = c
        .first_chunk_with_fourcc(fourcc::ANIM)
        .expect("ANIM chunk present");
    let h = parse_anim_header(anim.payload(ANIMATED_WITH_ALPHA))
        .expect("ANIM payload parses per §2.7.1.1");
    assert_eq!(h.background_color.blue, 0xFF);
    assert_eq!(h.background_color.green, 0xFF);
    assert_eq!(h.background_color.red, 0xFF);
    assert_eq!(h.background_color.alpha, 0xFF);
    assert_eq!(h.background_color.as_u32_le(), 0xFFFF_FFFF);
    assert_eq!(h.loop_count, 0);
    assert!(h.loops_forever());
}

#[test]
fn round5_lossy_fixture_payload_rewraps_into_byte_identical_riff_envelope() {
    // Round-5 surface: builder ↔ walker round-trip on a real fixture.
    // We rip the §2.5 `VP8 ` chunk's payload out of the libwebp-produced
    // lossy-1x1.webp, hand it back to the builder, and verify the
    // resulting bytes parse to the same chunk content + same
    // §2.4 File Size field. This is the encoder-replacement path —
    // demonstrates the builder is the algebraic inverse of the walker
    // for simple-lossy files.
    let c = parse_container(LOSSY_1X1).expect("lossy-1x1 fixture parses");
    assert_eq!(c.chunks.len(), 1);
    assert_eq!(c.chunks[0].fourcc, fourcc::VP8);
    let payload = c.chunks[0].payload(LOSSY_1X1);

    let rebuilt = build_webp_file(payload, ImageKind::Lossy, 0, 0)
        .expect("simple-lossy file builds from its own VP8 payload");
    let c2 = parse_container(&rebuilt).expect("rebuilt simple-lossy file parses");
    assert_eq!(c2.chunks.len(), 1);
    assert_eq!(c2.chunks[0].fourcc, fourcc::VP8);
    assert_eq!(c2.chunks[0].size, c.chunks[0].size);
    assert_eq!(c2.chunks[0].payload(&rebuilt), payload);
    assert_eq!(c2.riff_file_size, c.riff_file_size);
}

#[test]
fn round5_lossless_fixture_payload_rewraps_into_byte_identical_riff_envelope() {
    // Same as the lossy round-trip but on the §2.6 simple-lossless
    // fixture. Catches any LE / pad-byte / FourCC drift between the
    // walker and builder on the VP8L FourCC.
    let c = parse_container(LOSSLESS_1X1).expect("lossless-1x1 fixture parses");
    assert_eq!(c.chunks.len(), 1);
    assert_eq!(c.chunks[0].fourcc, fourcc::VP8L);
    let payload = c.chunks[0].payload(LOSSLESS_1X1);

    let rebuilt = build_webp_file(payload, ImageKind::Lossless, 0, 0)
        .expect("simple-lossless file builds from its own VP8L payload");
    let c2 = parse_container(&rebuilt).expect("rebuilt simple-lossless file parses");
    assert_eq!(c2.chunks.len(), 1);
    assert_eq!(c2.chunks[0].fourcc, fourcc::VP8L);
    assert_eq!(c2.chunks[0].payload(&rebuilt), payload);
    assert_eq!(c2.riff_file_size, c.riff_file_size);
}

#[test]
fn round6_lossy_1x1_fixture_extracts_to_typed_lossy_chunk_with_trace_dims() {
    // Round-6 surface: walker → typed lossy chunk. The
    // `lossy-1x1/trace.txt` golden output records the VP8 frame
    // header as
    //   VP8_FRAME_HEADER key_frame=1 profile=0 show=1
    //                    partition_length=11 width=1 height=1
    //                    xscale=0 yscale=0
    // The typed handle must surface every one of these fields,
    // and must also borrow the chunk payload verbatim so a
    // downstream VP8 decoder gets the exact bytes the walker saw.
    let handle = extract_lossy_chunk(LOSSY_1X1)
        .expect("lossy-1x1 fixture parses + extracts")
        .expect("lossy-1x1 fixture has a 'VP8 ' chunk");
    assert_eq!(handle.width(), 1, "trace width=1");
    assert_eq!(handle.height(), 1, "trace height=1");
    assert_eq!(handle.version(), 0, "trace profile=0");
    assert!(handle.show_frame(), "trace show=1");
    assert_eq!(
        handle.first_partition_size(),
        11,
        "trace partition_length=11"
    );
    assert_eq!(handle.horizontal_scale(), 0, "trace xscale=0");
    assert_eq!(handle.vertical_scale(), 0, "trace yscale=0");

    // Payload must include the §9.1 10-byte header at offset 0 plus
    // the entropy-coded remainder (the trace's CHUNK size=40 for the
    // VP8 chunk → 40 byte payload).
    assert_eq!(
        handle.bitstream().len(),
        40,
        "trace CHUNK fourcc=VP8 size=40"
    );
    assert!(handle.bitstream().len() >= VP8_KEYFRAME_HEADER_LEN);

    // The walker's chunk payload must match the handle's bitstream —
    // demonstrates the handle borrows out of the walker's slice
    // without modifying it.
    let c = parse_container(LOSSY_1X1).unwrap();
    let chunk = c.first_chunk_with_fourcc(fourcc::VP8).unwrap();
    assert_eq!(handle.bitstream(), chunk.payload(LOSSY_1X1));
}

#[test]
fn round6_lossy_with_alpha_extended_fixture_extracts_to_128x128_keyframe() {
    // Extended-format fixture: 'VP8X' + 'ALPH' + 'VP8 '. The §2.5
    // / §9.1 dims still come from the 'VP8 ' chunk's keyframe
    // header. The trace records width=128 height=128 partition=328.
    let handle = extract_lossy_chunk(LOSSY_WITH_ALPHA)
        .expect("lossy-with-alpha parses")
        .expect("'VP8 ' chunk present alongside 'VP8X' / 'ALPH'");
    assert_eq!(handle.width(), 128);
    assert_eq!(handle.height(), 128);
    assert_eq!(handle.first_partition_size(), 328);
    assert!(handle.show_frame());
    assert_eq!(handle.version(), 0);

    // The §2.7.1 VP8X-declared canvas must agree with the §9.1
    // keyframe-derived canvas — the typed handle does *not*
    // enforce that policy (cross-validation is the caller's job),
    // but the trace says they match for this fixture.
    let c = parse_container(LOSSY_WITH_ALPHA).unwrap();
    let vp8x = parse_vp8x_header(
        c.first_chunk_with_fourcc(fourcc::VP8X)
            .unwrap()
            .payload(LOSSY_WITH_ALPHA),
    )
    .unwrap();
    assert_eq!(vp8x.canvas_width, handle.width() as u32);
    assert_eq!(vp8x.canvas_height, handle.height() as u32);
}

#[test]
fn round6_lossless_fixture_extract_returns_none() {
    // §2.6 simple-lossless file has no 'VP8 ' chunk, only 'VP8L'.
    // `extract_lossy_chunk` must report that cleanly.
    let res = extract_lossy_chunk(LOSSLESS_1X1).expect("lossless-1x1 parses");
    assert!(res.is_none(), "lossless file carries no VP8 chunk");
}

#[test]
fn round6_lossy_chunk_payload_survives_round_trip_through_builder() {
    // Round-5 ↔ round-6 interplay: take the §2.5 'VP8 ' payload
    // surfaced by `extract_lossy_chunk`, re-wrap it via the
    // round-5 builder, and verify the re-extracted handle peeks
    // the same RFC 6386 §9.1 fields. This locks down the "router
    // ↔ container builder" contract for the encoder-replacement
    // pipeline.
    let original = extract_lossy_chunk(LOSSY_1X1).unwrap().unwrap();
    let bitstream = original.bitstream();

    let rebuilt = build_webp_file(bitstream, ImageKind::Lossy, 0, 0)
        .expect("simple-lossy file builds from extracted VP8 payload");
    let re_extracted = extract_lossy_chunk(&rebuilt)
        .expect("rebuilt file parses")
        .expect("rebuilt file carries a VP8 chunk");
    assert_eq!(re_extracted.width(), original.width());
    assert_eq!(re_extracted.height(), original.height());
    assert_eq!(re_extracted.version(), original.version());
    assert_eq!(
        re_extracted.first_partition_size(),
        original.first_partition_size()
    );
    assert_eq!(re_extracted.bitstream(), bitstream);
}

#[test]
fn round6_lossy_chunk_from_chunk_works_on_walker_output() {
    // Direct WebpLossyChunk::from_chunk path — bypasses the
    // top-level extract_lossy_chunk helper to verify the
    // chunk-level construction works too.
    let c = parse_container(LOSSY_1X1).unwrap();
    let vp8 = c.first_chunk_with_fourcc(fourcc::VP8).unwrap();
    let handle = WebpLossyChunk::from_chunk(LOSSY_1X1, vp8).expect("VP8 chunk handle constructs");
    assert_eq!(handle.width(), 1);
    assert_eq!(handle.height(), 1);
}

#[test]
fn round5_build_vp8x_chunk_round_trips_through_typed_parser_with_flags() {
    // §2.7.1 builder ↔ typed parser round-trip. We set every named
    // feature flag and a non-square canvas to catch any LE / bit-
    // position swap between writer and reader.
    let flags = Vp8xFlags {
        has_iccp: true,
        has_alpha: true,
        has_exif: false,
        has_xmp: true,
        has_animation: false,
    };
    let payload = build_vp8x_chunk(640, 480, flags).expect("VP8X payload builds");
    let h = parse_vp8x_header(&payload).expect("VP8X payload parses back");
    assert_eq!(h.canvas_width, 640);
    assert_eq!(h.canvas_height, 480);
    assert!(h.has_iccp);
    assert!(h.has_alpha);
    assert!(!h.has_exif);
    assert!(h.has_xmp);
    assert!(!h.has_animation);
    // The builder zero-fills every Reserved bit + the 24-bit reserved
    // field, so the parser's forward-compat signal must stay clear.
    assert!(!h.has_unknown);
}
