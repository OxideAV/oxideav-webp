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
use oxideav_webp::meta_prefix::{ImageRole, MetaPrefixCodes, MetaPrefixHeader};
use oxideav_webp::vp8_chunk::{WebpLossyChunk, VP8_KEYFRAME_HEADER_LEN};
use oxideav_webp::vp8l_chunk::{WebpLosslessChunk, VP8L_IMAGE_HEADER_LEN, VP8L_SIGNATURE};
use oxideav_webp::vp8l_decode::{decode_image, DecodeError};
use oxideav_webp::vp8l_prefix::PrefixCode;
use oxideav_webp::vp8l_stream::{BitReader, Transform, TransformList, TransformType};
use oxideav_webp::{
    build_vp8x_chunk, build_webp_file, extract_lossless_chunk, extract_lossy_chunk,
    parse_alph_header, parse_anim_header, parse_anmf_header, parse_container, parse_vp8x_header,
    read_vp8l_transform_list,
};

const LOSSY_1X1: &[u8] = include_bytes!("data/lossy-1x1.webp");
const LOSSLESS_1X1: &[u8] = include_bytes!("data/lossless-1x1.webp");
const LOSSLESS_32X32_RGBA: &[u8] = include_bytes!("data/lossless-32x32-rgba.webp");
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
fn round7_lossless_1x1_fixture_extracts_to_typed_lossless_chunk_with_trace_dims() {
    // Round-7 surface: walker → typed lossless chunk. The
    // `lossless-1x1/trace.txt` golden output records the VP8L
    // image-header as
    //   VP8L_HEADER magic=0x2f width=1 height=1 alpha_used=0 version=0
    // The typed handle must surface every one of these fields,
    // and must also borrow the chunk payload verbatim so a
    // downstream VP8L decoder gets the exact bytes the walker saw.
    let handle = extract_lossless_chunk(LOSSLESS_1X1)
        .expect("lossless-1x1 fixture parses + extracts")
        .expect("lossless-1x1 fixture has a 'VP8L' chunk");
    assert_eq!(handle.width(), 1, "trace width=1");
    assert_eq!(handle.height(), 1, "trace height=1");
    assert!(!handle.alpha_is_used(), "trace alpha_used=0");
    assert_eq!(handle.version(), 0, "trace version=0");

    // Payload must include the §3.4 / §7.1 5-byte image-header at
    // offset 0 plus the entropy-coded remainder (the trace's
    // CHUNK size=17 for the VP8L chunk → 17 byte payload).
    assert_eq!(handle.bitstream().len(), 17, "trace CHUNK size=17");
    assert!(handle.bitstream().len() >= VP8L_IMAGE_HEADER_LEN);
    assert_eq!(
        handle.bitstream()[0],
        VP8L_SIGNATURE,
        "VP8L payload byte 0 must be 0x2F"
    );

    // The walker's chunk payload must match the handle's bitstream —
    // demonstrates the handle borrows out of the walker's slice
    // without modifying it.
    let c = parse_container(LOSSLESS_1X1).unwrap();
    let chunk = c
        .first_chunk_with_fourcc(oxideav_webp::container::fourcc::VP8L)
        .unwrap();
    assert_eq!(handle.bitstream(), chunk.payload(LOSSLESS_1X1));
}

#[test]
fn round7_lossless_32x32_rgba_fixture_extracts_with_alpha_used_bit_set() {
    // `lossless-32x32-rgba/trace.txt` reports
    //   VP8L_HEADER magic=0x2f width=32 height=32 alpha_used=1 version=0
    // The fixture's RGBA source means the encoder set `alpha_is_used`
    // — this is the only fixture in the small in-crate corpus that
    // exercises the hint=1 path of the §3.4 / §7.1 bit decode.
    let handle = extract_lossless_chunk(LOSSLESS_32X32_RGBA)
        .expect("lossless-32x32-rgba fixture parses + extracts")
        .expect("lossless-32x32-rgba fixture has a 'VP8L' chunk");
    assert_eq!(handle.width(), 32, "trace width=32");
    assert_eq!(handle.height(), 32, "trace height=32");
    assert!(handle.alpha_is_used(), "trace alpha_used=1");
    assert_eq!(handle.version(), 0, "trace version=0");

    // Trace reports CHUNK size=121 — same length must hit
    // bitstream().
    assert_eq!(handle.bitstream().len(), 121, "trace CHUNK size=121");
    assert_eq!(handle.bitstream()[0], VP8L_SIGNATURE);
}

#[test]
fn round7_lossy_fixture_extract_lossless_returns_none() {
    // §2.5 simple-lossy file has no 'VP8L' chunk, only 'VP8 '.
    // `extract_lossless_chunk` must report that cleanly.
    let res = extract_lossless_chunk(LOSSY_1X1).expect("lossy-1x1 parses");
    assert!(res.is_none(), "lossy file carries no VP8L chunk");
}

#[test]
fn round7_lossless_chunk_payload_survives_round_trip_through_builder() {
    // Round-5 ↔ round-7 interplay: take the §2.6 'VP8L' payload
    // surfaced by `extract_lossless_chunk`, re-wrap it via the
    // round-5 builder, and verify the re-extracted handle peeks
    // the same §3.4 / §7.1 fields. Locks down the "router ↔
    // container builder" contract for the lossless path.
    let original = extract_lossless_chunk(LOSSLESS_1X1).unwrap().unwrap();
    let bitstream = original.bitstream();

    let rebuilt = build_webp_file(bitstream, ImageKind::Lossless, 0, 0)
        .expect("simple-lossless file builds from extracted VP8L payload");
    let re_extracted = extract_lossless_chunk(&rebuilt)
        .expect("rebuilt file parses")
        .expect("rebuilt file carries a VP8L chunk");
    assert_eq!(re_extracted.width(), original.width());
    assert_eq!(re_extracted.height(), original.height());
    assert_eq!(re_extracted.alpha_is_used(), original.alpha_is_used());
    assert_eq!(re_extracted.version(), original.version());
    assert_eq!(re_extracted.bitstream(), bitstream);
}

#[test]
fn round7_lossless_chunk_from_chunk_works_on_walker_output() {
    // Direct WebpLosslessChunk::from_chunk path — bypasses the
    // top-level extract_lossless_chunk helper to verify the
    // chunk-level construction works too.
    let c = parse_container(LOSSLESS_1X1).unwrap();
    let vp8l = c
        .first_chunk_with_fourcc(oxideav_webp::container::fourcc::VP8L)
        .unwrap();
    let handle =
        WebpLosslessChunk::from_chunk(LOSSLESS_1X1, vp8l).expect("VP8L chunk handle constructs");
    assert_eq!(handle.width(), 1);
    assert_eq!(handle.height(), 1);
    assert!(!handle.alpha_is_used());
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

#[test]
fn round99_lossless_1x1_transform_list_is_color_indexing_from_fixture() {
    // docs/image/webp/fixtures/lossless-1x1/trace.txt records:
    //   VP8L_TRANSFORM order=0 type=3 name=COLOR_INDEXING
    //   VP8L_TRANSFORM_PARAM type=3 num_colors=1 packed_bits=3
    // The §4 reader stops at the color-indexing transform's §5 body.
    let list = read_vp8l_transform_list(LOSSLESS_1X1)
        .expect("transform list reads")
        .expect("VP8L chunk present");
    assert_eq!(list.transforms().len(), 1);
    match list.transforms()[0] {
        Transform::ColorIndexing {
            color_table_size,
            width_bits,
        } => {
            assert_eq!(color_table_size, 1, "trace num_colors=1");
            assert_eq!(width_bits, 3, "trace packed_bits=3");
        }
        other => panic!("expected ColorIndexing, got {other:?}"),
    }
    assert_eq!(
        list.transforms()[0].transform_type(),
        TransformType::ColorIndexing
    );
    // Color-indexing carries a §5 color table — the reader halts there.
    assert!(list.stopped_at_entropy_body());
}

#[test]
fn round99_lossless_32x32_rgba_transform_list_matches_fixture_prefix() {
    // docs/image/webp/fixtures/lossless-32x32-rgba/trace.txt records:
    //   order=0 type=2 SUBTRACT_GREEN
    //   order=1 type=0 PREDICTOR  (TRANSFORM_PARAM bits=9)
    //   order=2 type=1 CROSS_COLOR
    // The §4 reader can advance past SUBTRACT_GREEN (no body) but
    // stops at PREDICTOR (§5 sub-resolution image), so it surfaces the
    // first two transforms only.
    let list = read_vp8l_transform_list(LOSSLESS_32X32_RGBA)
        .expect("transform list reads")
        .expect("VP8L chunk present");
    assert_eq!(list.transforms().len(), 2);
    assert_eq!(list.transforms()[0], Transform::SubtractGreen);
    match list.transforms()[1] {
        Transform::Predictor { size_bits } => assert_eq!(size_bits, 9, "trace bits=9"),
        other => panic!("expected Predictor, got {other:?}"),
    }
    assert!(list.stopped_at_entropy_body());
    // SUBTRACT_GREEN = 3 bits (present + 2-bit type), PREDICTOR =
    // present + 2-bit type + 3-bit size_bits = 6 bits; image-header is
    // 40 bits. So the §5 body begins at bit 40 + 3 + 6 = 49.
    assert_eq!(list.body_bit_position(), 49);
}

#[test]
fn round99_transform_list_returns_none_for_lossy_fixture() {
    // A simple-lossy file has no VP8L chunk, so the transform-list
    // reader returns Ok(None).
    assert!(read_vp8l_transform_list(LOSSY_1X1).unwrap().is_none());
}

#[test]
fn round104_lossless_1x1_color_table_prefix_group_matches_fixture_bytes() {
    // The round-99 §4 reader stops at the COLOR_INDEXING transform's §5
    // body. The body is the color-table image — itself a §5
    // entropy-coded stream that begins with the §5 color-cache info bit
    // and then a §6.2 prefix-code group (5 canonical prefix codes).
    //
    // This test resumes at the recorded body bit position and decodes
    // that prefix-code group with the round-104 §6.2.1 reader. The
    // golden values are derived purely from the fixture's VP8L payload
    // bytes (docs/image/webp/fixtures/lossless-1x1/input.webp) walked by
    // hand against the §2 bit-reader + §6.2.1 pseudocode:
    //   * color-cache use bit = 0 (matches trace VP8L_COLOR_CACHE
    //     color_cache_bits=0),
    //   * GREEN  = simple code, single symbol 60,
    //   * RED    = simple code, single symbol 180,
    //   * BLUE   = simple code, single symbol 90,
    //   * ALPHA  = simple code, single symbol 255,
    //   * DIST   = simple code, single symbol 0.
    // (ARGB = 255,180,60,90 — the single palette color of this 1×1 image.)
    let handle = extract_lossless_chunk(LOSSLESS_1X1)
        .expect("lossless-1x1 parses")
        .expect("lossless-1x1 has a VP8L chunk");
    let payload = handle.bitstream();

    // Read the §4 transform list to find where the COLOR_INDEXING §5
    // body begins.
    let mut reader = BitReader::new_after_image_header(payload);
    let list = TransformList::read(&mut reader).expect("transform list reads");
    assert!(list.stopped_at_entropy_body());
    assert!(matches!(
        list.transforms()[0],
        Transform::ColorIndexing { .. }
    ));

    // Resume at the body: the color-table image stream.
    reader.seek_to_bit(list.body_bit_position());

    // §5 color-cache info bit (and, here, no cache size since the bit is
    // 0). The color table is not in the ARGB role, so there is no
    // meta-Huffman bit — a single prefix-code group follows directly.
    let use_color_cache = reader.read_bit().expect("color-cache bit");
    assert!(!use_color_cache, "trace color_cache_bits=0");
    let cache_size = 0usize;

    // Prefix-code group: GREEN+length+cache, RED, BLUE, ALPHA, DIST.
    let green_alphabet = 256 + 24 + cache_size;
    let green = PrefixCode::read(&mut reader, green_alphabet).expect("GREEN code");
    assert_eq!(green.single_symbol(), Some(60), "GREEN single symbol");

    let red = PrefixCode::read(&mut reader, 256).expect("RED code");
    assert_eq!(red.single_symbol(), Some(180), "RED single symbol");

    let blue = PrefixCode::read(&mut reader, 256).expect("BLUE code");
    assert_eq!(blue.single_symbol(), Some(90), "BLUE single symbol");

    let alpha = PrefixCode::read(&mut reader, 256).expect("ALPHA code");
    assert_eq!(alpha.single_symbol(), Some(255), "ALPHA single symbol");

    let dist = PrefixCode::read(&mut reader, 40).expect("DIST code");
    assert_eq!(dist.single_symbol(), Some(0), "DIST single symbol");

    // Each single-symbol code consumes no bits when its symbol is read.
    let before = reader.bit_position();
    assert_eq!(green.read_symbol(&mut reader).unwrap(), 60);
    assert_eq!(red.read_symbol(&mut reader).unwrap(), 180);
    assert_eq!(blue.read_symbol(&mut reader).unwrap(), 90);
    assert_eq!(alpha.read_symbol(&mut reader).unwrap(), 255);
    assert_eq!(
        reader.bit_position(),
        before,
        "single-leaf reads consume 0 bits"
    );
}

#[test]
fn round106_lossless_1x1_color_table_meta_prefix_header_reads_single_group() {
    // Round-106 surface: the §5.2.3 color-cache info + §6.2.2 meta-prefix
    // + §6.2 5-prefix-code-group reader, exercised against the same
    // fixture round 104 used. The COLOR_INDEXING transform's color-table
    // image is an `entropy-coded-image` (§7.3 ABNF) — *not* the ARGB
    // role — so the §6.2.2 meta-prefix bit is NOT present; the reader
    // drops straight from the color-cache-info bit into the single
    // prefix-code group. Trace says `color_cache_bits=0` /
    // `num_htree_groups=1`.
    let handle = extract_lossless_chunk(LOSSLESS_1X1)
        .expect("lossless-1x1 parses")
        .expect("lossless-1x1 has a VP8L chunk");
    let payload = handle.bitstream();

    let mut reader = BitReader::new_after_image_header(payload);
    let list = TransformList::read(&mut reader).expect("transform list reads");
    assert!(list.stopped_at_entropy_body());
    let (color_table_w, color_table_h) = match list.transforms()[0] {
        Transform::ColorIndexing {
            color_table_size, ..
        } => (color_table_size as u32, 1u32),
        other => panic!("expected ColorIndexing, got {other:?}"),
    };
    reader.seek_to_bit(list.body_bit_position());

    // Read the §5.2.3 + §6.2 preamble for the color-table image as an
    // EntropyCoded role.
    let header = MetaPrefixHeader::read(
        &mut reader,
        ImageRole::EntropyCoded,
        color_table_w,
        color_table_h,
    )
    .expect("meta-prefix header reads");
    assert!(!header.color_cache.is_enabled());
    assert_eq!(header.color_cache.size(), 0);
    let group = header
        .codes
        .group()
        .expect("EntropyCoded → single group always");
    // ARGB palette = (255, 180, 60, 90) per round-104 derivation.
    assert_eq!(group.green.single_symbol(), Some(60));
    assert_eq!(group.red.single_symbol(), Some(180));
    assert_eq!(group.blue.single_symbol(), Some(90));
    assert_eq!(group.alpha.single_symbol(), Some(255));
    assert_eq!(group.distance.single_symbol(), Some(0));
}

#[test]
fn round106_meta_prefix_argb_single_group_synthetic_matches_trace_shape() {
    // Mirror the fixture-trace shape `VP8L_COLOR_CACHE color_cache_bits=0`
    // / `VP8L_HUFFMAN_GROUP meta_huffman=0 num_htree_groups=1` for an
    // ARGB-role read with no cache and no meta-Huffman split. The §6.2.2
    // dispatch bit IS read in this role; we set it to 0 and follow with
    // five simple single-symbol prefix codes.
    //
    // Synthetic byte stream (LSB-first per §2):
    //   bit 0: color-cache-info = 0      (cache disabled)
    //   bit 1: meta-prefix      = 0      (single group)
    //   then 5 × simple-code single-symbol codes (each 11 bits each:
    //     1 flag + 1 num + 1 is_first_8bits + 8 sym).
    fn write_bits(bytes: &mut Vec<u8>, bit_pos: &mut usize, mut value: u32, n: usize) {
        for _ in 0..n {
            let byte_idx = *bit_pos >> 3;
            if byte_idx >= bytes.len() {
                bytes.push(0);
            }
            let bit = (value & 1) as u8;
            bytes[byte_idx] |= bit << (*bit_pos & 7);
            *bit_pos += 1;
            value >>= 1;
        }
    }
    fn write_simple(bytes: &mut Vec<u8>, bit_pos: &mut usize, sym: u32) {
        write_bits(bytes, bit_pos, 1, 1); // simple
        write_bits(bytes, bit_pos, 0, 1); // num_symbols-1 = 0
        write_bits(bytes, bit_pos, 1, 1); // is_first_8bits = 1
        write_bits(bytes, bit_pos, sym, 8); // sym in 8 bits
    }

    let mut bytes: Vec<u8> = Vec::new();
    let mut bit_pos: usize = 0;
    write_bits(&mut bytes, &mut bit_pos, 0, 1); // color-cache-info=0
    write_bits(&mut bytes, &mut bit_pos, 0, 1); // meta-prefix=0
    for sym in [128u32, 64, 32, 255, 1] {
        write_simple(&mut bytes, &mut bit_pos, sym);
    }
    let mut reader = BitReader::new(&bytes);
    let header = MetaPrefixHeader::read(&mut reader, ImageRole::Argb, 32, 32).unwrap();
    assert!(!header.color_cache.is_enabled());
    let group = header.codes.group().expect("single group");
    assert_eq!(group.green.single_symbol(), Some(128));
    assert_eq!(group.red.single_symbol(), Some(64));
    assert_eq!(group.blue.single_symbol(), Some(32));
    assert_eq!(group.alpha.single_symbol(), Some(255));
    assert_eq!(group.distance.single_symbol(), Some(1));
}

#[test]
fn round106_meta_prefix_argb_multi_group_records_entropy_image_boundary() {
    // ARGB role, color-cache disabled, meta-prefix=1 with prefix_bits=4
    // (so block_size = 16). For a 128×128 ARGB image, ceil(128/16) = 8
    // → 8×8 entropy image. The reader records the boundary and STOPS
    // (the entropy image itself is a §5.2-encoded entropy-coded-image
    // that this round doesn't decode).
    fn write_bits(bytes: &mut Vec<u8>, bit_pos: &mut usize, mut value: u32, n: usize) {
        for _ in 0..n {
            let byte_idx = *bit_pos >> 3;
            if byte_idx >= bytes.len() {
                bytes.push(0);
            }
            let bit = (value & 1) as u8;
            bytes[byte_idx] |= bit << (*bit_pos & 7);
            *bit_pos += 1;
            value >>= 1;
        }
    }

    let mut bytes: Vec<u8> = Vec::new();
    let mut bit_pos: usize = 0;
    write_bits(&mut bytes, &mut bit_pos, 0, 1); // color-cache=0
    write_bits(&mut bytes, &mut bit_pos, 1, 1); // meta-prefix=1
    write_bits(&mut bytes, &mut bit_pos, 2, 3); // prefix_bits raw=2 → 4
                                                // pad another byte so the bit reader has slack past the boundary.
    bytes.push(0);

    let mut reader = BitReader::new(&bytes);
    let header = MetaPrefixHeader::read(&mut reader, ImageRole::Argb, 128, 128).unwrap();
    match header.codes {
        MetaPrefixCodes::EntropyImagePending {
            prefix_bits,
            image_width,
            image_height,
            entropy_image_bit_position,
        } => {
            assert_eq!(prefix_bits, 4);
            assert_eq!(image_width, 8);
            assert_eq!(image_height, 8);
            // 1 + 1 + 3 = 5 bits consumed
            assert_eq!(entropy_image_bit_position, 5);
        }
        other => panic!("expected EntropyImagePending, got {other:?}"),
    }
}

#[test]
fn round107_lossless_1x1_color_table_decodes_end_to_end_to_palette_pixel() {
    // End-to-end §5.2 decode of a real fixture's color-table image.
    //
    // The round-99 §4 reader stops at the COLOR_INDEXING transform's
    // §5 body — the color-table image, an `entropy-coded-image` of
    // width `color_table_size` and height 1. For `lossless-1x1.webp`,
    // `color_table_size = 1`, so this is a 1×1 entropy-coded image
    // holding a single palette color.
    //
    // This drives the full pipeline end-to-end:
    //   walk container → §4 transform list → resume at body →
    //   §5.2.3 + §6.2 meta-prefix header → §5.2 per-pixel decode loop.
    // The round-104/106 by-hand decode established the single palette
    // color (ARGB = 255,180,60,90). Here `decode_image` produces it
    // straight from the fixture's own VP8L payload bytes.
    let handle = extract_lossless_chunk(LOSSLESS_1X1)
        .expect("lossless-1x1 parses")
        .expect("lossless-1x1 has a VP8L chunk");
    let payload = handle.bitstream();

    let mut reader = BitReader::new_after_image_header(payload);
    let list = TransformList::read(&mut reader).expect("transform list reads");
    assert!(list.stopped_at_entropy_body());
    let color_table_w = match list.transforms()[0] {
        Transform::ColorIndexing {
            color_table_size, ..
        } => color_table_size as u32,
        other => panic!("expected ColorIndexing, got {other:?}"),
    };
    assert_eq!(color_table_w, 1, "lossless-1x1 has a single palette color");

    reader.seek_to_bit(list.body_bit_position());
    let header = MetaPrefixHeader::read(&mut reader, ImageRole::EntropyCoded, color_table_w, 1)
        .expect("meta-prefix header reads");
    // Trace: color_cache_bits=0 → no cache for this image.
    assert!(!header.color_cache.is_enabled());
    let group = header.codes.group().expect("EntropyCoded → single group");

    // Run the §5.2 per-pixel decode loop over the 1×1 color-table image.
    let img = decode_image(&mut reader, group, None, color_table_w, 1)
        .expect("color-table image decodes");
    assert_eq!(img.width(), 1);
    assert_eq!(img.height(), 1);
    // ARGB = (alpha=255, red=180, green=60, blue=90) → 0xFFB43C5A.
    assert_eq!(img.pixels(), &[0xFFB4_3C5Au32]);
}

#[test]
fn round107_decode_error_surfaces_through_crate_error() {
    // The §5.2 decode loop's `DecodeError` converts into the crate-wide
    // `oxideav_webp::Error` via `From`, so a higher-level decode entry
    // point can propagate it with `?`. This locks the wiring in lib.rs.
    let e = DecodeError::GreenSymbolOutOfRange {
        symbol: 280,
        alphabet_size: 280,
    };
    let wrapped: oxideav_webp::Error = e.into();
    assert!(matches!(wrapped, oxideav_webp::Error::Vp8lDecode(_)));
}
