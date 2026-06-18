//! Integration tests that drive [`oxideav_webp::parse_container`]
//! against a small in-crate copy of the WebP fixture corpus.
//!
//! The `tests/data/*.webp` files are byte-for-byte copies of the
//! `docs/image/webp/fixtures/<name>/input.webp` corpus. They live
//! inside the crate so that CI (which checks out only this repository,
//! not the umbrella workspace) can reach them via `include_bytes!`.
//!
//! Tests here fall in two layers. The early ones drive the structural
//! walker plus the `VP8X` / `ALPH` / `ANIM` header-field parsers over
//! *opaque* inputs. The later ones decode end-to-end through the full
//! §4–§6 lossless chain and check the reconstructed RGBA against each
//! fixture's committed `expected.png` ground truth — exact for every
//! lossless (and near-lossless) container, which the decoder
//! reconstructs bit-for-bit.

use oxideav_webp::alph::{AlphCompression, AlphFiltering, AlphPreprocessing};
use oxideav_webp::anmf::{BlendingMethod, DisposalMethod};
use oxideav_webp::build::{build_chunk, ImageKind, Vp8xFlags};
use oxideav_webp::container::fourcc;
use oxideav_webp::meta_prefix::{ImageRole, MetaPrefixCodes, MetaPrefixHeader};
use oxideav_webp::vp8_chunk::{WebpLossyChunk, VP8_KEYFRAME_HEADER_LEN};
use oxideav_webp::vp8l_chunk::{WebpLosslessChunk, VP8L_IMAGE_HEADER_LEN, VP8L_SIGNATURE};
use oxideav_webp::vp8l_decode::{
    decode_argb, decode_entropy_image, decode_image, DecodeError, MetaPrefixIndex,
};
use oxideav_webp::vp8l_prefix::PrefixCode;
use oxideav_webp::vp8l_stream::{BitReader, Transform, TransformList, TransformType};
use oxideav_webp::{
    build_vp8x_chunk, build_webp_file, decode_lossless_image, decode_webp, decode_webp_image,
    encode_webp_lossless, extract_lossless_chunk, extract_lossy_chunk, extract_metadata,
    parse_alph_header, parse_anim_header, parse_anmf_header, parse_container, parse_vp8x_header,
    read_vp8l_transform_list,
};

const LOSSY_1X1: &[u8] = include_bytes!("data/lossy-1x1.webp");
const LOSSLESS_1X1: &[u8] = include_bytes!("data/lossless-1x1.webp");
const LOSSLESS_32X32_RGBA: &[u8] = include_bytes!("data/lossless-32x32-rgba.webp");
const LOSSLESS_COLOR_INDEXING: &[u8] = include_bytes!("data/lossless-color-indexing-paletted.webp");
const EXTENDED_WITH_EXIF: &[u8] = include_bytes!("data/extended-with-exif.webp");
const EXTENDED_WITH_ICC: &[u8] = include_bytes!("data/extended-with-icc-profile.webp");
const EXTENDED_WITH_XMP: &[u8] = include_bytes!("data/extended-with-xmp.webp");
const LOSSY_WITH_ALPHA: &[u8] = include_bytes!("data/lossy-with-alpha-128x128.webp");
const ANIMATED_WITH_ALPHA: &[u8] = include_bytes!("data/animated-with-alpha.webp");
const LOSSLESS_32X32_RGB: &[u8] = include_bytes!("data/lossless-32x32-rgb.webp");
const LOSSLESS_COLOR_CACHE_STRESS: &[u8] = include_bytes!("data/lossless-color-cache-stress.webp");
const LOSSLESS_CROSS_COLOR_ACTIVE: &[u8] = include_bytes!("data/lossless-cross-color-active.webp");
const LOSSY_NEAR_LOSSLESS_Q40: &[u8] = include_bytes!("data/lossy-near-lossless-q40.webp");
const ANIMATED_3_FRAMES_RGB: &[u8] = include_bytes!("data/animated-3-frames-rgb.webp");

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
    // decode against a real reference-encoder-produced VP8X chunk. The
    // fixture's own `trace.txt` reports the same flags / dimensions.
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
fn round335_extract_metadata_exif_returns_the_corpus_exif_payload() {
    // §2.7.1.5 end-to-end EXIF *content* validation (the round-2 test
    // above only asserted the VP8X has_exif flag + chunk presence; this
    // pins the bytes `extract_metadata` hands back). The docs-corpus
    // `extended-with-exif` fixture carries a single 34-byte EXIF chunk
    // (its `trace.txt` records `CHUNK fourcc=VP8X` + an `EXIF` aux chunk).
    let meta = extract_metadata(EXTENDED_WITH_EXIF).expect("EXIF fixture metadata extracts");
    let exif = meta.exif.as_deref().expect("EXIF payload present");
    // Only EXIF is set on this fixture.
    assert!(meta.icc.is_none(), "no ICCP chunk on the EXIF fixture");
    assert!(meta.xmp.is_none(), "no XMP chunk on the EXIF fixture");

    // §2.7.1.5: the EXIF payload is a raw TIFF/Exif blob; this one opens
    // with the little-endian TIFF byte-order marker "II" (0x49 0x49) and
    // the 0x002A magic, i.e. `49 49 2A 00`.
    assert_eq!(exif.len(), 34, "EXIF chunk payload length");
    assert_eq!(
        &exif[..4],
        &[0x49, 0x49, 0x2a, 0x00],
        "TIFF little-endian header"
    );
    // Whole-payload digest lock — extracted bytes equal the chunk bytes
    // committed in the docs corpus (validator-free: the bytes are their
    // own ground truth, copied opaque from `fixtures/extended-with-exif`).
    assert_eq!(fnv1a64(exif), 0x2649_2d6d_ce37_6722);

    // Cross-check: the extracted payload is exactly the raw `EXIF` chunk
    // body the container walker observes (no header bytes leaked in/out).
    let c = parse_container(EXTENDED_WITH_EXIF).expect("EXIF fixture parses");
    let chunk = c
        .first_chunk_with_fourcc(fourcc::EXIF)
        .expect("EXIF chunk present");
    assert_eq!(exif, chunk.payload(EXTENDED_WITH_EXIF));
}

#[test]
fn round335_extract_metadata_iccp_returns_the_corpus_icc_profile() {
    // §2.7.1.4 end-to-end ICCP extraction against the docs corpus. The
    // `extended-with-icc-profile` fixture is a lossy `VP8 ` image wrapped
    // in a VP8X container with `has_icc=1` and a 292-byte `ICCP` chunk
    // (its `trace.txt`: `VP8X flags=0x00000020 has_icc=1`, `CHUNK
    // fourcc=ICCP size=292`). libwebp-style demuxers expose ICC data
    // verbatim without interpreting it; so does this crate.
    let meta = extract_metadata(EXTENDED_WITH_ICC).expect("ICCP fixture metadata extracts");
    let icc = meta.icc.as_deref().expect("ICCP payload present");
    assert!(meta.exif.is_none(), "no EXIF chunk on the ICCP fixture");
    assert!(meta.xmp.is_none(), "no XMP chunk on the ICCP fixture");

    // The VP8X flag byte agrees with the trace (has_icc only).
    let c = parse_container(EXTENDED_WITH_ICC).expect("ICCP fixture parses");
    let vp8x = c
        .first_chunk_with_fourcc(fourcc::VP8X)
        .expect("VP8X present");
    let h = parse_vp8x_header(vp8x.payload(EXTENDED_WITH_ICC)).expect("VP8X parses");
    assert_eq!((h.canvas_width, h.canvas_height), (128, 128));
    assert!(h.has_iccp);
    assert!(!h.has_exif && !h.has_xmp && !h.has_alpha && !h.has_animation);

    // Payload length + whole-payload digest lock (extracted bytes == the
    // corpus `ICCP` chunk body).
    assert_eq!(icc.len(), 292, "ICCP chunk payload length");
    assert_eq!(fnv1a64(icc), 0x104a_7b5d_d38a_ffd6);
    let chunk = c
        .first_chunk_with_fourcc(fourcc::ICCP)
        .expect("ICCP chunk present");
    assert_eq!(icc, chunk.payload(EXTENDED_WITH_ICC));
}

#[test]
fn round335_extract_metadata_xmp_returns_the_corpus_xmp_packet() {
    // §2.7.1.5 end-to-end XMP extraction against the docs corpus. The
    // `extended-with-xmp` fixture is a lossy `VP8 ` image wrapped in a
    // VP8X container with `has_xmp=1` and a 306-byte `XMP ` chunk (its
    // `trace.txt`: `VP8X flags=0x00000004 has_xmp=1`). The XMP payload is
    // an RDF/XML packet; this crate surfaces it as raw UTF-8 bytes.
    let meta = extract_metadata(EXTENDED_WITH_XMP).expect("XMP fixture metadata extracts");
    let xmp = meta.xmp.as_deref().expect("XMP payload present");
    assert!(meta.icc.is_none(), "no ICCP chunk on the XMP fixture");
    assert!(meta.exif.is_none(), "no EXIF chunk on the XMP fixture");

    let c = parse_container(EXTENDED_WITH_XMP).expect("XMP fixture parses");
    let vp8x = c
        .first_chunk_with_fourcc(fourcc::VP8X)
        .expect("VP8X present");
    let h = parse_vp8x_header(vp8x.payload(EXTENDED_WITH_XMP)).expect("VP8X parses");
    assert_eq!((h.canvas_width, h.canvas_height), (128, 128));
    assert!(h.has_xmp);
    assert!(!h.has_iccp && !h.has_exif && !h.has_alpha && !h.has_animation);

    // The packet opens with the standard XMP `<?xpacket begin=` processing
    // instruction and carries the fixture's single `dc:creator` field
    // (per `extended-with-xmp/notes.md`), proving the bytes are intact
    // UTF-8 XML and not truncated.
    assert_eq!(xmp.len(), 306, "XMP chunk payload length");
    assert!(
        xmp.starts_with(b"<?xpacket begin="),
        "XMP opens with the xpacket PI",
    );
    assert!(
        xmp.windows(b"dc:creator".len()).any(|w| w == b"dc:creator"),
        "XMP carries the dc:creator field",
    );
    assert_eq!(fnv1a64(xmp), 0x1d56_47f4_c563_aca6);
    let chunk = c
        .first_chunk_with_fourcc(fourcc::XMP)
        .expect("XMP chunk present");
    assert_eq!(xmp, chunk.payload(EXTENDED_WITH_XMP));
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
    // proper, decoded end-to-end by the round-110 test below.
    let bitstream = &alph.payload(LOSSY_WITH_ALPHA)[h.bitstream_offset()..];
    assert_eq!(bitstream.len(), (alph.size as usize) - 1);
}

/// FNV-1a (64-bit) over a byte slice — a small dependency-free digest
/// used to lock the full alpha plane to a known-good value.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn round110_lossy_with_alpha_decodes_full_alpha_plane_via_vp8l() {
    // §2.7.1.2 end-to-end: ALPH method=1 (headerless VP8L image-stream
    // with a PREDICTOR transform), filter=0 (none). The decoded alpha
    // is lifted from the GREEN channel and the (no-op) inverse filter
    // is applied. Dimensions are the §2.7.1 VP8X canvas (128x128).
    //
    // The full-plane FNV-1a digest and the corner / center / interior
    // sample anchors below are locked to the bytes produced by a
    // black-box reference-encoder alpha-extraction validator over this
    // same fixture (all 16384 bytes verified identical), per the round's
    // clean-room allow-list (validator output as ground truth; its
    // source unread).
    let plane = oxideav_webp::decode_alpha_plane(LOSSY_WITH_ALPHA)
        .expect("lossy-with-alpha decodes")
        .expect("ALPH chunk present");
    assert_eq!(plane.len(), 128 * 128, "full-resolution alpha plane");

    let w = 128usize;
    let at = |x: usize, y: usize| plane[y * w + x];

    // Fully-transparent border corners; opaque interior.
    assert_eq!(at(0, 0), 0);
    assert_eq!(at(127, 0), 0);
    assert_eq!(at(0, 127), 0);
    assert_eq!(at(127, 127), 0);
    assert_eq!(at(64, 64), 255);
    assert_eq!(at(32, 96), 255);

    // The plane spans the full 0..=255 range.
    assert_eq!(*plane.iter().min().unwrap(), 0);
    assert_eq!(*plane.iter().max().unwrap(), 255);
    assert_eq!(plane.iter().filter(|&&v| v == 255).count(), 7845);

    // Whole-plane digest lock (bit-exact vs. the validator output).
    assert_eq!(fnv1a64(&plane), 0x42e1_6029_2eb0_d472);
}

#[test]
fn round284_fixture_corpus_decode_digests_are_pinned() {
    // §6.2.1 read_symbol lookup-table fast path (round 284): the decoded
    // output of every decodable in-crate fixture is pinned byte-for-byte
    // to the output of the pre-r284 per-bit reference walk (the two
    // implementations were proven digest-identical over this corpus plus
    // the docs fixture set before the table landed). Any future
    // entropy-path rewrite must keep every digest below unchanged.
    //
    // Digest layout: FNV-1a-64 over `width_le ‖ height_le ‖
    // (frame_count as u32)_le ‖ frames[..].rgba` — dimensions and frame
    // count are folded in so a decoder that returns the right bytes at
    // the wrong geometry cannot collide.
    let cases: [(&str, &[u8], u64); 8] = [
        (
            "animated-with-alpha",
            ANIMATED_WITH_ALPHA,
            0x31c0_566b_ec09_d6a0,
        ),
        (
            "extended-with-exif",
            EXTENDED_WITH_EXIF,
            0x10bb_4321_d897_9d82,
        ),
        (
            "lossless-128x128-natural",
            include_bytes!("data/lossless-128x128-natural.webp"),
            0xf99f_a694_5bbc_f07c,
        ),
        ("lossless-1x1", LOSSLESS_1X1, 0x9653_f403_5cc4_a62f),
        (
            "lossless-32x32-rgba",
            LOSSLESS_32X32_RGBA,
            0xdbfc_b5cc_d1a0_d204,
        ),
        (
            "lossless-color-indexing-paletted",
            LOSSLESS_COLOR_INDEXING,
            0xbfc8_1c3b_8ce3_ca04,
        ),
        ("lossy-1x1", LOSSY_1X1, 0x9abb_50fd_9e15_d10d),
        (
            "lossy-with-alpha-128x128",
            LOSSY_WITH_ALPHA,
            0x00a4_e8ba_d528_34bf,
        ),
    ];
    for (name, bytes, want) in cases {
        let img = decode_webp(bytes).unwrap_or_else(|e| panic!("{name} decodes: {e}"));
        let mut buf = Vec::new();
        buf.extend_from_slice(&img.width.to_le_bytes());
        buf.extend_from_slice(&img.height.to_le_bytes());
        buf.extend_from_slice(&(img.frames.len() as u32).to_le_bytes());
        for f in &img.frames {
            buf.extend_from_slice(&f.rgba);
        }
        assert_eq!(fnv1a64(&buf), want, "{name} decoded-output digest");
    }
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
fn round118_decode_webp_animated_with_alpha_yields_three_rgba_frames() {
    // Round-118 surface: `decode_webp` assembles a genuine
    // reference-encoder-produced VP8L animation (3 ANMF frames, each a
    // VP8L sub-chunk) into N flat-RGBA
    // `WebpFrame`s, populating the ANIM background / loop count. The
    // per-frame headers (from `animated-with-alpha/trace.txt`) report
    // 64x64 / duration=100 / blend=1 / dispose=0 for all three frames; the
    // ANIM payload is bgcolor=0xffffffff (white opaque) loop_count=0.
    let img = decode_webp(ANIMATED_WITH_ALPHA).expect("animated VP8L file decodes");
    assert_eq!(img.frames.len(), 3, "one WebpFrame per ANMF chunk");
    for (i, frame) in img.frames.iter().enumerate() {
        assert_eq!(frame.width, 64, "frame {i} width");
        assert_eq!(frame.height, 64, "frame {i} height");
        assert_eq!(
            frame.duration_ms, 100,
            "frame {i} duration (ANMF native ms)"
        );
        // Flat-buffer invariant: tightly packed RGBA, no stride padding.
        assert_eq!(frame.rgba.len(), 64 * 64 * 4, "frame {i} flat RGBA length");
    }
    // ANIM globals surface on the WebpImage.
    assert_eq!(
        img.anim_background_rgba,
        Some([0xFF, 0xFF, 0xFF, 0xFF]),
        "ANIM background = white opaque (BGRA 0xffffffff → RGBA white)"
    );
    assert_eq!(
        img.anim_loop_count,
        Some(0),
        "ANIM loop_count = 0 (infinite)"
    );
    // The three frames are not all identical (the animation actually moves).
    assert_ne!(
        img.frames[0].rgba, img.frames[1].rgba,
        "frames 0 and 1 differ"
    );
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
    // We rip the §2.5 `VP8 ` chunk's payload out of the reference-encoder-
    // produced lossy-1x1.webp, hand it back to the builder, and verify the
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

/// LSB-first bit writer for the round-108 synthetic-stream tests.
struct R108BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}
impl R108BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn write_bits(&mut self, mut value: u32, n: usize) {
        for _ in 0..n {
            let byte_idx = self.bit_pos >> 3;
            if byte_idx >= self.bytes.len() {
                self.bytes.push(0);
            }
            let bit = (value & 1) as u8;
            self.bytes[byte_idx] |= bit << (self.bit_pos & 7);
            self.bit_pos += 1;
            value >>= 1;
        }
    }
    /// §6.2.1 simple-code single-symbol code (length-1 leaf, 8-bit form).
    fn simple_single(&mut self, sym: u32) {
        self.write_bits(1, 1); // simple flag
        self.write_bits(0, 1); // num_symbols - 1 = 0
        self.write_bits(1, 1); // is_first_8bits = 1
        self.write_bits(sym, 8);
    }
    /// §6.2.1 simple-code two-symbol code, both length 1.
    fn simple_two(&mut self, a: u32, b: u32) {
        self.write_bits(1, 1);
        self.write_bits(1, 1); // num_symbols - 1 = 1
        self.write_bits(1, 1);
        self.write_bits(a, 8);
        self.write_bits(b, 8);
    }
    fn group_single(&mut self, g: u32, r: u32, b: u32, a: u32, d: u32) {
        self.simple_single(g);
        self.simple_single(r);
        self.simple_single(b);
        self.simple_single(a);
        self.simple_single(d);
    }
    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[test]
fn round108_decode_argb_multi_group_through_public_api() {
    // The §6.2.2 multi-group ARGB decode reached through the public
    // `decode_argb` entry point. An 8×1 ARGB image, prefix_bits=2 (block
    // size 4) → 2 horizontal blocks. The entropy image (2×1) selects
    // group 0 for block 0 (pixels 0..3) and group 1 for block 1
    // (pixels 4..7). Group 0 emits green=111; group 1 emits green=222.
    let mut w = R108BitWriter::new();
    // spatially-coded-image header (ARGB role)
    w.write_bits(0, 1); // main color-cache-info = disabled
    w.write_bits(1, 1); // meta-prefix = 1 → multi-group
    w.write_bits(0, 3); // prefix_bits raw = 0 → 2 (block 4)
                        // entropy image: 2×1 entropy-coded-image
    w.write_bits(0, 1); // entropy color-cache-info = disabled
    w.simple_two(0, 1); // GREEN {0, 1}
    w.simple_single(0); // RED
    w.simple_single(0); // BLUE
    w.simple_single(0); // ALPHA
    w.simple_single(0); // DIST
    w.write_bits(0, 1); // entropy pixel 0 GREEN = 0 → meta-code 0
    w.write_bits(1, 1); // entropy pixel 1 GREEN = 1 → meta-code 1
                        // 2 prefix-code groups
    w.group_single(111, 0x10, 0x20, 0x30, 0);
    w.group_single(222, 0x40, 0x50, 0x60, 0);
    // main image data: 8 single-symbol literals → no bits.
    let data = w.into_bytes();

    let mut r = BitReader::new(&data);
    let img = decode_argb(&mut r, 8, 1).expect("multi-group ARGB decodes");
    assert_eq!(img.width(), 8);
    assert_eq!(img.height(), 1);
    let pixels = img.pixels();
    // First block: group 0; second block: group 1.
    for &p in &pixels[0..4] {
        assert_eq!(p & 0x0000_ff00, 111u32 << 8, "block 0 green should be 111");
    }
    for &p in &pixels[4..8] {
        assert_eq!(p & 0x0000_ff00, 222u32 << 8, "block 1 green should be 222");
    }
}

#[test]
fn round108_decode_argb_single_group_through_public_api() {
    // The single-group path of the same public `decode_argb`: meta-prefix
    // bit 0 → one group everywhere. A 3×1 image whose single-leaf GREEN
    // (no data bits) makes all three pixels the same literal.
    let mut w = R108BitWriter::new();
    w.write_bits(0, 1); // color-cache disabled
    w.write_bits(0, 1); // meta-prefix = 0 → single group
    w.group_single(0x33, 0x21, 0x22, 0x23, 0); // GREEN=0x33 single-leaf
    let data = w.into_bytes();
    let mut r = BitReader::new(&data);
    let img = decode_argb(&mut r, 3, 1).expect("single-group ARGB decodes");
    // All three pixels are the same literal:
    // (alpha=0x23, red=0x21, green=0x33, blue=0x22).
    let expected = (0x23u32 << 24) | (0x21u32 << 16) | (0x33u32 << 8) | 0x22u32;
    assert_eq!(img.pixels(), &[expected, expected, expected]);
}

#[test]
fn round108_decode_entropy_image_public_api_and_num_groups() {
    // `decode_entropy_image` is public: decode a 2×1 entropy image and
    // confirm the §6.2.2 meta-code extraction + num_prefix_groups.
    let mut w = R108BitWriter::new();
    w.write_bits(0, 1); // color-cache disabled
    w.simple_two(0, 4); // GREEN {0, 4}
    w.simple_single(0); // RED
    w.simple_single(0); // BLUE
    w.simple_single(0); // ALPHA
    w.simple_single(0); // DIST
    w.write_bits(0, 1); // pixel 0 GREEN = 0 → meta 0
    w.write_bits(1, 1); // pixel 1 GREEN = 4 → meta 4
    let data = w.into_bytes();
    let mut r = BitReader::new(&data);
    let index: MetaPrefixIndex = decode_entropy_image(&mut r, 2, 2, 1).unwrap();
    assert_eq!(index.meta_codes(), &[0, 4]);
    // num_prefix_groups = max(0,4) + 1 = 5 (max-based, not block count).
    assert_eq!(index.num_prefix_groups(), 5);
    assert_eq!(index.meta_code_for(0, 0), 0);
    assert_eq!(index.meta_code_for(4, 0), 4);
}

// ---- round 109: §4 inverse transforms, end-to-end ----

#[test]
fn round109_lossless_1x1_color_indexing_decodes_end_to_end() {
    // `lossless-1x1.webp` is a COLOR_INDEXING transform with a single
    // palette color (packed_bits=3). The full pipeline — §4 transform
    // list + §5 color-table body + §5/§6 main ARGB decode + §4.4 inverse
    // color-indexing — yields the palette color in the single pixel.
    let img = decode_lossless_image(LOSSLESS_1X1)
        .expect("lossless-1x1 decodes")
        .expect("lossless-1x1 has a VP8L chunk");
    assert_eq!(img.width(), 1);
    assert_eq!(img.height(), 1);
    // ARGB = (alpha=255, red=180, green=60, blue=90) → 0xFFB43C5A.
    assert_eq!(img.pixels(), &[0xFFB4_3C5Au32]);
}

#[test]
fn round109_lossless_color_indexing_paletted_decodes_end_to_end() {
    // 32×32 with an 8-color palette → COLOR_INDEXING with width_bits=1
    // (2 indices bundled per green byte). End-to-end decode against the
    // fixture's `expected.png` ARGB ground truth.
    let img = decode_lossless_image(LOSSLESS_COLOR_INDEXING)
        .expect("palette fixture decodes")
        .expect("palette fixture has a VP8L chunk");
    assert_eq!(img.width(), 32);
    assert_eq!(img.height(), 32);
    let px = img.pixels();
    assert_eq!(px.len(), 1024);
    // First row begins with 4 red, then 4 green pixels.
    assert_eq!(&px[0..4], &[0xFFFF_0000u32; 4]);
    assert_eq!(&px[4..8], &[0xFF00_FF00u32; 4]);
    // Interior + corner spot checks (from expected.png).
    assert_eq!(px[31], 0xFFDC_DCDCu32); // row 0, col 31
    assert_eq!(px[16 * 32 + 16], 0xFFFF_0000u32); // row 16, col 16
    assert_eq!(px[31 * 32], 0xFF00_FFFFu32); // row 31, col 0
    assert_eq!(px[1023], 0xFFFF_00FFu32); // last pixel (magenta)
}

#[test]
fn round109_lossless_32x32_rgba_full_transform_chain_decodes_end_to_end() {
    // `lossless-32x32-rgba.webp` exercises three transforms at once:
    // SUBTRACT_GREEN, then PREDICTOR (size_bits=9), then CROSS_COLOR
    // (size_bits=3), plus a level-1 color cache on the main image. The
    // inverse chain is applied in reverse read order. Validated against
    // the fixture's `expected.png` ARGB ground truth (with real alpha).
    let img = decode_lossless_image(LOSSLESS_32X32_RGBA)
        .expect("rgba fixture decodes")
        .expect("rgba fixture has a VP8L chunk");
    assert_eq!(img.width(), 32);
    assert_eq!(img.height(), 32);
    let px = img.pixels();
    assert_eq!(px.len(), 1024);
    // Top-left pixel is fully transparent (0x00000000).
    assert_eq!(px[0], 0x0000_0000u32);
    assert_eq!(px[1], 0x0808_0080u32);
    assert_eq!(px[7], 0x3838_0080u32);
    // Interior + corner spot checks (from expected.png).
    assert_eq!(px[31], 0xF8F8_0080u32); // row 0, col 31
    assert_eq!(px[16 * 32 + 16], 0x8080_8080u32); // row 16, col 16
    assert_eq!(px[31 * 32], 0x0000_0000u32); // row 31, col 0
    assert_eq!(px[1023], 0xF8F8_F880u32); // last pixel
}

#[test]
fn round109_decode_lossless_image_returns_none_for_lossy_file() {
    // A VP8-only (lossy) file has no VP8L chunk → Ok(None).
    let out = decode_lossless_image(LOSSY_1X1).expect("lossy file parses");
    assert!(out.is_none(), "lossy file has no VP8L chunk");
}

// ---------------------------------------------------------------------
// Round 111 — top-level `decode_webp` / `decode_webp_image` wiring.
//
// `decode_webp_image` walks the container, decodes the §2.6 VP8L
// lossless image through the full §4–§6 chain, optionally overrides
// alpha from a §2.7.1.2 ALPH chunk, and returns interleaved
// `[R, G, B, A]` bytes. `decode_webp` is the flat-buffer shorthand.
// ---------------------------------------------------------------------

/// Pull the **full** `VP8L` chunk payload (5-byte image header plus the
/// entropy-coded bitstream) out of a simple-lossless fixture, so a test
/// can re-wrap it in a synthetic `VP8X`-extended (or `ALPH`-bearing)
/// container using the crate's own §2.3 chunk builder. No external
/// container parsing — this reuses `parse_container` + the typed handle.
fn vp8l_payload(file: &[u8]) -> Vec<u8> {
    let chunk = extract_lossless_chunk(file)
        .expect("fixture walks")
        .expect("fixture has a VP8L chunk");
    // `bitstream()` is the full VP8L payload (header + entropy bytes).
    chunk.bitstream().to_vec()
}

#[test]
fn round111_decode_webp_image_simple_lossless_1x1_rgba() {
    // Simple lossless: one VP8L chunk, no VP8X. ARGB ground truth from
    // the round-109 decode is 0xFFB43C5A → RGBA [B4, 3C, 5A, FF].
    let img = decode_webp_image(LOSSLESS_1X1).expect("lossless-1x1 decodes to an image");
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
    assert_eq!(img.rgba, vec![0xB4, 0x3C, 0x5A, 0xFF]);
    // `decode_webp` returns the published `WebpImage` shape: one frame
    // carrying the same flat RGBA buffer.
    let webp = decode_webp(LOSSLESS_1X1).unwrap();
    assert_eq!(webp.frames.len(), 1);
    assert_eq!(webp.frames[0].rgba, img.rgba);
    assert_eq!(webp.frames[0].width, 1);
    assert_eq!(webp.frames[0].height, 1);
    assert_eq!(webp.frames[0].duration_ms, 0);
    assert_eq!(webp.anim_background_rgba, None);
    assert_eq!(webp.anim_loop_count, None);
}

#[test]
fn round111_decode_webp_image_color_indexing_paletted_rgba() {
    // 32x32 COLOR_INDEXING fixture; spot-check first/last pixels against
    // the round-109 ARGB ground truth (0xFFFF0000 red, 0xFFFF00FF
    // magenta) repacked to [R, G, B, A].
    let img = decode_webp_image(LOSSLESS_COLOR_INDEXING).expect("paletted fixture decodes");
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.rgba.len(), 32 * 32 * 4);
    // px[0] = 0xFFFF0000 → R=FF G=00 B=00 A=FF.
    assert_eq!(&img.rgba[0..4], &[0xFF, 0x00, 0x00, 0xFF]);
    // px[1023] (last) = 0xFFFF00FF → R=FF G=00 B=FF A=FF.
    let last = img.rgba.len() - 4;
    assert_eq!(&img.rgba[last..], &[0xFF, 0x00, 0xFF, 0xFF]);
}

#[test]
fn round111_decode_webp_image_32x32_rgba_carries_vp8l_alpha() {
    // The 32x32 RGBA fixture carries real (non-opaque) alpha inside the
    // VP8L stream itself. Top-left is fully transparent (ARGB 0x00000000),
    // so the repacked alpha byte must be 0x00 — proving alpha survives
    // the ARGB→RGBA repack.
    let img = decode_webp_image(LOSSLESS_32X32_RGBA).expect("rgba fixture decodes");
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    // Top-left transparent corner.
    assert_eq!(&img.rgba[0..4], &[0x00, 0x00, 0x00, 0x00]);
    // px[1023] ARGB 0xF8F8F880 → A=F8 R=F8 G=F8 B=80 → RGBA [F8, F8, 80, F8].
    let last = img.rgba.len() - 4;
    assert_eq!(&img.rgba[last..], &[0xF8, 0xF8, 0x80, 0xF8]);
}

#[test]
fn round111_decode_webp_image_extended_vp8x_plus_vp8l() {
    // §2.7 extended-lossless: re-wrap the lossless-1x1 VP8L payload in a
    // VP8X-fronted file via the crate's own builder, then decode. The
    // VP8X canvas dims and the decoded pixels must match the simple case.
    let payload = vp8l_payload(LOSSLESS_1X1);
    let extended = build_webp_file(&payload, ImageKind::ExtendedLossless, 1, 1)
        .expect("build VP8X + VP8L file");

    // Sanity: the synthesized file really carries a VP8X chunk.
    let c = parse_container(&extended).expect("extended file walks");
    assert!(
        c.first_chunk_with_fourcc(fourcc::VP8X).is_some(),
        "synthetic file has a VP8X chunk"
    );

    let img = decode_webp_image(&extended).expect("extended VP8X+VP8L decodes");
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
    assert_eq!(img.rgba, vec![0xB4, 0x3C, 0x5A, 0xFF]);
}

#[test]
fn round111_decode_webp_image_vp8x_vp8l_with_alph_overrides_alpha() {
    // §2.7.1.2: the spec discourages ALPH alongside VP8L ("SHOULD NOT"),
    // but does not forbid it. When present, the decoded alpha plane
    // overrides the VP8L per-pixel alpha. Hand-assemble a 1x1
    // VP8X + VP8L + ALPH(raw, filter=none) file: the VP8L pixel is
    // 0xFFB43C5A (RGB B4,3C,5A) and the raw alpha byte is 0x42, so the
    // result is RGBA [B4, 3C, 5A, 42] — alpha taken from ALPH, not FF.
    let vp8l = vp8l_payload(LOSSLESS_1X1);
    let vp8x_payload = build_vp8x_chunk(
        1,
        1,
        Vp8xFlags {
            has_alpha: true,
            ..Default::default()
        },
    )
    .expect("build VP8X payload");

    // ALPH payload: info byte 0x00 (method 0 raw, filter 0 none) + the
    // raw alpha plane (width*height = 1 byte).
    let alph_payload = vec![0x00u8, 0x42u8];

    let mut body = Vec::new();
    body.extend_from_slice(&build_chunk(fourcc::VP8X, &vp8x_payload).unwrap());
    body.extend_from_slice(&build_chunk(fourcc::ALPH, &alph_payload).unwrap());
    body.extend_from_slice(&build_chunk(fourcc::VP8L, &vp8l).unwrap());

    let mut file = Vec::new();
    file.extend_from_slice(&fourcc::RIFF);
    file.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
    file.extend_from_slice(&fourcc::WEBP);
    file.extend_from_slice(&body);

    let img = decode_webp_image(&file).expect("VP8X+VP8L+ALPH decodes");
    assert_eq!(img.width, 1);
    assert_eq!(img.height, 1);
    // RGB from VP8L, alpha overridden by the ALPH plane (0x42).
    assert_eq!(img.rgba, vec![0xB4, 0x3C, 0x5A, 0x42]);
}

#[test]
fn round124_decode_webp_lossy_1x1_decodes_via_oxideav_vp8() {
    // Round 124: the §2.5 simple-lossy `VP8 ` chunk is now decoded
    // through the `oxideav-vp8` sibling crate (previously a clean
    // Unsupported). Assert the reference-encoder-produced 1x1 fixture
    // decodes to a single 1x1 flat-RGBA frame with opaque alpha.
    let img = decode_webp(LOSSY_1X1).expect("lossy-1x1 decodes via oxideav-vp8");
    assert_eq!(img.frames.len(), 1, "still image yields one frame");
    let frame = &img.frames[0];
    assert_eq!(frame.width, 1);
    assert_eq!(frame.height, 1);
    assert_eq!(frame.rgba.len(), 4, "1x1 RGBA is exactly 4 bytes");
    assert_eq!(frame.rgba[3], 0xff, "no ALPH chunk → opaque alpha");
    // Non-animated: ANIM fields absent.
    assert_eq!(img.anim_background_rgba, None);
    assert_eq!(img.anim_loop_count, None);
}

#[test]
fn round124_decode_webp_lossy_with_alpha_decodes_to_128x128() {
    // The VP8X + ALPH + VP8 (lossy) fixture: the VP8 bitstream supplies
    // the opaque RGB picture and the ALPH chunk supplies the alpha plane.
    // Round 124 decodes both; assert the dimensions and the flat-buffer
    // length identity (pixel-exactness vs a reference decoder is not
    // required).
    let img = decode_webp(LOSSY_WITH_ALPHA).expect("lossy+alpha decodes");
    assert_eq!(img.frames.len(), 1);
    let frame = &img.frames[0];
    assert_eq!(frame.width, 128);
    assert_eq!(frame.height, 128);
    assert_eq!(
        frame.rgba.len(),
        128 * 128 * 4,
        "flat RGBA buffer, no stride padding"
    );
    // The fixture's ALPH chunk decodes a non-trivial alpha plane, so at
    // least one pixel must be non-opaque (the alpha override took effect).
    let has_transparency = frame.rgba.chunks_exact(4).any(|p| p[3] != 0xff);
    assert!(
        has_transparency,
        "ALPH plane should have introduced transparency"
    );
}

#[test]
fn round124_decode_webp_image_lossy_low_level_path() {
    // The low-level `decode_webp_image -> DecodedWebp` entry also routes
    // VP8 lossy through oxideav-vp8 (round 124), no longer Unsupported.
    let decoded = decode_webp_image(LOSSY_1X1).expect("low-level lossy decode");
    assert_eq!(decoded.width, 1);
    assert_eq!(decoded.height, 1);
    assert_eq!(decoded.rgba.len(), 4);
}

// ---------------------------------------------------------------------------
// Round 115 — VP8L lossless ENCODER round trips.
//
// The encoder produces a RIFF/WEBP file with a VP8L chunk that the existing
// decoder reads back pixel-exact. These tests drive a real fixture's decoded
// pixels through the encoder and confirm the re-decode is identical — the
// strongest end-to-end check available (encode and decode are independent
// code paths).
// ---------------------------------------------------------------------------

/// Decode a fixture, re-encode the RGBA, decode again, and assert the
/// pixels survived unchanged.
fn assert_encoder_round_trips(file: &[u8]) {
    let original = decode_webp_image(file).expect("fixture decodes");
    let reencoded = encode_webp_lossless(&original.rgba, original.width, original.height)
        .expect("RGBA re-encodes to VP8L");
    let redecoded = decode_webp_image(&reencoded).expect("re-encoded file decodes");
    assert_eq!(redecoded.width, original.width);
    assert_eq!(redecoded.height, original.height);
    assert_eq!(
        redecoded.rgba, original.rgba,
        "encoder round trip changed pixels"
    );
}

#[test]
fn round115_encoder_round_trips_lossless_1x1() {
    assert_encoder_round_trips(LOSSLESS_1X1);
}

#[test]
fn round115_encoder_round_trips_lossless_32x32_rgba() {
    // 32x32 with a wide spread of colors — exercises multi-bit canonical
    // codes for all four channels through the real decoder.
    assert_encoder_round_trips(LOSSLESS_32X32_RGBA);
}

#[test]
fn round115_encoder_round_trips_color_indexing_fixture() {
    // The paletted fixture decodes to few distinct colors; re-encoding it as
    // a flat literal stream still has to round-trip exactly.
    assert_encoder_round_trips(LOSSLESS_COLOR_INDEXING);
}

#[test]
fn round115_encoded_file_is_a_well_formed_simple_lossless_container() {
    let original = decode_webp_image(LOSSLESS_32X32_RGBA).unwrap();
    let file = encode_webp_lossless(&original.rgba, original.width, original.height).unwrap();
    // RIFF/WEBP framing with a single VP8L chunk, no VP8X.
    let c = parse_container(&file).expect("encoded file walks");
    assert!(c
        .first_chunk_with_fourcc(oxideav_webp::container::fourcc::VP8L)
        .is_some());
    assert!(c
        .first_chunk_with_fourcc(oxideav_webp::container::fourcc::VP8X)
        .is_none());
    // The typed lossless chunk handle agrees on the dimensions.
    let chunk = extract_lossless_chunk(&file).unwrap().unwrap();
    assert_eq!(chunk.width(), original.width);
    assert_eq!(chunk.height(), original.height);
}

// ---------------------------------------------------------------------
// Round 322 — end-to-end decode coverage for the remaining
// bit-exactly-reconstructible fixtures in the docs corpus.
//
// `docs/image/webp/fixtures/` ships eighteen fixtures, each with a
// committed `expected.png` decoded ground truth. Earlier rounds mirror
// a subset into `tests/data/` and validate them end-to-end; these four
// close the gap for the lossless (and near-lossless) inputs whose full
// §4–§6 reconstruction the decoder reproduces bit-for-bit:
//
//   * `lossless-32x32-rgb`          — opaque RGB, no alpha plane
//   * `lossless-color-cache-stress` — §5.2.3 color-cache heavy
//   * `lossless-cross-color-active` — §4.2 CROSS_COLOR transform active
//   * `lossy-near-lossless-q40`     — `cwebp -near_lossless 40`, which is
//                                     a VP8L (lossless container) encode,
//                                     so it decodes exactly despite the
//                                     residual-quantizing pre-pass name.
//
// Each ground-truth value below is read from the fixture's
// `expected.png` (RGBA, 8-bit). The decoder's flat `[R, G, B, A]`
// buffer is asserted to match those samples at a spread of edge +
// interior coordinates. (The true-lossy `VP8 ` fixtures — q1/q75/q100 —
// are deliberately *not* pixel-validated here: RFC 6386 §2 leaves the
// 4:2:0 chroma up-sampling kernel to the decoder, and this crate's
// spec-baseline nearest-neighbour kernel reconstructs values a few
// units off the `expected.png` produced by a fancier up-sampler — a
// quality choice, not a conformance defect, so it is not asserted as an
// equality.)
// ---------------------------------------------------------------------

/// Read the RGBA byte at `(x, y)` from a row-major `width`-wide flat
/// `[R, G, B, A]` buffer.
fn rgba_at(buf: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let o = ((y * width + x) * 4) as usize;
    [buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]
}

/// Assert `decode_webp` on `file` yields `w*h` and the listed
/// `(x, y) -> [R, G, B, A]` ground-truth samples (from `expected.png`).
fn assert_decode_matches(file: &[u8], w: u32, h: u32, samples: &[((u32, u32), [u8; 4])]) {
    let img = decode_webp(file).expect("fixture decodes");
    assert_eq!(img.frames.len(), 1, "still image is a single frame");
    let f = &img.frames[0];
    assert_eq!((f.width, f.height), (w, h), "dimensions");
    assert_eq!(f.rgba.len(), (w * h * 4) as usize, "tight RGBA buffer");
    for &((x, y), want) in samples {
        assert_eq!(rgba_at(&f.rgba, w, x, y), want, "pixel ({x}, {y}) mismatch");
    }
}

#[test]
fn round322_lossless_32x32_rgb_decodes_to_expected_png() {
    // Opaque RGB lossless: alpha is synthesised 0xff for every pixel.
    assert_decode_matches(
        LOSSLESS_32X32_RGB,
        32,
        32,
        &[
            ((0, 0), [0, 0, 0, 255]),
            ((31, 0), [248, 0, 124, 255]),
            ((0, 31), [0, 248, 124, 255]),
            ((31, 31), [248, 248, 248, 255]),
            ((16, 16), [128, 128, 128, 255]),
            ((1, 2), [8, 16, 12, 255]),
            ((3, 5), [24, 40, 32, 255]),
            ((7, 11), [56, 88, 72, 255]),
            ((10, 10), [80, 80, 80, 255]),
            ((21, 21), [168, 168, 168, 255]),
            ((8, 24), [64, 192, 128, 255]),
            ((30, 30), [240, 240, 240, 255]),
            ((13, 17), [104, 136, 120, 255]),
            ((27, 3), [216, 24, 120, 255]),
        ],
    );
}

#[test]
fn round322_lossless_color_cache_stress_decodes_to_expected_png() {
    // §5.2.3 color-cache-heavy stream: many pixels resolve through the
    // 2^color_cache_code_bits hash cache rather than a literal/back-ref.
    assert_decode_matches(
        LOSSLESS_COLOR_CACHE_STRESS,
        64,
        64,
        &[
            ((0, 0), [57, 12, 140, 255]),
            ((63, 0), [172, 52, 47, 255]),
            ((0, 63), [116, 148, 40, 255]),
            ((63, 63), [111, 119, 13, 255]),
            ((32, 32), [119, 51, 194, 255]),
            ((1, 2), [40, 150, 185, 255]),
            ((3, 5), [98, 35, 23, 255]),
            ((7, 11), [119, 51, 194, 255]),
            ((21, 21), [142, 79, 110, 255]),
            ((42, 42), [119, 51, 194, 255]),
            ((16, 48), [142, 79, 110, 255]),
            ((62, 62), [119, 51, 194, 255]),
            ((13, 17), [111, 119, 13, 255]),
            ((59, 3), [235, 63, 193, 255]),
        ],
    );
}

#[test]
fn round322_lossless_cross_color_active_decodes_to_expected_png() {
    // §4.2 CROSS_COLOR (color) transform active: the inverse must undo
    // the green-into-red / green-and-red-into-blue de-correlation.
    assert_decode_matches(
        LOSSLESS_CROSS_COLOR_ACTIVE,
        64,
        64,
        &[
            ((0, 0), [0, 0, 255, 255]),
            ((63, 0), [252, 189, 129, 255]),
            ((0, 63), [0, 63, 255, 255]),
            ((63, 63), [252, 252, 129, 255]),
            ((32, 32), [128, 128, 191, 255]),
            ((1, 2), [4, 5, 253, 255]),
            ((3, 5), [12, 14, 249, 255]),
            ((7, 11), [28, 32, 241, 255]),
            ((21, 21), [84, 84, 213, 255]),
            ((42, 42), [168, 168, 171, 255]),
            ((16, 48), [64, 96, 223, 255]),
            ((62, 62), [248, 248, 131, 255]),
            ((13, 17), [52, 56, 229, 255]),
            ((59, 3), [236, 180, 137, 255]),
        ],
    );
}

#[test]
fn round322_lossy_near_lossless_q40_decodes_exactly() {
    // `cwebp -near_lossless 40` emits a VP8L (lossless container) stream
    // whose residual pre-pass is baked into the encoded pixels, so the
    // decode is bit-exact against `expected.png` — the same exactness
    // every plain lossless fixture enjoys.
    assert_decode_matches(
        LOSSY_NEAR_LOSSLESS_Q40,
        128,
        128,
        &[
            ((0, 0), [133, 233, 241, 255]),
            ((127, 0), [191, 96, 238, 255]),
            ((0, 127), [135, 81, 1, 255]),
            ((127, 127), [189, 224, 0, 255]),
            ((64, 64), [251, 251, 152, 255]),
            ((1, 2), [151, 232, 238, 255]),
            ((3, 5), [181, 232, 225, 255]),
            ((7, 11), [235, 229, 196, 255]),
            ((42, 42), [0, 255, 44, 255]),
            ((85, 85), [76, 224, 236, 255]),
            ((32, 96), [214, 55, 37, 255]),
            ((126, 126), [196, 223, 0, 255]),
            ((13, 17), [250, 235, 147, 255]),
            ((123, 3), [206, 73, 248, 255]),
        ],
    );
}

#[test]
fn round322_lossless_32x32_rgb_carries_synthetic_opaque_alpha() {
    // An RGB (no-ALPH, no in-stream alpha) lossless file fills every
    // alpha byte with 0xff. Confirm across the whole buffer, not just
    // the spot-checked pixels.
    let img = decode_webp(LOSSLESS_32X32_RGB).expect("rgb fixture decodes");
    let f = &img.frames[0];
    assert!(
        f.rgba.chunks_exact(4).all(|p| p[3] == 0xff),
        "every alpha byte is opaque for an RGB lossless file"
    );
}

// ---------------------------------------------------------------------
// Round 327 — close the last *lossless* animation end-to-end pixel gap.
//
// `animated-with-alpha` (the other ANIM fixture in the in-crate corpus)
// is already geometry- + digest-validated above, but `animated-3-frames-
// rgb` — the docs corpus's 3-frame *opaque-RGB* VP8L animation — had no
// in-crate copy and was never pixel-validated against its committed
// `expected_NNNN.png` ground truth. It is a fully reconstructible
// lossless animation: per the fixture's `trace.txt`, VP8X declares a
// 64×64 canvas with the animation flag set, ANIM carries
// bgcolor=0xffffffff / loop_count=0, and three full-canvas (64×64 at
// offset 0,0) ANMF frames each wrap a VP8L lossless image. Frame 1 is
// NO_BLEND (flags_byte=0x02), frames 2 and 3 are BLEND (flags_byte=0x00);
// because every source frame is fully opaque, blending a full-canvas
// opaque frame over the canvas yields that frame's own pixels, so each
// composited output equals its per-frame decode — which is what the
// committed `expected_NNNN.png` frames hold.
//
// The frames are visually distinct (frame 0 = red field, frame 1 = green
// field, frame 2 = blue field, each with a white interior feature), so
// the per-frame ground-truth samples below pin not just the §4–§6 VP8L
// reconstruction but also the §2.7.1.1 frame ordering and the
// per-frame compositing result. All ground-truth bytes are read from the
// fixture's own `expected_0000/0001/0002.png` oracle outputs.
// ---------------------------------------------------------------------

#[test]
fn round327_animated_3_frames_rgb_decodes_three_distinct_frames_to_expected_pngs() {
    let img = decode_webp(ANIMATED_3_FRAMES_RGB).expect("animated-3-frames-rgb decodes");
    assert_eq!(img.frames.len(), 3, "three ANMF frames");
    // ANIM globals: white-opaque background, infinite loop.
    assert_eq!(
        img.anim_background_rgba,
        Some([0xFF, 0xFF, 0xFF, 0xFF]),
        "ANIM bgcolor=0xffffffff → RGBA white"
    );
    assert_eq!(img.anim_loop_count, Some(0), "loop_count=0 (infinite)");

    // Per-frame ground-truth samples lifted from each frame's committed
    // `expected_NNNN.png`. The field colour differs per frame; the
    // interior (32,32)/(42,42) is white in every frame, and (21,21) is
    // white in frames 0/1 but the field colour in frame 2 — so the
    // sample set uniquely distinguishes all three frames and their order.
    let frame0: &[((u32, u32), [u8; 4])] = &[
        ((0, 0), [255, 60, 60, 255]),
        ((63, 0), [255, 60, 60, 255]),
        ((0, 63), [255, 60, 60, 255]),
        ((63, 63), [255, 60, 60, 255]),
        ((32, 32), [255, 255, 255, 255]),
        ((1, 2), [255, 60, 60, 255]),
        ((3, 5), [255, 60, 60, 255]),
        ((7, 11), [255, 60, 60, 255]),
        ((21, 21), [255, 255, 255, 255]),
        ((42, 42), [255, 255, 255, 255]),
        ((16, 48), [255, 60, 60, 255]),
        ((62, 62), [255, 60, 60, 255]),
        ((13, 17), [255, 60, 60, 255]),
        ((59, 3), [255, 60, 60, 255]),
    ];
    let frame1: &[((u32, u32), [u8; 4])] = &[
        ((0, 0), [60, 255, 60, 255]),
        ((63, 0), [60, 255, 60, 255]),
        ((0, 63), [60, 255, 60, 255]),
        ((63, 63), [60, 255, 60, 255]),
        ((32, 32), [255, 255, 255, 255]),
        ((1, 2), [60, 255, 60, 255]),
        ((3, 5), [60, 255, 60, 255]),
        ((7, 11), [60, 255, 60, 255]),
        ((21, 21), [255, 255, 255, 255]),
        ((42, 42), [255, 255, 255, 255]),
        ((16, 48), [60, 255, 60, 255]),
        ((62, 62), [60, 255, 60, 255]),
        ((13, 17), [60, 255, 60, 255]),
        ((59, 3), [60, 255, 60, 255]),
    ];
    let frame2: &[((u32, u32), [u8; 4])] = &[
        ((0, 0), [60, 60, 255, 255]),
        ((63, 0), [60, 60, 255, 255]),
        ((0, 63), [60, 60, 255, 255]),
        ((63, 63), [60, 60, 255, 255]),
        ((32, 32), [255, 255, 255, 255]),
        ((1, 2), [60, 60, 255, 255]),
        ((3, 5), [60, 60, 255, 255]),
        ((7, 11), [60, 60, 255, 255]),
        ((21, 21), [60, 60, 255, 255]),
        ((42, 42), [255, 255, 255, 255]),
        ((16, 48), [60, 60, 255, 255]),
        ((62, 62), [60, 60, 255, 255]),
        ((13, 17), [60, 60, 255, 255]),
        ((59, 3), [60, 60, 255, 255]),
    ];
    let frame_samples = [frame0, frame1, frame2];

    for (i, samples) in frame_samples.iter().enumerate() {
        let f = &img.frames[i];
        assert_eq!((f.width, f.height), (64, 64), "frame {i} dimensions");
        assert_eq!(f.duration_ms, 100, "frame {i} duration (ANMF native ms)");
        assert_eq!(f.rgba.len(), 64 * 64 * 4, "frame {i} tight RGBA buffer");
        for &((x, y), want) in samples.iter() {
            assert_eq!(
                rgba_at(&f.rgba, 64, x, y),
                want,
                "frame {i} pixel ({x}, {y}) mismatch"
            );
        }
        // Every pixel is fully opaque (RGB source, alpha synthesised 0xff).
        assert!(
            f.rgba.chunks_exact(4).all(|p| p[3] == 0xff),
            "frame {i} alpha is opaque everywhere"
        );
    }

    // The three frames are pairwise distinct (the animation moves).
    assert_ne!(img.frames[0].rgba, img.frames[1].rgba, "frames 0/1 differ");
    assert_ne!(img.frames[1].rgba, img.frames[2].rgba, "frames 1/2 differ");
    assert_ne!(img.frames[0].rgba, img.frames[2].rgba, "frames 0/2 differ");
}

#[test]
fn round327_animated_3_frames_rgb_whole_buffer_digest_is_pinned() {
    // Whole-buffer lock over the same `width_le ‖ height_le ‖
    // frame_count_le ‖ frames[..].rgba` layout the round-284 corpus pin
    // uses, derived from concatenating the three `expected_NNNN.png`
    // oracle frames. Catches any reconstruction drift the 14-sample
    // spot-check above would miss.
    let img = decode_webp(ANIMATED_3_FRAMES_RGB).expect("animated-3-frames-rgb decodes");
    let mut buf = Vec::new();
    buf.extend_from_slice(&img.width.to_le_bytes());
    buf.extend_from_slice(&img.height.to_le_bytes());
    buf.extend_from_slice(&(img.frames.len() as u32).to_le_bytes());
    for f in &img.frames {
        buf.extend_from_slice(&f.rgba);
    }
    assert_eq!(
        fnv1a64(&buf),
        0xd565_0ed4_7ba8_f556,
        "decoded-output digest"
    );
}
