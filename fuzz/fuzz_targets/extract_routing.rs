#![no_main]

//! Consistency oracle on the public *chunk-routing* façade — the layer
//! between the §2.3 / §2.4 container walk and the decoders proper:
//! `extract_lossless_chunk`, `extract_lossy_chunk` and
//! `read_vp8l_transform_list`, all driven over the same
//! attacker-controlled byte buffer and cross-checked against each other
//! and against `container::parse`.
//!
//! ## Why this harness
//!
//! The `parse_container` sibling batters the §2.3 / §2.4 walker itself,
//! `parse_vp8_chunk` the bare §9.1 key-frame header parser and
//! `parse_transform_list` the §4 transform-list reader over a raw
//! zero-positioned bit stream — but the *routing* wrappers that stitch
//! those layers together through the public crate surface (walk the
//! container, pick the first `VP8L` / `VP8 ` chunk, peek its header,
//! hand out a borrowed bitstream slice) had no harness of their own.
//! Their contract is pure plumbing, which is exactly what makes a
//! divergence silent: a routing wrapper that picks a different chunk,
//! trims the payload slice differently, or maps an error differently
//! from the layer it wraps would corrupt every downstream consumer
//! while every layer-local harness stays green.
//!
//! ## The consistency contract
//!
//! With `c = container::parse(data)`:
//!
//! * `Err` ⇒ every routing wrapper is also `Err` (they all walk the
//!   container first and can only widen, never swallow, a container
//!   refusal).
//! * `Ok(c)` with no `VP8L` chunk ⇒ `extract_lossless_chunk` and
//!   `read_vp8l_transform_list` are `Ok(None)`.
//! * `Ok(c)` with a first `VP8L` chunk ⇒ `extract_lossless_chunk` is
//!   `Ok(Some(handle))` or `Err`, and on `Some` the handle is
//!   re-derived from the §3.4 / §7.1 wire bytes by this harness:
//!   `bitstream()` byte-identical to the chunk's payload slice,
//!   `payload[0] == 0x2F`, and `width` / `height` / `alpha_is_used` /
//!   `version` equal to the LSB-first unpack of the 32 bits packed in
//!   `payload[1..5]` (14 + 14 + 1 + 3), with the resolved dimensions
//!   inside the §3.4 `1..=16384` window.
//! * `Ok(c)` with no `VP8 ` chunk ⇒ `extract_lossy_chunk` is
//!   `Ok(None)`; with a first `VP8 ` chunk, `extract_lossy_chunk` must
//!   agree Ok/Err with `WebpLossyChunk::from_payload` over that chunk's
//!   payload slice, and on `Ok` carry field-identical width / height /
//!   scales / version / show_frame / first-partition-size and a
//!   byte-identical `bitstream()` echo (the field-level wire contract
//!   itself is the `parse_vp8_chunk` sibling's job).
//! * `read_vp8l_transform_list` must equal a manual
//!   `TransformList::read` over a `new_after_image_header` reader on
//!   the extracted lossless handle's bitstream — same Ok/Err split and
//!   an `==`-identical list on Ok — and be `Err` whenever
//!   `extract_lossless_chunk` is `Err` (never `Ok` on a file whose
//!   `VP8L` header the routing layer refused).
//!
//! A crash is a panic / debug-overflow / OOB anywhere in the walk or
//! the header peeks; an assertion failure is a real routing divergence.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{
    container, extract_lossless_chunk, extract_lossy_chunk, read_vp8l_transform_list, vp8_chunk,
    vp8l_stream,
};

fuzz_target!(|data: &[u8]| {
    let lossless = extract_lossless_chunk(data);
    let lossy = extract_lossy_chunk(data);
    let transforms = read_vp8l_transform_list(data);

    let Ok(c) = container::parse(data) else {
        // Container refusal: every routing wrapper walks the container
        // first, so none may succeed.
        assert!(
            lossless.is_err(),
            "extract_lossless_chunk must refuse a file the container walker refused",
        );
        assert!(
            lossy.is_err(),
            "extract_lossy_chunk must refuse a file the container walker refused",
        );
        assert!(
            transforms.is_err(),
            "read_vp8l_transform_list must refuse a file the container walker refused",
        );
        return;
    };

    // ── §2.6 VP8L routing ────────────────────────────────────────────
    let first_vp8l = c.first_chunk_with_fourcc(container::fourcc::VP8L);
    match (&lossless, first_vp8l) {
        (Ok(None), None) => {}
        (Ok(None), Some(_)) => {
            panic!("extract_lossless_chunk returned None with a VP8L chunk present");
        }
        (Ok(Some(_)) | Err(_), None) => {
            panic!("extract_lossless_chunk must be Ok(None) with no VP8L chunk present");
        }
        (Err(_), Some(_)) => {
            // A present-but-malformed §3.4 header: the transform-list
            // routing below must refuse too (checked after the match).
        }
        (Ok(Some(handle)), Some(chunk)) => {
            let payload = chunk.payload(data);
            assert_eq!(
                handle.bitstream(),
                payload,
                "the lossless handle's bitstream must echo the first VP8L chunk payload",
            );
            // Independent §3.4 / §7.1 re-derivation from the wire bytes.
            assert!(
                payload.len() >= 5 && payload[0] == 0x2F,
                "an accepted VP8L header must be >= 5 bytes behind the 0x2F signature",
            );
            let packed = u32::from(payload[1])
                | (u32::from(payload[2]) << 8)
                | (u32::from(payload[3]) << 16)
                | (u32::from(payload[4]) << 24);
            assert_eq!(
                handle.width(),
                (packed & 0x3FFF) + 1,
                "§3.4 width must be the 14-bit width-minus-one field plus one",
            );
            assert_eq!(
                handle.height(),
                ((packed >> 14) & 0x3FFF) + 1,
                "§3.4 height must be the 14-bit height-minus-one field plus one",
            );
            assert_eq!(
                handle.alpha_is_used(),
                (packed >> 28) & 1 == 1,
                "§3.4 alpha_is_used must be bit 28 of the packed header",
            );
            assert_eq!(
                u32::from(handle.version()),
                (packed >> 29) & 0x7,
                "§3.4 version must be bits 29..31 of the packed header",
            );
            assert!(
                (1..=16384).contains(&handle.width()) && (1..=16384).contains(&handle.height()),
                "§3.4 resolved dimensions must sit in [1, 16384]",
            );
        }
    }

    // ── §2.5 VP8 routing ─────────────────────────────────────────────
    let first_vp8 = c.first_chunk_with_fourcc(container::fourcc::VP8);
    match (&lossy, first_vp8) {
        (Ok(None), None) => {}
        (Ok(None), Some(_)) => {
            panic!("extract_lossy_chunk returned None with a VP8 chunk present");
        }
        (Ok(Some(_)) | Err(_), None) => {
            panic!("extract_lossy_chunk must be Ok(None) with no VP8 chunk present");
        }
        (routed, Some(chunk)) => {
            // Differential: routing through the container must agree
            // with the standalone §9.1 payload peek, field for field.
            let direct = vp8_chunk::WebpLossyChunk::from_payload(chunk.payload(data));
            match (routed, direct) {
                (Ok(Some(a)), Ok(b)) => {
                    assert_eq!(
                        a.bitstream(),
                        b.bitstream(),
                        "routed and standalone VP8 handles must borrow identical bitstreams",
                    );
                    assert_eq!(
                        (
                            a.width(),
                            a.height(),
                            a.horizontal_scale(),
                            a.vertical_scale()
                        ),
                        (
                            b.width(),
                            b.height(),
                            b.horizontal_scale(),
                            b.vertical_scale()
                        ),
                        "routed and standalone VP8 handles must agree on the §9.1 dimensions",
                    );
                    assert_eq!(
                        (a.version(), a.show_frame(), a.first_partition_size()),
                        (b.version(), b.show_frame(), b.first_partition_size()),
                        "routed and standalone VP8 handles must agree on the §9.1 frame tag",
                    );
                }
                (Err(_), Err(_)) => {}
                (a, b) => panic!(
                    "extract_lossy_chunk and WebpLossyChunk::from_payload diverged \
                     on the same chunk payload: routed {a:?} vs standalone {b:?}",
                ),
            }
        }
    }

    // ── §4 transform-list routing ────────────────────────────────────
    match (&transforms, &lossless) {
        (Ok(None), Ok(None)) => {}
        (_, Ok(None)) | (Ok(None), _) => {
            panic!("read_vp8l_transform_list must be Ok(None) exactly when there is no VP8L chunk");
        }
        (Err(_), Err(_)) => {}
        (Ok(Some(_)), Err(_)) => {
            panic!("read_vp8l_transform_list must refuse a file whose VP8L header routing refused");
        }
        (routed, Ok(Some(handle))) => {
            // Differential: the routed read must equal a manual §4 read
            // positioned past the handle's §3.4 image header.
            let mut reader = vp8l_stream::BitReader::new_after_image_header(handle.bitstream());
            match (routed, vp8l_stream::TransformList::read(&mut reader)) {
                (Ok(Some(a)), Ok(b)) => assert_eq!(
                    *a, b,
                    "routed and manual §4 transform-list reads must be identical",
                ),
                (Err(_), Err(_)) => {}
                (a, b) => panic!(
                    "read_vp8l_transform_list and the manual §4 read diverged: \
                     routed {a:?} vs manual {b:?}",
                ),
            }
        }
    }
});
