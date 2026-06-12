//! Criterion bench — §2.7 metadata-chunk walk (`extract_metadata`).
//!
//! [`oxideav_webp::extract_metadata`] is the published demux surface:
//! it parses the full RIFF/WEBP chunk list, then lifts the §2.7.1.4
//! `ICCP` and §2.7.1.5 `EXIF` / `XMP ` payloads (copying each into an
//! owned `Vec`). Its cost has two parts — the container chunk walk
//! (scales with chunk count) and the payload copies (scale with
//! metadata size) — and neither was benched. Three cells split them:
//!
//! * `metadata_walk_simple_nometa` — a simple-layout `VP8L` still
//!   image with no metadata: 1-chunk walk, three misses, no copies.
//!   This is the pure walk + miss floor.
//! * `metadata_walk_vp8x_full` — an extended-layout still image
//!   carrying ICC (3 KiB) + Exif (1 KiB) + XMP (2 KiB) built by
//!   [`oxideav_webp::encode_vp8l_argb_with_metadata`]: 5-chunk walk
//!   plus the three payload copies.
//! * `metadata_walk_anim64_full` — a 64-frame animation carrying the
//!   same three payloads, built by
//!   [`oxideav_webp::anim_encode::build_animated_webp_with_options`]:
//!   the `EXIF` / `XMP ` chunks sit after 64 `ANMF` chunks per the
//!   §2.7 chunk order, so the walk crosses ~68 chunks before the
//!   metadata is found. The spread between this cell and
//!   `vp8x_full` is the per-chunk walk cost.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench metadata_walk -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::anim_encode::{build_animated_webp_with_options, AnimEncoderOptions, AnimFrame};
use oxideav_webp::{
    encode_vp8l_argb_with_metadata, encode_webp_lossless, extract_metadata, WebpMetadata,
};

/// Deterministic LCG payload bytes (same constants as the other
/// benches) so the copies aren't of trivially-compressible zeros.
fn payload(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// A 64×64 opaque gradient as flat RGBA (still-image cells).
fn gradient_rgba_64() -> Vec<u8> {
    let mut buf = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64u32 {
        for x in 0..64u32 {
            buf.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, ((x ^ y) * 4) as u8, 0xff]);
        }
    }
    buf
}

/// The same gradient as packed ARGB (the metadata-embedding encoder
/// entry point takes ARGB words).
fn gradient_argb_64() -> Vec<u32> {
    gradient_rgba_64()
        .chunks_exact(4)
        .map(|px| {
            (u32::from(px[3]) << 24)
                | (u32::from(px[0]) << 16)
                | (u32::from(px[1]) << 8)
                | u32::from(px[2])
        })
        .collect()
}

fn bench_metadata_walk(c: &mut Criterion) {
    let icc = payload(3 * 1024, 0x11);
    let exif = payload(1024, 0x22);
    let xmp = payload(2 * 1024, 0x33);
    let meta = WebpMetadata {
        icc: Some(&icc),
        exif: Some(&exif),
        xmp: Some(&xmp),
    };

    // Cell 1: simple layout, no metadata.
    let simple = encode_webp_lossless(&gradient_rgba_64(), 64, 64).expect("encode simple");

    // Cell 2: VP8X still image with all three metadata chunks.
    let vp8x_full = encode_vp8l_argb_with_metadata(64, 64, &gradient_argb_64(), false, &meta)
        .expect("encode vp8x");

    // Cell 3: 64-frame animation with the same metadata — EXIF / XMP
    // sit after the ANMF run per the §2.7 chunk order.
    let frames: Vec<AnimFrame> = (0..64u32)
        .map(|i| {
            let px: Vec<u8> = (0..16 * 16)
                .flat_map(|p: u32| [(p + i) as u8, (p ^ i) as u8, i as u8, 0xff])
                .collect();
            AnimFrame::new(16, 16, px, 30)
        })
        .collect();
    let opts = AnimEncoderOptions {
        metadata: meta,
        ..AnimEncoderOptions::default()
    };
    let anim64_full = build_animated_webp_with_options(&frames, &opts).expect("encode anim");

    // Setup sanity: each cell yields exactly the payloads it embeds.
    let got = extract_metadata(&simple).expect("simple meta");
    assert!(got.icc.is_none() && got.exif.is_none() && got.xmp.is_none());
    for bytes in [&vp8x_full, &anim64_full] {
        let got = extract_metadata(bytes).expect("meta");
        assert_eq!(got.icc.as_deref(), Some(icc.as_slice()));
        assert_eq!(got.exif.as_deref(), Some(exif.as_slice()));
        assert_eq!(got.xmp.as_deref(), Some(xmp.as_slice()));
    }

    for (name, bytes) in [
        ("metadata_walk_simple_nometa", &simple),
        ("metadata_walk_vp8x_full", &vp8x_full),
        ("metadata_walk_anim64_full", &anim64_full),
    ] {
        c.bench_function(name, |b| {
            b.iter(|| {
                let meta = extract_metadata(black_box(bytes)).expect("extract");
                black_box(meta)
            })
        });
    }
}

criterion_group!(benches, bench_metadata_walk);
criterion_main!(benches);
