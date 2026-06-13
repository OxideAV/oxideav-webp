//! Criterion bench — the §2.7.1.2 `ALPH` (alpha-plane) decode path,
//! the **rank-1 webp-owned cost** on the lossy decode path.
//!
//! The round-289 `lossy_decode` hotspot map decomposed the public
//! `decode_webp` cost of a `VP8 `+`ALPH` still and attributed ≈52% of
//! the lossy end-to-end time to "container walk + `ALPH` plane layering"
//! — but that figure was obtained purely by *subtraction*
//! (`decode_webp_lossy_e2e` − `decode_lossy_rgba_extracted`), so the
//! `ALPH` decode itself had **no isolated harness**. This bench fills
//! that gap: it times [`oxideav_webp::alph::decode_alpha`] (and the
//! public [`oxideav_webp::decode_alpha_plane`] wrapper) directly, so the
//! rank-1 lossy webp-owned stage gets a measured number instead of an
//! estimate, and the §2.7.1.2 inverse-filter loop the crate can act on
//! gets a per-method A/B target.
//!
//! No behavior is measured beyond the public/`pub` API, and no decoded
//! bytes change — this is a measurement-only round.
//!
//! ## What the `ALPH` decode owns (§2.7.1.2)
//!
//! `decode_alpha` runs two stages:
//!
//! 1. **De-compression** (`C` field). Method 0 (`None`) copies the raw
//!    bytes; method 1 (`Lossless`) decodes a headerless §3 VP8L image
//!    stream and lifts the alpha out of each pixel's green channel. The
//!    `Lossless` path delegates the heavy lifting to the VP8L lossless
//!    decoder (already benched by `lossless_decode*` / `read_symbol`);
//!    the `None` path is a memcpy.
//! 2. **Inverse filtering** (`F` field). A per-pixel predictor
//!    (none / horizontal / vertical / gradient `clip(A+B−C)`) is added
//!    mod-256 to the de-compressed value over the *reconstructed* plane,
//!    with the §2.7.1.2 left-most / top-most edge cases. This is the
//!    fully crate-owned per-pixel loop — the closest analogue to the
//!    `inverse_predictor` lossless hot loop, and the cleanest A/B target
//!    for a future hoist of the per-pixel `(x, y)` match out of the loop.
//!
//! ## Cells
//!
//! * **`decode_alpha_plane_e2e`** — the public
//!   [`oxideav_webp::decode_alpha_plane`] over the committed 128×128
//!   `VP8 `+`ALPH` fixture: §2.3 RIFF walk + dimension resolution + the
//!   real `Lossless`/`None`-filter `ALPH` chunk the fixture carries. What
//!   a caller pays to recover the alpha plane of that still.
//! * **`decode_alpha_lossless_extracted`** — [`oxideav_webp::alph::decode_alpha`]
//!   on the fixture's *extracted* `ALPH` payload (RIFF walk removed),
//!   isolating the rank-1 webp-owned alpha stage (headerless VP8L decode,
//!   green-channel lift, inverse filter) from the container layer.
//!   Subtracting this from the e2e cell sizes the RIFF/dimension
//!   overhead.
//! * **`inverse_filter_{none,horizontal,vertical,gradient}_128x128`** —
//!   the §2.7.1.2 Stage-2 inverse-filter loop in isolation, driven through
//!   `decode_alpha` with a synthetic **uncompressed** (`C = None`) ALPH
//!   payload so the de-compression stage is a memcpy and the measured cost
//!   is the per-pixel predictor walk for one filter method. The four cells
//!   vary only the `F` field, so a future per-method hoist has direct A/B
//!   numbers (gradient — the `clip(A+B−C)` branch — is the deepest).
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench alpha_decode -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::alph::{decode_alpha, AlphFiltering};
use oxideav_webp::container::{self, fourcc};

/// The committed 128×128 extended-lossy fixture (`VP8 ` + `ALPH`). The
/// fixture's `ALPH` chunk is `Lossless` compression with `None`
/// filtering, decoding to a 16 384-byte plane. Same bytes the
/// `fixture_walks` / `vp8_lossy_roundtrip` integration tests decode;
/// included here so the bench needs no external file.
const LOSSY_128: &[u8] = include_bytes!("../tests/data/lossy-with-alpha-128x128.webp");

/// The fixture's plane dimensions (the `VP8 ` keyframe / `VP8X` canvas).
const W: u32 = 128;
const H: u32 = 128;

/// Extract the raw `ALPH` chunk payload from a full extended-lossy WebP
/// file, so the bench can drive [`decode_alpha`] without the RIFF walk
/// inside the measured interval.
fn extract_alph_payload(file: &[u8]) -> Vec<u8> {
    let cont = container::parse(file).expect("fixture is a well-formed RIFF container");
    let chunk = cont
        .first_chunk_with_fourcc(fourcc::ALPH)
        .expect("fixture carries an ALPH chunk");
    chunk.payload(file).to_vec()
}

/// Build a synthetic **uncompressed** (`C = None`) ALPH chunk payload of
/// `w × h` raw alpha bytes for the given §2.7.1.2 filter method.
///
/// The §2.7.1.2 info byte packs `Rsv|P|F|C` MSB-first; here `C = 0`
/// (None / raw) and `F` selects the filter method, so the payload is the
/// single info byte followed by exactly `w * h` raw plane bytes (the
/// `None`-compression path's length invariant). The plane is a
/// deterministic gradient so the inverse-filter predictor sees a spread
/// of neighbour values (not a constant the optimiser could fold), and the
/// content is identical across the four method cells so only the `F`
/// dispatch differs between them.
fn synthetic_uncompressed_alph(w: usize, h: usize, filtering: AlphFiltering) -> Vec<u8> {
    let f_bits: u8 = match filtering {
        AlphFiltering::None => 0,
        AlphFiltering::Horizontal => 1,
        AlphFiltering::Vertical => 2,
        AlphFiltering::Gradient => 3,
    };
    // C = 0 (None), P = 0, Rsv = 0 — only the F field is set.
    let info = f_bits << 2;
    let mut payload = Vec::with_capacity(1 + w * h);
    payload.push(info);
    for row in 0..h {
        for col in 0..w {
            payload.push(((row * 7 + col * 13) & 0xff) as u8);
        }
    }
    payload
}

fn bench_decode_alpha_plane_e2e(c: &mut Criterion) {
    // Sanity: the fixture's alpha plane decodes before we time it.
    let probe = oxideav_webp::decode_alpha_plane(LOSSY_128)
        .expect("fixture decodes")
        .expect("fixture carries an ALPH plane");
    assert_eq!(probe.len(), (W * H) as usize);
    c.bench_function("decode_alpha_plane_e2e", |b| {
        b.iter(|| {
            let plane = oxideav_webp::decode_alpha_plane(black_box(LOSSY_128))
                .expect("decode")
                .expect("alpha present");
            black_box(plane)
        })
    });
}

fn bench_decode_alpha_lossless_extracted(c: &mut Criterion) {
    let payload = extract_alph_payload(LOSSY_128);
    let probe = decode_alpha(&payload, W, H).expect("extracted ALPH payload decodes");
    assert_eq!(probe.len(), (W * H) as usize);
    c.bench_function("decode_alpha_lossless_extracted", |b| {
        b.iter(|| {
            let plane = decode_alpha(black_box(&payload), W, H).expect("decode");
            black_box(plane)
        })
    });
}

fn bench_inverse_filter(c: &mut Criterion) {
    let (w, h) = (W as usize, H as usize);
    for filtering in [
        AlphFiltering::None,
        AlphFiltering::Horizontal,
        AlphFiltering::Vertical,
        AlphFiltering::Gradient,
    ] {
        let payload = synthetic_uncompressed_alph(w, h, filtering);
        // Sanity: the synthetic uncompressed payload decodes to the plane.
        let probe = decode_alpha(&payload, W, H).expect("synthetic ALPH decodes");
        assert_eq!(probe.len(), w * h);
        let label = match filtering {
            AlphFiltering::None => "none",
            AlphFiltering::Horizontal => "horizontal",
            AlphFiltering::Vertical => "vertical",
            AlphFiltering::Gradient => "gradient",
        };
        let name = format!("inverse_filter_{}_{}x{}", label, W, H);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let plane = decode_alpha(black_box(&payload), W, H).expect("decode");
                black_box(plane)
            })
        });
    }
}

criterion_group!(
    benches,
    bench_decode_alpha_plane_e2e,
    bench_decode_alpha_lossless_extracted,
    bench_inverse_filter,
);
criterion_main!(benches);
