//! Criterion bench — full-file VP8L decode per elected §4 transform mix.
//!
//! The long-standing `lossless_decode` bench drives one fixture (a
//! 256×256 gradient) whose encoder elects the §4.1 predictor path, so
//! the end-to-end decode cost of the other three §4 inverse transforms
//! (§4.2 color, §4.3 subtract-green, §4.4 color-indexing) and of the
//! transform-free path was never visible at the public-entry-point
//! level — only through the per-pass `inverse_*` microbenches.
//!
//! This bench builds five 256×256 fixtures whose content steers the
//! encoder's chooser onto each §4 mix, asserts the elected transform
//! list at setup via [`oxideav_webp::read_vp8l_transform_list`] (so a
//! future chooser change that silently re-routes a cell fails loudly
//! instead of mislabeling the measurement), then measures the public
//! [`oxideav_webp::decode_webp`] entry point — RIFF walk + §6 entropy
//! decode + §4 inverse-transform chain + ARGB→RGBA repack.
//!
//! | Cell | Content | Elected transform list (asserted) |
//! |---|---|---|
//! | `predictor` | smooth gradient | §4.1 `Predictor` |
//! | `colorindex` | 4-color 8×8 blocks | §4.4 `ColorIndexing` |
//! | `crosscolor` | per-pixel random G, R≈G/2, B≈G/3+R/4 | §4.2 `Color` |
//! | `subgreen` | per-pixel random G, R≈G, B≈G | §4.3 `SubtractGreen` |
//! | `none` | uniform random noise | (empty) |
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench lossless_decode_mixes -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::vp8l_stream::TransformType;
use oxideav_webp::{decode_webp, encode_webp_lossless, read_vp8l_transform_list};

const W: u32 = 256;
const H: u32 = 256;

/// Deterministic LCG (same constants as the per-pass §4.x benches).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

/// Smooth gradient — the §4.1 predictor wins.
fn fixture_predictor() -> Vec<u8> {
    let mut buf = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            buf.extend_from_slice(&[x as u8, y as u8, ((x + y) / 2) as u8, 0xff]);
        }
    }
    buf
}

/// 4-color 8×8 block tiling — the §4.4 color-indexing transform wins
/// (palette of 4 ⇒ `width_bits = 2`, four indices per packed byte).
fn fixture_colorindex() -> Vec<u8> {
    const PALETTE: [[u8; 4]; 4] = [
        [0xe0, 0x30, 0x30, 0xff],
        [0x30, 0xe0, 0x30, 0xff],
        [0x30, 0x30, 0xe0, 0xff],
        [0xf0, 0xf0, 0xf0, 0xff],
    ];
    let mut buf = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            buf.extend_from_slice(&PALETTE[(((x / 8) + (y / 8)) % 4) as usize]);
        }
    }
    buf
}

/// Per-pixel random green with R ≈ G/2 and B ≈ G/3 + R/4 (plus ±7
/// noise) — spatially incompressible, but the channel correlation makes
/// the §4.2 cross-color transform win.
fn fixture_crosscolor() -> Vec<u8> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut buf = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..(W * H) {
        let n = lcg(&mut s);
        let g = (n & 0xff) as u8;
        let r = (g / 2).wrapping_add(((n >> 8) & 7) as u8);
        let b = (g / 3)
            .wrapping_add(r / 4)
            .wrapping_add(((n >> 11) & 7) as u8);
        buf.extend_from_slice(&[r, g, b, 0xff]);
    }
    buf
}

/// Per-pixel random green with R ≈ G and B ≈ G (plus ±3 noise) — the
/// §4.3 subtract-green transform wins (no entropy body, list read to
/// its terminating presence bit).
fn fixture_subgreen() -> Vec<u8> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut buf = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..(W * H) {
        let n = lcg(&mut s);
        let g = (n & 0xff) as u8;
        let r = g.wrapping_add(((n >> 8) & 3) as u8);
        let b = g.wrapping_add(((n >> 10) & 3) as u8);
        buf.extend_from_slice(&[r, g, b, 0xff]);
    }
    buf
}

/// Uniform random noise — no transform pays for itself; the elected
/// list is empty (ARGB literals straight through).
fn fixture_none() -> Vec<u8> {
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut buf = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..(W * H) {
        let n = lcg(&mut s);
        buf.extend_from_slice(&[
            (n & 0xff) as u8,
            ((n >> 8) & 0xff) as u8,
            ((n >> 16) & 0xff) as u8,
            0xff,
        ]);
    }
    buf
}

/// Encode `rgba` and assert the elected leading transform matches
/// `expect` (`None` = empty transform list). `read_vp8l_transform_list`
/// stops at the first §5-encoded transform body, so the leading entry
/// is exactly what it reports for every body-bearing mix; the
/// subtract-green and empty lists are read to the terminating bit.
fn encode_expecting(rgba: &[u8], expect: Option<TransformType>) -> Vec<u8> {
    let webp = encode_webp_lossless(rgba, W, H).expect("encode");
    let list = read_vp8l_transform_list(&webp)
        .expect("transform list")
        .expect("VP8L chunk");
    let got = list.transforms().first().map(|t| t.transform_type());
    assert_eq!(
        got, expect,
        "fixture no longer elects the documented transform mix; \
         relabel the bench cell"
    );
    webp
}

fn bench_lossless_decode_mixes(c: &mut Criterion) {
    let cells: [(&str, Vec<u8>, Option<TransformType>); 5] = [
        (
            "predictor",
            fixture_predictor(),
            Some(TransformType::Predictor),
        ),
        (
            "colorindex",
            fixture_colorindex(),
            Some(TransformType::ColorIndexing),
        ),
        (
            "crosscolor",
            fixture_crosscolor(),
            Some(TransformType::Color),
        ),
        (
            "subgreen",
            fixture_subgreen(),
            Some(TransformType::SubtractGreen),
        ),
        ("none", fixture_none(), None),
    ];
    for (name, rgba, expect) in cells {
        let webp = encode_expecting(&rgba, expect);
        c.bench_function(
            format!("lossless_decode_mix_{name}_256x256").as_str(),
            |b| {
                b.iter(|| {
                    let img = decode_webp(black_box(&webp)).expect("decode");
                    black_box(img)
                })
            },
        );
    }
}

criterion_group!(benches, bench_lossless_decode_mixes);
criterion_main!(benches);
