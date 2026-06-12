//! Criterion bench — §2.7.1.1 animation full-timeline decode.
//!
//! Every prior decode bench drives a single-frame still image; the
//! animated path — `ANIM`/`ANMF` chunk walk, per-frame §2.6 `VP8L`
//! decode, and the §2.7.1.1 canvas compositor (blend / dispose /
//! sub-frame placement) — had no coverage at the public-entry-point
//! level. This bench assembles two 12-frame 128×128 animations once at
//! setup via [`oxideav_webp::anim_encode::build_animated_webp`] and
//! measures [`oxideav_webp::decode_webp`] decoding the whole timeline:
//!
//! * `anim_decode_keyframes_12x128` — every frame a full-canvas
//!   lossless keyframe (`AnimFrameMode::Lossless`): 12 full 128×128
//!   VP8L decodes + 12 full-canvas overwrite composites. The
//!   per-frame content is a moving 32×32 square over a gradient so
//!   each keyframe encodes (and decodes) genuinely distinct pixels.
//! * `anim_decode_delta_12x128` — same timeline content with
//!   `AnimFrameMode::Delta`: frame 1 is a full keyframe, frames 2..12
//!   carry only the dirty rectangle the moving square sweeps, so the
//!   per-frame VP8L decode is small and the §2.7.1.1 sub-frame
//!   placement + compositor dominate.
//!
//! Both fixtures are verified at setup to decode to 12 frames on a
//! 128×128 canvas with identical final-frame pixels, so the two cells
//! measure the same visual timeline through the two `ANMF` layouts.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench anim_decode -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_webp::anim_encode::{build_animated_webp, AnimFrame, AnimFrameMode};
use oxideav_webp::decode_webp;

const W: u32 = 128;
const H: u32 = 128;
const FRAMES: u32 = 12;

/// Frame `i`: a gradient background with a 32×32 solid square whose
/// position advances 8 px per frame (wrapping), so successive frames
/// differ in a bounded dirty rect.
fn frame_pixels(i: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((W * H * 4) as usize);
    let sq_x = (i * 8) % (W - 32);
    let sq_y = (i * 6) % (H - 32);
    for y in 0..H {
        for x in 0..W {
            if x >= sq_x && x < sq_x + 32 && y >= sq_y && y < sq_y + 32 {
                buf.extend_from_slice(&[0xff, 0x40, (i * 20) as u8, 0xff]);
            } else {
                buf.extend_from_slice(&[x as u8, y as u8, ((x + y) / 2) as u8, 0xff]);
            }
        }
    }
    buf
}

fn build_timeline(mode: AnimFrameMode) -> Vec<u8> {
    let frames: Vec<AnimFrame> = (0..FRAMES)
        .map(|i| {
            let mut f = AnimFrame::new(W, H, frame_pixels(i), 40);
            f.mode = mode;
            f
        })
        .collect();
    build_animated_webp(&frames).expect("assemble animation")
}

fn bench_anim_decode(c: &mut Criterion) {
    let keyframes = build_timeline(AnimFrameMode::Lossless);
    let delta = build_timeline(AnimFrameMode::Delta);

    // Setup sanity: both layouts decode to the same 12-frame timeline.
    let img_k = decode_webp(&keyframes).expect("decode keyframes");
    let img_d = decode_webp(&delta).expect("decode delta");
    assert_eq!(img_k.frames.len(), FRAMES as usize);
    assert_eq!(img_d.frames.len(), FRAMES as usize);
    assert_eq!((img_k.width, img_k.height), (W, H));
    assert_eq!(
        img_k.frames.last().unwrap().rgba,
        img_d.frames.last().unwrap().rgba,
        "delta layout must composite to the keyframe timeline"
    );
    assert!(
        delta.len() < keyframes.len(),
        "delta layout should be smaller than all-keyframes \
         ({} vs {} bytes)",
        delta.len(),
        keyframes.len()
    );

    c.bench_function("anim_decode_keyframes_12x128", |b| {
        b.iter(|| {
            let img = decode_webp(black_box(&keyframes)).expect("decode");
            black_box(img)
        })
    });
    c.bench_function("anim_decode_delta_12x128", |b| {
        b.iter(|| {
            let img = decode_webp(black_box(&delta)).expect("decode");
            black_box(img)
        })
    });
}

criterion_group!(benches, bench_anim_decode);
criterion_main!(benches);
