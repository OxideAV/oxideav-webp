//! Helpers shared by the lossless-encoder integration tests.
//!
//! Each integration test file is its own crate and uses a different subset
//! of these helpers, so unused-item warnings are silenced per item.

/// A deterministic photo-like RGBA image: smooth gradients, a soft bright
/// blob and small per-channel noise. It compresses like a photograph (no
/// long exact repeats, strong spatial correlation) rather than like flat
/// synthetic art, which makes it the reference input for the lossless
/// encoder's speed and size checks. Fully opaque.
#[allow(dead_code)]
pub fn photo_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut px = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            // xorshift64 noise source.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let d = ((fx - 0.5).powi(2) + (fy - 0.4).powi(2)).sqrt();
            let blob = (1.0 - (d * 3.0).min(1.0)) * 60.0;
            let nr = (state & 0xF) as f32 - 8.0;
            let ng = ((state >> 4) & 0xF) as f32 - 8.0;
            let nb = ((state >> 8) & 0xF) as f32 - 8.0;
            let r = (40.0 + 180.0 * fx + blob + nr).clamp(0.0, 255.0) as u8;
            let g = (60.0 + 150.0 * fy + blob + ng).clamp(0.0, 255.0) as u8;
            let b = (200.0 - 120.0 * fx * fy + nb).clamp(0.0, 255.0) as u8;
            px.extend_from_slice(&[r, g, b, 255]);
        }
    }
    px
}

/// 64-bit FNV-1a over `bytes`: a dependency-free fingerprint for pinning
/// encoder output byte-for-byte.
#[allow(dead_code)]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The committed 128×128 natural-image fixture, decoded to packed RGBA.
#[allow(dead_code)]
pub fn natural_fixture_rgba() -> (u32, u32, Vec<u8>) {
    let bytes = include_bytes!("../data/lossless-128x128-natural.webp");
    let img = oxideav_webp::decode_rgba8(bytes).expect("natural fixture decodes");
    (img.width, img.height, img.data)
}
