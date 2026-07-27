//! Shared support code for the fuzz harnesses in `fuzz_targets/`.
//!
//! ## The declared-pixel budget (round 432)
//!
//! The §3.4 `VP8L` 14-bit dimension fields can declare up to
//! 16384 × 16384 pixels, and a §5.2.2 backward-reference stream expands
//! a ~40-byte chunk into that many decoded pixels *legally* — the
//! round-432 campaign minimised a 38-byte file that decodes to ~10^8
//! pixels in ~16 s and ~2.4 GiB peak RSS under the address-sanitized
//! fuzz build. That is a decompression-ratio property of the format,
//! not a decoder defect: the library rejects per-side dimensions above
//! its `MAX_DECODE_DIMENSION` ceiling before allocating (round 286),
//! and a 16384 × 16384 still is a spec-valid file a general-purpose
//! decoder must accept. A fuzz iteration, however, must stay well under
//! libFuzzer's RSS limit and its per-unit time budget, so the harnesses
//! that drive whole-file decode façades gate their decode tail on the
//! *declared* pixel load the container makes observable up front.

use oxideav_webp::{anmf, container, vp8_chunk, vp8l_chunk, vp8x};

/// Declared-pixel budget for one whole-file-decode fuzz iteration,
/// summed across every §2.6 / §2.5 / §2.7.1 / §2.7.1.1 dimension
/// declaration in the file. 1 << 22 (~4.2 M pixels) caps a single
/// decoded RGBA surface at ~16.8 MiB — with the up-to-three live
/// surfaces the differential harnesses hold, the worst-case iteration
/// stays two orders of magnitude under libFuzzer's 2 GiB RSS limit,
/// while every committed fixture (all well under a megapixel,
/// animations included) still runs its full oracle.
pub const MAX_DECLARED_PIXELS: u64 = 1 << 22;

/// Sum every pixel allocation a §2.3 container declares up front, so a
/// harness can gate its decode tail before any header-sized buffer
/// exists:
///
/// * the top-level §2.6 `VP8L` still dimensions (§3.4 header),
/// * the top-level §2.5 `VP8 ` still dimensions (§9.1 keyframe header),
/// * the §2.7.1 `VP8X` canvas multiplied by one-plus-the-`ANMF`-count
///   (the §2.7.1.1 compositor allocates one canvas plus one full-canvas
///   snapshot per frame),
/// * each `ANMF` frame's own §2.6 / §2.5 sub-bitstream dimensions (each
///   frame decodes at the *bitstream's* declared size before the
///   canvas-fit check runs), walked with the decoder's strict §2.3
///   padded sub-chunk traversal (stop on any truncated declaration).
///
/// Sources that fail their cheap structural parse contribute nothing:
/// the real decoder refuses those before allocating pixels.
pub fn declared_pixel_load(bytes: &[u8], c: &container::WebpContainer) -> u64 {
    let mut load: u64 = 0;

    if let Some(chunk) = c.first_chunk_with_fourcc(container::fourcc::VP8L) {
        if let Ok(h) = vp8l_chunk::WebpLosslessChunk::from_payload(chunk.payload(bytes)) {
            load = load.saturating_add(u64::from(h.width()) * u64::from(h.height()));
        }
    }

    if let Some(chunk) = c.first_chunk_with_fourcc(container::fourcc::VP8) {
        if let Ok(h) = vp8_chunk::WebpLossyChunk::from_payload(chunk.payload(bytes)) {
            load = load.saturating_add(u64::from(h.width()) * u64::from(h.height()));
        }
    }

    if let Some(chunk) = c.first_chunk_with_fourcc(container::fourcc::VP8X) {
        if let Ok(h) = vp8x::Vp8xHeader::parse(chunk.payload(bytes)) {
            let canvas = u64::from(h.canvas_width) * u64::from(h.canvas_height);
            let frames = c.chunks_with_fourcc(container::fourcc::ANMF).count() as u64;
            load = load.saturating_add(canvas.saturating_mul(frames + 1));
        }
    }

    for anmf_chunk in c.chunks_with_fourcc(container::fourcc::ANMF) {
        let payload = anmf_chunk.payload(bytes);
        let Ok(header) = anmf::AnmfHeader::parse(payload) else {
            continue;
        };
        let mut fd = &payload[header.frame_data_offset()..];
        while fd.len() >= 8 {
            let fourcc: [u8; 4] = fd[0..4].try_into().expect("4-byte slice");
            let size = u32::from_le_bytes(fd[4..8].try_into().expect("4-byte slice")) as usize;
            let Some(sub) = fd.get(8..8usize.saturating_add(size)) else {
                break;
            };
            if fourcc == container::fourcc::VP8L {
                if let Ok(h) = vp8l_chunk::WebpLosslessChunk::from_payload(sub) {
                    load = load.saturating_add(u64::from(h.width()) * u64::from(h.height()));
                }
            } else if fourcc == container::fourcc::VP8 {
                if let Ok(h) = vp8_chunk::WebpLossyChunk::from_payload(sub) {
                    load = load.saturating_add(u64::from(h.width()) * u64::from(h.height()));
                }
            }
            let advance = 8usize.saturating_add(size).saturating_add(size & 1);
            if advance > fd.len() {
                break;
            }
            fd = &fd[advance..];
        }
    }

    load
}

/// Convenience wrapper: parse the container and report whether the
/// file's declared pixel load exceeds [`MAX_DECLARED_PIXELS`]. A file
/// that does not parse as a §2.3 container declares nothing (the
/// decoder refuses it before any pixel allocation).
pub fn over_declared_pixel_budget(bytes: &[u8]) -> bool {
    match container::parse(bytes) {
        Ok(c) => declared_pixel_load(bytes, &c) > MAX_DECLARED_PIXELS,
        Err(_) => false,
    }
}
