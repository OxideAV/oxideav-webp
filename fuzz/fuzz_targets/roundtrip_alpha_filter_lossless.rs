#![no_main]

//! Round-trip differential oracle over the §2.7.1.2 ALPH **compression
//! method 1** (lossless-compressed) decode path
//! `oxideav_webp::alph::decode_alpha`.
//!
//! The sibling `roundtrip_alpha_filter` target pins the §2.7.1.2 Stage-2
//! inverse-filter *values*, but only over method-0 (no-compression)
//! payloads: it stores the forward-filtered residual as raw bytes. That
//! leaves the §2.7.1.2 method-1 decode chain — the one a real
//! lossy-with-alpha WebP almost always uses — completely outside any
//! structured oracle:
//!
//! 1. de-compress a **headerless** §3 VP8L image-stream
//!    (`decode_lossless_headerless`), and
//! 2. lift the residual out of the **green** channel of each decoded
//!    ARGB pixel, *then*
//! 3. apply the same §2.7.1.2 Stage-2 inverse filter.
//!
//! The dumb-byte `decode_alph` / `decode_alpha_plane` harnesses feed the
//! method-1 branch attacker bytes, but a coverage-guided mutator
//! essentially never grows a buffer into a self-consistent §3 / §4 / §6
//! entropy stream, so steps 1–2 are barely reached and their *values*
//! are never checked. This harness closes that gap.
//!
//! ## Construction (all clean-room, RFC 9649 §2.7.1.2 + §3)
//!
//! * Synthesise a known alpha plane.
//! * Apply the §2.7.1.2 **forward** filter `X = (alpha - predictor) mod
//!   256` in scan order (the algebraic inverse of the decoder's
//!   `alpha = (predictor + X) % 256` reconstruction). The forward filter
//!   and its per-method border rules are derived solely from RFC 9649
//!   §2.7.1.2 Figure 11 + the left-most-column / top-most-row /
//!   `(0, 0)`-corner special-case text — identical to the proven
//!   method-0 target's filter.
//! * Pack each residual byte into the **green** channel of an opaque
//!   ARGB pixel (`A = 0xff`, `R = B = 0`), exactly mirroring the
//!   decoder's "the alpha values live in the GREEN channel of the
//!   decoded ARGB quadruplets" extraction (§2.7.1.2).
//! * Encode that ARGB image as a §3 **headerless** VP8L bitstream via
//!   `encode_argb_literals_with_width` (the crate's own bare-bitstream
//!   writer — VP8L is lossless, so the decoded green channel is
//!   bit-for-bit the residual we packed).
//! * Wrap as a §2.7.1.2 ALPH payload with `C = 1` (lossless), `F =
//!   method`, `P = 0`, decode through `decode_alpha`, and assert the
//!   reconstructed plane is byte-identical to the original.
//!
//! Because VP8L is lossless and the forward filter is the exact inverse
//! of the decoder's, the round-trip is lossless iff every link in the
//! method-1 chain — headerless VP8L decode, green-channel extraction,
//! and the Stage-2 inverse filter — matches the spec. A mismatch is a
//! genuine §2.7.1.2 / §3 contract violation.
//!
//! ## Input layout
//!
//! * Byte 0 — `filter_sel`: `data[0] % 4` selects the §2.7.1.2 F field
//!   (0 None / 1 Horizontal / 2 Vertical / 3 Gradient).
//! * Byte 1 — `w_sel`: width  = `1 + (data[1] % 32)`  (1..=32).
//! * Byte 2 — `h_sel`: height = `1 + (data[2] % 32)`  (1..=32).
//! * Bytes 3.. — the alpha plane (first `w * h` bytes, zero-padded).
//!
//! Dimensions cap at 32×32 so a single iteration's forward-filter +
//! VP8L encode + decode stays well under a millisecond.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::alph::decode_alpha;
use oxideav_webp::vp8l_encode::encode_argb_literals_with_width;

/// §2.7.1.2: `clip(v)` clamps a predictor candidate to `[0, 255]`.
fn clip(v: i32) -> i32 {
    v.clamp(0, 255)
}

/// §2.7.1.2 Figure 11 predictor for scan position `(x, y)` read from the
/// original (== reconstructed) plane `p` of dimensions `w * h`. Mirrors
/// the decoder's per-method border rules exactly so that the forward
/// residual the decoder consumes is well-defined.
fn predictor(p: &[u8], w: usize, x: usize, y: usize, method: u8) -> i32 {
    // §2.7.1.2: the top-left value at (0, 0) uses 0 as the predictor.
    if x == 0 && y == 0 {
        return 0;
    }
    let idx = |xx: usize, yy: usize| p[yy * w + xx] as i32;
    match method {
        // Method 0 (None): predictor = 0 everywhere.
        0 => 0,
        // Method 1 (Horizontal): predictor = A (left). §2.7.1.2: the
        // left-most column (0, y>0) is predicted by (0, y-1) above.
        1 => {
            if x == 0 {
                idx(0, y - 1)
            } else {
                idx(x - 1, y)
            }
        }
        // Method 2 (Vertical): predictor = B (above). §2.7.1.2: the
        // top-most row (x>0, 0) is predicted by (x-1, 0) on the left.
        2 => {
            if y == 0 {
                idx(x - 1, 0)
            } else {
                idx(x, y - 1)
            }
        }
        // Method 3 (Gradient): predictor = clip(A + B - C). §2.7.1.2:
        // the left-most column falls back to (0, y-1) above; the
        // top-most row falls back to (x-1, 0) left.
        3 => {
            if x == 0 {
                idx(0, y - 1)
            } else if y == 0 {
                idx(x - 1, 0)
            } else {
                let a = idx(x - 1, y);
                let b = idx(x, y - 1);
                let c = idx(x - 1, y - 1);
                clip(a + b - c)
            }
        }
        _ => unreachable!("method is data[0] % 4, in 0..=3"),
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let method = data[0] % 4;
    let w = 1 + (data[1] % 32) as usize;
    let h = 1 + (data[2] % 32) as usize;
    let count = w * h;

    // Build the original alpha plane from the remaining bytes (zero-pad
    // if the fuzz buffer is short).
    let mut plane = vec![0u8; count];
    let tail = &data[3..];
    let n = tail.len().min(count);
    plane[..n].copy_from_slice(&tail[..n]);

    // §2.7.1.2 forward filter: X = (alpha - predictor) mod 256, read in
    // scan order so each predictor sees only already-emitted neighbours.
    let mut residual = vec![0u8; count];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let pred = predictor(&plane, w, x, y, method);
            residual[i] = ((plane[i] as i32 - pred) & 0xff) as u8;
        }
    }

    // §2.7.1.2 method-1 carrier: the residual lives in the GREEN channel
    // of a §3 ARGB image-stream. Build opaque pixels (A=0xff, R=B=0) so
    // only the green byte carries information, then encode a headerless
    // VP8L bitstream — VP8L is lossless, so the decoder's green channel
    // is bit-for-bit the residual packed here.
    let argb: Vec<u32> = residual
        .iter()
        .map(|&g| 0xff00_0000u32 | ((g as u32) << 8))
        .collect();
    let bitstream = encode_argb_literals_with_width(&argb, w as u32);

    // §2.7.1.2 ALPH payload: info byte then the alpha bitstream. The
    // info byte packs Rsv=0, P=0 (no preprocessing), F=method (bits
    // 3..2), C=1 (lossless compression, bits 1..0).
    let info = ((method & 0b11) << 2) | 0b01;
    let mut payload = Vec::with_capacity(1 + bitstream.len());
    payload.push(info);
    payload.extend_from_slice(&bitstream);

    let decoded = decode_alpha(&payload, w as u32, h as u32)
        .expect("§2.7.1.2 method-1 ALPH payload built from a valid VP8L stream must decode");

    assert_eq!(
        decoded.len(),
        count,
        "§2.7.1.2 decoded plane must be width * height bytes",
    );
    assert_eq!(
        decoded, plane,
        "§2.7.1.2 method-1 chain (VP8L decode → green extract → inverse \
         filter F={method}, {w}x{h}) did not reproduce the original plane",
    );
});
