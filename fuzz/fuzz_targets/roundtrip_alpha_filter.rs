#![no_main]

//! Round-trip differential oracle over the §2.7.1.2 ALPH inverse-filter
//! stage `oxideav_webp::alph::decode_alpha`.
//!
//! The existing `decode_alph` / `decode_alpha_plane` harnesses feed
//! attacker-controlled bytes straight into `decode_alpha` and assert
//! only the "always returns a `Result`; `Ok` plane is `width * height`
//! bytes" length contract. Neither pins the *values* the §2.7.1.2 Stage
//! 2 inverse filter reconstructs. This harness closes that gap: it
//! synthesises a known alpha plane, applies the §2.7.1.2 **forward**
//! filter (the algebraic inverse of the decoder's per-pixel reconstruct
//! step), wraps the residual as a §2.7.1.2 method-0 (no-compression)
//! `ALPH` payload, decodes it back through `decode_alpha`, and asserts
//! the reconstructed plane is byte-identical to the original.
//!
//! ## Why this is clean-room
//!
//! The forward filter below is derived solely from RFC 9649 §2.7.1.2
//! (Figure 11 + the "alpha = (predictor + X) % 256" reconstruction and
//! the left-most / top-most special-case text). The decoder computes
//!
//! ```text
//! alpha = (predictor + X) % 256
//! ```
//!
//! so the spec's own algebra inverts it to the residual the encoder
//! must store:
//!
//! ```text
//! X = (alpha - predictor) mod 256
//! ```
//!
//! where, per §2.7.1.2 Figure 11 and the special-case text:
//!
//! * `(0, 0)` predicts `0` for every method.
//! * Method 1 (Horizontal): predictor = `A` (left). The left-most
//!   column `(0, y>0)` falls back to the pixel just above `(0, y-1)`.
//! * Method 2 (Vertical): predictor = `B` (above). The top-most row
//!   `(x>0, 0)` falls back to the pixel on the left `(x-1, 0)`.
//! * Method 3 (Gradient): predictor = `clip(A + B - C)`. The left-most
//!   column falls back to `(0, y-1)` above; the top-most row falls back
//!   to `(x-1, 0)` left; `clip(v)` clamps to `[0, 255]`.
//! * Method 0 (None): predictor = `0` everywhere, so the residual is
//!   the plane itself (identity).
//!
//! Because the filter is causal — each predictor reads only neighbours
//! at scan positions strictly earlier than `X` — and a correct decode
//! reproduces the original plane exactly, the predictor the forward
//! pass reads from the *original* plane is bit-for-bit the predictor the
//! decoder reads from the *reconstructed* plane. The round-trip is
//! therefore lossless iff the decoder's §2.7.1.2 inverse filter matches
//! the spec, which is exactly the property under test.
//!
//! ## Input layout
//!
//! * Byte 0 — `filter_sel`: `data[0] % 4` selects the §2.7.1.2 F field
//!   (0 None / 1 Horizontal / 2 Vertical / 3 Gradient).
//! * Byte 1 — `w_sel`: width  = `1 + (data[1] % 32)`  (1..=32).
//! * Byte 2 — `h_sel`: height = `1 + (data[2] % 32)`  (1..=32).
//! * Bytes 3.. — the alpha plane. The first `w * h` bytes (zero-padded
//!   if the buffer is short) are the original alpha values in scan
//!   order. Capping the dimensions at 32×32 keeps a single iteration's
//!   forward-filter + decode under a few thousand byte operations.
//!
//! A buffer shorter than 3 bytes is ignored (nothing to round-trip).
//!
//! ## Contract under test, per RFC 9649 §2.7.1.2
//!
//! * `decode_alpha` returns `Ok` for a well-formed method-0 payload of
//!   exactly `1 + w * h` bytes (no panic, no overflow).
//! * The reconstructed plane equals the original plane byte-for-byte:
//!   the §2.7.1.2 Stage-2 inverse filter is the exact inverse of the
//!   spec's forward filter for all four F methods, across the interior,
//!   the left-most column, the top-most row, and the `(0, 0)` corner.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::alph::decode_alpha;

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

    // §2.7.1.2 ALPH payload: info byte then the alpha bitstream. The
    // info byte packs Rsv=0, P=0 (no preprocessing), F=method (bits
    // 3..2), C=0 (no compression, bits 1..0).
    let info = (method & 0b11) << 2;
    let mut payload = Vec::with_capacity(1 + count);
    payload.push(info);
    payload.extend_from_slice(&residual);

    let decoded = decode_alpha(&payload, w as u32, h as u32)
        .expect("§2.7.1.2 method-0 ALPH payload of length 1 + w*h must decode");

    assert_eq!(
        decoded.len(),
        count,
        "§2.7.1.2 decoded plane must be width * height bytes",
    );
    assert_eq!(
        decoded, plane,
        "§2.7.1.2 inverse filter (F={method}, {w}x{h}) did not reproduce the original plane",
    );
});
