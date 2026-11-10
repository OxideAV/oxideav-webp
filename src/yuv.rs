//! Exact integer Y′CbCr ⇄ RGB kernels for the lossy (`VP8 `) path.
//!
//! RFC 9649 §2.5: "The VP8 specification describes how to decode the image
//! into Y'CbCr format. To convert to RGB, Recommendation 601 [REC601] SHOULD
//! be used." RFC 6386 §9.2 names the colour space as "YUV color space
//! similar to the YCrCb color space defined in [ITU-R_BT.601]".
//!
//! Rec. ITU-R BT.601-7 §2.5.3 quantises the analogue signals to 8 bits as
//!
//! ```text
//!   Y  = 219 E'Y  + 16            (E'Y  = 0.299 E'R + 0.587 E'G + 0.114 E'B)
//!   CR = 224 E'CR + 128           (E'CR = (E'R − E'Y) / 1.402)
//!   CB = 224 E'CB + 128           (E'CB = (E'B − E'Y) / 1.772)
//! ```
//!
//! i.e. the *limited* (studio) range, Y′ in `16..=235`, Cb/Cr in
//! `16..=240`. Inverting those three lines for 8-bit RGB gives
//!
//! ```text
//!   R = 1.164384 (Y − 16) + 1.596027 (Cr − 128)
//!   G = 1.164384 (Y − 16) − 0.391762 (Cb − 128) − 0.812968 (Cr − 128)
//!   B = 1.164384 (Y − 16) + 2.017232 (Cb − 128)
//! ```
//!
//! (`255/219 = 1.164384`, `255/224 × 1.402 = 1.596027`,
//! `255/224 × 1.772 = 2.017232`, `255/224 × 0.114 × 1.772 / 0.587 =
//! 0.391762`, `255/224 × 0.299 × 1.402 / 0.587 = 0.812968`). The kernels
//! below evaluate those in Q16 fixed point with round-half-up and clamp
//! to `0..=255` (RFC 6386 §9.2 pixel-value clamping). A full-range
//! variant (the un-scaled BT.601 matrix) is kept for a caller that
//! explicitly tags its planes [`ColorRange::Full`](crate::ColorRange::Full).
//!
//! Chroma is upsampled nearest-neighbour: RFC 6386 only fixes the 4:2:0
//! geometry, not the reconstruction filter, so the pair of luma samples
//! `(2k, 2k+1)` shares chroma column `k` and rows `(2j, 2j+1)` share
//! chroma row `j`.

/// Rounding bias for the `>> 16` of a Q16 product.
const HALF: i32 = 1 << 15;

/// Q16 limited-range coefficients (see the module docs).
const L_Y: i32 = 76_309; // 1.164384
const L_RV: i32 = 104_597; // 1.596027
const L_GU: i32 = 25_675; // 0.391762
const L_GV: i32 = 53_279; // 0.812968
const L_BU: i32 = 132_201; // 2.017232

/// Q16 full-range coefficients (the un-scaled BT.601 matrix).
const F_RV: i32 = 91_881; // 1.402
const F_GU: i32 = 22_554; // 0.344136
const F_GV: i32 = 46_802; // 0.714136
const F_BU: i32 = 116_130; // 1.772

#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// The three Q16 chroma contributions (rounding bias folded in) shared by
/// every luma sample that maps to the chroma pair `(cb, cr)`.
#[inline]
fn chroma_offsets(cb: u8, cr: u8, full: bool) -> (i32, i32, i32) {
    let d = cb as i32 - 128;
    let e = cr as i32 - 128;
    if full {
        (
            F_RV * e + HALF,
            -F_GU * d - F_GV * e + HALF,
            F_BU * d + HALF,
        )
    } else {
        (
            L_RV * e + HALF,
            -L_GU * d - L_GV * e + HALF,
            L_BU * d + HALF,
        )
    }
}

/// The Q16 luma term.
#[inline]
fn luma_term(y: u8, full: bool) -> i32 {
    if full {
        (y as i32) << 16
    } else {
        L_Y * (y as i32 - 16)
    }
}

/// Convert one Y′CbCr sample to RGB (the per-pixel reference the tests
/// check the row kernel against).
#[cfg(test)]
#[inline]
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8, full: bool) -> [u8; 3] {
    let yl = luma_term(y, full);
    let (ro, go, bo) = chroma_offsets(cb, cr, full);
    [
        clamp_u8((yl + ro) >> 16),
        clamp_u8((yl + go) >> 16),
        clamp_u8((yl + bo) >> 16),
    ]
}

/// One 4:2:0 row to packed pixels of `BPP` bytes (3 or 4; the fourth
/// byte is set to `255`), `FULL` selecting the full-range matrix. The
/// loop body is two fixed-size pixels per chroma pair with no bounds
/// checks or range switches left in it.
#[inline(always)]
fn convert_row_kernel<const BPP: usize, const FULL: bool>(
    y_row: &[u8],
    u_row: &[u8],
    v_row: &[u8],
    out: &mut [u8],
) {
    #[inline(always)]
    fn put<const BPP: usize, const FULL: bool>(
        px: &mut [u8; BPP],
        y: u8,
        (ro, go, bo): (i32, i32, i32),
    ) {
        let yl = luma_term(y, FULL);
        px[0] = clamp_u8((yl + ro) >> 16);
        px[1] = clamp_u8((yl + go) >> 16);
        px[2] = clamp_u8((yl + bo) >> 16);
        if let Some(alpha) = px.get_mut(3) {
            *alpha = 0xff;
        }
    }
    let (y_pairs, y_rest) = y_row.as_chunks::<2>();
    let (pixels, _) = out.as_chunks_mut::<BPP>();
    let (pixel_pairs, pixel_rest) = pixels.as_chunks_mut::<2>();
    for (((ys, &cb), &cr), px) in y_pairs
        .iter()
        .zip(u_row.iter())
        .zip(v_row.iter())
        .zip(pixel_pairs.iter_mut())
    {
        let offsets = chroma_offsets(cb, cr, FULL);
        let [first, second] = px;
        put::<BPP, FULL>(first, ys[0], offsets);
        put::<BPP, FULL>(second, ys[1], offsets);
    }
    // Odd width: the last luma sample has a chroma pair to itself.
    if let (Some(&y), Some(&cb), Some(&cr), Some(px)) = (
        y_rest.first(),
        u_row.get(y_pairs.len()),
        v_row.get(y_pairs.len()),
        pixel_rest.first_mut(),
    ) {
        put::<BPP, FULL>(px, y, chroma_offsets(cb, cr, FULL));
    }
}

/// Convert one 4:2:0 row. `y_row` is `w` luma samples; `u_row` / `v_row`
/// are `⌈w/2⌉` chroma samples of the chroma row this luma row maps to;
/// `a_row`, when present, is `w` alpha samples. `out` is `w × out_bpp`
/// bytes (`out_bpp` 3 or 4); for `out_bpp == 4` the alpha byte is taken
/// from `a_row` or set to `255`.
pub(crate) fn convert_row(
    y_row: &[u8],
    u_row: &[u8],
    v_row: &[u8],
    a_row: Option<&[u8]>,
    out: &mut [u8],
    out_bpp: usize,
    full: bool,
) {
    let w = y_row.len();
    debug_assert!(out.len() >= w * out_bpp);
    debug_assert!(u_row.len() >= w.div_ceil(2) && v_row.len() >= w.div_ceil(2));
    let out = &mut out[..w * out_bpp];
    match (out_bpp, full) {
        (4, false) => convert_row_kernel::<4, false>(y_row, u_row, v_row, out),
        (4, true) => convert_row_kernel::<4, true>(y_row, u_row, v_row, out),
        (_, false) => convert_row_kernel::<3, false>(y_row, u_row, v_row, out),
        (_, true) => convert_row_kernel::<3, true>(y_row, u_row, v_row, out),
    }
    if out_bpp == 4
        && let Some(a_row) = a_row
    {
        for (px, &a) in out.as_chunks_mut::<4>().0.iter_mut().zip(a_row) {
            px[3] = a;
        }
    }
}

/// Convert tightly packed 4:2:0 planes to packed RGBA (alpha `255`).
pub(crate) fn yuv420_to_rgba(
    width: usize,
    height: usize,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    full: bool,
) -> Vec<u8> {
    let cw = width.div_ceil(2);
    let mut out = vec![0u8; width * height * 4];
    for row in 0..height {
        let y_row = &y[row * width..row * width + width];
        let cb = (row / 2) * cw;
        let u_row = &u[cb..cb + cw];
        let v_row = &v[cb..cb + cw];
        convert_row(
            y_row,
            u_row,
            v_row,
            None,
            &mut out[row * width * 4..(row + 1) * width * 4],
            4,
            full,
        );
    }
    out
}

// ─────────────────────────── forward (encode) side ───────────────────────

/// Q16 forward coefficients, limited range — the BT.601 §2.5.3
/// quantisation of `E'Y`, `E'CB`, `E'CR` scaled by `219 / 255` and
/// `224 / 255`:
///
/// ```text
///   Y  = 16  + (65.738 R + 129.057 G + 25.064 B) / 256
///   Cb = 128 + (−37.945 R − 74.494 G + 112.439 B) / 256
///   Cr = 128 + (112.439 R − 94.154 G − 18.285 B) / 256
/// ```
const FY: [i32; 3] = [16_829, 33_039, 6_416]; // 0.256788, 0.504129, 0.097906
const FU: [i32; 3] = [-9_714, -19_071, 28_784]; // −0.148223, −0.290993, 0.439216
const FV: [i32; 3] = [28_784, -24_103, -4_681]; // 0.439216, −0.367788, −0.071427

/// Forward limited-range BT.601 for one RGB sample.
#[inline]
pub(crate) fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> [u8; 3] {
    let (r, g, b) = (r as i32, g as i32, b as i32);
    let y = ((FY[0] * r + FY[1] * g + FY[2] * b + HALF) >> 16) + 16;
    let cb = ((FU[0] * r + FU[1] * g + FU[2] * b + HALF) >> 16) + 128;
    let cr = ((FV[0] * r + FV[1] * g + FV[2] * b + HALF) >> 16) + 128;
    [clamp_u8(y), clamp_u8(cb), clamp_u8(cr)]
}

/// One row of packed RGB(A) (`BPP` 3 or 4) to full-resolution Y′, Cb and
/// Cr samples (`y`, `cb`, `cr` each as long as the row has pixels). A
/// straight map over fixed-size pixels, so the compiler vectorises it.
#[inline]
fn row_to_ycbcr<const BPP: usize>(src: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]) {
    let (pixels, _) = src.as_chunks::<BPP>();
    for (((px, y), cb), cr) in pixels
        .iter()
        .zip(y.iter_mut())
        .zip(cb.iter_mut())
        .zip(cr.iter_mut())
    {
        let [yy, u, v] = rgb_to_ycbcr(px[0], px[1], px[2]);
        *y = yy;
        *cb = u;
        *cr = v;
    }
}

/// Rounded mean of each 2×2 block of two full-resolution chroma rows
/// (`top`, `bottom`, each `w` samples) into `out` (`⌈w/2⌉` samples); a
/// trailing odd column averages its two samples.
#[inline]
fn chroma_pair_mean(top: &[u8], bottom: &[u8], out: &mut [u8]) {
    let (top_pairs, top_rest) = top.as_chunks::<2>();
    let (bottom_pairs, bottom_rest) = bottom.as_chunks::<2>();
    for ((t, b), o) in top_pairs.iter().zip(bottom_pairs).zip(out.iter_mut()) {
        *o = ((u16::from(t[0]) + u16::from(t[1]) + u16::from(b[0]) + u16::from(b[1]) + 2) >> 2)
            as u8;
    }
    if let (Some(&t), Some(&b), Some(o)) = (top_rest.first(), bottom_rest.first(), out.last_mut()) {
        *o = ((u16::from(t) + u16::from(b) + 1) >> 1) as u8;
    }
}

/// Rounded mean of each horizontal pair of one full-resolution chroma
/// row (the last row of an odd-height picture); a trailing odd column is
/// copied.
#[inline]
fn chroma_row_mean(row: &[u8], out: &mut [u8]) {
    let (pairs, rest) = row.as_chunks::<2>();
    for (p, o) in pairs.iter().zip(out.iter_mut()) {
        *o = ((u16::from(p[0]) + u16::from(p[1]) + 1) >> 1) as u8;
    }
    if let (Some(&v), Some(o)) = (rest.first(), out.last_mut()) {
        *o = v;
    }
}

/// Convert a band of packed RGB(A) rows to 4:2:0. `data` starts at the
/// band's first row (an even picture row); `y` holds the band's luma
/// rows, `cb` / `cr` its chroma rows (`⌈rows / 2⌉`).
fn rgb_band_to_yuv420<const BPP: usize>(
    width: usize,
    data: &[u8],
    stride: usize,
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) {
    let cw = width.div_ceil(2);
    // Full-resolution chroma of the row pair being averaged:
    // [top Cb, top Cr, bottom Cb, bottom Cr].
    let mut scratch = vec![0u8; 4 * width];
    let (top, bottom) = scratch.split_at_mut(2 * width);
    let (top_cb, top_cr) = top.split_at_mut(width);
    let (bottom_cb, bottom_cr) = bottom.split_at_mut(width);
    for (pair, ((y_rows, cb_row), cr_row)) in y
        .chunks_mut(2 * width)
        .zip(cb.chunks_exact_mut(cw))
        .zip(cr.chunks_exact_mut(cw))
        .enumerate()
    {
        let row = 2 * pair;
        let src = &data[row * stride..row * stride + width * BPP];
        if y_rows.len() == 2 * width {
            let (y_top, y_bottom) = y_rows.split_at_mut(width);
            row_to_ycbcr::<BPP>(src, y_top, top_cb, top_cr);
            let src = &data[(row + 1) * stride..(row + 1) * stride + width * BPP];
            row_to_ycbcr::<BPP>(src, y_bottom, bottom_cb, bottom_cr);
            chroma_pair_mean(top_cb, bottom_cb, cb_row);
            chroma_pair_mean(top_cr, bottom_cr, cr_row);
        } else {
            row_to_ycbcr::<BPP>(src, y_rows, top_cb, top_cr);
            chroma_row_mean(top_cb, cb_row);
            chroma_row_mean(top_cr, cr_row);
        }
    }
}

/// Pictures below this many pixels are converted on the calling thread
/// whatever the budget: a thread costs more than it saves.
const PARALLEL_CONVERT_MIN_PIXELS: usize = 256 * 1024;

/// Convert packed 8-bit RGB(A) rows (`bpp` 3 or 4, `stride` bytes per row)
/// to tightly packed limited-range 4:2:0 planes `(y, cb, cr)`. Each chroma
/// sample is the rounded mean of the Cb / Cr of its 2×2 (or edge 2×1 /
/// 1×2 / 1×1) luma block.
///
/// With `threads > 1` a large picture is split into bands of whole row
/// pairs converted on scoped threads; every sample depends only on its
/// own block, so the planes are the same for every `threads`.
pub(crate) fn rgb_to_yuv420(
    width: usize,
    height: usize,
    data: &[u8],
    stride: usize,
    bpp: usize,
    threads: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut y_plane = vec![0u8; width * height];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];
    let band = |data: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8]| {
        if bpp == 3 {
            rgb_band_to_yuv420::<3>(width, data, stride, y, cb, cr);
        } else {
            rgb_band_to_yuv420::<4>(width, data, stride, y, cb, cr);
        }
    };
    let bands = if width * height < PARALLEL_CONVERT_MIN_PIXELS {
        1
    } else {
        threads.clamp(1, ch)
    };
    if bands == 1 {
        band(data, &mut y_plane, &mut cb_plane, &mut cr_plane);
        return (y_plane, cb_plane, cr_plane);
    }
    // Chroma rows (= row pairs) per band, the first `ch % bands` bands
    // taking one more.
    let (base, extra) = (ch / bands, ch % bands);
    let mut jobs = Vec::with_capacity(bands);
    let (mut y_rest, mut cb_rest, mut cr_rest) =
        (&mut y_plane[..], &mut cb_plane[..], &mut cr_plane[..]);
    let mut first_pair = 0;
    for index in 0..bands {
        let pairs = base + usize::from(index < extra);
        let luma_rows = (2 * pairs).min(height - 2 * first_pair);
        let (y_band, y_tail) = y_rest.split_at_mut(luma_rows * width);
        let (cb_band, cb_tail) = cb_rest.split_at_mut(pairs * cw);
        let (cr_band, cr_tail) = cr_rest.split_at_mut(pairs * cw);
        (y_rest, cb_rest, cr_rest) = (y_tail, cb_tail, cr_tail);
        jobs.push((&data[2 * first_pair * stride..], y_band, cb_band, cr_band));
        first_pair += pairs;
    }
    // Every thread (this one included) takes bands off the queue until
    // it is empty, so a thread that fails to start costs nothing but
    // time.
    let queue = std::sync::Mutex::new(jobs);
    let work = || {
        loop {
            let job = queue.lock().unwrap_or_else(|e| e.into_inner()).pop();
            let Some((src, y, cb, cr)) = job else { break };
            band(src, y, cb, cr);
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..bands {
            let _ = std::thread::Builder::new().spawn_scoped(scope, work);
        }
        work();
    });
    (y_plane, cb_plane, cr_plane)
}

/// The definition [`rgb_to_yuv420`] is checked against: per-block sums,
/// counts and a rounded division, one sample at a time.
#[cfg(test)]
fn rgb_to_yuv420_reference(
    width: usize,
    height: usize,
    data: &[u8],
    stride: usize,
    bpp: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut y_plane = vec![0u8; width * height];
    let mut cb_sum = vec![0u32; cw * ch];
    let mut cr_sum = vec![0u32; cw * ch];
    let mut cnt = vec![0u32; cw * ch];
    for row in 0..height {
        let src = &data[row * stride..row * stride + width * bpp];
        for (x, px) in src.chunks_exact(bpp).enumerate() {
            let [yy, cb, cr] = rgb_to_ycbcr(px[0], px[1], px[2]);
            y_plane[row * width + x] = yy;
            let ci = (row / 2) * cw + x / 2;
            cb_sum[ci] += cb as u32;
            cr_sum[ci] += cr as u32;
            cnt[ci] += 1;
        }
    }
    let mean = |sums: &[u32]| -> Vec<u8> {
        sums.iter()
            .zip(cnt.iter())
            .map(|(&s, &n)| ((s + n / 2) / n.max(1)) as u8)
            .collect()
    };
    (y_plane, mean(&cb_sum), mean(&cr_sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_range_anchors() {
        // BT.601 §2.5.3: black = (16, 128, 128), white = (235, 128, 128).
        assert_eq!(ycbcr_to_rgb(16, 128, 128, false), [0, 0, 0]);
        assert_eq!(ycbcr_to_rgb(235, 128, 128, false), [255, 255, 255]);
        // Neutral chroma stays grey; below-black / above-white clamp.
        assert_eq!(ycbcr_to_rgb(0, 128, 128, false), [0, 0, 0]);
        assert_eq!(ycbcr_to_rgb(255, 128, 128, false), [255, 255, 255]);
        let g = ycbcr_to_rgb(126, 128, 128, false);
        assert_eq!(g, [128, 128, 128]);
    }

    #[test]
    fn full_range_anchors() {
        for y in [0u8, 1, 64, 127, 128, 200, 255] {
            assert_eq!(ycbcr_to_rgb(y, 128, 128, true), [y, y, y]);
        }
        assert_eq!(ycbcr_to_rgb(128, 128, 255, true)[0], 255);
        assert_eq!(ycbcr_to_rgb(128, 255, 128, true)[2], 255);
    }

    #[test]
    fn forward_then_inverse_is_close_on_primaries() {
        // 8-bit limited-range quantisation loses at most a couple of
        // code values per channel on a round trip.
        for &(r, g, b) in &[
            (0u8, 0u8, 0u8),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (180, 60, 90),
            (17, 200, 33),
        ] {
            let [y, cb, cr] = rgb_to_ycbcr(r, g, b);
            let back = ycbcr_to_rgb(y, cb, cr, false);
            for (a, want) in back.iter().zip([r, g, b]) {
                assert!(
                    (*a as i32 - want as i32).abs() <= 2,
                    "({r},{g},{b}) -> ({y},{cb},{cr}) -> {back:?}"
                );
            }
        }
        // The BT.601 anchors.
        assert_eq!(rgb_to_ycbcr(0, 0, 0), [16, 128, 128]);
        assert_eq!(rgb_to_ycbcr(255, 255, 255), [235, 128, 128]);
    }

    #[test]
    fn convert_row_matches_per_pixel_for_odd_widths() {
        for w in 1..=9usize {
            let y: Vec<u8> = (0..w).map(|i| (i * 37 + 11) as u8).collect();
            let cw = w.div_ceil(2);
            let u: Vec<u8> = (0..cw).map(|i| (i * 53 + 200) as u8).collect();
            let v: Vec<u8> = (0..cw).map(|i| (i * 71 + 7) as u8).collect();
            let a: Vec<u8> = (0..w).map(|i| (i * 13) as u8).collect();
            let mut out = vec![0u8; w * 4];
            convert_row(&y, &u, &v, Some(&a), &mut out, 4, false);
            for x in 0..w {
                let want = ycbcr_to_rgb(y[x], u[x / 2], v[x / 2], false);
                assert_eq!(&out[x * 4..x * 4 + 3], &want, "w={w} x={x}");
                assert_eq!(out[x * 4 + 3], a[x]);
            }
            let mut out3 = vec![0u8; w * 3];
            convert_row(&y, &u, &v, None, &mut out3, 3, true);
            for x in 0..w {
                let want = ycbcr_to_rgb(y[x], u[x / 2], v[x / 2], true);
                assert_eq!(&out3[x * 3..x * 3 + 3], &want, "full w={w} x={x}");
            }
        }
    }

    #[test]
    fn rgb_to_yuv420_averages_chroma_blocks() {
        // 3×3 opaque RGBA: chroma grid is 2×2 with 4 / 2 / 2 / 1 samples.
        let w = 3;
        let mut rgba = Vec::new();
        for i in 0..9u8 {
            rgba.extend_from_slice(&[i * 20, 255 - i * 20, i * 7, 255]);
        }
        let (y, cb, cr) = rgb_to_yuv420(w, 3, &rgba, w * 4, 4, 1);
        assert_eq!(y.len(), 9);
        assert_eq!(cb.len(), 4);
        assert_eq!(cr.len(), 4);
        // Bottom-right chroma sample covers exactly pixel (2, 2).
        let [_, cb8, cr8] = rgb_to_ycbcr(160, 95, 56);
        assert_eq!(cb[3], cb8);
        assert_eq!(cr[3], cr8);
    }

    #[test]
    fn rgb_to_yuv420_matches_the_per_sample_definition() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        };
        // Odd / even sides, padded strides, both pixel sizes, and a
        // picture large enough to split into bands.
        for &(w, h) in &[(1, 1), (1, 2), (2, 1), (5, 4), (6, 7), (17, 33), (640, 411)] {
            for bpp in [3usize, 4] {
                let stride = w * bpp + 5;
                let data: Vec<u8> = (0..stride * h).map(|_| next()).collect();
                let want = rgb_to_yuv420_reference(w, h, &data, stride, bpp);
                for threads in [1usize, 2, 3, 7, 64] {
                    let got = rgb_to_yuv420(w, h, &data, stride, bpp, threads);
                    assert!(got == want, "{w}x{h} bpp {bpp} threads {threads}");
                }
            }
        }
    }
}
