//! Timing matrix: size x effort x threads -> milliseconds and bytes,
//! for lossy and lossless still encode and decode through the public
//! entry points (`encode_rgb8`, `decode_rgba8`), plus an eight-frame
//! animation (`encode_animation`).
//!
//! A plain `main` (no statistics harness): each cell is the median of a
//! few runs, printed as one Markdown table row, so the README table can
//! be regenerated with
//!
//! ```text
//! cargo bench --bench still_matrix
//! ```
//!
//! Environment:
//!
//! * `STILL_MATRIX_SIZES` — comma-separated `WxH` list (default
//!   `800x533,2400x1600`).
//! * `STILL_MATRIX_RUNS` — runs per cell (default 5; cells slower than
//!   two seconds run once).
//! * `STILL_MATRIX_SECTIONS` — any of `lossy`, `alpha`, `lossless`,
//!   `decode`, `animation` (default all five).
//! * `STILL_MATRIX_HASH` — when set, each encode row ends with the
//!   64-bit FNV-1a of the file, to compare outputs between builds.
//!
//! Without `--bench` on the command line (what `cargo test --benches`
//! does) it runs one tiny smoke pass instead of the matrix.

use std::time::{Duration, Instant};

use oxideav_webp::{
    EncodeOptions, Frame, WebpImage, decode_all, decode_rgba8, encode_animation, encode_rgb8,
    encode_rgba8,
};

/// Deterministic photo-like RGB picture: seven octaves of smooth lattice
/// noise per channel (correlated between channels, like a photograph's
/// luma), three shaded disks with soft edges, and a little per-sample
/// noise. No exact repeats, strong spatial correlation, a few edges.
fn photo_rgb(width: u32, height: u32) -> Vec<u8> {
    fn lattice(ix: u32, iy: u32, seed: u32) -> f32 {
        let mut h = ix
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add(iy.wrapping_mul(0x85EB_CA77))
            .wrapping_add(seed.wrapping_mul(0xC2B2_AE3D));
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        h = h.wrapping_mul(0x297A_2D39);
        h ^= h >> 15;
        (h & 0xFFFF) as f32 / 65535.0
    }
    fn smooth(x: f32, y: f32, seed: u32) -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
        let (ix, iy) = (x0 as u32, y0 as u32);
        let a = lattice(ix, iy, seed);
        let b = lattice(ix + 1, iy, seed);
        let c = lattice(ix, iy + 1, seed);
        let d = lattice(ix + 1, iy + 1, seed);
        let top = a + (b - a) * sx;
        let bottom = c + (d - c) * sx;
        top + (bottom - top) * sy
    }
    fn octaves(x: f32, y: f32, seed: u32) -> f32 {
        let mut sum = 0.0;
        let mut amplitude = 0.5;
        let mut total = 0.0;
        let mut frequency = 1.0;
        for octave in 0..7 {
            sum += amplitude * smooth(x * frequency, y * frequency, seed + octave);
            total += amplitude;
            amplitude *= 0.9;
            frequency *= 2.0;
        }
        sum / total
    }
    let (w, h) = (width as f32, height as f32);
    let scale = 14.0 / 800.0;
    let disks = [
        (0.30 * w, 0.35 * h, 0.18 * h, [230.0, 190.0, 120.0]),
        (0.68 * w, 0.55 * h, 0.25 * h, [70.0, 110.0, 190.0]),
        (0.50 * w, 0.85 * h, 0.12 * h, [200.0, 60.0, 80.0]),
    ];
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::with_capacity(width as usize * height as usize * 3);
    for py in 0..height {
        for px in 0..width {
            let (x, y) = (px as f32, py as f32);
            let luma = octaves(x * scale, y * scale, 11);
            let mut rgb = [
                40.0 + 170.0 * (0.7 * luma + 0.3 * octaves(x * scale, y * scale, 23)),
                50.0 + 160.0 * (0.8 * luma + 0.2 * octaves(x * scale, y * scale, 37)),
                60.0 + 150.0 * (0.6 * luma + 0.4 * octaves(x * scale, y * scale, 53)),
            ];
            for (cx, cy, radius, colour) in disks {
                let distance = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                let cover = ((radius - distance) / 2.0).clamp(0.0, 1.0);
                if cover > 0.0 {
                    let shade = 0.55 + 0.45 * ((cx - x + cy - y) / (2.0 * radius)).clamp(-1.0, 1.0);
                    for (channel, base) in rgb.iter_mut().zip(colour) {
                        *channel += cover * (base * shade - *channel);
                    }
                }
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            for (i, channel) in rgb.iter().enumerate() {
                let noise = ((state >> (8 * i)) & 0x7) as f32 - 3.5;
                out.push((channel + noise).clamp(0.0, 255.0) as u8);
            }
        }
    }
    out
}

/// Median wall time of `runs` calls (one call when the first takes two
/// seconds or more), with the last result.
fn time_median<T>(runs: usize, mut f: impl FnMut() -> T) -> (Duration, T) {
    let start = Instant::now();
    let mut result = f();
    let first = start.elapsed();
    if runs <= 1 || first >= Duration::from_secs(2) {
        return (first, result);
    }
    let mut samples = vec![first];
    for _ in 1..runs {
        let start = Instant::now();
        result = f();
        samples.push(start.elapsed());
    }
    samples.sort();
    (samples[samples.len() / 2], result)
}

/// 64-bit FNV-1a.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn ms(d: Duration) -> String {
    let v = d.as_secs_f64() * 1e3;
    if v >= 100.0 {
        format!("{v:.0}")
    } else if v >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// The photo-like picture with an alpha channel: an opaque disc with a
/// soft edge on a transparent ground, a quarter of it at half opacity.
fn photo_rgba_with_alpha(width: u32, height: u32) -> Vec<u8> {
    let rgb = photo_rgb(width, height);
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let radius = width.min(height) as f32 * 0.42;
    rgb.as_chunks::<3>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(i, px)| {
            let (x, y) = ((i as u32 % width) as f32, (i as u32 / width) as f32);
            let distance = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            let mut alpha = ((radius - distance) * 12.0).clamp(0.0, 255.0);
            if x > cx && y > cy {
                alpha *= 0.5;
            }
            [px[0], px[1], px[2], alpha as u8]
        })
        .collect()
}

fn encode_section(
    title: &str,
    base: &EncodeOptions,
    methods: &[u8],
    threads: &[usize],
    sizes: &[(u32, u32)],
    runs: usize,
) {
    encode_section_with(title, base, methods, threads, sizes, runs, false);
}

fn encode_section_with(
    title: &str,
    base: &EncodeOptions,
    methods: &[u8],
    threads: &[usize],
    sizes: &[(u32, u32)],
    runs: usize,
    alpha: bool,
) {
    println!("\n### {title}\n");
    print!("| size | effort |");
    for t in threads {
        print!(" {t} thread{} ms |", if *t == 1 { "" } else { "s" });
    }
    println!(" bytes |");
    println!("|---|---|{}---:|", "---:|".repeat(threads.len()));
    for &(w, h) in sizes {
        let pixels = if alpha {
            photo_rgba_with_alpha(w, h)
        } else {
            photo_rgb(w, h)
        };
        for &method in methods {
            print!("| {w}x{h} | {method} |");
            let mut bytes = None;
            let mut hash = 0u64;
            for &t in threads {
                let opts = base.clone().with_method(method).with_threads(t);
                let (d, out) = time_median(runs, || {
                    if alpha {
                        encode_rgba8(w, h, &pixels, &opts).expect("encode")
                    } else {
                        encode_rgb8(w, h, &pixels, &opts).expect("encode")
                    }
                });
                let this = fnv1a64(&out);
                match bytes {
                    None => (bytes, hash) = (Some(out.len()), this),
                    Some(_) => assert_eq!(hash, this, "thread budget changed the output"),
                }
                print!(" {} |", ms(d));
            }
            if std::env::var_os("STILL_MATRIX_HASH").is_some() {
                println!(" {} | {hash:016x} |", bytes.unwrap_or(0));
            } else {
                println!(" {} |", bytes.unwrap_or(0));
            }
        }
    }
}

fn decode_section(sizes: &[(u32, u32)], runs: usize) {
    println!("\n### Decode to RGBA (`decode_rgba8`)\n");
    println!("| size | file | ms | file bytes |");
    println!("|---|---|---:|---:|");
    for &(w, h) in sizes {
        let rgb = photo_rgb(w, h);
        let lossy = encode_rgb8(w, h, &rgb, &EncodeOptions::default().with_quality(80.0))
            .expect("lossy encode");
        let lossless = encode_rgb8(w, h, &rgb, &EncodeOptions::default().with_method(2))
            .expect("lossless encode");
        for (name, file) in [("lossy, quality 80", &lossy), ("lossless", &lossless)] {
            let (d, image) = time_median(runs, || decode_rgba8(file).expect("decode"));
            assert_eq!((image.width, image.height), (w, h));
            println!("| {w}x{h} | {name} | {} | {} |", ms(d), file.len());
        }
    }
}

/// Eight frames of the photo-like picture with a bright square moving
/// across it: most of each frame repeats the previous one.
fn animation_frames(w: u32, h: u32) -> Vec<Frame> {
    let rgb = photo_rgb(w, h);
    let side = (h / 6).max(2);
    (0..8u32)
        .map(|i| {
            let mut rgba: Vec<u8> = rgb
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|px| [px[0], px[1], px[2], 0xff])
                .collect();
            let (x0, y0) = ((w - side) * i / 8, (h - side) * i / 8);
            for y in y0..y0 + side {
                for x in x0..x0 + side {
                    let at = ((y * w + x) * 4) as usize;
                    rgba[at..at + 3].copy_from_slice(&[250, 240, 40]);
                }
            }
            let image = WebpImage::from_rgba8(w, h, rgba).expect("frame");
            Frame::new(image, Some(Duration::from_millis(40)))
        })
        .collect()
}

fn animation_section(sizes: &[(u32, u32)], threads: &[usize], runs: usize) {
    println!("\n### Animation encode, 8 frames (`encode_animation`)\n");
    print!("| size | frames |");
    for t in threads {
        print!(" {t} thread{} ms |", if *t == 1 { "" } else { "s" });
    }
    println!(" bytes |");
    println!("|---|---|{}---:|", "---:|".repeat(threads.len()));
    for &(w, h) in sizes {
        let frames = animation_frames(w, h);
        for (name, base) in [
            (
                "lossless, effort 2",
                EncodeOptions::default().with_method(2),
            ),
            (
                "lossy, quality 80, effort 4",
                EncodeOptions::default().with_quality(80.0),
            ),
            (
                "lossy, quality 80, effort 1",
                EncodeOptions::default().with_quality(80.0).with_method(1),
            ),
        ] {
            print!("| {w}x{h} | {name} |");
            let mut reference: Option<Vec<u8>> = None;
            for &t in threads {
                let opts = base.clone().with_threads(t);
                let (d, file) =
                    time_median(runs, || encode_animation(&frames, &opts).expect("encode"));
                match &reference {
                    None => reference = Some(file),
                    Some(first) => assert!(*first == file, "thread budget changed the output"),
                }
                print!(" {} |", ms(d));
            }
            let file = reference.unwrap_or_default();
            assert_eq!(decode_all(&file).expect("decode").len(), frames.len());
            println!(" {} |", file.len());
        }
    }
}

fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        // Smoke pass for `cargo test --benches`.
        let (w, h) = (48, 32);
        let rgb = photo_rgb(w, h);
        for opts in [
            EncodeOptions::default(),
            EncodeOptions::default().with_quality(80.0),
        ] {
            let file = encode_rgb8(w, h, &rgb, &opts).expect("encode");
            let image = decode_rgba8(&file).expect("decode");
            assert_eq!((image.width, image.height), (w, h));
        }
        return;
    }
    let sizes: Vec<(u32, u32)> = std::env::var("STILL_MATRIX_SIZES")
        .unwrap_or_else(|_| "800x533,2400x1600".into())
        .split(',')
        .filter_map(|s| {
            let (w, h) = s.trim().split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .collect();
    let runs: usize = std::env::var("STILL_MATRIX_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let sections = std::env::var("STILL_MATRIX_SECTIONS")
        .unwrap_or_else(|_| "lossy,alpha,lossless,decode,animation".into());
    let wants = |name: &str| sections.split(',').any(|s| s.trim() == name);
    if wants("lossy") {
        encode_section(
            "Lossy encode, quality 80 (`encode_rgb8`)",
            &EncodeOptions::default().with_quality(80.0),
            &[0, 1, 2, 4, 5, 6],
            &[1, 4, 8],
            &sizes,
            runs,
        );
    }
    if wants("alpha") {
        encode_section_with(
            "Lossy encode with an alpha plane, quality 80 (`encode_rgba8`)",
            &EncodeOptions::default().with_quality(80.0),
            &[0, 2, 4],
            &[1, 4, 8],
            &sizes,
            runs,
            true,
        );
    }
    if wants("lossless") {
        encode_section(
            "Lossless encode (`encode_rgb8`)",
            &EncodeOptions::default(),
            &[0, 2, 4, 6],
            &[1, 4, 8],
            &sizes,
            runs,
        );
    }
    if wants("decode") {
        decode_section(&sizes, runs);
    }
    if wants("animation") {
        animation_section(&sizes, &[1, 4, 8], runs.min(3));
    }
}
