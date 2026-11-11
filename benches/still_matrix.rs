//! Timing matrix: size x effort x threads -> milliseconds and bytes,
//! for lossy and lossless still encode and decode through the public
//! entry points (`encode_rgb8`, `encode_rgba8`, `decode_rgba8`), plus an
//! eight-frame animation (`encode_animation`).
//!
//! A plain `main` (no statistics harness) that prints the README's
//! Markdown tables:
//!
//! ```text
//! cargo bench --bench still_matrix
//! ```
//!
//! Every cell is timed once per *pass* over the whole matrix, so a
//! cell's samples are spread over the run rather than taken back to
//! back; the figure printed is their median. A cell that takes two
//! seconds or more is sampled on the first two passes only.
//!
//! Environment:
//!
//! * `STILL_MATRIX_SIZES` — comma-separated `WxH` list (default
//!   `800x533,2400x1600`).
//! * `STILL_MATRIX_RUNS` — passes over the matrix (default 5).
//! * `STILL_MATRIX_THREADS` — thread budgets, one column each (default
//!   `1,4,8`).
//! * `STILL_MATRIX_SECTIONS` — any of `lossy`, `alpha`, `lossless`,
//!   `decode`, `animation` (default all five).
//! * `STILL_MATRIX_HASH` — when set, each row ends with the 64-bit
//!   FNV-1a of the file, to compare outputs between builds.
//! * `STILL_MATRIX_STAT` — `median` (default), `min`, or `both`
//!   (`median (min)`): which of a cell's samples is printed. The
//!   minimum is the steadier figure on a machine that is also doing
//!   something else.
//!
//! Without `--bench` on the command line (what `cargo test --benches`
//! does) it runs one tiny smoke pass instead of the matrix.

use std::rc::Rc;
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

/// 64-bit FNV-1a.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// One timed operation of the matrix. `run` performs it once and
/// returns the size and hash of what it produced.
struct Cell {
    run: Box<dyn FnMut() -> (usize, u64)>,
    samples: Vec<Duration>,
    output: (usize, u64),
}

impl Cell {
    fn new(run: impl FnMut() -> (usize, u64) + 'static) -> Self {
        Self {
            run: Box::new(run),
            samples: Vec::new(),
            output: (0, 0),
        }
    }

    /// Take one more sample, unless this cell has shown itself slow and
    /// has two already.
    fn sample(&mut self) {
        if self.samples.len() >= 2 && self.samples[0] >= Duration::from_secs(2) {
            return;
        }
        let start = Instant::now();
        self.output = (self.run)();
        self.samples.push(start.elapsed());
    }

    /// The cell's figure as table text: the statistic
    /// `STILL_MATRIX_STAT` selects, in milliseconds.
    fn figure(&self) -> String {
        fn fmt(d: Duration) -> String {
            let v = d.as_secs_f64() * 1e3;
            if v >= 100.0 {
                format!("{v:.0}")
            } else if v >= 10.0 {
                format!("{v:.1}")
            } else {
                format!("{v:.2}")
            }
        }
        let mut sorted = self.samples.clone();
        sorted.sort();
        let (median, min) = (sorted[sorted.len() / 2], sorted[0]);
        match std::env::var("STILL_MATRIX_STAT").as_deref() {
            Ok("min") => fmt(min),
            Ok("both") => format!("{} ({})", fmt(median), fmt(min)),
            _ => fmt(median),
        }
    }
}

/// One table row: its leading labels and one cell per timing column.
struct Row {
    labels: Vec<String>,
    cells: Vec<Cell>,
}

/// One table of the matrix.
struct Table {
    title: String,
    label_heads: Vec<&'static str>,
    column_heads: Vec<String>,
    /// Rows end with the produced size (and, on request, hash); all the
    /// cells of a row must then produce the same bytes.
    output_head: Option<&'static str>,
    rows: Vec<Row>,
}

impl Table {
    fn print(&self) {
        println!("\n### {}\n", self.title);
        let mut head: Vec<String> = self.label_heads.iter().map(|h| h.to_string()).collect();
        head.extend(self.column_heads.iter().cloned());
        head.extend(self.output_head.map(str::to_string));
        println!("| {} |", head.join(" | "));
        let mut rule = vec!["---"; self.label_heads.len()];
        rule.extend(vec!["---:"; head.len() - self.label_heads.len()]);
        println!("|{}|", rule.join("|"));
        for row in &self.rows {
            let mut fields = row.labels.clone();
            fields.extend(row.cells.iter().map(Cell::figure));
            if self.output_head.is_some() {
                let (bytes, hash) = row.cells[0].output;
                for cell in &row.cells {
                    assert_eq!(
                        cell.output,
                        (bytes, hash),
                        "thread budget changed the output"
                    );
                }
                fields.push(bytes.to_string());
                if std::env::var_os("STILL_MATRIX_HASH").is_some() {
                    fields.push(format!("{hash:016x}"));
                }
            }
            println!("| {} |", fields.join(" | "));
        }
    }
}

fn thread_heads(threads: &[usize]) -> Vec<String> {
    threads
        .iter()
        .map(|t| format!("{t} thread{} ms", if *t == 1 { "" } else { "s" }))
        .collect()
}

/// A still-encode table: a row per size and `(effort label, options)`,
/// a column per thread budget. With `alpha` the input is RGBA with an
/// alpha plane, else RGB.
fn encode_table(
    title: &str,
    rows: &[(String, EncodeOptions)],
    threads: &[usize],
    sizes: &[(u32, u32)],
    alpha: bool,
) -> Table {
    let mut table = Table {
        title: title.to_string(),
        label_heads: vec!["size", "effort"],
        column_heads: thread_heads(threads),
        output_head: Some("bytes"),
        rows: Vec::new(),
    };
    for &(w, h) in sizes {
        let pixels = Rc::new(if alpha {
            photo_rgba_with_alpha(w, h)
        } else {
            photo_rgb(w, h)
        });
        for (label, options) in rows {
            let cells = threads
                .iter()
                .map(|&t| {
                    let (pixels, opts) = (Rc::clone(&pixels), options.clone().with_threads(t));
                    Cell::new(move || {
                        let file = if alpha {
                            encode_rgba8(w, h, &pixels, &opts)
                        } else {
                            encode_rgb8(w, h, &pixels, &opts)
                        }
                        .expect("encode");
                        (file.len(), fnv1a64(&file))
                    })
                })
                .collect();
            table.rows.push(Row {
                labels: vec![format!("{w}x{h}"), label.clone()],
                cells,
            });
        }
    }
    table
}

fn decode_table(sizes: &[(u32, u32)]) -> Table {
    let mut table = Table {
        title: "Decode to RGBA (`decode_rgba8`)".to_string(),
        label_heads: vec!["size", "file"],
        column_heads: vec!["ms".to_string()],
        output_head: None,
        rows: Vec::new(),
    };
    for &(w, h) in sizes {
        let rgb = photo_rgb(w, h);
        let rgba = photo_rgba_with_alpha(w, h);
        let lossy = EncodeOptions::default().with_quality(80.0);
        let files = [
            (
                "lossy, quality 80",
                encode_rgb8(w, h, &rgb, &lossy).expect("lossy encode"),
            ),
            (
                "lossy with alpha, quality 80",
                encode_rgba8(w, h, &rgba, &lossy.clone().with_method(2)).expect("lossy encode"),
            ),
            (
                "lossless",
                encode_rgb8(w, h, &rgb, &EncodeOptions::default().with_method(2))
                    .expect("lossless encode"),
            ),
        ];
        for (name, file) in files {
            let label = format!("{name} ({} bytes)", file.len());
            let cell = Cell::new(move || {
                let image = decode_rgba8(&file).expect("decode");
                assert_eq!((image.width, image.height), (w, h));
                (image.data.len(), 0)
            });
            table.rows.push(Row {
                labels: vec![format!("{w}x{h}"), label],
                cells: vec![cell],
            });
        }
    }
    table
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

fn animation_table(
    rows: &[(String, EncodeOptions)],
    threads: &[usize],
    sizes: &[(u32, u32)],
) -> Table {
    let mut table = Table {
        title: "Animation encode, 8 frames (`encode_animation`)".to_string(),
        label_heads: vec!["size", "frames"],
        column_heads: thread_heads(threads),
        output_head: Some("bytes"),
        rows: Vec::new(),
    };
    for &(w, h) in sizes {
        let frames = Rc::new(animation_frames(w, h));
        for (label, options) in rows {
            let cells = threads
                .iter()
                .map(|&t| {
                    let (frames, opts) = (Rc::clone(&frames), options.clone().with_threads(t));
                    Cell::new(move || {
                        let file = encode_animation(&frames, &opts).expect("encode");
                        (file.len(), fnv1a64(&file))
                    })
                })
                .collect();
            table.rows.push(Row {
                labels: vec![format!("{w}x{h}"), label.clone()],
                cells,
            });
        }
    }
    table
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
        let frames = animation_frames(w, h);
        let file = encode_animation(&frames, &EncodeOptions::default()).expect("encode");
        assert_eq!(decode_all(&file).expect("decode").len(), frames.len());
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
    let passes: usize = std::env::var("STILL_MATRIX_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let threads: Vec<usize> = std::env::var("STILL_MATRIX_THREADS")
        .unwrap_or_else(|_| "1,4,8".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let sections = std::env::var("STILL_MATRIX_SECTIONS")
        .unwrap_or_else(|_| "lossy,alpha,lossless,decode,animation".into());
    let wants = |name: &str| sections.split(',').any(|s| s.trim() == name);
    let efforts = |base: &EncodeOptions, levels: &[u8]| -> Vec<(String, EncodeOptions)> {
        levels
            .iter()
            .map(|&m| (m.to_string(), base.clone().with_method(m)))
            .collect()
    };
    let lossy = EncodeOptions::default().with_quality(80.0);
    let lossless = EncodeOptions::default();

    let mut tables = Vec::new();
    if wants("lossy") {
        tables.push(encode_table(
            "Lossy encode, quality 80 (`encode_rgb8`)",
            &efforts(&lossy, &[0, 1, 2, 4, 5, 6]),
            &threads,
            &sizes,
            false,
        ));
    }
    if wants("alpha") {
        tables.push(encode_table(
            "Lossy encode with an alpha plane, quality 80 (`encode_rgba8`)",
            &alpha_rows(&lossy),
            &threads,
            &sizes,
            true,
        ));
    }
    if wants("lossless") {
        tables.push(encode_table(
            "Lossless encode (`encode_rgb8`)",
            &efforts(&lossless, &[0, 2, 4, 6]),
            &threads,
            &sizes,
            false,
        ));
    }
    if wants("decode") {
        tables.push(decode_table(&sizes));
    }
    if wants("animation") {
        tables.push(animation_table(
            &[
                ("lossless, effort 2".into(), lossless.clone().with_method(2)),
                ("lossy, quality 80, effort 4".into(), lossy.clone()),
                (
                    "lossy, quality 80, effort 1".into(),
                    lossy.clone().with_method(1),
                ),
            ],
            &threads,
            &sizes,
        ));
    }
    for _ in 0..passes.max(1) {
        for table in &mut tables {
            for row in &mut table.rows {
                for cell in &mut row.cells {
                    cell.sample();
                }
            }
        }
    }
    for table in &tables {
        table.print();
    }
}

/// The rows of the alpha table: the picture's effort, and the default
/// picture with the quick alpha coder.
fn alpha_rows(lossy: &EncodeOptions) -> Vec<(String, EncodeOptions)> {
    vec![
        ("0".into(), lossy.clone().with_method(0)),
        ("2".into(), lossy.clone().with_method(2)),
        ("4, alpha 2".into(), lossy.clone().with_alpha_method(2)),
        ("4".into(), lossy.clone()),
    ]
}
