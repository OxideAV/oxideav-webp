//! RFC 9649 §4.1 predictor transform on the rightmost column (#33).
//!
//! "The pixels on the rightmost column are predicted by using the modes
//! [0..13], just like pixels not on the border, but the leftmost pixel on
//! the same row as the current pixel is instead used as the TR-pixel."
//!
//! Each test stream is assembled bit by bit from the RFC: a 17 x 121
//! `VP8L` image (4-pixel predictor blocks, so the last block column is
//! the single rightmost pixel column) whose predictor image names one
//! mode everywhere, and whose residuals are random picks between two
//! values per channel (two-symbol simple prefix codes, so no code-bit
//! order question arises). The expected pixels come from a reference
//! model written here from §4.1's text, independent of the crate's
//! decoder. Modes 3, 5, 9 and 10 read TR; every mode is checked.
//!
//! When `dwebp` is on `PATH` each stream also goes through it and must
//! decode to the same pixels.

use std::path::PathBuf;
use std::process::Command;

const W: usize = 17;
const H: usize = 121;
const SIZE_BITS: u32 = 2;

/// LSB-first bit writer (RFC 9649 §3.2 `ReadBits` order).
#[derive(Default)]
struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn put(&mut self, value: u32, bits: u32) {
        self.acc |= u64::from(value) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// §3.7.2.1.1 simple code with one symbol (`symbol < 256`).
fn simple_one(b: &mut Bits, symbol: u32) {
    b.put(1, 1); // simple code length code
    b.put(0, 1); // num_symbols - 1
    if symbol < 2 {
        b.put(0, 1);
        b.put(symbol, 1);
    } else {
        b.put(1, 1);
        b.put(symbol, 8);
    }
}

/// §3.7.2.1.1 simple code with two 8-bit symbols, both of length 1:
/// the smaller one is code `0`.
fn simple_two(b: &mut Bits, lo: u32, hi: u32) {
    assert!(lo < hi);
    b.put(1, 1);
    b.put(1, 1);
    b.put(1, 1); // is_first_8bits
    b.put(lo, 8);
    b.put(hi, 8);
}

/// Residual channel alphabets `[lo, hi]` for green, red, blue.
const PAIRS: [[u32; 2]; 3] = [[0x00, 0x5b], [0x07, 0xc2], [0x31, 0x90]];

/// A `RIFF`/`WEBP` file with one `VP8L` chunk: predictor transform with
/// every block on `mode`, residuals chosen by `picks` (3 bits a pixel:
/// green, red, blue), alpha residual 0.
fn crafted_file(mode: u32, picks: &[u8]) -> Vec<u8> {
    let mut b = Bits::default();
    // §3.4 header after the 0x2f signature.
    b.put((W - 1) as u32, 14);
    b.put((H - 1) as u32, 14);
    b.put(0, 1); // alpha_is_used
    b.put(0, 3); // version
    // §4 transform: predictor, size_bits.
    b.put(1, 1);
    b.put(0, 2);
    b.put(SIZE_BITS - 2, 3);
    // Predictor sub-image (entropy-coded image): no colour cache, five
    // one-symbol codes, so every pixel is green = mode, zero bits each.
    b.put(0, 1);
    simple_one(&mut b, mode);
    for _ in 0..4 {
        simple_one(&mut b, 0);
    }
    b.put(0, 1); // no further transform
    // Main ARGB image: no colour cache, one prefix code group.
    b.put(0, 1);
    b.put(0, 1);
    for [lo, hi] in PAIRS {
        simple_two(&mut b, lo, hi);
    }
    simple_one(&mut b, 0); // alpha
    simple_one(&mut b, 0); // distance
    for &p in picks {
        b.put(u32::from(p & 1), 1); // green
        b.put(u32::from((p >> 1) & 1), 1); // red
        b.put(u32::from((p >> 2) & 1), 1); // blue
    }
    let mut vp8l = vec![0x2f];
    vp8l.extend(b.finish());
    let pad = vp8l.len() & 1;
    let mut file = b"RIFF".to_vec();
    file.extend(((4 + 8 + vp8l.len() + pad) as u32).to_le_bytes());
    file.extend(b"WEBPVP8L");
    file.extend((vp8l.len() as u32).to_le_bytes());
    file.extend(&vp8l);
    file.resize(file.len() + pad, 0);
    file
}

/// Residual ARGB word of a pick.
fn residual(p: u8) -> u32 {
    let g = PAIRS[0][usize::from(p & 1)];
    let r = PAIRS[1][usize::from((p >> 1) & 1)];
    let bl = PAIRS[2][usize::from((p >> 2) & 1)];
    (r << 16) | (g << 8) | bl
}

fn ch(v: u32, c: u32) -> i32 {
    ((v >> (8 * c)) & 0xff) as i32
}

fn per_channel(f: impl Fn(i32, i32) -> i32, a: u32, b: u32) -> u32 {
    (0..4).fold(0, |acc, c| {
        acc | (((f(ch(a, c), ch(b, c)) as u32) & 0xff) << (8 * c))
    })
}

fn avg2(a: u32, b: u32) -> u32 {
    per_channel(|x, y| (x + y) / 2, a, b)
}

fn clamp(v: i32) -> i32 {
    v.clamp(0, 255)
}

/// §4.1 Table 2, from the RFC text.
fn predict(mode: u32, l: u32, t: u32, tr: u32, tl: u32) -> u32 {
    match mode {
        0 => 0xff00_0000,
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => avg2(avg2(l, tr), t),
        6 => avg2(l, tl),
        7 => avg2(l, t),
        8 => avg2(tl, t),
        9 => avg2(t, tr),
        10 => avg2(avg2(l, tl), avg2(t, tr)),
        11 => {
            let p = |c| ch(l, c) + ch(t, c) - ch(tl, c);
            let pl: i32 = (0..4).map(|c| (p(c) - ch(l, c)).abs()).sum();
            let pt: i32 = (0..4).map(|c| (p(c) - ch(t, c)).abs()).sum();
            if pl < pt { l } else { t }
        }
        12 => (0..4).fold(0, |acc, c| {
            acc | ((clamp(ch(l, c) + ch(t, c) - ch(tl, c)) as u32) << (8 * c))
        }),
        13 => {
            let a = avg2(l, t);
            (0..4).fold(0, |acc, c| {
                acc | ((clamp(ch(a, c) + (ch(a, c) - ch(tl, c)) / 2) as u32) << (8 * c))
            })
        }
        _ => unreachable!(),
    }
}

/// §4.1 inverse predictor over the residuals, RFC border rules included.
fn reference_decode(mode: u32, picks: &[u8]) -> Vec<u32> {
    let mut px = vec![0u32; W * H];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let pred = if x == 0 && y == 0 {
                0xff00_0000
            } else if y == 0 {
                px[i - 1]
            } else if x == 0 {
                px[i - W]
            } else {
                let tr = if x == W - 1 {
                    px[y * W] // leftmost pixel of the same row
                } else {
                    px[i - W + 1]
                };
                predict(mode, px[i - 1], px[i - W], tr, px[i - W - 1])
            };
            px[i] = per_channel(|a, b| a + b, pred, residual(picks[i]));
        }
    }
    px
}

fn to_rgba(px: &[u32]) -> Vec<u8> {
    px.iter()
        .flat_map(|&v| [(v >> 16) as u8, (v >> 8) as u8, v as u8, (v >> 24) as u8])
        .collect()
}

fn picks(seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..W * H)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 29) as u8
        })
        .collect()
}

#[test]
fn every_mode_decodes_like_the_rfc_on_the_rightmost_column() {
    for mode in 0..14u32 {
        let picks = picks(0x51ed_0000 + mode);
        let expected = to_rgba(&reference_decode(mode, &picks));
        let file = crafted_file(mode, &picks);
        let got = oxideav_webp::decode_rgba8(&file).expect("decode crafted stream");
        assert_eq!((got.width, got.height), (W as u32, H as u32));
        let bad: Vec<usize> = (0..W * H)
            .filter(|&i| got.data[i * 4..i * 4 + 4] != expected[i * 4..i * 4 + 4])
            .collect();
        assert!(
            bad.is_empty(),
            "mode {mode}: {} pixels differ from the §4.1 model (first at x={}, y={})",
            bad.len(),
            bad[0] % W,
            bad[0] / W
        );
    }
}

/// The TR rule matters for these streams: reading the row above instead
/// changes the rightmost column for every TR mode.
#[test]
fn the_crafted_streams_tell_the_two_tr_readings_apart() {
    for mode in [3u32, 5, 9, 10] {
        let picks = picks(0x51ed_0000 + mode);
        let rfc = reference_decode(mode, &picks);
        // Same walk, TR on the rightmost column read from the row above.
        let mut alt = vec![0u32; W * H];
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                let pred = if x == 0 && y == 0 {
                    0xff00_0000
                } else if y == 0 {
                    alt[i - 1]
                } else if x == 0 {
                    alt[i - W]
                } else {
                    let tr = if x == W - 1 {
                        alt[(y - 1) * W]
                    } else {
                        alt[i - W + 1]
                    };
                    predict(mode, alt[i - 1], alt[i - W], tr, alt[i - W - 1])
                };
                alt[i] = per_channel(|a, b| a + b, pred, residual(picks[i]));
            }
        }
        let differing = (0..W * H).filter(|&i| rfc[i] != alt[i]).count();
        assert!(differing >= H / 2, "mode {mode}: only {differing} differ");
    }
}

fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(bin))
        .find(|p| p.is_file())
}

#[test]
fn every_mode_decodes_like_the_reference_decoder() {
    let Some(dwebp) = which("dwebp") else {
        eprintln!("skip: `dwebp` not on PATH");
        return;
    };
    let dir = std::env::temp_dir().join(format!("oxideav-webp-tr-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    for mode in 0..14u32 {
        let picks = picks(0x51ed_0000 + mode);
        let file = crafted_file(mode, &picks);
        let src = dir.join(format!("m{mode}.webp"));
        let out = dir.join(format!("m{mode}.pam"));
        std::fs::write(&src, &file).expect("write stream");
        let status = Command::new(&dwebp)
            .arg(&src)
            .arg("-pam")
            .arg("-o")
            .arg(&out)
            .arg("-quiet")
            .status()
            .expect("run dwebp");
        assert!(status.success(), "mode {mode}: dwebp rejected the stream");
        let pam = std::fs::read(&out).expect("read PAM");
        let body = pam
            .windows(7)
            .position(|w| w == b"ENDHDR\n")
            .expect("PAM header")
            + 7;
        let ours = oxideav_webp::decode_rgba8(&file).expect("decode");
        assert!(
            pam[body..] == ours.data[..],
            "mode {mode}: reference decoder and this crate disagree"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
