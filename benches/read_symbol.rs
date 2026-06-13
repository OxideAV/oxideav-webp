//! Criterion bench — decoder-side §6.2.1 `PrefixCode::read_symbol`, the
//! per-symbol canonical-prefix-code reader that is the single largest
//! consumer of decode self-time in the crate (~82% post-round-284,
//! down from ~89% pre-table; see `BENCHMARKS.md` round-283 / round-284).
//!
//! `read_symbol` is invoked at least four times per decoded pixel in a
//! §5.2 single-group ARGB decode (GREEN + length, RED, BLUE, ALPHA,
//! plus a DISTANCE read on every backward reference), so its
//! per-call cost dominates entropy-bound streams. The round-284
//! primary lookup table turned codes of length ≤ `LOOKUP_BITS` (8) into
//! one peek + one table load + one cursor advance, but codes **longer
//! than 8 bits** still continue through the per-bit row walk from
//! length 9 — the round-284 post-change profile flagged "widen the fast
//! path's coverage of 9–11-bit codes" (a second-level spill table) as
//! the next ranked optimization candidate. The two existing prefix-code
//! benches measure the *table build*
//! (`prefix_from_code_lengths`) and the §3.6.2.2 *value expansion*
//! (`read_lz77_value`); the symbol-read hot loop itself — the actual
//! rank-1 decode symbol — had **no isolated harness**. This bench fills
//! that gap and gives the spill-table candidate direct A/B numbers at
//! every regime it would touch.
//!
//! The cells deliberately separate the two paths through the function:
//!
//! * **`short8` (uniform 8-bit)** — 256 symbols all at code length 8
//!   (Kraft-exact: 256·2⁻⁸ = 1). Every read resolves in the round-284
//!   primary table in one load: the pure fast-path lower bound. A
//!   spill-table change must leave this cell unchanged.
//! * **`short6` (uniform 6-bit, table built)** — 64 symbols all at code
//!   length 6 (64·2⁻⁶ = 1), `used = 64` above the `MIN_LOOKUP_USED = 32`
//!   gate so the table still builds. Short codes packed densely; the
//!   fast path at its cheapest per decoded symbol.
//! * **`long9_11` (9/10/11-bit only)** — 384 codes at length 9, 128 at
//!   length 10, 256 at length 11 (Kraft-exact: 384·2⁻⁹ + 128·2⁻¹⁰ +
//!   256·2⁻¹¹ = 1). **Every** read overshoots the 8-bit table and
//!   continues the per-bit walk from length 9 — this is the exact path
//!   the round-284 follow-up targets, isolated. This cell is the
//!   primary A/B reference for the second-level-spill-table candidate.
//! * **`dense256` (realistic literal channel)** — the 256-symbol 8-bit
//!   literal length table the encoder's `build_code_lengths` produces on
//!   the standard dense LCG frequency input (same constants as
//!   `prefix_from_code_lengths` / `canonical_codes`): a natural mix of
//!   lengths 7..13 peaking at 8, so most reads hit the table and a long
//!   tail spills — the blended cost a real RED/BLUE/ALPHA channel pays.
//! * **`belowgate` (no table)** — 16 symbols at length 4, `used = 16`
//!   below the `MIN_LOOKUP_USED = 32` gate, so no primary table is
//!   built and every read runs the pre-table per-bit `read_symbol_walk`.
//!   The walk-only baseline that animation delta frames / tiny
//!   transform images exercise; a spill-table change must leave it
//!   unchanged (the gate fences it off).
//!
//! ## Construction (public API only)
//!
//! Each cell builds a Kraft-complete per-symbol length table, derives
//! the canonical decoder via `PrefixCode::from_code_lengths`, and
//! derives the matching canonical *code values* via the encoder's
//! public `canonical_codes`. A deterministic LCG symbol stream is then
//! packed MSB-first-within-each-code into a `BitWriter` (the decoder
//! reads MSB-first within a code, so the high bit is laid down first;
//! `BitWriter` packs LSB-first across the stream, the exact inverse of
//! `BitReader`). The packed bytes are reconstructed into a fresh
//! `BitReader` inside the `iter_batched` setup so the measured interval
//! is `SYMS_PER_ITER` back-to-back `read_symbol` calls and nothing
//! else. No fixture files, no committed bytes — every input is built on
//! the fly from the same LCG constants the rest of the bench inventory
//! uses, so the cells are reproducible and cross-host comparable.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/oxideav-webp-bench-target \
//!   cargo bench -p oxideav-webp --bench read_symbol -- --quick
//! ```

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use oxideav_webp::vp8l_encode::{build_code_lengths, canonical_codes, BitWriter};
use oxideav_webp::vp8l_prefix::PrefixCode;
use oxideav_webp::vp8l_stream::BitReader;

/// Symbols read per measured iteration. Large enough to amortise the
/// per-iteration `BitReader` reconstruction (a pointer plus two
/// counters) to noise, small enough that the packed byte buffer stays
/// comfortably in L1/L2 for every cell.
const SYMS_PER_ITER: usize = 4096;

/// Deterministic LCG matching the constants used across the §3.x / §4.x
/// per-pass benches so the symbol stream here is reproducible across
/// runs and hosts and cross-bench comparable to the rest of the
/// inventory.
fn lcg_step(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    *seed
}

/// Dense frequency table over `alphabet` (every symbol used, frequency
/// `1..=255`), identical to the dense path in
/// `benches/build_code_lengths.rs` / `prefix_from_code_lengths.rs`.
fn build_freqs_dense(alphabet: usize) -> Vec<u32> {
    let mut seed: u32 = 0x1357_9bdf;
    let mut freqs = Vec::with_capacity(alphabet);
    for _ in 0..alphabet {
        freqs.push((lcg_step(&mut seed) % 255) + 1);
    }
    freqs
}

/// A Kraft-complete length table with every symbol at code length 8
/// (256 symbols ⇒ 256·2⁻⁸ = 1). All reads resolve in the round-284
/// primary table.
fn lengths_uniform8() -> Vec<u8> {
    vec![8u8; 256]
}

/// A length table where every used code is 9, 10, or 11 bits — the
/// long-code (> `LOOKUP_BITS`) continuation path the round-284 follow-up
/// targets. Kraft-exact: 384·2⁻⁹ + 128·2⁻¹⁰ + 256·2⁻¹¹ =
/// 0.75 + 0.125 + 0.125 = 1.
fn lengths_long9_11() -> Vec<u8> {
    let mut l = Vec::with_capacity(768);
    l.extend(std::iter::repeat(9u8).take(384));
    l.extend(std::iter::repeat(10u8).take(128));
    l.extend(std::iter::repeat(11u8).take(256));
    l
}

/// A short-code table above the `MIN_LOOKUP_USED = 32` table-build gate:
/// 64 symbols at code length 6 (64·2⁻⁶ = 1). Shortest realistic codes
/// with the primary table active — the fast path at its cheapest.
fn lengths_short6() -> Vec<u8> {
    vec![6u8; 64]
}

/// The realistic dense 8-bit literal length table the encoder produces
/// on the standard dense LCG frequency input — a mix of lengths 7..13
/// peaking at 8 (most reads table-hit, a tail spills to the walk).
fn lengths_dense256() -> Vec<u8> {
    build_code_lengths(&build_freqs_dense(256))
}

/// A length table below the `MIN_LOOKUP_USED = 32` gate: 16 symbols at
/// code length 4 (16·2⁻⁴ = 1, `used = 16`). No primary table is built;
/// every read runs the pre-table per-bit walk.
fn lengths_belowgate() -> Vec<u8> {
    vec![4u8; 16]
}

/// A maximally length-diverse table: one symbol at each of lengths
/// `1..=14` plus two at length 15 (Kraft-exact: `Σ 2⁻ⁱ` for `i∈1..=14`
/// `+ 2·2⁻¹⁵ = (1 − 2⁻¹⁴) + 2⁻¹⁴ = 1`). `used = 16` sits below the
/// `MIN_LOOKUP_USED = 32` gate, so every read runs the pure per-bit
/// walk — and because the 16 used symbols span all 15 distinct lengths,
/// the walk's per-bit "is there a row at this length?" test is the case
/// where the pre-`len_to_row` linear `length_rows.iter().find(..)` scan
/// was at its most expensive (up to 15 rows scanned at each consumed
/// bit, with codes running all the way to the 15-bit ceiling). This is
/// the cell the round-287 direct length→row side table targets: an
/// adversarial many-distinct-length code, the regime the uniform-length
/// `belowgate` cell can't exercise.
fn lengths_manylen() -> Vec<u8> {
    let mut l = vec![0u8; 16];
    for (i, slot) in l.iter_mut().take(14).enumerate() {
        *slot = (i + 1) as u8; // lengths 1..=14
    }
    l[14] = 15;
    l[15] = 15;
    l
}

/// Pack a deterministic LCG stream of `SYMS_PER_ITER` symbols, drawn
/// uniformly over the *used* symbols of `lengths`, into a byte buffer
/// using the canonical code values — MSB-first within each code (the
/// order `read_symbol` reads), LSB-first across the stream (the
/// `BitReader`/`BitWriter` contract).
fn pack_stream(lengths: &[u8]) -> Vec<u8> {
    let codes = canonical_codes(lengths);
    let used: Vec<usize> = lengths
        .iter()
        .enumerate()
        .filter(|(_, &l)| l > 0)
        .map(|(s, _)| s)
        .collect();
    debug_assert!(!used.is_empty(), "length table has no used symbols");

    let mut seed: u32 = 0xdead_beef;
    let mut w = BitWriter::new();
    for _ in 0..SYMS_PER_ITER {
        let sym = used[(lcg_step(&mut seed) as usize) % used.len()];
        let len = lengths[sym] as usize;
        let code = codes[sym];
        for i in 0..len {
            w.write_bits((code >> (len - 1 - i)) & 1, 1);
        }
    }
    w.into_bytes()
}

/// Bench `SYMS_PER_ITER` back-to-back `read_symbol` calls over a packed
/// stream of `lengths`. The `PrefixCode` and the packed bytes are built
/// once outside `b.iter`; only the cheap `BitReader` reconstruction
/// lives in the `iter_batched` setup, outside the measured interval.
fn bench_cell(c: &mut Criterion, lengths: Vec<u8>, label: &str) {
    let bytes = pack_stream(&lengths);
    let code =
        PrefixCode::from_code_lengths(lengths).expect("hand-built tables are Kraft-complete");
    let name = format!("read_symbol_{}", label);
    c.bench_function(&name, |b| {
        b.iter_batched(
            || BitReader::new(&bytes),
            |mut reader| {
                for _ in 0..SYMS_PER_ITER {
                    let sym = code
                        .read_symbol(&mut reader)
                        .expect("packed stream decodes within bounds");
                    black_box(sym);
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_short8(c: &mut Criterion) {
    bench_cell(c, lengths_uniform8(), "short8_uniform");
}

fn bench_short6(c: &mut Criterion) {
    bench_cell(c, lengths_short6(), "short6_uniform");
}

fn bench_long9_11(c: &mut Criterion) {
    bench_cell(c, lengths_long9_11(), "long9_11");
}

fn bench_dense256(c: &mut Criterion) {
    bench_cell(c, lengths_dense256(), "dense256");
}

fn bench_belowgate(c: &mut Criterion) {
    bench_cell(c, lengths_belowgate(), "belowgate16_walk");
}

fn bench_manylen(c: &mut Criterion) {
    bench_cell(c, lengths_manylen(), "manylen16_walk");
}

criterion_group!(
    benches,
    bench_short8,
    bench_short6,
    bench_long9_11,
    bench_dense256,
    bench_belowgate,
    bench_manylen,
);
criterion_main!(benches);
