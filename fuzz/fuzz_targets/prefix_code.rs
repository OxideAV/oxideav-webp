#![no_main]

//! Read arbitrary fuzz-supplied bits through the §6.2.1 *single canonical
//! prefix-code* reader standalone entry point
//! `oxideav_webp::vp8l_prefix::PrefixCode::read`.
//!
//! A §6.2.1 prefix code is one of the five canonical codes a §6.2
//! prefix-code group is built from (green / red / blue / alpha /
//! distance). `PrefixCode::read` reads one code's *code lengths* off the
//! wire — dispatching on the leading 1-bit flag between the §6.2.1
//! "Simple Code Length Code" (1 or 2 symbols, each at length 1) and the
//! §6.2.1 "Normal Code Length Code" (the lengths are themselves
//! entropy-coded with the 19-symbol code-length-code, read in the fixed
//! `kCodeLengthCodeOrder`, gated by `max_symbol`, and expanded through
//! the 16/17/18 repeat / zero-run codes) — and then builds the canonical
//! decoder via `PrefixCode::from_code_lengths` with its §6.2.1 Kraft
//! completeness gate and single-leaf-node exception.
//!
//! This twenty-fourth harness drives the §6.2.1 single-code surface
//! directly — the layer immediately *below* the round-274
//! `prefix_code_group` harness (§6.2). `PrefixCodeGroup::read` calls
//! `PrefixCode::read` exactly five times in green/red/blue/alpha/distance
//! order; this harness isolates one such call across an attacker-controlled
//! `(alphabet_size, bitstream)` cross-product. Sibling harnesses surround
//! this primitive but none drives it standalone:
//!
//! * `prefix_code_group` (round 274) drives all five reads as a unit, so
//!   a single code's parse outcome is entangled with the four that
//!   precede / follow it in the §6.2 group order and is only ever
//!   exercised at the fixed `{256, 40, green}` alphabet sizes.
//! * `parse_meta_prefix` (round 261) reaches the group — and thus this
//!   code — only after the §5.2.3 + §6.2.2 preamble, never standalone and
//!   never across a free alphabet-size sweep.
//!
//! The contract under test, per RFC 9649 §6.2.1 + §6.2.3:
//!
//! * `PrefixCode::read` always returns a `Result` — no panic, no
//!   debug-build integer overflow, no out-of-bounds index when the
//!   bitstream is empty, truncated, or arbitrarily long, and no
//!   allocation sized by a header field the §6.2.1 readers did not
//!   bound. Every read goes through
//!   [`oxideav_webp::vp8l_stream::BitReader`] whose EOF path raises a
//!   `PrefixError::Eof`, never an underflow panic.
//! * If the call returns `Ok(code)`:
//!     * §6.2.3 alphabet sizing: the per-symbol length table is exactly
//!       `alphabet_size` entries long (lengths for absent symbols are 0).
//!     * §6.2.1 the code-length ceiling: every nonzero length is `<= 15`
//!       (the `MAX_CODE_LENGTH` ceiling).
//!     * §6.2.1 single-leaf-node consistency: a code whose
//!       `single_symbol()` is `Some(s)` has exactly one nonzero entry in
//!       its length table (at symbol `s`), and a code whose
//!       `single_symbol()` is `None` has two or more.
//!     * §6.2.1 every successfully-built code is decodable: reading a
//!       symbol from a fresh all-zero reader yields a valid in-range
//!       symbol index (single-leaf node consumes no bits; a multi-symbol
//!       code follows the all-zero path). The symbol index must be
//!       `< alphabet_size`.
//!     * §6.2.1 the Kraft completeness gate: a multi-symbol `Ok` code is
//!       complete — rebuilding it from its own returned length table
//!       through `PrefixCode::from_code_lengths` succeeds and reproduces
//!       an equal code (the §6.2.1 `sum 2^-len == 1` invariant a built
//!       code must satisfy).
//!     * `BitReader` clamps every read at the slice end, so a successful
//!       read never advances past the slice's bit length.
//!     * Determinism: replaying the same bytes + `alphabet_size` yields
//!       an equal `PrefixCode` advanced to an identical bit position.
//! * Any `Err(_)` is a §6.2.1 refusal (truncated input; a simple-code
//!   symbol out of range; a `max_symbol` beyond the alphabet; a repeat /
//!   zero-run that overran the alphabet; an over-subscribed or incomplete
//!   normal code; a length beyond the 15-bit ceiling). The harness
//!   asserts only that the call returned a `Result` rather than panicking.
//!
//! Every assertion below is a real §6.2.1 / §6.2.3 carrier violation if
//! it ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — `alphabet_size` selector. The §6.2.3 VP8L alphabets are
//!   `40` (distance), `256` (red / blue / alpha), and `256 + 24 +
//!   color_cache_size` (green, `color_cache_size ∈ {0} ∪ {2 .. 2048}`).
//!   The selector picks one of those wire-reachable sizes so both the
//!   small distance alphabet, the byte alphabets, and the full green
//!   alphabet range are exercised against the same bitstream readers.
//! * Bytes `[1..]` — the §6.2.1 single-code bit sequence read by a
//!   zero-positioned `BitReader`. A short or empty tail raises an EOF
//!   refusal on the first §6.2.1 simple/normal flag bit.
//!
//! ## Iteration cost bound
//!
//! A single `PrefixCode::read` parses at most one §6.2.1 code-length
//! table (the 19-symbol meta code-length alphabet, then up to
//! `alphabet_size` per-symbol lengths). The largest selectable alphabet
//! is the green `256 + 24 + 2048 = 2328`; a single `read_symbol` walks at
//! most `MAX_CODE_LENGTH = 15` bits. `BitReader` indexes by `usize`
//! across the slice so every read is clamped at the slice end and a
//! single iteration completes in microseconds to milliseconds regardless
//! of input length.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_prefix::{PrefixCode, PrefixError};
use oxideav_webp::vp8l_stream::BitReader;

/// §6.2.1 code-length ceiling (`MAX_CODE_LENGTH`); re-declared here so the
/// harness depends only on the public surface.
const MAX_CODE_LENGTH: u8 = 15;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // §6.2.3 alphabet-size selector. The wire-reachable VP8L alphabets:
    // `40` (distance), `256` (red / blue / alpha), and the green
    // `256 + 24 + color_cache_size` with `color_cache_size ∈ {0} ∪
    // {2, 4, …, 2048}` (cache disabled, or `1 << code_bits` for
    // `code_bits ∈ [1, 11]`). Index this 14-wide table off byte 0.
    let alphabet_size: usize = match data[0] % 14 {
        0 => 40,                             // §6.2.3 distance alphabet
        1 => 256,                            // §6.2.3 red / blue / alpha alphabet
        2 => 256 + 24,                       // §6.2.3 green, cache disabled
        n => 256 + 24 + (1usize << (n - 2)), // green, cache size 2 .. 2048
    };

    let payload = &data[1..];
    let total_bits = payload.len() * 8;

    let mut reader = BitReader::new(payload);
    let result = PrefixCode::read(&mut reader, alphabet_size);

    match result {
        Ok(code) => {
            // §6.2.3 alphabet sizing: the per-symbol length table is
            // exactly `alphabet_size` entries long.
            assert_eq!(
                code.code_lengths().len(),
                alphabet_size,
                "§6.2.3 code-length table must have one entry per alphabet symbol",
            );

            // §6.2.1 the code-length ceiling: every nonzero length <= 15.
            let mut nonzero = 0usize;
            let mut last_nonzero: Option<usize> = None;
            for (sym, &len) in code.code_lengths().iter().enumerate() {
                assert!(
                    len <= MAX_CODE_LENGTH,
                    "§6.2.1 length {len} for symbol {sym} exceeds the {MAX_CODE_LENGTH}-bit ceiling",
                );
                if len != 0 {
                    nonzero += 1;
                    last_nonzero = Some(sym);
                }
            }

            // §6.2.1 single-leaf-node consistency.
            match code.single_symbol() {
                Some(sym) => {
                    assert_eq!(
                        nonzero, 1,
                        "§6.2.1 single_symbol() is Some but the length table has {nonzero} nonzero entries",
                    );
                    assert_eq!(
                        Some(sym as usize),
                        last_nonzero,
                        "§6.2.1 single_symbol() must point at the lone nonzero length entry",
                    );
                }
                None => {
                    assert!(
                        nonzero >= 2,
                        "§6.2.1 single_symbol() is None but the length table has only {nonzero} nonzero entries",
                    );
                }
            }

            // §6.2.1 every successfully-built code is decodable: a fresh
            // all-zero reader resolves to a valid in-range symbol index.
            let zeros = [0u8; 4];
            let mut zero_reader = BitReader::new(&zeros);
            let symbol = code
                .read_symbol(&mut zero_reader)
                .expect("§6.2.1 a successfully-built prefix code must decode the all-zero path");
            assert!(
                (symbol as usize) < alphabet_size,
                "§6.2.1 decoded symbol {symbol} is outside the {alphabet_size}-symbol alphabet",
            );

            // §6.2.1 the Kraft completeness gate: rebuilding a multi-symbol
            // code from its own returned length table must succeed (the
            // `sum 2^-len == 1` invariant a built code satisfies) and
            // reproduce an equal code. A single-leaf code is the documented
            // exception, but it too round-trips through `from_code_lengths`.
            let rebuilt = PrefixCode::from_code_lengths(code.code_lengths().to_vec())
                .expect("§6.2.1 a built code's own length table must rebuild");
            assert_eq!(
                rebuilt, code,
                "§6.2.1 rebuilding from the returned length table must reproduce the code",
            );

            // `BitReader` clamps every read at the slice end.
            assert!(
                reader.bit_position() <= total_bits,
                "§6.2.1 successful read advanced to bit {} beyond slice bit length {total_bits}",
                reader.bit_position(),
            );

            // Determinism: replaying the same bytes + `alphabet_size`
            // yields an equal code advanced to an identical bit position.
            let mut replay_reader = BitReader::new(payload);
            let replay = PrefixCode::read(&mut replay_reader, alphabet_size)
                .expect("§6.2.1 replay of a successful code read must also succeed");
            assert_eq!(
                replay, code,
                "§6.2.1 PrefixCode::read must be deterministic over the same bytes",
            );
            assert_eq!(
                replay_reader.bit_position(),
                reader.bit_position(),
                "§6.2.1 replay must advance the reader identically",
            );
        }
        Err(PrefixError::LengthTooLarge { length }) => {
            // §6.2.1: this variant is raised by `from_code_lengths` only
            // for a length strictly above the ceiling. The wire readers
            // cap literal lengths at the §6.2.1 3-bit code-length-code
            // field (`<= 7`) and the repeat codes copy a `prev_nonzero`
            // already `<= 15`, so a successfully-read table never exceeds
            // the ceiling — but the variant is part of the public refusal
            // surface, so cross-check the field is in fact over the limit.
            assert!(
                length > MAX_CODE_LENGTH as usize,
                "§6.2.1 LengthTooLarge.length {length} must exceed the {MAX_CODE_LENGTH}-bit ceiling",
            );
        }
        Err(_) => {
            // §6.2.1: a bitstream-level refusal (truncated input, a
            // simple-code symbol out of range, a `max_symbol` beyond the
            // alphabet, a repeat / zero-run overrun, or an over-subscribed
            // / incomplete normal code). The granular refusal modes are
            // cross-checked by the `prefix_code_group` / `parse_meta_prefix`
            // sibling harnesses through their own entry points; here the
            // contract under test is only that the call returned a `Result`
            // rather than panicking.
        }
    }
});
