#![no_main]

//! Differential oracle on the §6.2.1 canonical-decoder *symbol read*:
//! the round-284 primary-lookup-table fast path
//! (`vp8l_prefix::PrefixCode::read_symbol` — a 256-entry table keyed on
//! the next 8 peeked wire-order bits, with a per-bit continuation walk
//! for codes longer than 8 bits and per-bit fallbacks near EOF and
//! below the used-symbol amortization gate) against the crate's own
//! pre-table per-bit row walk, kept as the `#[doc(hidden)]`
//! `PrefixCode::read_symbol_reference` oracle.
//!
//! Two readers over the *same* bytes are advanced in lockstep — one
//! through `read_symbol`, one through `read_symbol_reference` — and
//! after **every** symbol the harness asserts:
//!
//! * the decoded symbol (or the typed refusal) is identical, including
//!   the `PrefixError::Eof` `bit_pos` / `wanted` / `available` fields
//!   (the fast path documents that near-EOF table outcomes are replayed
//!   through the per-bit walk precisely so those fields match);
//! * both cursors sit at the same bit position (the table path advances
//!   by the matched code's length in one step; the walk advances bit by
//!   bit — any drift is a real §6.2.1 fast-path bug);
//! * every decoded symbol is inside the code's alphabet;
//! * neither cursor ever passes the slice's bit length.
//!
//! The code under test is built two ways so the mutator controls both
//! the *table shape* and the *bit soup*:
//!
//! * **Wire mode** (even selector byte): the code is read off the fuzz
//!   bytes through `PrefixCode::read` (the §6.2.1 simple/normal
//!   code-length dispatch) at one of the wire-reachable §6.2.3 alphabet
//!   sizes, and the *remaining* bits of the same reader are the symbol
//!   stream — exactly the layout a VP8L entropy body has on disk.
//! * **Table mode** (odd selector byte): the per-symbol lengths are
//!   synthesised directly from fuzz bytes and *repaired* to an exact
//!   §6.2.1 Kraft sum (each desired length in `[1, 15]` is taken while
//!   budget remains, then the residual budget is filled exactly via its
//!   binary decomposition using reserved tail symbols), so
//!   `from_code_lengths` always succeeds and the mutator can steer the
//!   used-symbol count across the `MIN_LOOKUP_USED` gate and the length
//!   profile across the ≤ 8-bit fast path, the > 8-bit continuation
//!   path, and the 15-bit ceiling at will.
//!
//! The single-leaf-node tree (§6.2.1: reading consumes no bits) is
//! compared once instead of looped, since it never advances the cursor.
//!
//! Any assertion below is a real §6.2.1 fast-path/reference divergence
//! if it ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — mode selector: even → wire mode, odd → table mode.
//! * Byte `1` — §6.2.3 alphabet-size selector (same 14-entry table as
//!   the `prefix_code` harness: `40`, `256`, `256 + 24`, or
//!   `256 + 24 + 2^k` for `k ∈ [1, 11]`).
//! * Wire mode: bytes `[2..]` feed a zero-positioned `BitReader`;
//!   `PrefixCode::read` consumes the code-length header and the rest of
//!   the stream is the symbol soup.
//! * Table mode: bytes `2..3` (LE u16) pick the desired-symbol count;
//!   that many following bytes are per-symbol desired lengths
//!   (`b % 15 + 1`); the remaining bytes are the symbol soup read from
//!   bit 0.
//!
//! ## Iteration cost bound
//!
//! Each multi-symbol `read_symbol` consumes at least one bit, so the
//! lockstep loop is bounded by the input's bit length and ends with the
//! mandatory EOF-arm comparison. Table builds are at most one 2328-entry
//! length table plus the 256-entry lookup stamp. A single iteration
//! completes in microseconds regardless of input length.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_prefix::{PrefixCode, MAX_CODE_LENGTH};
use oxideav_webp::vp8l_stream::BitReader;

/// §6.2.1 Kraft denominator: the sum of `2^(MAX_CODE_LENGTH - len)`
/// over all used symbols of a complete code is exactly this.
const KRAFT_FULL: u64 = 1 << MAX_CODE_LENGTH;

/// Map the selector byte onto a wire-reachable §6.2.3 alphabet size:
/// `40` (distance), `256` (red / blue / alpha), `256 + 24` (green,
/// cache disabled), or `256 + 24 + 2^k` (green, cache size 2 .. 2048).
fn alphabet_size(selector: u8) -> usize {
    match selector % 14 {
        0 => 40,
        1 => 256,
        2 => 256 + 24,
        n => 256 + 24 + (1usize << (n - 2)),
    }
}

/// Repair fuzz-desired code lengths into an exact §6.2.1 Kraft-complete
/// table over `alphabet_size` symbols.
///
/// Greedy front fill: each desired length (already in `[1, 15]`) is
/// taken verbatim while its `2^(15 - len)` cost fits the remaining
/// budget, lengthened until it fits otherwise. The residual budget is
/// then consumed *exactly* via its binary decomposition — one tail
/// symbol of length `15 - k` per set bit `k` — using the
/// `MAX_CODE_LENGTH` tail slots the front fill never touches. A fully
/// unspent budget (no desired symbols) degenerates to the 2-leaf
/// length-1 code.
fn kraft_repair(alphabet_size: usize, desired: &[u8]) -> Vec<u8> {
    let mut lengths = vec![0u8; alphabet_size];
    // Reserve the last MAX_CODE_LENGTH symbols for the exact fill.
    let assignable = alphabet_size - MAX_CODE_LENGTH;
    let mut budget = KRAFT_FULL;
    for (slot, &d) in lengths.iter_mut().take(assignable).zip(desired) {
        if budget == 0 {
            break;
        }
        let mut len = (d as usize % MAX_CODE_LENGTH) + 1;
        let mut cost = 1u64 << (MAX_CODE_LENGTH - len);
        while cost > budget {
            len += 1;
            cost >>= 1;
        }
        *slot = len as u8;
        budget -= cost;
    }
    if budget == KRAFT_FULL {
        // Nothing assigned: the smallest multi-symbol complete code.
        lengths[alphabet_size - 1] = 1;
        lengths[alphabet_size - 2] = 1;
        return lengths;
    }
    let mut tail = alphabet_size - 1;
    for k in (0..MAX_CODE_LENGTH).rev() {
        if budget & (1u64 << k) != 0 {
            lengths[tail] = (MAX_CODE_LENGTH - k) as u8;
            tail -= 1;
        }
    }
    lengths
}

/// Lockstep-decode the same bytes through the table fast path and the
/// pre-table reference walk, asserting identity after every symbol.
fn lockstep_decode(code: &PrefixCode, payload: &[u8], start_bit: usize) {
    let total_bits = payload.len() * 8;
    let mut fast = BitReader::new(payload);
    fast.seek_to_bit(start_bit);
    let mut reference = fast.clone();

    if code.single_symbol().is_some() {
        // §6.2.1 single-leaf-node tree: reading consumes no bits, so a
        // lockstep loop would never terminate. One comparison pins both
        // paths' (identical, cursor-neutral) outcome.
        let a = code.read_symbol(&mut fast);
        let b = code.read_symbol_reference(&mut reference);
        assert_eq!(
            a, b,
            "§6.2.1 single-leaf fast path diverged from the reference walk"
        );
        assert_eq!(
            fast.bit_position(),
            reference.bit_position(),
            "§6.2.1 single-leaf read must not move either cursor",
        );
        return;
    }

    loop {
        let a = code.read_symbol(&mut fast);
        let b = code.read_symbol_reference(&mut reference);
        // Symbol-or-refusal identity, including the `Eof` position /
        // wanted / available fields the fast path's EOF fallback
        // documents as exactly reproduced.
        assert_eq!(
            a, b,
            "§6.2.1 lookup-table fast path diverged from the per-bit reference walk",
        );
        assert_eq!(
            fast.bit_position(),
            reference.bit_position(),
            "§6.2.1 fast path and reference walk consumed different bit counts",
        );
        assert!(
            fast.bit_position() <= total_bits,
            "§6.2.1 read_symbol advanced past the slice's bit length",
        );
        match a {
            Ok(symbol) => assert!(
                (symbol as usize) < code.code_lengths().len(),
                "§6.2.1 decoded symbol {symbol} is outside the alphabet",
            ),
            // The EOF arm has been compared; the stream is exhausted.
            Err(_) => break,
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let table_mode = data[0] % 2 == 1;
    let alphabet = alphabet_size(data[1]);
    let rest = &data[2..];

    if table_mode {
        // ---- Table mode: fuzz-shaped Kraft-complete length table, so
        // the used-symbol count (the MIN_LOOKUP_USED gate) and the
        // length profile (≤ 8-bit table hits vs > 8-bit continuations)
        // are both mutator-controlled. ----
        if rest.len() < 2 {
            return;
        }
        let desired_count = u16::from_le_bytes([rest[0], rest[1]]) as usize;
        let desired_end = (2 + desired_count).min(rest.len());
        let lengths = kraft_repair(alphabet, &rest[2..desired_end]);
        let code = PrefixCode::from_code_lengths(lengths)
            .expect("§6.2.1 a Kraft-repaired length table must always build");
        lockstep_decode(&code, &rest[desired_end..], 0);
    } else {
        // ---- Wire mode: the code-length header *and* the symbol soup
        // share one stream, exactly as a §5 entropy body is laid out on
        // disk. A header refusal is the `prefix_code` harness's domain;
        // here it just ends the iteration. ----
        let mut reader = BitReader::new(rest);
        if let Ok(code) = PrefixCode::read(&mut reader, alphabet) {
            lockstep_decode(&code, rest, reader.bit_position());
        }
    }
});
