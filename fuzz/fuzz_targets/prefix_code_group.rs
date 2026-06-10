#![no_main]

//! Read arbitrary fuzz-supplied bits through the §6.2 / §6.2.1
//! *prefix-code-group* reader standalone entry point
//! `oxideav_webp::meta_prefix::PrefixCodeGroup::read`.
//!
//! A §6.2 prefix-code group is the five canonical §6.2.1 prefix codes
//! every VP8L pixel is decoded with: code one is green plus backref-length
//! plus color-cache (alphabet `256 + 24 + color_cache_size` per §6.2.3),
//! codes two/three/four are red/blue/alpha (each alphabet `256`), and code
//! five is the backref distance (alphabet `40`). `PrefixCodeGroup::read`
//! reads the five in that bitstream order, each via `PrefixCode::read` then
//! the §6.2.1 simple/normal `read_code_lengths` dispatch and the §6.2.1
//! canonical-code `from_code_lengths` build with its Kraft completeness gate.
//!
//! This twenty-third harness drives the §6.2 prefix-code-group surface
//! directly — the layer immediately *below* the round-271
//! `decode_entropy_coded_image` (§7.3): that function reads a §5.2.3
//! color-cache-info bit and then exactly one `PrefixCodeGroup::read`
//! before running the §5.2 pixel loop. Sibling harnesses surround this
//! primitive but none drives it standalone across an attacker-controlled
//! `(color_cache_size, bitstream)` cross-product:
//!
//! * `parse_meta_prefix` (round 261) drives `MetaPrefixHeader::read`,
//!   which reaches `PrefixCodeGroup::read` only after the §5.2.3 +
//!   §6.2.2 preamble has selected the single-group branch, so the
//!   `color_cache_size` fed to the group is whatever the preamble's
//!   4-bit `color_cache_code_bits` produced — never the full
//!   attacker-chosen value range, and never the cache-disabled `0` and a
//!   wide cache size in the same corpus.
//! * `decode_entropy_coded_image` (round 271) and `decode_argb`
//!   (round 272) reach the group only through the §7.3 / §6.2.2 wrapper
//!   and immediately consume its symbols in the pixel loop, so a parse
//!   failure inside the group is indistinguishable from a later §5.2
//!   refusal at the harness boundary.
//!
//! The contract under test, per RFC 9649 §6.2 + §6.2.1 + §6.2.3:
//!
//! * `PrefixCodeGroup::read` always returns a `Result` — no panic, no
//!   debug-build integer overflow, no out-of-bounds index when the
//!   bitstream is empty, truncated, or arbitrarily long, and no
//!   allocation sized by a header field the §6.2.1 readers did not
//!   bound. Every read goes through
//!   [`oxideav_webp::vp8l_stream::BitReader`] whose EOF path raises a
//!   `MetaPrefixError::Eof`, never an underflow panic.
//! * If the call returns `Ok(group)`:
//!     * §6.2.3 alphabet sizing: each of the five codes never assigns a
//!       length to a symbol outside its alphabet, so the per-symbol
//!       code-length table length is bounded by its alphabet — green by
//!       `256 + 24 + color_cache_size`, red/blue/alpha by `256`, and
//!       distance by `40`. Cross-checked: every code's
//!       `code_lengths().len()` equals its alphabet size, and every
//!       nonzero length is `<= 15` (the §6.2.1 `MAX_CODE_LENGTH` ceiling).
//!     * §6.2.1 every code is decodable: `read_symbol` against a fresh
//!       all-zero reader yields a valid symbol *index* (the §6.2.1
//!       single-leaf-node code consumes no bits and returns its lone
//!       symbol; a multi-symbol code returns the symbol reached by the
//!       all-zero path). The returned symbol index must be `< alphabet`.
//!     * §6.2.1 single-leaf-node consistency: a code whose
//!       `single_symbol()` is `Some(s)` has exactly one nonzero entry in
//!       its length table (at symbol `s`), and a code whose
//!       `single_symbol()` is `None` has two or more.
//!     * `BitReader` clamps every read at the slice end, so a successful
//!       read never advances past the slice's bit length.
//!     * Determinism: replaying the same bytes + `color_cache_size`
//!       yields an equal `PrefixCodeGroup` advanced to an identical bit
//!       position.
//! * Any `Err(_)` is a §6.2.1 refusal (truncated input; a simple-code
//!   symbol out of range; an over-subscribed or incomplete normal code;
//!   a code length beyond the 15-bit ceiling). The harness asserts only
//!   that the call returned a `Result` rather than panicking.
//!
//! Every assertion below is a real §6.2 / §6.2.1 / §6.2.3 carrier
//! violation if it ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — `color_cache_size` selector. The §6.2.3 green alphabet
//!   is `256 + 24 + color_cache_size`; the §5.2.3 enabled cache size is
//!   `1 << color_cache_code_bits` with `code_bits ∈ [1, 11]`, i.e.
//!   `{2, 4, 8, …, 2048}`, plus the disabled `0`. The selector picks one
//!   of `{0, 2, 4, 8, 16, … 2048}` so both the cache-disabled and the
//!   full enabled-cache-size range are reached.
//! * Bytes `[1..]` — the §6.2 five-prefix-code bit sequence read by a
//!   zero-positioned `BitReader`. A short or empty tail raises an EOF
//!   refusal on the first §6.2.1 simple/normal flag bit.
//!
//! ## Iteration cost bound
//!
//! The five reads each parse at most one §6.2.1 code-length table (the
//! 19-symbol meta code-length alphabet, then up to `alphabet` per-symbol
//! lengths). The largest alphabet is the green `256 + 24 + 2048 = 2328`;
//! a single `read_symbol` walks at most `MAX_CODE_LENGTH = 15` bits.
//! `BitReader` indexes by `usize` across the slice so every read is
//! clamped at the slice end and a single iteration completes in
//! microseconds to milliseconds regardless of input length.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::meta_prefix::{MetaPrefixError, PrefixCodeGroup};
use oxideav_webp::vp8l_prefix::PrefixCode;
use oxideav_webp::vp8l_stream::BitReader;

/// §6.2.1 code-length ceiling (`MAX_CODE_LENGTH`); re-declared here so the
/// harness depends only on the public surface.
const MAX_CODE_LENGTH: u8 = 15;

/// Cross-check one decoded §6.2.1 prefix code against its alphabet.
fn check_code(code: &PrefixCode, alphabet: usize, label: &str) {
    // §6.2.3 alphabet sizing: the per-symbol length table is exactly the
    // alphabet size (lengths for absent symbols are zero).
    assert_eq!(
        code.code_lengths().len(),
        alphabet,
        "§6.2.3 {label} code-length table must have one entry per alphabet symbol",
    );

    // §6.2.1 the code-length ceiling: every nonzero length is <= 15.
    let mut nonzero = 0usize;
    let mut last_nonzero: Option<usize> = None;
    for (sym, &len) in code.code_lengths().iter().enumerate() {
        assert!(
            len <= MAX_CODE_LENGTH,
            "§6.2.1 {label} length {len} for symbol {sym} exceeds the {MAX_CODE_LENGTH}-bit ceiling",
        );
        if len != 0 {
            nonzero += 1;
            last_nonzero = Some(sym);
        }
    }

    // §6.2.1 single-leaf-node consistency: `single_symbol()` is `Some`
    // exactly when the code has one nonzero length entry, and it points
    // at that symbol.
    match code.single_symbol() {
        Some(sym) => {
            assert_eq!(
                nonzero, 1,
                "§6.2.1 {label} single_symbol() is Some but the length table has {nonzero} nonzero entries",
            );
            assert_eq!(
                Some(sym as usize),
                last_nonzero,
                "§6.2.1 {label} single_symbol() must point at the lone nonzero length entry",
            );
        }
        None => {
            assert!(
                nonzero >= 2,
                "§6.2.1 {label} single_symbol() is None but the length table has only {nonzero} nonzero entries",
            );
        }
    }

    // §6.2.1 every successfully-built code is decodable: reading a symbol
    // from a fresh all-zero reader yields a valid in-range symbol index
    // (single-leaf node consumes no bits; a multi-symbol code follows the
    // all-zero path). A built code never fails to resolve the zero path.
    let zeros = [0u8; 4];
    let mut zero_reader = BitReader::new(&zeros);
    let symbol = code
        .read_symbol(&mut zero_reader)
        .expect("§6.2.1 a successfully-built prefix code must decode the all-zero path");
    assert!(
        (symbol as usize) < alphabet,
        "§6.2.1 {label} decoded symbol {symbol} is outside the {alphabet}-symbol alphabet",
    );
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // §5.2.3 color-cache size selector: `{0}` (disabled) or one of the
    // enabled sizes `1 << code_bits` for `code_bits ∈ [1, 11]`, i.e.
    // `{2, 4, …, 2048}`. The §6.2.3 green alphabet is sized off this.
    let color_cache_size: usize = match data[0] % 12 {
        0 => 0,
        n => 1usize << n, // n ∈ [1, 11] → 2 .. 2048
    };
    let green_alphabet = PrefixCodeGroup::green_alphabet_size(color_cache_size);

    let payload = &data[1..];
    let total_bits = payload.len() * 8;

    let mut reader = BitReader::new(payload);
    let result = PrefixCodeGroup::read(&mut reader, color_cache_size);

    match result {
        Ok(group) => {
            // §6.2.3 / §6.2.1 per-code cross-checks against each alphabet.
            check_code(&group.green, green_alphabet, "green");
            check_code(&group.red, 256, "red");
            check_code(&group.blue, 256, "blue");
            check_code(&group.alpha, 256, "alpha");
            check_code(&group.distance, 40, "distance");

            // `BitReader` clamps every read at the slice end, so a
            // successful group read cannot have walked past the slice.
            assert!(
                reader.bit_position() <= total_bits,
                "§6.2 successful read advanced to bit {} beyond slice bit length {total_bits}",
                reader.bit_position(),
            );

            // Determinism: replaying the same bytes + `color_cache_size`
            // yields an equal group advanced to an identical bit position.
            let mut replay_reader = BitReader::new(payload);
            let replay = PrefixCodeGroup::read(&mut replay_reader, color_cache_size)
                .expect("§6.2 replay of a successful group read must also succeed");
            assert_eq!(
                replay, group,
                "§6.2 PrefixCodeGroup::read must be deterministic over the same bytes",
            );
            assert_eq!(
                replay_reader.bit_position(),
                reader.bit_position(),
                "§6.2 replay must advance the reader identically",
            );
        }
        Err(MetaPrefixError::InvalidColorCacheCodeBits { .. }) => {
            // §5.2.3: `PrefixCodeGroup::read` never reads a color-cache
            // info field — the cache size is supplied by the caller — so
            // this variant must never originate here.
            panic!("§6.2 PrefixCodeGroup::read must not raise InvalidColorCacheCodeBits",);
        }
        Err(_) => {
            // §6.2.1: a bitstream-level refusal (truncated input, a
            // simple-code symbol out of range, an over-subscribed or
            // incomplete normal code, or a length beyond the ceiling).
            // The granular refusal modes are cross-checked by the
            // `parse_meta_prefix` sibling harness through its own entry
            // point; here the contract under test is only that the call
            // returned a `Result` rather than panicking.
        }
    }
});
