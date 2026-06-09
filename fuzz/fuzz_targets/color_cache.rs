#![no_main]

//! Probe the §5.2.3 lossless-color-cache hash + insert + lookup state
//! machine `oxideav_webp::vp8l_decode::ColorCache` with arbitrary
//! `(code_bits, argb)` traffic.
//!
//! Every VP8L §5.2 GREEN symbol whose value is `>= 256 + 24` is a
//! color-cache code — the resolved index `S - (256 + 24)` is fed to
//! the cache's `lookup`, the returned ARGB is emitted as the pixel,
//! and that pixel is then re-inserted (`insert`) into the cache. Every
//! literal pixel and every backward-reference pixel is **also**
//! inserted into the cache as it is emitted. The cache itself is a
//! `1 << code_bits` array of ARGB entries; the slot of any color is
//! `(0x1e35a7bd * argb) >> (32 - code_bits)`. The §5.2.3 spec text
//! is explicit: "Only one lookup is done in a color cache; there is no
//! conflict resolution." Two colors that collide on the hash overwrite
//! each other in slot order — the most-recently-inserted wins.
//!
//! Sibling harnesses cover every surface that **feeds** the color
//! cache — `parse_meta_prefix` (§5.2.3 + §6.2.2 preamble that decides
//! whether the cache is enabled at all and at what `code_bits`),
//! `parse_transform_list` (§4 transform-list reader that runs before
//! the §5 entropy body), `parse_container` (§2.3 / §2.4 RIFF walker
//! that locates the §2.6 VP8L chunk), `decode` (full §2 RIFF + §3..§5
//! entry point that wraps every primitive), `roundtrip_lossless`
//! (encode→decode equality oracle on the full §3 lossless contract) —
//! but **none** of them reaches the §5.2.3 `ColorCache` operations
//! directly: they reach them only via whichever GREEN-symbol stream
//! the upstream prefix code produces, which means the actual
//! `(code_bits, argb)` patterns visited per iteration are bounded by
//! the entropy stream the upstream reader produces. This fourteenth
//! harness drives the §5.2.3 cache primitives directly across the
//! full attacker-reachable `code_bits ∈ [1, 11]` × `argb ∈ [0,
//! u32::MAX]` cross-product, with every hashed slot, every
//! insert-then-lookup round trip, every collision-overwrite, every
//! lookup-out-of-range refusal, and the §5.2.3 cache initialization
//! invariant cross-checked against the spec formula.
//!
//! The contract under test, per RFC 9649 §5.2.3:
//!
//! * `ColorCache::new(code_bits)` returns a cache of `size() == 1 <<
//!   code_bits` entries, each initialized to ARGB `0` (§5.2.3 cache
//!   initialization: "all entries in all color cache values are set
//!   to zero").
//! * `cache.hash(argb)` returns `(0x1e35a7bd * argb) >> (32 -
//!   code_bits)` as a `usize`, always in `[0, size())` (the upper
//!   `code_bits` bits of a u32 are at most `(1 << code_bits) - 1`).
//! * `cache.insert(argb)` writes `argb` into `cache.entries[hash(argb)]`
//!   — a single-slot single-write, no conflict resolution. Two colors
//!   `a != b` with `hash(a) == hash(b)` will, on `insert(a); insert(b)`,
//!   leave `lookup(hash(a)) == Some(b)` (the second insert overwrites
//!   the first; the most-recently-inserted color wins).
//! * `cache.lookup(index)` returns `Some(argb)` if `index < size()`
//!   and `None` otherwise. After `insert(argb)`, `lookup(hash(argb))
//!   == Some(argb)` (round-trip invariant).
//! * The cache is a pure function of the insert sequence — repeating
//!   the same operations on a fresh cache produces the same lookups.
//!
//! Every assertion below is a real §5.2.3 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Iteration cost bound
//!
//! Each color processed is a constant-time hash (multiply + shift +
//! cast) plus an indexed write and an indexed read. The harness fixes
//! `code_bits` from the first byte (masked to `[1, 11]` per §5.2.3),
//! then slices the rest of the fuzz buffer into 4-byte ARGB words and
//! processes at most `(data.len() - 1) / 4` of them per iteration.
//! With the libFuzzer 4 KiB default that's ~1023 colors; with the
//! 64 KiB cap it's ~16383.
//!
//! ## Input layout
//!
//! * Byte `[0]` — `code_bits_raw`. The §5.2.3 `color_cache_code_bits`
//!   wire field is 4 bits wide with the valid range `[1, 11]`; we
//!   mask the byte to its low nibble (`raw & 0x0F`) and remap any
//!   out-of-range value (`0`, `12..=15`) by `((raw & 0x0F).max(1)) %
//!   12` so the harness only exercises the spec-permitted window.
//!   The largest `code_bits` honoured is `11` (cache size `2048`).
//! * Bytes `[1..]` — repeated little-endian u32 `argb` words. Each
//!   is fed to the cache verbatim — every ARGB pattern is reachable
//!   on the §5.2 wire (the literal A/R/G/B channels are each 8 bits
//!   wide with no range restriction).

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_decode::{ColorCache, COLOR_CACHE_HASH_MULTIPLIER};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // §5.2.3 color_cache_code_bits ∈ [1, 11]. The raw 4-bit field can
    // encode 0 and 12..=15, but the spec mandates "Compliant decoders
    // MUST indicate a corrupted bitstream for other values" — so the
    // cache primitive itself is only ever called with code_bits in the
    // honoured range. Remap any out-of-range fuzz byte into the
    // permitted window so the harness only exercises the §5.2.3
    // reachable surface.
    let raw = data[0] & 0x0F;
    let code_bits = (raw.max(1)) % 12;
    // The above can produce 0 again only when raw == 12 (12 % 12 = 0);
    // bump back to 1 in that case so the permitted window is exactly
    // [1, 11].
    let code_bits = if code_bits == 0 {
        1u32
    } else {
        code_bits as u32
    };
    assert!(
        (1..=11).contains(&code_bits),
        "code_bits must be in §5.2.3 permitted range [1, 11]; got {code_bits}",
    );

    let mut cache = ColorCache::new(code_bits);
    let expected_size = 1usize << code_bits;
    check_initial_state(&cache, code_bits, expected_size);

    // A parallel "shadow" cache models the §5.2.3 single-slot
    // single-write semantics directly. After every insert into the
    // real cache we re-derive what the spec says the slot should hold
    // and assert the real cache matches.
    let mut shadow: Vec<u32> = vec![0u32; expected_size];

    for chunk in data[1..].chunks_exact(4) {
        let argb = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        check_one(&mut cache, &mut shadow, argb, code_bits, expected_size);
    }

    // §5.2.3 determinism: rebuilding the cache from the same insert
    // sequence on a fresh ColorCache produces identical lookups. This
    // catches any latent hidden state that would make the cache a
    // function of more than just `code_bits` + the insert sequence.
    let mut replay = ColorCache::new(code_bits);
    for chunk in data[1..].chunks_exact(4) {
        let argb = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        replay.insert(argb);
    }
    for i in 0..expected_size {
        assert_eq!(
            replay.lookup(i),
            cache.lookup(i),
            "§5.2.3 cache must be a pure function of (code_bits, insert sequence); replay differs at slot {i}",
        );
    }
});

/// Cross-check the §5.2.3 cache initialization invariant: a freshly
/// constructed cache has `size() == 1 << code_bits`, every slot reads
/// as `Some(0)`, and `lookup(size())` reads as `None`.
fn check_initial_state(cache: &ColorCache, code_bits: u32, expected_size: usize) {
    assert_eq!(
        cache.size(),
        expected_size,
        "§5.2.3 ColorCache::new({code_bits}) must produce size 1 << code_bits = {expected_size}; got {}",
        cache.size(),
    );

    // §5.2.3: "all entries in all color cache values are set to zero."
    for i in 0..expected_size {
        assert_eq!(
            cache.lookup(i),
            Some(0u32),
            "§5.2.3 fresh cache slot {i} must read as Some(0); got {:?}",
            cache.lookup(i),
        );
    }

    // §5.2.3 cache code S - (256 + 24) is bounded by color_cache_size;
    // a lookup at `size()` is an §5.2.3 ColorCacheIndexOutOfRange (the
    // GREEN symbol bound is exclusive). The cache's `lookup` returns
    // `None` for those indices.
    assert_eq!(
        cache.lookup(expected_size),
        None,
        "§5.2.3 lookup at index size() must return None (out of range)",
    );
    assert_eq!(
        cache.lookup(usize::MAX),
        None,
        "§5.2.3 lookup at usize::MAX must return None (out of range)",
    );
}

/// Cross-check a single `insert(argb); lookup(hash(argb))` round trip
/// against the §5.2.3 hash formula and the §5.2.3 single-slot
/// single-write spec text.
fn check_one(
    cache: &mut ColorCache,
    shadow: &mut [u32],
    argb: u32,
    code_bits: u32,
    expected_size: usize,
) {
    // §5.2.3 hash formula: `(0x1e35a7bd * argb) >> (32 - code_bits)`.
    let expected_hash = COLOR_CACHE_HASH_MULTIPLIER.wrapping_mul(argb) >> (32 - code_bits);
    let expected_hash = expected_hash as usize;
    let actual_hash = cache.hash(argb);
    assert_eq!(
        actual_hash, expected_hash,
        "§5.2.3 hash formula: ColorCache::hash(0x{argb:08X}) must equal (0x1e35a7bd * argb) >> (32 - {code_bits}) = {expected_hash}; got {actual_hash}",
    );

    // §5.2.3 carrier rule: the hash is always in [0, size()). The
    // upper `code_bits` of a u32 are bounded by `(1 << code_bits) - 1`.
    assert!(
        actual_hash < expected_size,
        "§5.2.3 hash {actual_hash} must be in [0, {expected_size}) for code_bits {code_bits}",
    );

    // Determinism: a pure-function hash must give the same answer on
    // the second call (catches any latent hidden state).
    let actual_hash2 = cache.hash(argb);
    assert_eq!(
        actual_hash, actual_hash2,
        "§5.2.3 ColorCache::hash must be deterministic; got {actual_hash} then {actual_hash2} for argb 0x{argb:08X}",
    );

    // Perform the §5.2.3 insert: writes `argb` into the slot at
    // `hash(argb)`. No conflict resolution — the spec text is
    // explicit on this.
    cache.insert(argb);
    shadow[actual_hash] = argb;

    // §5.2.3 round-trip invariant: after `insert(argb)`,
    // `lookup(hash(argb)) == Some(argb)`. This is the §5.2.3 carrier
    // contract every backward-reference-emitting decoder relies on.
    let looked_up = cache.lookup(actual_hash);
    assert_eq!(
        looked_up,
        Some(argb),
        "§5.2.3 insert/lookup round trip: after insert(0x{argb:08X}), lookup(hash) must be Some(0x{argb:08X}); got {looked_up:?}",
    );

    // §5.2.3 cross-slot invariant: every slot the cache has must agree
    // with the shadow model byte-for-byte. This catches any §5.2.3
    // violation where the insert touched a slot other than the hashed
    // one (e.g. open-addressing probe — explicitly forbidden by "Only
    // one lookup is done; there is no conflict resolution"). The shadow
    // length is `expected_size` by construction.
    assert_eq!(shadow.len(), expected_size);
    for (i, &expected_argb) in shadow.iter().enumerate() {
        assert_eq!(
            cache.lookup(i),
            Some(expected_argb),
            "§5.2.3 cache must agree with the single-slot single-write model at slot {i}; cache has {:?}, shadow has 0x{expected_argb:08X}",
            cache.lookup(i),
        );
    }
}
