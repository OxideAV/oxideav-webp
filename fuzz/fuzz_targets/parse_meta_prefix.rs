#![no_main]

//! Parse arbitrary fuzz-supplied bits through the §5.2.3 color-cache
//! info, §6.2.2 meta-prefix, and §6.2 prefix-code-group reader standalone
//! entry point `oxideav_webp::meta_prefix::MetaPrefixHeader::read`.
//!
//! The §3 image-header peek lands the §2.6 lossless bitstream at the
//! start of the §5.2.3 `color-cache-info` field. From there
//! `MetaPrefixHeader::read` walks:
//!
//! ```text
//! color-cache-info      =  %b0
//! color-cache-info      =/ (%b1 4BIT)   ; %b1 + color_cache_code_bits ∈ [1, 11]
//!
//! meta-prefix           =  %b0 / (%b1 entropy-image)   ; ARGB role only
//!
//! prefix-code-group     =  5prefix-code               ; only when meta-prefix = %b0
//! ```
//!
//! For the `Argb` role with the meta-prefix bit set the reader **stops**
//! at the §6.2.2 entropy-image boundary (recording `prefix_bits ∈ [2, 9]`
//! plus the derived `DIV_ROUND_UP` entropy-image dimensions and the bit
//! position the entropy image starts at) — the entropy image itself is a
//! sub-`entropy-coded-image` that the next §5.2 reader resumes. For the
//! `EntropyCoded` role the meta-prefix bit is absent entirely (a sub-image
//! always has exactly one prefix-code group); the reader drops straight
//! from the color-cache-info bit into the five canonical prefix codes.
//!
//! The function is a public standalone surface exported through
//! `pub mod meta_prefix` that downstream callers can invoke against any
//! byte slice they obtained from a different demuxer or an
//! attacker-controlled buffer of arbitrary length, including the empty
//! slice. The role flag and the `(image_width, image_height)` pair are
//! caller-controlled — when the entropy-image branch fires their product
//! drives the `DIV_ROUND_UP` arithmetic for the recorded entropy-image
//! dimensions.
//!
//! Sibling harnesses already cover the orthogonal parsers — `parse_vp8x`
//! (§2.7.1 Figure 7 octet), `parse_anmf` (§2.7.1.1 Figure 9 header),
//! `parse_anim` (§2.7.1.1 Figure 8 carrier), `parse_alph` (§2.7.1.2
//! Figure 10 info byte), `decode_alph` (§2.7.1.2 alpha plane),
//! `parse_transform_list` (§4 VP8L transform-list reader),
//! `extract_metadata` (§2 RIFF walk for ICCP/EXIF/XMP), `decode` (full
//! §2 RIFF + §3..§5 entry), and `roundtrip_animated` /
//! `roundtrip_lossless` (encode→decode equality oracles). This harness
//! widens fuzz coverage onto the §5.2.3 + §6.2.2 + §6.2 preamble path
//! the other harnesses only reach by way of the full §3 entry.
//!
//! The contract under test, per RFC 9649 §5.2.3 + §6.2.2 + §6.2:
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow, no out-of-bounds index when the input is empty
//!   or arbitrarily long. Every read goes through
//!   [`oxideav_webp::vp8l_stream::BitReader`] whose EOF path raises
//!   `BitReaderEof`, never an underflow.
//! * If the call returns `Ok(header)`:
//!     * §5.2.3 `header.color_cache.code_bits` is either `0` (cache
//!       disabled, the `%b0` color-cache-info branch) or in the
//!       compliant range `[1, 11]` (cache enabled, the `%b1 4BIT`
//!       branch with the §5.2.3 range gate accepting it).
//!     * §5.2.3 `header.color_cache.size()` matches the `code_bits`
//!       derivation: `0` when disabled, `1 << code_bits` when enabled.
//!       The §5.2.3 `is_enabled()` predicate matches `code_bits != 0`.
//!     * `header.codes` is `Single { group }` for the `EntropyCoded`
//!       role (the §6.2.2 meta-prefix bit is absent entirely — a
//!       sub-image always carries one group). The `Argb` role may
//!       produce either variant depending on the §6.2.2 meta-prefix
//!       bit; if `EntropyImagePending`, the `prefix_bits` is in `[2, 9]`
//!       (the `ReadBits(3) + 2` decode), the recorded `image_width` /
//!       `image_height` match the `DIV_ROUND_UP(caller_dim,
//!       1 << prefix_bits)` derivation, and the recorded
//!       `entropy_image_bit_position` does not exceed the slice's total
//!       bit length (the reader stops *at* the boundary, not past it).
//! * If the call returns `Err(MetaPrefixError::Eof(eof))`, the EOF is
//!   real: the captured `bit_pos + available` does not exceed the
//!   slice's bit length, and the failing read's `wanted` strictly
//!   exceeds `available`.
//! * If the call returns `Err(MetaPrefixError::InvalidColorCacheCodeBits
//!   { value })`, the §5.2.3 range gate fired: `value` is either `0` or
//!   in `[12, 15]` (the 4-bit field's range minus the compliant
//!   `[1, 11]` window).
//! * If the call returns `Err(MetaPrefixError::Prefix(_))`, one of the
//!   five §6.2.1 prefix codes inside the group failed to parse. The
//!   harness asserts only that the call returned (no panic); the
//!   precise variant is `PrefixError`-shaped and is exercised by
//!   downstream `decode` harnesses through the §3 entry.
//!
//! Every assertion below is a real §5.2.3 / §6.2.2 / §6.2 carrier
//! violation if it ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Iteration cost bound
//!
//! `MetaPrefixHeader::read` is bounded by the §6.2 5-prefix-code-group
//! read on its `Single` branch and by a fixed 3-bit read on the
//! `EntropyImagePending` branch. Each canonical prefix-code parse is
//! bounded by §3.7.2's 19-symbol code-length alphabet plus the per-code
//! alphabet (256 + 24 + cache for green, 256 for red/blue/alpha, 40 for
//! distance). The `EntropyImagePending` branch returns in microseconds
//! regardless of `image_width` / `image_height`. The `Single` branch's
//! worst-case fixed-field reads are bounded by the prefix-code reader's
//! own iteration cap; a single fuzz iteration completes in
//! milliseconds.
//!
//! ## Input layout
//!
//! * Byte 0 — role + image-dimension nibble. Bit 0 selects the §6.2.2
//!   `ImageRole`: `0` for `EntropyCoded` (no meta-prefix bit), `1` for
//!   `Argb` (the meta-prefix bit is read). Bits 1..8 give the upper byte
//!   of a small image-width.
//! * Byte 1 — upper byte of image-height. Width and height are bounded
//!   below at 1 (the §5 image-data block contract excludes zero
//!   dimensions; the harness reuses the §3 image-header `+ 1` rebias).
//! * Bytes 2.. — the §5.2.3 + §6.2.2 + §6.2 bit sequence read by the
//!   parser. If only the two header bytes are provided the bit reader
//!   sees an empty payload and raises `BitReaderEof` on the first
//!   color-cache-info bit.
//!
//! Image dimensions stay small (`< 2^16`) so the `EntropyImagePending`
//! branch's `DIV_ROUND_UP` arithmetic does not overflow when the
//! parser-derived `image_width` / `image_height` cross-product is
//! recomputed for the assertion. `BitReader` itself indexes by `usize`
//! across the slice so the *bit-position* arithmetic is bounded by the
//! slice length regardless.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::meta_prefix::{
    ColorCacheInfo, ImageRole, MetaPrefixCodes, MetaPrefixError, MetaPrefixHeader,
    COLOR_CACHE_BITS_MAX, COLOR_CACHE_BITS_MIN, PREFIX_BITS_MAX, PREFIX_BITS_MIN,
};
use oxideav_webp::vp8l_stream::BitReader;

/// Round-up division — matches the §4.1 `DIV_ROUND_UP` macro the
/// parser uses internally when the `EntropyImagePending` branch fires.
fn div_round_up(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // §6.2.2 role: ARGB carries the meta-prefix bit, EntropyCoded does
    // not. Bit 0 of byte 0 picks between them.
    let role = if data[0] & 1 == 1 {
        ImageRole::Argb
    } else {
        ImageRole::EntropyCoded
    };

    // Small image dimensions. The `+ 1` rebias matches the §3 image-
    // header `image_width_minus_one + 1` convention so the dimensions
    // are at least 1 (a zero canvas would short-circuit the §6.2.2
    // `DIV_ROUND_UP` numerator to 0).
    let image_width: u32 = u32::from(data[0] >> 1).saturating_add(1);
    let image_height: u32 = u32::from(data[1]).saturating_add(1);

    let payload = &data[2..];
    let mut reader = BitReader::new(payload);
    let total_bits = payload.len() * 8;

    match MetaPrefixHeader::read(&mut reader, role, image_width, image_height) {
        Ok(header) => {
            // §5.2.3: color_cache_code_bits is either 0 (disabled) or
            // in the compliant `[1, 11]` range (enabled).
            let code_bits = header.color_cache.code_bits;
            assert!(
                code_bits == 0 || (COLOR_CACHE_BITS_MIN..=COLOR_CACHE_BITS_MAX).contains(&code_bits),
                "§5.2.3 ColorCacheInfo.code_bits {code_bits} must be 0 or in [{COLOR_CACHE_BITS_MIN}, {COLOR_CACHE_BITS_MAX}]",
            );

            // §5.2.3: `is_enabled()` predicate matches `code_bits != 0`.
            assert_eq!(
                header.color_cache.is_enabled(),
                code_bits != 0,
                "§5.2.3 ColorCacheInfo.is_enabled() must match code_bits != 0",
            );

            // §5.2.3: size() is 0 when disabled, 1 << code_bits when
            // enabled. The size matches the canonical `ColorCacheInfo`
            // re-derivation from the same `code_bits`.
            let expected_size = if code_bits == 0 {
                0usize
            } else {
                1usize << code_bits
            };
            assert_eq!(
                header.color_cache.size(),
                expected_size,
                "§5.2.3 ColorCacheInfo.size() must be 0 when disabled and 1 << code_bits when enabled",
            );

            // §5.2.3: a fresh `ColorCacheInfo { code_bits }` has the
            // same observable surface as the parser-returned one.
            let canonical = ColorCacheInfo { code_bits };
            assert_eq!(
                canonical.size(),
                header.color_cache.size(),
                "§5.2.3 ColorCacheInfo.size() must be a pure function of code_bits",
            );
            assert_eq!(
                canonical.is_enabled(),
                header.color_cache.is_enabled(),
                "§5.2.3 ColorCacheInfo.is_enabled() must be a pure function of code_bits",
            );

            // §6.2.2 dispatch: the `EntropyCoded` role cannot reach
            // `EntropyImagePending` (the meta-prefix bit is absent
            // entirely for sub-images).
            if matches!(role, ImageRole::EntropyCoded) {
                assert!(
                    matches!(header.codes, MetaPrefixCodes::Single { .. }),
                    "§6.2.2 EntropyCoded role must produce a Single prefix-code group (no meta-prefix bit)",
                );
                assert!(
                    header.codes.is_single(),
                    "§6.2.2 EntropyCoded role's MetaPrefixCodes::is_single() must be true",
                );
                assert!(
                    header.codes.group().is_some(),
                    "§6.2.2 EntropyCoded role's MetaPrefixCodes::group() must yield Some",
                );
            }

            match &header.codes {
                MetaPrefixCodes::Single { group: _ } => {
                    // §6.2.2 Single branch: `is_single()` matches and
                    // the convenience getter returns the group. The
                    // group's prefix-code internals are validated by
                    // the per-code parse harnesses; here we only assert
                    // the dispatch shape.
                    assert!(
                        header.codes.is_single(),
                        "§6.2.2 MetaPrefixCodes::Single must satisfy is_single() == true",
                    );
                    assert!(
                        header.codes.group().is_some(),
                        "§6.2.2 MetaPrefixCodes::Single must satisfy group().is_some()",
                    );
                }
                MetaPrefixCodes::EntropyImagePending {
                    prefix_bits,
                    image_width: entropy_w,
                    image_height: entropy_h,
                    entropy_image_bit_position,
                } => {
                    // §6.2.2 EntropyImagePending: only the ARGB role
                    // ever reaches this branch.
                    assert!(
                        matches!(role, ImageRole::Argb),
                        "§6.2.2 EntropyImagePending requires the ARGB role (sub-images do not carry the meta-prefix bit)",
                    );

                    // §6.2.2: `prefix_bits = ReadBits(3) + 2` → range
                    // is `[2, 9]` (the `+ 2` minimum and the
                    // `7 + 2 = 9` maximum).
                    let pbits = u32::from(*prefix_bits);
                    assert!(
                        (PREFIX_BITS_MIN..=PREFIX_BITS_MAX).contains(&pbits),
                        "§6.2.2 prefix_bits {pbits} must lie in [{PREFIX_BITS_MIN}, {PREFIX_BITS_MAX}]",
                    );

                    // §6.2.2: the entropy-image dimensions are
                    // `DIV_ROUND_UP(image_width / image_height,
                    // 1 << prefix_bits)`. Recompute against the caller-
                    // controlled `(image_width, image_height)` pair and
                    // assert byte-equality with the recorded values.
                    let block_size = 1u32 << pbits;
                    let expected_w = div_round_up(image_width, block_size);
                    let expected_h = div_round_up(image_height, block_size);
                    assert_eq!(
                        *entropy_w, expected_w,
                        "§6.2.2 entropy-image width must equal DIV_ROUND_UP(image_width, 1 << prefix_bits)",
                    );
                    assert_eq!(
                        *entropy_h, expected_h,
                        "§6.2.2 entropy-image height must equal DIV_ROUND_UP(image_height, 1 << prefix_bits)",
                    );

                    // §6.2.2: the recorded bit position is where the
                    // entropy image *begins*, i.e. just past the
                    // `prefix_bits` field. It cannot exceed the slice's
                    // total bit length — `BitReader` clamps every read
                    // at the slice end, so a successful Ok must have
                    // walked to a position inside the slice.
                    assert!(
                        *entropy_image_bit_position <= total_bits,
                        "§6.2.2 entropy_image_bit_position {entropy_image_bit_position} exceeded slice bit length {total_bits}",
                    );

                    // §6.2.2 EntropyImagePending: the dispatch shape
                    // predicates report the variant correctly.
                    assert!(
                        !header.codes.is_single(),
                        "§6.2.2 MetaPrefixCodes::EntropyImagePending must satisfy is_single() == false",
                    );
                    assert!(
                        header.codes.group().is_none(),
                        "§6.2.2 MetaPrefixCodes::EntropyImagePending must satisfy group().is_none()",
                    );
                }
            }
        }
        Err(MetaPrefixError::Eof(eof)) => {
            // §5.2.3 / §6.2.2 / §6.2: the BitReader EOF coordinate must
            // be in range — the position the failing read started at
            // plus the bits it could see must not exceed the slice's
            // bit length.
            assert!(
                eof.bit_pos.saturating_add(eof.available) <= total_bits,
                "§5.2.3 / §6.2.2 BitReaderEof reported bit_pos {} + available {} > slice bit length {total_bits}",
                eof.bit_pos,
                eof.available,
            );
            // The reader only raises EOF if the demand strictly
            // exceeded what was left in the slice.
            assert!(
                eof.wanted > eof.available,
                "§5.2.3 / §6.2.2 BitReaderEof wanted {} <= available {} should not have raised EOF",
                eof.wanted,
                eof.available,
            );
        }
        Err(MetaPrefixError::InvalidColorCacheCodeBits { value }) => {
            // §5.2.3 range gate: the rejected value is a 4-bit field
            // (`ReadBits(4)`) so `value` lies in `[0, 15]`. The §5.2.3
            // compliant range is `[1, 11]`; the rejection fires iff
            // `value` is `0` or in `[12, 15]`.
            assert!(
                value < 16,
                "§5.2.3 color_cache_code_bits is a 4-bit field; rejected value {value} >= 16",
            );
            assert!(
                value == 0 || value > COLOR_CACHE_BITS_MAX,
                "§5.2.3 InvalidColorCacheCodeBits {value} must be 0 or > {COLOR_CACHE_BITS_MAX} (the compliant range is [{COLOR_CACHE_BITS_MIN}, {COLOR_CACHE_BITS_MAX}])",
            );
        }
        Err(MetaPrefixError::Prefix(_)) => {
            // §6.2.1: one of the five canonical prefix codes inside the
            // group failed to parse. The precise variant is the
            // `PrefixError` shape (alphabet overflow, simple-code
            // mismatch, code-length overflow, ...) which the sibling
            // `decode` / `decode_alph` / `roundtrip_lossless` harnesses
            // already exercise through the §3 entry. Here we only
            // assert the call returned at all (the harness reached this
            // arm, so the parser surfaced the §6.2.1 refusal cleanly).
        }
    }
});
