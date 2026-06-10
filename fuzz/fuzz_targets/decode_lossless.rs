#![no_main]

//! Decode arbitrary fuzz-supplied bits through the §4 + §5 + §6 full VP8L
//! lossless-bitstream decode path standalone entry points
//! `oxideav_webp::vp8l_transform::{decode_lossless, decode_lossless_headerless}`.
//!
//! `decode_lossless` is the §4 transform-list + main-image driver — the
//! layer immediately *above* the round-272 §6.2.2
//! `vp8l_decode::decode_argb`. It:
//!
//! * walks the §4 / §7.2 optional-transform loop (`while ReadBits(1)`),
//!   reading each transform's §4.x fixed fields and its §5-encoded body
//!   (the predictor / color sub-resolution images via the §7.3
//!   `decode_entropy_coded_image`, the §4.4 color-table via the same
//!   entry point, and the §4.4 `color_table_size` / `width_bits` width
//!   subsampling), refusing a repeated transform per §4 ("Each transform
//!   is allowed to be used only once.");
//! * decodes the main §5.1 ARGB image (`decode_argb`) at the
//!   possibly-subsampled width;
//! * applies the §4 inverse-transform chain in reverse read order (§4:
//!   "The inverse transforms are applied in the reverse order that they
//!   are read from the bitstream, that is, last one first.").
//!
//! `decode_lossless_headerless` is the §2.7.1.2 / §3 twin used by the
//! compressed `ALPH` alpha bitstream: the only difference is the 5-byte
//! §3.4 image-header is *not* skipped (the dimensions are already known
//! from the container), so the `BitReader` starts at bit 0 rather than
//! bit 40. Both share `decode_lossless_with_reader`.
//!
//! Sibling harnesses cover the layers *beneath* this driver but **none**
//! drives the assembled §4 transform-list + inverse-transform chain
//! across an attacker-controlled `(width, height, bitstream)`
//! cross-product:
//!
//! * `parse_transform_list` (round 260) drives `TransformList::read` —
//!   the §4 transform-presence loop's *header* fields — but stops *at*
//!   the §5 entropy body, never decoding a transform sub-image, never
//!   decoding the main ARGB image, and never applying an inverse pass.
//! * `inverse_predictor_color` (round 265) +
//!   `inverse_subtract_green_indexing` (round 266) drive the four §4.x
//!   inverse passes in isolation over fuzz-synthesised buffers, never
//!   reading them out of a §4 transform-list bitstream nor sequencing
//!   them in reverse read order.
//! * `decode_argb` (round 272) drives only the main §5.1 ARGB image — it
//!   is invoked *after* the §4 transform list is consumed, so it never
//!   reads the transform list, the §4.4 width subsampling, nor runs the
//!   reverse-order inverse chain.
//! * `decode` / `roundtrip_lossless` reach `decode_lossless` only through
//!   a complete §2 RIFF + §3.4 image-header walk, so the `(width, height)`
//!   pair is bounded by what an upstream §3.4 14-bit field already
//!   accepted and the bitstream is constrained to round-trip from the
//!   encoder; they never drive the raw §4 transform-list surface for the
//!   full attacker-reachable `(width, height, bitstream)` cross-product.
//!
//! This twenty-second harness drives the §4 transform-list + main-image
//! driver directly, in both the headerful and headerless forms.
//!
//! The contract under test, per RFC 9649 §4 + §5 + §6:
//!
//! * Both entry points always return a `Result` — no panic, no
//!   debug-build integer overflow, no out-of-bounds index when the
//!   bitstream is empty, truncated, or arbitrarily long, and no
//!   allocation sized by a field the §4 / §5 / §6 readers did not bound.
//!   Every read goes through [`oxideav_webp::vp8l_stream::BitReader`]
//!   whose EOF path raises a typed `DecodeError`, never an underflow
//!   panic.
//! * If a call returns `Ok(image)`:
//!     * §4 / §6.2.2 dimensions echo the carrier: `image.width() ==
//!       width`, `image.height() == height`. This holds even when a §4.4
//!       color-indexing transform subsampled the *internal* main-image
//!       width — the inverse pass un-bundles back to the canvas `width`.
//!     * §6.2.2 pixel count: the final image carries exactly
//!       `width * height` ARGB pixels in scan-line order, so
//!       `image.pixels().len() == width as usize * height as usize`.
//!     * The success path can only be reached with both dimensions ≥ 1;
//!       the harness clamps both carrier dimensions into `[1, 8]` so the
//!       success contract holds and the decode stays ≤ 64 pixels.
//!     * Determinism: replaying the same bytes + `(width, height)` yields
//!       a byte-identical pixel buffer.
//! * Any `Err(_)` is a §4 / §5 / §6 bitstream-level refusal (a duplicate
//!   §4 transform, a truncated transform sub-image, a meta-prefix /
//!   color-cache-info parse failure, a prefix-code parse failure, an
//!   out-of-range green symbol, or a color-cache / backward-reference
//!   fault); the harness asserts only that the call returned a `Result`
//!   rather than panicking — the granular refusal modes are cross-checked
//!   by the sibling primitive harnesses through their own entry points.
//!
//! Every assertion below is a real §4 / §6.2.2 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — `width`, clamped into `[1, 8]` (`data[0] % 8 + 1`). The
//!   carrier is always nonempty, mirroring the §3.4-validated dimensions
//!   `decode_lossless` is reachable with; the decoded image stays ≤ 64
//!   pixels so the §4 / §5 / §6 decode loop is bounded.
//! * Byte `1` — `height`, clamped into `[1, 8]` likewise.
//! * Bytes `[2..]` — the VP8L chunk-payload bytes. `decode_lossless`
//!   reads them with a `BitReader` positioned *past* the §3.4 5-byte
//!   image-header (so the first ≤ 40 bits are the skipped header and the
//!   §4 transform list begins at bit 40); `decode_lossless_headerless`
//!   reads the *same* bytes from bit 0 (no header skip), as the §2.7.1.2
//!   ALPH bitstream is read. Either short or empty tail raises an EOF
//!   refusal on the first transform-presence bit.
//!
//! ## Iteration cost bound
//!
//! The decoded image is at most `8 × 8 = 64` pixels, so the §5 / §6
//! decode loop is bounded by 64 emitted pixels plus the per-transform
//! §7.3 sub-image decode (each at most a `div_round_up(8, 4) × ...`
//! sub-grid for the smallest `size_bits`, ≤ 64 pixels) and the per-group
//! §3.7 prefix-code-table reads. The §4 loop runs at most four times (one
//! per distinct transform). A single fuzz iteration completes in
//! microseconds to milliseconds regardless of input length; `BitReader`
//! indexes by `usize` across the slice so every read is clamped at the
//! slice end.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_transform::{decode_lossless, decode_lossless_headerless};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // §4 / §6.2.2 carrier dimensions. `decode_lossless` is only ever
    // reached with dimensions an upstream §3.4 14-bit field already
    // validated as ≥ 1, so both are clamped into `[1, 8]`: always
    // nonempty (the success contract holds), and small so the §4 / §5 /
    // §6 decode loop is bounded.
    let width = u32::from(data[0] % 8) + 1;
    let height = u32::from(data[1] % 8) + 1;

    let payload = &data[2..];

    // ---- §4 headerful driver (`decode_lossless`): the `BitReader` is
    // positioned past the §3.4 5-byte image-header, so the first ≤ 40
    // payload bits are the skipped header and the §4 transform list
    // begins at bit 40. This is the form `decode_webp` routes the VP8L
    // chunk through. ----
    let headerful = decode_lossless(payload, width, height);
    if let Ok(ref image) = headerful {
        // §4 / §6.2.2 dimensions echo the carrier verbatim — even after a
        // §4.4 color-indexing transform subsampled the internal width,
        // the inverse pass un-bundles back to the canvas `width`.
        assert_eq!(
            image.width(),
            width,
            "§4 decoded lossless width must equal the carrier width",
        );
        assert_eq!(
            image.height(),
            height,
            "§4 decoded lossless height must equal the carrier height",
        );

        // §6.2.2 pixel count: the final image emits exactly
        // `width * height` ARGB pixels in scan-line order.
        let expected_pixels = (width as usize) * (height as usize);
        assert_eq!(
            image.pixels().len(),
            expected_pixels,
            "§4 decoded lossless image must carry width*height pixels",
        );

        // Determinism: replaying the same bytes + `(width, height)`
        // yields a byte-identical pixel buffer.
        let replay = decode_lossless(payload, width, height)
            .expect("§4 replay of a successful headerful decode must also succeed");
        assert_eq!(
            replay.pixels(),
            image.pixels(),
            "§4 decode_lossless must be deterministic over the same bytes",
        );
    }

    // ---- §2.7.1.2 / §3 headerless driver
    // (`decode_lossless_headerless`): the §2.7.1.2 ALPH form, where the
    // 5-byte image-header is omitted and the `BitReader` starts at bit 0.
    // Same shared `decode_lossless_with_reader`, exercised over the same
    // bytes so the only varied dimension is the header-skip offset. ----
    let headerless = decode_lossless_headerless(payload, width, height);
    if let Ok(ref image) = headerless {
        assert_eq!(
            image.width(),
            width,
            "§3 decoded headerless width must equal the carrier width",
        );
        assert_eq!(
            image.height(),
            height,
            "§3 decoded headerless height must equal the carrier height",
        );

        let expected_pixels = (width as usize) * (height as usize);
        assert_eq!(
            image.pixels().len(),
            expected_pixels,
            "§3 decoded headerless image must carry width*height pixels",
        );

        let replay = decode_lossless_headerless(payload, width, height)
            .expect("§3 replay of a successful headerless decode must also succeed");
        assert_eq!(
            replay.pixels(),
            image.pixels(),
            "§3 decode_lossless_headerless must be deterministic over the same bytes",
        );
    }

    // Both arms' refusals (`Err(_)`) are §4 / §5 / §6 bitstream-level
    // refusals — a duplicate §4 transform, a truncated transform
    // sub-image, a meta-prefix / color-cache-info parse failure, a
    // prefix-code parse failure, an out-of-range green symbol, or a
    // color-cache / backward-reference fault. The granular refusal modes
    // are cross-checked by the sibling primitive harnesses
    // (`parse_transform_list`, `inverse_predictor_color`,
    // `inverse_subtract_green_indexing`, `decode_argb`,
    // `decode_entropy_coded_image`, `parse_meta_prefix`, `distance_code`,
    // `color_cache`, `backward_reference`) through their own entry points;
    // here the contract under test is only that each call returned a
    // `Result` rather than panicking.
    let _ = &headerful;
    let _ = &headerless;
});
