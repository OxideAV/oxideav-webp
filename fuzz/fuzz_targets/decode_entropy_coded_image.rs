#![no_main]

//! Decode arbitrary fuzz-supplied bits through the §7.3
//! *entropy-coded-image* decode path standalone entry point
//! `oxideav_webp::vp8l_decode::decode_entropy_coded_image`.
//!
//! The §7.3 ABNF `entropy-coded-image` is the indivisible building block
//! every VP8L pixel surface is assembled from: a §5.2.3
//! `color-cache-info` bit, a single §6.2 prefix-code group (no §6.2.2
//! meta-prefix bit — that bit belongs to the §5.1 `spatially-coded-image`
//! ARGB role only), and the §5.2 LZ77 / literal / color-cache data that
//! emits exactly `width * height` ARGB pixels in scan-line order.
//! `decode_entropy_coded_image` is the function the §4.1 / §4.2 / §4.4
//! sub-resolution images (predictor / color / color-indexing) and the
//! §6.2.2 entropy image are all decoded through; it is the §7.3 surface
//! immediately beneath the §6.2.2 `decode_entropy_image` the round-270
//! harness drives (`decode_entropy_image` calls *this* function and then
//! folds each pixel's red+green channels into a per-block meta-code).
//!
//! Sibling harnesses cover the layers around this primitive but **none**
//! drives the §7.3 entropy-coded-image decode itself across an
//! attacker-controlled `(width, height, bitstream)` cross-product:
//!
//! * `parse_meta_prefix` (round 261) drives `MetaPrefixHeader::read` —
//!   the §5.2.3 `color-cache-info` + §6.2.2 meta-prefix preamble — but
//!   stops *at* the §5.2 entropy body, never decoding a pixel.
//! * `read_lz77_value` is reached only through the bench, not a harness;
//!   `distance_code` (round 263), `color_cache` (round 264), and
//!   `backward_reference` (round 267) each drive one §5.2 primitive in
//!   isolation, not the assembled §7.3 pixel loop.
//! * `decode_entropy_image` (round 270) drives the §6.2.2 entropy image,
//!   which *wraps* this function for the specific `prefix_image_width ×
//!   prefix_image_height` block grid and then folds the pixels — it does
//!   not expose the raw §7.3 ARGB pixels nor exercise the §7.3 surface
//!   for the full attacker-reachable `(width, height)` cross-product.
//! * `decode` / `roundtrip_lossless` reach §7.3 only through a complete
//!   §2 RIFF + §3 image-header walk, so the `(width, height)` pair is
//!   bounded by what an upstream §3.4 14-bit field already accepted and
//!   the bitstream is constrained to round-trip from the encoder.
//!
//! This twentieth harness drives the §7.3 surface directly.
//!
//! The contract under test, per RFC 9649 §7.3 + §5.2 + §6.2:
//!
//! * `decode_entropy_coded_image` always returns a `Result` — no panic,
//!   no debug-build integer overflow, no out-of-bounds index when the
//!   bitstream is empty, truncated, or arbitrarily long, and no
//!   allocation sized by a header field the §5 / §6.2 readers did not
//!   bound. Every read goes through
//!   [`oxideav_webp::vp8l_stream::BitReader`] whose EOF path raises a
//!   `DecodeError::Eof`/`MetaPrefix`/`Prefix`, never an underflow panic.
//! * If the call returns `Ok(image)`:
//!     * §7.3 dimensions echo the carrier: `image.width() == width`,
//!       `image.height() == height`.
//!     * §7.3 pixel count: the entropy-coded image emits exactly
//!       `width * height` ARGB pixels in scan-line order, so
//!       `image.pixels().len() == width as usize * height as usize`.
//!     * §7.3 the success path can only be reached with both dimensions
//!       ≥ 1 — a zero dimension short-circuits to `EmptyEntropyImage`
//!       before any header bit is read.
//!     * §6.2.2 fold consistency with the round-270 wrapper: folding
//!       each pixel's red+green channels (`(argb >> 8) & 0xffff`) and
//!       handing the result to `decode_entropy_image` for the same bytes
//!       must reproduce the same per-block meta-codes and advance the
//!       reader to the same bit position (the wrapper is a pure fold over
//!       this function's output). Cross-checked.
//!     * Determinism: replaying the same bytes + `(width, height)` yields
//!       a byte-identical pixel buffer advanced to an identical bit
//!       position.
//!     * `BitReader` clamps every read at the slice end, so a successful
//!       decode never advances past the slice's bit length.
//! * If the call returns `Err(DecodeError::EmptyEntropyImage { .. })`,
//!   the harness pins it to the §7.3 degenerate-dimension trigger (at
//!   least one of `width` / `height` is zero) with the offending
//!   dimensions echoed verbatim.
//! * Any other `Err(_)` is a §5.2 / §6.2 bitstream-level refusal
//!   (truncation, prefix-code parse failure, out-of-range green symbol,
//!   color-cache or backward-reference fault); the harness asserts only
//!   that the call returned a `Result` rather than panicking — the
//!   granular refusal modes are cross-checked by the sibling primitive
//!   harnesses through their own entry points.
//!
//! Every assertion below is a real §7.3 / §6.2.2 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — `width`, modulo 9 (`[0, 8]`; 0 reaches the §7.3
//!   degenerate-dimension `EmptyEntropyImage` refusal).
//! * Byte `1` — `height`, modulo 9 (likewise).
//! * Bytes `[2..]` — the §7.3 `entropy-coded-image` bit sequence read by
//!   a zero-positioned `BitReader`. A short or empty tail raises an EOF
//!   refusal on the first §5.2.3 color-cache-info bit.
//!
//! ## Iteration cost bound
//!
//! The decoded image is at most `8 × 8 = 64` pixels, so the §5.2 / §6.2
//! decode loop is bounded by 64 emitted pixels plus the per-group §3.7
//! prefix-code-table reads (the 19-symbol code-length alphabet plus the
//! per-code alphabets). A single fuzz iteration completes in microseconds
//! to milliseconds regardless of input length; `BitReader` indexes by
//! `usize` across the slice so every read is clamped at the slice end.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_decode::{decode_entropy_coded_image, decode_entropy_image, DecodeError};
use oxideav_webp::vp8l_stream::BitReader;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // §7.3 carrier dimensions. Kept small so the decode loop is bounded;
    // 0 reaches the degenerate-dimension `EmptyEntropyImage` refusal.
    let width = u32::from(data[0] % 9);
    let height = u32::from(data[1] % 9);

    let payload = &data[2..];
    let total_bits = payload.len() * 8;

    let mut reader = BitReader::new(payload);
    let result = decode_entropy_coded_image(&mut reader, width, height);

    match result {
        Ok(image) => {
            // §7.3 success is only reachable with both dimensions ≥ 1 (a
            // zero dimension short-circuits to `EmptyEntropyImage` before
            // any header bit is read).
            assert!(
                width >= 1 && height >= 1,
                "§7.3 decode_entropy_coded_image success implies a nonempty {width}x{height} image",
            );

            // §7.3 dimensions echo the carrier verbatim.
            assert_eq!(
                image.width(),
                width,
                "§7.3 decoded width must equal the carrier width",
            );
            assert_eq!(
                image.height(),
                height,
                "§7.3 decoded height must equal the carrier height",
            );

            // §7.3 pixel count: the entropy-coded image emits exactly
            // `width * height` ARGB pixels in scan-line order.
            let expected_pixels = (width as usize) * (height as usize);
            assert_eq!(
                image.pixels().len(),
                expected_pixels,
                "§7.3 entropy-coded image must carry width*height pixels",
            );

            // `BitReader` clamps every read at the slice end, so a
            // successful decode cannot have walked past the slice.
            assert!(
                reader.bit_position() <= total_bits,
                "§7.3 successful decode advanced to bit {} beyond slice bit length {total_bits}",
                reader.bit_position(),
            );

            // Determinism: replaying the same bytes + `(width, height)`
            // yields a byte-identical pixel buffer advanced to an
            // identical bit position.
            let mut replay_reader = BitReader::new(payload);
            let replay = decode_entropy_coded_image(&mut replay_reader, width, height)
                .expect("§7.3 replay of a successful decode must also succeed");
            assert_eq!(
                replay.pixels(),
                image.pixels(),
                "§7.3 decode_entropy_coded_image must be deterministic over the same bytes",
            );
            assert_eq!(
                replay_reader.bit_position(),
                reader.bit_position(),
                "§7.3 replay must advance the reader identically",
            );

            // §6.2.2 fold consistency with the round-270 wrapper. The
            // §6.2.2 entropy-image decoder `decode_entropy_image` calls
            // *this* function for the same `(width, height)` block grid
            // and then folds each pixel's red+green channels into a
            // per-block meta-code (`(entropy_pixel >> 8) & 0xffff`). The
            // wrapper is a pure fold over this function's output, so for
            // the same bytes it must reproduce the per-pixel fold of the
            // pixels decoded here and advance the reader identically.
            // (`prefix_bits` is an opaque carrier the wrapper records but
            // never re-derives a block size from; any in-range value
            // works — 2 lands inside the §6.2.2 `[2, 9]` wire window.)
            let expected_codes: Vec<u16> = image
                .pixels()
                .iter()
                .map(|&argb| ((argb >> 8) & 0xffff) as u16)
                .collect();
            let mut wrapper_reader = BitReader::new(payload);
            let wrapped = decode_entropy_image(&mut wrapper_reader, 2, width, height)
                .expect("§6.2.2 entropy-image wrapper over a successful §7.3 decode must succeed");
            assert_eq!(
                wrapped.meta_codes(),
                &expected_codes[..],
                "§6.2.2 wrapper meta-codes must be the (pixel >> 8) & 0xffff fold of the §7.3 pixels",
            );
            assert_eq!(
                wrapper_reader.bit_position(),
                reader.bit_position(),
                "§6.2.2 wrapper and §7.3 decode must advance the reader identically",
            );
            assert_eq!(
                wrapped.block_width(),
                width,
                "§6.2.2 wrapper block_width must equal the §7.3 width",
            );
            assert_eq!(
                wrapped.block_height(),
                height,
                "§6.2.2 wrapper block_height must equal the §7.3 height",
            );
        }
        Err(DecodeError::EmptyEntropyImage {
            prefix_image_width: w,
            prefix_image_height: h,
        }) => {
            // §7.3 degenerate-dimension refusal: reached iff at least one
            // dimension is zero, with the offending dimensions echoed
            // verbatim.
            assert_eq!(
                (w, h),
                (width, height),
                "§7.3 EmptyEntropyImage must echo the carrier dimensions",
            );
            assert!(
                width == 0 || height == 0,
                "§7.3 EmptyEntropyImage must imply a zero-dimension {width}x{height} image",
            );
        }
        Err(_) => {
            // §5.2 / §6.2: a bitstream-level refusal (truncated input,
            // prefix-code parse failure, out-of-range green symbol,
            // color-cache or backward-reference fault). The granular
            // refusal modes are cross-checked by the sibling primitive
            // harnesses (`parse_meta_prefix`, `distance_code`,
            // `color_cache`, `backward_reference`) through their own entry
            // points; here the contract under test is only that the call
            // returned a `Result` rather than panicking.
        }
    }
});
