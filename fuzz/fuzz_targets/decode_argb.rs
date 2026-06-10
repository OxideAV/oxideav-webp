#![no_main]

//! Decode arbitrary fuzz-supplied bits through the §6.2.2 top-level VP8L
//! ARGB main-image decode path standalone entry point
//! `oxideav_webp::vp8l_decode::decode_argb`.
//!
//! `decode_argb` is the §5.1 `spatially-coded-image` ARGB-role decoder —
//! the layer immediately *above* the round-270 §6.2.2
//! `decode_entropy_image` and the round-271 §7.3
//! `decode_entropy_coded_image`. It reads the round-106
//! `MetaPrefixHeader` for `ImageRole::Argb` (the §5.2.3
//! `color-cache-info` bit, then the §6.2.2 meta-prefix bit) and
//! dispatches:
//!
//! * **single group** (`meta-prefix = %b0`): one §6.2 prefix-code group
//!   drives the §6.2.3 decode loop everywhere.
//! * **multiple groups** (`meta-prefix = %b1`): the §6.2.2 entropy image
//!   is decoded (`decode_entropy_image`), `num_prefix_groups =
//!   max(entropy image) + 1` groups are read, and the §6.2.3 loop selects
//!   a group per pixel block via `MetaPrefixIndex::meta_code_for`. A
//!   single §5.2.3 color cache is maintained in stream order across the
//!   whole image, independent of which group emitted the pixel.
//!
//! Sibling harnesses cover the layers around this entry point but **none**
//! drives the assembled §6.2.2 ARGB image decode across an
//! attacker-controlled `(width, height, bitstream)` cross-product:
//!
//! * `parse_meta_prefix` (round 261) drives `MetaPrefixHeader::read` —
//!   the §5.2.3 + §6.2.2 preamble `decode_argb` reads first — but stops
//!   *at* the §5.2 entropy body, never decoding a pixel and never reading
//!   the per-group prefix-code groups nor running the §6.2.3 loop.
//! * `decode_entropy_image` (round 270) drives only the §6.2.2 entropy
//!   *sub*-image (the meta-code grid), and `decode_entropy_coded_image`
//!   (round 271) drives only the §7.3 building block beneath it — neither
//!   reads the §5.2.3 + §6.2.2 ARGB preamble, the single-vs-multi-group
//!   dispatch, the per-group prefix-code-group reads, nor the
//!   per-pixel-block group selection that `decode_argb` assembles on top
//!   of them.
//! * `decode` / `roundtrip_lossless` reach `decode_argb` only through a
//!   complete §2 RIFF + §3 image-header walk, so the `(width, height)`
//!   pair is bounded by what an upstream §3.4 14-bit field already
//!   accepted and the bitstream is constrained to round-trip from the
//!   encoder; they never drive the raw §6.2.2 ARGB surface for the full
//!   attacker-reachable `(width, height, bitstream)` cross-product.
//!
//! This twenty-first harness drives the §6.2.2 ARGB surface directly.
//!
//! The contract under test, per RFC 9649 §6.2.2 + §5.2 + §5.1:
//!
//! * `decode_argb` always returns a `Result` — no panic, no debug-build
//!   integer overflow, no out-of-bounds index when the bitstream is
//!   empty, truncated, or arbitrarily long, and no allocation sized by a
//!   header field the §5 / §6.2 readers did not bound. Every read goes
//!   through [`oxideav_webp::vp8l_stream::BitReader`] whose EOF path
//!   raises a typed `DecodeError`, never an underflow panic.
//! * If the call returns `Ok(image)`:
//!     * §6.2.2 dimensions echo the carrier: `image.width() == width`,
//!       `image.height() == height`.
//!     * §6.2.2 pixel count: the ARGB image carries exactly
//!       `width * height` ARGB pixels in scan-line order, so
//!       `image.pixels().len() == width as usize * height as usize`.
//!     * The success path can only be reached with both dimensions ≥ 1.
//!       `decode_argb` is only ever invoked with dimensions an upstream
//!       §3.4 field already validated as ≥ 1; the harness clamps both
//!       carrier dimensions into `[1, 8]` so the success contract holds.
//!     * Determinism: replaying the same bytes + `(width, height)` yields
//!       a byte-identical pixel buffer advanced to an identical bit
//!       position.
//!     * `BitReader` clamps every read at the slice end, so a successful
//!       decode never advances past the slice's bit length.
//! * Any `Err(_)` is a §5.2 / §6.2 bitstream-level refusal (truncation,
//!   meta-prefix/color-cache-info parse failure, entropy-image fault,
//!   prefix-code parse failure, out-of-range green symbol, color-cache or
//!   backward-reference fault, or a meta-prefix code beyond
//!   `num_prefix_groups`); the harness asserts only that the call
//!   returned a `Result` rather than panicking — the granular refusal
//!   modes are cross-checked by the sibling primitive harnesses through
//!   their own entry points.
//!
//! Every assertion below is a real §6.2.2 / §5.1 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Input layout
//!
//! * Byte `0` — `width`, clamped into `[1, 8]` (`data[0] % 8 + 1`). The
//!   carrier is always nonempty, mirroring the §3.4-validated dimensions
//!   `decode_argb` is reachable with; the decoded image stays ≤ 64 pixels
//!   so the §6.2.3 decode loop is bounded.
//! * Byte `1` — `height`, clamped into `[1, 8]` likewise.
//! * Bytes `[2..]` — the §6.2.2 ARGB image bit sequence read by a
//!   zero-positioned `BitReader`: the §5.2.3 `color-cache-info` bit, the
//!   §6.2.2 meta-prefix bit, then the body the dispatch selects. A short
//!   or empty tail raises an EOF refusal on the first preamble bit.
//!
//! ## Iteration cost bound
//!
//! The decoded image is at most `8 × 8 = 64` pixels, so the §5.2 / §6.2
//! decode loop is bounded by 64 emitted pixels plus the per-group §3.7
//! prefix-code-table reads and (on the multi-group path) the §6.2.2
//! entropy sub-image of the same ≤ 64-pixel bound. A single fuzz
//! iteration completes in microseconds to milliseconds regardless of
//! input length; `BitReader` indexes by `usize` across the slice so every
//! read is clamped at the slice end.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::vp8l_decode::decode_argb;
use oxideav_webp::vp8l_stream::BitReader;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // §6.2.2 carrier dimensions. `decode_argb` is only ever reached with
    // dimensions an upstream §3.4 14-bit field already validated as ≥ 1,
    // so both are clamped into `[1, 8]`: always nonempty (the success
    // contract holds), and small so the §6.2.3 decode loop is bounded.
    let width = u32::from(data[0] % 8) + 1;
    let height = u32::from(data[1] % 8) + 1;

    let payload = &data[2..];
    let total_bits = payload.len() * 8;

    let mut reader = BitReader::new(payload);
    let result = decode_argb(&mut reader, width, height);

    match result {
        Ok(image) => {
            // §6.2.2 dimensions echo the carrier verbatim.
            assert_eq!(
                image.width(),
                width,
                "§6.2.2 decoded ARGB width must equal the carrier width",
            );
            assert_eq!(
                image.height(),
                height,
                "§6.2.2 decoded ARGB height must equal the carrier height",
            );

            // §6.2.2 pixel count: the ARGB image emits exactly
            // `width * height` ARGB pixels in scan-line order.
            let expected_pixels = (width as usize) * (height as usize);
            assert_eq!(
                image.pixels().len(),
                expected_pixels,
                "§6.2.2 decoded ARGB image must carry width*height pixels",
            );

            // `BitReader` clamps every read at the slice end, so a
            // successful decode cannot have walked past the slice.
            assert!(
                reader.bit_position() <= total_bits,
                "§6.2.2 successful decode advanced to bit {} beyond slice bit length {total_bits}",
                reader.bit_position(),
            );

            // Determinism: replaying the same bytes + `(width, height)`
            // yields a byte-identical pixel buffer advanced to an
            // identical bit position.
            let mut replay_reader = BitReader::new(payload);
            let replay = decode_argb(&mut replay_reader, width, height)
                .expect("§6.2.2 replay of a successful ARGB decode must also succeed");
            assert_eq!(
                replay.pixels(),
                image.pixels(),
                "§6.2.2 decode_argb must be deterministic over the same bytes",
            );
            assert_eq!(
                replay_reader.bit_position(),
                reader.bit_position(),
                "§6.2.2 replay must advance the reader identically",
            );
        }
        Err(_) => {
            // §5.2 / §6.2: a bitstream-level refusal (truncated input,
            // meta-prefix/color-cache-info parse failure, entropy-image
            // fault, prefix-code parse failure, out-of-range green symbol,
            // color-cache or backward-reference fault, or a meta-prefix
            // code beyond `num_prefix_groups`). The granular refusal modes
            // are cross-checked by the sibling primitive harnesses
            // (`parse_meta_prefix`, `decode_entropy_image`,
            // `decode_entropy_coded_image`, `distance_code`, `color_cache`,
            // `backward_reference`, `meta_prefix_index`) through their own
            // entry points; here the contract under test is only that the
            // call returned a `Result` rather than panicking.
        }
    }
});
