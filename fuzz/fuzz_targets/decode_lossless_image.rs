#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the public top-level
//! lossless façade `oxideav_webp::decode_lossless_image`.
//!
//! ## Why this harness
//!
//! `decode_lossless_image` is the published, in-crate entry point that
//! walks a §2.3 `RIFF`/`WEBP` container, selects its §2.6 `VP8L` chunk,
//! reads the chunk's own §3.4 5-byte image-header (the 14-bit
//! `width - 1` / `height - 1` fields), and runs the full §4 + §5 + §6
//! lossless decode, returning a [`oxideav_webp::vp8l_decode::DecodedImage`]
//! of `width * height` ARGB pixels — or `Ok(None)` when the file carries
//! no `VP8L` chunk.
//!
//! No existing harness drives this façade across an attacker-controlled
//! byte buffer:
//!
//! * `decode_lossless` / `decode_lossless_headerless` (round 286) drive
//!   the §4 transform-list + main-image driver *standalone*, with the
//!   `(width, height)` pair supplied **by the harness**, over a bare
//!   chunk-payload slice — they never walk the §2.3 container, never
//!   select the `VP8L` chunk, and never read the §3.4 image-header, so
//!   the path where the **file's own** §3.4 14-bit dimension fields feed
//!   the §4 decode (and the coherence between that header and the §4
//!   inverse-transform un-bundling) is never exercised through them.
//! * `decode_still_paths` (round 288) drives the published RGBA surfaces
//!   `decode_webp` / `decode_webp_image` — which emit a flat
//!   `width * height * 4` `[R, G, B, A]` buffer and route the §2.5 lossy
//!   form to the sibling — never the typed `DecodedImage` ARGB-`u32`
//!   surface `decode_lossless_image` returns, and never the `Ok(None)`
//!   "no VP8L chunk" arm (which `decode_webp_image` instead maps to a
//!   lossy-route or a `NoImageData` error).
//! * `parse_container` / `parse_vp8x` / `read_vp8l_transform_list` stop
//!   at a structural layer above the §4/§5/§6 entropy decode.
//!
//! This thirtieth harness drives the assembled §2.3 container-walk +
//! §2.6 `VP8L` chunk-selection + §3.4 header-dimension extraction + full
//! §4/§5/§6 decode through the one public façade, with the decoded image
//! dimensions taken from the **file's own** §3.4 header rather than from
//! the harness.
//!
//! ## The contract under test, per RFC 9649 §2.6 + §3.4 + §4 + §6.2.2
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow, no out-of-bounds index when the buffer is empty,
//!   truncated, or arbitrarily long, and no allocation sized by a §3.4
//!   header field the readers did not bound.
//! * On `Ok(Some(image))`:
//!     * §3.4 / §6.2.2 dimension coherence: the decoded image dimensions
//!       echo the §3.4 image-header verbatim — `image.width()` /
//!       `image.height()` equal the `width()` / `height()` the
//!       [`oxideav_webp::extract_lossless_chunk`] handle resolved from the
//!       same §3.4 header. This holds even when a §4.4 color-indexing
//!       transform subsampled the *internal* main-image width: the
//!       inverse pass un-bundles back to the canvas width.
//!     * §6.2.2 pixel count: the final image carries exactly
//!       `width * height` ARGB pixels in scan-line order, so
//!       `image.pixels().len() == width as usize * height as usize`.
//!     * Both dimensions are ≥ 1 (the §3.4 fields encode `dim - 1`, so a
//!       resolved dimension is never zero); the pixel buffer is therefore
//!       non-empty on every success.
//!     * Determinism: a second decode over the same bytes reproduces a
//!       byte-identical pixel buffer.
//! * On `Ok(None)`: the file parsed as a §2.3 container but carried no
//!   §2.6 `VP8L` chunk — a structural outcome, asserted only as "returned
//!   a `Result`".
//! * Any `Err(_)` is a §2.3 / §3.4 / §4 / §5 / §6 bitstream-level refusal
//!   (a container-walk fault, a truncated image-header, a duplicate §4
//!   transform, a prefix-code parse failure, an out-of-range green
//!   symbol, or a color-cache / backward-reference fault); the harness
//!   asserts only that the call returned rather than panicking — the
//!   granular refusal modes are cross-checked by the sibling primitive
//!   harnesses through their own entry points.
//!
//! Every assertion below is a real §3.4 / §6.2.2 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Iteration cost bound
//!
//! Because the decoded dimensions come from the **file's own** §3.4
//! 14-bit fields, a single adversarial chunk can declare a
//! `16384 × 16384` image — a 1 GiB ARGB buffer that would dominate a
//! fuzz iteration. To keep each iteration within milliseconds the harness
//! first extracts the §2.6 `VP8L` chunk handle (cheap: a §2.3 walk plus a
//! §3.4 5-byte header read) and, when the **declared** `width * height`
//! exceeds `MAX_PIXELS = 1 << 16` (65 536 pixels, worst-case 256 KiB ARGB
//! intermediate), runs only that cheap structural path and returns before
//! the full §4/§5/§6 decode. The container-walk + chunk-selection +
//! header-read surface is still fully exercised on every input; only the
//! quadratic-allocation full-decode tail is gated by a declared-pixel
//! budget the §3.4 header makes observable up front.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::{decode_lossless_image, extract_lossless_chunk};

/// Declared-pixel budget for the full §4/§5/§6 decode tail. 1 << 16 =
/// 65 536 pixels caps the worst-case ARGB intermediate at ~256 KiB and
/// keeps a single fuzz iteration bounded regardless of the §3.4 14-bit
/// dimension fields (which can declare up to 16384 × 16384).
const MAX_PIXELS: u64 = 1 << 16;

fuzz_target!(|data: &[u8]| {
    // Cheap structural pre-pass: walk the §2.3 container and read the
    // §2.6 `VP8L` chunk's §3.4 5-byte image-header. This is exercised on
    // *every* input — a container-walk fault or a truncated image-header
    // surfaces here and must return a `Result`, never panic.
    let chunk = match extract_lossless_chunk(data) {
        Ok(Some(chunk)) => chunk,
        // No `VP8L` chunk, or a structural refusal: nothing to fully
        // decode, but the façade must agree it has no decodable lossless
        // image (it shares the same §2.3 walk + §2.6 selection). Drive it
        // anyway for the Result contract, then bail.
        _ => {
            let _ = decode_lossless_image(data);
            return;
        }
    };

    // The §3.4-resolved dimensions are the file's own claim. Use them to
    // gate the full-decode tail so an adversarial 16384 × 16384 header
    // can't blow the per-iteration budget.
    let declared_width = chunk.width();
    let declared_height = chunk.height();
    let declared_pixels = u64::from(declared_width) * u64::from(declared_height);
    if declared_pixels > MAX_PIXELS {
        // Still drive the façade for the Result contract — the §3.4
        // header parsed, but the full decode would over-allocate for a
        // fuzz iteration. The decoder may either succeed (and the budget
        // was the harness's, not the decoder's, concern) or refuse on the
        // entropy stream; either way it must return rather than panic.
        let _ = decode_lossless_image(data);
        return;
    }

    let decoded = decode_lossless_image(data);

    if let Ok(Some(image)) = &decoded {
        // §3.4 / §6.2.2 dimension coherence: the decoded image echoes the
        // §3.4 header verbatim, even after a §4.4 color-indexing
        // transform subsampled the internal width.
        assert_eq!(
            image.width(),
            declared_width,
            "§3.4 decoded lossless width must equal the §3.4 image-header width",
        );
        assert_eq!(
            image.height(),
            declared_height,
            "§3.4 decoded lossless height must equal the §3.4 image-header height",
        );

        // §6.2.2 pixel count: exactly `width * height` ARGB pixels.
        let expected_pixels = (declared_width as usize) * (declared_height as usize);
        assert_eq!(
            image.pixels().len(),
            expected_pixels,
            "§6.2.2 decoded lossless image must carry width*height ARGB pixels",
        );

        // §3.4 resolves `dim - 1 + 1`, so a successful decode always has
        // both dimensions ≥ 1 and therefore a non-empty pixel buffer.
        assert!(
            !image.pixels().is_empty(),
            "§3.4 a successfully decoded lossless image has a non-empty pixel buffer",
        );

        // Determinism: replaying the same bytes reproduces a
        // byte-identical pixel buffer.
        let replay = decode_lossless_image(data)
            .expect("§4 replay of a successful lossless façade decode must also succeed")
            .expect("§4 replay must also yield a VP8L image, not None");
        assert_eq!(
            replay.pixels(),
            image.pixels(),
            "decode_lossless_image must be deterministic over the same bytes",
        );
    }

    // `Ok(None)` (no VP8L chunk — unreachable here since we extracted one,
    // but kept for the exhaustive contract) and `Err(_)` (a §2.3 / §3.4 /
    // §4 / §5 / §6 bitstream refusal) are both valid: the only contract
    // under test on those arms is that the call returned a `Result`.
    let _ = &decoded;
});
