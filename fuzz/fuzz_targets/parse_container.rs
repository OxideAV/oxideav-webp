#![no_main]

//! Walk arbitrary fuzz-supplied bytes through the §2.3 / §2.4 RIFF/WEBP
//! chunk-walker standalone entry point `oxideav_webp::container::parse`.
//!
//! `container::parse` is the structural layer beneath every other WebP
//! entry point. It enforces RFC 9649 §2.3 (generic RIFF chunk: 4-byte
//! FourCC + 4-byte little-endian uint32 `Size` + payload + optional
//! 1-byte pad when `Size` is odd) and §2.4 (the 12-byte WebP file
//! header: `RIFF` + 4-byte little-endian `File Size` + `WEBP`), and
//! emits the ordered chunk list any §2.5 / §2.6 / §2.7 decode path
//! consumes downstream. The walker is *non-recovering*: it surfaces
//! the first structural problem it sees and stops.
//!
//! Every byte fed to the walker is attacker-controlled: the `File
//! Size` field at bytes 4..8, each chunk's `Size` field at offsets
//! `+4..+8` relative to its header, and the §2.3 pad byte whose
//! absence at the boundary of an odd-`Size` chunk triggers
//! `MissingPadByte`. The walker must never panic, never debug-build
//! integer-overflow, never index out of bounds, and never allocate a
//! `Vec<WebpChunk>` whose capacity is driven by an unbounded header
//! field. The pre-existing `decode` and `extract_metadata` harnesses
//! both wrap this walker — but their `WebpError` envelope flattens the
//! granular §2.3 / §2.4 refusal modes into a coarser shape and never
//! cross-checks the on-disk `(payload_start, payload_end)` ranges
//! against the original buffer. This harness drives the structural
//! walker directly and cross-checks every recorded chunk byte-for-byte.
//!
//! Sibling harnesses cover the layers *above* the walker — `parse_vp8x`
//! (§2.7.1 Figure 7 octet), `parse_anmf` (§2.7.1.1 Figure 9 header),
//! `parse_anim` (§2.7.1.1 Figure 8 carrier), `parse_alph` (§2.7.1.2
//! Figure 10 info byte), `decode_alph` (§2.7.1.2 alpha plane),
//! `parse_transform_list` (§4 VP8L transform-list reader),
//! `parse_meta_prefix` (§5.2.3 + §6.2.2 + §6.2 preamble),
//! `extract_metadata` (§2 RIFF walk for ICCP/EXIF/XMP), `decode` (full
//! §2 RIFF + §3..§5 entry), and `roundtrip_animated` /
//! `roundtrip_lossless` (encode→decode equality oracles). This twelfth
//! harness widens fuzz coverage onto the §2.3 + §2.4 RIFF chunk-walker
//! itself, the lowest-level structural surface every other path is
//! layered on top of, with attacker-controlled chunk-`Size` fields the
//! exact §2.3 hostility vector the brief calls out.
//!
//! The contract under test, per RFC 9649 §2.3 + §2.4:
//!
//! * The call must always return a `Result` — no panic, no debug-build
//!   integer overflow, no out-of-bounds index when the input is empty
//!   or arbitrarily long, no `Vec` allocation sized by an unbounded
//!   header field.
//! * If the call returns `Ok(container)`:
//!     * `container.riff_file_size` equals the little-endian uint32 at
//!       bytes 4..8 of the input.
//!     * For every recorded `WebpChunk`:
//!         * `chunk.payload_start` minus 8 is a valid offset inside the
//!           buffer (the 8 chunk-header bytes precede the payload, and
//!           the FourCC + LE uint32 at that header equal
//!           `chunk.fourcc` / `chunk.size`).
//!         * `chunk.payload_end - chunk.payload_start == chunk.size as
//!           usize` (the recorded range matches the declared length).
//!         * `chunk.payload_end <= buf.len()` and `chunk.payload_end
//!           <= 8 + riff_file_size as usize` (the recorded payload
//!           lies inside both the input buffer and the §2.4 declared
//!           RIFF payload window).
//!         * `chunk.payload(buf).len() == chunk.size as usize` (the
//!           helper accessor agrees with the declared length).
//!         * The convenience predicates `is_vp8_lossy`, `is_vp8_lossless`,
//!           `is_extended` agree with the FourCC.
//!     * Chunks are emitted in on-disk order —
//!       `chunks[i+1].payload_start - 8` is strictly greater than
//!       `chunks[i].payload_start - 8`, and `chunks[i+1].payload_start -
//!       8 >= chunks[i].payload_end + (chunks[i].size & 1) as usize`
//!       (the §2.3 pad byte sits between successive chunks when the
//!       predecessor's `Size` was odd).
//!     * `container.is_extended()` matches the first chunk's `is_extended()`.
//!     * The iterator helpers `chunks_with_fourcc` and
//!       `first_chunk_with_fourcc` agree with the recorded `chunks` list.
//! * If the call returns `Err(TooShortForHeader { got })`, `got` equals
//!   the buffer length and is strictly less than 12.
//! * If the call returns `Err(NotRiff { got })`, `got` equals `buf[0..4]`
//!   and is not `b"RIFF"` (the §2.4 opening tag refusal).
//! * If the call returns `Err(NotWebp { got })`, `got` equals `buf[8..12]`,
//!   bytes 0..4 *are* `b"RIFF"`, and `got` is not `b"WEBP"` (the §2.4
//!   form-type refusal).
//! * If the call returns `Err(RiffSizeOverflowsBuffer { declared,
//!   buffer_len })`, `declared` equals the LE uint32 at bytes 4..8,
//!   `buffer_len` equals the input length, and `8 + declared as usize`
//!   strictly exceeds `buffer_len` (the §2.4 `File Size` overflow
//!   refusal).
//! * If the call returns `Err(TruncatedChunkHeader { offset })`,
//!   `offset >= 12` (no chunk header begins inside the §2.4 file
//!   header), and `(8 + riff_file_size as usize) - offset < 8` (fewer
//!   than 8 bytes remain in the declared RIFF payload window for the
//!   FourCC + Size field).
//! * If the call returns `Err(ChunkPayloadOverflowsRiff { offset,
//!   declared, available })`, `offset >= 12`, `offset + 8 <= 8 +
//!   riff_file_size as usize` (the 8-byte header itself fit), `declared`
//!   equals the LE uint32 at `buf[offset+4..offset+8]`, `available`
//!   equals `(8 + riff_file_size as usize) - (offset + 8)`, and
//!   `declared as usize > available` (the §2.3 declared payload
//!   strictly exceeds the §2.4 declared remaining RIFF window).
//! * If the call returns `Err(MissingPadByte { offset })`, `offset >=
//!   12`, the chunk's declared `Size` at `buf[offset+4..offset+8]` is
//!   odd, and the §2.3 pad byte that would follow at
//!   `offset + 8 + size as usize` lies outside the declared RIFF payload
//!   window.
//!
//! Every assertion below is a real §2.3 / §2.4 carrier violation if it
//! ever fires; a panic short-circuits to libFuzzer.
//!
//! ## Iteration cost bound
//!
//! Each iteration of `container::parse`'s chunk loop reads exactly 8
//! header bytes, then advances the cursor by `8 + size + (size & 1)`.
//! With `size` clamped by the §2.4 declared payload window the loop
//! terminates in at most `declared_payload / 8` iterations — a 4 GiB
//! declared RIFF could iterate ~512 M times in the worst case, but the
//! `declared_payload > buf.len()` precondition fails first
//! (RiffSizeOverflowsBuffer), so the bounded iteration is in practice
//! capped at `buf.len() / 8`. A single fuzz iteration completes in
//! microseconds at the per-iteration libFuzzer 4 KiB default and in
//! milliseconds even at the 64 KiB limit.
//!
//! ## Input layout
//!
//! The entire fuzz buffer is forwarded verbatim to `container::parse` —
//! every byte feeds the walker (no per-harness rebias). This is the
//! lowest-level structural surface, so the lowest-level fuzz envelope
//! is the right shape: pure byte-in, structured-result-out.

use libfuzzer_sys::fuzz_target;
use oxideav_webp::container::{fourcc, parse, ContainerError, WebpContainer};

fuzz_target!(|data: &[u8]| {
    match parse(data) {
        Ok(container) => check_ok(data, &container),
        Err(err) => check_err(data, &err),
    }
});

/// Cross-check every invariant of a successful §2.3 / §2.4 walk
/// against the original input bytes.
fn check_ok(buf: &[u8], container: &WebpContainer) {
    // §2.4: the buffer must have been at least 12 bytes for an Ok
    // return — the walker enters the chunk loop only after the file
    // header.
    assert!(
        buf.len() >= 12,
        "§2.4 file header is 12 bytes; Ok return implies buf.len() {} >= 12",
        buf.len(),
    );

    // §2.4: bytes 0..4 are 'RIFF', bytes 8..12 are 'WEBP'.
    assert_eq!(
        &buf[0..4],
        &fourcc::RIFF,
        "§2.4 Ok return must have buf[0..4] == 'RIFF'",
    );
    assert_eq!(
        &buf[8..12],
        &fourcc::WEBP,
        "§2.4 Ok return must have buf[8..12] == 'WEBP'",
    );

    // §2.4: `riff_file_size` is the LE uint32 at bytes 4..8.
    let header_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    assert_eq!(
        container.riff_file_size, header_size,
        "§2.4 riff_file_size {} must equal LE uint32 at buf[4..8] = {}",
        container.riff_file_size, header_size,
    );

    // §2.4: the declared payload window ends at `8 + riff_file_size`
    // and must fit inside the buffer for an Ok return.
    let declared_end = 8usize
        .checked_add(container.riff_file_size as usize)
        .expect("§2.4 declared_end fits in usize on 64-bit platforms");
    assert!(
        declared_end <= buf.len(),
        "§2.4 declared payload end {declared_end} exceeded buffer length {} for Ok return",
        buf.len(),
    );

    // §2.3 per-chunk invariants. Walk the recorded list and check each
    // entry against the buffer bytes it points into.
    let mut last_header_offset: Option<usize> = None;
    let mut last_end_with_pad: usize = 12; // first chunk starts at 12
    for chunk in &container.chunks {
        // §2.3: the 8 chunk-header bytes precede the payload.
        let header_offset = chunk
            .payload_start
            .checked_sub(8)
            .expect("§2.3 payload_start >= 8 because the 8-byte chunk header precedes the payload");

        // §2.3: `chunk.fourcc` equals `buf[header_offset..+4]`.
        assert!(
            header_offset + 8 <= buf.len(),
            "§2.3 chunk header at offset {header_offset} extends past buffer length {}",
            buf.len(),
        );
        let on_disk_fourcc = &buf[header_offset..header_offset + 4];
        assert_eq!(
            on_disk_fourcc,
            &chunk.fourcc,
            "§2.3 recorded fourcc must equal LE bytes at buf[{header_offset}..{}]",
            header_offset + 4,
        );

        // §2.3: `chunk.size` equals the LE uint32 at `buf[header_offset
        // + 4..+8]`.
        let on_disk_size = u32::from_le_bytes([
            buf[header_offset + 4],
            buf[header_offset + 5],
            buf[header_offset + 6],
            buf[header_offset + 7],
        ]);
        assert_eq!(
            chunk.size,
            on_disk_size,
            "§2.3 recorded size {} must equal LE uint32 at buf[{}..{}] = {}",
            chunk.size,
            header_offset + 4,
            header_offset + 8,
            on_disk_size,
        );

        // §2.3: `payload_end - payload_start == size as usize`.
        let recorded_len = chunk
            .payload_end
            .checked_sub(chunk.payload_start)
            .expect("§2.3 payload_end >= payload_start");
        assert_eq!(
            recorded_len, chunk.size as usize,
            "§2.3 recorded payload range length {recorded_len} must equal size {}",
            chunk.size,
        );

        // §2.4: the recorded payload range lies inside both the buffer
        // and the declared RIFF payload window.
        assert!(
            chunk.payload_end <= buf.len(),
            "§2.3 recorded payload_end {} must lie inside buffer length {}",
            chunk.payload_end,
            buf.len(),
        );
        assert!(
            chunk.payload_end <= declared_end,
            "§2.4 recorded payload_end {} must lie inside declared RIFF window end {declared_end}",
            chunk.payload_end,
        );

        // §2.3: the `payload()` accessor agrees with the recorded
        // length.
        let payload = chunk.payload(buf);
        assert_eq!(
            payload.len(),
            chunk.size as usize,
            "§2.3 payload() length must equal recorded size",
        );

        // §2.3 convenience predicates: each is a pure function of the
        // FourCC.
        assert_eq!(
            chunk.is_vp8_lossy(),
            chunk.fourcc == fourcc::VP8,
            "§2.3 is_vp8_lossy() must equal fourcc == 'VP8 '",
        );
        assert_eq!(
            chunk.is_vp8_lossless(),
            chunk.fourcc == fourcc::VP8L,
            "§2.3 is_vp8_lossless() must equal fourcc == 'VP8L'",
        );
        assert_eq!(
            chunk.is_extended(),
            chunk.fourcc == fourcc::VP8X,
            "§2.3 is_extended() must equal fourcc == 'VP8X'",
        );

        // §2.3 walker ordering: chunk headers must appear strictly
        // forward in the buffer. The next header sits immediately
        // after the current chunk's payload + (size & 1) pad byte.
        if let Some(prev_offset) = last_header_offset {
            assert!(
                header_offset > prev_offset,
                "§2.3 chunk header at {header_offset} must be strictly past previous chunk header at {prev_offset}",
            );
        }
        assert_eq!(
            header_offset, last_end_with_pad,
            "§2.3 chunk header at {header_offset} must immediately follow previous chunk + pad at {last_end_with_pad}",
        );
        last_header_offset = Some(header_offset);
        last_end_with_pad = chunk
            .payload_end
            .checked_add((chunk.size & 1) as usize)
            .expect("§2.3 payload_end + pad fits in usize");
        assert!(
            last_end_with_pad <= declared_end,
            "§2.3 chunk + pad end {last_end_with_pad} must lie inside declared RIFF window end {declared_end}",
        );
    }

    // §2.7 dispatch hint: `is_extended()` matches the first chunk's
    // FourCC (or false on an empty chunk list).
    let expected_extended = container
        .chunks
        .first()
        .map(|c| c.fourcc == fourcc::VP8X)
        .unwrap_or(false);
    assert_eq!(
        container.is_extended(),
        expected_extended,
        "§2.7 is_extended() must match first chunk's fourcc == 'VP8X'",
    );

    // Iterator helpers: `chunks_with_fourcc` is a filter over the
    // recorded list, `first_chunk_with_fourcc` is its `.next()`.
    for fc in [
        fourcc::VP8X,
        fourcc::VP8,
        fourcc::VP8L,
        fourcc::ALPH,
        fourcc::ANIM,
        fourcc::ANMF,
        fourcc::ICCP,
        fourcc::EXIF,
        fourcc::XMP,
    ] {
        let manual: Vec<_> = container.chunks.iter().filter(|c| c.fourcc == fc).collect();
        let helper: Vec<_> = container.chunks_with_fourcc(fc).collect();
        assert_eq!(
            manual.len(),
            helper.len(),
            "§2.7 chunks_with_fourcc({:?}) count must equal manual filter count",
            fc,
        );
        for (m, h) in manual.iter().zip(helper.iter()) {
            assert_eq!(
                *m, *h,
                "§2.7 chunks_with_fourcc({:?}) item must equal manual filter item",
                fc,
            );
        }
        let first_helper = container.first_chunk_with_fourcc(fc);
        assert_eq!(
            first_helper,
            manual.first().copied(),
            "§2.7 first_chunk_with_fourcc({:?}) must equal first manual filter item",
            fc,
        );
    }
}

/// Cross-check every refusal-mode variant against the buffer state
/// that should have triggered it.
fn check_err(buf: &[u8], err: &ContainerError) {
    match err {
        ContainerError::TooShortForHeader { got } => {
            // §2.4: `got` is the buffer length and is < 12.
            assert_eq!(
                *got,
                buf.len(),
                "§2.4 TooShortForHeader.got {} must equal buf.len() {}",
                got,
                buf.len(),
            );
            assert!(
                *got < 12,
                "§2.4 TooShortForHeader.got {} must be < 12 (file header is 12 bytes)",
                got,
            );
        }
        ContainerError::NotRiff { got } => {
            // §2.4: `got` equals `buf[0..4]` and is not 'RIFF'. The
            // buffer is at least 12 bytes (the walker checked length
            // before reading the tag).
            assert!(
                buf.len() >= 12,
                "§2.4 NotRiff implies buf.len() {} >= 12",
                buf.len(),
            );
            assert_eq!(got, &buf[0..4], "§2.4 NotRiff.got must equal buf[0..4]",);
            assert_ne!(
                got,
                &fourcc::RIFF,
                "§2.4 NotRiff.got must not equal 'RIFF' (would have been Ok)",
            );
        }
        ContainerError::NotWebp { got } => {
            // §2.4: bytes 0..4 *are* 'RIFF' (the walker reached the
            // form-type check), and `got` equals `buf[8..12]` and is
            // not 'WEBP'.
            assert!(
                buf.len() >= 12,
                "§2.4 NotWebp implies buf.len() {} >= 12",
                buf.len(),
            );
            assert_eq!(
                &buf[0..4],
                &fourcc::RIFF,
                "§2.4 NotWebp implies buf[0..4] == 'RIFF' (walker reached form-type check)",
            );
            assert_eq!(got, &buf[8..12], "§2.4 NotWebp.got must equal buf[8..12]",);
            assert_ne!(
                got,
                &fourcc::WEBP,
                "§2.4 NotWebp.got must not equal 'WEBP' (would have been Ok)",
            );
        }
        ContainerError::RiffSizeOverflowsBuffer {
            declared,
            buffer_len,
        } => {
            // §2.4: the walker reached the declared-size check, so
            // bytes 0..4 == 'RIFF' and bytes 8..12 == 'WEBP'. `declared`
            // is the LE uint32 at bytes 4..8.
            assert!(
                buf.len() >= 12,
                "§2.4 RiffSizeOverflowsBuffer implies buf.len() {} >= 12",
                buf.len(),
            );
            assert_eq!(
                &buf[0..4],
                &fourcc::RIFF,
                "§2.4 RiffSizeOverflowsBuffer implies buf[0..4] == 'RIFF'",
            );
            assert_eq!(
                &buf[8..12],
                &fourcc::WEBP,
                "§2.4 RiffSizeOverflowsBuffer implies buf[8..12] == 'WEBP'",
            );
            let header_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
            assert_eq!(
                *declared, header_size,
                "§2.4 RiffSizeOverflowsBuffer.declared {} must equal LE uint32 at buf[4..8] {}",
                declared, header_size,
            );
            assert_eq!(
                *buffer_len,
                buf.len(),
                "§2.4 RiffSizeOverflowsBuffer.buffer_len {} must equal buf.len() {}",
                buffer_len,
                buf.len(),
            );
            // The refusal trigger: `8 + declared` strictly exceeds
            // `buffer_len`.
            let needed = 8u64 + u64::from(*declared);
            assert!(
                needed > *buffer_len as u64,
                "§2.4 RiffSizeOverflowsBuffer requires 8 + declared {} > buffer_len {}",
                needed,
                buffer_len,
            );
        }
        ContainerError::TruncatedChunkHeader { offset } => {
            // §2.3: a chunk header started at `offset` with fewer than
            // 8 bytes remaining in the declared RIFF payload window.
            // The walker only enters the chunk loop after the §2.4
            // header passes, so `offset >= 12`.
            assert!(
                *offset >= 12,
                "§2.3 TruncatedChunkHeader.offset {offset} must be >= 12 (chunks follow §2.4 header)",
            );
            assert!(
                buf.len() >= 12,
                "§2.3 TruncatedChunkHeader implies buf.len() {} >= 12",
                buf.len(),
            );
            assert_eq!(
                &buf[0..4],
                &fourcc::RIFF,
                "§2.3 TruncatedChunkHeader implies buf[0..4] == 'RIFF'",
            );
            assert_eq!(
                &buf[8..12],
                &fourcc::WEBP,
                "§2.3 TruncatedChunkHeader implies buf[8..12] == 'WEBP'",
            );
            let header_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
            let declared_end = 8usize
                .checked_add(header_size as usize)
                .expect("§2.4 declared_end fits in usize on 64-bit");
            assert!(
                declared_end <= buf.len(),
                "§2.3 TruncatedChunkHeader implies declared RIFF window {declared_end} fits in buffer {} (else RiffSizeOverflowsBuffer)",
                buf.len(),
            );
            assert!(
                *offset <= declared_end,
                "§2.3 TruncatedChunkHeader.offset {offset} must lie inside declared RIFF window end {declared_end}",
            );
            let remaining = declared_end - *offset;
            assert!(
                remaining < 8,
                "§2.3 TruncatedChunkHeader requires fewer than 8 bytes remaining at offset {offset} (got {remaining})",
            );
        }
        ContainerError::ChunkPayloadOverflowsRiff {
            offset,
            declared,
            available,
        } => {
            // §2.3: a chunk header at `offset` parsed cleanly (the
            // 8-byte header fit), but its declared payload size
            // exceeds the remaining declared RIFF payload window.
            assert!(
                *offset >= 12,
                "§2.3 ChunkPayloadOverflowsRiff.offset {offset} must be >= 12",
            );
            assert!(
                buf.len() >= 12,
                "§2.3 ChunkPayloadOverflowsRiff implies buf.len() {} >= 12",
                buf.len(),
            );
            assert_eq!(
                &buf[0..4],
                &fourcc::RIFF,
                "§2.3 ChunkPayloadOverflowsRiff implies buf[0..4] == 'RIFF'",
            );
            assert_eq!(
                &buf[8..12],
                &fourcc::WEBP,
                "§2.3 ChunkPayloadOverflowsRiff implies buf[8..12] == 'WEBP'",
            );
            let header_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
            let declared_end = 8usize
                .checked_add(header_size as usize)
                .expect("§2.4 declared_end fits in usize on 64-bit");
            assert!(
                declared_end <= buf.len(),
                "§2.3 ChunkPayloadOverflowsRiff implies declared RIFF window fits in buffer",
            );
            assert!(
                *offset + 8 <= declared_end,
                "§2.3 ChunkPayloadOverflowsRiff requires the 8-byte header to fit in the RIFF window (offset {offset} + 8 must be <= declared_end {declared_end})",
            );
            // Cross-check the LE uint32 at the chunk header.
            let on_disk_size = u32::from_le_bytes([
                buf[*offset + 4],
                buf[*offset + 5],
                buf[*offset + 6],
                buf[*offset + 7],
            ]);
            assert_eq!(
                *declared, on_disk_size,
                "§2.3 ChunkPayloadOverflowsRiff.declared {declared} must equal LE uint32 at buf[{}..{}] = {}",
                *offset + 4,
                *offset + 8,
                on_disk_size,
            );
            // `available` is exactly `(declared_end - (offset + 8))`.
            let expected_available = declared_end - (*offset + 8);
            assert_eq!(
                *available, expected_available,
                "§2.3 ChunkPayloadOverflowsRiff.available {available} must equal declared_end {declared_end} - (offset {offset} + 8) = {expected_available}",
            );
            // The refusal trigger: `declared > available`.
            assert!(
                *declared as u64 > *available as u64,
                "§2.3 ChunkPayloadOverflowsRiff requires declared {declared} > available {available}",
            );
        }
        ContainerError::MissingPadByte { offset } => {
            // §2.3: the chunk header at `offset` parsed cleanly and
            // its declared payload fit, but its declared `Size` is odd
            // and the §2.3 pad byte that should follow lies outside
            // the declared RIFF payload window.
            assert!(
                *offset >= 12,
                "§2.3 MissingPadByte.offset {offset} must be >= 12",
            );
            assert!(
                buf.len() >= 12,
                "§2.3 MissingPadByte implies buf.len() {} >= 12",
                buf.len(),
            );
            assert_eq!(
                &buf[0..4],
                &fourcc::RIFF,
                "§2.3 MissingPadByte implies buf[0..4] == 'RIFF'",
            );
            assert_eq!(
                &buf[8..12],
                &fourcc::WEBP,
                "§2.3 MissingPadByte implies buf[8..12] == 'WEBP'",
            );
            let header_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
            let declared_end = 8usize
                .checked_add(header_size as usize)
                .expect("§2.4 declared_end fits in usize on 64-bit");
            assert!(
                declared_end <= buf.len(),
                "§2.3 MissingPadByte implies declared RIFF window fits in buffer",
            );
            assert!(
                *offset + 8 <= declared_end,
                "§2.3 MissingPadByte requires the 8-byte header to fit in the RIFF window",
            );
            // The chunk's declared `Size` is odd (§2.3 pad byte only
            // applies to odd-sized chunks).
            let on_disk_size = u32::from_le_bytes([
                buf[*offset + 4],
                buf[*offset + 5],
                buf[*offset + 6],
                buf[*offset + 7],
            ]);
            assert!(
                on_disk_size & 1 == 1,
                "§2.3 MissingPadByte requires the chunk's Size {on_disk_size} to be odd",
            );
            // The pad-byte position lies outside the declared RIFF
            // payload window (the chunk payload itself did fit — the
            // §2.3 pad byte is the only thing missing).
            let payload_end = *offset + 8 + on_disk_size as usize;
            assert!(
                payload_end <= declared_end,
                "§2.3 MissingPadByte requires the chunk payload itself to fit (else ChunkPayloadOverflowsRiff)",
            );
            assert!(
                payload_end + 1 > declared_end,
                "§2.3 MissingPadByte requires the pad byte at {} to lie outside declared RIFF window end {declared_end}",
                payload_end + 1,
            );
        }
    }
}
