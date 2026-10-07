//! Allocation budget of the default lossless (`VP8L`) encode, the
//! single-pass path (`EncodeOptions::method` 4).
//!
//! A counting `#[global_allocator]` measures the bytes the encoder really
//! allocates, so the budget below is a measurement and not a guess. The
//! allocator lives in its own integration-test binary because it replaces
//! the allocator for the whole process.
//!
//! Only allocations made by the measuring thread inside the measured window
//! count: the libtest harness and any other thread are ignored, so the
//! number is stable under a parallel test run.
//!
//! This file holds the only `unsafe` in the crate's tests. A
//! `GlobalAlloc` implementation cannot be written without it; every
//! method forwards unchanged to [`System`].

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use oxideav_webp::{decode_rgba8, encode_rgba8, EncodeOptions};

/// Counts the bytes requested by the current thread while counting is on.
struct CountingAllocator;

thread_local! {
    /// `true` while the current thread is inside a measured window.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Bytes requested (`alloc`, `alloc_zeroed`, and the new size of every
    /// `realloc`) by the current thread while `COUNTING` is set.
    static BYTES: Cell<u64> = const { Cell::new(0) };
    /// Number of allocation calls counted with `BYTES`.
    static CALLS: Cell<u64> = const { Cell::new(0) };
}

fn record(size: usize) {
    // `try_with` because the allocator can run while thread-local storage
    // is being torn down; those allocations are outside any window.
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = BYTES.try_with(|b| b.set(b.get() + size as u64));
            let _ = CALLS.try_with(|c| c.set(c.get() + 1));
        }
    });
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged, so `System`'s guarantees carry over. `record` only touches
// const-initialised `Cell` thread-locals, which never allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Run `f` and return its result with the bytes and calls it allocated on
/// this thread.
fn measure<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    BYTES.with(|b| b.set(0));
    CALLS.with(|c| c.set(0));
    COUNTING.with(|on| on.set(true));
    let out = f();
    COUNTING.with(|on| on.set(false));
    (out, BYTES.with(Cell::get), CALLS.with(Cell::get))
}

/// Upper bound on the bytes one default 256 x 256 lossless encode may
/// allocate, as a multiple of the RGBA input size (4 bytes per pixel).
///
/// The single-pass encoder allocates its per-pixel buffers once per encode
/// and reuses them for every candidate it estimates. In units of the
/// 4-byte-per-pixel input:
///
/// * the ARGB copy of the input: 1x
/// * the transformed pixels it parses: 1x
/// * the LZ77 hash chain: 1x, plus 0.25x of hash heads at this size
/// * the longest-match table (16-bit length, 32-bit distance): 1.5x
/// * the cost-priced re-parse: 64-bit costs (2x) and two parses of a
///   16-bit length and 32-bit distance each (3x)
/// * the colour-cache hit table, when a cache is chosen: 0.5x
/// * the encoded file, written once in place: about 0.4x for this input
///
/// That is about 10 to 11x. The rest is about 2x at 256 x 256 and less on
/// larger images: prefix-code builds and the twelve colour-cache
/// histograms, which do not grow with the image, and the transform
/// sub-images, which grow with it at one entry per 16 x 16 block. The
/// measured total is about 12x; the bound leaves a little headroom.
const BUDGET_MULTIPLE: u64 = 16;

/// Encode the photo-like test image at `side x side`, after one warm-up
/// encode, and return its bytes, calls and input size.
fn measure_encode(side: u32) -> (u64, u64, u64) {
    let rgba = common::photo_rgba(side, side);
    let opts = EncodeOptions::default();

    // Warm-up: the first encode in a process also fills process-wide
    // lookup tables (for example the integer log2 table). Those are not
    // per-encode cost, so the measured encode is the second one.
    let warm = encode_rgba8(side, side, &rgba, &opts).expect("warm-up encode");

    let (file, bytes, calls) = measure(|| encode_rgba8(side, side, &rgba, &opts).expect("encode"));
    assert_eq!(file, warm, "the encoder is deterministic");
    let decoded = decode_rgba8(&file).expect("decode");
    assert_eq!(decoded.data, rgba, "lossless round trip");
    eprintln!(
        "{side}x{side} default lossless encode: {bytes} bytes in {calls} calls ({:.2}x the image)",
        bytes as f64 / rgba.len() as f64
    );
    (bytes, calls, rgba.len() as u64)
}

#[test]
fn default_lossless_encode_allocates_a_small_multiple_of_the_image() {
    let (bytes, _, image) = measure_encode(256);
    assert!(
        bytes <= BUDGET_MULTIPLE * image,
        "allocated {bytes} bytes, more than {BUDGET_MULTIPLE}x the {image}-byte image"
    );
}

/// Allocation calls must not grow with the image: no pass may allocate per
/// row or per pixel. Doubling the side adds 256 rows; one allocation per
/// row in any pass would add at least that many calls. The count still
/// drifts a little with content (prefix codes of other shapes), so the
/// bound is half the added rows.
#[test]
fn default_lossless_encode_allocation_calls_do_not_grow_with_the_image() {
    let (_, small, _) = measure_encode(256);
    let (_, large, _) = measure_encode(512);
    assert!(
        large < small + 128,
        "512x512 made {large} allocation calls, 256x256 made {small}: \
         the count grows with the image"
    );
}
