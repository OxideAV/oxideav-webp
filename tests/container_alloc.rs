//! Allocation budget of wrapping a finished bitstream in the RIFF
//! container (`build_webp_file`).
//!
//! A counting `#[global_allocator]` measures the bytes really allocated,
//! so the budget is a measurement and not a guess. The allocator lives in
//! its own integration-test binary because it replaces the allocator for
//! the whole process. Only allocations made by the measuring thread inside
//! the measured window count, so the libtest harness and other threads do
//! not disturb the number.
//!
//! A `GlobalAlloc` implementation cannot be written without `unsafe`;
//! every method forwards unchanged to [`System`].

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use oxideav_webp::build::ImageKind;
use oxideav_webp::build_webp_file;

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

/// `build_webp_file` wraps a finished bitstream (the lossy path's `VP8 `
/// keyframe, for one) in the RIFF container. It must copy the bitstream
/// once: the bytes allocated are the file (at most 38 bytes of headers
/// plus the bitstream and its pad byte) and, for the extended layouts, the
/// 10-byte `VP8X` payload. Two copies would be twice the bitstream.
#[test]
fn wrapping_a_bitstream_in_a_file_copies_it_once() {
    let payload = vec![0x5a_u8; 100_001];
    for kind in [
        ImageKind::Lossy,
        ImageKind::Lossless,
        ImageKind::ExtendedLossy,
        ImageKind::ExtendedLossless,
    ] {
        let (file, bytes, calls) =
            measure(|| build_webp_file(&payload, kind, 64, 64).expect("build"));
        assert!(
            file.ends_with(&[0x5a, 0]),
            "{kind:?}: the bitstream and its pad byte end the file"
        );
        let budget = payload.len() as u64 + 64;
        assert!(
            bytes <= budget,
            "{kind:?}: {bytes} bytes in {calls} calls; one copy of the bitstream is at most {budget}"
        );
    }
}
