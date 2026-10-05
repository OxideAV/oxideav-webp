#![no_main]

//! Framework-path fuzz harness: the bytes are a file handed to the WebP
//! container demuxer, every packet it cuts goes through the registered
//! decoder (whole-file or per-frame animation mode, as the stream's
//! `extradata` says), and the packets are muxed back.
//!
//! Contract: every call returns to its caller. A `panic!`, slice OOB,
//! integer overflow in debug, or OOM abort is a finding; `Err` on any
//! input is fine. The decoder is held to a small pixel budget through
//! `DecoderLimits` so a legal-but-huge canvas is refused, not allocated.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;
use oxideav_core::{DecoderLimits, Error, RuntimeContext};
use oxideav_webp::container_registry;

const MAX_PIXELS: u64 = 1 << 20; // 1 Mpx canvas budget per frame

fuzz_target!(|data: &[u8]| {
    let mut ctx = RuntimeContext::new();
    oxideav_webp::register(&mut ctx);

    let reader: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(data.to_vec()));
    let Ok(mut demux) = container_registry::open_demuxer(reader, &ctx.codecs) else {
        return;
    };
    let stream = demux.streams()[0].clone();
    let _ = demux.metadata();
    let _ = demux.duration_micros();

    let mut params = stream.params.clone();
    let mut limits = DecoderLimits::default();
    limits.max_pixels_per_frame = MAX_PIXELS;
    limits.max_alloc_bytes_per_frame = MAX_PIXELS * 4;
    params.limits = limits;
    let Ok(mut dec) = ctx.codecs.first_decoder(&params) else {
        return;
    };

    let mut packets = Vec::new();
    while let Ok(pkt) = demux.next_packet() {
        let _ = dec.send_packet(&pkt);
        loop {
            match dec.receive_frame() {
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(_) => break,
            }
        }
        packets.push(pkt);
    }
    let _ = dec.flush();
    while dec.receive_frame().is_ok() {}

    let sink: Box<dyn oxideav_core::WriteSeek> = Box::new(Cursor::new(Vec::new()));
    if let Ok(mut mux) = container_registry::open_muxer(sink, std::slice::from_ref(&stream)) {
        let _ = mux.write_header();
        for p in &packets {
            let _ = mux.write_packet(p);
        }
        let _ = mux.write_trailer();
    }
});
