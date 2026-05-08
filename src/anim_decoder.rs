//! Streaming animated WebP decoder.
//!
//! `WebpAnimDecoder` is the pull-driven counterpart to [`crate::decode_webp`]:
//! instead of decoding every `ANMF` chunk up-front and returning a
//! `WebpImage` with an owned `Vec<WebpFrame>`, it parses the container
//! once and then decodes one frame per `next_frame()` call. Callers that
//! only need the first N frames of a long animation (preview thumbnails,
//! progressive UI rendering, "show first 5 frames" tools) avoid paying
//! the full-decode cost.
//!
//! The shape mirrors libwebp's `WebPAnimDecoder` (`demux.h` public API),
//! but the implementation is wholly our own — the demuxer, VP8/VP8L
//! decoders, alpha-overlay path, disposal/blend state machine, and
//! `[B,G,R,A]`→RGBA conversion are all already present in this crate
//! (`crate::demux::parse_webp_body`, `crate::decoder` helpers,
//! `crate::demux::bgra_to_rgba`).
//!
//! Example:
//! ```ignore
//! let mut dec = oxideav_webp::WebpAnimDecoder::new(&bytes)?;
//! let info = dec.info();
//! while let Some(frame) = dec.next_frame()? {
//!     handle(&frame.rgba, info.canvas_width, info.canvas_height);
//!     if frame.pts_ms > 1000 { break; } // stop after ~1s of playback.
//! }
//! ```
//!
//! Disposal + blend rules: the decoder owns a single persistent RGBA
//! canvas. Each `next_frame()` composites the just-decoded tile onto it
//! per the ANMF flags (blend with previous canvas, or overwrite), then
//! clones the canvas into the returned [`WebpAnimFrame`]. A
//! `dispose_to_background` flag fills the just-rendered tile's bbox
//! with the ANIM background colour *after* we hand the frame to the
//! caller — exactly what the eager [`crate::decode_webp`] does.
//!
//! The decoder takes the bytes by reference *and copies the bitstreams
//! out into owned `ParsedFrame`s* (the existing `parse_webp_body`
//! behaviour). The copy lets the decoder work after the caller's bytes
//! buffer is gone — convenient for callers that read a `.webp` over
//! the network and want to drop the response buffer eagerly. Memory
//! cost is one allocation per ANMF chunk, ~equal to the file size for
//! animation-heavy inputs.

use crate::decoder::{canvas_filled, composite, decode_parsed_frame_to_rgba};
use crate::demux::{bgra_to_rgba, parse_webp_body, ParsedContainer, WebpFileMetadata};
use crate::error::{Result, WebpError as Error};

/// One frame emitted by [`WebpAnimDecoder::next_frame`]. The `rgba`
/// buffer is the **canvas state** at this point in playback — already
/// composited against the persistent canvas — sized
/// `canvas_width * canvas_height * 4` bytes, row-major.
///
/// `pts_ms` and `duration_ms` use millisecond units (the WebP ANMF
/// chunk's native time base; RFC 9649 §2.5). PTS is the cumulative sum
/// of prior frame durations, with the first frame at `pts_ms = 0`.
#[derive(Debug, Clone)]
pub struct WebpAnimFrame {
    /// Cumulative PTS in milliseconds. First frame is always `0`; each
    /// subsequent frame adds the previous frame's `duration_ms` (with a
    /// `1` floor for zero-duration frames, mirroring the demuxer's
    /// `Packet` PTS arithmetic).
    pub pts_ms: u64,
    /// On-disk frame duration, in milliseconds. `0` is a legal spec
    /// value (RFC 9649 §2.5: "May be zero") meaning "advance
    /// immediately"; we surface it verbatim.
    pub duration_ms: u32,
    /// Final RGBA canvas after this frame's composite. Length is
    /// `canvas_width * canvas_height * 4`. Each subsequent frame's
    /// canvas is computed by mutating the decoder's internal canvas in
    /// place — clone this `Vec` if you need to retain the snapshot.
    pub rgba: Vec<u8>,
    /// Canvas width in pixels — copied from [`WebpAnimDecoder::info`]
    /// for callers that want it on a per-frame basis.
    pub canvas_width: u32,
    /// Canvas height in pixels.
    pub canvas_height: u32,
    /// True when this is the first frame of the animation (or, post-
    /// [`WebpAnimDecoder::reset`], the first frame of the next playback
    /// pass). Mirrors the keyframe flag the demuxer attaches to its
    /// `Packet`s for the same stream.
    pub is_keyframe: bool,
    /// True when the ANMF flags requested alpha-blending against the
    /// pre-frame canvas (bit 0 of the ANMF flags byte = `0`). False
    /// means the tile overwrote the canvas region verbatim.
    pub blend_with_previous: bool,
    /// True when the ANMF flags requested "dispose to background after
    /// rendering" (bit 1 of the ANMF flags byte = `1`). The decoder
    /// has already applied the dispose by the time `next_frame()`
    /// returns, so this flag is mostly informational; consumers that
    /// want to render trails (e.g. trace each frame's bbox onto an
    /// external surface) can use it to know when the tile region was
    /// wiped.
    pub dispose_to_background: bool,
    /// Frame's logical bbox on the canvas (`x_offset`, `y_offset`, the
    /// stored `width` and `height` of the ANMF chunk). Useful for
    /// dirty-region rendering.
    pub frame_x: u32,
    pub frame_y: u32,
    pub frame_width: u32,
    pub frame_height: u32,
}

/// Container-level metadata available before any frame has been
/// decoded. Returned by [`WebpAnimDecoder::info`]; cheap to call
/// repeatedly (just clones a few primitive fields).
#[derive(Debug, Clone)]
pub struct WebpAnimInfo {
    /// Canvas width in pixels (VP8X header for animated/extended
    /// files; the still-image `VP8 ` / `VP8L` chunk's intrinsic size
    /// for simple-layout files).
    pub canvas_width: u32,
    /// Canvas height in pixels.
    pub canvas_height: u32,
    /// Number of frames in the file. A simple-layout `.webp` is
    /// treated as a 1-frame animation here; that's intentional so
    /// `WebpAnimDecoder` can wrap any `.webp` flavour. Animated files
    /// expose the count of `ANMF` chunks.
    pub frame_count: usize,
    /// Loop count from the `ANIM` chunk (RFC 9649 §2.5), `None` for
    /// non-animated files. `0` means "infinite".
    pub loop_count: Option<u16>,
    /// Animation background colour in RGBA byte order (already
    /// converted from the spec's on-disk `[B, G, R, A]` order). `None`
    /// for non-animated files.
    pub background_rgba: Option<[u8; 4]>,
    /// Auxiliary container-level metadata (`ICCP` / `EXIF` / `XMP `).
    /// All fields are `None` when the file omits them — including
    /// every simple-layout `.webp`.
    pub metadata: WebpFileMetadata,
}

/// Streaming animated-WebP decoder. Parses the container once on
/// construction, then yields one frame per [`Self::next_frame`] call.
/// Maintains the persistent RGBA canvas and ANMF disposal/blend state
/// internally, so consumers don't need to track it separately.
///
/// Cheap to construct (just walks the chunk list — no pixel decoding).
/// Memory cost is `canvas_w * canvas_h * 4` for the canvas + the
/// already-existing per-frame `ParsedFrame` storage from the demuxer.
pub struct WebpAnimDecoder {
    info: WebpAnimInfo,
    /// Pre-parsed frames in presentation order — same data the eager
    /// `decode_webp` path holds, stashed here so we can iterate
    /// lazily.
    parsed: ParsedContainer,
    /// Persistent RGBA canvas; mutated in place across `next_frame`
    /// calls per the ANMF blend/disposal rules.
    canvas: Vec<u8>,
    /// Background colour (already RGBA-converted) used for canvas init
    /// and dispose-to-background fills. `[0, 0, 0, 0]` for non-animated
    /// or files that lack an `ANIM` chunk — same fall-through behaviour
    /// as the eager decoder.
    bg_rgba: [u8; 4],
    /// Index of the next frame to decode. Bumped by `next_frame`,
    /// reset to `0` by `reset()`. `done()` returns true when this
    /// equals `parsed.frames.len()`.
    next_index: usize,
    /// Cumulative PTS in ms. Bumped by the previous frame's
    /// `max(duration_ms, 1)` after each `next_frame` (matching the
    /// demuxer's `Packet::pts` arithmetic so callers that compare
    /// with the framework path see identical timestamps).
    pts_ms: u64,
}

impl WebpAnimDecoder {
    /// Construct a streaming decoder over a complete `.webp` file
    /// sitting in `bytes`. Parses the RIFF/WEBP container once
    /// (validating the magic, walking the chunk list, capturing all
    /// metadata + ANIM bg/loop). No pixel data is decoded yet.
    ///
    /// Returns `Err` on malformed containers (bad magic, truncated
    /// chunks, unknown image chunk type). Empty animations with `0`
    /// frames are still constructed — `done()` returns `true`
    /// immediately and `next_frame` returns `Ok(None)`.
    pub fn new(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
            return Err(Error::invalid("WebP: bad RIFF/WEBP magic"));
        }
        let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let end = (8 + riff_size).min(bytes.len());
        let body = &bytes[12..end];
        let parsed = parse_webp_body(body)?;
        let (canvas_w, canvas_h) = parsed.canvas;
        let bg_rgba = parsed
            .anim_background_bgra
            .map(bgra_to_rgba)
            .unwrap_or([0, 0, 0, 0]);
        let canvas = canvas_filled(canvas_w as usize, canvas_h as usize, bg_rgba);
        let info = WebpAnimInfo {
            canvas_width: canvas_w,
            canvas_height: canvas_h,
            frame_count: parsed.frames.len(),
            loop_count: parsed.anim_loop_count,
            background_rgba: parsed.anim_background_bgra.map(bgra_to_rgba),
            metadata: parsed.metadata.clone(),
        };
        Ok(Self {
            info,
            parsed,
            canvas,
            bg_rgba,
            next_index: 0,
            pts_ms: 0,
        })
    }

    /// Container-level info captured at construction time. Cheap to
    /// call repeatedly — the canvas size, frame count, loop count, BG
    /// colour, and metadata don't change after `new()`.
    pub fn info(&self) -> &WebpAnimInfo {
        &self.info
    }

    /// `true` when every frame has been consumed and the next call to
    /// [`Self::next_frame`] will return `Ok(None)`. After [`Self::reset`]
    /// this returns `false` again until the frames are re-played.
    pub fn done(&self) -> bool {
        self.next_index >= self.parsed.frames.len()
    }

    /// Rewind the decoder to frame 0 — the canvas is re-filled with
    /// the ANIM background colour (or transparent black for non-
    /// animated files), `next_index` resets to `0`, and `pts_ms`
    /// resets to `0`. Useful for animations whose `loop_count` is
    /// non-zero (`info().loop_count`): callers that respect the loop
    /// limit can re-play the file by calling `reset()` and resuming
    /// `next_frame()`.
    pub fn reset(&mut self) {
        self.canvas = canvas_filled(
            self.info.canvas_width as usize,
            self.info.canvas_height as usize,
            self.bg_rgba,
        );
        self.next_index = 0;
        self.pts_ms = 0;
    }

    /// Decode the next frame, composite it onto the persistent canvas,
    /// and return a [`WebpAnimFrame`] snapshot. Returns `Ok(None)` once
    /// every frame has been consumed (use [`Self::reset`] to re-play).
    ///
    /// The returned frame's `rgba` buffer is a *clone* of the decoder's
    /// internal canvas — mutating the decoder (via further `next_frame`
    /// or `reset` calls) leaves prior `WebpAnimFrame` instances
    /// unaffected. That matches the eager [`crate::decode_webp`] path
    /// where each `WebpFrame` already owns its canvas snapshot.
    pub fn next_frame(&mut self) -> Result<Option<WebpAnimFrame>> {
        if self.next_index >= self.parsed.frames.len() {
            return Ok(None);
        }
        let frame_index = self.next_index;
        let f = &self.parsed.frames[frame_index];
        // Snapshot the per-frame fields we want to expose before we
        // hand the &ParsedFrame to the helper that owns the borrow.
        let duration_ms = f.duration_ms;
        let blend_with_previous = f.blend_with_previous;
        let dispose_to_background = f.dispose_to_background;
        let frame_x = f.x_offset;
        let frame_y = f.y_offset;
        let frame_w = f.width;
        let frame_h = f.height;

        // Decode the image chunk into a tile-sized RGBA buffer. This
        // is the only really expensive bit of `next_frame` — it runs
        // the VP8 / VP8L decoder + alpha-overlay path for *one* frame.
        let tile_rgba = decode_parsed_frame_to_rgba(f)?;

        // Composite onto the persistent canvas, honouring the ANMF
        // blend flag.
        composite(
            &mut self.canvas,
            self.info.canvas_width,
            self.info.canvas_height,
            &tile_rgba,
            frame_x,
            frame_y,
            frame_w,
            frame_h,
            blend_with_previous,
        );

        // Clone the canvas state for this frame's snapshot.
        let frame_rgba = self.canvas.clone();
        let pts_for_this_frame = self.pts_ms;
        // Advance PTS by `max(duration_ms, 1)` to mirror the demuxer's
        // packet-PTS arithmetic exactly. The `.max(1)` floor keeps a
        // legal `duration_ms = 0` frame from clobbering monotonicity.
        self.pts_ms = self.pts_ms.saturating_add(duration_ms.max(1) as u64);

        // Apply post-frame disposal — wipe the tile bbox to the BG
        // colour AFTER we've snapshotted the frame for the caller, so
        // the returned frame shows the rendered state and only the
        // *next* frame sees the disposed canvas.
        if dispose_to_background {
            let x0 = frame_x as usize;
            let y0 = frame_y as usize;
            let x1 = (x0 + frame_w as usize).min(self.info.canvas_width as usize);
            let y1 = (y0 + frame_h as usize).min(self.info.canvas_height as usize);
            let cw = self.info.canvas_width as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * cw + x) * 4;
                    self.canvas[i] = self.bg_rgba[0];
                    self.canvas[i + 1] = self.bg_rgba[1];
                    self.canvas[i + 2] = self.bg_rgba[2];
                    self.canvas[i + 3] = self.bg_rgba[3];
                }
            }
        }

        self.next_index += 1;
        Ok(Some(WebpAnimFrame {
            pts_ms: pts_for_this_frame,
            duration_ms,
            rgba: frame_rgba,
            canvas_width: self.info.canvas_width,
            canvas_height: self.info.canvas_height,
            is_keyframe: frame_index == 0,
            blend_with_previous,
            dispose_to_background,
            frame_x,
            frame_y,
            frame_width: frame_w,
            frame_height: frame_h,
        }))
    }

    /// Index of the frame [`Self::next_frame`] will decode next. `0`
    /// at construction and after [`Self::reset`]; bumped to
    /// `info().frame_count` after the last successful decode. Useful
    /// for progress reporting without forcing the caller to track an
    /// extra counter.
    pub fn next_frame_index(&self) -> usize {
        self.next_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_anim::{build_animated_webp, AnimFrame};

    const W: u32 = 8;
    const H: u32 = 8;

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
        let n = (width as usize) * (height as usize);
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&rgba);
        }
        v
    }

    fn three_frame_anim() -> Vec<u8> {
        let red = solid(W, H, [0xff, 0, 0, 0xff]);
        let green = solid(W, H, [0, 0xff, 0, 0xff]);
        let blue = solid(W, H, [0, 0, 0xff, 0xff]);
        let frames = [
            AnimFrame {
                width: W,
                height: H,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 30,
                blend: false,
                dispose_to_background: false,
                rgba: &red,
            },
            AnimFrame {
                width: W,
                height: H,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 40,
                blend: false,
                dispose_to_background: false,
                rgba: &green,
            },
            AnimFrame {
                width: W,
                height: H,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 50,
                blend: false,
                dispose_to_background: false,
                rgba: &blue,
            },
        ];
        build_animated_webp(W, H, [0, 0, 0, 0], 0, &frames).expect("encode")
    }

    #[test]
    fn streams_three_frames_in_order_with_pts() {
        let blob = three_frame_anim();
        let mut dec = WebpAnimDecoder::new(&blob).expect("new");
        assert_eq!(dec.info().frame_count, 3);
        assert!(!dec.done());
        let f0 = dec.next_frame().expect("ok").expect("Some");
        assert_eq!(f0.pts_ms, 0);
        assert_eq!(f0.duration_ms, 30);
        assert!(f0.is_keyframe);
        assert_eq!(&f0.rgba[0..4], &[0xff, 0, 0, 0xff], "F0 should be red");
        let f1 = dec.next_frame().expect("ok").expect("Some");
        assert_eq!(f1.pts_ms, 30);
        assert!(!f1.is_keyframe);
        assert_eq!(&f1.rgba[0..4], &[0, 0xff, 0, 0xff], "F1 should be green");
        let f2 = dec.next_frame().expect("ok").expect("Some");
        assert_eq!(f2.pts_ms, 70);
        assert_eq!(&f2.rgba[0..4], &[0, 0, 0xff, 0xff], "F2 should be blue");
        assert!(dec.done());
        assert!(dec.next_frame().expect("ok").is_none());
    }

    #[test]
    fn early_stop_does_not_decode_remaining_frames() {
        // The streaming guarantee: after pulling N frames, the decoder
        // hasn't done any work for the remaining frames. We can't
        // observe the absence of work directly, but `next_frame_index`
        // exposes the fact that we stopped early.
        let blob = three_frame_anim();
        let mut dec = WebpAnimDecoder::new(&blob).expect("new");
        let _ = dec.next_frame().expect("ok").expect("Some");
        assert_eq!(dec.next_frame_index(), 1);
        assert!(!dec.done());
    }

    #[test]
    fn reset_rewinds_to_frame_zero() {
        let blob = three_frame_anim();
        let mut dec = WebpAnimDecoder::new(&blob).expect("new");
        // Drain everything.
        while dec.next_frame().expect("ok").is_some() {}
        assert!(dec.done());
        dec.reset();
        assert!(!dec.done());
        let f0 = dec.next_frame().expect("ok").expect("Some");
        assert_eq!(f0.pts_ms, 0);
        assert!(f0.is_keyframe);
        assert_eq!(&f0.rgba[0..4], &[0xff, 0, 0, 0xff]);
    }

    #[test]
    fn info_exposes_anim_metadata() {
        // Use a non-zero BG so the BGRA→RGBA conversion is exercised
        // through the streaming path (not just decode_webp).
        let bg_bgra: [u8; 4] = [0x10, 0x20, 0x30, 0xff];
        let red_tile = solid(4, 4, [0xff, 0, 0, 0xff]);
        let frames = [AnimFrame {
            width: 4,
            height: 4,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &red_tile,
        }];
        let blob = build_animated_webp(W, H, bg_bgra, 7, &frames).expect("encode");
        let dec = WebpAnimDecoder::new(&blob).expect("new");
        let info = dec.info();
        assert_eq!(info.canvas_width, W);
        assert_eq!(info.canvas_height, H);
        assert_eq!(info.frame_count, 1);
        assert_eq!(info.loop_count, Some(7));
        assert_eq!(info.background_rgba, Some([0x30, 0x20, 0x10, 0xff]));
    }

    #[test]
    fn dispose_to_background_applies_between_streamed_frames() {
        // Mirrors the "anim_dispose_to_background_uses_bg_color_not_transparent_black"
        // test in tests/anim_background_color.rs but goes through the
        // streaming path.
        let bg_bgra: [u8; 4] = [0x40, 0x50, 0x60, 0xff];
        let bg_rgba_expected = [0x60, 0x50, 0x40, 0xff];
        let f0_tile = solid(W, H, [0xff, 0, 0, 0xff]);
        let f1_tile = solid(2, 2, [0, 0xff, 0, 0xff]);
        let frames = [
            AnimFrame {
                width: W,
                height: H,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 40,
                blend: false,
                dispose_to_background: true,
                rgba: &f0_tile,
            },
            AnimFrame {
                width: 2,
                height: 2,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 40,
                blend: false,
                dispose_to_background: false,
                rgba: &f1_tile,
            },
        ];
        let blob = build_animated_webp(W, H, bg_bgra, 0, &frames).expect("encode");
        let mut dec = WebpAnimDecoder::new(&blob).expect("new");
        let _f0 = dec.next_frame().expect("ok").expect("Some");
        let f1 = dec.next_frame().expect("ok").expect("Some");
        let stride = (W as usize) * 4;
        // Top-left 2×2: green.
        for y in 0..2 {
            for x in 0..2 {
                let i = y * stride + x * 4;
                assert_eq!(&f1.rgba[i..i + 4], &[0, 0xff, 0, 0xff]);
            }
        }
        // Outside: the BG colour, not transparent black.
        for y in 0..H as usize {
            for x in 0..W as usize {
                if x < 2 && y < 2 {
                    continue;
                }
                let i = y * stride + x * 4;
                assert_eq!(&f1.rgba[i..i + 4], &bg_rgba_expected);
            }
        }
    }

    #[test]
    fn still_webp_is_streamed_as_one_frame() {
        // `decode_webp` treats simple-layout files as a 1-frame
        // animation; the streaming decoder mirrors that so consumers
        // can use one code path for both.
        let argb = vec![0xff_80_40_20u32; (W * H) as usize];
        let blob = crate::encode_vp8l_argb_with_metadata(
            W,
            H,
            &argb,
            false,
            &crate::WebpMetadata::default(),
        )
        .expect("encode");
        let mut dec = WebpAnimDecoder::new(&blob).expect("new");
        assert_eq!(dec.info().frame_count, 1);
        assert_eq!(dec.info().loop_count, None);
        let f0 = dec.next_frame().expect("ok").expect("Some");
        assert!(f0.is_keyframe);
        assert!(dec.next_frame().expect("ok").is_none());
    }

    #[test]
    fn rejects_malformed_magic() {
        let bad = b"junk junk junk junk".to_vec();
        assert!(WebpAnimDecoder::new(&bad).is_err());
    }
}
