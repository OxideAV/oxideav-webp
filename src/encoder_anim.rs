//! Animated WebP encoder.
//!
//! Builds a `RIFF/WEBP/VP8X + ANIM + ANMF...ANMF` file from a sequence of
//! per-frame RGBA tiles. Each frame is encoded **per-frame** in either
//! VP8L (lossless) or VP8+ALPH (lossy) mode, whichever produces the
//! smaller ANMF payload — the file format permits mixing the two, and
//! the decoder already handles both shapes.
//!
//! Two factory entry points are exposed:
//!
//! * [`build_animated_webp`] — drives every frame through the
//!   lossless VP8L path. Bit-exact, no quality loss, larger files. Used
//!   by callers that need pixel-perfect playback.
//! * [`build_animated_webp_with_options`] — accepts an
//!   [`AnimEncoderOptions`] knob bag. With `mode_select = AUTO` (the
//!   default), each frame is encoded both ways and the byte-smallest
//!   payload wins, so animations with photographic / smoothly-varying
//!   frames get the lossy path's compression while sharp synthetic
//!   frames stay on the lossless path. The wrapper preserves the
//!   bit-exact behaviour of [`build_animated_webp`] when options are
//!   defaulted to `Lossless`.
//!
//! The reference for this is libwebp's per-frame `WebPAnimEncoderAdd`
//! decision: each frame calls into both encoders and the smallest
//! payload wins. We use raw byte count rather than a perceptual cost
//! model for now — closes #335.
//!
//! # Container layout
//!
//! ```text
//! RIFF <size> WEBP
//!   VP8X <10>   — flags (ANIM bit set), canvas_w-1, canvas_h-1
//!   ANIM <6>    — 4 BGRA bytes background, 2-byte loop count (0=infinite)
//!   ANMF <n>    — per-frame envelope (header + nested VP8L chunk)
//!   ANMF <n>    — ...
//! ```
//!
//! Per `ANMF` header (16 bytes before nested chunks):
//!
//! ```text
//!   3 bytes  X offset / 2          (must be even)
//!   3 bytes  Y offset / 2          (must be even)
//!   3 bytes  frame_w - 1
//!   3 bytes  frame_h - 1
//!   3 bytes  duration_ms
//!   1 byte   bit0 = blending (0=blend, 1=overwrite)
//!            bit1 = disposal (0=none,  1=dispose-to-background)
//! ```
//!
//! The nested chunk is a single `VP8L` (lossless) sub-chunk produced by
//! the existing per-frame VP8L encoder.

use crate::error::{Result, WebpError as Error};
use crate::riff::WebpMetadata;
use crate::vp8l::encode_vp8l_argb;

/// Per-frame mode-selection policy for [`build_animated_webp_with_options`].
///
/// `Eq` is intentionally not derived because the `Delta` variant carries
/// a `DeltaConfig` whose `max_bbox_fraction: f32` (and `auto_inner_quality`)
/// fields don't satisfy the total-equality contract — `PartialEq` is
/// sufficient for the mode-pattern matches the encoder relies on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AnimFrameMode {
    /// Always encode every frame as VP8L (lossless). Bit-exact, larger
    /// files. Matches the historical [`build_animated_webp`] behaviour.
    Lossless,
    /// Always encode every frame as VP8 + ALPH (lossy colour, lossless
    /// alpha). Smaller files for photographic / smoothly-varying
    /// frames; visible compression artefacts for sharp synthetic
    /// content.
    Lossy,
    /// **Default.** Encode every frame both ways and pick whichever
    /// produces the smaller ANMF sub-chunk payload. Mixed lossless +
    /// lossy output — the WebP container permits this and the decoder
    /// already handles both shapes. Mirrors libwebp's
    /// `WebPAnimEncoderAdd` per-frame mode decision.
    #[default]
    Auto,
    /// **Delta** (AVIF-style perceptual frame-merge): each non-first frame
    /// is compared against the prior frame on a block-by-block basis using
    /// a luminance-biased SAD cost model
    /// ([`DeltaConfig::block_cost`]). Blocks whose cost stays below
    /// [`DeltaConfig::threshold`] are presumed unchanged; the encoder
    /// computes the bounding box of all changed blocks, encodes only
    /// that sub-rectangle, and emits an ANMF with `blending_method = 1`
    /// (`DoNotBlend` — overwrite the prior canvas). Frames whose change
    /// region is at most [`DeltaConfig::max_bbox_fraction`] of the
    /// canvas (default 80%) take the delta path; otherwise the encoder
    /// falls back to encoding the full frame in `Auto` mode.
    ///
    /// Constraints (caller responsibility — checked at encode time):
    ///   * every frame must be canvas-sized (`width = canvas_w`,
    ///     `height = canvas_h`, `x_offset = y_offset = 0`),
    ///   * `dispose_to_background = false` and `blend = false`
    ///     (delta-mode output forces overwrite, and a dispose-to-bg
    ///     between frames invalidates the prior-canvas reference).
    ///
    /// The first frame is always emitted in full; subsequent frames may
    /// be partial sub-rectangles.
    Delta(DeltaConfig),
}

/// Tunable parameters for the [`AnimFrameMode::Delta`] frame-merge mode.
///
/// The cost model is a **luminance-biased sum-of-absolute-differences**
/// over fixed-size blocks. For each `block_size × block_size` block we
/// compute `sum_over_pixels |luma(prev) - luma(new)| + 0.25 * |R'-R| +
/// 0.25 * |G'-G| + 0.25 * |B'-B| + |A'-A|`, where `luma = 0.299R + 0.587G
/// + 0.114B` (BT.601). A block is considered **changed** if its cost
/// exceeds [`Self::threshold`].
///
/// Defaults are tuned for 8×8 blocks at threshold 32 (≈1 LSB per pixel
/// on a flat region — small enough to flag any real motion, large
/// enough to absorb codec rounding noise).
///
/// # Multi-rect emission
///
/// When the changed blocks form **multiple disjoint clusters** (e.g. a
/// UI with two independently-spinning indicators on a static
/// background), the encoder runs a 4-connected-component pass on the
/// dirty-block grid and emits **one ANMF sub-rect per cluster**.
///
/// The per-frame sub-rect cap is **adaptive by default**: the encoder
/// computes a `cluster_density = sum(cluster_pixels) / canvas_pixels`
/// metric per frame and picks a budget that scales inversely — lots of
/// scattered tiny clusters (low density) get a high budget (more rects
/// retained), one big near-canvas-wide cluster (high density) gets a
/// low budget (the merger squashes the long tail aggressively because
/// the super-rect collapse is cheap). Mapping (linear interpolation
/// inside the band):
///
/// ```text
///   density ≤ 5%   →  16 rects
///   density ≥ 30%  →   4 rects
/// ```
///
/// Callers can override the adaptive budget with a fixed value via
/// [`Self::max_components_override`] — when `Some(n)`, the cluster-density
/// branch is skipped and the budget is `n` regardless of frame content.
/// Set to `Some(1)` to force the historical single-bbox behaviour.
///
/// When the cluster count exceeds the effective budget, the smallest
/// clusters are iteratively merged with their nearest-neighbour cluster
/// (axis-aligned bbox-of-pair) until the count fits.
///
/// Each component within a logical input frame becomes a separate ANMF
/// in the output stream. Non-final sub-rects carry `duration_ms = 0`
/// (the decoder interprets this as "show for 1 ms", per the spec's
/// floor); the final sub-rect carries the input frame's `duration_ms`
/// so total display time stays correct. Decoded `WebpFrame` count will
/// therefore exceed the input frame count when multi-rect kicks in.
///
/// # Auto inner-encode (lossy fallback for large tiles)
///
/// By default Delta-mode sub-rects re-encode losslessly (VP8L) — the
/// tiles are typically tiny (a few KB raw RGBA) and the VP8 keyframe
/// fixed overhead would dominate. When a delta region is visually-busy
/// enough that lossless costs more than acceptable lossy bytes, set
/// [`Self::auto_inner_threshold_bytes`] to a byte cutoff: tiles whose
/// lossless payload exceeds the cutoff are also encoded as VP8 + ALPH
/// at [`Self::auto_inner_quality`] (default 75) and the byte-smaller
/// candidate wins on disk. Tiles below the cutoff stay lossless — no
/// quality loss for the common-case small-tile path.
///
/// # Cost model (SAD vs SSIM-lite)
///
/// The default cost model is the **luminance-biased SAD** described
/// above (cheap, but luminance-biased so it underweights pure chroma
/// shifts and low-contrast structural changes). When
/// [`Self::enable_ssim_cost`] is `true`, the encoder swaps in a
/// **single-scale SSIM-lite** cost — a perceptually-meaningful metric
/// derived from the standard SSIM formula at one scale (skipping the
/// multi-scale Gaussian-pyramid for speed):
///
/// ```text
///   SSIM = (2*µ_a*µ_b + C1)(2*σ_ab + C2)
///        / ((µ_a² + µ_b² + C1)(σ_a² + σ_b² + C2))
///   C1   = (0.01 * 255)² = 6.5025
///   C2   = (0.03 * 255)² = 58.5225
///   cost = round((1.0 - SSIM) * 10000)   // u64, scaled for integer threshold compare
/// ```
///
/// The threshold for the SSIM cost path is [`Self::ssim_threshold`]
/// (default 50, i.e. ≈ 0.005 SSIM gap — small enough to flag a
/// just-perceptible structural change, large enough to absorb 8-bit
/// rounding noise on a flat block). SAD is left as the default
/// (backwards-compatible behaviour); flip `enable_ssim_cost` on to opt
/// into the perceptual cost model.
///
/// Worked example demonstrating where SSIM beats SAD: a flat-luma
/// 8×8 block where every pixel shifts colour by 4 LSB but the
/// brightness mean stays constant. SAD's luminance term sees ≈ 0
/// (means cancel) and the chroma terms are >> 2 down-weighted, so
/// the block is flagged "similar" — but SSIM's covariance term picks
/// up the structural change because `σ_ab` collapses while `σ_a`
/// and `σ_b` stay non-trivial.
///
/// # MS-SSIM (multi-scale, 3 scales)
///
/// When [`Self::enable_msssim_cost`] is `true`, the encoder swaps in a
/// **3-scale MS-SSIM-lite** cost — a multi-scale generalisation of the
/// single-scale cost above, per Wang/Bovik 2003 ("Multi-scale
/// structural similarity for image quality assessment"):
///
/// ```text
///   scale 0: full SSIM at native block resolution (lum * contrast * struct)
///   scale 1: contrast * struct on a 2x-extended-region 2x-Gaussian-downsampled patch
///   scale 2: contrast * struct on a 4x-extended-region 4x-Gaussian-downsampled patch
///   MS-SSIM = SSIM_0^α * CS_1^β * CS_2^γ
///   cost    = round((1.0 - MS-SSIM) * 10000)
/// ```
///
/// The 3-scale empirical exponents (`α=0.2856, β=0.3001, γ=0.4143`,
/// summing to 1.0) are derived from the canonical 5-scale series
/// `{0.0448, 0.2856, 0.3001, 0.2363, 0.1333}` by fusing the bottom two
/// scales into γ (so the 5-scale weights collapse cleanly to a
/// 3-scale subset that still favours the larger spatial extents).
///
/// MS-SSIM catches **low-frequency structural drift** that single-scale
/// SSIM at 8×8 blocks can miss: a global gradient shift from one frame
/// to the next that perturbs the mean inside every 8×8 block by ≈ 0
/// but accumulates a clear DC change at the 32×32 (scale-2) extent.
/// Single-scale SSIM-lite scores ≈ 0 (no per-block change), but the
/// scale-2 contrast/structure terms collapse and MS-SSIM flags the
/// block as changed.
///
/// MS-SSIM supersedes [`Self::enable_ssim_cost`] when both are on (the
/// single-scale cost is the `SSIM_0` component of the multi-scale
/// product). Threshold for the MS-SSIM cost path is
/// [`Self::msssim_threshold`] (default 50, same scale as the
/// single-scale `(1 - SSIM) * 10000`).
///
/// Selects the downsampling kernel used by the MS-SSIM-lite cost path
/// when computing the contrast-structure terms at scale 1 (2×) and
/// scale 2 (4×). [`Self::Box`] is the historical default — a separable
/// box-average over each `factor × factor` source cell. [`Self::Gaussian`]
/// uses the canonical Wang/Bovik 2003 5-tap σ=0.8 kernel
/// (`[0.054, 0.244, 0.404, 0.244, 0.054]`) applied separably (horizontal
/// then vertical), then a 2× decimation. The 4× scale cascades two such
/// blur+decimate passes (a true Gaussian pyramid) — closer to the
/// Wang/Bovik reference and avoids the box kernel's high-frequency
/// undershoot on smooth gradients.
///
/// Box is kept as the default for backwards-compat — opt in to Gaussian
/// via [`DeltaConfig::msssim_downsample_kernel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownsampleKernel {
    /// Separable 2× box-average per `factor × factor` source cell —
    /// fast, but undershoots high-frequency content unevenly on smooth
    /// gradients vs the Wang/Bovik reference.
    Box,
    /// Separable 5-tap Gaussian σ=0.8 (weights
    /// `[0.054, 0.244, 0.404, 0.244, 0.054]`) followed by 2× decimation;
    /// the 4× scale cascades two passes for a true Gaussian pyramid.
    Gaussian,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaConfig {
    /// Block side length in pixels. Default 8. The encoder rounds
    /// the cost-model bbox up to a multiple of this and then up to
    /// the WebP-mandated even offset.
    pub block_size: u32,
    /// Luminance-biased SAD threshold per block — blocks with cost
    /// strictly greater than this are flagged as changed. Default 32.
    pub threshold: u32,
    /// If the changed-region bounding box covers more than this
    /// fraction of the canvas, the encoder bails out of the delta
    /// path for that frame and falls back to a full-frame encode in
    /// `Auto` mode. Range `0.0..=1.0`; default `0.8`.
    pub max_bbox_fraction: f32,
    /// Optional fixed cap on the number of disjoint sub-rect ANMFs
    /// the encoder may emit per logical input frame. When `None`
    /// (the default), the encoder picks the cap **adaptively** from
    /// the per-frame `cluster_density = sum_cluster_pixels /
    /// canvas_pixels` metric: density ≤ 5% → 16, density ≥ 30% → 4,
    /// linear interpolation in between. When `Some(n)`, the budget
    /// is `n` regardless of frame content. Set to `Some(1)` to force
    /// the historical single-bbox behaviour.
    pub max_components_override: Option<u32>,
    /// Optional byte cutoff on Delta-mode sub-rect tiles' lossless
    /// VP8L payload — tiles whose lossless bytes exceed this cutoff
    /// are *also* encoded via VP8 + ALPH at
    /// [`Self::auto_inner_quality`] and the byte-smaller candidate
    /// wins on disk. Tiles whose lossless bytes ≤ cutoff stay
    /// lossless (no quality loss).
    ///
    /// Default `Some(4096)` — small tiles (≤ 4 KB lossless) stay
    /// pixel-identical, larger noisy tiles trigger the lossy/lossless
    /// race that delivers the documented ~63% reduction on the
    /// 32×32 noisy sub-rect fixture. Set to `None` via
    /// [`Self::auto_inner_threshold_bytes`] to opt out and force every
    /// sub-rect tile to stay lossless regardless of size (preserves
    /// the pre-default-change pixel-identical round-trip semantics for
    /// every tile).
    pub auto_inner_threshold_bytes: Option<u32>,
    /// Quality knob for the lossy-fallback encode triggered by
    /// [`Self::auto_inner_threshold_bytes`] — same scale as
    /// [`AnimEncoderOptions::lossy_quality`] (`0.0..=100.0`, higher =
    /// better). Default 75. Ignored when `auto_inner_threshold_bytes`
    /// is `None`.
    pub auto_inner_quality: f32,
    /// When `true`, swap the default luminance-biased SAD cost model
    /// for a **single-scale SSIM-lite** perceptual cost. The cost
    /// scale changes (SSIM cost uses [`Self::ssim_threshold`] instead
    /// of [`Self::threshold`]). Default `false` — preserves the
    /// existing SAD-based behaviour for backwards compatibility.
    pub enable_ssim_cost: bool,
    /// Threshold for the SSIM-lite cost path (only consulted when
    /// [`Self::enable_ssim_cost`] is `true`). Cost is computed as
    /// `round((1.0 - SSIM) * 10000)`; blocks whose cost strictly
    /// exceeds this value are flagged as changed. Default 50
    /// (≈ 0.005 SSIM gap — picks up just-perceptible structural
    /// changes while absorbing 8-bit rounding noise). Independent
    /// of [`Self::threshold`] so the two modes' defaults stay clean.
    pub ssim_threshold: u32,
    /// When `true`, swap the cost model for the **3-scale MS-SSIM-lite**
    /// perceptual cost (Wang/Bovik 2003). Supersedes
    /// [`Self::enable_ssim_cost`] when both are on — the single-scale
    /// cost becomes the `SSIM_0` component of the multi-scale product.
    /// Cost scale matches the single-scale path
    /// (`round((1.0 - MS-SSIM) * 10000)`) and uses
    /// [`Self::msssim_threshold`]. Default `false` — preserves the
    /// existing SAD/single-scale-SSIM behaviour for backwards
    /// compatibility.
    pub enable_msssim_cost: bool,
    /// Threshold for the MS-SSIM cost path (only consulted when
    /// [`Self::enable_msssim_cost`] is `true`). Same scale as
    /// [`Self::ssim_threshold`] (`round((1.0 - MS-SSIM) * 10000)`);
    /// blocks whose cost strictly exceeds this value are flagged as
    /// changed. Default 50.
    pub msssim_threshold: u32,
    /// Downsample kernel used when reducing the 2×/4× extended
    /// regions for the MS-SSIM-lite cost path's scale-1 / scale-2
    /// CS terms. [`DownsampleKernel::Box`] (the default) keeps the
    /// pre-existing box-average behaviour for backwards-compat;
    /// [`DownsampleKernel::Gaussian`] swaps in the canonical
    /// Wang/Bovik 2003 5-tap σ=0.8 kernel + 2× decimation
    /// (cascaded twice for the 4× scale). Only consulted when
    /// [`Self::enable_msssim_cost`] is `true`.
    pub msssim_downsample_kernel: DownsampleKernel,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            block_size: 8,
            threshold: 32,
            max_bbox_fraction: 0.8,
            max_components_override: None,
            // Default Some(4096): small tiles (≤ 4 KB lossless) stay
            // pixel-identical, larger noisy tiles trigger the
            // lossy/lossless race. Opt out via
            // `DeltaConfig::auto_inner_threshold_bytes(None)`.
            auto_inner_threshold_bytes: Some(4096),
            auto_inner_quality: 75.0,
            enable_ssim_cost: false,
            ssim_threshold: 50,
            enable_msssim_cost: false,
            msssim_threshold: 50,
            msssim_downsample_kernel: DownsampleKernel::Box,
        }
    }
}

impl DeltaConfig {
    /// Builder-style setter for [`Self::max_components_override`] —
    /// sets a fixed per-frame sub-rect cap, bypassing the adaptive
    /// cluster-density branch.
    #[must_use]
    pub fn max_components_override(mut self, n: u32) -> Self {
        self.max_components_override = Some(n);
        self
    }

    /// Builder-style setter for [`Self::auto_inner_threshold_bytes`] —
    /// enables (or, with `None`, disables) the lossy-fallback
    /// inner-encode path for sub-rect tiles whose lossless VP8L
    /// payload exceeds `bytes`. Pass `None` to opt out of the
    /// per-sub-rect race entirely (every sub-rect tile stays
    /// lossless regardless of size — preserves the
    /// pre-default-change pixel-identical round-trip semantics).
    /// The default is `Some(4096)`.
    #[must_use]
    pub fn auto_inner_threshold_bytes(mut self, bytes: Option<u32>) -> Self {
        self.auto_inner_threshold_bytes = bytes;
        self
    }

    /// Builder-style setter for [`Self::enable_ssim_cost`] — turns on
    /// the SSIM-lite cost model (the SAD remains the default).
    #[must_use]
    pub fn enable_ssim_cost(mut self, on: bool) -> Self {
        self.enable_ssim_cost = on;
        self
    }

    /// Builder-style setter for [`Self::ssim_threshold`] — overrides
    /// the SSIM-lite cost-path threshold. Only consulted when
    /// [`Self::enable_ssim_cost`] is `true`.
    #[must_use]
    pub fn ssim_threshold(mut self, t: u32) -> Self {
        self.ssim_threshold = t;
        self
    }

    /// Builder-style setter for [`Self::enable_msssim_cost`] — turns
    /// on the 3-scale MS-SSIM-lite cost model. Supersedes
    /// [`Self::enable_ssim_cost`] when both are on.
    #[must_use]
    pub fn enable_msssim_cost(mut self, on: bool) -> Self {
        self.enable_msssim_cost = on;
        self
    }

    /// Builder-style setter for [`Self::msssim_threshold`] —
    /// overrides the MS-SSIM cost-path threshold. Only consulted when
    /// [`Self::enable_msssim_cost`] is `true`.
    #[must_use]
    pub fn msssim_threshold(mut self, t: u32) -> Self {
        self.msssim_threshold = t;
        self
    }

    /// Builder-style setter for [`Self::msssim_downsample_kernel`] —
    /// switches the MS-SSIM-lite scale-1 / scale-2 downsample kernel
    /// between the historical box-average ([`DownsampleKernel::Box`],
    /// the default) and the canonical Wang/Bovik 2003 5-tap σ=0.8
    /// Gaussian ([`DownsampleKernel::Gaussian`]). Only consulted when
    /// [`Self::enable_msssim_cost`] is `true`.
    #[must_use]
    pub fn msssim_downsample_kernel(mut self, kernel: DownsampleKernel) -> Self {
        self.msssim_downsample_kernel = kernel;
        self
    }
}

/// Map a per-frame `cluster_density` (sum of cluster pixel-areas
/// divided by total canvas area, in `[0.0, 1.0]`) to an effective
/// per-frame `max_components` budget. Linear ramp from `16` (at
/// `density ≤ 0.05`) down to `4` (at `density ≥ 0.30`); flat outside
/// the band. Pulled out as a freestanding helper so the test suite can
/// pin the mapping without going through the encoder.
///
/// Exposed via [`debug_adaptive_max_components`] (doc-hidden, public
/// only so integration tests can pin the ramp behaviour against an
/// end-to-end encode without re-deriving the density from the
/// internal cluster-density walk).
fn adaptive_max_components(density: f32) -> u32 {
    // Clamp to the documented range.
    let d = density.clamp(0.0, 1.0);
    const LO_DENSITY: f32 = 0.05;
    const HI_DENSITY: f32 = 0.30;
    const LO_BUDGET: f32 = 16.0;
    const HI_BUDGET: f32 = 4.0;
    if d <= LO_DENSITY {
        return LO_BUDGET as u32;
    }
    if d >= HI_DENSITY {
        return HI_BUDGET as u32;
    }
    // Linear interpolation in the band — `t = 0` at LO_DENSITY,
    // `t = 1` at HI_DENSITY. Round to nearest u32.
    let t = (d - LO_DENSITY) / (HI_DENSITY - LO_DENSITY);
    let budget = LO_BUDGET + t * (HI_BUDGET - LO_BUDGET);
    budget.round() as u32
}

/// Test-only probe that exposes [`adaptive_max_components`] for
/// integration tests. Hidden from the rendered docs (the function is
/// an internal implementation detail of the Delta-mode adaptive
/// budget; callers should rely on the documented density-to-budget
/// table in [`DeltaConfig`]'s docs rather than this raw function).
#[doc(hidden)]
pub fn debug_adaptive_max_components(density: f32) -> u32 {
    adaptive_max_components(density)
}

/// Test-only probe that runs the cost-model pixel walk on a
/// (prev, curr) RGBA pair and returns the same `cluster_density`
/// metric the encoder uses to drive the adaptive budget. Lets
/// integration tests pin the encoder's budget choice for a given
/// fixture without re-deriving the density math by hand. Hidden from
/// the rendered docs — see `debug_adaptive_max_components`.
#[doc(hidden)]
pub fn debug_cluster_density(
    prev: &[u8],
    curr: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    cfg: &DeltaConfig,
) -> f32 {
    let max_pixels = (canvas_w as u64).saturating_mul(canvas_h as u64);
    if max_pixels == 0 {
        return 0.0;
    }
    let (_, raw_pixels) =
        changed_block_components_with_density(prev, curr, canvas_w, canvas_h, cfg, u32::MAX);
    (raw_pixels as f32) / (max_pixels as f32)
}

/// Knob bag for [`build_animated_webp_with_options`]. Defaults pick the
/// per-frame mode-select strategy at quality 75 (libwebp's default).
///
/// File-level metadata chunks (`ICCP` / `EXIF` / `XMP `) can be attached
/// via [`metadata`](Self::metadata) — when any field of the inner
/// [`WebpMetadata`] is `Some`, the matching VP8X flag bit is set and the
/// chunk is written into the file body in the spec-mandated order
/// (ICCP immediately after VP8X; EXIF / XMP after the last ANMF).
#[derive(Clone, Debug)]
pub struct AnimEncoderOptions<'a> {
    /// Per-frame mode-selection policy. Defaults to [`AnimFrameMode::Auto`]
    /// (per-frame byte-smallest wins).
    pub mode: AnimFrameMode,
    /// Quality for the lossy path, on libwebp's `0.0..=100.0` scale
    /// (higher = better). Ignored when `mode = Lossless`. Default 75.
    pub lossy_quality: f32,
    /// Optional file-level auxiliary metadata (ICC profile, EXIF, XMP)
    /// to attach to the animation's VP8X header. Defaults to all `None`.
    pub metadata: WebpMetadata<'a>,
}

impl<'a> Default for AnimEncoderOptions<'a> {
    fn default() -> Self {
        Self {
            mode: AnimFrameMode::default(),
            lossy_quality: 75.0,
            metadata: WebpMetadata::default(),
        }
    }
}

/// One frame of an animation: an RGBA tile sized `width × height` rendered
/// at `(x_offset, y_offset)` on the canvas, displayed for `duration_ms`
/// before the next frame is composited.
///
/// `x_offset` and `y_offset` are stored on disk as half their value (the
/// spec mandates even offsets), so we silently round odd values down to
/// the next even number.
#[derive(Clone)]
pub struct AnimFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub x_offset: u32,
    pub y_offset: u32,
    pub duration_ms: u32,
    /// `true` → blend the frame's alpha onto the canvas. `false` → the
    /// frame overwrites the destination pixels (alpha included).
    pub blend: bool,
    /// `true` → after rendering, clear the frame's bbox to the background
    /// colour before drawing the next frame.
    pub dispose_to_background: bool,
    /// Row-major RGBA bytes for this tile — `width * height * 4` long.
    pub rgba: &'a [u8],
}

/// Build a complete animated `.webp` file from a slice of frames + a
/// canvas size. Every frame is encoded losslessly (VP8L) — for the
/// per-frame lossy/lossless mode-selection decision wired up by #335
/// see [`build_animated_webp_with_options`].
///
/// Loop count = 0 means infinite playback (the WebP default).
/// Background is BGRA; the spec writes B, G, R, A in that order — we
/// accept it the same way.
pub fn build_animated_webp(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
) -> Result<Vec<u8>> {
    build_animated_webp_with_options(
        canvas_w,
        canvas_h,
        background_bgra,
        loop_count,
        frames,
        AnimEncoderOptions {
            mode: AnimFrameMode::Lossless,
            ..AnimEncoderOptions::default()
        },
    )
}

/// Build an animated `.webp` file with explicit encoder options.
/// See [`AnimEncoderOptions`] for the knobs; the default policy is
/// per-frame mode auto-selection (whichever of VP8L / VP8+ALPH is
/// byte-smaller wins per frame).
pub fn build_animated_webp_with_options(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
    options: AnimEncoderOptions<'_>,
) -> Result<Vec<u8>> {
    if canvas_w == 0 || canvas_h == 0 {
        return Err(Error::invalid("animated WebP: zero canvas size"));
    }
    if canvas_w > 16384 || canvas_h > 16384 {
        return Err(Error::invalid("animated WebP: canvas exceeds 16384 px"));
    }
    if frames.is_empty() {
        return Err(Error::invalid("animated WebP: needs at least one frame"));
    }

    // Delta mode rewrites the input frame stream into per-frame sub-rect
    // tiles + an internal "auto"-fallback policy before falling through
    // to the standard layout loop. Doing the rewrite up front keeps the
    // RIFF body assembly path identical for every mode.
    if let AnimFrameMode::Delta(cfg) = options.mode {
        return build_animated_webp_delta(
            canvas_w,
            canvas_h,
            background_bgra,
            loop_count,
            frames,
            &options,
            cfg,
        );
    }

    // Pre-encode every frame's nested image sub-chunk(s) first. Doing
    // it up front lets us measure each chunk and lay out the RIFF body
    // in a single pass without a second iteration. Track whether *any*
    // frame carries non-opaque alpha — the VP8X ALPHA flag should only
    // be set when at least one frame actually needs alpha, otherwise
    // strict readers see the flag set with no real alpha and treat it
    // as a malformed file.
    let mut any_frame_has_alpha = false;
    let mut anmf_payloads: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for f in frames {
        if f.width == 0 || f.height == 0 {
            return Err(Error::invalid("animated WebP: zero frame size"));
        }
        if f.x_offset
            .checked_add(f.width)
            .map(|r| r > canvas_w)
            .unwrap_or(true)
            || f.y_offset
                .checked_add(f.height)
                .map(|r| r > canvas_h)
                .unwrap_or(true)
        {
            return Err(Error::invalid(
                "animated WebP: frame bbox extends past canvas",
            ));
        }
        if f.rgba.len() != (f.width as usize) * (f.height as usize) * 4 {
            return Err(Error::invalid(
                "animated WebP: frame rgba length mismatch frame_w*frame_h*4",
            ));
        }
        if f.duration_ms > 0x00FF_FFFF {
            return Err(Error::invalid(
                "animated WebP: duration_ms exceeds 24-bit field",
            ));
        }

        // Detect non-opaque alpha for the canvas-level VP8X flag. We
        // can't piggyback off the per-frame encode (the lossy path
        // checks `any(!= 0xff)` inside encode_lossy_anmf) because the
        // mode-select decision can drop that signal — scan once here
        // so the canvas flag is correct regardless of mode.
        if !any_frame_has_alpha && f.rgba.chunks_exact(4).any(|px| px[3] != 0xff) {
            any_frame_has_alpha = true;
        }

        // Per-frame mode selection: produce the requested encoding(s)
        // and pick whichever sub-chunk(s) lay out the smaller ANMF
        // payload. The choice is per-frame so an animation can mix
        // lossless and lossy frames depending on which wins on each.
        let chosen = encode_one_anmf_image(f, &options)?;

        // Build the ANMF payload (16-byte header + nested image sub-chunks).
        let nested_capacity = chosen.iter().map(|c| 8 + c.payload.len()).sum::<usize>();
        let mut payload = Vec::with_capacity(16 + nested_capacity);
        // Even offsets; the spec stores them divided by 2.
        write_u24_le(&mut payload, (f.x_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.y_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.width - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.height - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, f.duration_ms & 0x00FF_FFFF);
        // bit 0: blending — 0 = use alpha blending, 1 = overwrite.
        // bit 1: disposal — 0 = none, 1 = dispose-to-background.
        let mut flags: u8 = 0;
        if !f.blend {
            flags |= 0x01;
        }
        if f.dispose_to_background {
            flags |= 0x02;
        }
        payload.push(flags);

        // Nested image sub-chunk(s) inside the ANMF body. Either
        // [VP8L] or [ALPH, VP8 ] depending on the per-frame decision.
        for sub in &chosen {
            write_chunk(&mut payload, &sub.fourcc, &sub.payload);
        }
        anmf_payloads.push(payload);
    }

    // Assemble the body that lives between "WEBP" and the end of the
    // RIFF envelope: VP8X header + [ICCP] + ANIM + N x ANMF + [EXIF] + [XMP ].
    let mut body: Vec<u8> = Vec::new();

    // VP8X flags byte:
    //   bit 1 (0x02) = ANIM   — always set for an animation.
    //   bit 4 (0x10) = ALPHA  — set iff any frame carries non-opaque alpha.
    //   bit 5 (0x20) = ICCP   — set iff `meta.icc.is_some()`.
    //   bit 3 (0x08) = EXIF   — set iff `meta.exif.is_some()`.
    //   bit 2 (0x04) = XMP    — set iff `meta.xmp.is_some()`.
    let mut flags: u8 = 0x02; // ANIM
    if any_frame_has_alpha {
        flags |= 0x10;
    }
    if options.metadata.icc.is_some() {
        flags |= 0x20;
    }
    if options.metadata.exif.is_some() {
        flags |= 0x08;
    }
    if options.metadata.xmp.is_some() {
        flags |= 0x04;
    }
    let vp8x = vp8x_payload(flags, canvas_w, canvas_h);
    write_chunk(&mut body, b"VP8X", &vp8x);

    // ICCP must come immediately after VP8X per the WebP container spec.
    if let Some(icc) = options.metadata.icc {
        write_chunk(&mut body, b"ICCP", icc);
    }

    // ANIM chunk: 4 bytes BGRA + 2 bytes loop count.
    let mut anim = [0u8; 6];
    anim[0] = background_bgra[0];
    anim[1] = background_bgra[1];
    anim[2] = background_bgra[2];
    anim[3] = background_bgra[3];
    anim[4] = (loop_count & 0xff) as u8;
    anim[5] = ((loop_count >> 8) & 0xff) as u8;
    write_chunk(&mut body, b"ANIM", &anim);

    // ANMF chunks.
    for payload in &anmf_payloads {
        write_chunk(&mut body, b"ANMF", payload);
    }

    // EXIF / XMP follow the image-data chunks per the WebP container spec.
    if let Some(exif) = options.metadata.exif {
        write_chunk(&mut body, b"EXIF", exif);
    }
    if let Some(xmp) = options.metadata.xmp {
        write_chunk(&mut body, b"XMP ", xmp);
    }

    // RIFF envelope.
    let riff_size = 4 + body.len();
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(&body);
    Ok(out)
}

/// One nested image sub-chunk inside an `ANMF` payload (`VP8L`,
/// `VP8 `, or `ALPH`). Pre-assembled — `payload` is the bytes that go
/// after the chunk header.
struct AnmfSubChunk {
    fourcc: [u8; 4],
    payload: Vec<u8>,
}

/// Encode a single animated-frame image into the nested ANMF
/// sub-chunk(s) the per-frame mode policy selects. Returns either
/// `[VP8L]` (lossless) or `[ALPH, VP8 ]` (lossy colour + lossless
/// alpha) depending on `options.mode`. With `Auto`, both encodings
/// are produced and the byte-smaller wins (sum of sub-chunk header
/// + payload, mirroring the on-disk cost).
fn encode_one_anmf_image(
    f: &AnimFrame<'_>,
    options: &AnimEncoderOptions<'_>,
) -> Result<Vec<AnmfSubChunk>> {
    // Always produce the lossless candidate first — it's the
    // historic behaviour and the fallback when the lossy encode fails
    // (e.g. on too-small frames). `Delta` is rewritten upstream into
    // per-frame `Auto`-equivalent encodes — it should never reach
    // here, but treat it as `Auto` defensively.
    let lossless: Option<Vec<AnmfSubChunk>> = match options.mode {
        AnimFrameMode::Lossy => None,
        AnimFrameMode::Lossless | AnimFrameMode::Auto | AnimFrameMode::Delta(_) => {
            Some(encode_lossless_anmf(f)?)
        }
    };

    let lossy: Option<Vec<AnmfSubChunk>> = match options.mode {
        AnimFrameMode::Lossless => None,
        AnimFrameMode::Lossy | AnimFrameMode::Auto | AnimFrameMode::Delta(_) => {
            encode_lossy_anmf(f, options.lossy_quality)?
        }
    };

    match (lossless, lossy) {
        (None, None) => unreachable!("at least one mode must produce a candidate"),
        (Some(l), None) => Ok(l),
        (None, Some(l)) => Ok(l),
        (Some(ll), Some(ly)) => {
            // Auto mode: pick the smaller payload by total on-disk cost
            // (each sub-chunk costs `8 + payload + (payload & 1)`).
            let cost = |subs: &[AnmfSubChunk]| -> usize {
                subs.iter()
                    .map(|s| 8 + s.payload.len() + (s.payload.len() & 1))
                    .sum()
            };
            if cost(&ly) < cost(&ll) {
                Ok(ly)
            } else {
                Ok(ll)
            }
        }
    }
}

/// Encode a single frame as a lossless VP8L sub-chunk. Per-frame
/// `has_alpha` detection is done while scanning the RGBA buffer into
/// the packed-ARGB pixels the VP8L encoder consumes.
fn encode_lossless_anmf(f: &AnimFrame<'_>) -> Result<Vec<AnmfSubChunk>> {
    let mut pixels = Vec::with_capacity((f.width as usize) * (f.height as usize));
    let mut has_alpha = false;
    for px in f.rgba.chunks_exact(4) {
        let r = px[0] as u32;
        let g = px[1] as u32;
        let b = px[2] as u32;
        let a = px[3] as u32;
        if a != 0xff {
            has_alpha = true;
        }
        pixels.push((a << 24) | (r << 16) | (g << 8) | b);
    }
    let vp8l_bytes = encode_vp8l_argb(f.width, f.height, &pixels, has_alpha)?;
    Ok(vec![AnmfSubChunk {
        fourcc: *b"VP8L",
        payload: vp8l_bytes,
    }])
}

/// Encode a single frame as a lossy VP8 (+ optional ALPH) sub-chunk
/// pair. Mirrors the still-image encoder's RGBA → YUV420 + ALPH
/// orchestration: the colour planes go into a bare VP8 keyframe via
/// [`oxideav_vp8::encoder::encode_vp8_keyframe`], the alpha plane
/// (when not fully opaque) is compressed into an `ALPH` sub-chunk via
/// the same helper the still-image path uses.
///
/// Returns `Ok(None)` when the frame is too small for the VP8
/// encoder (e.g. <16 px on a side, where the keyframe would have
/// no MBs to emit) — the auto-mode caller falls back to lossless.
fn encode_lossy_anmf(f: &AnimFrame<'_>, quality: f32) -> Result<Option<Vec<AnmfSubChunk>>> {
    // VP8 needs at least a single 16×16 macroblock; smaller frames
    // can't go through the lossy path. Fall back to lossless silently.
    if f.width == 0 || f.height == 0 {
        return Ok(None);
    }
    let qindex = crate::encoder_vp8::quality_to_qindex(quality);

    let w = f.width as usize;
    let h = f.height as usize;
    let mut alpha_plane: Vec<u8> = Vec::with_capacity(w * h);
    let (y_plane, u_plane, v_plane) =
        crate::encoder_vp8::rgba_rows_to_yuv420(w, h, w * 4, f.rgba, &mut alpha_plane);

    // Detect "fully opaque" so we can skip the ALPH sub-chunk on
    // animations whose frames don't carry alpha (smaller payload).
    let has_alpha = alpha_plane.iter().any(|&a| a != 0xff);

    let vp8_frame = oxideav_vp8::Vp8Frame {
        width: f.width,
        height: f.height,
        pts: None,
        y: y_plane,
        u: u_plane,
        v: v_plane,
        y_stride: f.width,
        uv_stride: (f.width + 1) / 2,
    };
    let vp8_bytes =
        match oxideav_vp8::encoder::encode_vp8_keyframe(f.width, f.height, qindex, &vp8_frame) {
            Ok(b) => b,
            // VP8 keyframe encode failed (e.g. dimensions too small or
            // some other validation). Fall back to lossless.
            Err(_) => return Ok(None),
        };

    let mut subs: Vec<AnmfSubChunk> = Vec::with_capacity(2);
    if has_alpha {
        let alph = crate::encoder_vp8::encode_alph_chunk(f.width, f.height, &alpha_plane)
            .map_err(|e| Error::invalid(format!("animated WebP: ALPH encode: {e}")))?;
        // ALPH on disk: 1 header byte + payload bytes.
        let mut alph_payload = Vec::with_capacity(1 + alph.payload.len());
        alph_payload.push(alph.header_byte);
        alph_payload.extend_from_slice(&alph.payload);
        subs.push(AnmfSubChunk {
            fourcc: *b"ALPH",
            payload: alph_payload,
        });
    }
    subs.push(AnmfSubChunk {
        fourcc: *b"VP8 ",
        payload: vp8_bytes,
    });
    Ok(Some(subs))
}

/// AVIF-style delta-merge entry point — see [`AnimFrameMode::Delta`].
///
/// Validates Delta-mode caller constraints (canvas-sized full frames,
/// no per-frame disposal/blend), then for each non-first frame computes
/// the changed-region bounding box via [`changed_block_bbox`] and emits
/// either a sub-rect ANMF (cost-model says small change) or a full-frame
/// ANMF (cost-model says full repaint, or first frame). The returned
/// blob layout is byte-identical to the standard
/// [`build_animated_webp_with_options`] output — the only difference is
/// per-frame ANMF bbox/sub-rect placement and the `blending_method` bit.
fn build_animated_webp_delta(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
    options: &AnimEncoderOptions<'_>,
    cfg: DeltaConfig,
) -> Result<Vec<u8>> {
    // Validate cfg defensively first — caller-controlled.
    if cfg.block_size == 0 {
        return Err(Error::invalid(
            "animated WebP delta: block_size must be ≥ 1",
        ));
    }
    if !(cfg.max_bbox_fraction >= 0.0 && cfg.max_bbox_fraction <= 1.0) {
        return Err(Error::invalid(
            "animated WebP delta: max_bbox_fraction must be in [0.0, 1.0]",
        ));
    }
    if let Some(n) = cfg.max_components_override {
        if n == 0 {
            return Err(Error::invalid(
                "animated WebP delta: max_components_override must be ≥ 1",
            ));
        }
    }
    if let Some(t) = cfg.auto_inner_threshold_bytes {
        // Disallow a 0-byte threshold (would force every tile through
        // lossy and defeat the lossless-default round-trip semantics).
        if t == 0 {
            return Err(Error::invalid(
                "animated WebP delta: auto_inner_threshold_bytes must be ≥ 1",
            ));
        }
    }
    if cfg.auto_inner_threshold_bytes.is_some()
        && !(cfg.auto_inner_quality >= 0.0 && cfg.auto_inner_quality <= 100.0)
    {
        return Err(Error::invalid(
            "animated WebP delta: auto_inner_quality must be in [0.0, 100.0]",
        ));
    }

    // Validate caller constraints + collect per-frame full-canvas RGBA
    // for the cost-model comparison. We require frames that cover the
    // whole canvas because we have to reconstruct "what the prior
    // canvas looks like" to diff against, and the simplest invariant
    // is `prior_canvas[i] == frames[i-1].rgba` (which only holds when
    // each frame paints the entire canvas, blend=false, dispose=false).
    for (i, f) in frames.iter().enumerate() {
        if f.width != canvas_w || f.height != canvas_h {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must be canvas-sized ({canvas_w}x{canvas_h}), got {}x{}",
                f.width, f.height
            )));
        }
        if f.x_offset != 0 || f.y_offset != 0 {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must be at origin (0,0), got ({},{})",
                f.x_offset, f.y_offset
            )));
        }
        if f.blend {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must have blend=false (Delta mode forces overwrite)"
            )));
        }
        if f.dispose_to_background {
            return Err(Error::invalid(format!(
                "animated WebP delta: frame {i} must have dispose_to_background=false (would invalidate the prior-canvas reference)"
            )));
        }
        if f.duration_ms > 0x00FF_FFFF {
            return Err(Error::invalid(
                "animated WebP delta: duration_ms exceeds 24-bit field",
            ));
        }
        if f.rgba.len() != (f.width as usize) * (f.height as usize) * 4 {
            return Err(Error::invalid(
                "animated WebP delta: frame rgba length mismatch frame_w*frame_h*4",
            ));
        }
    }

    // Build the rewritten frame list. The first frame is always full-
    // canvas; each subsequent frame is either (a) a sub-rect tile sized
    // to the changed-block bbox, with `blend = false` so the decoder
    // overwrites the matching canvas region, or (b) a full-canvas
    // refresh when the cost-model bbox is too large to win.
    //
    // We carry the source RGBA for sub-rect frames in a transient
    // `Vec<u8>` since the sub-rect doesn't exist as a contiguous slice
    // in the original buffer (different stride). For full frames we
    // reuse the caller's slice — no copy.
    let max_pixels = (canvas_w as u64).saturating_mul(canvas_h as u64);
    let max_bbox_pixels = ((max_pixels as f64) * (cfg.max_bbox_fraction as f64)) as u64;
    let mut tile_storage: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    // For each output frame, the layout (offset, width, height, blend,
    // duration). The actual rgba slice is resolved at encode time from
    // either the caller's frame.rgba (full-canvas) or `tile_storage`
    // (sub-rect).
    struct PlannedFrame {
        x_offset: u32,
        y_offset: u32,
        width: u32,
        height: u32,
        duration_ms: u32,
        blend: bool,
        // index into either the caller's `frames` (rgba_kind=Full) or
        // `tile_storage` (rgba_kind=Tile).
        rgba_kind: RgbaKind,
        rgba_idx: usize,
    }
    enum RgbaKind {
        Full,
        Tile,
    }

    let mut planned: Vec<PlannedFrame> = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        if i == 0 {
            // First frame: always full-canvas.
            planned.push(PlannedFrame {
                x_offset: 0,
                y_offset: 0,
                width: f.width,
                height: f.height,
                duration_ms: f.duration_ms,
                blend: f.blend,
                rgba_kind: RgbaKind::Full,
                rgba_idx: i,
            });
            continue;
        }
        // Multi-rect cost-model: find the connected components of the
        // changed-block grid, capped at the per-frame budget (either
        // `cfg.max_components_override` if set, else
        // `adaptive_max_components(cluster_density)` chosen from the
        // raw cluster pixel-area sum returned by the cost-model walk).
        // We need the raw density to pick the budget, but the merge
        // step itself depends on the budget — chain the helper so the
        // walk runs once and the budget is plumbed through.
        let prior = &frames[i - 1];
        let budget = match cfg.max_components_override {
            Some(n) => n,
            None => {
                // Probe density first by running the flood-fill with
                // an "infinite" budget — `u32::MAX` skips the merge
                // loop entirely and we get the pre-merge cluster
                // pixel area for the density mapping. This is one
                // extra walk per frame, but it's pure-pixel-arith on
                // a block-grid (block-grid is canvas / block_size² in
                // size — a few hundred entries on typical content).
                let (_, raw_pixels) = changed_block_components_with_density(
                    prior.rgba,
                    f.rgba,
                    canvas_w,
                    canvas_h,
                    &cfg,
                    u32::MAX,
                );
                let density = if max_pixels == 0 {
                    0.0
                } else {
                    (raw_pixels as f32) / (max_pixels as f32)
                };
                adaptive_max_components(density)
            }
        };
        let components =
            changed_block_components(prior.rgba, f.rgba, canvas_w, canvas_h, &cfg, budget);

        if components.is_empty() {
            // Identical — emit a 1×1 (smallest the spec allows)
            // overwrite at (0,0) with the prior pixel: zero visible
            // change and the smallest possible payload.
            let p0 = &prior.rgba[..4];
            let mut tile = Vec::with_capacity(4);
            tile.extend_from_slice(p0);
            let tile_idx = tile_storage.len();
            tile_storage.push(tile);
            planned.push(PlannedFrame {
                x_offset: 0,
                y_offset: 0,
                width: 1,
                height: 1,
                duration_ms: f.duration_ms,
                blend: false, // overwrite (DoNotBlend)
                rgba_kind: RgbaKind::Tile,
                rgba_idx: tile_idx,
            });
            continue;
        }

        // Compute the union bbox (the single-bbox cover). If that
        // covers more than `max_bbox_fraction` of the canvas AND the
        // multi-rect total area is also that big, fall back to a
        // full-canvas refresh — neither shape is going to win.
        let union_bbox = bbox_union(&components);
        let union_pixels = (union_bbox.2 as u64) * (union_bbox.3 as u64);
        let multi_pixels: u64 = components
            .iter()
            .map(|&(_, _, w, h)| (w as u64) * (h as u64))
            .sum();

        // Multi-rect wins on wire when the sum of per-component
        // pixel counts is materially smaller than the single-bbox
        // cover (≥ 25% saving) AND we have more than one component.
        // Otherwise the per-ANMF chunk overhead + per-tile VP8L
        // header outweigh the bbox savings.
        let use_multi = components.len() > 1
            && multi_pixels.saturating_mul(4) < union_pixels.saturating_mul(3)
            && multi_pixels <= max_bbox_pixels;

        if !use_multi {
            // Single-bbox path (historical) or fall-back.
            if union_pixels > max_bbox_pixels {
                // Bbox too large — emit full-canvas refresh.
                planned.push(PlannedFrame {
                    x_offset: 0,
                    y_offset: 0,
                    width: f.width,
                    height: f.height,
                    duration_ms: f.duration_ms,
                    blend: f.blend,
                    rgba_kind: RgbaKind::Full,
                    rgba_idx: i,
                });
            } else {
                let (bx, by, bw, bh) = union_bbox;
                let tile = extract_subrect(f.rgba, canvas_w, bx, by, bw, bh);
                let tile_idx = tile_storage.len();
                tile_storage.push(tile);
                planned.push(PlannedFrame {
                    x_offset: bx,
                    y_offset: by,
                    width: bw,
                    height: bh,
                    duration_ms: f.duration_ms,
                    blend: false, // overwrite (DoNotBlend)
                    rgba_kind: RgbaKind::Tile,
                    rgba_idx: tile_idx,
                });
            }
            continue;
        }

        // Multi-rect emission: one ANMF per component, with
        // `duration_ms = 0` on every sub-rect except the last
        // (which carries the input frame's duration so total display
        // time is preserved). The decoder paints each tile in turn,
        // composing them onto the persistent canvas.
        let n = components.len();
        for (k, &(bx, by, bw, bh)) in components.iter().enumerate() {
            let tile = extract_subrect(f.rgba, canvas_w, bx, by, bw, bh);
            let tile_idx = tile_storage.len();
            tile_storage.push(tile);
            // Last sub-rect carries the input frame's duration; the
            // earlier ones are zero-duration "instant overwrites" so
            // the human-perceived frame timing is unchanged.
            let dur = if k + 1 == n { f.duration_ms } else { 0 };
            planned.push(PlannedFrame {
                x_offset: bx,
                y_offset: by,
                width: bw,
                height: bh,
                duration_ms: dur,
                blend: false, // overwrite (DoNotBlend)
                rgba_kind: RgbaKind::Tile,
                rgba_idx: tile_idx,
            });
        }
    }

    // Construct the rewritten frame list, borrowing into either the
    // original input slice (full-canvas frames) or the per-tile
    // `tile_storage` buffer (sub-rect frames). Track per-frame
    // "is sub-rect tile" alongside it — the auto-inner-encode threshold
    // only applies to sub-rect tiles, never to full-canvas frames (which
    // include the always-full first frame and the cost-model fallback
    // full-canvas refresh).
    let rewritten: Vec<AnimFrame<'_>> = planned
        .iter()
        .map(|p| {
            let rgba: &[u8] = match p.rgba_kind {
                RgbaKind::Full => frames[p.rgba_idx].rgba,
                RgbaKind::Tile => tile_storage[p.rgba_idx].as_slice(),
            };
            AnimFrame {
                width: p.width,
                height: p.height,
                x_offset: p.x_offset,
                y_offset: p.y_offset,
                duration_ms: p.duration_ms,
                blend: p.blend,
                dispose_to_background: false,
                rgba,
            }
        })
        .collect();
    let is_subrect_tile: Vec<bool> = planned
        .iter()
        .map(|p| matches!(p.rgba_kind, RgbaKind::Tile))
        .collect();

    // Without the auto-inner-encode threshold, every sub-rect re-encodes
    // losslessly (the historical behaviour). Sub-rect tiles produced by
    // Delta are typically tiny (≤ a few KB raw RGBA), so the VP8 keyframe
    // overhead would win on byte count and the rebuild would also incur
    // expensive RDO during the per-frame Auto-mode candidate evaluation.
    // Forcing lossless keeps Delta-mode encodes deterministic + fast and
    // preserves pixel-identical round-trip semantics.
    if cfg.auto_inner_threshold_bytes.is_none() {
        let inner_options = AnimEncoderOptions {
            mode: AnimFrameMode::Lossless,
            lossy_quality: options.lossy_quality,
            metadata: options.metadata.clone(),
        };
        return build_animated_webp_with_options(
            canvas_w,
            canvas_h,
            background_bgra,
            loop_count,
            &rewritten,
            inner_options,
        );
    }

    // Auto-inner-encode path: per-tile, encode lossless first; if the
    // payload exceeds `cfg.auto_inner_threshold_bytes`, also encode
    // lossy at `cfg.auto_inner_quality` and pick the byte-smaller
    // candidate. Tiles below the cutoff stay lossless (no quality loss
    // for the common-case small-tile path). The body assembly mirrors
    // `build_animated_webp_with_options` — kept inline because the
    // mode decision is per-tile, not per-call.
    let threshold = cfg.auto_inner_threshold_bytes.unwrap();
    let lossy_inner_quality = cfg.auto_inner_quality;
    build_animated_webp_inner_per_tile(
        canvas_w,
        canvas_h,
        background_bgra,
        loop_count,
        &rewritten,
        &is_subrect_tile,
        &options.metadata,
        threshold,
        lossy_inner_quality,
    )
}

/// Inner-assembly path for the [`AnimFrameMode::Delta`] flow when
/// `cfg.auto_inner_threshold_bytes` is set. Mirrors
/// [`build_animated_webp_with_options`]'s body-assembly loop, but the
/// per-tile encoding decision uses the Delta-mode "lossless first,
/// lossy only if lossless > threshold" rule. The `is_subrect_tile[i]`
/// flag gates the threshold per-frame: full-canvas frames (the first
/// frame + cost-model fallback) always stay lossless regardless of
/// size, only genuine sub-rect tiles can be promoted to the lossy
/// candidate.
#[allow(clippy::too_many_arguments)]
fn build_animated_webp_inner_per_tile(
    canvas_w: u32,
    canvas_h: u32,
    background_bgra: [u8; 4],
    loop_count: u16,
    frames: &[AnimFrame<'_>],
    is_subrect_tile: &[bool],
    metadata: &WebpMetadata<'_>,
    auto_inner_threshold_bytes: u32,
    auto_inner_quality: f32,
) -> Result<Vec<u8>> {
    debug_assert_eq!(
        frames.len(),
        is_subrect_tile.len(),
        "frames and is_subrect_tile must be parallel"
    );
    let mut any_frame_has_alpha = false;
    let mut anmf_payloads: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
    for (frame_idx, f) in frames.iter().enumerate() {
        // Validate identical to `build_animated_webp_with_options`'s
        // pre-loop. We've already validated the source frames upstream
        // — this loop sees the rewritten Delta tiles which the planner
        // built from validated inputs, so the bbox / size checks are
        // structural assertions here, not user-facing errors.
        if f.width == 0 || f.height == 0 {
            return Err(Error::invalid("animated WebP delta-inner: zero tile size"));
        }
        if f.x_offset
            .checked_add(f.width)
            .map(|r| r > canvas_w)
            .unwrap_or(true)
            || f.y_offset
                .checked_add(f.height)
                .map(|r| r > canvas_h)
                .unwrap_or(true)
        {
            return Err(Error::invalid(
                "animated WebP delta-inner: tile bbox extends past canvas",
            ));
        }
        if f.rgba.len() != (f.width as usize) * (f.height as usize) * 4 {
            return Err(Error::invalid(
                "animated WebP delta-inner: tile rgba length mismatch",
            ));
        }
        if !any_frame_has_alpha && f.rgba.chunks_exact(4).any(|px| px[3] != 0xff) {
            any_frame_has_alpha = true;
        }

        // Per-tile encode: always produce the lossless candidate.
        let lossless = encode_lossless_anmf(f)?;
        let lossless_bytes: usize = lossless
            .iter()
            .map(|s| 8 + s.payload.len() + (s.payload.len() & 1))
            .sum();
        // Only spend cycles on the lossy candidate when (a) this is a
        // genuine sub-rect tile (full-canvas frames always stay
        // lossless to preserve round-trip semantics on the canvas
        // baseline) AND (b) the lossless payload exceeds the threshold
        // (the common-case small tile skips lossy entirely → fast +
        // pixel-identical).
        let chosen: Vec<AnmfSubChunk> =
            if is_subrect_tile[frame_idx] && lossless_bytes > auto_inner_threshold_bytes as usize {
                match encode_lossy_anmf(f, auto_inner_quality)? {
                    Some(lossy) => {
                        let lossy_bytes: usize = lossy
                            .iter()
                            .map(|s| 8 + s.payload.len() + (s.payload.len() & 1))
                            .sum();
                        if lossy_bytes < lossless_bytes {
                            lossy
                        } else {
                            lossless
                        }
                    }
                    None => lossless,
                }
            } else {
                lossless
            };

        // Build the ANMF payload (same wire format as the standard
        // assembly loop).
        let nested_capacity = chosen.iter().map(|c| 8 + c.payload.len()).sum::<usize>();
        let mut payload = Vec::with_capacity(16 + nested_capacity);
        write_u24_le(&mut payload, (f.x_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.y_offset / 2) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.width - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, (f.height - 1) & 0x00FF_FFFF);
        write_u24_le(&mut payload, f.duration_ms & 0x00FF_FFFF);
        let mut flags: u8 = 0;
        if !f.blend {
            flags |= 0x01;
        }
        if f.dispose_to_background {
            flags |= 0x02;
        }
        payload.push(flags);
        for sub in &chosen {
            write_chunk(&mut payload, &sub.fourcc, &sub.payload);
        }
        anmf_payloads.push(payload);
    }

    // Body assembly identical to `build_animated_webp_with_options`.
    let mut body: Vec<u8> = Vec::new();
    let mut flags: u8 = 0x02; // ANIM
    if any_frame_has_alpha {
        flags |= 0x10;
    }
    if metadata.icc.is_some() {
        flags |= 0x20;
    }
    if metadata.exif.is_some() {
        flags |= 0x08;
    }
    if metadata.xmp.is_some() {
        flags |= 0x04;
    }
    let vp8x = vp8x_payload(flags, canvas_w, canvas_h);
    write_chunk(&mut body, b"VP8X", &vp8x);
    if let Some(icc) = metadata.icc {
        write_chunk(&mut body, b"ICCP", icc);
    }
    let mut anim = [0u8; 6];
    anim[0] = background_bgra[0];
    anim[1] = background_bgra[1];
    anim[2] = background_bgra[2];
    anim[3] = background_bgra[3];
    anim[4] = (loop_count & 0xff) as u8;
    anim[5] = ((loop_count >> 8) & 0xff) as u8;
    write_chunk(&mut body, b"ANIM", &anim);
    for payload in &anmf_payloads {
        write_chunk(&mut body, b"ANMF", payload);
    }
    if let Some(exif) = metadata.exif {
        write_chunk(&mut body, b"EXIF", exif);
    }
    if let Some(xmp) = metadata.xmp {
        write_chunk(&mut body, b"XMP ", xmp);
    }
    let riff_size = 4 + body.len();
    let mut out = Vec::with_capacity(8 + riff_size);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(&body);
    Ok(out)
}

/// Copy the `bw × bh` sub-rectangle starting at `(bx, by)` out of an
/// RGBA buffer with `canvas_w` pixels per row. Allocates a fresh
/// `Vec<u8>` of length `bw * bh * 4`. Caller guarantees the bbox stays
/// inside the canvas.
fn extract_subrect(rgba: &[u8], canvas_w: u32, bx: u32, by: u32, bw: u32, bh: u32) -> Vec<u8> {
    let canvas_w = canvas_w as usize;
    let bx = bx as usize;
    let by = by as usize;
    let bw = bw as usize;
    let bh = bh as usize;
    let mut out = Vec::with_capacity(bw * bh * 4);
    for row in 0..bh {
        let src_off = ((by + row) * canvas_w + bx) * 4;
        out.extend_from_slice(&rgba[src_off..src_off + bw * 4]);
    }
    out
}

/// Compute the boolean `n_bx × n_by` "changed" grid for the cost-model
/// (true ⇔ block cost > active threshold). Pulled out of
/// [`changed_block_bbox`] / [`changed_block_components`] so both share
/// one walk over the canvas pixels. Dispatches between the SAD,
/// single-scale SSIM-lite, and 3-scale MS-SSIM-lite cost models based
/// on `cfg.enable_msssim_cost` / `cfg.enable_ssim_cost` (MS-SSIM
/// supersedes single-scale when both are on).
fn compute_changed_grid(
    prev: &[u8],
    curr: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    cfg: &DeltaConfig,
) -> (Vec<bool>, u32, u32) {
    let bs = cfg.block_size;
    let cw = canvas_w as usize;
    let ch = canvas_h as usize;
    let n_bx = canvas_w.div_ceil(bs);
    let n_by = canvas_h.div_ceil(bs);
    let mut grid = vec![false; (n_bx as usize) * (n_by as usize)];
    let threshold = if cfg.enable_msssim_cost {
        cfg.msssim_threshold as u64
    } else if cfg.enable_ssim_cost {
        cfg.ssim_threshold as u64
    } else {
        cfg.threshold as u64
    };
    for by in 0..n_by {
        let y0 = (by * bs) as usize;
        let y1 = ((by + 1) * bs).min(canvas_h) as usize;
        for bx in 0..n_bx {
            let x0 = (bx * bs) as usize;
            let x1 = ((bx + 1) * bs).min(canvas_w) as usize;
            let cost = if cfg.enable_msssim_cost {
                block_cost_msssim(
                    prev,
                    curr,
                    cw,
                    ch,
                    x0,
                    y0,
                    x1,
                    y1,
                    cfg.msssim_downsample_kernel,
                )
            } else if cfg.enable_ssim_cost {
                block_cost_ssim(prev, curr, cw, x0, y0, x1, y1)
            } else {
                block_cost(prev, curr, cw, x0, y0, x1, y1)
            };
            if cost > threshold {
                grid[(by as usize) * (n_bx as usize) + bx as usize] = true;
            }
        }
    }
    (grid, n_bx, n_by)
}

/// Convert a block-grid bbox `(min_bx, min_by, max_bx, max_by)` (all
/// inclusive) into a pixel bbox `(px, py, pw, ph)` with even-aligned
/// offsets and dimensions clipped to the canvas. Centralises the
/// "block-grid → ANMF-spec-compliant pixel rect" step shared by every
/// component.
fn block_bbox_to_pixel_bbox(
    min_bx: u32,
    min_by: u32,
    max_bx: u32,
    max_by: u32,
    canvas_w: u32,
    canvas_h: u32,
    bs: u32,
) -> (u32, u32, u32, u32) {
    let mut px = min_bx * bs;
    let mut py = min_by * bs;
    let mut pw = ((max_bx + 1) * bs).min(canvas_w) - px;
    let mut ph = ((max_by + 1) * bs).min(canvas_h) - py;

    // ANMF spec mandates even offsets — round (px, py) down to even,
    // and grow (pw, ph) to compensate.
    if px % 2 != 0 {
        px -= 1;
        pw += 1;
    }
    if py % 2 != 0 {
        py -= 1;
        ph += 1;
    }
    if px + pw > canvas_w {
        pw = canvas_w - px;
    }
    if py + ph > canvas_h {
        ph = canvas_h - py;
    }
    (px, py, pw, ph)
}

/// Find the 4-connected components on the changed-block grid + apply
/// the merge-to-budget pass. Returns a list of even-aligned pixel
/// bboxes suitable for direct ANMF emission.
///
/// The `max_components` budget is determined by the caller (either the
/// explicit [`DeltaConfig::max_components_override`] or the adaptive
/// [`adaptive_max_components`] of the per-frame cluster density). To
/// pick the adaptive budget the caller needs the raw cluster area sum
/// — see [`changed_block_components_with_density`].
fn changed_block_components(
    prev: &[u8],
    curr: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    cfg: &DeltaConfig,
    max_components: u32,
) -> Vec<(u32, u32, u32, u32)> {
    changed_block_components_with_density(prev, curr, canvas_w, canvas_h, cfg, max_components).0
}

/// Same as [`changed_block_components`] but also returns the raw
/// (pre-merge) cluster pixel-area sum. The density-aware caller can
/// recompute the budget from this sum and re-call with a different
/// `max_components` value if needed (and we provide the
/// [`adaptive_max_components`] helper for that).
fn changed_block_components_with_density(
    prev: &[u8],
    curr: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    cfg: &DeltaConfig,
    max_components: u32,
) -> (Vec<(u32, u32, u32, u32)>, u64) {
    let bs = cfg.block_size;
    let (grid, n_bx, n_by) = compute_changed_grid(prev, curr, canvas_w, canvas_h, cfg);
    let nx = n_bx as usize;
    let ny = n_by as usize;
    if nx == 0 || ny == 0 {
        return (Vec::new(), 0);
    }

    // 4-connected flood fill — `label[i]` is the component id of
    // grid cell i (or `u32::MAX` for unset / non-changed).
    let mut label = vec![u32::MAX; nx * ny];
    let mut bboxes: Vec<(u32, u32, u32, u32)> = Vec::new(); // (min_bx, min_by, max_bx, max_by) per component
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for sy in 0..n_by {
        for sx in 0..n_bx {
            let idx = (sy as usize) * nx + sx as usize;
            if !grid[idx] || label[idx] != u32::MAX {
                continue;
            }
            // New component — flood-fill from (sx, sy).
            let comp_id = bboxes.len() as u32;
            let mut min_bx = sx;
            let mut min_by = sy;
            let mut max_bx = sx;
            let mut max_by = sy;
            stack.clear();
            stack.push((sx, sy));
            label[idx] = comp_id;
            while let Some((x, y)) = stack.pop() {
                if x < min_bx {
                    min_bx = x;
                }
                if x > max_bx {
                    max_bx = x;
                }
                if y < min_by {
                    min_by = y;
                }
                if y > max_by {
                    max_by = y;
                }
                // 4-neighbours.
                let ux = x as usize;
                let uy = y as usize;
                if ux + 1 < nx {
                    let n = uy * nx + ux + 1;
                    if grid[n] && label[n] == u32::MAX {
                        label[n] = comp_id;
                        stack.push((x + 1, y));
                    }
                }
                if ux > 0 {
                    let n = uy * nx + ux - 1;
                    if grid[n] && label[n] == u32::MAX {
                        label[n] = comp_id;
                        stack.push((x - 1, y));
                    }
                }
                if uy + 1 < ny {
                    let n = (uy + 1) * nx + ux;
                    if grid[n] && label[n] == u32::MAX {
                        label[n] = comp_id;
                        stack.push((x, y + 1));
                    }
                }
                if uy > 0 {
                    let n = (uy - 1) * nx + ux;
                    if grid[n] && label[n] == u32::MAX {
                        label[n] = comp_id;
                        stack.push((x, y - 1));
                    }
                }
            }
            bboxes.push((min_bx, min_by, max_bx, max_by));
        }
    }

    // Sum up the pre-merge cluster pixel-area (cluster_density numerator)
    // *before* the merge step rewrites bboxes. Each cluster's pixel area
    // is the bbox-on-pixel-grid area (clipped to canvas), which gives the
    // caller the right metric for the adaptive-budget mapping.
    let raw_cluster_pixels: u64 = bboxes
        .iter()
        .map(|&(mnx, mny, mxx, mxy)| {
            let (_, _, pw, ph) =
                block_bbox_to_pixel_bbox(mnx, mny, mxx, mxy, canvas_w, canvas_h, bs);
            (pw as u64) * (ph as u64)
        })
        .sum();

    // Budget enforcement: while we have more components than allowed,
    // pick the smallest (by block-area) component and merge it into
    // its nearest neighbour (axis-aligned inter-bbox Manhattan-style
    // distance — if the bboxes overlap, distance = 0 and that pair
    // wins immediately). Repeat until at-budget.
    while bboxes.len() > max_components as usize && bboxes.len() > 1 {
        // Find smallest by area.
        let smallest_idx = bboxes
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| {
                let w = (b.2 - b.0 + 1) as u64;
                let h = (b.3 - b.1 + 1) as u64;
                w * h
            })
            .map(|(i, _)| i)
            .unwrap();
        let small = bboxes[smallest_idx];
        // Find nearest neighbour by axis-aligned bbox gap.
        let mut best_other = usize::MAX;
        let mut best_dist = u64::MAX;
        for (j, b) in bboxes.iter().enumerate() {
            if j == smallest_idx {
                continue;
            }
            // Axis-aligned gap on each axis (0 if overlapping/touching).
            let gx = axis_gap(small.0, small.2, b.0, b.2);
            let gy = axis_gap(small.1, small.3, b.1, b.3);
            // Use squared-distance-like metric so a (1,1) gap beats
            // a (2,0) gap (the (2,0) merger creates a long thin bbox).
            let d = (gx as u64) * (gx as u64) + (gy as u64) * (gy as u64);
            if d < best_dist {
                best_dist = d;
                best_other = j;
            }
        }
        if best_other == usize::MAX {
            break; // shouldn't happen given len > 1 check
        }
        let other = bboxes[best_other];
        let merged = (
            small.0.min(other.0),
            small.1.min(other.1),
            small.2.max(other.2),
            small.3.max(other.3),
        );
        // Remove the higher index first to keep the lower one valid.
        let (hi, lo) = if smallest_idx > best_other {
            (smallest_idx, best_other)
        } else {
            (best_other, smallest_idx)
        };
        bboxes.swap_remove(hi);
        bboxes[lo] = merged;
    }

    // Convert each block-grid bbox to a pixel bbox.
    let pixel_rects: Vec<(u32, u32, u32, u32)> = bboxes
        .into_iter()
        .map(|(mnx, mny, mxx, mxy)| {
            block_bbox_to_pixel_bbox(mnx, mny, mxx, mxy, canvas_w, canvas_h, bs)
        })
        .collect();
    (pixel_rects, raw_cluster_pixels)
}

/// One-dimensional gap between two intervals `[a0, a1]` and `[b0, b1]`
/// (block-grid coordinates, both ends inclusive). Zero when the
/// intervals touch or overlap.
fn axis_gap(a0: u32, a1: u32, b0: u32, b1: u32) -> u32 {
    if a1 + 1 < b0 {
        b0 - a1 - 1
    } else if b1 + 1 < a0 {
        a0 - b1 - 1
    } else {
        0
    }
}

/// Bounding box of a list of pixel bboxes (the tightest axis-aligned
/// rect that contains every input rect). Caller guarantees non-empty
/// input. Used for the "single-bbox cover" fallback.
fn bbox_union(bboxes: &[(u32, u32, u32, u32)]) -> (u32, u32, u32, u32) {
    let mut x0 = u32::MAX;
    let mut y0 = u32::MAX;
    let mut x1 = 0u32;
    let mut y1 = 0u32;
    for &(x, y, w, h) in bboxes {
        if x < x0 {
            x0 = x;
        }
        if y < y0 {
            y0 = y;
        }
        if x + w > x1 {
            x1 = x + w;
        }
        if y + h > y1 {
            y1 = y + h;
        }
    }
    // Ensure x0 stays even (matches block_bbox_to_pixel_bbox output).
    let x0 = x0 & !1;
    let y0 = y0 & !1;
    (x0, y0, x1 - x0, y1 - y0)
}

/// Walk `prev` vs `curr` (both row-major canvas-sized RGBA) on
/// `cfg.block_size`-sized blocks; for each block compute the
/// luminance-biased SAD cost; return the bounding box (in pixels,
/// even-aligned + clipped to canvas) of all blocks whose cost exceeds
/// `cfg.threshold`. Returns `None` when no block is changed (frame
/// is bit-identical or the cost-model says it's all under threshold).
///
/// Output bbox is `(x, y, w, h)` with `x`/`y` rounded down to even
/// (WebP ANMF spec mandates even offsets) and `w`/`h` adjusted so the
/// bbox still encloses every changed block.
#[allow(dead_code)]
fn changed_block_bbox(
    prev: &[u8],
    curr: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    cfg: &DeltaConfig,
) -> Option<(u32, u32, u32, u32)> {
    let bs = cfg.block_size;
    let cw = canvas_w as usize;

    // Block-grid extents (last block may be shorter than `bs` on the
    // right/bottom edge — count it as a regular block, just iterate
    // fewer pixels in that case).
    let n_bx = canvas_w.div_ceil(bs);
    let n_by = canvas_h.div_ceil(bs);

    let mut min_bx = u32::MAX;
    let mut min_by = u32::MAX;
    let mut max_bx: i64 = -1;
    let mut max_by: i64 = -1;
    let ch = canvas_h as usize;
    let threshold = if cfg.enable_msssim_cost {
        cfg.msssim_threshold as u64
    } else if cfg.enable_ssim_cost {
        cfg.ssim_threshold as u64
    } else {
        cfg.threshold as u64
    };

    for by in 0..n_by {
        let y0 = (by * bs) as usize;
        let y1 = ((by + 1) * bs).min(canvas_h) as usize;
        for bx in 0..n_bx {
            let x0 = (bx * bs) as usize;
            let x1 = ((bx + 1) * bs).min(canvas_w) as usize;
            let cost = if cfg.enable_msssim_cost {
                block_cost_msssim(
                    prev,
                    curr,
                    cw,
                    ch,
                    x0,
                    y0,
                    x1,
                    y1,
                    cfg.msssim_downsample_kernel,
                )
            } else if cfg.enable_ssim_cost {
                block_cost_ssim(prev, curr, cw, x0, y0, x1, y1)
            } else {
                block_cost(prev, curr, cw, x0, y0, x1, y1)
            };
            if cost > threshold {
                if bx < min_bx {
                    min_bx = bx;
                }
                if by < min_by {
                    min_by = by;
                }
                if bx as i64 > max_bx {
                    max_bx = bx as i64;
                }
                if by as i64 > max_by {
                    max_by = by as i64;
                }
            }
        }
    }
    if max_bx < 0 || max_by < 0 {
        return None;
    }

    // Pixel bbox from block-grid bbox.
    let mut px = min_bx * bs;
    let mut py = min_by * bs;
    let mut pw = ((max_bx as u32 + 1) * bs).min(canvas_w) - px;
    let mut ph = ((max_by as u32 + 1) * bs).min(canvas_h) - py;

    // ANMF spec mandates even offsets — round (px, py) down to even,
    // and grow (pw, ph) to compensate.
    if px % 2 != 0 {
        px -= 1;
        pw += 1;
    }
    if py % 2 != 0 {
        py -= 1;
        ph += 1;
    }
    // Clamp width/height in case rounding pushed past the canvas.
    if px + pw > canvas_w {
        pw = canvas_w - px;
    }
    if py + ph > canvas_h {
        ph = canvas_h - py;
    }
    Some((px, py, pw, ph))
}

/// Luminance-biased SAD over a block in two RGBA canvas buffers. Both
/// `prev` and `curr` are row-major with `canvas_w` pixels per row;
/// the block spans `[x0, x1) × [y0, y1)`. Computes
/// `sum |luma(prev) - luma(curr)| + 0.25 * (|R'-R| + |G'-G| + |B'-B|) +
/// |A'-A|` per pixel using fixed-point integer math (the 0.25 weight is
/// a `>> 2`). Returns the accumulated cost as `u64` to avoid overflow on
/// 8×8 blocks of fully-saturated 8-bit deltas (max ≈ 8×8×(255+3*64+255)
/// ≈ 45k — fits in u32, but use u64 for safety vs larger blocks).
fn block_cost(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> u64 {
    let mut acc: u64 = 0;
    for y in y0..y1 {
        let row_off = y * canvas_w * 4;
        for x in x0..x1 {
            let off = row_off + x * 4;
            let pr = prev[off] as i32;
            let pg = prev[off + 1] as i32;
            let pb = prev[off + 2] as i32;
            let pa = prev[off + 3] as i32;
            let cr = curr[off] as i32;
            let cg = curr[off + 1] as i32;
            let cb = curr[off + 2] as i32;
            let ca = curr[off + 3] as i32;
            // BT.601 luma (integer-scaled): 0.299R + 0.587G + 0.114B
            // → (77*R + 150*G + 29*B + 128) >> 8 (sums to 256).
            let lp = (77 * pr + 150 * pg + 29 * pb + 128) >> 8;
            let lc = (77 * cr + 150 * cg + 29 * cb + 128) >> 8;
            let dl = (lp - lc).unsigned_abs() as u64;
            let dr = (pr - cr).unsigned_abs() as u64;
            let dg = (pg - cg).unsigned_abs() as u64;
            let db = (pb - cb).unsigned_abs() as u64;
            let da = (pa - ca).unsigned_abs() as u64;
            // Luma carries the bulk of the weight; chroma contributes
            // a quarter (>> 2) each; alpha gets full weight so changes
            // in transparency are flagged immediately.
            acc += dl + ((dr + dg + db) >> 2) + da;
        }
    }
    acc
}

/// Single-scale **SSIM-lite** cost over a block in two RGBA canvas
/// buffers. Uses the standard SSIM formula (Wang & Bovik 2004) at a
/// single scale, computed on the BT.601 luma channel only — skips the
/// multi-scale Gaussian-pyramid (the "MS-" prefix) for ≈ 4–10× lower
/// per-block cost. Returns `round((1.0 - SSIM) * 10000)` so the result
/// fits the same `u64` threshold-compare contract as [`block_cost`].
///
/// ```text
///   µ_a = mean(luma_a),  µ_b = mean(luma_b)
///   σ_a² = var(luma_a),  σ_b² = var(luma_b),  σ_ab = covar(a, b)
///   SSIM = (2*µ_a*µ_b + C1)(2*σ_ab + C2)
///        / ((µ_a² + µ_b² + C1)(σ_a² + σ_b² + C2))
///   C1   = (0.01 * 255)² = 6.5025
///   C2   = (0.03 * 255)² = 58.5225
///   cost = round((1.0 - SSIM) * 10000)
/// ```
///
/// Cost range is roughly `[0, ~20000]` (SSIM is in `[-1, 1]` so
/// `1 - SSIM` is in `[0, 2]`; the 10000 scale converts to integers
/// without overflowing `u64`). A cost of 50 corresponds to a SSIM
/// gap of ≈ 0.005 — small enough to flag a just-perceptible
/// structural change, large enough to absorb 8-bit rounding noise on
/// a flat block. Empty / zero-pixel blocks return cost = 0.
///
/// Single-channel (luma-only) is sufficient for the cost-model use
/// case: chroma-only changes that don't shift luma are rare in
/// natural-image animation content, and the cost-model already runs
/// on 8×8 blocks where the structural-change contribution dominates
/// any pure chroma drift.
fn block_cost_ssim(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> u64 {
    let n_pixels = (x1 - x0) * (y1 - y0);
    if n_pixels == 0 {
        return 0;
    }
    // First pass: accumulate sums of luma values for both blocks.
    let mut sum_a: i64 = 0;
    let mut sum_b: i64 = 0;
    for y in y0..y1 {
        let row_off = y * canvas_w * 4;
        for x in x0..x1 {
            let off = row_off + x * 4;
            let pa_r = prev[off] as i32;
            let pa_g = prev[off + 1] as i32;
            let pa_b = prev[off + 2] as i32;
            let pb_r = curr[off] as i32;
            let pb_g = curr[off + 1] as i32;
            let pb_b = curr[off + 2] as i32;
            // BT.601 luma (integer-scaled): same as block_cost.
            let la = (77 * pa_r + 150 * pa_g + 29 * pa_b + 128) >> 8;
            let lb = (77 * pb_r + 150 * pb_g + 29 * pb_b + 128) >> 8;
            sum_a += la as i64;
            sum_b += lb as i64;
        }
    }
    let n = n_pixels as f64;
    let mean_a = sum_a as f64 / n;
    let mean_b = sum_b as f64 / n;
    // Second pass: accumulate squared deviations and covariance.
    let mut var_a_acc: f64 = 0.0;
    let mut var_b_acc: f64 = 0.0;
    let mut cov_acc: f64 = 0.0;
    for y in y0..y1 {
        let row_off = y * canvas_w * 4;
        for x in x0..x1 {
            let off = row_off + x * 4;
            let pa_r = prev[off] as i32;
            let pa_g = prev[off + 1] as i32;
            let pa_b = prev[off + 2] as i32;
            let pb_r = curr[off] as i32;
            let pb_g = curr[off + 1] as i32;
            let pb_b = curr[off + 2] as i32;
            let la = ((77 * pa_r + 150 * pa_g + 29 * pa_b + 128) >> 8) as f64;
            let lb = ((77 * pb_r + 150 * pb_g + 29 * pb_b + 128) >> 8) as f64;
            let da = la - mean_a;
            let db = lb - mean_b;
            var_a_acc += da * da;
            var_b_acc += db * db;
            cov_acc += da * db;
        }
    }
    // Use sample-population variance (divide by N, not N-1) — matches
    // the standard SSIM-lite formulation used in image-quality metrics
    // libraries (incl. Wang/Bovik's reference impl) and keeps the
    // n=1 edge case (single-pixel block) numerically clean.
    let var_a = var_a_acc / n;
    let var_b = var_b_acc / n;
    let cov_ab = cov_acc / n;
    // SSIM constants for 8-bit data (L = 255).
    const C1: f64 = 6.5025; // (0.01 * 255)²
    const C2: f64 = 58.5225; // (0.03 * 255)²
    let numer = (2.0 * mean_a * mean_b + C1) * (2.0 * cov_ab + C2);
    let denom = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);
    // `denom` is strictly > 0 (both factors carry +C1 / +C2), so the
    // division is well-defined for every input.
    let ssim = numer / denom;
    let cost_f = (1.0 - ssim) * 10_000.0;
    // Clamp to non-negative + cap at u64 range. SSIM can dip slightly
    // negative on adversarial content (means anti-correlate); we
    // floor cost at 0 to match the SAD's [0, ∞) range.
    if cost_f <= 0.0 {
        0
    } else {
        cost_f.round() as u64
    }
}

/// 3-scale **MS-SSIM-lite** cost over a block in two RGBA canvas
/// buffers. Per Wang/Bovik 2003 "Multi-scale structural similarity
/// for image quality assessment" — cascade SSIM at three spatial
/// scales (the canonical 5-scale series collapsed to 3 by fusing
/// the bottom-two empirical exponents into the largest scale's γ),
/// fuse via the standard `prod(...)^weight` form. Computed on the
/// BT.601 luma channel only (single-channel matches the single-scale
/// path's design choice).
///
/// ```text
///   scale 0: native block — full SSIM (luminance × contrast × structure)
///   scale 1: 2×-extended region around the block, downsampled 2× via separable Gaussian
///            → contrast × structure
///   scale 2: 4×-extended region, downsampled 4× → contrast × structure
///   MS-SSIM = SSIM_0^α * CS_1^β * CS_2^γ
///   α = 0.2856, β = 0.3001, γ = 0.4143  (sum = 1.0)
///   cost = round((1.0 - MS-SSIM) * 10000)
/// ```
///
/// Empirical-exponent derivation: the Wang/Bovik 5-scale series is
/// `{0.0448, 0.2856, 0.3001, 0.2363, 0.1333}` (sum 1.0). Our 3-scale
/// subset takes scales 1, 2, and {3+4+5} fused: `α = 0.2856,
/// β = 0.3001, γ = 0.0448 + 0.2363 + 0.1333 = 0.4144` (rounded to
/// 0.4143 to keep the trio at sum = 1.0). The largest-scale γ is
/// dominant — multi-scale weighting deliberately biases toward the
/// coarse spatial extents because that's where single-scale SSIM
/// blind spots live (low-frequency global drift).
///
/// Returns `round((1.0 - MS-SSIM) * 10000)` (same scale as the
/// single-scale path's `(1 - SSIM) * 10000`); blocks whose cost
/// strictly exceeds [`DeltaConfig::msssim_threshold`] are flagged
/// changed.
///
/// Edge clipping: when the block is near the canvas edge and the
/// 2×/4× extended region would extend past the canvas, the extended
/// region is clipped to canvas bounds rather than mirrored / padded.
/// Pixel counts inside the clipped region are tracked so the
/// downsampled patch still has the right size for the per-scale
/// SSIM/CS computation.
fn block_cost_msssim(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    canvas_h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    kernel: DownsampleKernel,
) -> u64 {
    let n_pixels = (x1 - x0) * (y1 - y0);
    if n_pixels == 0 {
        return 0;
    }
    // Scale 0: full SSIM on the native block.
    let ssim0 = ssim_components_native(prev, curr, canvas_w, x0, y0, x1, y1);

    // Scale 1: 2×-extended region around the block, 2× downsampled
    // (kernel-dependent — Box keeps the historical separable box-
    // average; Gaussian uses a separable 5-tap σ=0.8 kernel).
    let bw = x1 - x0;
    let bh = y1 - y0;
    let ext_x0 = x0.saturating_sub(bw / 2);
    let ext_y0 = y0.saturating_sub(bh / 2);
    let ext_x1 = (x1 + bw / 2).min(canvas_w);
    let ext_y1 = (y1 + bh / 2).min(canvas_h);
    let cs1 = downsampled_cs(
        prev, curr, canvas_w, ext_x0, ext_y0, ext_x1, ext_y1, 2, kernel,
    );

    // Scale 2: 4×-extended region, 4× downsampled. The Gaussian path
    // cascades two blur+decimate-by-2 passes so the pyramid is a true
    // Wang/Bovik MS-SSIM pyramid (not a one-shot 4× decimation).
    let ext2_x0 = x0.saturating_sub(3 * bw / 2);
    let ext2_y0 = y0.saturating_sub(3 * bh / 2);
    let ext2_x1 = (x1 + 3 * bw / 2).min(canvas_w);
    let ext2_y1 = (y1 + 3 * bh / 2).min(canvas_h);
    let cs2 = downsampled_cs(
        prev, curr, canvas_w, ext2_x0, ext2_y0, ext2_x1, ext2_y1, 4, kernel,
    );

    // Empirical exponents from Wang/Bovik 2003, fused 5→3 scales.
    const ALPHA: f64 = 0.2856; // scale 0 weight
    const BETA: f64 = 0.3001; // scale 1 weight
    const GAMMA: f64 = 0.4143; // scale 2 weight (absorbs scales 3+4 of the 5-scale series)

    // Clamp components to (0, 1] for the powf — SSIM/CS in (-∞, 1]
    // theoretically, but our integer-noise content keeps them in
    // [~0, 1]. Negative or zero values would make the powf return
    // NaN / 0, collapsing the product; clamp to a tiny epsilon so
    // the cost stays in a usable range on adversarial content.
    let s0 = ssim0.clamp(1e-6, 1.0);
    let s1 = cs1.clamp(1e-6, 1.0);
    let s2 = cs2.clamp(1e-6, 1.0);
    let msssim = s0.powf(ALPHA) * s1.powf(BETA) * s2.powf(GAMMA);
    let cost_f = (1.0 - msssim) * 10_000.0;
    if cost_f <= 0.0 {
        0
    } else {
        cost_f.round() as u64
    }
}

/// Compute the **full SSIM** value (luminance × contrast × structure)
/// over a native-resolution block in two RGBA canvas buffers, on the
/// BT.601 luma channel. Returns SSIM in `(-∞, 1]` — caller is
/// responsible for clamping if it needs a positive value.
///
/// Shares the per-block math with [`block_cost_ssim`] but returns the
/// raw SSIM scalar (rather than the `(1-SSIM)*10000` cost) so the
/// MS-SSIM cost can fold it into the multi-scale product.
fn ssim_components_native(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> f64 {
    let n_pixels = (x1 - x0) * (y1 - y0);
    if n_pixels == 0 {
        return 1.0;
    }
    let mut sum_a: i64 = 0;
    let mut sum_b: i64 = 0;
    for y in y0..y1 {
        let row_off = y * canvas_w * 4;
        for x in x0..x1 {
            let off = row_off + x * 4;
            let pa_r = prev[off] as i32;
            let pa_g = prev[off + 1] as i32;
            let pa_b = prev[off + 2] as i32;
            let pb_r = curr[off] as i32;
            let pb_g = curr[off + 1] as i32;
            let pb_b = curr[off + 2] as i32;
            let la = (77 * pa_r + 150 * pa_g + 29 * pa_b + 128) >> 8;
            let lb = (77 * pb_r + 150 * pb_g + 29 * pb_b + 128) >> 8;
            sum_a += la as i64;
            sum_b += lb as i64;
        }
    }
    let n = n_pixels as f64;
    let mean_a = sum_a as f64 / n;
    let mean_b = sum_b as f64 / n;
    let mut var_a_acc: f64 = 0.0;
    let mut var_b_acc: f64 = 0.0;
    let mut cov_acc: f64 = 0.0;
    for y in y0..y1 {
        let row_off = y * canvas_w * 4;
        for x in x0..x1 {
            let off = row_off + x * 4;
            let pa_r = prev[off] as i32;
            let pa_g = prev[off + 1] as i32;
            let pa_b = prev[off + 2] as i32;
            let pb_r = curr[off] as i32;
            let pb_g = curr[off + 1] as i32;
            let pb_b = curr[off + 2] as i32;
            let la = ((77 * pa_r + 150 * pa_g + 29 * pa_b + 128) >> 8) as f64;
            let lb = ((77 * pb_r + 150 * pb_g + 29 * pb_b + 128) >> 8) as f64;
            let da = la - mean_a;
            let db = lb - mean_b;
            var_a_acc += da * da;
            var_b_acc += db * db;
            cov_acc += da * db;
        }
    }
    let var_a = var_a_acc / n;
    let var_b = var_b_acc / n;
    let cov_ab = cov_acc / n;
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    let numer = (2.0 * mean_a * mean_b + C1) * (2.0 * cov_ab + C2);
    let denom = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);
    numer / denom
}

/// Downsample a region of the BT.601 luma channel by `factor` (2× or
/// 4×) using the kernel selected by `kernel`, then compute the
/// **contrast × structure** component of SSIM (drop the luminance term
/// so the MS-SSIM product doesn't double-count brightness drift across
/// scales — only the native-scale `SSIM_0` factor carries the
/// luminance term).
///
/// [`DownsampleKernel::Box`] uses the historical separable box-average
/// over each `factor × factor` source cell. [`DownsampleKernel::Gaussian`]
/// uses the canonical Wang/Bovik 2003 5-tap σ=0.8 kernel
/// (`[0.054, 0.244, 0.404, 0.244, 0.054]`) applied separably (horizontal
/// then vertical) followed by 2× decimation; the 4× scale cascades two
/// blur+decimate passes for a true Gaussian pyramid.
///
/// Returns `(2 σ_ab + C2) / (σ_a² + σ_b² + C2)` clamped to `(-∞, 1]`,
/// which is the standard "CS" component used in MS-SSIM. Returns 1.0
/// for empty / single-pixel-after-downsample regions (no contrast
/// signal available — treat as "matches" to keep the multi-scale
/// product well-behaved).
fn downsampled_cs(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    factor: usize,
    kernel: DownsampleKernel,
) -> f64 {
    if x1 <= x0 || y1 <= y0 || factor == 0 {
        return 1.0;
    }
    let (prev_ds, curr_ds) = match kernel {
        DownsampleKernel::Box => box_downsample_luma(prev, curr, canvas_w, x0, y0, x1, y1, factor),
        DownsampleKernel::Gaussian => {
            gaussian_downsample_luma(prev, curr, canvas_w, x0, y0, x1, y1, factor)
        }
    };
    let n = prev_ds.len();
    if n == 0 {
        return 1.0;
    }
    // Compute means + variances + covariance on the downsampled
    // patch (single block — no further blocking).
    let nf = n as f64;
    let mean_a: f64 = prev_ds.iter().sum::<f64>() / nf;
    let mean_b: f64 = curr_ds.iter().sum::<f64>() / nf;
    let mut var_a_acc = 0.0;
    let mut var_b_acc = 0.0;
    let mut cov_acc = 0.0;
    for i in 0..n {
        let da = prev_ds[i] - mean_a;
        let db = curr_ds[i] - mean_b;
        var_a_acc += da * da;
        var_b_acc += db * db;
        cov_acc += da * db;
    }
    let var_a = var_a_acc / nf;
    let var_b = var_b_acc / nf;
    let cov_ab = cov_acc / nf;
    const C2: f64 = 58.5225; // (0.03 * 255)²
                             // CS component (no luminance term) — only the native-scale
                             // `SSIM_0` carries the brightness-drift comparison.
    let numer = 2.0 * cov_ab + C2;
    let denom = var_a + var_b + C2;
    numer / denom
}

/// Box-average downsample helper — extracts BT.601 luma from each
/// `factor × factor` source cell (clipping trailing partial cells along
/// the right / bottom edge to the actual pixel count) and emits the
/// downsampled patch as a pair of `out_w × out_h` `f64` buffers, one for
/// `prev` and one for `curr`.
fn box_downsample_luma(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    factor: usize,
) -> (Vec<f64>, Vec<f64>) {
    let region_w = x1 - x0;
    let region_h = y1 - y0;
    let out_w = region_w.div_ceil(factor);
    let out_h = region_h.div_ceil(factor);
    if out_w == 0 || out_h == 0 {
        return (Vec::new(), Vec::new());
    }
    let n = out_w * out_h;
    let mut prev_ds = vec![0.0f64; n];
    let mut curr_ds = vec![0.0f64; n];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let sy0 = y0 + oy * factor;
            let sy1 = (sy0 + factor).min(y1);
            let sx0 = x0 + ox * factor;
            let sx1 = (sx0 + factor).min(x1);
            let cell_pixels = (sy1 - sy0) * (sx1 - sx0);
            if cell_pixels == 0 {
                continue;
            }
            let mut sum_p: u64 = 0;
            let mut sum_c: u64 = 0;
            for y in sy0..sy1 {
                let row_off = y * canvas_w * 4;
                for x in sx0..sx1 {
                    let off = row_off + x * 4;
                    let pa_r = prev[off] as i32;
                    let pa_g = prev[off + 1] as i32;
                    let pa_b = prev[off + 2] as i32;
                    let pb_r = curr[off] as i32;
                    let pb_g = curr[off + 1] as i32;
                    let pb_b = curr[off + 2] as i32;
                    let la = (77 * pa_r + 150 * pa_g + 29 * pa_b + 128) >> 8;
                    let lb = (77 * pb_r + 150 * pb_g + 29 * pb_b + 128) >> 8;
                    sum_p += la as u64;
                    sum_c += lb as u64;
                }
            }
            let denom_pix = cell_pixels as f64;
            prev_ds[oy * out_w + ox] = sum_p as f64 / denom_pix;
            curr_ds[oy * out_w + ox] = sum_c as f64 / denom_pix;
        }
    }
    (prev_ds, curr_ds)
}

/// Gaussian-pyramid downsample helper — extracts the BT.601 luma plane
/// for the source region into native-resolution `f64` buffers, then
/// applies log2(`factor`) cascaded passes of separable 5-tap σ=0.8
/// Gaussian blur + 2× decimation. The kernel weights
/// `[0.054, 0.244, 0.404, 0.244, 0.054]` are the canonical Wang/Bovik
/// 2003 reference; edge taps are renormalised to sum to 1.0 over the
/// in-bounds samples (clamp-to-edge / replicate-boundary semantics).
///
/// Only `factor` ∈ {1, 2, 4} is exercised by the MS-SSIM cost path. A
/// non-power-of-two `factor` falls back to box semantics.
fn gaussian_downsample_luma(
    prev: &[u8],
    curr: &[u8],
    canvas_w: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    factor: usize,
) -> (Vec<f64>, Vec<f64>) {
    let region_w = x1 - x0;
    let region_h = y1 - y0;
    if region_w == 0 || region_h == 0 || factor == 0 {
        return (Vec::new(), Vec::new());
    }
    // Gaussian pyramid only supports 2^k decimation. Anything else
    // falls back to box semantics so the cost path stays
    // well-behaved if a future caller asks for an odd factor.
    if !factor.is_power_of_two() {
        return box_downsample_luma(prev, curr, canvas_w, x0, y0, x1, y1, factor);
    }
    // Extract native-resolution luma planes for prev and curr.
    let mut prev_l = vec![0.0f64; region_w * region_h];
    let mut curr_l = vec![0.0f64; region_w * region_h];
    for y in 0..region_h {
        let src_row = (y0 + y) * canvas_w * 4;
        let dst_row = y * region_w;
        for x in 0..region_w {
            let off = src_row + (x0 + x) * 4;
            let pa_r = prev[off] as i32;
            let pa_g = prev[off + 1] as i32;
            let pa_b = prev[off + 2] as i32;
            let pb_r = curr[off] as i32;
            let pb_g = curr[off + 1] as i32;
            let pb_b = curr[off + 2] as i32;
            let la = (77 * pa_r + 150 * pa_g + 29 * pa_b + 128) >> 8;
            let lb = (77 * pb_r + 150 * pb_g + 29 * pb_b + 128) >> 8;
            prev_l[dst_row + x] = la as f64;
            curr_l[dst_row + x] = lb as f64;
        }
    }
    // Cascade log2(factor) blur+decimate-by-2 passes.
    let mut w = region_w;
    let mut h = region_h;
    let mut p = prev_l;
    let mut c = curr_l;
    let mut step = 2usize;
    while step <= factor {
        let (np, nw, nh) = gaussian_blur_decimate2(&p, w, h);
        let (nc, _, _) = gaussian_blur_decimate2(&c, w, h);
        p = np;
        c = nc;
        w = nw;
        h = nh;
        if nw == 0 || nh == 0 {
            return (Vec::new(), Vec::new());
        }
        step *= 2;
    }
    (p, c)
}

/// One pass of separable 5-tap σ=0.8 Gaussian blur followed by 2×
/// decimation. Edge taps are renormalised over the in-bounds weights
/// (clamp-to-edge), so a 1-pixel-wide input stays meaningful (the kernel
/// collapses to a single tap at weight 1.0). Returns `(out, out_w, out_h)`
/// where `out_w = ceil(w / 2)`, `out_h = ceil(h / 2)`.
fn gaussian_blur_decimate2(input: &[f64], w: usize, h: usize) -> (Vec<f64>, usize, usize) {
    // Wang/Bovik 2003 5-tap σ ≈ 0.8 kernel.
    const K: [f64; 5] = [0.054, 0.244, 0.404, 0.244, 0.054];
    if w == 0 || h == 0 {
        return (Vec::new(), 0, 0);
    }
    // Horizontal blur into a `w × h` scratch buffer.
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for k in 0..5 {
                let dx = k as isize - 2;
                let sx = x as isize + dx;
                if sx >= 0 && (sx as usize) < w {
                    acc += input[row + sx as usize] * K[k];
                    wsum += K[k];
                }
            }
            tmp[row + x] = if wsum > 0.0 {
                acc / wsum
            } else {
                input[row + x]
            };
        }
    }
    // Vertical blur into a `w × h` blurred buffer (will be decimated
    // by 2 in the next step).
    let mut blurred = vec![0.0f64; w * h];
    for x in 0..w {
        for y in 0..h {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for k in 0..5 {
                let dy = k as isize - 2;
                let sy = y as isize + dy;
                if sy >= 0 && (sy as usize) < h {
                    acc += tmp[(sy as usize) * w + x] * K[k];
                    wsum += K[k];
                }
            }
            blurred[y * w + x] = if wsum > 0.0 {
                acc / wsum
            } else {
                tmp[y * w + x]
            };
        }
    }
    // Decimate by 2 (take samples at even indices).
    let out_w = w.div_ceil(2);
    let out_h = h.div_ceil(2);
    let mut out = vec![0.0f64; out_w * out_h];
    for oy in 0..out_h {
        let sy = oy * 2;
        for ox in 0..out_w {
            let sx = ox * 2;
            out[oy * out_w + ox] = blurred[sy * w + sx];
        }
    }
    (out, out_w, out_h)
}

/// VP8X payload: 1 byte flags, 3 bytes reserved, 3 bytes canvas_w-1,
/// 3 bytes canvas_h-1.
fn vp8x_payload(flags: u8, canvas_w: u32, canvas_h: u32) -> [u8; 10] {
    let mut out = [0u8; 10];
    out[0] = flags;
    let w_minus_1 = canvas_w.saturating_sub(1) & 0x00FF_FFFF;
    let h_minus_1 = canvas_h.saturating_sub(1) & 0x00FF_FFFF;
    out[4] = (w_minus_1 & 0xff) as u8;
    out[5] = ((w_minus_1 >> 8) & 0xff) as u8;
    out[6] = ((w_minus_1 >> 16) & 0xff) as u8;
    out[7] = (h_minus_1 & 0xff) as u8;
    out[8] = ((h_minus_1 >> 8) & 0xff) as u8;
    out[9] = ((h_minus_1 >> 16) & 0xff) as u8;
    out
}

fn write_u24_le(out: &mut Vec<u8>, v: u32) {
    out.push((v & 0xff) as u8);
    out.push(((v >> 8) & 0xff) as u8);
    out.push(((v >> 16) & 0xff) as u8);
}

fn write_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w as usize) * (h as usize) * 4);
        for _ in 0..(w * h) {
            v.extend_from_slice(&rgba);
        }
        v
    }

    #[test]
    fn build_animated_emits_vp8x_anim_anmf_in_order() {
        let f0 = solid_frame(8, 8, [0xff, 0, 0, 0xff]);
        let f1 = solid_frame(8, 8, [0, 0xff, 0, 0xff]);
        let frames = [
            AnimFrame {
                width: 8,
                height: 8,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 100,
                blend: false,
                dispose_to_background: false,
                rgba: &f0,
            },
            AnimFrame {
                width: 8,
                height: 8,
                x_offset: 0,
                y_offset: 0,
                duration_ms: 200,
                blend: false,
                dispose_to_background: false,
                rgba: &f1,
            },
        ];
        let out = build_animated_webp(8, 8, [0; 4], 0, &frames).expect("build");
        // RIFF / WEBP magic at the front.
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WEBP");
        // VP8X first, with ANIM bit (0x02) set.
        assert_eq!(&out[12..16], b"VP8X");
        assert_ne!(out[20] & 0x02, 0, "ANIM flag must be set in VP8X");
        // ANIM next.
        let vp8x_chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]) as usize;
        let anim_off = 12 + 8 + vp8x_chunk_len + (vp8x_chunk_len & 1);
        assert_eq!(&out[anim_off..anim_off + 4], b"ANIM");
        // First ANMF after ANIM.
        let anim_chunk_len = u32::from_le_bytes([
            out[anim_off + 4],
            out[anim_off + 5],
            out[anim_off + 6],
            out[anim_off + 7],
        ]) as usize;
        let anmf0_off = anim_off + 8 + anim_chunk_len + (anim_chunk_len & 1);
        assert_eq!(&out[anmf0_off..anmf0_off + 4], b"ANMF");
    }

    #[test]
    fn rejects_oversized_frame_bbox() {
        let f = solid_frame(8, 8, [0; 4]);
        let frames = [AnimFrame {
            width: 8,
            height: 8,
            x_offset: 4,
            y_offset: 4,
            duration_ms: 0,
            blend: false,
            dispose_to_background: false,
            rgba: &f,
        }];
        // 8x8 frame at (4,4) on a 8x8 canvas — extends past edge.
        let r = build_animated_webp(8, 8, [0; 4], 0, &frames);
        assert!(r.is_err(), "expected oversized-bbox to be rejected");
    }

    #[test]
    fn auto_mode_picks_smaller_of_the_two_candidates() {
        // 96×96 noisy photographic-style frame: the per-pixel value
        // varies enough that VP8L can't collapse it to a tiny literal
        // run, while VP8 lossy at q=75 compresses the smooth-noise
        // structure to a fraction of the size. Auto mode must end up
        // at min(lossless, lossy) — modulo bit-for-bit equality on the
        // mode-specific candidate payload.
        let w = 96u32;
        let h = 96u32;
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        // Pseudo-random but reproducible: an xorshift-ish hash per pixel
        // gives the VP8L Huffman alphabet a wide distribution that
        // doesn't compress to a tiny payload, and gives VP8 a smooth-
        // ish tile pattern (since adjacent pixels share most of their
        // hash bits) that DCT handles very well.
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let mut s = y.wrapping_mul(0x9E37_79B9) ^ x.wrapping_mul(0x85EB_CA77);
                s ^= s.wrapping_shr(13);
                s = s.wrapping_mul(0xC2B2_AE35);
                s ^= s.wrapping_shr(16);
                rgba[i] = ((s >> 0) & 0xff) as u8;
                rgba[i + 1] = ((s >> 8) & 0xff) as u8;
                rgba[i + 2] = ((s >> 16) & 0xff) as u8;
                rgba[i + 3] = 0xff;
            }
        }
        let frames = [AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &rgba,
        }];

        // Force lossless and force lossy, then run auto: auto should
        // not be larger than the smaller of the two forced encodings.
        let lossless = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions {
                mode: AnimFrameMode::Lossless,
                ..Default::default()
            },
        )
        .expect("encode lossless");
        let lossy = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions {
                mode: AnimFrameMode::Lossy,
                ..Default::default()
            },
        )
        .expect("encode lossy");
        let auto = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions::default(),
        )
        .expect("encode auto");

        eprintln!(
            "anim sizes (noise 96x96): lossless={} lossy={} auto={}",
            lossless.len(),
            lossy.len(),
            auto.len()
        );
        // Auto must be ≤ the smaller candidate. (Modulo a few-byte
        // wiggle room for the optional ALPH sub-chunk's even-length
        // padding — if the ALPH/VP8 split happens to round differently
        // than the bare VP8L would, the comparison can be off by 1
        // byte. So compare to `min + 2` for slack.)
        let smaller = lossless.len().min(lossy.len());
        assert!(
            auto.len() <= smaller + 2,
            "auto ({}) > min(lossless={}, lossy={}) + 2 — mode-selection broken",
            auto.len(),
            lossless.len(),
            lossy.len(),
        );
    }

    #[test]
    fn auto_mode_picks_lossless_for_palette_frame() {
        // Build a small flat-colour frame: 32×32 of a single solid
        // colour. VP8L collapses this to ≤ 30 bytes (a single literal +
        // run). VP8 spends a fixed overhead on the keyframe header +
        // partition data + entropy-default tables that's much larger.
        // Auto mode should therefore pick lossless on this fixture.
        let w = 32u32;
        let h = 32u32;
        let rgba = solid_frame(w, h, [0x80, 0x40, 0x20, 0xff]);
        let frames = [AnimFrame {
            width: w,
            height: h,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 50,
            blend: false,
            dispose_to_background: false,
            rgba: &rgba,
        }];
        let auto = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions::default(),
        )
        .expect("encode auto");
        let lossless = build_animated_webp_with_options(
            w,
            h,
            [0; 4],
            0,
            &frames,
            AnimEncoderOptions {
                mode: AnimFrameMode::Lossless,
                ..Default::default()
            },
        )
        .expect("encode lossless");
        // On a solid colour the lossless path is the byte-smaller
        // candidate, so auto must match it (no inflation).
        assert_eq!(
            auto.len(),
            lossless.len(),
            "auto mode failed to pick lossless on a flat-colour fixture"
        );
    }

    #[test]
    fn ssim_cost_block_low_contrast_diff_picks_higher_cost_than_sad() {
        // Build two 8×8 RGBA "blocks" (laid out as a 1-block 8×8 canvas
        // pair). `prev` contains a faint vertical-stripe pattern: 8
        // pixels at luma 130 in one column, 56 pixels at luma 128
        // elsewhere. `curr` is uniform luma 128. Mean-luma drift is
        // tiny (≈ 0.25) so SAD's per-pixel diff sums to ≈ 24 over the
        // block (well under the default SAD threshold of 32 → SAD
        // says "not changed"), but the structural correlation
        // collapses entirely (curr is uniform → covariance with the
        // striped prev is zero → SSIM ≈ 0.993, cost ≈ 73 → SSIM cost
        // > the default SSIM threshold of 50 → SSIM says "changed").
        //
        // Worked example, integers (R=G=B so luma == channel value):
        //   prev: 8 stripe pixels at 130, 56 background pixels at 128
        //         → luma sum = 8*130 + 56*128 = 8208, mean = 128.25
        //         → var      = (8*(130-128.25)² + 56*(128-128.25)²)/64
        //                    ≈ 28/64 = 0.4375
        //   curr: uniform 128
        //         → mean = 128, var = 0, covar(prev, curr) = 0
        //
        //   SAD per stripe pixel  = |130-128| + ((|2|+|2|+|2|) >> 2)
        //                         = 2 + 1 = 3
        //   SAD per bg pixel      = 0
        //   total SAD             = 8*3 + 56*0 = 24    (< 32 → unchanged)
        //
        //   SSIM ≈ (2*128.25*128 + C1)(2*0 + C2)
        //        / ((128.25² + 128² + C1)(0.4375 + 0 + C2))
        //       ≈ 0.99263
        //   cost ≈ round((1 - 0.99263) * 10000) ≈ 73   (> 50 → changed)
        let mut prev = vec![0u8; 8 * 8 * 4];
        let mut curr = vec![0u8; 8 * 8 * 4];
        // curr: uniform luma 128 (R=G=B=128, alpha 255).
        for px in curr.chunks_exact_mut(4) {
            px[0] = 128;
            px[1] = 128;
            px[2] = 128;
            px[3] = 255;
        }
        // prev: same uniform 128 background everywhere first.
        for px in prev.chunks_exact_mut(4) {
            px[0] = 128;
            px[1] = 128;
            px[2] = 128;
            px[3] = 255;
        }
        // prev: replace one column (x = 3) with luma 130 → 8 stripe
        // pixels. (One column = 8 pixels; matches the worked-example
        // counts of 8 stripe + 56 background.)
        for y in 0..8 {
            let off = (y * 8 + 3) * 4;
            prev[off] = 130;
            prev[off + 1] = 130;
            prev[off + 2] = 130;
            prev[off + 3] = 255;
        }
        let sad = block_cost(&prev, &curr, 8, 0, 0, 8, 8);
        let ssim = block_cost_ssim(&prev, &curr, 8, 0, 0, 8, 8);
        // SAD stays under the default SAD threshold (32) — the cost
        // model says "not changed".
        let sad_threshold = DeltaConfig::default().threshold as u64;
        assert!(
            sad <= sad_threshold,
            "SAD should be ≤ default threshold ({sad_threshold}) for low-contrast structural change, got {sad}"
        );
        // SSIM exceeds the default SSIM threshold (50) — the cost
        // model says "changed". This is the key inequality the SSIM
        // path was added to satisfy.
        let ssim_threshold = DeltaConfig::default().ssim_threshold as u64;
        assert!(
            ssim > ssim_threshold,
            "SSIM should be > default ssim_threshold ({ssim_threshold}) for low-contrast structural change, got {ssim}"
        );
        // And SSIM clearly disagrees with SAD on this fixture.
        assert!(
            ssim > sad,
            "SSIM ({ssim}) should rate this block as more 'different' than SAD ({sad})"
        );
    }

    #[test]
    fn ssim_cost_block_identical_inputs_returns_zero() {
        // Identical blocks → SSIM = 1.0 → cost = 0.
        let mut buf = vec![0u8; 8 * 8 * 4];
        // Some non-trivial content so the variance terms are non-zero.
        for (i, px) in buf.chunks_exact_mut(4).enumerate() {
            px[0] = ((i * 13) & 0xff) as u8;
            px[1] = ((i * 19) & 0xff) as u8;
            px[2] = ((i * 23) & 0xff) as u8;
            px[3] = 255;
        }
        let cost = block_cost_ssim(&buf, &buf, 8, 0, 0, 8, 8);
        assert_eq!(
            cost, 0,
            "SSIM cost on identical blocks should be 0, got {cost}"
        );
    }

    #[test]
    fn ssim_cost_threshold_dispatches_correctly_via_config() {
        // Confirm the cost-model dispatcher (compute_changed_grid)
        // honours the cfg.enable_ssim_cost flag — the same input pair
        // should produce different "changed" maps when the SSIM cost
        // is on vs off, on a fixture where the two cost models
        // disagree.
        //
        // Build a 16×8 canvas pair with the left-half (8×8 block 0)
        // identical (cost = 0 either way) and the right-half (8×8
        // block 1) carrying the SSIM-vs-SAD disagreement fixture from
        // the unit test above.
        let canvas_w = 16u32;
        let canvas_h = 8u32;
        let mut prev = vec![0u8; (canvas_w * canvas_h * 4) as usize];
        let mut curr = vec![0u8; (canvas_w * canvas_h * 4) as usize];
        // Both halves: uniform luma 128 background.
        for px in prev.chunks_exact_mut(4) {
            px[0] = 128;
            px[1] = 128;
            px[2] = 128;
            px[3] = 255;
        }
        curr.copy_from_slice(&prev);
        // prev: stripe (luma 130) at x = 11 (column inside the
        // right-half block, block index = 1 at block_size 8).
        for y in 0..8 {
            let off = (y * canvas_w as usize + 11) * 4;
            prev[off] = 130;
            prev[off + 1] = 130;
            prev[off + 2] = 130;
            prev[off + 3] = 255;
        }
        let cfg_sad = DeltaConfig::default();
        let cfg_ssim = DeltaConfig::default().enable_ssim_cost(true);
        let (grid_sad, n_bx, _) = compute_changed_grid(&prev, &curr, canvas_w, canvas_h, &cfg_sad);
        let (grid_ssim, _, _) = compute_changed_grid(&prev, &curr, canvas_w, canvas_h, &cfg_ssim);
        assert_eq!(n_bx, 2, "16/8 = 2 block columns");
        // SAD: right block stays "unchanged" (cost ≤ 32).
        assert!(
            !grid_sad[1],
            "SAD-mode right block should NOT be flagged (low-contrast structural change)"
        );
        // SSIM: right block is flagged "changed" (cost > 50).
        assert!(
            grid_ssim[1],
            "SSIM-mode right block SHOULD be flagged (structural correlation collapsed)"
        );
        // Left block is identical in both inputs → unchanged in both
        // modes (sanity check the dispatcher doesn't mis-flag the
        // identical block).
        assert!(!grid_sad[0]);
        assert!(!grid_ssim[0]);
    }

    #[test]
    fn msssim_cost_block_identical_inputs_returns_zero() {
        // Identical inputs at every scale → MS-SSIM = 1.0 → cost = 0.
        // Build a 32×32 canvas (large enough for the 4×-extended scale-2
        // region to fit) with non-trivial content so the variance terms
        // are non-zero at every scale.
        let cw = 32usize;
        let ch = 32usize;
        let mut buf = vec![0u8; cw * ch * 4];
        for (i, px) in buf.chunks_exact_mut(4).enumerate() {
            px[0] = ((i * 13) & 0xff) as u8;
            px[1] = ((i * 19) & 0xff) as u8;
            px[2] = ((i * 23) & 0xff) as u8;
            px[3] = 255;
        }
        // Score the centre 8×8 block — well inside the canvas so the
        // 4× extension (32×32 region) doesn't get clipped.
        let cost = block_cost_msssim(&buf, &buf, cw, ch, 12, 12, 20, 20, DownsampleKernel::Box);
        assert_eq!(
            cost, 0,
            "MS-SSIM cost on identical blocks should be 0 at every scale, got {cost}"
        );
        // Same block under the Gaussian kernel must also score 0 —
        // identical inputs collapse every scale's CS term to 1.0
        // regardless of the downsample kernel.
        let cost_g = block_cost_msssim(
            &buf,
            &buf,
            cw,
            ch,
            12,
            12,
            20,
            20,
            DownsampleKernel::Gaussian,
        );
        assert_eq!(
            cost_g, 0,
            "MS-SSIM cost on identical blocks should be 0 under the Gaussian kernel, got {cost_g}"
        );
    }

    #[test]
    fn msssim_cost_catches_low_freq_drift_single_scale_misses() {
        // Construct a fixture where SINGLE-scale SSIM at the 8×8 block
        // resolution scores ≈ 0 (no per-block change) but a coarse-scale
        // (32×32) extent picks up a clear DC drift between prev and curr.
        //
        // Recipe: tile a 32×32 canvas with a high-frequency checkerboard
        // pattern (luma alternates 0/255 per pixel). Then in `curr`,
        // shift one half of the canvas (e.g. the left half) brightness
        // up by a small amount that DOES NOT change the per-8×8 mean
        // visibly (the checkerboard already has mean ≈ 127, and we
        // shift by an amount that the modulo-channel-clip absorbs into
        // the structural variance). At the COARSE scale (32×32 box-
        // averaged → single value), the half-canvas DC step IS visible.
        //
        // Actually the cleanest construction: a 32×32 canvas with the
        // 8×8 block at the centre (12..20, 12..20) IDENTICAL in prev
        // vs curr (so single-scale at that block scores 0), but the
        // SURROUNDING context (used by MS-SSIM's scale-1 / scale-2
        // extended regions) carries a clear structural change that
        // collapses the coarse-scale CS terms.
        //
        // Build this: prev = uniform luma 128 everywhere. curr = same
        // 128 in the centre 8×8 block, but the surrounding ring carries
        // a strong vertical-stripe pattern that wasn't there in prev.
        // Single-scale SSIM on the centre 8×8 == 1.0 (identical pixels);
        // MS-SSIM scale-1 (16×16 region) and scale-2 (32×32 region)
        // both see the surrounding stripes → CS terms < 1 → MS-SSIM < 1.
        let cw = 32usize;
        let ch = 32usize;
        let mut prev = vec![0u8; cw * ch * 4];
        let mut curr = vec![0u8; cw * ch * 4];
        // Both: uniform luma 128 base.
        for px in prev.chunks_exact_mut(4) {
            px[0] = 128;
            px[1] = 128;
            px[2] = 128;
            px[3] = 255;
        }
        curr.copy_from_slice(&prev);
        // curr: high-contrast stripes EVERYWHERE EXCEPT the centre 8×8
        // block (12..20, 12..20). This puts a strong structural signal
        // in the scale-1 and scale-2 extended regions while leaving
        // the centre block bit-identical to prev.
        for y in 0..ch {
            for x in 0..cw {
                let centre = (12..20).contains(&y) && (12..20).contains(&x);
                if !centre && (x % 2 == 0) {
                    let off = (y * cw + x) * 4;
                    curr[off] = 200;
                    curr[off + 1] = 200;
                    curr[off + 2] = 200;
                }
            }
        }
        // Single-scale SSIM at the centre block: identical pixels → 0.
        let ssim_single = block_cost_ssim(&prev, &curr, cw, 12, 12, 20, 20);
        // MS-SSIM at the centre block: scale-0 = 0 (centre identical),
        // but scale-1 / scale-2 see the surrounding stripes → CS < 1 →
        // (1 - MS-SSIM) > 0 → cost > 0 → strictly larger than the
        // single-scale 0 cost.
        let msssim = block_cost_msssim(&prev, &curr, cw, ch, 12, 12, 20, 20, DownsampleKernel::Box);
        assert_eq!(
            ssim_single, 0,
            "single-scale SSIM at the identical centre block should be 0"
        );
        assert!(
            msssim > 0,
            "MS-SSIM SHOULD pick up the surrounding-context structural change (cost > 0), got {msssim}"
        );
        // And the MS-SSIM cost should exceed the default ms-ssim
        // threshold (50) — this is the key property the multi-scale
        // path was added to demonstrate.
        let msssim_threshold = DeltaConfig::default().msssim_threshold as u64;
        assert!(
            msssim > msssim_threshold,
            "MS-SSIM cost ({msssim}) should exceed default msssim_threshold ({msssim_threshold})"
        );
    }

    #[test]
    fn msssim_cost_threshold_dispatches_correctly_via_config() {
        // Confirm `compute_changed_grid` honours the `enable_msssim_cost`
        // flag and that MS-SSIM supersedes single-scale SSIM when both
        // are enabled.
        //
        // Use the same surrounding-context-stripes fixture as the
        // per-block test above on a 32×32 canvas — the centre 8×8 block
        // is identical in prev/curr (so single-scale SSIM scores 0,
        // single-scale flag stays false), but the surrounding context
        // carries the stripes (so MS-SSIM flags it).
        let canvas_w = 32u32;
        let canvas_h = 32u32;
        let cw = canvas_w as usize;
        let ch = canvas_h as usize;
        let mut prev = vec![0u8; cw * ch * 4];
        let mut curr = vec![0u8; cw * ch * 4];
        for px in prev.chunks_exact_mut(4) {
            px[0] = 128;
            px[1] = 128;
            px[2] = 128;
            px[3] = 255;
        }
        curr.copy_from_slice(&prev);
        for y in 0..ch {
            for x in 0..cw {
                let centre = (12..20).contains(&y) && (12..20).contains(&x);
                if !centre && (x % 2 == 0) {
                    let off = (y * cw + x) * 4;
                    curr[off] = 200;
                    curr[off + 1] = 200;
                    curr[off + 2] = 200;
                }
            }
        }
        // Block layout: 32 / 8 = 4 columns × 4 rows = 16 blocks.
        // Centre is block (1, 1) at (8..16, 8..16) — the centre 8×8 in
        // the test fixture is at (12..20, 12..20), which crosses 4
        // blocks: (1,1), (2,1), (1,2), (2,2). So we pick a different
        // probe — block (1,1) IS partially identical (top-left quadrant
        // of the centre identical region), partially noise. Move the
        // identical region to align with block (1,1) for the test:
        // identical at exactly (8..16, 8..16) instead of (12..20, 12..20).
        let mut prev = vec![0u8; cw * ch * 4];
        let mut curr = vec![0u8; cw * ch * 4];
        for px in prev.chunks_exact_mut(4) {
            px[0] = 128;
            px[1] = 128;
            px[2] = 128;
            px[3] = 255;
        }
        curr.copy_from_slice(&prev);
        for y in 0..ch {
            for x in 0..cw {
                let centre = (8..16).contains(&y) && (8..16).contains(&x);
                if !centre && (x % 2 == 0) {
                    let off = (y * cw + x) * 4;
                    curr[off] = 200;
                    curr[off + 1] = 200;
                    curr[off + 2] = 200;
                }
            }
        }
        let cfg_ssim = DeltaConfig::default().enable_ssim_cost(true);
        let cfg_msssim = DeltaConfig::default().enable_msssim_cost(true);
        let (grid_ssim, n_bx, _) =
            compute_changed_grid(&prev, &curr, canvas_w, canvas_h, &cfg_ssim);
        let (grid_msssim, _, _) =
            compute_changed_grid(&prev, &curr, canvas_w, canvas_h, &cfg_msssim);
        assert_eq!(n_bx, 4, "32/8 = 4 block columns");
        // Centre block (1,1): identical pixels in both → single-scale
        // SSIM cost = 0 → flag = false.
        let centre_idx = 1 * (n_bx as usize) + 1;
        assert!(
            !grid_ssim[centre_idx],
            "single-scale SSIM should NOT flag identical centre block"
        );
        // Same centre block under MS-SSIM: surrounding stripes drive
        // scale-1 / scale-2 CS terms below 1 → MS-SSIM cost > threshold
        // → flag = true.
        assert!(
            grid_msssim[centre_idx],
            "MS-SSIM SHOULD flag the centre block due to surrounding context drift"
        );
    }

    #[test]
    fn msssim_gaussian_kernel_disagrees_with_box_on_smooth_gradient() {
        // Real-Gaussian (5-tap σ=0.8 separable) vs box-average kernel:
        // diverges most on content with high-frequency structure where
        // the pillbox box-average over-smooths and the Gaussian kernel
        // weights the centre tap (0.404) more heavily than the edge
        // taps (0.054). The two kernels produce different downsampled
        // luma values at every output pixel, which propagates into a
        // different (1 - CS) at each MS-SSIM scale and hence a
        // different final cost.
        //
        // Fixture: 32×32 canvas with a luma checkerboard (alternates
        // ~64/192 per pixel — high-frequency content the box pillbox
        // over-flattens). `prev` is the clean checkerboard; `curr`
        // shifts the centre 8×8 block's mean by a constant +20 (so
        // SSIM_0 picks up a small luminance change AND the surrounding
        // pixels stay structurally identical, isolating the
        // kernel-dependent CS terms).
        let cw = 32usize;
        let ch = 32usize;
        let mut prev = vec![0u8; cw * ch * 4];
        let mut curr = vec![0u8; cw * ch * 4];
        for y in 0..ch {
            for x in 0..cw {
                let v = if (x + y) & 1 == 0 { 64u8 } else { 192u8 };
                let off = (y * cw + x) * 4;
                prev[off] = v;
                prev[off + 1] = v;
                prev[off + 2] = v;
                prev[off + 3] = 255;
                let centre = (12..20).contains(&y) && (12..20).contains(&x);
                let cv = if centre { v.saturating_add(20) } else { v };
                curr[off] = cv;
                curr[off + 1] = cv;
                curr[off + 2] = cv;
                curr[off + 3] = 255;
            }
        }
        let cost_box =
            block_cost_msssim(&prev, &curr, cw, ch, 12, 12, 20, 20, DownsampleKernel::Box);
        let cost_gauss = block_cost_msssim(
            &prev,
            &curr,
            cw,
            ch,
            12,
            12,
            20,
            20,
            DownsampleKernel::Gaussian,
        );
        // Both kernels should detect that the frames differ (cost > 0).
        assert!(
            cost_box > 0,
            "Box-kernel MS-SSIM cost should detect the centre-block luminance shift, got {cost_box}"
        );
        assert!(
            cost_gauss > 0,
            "Gaussian-kernel MS-SSIM cost should detect the centre-block luminance shift, got {cost_gauss}"
        );
        // The kernels must produce *different* costs — the whole point
        // of the Gaussian opt-in is that it reports a different (more
        // perceptually-aligned) cost than the box pillbox. On a
        // high-frequency checkerboard with a localised perturbation the
        // two paths' downsampled CS terms diverge by enough integer
        // counts to distinguish them.
        assert_ne!(
            cost_box, cost_gauss,
            "Box ({cost_box}) and Gaussian ({cost_gauss}) MS-SSIM costs must differ on a checkerboard with localised perturbation \
             — if they match, the Gaussian path collapsed to box semantics"
        );
    }

    #[test]
    fn adaptive_max_components_pins_documented_density_band() {
        // Below LO_DENSITY (5%): clamps to LO_BUDGET (16).
        assert_eq!(adaptive_max_components(0.0), 16);
        assert_eq!(adaptive_max_components(0.01), 16);
        assert_eq!(adaptive_max_components(0.05), 16);
        // Above HI_DENSITY (30%): clamps to HI_BUDGET (4).
        assert_eq!(adaptive_max_components(0.30), 4);
        assert_eq!(adaptive_max_components(0.50), 4);
        assert_eq!(adaptive_max_components(1.00), 4);
        // Mid-band (linear interpolation): density = 0.175 → t = 0.5 →
        // budget = 16 + 0.5*(4-16) = 10.
        assert_eq!(adaptive_max_components(0.175), 10);
        // Just inside the band on each end.
        let near_lo = adaptive_max_components(0.06);
        let near_hi = adaptive_max_components(0.29);
        assert!(
            near_lo > near_hi,
            "monotonic: lower density yields larger budget, got {near_lo} vs {near_hi}"
        );
        assert!(
            (4..=16).contains(&near_lo) && (4..=16).contains(&near_hi),
            "in-band budgets stay inside [4, 16]"
        );
    }

    #[test]
    fn loop_count_and_background_round_trip_on_disk() {
        let f = solid_frame(4, 4, [0; 4]);
        let frames = [AnimFrame {
            width: 4,
            height: 4,
            x_offset: 0,
            y_offset: 0,
            duration_ms: 1,
            blend: false,
            dispose_to_background: false,
            rgba: &f,
        }];
        let out = build_animated_webp(4, 4, [0x12, 0x34, 0x56, 0x78], 7, &frames).expect("build");
        let vp8x_chunk_len = u32::from_le_bytes([out[16], out[17], out[18], out[19]]) as usize;
        let anim_off = 12 + 8 + vp8x_chunk_len + (vp8x_chunk_len & 1);
        // ANIM payload starts at anim_off + 8.
        let anim_payload = &out[anim_off + 8..anim_off + 8 + 6];
        assert_eq!(&anim_payload[0..4], &[0x12, 0x34, 0x56, 0x78]);
        let lc = u16::from_le_bytes([anim_payload[4], anim_payload[5]]);
        assert_eq!(lc, 7);
    }
}
