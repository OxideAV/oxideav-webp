//! Single-pass `VP8L` encoder: lossless efforts `0..=5`
//! ([`crate::EncodeOptions::method`]).
//!
//! The exhaustive search ([`super::encode_argb_with_predictor_chooser`],
//! effort `6`) encodes dozens of complete candidate streams (every
//! transform stack, every predictor chooser, every §3.6.2.3 colour-cache
//! size) and keeps the smallest. This module chooses the transform stack
//! from histogram cost estimates instead, the way libwebp's default
//! method does (`EncoderAnalyze` / `AnalyzeEntropy` in
//! `src/enc/vp8l_enc.c`), and then encodes the image once. The bitstream
//! pieces are the parent module's: the same transform passes, LZ77
//! matcher, cost-priced re-parse, prefix-code builder and writer.
//!
//! ## Choosing the transform stack
//!
//! The candidates are the RFC 9649 §3.5 transform stacks that pay off on real
//! content: no transform, §3.5.3 subtract-green, the §3.5.1 predictor,
//! subtract-green followed by the predictor and, when the image has at
//! most 256 colours, §3.5.4 colour indexing with and without a predictor
//! over the packed indices.
//!
//! For each candidate the forward transform runs into one reused buffer
//! and a greedy §3.6.2.2 LZ77 parse of the result streams into symbol
//! histograms; no token stream is stored. The estimate is the exact size
//! of that greedy stream: transform header and sub-image bits, the five
//! prefix-code tables, the symbols and the extra bits, priced with the
//! encoder's own prefix-code builder.
//!
//! libwebp's `AnalyzeEntropy` ranks the stacks from pixel histograms
//! alone, using "pixel minus left neighbour" as the spatial proxy. That
//! is cheaper, but it cannot see what LZ77 does with the residuals, and on
//! smooth or repetitive content it ranks the stacks wrongly: on the
//! committed 128 x 128 natural fixture it prefers subtract-green plus
//! predictor, whose stream is 55% larger than the predictor alone.
//!
//! ## Encoding once
//!
//! The winning stack is encoded with the exhaustive path's token planner:
//! the greedy parse, then up to two cost-priced dynamic programming
//! re-parses, keeping whichever the exact cost mirror finds smallest.
//! Here the planner keeps its state in compact per-pixel arrays (a 16-bit
//! length and a 32-bit distance per position) that live for the whole
//! encode, where the exhaustive path's planner stores `Vec<Token>`
//! streams and per-position match tables. For the same pixels and stream
//! width the two planners choose the same tokens; the
//! `planner_matches_the_exhaustive_planner` test pins that. Unlike the
//! exhaustive path's matcher, this one never returns a backward reference
//! farther than the RFC 9649 §3.6.2.2 distance codes reach
//! ([`MAX_BACKWARD_DISTANCE`]).
//!
//! ## Rightmost-column predictor modes
//!
//! This crate's §3.5.1 encoder and decoder take the top-right (TR) pixel of
//! the rightmost column from the leftmost pixel of the row above. RFC
//! 9649 §3.5.1 and libwebp use the leftmost pixel of the current row. The
//! two only agree while the rightmost column never uses a mode that reads
//! TR (modes 3, 5, 9 and 10), so this encoder never assigns those modes
//! to the last column of predictor blocks. Its files then decode the same
//! in this crate and in libwebp.

use super::*;

/// §3.5.1 modes that never read the top-right neighbour; the only modes the
/// last column of predictor blocks may use (see the module docs).
const TR_FREE_MODES: [u8; 10] = [0, 1, 2, 4, 6, 7, 8, 11, 12, 13];

/// Every §3.5.1 predictor mode.
const ALL_MODES: [u8; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

/// `match_len` value for a position the greedy parse never probed.
const NOT_PROBED: u16 = u16::MAX;

/// A transform stack the encoder can choose, in the order it estimates
/// them (on equal estimates the earlier, simpler stack wins).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stack {
    /// No transform.
    Plain,
    /// §3.5.3 subtract-green.
    SubtractGreen,
    /// §3.5.1 predictor.
    Predictor,
    /// §3.5.3 subtract-green, then the §3.5.1 predictor over its output.
    SubtractGreenPredictor,
    /// §3.5.4 colour indexing.
    Palette,
    /// §3.5.4 colour indexing, then the §3.5.1 predictor over the packed
    /// indices.
    PalettePredictor,
}

impl Stack {
    const ALL: [Stack; 6] = [
        Stack::Plain,
        Stack::SubtractGreen,
        Stack::Predictor,
        Stack::SubtractGreenPredictor,
        Stack::Palette,
        Stack::PalettePredictor,
    ];
}

/// The §3.5.4 palette of an image with at most 256 colours, built once.
struct PaletteInfo {
    /// Colours in the order written to the colour table: the greedy
    /// nearest-neighbour chain, as libwebp's default `kMinimizeDelta`
    /// sorting does, so the subtraction-coded table stays small.
    palette: Vec<u32>,
    index_of: std::collections::HashMap<u32, u32>,
    /// §3.5.4 index-bundling width (3, 2, 1 or 0).
    width_bits: u8,
}

impl PaletteInfo {
    fn collect(pixels: &[u32]) -> Option<Self> {
        let (sorted, _) = collect_palette(pixels)?;
        let palette = order_palette(&sorted, pixels, PaletteOrdering::MinDeltaChain);
        let index_of = palette_index_map(&palette);
        let width_bits = crate::vp8l_transform::color_indexing_width_bits(palette.len());
        Some(Self {
            palette,
            index_of,
            width_bits,
        })
    }

    /// Write the §3.5.4 transform header and subtraction-coded colour table.
    fn write_transform(&self, w: &mut BitWriter) {
        w.write_bit(true);
        w.write_bits(crate::vp8l_stream::TransformType::ColorIndexing as u32, 2);
        w.write_bits((self.palette.len() - 1) as u32, 8);
        let mut table = self.palette.clone();
        forward_color_table(&mut table);
        write_entropy_coded_image_literals(w, &table);
    }
}

/// Pick the §3.5.1 mode for one block: the [`pick_block_mode_with_hint_slack`]
/// rule restricted to `modes`. The cheapest mode by the folded-L1 residual
/// proxy wins, and the preferred neighbour mode is taken instead when it
/// is allowed and costs at most `slack` more.
#[allow(clippy::too_many_arguments)]
fn pick_block_mode_from(
    pixels: &[u32],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    size: usize,
    modes: &[u8],
    prefer_mode: Option<u8>,
    slack: u64,
) -> u8 {
    let mut best_mode = modes[0];
    let mut best_cost = u64::MAX;
    for &mode in modes {
        let cost =
            block_mode_cost_capped(pixels, width, height, x0, y0, size, size, mode, best_cost);
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }
    if let Some(m) = prefer_mode {
        if m != best_mode && modes.contains(&m) {
            let cost = block_mode_cost(pixels, width, height, x0, y0, size, size, m);
            if cost <= best_cost.saturating_add(slack) {
                best_mode = m;
            }
        }
    }
    best_mode
}

/// Build the §3.5.1 predictor sub-image for this encoder: the round-160
/// slack chooser ([`build_predictor_image_with_slack`]) with a slack of
/// one residual unit per block pixel, and only [`TR_FREE_MODES`] in the
/// last block column.
///
/// The slack lets a block keep its left (or top) neighbour's mode when
/// that costs little more than the best mode. Runs of equal modes make
/// the residual stream more repetitive, which LZ77 then exploits: on the
/// 256 x 256 `(x, y, x ^ y)` gradient of the `lossless_encode` bench the
/// strict chooser's stream is nine times larger.
fn build_predictor_image_single_pass(
    pixels: &[u32],
    width: u32,
    height: u32,
    size_bits: u8,
) -> (Vec<u32>, u32) {
    let block = 1u32 << size_bits;
    let tw = predictor_div_round_up(width, block);
    let th = predictor_div_round_up(height, block);
    let (w, h, bsz) = (width as usize, height as usize, block as usize);
    let slack = (bsz * bsz) as u64;
    let mut img = Vec::with_capacity((tw * th) as usize);
    let mut prev_row: Vec<Option<u8>> = vec![None; tw as usize];
    for by in 0..th as usize {
        let mut left_mode: Option<u8> = None;
        for (bx, top_slot) in prev_row.iter_mut().enumerate() {
            let modes: &[u8] = if bx + 1 == tw as usize {
                &TR_FREE_MODES
            } else {
                &ALL_MODES
            };
            let prefer = left_mode.or(*top_slot);
            let mode =
                pick_block_mode_from(pixels, w, h, bx * bsz, by * bsz, bsz, modes, prefer, slack);
            img.push(0xff00_0000 | ((mode as u32) << 8));
            left_mode = Some(mode);
            *top_slot = Some(mode);
        }
    }
    (img, tw)
}

/// [`apply_forward_predictor`] in place: replace every pixel of `buf`
/// with its §3.5.1 residual. The walk runs from the last pixel to the
/// first, so every neighbour a prediction reads (left, top, top-left,
/// top-right, all earlier in scan order) still holds its original value.
fn apply_forward_predictor_in_place(
    buf: &mut [u32],
    width: u32,
    height: u32,
    predictor_image: &[u32],
    transform_width: u32,
    size_bits: u8,
) {
    let w = width as usize;
    for y in (0..height as usize).rev() {
        for x in (0..w).rev() {
            let mode = if x == 0 || y == 0 {
                0
            } else {
                let block =
                    ((y as u32 >> size_bits) * transform_width + (x as u32 >> size_bits)) as usize;
                ((predictor_image[block] >> 8) & 0xff) as u8
            };
            let pred = predictor_at(buf, w, x, y, mode);
            let idx = y * w + x;
            buf[idx] = predictor_subtract(buf[idx], pred);
        }
    }
}

/// Write the §3.5.1 transform header and sub-image for `predictor_image`.
fn write_predictor_transform(w: &mut BitWriter, size_bits: u8, predictor_image: &[u32]) {
    w.write_bit(true);
    w.write_bits(crate::vp8l_stream::TransformType::Predictor as u32, 2);
    w.write_bits((size_bits - 2) as u32, 3);
    write_entropy_coded_image_literals(w, predictor_image);
}

/// Run `stack`'s forward transforms and write its §3.8.2 transform list,
/// terminator included, to `w`.
///
/// Returns the pixels the spatially-coded image covers and their width:
/// `pixels` itself for [`Stack::Plain`], otherwise `buf`, which receives
/// the transformed pixels. Returns `None` when the stack does not apply:
/// no palette, or an image smaller than one predictor block.
fn apply_stack<'a>(
    stack: Stack,
    pixels: &'a [u32],
    width: u32,
    height: u32,
    palette: Option<&PaletteInfo>,
    buf: &'a mut Vec<u32>,
    w: &mut BitWriter,
) -> Option<(&'a [u32], u32)> {
    let size_bits = DEFAULT_PREDICTOR_SIZE_BITS;
    let block = 1u32 << size_bits;
    let fits_predictor = |stream_width: u32| stream_width >= block && height >= block;
    match stack {
        Stack::Plain => {
            w.write_bit(false);
            Some((pixels, width))
        }
        Stack::SubtractGreen | Stack::SubtractGreenPredictor => {
            let with_predictor = stack == Stack::SubtractGreenPredictor;
            if with_predictor && !fits_predictor(width) {
                return None;
            }
            buf.clear();
            buf.extend_from_slice(pixels);
            apply_subtract_green(buf);
            w.write_bit(true);
            w.write_bits(crate::vp8l_stream::TransformType::SubtractGreen as u32, 2);
            if with_predictor {
                let (image, tw) = build_predictor_image_single_pass(buf, width, height, size_bits);
                write_predictor_transform(w, size_bits, &image);
                apply_forward_predictor_in_place(buf, width, height, &image, tw, size_bits);
            }
            w.write_bit(false);
            Some((&buf[..], width))
        }
        Stack::Predictor => {
            if !fits_predictor(width) {
                return None;
            }
            let (image, tw) = build_predictor_image_single_pass(pixels, width, height, size_bits);
            write_predictor_transform(w, size_bits, &image);
            w.write_bit(false);
            buf.clear();
            buf.resize(pixels.len(), 0);
            apply_forward_predictor(pixels, buf, width, height, &image, tw, size_bits);
            Some((&buf[..], width))
        }
        Stack::Palette | Stack::PalettePredictor => {
            let palette = palette?;
            let (packed, packed_width) = pack_indices_into_bundled_image(
                pixels,
                &palette.index_of,
                width,
                height,
                palette.width_bits,
            );
            let with_predictor = stack == Stack::PalettePredictor;
            if with_predictor && !fits_predictor(packed_width) {
                return None;
            }
            palette.write_transform(w);
            buf.clear();
            if with_predictor {
                let (image, tw) =
                    build_predictor_image_single_pass(&packed, packed_width, height, size_bits);
                write_predictor_transform(w, size_bits, &image);
                buf.resize(packed.len(), 0);
                apply_forward_predictor(&packed, buf, packed_width, height, &image, tw, size_bits);
            } else {
                buf.extend_from_slice(&packed);
            }
            w.write_bit(false);
            Some((&buf[..], packed_width))
        }
    }
}

/// Exact bits of coding `freqs` with its own prefix code: the
/// code-length table plus the symbols, as [`CostLengths`] prices them.
fn histogram_bits(freqs: &[u32]) -> u64 {
    let code = CostLengths::from_freqs(freqs);
    let mut bits = code.code_lengths_bits() as u64;
    for (sym, &f) in freqs.iter().enumerate() {
        bits += u64::from(f) * code.sym_bits(sym) as u64;
    }
    bits
}

/// Symbol histograms of one greedy parse (no colour cache), plus the
/// length and distance extra bits.
struct GreedyHistograms {
    freqs: Frequencies,
    extra_bits: u64,
}

impl GreedyHistograms {
    fn new() -> Self {
        Self {
            freqs: Frequencies::new(0),
            extra_bits: 0,
        }
    }

    fn reset(&mut self) {
        let f = &mut self.freqs;
        for table in [
            &mut f.green,
            &mut f.red,
            &mut f.blue,
            &mut f.alpha,
            &mut f.distance,
        ] {
            table.fill(0);
        }
        self.extra_bits = 0;
    }

    fn add(&mut self, tok: Token, stream_width: u32) {
        if let Token::Copy { length, distance } = tok {
            let (_, len_extra, _) = value_to_prefix(length as u32);
            let code = pixel_distance_to_distance_code(distance, stream_width);
            let (_, dist_extra, _) = value_to_prefix(code);
            self.extra_bits += u64::from(len_extra + dist_extra);
        }
        self.freqs.count_token(tok, stream_width);
    }

    /// Exact `spatially-coded-image` size in bits: the colour-cache-info
    /// bit (no cache), the meta-prefix bit, the five prefix-code tables,
    /// the symbols and the extra bits.
    fn bits(&self) -> u64 {
        let f = &self.freqs;
        let mut total = 1 + 1 + self.extra_bits;
        for table in [&f.green, &f.red, &f.blue, &f.alpha, &f.distance] {
            total += histogram_bits(table);
        }
        total
    }
}

/// A token stream stored one entry per pixel position: at a position the
/// stream reaches, `len == 0` is a literal and `len > 0` a backward
/// reference of that length at distance `dist`. Entries the stream skips
/// over (inside a backward reference) are never read.
#[derive(Debug, Default)]
struct Parse {
    len: Vec<u16>,
    dist: Vec<u32>,
}

impl Parse {
    fn resize(&mut self, n: usize) {
        self.len.resize(n, 0);
        self.dist.resize(n, 0);
    }

    /// The parse as a token stream over `pixels`.
    fn tokens<'a>(&'a self, pixels: &'a [u32]) -> ParseTokens<'a> {
        ParseTokens {
            pixels,
            parse: self,
            pos: 0,
        }
    }
}

/// Iterator behind [`Parse::tokens`].
#[derive(Clone)]
struct ParseTokens<'a> {
    pixels: &'a [u32],
    parse: &'a Parse,
    pos: usize,
}

impl Iterator for ParseTokens<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        let pos = self.pos;
        if pos >= self.pixels.len() {
            return None;
        }
        let len = self.parse.len[pos];
        if len == 0 {
            self.pos += 1;
            Some(Token::Literal(self.pixels[pos]))
        } else {
            self.pos += len as usize;
            Some(Token::Copy {
                length: len as usize,
                distance: self.parse.dist[pos] as usize,
            })
        }
    }
}

/// The token planner's per-pixel state, allocated once per encode.
#[derive(Debug, Default)]
struct Planner {
    lz77: Lz77Buffers,
    /// Longest match per position (`0` = none, [`NOT_PROBED`] before the
    /// match table is complete).
    match_len: Vec<u16>,
    match_dist: Vec<u32>,
    /// `cost[i]`: model bits to code `pixels[i..]` (dynamic programming).
    cost: Vec<u64>,
    /// The greedy parse, then the second re-parse.
    parse_a: Parse,
    /// The first re-parse.
    parse_b: Parse,
}

/// Which [`Planner`] parse holds the chosen token stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chosen {
    A,
    B,
}

impl Planner {
    /// Plan the token stream for `pixels` (a spatially-coded image
    /// `width` wide, no colour cache): the greedy parse plus up to two
    /// cost-priced re-parses, the cheapest by exact size winning. This is
    /// [`best_stream_tokens_with_cost`] without a cache. Returns the
    /// chosen parse and its exact `prefix-codes + lz77-coded-image` size
    /// in bits.
    fn plan(&mut self, pixels: &[u32], width: u32) -> (Chosen, usize) {
        let n = pixels.len();
        self.match_len.clear();
        self.match_len.resize(n, NOT_PROBED);
        self.match_dist.clear();
        self.match_dist.resize(n, 0);
        self.parse_a.resize(n);
        self.parse_b.resize(n);

        // Greedy parse into `parse_a`, recording every probe into the
        // match table (the round-440 fusion of the two matcher passes).
        let mut matcher = Lz77Matcher::with_buffers(
            pixels,
            MAX_BACKWARD_DISTANCE,
            std::mem::take(&mut self.lz77),
        );
        {
            let (match_len, match_dist) = (&mut self.match_len, &mut self.match_dist);
            let parse = &mut self.parse_a;
            let mut pos = 0usize;
            lz77_parse(
                &mut matcher,
                LAZY_DEPTH_DEFAULT,
                |p, found| {
                    let (len, dist) = found.unwrap_or((0, 0));
                    match_len[p] = len as u16;
                    match_dist[p] = dist as u32;
                },
                |tok| match tok {
                    Token::Copy { length, distance } => {
                        parse.len[pos] = length as u16;
                        parse.dist[pos] = distance as u32;
                        pos += length;
                    }
                    _ => {
                        parse.len[pos] = 0;
                        pos += 1;
                    }
                },
            );
        }
        let mut matcher =
            Lz77Matcher::with_buffers(pixels, MAX_BACKWARD_DISTANCE, matcher.into_buffers());
        self.complete_match_table(&mut matcher);
        self.lz77 = matcher.into_buffers();

        let price = |parse: &Parse| -> (StreamCostTables, usize) {
            let tokens = parse.tokens(pixels);
            let tables = StreamCostTables::from_frequencies(&count_token_stream_frequencies(
                tokens.clone(),
                0,
                width,
            ));
            let bits = token_stream_bits_from(&tables, tokens, width);
            (tables, bits)
        };

        let (greedy_tables, greedy_bits) = price(&self.parse_a);
        dp_refine_parse(
            pixels,
            width,
            &self.match_len,
            &self.match_dist,
            &greedy_tables,
            &mut self.cost,
            &mut self.parse_b,
        );
        let (dp1_tables, dp1_bits) = price(&self.parse_b);
        if dp1_bits >= greedy_bits {
            return (Chosen::A, greedy_bits);
        }
        // The greedy parse lost; its buffer takes the second re-parse.
        dp_refine_parse(
            pixels,
            width,
            &self.match_len,
            &self.match_dist,
            &dp1_tables,
            &mut self.cost,
            &mut self.parse_a,
        );
        let (_, dp2_bits) = price(&self.parse_a);
        if dp2_bits < dp1_bits {
            (Chosen::A, dp2_bits)
        } else {
            (Chosen::B, dp1_bits)
        }
    }

    /// Fill the match-table positions the greedy parse never probed, with
    /// the run inheritance of [`compute_dp_matches_with_probes`] (whose
    /// table this reproduces entry for entry).
    fn complete_match_table(&mut self, matcher: &mut Lz77Matcher<'_>) {
        let n = matcher.pixels.len();
        let mut pos = 0usize;
        while pos < n {
            let found = if self.match_len[pos] == NOT_PROBED {
                matcher.find(pos)
            } else {
                let recorded = match self.match_len[pos] {
                    0 => None,
                    len => Some((len as usize, self.match_dist[pos] as usize)),
                };
                debug_assert_eq!(
                    recorded,
                    matcher.find(pos),
                    "recorded probe diverged from fresh search at {pos}"
                );
                recorded
            };
            let (len, dist) = found.unwrap_or((0, 0));
            self.match_len[pos] = len as u16;
            self.match_dist[pos] = dist as u32;
            matcher.insert(pos);
            if len >= DP_LONG_MATCH_INHERIT {
                let inherit = len - DP_LONG_MATCH_INHERIT;
                for k in 1..=inherit {
                    self.match_len[pos + k] = (len - k) as u16;
                    self.match_dist[pos + k] = dist as u32;
                    matcher.insert(pos + k);
                }
                pos += inherit + 1;
                continue;
            }
            pos += 1;
        }
    }

    fn parse(&self, chosen: Chosen) -> &Parse {
        match chosen {
            Chosen::A => &self.parse_a,
            Chosen::B => &self.parse_b,
        }
    }
}

/// [`dp_refine_tokens`] over the planner's compact state: the same
/// backward dynamic programme, candidate order and tie-breaks, writing
/// the chosen parse to `out` instead of a token vector.
///
/// The longest-match decompositions are computed per position from
/// `match_len` / `match_dist` instead of read from a precomputed
/// [`DpMatch`] table, and the fixed-distance runs of
/// [`compute_special_matches`] are carried as one running length per
/// distance, updated as the walk moves back. Both give the values the
/// tables would hold.
fn dp_refine_parse(
    pixels: &[u32],
    width: u32,
    match_len: &[u16],
    match_dist: &[u32],
    cost_model: &StreamCostTables,
    cost: &mut Vec<u64>,
    out: &mut Parse,
) {
    /// Longest run a fixed-distance candidate may cover (the §3.6.2.2
    /// maximum, as in [`compute_special_matches`]).
    const MAX_RUN: u32 = 4096;
    let n = pixels.len();
    let expand = |lengths: &[u8]| -> Vec<u32> {
        lengths
            .iter()
            .map(|&l| {
                if l == 0 {
                    DP_UNSEEN_SYMBOL_COST
                } else {
                    l as u32
                }
            })
            .collect()
    };
    let green_cost = expand(&cost_model.green.lengths);
    let red_cost = expand(&cost_model.red.lengths);
    let blue_cost = expand(&cost_model.blue.lengths);
    let alpha_cost = expand(&cost_model.alpha.lengths);
    let dist_cost = expand(&cost_model.distance.lengths);
    let mut ladder_bits = [0u64; DP_LENGTH_LADDER.len()];
    for (bits, &l) in ladder_bits.iter_mut().zip(DP_LENGTH_LADDER.iter()) {
        let (len_prefix, len_extra, _) = value_to_prefix(l as u32);
        *bits = (green_cost[256 + len_prefix as usize] + len_extra) as u64;
    }
    // (distance, decomposition template, running match length).
    let mut specials: Vec<(usize, DpMatch, u32)> = special_distances(width, n)
        .into_iter()
        .map(|d| (d, DpMatch::new(1, d, width), 0))
        .collect();

    cost.clear();
    cost.resize(n + 1, 0);
    for i in (0..n).rev() {
        let p = pixels[i];
        let a = ((p >> 24) & 0xff) as usize;
        let r = ((p >> 16) & 0xff) as usize;
        let g = ((p >> 8) & 0xff) as usize;
        let b = (p & 0xff) as usize;
        let lit_bits = (green_cost[g] + red_cost[r] + blue_cost[b] + alpha_cost[a]) as u64;
        let mut best = lit_bits + cost[i + 1];
        let mut best_len = 0u32;
        let mut best_dist = 0u32;
        if match_len[i] > 0 {
            let m = DpMatch::new(match_len[i] as usize, match_dist[i] as usize, width);
            let dist_bits = (dist_cost[m.dist_prefix as usize] + m.dist_extra_bits as u32) as u64;
            let max_len = m.len as usize;
            let full_bits =
                (green_cost[256 + m.len_prefix as usize] + m.len_extra_bits as u32) as u64;
            let total = full_bits + dist_bits + cost[i + max_len];
            if total < best {
                best = total;
                best_len = m.len;
                best_dist = m.dist;
            }
            for (k, &l) in DP_LENGTH_LADDER.iter().enumerate() {
                if l >= max_len {
                    break;
                }
                let total = ladder_bits[k] + dist_bits + cost[i + l];
                if total < best {
                    best = total;
                    best_len = l as u32;
                    best_dist = m.dist;
                }
            }
        }
        for (d, template, run) in specials.iter_mut() {
            *run = if i >= *d && pixels[i] == pixels[i - *d] {
                (*run + 1).min(MAX_RUN)
            } else {
                0
            };
            if *run == 0 {
                continue;
            }
            let m = template.with_length(*run as usize);
            let dist_bits = (dist_cost[m.dist_prefix as usize] + m.dist_extra_bits as u32) as u64;
            let full_bits =
                (green_cost[256 + m.len_prefix as usize] + m.len_extra_bits as u32) as u64;
            let total = full_bits + dist_bits + cost[i + m.len as usize];
            if total < best {
                best = total;
                best_len = m.len;
                best_dist = m.dist;
            }
        }
        cost[i] = best;
        out.len[i] = best_len as u16;
        out.dist[i] = best_dist;
    }
}

/// Estimate every applicable [`Stack`] and return the cheapest.
fn choose(
    pixels: &[u32],
    width: u32,
    height: u32,
    palette: Option<&PaletteInfo>,
    buf: &mut Vec<u32>,
    lz77: &mut Lz77Buffers,
) -> Stack {
    let mut histograms = GreedyHistograms::new();
    let mut best: Option<(Stack, u64)> = None;
    for stack in Stack::ALL {
        let mut prelude = BitWriter::new();
        let Some((stream, stream_width)) =
            apply_stack(stack, pixels, width, height, palette, buf, &mut prelude)
        else {
            continue;
        };
        histograms.reset();
        let mut matcher =
            Lz77Matcher::with_buffers(stream, MAX_BACKWARD_DISTANCE, std::mem::take(lz77));
        lz77_parse(
            &mut matcher,
            LAZY_DEPTH_DEFAULT,
            |_, _| {},
            |tok| histograms.add(tok, stream_width),
        );
        *lz77 = matcher.into_buffers();
        let bits = prelude.bit_position() as u64 + histograms.bits();
        if best.map_or(true, |(_, b)| bits < b) {
            best = Some((stack, bits));
        }
    }
    best.expect("the plain stack always applies").0
}

/// Encode `pixels` (`width * height` ARGB values in scan-line order) as a
/// §3.8.1 image stream appended to `w`: the transform list, then the
/// spatially-coded image.
pub(super) fn encode_image_stream(pixels: &[u32], width: u32, height: u32, w: &mut BitWriter) {
    debug_assert_eq!(pixels.len(), width as usize * height as usize);
    let palette = PaletteInfo::collect(pixels);
    let mut buf: Vec<u32> = Vec::with_capacity(pixels.len());
    let mut planner = Planner::default();

    let stack = choose(
        pixels,
        width,
        height,
        palette.as_ref(),
        &mut buf,
        &mut planner.lz77,
    );
    let (stream, stream_width) =
        apply_stack(stack, pixels, width, height, palette.as_ref(), &mut buf, w)
            .expect("the chosen stack applies");
    let (chosen, _) = planner.plan(stream, stream_width);
    let tokens = planner.parse(chosen).tokens(stream);
    write_spatially_coded_token_stream(w, tokens, None, stream_width);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic photo-like ARGB: gradients plus small noise.
    fn photo(width: u32, height: u32, seed: u32) -> Vec<u32> {
        let mut state = seed;
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = state >> 28;
                let r = (x * 3 + noise) & 0xff;
                let g = (y * 2 + x + noise) & 0xff;
                let b = (200 + noise).wrapping_sub(y) & 0xff;
                out.push(0xff00_0000 | (r << 16) | (g << 8) | b);
            }
        }
        out
    }

    /// Flat bands and exact row repeats, so LZ77 carries real weight.
    fn banded(width: u32, height: u32) -> Vec<u32> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let band = (y / 5 + x / 9) % 7;
                    0xff00_0000 | (band * 0x0024_1b30) & 0x00ff_ffff
                })
            })
            .collect()
    }

    #[test]
    fn in_place_forward_predictor_matches_the_two_buffer_pass() {
        let (w, h) = (45u32, 38u32);
        let px = photo(w, h, 7);
        let (image, tw) = build_predictor_image_single_pass(&px, w, h, 4);
        let mut expected = vec![0u32; px.len()];
        apply_forward_predictor(&px, &mut expected, w, h, &image, tw, 4);
        let mut in_place = px.clone();
        apply_forward_predictor_in_place(&mut in_place, w, h, &image, tw, 4);
        assert_eq!(in_place, expected);
    }

    #[test]
    fn last_block_column_never_reads_the_top_right_pixel() {
        for (w, h) in [(48u32, 32u32), (45, 38), (16, 16), (17, 40)] {
            let px = photo(w, h, w * 31 + h);
            let (image, tw) = build_predictor_image_single_pass(&px, w, h, 4);
            for (i, &entry) in image.iter().enumerate() {
                let mode = ((entry >> 8) & 0xff) as u8;
                if i as u32 % tw == tw - 1 {
                    assert!(
                        TR_FREE_MODES.contains(&mode),
                        "{w}x{h}: block {i} in the last column uses mode {mode}"
                    );
                }
            }
        }
    }

    #[test]
    fn greedy_histograms_price_the_greedy_stream_exactly() {
        for (pixels, width) in [(photo(40, 30, 3), 40u32), (banded(64, 48), 64)] {
            let tokens = tokenize_lz77(&pixels);
            let mut histograms = GreedyHistograms::new();
            for &tok in &tokens {
                histograms.add(tok, width);
            }
            let expected = 1 + 1 + prefix_codes_and_tokens_bits(&tokens, 0, width) as u64;
            assert_eq!(histograms.bits(), expected);
        }
    }

    #[test]
    fn planner_matches_the_exhaustive_planner() {
        // The planner must pick exactly the tokens
        // `best_stream_tokens_with_cost` picks, at the same exact cost.
        for (pixels, width) in [
            (photo(48, 40, 11), 48u32),
            (banded(96, 64), 96),
            (vec![0xff10_2030; 32 * 8], 32),
        ] {
            let mut planner = Planner::default();
            let (expected_tokens, expected_bits) =
                best_stream_tokens_with_cost(&pixels, width, None);
            let (chosen, bits) = planner.plan(&pixels, width);
            let tokens: Vec<Token> = planner.parse(chosen).tokens(&pixels).collect();
            assert_eq!(tokens, expected_tokens, "{width} wide");
            assert_eq!(bits, expected_bits, "{width} wide");
        }
    }

    #[test]
    fn capped_matcher_never_reaches_past_the_distance_alphabet() {
        // Unique pixels, except a 16-pixel run repeated farther back than
        // §3.6.2.2 can code.
        let far = MAX_BACKWARD_DISTANCE + 100;
        let mut pixels: Vec<u32> = (0..far as u32 + 64).collect();
        for k in 0..16 {
            pixels[far + k] = pixels[k];
        }
        let find_at = |max_distance: usize| {
            let mut matcher =
                Lz77Matcher::with_buffers(&pixels, max_distance, Lz77Buffers::default());
            for pos in 0..far {
                matcher.insert(pos);
            }
            matcher.find(far)
        };
        // Uncapped (the exhaustive path's matcher) it finds the repeat ...
        assert_eq!(find_at(usize::MAX), Some((16, far)));
        // ... capped it must not.
        assert_eq!(find_at(MAX_BACKWARD_DISTANCE), None);
        let (prefix, _, _) =
            value_to_prefix(pixel_distance_to_distance_code(MAX_BACKWARD_DISTANCE, 2048));
        assert!((prefix as usize) < 40, "the cap itself must be codable");
    }

    #[test]
    fn single_pass_streams_decode_to_the_source() {
        let cases: Vec<(Vec<u32>, u32, u32)> = vec![
            (photo(64, 48, 5), 64, 48),
            (banded(80, 33), 80, 33),
            (photo(7, 5, 9), 7, 5),
            (vec![0x8040_2010; 1], 1, 1),
        ];
        for (pixels, width, height) in cases {
            let mut w = BitWriter::new();
            for b in build_image_header(width, height, true) {
                w.write_bits(u32::from(b), 8);
            }
            encode_image_stream(&pixels, width, height, &mut w);
            let payload = w.into_bytes();
            let decoded = crate::vp8l_transform::decode_lossless(&payload, width, height).unwrap();
            assert_eq!(decoded.pixels(), &pixels[..], "{width}x{height}");
        }
    }
}
