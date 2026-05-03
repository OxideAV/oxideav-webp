//! LSB-first bit reader for VP8L.
//!
//! VP8L packs bits with the low bit of each byte coming out first — the
//! opposite convention from VP8's boolean arithmetic decoder. This reader
//! is a straightforward buffered shift accumulator: whenever fewer than 32
//! bits remain in the buffer we pull one more byte in.

use oxideav_core::Result;

pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Next byte to consume.
    byte_pos: usize,
    /// Accumulator of pending bits (LSB-first).
    bits: u64,
    /// Valid bits currently in `bits`.
    nbits: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            byte_pos: 0,
            bits: 0,
            nbits: 0,
        }
    }

    /// Read `n` bits (0..=32) and return them as a u32, LSB-first. Past
    /// end-of-buffer reads return zero bits — matching libwebp's
    /// well-defined trailing-zero behaviour. Callers that care about
    /// catching truncation should watch [`Self::at_end`].
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        debug_assert!(n <= 32);
        while self.nbits < n as u32 {
            if self.byte_pos >= self.buf.len() {
                // Inject zeros — we deliberately do *not* error here.
                self.nbits += 8;
                continue;
            }
            self.bits |= (self.buf[self.byte_pos] as u64) << self.nbits;
            self.byte_pos += 1;
            self.nbits += 8;
        }
        let mask = if n == 0 { 0u64 } else { (1u64 << n) - 1 };
        let v = (self.bits & mask) as u32;
        self.bits >>= n;
        self.nbits -= n as u32;
        Ok(v)
    }

    /// Read exactly one bit.
    pub fn read_bit(&mut self) -> Result<u32> {
        self.read_bits(1)
    }

    /// True if we've read past the physical end of the underlying buffer.
    pub fn at_end(&self) -> bool {
        self.byte_pos >= self.buf.len()
    }

    /// Current byte position (useful for debugging).
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_lsb_first() {
        // 0b1011_0001 → LSB-first yields 1, 0, 0, 0, 1, 1, 0, 1
        let buf = [0b1011_0001u8];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(1).unwrap(), 1);
        assert_eq!(br.read_bits(1).unwrap(), 0);
        assert_eq!(br.read_bits(1).unwrap(), 0);
        assert_eq!(br.read_bits(1).unwrap(), 0);
        assert_eq!(br.read_bits(4).unwrap(), 0b1011);
    }

    #[test]
    fn crosses_byte_boundaries() {
        let buf = [0xff, 0x01];
        let mut br = BitReader::new(&buf);
        let v = br.read_bits(12).unwrap();
        assert_eq!(v, 0x1ff);
    }

    #[test]
    fn read_16_bits() {
        let buf = [0x34, 0x12];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(16).unwrap(), 0x1234);
    }

    #[test]
    fn read_zero_bits_consumes_nothing() {
        let buf = [0xab, 0xcd];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(0).unwrap(), 0);
        assert_eq!(br.byte_pos(), 0);
        assert_eq!(br.read_bits(8).unwrap(), 0xab);
    }

    #[test]
    fn read_24_bits_three_byte_span() {
        // 0x563412 packed across 3 bytes — exercises the buffered
        // accumulator beyond a single u32 refill cycle.
        let buf = [0x12u8, 0x34, 0x56];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(24).unwrap(), 0x56_3412);
    }

    #[test]
    fn read_32_bits_max() {
        // Boundary case: the spec promises read_bits(32) is supported.
        let buf = [0x78u8, 0x56, 0x34, 0x12];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(32).unwrap(), 0x1234_5678);
    }

    #[test]
    fn read_3_then_5_then_byte() {
        // Three reads totalling 8+8 = 16 bits across two bytes; 3+5
        // exhaust byte 0, the final 8-bit read pulls in byte 1.
        let buf = [0b1010_1101u8, 0b0000_0011];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(3).unwrap(), 0b101); // bits 0..3 of byte 0
        assert_eq!(br.read_bits(5).unwrap(), 0b10101); // bits 3..8 of byte 0
        assert_eq!(br.read_bits(8).unwrap(), 0b0000_0011); // byte 1
    }

    #[test]
    fn read_across_byte_boundary_5_then_5() {
        // 5 bits from byte 0 (low 5 bits), then 5 bits = remaining 3 bits
        // of byte 0 (high) + low 2 bits of byte 1.
        let buf = [0b1010_1010u8, 0b1111_0011];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(5).unwrap(), 0b0_1010);
        // Remaining bits in byte 0 are the top 3: 0b101 (bits 5,6,7 = 1,0,1).
        // Plus low 2 bits of byte 1 = 0b11. Combined LSB-first → 0b11_101 = 0x1d.
        assert_eq!(br.read_bits(5).unwrap(), 0b11101);
    }

    #[test]
    fn read_past_eof_yields_zero_bits() {
        // Spec/libwebp behaviour: reads past the physical buffer return
        // zero-filled bits without erroring. This is part of the wire
        // contract we share with libwebp — callers detect truncation via
        // `at_end()`.
        let buf = [0xffu8];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.read_bits(8).unwrap(), 0xff);
        assert!(br.at_end());
        // Subsequent reads must succeed but return zero.
        assert_eq!(br.read_bits(8).unwrap(), 0);
        assert_eq!(br.read_bits(16).unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 0);
    }

    #[test]
    fn at_end_only_after_reading_buffer() {
        let buf = [0u8; 2];
        let mut br = BitReader::new(&buf);
        assert!(!br.at_end());
        let _ = br.read_bits(8).unwrap();
        // After 8 bits read, byte_pos = 1. Not at end yet.
        assert!(!br.at_end());
        let _ = br.read_bits(8).unwrap();
        assert!(br.at_end());
    }

    #[test]
    fn read_one_bit_at_a_time_full_byte() {
        // 0x96 = 0b1001_0110 → LSB-first 0,1,1,0,1,0,0,1
        let buf = [0x96u8];
        let mut br = BitReader::new(&buf);
        let expect = [0u32, 1, 1, 0, 1, 0, 0, 1];
        for (i, &e) in expect.iter().enumerate() {
            assert_eq!(br.read_bit().unwrap(), e, "bit {i}");
        }
    }

    #[test]
    fn empty_buffer_reads_yield_zero() {
        let buf: [u8; 0] = [];
        let mut br = BitReader::new(&buf);
        assert!(br.at_end());
        assert_eq!(br.read_bits(1).unwrap(), 0);
        assert_eq!(br.read_bits(7).unwrap(), 0);
        assert_eq!(br.read_bits(32).unwrap(), 0);
    }
}
