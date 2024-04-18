use std::sync::Once;

use log::LevelFilter;
use simple_logger::SimpleLogger;

static INIT: Once = Once::new();

pub fn setup() {
    INIT.call_once(|| {
        SimpleLogger::new()
            .with_level(LevelFilter::Info)
            .without_timestamps()
            .init()
            .unwrap();
    });
}

pub fn bits_to_bit_vec_u16(bits: &u16) -> Vec<bool> {
    let mut result = Vec::new();

    for i in (0..16).rev() {
        result.push(((bits >> i) & 1) == 1);
    }

    result
}

pub fn bits_to_bit_vec_u8(bits: &u8) -> Vec<bool> {
    let mut result = Vec::new();

    for i in (0..8).rev() {
        result.push(((bits >> i) & 1) == 1);
    }

    result
}

pub fn bits_to_u16(bits: &[bool]) -> u16 {
    let mut result = 0;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            result |= 1 << (bits.len() - 1 - i);
        }
    }
    result
}

pub fn bits_to_u8(bits: &[bool]) -> u8 {
    let mut result = 0;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            result |= 1 << (bits.len() - 1 - i);
        }
    }
    result
}

#[cfg(test)]
mod common_tests {
    use super::*;

    #[test]
    fn test_bits_to_bit_vec() {
        // Test case 1: All bits are 0
        let bits = 0u16;
        let expected = vec![false; 16];
        assert_eq!(bits_to_bit_vec_u16(&bits), expected);

        // Test case 2: All bits are 1
        let bits = 0xFFFFu16;
        let expected = vec![true; 16];
        assert_eq!(bits_to_bit_vec_u16(&bits), expected);

        // Test case 3: Alternating bits
        let bits = 0x5555u16;
        let expected = vec![
            false, true, false, true, false, true, false, true, false, true, false, true, false,
            true, false, true,
        ];
        assert_eq!(bits_to_bit_vec_u16(&bits), expected);
    }

    #[test]
    fn test_bits_to_u32() {
        // Test case 1: Empty vector
        let bits: Vec<bool> = vec![];
        assert_eq!(bits_to_u16(&bits), 0);

        // Test case 2: All bits are false
        let bits = vec![false; 16];
        assert_eq!(bits_to_u16(&bits), 0);

        // Test case 3: All bits are true
        let bits = vec![true; 16];
        assert_eq!(bits_to_u16(&bits), 0xFFFF);

        // Test case 4: Alternating bits
        let bits = vec![
            false, true, false, true, false, true, false, true, false, true, false, true, false,
            true, false, true,
        ];
        assert_eq!(bits_to_u16(&bits), 0x5555);
    }
}