use std::{
    io::Read,
    error::Error,
    fs::File,
    sync::Once,
};


use log::LevelFilter;
use simple_logger::SimpleLogger;

type Dataset = Vec<(Vec<f32>, f32)>;

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

pub fn dataset_from_csv<R: Read>(reader: R, has_headers: bool, delimiter: char) -> Result<Dataset, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(delimiter as u8)
        .flexible(false)
        .from_reader(reader);

    let mut dataset = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let mut record_iter = record.iter();

        if let Some(target) = record_iter.next_back() {
            let target: f32 = target.parse()?;
            let features: Result<Vec<_>, _> = record_iter.map(|cell| cell.parse::<f32>()).collect();
            dataset.push((features?, target));
        }
    }

    Ok(dataset)
}

#[cfg(test)]
mod util_tests {
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

    #[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_dataset_from_csv() {
        // Sample CSV data
        let csv_data = "1.0,2.0,3.0,4.0
4.0,5.0,6.0,7.0
8.0,9.0,10.0,11.0";

        // Create a cursor from the CSV data
        let reader = Cursor::new(csv_data.as_bytes());

        // Call the function with the cursor, no headers, and comma delimiter
        let dataset = dataset_from_csv(reader, false, ',').unwrap();

        // Expected dataset
        let expected_dataset = vec![
            (vec![1.0, 2.0, 3.0], 4.0),
            (vec![4.0, 5.0, 6.0], 7.0),
            (vec![8.0, 9.0, 10.0], 11.0),
        ];

        // Assert that the dataset matches the expected dataset
        assert_eq!(dataset, expected_dataset);
    }
}
}