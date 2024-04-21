#![allow(dead_code)]

use std::{error::Error, io::Read, sync::Once};

use flexi_logger::{FileSpec, Logger};
use log::LevelFilter;

pub type Dataset = Vec<(Vec<f64>, f64)>;

static INIT: Once = Once::new();

pub fn setup(log_level: &LevelFilter, log_directory: Option<&str>, is_headless: bool) {
    INIT.call_once(|| {
        let logger = Logger::try_with_str(log_level.as_str()).unwrap();

        let logger_configured = match log_directory {
            Some(log_directory_val) => {
                let file_spec = FileSpec::default()
                    .directory(log_directory_val)
                    .suppress_timestamp();
                if is_headless {
                    logger
                        .log_to_file(file_spec)
                        .duplicate_to_stdout(flexi_logger::Duplicate::All)
                } else {
                    logger.log_to_file(file_spec)
                }
            }
            None => logger.log_to_stdout(),
        };

        logger_configured.start().unwrap();
    });
}

pub fn dataset_from_csv<R: Read>(
    reader: R,
    has_headers: bool,
    delimiter: char,
) -> Result<Dataset, Box<dyn Error>> {
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
            let target: f64 = target.parse()?;
            let features: Result<Vec<_>, _> = record_iter.map(|cell| cell.parse::<f64>()).collect();
            dataset.push((features?, target));
        }
    }

    Ok(dataset)
}

#[cfg(test)]
mod util_tests {
    use super::*;

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
