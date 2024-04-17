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