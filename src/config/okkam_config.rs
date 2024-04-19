use std::str::FromStr;

use config::{Config, ConfigError, Environment, File};
use log::{debug, LevelFilter};
use serde::{Deserialize, Deserializer};

use super::{ga_config::GaConfig, polynomial_config::PolynomialConfig};

#[derive(Debug)]
pub struct LevelFilterWrapper(pub LevelFilter);

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct OkkamConfig {
    pub log_level: LevelFilterWrapper,
    pub dataset_path: Box<str>,
    pub result_path: Box<str>,
    pub ga: GaConfig,
    pub polynomial: PolynomialConfig
}

impl OkkamConfig {
    const ENV_VAR_PREFIX: &'static str = "okkam";

    pub fn new(config_path: &str) -> Result<Self, ConfigError> {
        let s = Config::builder()
            .add_source(File::with_name(config_path))
            .add_source(Environment::with_prefix(Self::ENV_VAR_PREFIX))
            .build()?;

        debug!("Using config: {:?}", s);

        s.try_deserialize()
    }
}

// Newtype wrapper for LevelFilter

impl<'de> Deserialize<'de> for LevelFilterWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let level_str = String::deserialize(deserializer)?;
        LevelFilter::from_str(&level_str)
            .map(LevelFilterWrapper)
            .map_err(|_| serde::de::Error::custom("Invalid log level"))
    }
}