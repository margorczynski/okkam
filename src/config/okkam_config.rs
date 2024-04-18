use config::{Config, ConfigError, Environment, File};
use log::debug;
use serde::Deserialize;

use super::{ga_config::GaConfig, polynomial_config::PolynomialConfig};

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct OkkamConfig {
    pub dataset_path: String,
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