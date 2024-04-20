use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[allow(unused)]
pub struct GaConfig {
    pub population_size: usize,
    pub tournament_size: usize,
    pub mutation_rate: f32,
    pub elite_factor: f32,
}
