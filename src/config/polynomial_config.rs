use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct PolynomialConfig {
    pub terms_num: usize,
    pub degree_bits_num: usize,
}
