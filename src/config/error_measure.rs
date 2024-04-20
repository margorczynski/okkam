use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub enum ErrorMeasure {
    MAE,
    MAPE,
    RMSE
}