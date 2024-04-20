use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub enum ErrorMeasure {
    MAE,
    MAPE,
    RMSE,
}
