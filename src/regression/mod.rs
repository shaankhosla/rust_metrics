//! Regression metrics
//!
//! All types implement [`Metric`](crate::core::Metric) for batched updates.

pub mod mae;
pub mod mape;
pub mod mse;
pub mod nrmse;
pub mod r2;

pub use mae::MeanAbsoluteError;
pub use mape::MeanAbsolutePercentageError;
pub use mse::MeanSquaredError;
pub use nrmse::{NormalizationType, NormalizedRootMeanSquaredError};
pub use r2::R2Score;
