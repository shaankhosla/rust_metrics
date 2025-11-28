//! Regression metrics: mean squared error, mean absolute error
//!
//! All types implement [`Metric`](crate::core::Metric) for batched updates.

pub mod mae;
pub mod mse;
pub mod r2;

pub use mae::MeanAbsoluteError;
pub use mse::MeanSquaredError;
pub use r2::R2Score;
