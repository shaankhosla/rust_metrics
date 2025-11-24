pub mod classification;
pub mod core;

pub use classification::{BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy};
pub use core::{Metric, MetricError};
