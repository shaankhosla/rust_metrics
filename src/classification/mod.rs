//! Classification metrics inspired by TorchMetrics.
//!
//! Every struct in this module implements [`Metric`](crate::core::Metric) and therefore supports
//! batched updates plus `reset`/`compute` semantics.

pub mod accuracy;
pub mod auroc;
pub mod hinge;
pub mod precision_recall;

pub use accuracy::{BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy};
pub use auroc::BinaryAuroc;
pub use hinge::BinaryHinge;
pub use precision_recall::{BinaryPrecision, BinaryRecall};
