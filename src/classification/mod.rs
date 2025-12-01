//! Classification metrics inspired by TorchMetrics.
//!
//! Every struct in this module implements [`Metric`](crate::core::Metric) and therefore supports
//! batched updates plus `reset`/`compute` semantics.

pub mod accuracy;
pub mod auroc;
pub mod confusion_matrix;
pub mod f1;
pub mod hinge;
pub mod precision_recall;
pub mod stat_scores;

pub use accuracy::{BinaryAccuracy, MulticlassAccuracy};
pub use auroc::BinaryAuroc;
pub use confusion_matrix::BinaryConfusionMatrix;
pub use f1::BinaryF1Score;
pub use hinge::BinaryHinge;
pub use precision_recall::{BinaryPrecision, BinaryRecall};
