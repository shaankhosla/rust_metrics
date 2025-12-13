//! Clustering metrics inspired by TorchMetrics.
//!
//! Every struct in this module implements [`Metric`](crate::core::Metric) and therefore supports
//! batched updates plus `reset`/`compute` semantics.

pub mod mutual_info_score;

pub use mutual_info_score::MutualInfoScore;
