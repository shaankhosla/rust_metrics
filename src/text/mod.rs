//! Text generation metrics: BLEU, edit distance, and optional BERT similarities.
//!
//! All types implement [`Metric`](crate::core::Metric) for batched updates.

#[cfg(feature = "text-bert")]
pub mod bert;

#[cfg(feature = "text-bert")]
pub use bert::SentenceEmbeddingSimilarity;

pub mod bleu;
pub mod edit;

pub use bleu::Bleu;
pub use edit::EditDistance;
