pub mod classification;
pub mod core;
pub mod text;
pub mod utils;

pub use classification::{
    BinaryAccuracy, BinaryAuroc, BinaryPrecision, BinaryRecall, MulticlassAccuracy,
    MultilabelAccuracy,
};
pub use core::{Metric, MetricError};

pub use text::Bleu;

#[cfg(feature = "text-bert")]
pub use text::SentenceEmbeddingSimilarity;
