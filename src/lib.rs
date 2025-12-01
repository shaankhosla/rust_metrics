#![doc = include_str!("../README.md")]

pub mod classification;
pub mod core;
pub mod regression;
pub mod text;
pub mod utils;

pub use classification::{
    BinaryAccuracy, BinaryAuroc, BinaryConfusionMatrix, BinaryF1Score, BinaryHinge,
    BinaryPrecision, BinaryRecall, MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
};
pub use core::{Metric, MetricError};
pub use regression::{MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score};

pub use text::{Bleu, EditDistance};

#[cfg_attr(docsrs, doc(cfg(feature = "text-bert")))]
#[cfg(feature = "text-bert")]
pub use text::SentenceEmbeddingSimilarity;
