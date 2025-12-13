#![doc = include_str!("../README.md")]

pub mod classification;
pub mod clustering;
pub mod core;
pub mod regression;
pub mod text;
pub mod utils;

pub use classification::{
    BinaryAccuracy, BinaryAuroc, BinaryConfusionMatrix, BinaryF1Score, BinaryHingeLoss,
    BinaryJaccardIndex, BinaryPrecision, BinaryRecall, MulticlassAccuracy, MulticlassF1Score,
    MulticlassHingeLoss, MulticlassJaccardIndex, MulticlassPrecision,
};
pub use clustering::MutualInfoScore;
pub use core::{Metric, MetricError};
pub use regression::{
    MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError,
    NormalizedRootMeanSquaredError, R2Score,
};

pub use text::{Bleu, EditDistance, RougeScore};
pub use utils::Reduction;

#[cfg_attr(docsrs, doc(cfg(feature = "text-bert")))]
#[cfg(feature = "text-bert")]
pub use text::SentenceEmbeddingSimilarity;
