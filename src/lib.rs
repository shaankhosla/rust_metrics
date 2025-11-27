#![doc = include_str!("../README.md")]

pub mod classification;
pub mod core;
pub mod text;
pub mod utils;

pub use classification::{
    BinaryAccuracy, BinaryAuroc, BinaryPrecision, BinaryRecall, MulticlassAccuracy,
    MultilabelAccuracy,
};
pub use core::{Metric, MetricError};

pub use text::{Bleu, EditDistance};

#[cfg_attr(docsrs, doc(cfg(feature = "text-bert")))]
#[cfg(feature = "text-bert")]
pub use text::SentenceEmbeddingSimilarity;
