#[cfg(feature = "text-bert")]
pub mod bert;

#[cfg(feature = "text-bert")]
pub use bert::SentenceEmbeddingSimilarity;
