use std::sync::{Arc, Mutex};

use fastembed::TextEmbedding;

use crate::{
    core::{Metric, MetricError},
    utils::cosine_similarity,
};

pub struct SentenceEmbeddingSimilarity {
    model: Arc<Mutex<TextEmbedding>>,
    predictions: Vec<String>,
    targets: Vec<String>,
}

impl Default for SentenceEmbeddingSimilarity {
    fn default() -> Self {
        let model =
            TextEmbedding::try_new(Default::default()).expect("Failed to initialize TextEmbedding");
        Self::new(Arc::new(Mutex::new(model)))
    }
}

impl SentenceEmbeddingSimilarity {
    pub fn new(model: Arc<Mutex<TextEmbedding>>) -> Self {
        Self {
            model,
            predictions: Vec::new(),
            targets: Vec::new(),
        }
    }

    fn embed_sentences(&self, sentences: Vec<String>) -> Vec<Vec<f32>> {
        let mut model = self.model.lock().expect("TextEmbedding lock poisoned");
        model
            .embed(sentences, None)
            .expect("Failed to embed sentences")
    }
}

impl Metric<(&[&str], &[&str])> for SentenceEmbeddingSimilarity {
    type Output = Vec<f64>;

    fn update(&mut self, (predictions, targets): (&[&str], &[&str])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        self.predictions
            .extend(predictions.iter().map(|s| s.to_string()));
        self.targets.extend(targets.iter().map(|s| s.to_string()));

        Ok(())
    }

    fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
    }

    fn compute(&self) -> Self::Output {
        if self.predictions.is_empty() {
            return Vec::new();
        }

        let prediction_embeddings = self.embed_sentences(self.predictions.clone());
        let target_embeddings = self.embed_sentences(self.targets.clone());

        prediction_embeddings
            .iter()
            .zip(target_embeddings.iter())
            .map(|(pred, tgt)| cosine_similarity(pred, tgt))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::SentenceEmbeddingSimilarity;
    use crate::core::Metric;

    #[test]
    fn bert_score_batches() {
        let mut bert_score = SentenceEmbeddingSimilarity::default();

        bert_score
            .update((&["hello world", "ping"], &["hi there world!", "pong"]))
            .expect("lengths should match");
        let expected_result = [0.6906931228059713, 0.6256474482252247];
        let result = bert_score.compute();
        assert_eq!(result, expected_result);

        bert_score.reset();
        assert_eq!(bert_score.compute().len(), 0);
    }
}
