use std::sync::{Arc, Mutex};

use fastembed::TextEmbedding;

use crate::{
    core::{Metric, MetricError},
    utils::cosine_similarity,
};

pub struct SentenceEmbeddingSimilarity {
    model: Arc<Mutex<TextEmbedding>>,
    prediction_embeddings: Vec<Vec<f32>>,
    target_embeddings: Vec<Vec<f32>>,
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
            prediction_embeddings: Vec::new(),
            target_embeddings: Vec::new(),
        }
    }

    fn embed_sentences(&self, sentences: &[&str]) -> Vec<Vec<f32>> {
        let inputs: Vec<String> = sentences.iter().map(|s| (*s).to_string()).collect();
        let mut model = self.model.lock().expect("TextEmbedding lock poisoned");
        model
            .embed(inputs, None)
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

        let prediction_embeddings = self.embed_sentences(predictions);
        let target_embeddings = self.embed_sentences(targets);
        self.prediction_embeddings.extend(prediction_embeddings);
        self.target_embeddings.extend(target_embeddings);

        Ok(())
    }

    fn reset(&mut self) {
        self.prediction_embeddings = Vec::new();
        self.target_embeddings = Vec::new();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.prediction_embeddings.is_empty() {
            return None;
        }

        Some(
            self.prediction_embeddings
                .iter()
                .zip(self.target_embeddings.iter())
                .map(|(pred, tgt)| cosine_similarity(pred, tgt))
                .collect(),
        )
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
        let result = bert_score.compute().unwrap();
        assert_eq!(result, expected_result);

        bert_score.reset();
        assert_eq!(bert_score.compute(), None);
    }
}
