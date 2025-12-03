use crate::core::{Metric, MetricError};
use crate::utils::{count_ngrams, normalize, tokenize};
use std::collections::HashMap;

/// Calculate Rouge Score, used for automatic summarization.
///
/// Normalizes text by replacing any non-alpha-numeric characters
/// with spaces and converts to lowercase.
/// Tokenizes text by splitting on spaces.
///
/// ```
/// use rust_metrics::{text::rouge::RougeKey, Metric, RougeScore};
///
/// let preds = vec!["My name is John"];
/// let targets = vec!["Is your name John"];
///
/// let mut metric = RougeScore::new(vec![RougeKey::Rouge1, RougeKey::Rouge2]);
/// metric.update((&preds, &targets)).unwrap();
/// let score = metric.compute().unwrap();
/// assert_eq!(score.get(&RougeKey::Rouge1).unwrap().precision, 0.75);
/// assert_eq!(score.get(&RougeKey::Rouge1).unwrap().recall, 0.75);
/// assert_eq!(score.get(&RougeKey::Rouge1).unwrap().fmeasure, 0.75);
/// ```
pub struct RougeScore {
    rouge_keys: Vec<RougeKey>,
    stats: HashMap<RougeKey, RougeStats>,
    total: usize,
}

impl Default for RougeScore {
    fn default() -> Self {
        Self::new(vec![RougeKey::Rouge1, RougeKey::Rouge2])
    }
}

impl RougeScore {
    pub fn new(rouge_keys: Vec<RougeKey>) -> Self {
        Self {
            rouge_keys,
            stats: HashMap::new(),
            total: 0,
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub enum RougeKey {
    Rouge1,
    Rouge2,
    Rouge3,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RougeStats {
    pub precision: f64,
    pub recall: f64,
    pub fmeasure: f64,
}

//impl RunningStats {
//    fn finalize(&mut self, total: usize) {
//        let denom = total as f64;
//        self.precision = self.precision / denom;
//        self.recall = self.recall / denom;
//        self.fmeasure = self.fmeasure / denom;
//    }
//}

impl Metric<(&[&str], &[&str])> for RougeScore {
    type Output = HashMap<RougeKey, RougeStats>;

    fn update(&mut self, (predictions, targets): (&[&str], &[&str])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        for (prediction, target) in predictions.iter().zip(targets.iter()) {
            let prediction_norm = normalize(prediction);
            let target_norm = normalize(target);

            let prediction_tokens = tokenize(&prediction_norm);
            let target_tokens = tokenize(&target_norm);
            for rouge_key in &self.rouge_keys {
                let rouge = match rouge_key {
                    RougeKey::Rouge1 => rouge_n(&prediction_tokens, &target_tokens, 1),
                    RougeKey::Rouge2 => rouge_n(&prediction_tokens, &target_tokens, 2),
                    RougeKey::Rouge3 => rouge_n(&prediction_tokens, &target_tokens, 3),
                };

                if let Some(rouge) = rouge {
                    self.stats
                        .entry(*rouge_key)
                        .and_modify(|stats| {
                            stats.precision += rouge.precision;
                            stats.recall += rouge.recall;
                            stats.fmeasure += rouge.fmeasure;
                        })
                        .or_insert(rouge);
                }
            }
            self.total += 1;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.total = 0;
        self.stats.clear();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }

        let mut stats_to_return = HashMap::new();
        for (rouge_key, rouge) in &self.stats {
            stats_to_return.insert(
                *rouge_key,
                RougeStats {
                    precision: rouge.precision / self.total as f64,
                    recall: rouge.recall / self.total as f64,
                    fmeasure: rouge.fmeasure / self.total as f64,
                },
            );
        }
        Some(stats_to_return)
    }
}

fn rouge_n(pred_tokens: &[&str], target_tokens: &[&str], n: usize) -> Option<RougeStats> {
    if pred_tokens.len() < n || target_tokens.len() < n {
        return None;
    }

    let pred_counts = count_ngrams(pred_tokens, n);
    let target_counts = count_ngrams(target_tokens, n);
    let overlap: usize = pred_counts
        .iter()
        .map(|(ngram, &count)| count.min(*target_counts.get(ngram).unwrap_or(&0)))
        .sum();

    let pred_total = pred_tokens.len() + 1 - n;
    let target_total = target_tokens.len() + 1 - n;
    if pred_total == 0 || target_total == 0 {
        return None;
    }

    let precision = overlap as f64 / pred_total as f64;
    let recall = overlap as f64 / target_total as f64;
    let fmeasure = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Some(RougeStats {
        precision,
        recall,
        fmeasure,
    })
}

#[cfg(test)]
mod tests {
    use super::{RougeKey, RougeScore};
    use crate::core::Metric;

    #[test]
    fn rouge() {
        let mut metric = RougeScore::default();

        let preds = vec!["My name is John"];
        let targets = vec!["Is your name John"];

        metric.update((&preds, &targets)).unwrap();
        let score = metric.compute().unwrap();
        assert_eq!(score.get(&RougeKey::Rouge1).unwrap().precision, 0.75);
        assert_eq!(score.get(&RougeKey::Rouge1).unwrap().recall, 0.75);
        assert_eq!(score.get(&RougeKey::Rouge1).unwrap().fmeasure, 0.75);

        assert_eq!(score.get(&RougeKey::Rouge2).unwrap().precision, 0.0);

        metric.reset();
        let result = metric.compute();
        assert!(result.is_none());
    }
}
