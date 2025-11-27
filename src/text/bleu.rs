use crate::core::{Metric, MetricError};
use crate::utils::tokenize;
use std::collections::HashMap;

/// Cumulative BLEU score with optional smoothing and arbitrary n-gram depth.
///
/// ```
/// use rust_metrics::{Bleu, Metric};
///
/// let preds = ["the cat is on the mat"];
/// let targets = ["the cat is on the mat"];
/// let mut bleu = Bleu::default();
/// bleu.update((&preds, &targets)).unwrap();
/// assert_eq!(bleu.compute(), Some(1.0));
/// ```
#[derive(Debug, Clone)]
pub struct Bleu {
    n_gram: usize,
    preds_len: usize,
    targets_len: usize,
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    smooth: bool,
}

impl Default for Bleu {
    fn default() -> Self {
        Self::new(4, false)
    }
}

impl Bleu {
    pub fn new(n_gram: usize, smooth: bool) -> Self {
        Self {
            n_gram,
            smooth,
            numerator: vec![0.0; n_gram],
            denominator: vec![0.0; n_gram],
            preds_len: 0,
            targets_len: 0,
        }
    }
}

fn count_ngrams<'a>(tokens: &[&'a str], n: usize) -> HashMap<Vec<&'a str>, usize> {
    let mut map = HashMap::new();
    if tokens.len() < n {
        return map;
    }
    for i in 0..=(tokens.len() - n) {
        let key = tokens[i..i + n].to_vec();
        *map.entry(key).or_insert(0) += 1;
    }
    map
}

impl Metric<(&[&str], &[&str])> for Bleu {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[&str], &[&str])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_tokens = tokenize(pred);
            let target_tokens = tokenize(target);
            self.preds_len += pred_tokens.len();
            self.targets_len += target_tokens.len();

            for n in 1..=self.n_gram {
                let pred_counts = count_ngrams(&pred_tokens, n);
                let target_counts = count_ngrams(&target_tokens, n);

                let mut clipped = 0usize;
                let mut total = 0usize;

                for (ngram, &p_count) in &pred_counts {
                    total += p_count;
                    if let Some(&t_count) = target_counts.get(ngram) {
                        clipped += p_count.min(t_count);
                    }
                }
                self.numerator[n - 1] += clipped as f64;
                self.denominator[n - 1] += total as f64;
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.numerator.fill(0.0);
        self.denominator.fill(0.0);
        self.preds_len = 0;
        self.targets_len = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.preds_len == 0 || self.targets_len == 0 {
            return None;
        }

        if self.numerator.first().copied().unwrap_or(0.0) == 0.0 {
            return Some(0.0);
        }

        if !self.smooth && self.numerator.contains(&0.0) {
            return Some(0.0);
        }

        let precision_scores: Vec<f64> = if self.smooth {
            let mut precisions: Vec<f64> = self
                .numerator
                .iter()
                .zip(&self.denominator)
                .map(|(&num, &den)| (num + 1.0) / (den + 1.0))
                .collect();

            if let (Some(first), Some(&den)) = (precisions.get_mut(0), self.denominator.first()) {
                *first = self.numerator[0] / den;
            }

            precisions
        } else {
            self.numerator
                .iter()
                .zip(&self.denominator)
                .map(|(&num, &den)| num / den)
                .collect()
        };

        if precision_scores.iter().any(|&p| p <= 0.0) {
            return Some(0.0);
        }

        let log_precision_sum: f64 = precision_scores
            .iter()
            .map(|&p| p.ln() / self.n_gram as f64)
            .sum();
        let geo_mean = log_precision_sum.exp();

        let c = self.preds_len as f64;
        let r = self.targets_len as f64;
        let bp = if c > r { 1.0 } else { (1.0 - r / c).exp() };

        Some(bp * geo_mean)
    }
}

#[cfg(test)]
mod tests {
    use super::Bleu;
    use crate::core::Metric;

    #[test]
    fn bleu_over_batches() {
        let mut bleu = Bleu::default();

        let preds = vec!["the cat is on the mat"];
        let targets = vec!["the cat is on the mat"];

        bleu.update((&preds, &targets)).unwrap();
        let score = bleu.compute().unwrap();
        assert!((score - 1.0).abs() < f64::EPSILON);

        bleu.reset();
        assert_eq!(bleu.compute(), None);

        let preds = vec!["the cat on the mat"];
        let targets = vec!["the cat on the rug"];
        bleu.update((&preds, &targets)).unwrap();
        let score = bleu.compute().unwrap();
        assert!((score - 0.668740304976422).abs() < f64::EPSILON);
    }

    #[test]
    fn smoothing_prevents_zero_score() {
        let preds = vec!["the cat sits"];
        let targets = vec!["the dog sits"];

        let mut bleu = Bleu::new(2, false);
        bleu.update((&preds, &targets)).unwrap();
        assert_eq!(bleu.compute().unwrap(), 0.0);

        let mut smoothed = Bleu::new(2, true);
        smoothed.update((&preds, &targets)).unwrap();
        assert!(smoothed.compute().unwrap() > 0.0);
    }
}
