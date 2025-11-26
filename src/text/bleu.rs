use crate::core::{Metric, MetricError};
use crate::utils::tokenize;
use std::collections::HashMap;

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
        if self.denominator.contains(&0.0) {
            return None;
        }

        let precisions: Vec<f64> = self
            .numerator
            .iter()
            .zip(&self.denominator)
            .map(|(&num, &den)| {
                if den == 0.0 {
                    0.0
                } else if num == 0.0 && self.smooth {
                    1e-9
                } else {
                    num / den
                }
            })
            .collect();

        let c = self.preds_len as f64;
        let r = self.targets_len as f64;
        let bp = if c < r { (-1.0 + r / c).exp() } else { 1.0 };

        let mut log_sum = 0.0;
        let valid_precisions: Vec<f64> = precisions.iter().cloned().filter(|&p| p > 0.0).collect();
        if valid_precisions.is_empty() {
            return Some(0.0);
        }

        for p in &precisions {
            log_sum += p.max(1e-16).ln();
        }
        let geo_mean = (log_sum / self.n_gram as f64).exp();

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
}
