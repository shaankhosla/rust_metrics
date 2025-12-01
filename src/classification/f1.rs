use crate::core::{Metric, MetricError};

use super::stat_scores::BinaryStatScores;

/// Online F1 Score for binary classification.
///
/// ```
/// use rust_metrics::{BinaryF1Score, Metric};
///
/// let mut f1 = BinaryF1Score::default();
/// f1.update((&[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
/// &[0_usize, 1, 0, 1, 0, 1])).unwrap();
/// assert!(f1.compute().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryF1Score {
    stat_scores: BinaryStatScores,
}

impl BinaryF1Score {
    pub fn new(threshold: f64) -> Self {
        let stat_scores = BinaryStatScores::new(threshold);
        Self { stat_scores }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryF1Score {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[usize])) -> Result<(), MetricError> {
        self.stat_scores.update((predictions, targets))?;

        Ok(())
    }

    fn reset(&mut self) {
        self.stat_scores.reset();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.stat_scores.total == 0 {
            return None;
        }
        let precision = self.stat_scores.true_positive as f64
            / (self.stat_scores.true_positive + self.stat_scores.false_positive) as f64;
        let recall = self.stat_scores.true_positive as f64
            / (self.stat_scores.true_positive + self.stat_scores.false_negative) as f64;
        Some(2.0 * precision * recall / (precision + recall))
    }
}

#[cfg(test)]
mod tests {
    use super::BinaryF1Score;
    use crate::core::Metric;

    #[test]
    fn f1_computes_over_batches() {
        let mut f1 = BinaryF1Score::default();

        f1.update((&[0.0, 0.0, 1.0, 1.0, 0.0, 1.0], &[0_usize, 1, 0, 1, 0, 1]))
            .expect("update should succeed");
        assert!((f1.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);
        f1.update((&[0.7], &[1_usize]))
            .expect("update should succeed");
        assert_eq!(f1.compute().unwrap(), 0.75);

        f1.reset();
        assert_eq!(f1.compute(), None);
    }
}
