use crate::core::{Metric, MetricError};
use crate::utils::{ConfusionMatrix, verify_range};

/// Online F1 Score for binary classification.
///
/// ```
/// use rust_metrics::{BinaryF1Score, Metric};
///
/// let mut f1 = BinaryF1Score::default();
/// f1.update((&[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
/// &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0])).unwrap();
/// assert!(f1.compute().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct BinaryF1Score {
    confusion_matrix: ConfusionMatrix,
}

impl Default for BinaryF1Score {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl BinaryF1Score {
    pub fn new(threshold: f64) -> Self {
        let confusion_matrix = ConfusionMatrix::new(threshold);
        Self { confusion_matrix }
    }
}

impl Metric<(&[f64], &[f64])> for BinaryF1Score {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            self.confusion_matrix.update(prediction, target)?;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.confusion_matrix.reset();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.confusion_matrix.total == 0 {
            return None;
        }
        let precision = self.confusion_matrix.true_positive as f64
            / (self.confusion_matrix.true_positive + self.confusion_matrix.false_positive) as f64;
        let recall = self.confusion_matrix.true_positive as f64
            / (self.confusion_matrix.true_positive + self.confusion_matrix.false_negative) as f64;
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

        f1.update((
            &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ))
        .expect("update should succeed");
        assert!((f1.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);
        f1.update((&[0.7], &[1.0])).expect("update should succeed");
        assert_eq!(f1.compute().unwrap(), 0.75);

        f1.reset();
        assert_eq!(f1.compute(), None);
    }
}
