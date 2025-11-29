use crate::core::{Metric, MetricError};
use crate::utils::ConfusionMatrix;

/// Thresholded precision for binary classification probabilities.
///
/// ```
/// use rust_metrics::{BinaryPrecision, Metric};
///
/// let mut precision = BinaryPrecision::new(0.5);
/// precision.update((&[0.9, 0.4], &[1_usize, 0])).unwrap();
/// assert_eq!(precision.compute(), Some(1.0));
/// ```
#[derive(Debug, Clone)]
pub struct BinaryPrecision {
    confusion_matrix: ConfusionMatrix,
}

impl Default for BinaryPrecision {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl BinaryPrecision {
    pub fn new(threshold: f64) -> Self {
        let confusion_matrix = ConfusionMatrix::new(threshold);
        Self { confusion_matrix }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryPrecision {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[usize])) -> Result<(), MetricError> {
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
        Some(
            self.confusion_matrix.true_positive as f64
                / (self.confusion_matrix.true_positive + self.confusion_matrix.false_positive)
                    as f64,
        )
    }
}

/// Thresholded recall for binary classification probabilities.
///
/// ```
/// use rust_metrics::{BinaryRecall, Metric};
///
/// let mut recall = BinaryRecall::new(0.5);
/// recall.update((&[0.9, 0.4], &[1_usize, 1])).unwrap();
/// assert_eq!(recall.compute(), Some(0.5));
/// ```
#[derive(Debug, Clone)]
pub struct BinaryRecall {
    confusion_matrix: ConfusionMatrix,
}

impl Default for BinaryRecall {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl BinaryRecall {
    pub fn new(threshold: f64) -> Self {
        let confusion_matrix = ConfusionMatrix::new(threshold);
        Self { confusion_matrix }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryRecall {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[usize])) -> Result<(), MetricError> {
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
        Some(
            self.confusion_matrix.true_positive as f64
                / (self.confusion_matrix.true_positive + self.confusion_matrix.false_negative)
                    as f64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryPrecision, BinaryRecall};
    use crate::core::{Metric, MetricError};

    #[test]
    fn binary_precision_computes_over_batches() {
        let mut precision = BinaryPrecision::default();

        precision
            .update((&[0.8, 0.6, 0.3, 0.1], &[1_usize, 0, 1, 0]))
            .expect("update should succeed");
        precision
            .update((&[0.7], &[1_usize]))
            .expect("update should succeed");
        assert_eq!(precision.compute().unwrap(), 2.0 / 3.0);

        precision.reset();
        assert_eq!(precision.compute(), None);
    }

    #[test]
    fn binary_precision_validates_targets() {
        let mut precision = BinaryPrecision::default();
        let err = precision
            .update((&[0.8], &[2_usize]))
            .expect_err("invalid targets should fail");
        assert_eq!(
            err,
            MetricError::IncompatibleInput {
                expected: "target must be 0 or 1",
                got: "other",
            }
        );
    }

    #[test]
    fn binary_recall_computes_over_batches() {
        let mut recall = BinaryRecall::default();

        recall
            .update((&[0.8, 0.6, 0.3, 0.1], &[1_usize, 0, 1, 0]))
            .expect("update should succeed");
        recall
            .update((&[0.7], &[1_usize]))
            .expect("update should succeed");
        assert_eq!(recall.compute().unwrap(), 2.0 / 3.0);

        recall.reset();
        assert_eq!(recall.compute(), None);
    }

    #[test]
    fn binary_recall_validates_targets() {
        let mut recall = BinaryRecall::default();
        let err = recall
            .update((&[0.8], &[2_usize]))
            .expect_err("invalid targets should fail");
        assert_eq!(
            err,
            MetricError::IncompatibleInput {
                expected: "target must be 0 or 1",
                got: "other",
            }
        );
    }
}
