use crate::core::{Metric, MetricError};

/// Running accuracy for binary classification tasks (`0/1` labels).
///
/// ```
/// use rust_metrics::{BinaryAccuracy, Metric};
///
/// let predictions = [0_usize, 1, 1];
/// let targets = [0_usize, 1, 0];
/// let mut metric = BinaryAccuracy::new();
/// metric.update((&predictions, &targets)).unwrap();
/// assert_eq!(metric.compute(), Some(2.0 / 3.0));
/// ```
#[derive(Debug, Default, Clone)]
pub struct BinaryAccuracy {
    correct: usize,
    total: usize,
}

impl BinaryAccuracy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric<(&[usize], &[usize])> for BinaryAccuracy {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[usize], &[usize])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        self.total += predictions.len();
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            if prediction > 1 || target > 1 {
                return Err(MetricError::IncompatibleInput {
                    expected: "0 or 1",
                    got: "other",
                });
            }
            if prediction == target {
                self.correct += 1;
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            None
        } else {
            Some(self.correct as f64 / self.total as f64)
        }
    }
}

/// Multiclass accuracy that validates class indices on every batch.
///
/// ```
/// use rust_metrics::{Metric, MulticlassAccuracy};
///
/// let mut metric = MulticlassAccuracy::new(3);
/// metric.update((&[0_usize, 2], &[0_usize, 1])).unwrap();
/// assert_eq!(metric.compute(), Some(0.5));
/// ```
#[derive(Debug, Clone)]
pub struct MulticlassAccuracy {
    num_classes: usize,
    correct: usize,
    total: usize,
}

impl MulticlassAccuracy {
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes >= 2, "num_classes must be at least 2");

        Self {
            num_classes,
            correct: 0,
            total: 0,
        }
    }
}

impl Metric<(&[usize], &[usize])> for MulticlassAccuracy {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[usize], &[usize])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        let mut batch_correct = 0usize;
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            if prediction >= self.num_classes {
                return Err(MetricError::InvalidClassIndex {
                    class: prediction,
                    num_classes: self.num_classes,
                });
            }

            if target >= self.num_classes {
                return Err(MetricError::InvalidClassIndex {
                    class: target,
                    num_classes: self.num_classes,
                });
            }

            if prediction == target {
                batch_correct += 1;
            }
        }

        self.total += predictions.len();
        self.correct += batch_correct;
        Ok(())
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            None
        } else {
            Some(self.correct as f64 / self.total as f64)
        }
    }
}

/// Multilabel accuracy over flattened bool tensors.
///
/// ```
/// use rust_metrics::{Metric, MultilabelAccuracy};
///
/// let preds = [true, false, true, false];
/// let targets = [true, true, false, false];
/// let mut metric = MultilabelAccuracy::new(2);
/// metric.update((&preds, &targets)).unwrap();
/// assert_eq!(metric.compute(), Some(0.5));
/// ```
#[derive(Debug, Clone)]
pub struct MultilabelAccuracy {
    num_labels: usize,
    correct: usize,
    total: usize,
}

impl MultilabelAccuracy {
    pub fn new(num_labels: usize) -> Self {
        assert!(num_labels >= 1, "num_labels must be at least 1");

        Self {
            num_labels,
            correct: 0,
            total: 0,
        }
    }
}

impl Metric<(&[bool], &[bool])> for MultilabelAccuracy {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[bool], &[bool])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        if predictions.len() % self.num_labels != 0 {
            return Err(MetricError::InvalidLabelShape {
                total_labels: predictions.len(),
                num_labels: self.num_labels,
            });
        }

        self.total += predictions.len();
        self.correct += predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| pred == target)
            .count();

        Ok(())
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            None
        } else {
            Some(self.correct as f64 / self.total as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy};
    use crate::core::{Metric, MetricError};

    #[test]
    fn binary_accuracy_computes_over_batches() {
        let mut accuracy = BinaryAccuracy::new();

        accuracy.update((&[0, 1, 0], &[0, 1, 1])).unwrap();
        assert!((accuracy.compute().unwrap() - (2.0 / 3.0)).abs() < f64::EPSILON);

        accuracy.update((&[1, 0], &[1, 0])).unwrap();
        assert!((accuracy.compute().unwrap() - 0.8).abs() < f64::EPSILON);

        accuracy.reset();
        assert_eq!(accuracy.compute(), None);
    }

    #[test]
    fn binary_accuracy_rejects_mismatched_batches() {
        let mut accuracy = BinaryAccuracy::new();
        let err = accuracy
            .update((&[0, 1], &[0]))
            .expect_err("length mismatch should be rejected");

        assert_eq!(
            err,
            MetricError::LengthMismatch {
                predictions: 2,
                targets: 1
            }
        );
    }

    #[test]
    fn multiclass_accuracy_requires_valid_classes() {
        let mut accuracy = MulticlassAccuracy::new(3);

        let err = accuracy
            .update((&[0, 1, 2], &[0, 1, 3]))
            .expect_err("class index should be validated");

        assert_eq!(
            err,
            MetricError::InvalidClassIndex {
                class: 3,
                num_classes: 3
            }
        );
    }

    #[test]
    fn multiclass_accuracy_computes_correctly() {
        let mut accuracy = MulticlassAccuracy::new(3);

        accuracy.update((&[0, 1, 2, 2], &[0, 1, 1, 2])).unwrap();
        assert!((accuracy.compute().unwrap() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn multilabel_accuracy_validates_shape() {
        let mut accuracy = MultilabelAccuracy::new(3);

        let err = accuracy
            .update((&[true, false], &[true, true]))
            .expect_err("input length must align with num_labels");

        assert_eq!(
            err,
            MetricError::InvalidLabelShape {
                total_labels: 2,
                num_labels: 3
            }
        );
    }

    #[test]
    fn multilabel_accuracy_computes_correctly() {
        let mut accuracy = MultilabelAccuracy::new(3);
        let preds = [true, false, true, false, true, false];
        let targets = [true, true, true, false, false, false];

        accuracy
            .update((&preds, &targets))
            .expect("valid multilabel batch");
        assert!((accuracy.compute().unwrap() - (4.0 / 6.0)).abs() < f64::EPSILON);
    }
}
