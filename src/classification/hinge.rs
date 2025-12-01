use crate::core::{Metric, MetricError};
use crate::utils::{verify_binary_label, verify_label, verify_range};

/// Online hinge loss for binary classification.
///
///
/// ```
/// use rust_metrics::{BinaryHingeLoss, Metric};
///
/// let preds = [0.25, 0.25, 0.55, 0.75, 0.75];
/// let target = [0, 0, 1, 1, 1];
///
/// let mut hinge = BinaryHingeLoss::default();
/// hinge.update((&preds, &target)).unwrap();
/// assert!((hinge.compute().unwrap() - 0.69).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct BinaryHingeLoss {
    squared: bool,
    measures: f64,
    total: usize,
}

impl Default for BinaryHingeLoss {
    fn default() -> Self {
        Self::new(false)
    }
}

impl BinaryHingeLoss {
    pub fn new(squared: bool) -> Self {
        Self {
            squared,
            measures: 0.0,
            total: 0,
        }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryHingeLoss {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[usize])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        self.total += predictions.len();
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            verify_range(prediction, 0.0, 1.0)?;
            verify_binary_label(target)?;

            let y = if target == 1 { 1.0 } else { -1.0 };
            let mut measure = (1.0 - prediction * y).max(0.0);
            if self.squared {
                measure *= measure;
            }
            self.measures += measure;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.measures = 0.0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        Some(self.measures / self.total as f64)
    }
}

/// Online hinge loss for multiclass classification. Currently only the Crammer-Singer loss is supported
///
///
/// ```
/// use rust_metrics::{MulticlassHingeLoss, Metric};
///
/// let mut hinge = MulticlassHingeLoss::new(3, false);
/// let preds: [&[f64]; 4] = [
/// &[0.25, 0.20, 0.55][..],
/// &[0.55, 0.05, 0.40][..],
/// &[0.10, 0.30, 0.60][..],
/// &[0.90, 0.05, 0.05][..],
/// ];
///
/// let target = [0, 1, 2, 0];
///
/// hinge.update((&preds, &target)).unwrap();
/// assert!((hinge.compute().unwrap() - 0.9125).abs() < 1e-12);
///
///
/// let mut hinge = MulticlassHingeLoss::new(3, true);
/// hinge.update((&preds, &target)).unwrap();
/// assert!((hinge.compute().unwrap() - 1.1131250000000001).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct MulticlassHingeLoss {
    num_classes: usize,
    squared: bool,
    measures: f64,
    total: usize,
}

impl MulticlassHingeLoss {
    pub fn new(num_classes: usize, squared: bool) -> Self {
        assert!(num_classes >= 2, "num_classes must be at least 2");
        Self {
            num_classes,
            squared,
            measures: 0.0,
            total: 0,
        }
    }
}

impl Metric<(&[&[f64]], &[usize])> for MulticlassHingeLoss {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[&[f64]], &[usize])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        for (&prediction_batch, &target) in predictions.iter().zip(targets.iter()) {
            if prediction_batch.len() != self.num_classes {
                return Err(MetricError::LengthMismatch {
                    predictions: prediction_batch.len(),
                    targets: self.num_classes,
                });
            }

            verify_label(target, self.num_classes)?;
            let true_score = prediction_batch[target];
            let mut max_other_score: f64 = -1.0;
            for (i, &prediction) in prediction_batch.iter().enumerate() {
                verify_range(prediction, 0.0, 1.0)?;
                if i == target {
                    continue;
                }
                max_other_score = max_other_score.max(prediction);
            }
            let mut loss = (1.0 - true_score + max_other_score).max(0.0);
            if self.squared {
                loss *= loss;
            }
            self.measures += loss;
            self.total += 1;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.measures = 0.0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        Some(self.measures / self.total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryHingeLoss, MulticlassHingeLoss};
    use crate::core::Metric;

    #[test]
    fn binary_hinge_computes_over_batches() {
        let mut hinge = BinaryHingeLoss::default();

        let preds = &[0.25, 0.25, 0.55, 0.75, 0.75];
        let target = &[0, 0, 1, 1, 1];

        hinge.update((preds, target)).unwrap();
        dbg!(hinge.compute().unwrap());
        assert!((hinge.compute().unwrap() - 0.69).abs() < 1e-12);

        hinge.reset();
        assert_eq!(hinge.compute(), None);

        let mut hinge = BinaryHingeLoss::new(true);

        hinge.update((preds, target)).unwrap();
        assert!((hinge.compute().unwrap() - 0.6905).abs() < 1e-12);
    }

    #[test]
    fn multiclass_hinge() {
        let mut hinge = MulticlassHingeLoss::new(3, false);
        let preds: [&[f64]; 4] = [
            &[0.25, 0.20, 0.55][..],
            &[0.55, 0.05, 0.40][..],
            &[0.10, 0.30, 0.60][..],
            &[0.90, 0.05, 0.05][..],
        ];

        let target = [0, 1, 2, 0];

        hinge.update((&preds, &target)).unwrap();
        assert!((hinge.compute().unwrap() - 0.9125).abs() < 1e-12);

        hinge.reset();
        assert_eq!(hinge.compute(), None);

        let mut hinge = MulticlassHingeLoss::new(3, true);
        hinge.update((&preds, &target)).unwrap();
        assert!((hinge.compute().unwrap() - 1.1131250000000001).abs() < 1e-12);
    }
}
