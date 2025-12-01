use crate::core::{Metric, MetricError};
use crate::utils::{verify_binary_label, verify_range};

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

#[cfg(test)]
mod tests {
    use super::BinaryHingeLoss;
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
        dbg!(hinge.compute().unwrap());
        assert!((hinge.compute().unwrap() - 0.6905).abs() < 1e-12);
    }
}
