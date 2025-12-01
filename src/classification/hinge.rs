use crate::core::{Metric, MetricError};
use crate::utils::verify_range;

/// Online hinge loss for binary classification.
///
/// Predictions stay within `[-1, 1]` and
/// labels must be encoded as `-1.0` (negative) or `1.0` (positive).
///
/// ```
/// use rust_metrics::{BinaryHinge, Metric};
///
/// let preds = [0.25, 0.25, 0.55, 0.75, 0.75];
/// let targets = [-1.0, -1.0, 1.0, 1.0, 1.0];
///
/// let mut hinge = BinaryHinge::default();
/// hinge.update((&preds, &targets)).unwrap();
/// assert!((hinge.compute().unwrap() - 0.69).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct BinaryHinge {
    squared: bool,
    measures: f64,
    total: usize,
}

impl Default for BinaryHinge {
    fn default() -> Self {
        Self::new(false)
    }
}

impl BinaryHinge {
    pub fn new(squared: bool) -> Self {
        Self {
            squared,
            measures: 0.0,
            total: 0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for BinaryHinge {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        self.total += predictions.len();
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            verify_range(prediction, -1.0, 1.0)?;
            verify_range(target, -1.0, 1.0)?;
            let mut measure = (1.0 - prediction * target).max(0.0);
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
    use super::BinaryHinge;
    use crate::core::Metric;

    #[test]
    fn binary_hinge_computes_over_batches() {
        let mut hinge = BinaryHinge::default();

        hinge
            .update((&[0.25, 0.25, 0.55], &[-1.0, -1.0, 1.0]))
            .expect("update should succeed");
        hinge
            .update((&[0.75, 0.75], &[1.0, 1.0]))
            .expect("update should succeed");
        assert!((hinge.compute().unwrap() - 0.69).abs() < 1e-12);

        hinge.reset();
        assert_eq!(hinge.compute(), None);
    }
}
