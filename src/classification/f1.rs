use crate::core::{Metric, MetricError};
use crate::utils::verify_range;

/// Online F1 Score for binary classification.
///
/// ```
/// use rust_metrics::{BinaryF1Score, Metric};
///
/// let mut f1 = BinaryF1Score::default();
/// f1.update((&[0.8, -0.6], &[1.0, -1.0])).unwrap();
/// assert!(f1.compute().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct BinaryF1Score {
    threshold: f64,
    squared: bool,
    measures: f64,
    total: usize,
}

impl Default for BinaryF1Score {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl BinaryF1Score {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            measures: 0.0,
            total: 0,
        }
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
    use super::BinaryF1Score;
    use crate::core::Metric;

    #[test]
    fn binary_hinge_computes_over_batches() {
        let mut hinge = BinaryHinge::default();

        hinge
            .update((&[0.8, -0.6, 0.3, 0.1], &[1.0, -1.0, 1.0, -1.0]))
            .expect("update should succeed");
        assert_eq!(hinge.compute().unwrap(), 0.6);
        hinge
            .update((&[0.7], &[1.0]))
            .expect("update should succeed");
        assert_eq!(hinge.compute().unwrap(), 0.54);

        hinge.reset();
        assert_eq!(hinge.compute(), None);
    }
}
