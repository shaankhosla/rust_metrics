use crate::core::{Metric, MetricError};
use crate::utils::verify_range;

/// Online hinge loss for binary classification.
///
/// ```
/// use rust_metrics::{BinaryHinge, Metric};
///
/// let mut hinge = BinaryHinge::default();
/// hinge.update((&[0.8, -0.6], &[1.0, -1.0])).unwrap();
/// assert!(hinge.compute().unwrap() >= 0.0);
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
