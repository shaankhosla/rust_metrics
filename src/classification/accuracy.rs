use crate::core::{Metric, MetricError};

#[derive(Debug, Default, Clone)]
pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric<(&[f64], &[f64])> for Accuracy {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
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

    fn compute(&self) -> Self::Output {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Accuracy;
    use crate::core::{Metric, MetricError};

    #[test]
    fn computes_accuracy_over_batches() {
        let mut accuracy = Accuracy::new();

        accuracy
            .update((&[0.0, 1.0, 2.0], &[0.0, 2.0, 2.0]))
            .expect("update should succeed");
        assert!((accuracy.compute() - (2.0 / 3.0)).abs() < f64::EPSILON);

        accuracy
            .update((&[1.0, 2.0], &[1.0, 2.0]))
            .expect("update should succeed");
        assert!((accuracy.compute() - 0.8).abs() < f64::EPSILON);

        accuracy.reset();
        assert_eq!(accuracy.compute(), 0.0);
    }

    #[test]
    fn rejects_mismatched_batches() {
        let mut accuracy = Accuracy::new();
        let err = accuracy
            .update((&[0.0, 1.0], &[0.0]))
            .expect_err("length mismatch should be rejected");

        assert_eq!(
            err,
            MetricError::LengthMismatch {
                predictions: 2,
                targets: 1
            }
        );
    }
}
