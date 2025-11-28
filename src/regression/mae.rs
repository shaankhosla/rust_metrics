use crate::core::{Metric, MetricError};

/// Online MeanAbsoluteError
///
/// ```
/// use rust_metrics::{MeanAbsoluteError, Metric};
///
/// let mut mae = MeanAbsoluteError::default();
/// mae.update((&[3.0, 5.0, 2.5, 7.0], &[2.5, 5.0, 4.0, 8.0])).unwrap();
/// assert_eq!(mae.compute().unwrap(), 0.75);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MeanAbsoluteError {
    sum_abs_error: f64,
    total: usize,
}

impl MeanAbsoluteError {
    pub fn new() -> Self {
        Self {
            sum_abs_error: 0.0,
            total: 0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for MeanAbsoluteError {
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
            let err = prediction - target;
            self.sum_abs_error += err.abs();
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.sum_abs_error = 0.0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        Some(self.sum_abs_error / self.total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{MeanAbsoluteError, Metric};

    #[test]
    fn mae_computes_over_batches() {
        let mut mae = MeanAbsoluteError::default();
        mae.update((&[3.0, 5.0, 2.5, 7.0], &[2.5, 5.0, 4.0, 8.0]))
            .unwrap();
        assert_eq!(mae.compute().unwrap(), 0.75);
    }
}
