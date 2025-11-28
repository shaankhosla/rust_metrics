use crate::core::{Metric, MetricError};

/// Online MeanSquaredError
///
/// ```
/// use rust_metrics::{MeanSquaredError, Metric};
///
/// let mut mse = MeanSquaredError::default();
/// mse.update((&[3.0, 5.0, 2.5, 7.0], &[2.5, 5.0, 4.0, 8.0])).unwrap();
/// assert_eq!(mse.compute().unwrap(), 0.8750);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MeanSquaredError {
    sum_squared_error: f64,
    total: usize,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            total: 0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for MeanSquaredError {
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
            self.sum_squared_error += err * err;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        Some(self.sum_squared_error / self.total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{MeanSquaredError, Metric};

    #[test]
    fn mse_computes_over_batches() {
        let mut mse = MeanSquaredError::default();
        mse.update((&[3.0, 5.0, 2.5, 7.0], &[2.5, 5.0, 4.0, 8.0]))
            .unwrap();
        assert_eq!(mse.compute().unwrap(), 0.8750);
    }
}
