use crate::core::{Metric, MetricError};

/// Online R2Score
///
/// ```
/// use rust_metrics::{Metric, R2Score};
///
/// let preds = [2.5, 0.0, 2.0, 8.0];
/// let target = [3.0, -0.5, 2.0, 7.0];
///
/// let mut r2 = R2Score::default();
/// r2.update((&preds, &target)).unwrap();
/// assert!((r2.compute().unwrap() - 0.9486081370449679).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct R2Score {
    sum_squared_error: f64,
    sum_error: f64,
    residual: f64,
    total: usize,
}

impl R2Score {
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            sum_error: 0.0,
            residual: 0.0,
            total: 0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for R2Score {
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
            self.sum_error += target;
            self.residual += target * target;
            let error = prediction - target;
            self.sum_squared_error += error * error;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.sum_error = 0.0;
        self.residual = 0.0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        let target_mean = self.sum_error / self.total as f64;
        let sum_squares = self.residual - (self.total as f64) * target_mean * target_mean;
        let r2 = 1.0 - self.sum_squared_error / sum_squares;
        Some(r2)
    }
}

#[cfg(test)]
mod tests {
    use super::{Metric, R2Score};

    #[test]
    fn r2_compute_over_batches() {
        let mut r2 = R2Score::default();
        r2.update((&[2.5, 0.0, 2.0, 8.0], &[3.0, -0.5, 2.0, 7.0]))
            .unwrap();
        assert!((r2.compute().unwrap() - 0.9486081370449679).abs() < f64::EPSILON);
    }
}
