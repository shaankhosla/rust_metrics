use crate::core::{Metric, MetricError};

/// Online MeanAbsolutePercentageError
///
/// ```
/// use rust_metrics::{MeanAbsolutePercentageError, Metric};
///
/// let mut mape = MeanAbsolutePercentageError::default();
///mape.update((&[0.9, 15.0, 1200000.0], &[1.0, 10.0, 1000000.0]))
///    .unwrap();
///assert!((mape.compute().unwrap() - 0.26666666666666666).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MeanAbsolutePercentageError {
    sum_abs_per_error: f64,
    total: usize,
}

impl MeanAbsolutePercentageError {
    pub fn new() -> Self {
        Self {
            sum_abs_per_error: 0.0,
            total: 0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for MeanAbsolutePercentageError {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            if target == 0.0 {
                continue;
            }
            self.sum_abs_per_error += (prediction - target).abs() / target.abs();
            self.total += 1;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.sum_abs_per_error = 0.0;
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        Some(self.sum_abs_per_error / self.total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{MeanAbsolutePercentageError, Metric};

    #[test]
    fn mape_computes_over_batches() {
        let mut mape = MeanAbsolutePercentageError::default();
        mape.update((&[0.9, 15.0, 1200000.0], &[1.0, 10.0, 1000000.0]))
            .unwrap();
        assert!((mape.compute().unwrap() - 0.26666666666666666).abs() < f64::EPSILON);
    }
}
