use crate::core::{Metric, MetricError};

#[derive(Debug, Clone, Default)]
pub enum NormalizationType {
    #[default]
    Mean,
    Range,
    Std,
    L2,
}

/// Calculates the `Normalized Root Mean Squared Error`_ (NRMSE) also know as scatter index.
///
/// The metric is defined as:
///
///    .. math::
/// \text{NRMSE} = \frac{\text{RMSE}}{\text{denom}}
///
/// where RMSE is the root mean squared error and `denom` is the normalization factor. The normalization factor can be
/// either be the mean, range, standard deviation or L2 norm of the target, which can be set using `NormalizationType`
///
///
/// ```
/// use rust_metrics::{regression::nrmse::NormalizationType, Metric, NormalizedRootMeanSquaredError};
///
/// let preds = [3.0, 5.0, 2.5, 7.0];
/// let target = [2.5, 5.0, 4.0, 8.0];
///
/// let mut mse = NormalizedRootMeanSquaredError::new(NormalizationType::Mean);
/// mse.update((&preds, &target)).unwrap();
/// assert!((mse.compute().unwrap() - 0.19187986598840726).abs() < f64::EPSILON);
///
/// let mut metric = NormalizedRootMeanSquaredError::new(NormalizationType::Range);
/// metric.update((&preds, &target)).unwrap();
/// let result = metric.compute().unwrap();
/// assert!((result - 0.17007533576245187).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct NormalizedRootMeanSquaredError {
    normalization_type: NormalizationType,
    sum_squared_error: f64,
    total: usize,
    min_val: Option<f64>,
    max_val: Option<f64>,
    target_squared: f64,
    mean_val: f64,
    var_val: f64,
}

impl NormalizedRootMeanSquaredError {
    pub fn new(normalization_type: NormalizationType) -> Self {
        Self {
            normalization_type,
            sum_squared_error: 0.0,
            total: 0,
            min_val: None,
            max_val: None,
            target_squared: 0.0,
            mean_val: 0.0,
            var_val: 0.0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for NormalizedRootMeanSquaredError {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            let error = prediction - target;
            self.sum_squared_error += error * error;
            self.target_squared += target * target;

            self.min_val = Some(self.min_val.map_or(target, |min| min.min(target)));
            self.max_val = Some(self.max_val.map_or(target, |max| max.max(target)));

            self.total += 1;
            let delta = target - self.mean_val;
            self.mean_val += delta / self.total as f64;
            let delta2 = target - self.mean_val;
            self.var_val += delta * delta2;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.total = 0;
        self.min_val = None;
        self.max_val = None;
        self.target_squared = 0.0;
        self.mean_val = 0.0;
        self.var_val = 0.0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        let denom = match self.normalization_type {
            NormalizationType::Mean => self.mean_val,
            NormalizationType::Range => self.max_val? - self.min_val?,
            NormalizationType::Std => (self.var_val / self.total as f64).sqrt(),
            NormalizationType::L2 => self.target_squared.sqrt(),
        };
        let mse = self.sum_squared_error / self.total as f64;
        let rmse = mse.sqrt();
        Some(rmse / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::{NormalizationType, NormalizedRootMeanSquaredError};
    use crate::core::Metric;

    #[test]
    fn nrmse() {
        let mut metric = NormalizedRootMeanSquaredError::default();
        let targets = [2.5, 5.0, 4.0, 8.0];
        let preds = [3.0, 5.0, 2.5, 7.0];
        metric.update((&preds, &targets)).unwrap();
        let result = metric.compute().unwrap();
        assert!((result - 0.19187986598840726).abs() < f64::EPSILON);

        let mut metric = NormalizedRootMeanSquaredError::new(NormalizationType::Range);
        metric.update((&preds, &targets)).unwrap();
        let result = metric.compute().unwrap();
        assert!((result - 0.17007533576245187).abs() < f64::EPSILON);

        metric.reset();
        assert_eq!(metric.compute(), None);
    }
}
