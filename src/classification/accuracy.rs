use crate::core::{Metric, MetricError};
use crate::utils::AverageMethod;

use super::stat_scores::{BinaryStatScores, MulticlassStatScores};

/// Binary accuracy over thresholded probabilities.
///
/// ```
/// use rust_metrics::{BinaryAccuracy, Metric};
///
/// let target = [0_usize, 1, 0, 1, 0, 1];
/// let preds = [0.11, 0.22, 0.84, 0.73, 0.33, 0.92];
///
/// let mut metric = BinaryAccuracy::default();
/// metric.update((&preds, &target)).unwrap();
/// assert!((metric.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryAccuracy {
    stat_scores: BinaryStatScores,
}

impl BinaryAccuracy {
    pub fn new(threshold: f64) -> Self {
        let stat_scores = BinaryStatScores::new(threshold);
        Self { stat_scores }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryAccuracy {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[usize])) -> Result<(), MetricError> {
        self.stat_scores.update((predictions, targets))?;
        Ok(())
    }

    fn reset(&mut self) {
        self.stat_scores.reset();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.stat_scores.total == 0 {
            return None;
        }
        let correct = self.stat_scores.true_positive + self.stat_scores.true_negative;
        let total = self.stat_scores.total;
        Some(correct as f64 / total as f64)
    }
}

/// Macro/micro accuracy for multi-class classification.
/// # Example
///
/// ```
/// use rust_metrics::{Metric, MulticlassAccuracy};
/// use rust_metrics::utils::AverageMethod;
///
/// let targets = [2, 1, 0, 0];
/// let preds: [&[f64]; 4] = [
///     &[0.16, 0.26, 0.58],
///     &[0.22, 0.61, 0.17],
///     &[0.71, 0.09, 0.20],
///     &[0.05, 0.82, 0.13],
/// ];
///
/// let mut metric = MulticlassAccuracy::new(3, AverageMethod::Macro);
/// metric.update((&preds, &targets)).unwrap();
/// assert!((metric.compute().unwrap() - 0.8333333333333334).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone)]
pub struct MulticlassAccuracy {
    stat_scores: MulticlassStatScores,
    average_method: AverageMethod,
}

impl MulticlassAccuracy {
    pub fn new(num_classes: usize, average_method: AverageMethod) -> Self {
        let stat_scores = MulticlassStatScores::new(num_classes);

        Self {
            stat_scores,
            average_method,
        }
    }
}

impl Metric<(&[&[f64]], &[usize])> for MulticlassAccuracy {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[&[f64]], &[usize])) -> Result<(), MetricError> {
        self.stat_scores.update((predictions, targets))?;

        Ok(())
    }

    fn reset(&mut self) {
        self.stat_scores.reset();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.stat_scores.total == 0 {
            return None;
        }
        let num_classes = self.stat_scores.num_classes;

        match self.average_method {
            AverageMethod::Micro => {
                let tp_sum: usize = self.stat_scores.true_positive.iter().sum();
                let tn_sum: usize = self.stat_scores.true_negative.iter().sum();
                Some((tp_sum + tn_sum) as f64 / self.stat_scores.total as f64)
            }

            AverageMethod::Macro => {
                let mut accuracies = 0.0;
                for class in 0..num_classes {
                    let accuracy = (self.stat_scores.true_positive[class] as f64
                        + self.stat_scores.true_negative[class] as f64)
                        / self.stat_scores.total_per_class[class] as f64;
                    accuracies += accuracy;
                }
                Some(accuracies / num_classes as f64)
            }

            AverageMethod::Weighted => {
                let mut weighted_sum = 0.0;
                let mut total_support = 0usize;
                for k in 0..num_classes {
                    let support =
                        self.stat_scores.true_positive[k] + self.stat_scores.false_negative[k];

                    let accuracy = self.stat_scores.true_positive[k] as f64
                        + self.stat_scores.true_negative[k] as f64
                            / self.stat_scores.total_per_class[k] as f64;
                    weighted_sum += accuracy * support as f64;
                    total_support += support;
                }
                Some(weighted_sum / total_support as f64)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BinaryAccuracy;
    use super::MulticlassAccuracy;
    use crate::core::Metric;

    #[test]
    fn binary_accuracy() {
        let mut metric = BinaryAccuracy::default();
        let targets = [0, 1, 0, 1, 0, 1];
        let preds = [0.11, 0.22, 0.84, 0.73, 0.33, 0.92];
        metric.update((&preds, &targets)).unwrap();
        let result = metric.compute().unwrap();
        assert!((result - (2.0 / 3.0)).abs() < f64::EPSILON);

        metric.reset();
        assert_eq!(metric.compute(), None);
    }

    #[test]
    fn multiclass_accuracy() {
        let mut metric = MulticlassAccuracy::new(3, super::AverageMethod::Macro);
        let targets = [2, 1, 0, 0];
        let preds: [&[f64]; 4] = [
            &[0.16, 0.26, 0.58][..],
            &[0.22, 0.61, 0.17][..],
            &[0.71, 0.09, 0.20][..],
            &[0.05, 0.82, 0.13][..],
        ];
        metric.update((&preds, &targets)).unwrap();
        let result = metric.compute().unwrap();
        assert!((result - 0.8333333333333334).abs() < f64::EPSILON);

        metric.reset();
        assert_eq!(metric.compute(), None);
    }
}
