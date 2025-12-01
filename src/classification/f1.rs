use crate::core::{Metric, MetricError};
use crate::utils::AverageMethod;

use super::stat_scores::{BinaryStatScores, MulticlassStatScores};

/// Binary F1 Score, computed as the harmonic mean of precision and recall.
///
///
/// ```
/// use rust_metrics::{BinaryF1Score, Metric};
///
/// let target = [0_usize, 1, 0, 1, 0, 1];
/// let preds = [0.11, 0.22, 0.84, 0.73, 0.33, 0.92];
///
/// let mut f1 = BinaryF1Score::default();
/// f1.update((&preds, &target)).unwrap();
/// assert!((f1.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryF1Score {
    stat_scores: BinaryStatScores,
}

impl BinaryF1Score {
    pub fn new(threshold: f64) -> Self {
        let stat_scores = BinaryStatScores::new(threshold);
        Self { stat_scores }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryF1Score {
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
        let precision = self.stat_scores.true_positive as f64
            / (self.stat_scores.true_positive + self.stat_scores.false_positive) as f64;
        let recall = self.stat_scores.true_positive as f64
            / (self.stat_scores.true_positive + self.stat_scores.false_negative) as f64;
        Some(2.0 * precision * recall / (precision + recall))
    }
}

/// Compute F-1 score for multiclass tasks.
///
///
/// ```
/// use rust_metrics::{MulticlassF1Score, Metric};
/// use rust_metrics::utils::AverageMethod;
///
/// let mut metric = MulticlassF1Score::new(3, AverageMethod::Macro);
/// let target = [2, 1, 0, 0];
/// let preds: [&[f64]; 4] = [
/// &[0.16, 0.26, 0.58][..],
/// &[0.22, 0.61, 0.17][..],
/// &[0.71, 0.09, 0.20][..],
/// &[0.05, 0.82, 0.13][..],
/// ];
///
/// metric.update((&preds, &target)).unwrap();
/// let result = metric.compute().unwrap();
/// assert!((result - 0.7777777777777777).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone)]
pub struct MulticlassF1Score {
    stat_scores: MulticlassStatScores,
    average_method: AverageMethod,
}

impl MulticlassF1Score {
    pub fn new(num_classes: usize, average_method: AverageMethod) -> Self {
        let stat_scores = MulticlassStatScores::new(num_classes);
        Self {
            stat_scores,
            average_method,
        }
    }
}

impl Metric<(&[&[f64]], &[usize])> for MulticlassF1Score {
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

        let tp = &self.stat_scores.true_positive;
        let fp = &self.stat_scores.false_positive;
        let fn_counts = &self.stat_scores.false_negative;
        let num_classes = self.stat_scores.num_classes;

        match self.average_method {
            AverageMethod::Micro => {
                let total_tp: usize = tp.iter().sum();
                let total_fp: usize = fp.iter().sum();
                let total_fn: usize = fn_counts.iter().sum();
                let denom = 2 * total_tp + total_fp + total_fn;

                if denom == 0 {
                    None
                } else {
                    Some(2.0 * total_tp as f64 / denom as f64)
                }
            }

            AverageMethod::Macro => {
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..num_classes {
                    let denom = 2 * tp[i] + fp[i] + fn_counts[i];
                    if denom > 0 {
                        sum += 2.0 * tp[i] as f64 / denom as f64;
                        count += 1;
                    }
                }

                if count == 0 {
                    None
                } else {
                    Some(sum / count as f64)
                }
            }

            AverageMethod::Weighted => {
                let mut weighted_sum = 0.0;
                let mut support_sum = 0usize;
                for i in 0..num_classes {
                    let denom = 2 * tp[i] + fp[i] + fn_counts[i];
                    let support = tp[i] + fn_counts[i];
                    if denom > 0 && support > 0 {
                        let f1_i = 2.0 * tp[i] as f64 / denom as f64;
                        weighted_sum += f1_i * support as f64;
                        support_sum += support;
                    }
                }

                if support_sum == 0 {
                    None
                } else {
                    Some(weighted_sum / support_sum as f64)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AverageMethod, BinaryF1Score, MulticlassF1Score};
    use crate::core::Metric;

    #[test]
    fn f1_computes_over_batches() {
        let mut f1 = BinaryF1Score::default();

        f1.update((&[0.11, 0.22, 0.84], &[0_usize, 1, 0]))
            .expect("update should succeed");
        f1.update((&[0.73, 0.33, 0.92], &[1_usize, 0, 1]))
            .expect("update should succeed");
        assert!((f1.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);

        f1.reset();
        assert_eq!(f1.compute(), None);
    }

    #[test]
    fn f1_multiclass() {
        let mut metric = MulticlassF1Score::new(3, AverageMethod::Macro);
        let target = [2, 1, 0, 0];
        let preds: [&[f64]; 4] = [
            &[0.16, 0.26, 0.58][..],
            &[0.22, 0.61, 0.17][..],
            &[0.71, 0.09, 0.20][..],
            &[0.05, 0.82, 0.13][..],
        ];

        metric.update((&preds, &target)).unwrap();
        let result = metric.compute().unwrap();
        assert!((result - 0.7777777777777777).abs() < f64::EPSILON);

        metric.reset();
        assert_eq!(metric.compute(), None);
    }
}
