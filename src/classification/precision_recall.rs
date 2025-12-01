use crate::core::{Metric, MetricError};
use crate::utils::AverageMethod;

use super::stat_scores::{BinaryStatScores, MulticlassStatScores};

/// Thresholded precision for binary classification probabilities.
///
/// ```
/// use rust_metrics::{BinaryPrecision, Metric};
///
/// let target = [0_usize, 1, 0, 1, 0, 1];
/// let preds = [0.11, 0.22, 0.84, 0.73, 0.33, 0.92];
///
/// let mut precision = BinaryPrecision::default();
/// precision.update((&preds, &target)).unwrap();
/// assert!((precision.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryPrecision {
    stat_scores: BinaryStatScores,
}

impl BinaryPrecision {
    pub fn new(threshold: f64) -> Self {
        let stat_scores = BinaryStatScores::new(threshold);
        Self { stat_scores }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryPrecision {
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
        Some(
            self.stat_scores.true_positive as f64
                / (self.stat_scores.true_positive + self.stat_scores.false_positive) as f64,
        )
    }
}

#[derive(Debug, Clone)]
pub struct MulticlassPrecision {
    stat_scores: MulticlassStatScores,
    average_method: AverageMethod,
}

/// Macro/micro precision for multi-class classification.
///
///
/// ```
/// use rust_metrics::{Metric, MulticlassPrecision};
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
/// let mut metric = MulticlassPrecision::new(3, AverageMethod::Macro);
/// metric.update((&preds, &targets)).unwrap();
/// assert!((metric.compute().unwrap() - 0.8333333333333334).abs() < f64::EPSILON);
/// ```
impl MulticlassPrecision {
    pub fn new(num_classes: usize, average_method: AverageMethod) -> Self {
        let stat_scores = MulticlassStatScores::new(num_classes);
        Self {
            stat_scores,
            average_method,
        }
    }
}

impl Metric<(&[&[f64]], &[usize])> for MulticlassPrecision {
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
        let tp = &self.stat_scores.true_positive;
        let fp = &self.stat_scores.false_positive;
        let total_per_class = &self.stat_scores.total_per_class;

        match self.average_method {
            AverageMethod::Micro => {
                let total_tp: usize = tp.iter().sum();
                let total_fp: usize = fp.iter().sum();

                if total_tp + total_fp == 0 {
                    return None;
                }
                Some(total_tp as f64 / (total_tp + total_fp) as f64)
            }

            AverageMethod::Macro => {
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..num_classes {
                    let denom = tp[i] + fp[i];
                    if denom > 0 {
                        sum += tp[i] as f64 / denom as f64;
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
                let mut numerator = 0.0;
                let mut denom_total = 0.0;
                for i in 0..num_classes {
                    let denom = tp[i] + fp[i];
                    if denom > 0 {
                        let support = total_per_class[i] as f64;
                        numerator += support * (tp[i] as f64 / denom as f64);
                        denom_total += support;
                    }
                }
                if denom_total == 0.0 {
                    None
                } else {
                    Some(numerator / denom_total)
                }
            }
        }
    }
}

/// Binary recall (`TP / (TP + FN)`) over thresholded probabilities.
///
/// ```
/// use rust_metrics::{BinaryRecall, Metric};
///
/// let target = [0_usize, 1, 0, 1, 0, 1];
/// let preds = [0.11, 0.22, 0.84, 0.73, 0.33, 0.92];
///
/// let mut recall = BinaryRecall::default();
/// recall.update((&preds, &target)).unwrap();
/// assert!((recall.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryRecall {
    stat_scores: BinaryStatScores,
}

impl BinaryRecall {
    pub fn new(threshold: f64) -> Self {
        let stat_scores = BinaryStatScores::new(threshold);
        Self { stat_scores }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryRecall {
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
        Some(
            self.stat_scores.true_positive as f64
                / (self.stat_scores.true_positive + self.stat_scores.false_negative) as f64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryPrecision, BinaryRecall, MulticlassPrecision};
    use crate::core::{Metric, MetricError};

    #[test]
    fn binary_precision_computes_over_batches() {
        let mut precision = BinaryPrecision::default();

        precision
            .update((&[0.11, 0.22, 0.84], &[0_usize, 1, 0]))
            .expect("update should succeed");
        precision
            .update((&[0.73, 0.33, 0.92], &[1_usize, 0, 1]))
            .expect("update should succeed");
        assert!((precision.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);

        precision.reset();
        assert_eq!(precision.compute(), None);
    }

    #[test]
    fn binary_precision_validates_targets() {
        let mut precision = BinaryPrecision::default();
        let err = precision
            .update((&[0.8], &[2_usize]))
            .expect_err("invalid targets should fail");
        match err {
            MetricError::IncompatibleInput { .. } => {} // OK: variant matches
            other => panic!("Expected IncompatibleInput error, got: {:?}", other),
        }
    }
    #[test]
    fn mutliclass_precision() {
        let mut metric = MulticlassPrecision::new(3, super::AverageMethod::Macro);
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

    #[test]
    fn binary_recall_computes_over_batches() {
        let mut recall = BinaryRecall::default();

        recall
            .update((&[0.11, 0.22, 0.84], &[0_usize, 1, 0]))
            .expect("update should succeed");
        recall
            .update((&[0.73, 0.33, 0.92], &[1_usize, 0, 1]))
            .expect("update should succeed");
        assert!((recall.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);

        recall.reset();
        assert_eq!(recall.compute(), None);
    }

    #[test]
    fn binary_recall_validates_targets() {
        let mut recall = BinaryRecall::default();
        let err = recall
            .update((&[0.8], &[2_usize]))
            .expect_err("invalid targets should fail");
        match err {
            MetricError::IncompatibleInput { .. } => {} // OK: variant matches
            other => panic!("Expected IncompatibleInput error, got: {:?}", other),
        }
    }
}
