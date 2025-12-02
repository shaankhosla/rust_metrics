use crate::core::{Metric, MetricError};
use crate::utils::AverageMethod;

use super::stat_scores::{BinaryStatScores, MulticlassStatScores};

/// Calculate the Jaccard index for binary tasks.
/// The `Jaccard index`_ (also known as the intersection over union or jaccard similarity coefficient) is an statistic
/// that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
/// intersection divided by the union of the sample sets:
///
/// math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}
///
///
/// ```
/// use rust_metrics::{BinaryJaccardIndex, Metric};
///
/// let preds = [0.35, 0.85, 0.48, 0.01];
/// let target = [1, 1, 0, 0];
///
/// let mut metric = BinaryJaccardIndex::default();
/// metric.update((&preds, &target)).unwrap();
/// assert!((metric.compute().unwrap() - 0.50).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BinaryJaccardIndex {
    stat_scores: BinaryStatScores,
}

impl BinaryJaccardIndex {
    pub fn new(threshold: f64) -> Self {
        let stat_scores = BinaryStatScores::new(threshold);
        Self { stat_scores }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryJaccardIndex {
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
        let denom = self.stat_scores.true_positive
            + self.stat_scores.false_positive
            + self.stat_scores.false_negative;
        if denom == 0 {
            return None;
        }
        Some(self.stat_scores.true_positive as f64 / denom as f64)
    }
}

/// Calculate the Jaccard index for multiclass tasks.
/// The `Jaccard index`_ (also known as the intersection over union or jaccard similarity coefficient) is an statistic
/// that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
/// intersection divided by the union of the sample sets:
///
/// .. math:: J(A,B) = \frac{|A\cap B|}{|A\cup B|}
///
///
/// ```
/// use rust_metrics::{MulticlassJaccardIndex, Metric};
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
/// let mut metric = MulticlassJaccardIndex::new(3, AverageMethod::Macro);
/// metric.update((&preds, &targets)).unwrap();
/// assert!((metric.compute().unwrap() - (2.0/3.0)).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct MulticlassJaccardIndex {
    stat_scores: MulticlassStatScores,
    average_method: AverageMethod,
}

impl MulticlassJaccardIndex {
    pub fn new(num_classes: usize, average_method: AverageMethod) -> Self {
        let stat_scores = MulticlassStatScores::new(num_classes);

        Self {
            stat_scores,
            average_method,
        }
    }
}

impl Metric<(&[&[f64]], &[usize])> for MulticlassJaccardIndex {
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
                let fp_sum: usize = self.stat_scores.false_positive.iter().sum();
                let fn_sum: usize = self.stat_scores.false_negative.iter().sum();
                let denom = tp_sum + fp_sum + fn_sum;
                Some(tp_sum as f64 / denom as f64)
            }

            AverageMethod::Macro => {
                let mut jaccard_sum = 0.0;
                for class in 0..num_classes {
                    let denom = self.stat_scores.true_positive[class]
                        + self.stat_scores.false_positive[class]
                        + self.stat_scores.false_negative[class];
                    let class_jaccard = self.stat_scores.true_positive[class] as f64 / denom as f64;
                    jaccard_sum += class_jaccard;
                }
                Some(jaccard_sum / num_classes as f64)
            }

            AverageMethod::Weighted => {
                let mut weighted_sum = 0.0;
                let mut total_support = 0usize;
                for class in 0..num_classes {
                    let support = self.stat_scores.true_positive[class]
                        + self.stat_scores.false_negative[class];

                    let denom = self.stat_scores.true_positive[class]
                        + self.stat_scores.false_positive[class]
                        + self.stat_scores.false_negative[class];
                    let class_jaccard = self.stat_scores.true_positive[class] as f64 / denom as f64;
                    weighted_sum += class_jaccard * support as f64;
                    total_support += support;
                }
                Some(weighted_sum / total_support as f64)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BinaryJaccardIndex;
    use super::MulticlassJaccardIndex;
    use crate::core::Metric;

    #[test]
    fn binary_accuracy() {
        let mut metric = BinaryJaccardIndex::default();
        let preds = [0.35, 0.85, 0.48, 0.01];
        let targets = [1, 1, 0, 0];
        metric.update((&preds, &targets)).unwrap();
        let result = metric.compute().unwrap();
        assert!((result - 0.50).abs() < f64::EPSILON);

        metric.reset();
        assert_eq!(metric.compute(), None);
    }

    #[test]
    fn multiclass_accuracy() {
        let mut metric = MulticlassJaccardIndex::new(3, super::AverageMethod::Macro);
        let targets = [2, 1, 0, 0];
        let preds: [&[f64]; 4] = [
            &[0.16, 0.26, 0.58][..],
            &[0.22, 0.61, 0.17][..],
            &[0.71, 0.09, 0.20][..],
            &[0.05, 0.82, 0.13][..],
        ];
        metric.update((&preds, &targets)).unwrap();
        let result = metric.compute().unwrap();
        assert!((result - (2.0 / 3.0)).abs() < f64::EPSILON);

        metric.reset();
        assert_eq!(metric.compute(), None);
    }
}
