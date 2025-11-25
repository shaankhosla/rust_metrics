use crate::core::{Metric, MetricError};
use crate::utils::verify_range;

#[derive(Debug, Clone)]
pub struct BinaryPrecision {
    threshold: f64,
    true_positive_ct: usize,
    false_positive_ct: usize,
}

impl Default for BinaryPrecision {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl BinaryPrecision {
    pub fn new(threshold: f64) -> Self {
        verify_range(threshold, 0.0, 1.0).unwrap();
        Self {
            threshold,
            true_positive_ct: 0,
            false_positive_ct: 0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for BinaryPrecision {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            verify_range(prediction, 0.0, 1.0)?;
            if prediction > self.threshold {
                if target == 1.0 {
                    self.true_positive_ct += 1;
                }
                self.false_positive_ct += 1;
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.true_positive_ct = 0;
        self.false_positive_ct = 0;
    }

    fn compute(&self) -> Self::Output {
        if self.true_positive_ct + self.false_positive_ct == 0 {
            return 0.0;
        }
        self.true_positive_ct as f64 / (self.true_positive_ct + self.false_positive_ct) as f64
    }
}

#[derive(Debug, Clone)]
pub struct BinaryRecall {
    threshold: f64,
    true_positive_ct: usize,
    false_negative_ct: usize,
}

impl Default for BinaryRecall {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl BinaryRecall {
    pub fn new(threshold: f64) -> Self {
        verify_range(threshold, 0.0, 1.0).unwrap();
        Self {
            threshold,
            true_positive_ct: 0,
            false_negative_ct: 0,
        }
    }
}

impl Metric<(&[f64], &[f64])> for BinaryRecall {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            verify_range(prediction, 0.0, 1.0)?;
            if target == 1.0 {
                if prediction > self.threshold {
                    self.true_positive_ct += 1;
                }
                self.false_negative_ct += 1;
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.true_positive_ct = 0;
        self.false_negative_ct = 0;
    }

    fn compute(&self) -> Self::Output {
        if self.true_positive_ct + self.false_negative_ct == 0 {
            return 0.0;
        }
        self.true_positive_ct as f64 / (self.true_positive_ct + self.false_negative_ct) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryPrecision, BinaryRecall};
    use crate::core::Metric;

    #[test]
    fn binary_precision_computes_over_batches() {
        let mut precision = BinaryPrecision::default();

        precision
            .update((&[0.8, 0.6, 0.3, 0.1], &[1.0, -1.0, 1.0, -1.0]))
            .expect("update should succeed");
        precision
            .update((&[0.7], &[1.0]))
            .expect("update should succeed");
        assert_eq!(precision.compute(), 0.4);

        precision.reset();
        assert_eq!(precision.compute(), 0.0);
    }

    #[test]
    fn binary_recall_computes_over_batches() {
        let mut recall = BinaryRecall::default();

        recall
            .update((&[0.8, 0.6, 0.3, 0.1], &[1.0, -1.0, 1.0, -1.0]))
            .expect("update should succeed");
        recall
            .update((&[0.7], &[1.0]))
            .expect("update should succeed");
        assert_eq!(recall.compute(), 0.4);

        recall.reset();
        assert_eq!(recall.compute(), 0.0);
    }
}
