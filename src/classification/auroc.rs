use crate::core::{Metric, MetricError};

#[derive(Debug, Default, Clone)]
pub struct BinaryAuroc {
    bins: usize,
    pos_hist: Vec<u64>,
    neg_hist: Vec<u64>,
}

impl BinaryAuroc {
    pub fn new(bins: usize) -> Self {
        assert!(bins >= 1, "bins must be at least 1");
        Self {
            bins,
            pos_hist: vec![0; bins],
            neg_hist: vec![0; bins],
        }
    }
}

impl Metric<(&[f64], &[f64])> for BinaryAuroc {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[f64], &[f64])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        let max_bin_idx = (self.bins - 1) as f64;
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            if !(0.0..=1.0).contains(&prediction) {
                return Err(MetricError::IncompatibleInput {
                    expected: "prediction must be between 0 and 1",
                    got: "other",
                });
            }

            let bin_index = ((prediction * max_bin_idx).round()) as usize;
            if target != 0.0 && target != 1.0 {
                return Err(MetricError::IncompatibleInput {
                    expected: "target must be 0 or 1",
                    got: "other",
                });
            }
            if target == 1.0 {
                self.pos_hist[bin_index] += 1;
            } else {
                self.neg_hist[bin_index] += 1;
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.pos_hist = vec![0; self.bins];
        self.neg_hist = vec![0; self.bins];
    }

    fn compute(&self) -> Self::Output {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let total_pos: f64 = self.pos_hist.iter().sum::<u64>() as f64;
        let total_neg: f64 = self.neg_hist.iter().sum::<u64>() as f64;
        if total_pos == 0.0 && total_neg == 0.0 {
            return 0.0;
        }
        let mut auc = 0.0;

        for (p, n) in self.pos_hist.iter().zip(self.neg_hist.iter()).rev() {
            let prev_tp = tp;
            let prev_fp = fp;
            tp += *p as f64;
            fp += *n as f64;
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
        }

        auc / (total_pos * total_neg)
    }
}

#[cfg(test)]
mod tests {
    use super::BinaryAuroc;
    use crate::core::Metric;

    #[test]
    fn binary_auroc() {
        let mut auc = BinaryAuroc::new(100);
        let _ = auc.update((&[0.9, 0.8, 0.7, 0.4, 0.2], &[1.0, 1.0, 0.0, 0.0, 1.0]));
        assert!((auc.compute() - (2.0 / 3.0)).abs() < f64::EPSILON);

        auc.reset();
        assert_eq!(auc.compute(), 0.0);
    }
}
