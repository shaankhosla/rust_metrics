use std::cmp::Ordering;

use crate::core::{Metric, MetricError};
use crate::utils::{verify_binary_label, verify_range};

#[derive(Debug, Clone)]
enum BinaryAurocMode {
    Exact {
        samples: Vec<(f64, bool)>,
    },
    Binned {
        bins: usize,
        pos_hist: Vec<u64>,
        neg_hist: Vec<u64>,
    },
}

/// ROC AUC for binary classification with exact or histogrammed accumulation.
///
/// ```
/// use rust_metrics::{BinaryAuroc, Metric};
///
/// let mut auroc = BinaryAuroc::new(0); // 0 => exact mode
/// auroc.update((&[0.9, 0.2], &[1.0, 0.0])).unwrap();
/// assert!(auroc.compute().unwrap() > 0.9);
/// ```
#[derive(Debug, Clone)]
pub struct BinaryAuroc {
    mode: BinaryAurocMode,
}

impl Default for BinaryAuroc {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl BinaryAuroc {
    pub fn new(bins: usize) -> Self {
        let mode = match bins {
            0 => BinaryAurocMode::Exact {
                samples: Vec::new(),
            },
            1 => panic!("bins must be 0 (exact) or greater than 1 (binned)"),
            _ => BinaryAurocMode::Binned {
                bins,
                pos_hist: vec![0; bins],
                neg_hist: vec![0; bins],
            },
        };

        Self { mode }
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

        match &mut self.mode {
            BinaryAurocMode::Exact { samples } => {
                for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
                    verify_range(prediction, 0.0, 1.0)?;
                    verify_binary_label(target)?;
                    let target_bool = target == 1.0;
                    samples.push((prediction, target_bool));
                }
                Ok(())
            }
            BinaryAurocMode::Binned {
                bins,
                pos_hist,
                neg_hist,
            } => {
                let max_bin_idx = (*bins - 1) as f64;
                for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
                    verify_range(prediction, 0.0, 1.0)?;
                    verify_binary_label(target)?;
                    let bin_index = ((prediction * max_bin_idx).round()) as usize;
                    if target == 1.0 {
                        pos_hist[bin_index] += 1;
                    } else {
                        neg_hist[bin_index] += 1;
                    }
                }
                Ok(())
            }
        }
    }

    fn reset(&mut self) {
        match &mut self.mode {
            BinaryAurocMode::Exact { samples } => samples.clear(),
            BinaryAurocMode::Binned {
                pos_hist, neg_hist, ..
            } => {
                for value in pos_hist.iter_mut() {
                    *value = 0;
                }
                for value in neg_hist.iter_mut() {
                    *value = 0;
                }
            }
        }
    }

    fn compute(&self) -> Option<Self::Output> {
        match &self.mode {
            BinaryAurocMode::Exact { samples } => {
                if samples.is_empty() {
                    return None;
                }

                let mut sorted = samples.to_vec();
                sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

                let total_pos = sorted.iter().filter(|(_, t)| *t).count() as f64;
                let total_neg = sorted.len() as f64 - total_pos;

                if total_pos == 0.0 || total_neg == 0.0 {
                    return None;
                }

                let mut tp = 0.0;
                let mut fp = 0.0;
                let mut auc = 0.0;
                let mut idx = 0;

                while idx < sorted.len() {
                    let current_score = sorted[idx].0;
                    let prev_tp = tp;
                    let prev_fp = fp;

                    let mut group_pos = 0.0;
                    let mut group_neg = 0.0;

                    while idx < sorted.len() && sorted[idx].0 == current_score {
                        if sorted[idx].1 {
                            group_pos += 1.0;
                        } else {
                            group_neg += 1.0;
                        }
                        idx += 1;
                    }

                    tp += group_pos;
                    fp += group_neg;
                    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
                }

                Some(auc / (total_pos * total_neg))
            }
            BinaryAurocMode::Binned {
                pos_hist, neg_hist, ..
            } => {
                let mut tp = 0.0;
                let mut fp = 0.0;
                let total_pos: f64 = pos_hist.iter().sum::<u64>() as f64;
                let total_neg: f64 = neg_hist.iter().sum::<u64>() as f64;
                if total_pos == 0.0 && total_neg == 0.0 {
                    return None;
                }
                let mut auc = 0.0;

                for (p, n) in pos_hist.iter().zip(neg_hist.iter()).rev() {
                    let prev_tp = tp;
                    let prev_fp = fp;
                    tp += *p as f64;
                    fp += *n as f64;
                    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
                }

                Some(auc / (total_pos * total_neg))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BinaryAuroc;
    use crate::core::Metric;

    #[test]
    fn binary_auroc_binned() {
        let mut auc = BinaryAuroc::new(100);
        let _ = auc.update((&[0.9, 0.8, 0.7, 0.4, 0.2], &[1.0, 1.0, 0.0, 0.0, 1.0]));
        assert!((auc.compute().unwrap() - (2.0 / 3.0)).abs() < f64::EPSILON);

        auc.reset();
        assert_eq!(auc.compute(), None);
    }

    #[test]
    fn binary_auroc_exact() {
        let mut auc = BinaryAuroc::new(0);
        let _ = auc.update((&[0.9, 0.8, 0.7, 0.4, 0.2], &[1.0, 1.0, 0.0, 0.0, 1.0]));
        let _ = auc.update((&[0.9, 0.8, 0.7, 0.4, 0.2], &[1.0, 1.0, 0.0, 0.0, 1.0]));
        assert!((auc.compute().unwrap() - (2.0 / 3.0)).abs() < f64::EPSILON);
    }
}
