use crate::core::{Metric, MetricError};
use std::collections::HashMap;

/// Mutual information between predicted and target cluster assignments.
///
/// ```
/// use rust_metrics::{Metric, MutualInfoScore};
///
/// let preds = [2, 1, 0, 1, 0];
/// let target = [0, 2, 1, 1, 0];
///
/// let mut metric = MutualInfoScore::default();
/// metric.update((&preds, &target)).unwrap();
/// assert!((metric.compute().unwrap() - 0.500402423538188).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MutualInfoScore {
    preds: Vec<usize>,
    targets: Vec<usize>,
}

impl MutualInfoScore {
    pub fn new() -> Self {
        Self {
            preds: Vec::new(),
            targets: Vec::new(),
        }
    }
}

impl Metric<(&[usize], &[usize])> for MutualInfoScore {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[usize], &[usize])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        self.preds.extend(predictions);
        self.targets.extend(targets);

        Ok(())
    }

    fn reset(&mut self) {
        self.preds.clear();
        self.targets.clear();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.preds.is_empty() {
            return None;
        }
        let total = self.preds.len() as f64;

        let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for (&target, &pred) in self.targets.iter().zip(self.preds.iter()) {
            *joint_counts.entry((target, pred)).or_insert(0) += 1;
        }

        let mut target_counts: HashMap<usize, usize> = HashMap::new();
        let mut pred_counts: HashMap<usize, usize> = HashMap::new();
        for &t in self.targets.iter() {
            *target_counts.entry(t).or_insert(0) += 1;
        }
        for &p in self.preds.iter() {
            *pred_counts.entry(p).or_insert(0) += 1;
        }

        let mut mi = 0.0;
        for ((target, pred), &count) in joint_counts.iter() {
            let count = count as f64;
            let target_count = *target_counts.get(target)? as f64;
            let pred_count = *pred_counts.get(pred)? as f64;

            let term = (count / total) * ((total * count) / (target_count * pred_count)).ln();
            mi += term;
        }

        Some(mi)
    }
}

#[cfg(test)]
mod tests {
    use super::{Metric, MutualInfoScore};

    #[test]
    fn mutual_info() {
        let mut metric = MutualInfoScore::default();

        let preds = [2, 1, 0, 1, 0];
        let target = [0, 2, 1, 1, 0];
        metric.update((&preds, &target)).unwrap();
        dbg!(metric.compute());
        assert!((metric.compute().unwrap() - 0.500402423538188).abs() < f64::EPSILON);
    }
}
