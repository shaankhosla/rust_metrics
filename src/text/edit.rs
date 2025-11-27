use crate::core::{Metric, MetricError};
use crate::utils::levenshtein_distance;

/// Reduction strategy used by [`EditDistance`].
#[derive(Debug, Clone, PartialEq)]
pub enum Reduction {
    /// Return the sum of edit distances in the buffer.
    Sum,
    /// Return the mean edit distance over observed pairs.
    Mean,
}

/// Streaming Levenshtein distance.
///
/// ```
/// use rust_metrics::{EditDistance, Metric};
///
/// let preds = ["kitten"];
/// let targets = ["sitting"];
/// let mut edit = EditDistance::default();
/// edit.update((&preds, &targets)).unwrap();
/// assert_eq!(edit.compute(), Some(3.0));
/// ```
#[derive(Debug, Clone)]
pub struct EditDistance {
    reduction: Reduction,
    edit_scores: Vec<f64>,
    total: usize,
}

impl Default for EditDistance {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

impl EditDistance {
    pub fn new(reduction: Reduction) -> Self {
        Self {
            reduction,
            edit_scores: vec![],
            total: 0,
        }
    }
}

impl Metric<(&[&str], &[&str])> for EditDistance {
    type Output = f64;

    fn update(&mut self, (predictions, targets): (&[&str], &[&str])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        self.total += predictions.len();
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            let edit_distance = levenshtein_distance(prediction, target) as f64;
            self.edit_scores.push(edit_distance);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.total = 0;
        self.edit_scores.clear();
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }
        if self.reduction == Reduction::Sum {
            return Some(self.edit_scores.iter().sum::<f64>());
        }
        Some(self.edit_scores.iter().sum::<f64>() / self.total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::EditDistance;
    use crate::core::Metric;

    #[test]
    fn edit_over_batches() {
        let mut edit_distance = EditDistance::default();

        let preds = vec!["the cat is on the mat"];
        let targets = vec!["the cat is on the mat"];

        edit_distance.update((&preds, &targets)).unwrap();
        let score = edit_distance.compute().unwrap();
        assert_eq!(score, 0.0);

        edit_distance.reset();
        let score = edit_distance.compute();
        assert_eq!(score, None);

        let mut edit_distance = EditDistance::default();
        let preds = vec!["the cat is on the bath"];
        let targets = vec!["the cat is on the mat"];
        edit_distance.update((&preds, &targets)).unwrap();
        let score = edit_distance.compute().unwrap();
        assert_eq!(score, 2.0);

        let preds = vec!["the cat is on the mat"];
        let targets = vec!["the cat is on the mat"];
        edit_distance.update((&preds, &targets)).unwrap();
        let score = edit_distance.compute().unwrap();
        assert_eq!(score, 1.0);
    }
}
