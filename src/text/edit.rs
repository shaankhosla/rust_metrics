use crate::core::{Metric, MetricError};
use crate::utils::{levenshtein_distance, MetricAggregator, Reduction};

/// Streaming Levenshtein distance.
///
/// ```
/// use rust_metrics::{EditDistance, Metric};
///
/// let preds = ["rain"];
/// let targets = ["shine"];
/// let mut edit = EditDistance::default();
/// edit.update((&preds, &targets)).unwrap();
/// assert_eq!(edit.compute(), Some(3.0));
/// ```
#[derive(Debug, Clone)]
pub struct EditDistance {
    metric_aggregator: MetricAggregator,
}

impl Default for EditDistance {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

impl EditDistance {
    pub fn new(reduction: Reduction) -> Self {
        Self {
            metric_aggregator: MetricAggregator::new(reduction),
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
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            let edit_distance = levenshtein_distance(prediction, target) as f64;
            self.metric_aggregator.update(edit_distance);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.metric_aggregator.reset();
    }

    fn compute(&self) -> Option<Self::Output> {
        self.metric_aggregator.compute()
    }
}

#[cfg(test)]
mod tests {
    use super::EditDistance;
    use crate::core::Metric;

    #[test]
    fn edit_over_batches() {
        let mut edit_distance = EditDistance::default();

        let preds = vec!["rain"];
        let targets = vec!["shine"];

        edit_distance.update((&preds, &targets)).unwrap();
        let score = edit_distance.compute().unwrap();
        assert_eq!(score, 3.0);

        edit_distance.reset();
        let score = edit_distance.compute();
        assert_eq!(score, None);

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
