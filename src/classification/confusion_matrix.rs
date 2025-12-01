use crate::core::{Metric, MetricError};

use super::stat_scores::BinaryStatScores;

#[derive(Debug, Clone, Default)]
pub struct BinaryConfusionMatrix {
    pub stat_scores: BinaryStatScores,
}

impl BinaryConfusionMatrix {
    pub fn new(threshold: f64) -> Self {
        let stat_scores = BinaryStatScores::new(threshold);
        Self { stat_scores }
    }
}

impl Metric<(&[f64], &[usize])> for BinaryConfusionMatrix {
    type Output = [[usize; 2]; 2];

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
        let confusion_matrix = [
            [
                self.stat_scores.true_positive,
                self.stat_scores.false_positive,
            ],
            [
                self.stat_scores.false_negative,
                self.stat_scores.true_negative,
            ],
        ];
        Some(confusion_matrix)
    }
}
