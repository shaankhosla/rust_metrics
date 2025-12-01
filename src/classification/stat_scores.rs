use crate::core::MetricError;
use crate::utils::{verify_binary_label, verify_label, verify_range};

#[derive(Debug, Clone)]
pub struct BinaryStatScores {
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    pub true_negative: usize,
    pub total: usize,
    threshold: f64,
}
impl Default for BinaryStatScores {
    fn default() -> Self {
        Self::new(0.5)
    }
}
impl BinaryStatScores {
    pub fn new(threshold: f64) -> Self {
        verify_range(threshold, 0.0, 1.0).unwrap();
        Self {
            true_positive: 0,
            false_positive: 0,
            false_negative: 0,
            true_negative: 0,
            total: 0,
            threshold,
        }
    }

    pub fn update(
        &mut self,
        (predictions, targets): (&[f64], &[usize]),
    ) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            verify_range(prediction, 0.0, 1.0)?;
            verify_binary_label(target)?;

            let prediction: bool = prediction > self.threshold;
            let actual: bool = target == 1;

            match (prediction, actual) {
                (true, true) => self.true_positive += 1,
                (true, false) => self.false_positive += 1,
                (false, true) => self.false_negative += 1,
                (false, false) => self.true_negative += 1,
            }

            self.total += 1;
        }
        Ok(())
    }
    pub fn reset(&mut self) {
        self.true_positive = 0;
        self.false_positive = 0;
        self.false_negative = 0;
        self.true_negative = 0;
        self.total = 0;
    }
}

#[derive(Debug, Clone)]
pub struct MulticlassStatScores {
    pub true_positive: Vec<usize>,
    pub false_positive: Vec<usize>,
    pub false_negative: Vec<usize>,
    pub true_negative: Vec<usize>,
    pub total_per_class: Vec<usize>,
    pub total: usize,
    pub num_classes: usize,
}
impl MulticlassStatScores {
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes >= 2, "num_classes must be at least 2");
        Self {
            true_positive: vec![0; num_classes],
            false_positive: vec![0; num_classes],
            false_negative: vec![0; num_classes],
            true_negative: vec![0; num_classes],
            total_per_class: vec![0; num_classes],
            total: 0,
            num_classes,
        }
    }

    pub fn update(
        &mut self,
        (predictions, targets): (&[&[f64]], &[usize]),
    ) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            verify_label(target, self.num_classes)?;

            if prediction.len() != self.num_classes {
                return Err(MetricError::IncompatibleInput {
                    expected: format!(
                        "length of predictions must be equal to number of classes: {}",
                        self.num_classes
                    ),
                    got: format!("got {}", prediction.len()),
                });
            }
            let prediction_idx = prediction
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .expect("Vector is empty");
            for class_idx in 0..self.num_classes {
                if class_idx == target {
                    if class_idx == prediction_idx {
                        self.true_positive[class_idx] += 1;
                    } else {
                        self.false_negative[class_idx] += 1;
                    }
                } else if class_idx == prediction_idx {
                    self.false_positive[class_idx] += 1;
                } else {
                    self.true_negative[class_idx] += 1;
                }
                self.total_per_class[class_idx] += 1;
            }

            self.total += 1;
        }
        Ok(())
    }
    pub fn reset(&mut self) {
        self.true_positive = vec![0; self.num_classes];
        self.false_positive = vec![0; self.num_classes];
        self.false_negative = vec![0; self.num_classes];
        self.true_negative = vec![0; self.num_classes];
        self.total_per_class = vec![0; self.num_classes];
        self.total = 0;
    }
}
