#[derive(Debug, PartialEq, Eq)]
pub enum MetricError {
    LengthMismatch { predictions: usize, targets: usize },
    InvalidClassIndex { class: usize, num_classes: usize },
    InvalidLabelShape { total_labels: usize, num_labels: usize },
    IncompatibleInput { expected: &'static str, got: &'static str },
}

pub trait Metric<Input> {
    type Output;

    fn update(&mut self, input: Input) -> Result<(), MetricError>;

    fn reset(&mut self);

    fn compute(&self) -> Self::Output;
}
