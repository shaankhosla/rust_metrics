#[derive(Debug, PartialEq, Eq)]
pub enum MetricError {
    LengthMismatch { predictions: usize, targets: usize },
}

pub trait Metric<Input> {
    type Output;

    fn update(&mut self, input: Input) -> Result<(), MetricError>;

    fn reset(&mut self);

    fn compute(&self) -> Self::Output;
}
