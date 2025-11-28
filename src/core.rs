/// Errors emitted by metrics when incoming batches cannot be processed.
#[derive(Debug, PartialEq, Eq)]
pub enum MetricError {
    /// `predictions.len()` and `targets.len()` differ.
    LengthMismatch { predictions: usize, targets: usize },
    /// A class index is outside the configured range.
    InvalidClassIndex { class: usize, num_classes: usize },
    /// Multilabel inputs do not align with the expected shape.
    InvalidLabelShape {
        total_labels: usize,
        num_labels: usize,
    },
    /// Inputs fail additional validation (value ranges, binary labels, etc.).
    IncompatibleInput {
        expected: &'static str,
        got: &'static str,
    },
}

/// Common interface implemented by every streaming metric.
///
/// Metrics accept batched `Input` values via [`update`](Metric::update), can be cleared with
/// [`reset`](Metric::reset), and report an [`Output`](Metric::Output) when enough data has been
/// observed.
pub trait Metric<Input> {
    type Output;

    /// Incorporate another batch of predictions/targets.
    fn update(&mut self, input: Input) -> Result<(), MetricError>;

    /// Drop any accumulated state.
    fn reset(&mut self);

    /// Compute the final value; returns `None` until at least one batch was seen.
    fn compute(&self) -> Option<Self::Output>;
}
