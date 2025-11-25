use crate::core::MetricError;

pub fn verify_range(input: f64, min: f64, max: f64) -> Result<(), MetricError> {
    if (min..=max).contains(&input) {
        Ok(())
    } else {
        Err(MetricError::IncompatibleInput {
            expected: "prediction must be between 0 and 1",
            got: "other",
        })
    }
}
