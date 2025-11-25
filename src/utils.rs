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

pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f64 {
    let dot: f64 = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let norm1 = (v1.iter().map(|a| (*a as f64).powi(2)).sum::<f64>()).sqrt();
    let norm2 = (v2.iter().map(|a| (*a as f64).powi(2)).sum::<f64>()).sqrt();
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot / (norm1 * norm2)
    }
}
