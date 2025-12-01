use crate::core::MetricError;

pub fn verify_range(input: f64, min: f64, max: f64) -> Result<(), MetricError> {
    if (min..=max).contains(&input) {
        Ok(())
    } else {
        Err(MetricError::IncompatibleInput {
            expected: format!("value must be within the range [{}, {}]", min, max),
            got: format!("{}", input),
        })
    }
}

pub fn verify_label(input: usize, num_classes: usize) -> Result<(), MetricError> {
    if input < num_classes {
        Ok(())
    } else {
        Err(MetricError::IncompatibleInput {
            expected: format!("label index must be less than {}", num_classes),
            got: format!("{}", input),
        })
    }
}

pub fn verify_binary_label(input: usize) -> Result<(), MetricError> {
    verify_label(input, 2)
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

pub fn tokenize(input: &str) -> Vec<&str> {
    input.split_whitespace().collect()
}

pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    if s1.is_empty() {
        return s2.chars().count();
    }
    if s2.is_empty() {
        return s1.chars().count();
    }

    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    let mut dp = vec![vec![0usize; len2 + 1]; len1 + 1];

    for (i, item) in dp.iter_mut().enumerate().take(len1 + 1) {
        item[0] = i;
    }
    for j in 0..=len2 {
        dp[0][j] = j;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = *[dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost]
                .iter()
                .min()
                .unwrap();
        }
    }

    dp[len1][len2]
}

#[derive(Debug, Clone, Default)]
pub enum AverageMethod {
    Micro,
    #[default]
    Macro,
    Weighted,
}
