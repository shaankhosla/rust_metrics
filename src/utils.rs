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

    for i in 0..=len1 {
        dp[i][0] = i;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    Sum,
    #[default]
    Mean,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MetricAggregator {
    reduction: Reduction,
    total: usize,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl MetricAggregator {
    pub fn new(reduction: Reduction) -> Self {
        Self {
            reduction,
            total: 0,
            sum: 0.0,
            min: None,
            max: None,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.total += 1;
        self.sum += value;
        self.min = Some(self.min.map_or(value, |m| m.min(value)));
        self.max = Some(self.max.map_or(value, |m| m.max(value)));
    }
    pub fn reset(&mut self) {
        self.total = 0;
        self.sum = 0.0;
        self.min = None;
        self.max = None;
    }

    pub fn compute(&self) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        match self.reduction {
            Reduction::Sum => Some(self.sum),
            Reduction::Mean => Some(self.sum / self.total as f64),
            Reduction::Min => self.min,
            Reduction::Max => self.max,
        }
    }
}
