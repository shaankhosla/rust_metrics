use crate::core::{Metric, MetricError};
use crate::utils::tokenize;
use std::collections::{BTreeMap, HashMap};

/// Streaming ROUGE scores with TorchMetrics-style outputs.
///
/// This simplified version mirrors TorchMetrics defaults (no custom tokenizers or stemmers) and
/// reports the `precision`, `recall`, and `fmeasure` for `rouge1`, `rouge2`, `rougeL`, and
/// `rougeLsum`.
///
/// ```
/// use rust_metrics::{Metric, RougeScore};
///
/// let preds = ["My name is John"];
/// let targets = ["Is your name John"];
/// let mut rouge = RougeScore::default();
/// rouge.update((&preds, &targets)).unwrap();
/// let scores = rouge.compute().unwrap();
/// assert!((scores["rouge1_fmeasure"] - 0.75).abs() < 1e-9);
/// ```
#[derive(Debug, Clone)]
pub struct RougeScore {
    stats: [RougeAccumulator; ROUGE_KINDS.len()],
    total: usize,
}

impl Default for RougeScore {
    fn default() -> Self {
        Self {
            stats: [RougeAccumulator::default(); ROUGE_KINDS.len()],
            total: 0,
        }
    }
}

impl Metric<(&[&str], &[&str])> for RougeScore {
    type Output = BTreeMap<String, f64>;

    fn update(&mut self, (predictions, targets): (&[&str], &[&str])) -> Result<(), MetricError> {
        if predictions.len() != targets.len() {
            return Err(MetricError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            });
        }

        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            let pred_tokens = tokenize_words(prediction);
            let target_tokens = tokenize_words(target);

            let rouge1 = rouge_n(&pred_tokens, &target_tokens, 1);
            self.stats[RougeKind::Rouge1.index()].add(rouge1);

            let rouge2 = rouge_n(&pred_tokens, &target_tokens, 2);
            self.stats[RougeKind::Rouge2.index()].add(rouge2);

            let rouge_l = rouge_l_tokens(&pred_tokens, &target_tokens);
            self.stats[RougeKind::RougeL.index()].add(rouge_l);

            let pred_lsum = tokenize_with_newlines(prediction);
            let target_lsum = tokenize_with_newlines(target);
            let rouge_lsum = rouge_l_tokens(&pred_lsum, &target_lsum);
            self.stats[RougeKind::RougeLsum.index()].add(rouge_lsum);

            self.total += 1;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.stats = [RougeAccumulator::default(); ROUGE_KINDS.len()];
        self.total = 0;
    }

    fn compute(&self) -> Option<Self::Output> {
        if self.total == 0 {
            return None;
        }

        let mut scores = BTreeMap::new();
        let denom = self.total as f64;

        for kind in ROUGE_KINDS {
            let acc = self.stats[kind.index()];
            let prefix = kind.label();
            scores.insert(format!("{prefix}_precision"), acc.precision_sum / denom);
            scores.insert(format!("{prefix}_recall"), acc.recall_sum / denom);
            scores.insert(format!("{prefix}_fmeasure"), acc.fmeasure_sum / denom);
        }

        Some(scores)
    }
}

#[derive(Debug, Clone, Copy)]
struct RougeAccumulator {
    precision_sum: f64,
    recall_sum: f64,
    fmeasure_sum: f64,
}

impl RougeAccumulator {
    fn add(&mut self, (precision, recall, fmeasure): (f64, f64, f64)) {
        self.precision_sum += precision;
        self.recall_sum += recall;
        self.fmeasure_sum += fmeasure;
    }
}

impl Default for RougeAccumulator {
    fn default() -> Self {
        Self {
            precision_sum: 0.0,
            recall_sum: 0.0,
            fmeasure_sum: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RougeKind {
    Rouge1,
    Rouge2,
    RougeL,
    RougeLsum,
}

impl RougeKind {
    fn index(self) -> usize {
        match self {
            RougeKind::Rouge1 => 0,
            RougeKind::Rouge2 => 1,
            RougeKind::RougeL => 2,
            RougeKind::RougeLsum => 3,
        }
    }

    fn label(self) -> &'static str {
        match self {
            RougeKind::Rouge1 => "rouge1",
            RougeKind::Rouge2 => "rouge2",
            RougeKind::RougeL => "rougeL",
            RougeKind::RougeLsum => "rougeLsum",
        }
    }
}

const ROUGE_KINDS: [RougeKind; 4] = [
    RougeKind::Rouge1,
    RougeKind::Rouge2,
    RougeKind::RougeL,
    RougeKind::RougeLsum,
];

fn normalize_text(input: &str) -> String {
    let mut normalized = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
        } else {
            normalized.push(' ');
        }
    }
    normalized
}

fn tokenize_words(text: &str) -> Vec<String> {
    let normalized = normalize_text(text);
    tokenize(&normalized)
        .into_iter()
        .map(|token| token.to_string())
        .collect()
}

fn tokenize_with_newlines(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for sentence in text.split('\n') {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }
        tokens.extend(tokenize_words(trimmed));
        tokens.push("<n>".to_string());
    }
    if matches!(tokens.last(), Some(last) if last == "<n>") {
        tokens.pop();
    }
    tokens
}

fn rouge_n(pred_tokens: &[String], target_tokens: &[String], n: usize) -> (f64, f64, f64) {
    if pred_tokens.len() < n || target_tokens.len() < n {
        return (0.0, 0.0, 0.0);
    }

    let pred_counts = ngram_counts(pred_tokens, n);
    let target_counts = ngram_counts(target_tokens, n);

    let mut overlap = 0usize;
    for (ngram, &count) in &pred_counts {
        if let Some(&target_count) = target_counts.get(ngram) {
            overlap += count.min(target_count);
        }
    }

    let pred_total = pred_tokens.len() + 1 - n;
    let target_total = target_tokens.len() + 1 - n;
    precision_recall_fmeasure(overlap, pred_total, target_total)
}

fn ngram_counts(tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
    let mut counts = HashMap::new();
    for window in tokens.windows(n) {
        let key: Vec<String> = window.iter().cloned().collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn rouge_l_tokens(pred_tokens: &[String], target_tokens: &[String]) -> (f64, f64, f64) {
    if pred_tokens.is_empty() || target_tokens.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let lcs = lcs_length(pred_tokens, target_tokens);
    precision_recall_fmeasure(lcs, pred_tokens.len(), target_tokens.len())
}

fn lcs_length(pred_tokens: &[String], target_tokens: &[String]) -> usize {
    let pred_len = pred_tokens.len();
    let target_len = target_tokens.len();

    if pred_len == 0 || target_len == 0 {
        return 0;
    }

    let mut dp = vec![vec![0usize; target_len + 1]; pred_len + 1];
    for i in 1..=pred_len {
        for j in 1..=target_len {
            if pred_tokens[i - 1] == target_tokens[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[pred_len][target_len]
}

fn precision_recall_fmeasure(
    overlap: usize,
    pred_total: usize,
    target_total: usize,
) -> (f64, f64, f64) {
    if overlap == 0 || pred_total == 0 || target_total == 0 {
        return (0.0, 0.0, 0.0);
    }
    let precision = overlap as f64 / pred_total as f64;
    let recall = overlap as f64 / target_total as f64;
    let fmeasure = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (precision, recall, fmeasure)
}

#[cfg(test)]
mod tests {
    use super::RougeScore;
    use crate::core::Metric;

    fn approx_equal(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn matches_torch_example() {
        let preds = ["My name is John"];
        let targets = ["Is your name John"];

        let mut rouge = RougeScore::default();
        rouge.update((&preds, &targets)).unwrap();
        let scores = rouge.compute().unwrap();

        assert!(approx_equal(scores["rouge1_precision"], 0.75));
        assert!(approx_equal(scores["rouge1_recall"], 0.75));
        assert!(approx_equal(scores["rouge1_fmeasure"], 0.75));
        assert!(approx_equal(scores["rouge2_precision"], 0.0));
        assert!(approx_equal(scores["rougeL_fmeasure"], 0.5));
        assert!(approx_equal(scores["rougeLsum_fmeasure"], 0.5));
    }

    #[test]
    fn aggregates_over_batches() {
        let mut rouge = RougeScore::default();

        let preds = ["the cat is on the mat", "the fast cat slept"];
        let targets = ["the cat is on the mat", "the slow dog slept"];

        rouge.update((&preds, &targets)).unwrap();
        let scores = rouge.compute().unwrap();

        assert!(scores["rouge1_precision"] <= 1.0);
        assert!(scores["rouge1_precision"] >= 0.5);
        assert!(scores["rouge2_recall"] <= 1.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut rouge = RougeScore::default();
        let preds = ["hello there"];
        let targets = ["general kenobi"];
        rouge.update((&preds, &targets)).unwrap();
        assert!(rouge.compute().is_some());
        rouge.reset();
        assert!(rouge.compute().is_none());
    }
}
