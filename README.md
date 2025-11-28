# rust_metrics

`rust_metrics` is an ML evaluation toolkit that brings [Torchmetrics](https://github.com/Lightning-AI/torchmetrics)-style metrics to Rust. Each metric implements the same incremental
`Metric` trait, so you can feed batched predictions over time and ask for the final score when ready.

## Getting started

Add the crate to your project:

```bash
cargo add rust_metrics
# or enable the BERT-based similarity metric
cargo add rust_metrics --features text-bert
```

Evaluate batched predictions:

```rust
use rust_metrics::{BinaryAccuracy, BinaryAuroc, Metric};

let predictions = [0, 1, 1, 0];
let targets = [0, 1, 0, 0];

let mut accuracy = BinaryAccuracy::new();
accuracy.update((&predictions, &targets)).unwrap();
assert_eq!(accuracy.compute(), Some(0.75));

let scores = [0.9, 0.6, 0.1, 0.2];
let mut auroc = BinaryAuroc::new(0); // 0 => compute exact ROC AUC
auroc.update((&scores, &targets.map(|t| t as f64))).unwrap();
assert!(auroc.compute().unwrap() > 0.6);
```

## Available metrics

### Classification

- `BinaryAccuracy`, `MulticlassAccuracy`, `MultilabelAccuracy`
- `BinaryPrecision`, `BinaryRecall`
- `BinaryHinge`
- `BinaryAuroc` (exact or binned ROC AUC)
- `BinaryF1Score`

### Regression

- `MeanSquaredError`
- `MeanAbsoluteError`
- `MeanAbsolutePercentageError`
- `R2Score`

### Text

- `Bleu` with optional smoothing and arbitrary n-gram depth
- `EditDistance` with sum or mean reduction
- `SentenceEmbeddingSimilarity` (requires the `text-bert` feature) backed by [`fastembed`]. This
  metric embeds each sentence pair with lightweight BERT embeddings and reports cosine similarity
  scores.

[`fastembed`]: https://crates.io/crates/fastembed

## Feature flags

| Feature    | Default | Description                                                  |
| --------- | ------- | ------------------------------------------------------------ |
| `text-bert` | no    | Enables BERT sentence embedding similarity via `fastembed`. |

