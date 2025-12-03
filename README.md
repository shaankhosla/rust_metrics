# rust_metrics

`rust_metrics` is an ML evaluation toolkit that brings [Torchmetrics](https://github.com/Lightning-AI/torchmetrics)-style metrics to Rust. Each metric implements the same incremental
`Metric` trait, so you can feed batched predictions over time and ask for the final score when ready.
Every metric shares the same Torch-inspired test cases and examples so that the
crate docs mirror the upstream behavior where the functionality matches.

Some benefits of `rust-metrics`:

- A standardized interface to increase reproducibility
- Reduces Boilerplate
- Rigorously tested
- Automatic accumulation over batches


## Getting started

Add the crate to your project:

```bash
cargo add rust_metrics
# or enable the BERT-based similarity metric
cargo add rust_metrics --features text-bert
```

## TorchMetrics-aligned examples

All snippets below (and method examples) reuse the exact inputs from the public TorchMetrics docs so you can cross-check the expected values.

### Classification

```rust
use rust_metrics::{BinaryAccuracy, BinaryAuroc, Metric};

let target = [0_usize, 1, 0, 1, 0, 1];
let preds = [0.11, 0.22, 0.84, 0.73, 0.33, 0.92];

let mut acc = BinaryAccuracy::default();
acc.update((&preds[..], &target[..])).unwrap();
assert!((acc.compute().unwrap() - 2.0 / 3.0).abs() < f64::EPSILON);

let mut auroc = BinaryAuroc::new(0); 
let auroc_scores = [0.0, 0.5, 0.7, 0.8];
let auroc_target = [0_usize, 1, 1, 0];
auroc.update((&auroc_scores, &auroc_target)).unwrap();
assert!((auroc.compute().unwrap() - 0.5).abs() < f64::EPSILON);
```

### Regression

```rust
use rust_metrics::{MeanSquaredError, MeanAbsoluteError, Metric};

let mut mse = MeanSquaredError::default();
mse.update((&[3.0, 5.0, 2.5, 7.0], &[2.5, 5.0, 4.0, 8.0])).unwrap();
assert!((mse.compute().unwrap() - 0.875).abs() < f64::EPSILON);

let mut mae = MeanAbsoluteError::default();
mae.update((&[2.5, 0.0, 2.0, 8.0], &[3.0, -0.5, 2.0, 7.0])).unwrap();
assert!((mae.compute().unwrap() - 0.5).abs() < f64::EPSILON);
```

### Text

```rust
use rust_metrics::{Bleu, EditDistance, Metric};

let preds = ["the cat is on the mat"];
let targets = ["a cat is on the mat"];

let mut bleu = Bleu::default();
bleu.update((&preds, &targets)).unwrap();
assert!(bleu.compute().unwrap() > 0.5); 

let mut edit = EditDistance::default();
edit.update((&["rain"], &["shine"])).unwrap();
assert_eq!(edit.compute(), Some(3.0));
```

For `SentenceEmbeddingSimilarity` enable the `text-bert` feature; it mirrors the `BERTScore` example sentences and
reports cosine similarities for each pair instead of precision/recall triples.

## Implemented metrics

### Classification

- `BinaryAccuracy`, `MulticlassAccuracy`
- `BinaryPrecision`, `BinaryRecall`, `MulticlassPrecision` 
- `BinaryF1Score`, `MulticlassF1Score`
- `BinaryHingeLoss`, `MulticlassHingeLoss`
- `BinaryJaccardIndex`, `MulticlassJaccardIndex`
- `BinaryConfusionMatrix`
- `BinaryAuroc`

### Regression

- `MeanSquaredError`
- `NormalizedRootMeanSquaredError`
- `MeanAbsoluteError`
- `MeanAbsolutePercentageError`
- `R2Score`

### Text

- `Bleu` with optional smoothing and arbitrary n-gram depth
- `EditDistance` with sum or mean reduction
- `RougeScore` 
- `SentenceEmbeddingSimilarity` (requires the `text-bert` feature) backed by [`fastembed`]. This
  metric embeds each sentence pair with lightweight BERT embeddings and reports cosine similarity
  scores.

[`fastembed`]: https://crates.io/crates/fastembed

## Feature flags

| Feature    | Default | Description                                                  |
| --------- | ------- | ------------------------------------------------------------ |
| `text-bert` | no    | Enables BERT sentence embedding similarity via `fastembed`. |
