use rust_metrics::{BinaryAccuracy, Metric};

fn main() {
    let predictions = [0, 1, 1, 0];
    let targets = [0, 1, 0, 0];

    let mut accuracy = BinaryAccuracy::new();
    accuracy
        .update((&predictions, &targets))
        .expect("lengths should match");

    println!("Accuracy: {:.2}%", accuracy.compute() * 100.0);
}
