use rust_metrics::{Accuracy, Metric};

fn main() {
    let predictions = [0.0, 1.0, 2.0, 2.0];
    let targets = [0.0, 2.0, 2.0, 2.0];

    let mut accuracy = Accuracy::new();
    accuracy
        .update((&predictions, &targets))
        .expect("lengths should match");

    println!("Accuracy: {:.2}%", accuracy.compute() * 100.0);
}
