use rust_metrics::{BinaryAccuracy, BinaryAuroc, Metric};

fn main() {
    let predictions = [0, 1, 1, 0];
    let targets = [0, 1, 0, 0];
    let mut accuracy = BinaryAccuracy::new();
    accuracy
        .update((&predictions, &targets))
        .expect("lengths should match");
    println!("Accuracy: {:.2}%", accuracy.compute() * 100.0);

    let predictions = [0.3, 0.2, 0.2, 0.5];
    let targets = [0.0, 1.0, 0.0, 0.0];
    let mut auc = BinaryAuroc::new(10000);
    auc.update((&predictions, &targets))
        .expect("lengths should match");
    println!("AUC: {:.2}%", auc.compute() * 100.0);
}
