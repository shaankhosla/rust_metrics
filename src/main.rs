use rust_metrics::{BinaryAccuracy, BinaryAuroc, Metric};

#[cfg(feature = "text-bert")]
use rust_metrics::SentenceEmbeddingSimilarity;

fn main() {
    let predictions = [0.0, 1.0, 0.6, 0.0];
    let targets = [0, 1, 0, 0];
    let mut accuracy = BinaryAccuracy::default();
    accuracy
        .update((&predictions, &targets))
        .expect("lengths should match");
    println!("Accuracy: {:.2}%", accuracy.compute().unwrap() * 100.0);

    let predictions = [0.3, 0.2, 0.2, 0.5];
    let targets = [0_usize, 1, 0, 0];
    let mut auc = BinaryAuroc::new(10000);
    auc.update((&predictions, &targets))
        .expect("lengths should match");
    println!("Approximated AUC: {:.2}%", auc.compute().unwrap() * 100.0);

    let mut auc = BinaryAuroc::new(0);
    auc.update((&predictions, &targets))
        .expect("lengths should match");
    println!("AUC: {:.2}%", auc.compute().unwrap() * 100.0);

    #[cfg(feature = "text-bert")]
    {
        let mut bert_score = SentenceEmbeddingSimilarity::default();
        bert_score
            .update((
                &[
                    "hello world",
                    "ping",
                    "the quick brown fox jumped over the ",
                ],
                &["hi there world!", "pong", "gnop"],
            ))
            .expect("lengths should match");
        dbg!(bert_score.compute());
    }
}
