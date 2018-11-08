extern crate rust_nb;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use rust_nb::{Feature, FeatureType, Model};

fn main() {
    let mut model = Model::new().with_stop_words_file("examples/data/english-stop-words-large.txt");

    let train_data = load_txt("examples/data/20newsgroup_train.txt");
    let test_data = load_txt("examples/data/20newsgroup_test.txt");
    let (test_labels, test_features): (Vec<&str>, Vec<&Vec<Feature>>) =
        test_data.iter().map(|(s, v)| (s.as_str(), v)).unzip();

    println!(
        "Train size: {}, test size: {}",
        train_data.len(),
        test_data.len()
    );

    model.train("20newsgroup_model", &train_data);
    println!("Training finished");

    let predicts = model.predict_batch("20newsgroup_model", &test_features);
    println!("Testing finished");

    let total_test_score: f64 = test_labels
        .iter()
        .zip(predicts.iter())
        .map(|(test_label, predict)| {
            let (pred_label, _test_score) = predict
                .iter()
                .max_by(|(_ka, va), (_kb, vb)| va.partial_cmp(vb).unwrap())
                .unwrap();
            if test_label == pred_label {
                1.0
            } else {
                0.0
            }
        })
        .sum();
    let score = total_test_score / test_data.len() as f64;

    println!("test score: {}", score);
    assert!((0.6456452469463622 - score).abs() < 1e-10);
}

fn load_txt(file_name: &str) -> Vec<(String, Vec<Feature>)> {
    let f = File::open(file_name).unwrap();
    let f = BufReader::new(&f);

    let mut v = vec![];

    for line in f.lines() {
        let line = line.unwrap();
        let mut splitter = line.splitn(2, ' ');
        let first = splitter.next().unwrap();
        let second = splitter.next().unwrap();

        v.push((
            first.to_owned(),
            vec![Feature {
                feature_type: FeatureType::Text,
                name: "20newsgroup text".to_owned(),
                value: second.to_owned(),
            }],
        ));
    }
    v
}
