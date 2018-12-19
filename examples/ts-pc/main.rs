extern crate rust_nb;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use rust_nb::{Feature, FeatureType, Model};

fn main() {
    let mut model = Model::new().with_pseudo_count(0.01);

    let train_data = load_txt(".hidden/train.csv");
    let test_data = load_txt(".hidden/test.csv");
    let (test_labels, test_features): (Vec<String>, Vec<Vec<Feature>>) =
        test_data.into_iter().map(|(s, v)| (s, v)).unzip();

    println!(
        "Train size: {}, test size: {}",
        train_data.len(),
        test_features.len()
    );

    model.train("pc", &train_data);
    println!("Training finished");

    let predicts = model.predict_batch("pc", &test_features);
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
    let score = total_test_score / test_labels.len() as f64;

    println!("test score: {}", score);
}

fn load_txt(file_name: &str) -> Vec<(String, Vec<Feature>)> {
    let f = File::open(file_name).unwrap();
    let f = BufReader::new(&f);

    let mut v = vec![];

    for line in f.lines() {
        let line = line.unwrap();
        let line_v: Vec<&str> = line.rsplitn(2, ",").collect();
        
        let unspsc = line_v.get(0).unwrap_or(&"UNKNOWN").to_string();
        let product_name  = line_v.get(1).unwrap_or(&"").to_string();

        // print!("unspsc: {}, ", unspsc);
        // println!("product name: {}", product_name);

        v.push((
            unspsc,
            vec![Feature {
                feature_type: FeatureType::Text,
                name: "product name".to_owned(),
                value: product_name,
            }],
        ));
    }
    v
}
