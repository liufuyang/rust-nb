extern crate rayon;
extern crate rust_nb;

use rayon::prelude::*;
use rust_nb::{Feature, FeatureType, Model};
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

fn main() {
    let mut model = Model::new().with_pseudo_count(0.01);

    let train_data = load_txt(".hidden/train.csv");
    let test_data = load_txt(".hidden/test.csv");

    let test_data_len = test_data.len();

    println!(
        "Train size: {}, test size: {}",
        train_data.len(),
        test_data.len()
    );

    model.train("pc", &train_data);
    println!("Training finished");

    let total_test_score: f64 = test_data
        .into_par_iter()
        .map(|(test_label, features)| {
            let predict = model.predict("pc", &features);
            let (pred_label, _test_score) = predict
                .iter()
                .max_by(|(_ka, va), (_kb, vb)| va.partial_cmp(vb).unwrap())
                .unwrap();
            if &test_label == pred_label {
                1.0
            } else {
                0.0
            }
        })
        .sum();

    println!("Testing finished");

    let score = total_test_score / test_data_len as f64;

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
        let product_name = line_v.get(1).unwrap_or(&"").to_string();

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
