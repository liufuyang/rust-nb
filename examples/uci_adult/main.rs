extern crate rust_nb;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use rust_nb::{Feature, FeatureType, Model};

static FEATURE_COLUMN_NAMES: &'static [(&str, FeatureType)] = &[
    ("age", FeatureType::Gaussian),            // 0
    ("workclass", FeatureType::Category),      // 1
    ("fnlwgt", FeatureType::Gaussian),         // 2, final weight
    ("education", FeatureType::Category),      // 3
    ("education-num", FeatureType::Gaussian),  // 4
    ("marital-status", FeatureType::Category), // 5
    ("occupation", FeatureType::Category),     // 6
    ("relationship", FeatureType::Category),   // 7
    ("race", FeatureType::Category),           // 8
    ("sex", FeatureType::Category),            // 9
    ("capital-gain", FeatureType::Gaussian),   // 10
    ("capital-loss", FeatureType::Gaussian),   // 11
    ("hours-per-week", FeatureType::Gaussian), // 12
    ("native-country", FeatureType::Category), // 13
];

fn main() {
    let mut model = Model::new()
        .with_prior_factor(1.0)
        .with_pseudo_count(0.1)
        .with_default_gaussian_sigma_factor(0.05);

    let train_data = load_txt("examples/data/adult.data");
    let test_data = load_txt("examples/data/adult.test");
    let (test_labels, test_features): (Vec<String>, Vec<Vec<Feature>>) =
        test_data.into_iter().map(|(s, v)| (s, v)).unzip();

    println!(
        "Train size: {}, test size: {}",
        train_data.len(),
        test_features.len()
    );

    model.train("uci_adult", &train_data);
    println!("Training finished");

    let predicts = model.predict_batch("uci_adult", &test_features);
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
    assert!((0.83 - score).abs() < 1e-2);
}

fn load_txt(file_name: &str) -> Vec<(String, Vec<Feature>)> {
    let f = File::open(file_name).unwrap();
    let f = BufReader::new(&f);

    let mut v = vec![];

    for line in f.lines() {
        let line = line.unwrap();
        let splitter = line.split(",");

        let mut features = vec![];
        let mut outcome = "Unknown";

        for (i, item) in splitter.enumerate() {
            let item = item.trim();
            if i < 14 {
                if item.is_empty() || item == "?" {
                    continue;
                }

                let (feature_name, feature_type) = &FEATURE_COLUMN_NAMES[i];
                features.push(Feature {
                    feature_type: feature_type.clone(),
                    name: feature_name.to_string(),
                    value: item.to_string(),
                });
            } else {
                // last, or the 14th is outcome
                outcome = item;
            }
        }

        v.push((outcome.replace(".", ""), features));
    }

    v
}
