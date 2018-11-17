extern crate rust_nb;

use rust_nb::{Feature, FeatureType, Model};

fn main() {
    let mut model = Model::new();

    let input_train = vec![
        (
            "spam".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Text,
                    name: "email.body".to_owned(),
                    value: "Good day dear beneficiary. This is Secretary to president of Benin republic is writing this email ... heritage, tax, dollars, money, credit card...".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "email.domain".to_owned(),
                    value: "evil.com".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "email.n_words".to_owned(),
                    value: "482".to_owned(),
                },
            ],
        ),
        (
            "not spam".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Text,
                    name: "email.body".to_owned(),
                    value: "Hey bro, how's work these days, wanna join me for hotpot next week?".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "email.domain".to_owned(),
                    value: "gmail.com".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "email.n_words".to_owned(),
                    value: "42".to_owned(),
                },
            ],
        ),
    ];

    model.train("Spam checker", &input_train);

    // test example 1
    let result = model.predict(
        "Spam checker",
        &vec![
            Feature {
                feature_type: FeatureType::Text,
                name: "email.body".to_owned(),
                value: "Hey bro, This is Secretary to president want to give you some money. Please give me your credit card number ..."
                    .to_owned(),
            },
            Feature {
                feature_type: FeatureType::Category,
                name: "email.domain".to_owned(),
                value: "example.com".to_owned(),
            },
            Feature {
                feature_type: FeatureType::Gaussian,
                name: "email.n_words".to_owned(),
                value: "288".to_owned(),
            },
        ],
    );

    println!("{:?}\n", result);
    assert!(result.get("spam").unwrap().abs() > 0.9);
    // result will be:
    // {"not spam": 0.02950007253794831, "spam": 0.9704999274620517}

    // test example 2
    let result = model.predict(
        "Spam checker",
        &vec![
            Feature {
                feature_type: FeatureType::Text,
                name: "email.body".to_owned(),
                value: "Hey bro, hotpot again?".to_owned(),
            },
            Feature {
                feature_type: FeatureType::Category,
                name: "email.domain".to_owned(),
                value: "gmail.com".to_owned(),
            },
            Feature {
                feature_type: FeatureType::Gaussian,
                name: "email.n_words".to_owned(),
                value: "10".to_owned(),
            },
        ],
    );

    println!("{:?}\n", result);
    assert!(result.get("not spam").unwrap().abs() > 0.9);
    // result will be:
    // {"not spam": 0.9976790459980525, "spam": 0.002320954001947624}
}
