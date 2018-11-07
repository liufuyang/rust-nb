extern crate rust_nb;

use rust_nb::{Feature, FeatureType, Model};

fn main() {
    let mut model = Model::new();

    let input_train = vec![
        (
            "go play well".to_owned(),
            vec![Feature {
                feature_type: FeatureType::Category,
                name: "weather.title".to_owned(),
                value: "sunny".to_owned(),
            }],
        ),
        (
            "go play well".to_owned(),
            vec![Feature {
                feature_type: FeatureType::Category,
                name: "weather.title".to_owned(),
                value: "cloudy".to_owned(),
            }],
        ),
        (
            "take umbrella".to_owned(),
            vec![Feature {
                feature_type: FeatureType::Category,
                name: "weather.title".to_owned(),
                value: "rainy".to_owned(),
            }],
        ),
        (
            "take umbrella".to_owned(),
            vec![Feature {
                feature_type: FeatureType::Category,
                name: "weather.title".to_owned(),
                value: "rainy".to_owned(),
            }],
        ),
        (
            "wear more cloth".to_owned(),
            vec![Feature {
                feature_type: FeatureType::Category,
                name: "weather.title".to_owned(),
                value: "cloudy".to_owned(),
            }],
        ),
        (
            "wear more cloth".to_owned(),
            vec![Feature {
                feature_type: FeatureType::Category,
                name: "weather.title".to_owned(),
                value: "sunny".to_owned(),
            }],
        ),
    ];
    model.train("Mom's word to me before I go out", input_train);

    // note: model is trained only with "weather.title" as a category feature,
    // the following prediction test only will take account the "weather.title" feature in the input
    // test example 1
    let result = model
        .predict(
            "Mom's word to me before I go out",
            &vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "0.0".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "sunny".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "2".to_owned(),
                },
            ],
        ).unwrap();

    println!("{:?}\n", result);
    assert!(result.get("wear more cloth").unwrap().abs() >= 0.4);
    assert!(result.get("go play well").unwrap().abs() >= 0.4);

    // test example 2
    let result = model
        .predict(
            "Mom's word to me before I go out",
            &vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "22".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "rainy".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "5".to_owned(),
                },
            ],
        ).unwrap();

    println!("{:?}\n", result);
    assert!(result.get("take umbrella").unwrap().abs() > 0.5);

    // test example 3
    let result = model
        .predict(
            "Mom's word to me before I go out",
            &vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "25".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "cloudy".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "3".to_owned(),
                },
            ],
        ).unwrap();

    println!("{:?}\n", result);
    assert!(result.get("wear more cloth").unwrap().abs() >= 0.4);
    assert!(result.get("go play well").unwrap().abs() >= 0.4);
}
