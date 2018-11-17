extern crate rust_nb;

use rust_nb::{Feature, FeatureType, Model};

fn main() {
    let mut model = Model::new().with_default_gaussian_m2(100.0);

    let input_train = vec![
        (
            "go play well".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "32".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "sunny".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "3".to_owned(),
                },
            ],
        ),
        (
            "go play well".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "24".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "cloudy".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "1".to_owned(),
                },
            ],
        ),
        (
            "take umbrella".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "5".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "rainy".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "3".to_owned(),
                },
            ],
        ),
        (
            "take umbrella".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "25".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "rainy".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "4".to_owned(),
                },
            ],
        ),
        (
            "wear more cloth".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "-2".to_owned(),
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
        ),
        (
            "wear more cloth".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.degree".to_owned(),
                    value: "2".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "weather.title".to_owned(),
                    value: "sunny".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "weather.wind.level".to_owned(),
                    value: "3".to_owned(),
                },
            ],
        ),
    ];

    model.train("Mom's word to me before I go out", &input_train);

    // test example 1
    let result = model.predict(
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
    );

    println!("{:?}\n", result);
    assert!(result.get("wear more cloth").unwrap().abs() > 0.7);
    // result will be:
    // {"wear more cloth": 0.8145285759525068, "go play well": 0.1310511820033621, "take umbrella": 0.05442024204413106}

    // test example 2
    let result = model.predict(
        "Mom's word to me before I go out",
        &vec![
            Feature {
                feature_type: FeatureType::Gaussian,
                name: "weather.degree".to_owned(),
                value: "28".to_owned(),
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
    );

    println!("{:?}\n", result);
    assert!(result.get("take umbrella").unwrap().abs() > 0.6);
    // result will be:
    // {"wear more cloth": 0.040777064361781155, "take umbrella": 0.6929647650603867, "go play well": 0.2662581705778321}

    // test example 3
    let result = model.predict(
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
    );

    println!("{:?}\n", result);
    assert!(result.get("go play well").unwrap().abs() > 0.5);
    // result will be:
    // {"go play well": 0.6267604626518958, "wear more cloth": 0.14149599917558417, "take umbrella": 0.23174353817252016}
}
