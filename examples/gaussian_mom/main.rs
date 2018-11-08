extern crate rust_nb;

use rust_nb::{Feature, FeatureType, Model};

fn main() {
    let mut model = Model::new();

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
                    value: "2".to_owned(),
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
                    value: "20".to_owned(),
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
    assert!(result.get("wear more cloth").unwrap().abs() > 0.9);
    // result will be:
    // {
    //     "wear more cloth": 0.9655852714584366,
    //     "go play well": 0.0000033613494482027493,
    //     "take umbrella": 0.03441136719211526
    // }

    // test example 2
    let result = model.predict(
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
    );

    println!("{:?}\n", result);
    assert!(result.get("take umbrella").unwrap().abs() > 0.8);
    // result will be:
    // {
    //     "go play well": 0.0035707854476099555,
    //     "wear more cloth": 0.00000000000005651204952913297,
    //     "take umbrella": 0.9964292145523336
    // }

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
    // {
    //     "take umbrella": 0.294095527044516,
    //     "wear more cloth": 0.00000000000000005430023183112072,
    //     "go play well": 0.7059044729554839
    // }
}
