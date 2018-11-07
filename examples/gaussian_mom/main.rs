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
                    value: "5".to_owned(),
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
                    value: "2".to_owned(),
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
                    value: "2".to_owned(),
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
                    value: "2".to_owned(),
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
                    value: "2".to_owned(),
                },
            ],
        ),
    ];

    model.train("Mom's word to me before I go out", input_train);

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
    assert!(result.get("wear more cloth").unwrap().abs() > 0.9);
    // result will be:
    // {
    //     "take umbrella":0.06242401106172323,
    //     "go play well":0.00000032854821646191724,
    //     "wear more cloth":0.9375756603900602
    // }

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
    assert!(result.get("take umbrella").unwrap().abs() > 0.9);
    // result will be:
    // {
    //     "take umbrella":0.9277325553119831,
    //     "go play well":0.07226744468789072,
    //     "wear more cloth":0.00000000000012621909863350682
    // }

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
    assert!(result.get("go play well").unwrap().abs() > 0.5);
    // result will be:
    // {
    //     "wear more cloth":0.00000000000000007428571894758607,
    //     "take umbrella":0.45590974640701326,
    //     "go play well":0.5440902535929867
    // }
}
