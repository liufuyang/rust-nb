#[cfg(test)]
mod rust_nb {
    use rust_nb::{Feature, FeatureType, Model};

    #[test]
    fn model_works_simple_case() {
        let mut model = Model::new();

        let input_train = vec![
            (
                "happy".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Text,
                    name: "my_words".to_owned(),
                    value: "good".to_owned(),
                }],
            ),
            (
                "sad".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Text,
                    name: "my_words".to_owned(),
                    value: "bad".to_owned(),
                }],
            ),
        ];
        model.train("test_model", &input_train);

        let input_test = vec![Feature {
            feature_type: FeatureType::Text,
            name: "my_words".to_owned(),
            value: "no opinion".to_owned(),
        }];
        let result = model.predict("test_model", &input_test);
        assert_eq!(0.5, *result.get("happy").unwrap());
        assert_eq!(0.5, *result.get("sad").unwrap());

        let input_test = vec![Feature {
            feature_type: FeatureType::Text,
            name: "my_words".to_owned(),
            value: "GOOD".to_owned(),
        }];
        let result = model.predict("test_model", &input_test);
        assert!((0.6666666666666666 - *result.get("happy").unwrap()).abs() < 1e-10);
        assert!((0.3333333333333333 - *result.get("sad").unwrap()).abs() < 1e-10);
    }

    #[test]
    fn model_works() {
        let mut model = Model::new();

        let input_train = vec![
            (
                "happy".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Text,
                    name: "my_words".to_owned(),
                    value: "The weather is so good".to_owned(),
                }],
            ),
            (
                "sad".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Text,
                    name: "my_words".to_owned(),
                    value: "that food tastes so bad".to_owned(),
                }],
            ),
        ];
        model.train("test_model", &input_train);

        let input_test = vec![Feature {
            feature_type: FeatureType::Text,
            name: "my_words".to_owned(),
            value: "thinking about the weather ...".to_owned(),
        }];
        let result = model.predict("test_model", &input_test);

        assert!((0.8 - *result.get("happy").unwrap()).abs() < 1e-10);
        assert!((0.2 - *result.get("sad").unwrap()).abs() < 1e-10);
    }

    #[test]
    fn model_gaussian_works() {
        let mut model = Model::new();

        let input_train = vec![
            (
                "eur".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "age".to_owned(),
                    value: "20".to_owned(),
                }],
            ),
            (
                "eur".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "age".to_owned(),
                    value: "30".to_owned(),
                }],
            ),
            (
                "usd".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "age".to_owned(),
                    value: "40".to_owned(),
                }],
            ),
            (
                "usd".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "age".to_owned(),
                    value: "50".to_owned(),
                }],
            ),
            (
                "eur".to_owned(),
                vec![Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "age".to_owned(),
                    value: "40".to_owned(),
                }],
            ),
        ];
        model.train("test_model", &input_train);

        let input_test = vec![Feature {
            feature_type: FeatureType::Gaussian,
            name: "age".to_owned(),
            value: "23".to_owned(),
        }];
        let result = model.predict("test_model", &input_test);

        println!("{:?}", result);

        let mut p_eur = 0.0312254; // https://www.wolframalpha.com/input/?i=normalpdf(23.0,+30.0,+10.0)
        let mut p_usd = 0.000446108; // https://www.wolframalpha.com/input/?i=normalpdf(23.0,+45.0,+7.0710678118654755)

        p_usd = p_usd * (2.0 / 5.0);
        p_eur = p_eur * (3.0 / 5.0);

        p_usd = p_usd / (p_usd + p_eur);
        p_eur = 1.0 - p_usd;

        assert!((p_usd - *result.get("usd").unwrap()).abs() < 1e-6);
        assert!((p_eur - *result.get("eur").unwrap()).abs() < 1e-6);
    }

}
