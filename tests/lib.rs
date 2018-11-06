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
        model.train("test_model", input_train);
        // println!("{:?}", model.get_store());

        let input_test = vec![Feature {
            feature_type: FeatureType::Text,
            name: "my_words".to_owned(),
            value: "no opinion".to_owned(),
        }];
        let result = model.predict("test_model", &input_test).unwrap();
        assert_eq!(0.5, *result.get("happy").unwrap());
        assert_eq!(0.5, *result.get("sad").unwrap());

        let input_test = vec![Feature {
            feature_type: FeatureType::Text,
            name: "my_words".to_owned(),
            value: "GOOD".to_owned(),
        }];
        let result = model.predict("test_model", &input_test).unwrap();
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
        model.train("test_model", input_train);

        let input_test = vec![Feature {
            feature_type: FeatureType::Text,
            name: "my_words".to_owned(),
            value: "thinking about the weather ...".to_owned(),
        }];
        let result = model.predict("test_model", &input_test).unwrap();

        assert!((0.8 - *result.get("happy").unwrap()).abs() < 1e-10);
        assert!((0.2 - *result.get("sad").unwrap()).abs() < 1e-10);
    }

}
