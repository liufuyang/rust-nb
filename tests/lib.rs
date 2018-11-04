#[cfg(test)]
mod tests {
    use rust_nb::{Feature, Model, ModelHashMapStore, ModelStore};
    use std::collections::HashMap;

    #[test]
    fn model_works() {
        let mut model = Model::new();

        let input_train = vec![
            (
                "happy".to_owned(),
                vec![Feature {
                    is_text: true,
                    name: "my_words".to_owned(),
                    value: "The weather is so good".to_owned(),
                }],
            ),
            (
                "sad".to_owned(),
                vec![Feature {
                    is_text: true,
                    name: "my_words".to_owned(),
                    value: "The food tastes so bad".to_owned(),
                }],
            ),
        ];
        model.train("test_model", input_train);

        let input_test = vec![Feature {
            is_text: true,
            name: "my_words".to_owned(),
            value: "The weather is so good".to_owned(),
        }];
        let result = model.predict("test_model", &input_test);

        println!(">>>>>>>>>>>>>>>>result: {:?}", result);
    }

    #[test]
    fn model_hashmap_store_works() {
        let model_store = ModelHashMapStore::new();
        assert_eq!(0, model_store.get_total_data_count("test_model"));
        let mut model_store = model_store;
        model_store.add_to_total_data_count("test_model", 1);
        assert_eq!(1, model_store.get_total_data_count("test_model"));
        model_store.add_to_total_data_count("test_model", 10);
        assert_eq!(11, model_store.get_total_data_count("test_model"));
    }

    #[test]
    fn normalize_works() {
        let mut map = HashMap::new();
        map.insert("a".to_owned(), 1.0);
        map.insert("b".to_owned(), 5.0);

        let map = rust_nb::normalize(map);
        println!("{:?}", map);
    }

    #[test]
    fn model_class_map_works() {
        let model_store = ModelHashMapStore::new();
        assert_eq!(
            0,
            model_store.get_priors_count_of_class("test_model", "class_1")
        );
        assert_eq!(
            0,
            model_store.get_priors_count_of_class("test_model", "class_2")
        );
        assert_eq!(None, model_store.get_all_classes("test_model"));

        let mut model_store = model_store;

        model_store.add_to_priors_count_of_class("test_model", "class_1", 1);
        assert_eq!(
            1,
            model_store.get_priors_count_of_class("test_model", "class_1")
        );
        assert_eq!(1, model_store.get_all_classes("test_model").unwrap().len());

        model_store.add_to_priors_count_of_class("test_model", "class_1", 10);
        assert_eq!(
            11,
            model_store.get_priors_count_of_class("test_model", "class_1")
        );
        assert_eq!(1, model_store.get_all_classes("test_model").unwrap().len());

        model_store.add_to_priors_count_of_class("test_model", "class_2", 10);
        assert_eq!(
            10,
            model_store.get_priors_count_of_class("test_model", "class_2")
        );
        assert_eq!(2, model_store.get_all_classes("test_model").unwrap().len());

        model_store.add_to_priors_count_of_class("test_model", "class_2", 10);
        assert_eq!(
            20,
            model_store.get_priors_count_of_class("test_model", "class_2")
        );
        assert_eq!(2, model_store.get_all_classes("test_model").unwrap().len());
    }
}
