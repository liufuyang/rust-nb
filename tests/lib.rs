#[cfg(test)]
mod tests {
    use rust_nb::{Model, ModelHashMapStore, ModelStore};
    use std::collections::HashMap;

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

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn it_works_2() {
        rust_nb::add()
    }
}
