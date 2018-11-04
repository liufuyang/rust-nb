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
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn it_works_2() {
        rust_nb::add()
    }
}
