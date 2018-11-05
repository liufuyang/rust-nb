#[cfg(test)]
mod test_20_newsgroup {
    use rust_nb::{ModelHashMapStore, ModelStore};

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

}
