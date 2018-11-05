#[cfg(test)]
mod test_20_newsgroup {
    use rust_nb::{ModelHashMapStore, ModelStore};

    #[test]
    fn model_hashmap_store_works() {
        let model_store = ModelHashMapStore::new();
        assert_eq!(0, model_store.get_total_data_count());
        let mut model_store = model_store;
        model_store.add_to_total_data_count(1);
        assert_eq!(1, model_store.get_total_data_count());
        model_store.add_to_total_data_count(10);
        assert_eq!(11, model_store.get_total_data_count());
    }

}
