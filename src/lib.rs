use std::collections::HashMap;

pub struct Category {
    pub name: String,
    pub value: String,
}

pub struct Text {
    pub name: String,
    pub value: String,
}

pub struct Model<T: ModelStore> {
    model_store: T,
}

impl Model<ModelHashMapStore> {
    pub fn new() -> Model<ModelHashMapStore> {
        Model::<ModelHashMapStore> {
            model_store: ModelHashMapStore::new(),
        }
    }
}

pub trait ModelStore {
    fn add_to_total_data_count(&mut self, model_name: &str, v: usize);
    fn get_total_data_count(&self, model_name: &str) -> usize;
}

// A in memory ModelStore implementation ModelHashMapStore
pub struct ModelHashMapStore {
    total_data_count: HashMap<String, usize>,
    count_of_word_in_class: HashMap<String, usize>,
    count_of_all_word_in_class: HashMap<String, usize>,
    words_appeared: HashMap<String, usize>,
}

impl ModelHashMapStore {
    pub fn new() -> ModelHashMapStore {
        ModelHashMapStore {
            total_data_count: HashMap::new(),
            count_of_word_in_class: HashMap::new(),
            count_of_all_word_in_class: HashMap::new(),
            words_appeared: HashMap::new(),
        }
    }

    pub fn new_large() -> ModelHashMapStore {
        ModelHashMapStore {
            total_data_count: HashMap::with_capacity(32),
            count_of_word_in_class: HashMap::with_capacity(262_144), // 2^18
            count_of_all_word_in_class: HashMap::with_capacity(256),
            words_appeared: HashMap::with_capacity(32),
        }
    }
}

impl ModelStore for ModelHashMapStore {
    fn add_to_total_data_count(&mut self, model_name: &str, v: usize) {
        let total_count = self
            .total_data_count
            .entry(model_name.to_string())
            .or_insert(0);
        *total_count += v;
    }

    fn get_total_data_count(&self, model_name: &str) -> usize {
        *self.total_data_count.get(model_name).unwrap_or_else(|| &0)
    }
}

pub fn add() {
    println!("add works")
}
