use std::collections::{HashMap, HashSet};

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
    fn add_to_priors_count_of_class(&mut self, model_name: &str, c: &str, v: usize);
    fn get_priors_count_of_class(&self, model_name: &str, c: &str) -> usize;
    fn get_all_classes(&self, model_name: &str) -> Option<&HashSet<String>>;

    fn add_to_total_data_count(&mut self, model_name: &str, v: usize);
    fn get_total_data_count(&self, model_name: &str) -> usize;
}

// A in memory ModelStore implementation ModelHashMapStore
pub struct ModelHashMapStore {
    map: HashMap<String, usize>,
    class_map: HashMap<String, HashSet<String>>, // model_name to list of class
    word_map: HashMap<String, usize>,            // a very large hash map for look up word counts
}

impl ModelHashMapStore {
    pub fn new() -> ModelHashMapStore {
        ModelHashMapStore {
            map: HashMap::new(),
            class_map: HashMap::new(),
            word_map: HashMap::new(),
        }
    }

    pub fn new_large() -> ModelHashMapStore {
        ModelHashMapStore {
            map: HashMap::with_capacity(1024),
            class_map: HashMap::with_capacity(1024),
            word_map: HashMap::with_capacity(262_144), // 2^18
        }
    }

    fn map_add(&mut self, model_name: &str, prefix: &str, v: usize) {
        let key = format!("{}|%{}%", model_name, prefix);
        let total_count = self.map.entry(key).or_insert(0);
        *total_count += v;
    }

    fn map_get(&self, model_name: &str, prefix: &str) -> usize {
        let key = format!("{}|%{}%", model_name, prefix);
        *self.map.get(&key).unwrap_or_else(|| &0)
    }
}

impl ModelStore for ModelHashMapStore {
    fn add_to_priors_count_of_class(&mut self, model_name: &str, c: &str, v: usize) {
        self.map_add(model_name, &format!("priors_count_of_class|%{}%", c), v);

        let class_vec = self
            .class_map
            .entry(model_name.to_string())
            .or_insert(HashSet::new());
        class_vec.insert(c.to_string());
    }

    fn get_priors_count_of_class(&self, model_name: &str, c: &str) -> usize {
        self.map_get(model_name, &format!("priors_count_of_class|%{}%", c))
    }

    fn get_all_classes(&self, model_name: &str) -> Option<&HashSet<String>> {
        self.class_map.get(model_name)
    }

    fn add_to_total_data_count(&mut self, model_name: &str, v: usize) {
        self.map_add(model_name, "total_data_count", v);
    }

    fn get_total_data_count(&self, model_name: &str) -> usize {
        self.map_get(model_name, "total_data_count")
    }
}

pub fn add() {
    println!("add works")
}
