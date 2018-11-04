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

    fn add_to_count_of_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        word: &str,
        v: usize,
    );
    fn get_count_of_word_in_class(
        &self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        word: &str,
    ) -> usize;

    fn add_to_count_of_all_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        v: usize,
    );
    fn get_count_of_all_word_in_class(
        &self,
        model_name: &str,
        feature_name: &str,
        c: &str,
    ) -> usize;

    fn get_appeared_words(&self, model_name: &str, feature_name: &str) -> Option<&HashSet<String>>;
}

// A in memory ModelStore implementation ModelHashMapStore
pub struct ModelHashMapStore {
    map: HashMap<String, usize>,
    class_map: HashMap<String, HashSet<String>>, // model_name to list of class
    words_appeared_map: HashMap<String, HashSet<String>>, // model_name|feature_name to list of appeared words
    word_count_map: HashMap<String, usize>, // a very large hash map for look up word counts
    words_in_class_count_map: HashMap<String, usize>, // a hash map for look up word counts for each feature and class
}

impl ModelHashMapStore {
    pub fn new() -> ModelHashMapStore {
        ModelHashMapStore {
            map: HashMap::new(),
            class_map: HashMap::new(),
            words_appeared_map: HashMap::new(),
            word_count_map: HashMap::new(),
            words_in_class_count_map: HashMap::new(),
        }
    }

    pub fn new_large() -> ModelHashMapStore {
        ModelHashMapStore {
            map: HashMap::with_capacity(1024),
            class_map: HashMap::with_capacity(1024),
            words_appeared_map: HashMap::with_capacity(32),
            word_count_map: HashMap::with_capacity(262_144), // 2^18
            words_in_class_count_map: HashMap::with_capacity(1024),
        }
    }

    fn map_add(&mut self, model_name: &str, prefix: &str, v: usize) {
        let key = format!("{}|%{}", model_name, prefix);
        let total_count = self.map.entry(key).or_insert(0);
        *total_count += v;
    }

    fn map_get(&self, model_name: &str, prefix: &str) -> usize {
        let key = format!("{}|%{}", model_name, prefix);
        *self.map.get(&key).unwrap_or_else(|| &0)
    }
}

impl ModelStore for ModelHashMapStore {
    fn add_to_priors_count_of_class(&mut self, model_name: &str, c: &str, v: usize) {
        self.map_add(model_name, &format!("priors_count_of_class|%{}", c), v);

        // add class to class_map
        let class_vec = self
            .class_map
            .entry(model_name.to_string())
            .or_insert(HashSet::new());
        class_vec.insert(c.to_string());
    }

    fn get_priors_count_of_class(&self, model_name: &str, c: &str) -> usize {
        self.map_get(model_name, &format!("priors_count_of_class|%{}", c))
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

    fn add_to_count_of_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        word: &str,
        v: usize,
    ) {
        let key = format!("{}|%{}%|%{}%|%{}", model_name, feature_name, c, word);
        let word_count = self.word_count_map.entry(key).or_insert(0);
        *word_count += v;

        // add this word with this feature as well, no matter which class it is
        let word_set = self
            .words_appeared_map
            .entry(format!("{}|%{}", model_name, feature_name))
            .or_insert(HashSet::with_capacity(262_144)); // 2^18
        word_set.insert(word.to_string());
    }

    fn get_count_of_word_in_class(
        &self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        word: &str,
    ) -> usize {
        let key = format!("{}|%{}%|%{}%|%{}", model_name, feature_name, c, word);
        *self.word_count_map.get(&key).unwrap_or_else(|| &0)
    }

    fn add_to_count_of_all_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        v: usize,
    ) {
        let key = format!("{}|%{}%|%{}", model_name, feature_name, c);
        let words_count = self.words_in_class_count_map.entry(key).or_insert(0);
        *words_count += v;
    }

    fn get_count_of_all_word_in_class(
        &self,
        model_name: &str,
        feature_name: &str,
        c: &str,
    ) -> usize {
        let key = format!("{}|%{}%|%{}%", model_name, feature_name, c);
        *self
            .words_in_class_count_map
            .get(&key)
            .unwrap_or_else(|| &0)
    }

    fn get_appeared_words(&self, model_name: &str, feature_name: &str) -> Option<&HashSet<String>> {
        self.words_appeared_map
            .get(&format!("{}|%{}%", model_name, feature_name))
    }
}
