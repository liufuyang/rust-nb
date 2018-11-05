extern crate regex;

use regex::Regex;
use std::collections::{HashMap, HashSet};

///
#[derive(Debug)]
pub struct Feature {
    // setting `is_text` as true will considering this feature as a multinomial feature and do word counting on feature.value.
    // setting `is_text` as true will considering this feature as categorical feature and will use feature.value as whole word with count 1
    pub is_text: bool,

    pub name: String,
    pub value: String,
}

pub struct Model<T: ModelStore> {
    model_store: T,
    regex: Regex,
}

impl<T: ModelStore> Model<T> {
    fn cal_log_prob(
        &self,
        model_name: &str,
        feature_name: &str,
        outcome: &str,
        count_of_unique_word: usize,
        count_of_all_word_in_class: usize,
        count_of_word: usize,
        word: &str,
    ) -> f64 {
        let count_of_word_in_class =
            self.model_store
                .get_count_of_word_in_class(model_name, feature_name, outcome, word);
        log_prob(
            count_of_word,
            count_of_word_in_class,
            count_of_all_word_in_class,
            count_of_unique_word,
        )
    }

    pub fn predict(
        &self,
        model_name: &str,
        features: &Vec<Feature>,
    ) -> Option<HashMap<String, f64>> {
        let mut result = HashMap::new();

        let outcomes = self.model_store.get_all_classes(model_name)?;

        for outcome in outcomes {
            let priors_count_of_class = self
                .model_store
                .get_priors_count_of_class(model_name, outcome);
            let total_data_count = self.model_store.get_total_data_count(model_name);

            let mut lp = 0.0;
            let tmp_hash_set = HashSet::new();

            for feature in features {
                let known_features_in_table = match self
                    .model_store
                    .get_appeared_words(model_name, &feature.name)
                {
                    Some(set) => set,
                    None => &tmp_hash_set,
                };
                let count_of_unique_word = known_features_in_table.len();
                let count_of_all_word_in_class = self.model_store.get_count_of_all_word_in_class(
                    model_name,
                    &feature.name,
                    &outcome,
                );

                if feature.is_text {
                    let word_counts_for_current_feature = count(&feature.value, &self.regex);
                    let current_words_set: HashSet<String> =
                        word_counts_for_current_feature.keys().cloned().collect();

                    let known_words: HashSet<&String> = current_words_set
                        .intersection(&known_features_in_table)
                        .collect();

                    for word in known_words {
                        let count = match word_counts_for_current_feature.get(word) {
                            Some(v) => *v,
                            None => 0,
                        };

                        lp += self.cal_log_prob(
                            model_name,
                            &feature.name,
                            outcome,
                            count_of_unique_word,
                            count_of_all_word_in_class,
                            count,
                            word,
                        )
                    }
                } else {
                    if known_features_in_table.contains(&feature.value) {
                        lp += self.cal_log_prob(
                            model_name,
                            &feature.name,
                            outcome,
                            count_of_unique_word,
                            count_of_all_word_in_class,
                            1,
                            &feature.value,
                        )
                    }
                }
            }

            let final_log_p =
                (priors_count_of_class as f64).ln() - (total_data_count as f64).ln() + lp;
            result.insert(outcome.to_owned(), final_log_p);
        }

        Some(normalize(result))
    }

    pub fn train(&mut self, model_name: &str, outcome_feature_pairs: Vec<(String, Vec<Feature>)>) {
        let mut c = 0;

        for (outcome, features) in outcome_feature_pairs {
            c += 1;
            println!("{}", c);

            for feature in features {
                self.model_store
                    .add_to_priors_count_of_class(model_name, &outcome, 1);
                self.model_store.add_to_total_data_count(model_name, 1);

                if feature.is_text {
                    let word_counts = count(&feature.value, &self.regex);
                    for (word, count) in word_counts {
                        self.model_store.add_to_count_of_word_in_class(
                            model_name,
                            &feature.name,
                            &outcome,
                            &word,
                            count,
                        );

                        self.model_store.add_to_count_of_all_word_in_class(
                            model_name,
                            &feature.name,
                            &outcome,
                            count,
                        )
                    }
                } else {
                    self.model_store.add_to_count_of_word_in_class(
                        model_name,
                        &feature.name,
                        &outcome,
                        &feature.value,
                        1,
                    );
                    self.model_store.add_to_count_of_all_word_in_class(
                        model_name,
                        &feature.name,
                        &outcome,
                        1,
                    )
                }
            }
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

#[derive(Debug)]
pub struct ModelHashMapStore {
    map: HashMap<String, usize>,
    class_map: HashMap<String, HashSet<String>>, // model_name to list of class
    words_appeared_map: HashMap<String, HashSet<String>>, // model_name|feature_name to list of appeared words
    word_count_map: HashMap<String, usize>, // a very large hash map for look up word counts
    words_in_class_count_map: HashMap<String, usize>, // a hash map for look up word counts for each feature and class
}

impl Model<ModelHashMapStore> {
    pub fn new() -> Model<ModelHashMapStore> {
        Model::<ModelHashMapStore> {
            model_store: ModelHashMapStore::new(),
            regex: Regex::new(r"[^a-zA-Z]+").unwrap(), // only keep any kind of letter from any language, others become space
        }
    }

    pub fn new_large() -> Model<ModelHashMapStore> {
        Model::<ModelHashMapStore> {
            model_store: ModelHashMapStore::new_large(),
            regex: Regex::new(r"[^a-zA-Z]+").unwrap(), // only keep any kind of letter from any language, others become space
        }
    }

    pub fn get_store(&self) -> &ModelHashMapStore {
        &self.model_store
    }
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
        let key = format!("{}|%{}%|%{}", model_name, feature_name, c);
        *self
            .words_in_class_count_map
            .get(&key)
            .unwrap_or_else(|| &0)
    }

    fn get_appeared_words(&self, model_name: &str, feature_name: &str) -> Option<&HashSet<String>> {
        self.words_appeared_map
            .get(&format!("{}|%{}", model_name, feature_name))
    }
}

///
/// private util functions
///
fn count(text: &str, regex: &Regex) -> HashMap<String, usize> {
    let text = text.to_lowercase();
    let text = regex.replace_all(&text, " ");

    let words = text.split(" ");

    // words.iter().fold(HashMap::new(), |mut acc, w| {
    //     *acc.entry(w.to_string()).or_insert(0) += 1; // seems a slow operation?
    //     acc
    // })

    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in words {
        *counts.entry(word.to_owned()).or_insert(0) += 1;
    }
    // TODO: delete later
    // counts.insert("good".to_owned(), 1);
    return counts;
}

fn log_prob(count: usize, c_f_c: usize, c_c: usize, num_of_unique_word: usize) -> f64 {
    let pseudo_count = 1.0;

    let count = count as f64;
    let c_f_c = c_f_c as f64;
    let c_c = c_c as f64;
    let num_of_unique_word = num_of_unique_word as f64;

    count * ((c_f_c + pseudo_count).ln() - (c_c + num_of_unique_word * pseudo_count).ln())
}

fn normalize(mut predictions: HashMap<String, f64>) -> HashMap<String, f64> {
    let max = &predictions
        .values()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .clone();

    for (_, v) in &mut predictions {
        *v = (*v - max).exp();
    }

    let norm: f64 = predictions.values().sum();

    for (_, v) in &mut predictions {
        *v = *v / norm;
    }

    predictions
}

///
///
///
#[test]
fn count_works() {
    let regex = Regex::new(r"[^a-zA-Z]+").unwrap();
    let result = count("This is good good ... Rust rust RUST", &regex);
    assert_eq!(2, result["good"]);
    assert_eq!(1, result["this"]);
    assert_eq!(1, result["is"]);
    assert_eq!(3, result["rust"]);
}

#[test]
fn normalize_works() {
    let mut map = HashMap::new();
    map.insert("a".to_owned(), 1.0);
    map.insert("b".to_owned(), 5.0);

    let map = normalize(map);
    assert_eq!(0.017986209962091555, *map.get("a").unwrap());
    assert_eq!(0.9820137900379085, *map.get("b").unwrap());
}
