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
        count_of_unique_words_in_feature: usize, // |V|
        count_of_all_word_in_class: usize,
        count_of_word: usize,
        word: &str,
    ) -> f64 {
        let count_of_word_in_class =
            self.model_store
                .get_count_of_word_in_class(model_name, feature_name, outcome, word);

        log_prob(
            count_of_word,                    // t_i
            count_of_word_in_class,           // c_f_c
            count_of_all_word_in_class,       // c_c
            count_of_unique_words_in_feature, // |V|
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
            // let tmp_hash_set = HashSet::new();

            for feature in features {
                let count_of_unique_words_in_feature = self
                    .model_store
                    .get_count_of_unique_words_in_feature(model_name, &feature.name);

                let count_of_all_word_in_class = self.model_store.get_count_of_all_word_in_class(
                    model_name,
                    &feature.name,
                    &outcome,
                );

                // println!("{}:{}", &outcome, count_of_all_word_in_class);

                if feature.is_text {
                    for (word, count) in count(&feature.value, &self.regex) {
                        if self.model_store.is_word_appeared_in_feature(
                            model_name,
                            &feature.name,
                            &word,
                        ) {
                            lp += self.cal_log_prob(
                                model_name,
                                &feature.name,
                                outcome,
                                count_of_unique_words_in_feature,
                                count_of_all_word_in_class,
                                count,
                                &word,
                            )
                        }
                    }
                } else {
                    if self.model_store.is_word_appeared_in_feature(
                        model_name,
                        &feature.name,
                        &feature.value,
                    ) {
                        lp += self.cal_log_prob(
                            model_name,
                            &feature.name,
                            outcome,
                            count_of_unique_words_in_feature,
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
        for (outcome, features) in outcome_feature_pairs {
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

    fn add_unique_word_in_feature(&mut self, model_name: &str, feature_name: &str, word: &str);
    fn is_word_appeared_in_feature(&self, model_name: &str, feature_name: &str, word: &str)
        -> bool;

    fn get_count_of_unique_words_in_feature(&self, model_name: &str, feature_name: &str) -> usize;
}

// A in memory ModelStore implementation ModelHashMapStore

#[derive(Debug)]
pub struct ModelHashMapStore {
    map: HashMap<String, usize>,
    class_map: HashMap<String, HashSet<String>>, // model_name to list of class
}

impl Model<ModelHashMapStore> {
    pub fn new() -> Model<ModelHashMapStore> {
        Model::<ModelHashMapStore> {
            model_store: ModelHashMapStore::new(),
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
        }
    }

    fn map_add(&mut self, model_name: &str, prefix: &str, v: usize) {
        let key = format!("{}|{}", model_name, prefix);
        let total_count = self.map.entry(key).or_insert(0);
        *total_count += v;
    }

    fn map_get(&self, model_name: &str, prefix: &str) -> usize {
        let key = format!("{}|{}", model_name, prefix);
        *self.map.get(&key).unwrap_or_else(|| &0)
    }
}

impl ModelStore for ModelHashMapStore {
    fn add_to_priors_count_of_class(&mut self, model_name: &str, c: &str, v: usize) {
        self.map_add(model_name, &format!("_Ncn|{}", c), v); // _Ncn: priors_count_of_class

        // add class to class_map
        let class_vec = self
            .class_map
            .entry(model_name.to_string())
            .or_insert(HashSet::new());
        class_vec.insert(c.to_string());
    }

    fn get_priors_count_of_class(&self, model_name: &str, c: &str) -> usize {
        self.map_get(model_name, &format!("_Ncn|{}", c)) // _Ncn: priors_count_of_class
    }

    fn get_all_classes(&self, model_name: &str) -> Option<&HashSet<String>> {
        self.class_map.get(model_name)
    }

    fn add_to_total_data_count(&mut self, model_name: &str, v: usize) {
        self.map_add(model_name, "_N", v); //_N: sum of all prior count N_cn of all classes c_n.
    }

    fn get_total_data_count(&self, model_name: &str) -> usize {
        self.map_get(model_name, "_N") //_N: sum of all prior count N_cn of all classes c_n.
    }

    fn add_to_count_of_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        word: &str,
        v: usize,
    ) {
        self.map_add(
            model_name,
            &format!("_c_f_c|{}|{}|{}", feature_name, c, word),
            v,
        );

        self.add_unique_word_in_feature(model_name, feature_name, word);
    }

    fn get_count_of_word_in_class(
        &self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        word: &str,
    ) -> usize {
        self.map_get(
            model_name,
            &format!("_c_f_c|{}|{}|{}", feature_name, c, word),
        )
    }

    fn add_to_count_of_all_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        v: usize,
    ) {
        let key = format!("{}|_c_c|{}|{}", model_name, feature_name, c);
        let words_count = self.map.entry(key).or_insert(0);
        *words_count += v;
    }

    fn get_count_of_all_word_in_class(
        &self,
        model_name: &str,
        feature_name: &str,
        c: &str,
    ) -> usize {
        let key = format!("{}|_c_c|{}|{}", model_name, feature_name, c);
        *self.map.get(&key).unwrap_or_else(|| &0)
    }

    fn add_unique_word_in_feature(&mut self, model_name: &str, feature_name: &str, word: &str) {
        if !self.is_word_appeared_in_feature(model_name, feature_name, word) {
            self.map_add(
                model_name,
                &format!("_Vw|{}|{}", feature_name, word), // _Vw: marker for unique word in feature
                1,
            );
            self.map_add(model_name, &format!("_V|{}", feature_name), 1);
        }
    }
    fn is_word_appeared_in_feature(
        &self,
        model_name: &str,
        feature_name: &str,
        word: &str,
    ) -> bool {
        0 != self.map_get(model_name, &format!("_Vw|{}|{}", feature_name, word)) // _Vw: marker for unique word in feature
    }
    fn get_count_of_unique_words_in_feature(&self, model_name: &str, feature_name: &str) -> usize {
        self.map_get(model_name, &format!("_V|{}", feature_name))
    }
}

///
/// private util functions
///
fn count(text: &str, regex: &Regex) -> HashMap<String, usize> {
    // let text = text.to_lowercase();
    let text = regex.replace_all(&text, " ");

    let text = text.trim().to_lowercase();
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

/// c_f_c: count_of_word_in_class
/// c_c:
/// v: count_of_unique_words_in_feature
fn log_prob(count: usize, c_f_c: usize, c_c: usize, v: usize) -> f64 {
    let pseudo_count = 1.0;

    let count = count as f64;
    let c_f_c = c_f_c as f64;
    let c_c = c_c as f64;
    let v = v as f64;

    count * ((c_f_c + pseudo_count).ln() - (c_c + v * pseudo_count).ln())
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

    // Noticing an interesting character for normalize:
    // If you add same value onto the original values, the normalized result won't change.
    // We utilize this character on prediction to allow log probability calculated also for those
    // unknown words
    let mut map = HashMap::new();
    map.insert("a".to_owned(), 1.0 + 500.0);
    map.insert("b".to_owned(), 5.0 + 500.0);

    let map = normalize(map);
    assert_eq!(0.017986209962091555, *map.get("a").unwrap());
    assert_eq!(0.9820137900379085, *map.get("b").unwrap());
}
