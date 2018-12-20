extern crate rayon;
extern crate regex;
#[macro_use]
extern crate serde_derive;

use rayon::prelude::*;
use regex::Regex;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use std::marker::Sync;

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub feature_type: FeatureType,
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    /// A multinomial feature and do word counting on feature.value.
    Text,

    /// A categorical feature and will use feature.value as whole word with count 1
    Category,

    /// A gaussian feature that can take in continues values (f.g. 1.0, 4.2).
    /// It is calculated as an proximate gaussian distribution, with sigma calculated simply via
    /// (max - min) * default_gaussian_sigma_factor, with default_gaussian_sigma_factor as 1.0/6.0 by default
    /// and the sigma is always the same for the same Gaussian feature that is among different classes.
    /// Use this features in case you have other features such as Text or Category at the same time.
    /// Comparing with GaussianStd type below, this feature is more stable and practically more useful.
    /// Noticing that this Gaussian will be heavily influenced by prior prob if you have very unbalanced
    /// classes in training time.
    Gaussian,

    /// A standard gaussian feature that can take in continues values (f.g. 1.0, 4.2).
    /// Sigma is calculated based on the known values of this feature of a certain class.
    /// Be sure your input is really (or very close) to a gaussian
    /// distribution otherwise this feature might easily dominate other features.
    /// If you are not sure, use the `Gaussian` type above.
    GaussianStd,
}

pub trait ModelStore {
    fn map_add(&mut self, model_name: &str, prefix: &str, v: f64) -> f64;

    fn map_add_with_default(&mut self, model_name: &str, prefix: &str, v: f64, default: f64)
        -> f64;

    fn map_get(&self, model_name: &str, prefix: &str) -> f64;

    fn save_class(&mut self, model_name: &str, class: &str);

    fn get_all_classes(&self, model_name: &str) -> Option<BTreeSet<String>>;
}

pub struct Model<T: ModelStore + Sync> {
    // m2 init value, std at the beginning will be sqrt(default_gaussian_m2)
    default_gaussian_m2: f64,
    default_gaussian_sigma_factor: f64,
    model_store: T,
    regex: Regex,
    // Regex used on features, matches will be replaces by empty space. By default we use r"[^a-zA-Z]+" to replace every char not in English as space
    stop_words: Option<HashSet<String>>,
    pseudo_count: f64, // pseudo count of words in text feature
    prior_factor: f64, // influence of prior prob, default 1.0
}

impl<T: ModelStore + Sync> Model<T> {
    pub fn with_stop_words_file(mut self, stop_words_file: &str) -> Self {
        let f = File::open(stop_words_file).unwrap();
        let f = BufReader::new(&f);

        let mut stop_words = HashSet::new();

        for line in f.lines() {
            let line = line.unwrap();
            stop_words.insert(line);
        }

        self.stop_words = Some(stop_words);

        self
    }

    pub fn with_default_gaussian_m2(mut self, default_gaussian_m2: f64) -> Self {
        self.default_gaussian_m2 = default_gaussian_m2;
        self
    }

    pub fn with_pseudo_count(mut self, pseudo_count: f64) -> Self {
        self.pseudo_count = pseudo_count;
        self
    }

    pub fn with_prior_factor(mut self, prior_factor: f64) -> Self {
        self.prior_factor = prior_factor;
        self
    }

    pub fn with_default_gaussian_sigma_factor(
        mut self,
        default_gaussian_sigma_factor: f64,
    ) -> Self {
        self.default_gaussian_sigma_factor = default_gaussian_sigma_factor;
        self
    }

    pub fn train(&mut self, model_name: &str, class_feature_pairs: &[(String, Vec<Feature>)]) {
        for (class, features) in class_feature_pairs {
            for f in features {
                self.add_to_priors_count_of_class(model_name, &class, 1.0);
                self.add_to_total_data_count(model_name, 1.0);

                match f.feature_type {
                    FeatureType::Text => {
                        let feature_value = clean_text(&f.value, &self.regex);
                        let word_counts = count(&feature_value, &self.stop_words);
                        for (word, count) in word_counts {
                            self.add_to_count_of_word_in_class(
                                model_name,
                                &f.name,
                                &class,
                                word,
                                count as f64,
                            );
                            self.add_to_count_of_all_word_in_class(
                                model_name,
                                &f.name,
                                &class,
                                count as f64,
                            )
                        }
                    }
                    FeatureType::Category => {
                        self.add_to_count_of_word_in_class(
                            model_name, &f.name, &class, &f.value, 1.0,
                        );
                        self.add_to_count_of_all_word_in_class(model_name, &f.name, &class, 1.0)
                    }
                    FeatureType::GaussianStd => match &f.value.parse::<f64>() {
                        Ok(v) => self.gaussian_std_add(model_name, &f.name, &class, v.clone()),
                        Err(e) => {
                            println!(
                                "FeatureType::GaussianStd parsing '{}' gives an error: {}. ",
                                &f.value, e,
                            );
                            ()
                        }
                    },
                    FeatureType::Gaussian => match &f.value.parse::<f64>() {
                        Ok(v) => self.gaussian_add(model_name, &f.name, &class, v.clone()),
                        Err(e) => {
                            println!(
                                "FeatureType::Gaussian parsing '{}' gives an error: {}. ",
                                &f.value, e,
                            );
                            ()
                        }
                    },
                }
            }
        }
    }

    pub fn predict(&self, model_name: &str, features: &[Feature]) -> HashMap<String, f64> {
        self.predict_batch(&model_name, &[Vec::from(features)])
            .remove(0)
    }

    pub fn predict_batch(
        &self,
        model_name: &str,
        features_vec: &[Vec<Feature>],
    ) -> Vec<HashMap<String, f64>> {
        let outcomes = match self.model_store.get_all_classes(model_name) {
            Some(c) => c,
            None => return vec![HashMap::new()],
        };

        let total_data_count = self.get_total_data_count(model_name);

        let outcomes_to_priors_counts: Vec<(String, f64)> = outcomes
            .into_iter()
            .map(|outcome| {
                let priors_count_of_class = self.get_priors_count_of_class(model_name, &outcome);
                (outcome, priors_count_of_class)
            })
            .collect();

        let results: Vec<HashMap<String, f64>> = features_vec
            .par_iter() // use rayon for predicting in parallel
            .map(|features| {
                let mut result = HashMap::new();

                for (outcome, priors_count_of_class) in &outcomes_to_priors_counts {
                    let mut lp = 0.0;

                    for f in features {
                        let count_of_unique_words_in_feature =
                            self.get_count_of_unique_words_in_feature(model_name, &f.name);

                        let count_of_all_word_in_class =
                            self.get_count_of_all_word_in_class(model_name, &f.name, &outcome);

                        match f.feature_type {
                            FeatureType::Text => {
                                let feature_value = clean_text(&f.value, &self.regex);
                                for (word, count) in count(&feature_value, &self.stop_words) {
                                    if self.is_word_appeared_in_feature(model_name, &f.name, &word)
                                    {
                                        lp += self.cal_log_prob(
                                            model_name,
                                            &f.name,
                                            &outcome,
                                            count_of_unique_words_in_feature,
                                            count_of_all_word_in_class,
                                            count as f64,
                                            word,
                                        )
                                    }
                                }
                            }
                            FeatureType::Category => {
                                if self.is_word_appeared_in_feature(model_name, &f.name, &f.value) {
                                    lp += self.cal_log_prob(
                                        model_name,
                                        &f.name,
                                        &outcome,
                                        count_of_unique_words_in_feature,
                                        count_of_all_word_in_class,
                                        1.0,
                                        &f.value,
                                    )
                                }
                            }
                            FeatureType::GaussianStd => match &f.value.parse::<f64>() {
                                Ok(v) => {
                                    lp += self.cal_log_prob_gaussian_std(
                                        model_name,
                                        &f.name,
                                        &outcome,
                                        v.clone(),
                                    )
                                }
                                Err(e) => {
                                    println!(
                                        "FeatureType::GaussianStd parsing '{}' gives an error: {}. ",
                                        &f.value, e,
                                    );
                                    ()
                                }
                            },
                            FeatureType::Gaussian => match &f.value.parse::<f64>() {
                                Ok(v) => {
                                    lp += self.cal_log_prob_gaussian(
                                        model_name,
                                        &f.name,
                                        &outcome,
                                        v.clone(),
                                    )
                                }
                                Err(e) => {
                                    println!(
                                        "FeatureType::Gaussian parsing '{}' gives an error: {}. ",
                                        &f.value, e,
                                    );
                                    ()
                                }
                            },
                        };
                    }

                    let final_log_p = self.prior_factor
                        * ((priors_count_of_class).ln() - (total_data_count).ln())
                        + lp;
                    result.insert(outcome.to_owned(), final_log_p);
                }

                normalize(result)
            })
            .collect();

        results
    }

    fn add_to_priors_count_of_class(&mut self, model_name: &str, c: &str, v: f64) {
        self.model_store
            .map_add(model_name, &format!("_Ncn|{}", c), v); // _Ncn: priors_count_of_class

        self.model_store.save_class(model_name, c);
    }

    fn get_priors_count_of_class(&self, model_name: &str, c: &str) -> f64 {
        self.model_store.map_get(model_name, &format!("_Ncn|{}", c)) // _Ncn: priors_count_of_class
    }

    fn add_to_total_data_count(&mut self, model_name: &str, v: f64) {
        self.model_store.map_add(model_name, "_N", v); //_N: sum of all prior count N_cn of all classes c_n.
    }

    fn get_total_data_count(&self, model_name: &str) -> f64 {
        self.model_store.map_get(model_name, "_N") //_N: sum of all prior count N_cn of all classes c_n.
    }

    fn add_to_count_of_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        word: &str,
        v: f64,
    ) {
        self.model_store.map_add(
            model_name,
            &["_c_f_c", feature_name, c, word].join("|"),
            // &format!("_c_f_c|{}|{}|{}", feature_name, c, word),
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
    ) -> f64 {
        self.model_store.map_get(
            model_name,
            &["_c_f_c", feature_name, c, word].join("|"),
            // &format!("_c_f_c|{}|{}|{}", ),
        )
    }

    fn add_to_count_of_all_word_in_class(
        &mut self,
        model_name: &str,
        feature_name: &str,
        c: &str,
        v: f64,
    ) {
        self.model_store
            .map_add(model_name, &["_c_c", feature_name, c].join("|"), v); // &format!("_c_c|{}|{}", feature_name, c)
    }

    fn get_count_of_all_word_in_class(&self, model_name: &str, feature_name: &str, c: &str) -> f64 {
        self.model_store
            .map_get(model_name, &["_c_c", feature_name, c].join("|")) // &format!("_c_c|{}|{}", feature_name, c)
    }

    fn add_unique_word_in_feature(&mut self, model_name: &str, feature_name: &str, word: &str) {
        if !self.is_word_appeared_in_feature(model_name, feature_name, word) {
            self.model_store.map_add(
                model_name,
                &["_Vw", feature_name, word].join("|"), // _Vw: marker for unique word in feature
                // &format!("_Vw|{}|{}", feature_name, word),
                1.0,
            );
            self.model_store
                .map_add(model_name, &format!("_V|{}", feature_name), 1.0);
        }
    }
    fn is_word_appeared_in_feature(
        &self,
        model_name: &str,
        feature_name: &str,
        word: &str,
    ) -> bool {
        0 != self
            .model_store
            .map_get(model_name, &["_Vw", feature_name, word].join("|")) as usize // _Vw: marker for unique word in feature
                                                                                  // &format!("_Vw|{}|{}", feature_name, word)
    }
    fn get_count_of_unique_words_in_feature(&self, model_name: &str, feature_name: &str) -> f64 {
        self.model_store
            .map_get(model_name, &["_V", feature_name].join("|")) // &format!("_V|{}", feature_name)
    }

    ///
    /// GaussianStd session
    ///
    fn gaussian_std_add(
        &mut self,
        model_name: &str,
        feature_name: &str,
        outcome: &str,
        value: f64,
    ) {
        //     count += 1
        //     val delta = x - mean
        //     mean += delta / count
        //     val delta2 = x - mean
        //     m2 += delta * delta2
        let count = self.model_store.map_add(
            model_name,
            &format!("_Gstd_count|{}|{}", feature_name, outcome),
            1.0,
        );

        let mean = self.model_store.map_get(
            model_name,
            &format!("_Gstd_mean|{}|{}", feature_name, outcome),
        );
        let delta = value - mean;

        let mean = self.model_store.map_add(
            model_name,
            &format!("_Gstd_mean|{}|{}", feature_name, outcome),
            delta / count,
        ); // mean += delta / count

        let delta2 = value - mean;

        self.model_store.map_add_with_default(
            model_name,
            &format!("_Gstd_m2|{}|{}", feature_name, outcome),
            delta * delta2,
            self.default_gaussian_m2, // m2 init value, std at the beginning will be sqrt(default_gaussian_m2)
        );
    }

    fn cal_log_prob_gaussian_std(
        &self,
        model_name: &str,
        feature_name: &str,
        outcome: &str,
        value: f64,
    ) -> f64 {
        // private fun logPropabilityOutcome(outcome: Outcome, value: Double): Double {
        // // p(x|mu,sigma)        = 1/sqrt(2*pi*sigma^2)              * exp(-(x-mu)^2/(2*sigma^2))
        // // log(p(x|mu, sigma)   = log(1) - log(sqrt(2*pi*sigma^2))  - (x-mu)^2/(2*sigma^2)
        // //                      = -log(sqrt(2*pi*sigma^2))          - (x-mu)^2/(2*sigma^2)
        // //                      = -log(sigma*sqrt(2*pi))            - (x-mu)^2/(2*sigma^2)
        // //                      = -log(sigma) - log(sqrt(2*pi))     - (x-mu)^2/(2*sigma^2)
        //     val (mu, sigma) = estimators[outcome] ?: return 0.0
        //     if (sigma == 0.0) {
        //         return 0.0
        //     }
        //     return -ln(sigma) - ln(sqrt(2 * PI)) - (value - mu).pow(2).div(2 * sigma.pow(2))
        // }

        let mu = self.model_store.map_get(
            model_name,
            &format!("_Gstd_mean|{}|{}", feature_name, outcome),
        );
        let count = self.model_store.map_get(
            model_name,
            &format!("_Gstd_count|{}|{}", feature_name, outcome),
        );
        let m2 = self.model_store.map_get(
            model_name,
            &format!("_Gstd_m2|{}|{}", feature_name, outcome),
        );

        let mut sigma;
        if count >= 2.0 {
            sigma = (m2 / (count - 1.0)).sqrt();
        } else {
            sigma = 1.0; // simple assumption to make prediction work better even if only trained once
        }
        if sigma < 1.0 {
            sigma = 1.0; // do not allow sigma smaller than ore, prevent over taken other features
                         // when using Gaussian feature here, one should make the input larger than 1
                         // different with blayze, where it lets function returns 0 if happens
        }

        // from Kotlin blayze code:
        // -ln(sigma) - ln(sqrt(2 * PI)) - (value - mu).pow(2).div(2 * sigma.pow(2))
        // switch the last part, google "-log(1+x**2), -(x**2)" to see difference
        -sigma.ln()
            - (2.0 * PI).sqrt().ln()
            - (1.0 + (value - mu).powi(2) / (2.0 * sigma.powi(2))).ln()
    }
    /// end of GaussianStd session

    ///
    /// Gaussian session
    ///
    fn gaussian_add(&mut self, model_name: &str, feature_name: &str, outcome: &str, value: f64) {
        //     count += 1
        //     val delta = x - mean
        //     mean += delta / count
        //     sigma = (max - min) / 6
        let count = self.model_store.map_add(
            model_name,
            &format!("_G_count|{}|{}", feature_name, outcome),
            1.0,
        );

        let mean = self
            .model_store
            .map_get(model_name, &format!("_G_mean|{}|{}", feature_name, outcome));
        let delta = value - mean;

        self.model_store.map_add(
            model_name,
            &format!("_G_mean|{}|{}", feature_name, outcome),
            delta / count,
        ); // mean += delta / count

        // add max
        let max = self
            .model_store
            .map_get(model_name, &format!("_G_max|{}", feature_name));
        if value > max {
            self.model_store
                .map_add(model_name, &format!("_G_max|{}", feature_name), value - max);
        }

        // add min
        let min = self
            .model_store
            .map_get(model_name, &format!("_G_min|{}", feature_name));
        if value < min {
            self.model_store
                .map_add(model_name, &format!("_G_min|{}", feature_name), value - min);
        }
    }

    fn cal_log_prob_gaussian(
        &self,
        model_name: &str,
        feature_name: &str,
        outcome: &str,
        value: f64,
    ) -> f64 {
        let mu = self
            .model_store
            .map_get(model_name, &format!("_G_mean|{}|{}", feature_name, outcome));

        let max = self
            .model_store
            .map_get(model_name, &format!("_G_max|{}", feature_name));
        let min = self
            .model_store
            .map_get(model_name, &format!("_G_min|{}", feature_name));

        let sigma = ((max - min) * self.default_gaussian_sigma_factor).max(1e-3);

        // from Kotlin blayze code:
        // -ln(sigma) - ln(sqrt(2 * PI)) - (value - mu).pow(2).div(2 * sigma.pow(2))
        // switch the last part, google "-log(1+x**2), -(x**2)" to see difference
        -sigma.ln()
            - (2.0 * PI).sqrt().ln()
            - (1.0 + (value - mu).powi(2) / (2.0 * sigma.powi(2))).ln()
    }
    /// end of Gaussian session

    fn cal_log_prob(
        &self,
        model_name: &str,
        feature_name: &str,
        outcome: &str,
        count_of_unique_words_in_feature: f64, // |V|
        count_of_all_word_in_class: f64,
        count_of_word: f64,
        word: &str,
    ) -> f64 {
        let count_of_word_in_class =
            self.get_count_of_word_in_class(model_name, feature_name, outcome, word);

        log_prob(
            count_of_word,                    // t_i
            count_of_word_in_class,           // c_f_c
            count_of_all_word_in_class,       // c_c
            count_of_unique_words_in_feature, // |V|
            self.pseudo_count,
        )
    }
}

// A in memory ModelStore implementation ModelHashMapStore

#[derive(Debug)]
pub struct ModelHashMapStore {
    map: HashMap<String, f64>,
    class_map: HashMap<String, BTreeSet<String>>, // model_name to list of class
}

impl Model<ModelHashMapStore> {
    pub fn new() -> Model<ModelHashMapStore> {
        Model::<ModelHashMapStore> {
            model_store: ModelHashMapStore {
                map: HashMap::new(),
                class_map: HashMap::new(),
            },
            regex: Regex::new(r"[^a-zA-Z]+").unwrap(), // only keep any kind of letter from any language, others become space
            stop_words: None,
            default_gaussian_m2: 0.0,
            default_gaussian_sigma_factor: 1.0 / 6.0,
            pseudo_count: 1.0,
            prior_factor: 1.0,
        }
    }
}

impl ModelStore for ModelHashMapStore {
    fn map_add(&mut self, model_name: &str, prefix: &str, v: f64) -> f64 {
        self.map_add_with_default(model_name, prefix, v, 0.0)
    }

    fn map_add_with_default(
        &mut self,
        model_name: &str,
        prefix: &str,
        v: f64,
        default: f64,
    ) -> f64 {
        let key = format!("{}|{}", model_name, prefix);
        let old_value = self.map.entry(key).or_insert(default);
        *old_value += v;
        *old_value
    }

    fn map_get(&self, model_name: &str, prefix: &str) -> f64 {
        let key = format!("{}|{}", model_name, prefix);
        *self.map.get(&key).unwrap_or_else(|| &0.0)
    }

    fn save_class(&mut self, model_name: &str, class: &str) {
        // add class to class_map
        let class_vec = self
            .class_map
            .entry(model_name.to_string())
            .or_insert(BTreeSet::new());
        class_vec.insert(class.to_string());
    }

    fn get_all_classes(&self, model_name: &str) -> Option<BTreeSet<String>> {
        self.class_map.get(model_name).cloned()
    }
}

///
/// private util functions
///

fn clean_text(text: &str, regex: &Regex) -> String {
    let text = regex.replace_all(&text, " ");
    let text = text.trim().to_lowercase();
    text
}

fn count<'a>(text: &'a str, stop_words: &Option<HashSet<String>>) -> HashMap<&'a str, usize> {
    let counts = text
        .split(" ")
        .filter(|w| match stop_words {
            Some(stop_words_set) => !stop_words_set.contains(*w),
            None => true,
        })
        .fold(HashMap::new(), |mut acc, w| {
            *acc.entry(w).or_insert(0) += 1;
            acc
        });
    counts
}

/// c_f_c: count_of_word_in_class
/// c_c:
/// v: count_of_unique_words_in_feature
fn log_prob(count: f64, c_f_c: f64, c_c: f64, v: f64, pseudo_count: f64) -> f64 {
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
    let result = count("This is good good ... Rust Rust Rust", &None);
    assert_eq!(2, result["good"]);
    assert_eq!(1, result["This"]);
    assert_eq!(1, result["is"]);
    assert_eq!(3, result["Rust"]);
}

#[test]
fn clean_text_works() {
    let text = "This is &/some weird TEXT";

    let cleaned_text = clean_text(text, &Regex::new(r"[^a-zA-Z]+").unwrap());
    assert_eq!("this is some weird text", cleaned_text);
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

#[test]
fn model_hashmap_store_works() {
    let model = Model::new();
    assert_eq!(0, model.get_total_data_count("test_model") as usize);
    let mut model = model;
    model.add_to_total_data_count("test_model", 1.0);
    assert_eq!(1, model.get_total_data_count("test_model") as usize);
    model.add_to_total_data_count("test_model", 10.0);
    assert_eq!(11, model.get_total_data_count("test_model") as usize);
}

#[test]
fn model_count_classes_works() {
    let model = Model::new();
    assert_eq!(
        0,
        model.get_priors_count_of_class("test_model", "class_1") as usize
    );
    assert_eq!(
        0,
        model.get_priors_count_of_class("test_model", "class_2") as usize
    );
    assert_eq!(None, model.model_store.get_all_classes("test_model"));

    let mut model = model;

    model.add_to_priors_count_of_class("test_model", "class_1", 1.0);
    assert_eq!(
        1,
        model.get_priors_count_of_class("test_model", "class_1") as usize
    );
    assert_eq!(
        1,
        model
            .model_store
            .get_all_classes("test_model")
            .unwrap()
            .len()
    );

    model.add_to_priors_count_of_class("test_model", "class_1", 10.0);
    assert_eq!(
        11,
        model.get_priors_count_of_class("test_model", "class_1") as usize
    );
    assert_eq!(
        1,
        model
            .model_store
            .get_all_classes("test_model")
            .unwrap()
            .len()
    );

    model.add_to_priors_count_of_class("test_model", "class_2", 10.0);
    assert_eq!(
        10,
        model.get_priors_count_of_class("test_model", "class_2") as usize
    );
    assert_eq!(
        2,
        model
            .model_store
            .get_all_classes("test_model")
            .unwrap()
            .len()
    );

    model.add_to_priors_count_of_class("test_model", "class_2", 10.0);
    assert_eq!(
        20,
        model.get_priors_count_of_class("test_model", "class_2") as usize
    );
    assert_eq!(
        2,
        model
            .model_store
            .get_all_classes("test_model")
            .unwrap()
            .len()
    );
}
