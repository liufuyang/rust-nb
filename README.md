# rust-bn
<a href="https://crates.io/crates/rust_nb"><img src="https://img.shields.io/crates/v/rust_nb.svg"></a>
<a href="https://travis-ci.org/liufuyang/rust-nb"><img src="https://travis-ci.org/liufuyang/rust-nb.svg?branch=master"></a>

A simple Naive Bayes Model in Rust. 

Basic idea is from [blayze](https://github.com/Tradeshift/blayze) but
`rust-bn` is a Rust implementation and use a very simple 
Rust `HashMap<String, f64>` to save model in memory. 

Later it should be very simple to enhance it with a hard drive
key-value store to make model persist.

## How To Use

Simply checkout this repo and run some examples locally:

```
git clone git@github.com:liufuyang/rust-nb.git

cargo run --example spam

# or run a more complex example, use --release to speed up train/test process
cargo run --example 20newsgroup_stopwords --release
```

And then you can modify those examples in the `examples` folder
and perhaps from there build your own models.

Or you can use this package in your application by setting in Cargo.toml:
```
[dependencies]
...
rust_nb = "0.1.0"
```

Just take make a main function looks like below. See how a simple email spam model might look like when you train and predict on it.

```rust
extern crate rust_nb;

use rust_nb::{Feature, FeatureType, Model};

fn main() {
    let mut model = Model::new();

    let input_train = vec![
        (
            "spam".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Text,
                    name: "email.body".to_owned(),
                    value: "Good day dear beneficiary. This is Secretary to president of Benin republic is writing this email ... heritage, tax, dollars, money, credit card...".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "email.domain".to_owned(),
                    value: "evil.com".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "email.n_words".to_owned(),
                    value: "482".to_owned(),
                },
            ],
        ),
        (
            "not spam".to_owned(),
            vec![
                Feature {
                    feature_type: FeatureType::Text,
                    name: "email.body".to_owned(),
                    value: "Hey bro, how's work these days, wanna join me for hotpot next week?".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Category,
                    name: "email.domain".to_owned(),
                    value: "gmail.com".to_owned(),
                },
                Feature {
                    feature_type: FeatureType::Gaussian,
                    name: "email.n_words".to_owned(),
                    value: "42".to_owned(),
                },
            ],
        ),
    ];

    model.train("Spam checker", &input_train);

    // test example 1
    let result = model.predict(
        "Spam checker",
        &vec![
            Feature {
                feature_type: FeatureType::Text,
                name: "email.body".to_owned(),
                value: "Hey bro, This is Secretary to president want to give you some money. Please give me your credit card number ..."
                    .to_owned(),
            },
            Feature {
                feature_type: FeatureType::Category,
                name: "email.domain".to_owned(),
                value: "example.com".to_owned(),
            },
            Feature {
                feature_type: FeatureType::Gaussian,
                name: "email.n_words".to_owned(),
                value: "288".to_owned(),
            },
        ],
    );

    println!("{:?}\n", result);
    assert!(result.get("spam").unwrap().abs() > 0.9);
    // result will be:
    // {"not spam": 0.04228956359881729, "spam": 0.9577104364011828}

    // test example 2
    let result = model.predict(
        "Spam checker",
        &vec![
            Feature {
                feature_type: FeatureType::Text,
                name: "email.body".to_owned(),
                value: "Hey bro, hotpot again?".to_owned(),
            },
            Feature {
                feature_type: FeatureType::Category,
                name: "email.domain".to_owned(),
                value: "gmail.com".to_owned(),
            },
            Feature {
                feature_type: FeatureType::Gaussian,
                name: "email.n_words".to_owned(),
                value: "10".to_owned(),
            },
        ],
    );

    println!("{:?}\n", result);
    assert!(result.get("not spam").unwrap().abs() > 0.9);
    // result will be:
    // {"spam": 0.03786816269284711, "not spam": 0.9621318373071529}
}
```

## About Naive Bayes Model (and how to understand the code)

Firstly let's take a look at the Bayes equations for only 2 classes and a feature

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_1&space;|&space;x)&space;=&space;\frac{&space;p(x&space;|&space;c_1)&space;p(c_1)&space;}{&space;p(x&space;|&space;c_1)&space;p(c_1)&space;&plus;&space;p(x&space;|&space;c_2)&space;p(c_2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_1&space;|&space;x)&space;=&space;\frac{&space;p(x&space;|&space;c_1)&space;p(c_1)&space;}{&space;p(x&space;|&space;c_1)&space;p(c_1)&space;&plus;&space;p(x&space;|&space;c_2)&space;p(c_2)}" title="p(c_1 | x) = \frac{ p(x | c_1) p(c_1) }{ p(x | c_1) p(c_1) + p(x | c_2) p(c_2)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_2&space;|&space;x)&space;=&space;\frac{&space;p(x&space;|&space;c_2)&space;p(c_2)&space;}{&space;p(x&space;|&space;c_1)&space;p(c_1)&space;&plus;&space;p(x&space;|&space;c_2)&space;p(c_2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_2&space;|&space;x)&space;=&space;\frac{&space;p(x&space;|&space;c_2)&space;p(c_2)&space;}{&space;p(x&space;|&space;c_1)&space;p(c_1)&space;&plus;&space;p(x&space;|&space;c_2)&space;p(c_2)}" title="p(c_2 | x) = \frac{ p(x | c_2) p(c_2) }{ p(x | c_1) p(c_1) + p(x | c_2) p(c_2)}" /></a>

As we can see the denominator is same for both probabilities of class 1 and 2 based on input x (and they sum equal to 1).

Thus we could simply only focusing calculating only the numerator part for each classes then normalized them all in the end
to get probabilities for each class prediction.

This also generalize to the classes number greater than 2 as well.

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_n&space;|&space;x)&space;<=&space;{&space;p(x&space;|&space;c_n)&space;p(c_n)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_n&space;|&space;x)&space;<=&space;{&space;p(x&space;|&space;c_n)&space;p(c_n)&space;}" title="p(c_n | x) <= { p(x | c_n) p(c_n) }" /></a>

Note here we use `<=` notation meaning we can infer `p(c_n | x)` by the value on the right later on, after we have all 
calculations of classes with number index as `n.

Now expand this to situations where we have multiple features, index of features noted with `i`, and let `X = x_1, x_2, ... x_i`,
we have:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_n&space;|&space;X)&space;<=&space;{&space;p(X&space;|&space;c_n)&space;p(c_n)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_n&space;|&space;X)&space;<=&space;{&space;p(X&space;|&space;c_n)&space;p(c_n)&space;}" title="p(c_n | X) <= { p(X | c_n) p(c_n) }" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_n&space;|&space;X)&space;<=&space;{&space;p(x_1,&space;x_2,&space;...&space;x_i&space;|&space;c_n)&space;p(c_n)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_n&space;|&space;X)&space;<=&space;{&space;p(x_1,&space;x_2,&space;...&space;x_i&space;|&space;c_n)&space;p(c_n)&space;}" title="p(c_n | X) <= { p(x_1, x_2, ... x_i | c_n) p(c_n) }" /></a>

As "Naive" way of thinking, each feature appearance x_i is independent, so we could have:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_n&space;|&space;X)&space;<=&space;p(c_n)&space;\:&space;p(x_1|&space;c_n)&space;\,&space;p(x_2|&space;c_n)&space;\,&space;p(x_3|&space;c_n)&space;\,...\,&space;p(x_i&space;|&space;c_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_n&space;|&space;X)&space;<=&space;p(c_n)&space;\:&space;p(x_1|&space;c_n)&space;\,&space;p(x_2|&space;c_n)&space;\,&space;p(x_3|&space;c_n)&space;\,...\,&space;p(x_i&space;|&space;c_n)" title="p(c_n | X) <= p(c_n) \: p(x_1| c_n) \, p(x_2| c_n) \, p(x_3| c_n) \,...\, p(x_i | c_n)" /></a>

## Feature type: Multinomial and Categorical

Currently we support two feature type:
* Categorical: each x_i in above equations are different value
* Multinomial (a.k.a Text feature): each x_i in above equations could be the same. For example, in the case of counting words to predict document class,
a word `apple` as `x_i` can appear multiple times, let's denote it as `t_i` (as in code it's called `inputFeatureCounts`.)

One can also think of `Categorical` feature being as `Multinomial` feature but all `t_i` is 1. 

So we for now just look at equations for `Multinomial`. Suppose now our `x_i` as unique words, the equations becomes as:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_n&space;|&space;X)&space;<=&space;p(c_n)&space;\:&space;p(x_1|&space;c_n)^{t_1}&space;\,&space;p(x_2|&space;c_n)^{t_2}&space;\,&space;p(x_3|&space;c_n)^{t_3}&space;\,...\,&space;p(x_i&space;|&space;c_n)&space;^{t_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_n&space;|&space;X)&space;<=&space;p(c_n)&space;\:&space;p(x_1|&space;c_n)^{t_1}&space;\,&space;p(x_2|&space;c_n)^{t_2}&space;\,&space;p(x_3|&space;c_n)^{t_3}&space;\,...\,&space;p(x_i&space;|&space;c_n)&space;^{t_i}" title="p(c_n | X) <= p(c_n) \: p(x_1| c_n)^{t_1} \, p(x_2| c_n)^{t_2} \, p(x_3| c_n)^{t_3} \,...\, p(x_i | c_n) ^{t_i}" /></a>

There are many multiplications with values less than one. To prevent the number gets to small to be presented as double 
in computers, we can calculate the log value on each side instead:

<a href="https://www.codecogs.com/eqnedit.php?latex=log(p(c_n&space;|&space;X))&space;<=&space;log(&space;p(c_n))&space;&plus;&space;\:&space;t_1&space;log(&space;p(x_1|&space;c_n))&space;&plus;&space;\:&space;t_2&space;log(p(x_2|&space;c_n))&space;&plus;&space;\:&space;t_3&space;log(&space;p(x_3|&space;c_n))&space;\,...\,&space;&plus;&space;\:&space;t_i&space;log(p(x_i&space;|&space;c_n)&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log(p(c_n&space;|&space;X))&space;<=&space;log(&space;p(c_n))&space;&plus;&space;\:&space;t_1&space;log(&space;p(x_1|&space;c_n))&space;&plus;&space;\:&space;t_2&space;log(p(x_2|&space;c_n))&space;&plus;&space;\:&space;t_3&space;log(&space;p(x_3|&space;c_n))&space;\,...\,&space;&plus;&space;\:&space;t_i&space;log(p(x_i&space;|&space;c_n)&space;)" title="log(p(c_n | X)) <= log( p(c_n)) + \: t_1 log( p(x_1| c_n)) + \: t_2 log(p(x_2| c_n)) + \: t_3 log( p(x_3| c_n)) \,...\, + \: t_i log(p(x_i | c_n) )" /></a>

or

<a href="https://www.codecogs.com/eqnedit.php?latex=log(p(c_n&space;|&space;X))&space;<=&space;log(&space;p(c_n))&space;&plus;&space;\:&space;\sum_{i}&space;t_i&space;log(p(x_i|&space;c_n))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log(p(c_n&space;|&space;X))&space;<=&space;log(&space;p(c_n))&space;&plus;&space;\:&space;\sum_{i}&space;t_i&space;log(p(x_i|&space;c_n))" title="log(p(c_n | X)) <= log( p(c_n)) + \: \sum_{i} t_i log(p(x_i| c_n))" /></a>

To calculate the priors `p(c_n)` and conditional probabilities `p(x_i| c_n)`:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(c_n)&space;=&space;\frac{N_{c_n}}{N}&space;,\:&space;\:&space;p(x_i|&space;c_n)&space;=&space;\frac{count(x_i,&space;c_n)&space;&plus;&space;\epsilon&space;}{count(c_n)&space;&plus;&space;|V|&space;*&space;\epsilon&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(c_n)&space;=&space;\frac{N_{c_n}}{N}&space;,\:&space;\:&space;p(x_i|&space;c_n)&space;=&space;\frac{count(x_i,&space;c_n)&space;&plus;&space;\epsilon&space;}{count(c_n)&space;&plus;&space;|V|&space;*&space;\epsilon&space;}" title="p(c_n) = \frac{N_{c_n}}{N} ,\: \: p(x_i| c_n) = \frac{count(x_i, c_n) + \epsilon }{count(c_n) + |V| * \epsilon }" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;log(p(c_n&space;|&space;X))&space;<=&space;\:&space;&&space;log(&space;N_{c_n})&space;-&space;log(N)&space;&plus;\\&space;&&space;\sum_{i}&space;t_i&space;(log(count(x_i,&space;c_n)&space;&plus;&space;\epsilon)&space;-&space;log(count(c_n)&space;&plus;&space;|V|&space;*&space;\epsilon))&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;log(p(c_n&space;|&space;X))&space;<=&space;\:&space;&&space;log(&space;N_{c_n})&space;-&space;log(N)&space;&plus;\\&space;&&space;\sum_{i}&space;t_i&space;(log(count(x_i,&space;c_n)&space;&plus;&space;\epsilon)&space;-&space;log(count(c_n)&space;&plus;&space;|V|&space;*&space;\epsilon))&space;\end{align*}" title="\begin{align*} log(p(c_n | X)) <= \: & log( N_{c_n}) - log(N) +\\ & \sum_{i} t_i (log(count(x_i, c_n) + \epsilon) - log(count(c_n) + |V| * \epsilon)) \end{align*}" /></a>

So in the end we have those things need to calculate during training and predicting:
* During Training and Predicting, save or access these parameters:
    * `N_cn` : prior count of class c_n. Calculated via function `logPrior` in code.
    * `N`: sum of all prior count `N_cn` of all class c_n. Calculated via function `logPrior` in code.
    * `count(x_i, c_n)`: count of word/feature i's appearance in class c_n `countFeatureAppearsInOutcome`
    * `count(c_n)`: count of total numbers of word/feature appeared in class c_n `totalFeatureCountInOutcome`
    * `|V|`: count of the unique words/feature/vocabulary among all classes. In code it is called `numOfUniqueFeaturesSeen`
* Only During Predicting, also calculate:
    * `t_i`: number of time word/feature i appears in the incoming data for prediction. In code it is called `inputFeatureCounts`
* Constant:
    * `epsilon`: pseudocount, no probability is ever set to be exactly zero. By default we set it as 1, This way of regularizing naive Bayes is called Laplace smoothing when the pseudocount is one

![pic-1](docs/pics/naive-bayes-1.png)