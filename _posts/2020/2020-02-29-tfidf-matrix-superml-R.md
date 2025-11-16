---
title: "How to create TF-IDF matrix using ngrams in R?"
date: '2020-02-29'
tags:
- machine learning
- r
- superml
excerpt: Create tfidf matrix in R just like using scikit-learn
header:
  teaser: /assets/images/R_logo.png
  overlay_image: /assets/images/R_logo.png
  overlay_filter: 0.5
---
## Introduction

Term Document/Inverse Document Frequency(TF-IDF) is a powerful text analysis technique to find similar documents based their vector representations. In simple words, it weights each token, not only on how many times it has occured in a particular document, but also across all set of documents (also called as corpus), to ensure that we put a lower weight to a token if it occurs too frequently (like stopwords).

Let's see how we can create tf-idf matrix using ngrams. Later, we'll train a simple random forest model on features generated from tf-idf. 

You can install the package by doing:


```r
install.packages("superml", dependencies = TRUE)
```



## TF-IDF Matrix with superml

Superml following a scikit-learn style api, so if you are familiar with it, superml should be easy for you. In case you are new to it, just follow the explanation below. Superml is based on C++ optimised functions, hence it should be quite fast as well. 

First, we'll try to get a dummy dataset.  


```r
# download data, takes a few seconds
data <- readLines("https://www.r-bloggers.com/wp-content/uploads/2016/01/vent.txt")
df <- data.frame(data)
colnames(df) <- 'text'

# also let's add a target variable, so later we can train a model on tfidf features
df$target <- sample(c(0,1), size = nrow(df), replace = T)
```


Now, we create tfidf features.


```r
library(superml)
```

```
## Loading required package: R6
```

```r
tf <- TfIdfVectorizer$new(ngram_range = c(1,3), max_df = 0.7)
tf_feats <- tf$fit_transform(df$text)
```

**Explanation**

*  `ngram_range` as `c(1,3)` means 1 is the minimum length of ngram and 3 is the maximum length of ngram.
*  `max_df = 0.7` means ignore token (or word) should appear in more than 40% of the documents. 
* `$new()` method initialises the instance of `TfIdfVectorizer` class.
* `$fit_transform()` return a matrix of tfidf features.

Let's see how the matrix looks like:


```r
dim(tf_feats)
```

```
## [1]   83 2059
```

We get 2059 features for the given data. Let's look at the tokens.


```r
colnames(tf_feats)[1:30]
```

```
##  [1] "s"             "t"             "charleston"    "west"         
##  [5] "don"           "people"        "don t"         "see"          
##  [9] "virginia"      "west virginia" "just"          "news"         
## [13] "please"        " liar"         " s"            "liar"         
## [17] "obama"         "republicans"   "want"          "can"          
## [21] "good"          "hillary"       "problem"       "re"           
## [25] "show"          "trump"         "us"            " liar "       
## [29] "breaking"      "breaking news"
```

We see some text processing would be great before passing calculating the tfidf features. Let's tke a look at the matrix.


```r
head(tf_feats[1:3,50:60])
```

```
##       problem   anyone ben ben carson c carson clear     deal drive every
## [1,]        0 0.000000   0          0 0      0     0 0.000000     0     0
## [2,]        0 0.166419   0          0 0      0     0 0.166419     0     0
## [3,]        0 0.000000   0          0 0      0     0 0.000000     0     0
##          iran
## [1,] 0.000000
## [2,] 0.166419
## [3,] 0.000000
```

Now, let's train a random forest model on these features. But, before that let's split the data into train and test.


```r
n_rows <- nrow(df) %/% 2
train <- data.frame(tf_feats[1:n_rows,])
train$target_var <- df[1:n_rows,]$target

test <- data.frame(tf_feats[n_rows:nrow(df),])
test$target_var <- df[n_rows:nrow(df),'target']

dim(train)
```

```
## [1]   41 2060
```

```r
dim(test)
```

```
## [1]   43 2060
```

Ideally, we should have dont stratified sampling because the target variable is binary. Here, the idea is to demonstrate the use of superml for building machine learning models on text data.

Let's train a random forest model. All machine learning models in superml are known as `Trainer`.

```r
rf <- RFTrainer$new(max_features = 50, n_estimators = 100)
rf$fit(train, 'target_var')

preds <- rf$predict(test)
print(preds[1:10])
```

```
##  [1] 0 0 0 0 0 0 0 0 0 0
## Levels: 0 1
```

## Summary

In this tutorial, we learned to train a random forest model using tfidf ngram features in R. Next, we'll see how to create a simple ngram bag of words features model in R.




