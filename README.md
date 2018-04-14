# Sentiment Analysis on Movie Reviews

For DAT607 project 4, Text analysis, I chose kaggle project called `Sentiment analysis on movie reviews` from Kaggle.

https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
 
The challenge is to classify the sentiment of sentences from the Rotten Tomatoes dataset.


## Overview

The Rotten Tomatoes movie review dataset is a corpus of movie reviews used for sentiment analysis, originally collected by Pang and Lee [1]. In their work on sentiment treebanks, Socher et al. [2] used Amazon's Mechanical Turk to create fine-grained labels for all parsed phrases in the corpus. This competition presents a chance to benchmark your sentiment-analysis ideas on the Rotten Tomatoes dataset. 
You are asked to label phrases on a scale of five values: negative, somewhat negative, neutral, somewhat positive, positive. 
Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.


### Data

- train.tsv
- test.tsv
- sampleSubmission.csv

The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. 
The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. 
Each Sentence has been parsed into many phrases by the Stanford parser. 
Each phrase has a PhraseId. 
Each sentence has a SentenceId. 
Phrases that are repeated (such as short/common words) are only included once in the data.

train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
test.tsv contains just phrases. You must assign a sentiment label to each phrase.
The sentiment labels are:

* 0 - negative
* 1 - somewhat negative
* 2 - neutral
* 3 - somewhat positive
* 4 - positive


### Preprocessing

#####cleantext1

Use nltk.SnowballStemmer to stem words. e.g.
```
stemmer.stem("disappointments") 
#disappointment
```

#####cleantext2

Create function to manually change certain word. e.g.

```
n`t -> not
not -> no
hopeless -> bad
```

##### one hot encode the phrases

It is possible to use `CountVectorizer` and manually set stopwords, max_features and ngram in order to create a dictionary. 

Visualization is created to see the word frequency in graphic. This could contribute to insights for developing the CountVectorizer.

However, for the better performance, `TfidfVectorizer` was used.

TfidfVectorizer combines all the options of CountVectorizer and TfidfTransformer in a single model.


TF-IDF:

* TF: Term Frequency (per sentence)
* IDF: Document Frequency (per document)
e.g.
```
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["This is very strange",
          "This is very nice"]
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
print dict(zip(vectorizer.get_feature_names(), idf))


# TF/IDF, the value uniqueness 1 > 0 (a, the, etc...)
#{u'is': 1.0,
# u'nice': 1.4054651081081644,
# u'strange': 1.4054651081081644,
# u'this': 1.0,
# u'very': 1.0}
```

We use TfidfVectorizer and make char_vectorizer to analyze on character level, and word_vectorizer on word level. (It is possible to create on feature-level.)
I set max_features 10,000 and 30,000 respectively. It will take the amount to create vocabulary upto max_features amount. The ngram and max_features should get adjusted simultaneously.


http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

### Fit and Transform

Fitting finds the internal parameters of a model to transform data.
Transforming applies the parameters to data.

You may fit a model to one set of a data, and then transform it on a completely different data.

Use the word_vectorizer and char_vectorizer to transform the data and merge them together using hstack for both train, test dataset.


### Score

The score represents the accuracy of prediction.

* SGD classifier:

SGD Classifier implements regularised linear models with Stochastic Gradient Descent.

`score = 59.2452%`

Group Kfold iterator variant with non-overlapping groups -- i.e. The same group will not appear in two different folds.

`score = 59.852%`

* XGboost:

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.
It is short for “Extreme Gradient Boosting” and used for superviosed learning problems. It is fast and shows good results.

The score is calculated from kaggle at

`score = 63.426%`