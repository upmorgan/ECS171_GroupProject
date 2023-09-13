# Barbie or Oppenheimer

## Abstract

We will be using review data for 2 movies, Barbie and Oppenheimer. The data will be taken from Kaggle, originally from 
IMDB. The data will include a text review and a 1-10 scale score of the movie. Our goal is to identify which movie a 
given review is from based on this data, using linear regression or deep learning models, supervised models. We will 
parse this information further in order to create new variables potentially more useful, variables like word count, and 
average word length. We may need to make an additional ML model, unsupervised, to find patterns in the data we want to 
use for our classification. Potential problems that can arise, include the use of the word itself in the comment, 
Oppenheimer will most likely be mentioned in a review about Barbie and vice versa, we may have to clean the data more, 
for emojis, unique values can create unnecessary bias.

Datasets used:

1. IMBD Oppenheimer Review: https://www.kaggle.com/datasets/ibrahimonmars/84k-reviews-on-oppenheimer-dataset/discussion
2. IMBD Barbie: https://www.kaggle.com/datasets/ibrahimonmars/imdb-reviews-on-barbie?select=imdb_barbie_Uncleaned.csv

**Notice: The oringal dataset contain large files (>100MB), and thus not able to be uploaded to github. The user should 
download and unzip the original dataset to the local machine and put them in the `./data` directory.**

## Introduction

In NLP, it is important to identify the subject of conversation. Some times multiple subject might get involved, and 
some times different subjects are compared either implicitly or explicitly. It is crucial for the algorithm to be able 
to identify the correct subject even when it is compared with other similar subjects.

Since Barbie and Oppenheimer are the two most popular films right now, we decided to use them as our testing subjects.
Reviews from IMDB are acquired and cleaned. Many of them are talking about each other in the review instead of only 
dicussing one of them. This is a scenario closer to real life conversation.

If a model can predict the subject correctly (whether the given review are talking about Barbie or Oppenheimer), we 
can then asure that the machine knows what are we talking about instead of just giving some text based on merely 
statiscal data that can match either of them.

## Methods

Three different models are built and tuned: BERT classifier, Bag of Words Decision Tree, and ANN based on present of 
certain keywords (name of character, place, certain punctuation). One novel point of it is that, similar to random 
forest, each of them are taking different features. The BERT take in the review text directly, Bag of Words take in 
vectorized frequency data, and ANN extract the frequency of certain words that human mannually put into it.

The ANN model is kind of similar to the "human in the loop" algorithms. Human being extract the words that is mostly 
representitive, and feed them to the algorithm directly. This largely accelerate the speed of algorithm.

## Results

## Dicussion

## Conclusion

## Collaboration Section