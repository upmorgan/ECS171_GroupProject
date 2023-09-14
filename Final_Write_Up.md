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

The original datasets were given by python scripts. It contains two columns: text data and score. We used regular 
expression to extract data that can be used:

```
| Score | Title | Username | Date | People_Found_Helpful | People_Not_Found_Helpful | Review | isBarbie? |
```

In data exploratation, we found the raw data we extracted are not strongly related except for the `score`.
Hence, more features extraction methods are needed.

Three different models are applied: BERT, Bag of Words, and word frequency of featured vocabulary.

### BERT

In BERT model, `review` are used as features. Becuase of the computation required for training, we cannot tuned
the model very well since a 10 epoch, 512 batch_size require more than 10 mins to train (in a 16 threads 4GHz machine)

By reviewing the history of training, we found in 512 batch_size, 10 epoch does not show significant overfitting 
nor underfitting.

### Bag of Words

The `review` is first vectorized with built in method of SKlearn feature extraction module which produce vectorized 
data of word count in each observation. Then, a decision tree is trained on it. To determine the best depth, 
of the tree, we graphed the error of train/test vs depth and decided the max_depth = 5.

![image](https://github.com/upmorgan/ECS171_GroupProject/assets/45218090/1f015967-313f-4abf-830d-7d48fb41c6df)


### ANN

The word frequency for certain manually decided words are calcualted for each observation and put in as features 
in an ANN. The list of words can be found in the notebook.

By graphing the error of train/test vs epoch, we found epoch = 60 to be the crossing point. Another 
interesting property we observed is that the training loss is higher than the validation loss when it is not 
overfitting. With manual parameter searching, we found Adam optimizer with learning rate of 0.001 to be the 
best perfromed.

![image](https://github.com/upmorgan/ECS171_GroupProject/assets/45218090/d8946b18-01b1-431e-a057-ed7df8041a08)

### Ensemble

After all the model is trained, a new 2 layer ANN is trained based on the predicted data of previous three 
models. This is a customized ensemble network supposed to add weights on each models's vote.

## Results

Train/validation error vs Epocs of ensemble:

![image](https://github.com/upmorgan/ECS171_GroupProject/assets/45218090/bf8b9c38-c992-4502-b859-b23319a84495)

Ensemble model report:

```
              precision    recall  f1-score   support

           0       1.00      0.74      0.85       560
           1       0.00      0.00      0.00         0

    accuracy                           0.74       560
   macro avg       0.50      0.37      0.43       560
weighted avg       1.00      0.74      0.85       560
```

Simple Vote with only BOW and ANN

```
              precision    recall  f1-score   support

           0       0.99      0.74      0.85       555
           1       0.01      0.40      0.03         5

    accuracy                           0.74       560
   macro avg       0.50      0.57      0.44       560
weighted avg       0.98      0.74      0.84       560
```

Report from single models:

```
BERT Test
              precision    recall  f1-score   support

           0       0.17      0.96      0.29        75
           1       0.98      0.29      0.45       485

    accuracy                           0.38       560
   macro avg       0.58      0.63      0.37       560
weighted avg       0.87      0.38      0.43       560

========================================
BoW Test
              precision    recall  f1-score   support

           0       0.94      0.74      0.83       530
           1       0.05      0.23      0.08        30

    accuracy                           0.71       560
   macro avg       0.50      0.49      0.46       560
weighted avg       0.90      0.71      0.79       560

========================================
ANN Test
              precision    recall  f1-score   support

           0       1.00      0.91      0.95       447
           1       0.74      0.98      0.84       113

    accuracy                           0.93       560
   macro avg       0.87      0.95      0.90       560
weighted avg       0.94      0.93      0.93       560
```

## Discussion

The ANN based on present of certain keywords (name of character, place, certain punctuation). One novel point 
of it is that, similar to random forest, each of them are taking different features. The BERT take in the 
review text directly, Bag of Words take in vectorized frequency data, and ANN extract the frequency of certain 
words that human mannually put into it. This makes the model kind of similar to the "human in the loop"
algorithms. Human being extract the words that is mostly representitive, and feed them to the algorithm 
directly. This largely accelerate the speed of algorithm, and it showed really good results with 90% accuracy 
on testing dataset.

However, the BERT model requires a lot computation in training and predicting phase which made it hard to be 
used in our case due to the low computation resources.

From the recall and precision we saw that the unbalanced data does made the accuracy less helpful in 
scoring the models. A resampling method can be applied or with more data collected to make the decision tree 
better in prediction.

Since the ANN with words manually assigned shows great performance in comparing the others, the next step of 
of this study will be exploring method to identify those words that is mostly related to the subject. This can 
be done offline with only statistical analysis of words or can be done with online seraching engine that might
gives out more related words than just the review.

## Conclusion



## Collaboration Section

Yuxin Ren: Data cleaning with regular expression, first milestone write up, final write up, ensemble, debugging, 
model analysis suggestion.

Ulysses Morgan: Create Github repo, first milestone write up, Bag of Words model

Ivan Karpov: ANN model and tuning

Evan Tan: Data cleaning, BERT model

Jason Yoo: hyper parameter tuning of Bag of Words model, 
