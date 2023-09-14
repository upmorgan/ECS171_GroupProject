# ECS171_GroupProject

> Please see the 'Final_Write_Up.md' for the final report.

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
 
## Dataset Description and Preprocessing

The two original datasets contains "cleaned" and "uncleaned" data stored as csv files. After examination of cleaned 
version, we found it is not usable due to the author's poor data cleaning. The score is not properly extracted, many 
duplication, and unextracted information (such as username and date). Therefore the "cleaned" version is abondoned.

The uncleaned version contain only one column, which including score, username, date, review, number of people found 
helpful. These features are extracted through regular expression.

Some reviews has no score due to the script used by original author. An `nan` will be assigned for missing numerical values.

The script does not expand any review contains "spoiler warning." Thus, the review for those will be assigned as `NULL`

The uncleaned version contains 13098 rows for barbie and 84049 rows for oppenheimer. After cleaning and dropping 
duplicate rows, we have 784 rows for barbie and 2013 rows for oppenheimer.

The data structure is as following:

```
| Score | Title | Username | Date | People_Found_Helpful | People_Not_Found_Helpful | Review | isBarbie? |
```

## Feature Description

Score is integer range from 1 to 10.

Title, username, review are string.

Date will be converted to pd.datetime object.

People_Found_Helpful are the number of people who viewed review and clicked on the "helpful" button.

People_Not_Found_Helpful are the number of people who viewed review and didn't click on the "helpful" button.

isBarbie? is a boolean value indicating whether the review is for barbie or oppenheimer. This is the target variable.

## Added Feature

From the text provided, we created new numerical data frame. Word frequency and sentiment analysis is under construction.

The numerical data frame contains following extra information:

```
| Length_of_Title | Length_of_Username | Length_of_Review | Helpful_Ratio |
```

## Preprocssing Planning

Score will be normalized to 0~1 by multiplying 1/10. The length of title, length of username, leng of review will be 
normalized basedon sklearn.MinMaxScaler. Depending on result and furthur exploration, all above value might be standardized.

Frequency of words will be normalized with sklearn.MinMaxScaler once constructed. The coordination will be caclated and 
the first 100~ related token will be selected for future use. Too much words might result in slow trainning speed, and thus
the actual number of words selected might subject to change.

Sentiment data is still under construction. Once constructed, they will be encoded with one-hot-encoding.

In exmination of data, we found that reviews of both oppenheimer and barbie tend to invole in each other for comparision, thus 
we are particularly intersted in certain words such as names of directors, names of actors, names of characters or places in 
both movie will be selected out and examed in detaile.

## Data Example:

### Raw Data

```text
9/10

Murphy is exceptional

Orlando_Gardner19 July 2023

You'll have to have your wits about you and your brain fully switched on watching Oppenheimer as it could easily get 
away from a nonattentive viewer. This is intelligent filmmaking which shows it's audience great respect. It fires 
dialogue packed with information at a relentless pace and jumps to very different times in Oppenheimer's life 
continuously through it's 3 hour runtime. There are visual clues to guide the viewer through these times but again 
you'll have to get to grips with these quite quickly. This relentlessness helps to express the urgency with which the 
US attacked it's chase for the atomic bomb before Germany could do the same. An absolute career best performance from 
(the consistenly brilliant) Cillian Murphy anchors the film. This is a nailed on Oscar performance. In fact the whole 
cast are fantastic (apart maybe for the sometimes overwrought Emily Blunt performance). RDJ is also particularly 
brilliant in a return to proper acting after his decade or so of calling it in. The screenplay is dense and layered 
(I'd say it was a thick as a Bible), cinematography is quite stark and spare for the most part but imbued with rich, 
lucious colour in moments (especially scenes with Florence Pugh), the score is beautiful at times but mostly anxious 
and oppressive, adding to the relentless pacing. The 3 hour runtime flies by. All in all I found it an intense, taxing 
but highly rewarding watch. This is film making at it finest. A really great watch.

1,413 out of 1,597 found this helpful. Was this review helpful? Sign in to vote.

Permalink<>?
```

### Cleaned Data

```
[9, "Murphy is exceptional", "Orlando_Gardner", 2023-07-19, 1413, 1587, "You'll have ... great watch.", 0]
```

## Miltestone 2

> We are doing a ensemble model, each model are build and tested on different branch. Since we haven't finish the building,
> and testing phase of them, we are not merging them to the `main` branch. If you want to look through the codes, please go
> to their branch: `Decision_Tree` and `text_process_model`. A neruonetwork is also being built but the group member has
> some issue with github. We will try to fix it on Monday.

Becuase the data is very unbalanced (700 for barbie, 2000+ for oppenheimer) we had issue for model training becuase they tend 
to classify everyone as oppenheimer (which is cheating). And we are at the underfitting side of it since it neither perform 
well on training nor on the testing set. Two solution are proposed, increase learing rate and use `Adam` optimizer which should 
auto ajust the learning rate depending the result from previous epochs. The other solution is to increase epochs. However, `BERT`
model require lots of computation to tune the parameters, increasing epochs or decreasing batch size is not feasiblely possible.
Luckily, the `Adam` work and didn't overfit after examing the test cross-entropy and training cross-entropy. Kfold validation will 
be run later (it took long time). 

The other model is build with decision tree, giving us high accuracy. The model uses Bag Of Words, created from review column, this
is the only feature used to train the decision tree. But we were able to have a high accuracy(0.9) with this method. Another interesting
point is that this approach did not account for data unabalance, despite this the model was able to perform pretty well. The test MSE 
for this model was 0.098 and the Train MSE was 0.06 suggesting no overfitting of the model. When we tested via cross entropy Test Log_loss was 3.5
and Train log_loss was 2.4 a difference of 1(0.14 ish), this is somewhat inconclusive, plotting seems overfitting is possible. Further research required.


