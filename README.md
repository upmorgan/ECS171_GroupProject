# ECS171_GroupProject

# Abstract
We will be using review data for 2 movies, Barbie and Oppenheimer. The data will be taken from Kaggle, originally from IMDB. The data will include a text review and a 1-10 scale score of the movie. Our goal is to identify which movie a given review is from based on this data, using linear regression or deep learning models, supervised models. We will parse this information further in order to create new variables potentially more useful, variables like word count, and average word length. We may need to make an additional ML model, unsupervised, to find patterns in the data we want to use for our classification. Potential problems that can arise, include the use of the word itself in the comment, Oppenheimer will most likely be mentioned in a review about Barbie and vice versa, we may have to clean the data more, for emojis, unique values can create unnecessary bias.
Datasets used:

1. IMBD Oppenheimer Review: https://www.kaggle.com/datasets/ibrahimonmars/84k-reviews-on-oppenheimer-dataset/discussion
2. IMBD Barbie: https://www.kaggle.com/datasets/ibrahimonmars/imdb-reviews-on-barbie?select=imdb_barbie_Uncleaned.csv



 
# Dataset Description and Preprocessing

Our dataset contains reviews for a movie named "Oppenheimer." Each review consists of several components that provide valuable information about the review and the reviewer's sentiment. We have both an uncleaned and a cleaned version of the data, and we'll outline the format of the uncleaned data followed by our preprocessing steps. Both the Barbie and Oppenheimer datasets have 7 columns, which are score (movie rating out of 10), title, username, date, the number of users who found this review helpful, the number of users who did not find this review helpful, and the review itself. The original dataset for Barbie had 796 reviews, while after cleaning it went down to 784 reviews, while for Oppenheimer, there were 2035 reviews and 2013 after cleaning. 

In order to clean the datasets, we have used regular expressions that would extract the 7 pieces of information from each review and create a dataframe with 7 columns. 

## Uncleaned Data Example:

`9/10
Murphy is exceptional
Orlando_Gardner19 July 2023
You'll have to have your wits about you and your brain fully switched on watching Oppenheimer as it could easily get away from a nonattentive viewer. This is intelligent filmmaking which shows it's audience great respect. It fires dialogue packed with information at a relentless pace and jumps to very different times in Oppenheimer's life continuously through it's 3 hour runtime. There are visual clues to guide the viewer through these times but again you'll have to get to grips with these quite quickly. This relentlessness helps to express the urgency with which the US attacked it's chase for the atomic bomb before Germany could do the same. An absolute career best performance from (the consistenly brilliant) Cillian Murphy anchors the film. This is a nailed on Oscar performance. In fact the whole cast are fantastic (apart maybe for the sometimes overwrought Emily Blunt performance). RDJ is also particularly brilliant in a return to proper acting after his decade or so of calling it in. The screenplay is dense and layered (I'd say it was a thick as a Bible), cinematography is quite stark and spare for the most part but imbued with rich, lucious colour in moments (especially scenes with Florence Pugh), the score is beautiful at times but mostly anxious and oppressive, adding to the relentless pacing. The 3 hour runtime flies by. All in all I found it an intense, taxing but highly rewarding watch. This is film making at it finest. A really great watch.
1,413 out of 1,597 found this helpful. Was this review helpful? Sign in to vote.
Permalink<>?`

The cleaned version keeps a majority of the same information, with the exception of the score being in a separate column, but after a comparison with the uncleaned version we’ve realized there are inaccuracies in the score in the cleaned version, namely, all 10 scores were converted to 1, this can be seen clearly on the Kaggle page for Oppenheimer. 50% of the scores are labeled as 1, something we can see is clearly false with a quick look at the raw data. Thus, the cleaned version is unreliable.

Duplicate reviews were identified and removed from both datasets to enhance data accuracy.

The data has 7 components, we will examine these using the above example


The uncleaned data comprises the following components:

## Score: 
**Ex: 9/10** 
A numerical score ranging from 1 to 10.
### Preprocessing Steps:
We converted the score to an integer by extracting the numeric part (e.g., "9/10" becomes 9).

## Review Title: 
**Ex: Murphy is exceptional** 
A concise subtitle summarizing the review.
### Preprocessing Steps:
No changes were made to this component. 

## Username:
 **Ex: Orlando_Gardner** 
The reviewer's username.
### Preprocessing Steps:
No changes were made to this component.
 
## Date of Review: 
**Ex: 19 July 2023** 
The date when the review was posted.
### Preprocessing Steps:
We excluded this feature, as it does not significantly contribute to the analysis due to the overlapping movie release dates.
 
## Review Text:
 **Ex: You'll have to have…… really great watch.** 
The main review content.
### Preprocessing Steps:
We retained the main review content for analysis but will be the main source of feature extraction.

## Helpful Score:
#Ex: 1,413 out of 1,597 found this helpful. Was this review helpful? Sign in to vote.** 
The count of users who found the review helpful. 
### Preprocessing Steps:
We extracted the count of helpful votes from this field.

## Permalink:  
**Ex: Permalink<>?** 
A link irrelevant for analysis.
### Preprocessing Steps:
We excluded this component as it does not provide relevant information.




