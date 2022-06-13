# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 08:45:54 2022
For this project we'll use the Cornell University Movie 
Review polarity dataset v2.0 obtained 
from http://www.cs.cornell.edu/people/pabo/movie-review-data/

MODEL : Naïve Bayes
    
    
    
@author: manish
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


df = pd.read_csv('./moviereviews.tsv', sep='\t')
#df.head()

#########PReprocessing ################

#We look at a typical review
from IPython.display import Markdown, display
print(display(Markdown('> '+df['review'][0])))

# Check for the existence of NaN values in a cell:
print(df.isnull().sum())

#35 records show **NaN** (this stands for "not a number" and is equivalent to *None*). These are easily removed using the `.dropna()` pandas function.
df.dropna(inplace=True)

#Detect & remove empty strings
blanks = []  # start with an empty list

for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
        
print(len(blanks), 'blanks: ', blanks)
df.drop(blanks, inplace=True)

## Look at the dataset now
print(df['label'].value_counts())


### SPLIT THE DATA 
from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

######Build pipeline



# Naïve Bayes:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),])

# Linear SVC:
#text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
 #                    ('clf', LinearSVC()),])

## Fit the Naive Bayes 
text_clf_nb.fit(X_train, y_train)

## Fit the  Linear SVC
#text_clf_lsvc.fit(X_train, y_train)

# Form a prediction set
predictions = text_clf_nb.predict(X_test)

# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))


# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


filename = 'nb_model.pkl'
pickle.dump(text_clf_nb, open(filename, 'wb'))

'''
### TEST 

myreview = "A movie I really wanted to love was terrible. \
I'm sure the producers had the best intentions, but the execution was lacking."
print('\n\n\n')
print(text_clf_nb.predict([myreview]))

'''
