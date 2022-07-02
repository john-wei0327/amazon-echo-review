#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# In[2]:
reviews = pd.read_csv(r'C:\Users\User\Desktop\Amazon Echo Sentiment Analysis\amazon_reviews.csv')
reviews = reviews.drop(['date'], axis = 1)
reviews


# In[3]:
reviews.info()


# In[4]:
#countplot of ratings
sns.countplot(x = reviews['rating'])


# In[5]:
#countplot of positive and negative reviews with 1 being positive
sns.countplot(x = reviews['feedback'])


# In[6]:
#calculate the length of each entry in the verified_reviews column
reviews['length'] = reviews['verified_reviews'].apply(len)
reviews


# In[7]:
#shows frequency of review length
reviews['length'].plot(bins = 100, kind = 'hist', figsize = (20,10))


# In[8]:
reviews.describe()


# In[9]:
#separating reviews into positive and negative
positive = reviews[reviews['feedback'] == 1]
negative = reviews[reviews['feedback'] == 0]

#converting all positive and negative reviews into sentences
pos_sent = positive['verified_reviews'].tolist()
neg_sent = negative['verified_reviews'].tolist()

#converting all sentences into a single string
pos_string = " ".join(pos_sent)
neg_string = " ".join(neg_sent)


# In[10]:
from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(pos_string))


# In[11]:
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(neg_string))


# In[12]:
#creating a pipeline to remove punctuations, stopwords and perform count vectorisation
import string
import nltk
from nltk.corpus import stopwords

#removing punctuation, joining text and cleaning stopwords
def review_cleaning(text):
    punc_removed = [char for char in text if char not in string.punctuation]
    text_joined = ''.join(punc_removed)
    text_cleaned = [word for word in text_joined.split() if word.lower() not in stopwords.words('english')]
    return text_cleaned


# In[13]:
reviews_cleaned = reviews['verified_reviews'].apply(review_cleaning)


# In[14]:
#comparing cleaned and original review
print(reviews_cleaned[5])
print(reviews['verified_reviews'][5])


# In[15]:
from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = review_cleaning, dtype = np.uint8)
reviews_countvectorizer = vectorizer.fit_transform(reviews['verified_reviews'])


# In[16]:
#splitting the dataset into X and y which are reviews and labels for pos/neg reviews respectively 
X = pd.DataFrame(reviews_countvectorizer.toarray())
y = reviews['feedback']


# In[17]:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[18]:
from sklearn.metrics import classification_report, confusion_matrix

#using naive bayes classifiers
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))


# In[19]:
#using logistic regression 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_pred, y_test))


# In[20]:
#using gradient boosted trees
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))
