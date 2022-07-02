# Amazon Echo Review Sentiment Analysis
Conducting sentiment analysis on Amazon Echo reviews

## Exploratory Data Analysis
I first imported the relevant libraries and created several summary graphs to visualise the distribution of ratings, positive/negative reviews and review lengths. 
Next I created word clouds for positive and negative reviews and identified key indicators for negative experiences such as price, bulb and screen. 

## Data Cleaning
I then created a pipeline to remove punctuations, stopwords and performed count vectorisation on the dataset. 

## Model training 
After splitting the data into test and training data, I employed Naive Bayes classifier, logistic regression and gradient boosting models on the dataset. 
This yield great precision results and weight averages of 90+ for all 3 cases.
