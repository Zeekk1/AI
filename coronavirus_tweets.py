# Part 3: Text mining.
import nltk
# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
import pandas as pd 
import requests
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer



def read_csv_3(data_file):
	# Read the CSV file with 'latin-1' encoding
    df = pd.read_csv(data_file, encoding='latin-1')
    return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return df['Sentiment'].unique().tolist()


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	#in pandase series is a 1 dimensional array
	# Count the occurrences of each sentiment
    sentiment_counts = df['Sentiment'].value_counts()
    
    # Get the second most frequent sentiment
    if len(sentiment_counts) > 1:
        return sentiment_counts.index[1]  # Second most common sentiment
    else:
        return None  # If there's only one unique sentiment, return None

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.

"""Handle the exception if it is a tie then what happens but I dont think so that is neccessary here"""
def date_most_popular_tweets(df):
	# Filter only rows where the sentiment is "Extremely Positive"
    positive_tweets = df[df['Sentiment'] == 'Extremely Positive']
    
    # Count occurrences of each date
    date_counts = positive_tweets['TweetAt'].value_counts()
    
    # Get the date with the maximum count
    if not date_counts.empty:
        return date_counts.idxmax()  # Returns the date with the highest number of "Extremely Positive" tweets
    else:
        return None  # If no "Extremely Positive" tweets exist, return None

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda text: ''.join(char if char.isalpha() or char.isspace() else ' ' for char in text))
    return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda text: ' '.join(text.split()))
    return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	#first step is to extract
	return df['OriginalTweet'].str.split()
	#maybe I can add them into list 


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions
def count_words_with_repetitions(tdf):
    if isinstance(tdf, pd.DataFrame):
        token_series = tdf['OriginalTweet']
    else:
        token_series = tdf
    return token_series.explode().count()
# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    all_words = tdf.explode()
    unique_words = set(all_words)
    return len(unique_words)


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
    word_counts = tdf.explode().value_counts()
    
    # Return the top k most frequent words as a list
    return word_counts.head(k).index.tolist()
	

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
#https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
#we need to import this thing


#removing stem words is doen 
url = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
def remove_stop_words(tdf):
    stop_words = set(requests.get(url).text.split())
    return tdf.apply(lambda tokens: [word for word in tokens if isinstance(word, str) and word not in stop_words and len(word) > 2])

    
    

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.

#stemming is done
def stemming(tdf):
    stemmer = PorterStemmer()
    return tdf.apply(lambda words: [stemmer.stem(word) if isinstance(word, str) else word for word in words])


def mnb_predict(df):
    # Convert tweets into feature vectors using CountVectorizer (Bag-of-Words)
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    #vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['OriginalTweet'])  # Transform text data into numeric form
    
    # Get target labels (Sentiment column)
    y = df['Sentiment']
    
    # Train a Multinomial Naive Bayes classifier
    model = MultinomialNB(alpha=0.1)
    model.fit(X, y)
 
    
    # Predict sentiments for the training set

    return model.predict(X),y # Returns predictions as a numpy.ndarray


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.

def mnb_accuracy(y_pred,y_true):
    # Compute classification accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return round(accuracy,3)

