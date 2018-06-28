import numpy as np
import operator
import pandas as pd
import re
import requests
import sys
import tweepy

from bs4 import BeautifulSoup
from collections import Counter
from requests.exceptions import ConnectionError, ContentDecodingError
from sklearn.metrics import accuracy_score


def loadkeys(filename):
    """"
    Load Twitter API keys/tokens from CSV file with format:
    consumer_key, consumer_secret, access_token, access_token_secret
    """
    with open(filename) as f:
        items = f.readline().strip().split(', ')
        return items


def authenticate(twitter_auth_filename):
    """
    Given a file name containing the Twitter keys and tokens, create and return a tweepy API object.
    """
    twitter_auth = loadkeys(twitter_auth_filename)
    auth = tweepy.OAuthHandler(twitter_auth[0], twitter_auth[1])
    auth.set_access_token(twitter_auth[2], twitter_auth[3])
    api_object = tweepy.API(auth)
    return api_object


def fetch_tweets(api_object, name):
    """
    Given an API object and the name of a Twitter account, retrieve the full text of all tweets from the account
    """
    tweet_list = list()
    for status in tweepy.Cursor(api_object.user_timeline, screen_name=name, tweet_mode='extended').items():
        tweet_list.append(status._json['full_text'])
    return tweet_list


def answers_only(row):
    """
    Remove leading numbers in the answers
    """
    row['A'] = row['A'][3:]
    row['B'] = row['B'][3:]
    row['C'] = row['C'][3:]
    return row


def correct_answer(row):
    """
    Find which of the columns contains a checkmark, indicating this is the correct answer.  Store the correct answer
    and remove the checkmark from the text.
    """
    if row['A'].endswith('✓'):
        row['Correct'] = 'A'
        row['A'] = row['A'][:-2]
    elif row['B'].endswith('✓'):
        row['Correct'] = 'B'
        row['B'] = row['B'][:-2]
    else:
        row['Correct'] = 'C'
        row['C'] = row['C'][:-2]
    return row


def get_word_count(questions):
    """
    Find the word count of all words in the training set questions
    """
    word_count = Counter()
    for question in questions:
        words = question.split()
        words = [re.sub('[^0-9a-z*]', '', x.lower()) for x in words]
        for word in words:
            word_count[word] += 1
    return word_count


def get_search_terms(question, word_count):
    """
    Find the words that appear the least in previous questions
    """
    words = question.split()
    words = [re.sub('[^0-9a-z*]', '', x.lower()) for x in words]
    question_dict = dict()
    for word in words:
        question_dict[word] = word_count[word]
    search_terms = [k for k, v in question_dict.items() if v < 20]
    return search_terms


def add_search_terms(df, word_count):
    """
    Add the search terms to the dataframe
    """
    df['Terms'] = df.apply(lambda row: get_search_terms(row['Question'], word_count), axis=1)
    return df


def get_google_results(search_terms):
    """
    Get the HTML of the first page from Google based on the search terms
    """
    query = '+'.join(search_terms)
    r = requests.get("https://www.google.com/search?q="+query)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


def get_links(search_result):
    """
    Get the links from the results on the first page
    """
    links = re.findall('a href="\/url\?q=[^&]*', str(search_result))
    links = [link.replace('a href="/url?q=', '') for link in links]
    return links


def add_links(df):
    df['Links'] = df.apply(lambda row: get_links(get_google_results(row['Terms'])), axis=1)
    return df


def get_words(text):
    """
    Get all words from a text. Normalize to lowercase. Used in Lab 4.
    """
    return re.findall('[a-z]+', text.lower())


def score(counts, answer):
    """
    Get the likelihood of a given word using Laplacian Smoothing
    """
    return (counts[answer] + 1)/(sum(counts.values()) + len(counts.values()))


def prediction(row):
    """
    Generate a prediction for a certain row
    """
    a_score = 0
    b_score = 0
    c_score = 0
    print(type(row['Links']))
    for link in row['Links'][:3]:
        if not link.endswith('.html'):
            try:
                print(link)
                r = requests.get(link)
                soup = BeautifulSoup(r.text, 'html.parser')
                link_words = get_words(str(soup))
                word_count = Counter(link_words)
                a_score += score(word_count, row['A'].lower())
                b_score += score(word_count, row['B'].lower())
                c_score += score(word_count, row['C'].lower())
            except (ConnectionError, ContentDecodingError):
                pass
    if a_score == max(a_score, b_score, c_score):
        return 'A'
    elif b_score == max(a_score, b_score, c_score):
        return 'B'
    else:
        return 'C'


def add_predictions(df):
    df['Prediction'] = df.apply(lambda row: prediction(row), axis=1)
    return df


if __name__ == '__main__':
    # Authenticate and create API object
    api = authenticate("/Users/jtoffler/Documents/Licenses/twitter.csv")
    all_tweets = fetch_tweets(api, "HQTriviaScribe")

    # Create DataFrame with question and three possible answers
    # Only include Tweets that have 4 lines (i.e. Q/A/B/C structure)
    q_list = [tweet.split('\n') for tweet in all_tweets if len(tweet.split('\n')) == 4]
    hq = pd.DataFrame(q_list, columns=['Question', 'A', 'B', 'C'])

    # Create column to store correct answer and remove answer numbers
    hq['Correct'] = ''
    hq = hq.apply(lambda row: correct_answer(row), axis=1)
    hq = hq.apply(lambda row: answers_only(row), axis=1)

    # Generate the search terms
    count_of_words = get_word_count(hq['Question'])
    hq = add_search_terms(hq, count_of_words)

    # Get relevant links from the search terms
    hq = add_links(hq)

    # Generate predictions
    hq_preds = add_predictions(hq)

    # Calculate accuracy
    print(accuracy_score(hq_preds['Correct'], hq_preds['Prediction']))
