import numpy as np
import pandas as pd
import re


tweets_df = pd.read_csv('data/pre-processed/aapl_spy_tweets.csv')
#tweets_df = pd.read_csv('data/pre-processed/test_tweets.csv')

total_number_of_tweets = 0
for tweets in tweets_df['tweets']:

    daily_tweets = []

    for match in re.finditer(r"([\s\S]+?)(\|\||$)", tweets):
        daily_tweets.append(match.group(1))

    total_number_of_tweets = total_number_of_tweets + len(daily_tweets)


print("the total number of tweets is: " + str(total_number_of_tweets))
