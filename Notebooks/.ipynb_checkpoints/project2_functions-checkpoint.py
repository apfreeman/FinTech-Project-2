# Import libraries and dependencies
import requests
import os
import pandas as pd
import time
from datetime import date
from dotenv import load_dotenv

class functions:

    def get_sentiment_df():
        ss_key=os.getenv("SS_API_KEY")
        headers = {
            'Authorization' : f'Token {ss_key}',
            'Accept': 'application/json',
        }

        sentiment_response = requests.get('https://socialsentiment.io/api/v1/stocks/sentiment/daily/', headers=headers)
        sentiment_dict = sentiment_response.json()['results']
        sentiment_df = pd.DataFrame.from_dict(sentiment_dict)
        line_count = sentiment_response.json()['count']
        page_count = int(line_count / 50) + (line_count % 50 > 0)
        page=2

        while page <= page_count:
            # Loop through all pages available from API and construct dataframe for sentiment data
            sentiment_url = "https://socialsentiment.io/api/v1/stocks/sentiment/daily/?page=%s"%page
            sentiment_response = requests.get(sentiment_url, headers=headers)
            sentiment_dict = sentiment_response.json()['results']
            sentiment_df_loop = pd.json_normalize(sentiment_dict)
            sentiment_df = pd.concat([sentiment_df, sentiment_df_loop], axis=0)
            page += 1
            time.sleep(1)

        sentiment_df.reset_index(inplace=True, drop=True)
        path = (f'../Resources/sentiment_{date.today()}.csv')
        sentiment_df.to_csv(path)
        return sentiment_df

    def get_sentiment_trending_df():
        ss_key=os.getenv("SS_API_KEY")
        headers = {
            'Authorization' : f'Token {ss_key}',
            'Accept': 'application/json',
        }
        trending_response = requests.get('https://socialsentiment.io/api/v1/stocks/trending/daily/', headers=headers)
        trending_dict = trending_response.json()
        trending_df = pd.DataFrame.from_dict(trending_dict)
        path = (f'../Resources/sentiment_trending_{date.today()}.csv')
        trending_df.to_csv(path)
        return trending_df