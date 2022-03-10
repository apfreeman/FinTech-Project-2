# Functions defined for project 2
import requests
import os
import pandas as pd
import numpy as np
import time
import hvplot.pandas
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date
from dotenv import load_dotenv
#Set API Key from env variable
ss_key=os.getenv("SS_API_KEY")


def get_sentiment_df():
    # Create headers variable containing API Key 
    headers = {
        'Authorization' : f'Token {ss_key}',
        'Accept': 'application/json',
    }
    # Get data for daily stock sentiment from API
    sentiment_response = requests.get('https://socialsentiment.io/api/v1/stocks/sentiment/daily/', headers=headers)
    
    # Create dict from sentiment_response['results'] and then a data frame from this dict 
    sentiment_dict = sentiment_response.json()['results']
    sentiment_df = pd.DataFrame.from_dict(sentiment_dict)
    
    # Determine how many lines and then pages are in the sentiment response
    line_count = sentiment_response.json()['count']
    page_count = int(line_count / 50) + (line_count % 50 > 0)
    page=2
    
    # Loop through each page and gather the sentiment data from the API
    while page <= page_count:
        # Loop through all pages available from API and construct dataframe for sentiment data
        sentiment_url = "https://socialsentiment.io/api/v1/stocks/sentiment/daily/?page=%s"%page
        sentiment_response = requests.get(sentiment_url, headers=headers)
        sentiment_dict = sentiment_response.json()['results']
        sentiment_df_loop = pd.json_normalize(sentiment_dict)
        sentiment_df = pd.concat([sentiment_df, sentiment_df_loop], axis=0)
        page += 1
        time.sleep(1)

    # Reset the index on the sentiment df and drop the old index
    sentiment_df.reset_index(inplace=True, drop=True)
    
    # Output the sentiment data for today to a csv file
    path = (f'../Resources/sentiment_{date.today()}.csv')
    sentiment_df.to_csv(path)
    
    return sentiment_df

def get_sentiment_trending_df():
    # Create headers variable containing API Key 
    headers = {
        'Authorization' : f'Token {ss_key}',
        'Accept': 'application/json',
    }
    # Get data for trending stock sentiment from API
    trending_response = requests.get('https://socialsentiment.io/api/v1/stocks/trending/daily/', headers=headers)
    
    # Create dict from trending_response and then a data frame from this dict 
    trending_dict = trending_response.json()
    trending_df = pd.DataFrame.from_dict(trending_dict)
    
    # Output the sentiment data for today to a csv file
    path = (f'../Resources/sentiment_trending_{date.today()}.csv')
    trending_df.to_csv(path)
    
    return trending_df

def create_technical_analysis_df():
    # Create the Alpaca API object, specifying use of the paper trading account:
    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        base_url = 'https://paper-api.alpaca.markets',
        api_version = "v2"
    )
    
    # Set the list of tickers to the top class stock
    tickers = top_class_stocks.index
    beg_date = '2022-03-04'
    end_date = '2022-03-04'
    timeframe='1Min'
    start =  pd.Timestamp(f'{beg_date} 09:30:00-0400', tz='America/New_York').replace(hour=9, minute=30, second=0).astimezone('GMT').isoformat()[:-6]+'Z'
    end =  pd.Timestamp(f'{end_date} 16:00:00-0400', tz='America/New_York').replace(hour=16, minute=0, second=0).astimezone('GMT').isoformat()[:-6]+'Z'

    portfolio_df = api.get_bars(tickers, timeframe, start=start, end=end).df

    #This calculates the rolling 5 period and 15 period high/low MA for each symbol in portfolio_df
    portfolio_df['Lowest_5D'] = portfolio_df.groupby('symbol')['low'].transform(lambda x: x.rolling(window = 12).min())
    portfolio_df['High_5D'] = portfolio_df.groupby('symbol')['high'].transform(lambda x: x.rolling(window = 12).max())
    portfolio_df['Lowest_15D'] = portfolio_df.groupby('symbol')['low'].transform(lambda x: x.rolling(window = 26).min())
    portfolio_df['High_15D'] = portfolio_df.groupby('symbol')['high'].transform(lambda x: x.rolling(window = 26).max())

    portfolio_df['Stochastic_5'] = ((portfolio_df['close'] - portfolio_df['Lowest_5D'])/(portfolio_df['High_5D'] - portfolio_df['Lowest_5D']))*100
    portfolio_df['Stochastic_15'] = ((portfolio_df['close'] - portfolio_df['Lowest_15D'])/(portfolio_df['High_15D'] - portfolio_df['Lowest_15D']))*100

    portfolio_df['Stochastic_%D_5'] = portfolio_df['Stochastic_5'].rolling(window = 5).mean()
    portfolio_df['Stochastic_%D_15'] = portfolio_df['Stochastic_5'].rolling(window = 15).mean()

    portfolio_df['Stochastic_Ratio'] = portfolio_df['Stochastic_%D_5']/portfolio_df['Stochastic_%D_15']

    portfolio_df['TP'] = (portfolio_df['high'] + portfolio_df['low'] + portfolio_df['close']) / 3
    portfolio_df['sma'] = portfolio_df.groupby('symbol')['TP'].transform(lambda x: x.rolling(window=26).mean())
    portfolio_df['mad'] = portfolio_df['TP'].rolling(window=26).apply(lambda x: pd.Series(x).mad()) #Calculates Mean Absolute Deviation of 'TP' using a 21 period and returns a pandas series
    portfolio_df['CCI'] = (portfolio_df['TP'] - portfolio_df['sma']) / (0.015 * portfolio_df['mad'])
        #Note since .groupby is in 'sma' calc, the CCI Indicator cannot be calculated until 21 periods into next stock, therefore .groupby is not required in following 'mad' and 'cci' calculation.

    portfolio_df['prev_close'] = portfolio_df.groupby('symbol')['close'].shift(1)
    portfolio_df['TR'] = np.maximum((portfolio_df['high'] - portfolio_df['low']),
                                np.maximum(abs(portfolio_df['high'] - portfolio_df['prev_close']), 
                                abs(portfolio_df['prev_close'] - portfolio_df['low'])))

    for i in portfolio_df['symbol'].unique():
        ATR_12 = []
        ATR_26 = []
        TR_data = portfolio_df[portfolio_df.symbol == i].copy()
        portfolio_df.loc[portfolio_df.symbol==i,'ATR_12'] = (TR_data['TR']).rolling(window=12).mean()
        portfolio_df.loc[portfolio_df.symbol==i,'ATR_26'] = (TR_data['TR']).rolling(window=26).mean()


    portfolio_df['ATR_Ratio'] = portfolio_df['ATR_12'] / portfolio_df['ATR_26']
        #Note this ATR loops through each 'symbol' in portfolio_df and calculates an ATR Ratio on a fast / slow moving ATR.

    portfolio_df.reset_index(inplace=True)
    portfolio_df.set_index(['symbol', 'timestamp'], inplace=True)
    path = (f'../Resources/portfolio_indicators_{date.today()}.csv')
    portfolio_df.to_csv(path)
    return portfolio_df