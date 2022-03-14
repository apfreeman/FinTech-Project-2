import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#Set API Key from env variable
ss_key=os.getenv("SS_API_KEY")


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
    
    return trending_df