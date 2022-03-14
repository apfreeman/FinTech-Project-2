# FinTech-Project-2
## FinTech Project 2

# Project Title
## Meme Trading - How to catch a Gamestop

![Wall Street Bets - Gamestop](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/wsb_gme.jpg?raw=true)"

# Team Members
- Adam Freeman
- Aidan Laird
- Jeryl Lim
- Kimberley Ng
- Lincoln Luther

# Project Description/Outline

- Could the Gamestop phenomenom happen again ? If so how can we detect it and be a part of it

- Could we create a trading algorithm that indentifies stocks based on Social Media activity and positive sentiment from popular social media sources like Twitter and WallStreetBets. 

- Could we use Technical analysis along with deep learninig module to predict price movement of these hyped stocks.

- Could we trade these predictions and actually make a profit ?

Our team has managed to create a tool that will accomplish all of these tasks. This document will outline how an ordinary person could interact and use this tool to trade based on social media hype and achieve the objective of catching the next "Gamestop Phenomenon". This tool is designed to be used by an everyday person living in Australia who is enaged in study or work. This person is intersted in investing from a unique social media hype angle and would use this tool with minimal input alongside their day job.

## 9:30pm - Run the Meme Trading Notebook
## 10:00pm - Fundemental Analysis - Examine the following outputs
- Stock Sentiment Plot

![Stock Sentiment](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/sentiment_trending_df_scores_example.PNG?raw=true)

- Stock Sentiment DataFrame (This is example only of first 5 rows)

![Stock Sentiment Trending DF](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/sentiment_trending_df_example.PNG?raw=true)

- Stock Sentiment Clusters

![Stock Sentiment Clusters](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/classification_plot_view1.PNG?raw=true)

![Stock Sentiment Clusters](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/classification_plot_view2.PNG?raw=true)

- List of Top Class of stocks according to classification algorithm

![Stock Sentiment Clusters](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/top_class_stocks.PNG?raw=true)

## 10:05pm - Technical Analysis - Examine the following output

- Technical Anlysis DataFrame (This is example only of first 5 rows)

![TA Analysis](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/TA_Analysis.PNG?raw=true)


## 10:10pm - Review the LTSM Deep Learning outputs and buy predictions

- LTSM Output example (This is example only the first ticker)

![DL Output](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/dl_output.PNG?raw=true)

- Buy Predictions

![BUY Predictions](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/dl_buy_predict.PNG?raw=true)

- Model Training vs Validation Loss

![Training vs Validation Loss](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/training_vs_validation_loss.PNG?raw=true)

- Returns 

![Actual vs Predicted](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/dl_actual_vs_predicted.PNG?raw=true)

![Strategy vs Actual](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/strategy_vs_actual.PNG?raw=true)

- Optional - Examine model more closely. If required enter the TICKER name and the code will produce the model summary for further investigation

![Model Summary](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/LTSM_model_summary_example.PNG?raw=true)

## 10:15pm - This step can be done at the end of the day or now to look at historical information for any previous day. Exampe below is for stock ticker HYMC for 2022/03/11

-  Enter stock ticker and data and the code will create the trade evaluation data for the time and stock specified

![Evaluation](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/evalutation.PNG?raw=true)

![Entry and Exit](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/entry_exit_example.PNG?raw=true)

## 10:30pm - Run the remaining code and head off to bed. 

- The remaining code will take all the analysis as inputs and any stocks identied will be traded at US market open and sold at 30 mins prior to market close. 

![trades](https://github.com/apfreeman/FinTech-Project-2/blob/main/Images/trade.PNG?raw=true)

## Wake up the next day and grab a coffee, review your trades before heading off to work

