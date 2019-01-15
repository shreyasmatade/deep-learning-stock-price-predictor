# Machine Learning Engineer Nanodegree
## Capstone Proposal
Shreyas Matade  
December 1st, 2018

## Proposal
Deep Learning Stock Price Predictor

### Project intuition

Like most of the tech engineers, I was impressed and surprised by the bitcoin-cryptocurrency boom. Like any other human, I was drawn to FOMO and started investing in crypto (and I can say proudly, i did not lost nor gained ). This is where I started my curiosity toward trading bots.  
After taking Machine learning nanodegree course, I learned a lot about stats, data analysis and various machine learning models, methodologies.  
I really wanted to apply my newly learned skills to work. However, cryptocurrency market is highly volatile and as of now it is becoming unreliable. Hence, just to get the taste of analyzing time series data I thought of exploring "comparatively" more reliable Stock Market.

### Project outline

In this project, I am planning to apply and explore different versions of LSTM RNN models for time series prediction of target Adjusted Close Price.   
While working on this project, I found that a lot of academic work has been done applying RNNs for time series data.   
**Data and Inputs**  
For any machine learning problem finding appropriate data is vital, fortunately python library 'fix-yahoo-finance' was able to provide required dataset.
I have used data from year 2000 till current date.  
This data is already **sorted**, meaning 1st record is of date 2000-01-01 and last record is of current date. 

### Problem Statement

				For a given a stock ticker, train a predictive model, to predict the close value for next 7 open market days. 
 
The model shall be trained on historical time series data of a given stock and the model shall be backtested to compare its accuracy over the span of time period.
This model then shall be applied different stocks to see its performance.    
Considering problem statement we will be interested in training models in **Univariate Multistep** and **Multivariate Multistep Step**.

### Project Organization

Project Submissions
- report/capstone_report_final.pdf

Project Development
- dev/Data Preprocessing
- dev/Multi Step Models
- dev/Single Step Models

Project Data
 - data/AAPL
 - data/AAPL_final
 - data/AAPL_processed
 - data/IBM
 - data/IBM_final
 - data/IBM_processed
 - data/MSFT
 - data/MSFT_final
 - data/MSFT_processed 
 - data/AMZN
 - data/AMZN_final
 - data/AMZN_processed

Project saved work 
- reports/HTMLs - saved ipynb files
- res - images and screenshots


------------

**Links**  
1. [What is Stock Market](https://en.wikipedia.org/wiki/Stock_market)  
2. [Stocks Tutorials](https://www.khanacademy.org/economics-finance-domain/core-finance/stock-and-bonds)    
3. [Stock Trading vs Stock Investment](https://www.nerdwallet.com/blog/investing/stock-trading-vs-investing-whats-the-difference/)    
4. [Different Market Index and Why I choose S&P 500](https://www.youtube.com/watch?v=LRC7N-qD3XA)  
5 . [https://sigmoidal.io/machine-learning-for-trading/](https://sigmoidal.io/machine-learning-for-trading/)  
6. [https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)  
7. [https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data](https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data)  
8. [Understanding LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
9. https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
10. https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
11. https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
12. https://stackoverflow.com/questions/24901637/what-is-a-recurrent-neural-network-what-is-a-long-short-term-memory-lstm-netw
13. https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
14. https://www.investopedia.com/terms/m/macd.asp
15. https://www.investopedia.com/terms/s/stochasticoscillator.asp
16. https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

