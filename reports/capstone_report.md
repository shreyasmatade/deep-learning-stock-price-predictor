# Machine Learning Engineer Nanodegree
## Capstone Project
Shreyas Matade  
January 13st, 2019

## I. Definition
Deep Learning Stock Price Predictor  

### Project intuition

Like most of the tech engineers, I was impressed and surprised by the bitcoin-cryptocurrency boom. Like any other human, I was drawn to FOMO and started investing in crypto (and I can say proudly, i did not lost nor gained ). This is where I started my curiosity toward trading bots.  
After taking Machine learning nanodegree course, I learned a lot about stats, data analysis and various machine learning models, methodologies.  
I really wanted to apply my newly learned skills to work. However, cryptocurrency market is highly volatile and as of now it is becoming unreliable. Hence, just to get the taste of analyzing time series data I thought of exploring "comparatively" more reliable Stock Market.

### Domain background
**Stocks Market Trading**   

Stock markets are getting more and more influenced by the **Algorithms in Trading - Computational Trading**. When algorithmic trading strategies were first introduced, they were wildly profitable and swiftly gained market share. In May 2017, capital market research firm Tabb Group said that high-frequency trading (HFT) accounted for 52% of average daily trading volume. But as competition has increased, profits have declined. In this increasingly difficult environment, traders need a new tool to give them a competitive advantage and increase profits. Algorithms like simple moving average, Exponential moving average are fair to predict stock prices for shorter period in future but these techniques has limitations to predict prices at far time in future. These limitations can be avoided by momentum based algorithms. But more appropriate solution to this problem would be **Machine Learning.**  

**Machine Learning**  

Machine learning techniques are much more capable of learning hidden patterns in the historical data and come up with better and profitable strategy. We can use Supervised Machine learning algorithms 
 linear regressions, neural networks, support vector machines, and naive Bayes, to name a few.
However, time series prediction problems are a difficult type of predictive modeling problem. This Time series nature of stocks adds complexity of a sequence dependence among the input variables. Such a complexity can be handled by better by applying **Recurrent Neural networks** a.k.a **RNN**. A powerful type of neural network designed to handle sequence dependence is called recurrent neural networks. The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.

###Project outline

In this project, I am applied from simple supervised learning methods that I learned over this machine learning coursework to the LSTM RNN models.   
While working on this project, I found that a lot of academic work has been done applying RNNs for time series data.   
**Data and Inputs**  
For any machine learning problem finding appropriate data is vital, fortunately python library 'fix-yahoo-finance' was able to provide required dataset.
I have used data from year 2000 till current date.  
This data is already **sorted**, meaning 1st record is of date 2000-01-01 and last record is of current date. 

1. Apple(AAPL)  
2. Microsoft(MSFT)  
3. Amazon(AMZN)  
4. IBM(IBM)     
  
This data is logged on daily basis. We will discuss inputs in this data in later part in detail.  
For now,  
Our target variable is **"Adj Close Price"**.  
Our features (input variables) are derived from basic input variables in initial data We will elaborate on that in later part.  
**Date Prepossessing**  
Since, we are using LSTM RNN for our forcasting target and the fact that we are dealing with Time Series data I felt prepossessing data and come up with significant features was challenging. We will elaborate on this in later part.  
**Predictive Modeling**  
As discussed, we are trying to predict "Adj Close" price for given stock in future. As we are interested in predicting not only next day data but multiple day data, this makes this problem more complicated. There are several models we can apply, we are concentrating on RNN model LSTM. Considering the nature of inputs and nature of target we are planning to predict, that makes this problem Multistep Prediction problem. 

We will apply and use multiple models and see comes best on our metric, RMSE (root mean square error) 

We also compare between Univariate Models (where single feature is used ) versus Multivariate Models (where multiple features are used ) and see added complexity is worth to achieve any better metric.     

### Problem Statement

				For a given a stock ticker, train a predictive model, to predict the close value for next 7 open market days. 
 
The model shall be trained on historical time series data of a given stock and the model shall be backtested to compare its accuracy over the span of time period.
This model then shall be applied different stocks to see its performance.    
Considering problem statement we will be interested in training models in **Univariate Multistep** and **Multivariate Multistep Step**.


### Metrics

Our forecast will be comprised of seven values, since we are predicting for 7 days in future.It is common with multi-step forecasting problems to evaluate each forecasted time step separately. 

This is helpful for a few reasons:
1. To comment on the skill at a specific lead time (e.g. +1 day vs +3 days).
2. To contrast models based on their skills at different lead times (e.g. models good at +1 day vs models good at days +5).

Since our target variable is in USD, our forecasted value should be in USD as well
Mean Absolute Error (MSE) and Root Mean Square Error (RMSE) both are applicable here as metric. I choose RMSE since it is more punishing of forecast errors.

The performance metric for this problem will be the RMSE for each lead time from day 1 to day 7.


The function evaluate_forecasts() is implemented to calculate RMSE and return the performance of a model based on multiple seven-day forecasts.

**1 Week or 5 Days ?**  
Since, Market is open only 5 days in a week and there is some discontinuity in the time series data. However, the problem statement makes it clear to predict Adj Close value on next 7 days when Market is open. So we should be OK to use this metric and forecast evaluation. And discounitnuity in data shall be taken care off by our predictive model, since it is occurring relatively periodically. This can be seen distribution of one of our feature variable.

![Distribution of Days Since Market Open](https://github.com/shreyasmatade/capstone/blob/master/res/dslo.PNG "Logo Title Text 1")

So, applying this forecast evaluation we can choose model which performs best and given best RMSE value over the 7 day prediction.


## II. Analysis

### Data Exploration
#### Datasets
I will be using [fix-yahoo-finance](https://pypi.org/project/fix-yahoo-finance/) python package to get the stock data. Below are the variables of the time series data of a single stock.  
Solution shall be able to get the historical data given a Ticker symbol of the desired stock. 
I will be using data for following tickers from `START_DATE = '2000-01-01'` till current date.

1. Apple(AAPL)
2. Microsoft(MSFT)
3. Amazon(AMZN)
4. IBM(IBM) 
 
#### Dataset variables

The datasets we get from fix-yahoo-finance has below columns. This is a time series daily data, **One record for one day**
Date is the index for the dataset. Ex: 2000-01-01

**1. Open**       - The opeing price of the stock for given day.     
**2. High**       - Maximum price of the stock for given day.   
**3. Low**        - Minimum price of the stock for given day.  
**4. Close**      - Price of the stock at which day was closed for market.  
**5. Adj Close**  - An adjusted closing price is a stock's closing price on any given day of trading that has been amended to                         include any distributions and corporate actions that occurred at any time before the next day's open. 
                    **This will be our Target variable in prediction**  
**6. Volume**     - This value is the indicator of total transactions of the stock market buys/sells for given ticker.

I have used 'Adj Close' as my target variable. And for the features i will using techinical indicators which are derived from above mentioned inputs. 

#### Storing Data

We are using **fix-yahoo-finance** library. To avoid frequent query to this library database and downloads I am saving data locally in a csv file. I am using 'ALLOWED_DELTA' constant to see if there is any need to get latest data. if the time delta is greater than this constant  then only I pull data from fix-yahoo-finance else I use pre-stored data from csv file.

`# Max allowed time delta`    
`ALLOWED_DELTA = datetime.timedelta(days=7) `  
`START_DATE = '2000-01-01'`

All csv file are stored in `data` folder

#### Impute Data

Since we are dealing with the time series data, we cannot just replace 'nan' or missing values by over all mean, since data is sequence dependent and overall mean will not appropriate value to replace any missing values in data.
There are two methods provided by `pandas` to resolve this problem. We need to be careful in using these techniques though. First we have to apply 'Fill Forward' and then apply 'Fill Backwards' in order to avoid any `Look ahead` issues.  

Hence, I am using `fill forward` and then `fill backward` method to replace the `nan` in the data.

#### Need for Feature Exploration

Finding a better feature for Stock Market prediction is a topic for more exploration and research. I found more I explore more different indicators I find. It is only when I apply them in model, I would get better idea about whether they are helping the model learn or making it worst.

I am considering to use 5 features, shifted version **Adj Close**  (Shift depends on the `lookback` factor ) and 3 technical indicators and one functional indicator.

- MACD (Trend)
- MOMETM (Momentum)
- Average True Range (Volume)
- Days since market open (Functional Indicator) 

#### Details on Feature Calculations

**Exponential Moving Average:** An exponential moving average (EMA) is a type of moving average (MA) that places a greater weight and significance on the most recent data points.

`Initial SMA: 10-period sum / 10 `  
`Multiplier: (2 / (Time periods + 1) ) = (2 / (10 + 1) ) = 0.1818 (18.18%)`  
`EMA: {Close - EMA(previous day)} x multiplier + EMA(previous day).`  

https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages

**MACD:**  The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. 

https://www.investopedia.com/terms/m/macd.asp


**Stochastics oscillator:** The stochastic oscillator is a momentum indicator comparing the closing price of a security to the range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result.

https://www.investopedia.com/terms/s/stochasticoscillator.asp

**Average True Range:** Is an indicator to measure the volatility (NOT price direction). The largest of:
- Method A: Current High less the current Low
- Method B: Current High less the previous Close (absolute value)
- Method C: Current Low less the previous Close (absolute value)

https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

Calculation:

![alt text](http://i68.tinypic.com/e0kggi.png "Logo Title Text 1")

**DSLO** : 
	It is the days since market is open. This might affect the prices if market was closed for multiple days. My gut feeling tells me this might come handy in predicting target.

	We calculate this indictor by subtracting consucutive timestamp and take the day difference.

`def DaySinceLastOpen(df):` 
    `"""This function assumes that timeseries df indexed by date is passed """`  
    `DSLO = [] ` 
    `DSLO.append(1.0)`      
    `for i in range(df.shape[0]-1):`  
        `diff = pd.to_datetime(df.index[i+1]) - pd.to_datetime(df.index[i])`  
        `DSLO.append(float(diff.days))`  
    `return pd.Series(DSLO, index=df.index)` 		
	  
    Since `dslo` is a categorical variable we need to apply one hot encoding on this variable before we use this as a feature. We use one-hot encoding to convert this categorical variable.

#### Sample Data and Stats
For IBM ticker
https://github.com/shreyasmatade/capstone/blob/master/res/sample_data.PNG

**Data Info**
`final_data.info()`
`<class 'pandas.core.frame.DataFrame'>`
`DatetimeIndex: 4774 entries, 2000-01-21 to 2019-01-11`
`Data columns (total 12 columns):`
`Adj Close    4774 non-null float64`
`MACD         4774 non-null float64`
`MOMETM       4774 non-null float64`
`ATR          4774 non-null float64`
`dslo_0       4774 non-null float32`
`dslo_1       4774 non-null float32`
`dslo_2       4774 non-null float32`
`dslo_3       4774 non-null float32`
`dslo_4       4774 non-null float32`
`dslo_5       4774 non-null float32`
`dslo_6       4774 non-null float32`
`dslo_7       4774 non-null float32`
`dtypes: float32(8), float64(4)`
`memory usage: 335.7 KB`

**Basic Statistics**
https://github.com/shreyasmatade/capstone/blob/master/res/basic_stats.PNG

From Initial analysis, it is very clear that feature 'dslo_6' and 'dslo_0' are not really required in the analysis as its value is 0 for all records.
So, we drop these two columns from our final data.


### Exploratory Visualization

We plot features 'ATR', 'MACD' and 'MOMETM' along with Target variable 'Adj Close'. We avoid functional indicator 'DSLO' since it will be difficult to see its affect on target variable by visualization, we rely on predictive models for that.

https://github.com/shreyasmatade/capstone/blob/master/res/eda_ibm.PNG

https://github.com/shreyasmatade/capstone/blob/master/res/eda_aapl.PNG

https://github.com/shreyasmatade/capstone/blob/master/res/eda_msft.PNG

https://github.com/shreyasmatade/capstone/blob/master/res/eda_amzn.PNG

By observing above visualization, we can see there is obvious relation between ATR, MACD and Target variable, Adj Close. When there is movement in target variable there is corresponding fluctuation in ATR and MACD. It will be interesting to see how it helps the model for better prediction.
MOMETM seems highly fluctuating for all values of target. We will later explore if really need this variable.  

### Algorithms and Techniques

We have already discussed feature exploration and how we calculated different technical indicators and prepared our dataset. These techniques were in detail discussed above.

**Algorithms**

We are using Persistence model as a baseline model.  

**Persistance Model Algorithm** 
A baseline in forecast performance provides a point of comparison.
It is a point of reference for all other modeling techniques on your problem. If a model achieves performance at or below the baseline, the technique should be fixed or abandoned.
I am using Persistance model as my baseline model. 
I have chosen this as my baseline model as it has following three characteristics  
**Simple:** A method that requires little or no training or intelligence.  
**Fast:** A method that is fast to implement and computationally trivial to make a prediction.  
**Repeatable:** A method that is deterministic, meaning that it produces an expected output given the same input. 
   
As name suggest persistance alorithm is easy and simple, to predict value at timestamp `t` it simple repeats value at timestamp `t-1`.

https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/

We have used Long Short Temporal Memory (LSTM) model for predicting target for given time series.


**LSTM Network**   
The Long Short-Term Memory network, or LSTM network, is a recurrent neural network that is trained using Backpropagation Through Time and overcomes the vanishing gradient problem.

As such, it can be used to create large recurrent networks that in turn can be used to address difficult sequence problems in machine learning and achieve state-of-the-art results.

Instead of neurons, LSTM networks have memory blocks that are connected through layers.

A block has components that make it smarter than a classical neuron and a memory for recent sequences. A block contains gates that manage the block’s state and output. A block operates upon an input sequence and each gate within a block uses the sigmoid activation units to control whether they are triggered or not, making the change of state and addition of information flowing through the block conditional.

There are three types of gates within a unit:

	Forget Gate: conditionally decides what information to throw away from the block.
	Input Gate: conditionally decides which values from the input to update the memory state.
	Output Gate: conditionally decides what to output based on input and the memory of the block.  
Each unit is like a mini-state machine where the gates of the units have weights that are learned during the training procedure.

In this project, we have used LSTM for a regression problem. We will explore data preparation and forecast evaluation two most challenging aspects of using LSTM models in later sections.


### Benchmark
As we discussed on the Metric section, we are using RMSE as metric for the model. When we run our baseline model.

Following results were obtained when we ran persistence model, to predict target variable for next 7 days.

**For IBM**             
t+1 RMSE: 2.399258
t+2 RMSE: 2.705824
t+3 RMSE: 3.096104
t+4 RMSE: 4.083876
t+5 RMSE: 4.997928
t+6 RMSE: 5.853293
t+7 RMSE: 7.019526
Overall RMSE: 4.307973     
**For Microsoft**  
t+1 RMSE: 3.264457
t+2 RMSE: 3.248963
t+3 RMSE: 2.899928
t+4 RMSE: 3.381786
t+5 RMSE: 3.480314
t+6 RMSE: 2.418524
t+7 RMSE: 3.553691
Overall RMSE: 3.178238    
**For Apple**  
t+1 RMSE: 7.298900
t+2 RMSE: 8.191651
t+3 RMSE: 8.518772
t+4 RMSE: 8.911374
t+5 RMSE: 9.241080
t+6 RMSE: 6.401026
t+7 RMSE: 5.919230
Overall RMSE: 7.783148   
**For Amazon**    
t+1 RMSE: 54.841590
t+2 RMSE: 64.709003
t+3 RMSE: 81.138913
t+4 RMSE: 107.418907
t+5 RMSE: 135.605513
t+6 RMSE: 151.213724
t+7 RMSE: 168.519983
Overall RMSE: 109.06394  

https://github.com/shreyasmatade/capstone/blob/master/res/baseline_ibm_result.PNG
https://github.com/shreyasmatade/capstone/blob/master/res/baseline_msft_result.PNG
https://github.com/shreyasmatade/capstone/blob/master/res/baseline_aapl_result.PNG
https://github.com/shreyasmatade/capstone/blob/master/res/baseline_amzn_result.PNG

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

This was the most challanging part of the project. Since, we are dealing with Time Series data 

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
