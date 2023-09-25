### ARIMA
_(Autoregressive Integrated Moving Average)_
- A statistical model is autoregressive if it predicts future values based on past values. Eg. An ARIMA  model might seek to predict a stock's future prices based on it's past performance or forecast a company's earnings based on past periods. 
- Autoregressive models implicitly assume that the future will resemble the past hence they can prove inaccurate under certain conditions such as rapid changes. Also, many time series models, including ARIMA  , assume that the data's statistical properties do not change over time. As such, they work best with stationary time series. 

#### Stationarity
- A stationary time series is one where statistical properties, such as mean, variance and autocorrelation, remain constant over time. I.e, the data doesn't exhibit any significant trend, seasonality or structural changes. 
- One of the techniques used to achieve stationarity is differencing. Differencing involves subtracting a previous observation from the current observation in order to remove the effects of trends or seasonality. The "Integrated" component in ARIMA(denoted as ARIMA(p,d,q)) refers to the order of differencing (d) needed to make the time series stationary.
- One reason why ARIMA models better perform with stationary data is that Non-stationary data with trends or changing variances can lead to erratic patterns and inaccurate model forecasts. Stationary data helps stabilize these characteristics, making it easier to identify meaningful patterns.
<strong>Note however that incorrectly applying differencing can lead to overfitting or underfitting. Eg if you apply one-step differencing to the series [3, 5, 7, 9, 11], you get the differenced time series [2, 2, 2, 2]. If the original time series has a quadratic trend instead of a linear trend, then a single round of differencing will not be enough. Eg, the series [1, 4, 9, 16, 25, 36] becomes [3, 5, 7, 9, 11] after one round of differencing, but if you run differencing for a second round, then you get [2, 2, 2, 2]. Running two rounds of differencing will eliminate quadratic trends. More generally, running d consecutive rounds of differencing computes an approximation of the dth order derivative of the time series, so it will eliminate polynomial trends up to degree d. d is called the order of integration.</strong>

ARIMA models have strong points and are good at short term forecasting based on past circumstances, but there are more reasons to be cautious when using ARIMA. In stark contrast to investing disclaimers(when used to forecast financial data) that state "past performance is not an indicator of future performance...," ARIMA models assume that past values have some residual effect on current or future values and use data from the past to forecast future events.

#### Appropriate orders of integration.
While there's no one-size-fits-all technique, some of the approaches one can use to decide the order of integration include:
1. Visual Inspection: Plot the time series data and examine its behaviour. Look for any trends(A trending series might indicate non-stationarity), seasonality, or irregular patterns. If there are clear trends, differencing might be necessary to remove them. The number of times you need to difference to achieve stationarity can give you an initial idea of the order of integration. 
2. Augmented Dickey-Fuller(ADF) test: This statistical test helps determine whether differencing is needed to make a time series stationary. It provides a p-value that indicates the significance of differencing. A low p-value suggests that differencing is needed. You can use the test iteratively with increasing orders of differencing until you achieve stationarity.
3. Partial Autocorrelation Function(PACF): The PACF helps you identify the order of differencing needed. If the PACF drops off after a certain lag, it indicates that the data becomes stationary after that lag, implying the order of integration needed.
4. Domain Knowledge: Your understanding of the underlying process and the data's generating mechanisms can provide insights into the appropriate order of integration. For example, economic data might have a known seasonal pattern that guides the choice of differencing.
5. Experimentation: Try different orders of differencing and evaluate the results using diagnostic plots and information criteria. Iterate until you find a model that achieves stationarity without over-differencing.
6. Information Criteria: Criteria like the Akaike Information Criterion(AIC) and Bayesian Information Criterion(BiC) are used to select the optimal model parameters. When fitting ARIMA models with different orders of integration, you can compare AIC and BIC to find the best fitting model.
7. Automated Approaches: Some libraries, like the 'auto_arima' function in the 'pmdarima' package in python, can automatically determine the order of integration based on statistical tests and optimization criteria.

The "d" parameter in ARIMA specifies the order of differencing that the model should apply to the data before fitting the model. When you specify a value for "d," it instructs the ARIMA model to perform differencing on the data a certain number of times before using the resulting differenced series for analysis.
In this sense, you can think of the "d" parameter as a way of telling the ARIMA model how much pre-processing is required to achieve stationarity. The model itself does apply the differencing, and the differenced data is what the model uses for analysis. So, in this aspect, the model does handle the process of differencing.
However, the model doesn't automatically make the data stationary without you specifying the appropriate order of differencing. You need to provide the "d" parameter value based on your assessment of the data's stationarity. If you choose an incorrect value for "d," the model might not perform well, and the resulting analysis could be affected.

**ARIMA formula:**
$$\hat{y}_{(t)} = \sum_{i=1}^p \alpha_i y_{(t-i)} + \sum_{j=1}^q \theta_i \epsilon_{(t-j)} + C$$
$$with$$ 
$$\epsilon_t = y_t - \hat{y}_t$$

Where
- C is a constant. 
- $\hat{y}_t$ is the model's forecast for time step t.
- $y_t$ is the time series' value at time step t.
- The First sum in the equation is the weighted sum of the past p values of the time series, using the learned weights $\alpha_i$. The number p is a hyperparameter, and it determines how far back into the past the model should look. This sum is the AutoRegressive component of the model: it performs regression based on past values.
- The second sum is the weighted sum over the past q forecast errors $\epsilon_t$, using the learned weights $\theta_i$. The number q is a hyperparameter which determines how far back into the past the model should look when summing forecast errors using the learned weights. This sum is the moving average component of the model.

#### Endogenous Features
Endogenous variables are primary variables of interest in the time series analysis, they are the variables we want to predict or model. Eg, if you are building a time series forecasting model for daily sales data, the daily sales figures would be the endogenous variable.

#### Exogenous Features
Exogenous variables are external or additional variables that are not part of the primary time series but can influence or provide additional information for the modeling process. These variables are considered external to the time series and are often used as predictors or covariates to improve the accuracy of the time series model. They can be thought of as "input" features that help the model make better predictions and may include factors such as economic indicators, weather data, holiday calendars or other relevant data sources that can impact the endogenous time series.

