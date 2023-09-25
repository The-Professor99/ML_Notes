### Stationarity
The requirement of stationary data is particularly emphasized in traditional time series models like ARIMA(check arima notes for more info) due to their explicit assumptions about the data's statistical properties. However, when it comes to more complex models like RNNs, the situation is a bit different.
Recurrent Neural Networks, including LSTM and Gated Recurrent Unit(GRU) networks, are capable of capturing intricate temporal dependencies within data, even if it's non-stationary to some extent. RNNs inherently handle sequences and can learn to adapt to changing patterns and trends over time, making them more flexible in handling non-stationary sequences. As such, while stationarity is beneficial for many modeling techniques, including RNNs, it's not always a strict requirement.
Also, RNNs often benefit from input data that is well-scaled and normalized. This also helps mitigate the impact of non-stationarity to some extent.

However, while mild non-stationarity might not pose a significant problem for RNNs, extreme non-stationary, stron trends or drastic changes can still affect the model's ability to learn effectively.

### Time Step Features.
Time step features, also known as time-step embeddings or time-step indicators are a type of feature engineering technique used in time series analysis and forecasting. They aim to encode information about the sequential order or time steps of observations within a time series and allow linear regression models to account for systematic variations that occur at specific time intervals. Time step features lets you model **time dependence**. A series is time dependent if its values can be predicted from the time they occured.
#### Time Dummies:
A time dummy basically counts off time steps in a series from the beginning to the end and are used to represent the effect of categorical time-related factors such as days of the week, months, or seasons. 
#### Ways to create time step features:
- Ordinal Encoding: You can assign a unique numerical value to each time step starting from the beginning of the time series. 
- One-Hot Encoding: Instead of using numerical values, you can represent time steps using one hot encoding. Each time corresponds to a binary indicator variable (0 or 1) where only one indicator is "on" for each time step. For example, in a time series of daily data, you'd have 365binary features(one for each day) and the feature corresponding to the current time step would be on while the rest would be off.
- Periodic Encoding: If the time series exhibits strong seasonality, you might encode time steps in a way that captures the cyclic nature of the data. For example, one could use sine and cosine functions to represent time steps and capture seasonal patterns.


### Trend
The trend component of a time series represents a persistent, long term change in the mean of the series. (Moving averages).


#### Detrending
Detrending a time series involves removing the underlying trend component from the data, leaving behind the seasonality(if any( and irregular components. It is usually performed to better understand and model the cyclical and irregular patterns in the data. Methods of detrendig
1. Moving Averages(see deseasoning)
2. Polynomial Regression: Fit a polynomial regression model to the data, where the degree of the polynomial depends on the complexity of the trend. eg, for a linear trend, use a 1st degree polynomial. subtract the predicted trend values from the original data.
3. Differencing(see deseasoning)


### Seasonality
Seasonal indicators are binary featurs that represent seasonal differences in the level of a time series. Eg. By one-hot-encoding days of the week, we get weekly seasonal indicators. 
Adding seasonal indicators to the training data helps models distinguish means within a seasonal period.
Another kind of feature for seasonality is Fourier Features. They are better suited for long seasons over many observations where indicators would be impractical. Instead of creating a feature for each date, Fourier features try to capture the overall shape of the seasonal curve with just a few features. The idea is to include in our training data, periodic curves having the same frequencies as the season we are trying to model. The curves we use are trigonometric functions sine and cosine. Fourier features are pairs of sine and cosine curves, one pair for each potential frequency in the season starting with the longest. By modeling only the "main effect" of seasonality with Fourier Features, you'll usually need to add far fewer features to your training data which means reduced computation time and less risk of overfitting.

#### Choosing Fourier features
A periodogram tells the strength of frequencies in a time series, as such, can be used to tell how many fourier pairs should be included in a feature set.


### Lag features
Lag features represent past values of the target variable or other relevant features as predictors. They allow a model to capture how the current value depends on its previous values.
To make a lag feature we shift the observations of the target series so that they appear to have occured later in time. 
<strong>Autocorrelation</strong> means the correlation a time series has with one of its lags and its commonly used to measure serial dependence. If the lag feature correlates with the target, it means the lag feature will be useful

While Time Step Features model **time dependence**, Lag features model **serial dependence**. A time series has serial dependence when an observation can be predicted from previous observations.

One common way for serial dependence to manifest is in cycles. Cycles are patterns of growth and decay in a time series and are associated with how the value in a series at one time depends on values at previous times, but not necessarily on the time step itself. Cyclic behaviour is  characteristic of systems that can affect themselves or whose reactions persist over time.
<strong>One common difference between cyclic behaviour and seasonality is that cycles are not necessarily time dependent, as seasons are. What happens in a cycle is less about the particular date of occurence and more about what has happened in recent past.</strong> This independence from time means that cyclic behaviour may be much more irregular than seasonality.

#### Choosing Lag Features
When choosing lags as features, it generally wont be useful to include every lag with a large autocorrelation, doing this may result in multicollinearity and also, the newer lag features may just be decayed information(correlation carried over) from previous lag features. For this reason, <strong>Partial autocorrelation is more useful</strong>. 

Partial autocorrelation calculates the correlation between a specific time point and another time point at a certain lag while "partialing out" the influence of the time points in between(i.e it removes the indirect influence of the time steps in between and only considers the influence of the specific lag). The PACF(partial autocorrelation function) at lag k measures the correlation between the time series at time t and the time series at time t-k, with the influence of time points t-1, t-1, ..., t - k+1 removed, thus isolating the direct influence of time point t-k on time point t.
- A high PACF value at a particular lag suggests a strong direct influence of the time point at lag k on the current time point while a low PACF value at a lag indicates that the direct influence of that lag on the current time point is minimal once the influence of intermediate lags is taken into account.

A correlogram  is used in choosing lag features in the same manner as a periodogram is used in choosing fourier features. Autocorrelation and partial autocorrelation are measures of linear dependence, however, real-world series often have substantial non-linear dependencies, as succh, it is best to look at a lag plot(a scatter plot of a time series against one of its lagged values) when choosing lag features as they allow you to visually identify patterns or trends in the data and you can spot linear and non-linear dependencies between the target variable and its lagged versions.

#### Deseasoning
Deseasoning a time series involves removing the seasonal component from the data to isolate the underlying trend and any remaining irregular or residual components. This process helps to study and model the non-seasonal patterns in the data more effectively.
Common ways to deseason a time series include:
1. Moving Averages(simple, weighted): calculate the moving average of the time series with a window size that matches the seasonal cycle's length. Eg, if you have monthly data with an annual seasonality, use a 12month ma, then subtract the ma from the original data to deseason it. When the seasonality is not strictly regular, you can use a weighted ma to give more importance to recent observations.
2. Seasonal Decomposition(Additive, multiplicative): Decompose the time series into 3 components: trend, seasonality and residual(using statsmodel's `seasonal_decompose` function. Once you have the decomposition, subtract the seasonal component from the original data to obtain the deasoned series.
3. Seasonal Differencing: Take the difference between the time series and a lagged version of itself corresponding to the seasonal cycle's length. This can be effective when seasonality is stable over time.
4. Regression Models: Fit a regression model to the time series data, including seasonal indicators representing the seasonality. The coefficients of these seasonal indicators capture the seasonal effect. subtract them from the data.

We could imagine learning the components of a time series as an iterative process: first learn the trend and subtract it out from the series, then learn the seasonality from the detrended residuals and subtract the seasons out, then learn the cycles and subtract the cycles out, and finally only the unpredictable error remains.

### Defining the Forecasting Task
Two things to establish before designing a forecast model:
- what information is available at the time a forecast is made(features).
- the time period during which you require forecasted value(target)
**Forecast origin** is the time at which you are making a forecast. Everything up to the origin can be used to create features.
**Forecast horizon** is the time for which you are making a forecast. This describes the target.
**Lead time** is the time between the origin and the horizon and in practice, it may be necessary for a forecast to begin multiple steps ahead of the origin because of delays in data acquisition or processing.


