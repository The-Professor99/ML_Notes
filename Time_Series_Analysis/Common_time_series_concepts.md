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

### Lag features
Lag features represent past values of the target variable or other relevant features as predictors. They allow a model to capture how the current value depends on its previous values.
To make a lag feature we shift the observations of the target series so that they appear to have occured later in time. If the lag feature correlates with the target, it means the lag feature will be useful.

While Time Step Features model **time dependence**, Lag features model **serial dependence**. A time series has serial dependence when an observation can be predicted from previous observations.

### Trend
The trend component of a time series represents a persistent, long term change in the mean of the series. (Moving averages).

### Seasonality
Seasonal indicators are binary featurs that represent seasonal differences in the level of a time series. Eg. By one-hot-encoding days of the week, we get weekly seasonal indicators. 
Adding seasonal indicators to the training data helps models distinguish means within a seasonal period.
Another kind of feature for seasonality is Fourier Features. They are better suited for long seasons over many observations where indicators would be impractical. Instead of creating a feature for each date, Fourier features try to capture the overall shape of the seasonal curve with just a few features. The idea is to include in our training data, periodic curves having the same frequencies as the season we are trying to model. The curves we use are trigonometric functions sine and cosine. Fourier features are pairs of sine and cosine curves, one pair for each potential frequency in the season starting with the longest. By modeling only the "main effect" of seasonality with Fourier Features, you'll usually need to add far fewer features to your training data which means reduced computation time and less risk of overfitting.

#### Choosing Fourier features
A periodogram tells the strength of frequencies in a time series, as such, can be used to tell how many fourier pairs should be included in a feature set.