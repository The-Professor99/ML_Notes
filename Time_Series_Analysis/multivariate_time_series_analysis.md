
### Multivariate Time Series Analysis:
A Multivariate time series refers to a dataset where multiple variables are recorded over successive time intervals. In other words, it is a collection of observations where each observation consists of measurements of multiple variables taken at specific points in time. This is in contrast to univariate time series where you have only one variable changing over time.
For Example, consider a scenario where you want to analyze the economic health of a country. Instead of just looking at a single variable like the Gross Domestic Product(GDP), you might consider a range of economic indicators such as unemployment rate, inflation rate, consumer spending and government debt. All measured at different time points(eg. monthly or quaterly).
These variables could be related, influencing each other, or responding to similar external factors and hence, applications of multivariate time series analysis range from forecasting, understanding causal relationships, identifying patterns and anomalies, and making informed decisions based on the interactions between variables.
One way to model multivariate time series is using VARIMA(Vector Autoregressive Integrated Moving Average) which combines VAR and ARIMA concepts to model both autoregressive and moving average relationships in a multivariate context.

#### Vector Autoregression(VAR):
This model assumes that each variable in the time series is influenced by its past values as well as the past values of other variables in the system.