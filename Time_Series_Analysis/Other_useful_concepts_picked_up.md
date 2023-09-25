### Deterministic Terms and Deterministic Process

A deterministic process is a model or system where the output is entirely determined by known relationships, fixed inputs and deterministic functions, with no randomness or uncertainty involved.
Key characteristics of a deterministic process include:

- The output is entirely predictable based on the input and the known relationships.
- There is no inherent randomness or variability in the process.
- The same set of inputs will always produce the same output.

Deterministic terms are components of a model that are explicitly specified based on prior knowledge or theoretical considerations. These terms are not estimated from the data but are included to represent known relationships or patterns.
Examples of Deterministic terms:

- Intercept Term: A Constant term(intercept) in linear regression models.
- Time trends: Deterministic trends may be added to capture known linear and nonlinear patterns in the data.
- Seasonal Effects: Seasonal deterministic terms represent known patterns that repeat at regular intervals such as daily, weekly or yearly seasonal effects in time series data.
- Categorical variables: When modeling categorical variables with fixed levels,dummy variables(binary indicators) are often used as deterministic terms to represent the categories.

### Linear Dependence

Linear dependence refers to the degree to which one variable can be predicted or explained by a linear combination of other variables.

### Designing hybrid models in time series

_(check Designing_hybrid_ensemble_models.md for a build up to this)_
In the context of time series, If the time dummy continues counting time steps, linear regression(feature transforming algorithm) models will continue to draw a trend line. However, given this same time dummy, a decision tree(target transforming algorithm) will predict the trend indicated by the last step of the training data of the training data into the future forever as decision trees cannot extrapolate trends. Random Forests and gradient boosted decision trees (like XGBBoost) are ensembles of decision trees hence they can also not extrapolate trends.
We'll usually want to use different feature sets (X_train_1 and X_train_2 above) depending on what we want each model to learn. If we use the first model to learn the trend, we generally wouldn't need a trend feature for the second model, for example.
When designing hybrid models to handle time series, you may want to use feature-transforming models(eg Linear Regression) to extrapolate the trend, transform the target to remove the trend and then apply a target-transforming algorithm(eg XGBoost) to the detrended residuals. To hybridize a neural net, you could instead include the predictions of another model as a feature which the neural net would then include as part of its own predictions.
While it's possible to use more than two models, in practice it doesn't seem to be especially helpful. The strategy of using a simple(usually linear) learning algorithm followed by a complex, non-linear learner like GBDTs or a deep neural net with the simple model typically designed as a "helper" for the powerful algorithm is quite common.
