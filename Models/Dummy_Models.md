When doing supervised learning, a simple sanity check consists of comparing one’s estimator against simple rules of thumb. DummyClassifier and DummyRegressor implements several such simple strategies for classification and regression respectively. These strategies completely ignore the input data.

DummyClassifier and DummyRegressor (sklearn.dummy) makes predictions that ignore the input features. This serves as a simple baseline to compare against other more complex models. The specific behavior of the baseline is selected with the strategy parameter. All strategies make predictions that ignore the input feature values passed as the X argument to fit and predict. The predictions, however, typically depend on values observed in the y parameter passed to fit. If a complex model cannot outperform a DummyClassifier or DummyRegressor, then there may be issues with the feature engineering process or model selection.

The “stratified” and “uniform” strategies lead to non-deterministic predictions that can be rendered deterministic by setting the random_state parameter if needed. The other strategies are naturally deterministic and, once fit, always return the same constant prediction for any value of X

Links:
[DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier)
[DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor)