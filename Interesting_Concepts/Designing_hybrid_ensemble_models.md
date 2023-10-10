There are generally two ways a regression algorithm can make predictions: 
1. By transforming the _features_: Feature transforming algorithms learn some mathematical function that takes features as an input and then combines and transforms them to produce an output that matches the target values in the training set. Linear Regression and Neural Nets are of this kind.

2. By transforming the _target_: Target transforming algorithms use the features to group the target values in the training set and make predictions by averaging values in a group. A set of features just indicates which group to average. Decision trees and nearest neighbors are of this kind.

The important thing is: Feature transformers generally can extrapolate target values beyond the training set given appropriate features as inputs, but the predictions of target transformers will always be bound within the range of the training set. 

#### Multioutput-problems.
A multi-output problem is a supervised learning problem with several outputs to predict, that is when Y is a 2d array of shape (n_samples, n_outputs). When there is no correlation between the outputs, a very simple way to solve this kind of problem is to build n independent models, i.e. one for each output, and then to use those models to independently predict each one of the n outputs. However, because it is likely that the output values related to the same input are themselves correlated, an often better way is to build a single model capable of predicting simultaneously all n outputs. First, it requires lower training time since only a single estimator is built. Second, the generalization accuracy of the resulting estimator may often be increased.
