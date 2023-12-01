[see sklearn's decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
Decision Trees are versatile machine learning algorithms that can perform both classification and regression tasks, and even multioutput tasks. One of the many qualities of Decision trees is that they require very little data preparation. in fact, they don't require feature scaling or centering at all.

Decision Trees make very few assumptions about the training data. If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely, indeed, most likely overfitting it. This kind of model is often called a nonparametric model because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data. In contrast, a parametric model, such as a linear model, has a predetermined number of parameters so its degree of freedom is limited, thus reducing the risk of overfitting but increasing the risk of underfitting.

To avoid overfitting the training data, you need to restrict the Decision Tree's freedom during training. **In sklearn.tree's Decision tree, increasing min_* hyperparameters or reducing max_* hyperparameters will regularize the model.** Also, this model uses the CART(Classification and Regression Tree) algorithm which produces only binary trees: nonleaf nodes always have two children(i.e questions always have yes/no answers). However, other algorithms such as ID3 can produce Decision Trees with nodes that have more than 2 children. Binary splits are simpler and faster, but they may not capture the best separation of the data. Multi-way splits are more flexible and accurate, but they may increase the tree size and reduce the generalization ability.. **While Tree-based methods can handle categorical variables directly, without the need for encoding or transformation, sklearn.tree.DecisionTree cannot, at the time of this writing, handle categorical variables**


#### Parameters
- min_samples_split (the minimum number of samples a node must have before it can be split), 
- min_samples_leaf (the minimum number of samples a leaf node must have), 
- min_weight_fraction_leaf (same as min_samples_leaf but expressed as a fraction of the total number of weighted instances), 
- max_leaf_nodes (the maximum number of leaf nodes), and 
- max_features (the maximum number of features that are evaluated for splitting at each node).

**Notes**
- Decision trees are target transforming algorithms and as such, cannot extrapolate target values beyond the training set.
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
- sklearn.tree.DecisionTree can readily be used to support multi-output problems
- Decision trees tend to overfit on data with a large number of features. Getting the right ratio of samples to number of features is important, since a tree with few samples in high dimensional space is very likely to overfit.
- Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand to give your tree a better chance of finding features that are discriminative.
- The number of samples required to populate the tree doubles for each additional level the tree grows to. Use max_depth=3 as an initial tree depth to get a feel for how the tree is fitting to your data, and then increase the depth. This helps prevent overfitting.
- For regression problems, try min_samples_leaf=5 as an initial value. If the sample size varies greatly, a float number can be used as percentage in these two parameters. For classification with few classes, min_samples_leaf=1 is often the best choice.
- Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. Class balancing can be done by sampling an equal number of samples from each class, or preferably by normalizing the sum of the sample weights (sample_weight) for each class to the same value. 
- Checkout sklearn's [tips on practical use](https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use) for more help on using its DecisionTree Algorithm

By default, sklearn.tree.DecisionTreeClassifier uses the gini impurity as it's function to measure the quality of a split but one can also select the entropy impurity measure. A node is "pure"(gini=0) if all training instances it applies to belong to the same class. Most of the time, [it does not make a big difference using either gini impurity or entropy](https://homl.info/19). Gini impurity is slightly faster to compute, so it is a good default. However, when they differ, gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced tree.

