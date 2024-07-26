## Notes on Latent Semantic Analysis (LSA)

### Introduction
Text data often suffer from high dimensionality. Latent Semantic Analysis (LSA) is a popular dimensionality-reduction technique, similar to Principal Component Analysis (PCA), but specifically designed to handle sparse data typical in natural language processing (NLP). LSA reformulates text data in terms of r latent(i.e. hidden) features, where r is less than m(the number of terms in the data).

LSA is very similar to PCA but it operates better on sparse data than PCA does(and text data is almost always sparse). While PCA performs decomposition on the correlation/covariance matrix of a dataset, SVD/LSA performs decomposition directly on the dataset as it is, it is for this reason that LSA is used majorly in NLP tasks as LSA is free from any normality assumption of data(covariance calculation assumes a normal distribution of data).

LSA involves creating a term-document matrix where rows represent terms(words) and columns represent documents. This matrix is then decomposed using SVD and the decomposed matrices help to identify patterns and relationships between terms and documents.

LSA is different from SVD in the sense that LSA applies SVD to a term-document matrix in a specific way to uncover latent semantic structures. After applying SVD to this term-document matrix, dimensionality is reduced by selecting the top k singular values in $\sum$ and their corresponding singular vectors in $U$ and $V^T$. This results in a lower-rank approximation of the original matrix, capturing the most significant semantic information and reducing the noise.

### Implementation in SKLearn
In SKLearn, LSA is implemented as TruncatedSVD(which implements a variant of SVD that only computes the k(called n_components in SKlearn's api) largest singular values, where k is a user-specified parameter. TruncatedSVD is very similar to PCA, but differs in that the matrix X does not need to be centered. When the columnwise(per-feature) means of X are subtracted from the feature values, truncated SVD on the resulting matrix is equivalent to PCA. 

When TruncatedSVD is applied to term-document matrices(as returned by CountVectorizer or TfidfVectorizer), the transformation is known as LSA because it transforms such matrices to a "semantic" space of low dimensionality. In particular, LSA is known to combat the effects of synonymy and polysemy(both of which roughly mean there are multiple meanings per word), which cause term-document matrices to be overly sparse and exhibit poor similarity under measures such as cosine similarity.

### Recommendations for Using TruncatedSVD
While the TruncatedSVD transformer works with any feature matrix, using it on tf-idf matrices is recommended over raw frequency counts in an LSA/document processing setting. In particular, sublinear scaling and inverse document frequency should be turned on (sublinear_tf=True, use_idf=True) to bring the feature values closer to a Gaussian distribution, compensating for LSA's erroneous assumptions about textual data.

### Additional Notes
- LSA is also known as latent semantic indexing, LSI, though strictly that refers to its use in persistent indexes for information retrieval purposes.
- Using SVD to reduce the dimensionality of TF-IDF document vectors is often known as Latent Semantic Analysis(LSA) in the information retrieval and text mining literature.
- LSA is typically used with term-document matrices. 