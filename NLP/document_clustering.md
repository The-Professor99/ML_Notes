# Document Clustering(aka text clustering)
## Introduction
Document clustering is a method used to group a set of textual documents into clusters. Documents in the same cluster are more similar to each other than to those in other clusters. This technique is widely used in data mining, information retrieval, and pattern recognition. It helps organize and analyze large textual datasets efficiently, making it easier to identify patterns and trends that would otherwise be difficult to discern.

## Importance
In today's data-driven landscape, the vast amount of textual data generated from sources like social media platforms (e.g., Facebook, Twitter) makes data analysis and retrieval challenging. Document clustering streamlines the organization and analysis of large textual datasets, addressing the growing need for efficient and scalable clustering methods as data volumes surge.

## Applications
Document clustering has various applications:

- Web Document Clustering: A web search engine often returns thousands of pages in response to a broad query, making it difficult for users to browse or to identify relevant information, clustering methods can be used to automatically group the retrieved documents into a list of meaningful categories. 
- Automatic Document Organization: Automatically organizes large collections of documents.
- Topic Extraction: Identifies main topics within a set of documents.
- Fast Information Retrieval or Filtering: Enhances the speed and efficiency of retrieving or filtering information.

Applications can be categorized into two types:

Online Applications: Constrained by efficiency problems.
Offline Applications: Typically more robust but may not operate in real-time.

Clustering Algorithms
There are multiple clustering algorithms with different strategies (density-based, distance-based) and approaches (hierarchical, non-hierarchical). The choice of which algorithm to use depends on the application and our understanding of the data distribution.
- Hierarchical Based Algorithm.
- K-Means Algorithm.

Hierarchical algorithms produce more in-depth information for detailed analyses, while algorithms based around variants of the k-means algorithm are more efficient and provide sufficient information for most purposes.

Document Clustering Steps
1. Preprocessing
	- Eliminate handles and URLs.
	- Removing Stop Words and Punctuation: Eliminating common but uninformative words.
	- Stemming and Lemmatization: Perform stemming to convert words to their base form (e.g., dancer, dancing, danced become "danc").
	- Convert all words to lowercase.
	- Tokenization: Tokenize the string into words.
2. Feature Extraction
Convert textual content into numerical values using vectorizers like TfidfVectorizer, CountVectorizer, and HashingVectorizer. This step shapes and determines the final data distribution.
3. Dimensionality Reduction
Reduce the number of features to simplify the model and improve performance.
4. Clustering Algorithm
Apply the chosen clustering algorithm (e.g., K-means, hierarchical) to group documents.
5. Evaluation and Visualization
Assess cluster quality and present results.

Side Notes:
Distributional Hypothesis
The distributional hypothesis states that words with similar meanings appear in similar contexts. For example, "The box is on the shelf" and "The box is under the shelf" have interchangeable prepositions that still make sense. This hypothesis is used in creating word embeddings, which map words onto an n-dimensional vector space where words with similar contexts appear in the same area.


Useful Links: \
https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04