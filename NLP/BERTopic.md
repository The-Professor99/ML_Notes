### Important Notes On Using BERTopic

### BERTopic Topic Modeling Process
1. Get the data
2. Preprocess Data(if need be)
3. Convert the documents to numerical data(Embeddings): BERTopic uses sentence transformers by default but other methods can be used as well. New embedding models are released frequently and their performance keeps getting better. To keep track of the best embedding models, visit the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). It is an excellent place for selecting the embedding that works best for you. Eg, if you want the best of the best, then the top 5 models might be the place to look. [Scikit-Learn Embeddings](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html#scikit-learn-embeddings) are relatively lightweight and do not require a GPU. While the representations may be less expensive than many BERT models, the fact that it runs much faster can make it a relevant candidate to consider.
4. Dimensionality Reduction: This is inorder to avoid the curse of dimensionality. BERTopic uses UMAP for this dimensionality reduction process but other methods can be used to. Keep in mind that a too low dimensionality results in a loss of information while a too high dimensionality results in poorer clustering results. You could skip the dimensionality reduction step if you use a clustering algorithm that can handle high dimensionality like a cosine-based k-Means.
5. Clustering: BERTopic uses HDBSCAN(according to the BERTopic author, it works quite well with UMAP since UMAP maintains a lot of local structure even in a lower dimensional space. HDBSCAN does not force data points to cluster as it considers them outliers(one can use KMeans clustering if they want to force data points to cluster)). (KMeans Clustering and PCA perform well together).
6. Topic Creation and Representation: BERTopic uses the [class-based TF-IDF(c-TF-IDF)](../General_Concepts/tf_idf.md) by default for the topic extraction process. The bm25_weighting boolean parameter of the  ClassTfidfTransformer indicates whether a class-based BM-25 weighting measure is used instead of the default method as defined in the c-TF-IDF formula. At smaller datasets, this variant can be more robust to stop words that appear in your data. After having generated our topics with c-TF-IDF, we might want to do some fine-tuning based on the semantic relationship between keywords/keyphrases and the set of documents in each topic. [See how to improve default representation](https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html#controlling-number-of-topics) 
7. Topic Reduction

#### Preprocessing
1. [Removing stop words](https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#removing-stop-words): At times, stop words might end up in our topic representations, this is something we typically want to avoid as they contribute little to the interpretation of the topics. However, removing stop words as a preprocessing step is not advised as the transformer-based embeddings models used in BERTopic needs to use the full context in order to create accurate embeddings. In this practice, we use the `CountVectorizer` to preprocess our documents after having generated embeddings and clustered our documents. Check the link for other ways to work around the problem.

### Embedding Process
1. Pre-compute Embeddings: Typically, we want to iterate fast over different versions of our BERTopic model while trying to optimize it to a specific use case. To speed up this process, we can precompute the embeddings, save them, and pass them to BERTopic so it doesn't need to calculate the embeddings each time.

### Topic Modeling Process
1. [Diversify Topic Representation](https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#diversify-topic-representation): After having calculated our top n words per topic, there might be many words that essentially mean the same thing. We can use `bertopic.representation.MaximalMarginalRelevance` to diversify words in each topic such that we limit the number of duplicate words we find in each topic. The algorithm used(Maximal Marginal Relevance) compares word embeddings with the topic embedding. We do this by specifying a value between 0 and 1, with 0 being not at all diverse and 1 being completely diverse.
2. Reduce Topics: Specifying the `nr_topics` parameter will reduce the initial number of topics to the value specified. This reduction can take a while as each reduction in topics (-1) activates a c-TF-IDF calculation. If this is set to None, no reduction is applied. Use "auto" to automatically reduce topics using HDBSCAN. NOTE: Controlling the number of topics is best done by adjusting min_topic_size first before adjusting this parameter. A higher min_cluster_size will generate fewer topics and a lower min_cluster_size will generate more topics. BERTopic's min_topic_size is what's passed to HDBSCAN's min_cluster_size. We can further reduce the topics by calling the reduce_topics method.
3. Reuce Frequent Words: [See removing stop words above](#). We can also reduce frequent words by setting the `reduce_frequent_words` parameter of ClassTfidfTransformer to True.


Notes:
---
BERTopic uses Transformer models(SentenceTransformer) by default to create embeddings. Since transformer models have a token limit, one might run into some errors when inputting large documents. In that case, you can consider splitting the documents into paragraphs or sentences. A nice way to do so is by using NLTK's sentence splitter like the below:
```
#Example splitting
from nltk.tokenize import sent_tokenize, word_tokenize
sentences = [sent_tokenize(doc) for doc in X]
sentences = [sentence for doc in sentences for sentence in doc]
# After splitting, there is a question of how we'll assign labels to the new sentences
# Let's leave that for that given it's a clustering task and we'll not always have labels.
```

Some Few Random Notes on CountVectorizer
- vectorizer_model=CountVectorizer(stop_words="english") helps quite a bit to improve topic representation.

- ngram_range: The ngram_range parameter allows us to decide how many tokens each entity is in a topic representation. For example, we have words like game and team with a length of 1 in a topic but it would also make sense to have words like hockey league with a length of 2. To allow for these words to be generated, we can set the ngram_range parameter:
```
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
```
Before starting, it should be noted that you can pass the CountVectorizer before and after training your topic model. Passing it before training allows you to minimize the size of the resulting c-TF-IDF matrix.

- min_df: One important parameter to keep in mind is the min_df. This is typically an integer representing how frequent a word must be before being added to our representation. You can imagine that if we have a million documents and a certain word only appears a single time across all of them, then it would be highly unlikely to be representative of a topic. Typically, the c-TF-IDF calculation removes that word from the topic representation but when you have millions of documents, that will also lead to a very large topic-term matrix. To prevent a huge vocabulary, we can set the min_df to only accept words that have a minimum frequency.
When you have millions of documents or error issues, I would advise increasing the value of min_df as long as the topic representations might sense.

- max_features: A parameter similar to min_df is max_features which allows you to select the top n most frequent words to be used in the topic representation. Setting this, for example, to 10_000 creates a topic-term matrix with 10_000 terms. This helps you control the size of the topic-term matrix directly without having to fiddle around with the min_df parameter: