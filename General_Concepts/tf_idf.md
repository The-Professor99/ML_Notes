### Notes on Term Frequency - Inverse Document Frequency 

#### Term Frequency - Inverse Document Frequency(TF-IDF)
This basically measures how relevant a word is with respect to the document that contains it, and the entire collection of documents.
##### Components
1. Term Frequency (TF)
The term frequency is simply the number of occurrences of a word in a specific document. If our document is "I love chocolates and chocolates love me", the term frequency of the word love would be 2. This value is often normalized by dividing it by the highest term frequency in the given document, resulting in term frequency values between 0(for words not appearing in the document) and 1(for the most frequent word in the document). The term frequencies are calculated per word and document.
$$TF(t, d) = \text{Number of occurrences of term t in document d} \over \text{Max number of term occurrences in d}$$

2. The Inverse Document Frequency (IDF)
The inverse document frequency is only calculated per word. It indicates how frequently a word appears in the entire corpus. This value is inversed by taking the logarithm of it. Put simply, it is the natural logarithm of the total number of documents divided by the total number of documents that contain this particular word 
$$IDF(t, D) = \ln({\text{Total number of documents} \over \text{Number of documents containing word t}})$$
The Scikit library adds 1 to this value.

##### TF-IDF Calculation:
$$TF-IDF(t, d, D) = TF(t, d)*IDF(t, D)$$

With TF-IDF, we can compare documents, find similar documents, find opposite documents, find similarities in documents.


#### Class Based TF-IDF(c-TF-IDF)
Overview
---
When you apply TF-IDF as usual on a set documents, what you're basically doing is comparing the importance of words between documents, what if, we instead treat all documents in a single category(e.g., a cluster) as a single document and then apply TF-IDF? the result would be a very long document per category and the resulting TF-IDF score would demonstrate the important words in a topic.

c-TF-IDF(Class-based TF-IDF) is an adaptation of TF-IDF for topic modeling or clustering tasks. Instead of evaluating the importance of terms in individual documents, it evaluates the importance of terms across groups of documents(e.g.. clusters or classes)

##### Components
1. Class Based Term Frequency(c-TF):
- Instead of computing term frequency for individual documents, compute it for groups of documents that belong to the same class or cluster.
for a term t in a class C, c-TF is calculated as:
$$c-TF(t, C) = {\text{Number of times term t appears in class C} \over \text{Total number of terms in class C}}$$
2. Inverse Document Frequency(IDF):
- IDF remains the same as in traditional TF-IDF, computed across the entire corpus.

##### c-TF-IDF calculation:
- The c-TF-IDF score for a term t in a class C is calculated as 
$$c-TF-IDF(t, C) = c-TF(t, C) * IDF(t)$$


| ![c-TF-IDF formula](../general_concepts/c-TF-IDF.svg "c-TF-IDF formula") | 
|:--:| 
| *c-TF-IDF formula* |


Because c-TF-IDF takes into account the distribution of terms within classes or clusters, it provides a more context aware representation.

Useful Links:
---

For more detailed information on c-TF-IDF, see [BERTopic documentation](https://maartengr.github.io/BERTopic/api/ctfidf.html#bertopic.vectorizers._ctfidf.ClassTfidfTransformer)