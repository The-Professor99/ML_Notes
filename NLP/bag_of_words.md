### Notes on Bag of Words
The bag of words model is a model of text which uses a representation of text that is based on an unordered collection (or "bag") of words. 

Example Implementation
---
Consider two simple text documents:
1. John likes to watch movies. Mary likes movies too.
2. Mary also likes to watch football games.

Based on these two text documents, a list is constructed as follows for each document:
"John", "likes", "to", "watch", "movies", "Mary", "likes", "movies", "too"
"Mary", "also", "likes", "to", "watch", "football", "games"

Representing each bag-of-words as a JSON object:
BoW1 = {"John": 1, "likes": 2, "to": 1, "watch": 1, "movies": 2, "Mary": 1, "too": 1}
BoW2 = {"Mary": 1, "also": 1, "likes": 1, "to": 1, "watch": 1, "football": 1, "games": 1}

Each key is the word and each value is the number of occurrences of that word in the given text document.
The order of elements is free, so, for example, {"too": 1, "Mary": 1, "movies": 2, "John": 1, "watch": 1, "likes": 2, "to": 1} is also equivalent to BoW1.
The "union" of two documents in the bag of words representation is, formally, the disjoint union, summing the multiplicities of each element.

Implementations of the bag-of-words model might involve using frequencies of words in a document to represent its contents. The frequencies can be "normalized" by the inverse of document frequency, or tf-idf. Additionally, for the specific purpose of classification, supervised alternatives have been developed to account for the class label of a document. Also, binary(presence/absence or 1/0) weighting is used in place of frequencies for some problems.
