from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import string
from nltk.corpus import stopwords
import math
from collections import Counter

nltk.download("stopwords")
nltk.download("punkt")

STEMMER = PorterStemmer()


def tokenize(text):
    """
    It tokenizes and stems an input text.

    :param text: str, with the input text
    :return: list, of the tokenized and stemmed tokens.
    """
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return [
        STEMMER.stem(word.lower())
        for word in tokens
        if word not in stopwords.words("english")
    ]


def idf_values(vocabulary, documents):
    """
    It computes IDF scores, storing idf values in a dictionary.

    :param vocabulary: list of str, with the unique tokens of the vocabulary.
    :param documents: list of lists of str, with tokenized sentences.
    :return: dict with the idf values for each vocabulary word.
    """
    idf = {}
    num_documents = len(documents)
    for i, term in enumerate(vocabulary):
        idf[term] = math.log(
            num_documents / sum(term in document for document in documents), math.e
        )
    return idf


def vectorize(document, vocabulary, idf):
    """
    It generates the vector for an input document (with normalization).

    :param document: list of str with the tokenized documents.
    :param vocabulary: list of str, with the unique tokens of the vocabulary.
    :param idf: dict with the idf values for each vocabulary word.
    :return: list of floats
    """
    vector = [0] * len(vocabulary)
    counts = Counter(document)
    max_count = counts.most_common(1)[0][1]
    for i, term in enumerate(vocabulary):
        vector[i] = idf[term] * counts[term] / max_count
    return vector


def cosine_similarity(v1, v2):
    """
    It computes cosine similarity.

    :param v1: list of floats, with the vector of a document.
    :param v2: list of floats, with the vector of a document.
    :return: float
    """
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    if sumxy == 0:
        result = 0
    else:
        result = sumxy / math.sqrt(sumxx * sumyy)
    return result


def get_top_k(corpus, query, top_k):
    """
    It computes the search result (get the top_k documents).

    :param query: str
    :param top_k: int
    """
    original_documents, documents, document_vectors, vocabulary, idf = tfidf(corpus)

    q = query.split()
    q = [STEMMER.stem(w) for w in q]
    query_vector = vectorize(q, vocabulary, idf)
    scores = [
        [cosine_similarity(query_vector, document_vectors[d]), d]
        for d in range(len(documents))
    ]
    scores.sort(key=lambda x: -x[0])
    return [original_documents[scores[i][1]] for i in range(min(top_k, len(scores)))]


def tfidf(corpus):
    # Tokenize sentences
    original_documents = [x.strip() for x in corpus]
    documents = [tokenize(d) for d in original_documents]

    # create the vocabulary
    vocabulary = list(set([item for sublist in documents for item in sublist]))
    vocabulary.sort()

    # Compute IDF values and vectors
    idf = idf_values(vocabulary, documents)
    document_vectors = [vectorize(s, vocabulary, idf) for s in documents]

    return original_documents, documents, document_vectors, vocabulary, idf


def compute_recall_at_k(predict, gt, k):
    """
    It computes the recall score at a defined set of retrieved documents.

    :param predict: list of predictions
    :param gt: list of actual data
    :param k: int
    """
    correct_recall = set(predict[:k]).intersection(set(gt))
    return len(correct_recall) / len(gt)


def compute_precision_at_k(predict, gt, k):
    """
    It computes the precision score at a defined set of retrieved documents.

    :param predict: list of predictions
    :param gt: list of actual data
    :param k: int
    """
    correct_predict = set(predict[:k]).intersection(set(gt))
    return len(correct_predict) / k


corpus = [
    "Topic sentences are similar to mini thesis statements.\
        Like a thesis statement, a topic sentence has a specific \
        main point. Whereas the thesis is the main point of the essay",
    "the topic sentence is the main point of the paragraph.\
        Like the thesis statement, a topic sentence has a unifying function. \
        But a thesis statement or topic sentence alone doesn’t guarantee unity.",
    "An essay is unified if all the paragraphs relate to the thesis,\
        whereas a paragraph is unified if all the sentences relate to the topic sentence.",
]


print(get_top_k(corpus, "sentence", 5))
original_documents, documents, document_vectors, vocabulary, idf = tfidf(corpus)
print(document_vectors)
print(vocabulary)
