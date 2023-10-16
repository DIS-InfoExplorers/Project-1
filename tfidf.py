from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import string
from nltk.corpus import stopwords
import math
from collections import Counter
import re
from nltk.corpus import wordnet as wn
from multiprocessing import Pool
import pandas as pd
from math import log
import numpy as np

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

# Precompile regex patterns for efficiency
HTML_PATTERN = re.compile("(<.*?>)")
NON_ASCII_DIGITS_PATTERN = re.compile("(\\W|\\d)")
NON_ASCII_CHARS_PATTERN = re.compile(r'[^\x00-\x7F]+')

# Convert stopwords list to set for faster lookup
STOPWORDS_SET = set(stopwords.words("english"))

def preprocess_text(text):
    """Optimized text preprocessing function."""
    
    # Cleaning
    text = HTML_PATTERN.sub("", text)
    text = NON_ASCII_DIGITS_PATTERN.sub(" ", text)
    text = NON_ASCII_CHARS_PATTERN.sub('', text)
    text = "".join([ch for ch in text if ch not in string.punctuation])
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords, and then perform Stemming and Lemmatization
    preprocessed_tokens = [
        STEMMER.stem(LEMMATIZER.lemmatize(word))
        for word in tokens
        if word.lower() not in STOPWORDS_SET
    ]
    
    return preprocessed_tokens


def parallel_preprocess_texts(texts):
    with Pool() as pool:
        preprocessed_batches = pool.map(preprocess_text, texts)
    return preprocessed_batches

    


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


def vectorize_query(query, vocabulary, idf):
    q = query.split()
    q = [STEMMER.stem(w) for w in q]
    query_vector = vectorize(q, vocabulary, idf)
    return query_vector


def get_top_k(corpus, query, k):
    """
    It computes the search result (get the top k documents).

    :param corpus: corpus of documents
    :param query: str
    :param k: int
    """
    original_documents, documents, document_vectors, vocabulary, idf = tfidf(corpus)

    query_vector = vectorize_query(query, vocabulary, idf)
    scores = [
        [cosine_similarity(query_vector, document_vectors[d]), d]
        for d in range(len(documents))
    ]
    scores.sort(key=lambda x: -x[0])
    ans = []
    indices = []
    for i in range(min(k, len(original_documents))):
        ans.append(original_documents[scores[i][1]])
        indices.append(scores[i][1])
    return ans, indices, query_vector


def tfidf(corpus):
    # Tokenize sentences
    original_documents = [x.strip() for x in corpus]
    documents = batch_preprocess_texts(original_documents)

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


# Expand query function for Rocchio's Algorithm
def expand_query(
    relevant_doc_vecs, non_relevant_doc_vecs, query_vector, alpha, beta, gamma
):
    # Note: relevant_doc_vecs and non_relevant_doc_vecs are list of vectors, vectors are also lists in this case.
    # We are using (zip(*list)) to columnwise addition. i.e. [[1,2,3], [4,5,6]] iterate over tuples (1,4),(2,5),(3,6)
    # Check here: https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist
    # You can use numpy if you want to go fancier

    num_rel = len(relevant_doc_vecs)
    num_non_rel = len(non_relevant_doc_vecs)

    # Compute the first term in the Rocchio equation
    norm_query_vector = [alpha * weight for weight in query_vector]

    # Compute the second term in the Rocchio equation
    norm_sum_relevant = [beta * sum(x) / num_rel for x in zip(*relevant_doc_vecs)]

    # Compute the last term in the Rocchio equation
    norm_sum_non_relevant = [
        -gamma * sum(x) / num_non_rel for x in zip(*non_relevant_doc_vecs)
    ]

    # Sum all the terms
    modified_query_vector = [
        sum(x) for x in zip(norm_sum_relevant, norm_sum_non_relevant, norm_query_vector)
    ]

    # Ignore negative elements
    modified_query_vector = [x if x > 0 else 0 for x in modified_query_vector]
    return modified_query_vector

def populate_tfidf_dataframe(documents, vocabulary):
    # Create a list of dictionaries with term frequencies
    list_of_dicts = [Counter(doc) for doc in documents]
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(list_of_dicts).fillna(0)
    
    # Reorder columns according to the vocabulary and fill missing columns with 0
    df = df.reindex(columns=vocabulary, fill_value=0)
    
    return df

def tfidf_with_pandas(corpus):
    # Parallel tokenization and preprocessing
    print("Process docs ...")
    documents = parallel_preprocess_texts(corpus)

    print("Create vocab ...")
    # Create the vocabulary
    vocabulary = list(set(word for doc in documents for word in doc))
    vocabulary.sort()
    
    # Use the helper function to create and populate the DataFrame for term frequencies
    print("Compute tf ...")
    df = populate_tfidf_dataframe(documents, vocabulary)
            
    # Compute IDF values
    print("Compute idf ...")
    doc_count = len(documents)
    idf = df.sum().apply(lambda x: log(doc_count / x))
    
    # Compute TF-IDF values
    print("Compute tf-idf ...")
    tfidf_df = df.apply(lambda x: x / x.sum(), axis=1).multiply(idf)
    
    return original_documents, documents, tfidf_df.values, vocabulary, idf

def vectorize_queries(queries, vocabulary, idf):
    """Vectorize a list of queries."""
    return np.array([vectorize_query(query, vocabulary, idf) for query in queries])

def batch_query(tfidf_matrix_normalized, query_vectors):
    """Process multiple queries and return ranked document indices for each query."""
    # Compute cosine similarities using matrix operations
    similarity_matrix = np.dot(query_vectors, tfidf_matrix_normalized.T)
    
    # Get document indices ranked by relevance for each query
    ranked_doc_indices = np.argsort(-similarity_matrix)
    
    return ranked_doc_indices
    
def preprocess_query(query):
    """Tokenize, stem, and remove stopwords from the query."""
    return preprocess_text(query)  # Assuming preprocess_text() is defined as before

def vectorize_query(query, vocabulary, idf):
    """Convert the query into its TF-IDF vector."""
    query_tf = Counter(preprocess_query(query))
    query_vector = [query_tf.get(term, 0) * idf[term] for term in vocabulary]
    return np.array(query_vector)
    


