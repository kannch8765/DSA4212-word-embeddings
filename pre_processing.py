import numpy as np
import math
from collections import Counter

def tokenize(corpus):
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    return tokenized_corpus

def word_to_vector_word_word(corpus, window_size=4):
    # Tokenize and build vocabulary
    tokenized_corpus = tokenize(corpus)
    vocabulary = sorted(set(word for doc in tokenized_corpus for word in doc))
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    
    # Initialize the word-word co-occurrence matrix
    co_occurrence_matrix = np.zeros((len(vocabulary), len(vocabulary)))
    
    # Populate the matrix based on co-occurrences within the context window
    for doc in tokenized_corpus:
        for idx, word in enumerate(doc):
            word_idx = vocab_index[word]
            # Determine the context window boundaries (careful at the edges of the document)
            start = max(idx - window_size, 0)
            end = min(idx + window_size + 1, len(doc))
            
            # Iterate through context window
            for ctx_idx in range(start, end):
                if ctx_idx != idx:  # Exclude the target word itself
                    context_word = doc[ctx_idx]
                    context_word_idx = vocab_index[context_word]
                    co_occurrence_matrix[word_idx, context_word_idx] += 1
    
    return co_occurrence_matrix, vocab_index

def cosine_similarity(v, w):
    # Compute the dot product between v and w
    dot_product = np.dot(v, w)
    
    # Compute the magnitude (length) of v and w
    magnitude_v = np.sqrt(np.dot(v, v))
    magnitude_w = np.sqrt(np.dot(w, w))
    
    # Compute the cosine similarity
    if magnitude_v == 0 or magnitude_w == 0:
        # Return 0 if one of the vectors is a zero vector
        return 0
    else:
        return dot_product / (magnitude_v * magnitude_w)


def compute_tf_idf(corpus):
    # Tokenize the corpus and build the vocabulary
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    vocabulary = sorted(set(word for doc in tokenized_corpus for word in doc))
    vocab_index = {word: i for i, word in enumerate(vocabulary)}

    # Initialize matrices and document frequency dictionary
    tf = np.zeros((len(corpus), len(vocabulary)))
    df = Counter()
    idf = np.zeros(len(vocabulary))
    
    # Compute term frequency (TF) and document frequency (DF)
    for doc_idx, doc in enumerate(tokenized_corpus):
        doc_counter = Counter(doc)
        for word, count in doc_counter.items():
            word_idx = vocab_index[word]
            tf[doc_idx, word_idx] = 1 + math.log10(count)
            df[word] += 1
    
    # Compute inverse document frequency (IDF)
    for word, word_idx in vocab_index.items():
        idf[word_idx] = math.log10(len(corpus) / df[word])
    
    # Compute TF-IDF
    tf_idf = tf * idf
    
    return tf_idf, vocab_index