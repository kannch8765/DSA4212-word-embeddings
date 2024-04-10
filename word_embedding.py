import numpy as np
import math
from collections import Counter

def tokenize_with_pos(tagged_corpus):
    # Tokenize the corpus and include POS tags
    # Correctly unpack each (word, pos) tuple in the sentence
    tokenized_corpus = [f"{word.lower()}_{pos}" for sentence in tagged_corpus for word, pos in sentence]
    return tokenized_corpus




def word2vec_pos(tagged_corpus, window_size=4):
    # Tokenize with POS tags
    tokenized_corpus = [tokenize_with_pos(sentence) for sentence in tagged_corpus]
    
    # Flatten the list of token lists to create a single list of tokens
    flat_tokenized_corpus = [token for sentence in tokenized_corpus for token in sentence]
    
    # Now you can create a set from the flat list of tokens
    vocabulary = sorted(set(flat_tokenized_corpus))
    vocab_index = {token: i for i, token in enumerate(vocabulary)}
    
    # Initialize the word-word co-occurrence matrix
    co_occurrence_matrix = np.zeros((len(vocabulary), len(vocabulary)))
    
    # Populate the matrix based on co-occurrences within the context window
    for tokens in tokenized_corpus:
        for idx, token in enumerate(tokens):
            token_idx = vocab_index[token]
            start = max(idx - window_size, 0)
            end = min(idx + window_size + 1, len(tokens))
            
            for ctx_idx in range(start, end):
                if ctx_idx != idx:  # Exclude the target word itself
                    context_token = tokens[ctx_idx]
                    context_token_idx = vocab_index[context_token]
                    co_occurrence_matrix[token_idx, context_token_idx] += 1
                    
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

def find_nearest_neighbors(word, co_occurrence_matrix, vocab_index, top_n=5):
    word_idx = vocab_index.get(f"{word.lower()}_NN", None)  # Example: looking for nouns
    if word_idx is None:
        return []

    similarities = []
    word_vector = co_occurrence_matrix[word_idx, :]

    for other_word, other_idx in vocab_index.items():
        if other_word != word:
            other_vector = co_occurrence_matrix[other_idx, :]
            sim = cosine_similarity(word_vector, other_vector)
            similarities.append((other_word, sim))

    # Sort by similarity
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
