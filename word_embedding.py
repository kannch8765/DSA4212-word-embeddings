import numpy as np
import math
from collections import Counter

#sample sentence: [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
def tokenize_with_pos(tagged_sentence):
    tokenized_sentence = []
    for word, tag in tagged_sentence:
        tokenized_sentence.append(f"{word.lower()}_{tag}")
    return tokenized_sentence



def word2vec_pos(tagged_corpus, window_size=4, pos_weighting=None, pos_weights=None):
    if pos_weights is None:
        pos_weights = {"_VB": 1.5}

    tokenized_corpus = [tokenize_with_pos(sentence) for sentence in tagged_corpus]
    flat_tokenized_corpus = [token for sentence in tokenized_corpus for token in sentence]
    
    vocabulary = sorted(set(flat_tokenized_corpus))
    vocab_index = {token: i for i, token in enumerate(vocabulary)}
    
    co_occurrence_matrix = np.zeros((len(vocabulary), len(vocabulary)))

    for tokens in tokenized_corpus:
        for idx, token in enumerate(tokens):
            token_idx = vocab_index[token]
            start = max(idx - window_size, 0)
            end = min(idx + window_size + 1, len(tokens))

            for ctx_idx in range(start, end):
                if ctx_idx != idx:  # Exclude the target word itself
                    context_token = tokens[ctx_idx]
                    context_token_idx = vocab_index[context_token]
                    weight = 1
                    _, pos = context_token.rsplit('_', 1)
                    weight = pos_weights.get(f"_{pos}", 1)

                    co_occurrence_matrix[token_idx, context_token_idx] += weight
                    
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
    
def compute_ppmi(co_occurrence_matrix):
    # Calculate the probabilities of each word and context
    word_prob = np.sum(co_occurrence_matrix, axis=1) / np.sum(co_occurrence_matrix)
    context_prob = np.sum(co_occurrence_matrix, axis=0) / np.sum(co_occurrence_matrix)
    
    # Calculate the joint probability matrix for words and contexts
    joint_prob_matrix = co_occurrence_matrix / np.sum(co_occurrence_matrix)
    
    # Calculate PPMI
    ppmi_matrix = np.maximum(np.log2(joint_prob_matrix / (word_prob[:, None] * context_prob[None, :])), 0)
    
    # Replace NaN and -Inf values with 0 (resulting from 0/0 or log(0) in calculations)
    ppmi_matrix = np.nan_to_num(ppmi_matrix)
    
    return ppmi_matrix


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

def find_nearest_neighbors(word_vector, ppmi_matrix, vocab_index, pos_tag=None, top_n=5):
    similarities = []

    for other_word, other_idx in vocab_index.items():
        # Check for matching POS tags if pos_tag is specified
        if pos_tag is None or other_word.endswith(f"_{pos_tag}"):
            other_vector = ppmi_matrix[other_idx, :]
            sim = cosine_similarity(word_vector, other_vector)
            similarities.append((other_word, sim))

    nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return nearest_neighbors


