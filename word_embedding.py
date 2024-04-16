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
                if ctx_idx != idx:  
                    context_token = tokens[ctx_idx]
                    context_token_idx = vocab_index[context_token]
                    weight = 1
                    _, pos = context_token.rsplit('_', 1)
                    weight = pos_weights.get(f"_{pos}", 1)

                    co_occurrence_matrix[token_idx, context_token_idx] += weight
                    
    return co_occurrence_matrix, vocab_index




def cosine_similarity(v, w):
    dot_product = np.dot(v, w)
    magnitude_v = np.sqrt(np.dot(v, v))
    magnitude_w = np.sqrt(np.dot(w, w))

    if magnitude_v == 0 or magnitude_w == 0:
        return 0
    else:
        return dot_product / (magnitude_v * magnitude_w)
    
def compute_ppmi(co_occurrence_matrix):
    word_prob = np.sum(co_occurrence_matrix, axis=1) / np.sum(co_occurrence_matrix)
    context_prob = np.sum(co_occurrence_matrix, axis=0) / np.sum(co_occurrence_matrix)
    
    joint_prob_matrix = co_occurrence_matrix / np.sum(co_occurrence_matrix)
    ppmi_matrix = np.maximum(np.log2(joint_prob_matrix / (word_prob[:, None] * context_prob[None, :])), 0)

    ppmi_matrix = np.nan_to_num(ppmi_matrix)
    
    return ppmi_matrix


def find_nearest_neighbors(word_with_pos, co_occurrence_matrix, vocab_index, top_n=5):
    _, target_pos = word_with_pos.rsplit('_', 1)  
    word_idx = vocab_index[word_with_pos]  
    word_vector = co_occurrence_matrix[word_idx, :]  

    similarities = []
    for other_word, other_idx in vocab_index.items():
        _, other_pos = other_word.rsplit('_', 1)  
        if other_pos == target_pos:  
            other_vector = co_occurrence_matrix[other_idx, :]  
            sim = cosine_similarity(word_vector, other_vector)  
            similarities.append((other_word, sim))

    nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return nearest_neighbors



