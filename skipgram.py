import numpy as np

def tokenize_with_pos(tagged_sentence):
    tokenized_sentence = [f"{word.lower()}_{tag}" for word, tag in tagged_sentence]
    return tokenized_sentence

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_pos(word_with_tag):
    return word_with_tag.split('_')[-1]


def build_pos_index(flattened_corpus_with_tags, vocab_size):
    pos_index = {}
    for idx, word_with_tag in enumerate(flattened_corpus_with_tags):
        if idx >= vocab_size:
            break  
        pos = get_pos(word_with_tag)
        if pos not in pos_index:
            pos_index[pos] = []
        pos_index[pos].append(idx)
    return pos_index


def sample_neg_word_with_same_pos(word_pos, pos_index, vocab_size, neg_samples):
    available_indices = pos_index.get(word_pos, [])
    if available_indices:
        valid_indices = [idx for idx in available_indices if idx < vocab_size]
        if len(valid_indices) >= neg_samples:
            return np.random.choice(valid_indices, neg_samples, replace=False)
    return np.random.choice(range(vocab_size), neg_samples, replace=False)


def skipgram(vocab_size, emb_dim, flattened_corpus_with_tags, window_size=5, neg_samples=5, epochs=1, learning_rate=0.01):
    W = np.random.rand(vocab_size, emb_dim)  # Target word embeddings
    C = np.random.rand(vocab_size, emb_dim)  # Context word embeddings
    pos_index = build_pos_index(flattened_corpus_with_tags, vocab_size)

    for epoch in range(epochs):
        for i, word_with_tag in enumerate(flattened_corpus_with_tags):
            if i >= vocab_size:  
                continue
            word_pos = get_pos(word_with_tag)

            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(flattened_corpus_with_tags))

            for j in range(start, end):
                if j != i and j < vocab_size: 
                    ctx_word_with_tag = flattened_corpus_with_tags[j]
                    ctx_pos = get_pos(ctx_word_with_tag)

                    if word_pos == ctx_pos:
                        z = np.dot(W[i], C[j])
                        p_pos = sigmoid(z)
                        grad_pos = p_pos - 1

                        W[i] -= learning_rate * grad_pos * C[j]
                        C[j] -= learning_rate * grad_pos * W[i]

                        neg_indices = sample_neg_word_with_same_pos(word_pos, pos_index, vocab_size, neg_samples)
                        for neg_idx in neg_indices:
                            if neg_idx < vocab_size:  
                                z_neg = np.dot(W[i], C[neg_idx])
                                p_neg = sigmoid(z_neg)
                                grad_neg = p_neg

                                W[i] -= learning_rate * grad_neg * C[neg_idx]
                                C[neg_idx] -= learning_rate * grad_neg * W[i]

        print(f"Epoch {epoch + 1}/{epochs} completed.")

    return W




def train(vocab_size, emb_dim, word_indices, window_size, neg_samples, epochs, learning_rate):
    word_embeddings = skipgram(vocab_size, emb_dim, word_indices, window_size, neg_samples, epochs, learning_rate)
    return word_embeddings

def cosine_similarity(v, w):
    dot_product = np.dot(v, w)
    magnitude_v = np.sqrt(np.dot(v, v))
    magnitude_w = np.sqrt(np.dot(w, w))
    if magnitude_v == 0 or magnitude_w == 0:
        return 0
    else:
        return dot_product / (magnitude_v * magnitude_w)
    

def find_nearest_neighbors(word_index, embeddings, index_to_word, top_n=5):
    target_word_with_tag = index_to_word[word_index]
    target_pos = target_word_with_tag.split('_')[-1]

    similarities = []
    for i, embedding in enumerate(embeddings):
        if i != word_index: 
            candidate_word_with_tag = index_to_word[i]
            candidate_pos = candidate_word_with_tag.split('_')[-1]
            if target_pos == candidate_pos:
                sim = cosine_similarity(embeddings[word_index], embedding)  
                similarities.append((candidate_word_with_tag, sim))
                
    nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    return nearest_neighbors
