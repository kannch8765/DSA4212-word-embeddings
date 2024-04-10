import numpy as np

def tokenize_with_pos(tagged_sentence):
    tokenized_sentence = [f"{word.lower()}_{tag}" for word, tag in tagged_sentence]
    return tokenized_sentence

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_pos(word_with_tag):
    """Extract the POS tag from a word."""
    return word_with_tag.split('_')[-1]


def build_pos_index(flattened_corpus_with_tags, vocab_size):
    pos_index = {}
    for idx, word_with_tag in enumerate(flattened_corpus_with_tags):
        if idx >= vocab_size:
            break  # Skip indices that are outside the vocabulary size
        pos = get_pos(word_with_tag)
        if pos not in pos_index:
            pos_index[pos] = []
        pos_index[pos].append(idx)
    return pos_index


def sample_neg_word_with_same_pos(word_pos, pos_index, vocab_size, neg_samples):
    available_indices = pos_index.get(word_pos, [])
    if available_indices:
        # Ensure we only sample valid indices (within vocab_size)
        valid_indices = [idx for idx in available_indices if idx < vocab_size]
        if len(valid_indices) >= neg_samples:
            return np.random.choice(valid_indices, neg_samples, replace=False)
    # Fallback: sample any word in the vocab if no same POS or not enough words with the same POS
    return np.random.choice(range(vocab_size), neg_samples, replace=False)


def skipgram(vocab_size, emb_dim, flattened_corpus_with_tags, window_size=5, neg_samples=5, epochs=1, learning_rate=0.01):
    W = np.random.rand(vocab_size, emb_dim)  # Target word embeddings
    C = np.random.rand(vocab_size, emb_dim)  # Context word embeddings
    pos_index = build_pos_index(flattened_corpus_with_tags, vocab_size)

    for epoch in range(epochs):
        for i, word_with_tag in enumerate(flattened_corpus_with_tags):
            if i >= vocab_size:  # Ensure the target word index is within bounds
                continue
            word_pos = get_pos(word_with_tag)

            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(flattened_corpus_with_tags))

            for j in range(start, end):
                if j != i and j < vocab_size:  # Check context index is within bounds
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
                            if neg_idx < vocab_size:  # Check negative sample index is within bounds
                                z_neg = np.dot(W[i], C[neg_idx])
                                p_neg = sigmoid(z_neg)
                                grad_neg = p_neg

                                W[i] -= learning_rate * grad_neg * C[neg_idx]
                                C[neg_idx] -= learning_rate * grad_neg * W[i]

        print(f"Epoch {epoch + 1}/{epochs} completed.")

    return W




def train(vocab_size, emb_dim, word_indices, window_size, neg_samples, epochs, learning_rate):
    # Assuming skipgram_simple is implemented as provided earlier
    word_embeddings = skipgram(vocab_size, emb_dim, word_indices, window_size, neg_samples, epochs, learning_rate)
    return word_embeddings

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
    

def find_nearest_neighbors(word_index, embeddings, index_to_word, top_n=5):
    # Get the target word and its POS tag
    target_word_with_tag = index_to_word[word_index]
    target_pos = target_word_with_tag.split('_')[-1]

    similarities = []
    for i, embedding in enumerate(embeddings):
        if i != word_index:  # Skip the target word itself
            # Get the candidate neighbor word and its POS tag
            candidate_word_with_tag = index_to_word[i]
            candidate_pos = candidate_word_with_tag.split('_')[-1]
            
            # Ensure the POS tags match
            if target_pos == candidate_pos:
                sim = cosine_similarity(embeddings[word_index], embedding)  # Use your cosine_similarity function
                similarities.append((candidate_word_with_tag, sim))

    # Sort by similarity in descending order
    nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    return nearest_neighbors
