import numpy as np

def tokenize_with_pos(tagged_sentence):
    tokenized_sentence = []
    for word, tag in tagged_sentence:
        tokenized_sentence.append(f"{word.lower()}_{tag}")
    return tokenized_sentence

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def skipgram(vocab_size, emb_dim, corpus, window_size=2, neg_samples=5, epochs=1, learning_rate=0.01):
    # Randomly initialize W (target) and C (context) matrices
    W = np.random.rand(vocab_size, emb_dim)
    C = np.random.rand(vocab_size, emb_dim)

    for epoch in range(epochs):
        for word_idx, word in enumerate(corpus):
            # Define the context window
            start = max(word_idx - window_size, 0)
            end = min(word_idx + window_size + 1, len(corpus))

            # Iterate through the context window
            for ctx_idx in range(start, end):
                if ctx_idx != word_idx:  # Skip the target word itself
                    ctx_word = corpus[ctx_idx]

                    # Positive sample
                    z = np.dot(W[word], C[ctx_word])
                    p_pos = sigmoid(z)
                    loss_pos = -np.log(p_pos)
                    grad_pos = p_pos - 1

                    # Update W and C for positive sample
                    W[word] -= learning_rate * grad_pos * C[ctx_word]
                    C[ctx_word] -= learning_rate * grad_pos * W[word]

                    # Negative sampling
                    for _ in range(neg_samples):
                        neg_word = np.random.randint(vocab_size)
                        z_neg = np.dot(W[word], C[neg_word])
                        p_neg = sigmoid(z_neg)
                        loss_neg = -np.log(1 - p_neg)
                        grad_neg = p_neg

                        # Update W and C for negative sample
                        W[word] -= learning_rate * grad_neg * C[neg_word]
                        C[neg_word] -= learning_rate * grad_neg * W[word]

        print(f"Epoch {epoch+1}/{epochs} completed.")

    return W  # Return the final target word embeddings

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
    target_embedding = embeddings[word_index]  # Get the target word's embedding

    similarities = []
    for i, embedding in enumerate(embeddings):
        if i != word_index:  # Skip the target word itself
            sim = cosine_similarity(target_embedding, embedding)  # Use your cosine_similarity function
            similarities.append((i, sim))

    # Sort by similarity in descending order
    nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    # Convert indices to words
    nearest_neighbors = [(index_to_word[idx], sim) for idx, sim in nearest_neighbors]
    
    return nearest_neighbors