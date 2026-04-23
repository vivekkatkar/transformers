import numpy as np

from tokenizer import Tokenizer
from data import corpus 


class Word2Vec:
    def __init__(self, embedding_dim=10, window_size=2, lr=0.01, epochs=5000):
        self.tokenizer = Tokenizer()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = lr
        self.epochs = epochs

    def data_pre(self):
        self.tokens = self.tokenizer.tokens(corpus)  
        
        all_words = [word for sentence in self.tokens for word in sentence]
        
        self.vocab = list(set(all_words))
        self.vocab_size = len(self.vocab)

        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def generate_skip_gram(self):
        pairs = []

        for sentence in self.tokens:
            for i, target in enumerate(sentence):
                for j in range(-self.window_size, self.window_size + 1):
                    if j != 0 and 0 <= i + j < len(sentence):
                        context = sentence[i + j]
                        pairs.append((target, context))

        return pairs

    def one_hot(self, word):
        vec = np.zeros(self.vocab_size)
        vec[self.word2idx[word]] = 1
        return vec

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def train(self):
        self.data_pre()
        pairs = self.generate_skip_gram()

        self.W1 = np.random.randn(self.vocab_size, self.embedding_dim)
        self.W2 = np.random.randn(self.embedding_dim, self.vocab_size)

        for epoch in range(self.epochs):
            loss = 0

            for target, context in pairs:
                x = self.one_hot(target)

                # Forward
                h = np.dot(x, self.W1)
                u = np.dot(h, self.W2)
                y_pred = self.softmax(u)

                # True
                y_true = self.one_hot(context)

                # Loss
                loss -= np.log(y_pred[self.word2idx[context]] + 1e-9)

                # Backprop
                e = y_pred - y_true

                dW2 = np.outer(h, e)
                dW1 = np.outer(x, np.dot(self.W2, e))

                self.W1 -= self.lr * dW1
                self.W2 -= self.lr * dW2

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def get_vector(self, word):
        return self.W1[self.word2idx[word]]

    def most_similar(self, word):
        target_vec = self.get_vector(word)
        similarities = {}

        for w in self.vocab:
            vec = self.get_vector(w)
            sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec))
            similarities[w] = sim

        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[1:5]


model = Word2Vec()
model.train()

print("\nSimilar to 'human':")
print(model.most_similar("human"))