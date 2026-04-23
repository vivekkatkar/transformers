from gensim.models import KeyedVectors

import gensim.downloader as api

# Load Google's pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

# Get vector for a word
word_vector = model['python']
print(f"Vector for 'python': {word_vector}")

# Find most similar words
similar_words = model.most_similar('python', topn=5)
print(f"Words similar to 'python': {similar_words}")

# Calculate similarity between two words
similarity = model.similarity('python', 'programming')
print(f"Similarity between 'python' and 'programming': {similarity}")

# Check if word exists in vocabulary
if 'machine' in model:
    print("'machine' is in vocabulary")

# Perform vector arithmetic
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"King - man + woman = {result}")