import torch 
import numpy as np
from collections import defaultdict

def get_random_norm_vec(dim):
    vec = np.random.randn(dim)
    return vec

def create_word2vec(word_to_ix, embedding):
    unknown_word_vec = get_random_norm_vec(embedding.shape[1])
    word2vec = defaultdict(lambda: unknown_word_vec) # unkown words will have a specific vec id
    for word, index in word_to_ix.items():
        word2vec[word] = embedding[index]
    
    return word2vec

embed = torch.load("NewResults/Trump'sTweets/embedding")
word_to_ix = torch.load("NewResults/Trump'sTweets/word_to_ix")

word2vec = create_word2vec(word_to_ix=word_to_ix, embedding=embed)

