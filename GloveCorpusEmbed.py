import torch
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from EM_utils import prepare_data, get_filtered_vocabulary, load_file_txt, save_file_txt
import sys
# from preprocessing import prepare_cleaner_datd

"""
This file create the word embedding vectors according to words in a given corpus.
Saves the embeddings and the word_to_ix.

"""

class myEmbeddings:
    def __init__(self,
        embeddings: torch.FloatTensor,
        word_to_ix: dict,
        name: str,
        vec_size: int,
        num_of_vec: int):
    
        self.embedding = embeddings
        self.word_to_ix = word_to_ix
        self.name = name
        self.vec_size = vec_size
        self.num_of_vec = num_of_vec

    def get_word2vec(self):
        unknown_word_vec = get_random_norm_vec(self.vec_size)
        word2vec = defaultdict(lambda: unknown_word_vec) # unkown words will have a specific vec id
        for word, index in self.word_to_ix.items():
            word2vec[word] = self.embeddings[index]
    
        return word2vec



def load_glove_embeddings(file_path, embedding_dim):
    """
    No random vectors here.

    """
    word_to_index = defaultdict(lambda: 0)  # unknown word is index 0
    embeddings = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            
            word_to_index[word] = i
            embeddings.append(vector)
    

    return word_to_index, torch.FloatTensor(embeddings)


def get_index_and_embedding(path, dim):
    # Load d-dimensional GloVe embeddings
    word_to_index, embeddings = load_glove_embeddings(path, dim)
    return word_to_index, embeddings


def get_random_norm_vec(dim):
    vec = np.random.randn(dim)
    return vec


def get_word2vec(word_to_index, embeddings):
    unknown_word_vec = get_random_norm_vec(embeddings.shape[1])
    word2vec = defaultdict(lambda: unknown_word_vec) # unkown words will have a specific vec id
    for word, index in word_to_index.items():
        word2vec[word] = embeddings[index]
    
    return word2vec


def get_embed_letters(word, word_to_ix, embedding):
    """
    Unknown words will be the sum of their letters embeddings
    """
    letters = [char for char in word]
    embed = torch.zeros(embedding.shape[1])

    for char in letters:
        ix = word_to_ix[char]
        embed += embedding[ix]

    return embed


def get_embed_bigram(word, word_to_ix, embedding):

    bigrams = [word[i:i+2] for i in range(len(word) - 1)]
    embed = torch.zeros(embedding.shape[1])

    for pair in bigrams:
        ix = word_to_ix[pair]
        embed += embedding[ix]

    return embed


def get_final_embedding(word_to_ix, embed):
    """
    Take the origin Embedding (Glove e.g.) and keep only word embeddings in the given corpus.

    Params:
    word_to_ix: origin Embed dictionary
    embed: origin Embed

    """
    vocabulary = load_file_txt("Results/CorpusFilter/clean_vocab")

    print(f"Number of words: {len(vocabulary)}")

    # Init the dictionary of the corpus and the embedding of the corpus.
    corpus_to_ix = dict()
    corpus_embed = dict()

    index = 0
    words = []
    # insert word from corpus to pre_trained embedding
    for word in vocabulary:
        # Get the index of the word in origin embedding - use it to fill the embedding of corpus.
        ix = word_to_ix[word]

        if not ix: # remove words that are in Corpus but not in Glove
            continue

        # Fill the dictionary and the embedding of the corpus.
        corpus_to_ix[word] = index
        corpus_embed[index] = embed[ix]
        index += 1
        words.append(word)

    embed_list = list(corpus_embed.values())
    save_file_txt("Results/CorpusFilter/filter_words", words)
    return corpus_to_ix, torch.stack(embed_list)


def save(obj, name):
    torch.save(obj, name)


def main():
    args = sys.argv

    if len(args) == 1:
        # If new embedding
        embed_dim = 100

        word_to_ix, embed = get_index_and_embedding(path, embed_dim)
        corpus_to_ix, corpus_embed = get_final_embedding(word_to_ix, embed)

        print(f"Length of corpus embedding: {corpus_embed.shape[0]}")
        save(corpus_embed, "Results/CorpusFilter/embedding")
        save(corpus_to_ix, "Results/CorpusFilter/word_to_ix")

        return 0
    
    # load exist embedding
    
    


path = 'glove.6B/glove.6B.100d.txt'

main()

###
# validation
# word_to_ix = torch.load("Results/BBCGlove/word_to_ix")
# Embedding = torch.load("Results/BBCGlove/embedding")
# word2vec = get_word2vec(word_to_index=word_to_ix, embeddings=Embedding)
# passed
###
