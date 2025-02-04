# Preprocess data - remove puncuation words, too long/short words etc.

# Create embedding and word_to_ix from Glove according to words in corpus - keep only the intersection

# Get predictions from the embeddings - according to number of topics

# Create prior from predictions

# save the words in topics in a csv file

import string
import json
import preprocessing as pp
import numpy as np
from collections import defaultdict
import torch
from itertools import chain
from spectralnet import SpectralNet
from scase import ScaSE, SpectralNet as SN
from sklearn.mixture import GaussianMixture as GMM
from sklearn.manifold import SpectralEmbedding as SE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from pydiffmap import diffusion_map as dm
from helper_functions import diffusion_maps as dm


np.random.seed(42)

def get_random_norm_vec(dim):
    vec = np.random.randn(dim)
    return vec


def get_intersection(list1, list2):
    return list(set(list1) & set(list2))


def get_norm_vec(embeddings):
    # Compute the L2 norm for each word vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Normalize each word vector by dividing by its norm
    normalized_embeddings = embeddings / norms
    return normalized_embeddings


def get_update_embed(vec, val):
    eigenvectors = vec.T

    norm_val = 1 - val
    norm_val = norm_val[norm_val > 0]
    eigenvalues = norm_val[:, np.newaxis]

    eigenvectors = eigenvectors[:eigenvalues.shape[0]]
    return (eigenvectors * eigenvalues).T

class Preprocessor:
    def __init__(self, corpus_path: string, embedding: torch.tensor = None):
        self.corpus_path = corpus_path
        self.embedding = embedding
    
    def process_data(self, data_type):
        """
        This function should load corpus and get the cleaned corpus and vocabulary out of it.
        Update embedding and word_to_ix - The embedding should contain only the words that are in the 
        processor's vocabulary.
        """
        if data_type == "csv":
            sentences, labels = pp.prepare_cleaner_csv_data(self.corpus_path)
        elif data_type == "json":
            sentences = pp.prepare_cleaner_data(self.corpus_path)
        elif data_type == "txt":
            sentences = pp.prepare_cleaner_txt_data(self.corpus_path)
        else:
            print("Can't process this data type!")
            return

        words_embed = self.word_to_ix.keys()  # words in glove embedding
        words_corpus = list(chain(*sentences))
        words = get_intersection(words_embed, words_corpus)
        # pp.save_file_txt("20NewsGroupWords", words)
        _, corpus = pp.get_filtered_corpus(sentences, words, word_embed_type="glove")
        corpus_input = [doc.split() for doc in corpus]
        vocabFilter = pp.get_filtered_vocabulary(corpus_input)
        self.vocabulary = list(vocabFilter)
        self.corpus = corpus
        self.__update_embed_dict(self.vocabulary)


    def get_corpus(self):
        return self.corpus
    

    def save_corpus(self):
        with open(f"{DATASET}_corpus.json", "w") as f:
            json.dump(self.corpus, f)


    def generate_embedding_and_dictionaty(self, embed_path):
        """
        This function load the Embedding in the given path.
        Create also the word_to_ix of these embedding vectors.
        """
        word_to_index = defaultdict(lambda: 0)  # unknown word is index 0
        embeddings = []
        
        with open(embed_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                
                word_to_index[word] = i
                embeddings.append(vector)
        

        self.word_to_ix = word_to_index
        self.embedding = torch.FloatTensor(np.array(embeddings))


    def create_word2vec(self):
        unknown_word_vec = get_random_norm_vec(self.embedding.shape[1])
        word2vec = defaultdict(lambda: unknown_word_vec) # unkown words will have a specific vec id
        for word, index in self.word_to_ix.items():
            word2vec[word] = self.embedding[index]
        
        self.word2vec = word2vec


    def save_obj_in_file(self, obj, type, file_path):
        if type == "txt":
            pp.save_file_txt(file_path, obj)
        elif type == "csv":
            pass
        elif type == "torch":
            torch.save(obj, file_path)
        else:
            print("CAN'T SAVE OBJ FORMAT")


    def __update_embed_dict(self, corpus_words):
         # Init the dictionary of the corpus and the embedding of the corpus.
        corpus_to_ix = dict()
        corpus_embed = dict()

        index = 0
        words = []
        # insert word from corpus to pre_trained embedding
        for word in corpus_words:
            # Get the index of the word in origin embedding - use it to fill the embedding of corpus.
            ix = self.word_to_ix[word]

            if not ix: # remove words that are in Corpus but not in Glove
                continue

            # Fill the dictionary and the embedding of the corpus.
            corpus_to_ix[word] = index
            corpus_embed[index] = self.embedding[ix]
            index += 1
            words.append(word)

        embed_list = list(corpus_embed.values()) 

        self.embedding = torch.stack(embed_list)
        self.word_to_ix = corpus_to_ix


class Predictor:
    def __init__(self, mode, X, n_predictions) -> None:
        """
        mode: type of prediction. could be GMM or SpectralNet GMM
        n_predictions: the number of predictions. hyperparam for n_component/n_clusters.

        """
        self.mode = mode
        self.X = X
        self.n_predictions = n_predictions


    def get_X(self):
        return self.X
    

    def __fit(self):
        """
        This function fits the data to the model.
        Fit according to model mode.
        """
        if self.mode == "SN":
            model = SpectralNet(self.n_predictions, spectral_epochs=30)
            model.fit(self.X)
        elif self.mode == "ScaSE":
            if not has_embed: 
                model = ScaSE(10, spectral_lr=0.001, spectral_max_epochs=50)
                eigenvec = model.fit_transform(self.X)
                eigval = model.get_eigenvalues()
                # embed = get_update_embed(eigenvec, eigval)
                # np.save("embed.npy", embed)
                embed = eigenvec
            else:
                embed = np.load("embed.npy")
            model = GMM(n_components=self.n_predictions, 
                        random_state=42,
                        n_init=5, 
                        verbose=1)
            model.fit(embed)
            return model, embed
            # model = SN(self.n_predictions, spectral_lr=0.001)
            # model.fit(self.X)
        elif self.mode == "DM":
            diffusion_coordinats = dm(data=self.X, n_components=100, k=20, epsilon=35)
            self.X = diffusion_coordinats
            model = GMM(
                n_components=self.n_predictions,  # adjust based on your needs
                random_state=42,
                n_init=5, 
                verbose=1
            )
            model.fit(self.X)       
        else:
            pca = PCA(n_components=50)  # You can adjust this
            self.X = pca.fit_transform(self.X)
            model = GMM(
                n_components=self.n_predictions,  # adjust based on your needs
                random_state=42,
                n_init=5, 
                verbose=1
            )
            model.fit(self.X)

        return model

    def __prediction(self, model):
        """
        This function predict according to models mode.
        """ 
        pred = model.predict_proba(self.X)
        
        return pred

    def __soft_predictions(self):
        # Define epsilon
        epsilon = 0.01
        soft_predictions = self.predictions + epsilon  # Add epsilon to each element
        soft_predictions /= soft_predictions.sum(axis=1, keepdims=True) # Normalize each row so that the sum is 1

        return soft_predictions


    def predict(self):
        self.predictions = self.__prediction(self.__fit())

    def save_predictions(self, file_name):
        torch.save(self.predictions, file_name)

    def calculte_prior(self):
        self.prior = self.__soft_predictions()


    def save_prior_to_file(self, file_path):
        torch.save(self.prior.T, file_path)

is_first = False
HOME_DIR = "NewResults"
DATASET = "20NewsGroup"
DATASET_PATH = "20NewsGroup"
FILE_TYPE = "json"

MODES = ["DM"]
TOPICS_LIST = [100]
DATA_PATH = f"{DATASET_PATH}.{FILE_TYPE}"

for MODE in MODES:
    for TOPICS in TOPICS_LIST:

        if is_first:
            # Create embedding according to data
            pprocessor = Preprocessor(DATA_PATH)
            pprocessor.generate_embedding_and_dictionaty('glove.6B/glove.6B.100d.txt')
            pprocessor.process_data(FILE_TYPE)
            torch.save(pprocessor.embedding, f"{HOME_DIR}/{DATASET}/embedding")
            torch.save(pprocessor.word_to_ix, f"{HOME_DIR}/{DATASET}/word_to_ix")
            is_first = False
        else: 
            pprocessor = Preprocessor(DATA_PATH, torch.load(f"NewResults/{DATASET}/embedding"))

        # Calculate topics-words distribution (prior)
        has_embed = False
        predictor = Predictor(mode=MODE, X=pprocessor.embedding, n_predictions=TOPICS)
        predictor.predict()
        predictor.calculte_prior()
        predictor.save_predictions(f"NewResults/{DATASET}/{TOPICS}_CHANGEpred_{MODE}")
        predictor.save_prior_to_file(f"NewResults/{DATASET}/{TOPICS}_CHANGEprior_{MODE}")
