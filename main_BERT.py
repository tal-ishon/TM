# Preprocess data - remove puncuation words, too long/short words etc.

# Create embedding and word_to_ix from Glove according to words in corpus - keep only the intersection

# Get predictions from the embeddings - according to number of topics

# Create prior from predictions

# save the words in topics in a csv file

import string
import preprocessing as pp
import numpy as np
from collections import defaultdict
import torch
from itertools import chain
from spectralnet import SpectralNet
from spectralnet._utils import get_affinity_matrix
from scase import ScaSE, SpectralNet as SN
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

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


    def init_BERT(self, model_id="bert-base-uncased"):
        # Load the pre-trained BERT model and tokenizer from Hugging Face
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        model = BertModel.from_pretrained(model_id)
        model.eval()  # Set the model to evaluation mode
        self.model = model
    

    def get_word_embeddings(self, sentences):
        word_embeddings = {}

        for sentence in sentences:
            # Tokenize and encode the sentence
            tokens = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
                hidden_states = outputs.last_hidden_state[0]  # Shape: [seq_len, hidden_dim]

                # Convert token IDs to words
                token_words = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
                current_word = None
                current_embedding = []

                for i, token in enumerate(token_words):
                    word = token.replace("##", "").lower()
                    if not token.startswith("##"):
                        # Append the accumulated embeddings of the previous word
                        if current_word and current_embedding:
                            if current_word not in word_embeddings:
                                word_embeddings[current_word] = []
                            word_embeddings[current_word].append(np.mean(current_embedding, axis=0))
                        # Start a new word
                        current_word = word
                        current_embedding = []
                    current_embedding.append(hidden_states[i].numpy())

                # Handle the last word in the sentence
                if current_word and current_embedding:
                    if current_word not in word_embeddings:
                        word_embeddings[current_word] = []
                    word_embeddings[current_word].append(np.mean(current_embedding, axis=0))

        return word_embeddings

    
    
    def save_embed_word2ix(self, aggregated_word_embeddings):
        # Step 1: Create word_to_ix and embeddings_list
        word_to_ix = {}
        embeddings_list = []

        for i, (word, embedding) in enumerate(aggregated_word_embeddings.items()):
            word_to_ix[word] = i  # Map word to index
            embeddings_list.append(embedding)  # Collect the embedding

        # Step 2: Check for consistent embedding shapes
        embedding_dim = len(embeddings_list[0])
        assert all(len(embedding) == embedding_dim for embedding in embeddings_list), \
            "Inconsistent embedding dimensions detected!"

        # Step 3: Convert embeddings to PyTorch tensor
        self.embedding = torch.FloatTensor(np.array(embeddings_list))
        self.word_to_ix = word_to_ix

        # Optional: Save to file for persistence
        torch.save({"embeddings": self.embedding, "word_to_ix": self.word_to_ix}, "embeddings.pt")

        print(f"Embeddings saved with {len(word_to_ix)} words and embedding size {embedding_dim}.")
        return self.embedding, self.word_to_ix



    def aggregate_embeddings(self, word_embeddings):
        aggregated_embeddings = {}

        for word, embeddings in word_embeddings.items():
            embeddings = np.array(embeddings)
            if len(embeddings) < 2:
                # If fewer than 2 embeddings, use the average directly
                aggregated_embedding = np.mean(embeddings, axis=0)
            else:
                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
                kmeans.fit(embeddings)
                cluster_centers = kmeans.cluster_centers_

                # Average the cluster centers
                aggregated_embedding = np.mean(cluster_centers, axis=0)

            aggregated_embeddings[word] = aggregated_embedding

        return aggregated_embeddings




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

        words_corpus = list(chain(*sentences))
        # pp.save_file_txt("20NewsGroupWords", words)
        _, corpus = pp.get_filtered_corpus(sentences, words_corpus)
        corpus_input = [doc.split() for doc in corpus]
        vocabFilter = pp.get_filtered_vocabulary(corpus_input)
        self.vocabulary = list(vocabFilter)
        self.corpus = corpus
        word_embeddings = self.get_word_embeddings(self.corpus)
        aggregated_word_embeddings = self.aggregate_embeddings(word_embeddings)
        embeddings_df = pd.DataFrame.from_dict(aggregated_word_embeddings, orient='index')

        # Save the DataFrame to a CSV file
        embeddings_df.to_csv('word_embeddings_BERT.csv')
        self.save_embed_word2ix(aggregated_word_embeddings)



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
                embed = get_update_embed(eigenvec, eigval)
                np.save("embed.npy", embed)
            else:
                embed = np.load("embed.npy")
            model = GMM(self.n_predictions, 
                    covariance_type='full',
                    n_init=10,
                    max_iter=100,
                    random_state=42)
            model.fit(embed)
            return model, embed
            # model = SN(self.n_predictions, spectral_lr=0.001)
            # model.fit(self.X)
        elif self.mode == "PCA":
            pca = PCA(n_components=20)
            X_transformed = pca.fit_transform(self.X)
            model = GMM(self.n_predictions, n_init=1)
            model.fit(X_transformed)
            return model, X_transformed
        elif self.mode == "UMAP":
            import umap
            model = umap.UMAP(n_components=20)
            X_transformed = model.fit_transform(self.X)
            gmm = GMM(self.n_predictions, n_init=1)
            gmm.fit(X_transformed)
            return gmm, X_transformed
        elif self.mode == "RW":
            from scipy.linalg import eigh
            A = np.array(get_affinity_matrix(X=self.X, n_neighbors=10, device="cpu"))
            degree_matrix = np.diag(A.sum(axis=1))
            D_inv = np.linalg.inv(degree_matrix)  # Inverse of the degree matrix
            I = np.eye(A.shape[0])  # Identity matrix
            L_rw = I - D_inv @ A  # Random walk Laplacian
            eigenvalues, eigenvectors = eigh(L_rw)
            embed = get_update_embed(eigenvectors, eigenvalues)
            model = GMM(self.n_predictions, n_init=1)
            model.fit(embed)
            return model, embed           
        else:
            # pca = PCA(n_components=75)  # You can adjust this
            # reduced_embeddings = pca.fit_transform(self.X)
            model = GMM(
                n_components=self.n_predictions,  # adjust based on your needs
                covariance_type='full',
                n_init=10,
                max_iter=100,
                random_state=42
                )
            model.fit(self.X)

        return model, self.X

    def __prediction(self, model):
        """
        This function predict according to models mode.
        """
        if self.mode == "SN":
            pred = model.predict(self.X)
        elif self.mode == "ScaSE":
            model, embed = model[0], model[1] 
            pred = model.predict_proba(embed)
        elif self.mode == "PCA":
            model, embed = model[0], model[1] 
            pred = model.predict_proba(embed)
        elif self.mode == "UMAP":
            model, embed = model[0], model[1] 
            pred = model.predict_proba(embed)
        elif self.mode == "RW":
            model, embed = model[0], model[1] 
            pred = model.predict_proba(embed)
        else:
            model, embed = model[0], model[1] 
            pred = model.predict_proba(embed)
        
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
HOME_DIR = "BERTNewResults"
DATASET = "Trump'sTweets"
DATASET_PATH = "Trump'sTweets"
FILE_TYPE = "csv"

MODE = "GMM"
DATA_PATH = f"{DATASET_PATH}.{FILE_TYPE}"
TOPICS = 200

###################
# RUN WITH SCRIPT #
###################

# import argparse

# # Initialize the argument parser
# parser = argparse.ArgumentParser(description="Run a model with the specified parameters.")
# parser.add_argument("--mode", type=str, required=True, help="The type of DR to run")
# parser.add_argument("--dataset", type=str, required=True, help="The dataset to use")
# parser.add_argument("--topics", type=int, required=True, help="The number of topics")

# global MODE, TOPIC_NUM
# # Parse the arguments
# args = parser.parse_args()
# MODE = args.mode
# DATASET = args.dataset
# DATASET_PATH = DATASET
# TOPICS = args.topics

# if DATASET == "BBC" or DATASET == "Trump'sTweets":
#     FILE_TYPE = "csv"
# else:
#     FILE_TYPE = "json"

# DATA_PATH = f"{DATASET_PATH}.{FILE_TYPE}"

#######################

if is_first:
    # Create embedding according to data
    pprocessor = Preprocessor(DATA_PATH)
    pprocessor.init_BERT()
    pprocessor.process_data(FILE_TYPE)
    torch.save(pprocessor.embedding, f"{HOME_DIR}/{DATASET}/embedding")
    torch.save(pprocessor.word_to_ix, f"{HOME_DIR}/{DATASET}/word_to_ix")
else: 
    pprocessor = Preprocessor(DATA_PATH, torch.load(f"{HOME_DIR}/{DATASET}/embedding"))

# Calculate topics-words distribution (prior)
has_embed = False
predictor = Predictor(mode=MODE, X=pprocessor.embedding, n_predictions=TOPICS)
predictor.predict()
predictor.calculte_prior()
predictor.save_predictions(f"{HOME_DIR}/{DATASET}/pred_{MODE}")
predictor.save_prior_to_file(f"{HOME_DIR}/{DATASET}/prior_{MODE}")
