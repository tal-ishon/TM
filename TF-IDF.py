
from tfidf_utils import prepare_data, get_corpus, get_filtered_corpus, generate_tfidf_matrix, load_file_txt
import networkx as nx
from scase._cluster import SpectralNet
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import SpectralEmbedding
from preprocessing import prepare_cleaner_data

# Create a graph from the TF-IDF matrix
def create_tfidf_graph(tfidf_matrix, feature_names):
    G = nx.Graph()

    for doc_index in range(tfidf_matrix.shape[0]):
        feature_index = tfidf_matrix[doc_index].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc_index, x] for x in feature_index])
        
        for i, (word_index1, score1) in enumerate(tfidf_scores):
            for word_index2, score2 in tfidf_scores:
                if word_index1 != word_index2:
                    word1 = feature_names[word_index1]
                    word2 = feature_names[word_index2]
                    if score1 > score2:
                        weight = score2 / score1
                    else:
                        weight = score1 / score2
                    # weight = (score1 + score2) / 2  # Or any other function of scores you prefer
                    if G.has_edge(word1, word2):
                        G[word1][word2]['weight'] += weight
                    else:
                        G.add_edge(word1, word2, weight=weight)

    return G



# Apply Gaussian Mixture Model to TF-IDF graph
def apply_SN_to_graph(G, words_path):
    # Extract edge weights as features
    edge_features = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]


    words = list(G.nodes())
    n_words = len(words)


    # Create a feature matrix
    X = np.zeros((n_words, n_words), dtype=np.float32)

    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if G.has_edge(word1, word2):
                X[i, j] = G[word1][word2]['weight']

    
    print("FINISH X")
    HOME = "Results/TF-IDF"
    X = torch.tensor(X).float()
    torch.save(X, f"{HOME}/Tal_Affinity_matrix")

    exit(0)
    ###
    
    # save words order to be able to assign each prediction to each word

    # Define the file path
    file_path = words_path

    # Write the string representation to a text file
    with open(file_path, 'w') as file:
        for word in words:
            file.write(f"{word}\n")

    ###


    SE = SpectralEmbedding(n_components=1000)
    eigen_vec = SE.fit_transform(X)
    torch.save(eigen_vec, f"{HOME}/SE_TFIDF")

    exit(0)

    sn = SpectralNet(n_clusters=104)
    sn.fit(X)

    torch.save(sn, SN_model_path)



# Main function to process and visualize TF-IDF graph with GMM clusters
def main():
    data = prepare_cleaner_data()
    word_to_ix, corpus = get_filtered_corpus(data)
    torch.save(word_to_ix, "cleaner_tfidf_word_to_ix")
    tfidf_matrix, feature_names = generate_tfidf_matrix(corpus)
    G = create_tfidf_graph(tfidf_matrix, feature_names)
    apply_SN_to_graph(G, SN_model_path="SN_model_cleaner", words_path="words_order_cleaner.txt")


def calculate_predictions(X_path, sn_model_path, save_predictions_path):
    X = torch.load(X_path)
    sn = torch.load(sn_model_path)

    print("FINISH LOADING")

    predictions = sn.predict(X)
    torch.save(predictions, save_predictions_path)


def calculate_pred(save_predictions_path):
    print("Start Predicting")
    predictions = sn.predict(X)
    torch.save(predictions, save_predictions_path)


# main()
corpus = load_file_txt("Results/clean_corpus")
tfidf_matrix, feature_names = generate_tfidf_matrix(corpus)
G = create_tfidf_graph(tfidf_matrix, feature_names)
apply_SN_to_graph(G, "Results/TF-IDF/Tal_words.txt")