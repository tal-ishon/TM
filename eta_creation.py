import torch
from utils import soft_predictions, get_lda_input
from preprocessing import prepare_cleaner_data as prepare_data
import numpy as np


def read_words_file(path):
    # Read the file and create a list of words
    with open(path, 'r') as file:
        eta_words = [line.strip() for line in file]

    return eta_words


def get_words_from_dict(path_words_dict):
    words_dict = torch.load(path_words_dict)
    words = [word for word in words_dict.keys()]
    return words


def create_eta(load_path):
    pred = torch.load(load_path)
    soft_pred = soft_predictions(pred)
    words = get_words_from_dict(path_words_order)
    _, dictionary = get_lda_input(prepare_data())

    ordered_matrix = get_sorted_matrix_according_corpus_order(soft_pred, words, dictionary)

    return ordered_matrix
    torch.save(soft_pred, save_path)
    beta = torch.load(save_path) # validte saving well

    print(beta[:5])


def get_sorted_matrix_according_corpus_order(matrix, words, dictionary):
    """
    making sure words are ordered according dictionary words order

    """
    ordered_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    word_to_id = dictionary.token2id

    for i, word in enumerate(words):
        if word in word_to_id:
            dict_index = word_to_id[word]
            ordered_matrix[dict_index] = matrix[i]

    return ordered_matrix


def save_words_WE(load_path, save_path):
    word_to_ix = torch.load(load_path)
    keys = word_to_ix.keys()
    file_path = save_path

    # Write the string representation to a text file
    with open(file_path, 'w') as file:
        for word in keys:
            file.write(f"{word}\n")


HOME = "Results/GloveCorpusEmbedding"
TOP = 100

path_words_order = f"{HOME}/word_to_ix"
path_predictions = f"{HOME}/{TOP}_WE_predictions"


ordered_mat = create_eta(path_predictions)
torch.save(ordered_mat.T, f"{HOME}/{TOP}_WE_prior")

