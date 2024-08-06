import torch
from utils import soft_predictions, get_lda_input
from preprocessing import prepare_cleaner_data as prepare_data
import numpy as np


def read_words_file(path):
    # Read the file and create a list of words
    with open(path, 'r') as file:
        eta_words = [line.strip() for line in file]

    return eta_words


def get_words_dict(path_words_dict):
    words_dict = torch.load(path_words_dict)
    return words_dict


def create_prior(load_path):
    pred = torch.load(load_path)
    soft_pred = soft_predictions(pred)
    word_to_ix = get_words_dict(path_words_order)

    return soft_pred



# def get_sorted_matrix_according_corpus_order(matrix, word_to_ix):
#     """
#     making sure words are ordered according dictionary words order

#     """
#     ordered_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
#     word_to_id = dictionary.token2id

#     for i, word in enumerate(words):
#         if word in word_to_id:
#             dict_index = word_to_id[word]
#             ordered_matrix[dict_index] = matrix[i]

#     return ordered_matrix


def save_words_WE(load_path, save_path):
    word_to_ix = torch.load(load_path)
    keys = word_to_ix.keys()
    file_path = save_path

    # Write the string representation to a text file
    with open(file_path, 'w') as file:
        for word in keys:
            file.write(f"{word}\n")


HOME = "Results/BBCGlove"
TOP = 5

path_words_order = f"{HOME}/word_to_ix"
path_predictions = f"{HOME}/{TOP}_WE_predictions"


ordered_mat = create_prior(path_predictions)
torch.save(ordered_mat.T, f"{HOME}/{TOP}_WE_prior")

