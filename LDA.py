from utils import prepare_data, get_lda_input
import torch
import numpy as np
from gensim.models import LdaModel

num_topics = 104


def read_words_file(path):
    # Read the file and create a list of words
    with open(path, 'r') as file:
        eta_words = [line.strip() for line in file]

    return eta_words


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

def run_lda(eta_type="", eta='symmetric'):
    
    temp = dictionary[0]  # This is only to "load" the dictionary.
    lda = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary.id2token, passes=10, eta=eta)

    if eta == 'symmetric':
        lda.save("lda_model")
    else:
        lda.save(f"lda_model_with_eta_{eta_type}")

HOME_PATH = "/home/tal/dev/TopicModeling"

words_no_filter = read_words_file(f"{HOME_PATH}/words_WE_no_filter.txt")
doc_term_matrix, dictionary = get_lda_input(prepare_data())


ordered_mat = get_sorted_matrix_according_corpus_order(matrix=torch.load(f"{HOME_PATH}/beta/beta_no_filter_WE"), 
                                                       words=words_no_filter, 
                                                       dictionary=dictionary)

torch.save(ordered_mat.T, f"{HOME_PATH}/eta/WE_no_filter_eta")









