#  #  #  #  #  #  #  #  #
## test LDA performance #
#  #  #  #  #  #  #  #  #

from gensim.models import LdaModel
import preprocessing as pp
from itertools import chain
import torch
from gensim.models.coherencemodel import CoherenceModel
import sys
import numpy as np
import random
import pandas as pd
from gensim import corpora
import pickle


CORPUS_PATH = None
IS_FIRST = None
IS_SAVED = None

# Setting seeds for reproducibility
random.seed(42)
np.random.seed(42)

def compute_umass_coherence(model_name, model):
  from gensim.models.coherencemodel import CoherenceModel
  # Compute c_v coherence
  coherence_model_cv = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
  coherence_cv = coherence_model_cv.get_coherence()
  print(f"{model_name} u_mass Coherence: {round(coherence_cv, 5)}")


def compute_cv_coherence(model_name, model):
  # Compute c_v coherence
#   coherence_model_cv = CoherenceModel(topics=topics, texts=texts, dictionary=pp.dictionary, coherence='c_v')
  coherence_model_cv = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
  coherence_cv = coherence_model_cv.get_coherence()
  print(f"{model_name} c_v Coherence: {round(coherence_cv, 5)}")


def get_intersection(list1, list2):
    return list(set(list1) & set(list2))


def process_data(corpus_path, word_to_ix, data_type):
    """
    This function should load corpus and get the cleaned corpus and vocabulary out of it.
    Update embedding and word_to_ix - The embedding should contain only the words that are in the 
    processor's vocabulary.
    """
    if data_type == "csv":
        sentences, labels = pp.prepare_cleaner_csv_data(corpus_path)
    elif data_type == "json":
        sentences = pp.prepare_cleaner_data(corpus_path)
    elif data_type == "txt":
        sentences = pp.prepare_cleaner_txt_data(corpus_path)

    else:
        print("Can't process this data type!")
        return

    words_embed = word_to_ix.keys()  # words in glove embedding
    words_corpus = list(chain(*sentences))
    words = get_intersection(words_embed, words_corpus)
    # pp.save_file_txt("20NewsGroupWords", words)
    _, _ = pp.get_filtered_corpus(sentences, words)


def get_topics(LDA):
   return [[word for word, _ in LDA.show_topic(topicid, topn=10)] for topicid in range(LDA.num_topics)]


def create_random_prior():
    # Create a random word-to-topic assignment matrix (topic-word prior)
    random_prior = np.random.rand(TOPIC_NUM, len(dictionary.id2token))

    # Normalize each row to sum to 1 (so they form valid probability distributions)
    random_prior = random_prior / random_prior.sum(axis=1, keepdims=True)

    return random_prior


def create_false_prior(k, V):
    # Each row represents a topic, and each column represents a word.

    # Step 1: Create a matrix of zeros with shape (k, V)
    word_topic_matrix = np.zeros((k, V))

    # Step 2: Set all values in the first column to 1
    word_topic_matrix[1] = 0.5
    word_topic_matrix[2] = 0.5

    return word_topic_matrix



def save_learned_eta(model_type, model):
    num_words = TOPIC_NUM  # Set the number of words per topic

    # Get topics in a list of tuples (topic_id, words)
    topics_words = model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

    # Convert the list of topics and words into a list of word lists (one list per topic)
    topics_words_list = [[word for word, prob in topic_words] for _, topic_words in topics_words]

    # Create a pandas DataFrame with topics as rows and terms as columns
    df = pd.DataFrame(topics_words_list)

    df = pd.DataFrame({
            'Topic': [f'Topic {i}' for i in range(len(topics_words_list))],
            'Words': [', '.join(words) for words in topics_words_list]
        })

    # Save the DataFrame to a CSV file
    df.to_csv('{}_topic_term_matrix.csv'.format(model_type), index=False, header=False)
    # df.to_csv('{}_check.csv'.format(model_type), index=False, header=False)


def calculate_bertopic_coherence(topic_words, corpus):
    # Compute c_v coherence
    coherence_model_cv = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()

    # Compute u_mass coherence
    coherence_model_umass = CoherenceModel(topics=topic_words, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_umass = coherence_model_umass.get_coherence()

    print(f"c_v Coherence: {coherence_cv}")
    print(f"u_mass Coherence: {coherence_umass}")


def save_learned_topic_word_distribution(topic_words, num_topics):
        # Create a DataFrame
        topic_df = pd.DataFrame({
            'Topic': [f'Topic {i}' for i in num_topics],
            'Words': [', '.join(words) for words in topic_words]
        })

        # Save to CSV
        topic_df.to_csv('topic_words.csv', index=False)


def run_bertopic():
    from bertopic import BERTopic
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    bertopic_model = BERTopic()
    topics, probabilities = bertopic_model.fit_transform(filtered_corpus)

    for topic_id in range(5):
        print(f"Topic {topic_id + 1}: \n{bertopic_model.get_topic(topic_id)}")

    num_topics = len(bertopic_model.get_topics())

    # Create topic words list with error handling
    topic_words = []
    valid_topic_ids = []  # Keep track of valid topic IDs

    for topic_id in range(num_topics):
        topic = bertopic_model.get_topic(topic_id)
        if isinstance(topic, list):
            words = [word for word, _ in topic]
            topic_words.append(words)
            valid_topic_ids.append(topic_id)  # Track valid topic IDs
        else:
            print(f"Skipping invalid topic ID: {topic_id}")

    calculate_bertopic_coherence(topic_words=topic_words, corpus=doc_term_matrix)
    save_learned_topic_word_distribution(topic_words=topic_words, num_topics=valid_topic_ids)


def run_lda_models():
    prior_type = ["prior_GMM"]
    models = []
    eta_weight = 70

    for type in prior_type:
        if not IS_SAVED:
            alpha = "auto"
            if type == "lda":
                LDA = LdaModel(doc_term_matrix, 
                            num_topics=TOPIC_NUM, 
                            id2word=dictionary.id2token, 
                            passes=10, 
                            alpha=alpha)
            elif type == "random":
                random_prior = create_random_prior()
                LDA = LdaModel(doc_term_matrix, 
                            num_topics=TOPIC_NUM, 
                            id2word=dictionary.id2token, 
                            passes=10, 
                            eta=random_prior * eta_weight, 
                            alpha=alpha)
            elif type == "false":
                false_prior = create_false_prior(k=TOPIC_NUM, V=len(dictionary.id2token))
                print("FINISH CREATING FALSE PRIOR")
                LDA = LdaModel(doc_term_matrix, 
                            num_topics=TOPIC_NUM, 
                            id2word=dictionary.id2token, 
                            passes=10, 
                            eta=false_prior * eta_weight, 
                            alpha=alpha)
            else:
                prior = torch.load(f"{HOME}/{type}")
                LDA = LdaModel(doc_term_matrix, 
                            num_topics=TOPIC_NUM, 
                            id2word=dictionary.id2token, 
                            passes=10, 
                            eta=prior * eta_weight, 
                            alpha=alpha)
            
            # LDA.save(f"{type}")

        else:
            LDA = LdaModel.load(f"{type}")

        models.append(LDA)
        
        print("\n ###### {} ###### \n ".format(type))
        for topic_id in range(5):
            print(f"Topic {topic_id + 1}: \n{LDA.show_topic(topic_id, topn=10)}")


    models_type = ["LDA_GMM"]

    for model_type, model in zip(models_type, models):
        # topics = get_topics(model)
        compute_umass_coherence(model_type, model)
        compute_cv_coherence(model_type, model)
        save_learned_eta(model_type=model_type, model=model)


def init(topic_num = 100, home_dir='NewResults', dataset_name="20NewsGroup", is_first=False, is_model_saved=False):
    global TOPIC_NUM, HOME, DATASET_NAME, DATASET_TYPE, CORPUS_PATH, word_to_ix, IS_FIRST, IS_SAVED

    TOPIC_NUM = topic_num
    HOME = f'{home_dir}/{dataset_name}'
    DATASET_NAME = dataset_name

    if DATASET_NAME == "20NewsGroup":
        DATASET_TYPE = "json"
    elif DATASET_NAME == "BBC" or DATASET_NAME == "Trump'sTweets":
        DATASET_TYPE = "csv"
    else:
        DATASET_TYPE = "txt"

    CORPUS_PATH = f'{DATASET_NAME}.{DATASET_TYPE}'
    
    word_to_ix = torch.load(f"{HOME}/word_to_ix")
    IS_FIRST = is_first
    IS_SAVED = is_model_saved


def main():
    argv = sys.argv
    argc = len(argv)

    if argc == 2:
        topic_num = argv[1]
        init(topic_num=topic_num)
    elif argc == 3:
        topic_num, home_dir = int(argv[1]), argv[2]
        init(topic_num=topic_num, home_dir=home_dir)
    elif argc == 4:
        topic_num, home_dir, dataset_name = argv[1], argv[2], argv[3]
        init(topic_num=topic_num, home_dir=home_dir, dataset_name=dataset_name)
    elif argc == 5:
        topic_num, home_dir, dataset_name, is_first = argv[1], argv[2], argv[3], argv[4]
        init(topic_num=topic_num, home_dir=home_dir, dataset_name=dataset_name, is_first=bool(is_first))
    else:
        init()


main()

if IS_FIRST:
    process_data(corpus_path=CORPUS_PATH, word_to_ix=word_to_ix, data_type=DATASET_TYPE)
    print("FINISH PROCESSING")
    pp.dictionary.save("lda_dictionary.gensim")
    corpora.MmCorpus.serialize("lda_corpus.mm" ,pp.doc_term_matrix)
    with open('filtered_corpus.pkl', 'wb') as f:
        pickle.dump(pp.filtered_corpus, f)

    doc_term_matrix = pp.doc_term_matrix
    dictionary = pp.dictionary
    filtered_corpus = pp.filtered_corpus
    print("finish")

else:
    # Load the dictionary from a file
    dictionary = corpora.Dictionary.load('lda_dictionary.gensim')

    # Load the BoW corpus from a Matrix Market format file
    bow_corpus = corpora.MmCorpus('lda_corpus.mm')
    doc_term_matrix = [list(doc) for doc in bow_corpus]

    with open('filtered_corpus.pkl', 'rb') as f:
        filtered_corpus = pickle.load(f)


texts = [
        list((dictionary[word_id] for word_id, freq in bow))
        for bow in doc_term_matrix
    ]

run_lda_models()
# run_bertopic()


# prior : topic - word distribution. 
# For a given word, we first calculate the probability of being "clustered" to each topic.
# Thus, the sum of the column which represents the word should be 1. The topic sum can be much higher.

    