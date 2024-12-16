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
import os
import little_mallet_wrapper as lmw
# from bertopic import BERTopic
    

CORPUS_PATH = None
IS_FIRST = None
IS_SAVED = None
MODEL_TYPE = None

# Setting seeds for reproducibility
random.seed(42)
np.random.seed(42)
# Set environment variable for Python hash seed
os.environ['PYTHONHASHSEED'] = str(42)

# Force single thread operation
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def save_distributions(model, model_type, corpus):
    path = f"Distributions-Results/{DATASET_NAME}"
    # topic-word

    # Get topic-word matrix
    topic_word_matrix = model.get_topics()  # shape (num_topics, num_words)

    # Save to CSV
    topic_word_df = pd.DataFrame(topic_word_matrix, columns=[dictionary[i] for i in range(topic_word_matrix.shape[1])])
    topic_word_df.to_csv(f"{path}/{model_type}_topic_word_distribution.csv", index_label="Topic")

    # document-topic
    # Convert to dense matrix
    doc_topic_matrix = np.zeros((len(corpus), model.num_topics))
    for i, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow):
            doc_topic_matrix[i, topic_id] = prob.round(5)

    # Save to CSV
    doc_topic_df = pd.DataFrame(doc_topic_matrix)
    doc_topic_df.to_csv(f"{path}/{model_type}_document_topic_distribution.csv", index_label="Document")


def saveBERTopic_distributions(model, probs):
    # Assume `documents` is your list of original documents

    path = f"Distributions-Results/{DATASET_NAME}"

   # Initialize dictionary to hold topic-word distributions
    topic_word_distributions = {}
    for topic_id in model.get_topic_info().Topic:
        if topic_id != -1:  # -1 usually represents outliers in BERTopic
            words, scores = zip(*model.get_topic(topic_id))  # Get words and their scores for the topic
            topic_word_distributions[topic_id] = dict(zip(words, scores))

    # Convert to DataFrame for saving
    topic_word_df = pd.DataFrame(topic_word_distributions).fillna(0)  # Replace NaN with 0 for missing words
    topic_word_df = topic_word_df.transpose()  # Transpose to have topics as rows

    # Save to CSV
    topic_word_df.to_csv(f"{path}/BERTopic_topic_word_distribution.csv", index_label="Topic")

    # Convert to DataFrame
    doc_topic_df = pd.DataFrame(probs.round(5))
    doc_topic_df.columns = [i for i in range(probs.shape[1])]  # Name columns by topic
    
    # Save to CSV
    doc_topic_df.to_csv(f"{path}/BERTopic_document_topic_distribution.csv", index_label="Document")

# Function to calculate perplexity
def calculate_perplexity(model, test_corpus):
    return model.log_perplexity(test_corpus)


# Function to calculate NPMI
def calculate_npmi(topics, texts, model_name):
    npmi_metric = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_npmi') 
    cohenrence_npmi = npmi_metric.get_coherence()
    print(f"{model_name} npmi Coherence: {round(cohenrence_npmi, 5)}")
    return cohenrence_npmi


def compute_umass_coherence(model_name, model):
    from gensim.models.coherencemodel import CoherenceModel
    # Compute u_mass coherence
    coherence_model_umass = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
    coherence_umass = coherence_model_umass.get_coherence()
    print(f"{model_name} u_mass Coherence: {round(coherence_umass, 5)}")


def compute_cv_coherence(model_name, model):
    coherence_model_cv = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()
    print(f"{model_name} c_v Coherence: {round(coherence_cv, 5)}")
    cv.append(coherence_cv)


def compute_topic_diversity_coherence(model_name, model, top_n):
    """
    Calculate topic diversity for a given LDA model.

    Parameters:
    lda_model (gensim.models.LdaModel): Trained LDA model
    top_n (int): Number of top words per topic to consider

    Returns:
    float: Topic diversity score
    """
    if model_name == "BERTopic":
        # Get top N words for each topic
        topics = model.get_topics()
        
        # Exclude the '-1' topic if present, which refers to outliers/noise
        if -1 in topics:
            del topics[-1]

        top_words = [word for topic_id in topics for word, _ in topics[topic_id][:top_n]]

    else:
        # Get top N words for each topic
        topics = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)
        # Flatten the list of words and calculate unique words
        top_words = [word for topic in topics for word, _ in topic[1]]
        

    unique_words = set(top_words)

    # Calculate topic diversity
    total_words = len(top_words)
    unique_words_count = len(unique_words)
    topic_diversity_coherence = unique_words_count / total_words

    print(f"{model_name} Topic - Diversity Coherence: {round(topic_diversity_coherence, 5)}")
    topic_diversity.append(topic_diversity_coherence)


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
    _, _ = pp.get_filtered_corpus(sentences, words, "glove")


def get_topics(LDA):
   return [[word for word, _ in LDA.show_topic(topicid, topn=10)] for topicid in range(LDA.num_topics)]

def calculate_avg_std(type, values):
    import statistics

    # Calculate the mean
    average = statistics.mean(values)

    # Calculate the standard deviation
    std_dev = statistics.stdev(values)

    print("### Type: {} ###".format(type))
    print("Average:", average)
    print("Standard Deviation:", std_dev)


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
    num_words = 10  # Set the number of words per topic

    # Get topics in a list of tuples (topic_id, words)
    topics_words = model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

    # Convert the list of topics and words into a list of word lists (one list per topic)
    topics_words_list = [[f'{word}: {round(float(prob), 5)}' for word, prob in topic_words] for _, topic_words in topics_words]

    # Create a pandas DataFrame with topics as rows and terms as columns
    df = pd.DataFrame(topics_words_list)

    df = pd.DataFrame({
            'Topic': [f'Topic {i}' for i in range(len(topics_words_list))],
            'Words': [', '.join(words) for words in topics_words_list]
        })

    # Save the DataFrame to a CSV file
    # df.to_csv('{}_topic_term_matrix.csv'.format(model_type), index=False, header=False)
    df.to_csv('Learned_eta/{}/200{}_100_eta.csv'.format(DATASET_NAME, model_type), index=False, header=False)


def calculate_bertopic_coherence(topic_words, corpus):
    # Compute c_v coherence
    coherence_model_cv = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_cv = coherence_model_cv.get_coherence()

    # Compute u_mass coherence
    coherence_model_umass = CoherenceModel(topics=topic_words, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_umass = coherence_model_umass.get_coherence()

    print(f"c_v Coherence: {coherence_cv}")
    print(f"u_mass Coherence: {coherence_umass}")
    cv.append(coherence_cv)


def save_learned_topic_word_distribution(topic_words, num_topics):
        # Create a DataFrame
        topic_df = pd.DataFrame({
            'Topic': [f'Topic {i}' for i in num_topics],
            'Words': [', '.join(words) for words in topic_words]
        })

        # Save to CSV
        topic_df.to_csv('Learned_eta/{}/topic_words.csv'.format(DATASET_NAME), index=False)


def run_bertopic():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    bertopic_model = BERTopic(calculate_probabilities=True)
    topics, probabilities = bertopic_model.fit_transform(filtered_corpus)

    for topic_id in range(5):
        print(f"Topic {topic_id + 1}: \n{bertopic_model.get_topic(topic_id)}")

    num_topics = len(bertopic_model.get_topics())

    # Create topic words list with error handling
    topic_words = []
    valid_topic_ids = []  # Keep track of valid topic IDs
    topic_words_save = []

    for topic_id in range(num_topics):
        topic = bertopic_model.get_topic(topic_id)
        if isinstance(topic, list):
            words_save = [f'{word}: {round(prob, 5)}' for word, prob in topic]
            words = [word for word, _ in topic]
            topic_words.append(words)
            topic_words_save.append(words_save)
            valid_topic_ids.append(topic_id)  # Track valid topic IDs
        else:
            print(f"Skipping invalid topic ID: {topic_id}")

    # calculate_bertopic_coherence(topic_words=topic_words, corpus=doc_term_matrix)
    # compute_topic_diversity_coherence(model_name="BERTopic", model=bertopic_model, top_n=10)
    # save_learned_topic_word_distribution(topic_words=topic_words_save, num_topics=valid_topic_ids)
    # calculate_npmi(topics=topic_words, texts=texts, model_name="BERTopic")
    saveBERTopic_distributions(model=bertopic_model, probs=probabilities)

def run_lda_models(prior_type="lda"):
    eta_weight = 10
    
    alpha = "auto"
    if prior_type == "lda":
        LDA = LdaModel(doc_term_matrix, 
                    num_topics=TOPIC_NUM, 
                    id2word=dictionary.id2token, 
                    passes=10, 
                    alpha=alpha)
    elif prior_type == "random":
        random_prior = create_random_prior()
        LDA = LdaModel(doc_term_matrix, 
                    num_topics=TOPIC_NUM, 
                    id2word=dictionary.id2token, 
                    passes=10, 
                    eta=random_prior * eta_weight, 
                    alpha=alpha)
    elif prior_type == "false":
        false_prior = create_false_prior(k=TOPIC_NUM, V=len(dictionary.id2token))
        print("FINISH CREATING FALSE PRIOR")
        LDA = LdaModel(doc_term_matrix, 
                    num_topics=TOPIC_NUM, 
                    id2word=dictionary.id2token, 
                    passes=10, 
                    eta=false_prior * eta_weight, 
                    alpha=alpha)
    else:
        prior = torch.load(f"{HOME}/{prior_type}")
        LDA = LdaModel(doc_term_matrix, 
                    num_topics=TOPIC_NUM, 
                    id2word=dictionary.id2token, 
                    passes=10, 
                    eta=prior * eta_weight, 
                    alpha=alpha)
            
            # LDA.save(f"{type}")
    # else:
    #     LDA = LdaModel.load(f"{prior_type}")
    
    print("\n ###### {} ###### \n ".format(prior_type))
    for topic_id in range(5):
        print(f"Topic {topic_id + 1}: \n{LDA.show_topic(topic_id, topn=10)}")


    # models_type = ["LDA", "RANDOM", "GMM", "ScaSE"]

# for model_type, model in zip(models_type, models):
    # topics = get_topics(model)
    model = LDA
    model_type = prior_type
    # compute_umass_coherence(model_type, model)
    # compute_cv_coherence(model_type, model)

    # compute_topic_diversity_coherence(model_type, model, top_n=10)
    # calculate_npmi(topics=get_topics(model), texts=texts, model_name=model_type)
    # save_learned_eta(model_type=model_type, model=model)
    save_distributions(model=model, model_type=model_type, corpus=bow_corpus)



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
    """
    Params -
        argv[1]: dataset name
        argv[2]: number of topics
        argv[3]: is first - deafult True
    """
    # argv = sys.argv
    # argc = len(argv)

    # if argc == 2:
    #     dataset = argv[1]
    #     init(dataset_name=dataset)
    # elif argc == 3:
    #     dataset = argv[1]
    #     topic_num = int(argv[2])
    #     init(dataset_name=dataset, topic_num=topic_num)
    # elif argc == 4:
    #     dataset = argv[1]
    #     topic_num = int(argv[2])
    #     is_first=bool(int(argv[3]))
    #     init(dataset_name=dataset, topic_num=topic_num, is_first=is_first)
    # else:
    #     init()

    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run a model with the specified parameters.")
    parser.add_argument("--model_type", type=str, required=True, help="The type of model to run")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to use")
    parser.add_argument("--num_of_topics", type=int, required=True, help="The number of topics")

    global MODEL_TYPE, TOPIC_NUM
    # Parse the arguments
    args = parser.parse_args()
    MODEL_TYPE = args.model_type
    dataset = args.dataset
    TOPIC_NUM = args.num_of_topics

    print(f"Dataset: {dataset}")

    init(dataset_name=dataset, topic_num=TOPIC_NUM)

        
        
main()

path_save_data = f"ProcessedData/{DATASET_NAME}"

if IS_FIRST:
    process_data(corpus_path=CORPUS_PATH, word_to_ix=word_to_ix, data_type=DATASET_TYPE)
    print("FINISH PROCESSING")
    pp.dictionary.save(f"{path_save_data}/lda_dictionary.gensim")
    corpora.MmCorpus.serialize(f"{path_save_data}/lda_corpus.mm" ,pp.doc_term_matrix)
    with open(f'{path_save_data}/filtered_corpus.pkl', 'wb') as f:
        pickle.dump(pp.filtered_corpus, f)

    doc_term_matrix = pp.doc_term_matrix
    dictionary = pp.dictionary
    filtered_corpus = pp.filtered_corpus
    print("finish")

else:
    # Load the dictionary from a file
    dictionary = corpora.Dictionary.load(f'{path_save_data}/lda_dictionary.gensim')

    # Load the BoW corpus from a Matrix Market format file
    bow_corpus = corpora.MmCorpus(f'{path_save_data}/lda_corpus.mm')
    doc_term_matrix = [list(doc) for doc in bow_corpus]

    with open(f'{path_save_data}/filtered_corpus.pkl', 'rb') as f:
        filtered_corpus = pickle.load(f)


texts = [
        list((dictionary[word_id] for word_id, freq in bow))
        for bow in doc_term_matrix
    ]

topic_diversity = []
cv = []

if MODEL_TYPE == "BERTopic":
    run_bertopic()
else:
    run_lda_models(prior_type=MODEL_TYPE)




# calculate_avg_std("TD", topic_diversity)
# calculate_avg_std("cv", cv)



# prior : topic - word distribution. 
# For a given word, we first calculate the probability of being "clustered" to each topic.
# Thus, the sum of the column which represents the word should be 1. The topic sum can be much higher.