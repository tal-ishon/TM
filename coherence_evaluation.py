import helper_functions as hf
import pandas as pd
from gensim import corpora
import pickle
import re

DATASET_NAME = "20NewsGroup"
PATH_DF = f"/home/dsi/ishonta/TM/Distributions-Results/{DATASET_NAME}"
PATH_DATA = f"ProcessedData/{DATASET_NAME}"
MODELS = ["100_prior_DM"]

top_k = 20

# LOAD DOCUMENTS DATA - corpus, dictionary
# Load the dictionary from a file
dictionary = corpora.Dictionary.load(f'{PATH_DATA}/lda_dictionary.gensim')

# Load the BoW corpus from a Matrix Market format file
bow_corpus = corpora.MmCorpus(f'{PATH_DATA}/lda_corpus.mm')
doc_term_matrix = [list(doc) for doc in bow_corpus]

with open(f'{PATH_DATA}/filtered_corpus.pkl', 'rb') as f:
    filtered_corpus = pickle.load(f)


texts = [
    list((dictionary[word_id] for word_id, freq in bow))
    for bow in doc_term_matrix
]

pattern = r"\d+GMM"

for MODEL in MODELS:
    for k in [10, 15, 20]:
        topics_df = pd.read_csv(f"{PATH_DF}/{MODEL}_topic_word_distribution.csv")

        if re.findall(pattern, MODEL):
            topic_word_lists = [list(topic[:k]) for topic in topics_df.values]
        else:
            topic_word_lists = [
            topics_df.loc[topic].nlargest(k).index.tolist()  # Get top k indices (words) for the topic
            for topic in topics_df.index
            ]

        if MODEL != "BERTopic":
            cv = hf.coherence_model_evaluation(topics=topic_word_lists, texts=texts, dictionary=dictionary, coherence="c_v")
            # umass = hf.coherence_model_evaluation(topics=topic_word_lists, texts=filtered_corpus, dictionary=dictionary, coherence="umass")
        else:
            cv = hf.coherence_model_evaluation(topics=topic_word_lists, texts=texts, dictionary=dictionary, coherence="c_v")
            # umass = hf.coherence_model_evaluation(topics=topic_word_lists, texts=texts, dictionary=dictionary, coherence="umass")

        print(f"Top {k} - {MODEL} cv Coherence: {cv}")


    
