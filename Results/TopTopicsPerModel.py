import pandas as pd
import random
import sys
import os
random.seed(0)

argv = sys.argv
if len(argv) < 2:
    CURRENT_DIR = "Octis/BBC/dm_pmi/5/20k_15W"
    MODELS = ["pmi"]
    print("MODELS: {}".format(MODELS))
else:
    raise ValueError("Must provide a directory from which code load distributions")


def get_top_words_for_topics(topic_word_df, top_n=5):
    top_words_by_topic = {}
    for topic_id in topic_word_df['Topic']:
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0]
        # Sort the row values in descending order and get the top N columns (words)
        top_words = topic_row.sort_values(ascending=False).head(top_n)

        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = top_words.index.tolist()

    return top_words_by_topic


import re
ks = [5, 10, 15, 20]
DATA_SET = "20NewsGroup"


for MODEL in MODELS:
    pattern1 = re.fullmatch(r"\d*lda", MODEL)
    pattern2 = re.fullmatch(r"\d*ProdLDA", MODEL)
    if MODEL == "BERTopic" or pattern1 or pattern2:
        dir_path = f"/home/dsi/ishonta/TM/helper/{CURRENT_DIR}"
        HOME_DIR = f"/home/dsi/ishonta/TM/Distributions-Results/{CURRENT_DIR}"

    else:
        dir_path = f"/home/dsi/ishonta/TM/WordsGraph/{CURRENT_DIR}/helper"
        HOME_DIR = f"/home/dsi/ishonta/TM/WordsGraph/{CURRENT_DIR}/distributions"

    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    df = pd.read_csv(f"{HOME_DIR}/{MODEL}_topic_word_distribution.csv")
    df = df.round(10)
    
    top_words_by_topic = get_top_words_for_topics(df, top_n=20)
    df = pd.DataFrame(top_words_by_topic)
    df.to_csv(f"Results/{MODEL}_top_words.csv")