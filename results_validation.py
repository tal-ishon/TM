import torch
import os
import numpy as np
import csv
import pandas as pd


def save_topics_words(lists):
    with open(f"{HOME}/validations/{TOP}_GMM.csv", "w") as f:
        wr = csv.writer(f)
        for list in lists:
            f.write(f'Number of words in topic: {len(list)}')
            f.write(f'\n')

            for word in list:
                f.write(word + '\n')
            f.write('\n')

def save_df_topics_words(lists):
    # Create a DataFrame where each row is a topic and each column is a word
    df = pd.DataFrame(lists)

    # Save to CSV file (index=False to avoid saving index numbers)
    df.to_csv('GMM_eta_prior.csv', index=False, header=False)


TOP = 100
HOME = "NewResults/20NewsGroup"
pred_path = f"{HOME}/pred_GMM"
word2ix_path = f"{HOME}/word_to_ix"

pred = torch.load(pred_path).T
word_to_ix = torch.load(word2ix_path)
ix_to_word = {v: k for k, v in word_to_ix.items()}

num_of_topics = pred.shape[0]

words_per_topic = []

for i, _ in enumerate(pred):
    topic_pred = pred[i].round(10)

    # keep only indicies that are higher than 0.1
    x = np.where(topic_pred > 0.1)
    shape = x[0].shape[0]
    sorted = np.argsort(-topic_pred)
    words_ix = sorted[:shape]

    print(f"Number of words in topic = {shape}")
    
    topic_words = []
    for word_ix in words_ix:
        topic_words.append(ix_to_word[word_ix])
    
    words_per_topic.append(topic_words)

save_df_topics_words(words_per_topic)



