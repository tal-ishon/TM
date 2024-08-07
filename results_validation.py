import torch
import os
import numpy as np
import csv


def save_topics_words(lists):
    with open(f"{TOP}_WE_words.csv", "w") as f:
        wr = csv.writer(f)
        for list in lists:
            f.write(f'Number of words in topic: {len(list)}')
            f.write(f'\n')

            for word in list:
                f.write(word + '\n')
            f.write('\n')

TOP = 5
HOME = "Results/BBCGlove"
pred_path = f"{HOME}/{TOP}_WE_predictions"
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

save_topics_words(words_per_topic)



