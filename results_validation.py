import torch
import os
import numpy as np
import csv
import pandas as pd


def save_topics_words(lists):
    with open(f"{HOME}/validations/{TOP}_{prior_type}_pca.csv", "w") as f:
        wr = csv.writer(f)
        for list in lists:
            f.write(f'Number of words in topic: {len(list)}')
            f.write(f'\n')

            for word in list:
                f.write(f"{word}\n")
            f.write('\n')

def save_df_topics_words(lists):
    # Create a DataFrame where each row is a topic and each column is a word
    df = pd.DataFrame(lists)

    # Save to CSV file (index=False to avoid saving index numbers)
    df.to_csv(f'{prior_type}_prior.csv', index=False, header=False)


def save_topic_word_distributions():
    output_file = f"{prior_type}_topic_word_distribution.csv"  # File to save the distribution

    # Open a CSV file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row: "Topic", followed by all words
        header = ["Topic"] + [ix_to_word[ix] for ix in range(len(ix_to_word))]
        writer.writerow(header)
        
        # Write each topic's word probabilities
        for topic_idx in range(num_of_topics):
            topic_pred = pred[topic_idx].tolist()  # Convert the tensor to a list
            writer.writerow([f"{topic_idx}"] + topic_pred)

    print(f"Topic-word distribution saved to {output_file}")

prior_type = "GMM"
TOP = 200
HOME = "NewResults/20NewsGroup"
pred_path = f"{HOME}/200prior_{prior_type}"
word2ix_path = f"{HOME}/word_to_ix"

pred = torch.load(pred_path)
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

    print(f"Number of words in topic {i} = {shape}")
    
    topic_words = []
    for word_ix in words_ix:
        topic_words.append((ix_to_word[word_ix])) # use also topic_pred[word_ix] to know the probability of that word in that topic
    
    words_per_topic.append(topic_words)

save_df_topics_words(words_per_topic)
# save_topics_words(words_per_topic)

# if want to save topic-word distribution of prior file (before using LDA)
# save_topic_word_distributions()


