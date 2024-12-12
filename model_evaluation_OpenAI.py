import pandas as pd
import random
import json
import os
import sys
import openai

# Set your OpenAI API key here
openai.api_key = "OpenAI_TI_model"

# -------------------------
# Load Data from CSV Files
# -------------------------

def load_topic_word_distribution(filepath):
    return pd.read_csv(filepath)

def load_doc_topic_distribution(filepath):
    return pd.read_csv(filepath)

# -------------------------
# Prepare Data for Tasks
# -------------------------

def get_top_words_for_topics(topic_word_df, top_n=8):
    top_words_by_topic = {}
    for topic_id in topic_word_df['Topic']:
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0]
        # Sort the row values in descending order and get the top N columns (words)
        top_words = topic_row.sort_values(ascending=False).head(top_n)
        
        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = top_words.index.tolist()
    
    return top_words_by_topic

def get_bottom_words_for_topics(topic_word_df, top_n=10):
    top_words_by_topic = {}
    for topic_id in topic_word_df['Topic']:
        # Get the row corresponding to the current topic (excluding the 'Topic' column)
        topic_row = topic_word_df.loc[topic_word_df['Topic'] == topic_id].drop(columns='Topic').iloc[0]
        # Sort the row values in descending order and get the top N columns (words)
        top_words = topic_row.sort_values(ascending=True).head(top_n)
        
        # Collect the top N words (column names)
        top_words_by_topic[topic_id] = top_words.index.tolist()
    
    return top_words_by_topic


def get_doc_topics(doc_topic_df, prob_threshold=0.0):
    doc_topics = {}
     # Iterate through each document (row)
    for doc_id, row in doc_topic_df.iterrows():
        # Exclude the 'Document' column (first column) and get the topic probabilities
        topic_probabilities = row.drop('Document')
        # Identify topics where the probability exceeds the threshold
        topics_above_threshold = topic_probabilities[topic_probabilities > prob_threshold].index.tolist()
        # Convert the topic names (strings) to integers if necessary (as column names are strings)
        topics_above_threshold = [int(topic) for topic in topics_above_threshold]
        # Store the result in the dictionary
        doc_topics[doc_id] = topics_above_threshold
    
    return doc_topics

# -------------------------
# Word Intrusion Task
# -------------------------

# -------------------------
# Word Intrusion Task
# -------------------------

def word_intrusion(top_words_by_topic, save_for_human_eval=False):
    word_intrusion_results = []

    for topic_id, words in top_words_by_topic.items():
        # Add an intruder word that does not belong
        all_other_words = sum(top_words_by_topic.values(), [])
        intruder_word = random.choice([word for word in all_other_words if word not in words])
        
        # Setup words including intruder
        word_list = words + [intruder_word]
        random.shuffle(word_list)
        
        # Construct the prompt for word intrusion task
        prompt = f"Identify the intruder in the following list: {', '.join(word_list)}. Respond with only one word - the intruder."
        
        # Call the OpenAI API to get a response
        response = openai.Completion.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "user", "content": "Hello, how are you?"}])
            
        word_result = response.choices[0].text.strip()  # Get the intruder word
        intrusion_result = {
            "topic_id": topic_id,
            "prompt": prompt,
            "model_response": word_result,
            "intruder": intruder_word
        }
        
        word_intrusion_results.append(intrusion_result)
        
        # Optionally save for human evaluation
        if save_for_human_eval:
            with open(f"word_intrusion_topic_{topic_id}.json", "w") as f:
                json.dump(intrusion_result, f)

    with open("word_intrusion_results.json", "w") as f:
        json.dump(word_intrusion_results, f)

    return word_intrusion_results

# -------------------------
# Topic Intrusion Task
# -------------------------

def topic_intrusion(doc_topics, top_words_by_topic, save_for_human_eval=False):
    topic_intrusion_results = []

    for doc_id, topics in doc_topics.items():
        all_topic_ids = list(top_words_by_topic.keys())
        intruder_topic_id = random.choice([t for t in all_topic_ids if t not in topics])
        intruder_topic_words = top_words_by_topic[intruder_topic_id]
        
        topic_words_list = [top_words_by_topic[topic_id] for topic_id in topics] + [intruder_topic_words]
        random.shuffle(topic_words_list)
        
        # Construct the prompt for topic intrusion task
        prompt = f"For document {doc_id}, which topic does not belong: " + \
                 " | ".join([", ".join(words) for words in topic_words_list]) + "?"
        
        # Call the OpenAI API to get a response
        response = openai.Completion.create(
            engine="text-davinci-003",  # or "gpt-3.5-turbo" for ChatGPT model
            prompt=prompt,
            max_tokens=10,  # Limit response to a single word (the intruder topic)
            temperature=0.2,  # Lower temperature for more deterministic responses
        )
        
        result = response.choices[0].text.strip()  # Get the intruder topic ID
        intrusion_result = {
            "doc_id": doc_id,
            "prompt": prompt,
            "model_response": result,
            "intruder_topic_id": intruder_topic_id
        }
        
        topic_intrusion_results.append(intrusion_result)
        
        # Optionally save for human evaluation
        if save_for_human_eval:
            with open(f"topic_intrusion_doc_{doc_id}.json", "w") as f:
                json.dump(intrusion_result, f)

    with open("topic_intrusion_results.json", "w") as f:
        json.dump(topic_intrusion_results, f)

    return topic_intrusion_results

# -------------------------
# Evaluate Model Performance
# -------------------------

def evaluate_intrusion_tasks(word_intrusion_results, topic_intrusion_results):
    word_intrusion_correct = sum(1 for res in word_intrusion_results if res['model_response'] == res['intruder'])
    topic_intrusion_correct = sum(1 for res in topic_intrusion_results if res['model_response'] == str(res['intruder_topic_id']))

    word_intrusion_accuracy = word_intrusion_correct / len(word_intrusion_results)
    topic_intrusion_accuracy = topic_intrusion_correct / len(topic_intrusion_results)

    print(f"Word Intrusion Task Accuracy: {word_intrusion_accuracy * 100:.2f}%")
    print(f"Topic Intrusion Task Accuracy: {topic_intrusion_accuracy * 100:.2f}%")
    
    return {
        "word_intrusion_accuracy": word_intrusion_accuracy,
        "topic_intrusion_accuracy": topic_intrusion_accuracy
    }

# -------------------------
# Main Execution
# -------------------------


if len(sys.argv) < 2:
    dataset = "20NewsGroup"
else:
    dataset = sys.argv[1]

path = "Distributions-Results/{}".format(dataset)
# Load distributions
topic_word_df = load_topic_word_distribution(f"{path}/lda_topic_word_distribution.csv")
doc_topic_df = load_doc_topic_distribution(f"{path}/lda_document_topic_distribution.csv")

# Prepare data
top_words_by_topic = get_top_words_for_topics(topic_word_df, top_n=8)
bottom_words_by_topic = get_bottom_words_for_topics(topic_word_df, top_n=10)
doc_topics = get_doc_topics(doc_topic_df, prob_threshold=0.1)

# Run word intrusion task
word_intrusion_results = word_intrusion(top_words_by_topic, save_for_human_eval=True)

# Run topic intrusion task
topic_intrusion_results = topic_intrusion(doc_topics, top_words_by_topic, save_for_human_eval=True)

# Evaluate results
evaluation_results = evaluate_intrusion_tasks(word_intrusion_results, topic_intrusion_results)

# Save evaluation results to a file
with open("evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f)
