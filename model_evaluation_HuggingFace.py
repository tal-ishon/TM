from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import random
import json
import os
import sys
import torch
import re

os.environ["HUGGINGFACE_API_TOKEN"] = "TI_access_model"

# Initialize the LLM pipeline
model_name="meta-llama/Llama-3.3-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Define prompt
temp=0.7
llm_model = pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                model_kwargs={"torch_dtype": torch.bfloat16},
            )


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

def word_intrusion(top_words_by_topic, model, save_for_human_eval=False):
    word_intrusion_results = []

    print("\nPrompt #2:\n")
    i = 0
    for topic_id, words in top_words_by_topic.items():
        
        # Add an intruder word that does not belong
        all_other_words = sum(bottom_words_by_topic.values(), [])
        intruder_word = random.choice([word for word in all_other_words if word not in words])
        
        # Setup words including intruder
        word_list = words + [intruder_word]
        random.shuffle(word_list)
        
        numbered_word_list = ""
        for j, word in enumerate(word_list):
                numbered_word_list += f"{j + 1}. {word}\r\n"
        input_text = f"You are an assistant in understanding which word is the intruder among other words in a given list. Identify from the following list of words, which word does not belong with the others: {numbered_word_list}. In your response, return only the number of the intruder word from the list.\nFor example - Given the following list: ['card', 'driver', 'ethernet', 'mode', 'bothering', 'resolution', 'support', 'detector', 'radar'] the intruder word is: 'bothering'.\nAnother example - Given the following list of words: ['weapon', 'crime', 'rate', 'sickle', 'bill', 'license', 'control', 'carry', 'firearm'] the intruder word is: 'sickle'."       
        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]
        # Initialize pipeline with the loaded model

        # Generate output with constraints
        result = pipeline(
            messages,
            max_new_tokens=20,  # Limit to a few tokens to get a short response
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only show generated text, not the prompt
            temperature = temp
        )

        clean_result = result[0]['generated_text'].strip()
        answer_ix = int(re.sub(r'[^0-9]', '', clean_result)) - 1
        word_result = word_list[answer_ix]
        intrusion_result = {
            "topic_id": topic_id,
            "prompt": input_text,
            "model_response": word_result,
            "intruder": intruder_word
        }
        
        print("Words List: {}\nModel Intuder: {}\nReal Intruder: {}".format(word_list, word_result, intruder_word))
        if i > 5:
            print("\nDone! Examples for word intrution task appear above\n")
            exit(0)
        
        # word_intrusion_results.append(intrusion_result)
        i+=1
        
    # Optionally save for human evaluation   
    if save_for_human_eval:
        with open("word_intrusion_results.json", "w") as f:
            json.dump(word_intrusion_results, f)

    return word_intrusion_results

# -------------------------
# Topic Intrusion Task
# -------------------------

def topic_intrusion(doc_topics, top_words_by_topic, model, save_for_human_eval=False):
    topic_intrusion_results = []

    for doc_id, topics in doc_topics.items():
        all_topic_ids = list(top_words_by_topic.keys())
        intruder_topic_id = random.choice([t for t in all_topic_ids if t not in topics])
        intruder_topic_words = top_words_by_topic[intruder_topic_id]
        
        topic_words_list = [top_words_by_topic[topic_id] for topic_id in topics] + [intruder_topic_words]
        random.shuffle(topic_words_list)
        
        input_text = f"From the following list of topics, identify the one topic that does not belong with the others: {topic_words_list}. In your response, use only the intruder topic from the list without any additional explanation."
        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]
        # Initialize pipeline with the loaded model

        # Generate output with constraints
        result = pipeline(
            messages,
            max_new_tokens=20,  # Limit to a few tokens to get a short response
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only show generated text, not the prompt
            temperature = temp
        )

        topic_result = result[0]['generated_text'].strip()
        intrusion_result = {
            "doc_id": doc_id,
            "prompt": input_text,
            "model_response": topic_result,
            "intruder_topic_id": intruder_topic_id
        }
        
        topic_intrusion_results.append(intrusion_result)
        

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
# Test LLM
# -------------------------

def test_llm_performance(words_lists, intruders):
    for intruder, word_list in zip(intruders, words_lists):
        
        numbered_word_list = ""
        for j, word in enumerate(word_list):
                numbered_word_list += f"{j + 1}. {word}\r\n"
        input_text = f"Your task is understanding which word is the intruder among other words in a given list. Identify from the following list of words, which word is least related to the others: {numbered_word_list}.\nFor example - Given the following list: ['card', 'driver', 'ethernet', 'mode', 'bothering', 'resolution', 'support', 'detector', 'radar'] the intruder word is: 'bothering'.\nAnother example - Given the following list of words: ['weapon', 'crime', 'rate', 'sickle', 'bill', 'license', 'control', 'carry', 'firearm'] the intruder word is: 'sickle'. Your response should be only the word you believe to be the intruder word. No additional explanations"
        messages = [
            {"role": "user", "content": f"{input_text}"},
        ]
        # Initialize pipeline with the loaded model

        # Generate output with constraints
        result = pipeline(
            messages,
            max_new_tokens=20,  # Limit to a few tokens to get a short response
            no_repeat_ngram_size=2,
            return_full_text=False,  # Only show generated text, not the prompt
            temperature = temp
        )

        clean_result = result[0]['generated_text'].strip()
        # index = int(clean_result) - 1
        # words_result = word_list[index]

        intrusion_result = {
            "prompt": input_text,
            "model_response": clean_result,
            "intruder": intruder
        }

        print("Words List: {}\nModel Intuder: {}\nReal Intruder: {}".format(word_list, clean_result, intruder))


# -------------------------
# Main Execution
# -------------------------


# if len(sys.argv) < 2:
#     dataset = "20NewsGroup"
# else:
#     dataset = sys.argv[1]

# path = "Distributions-Results/{}".format(dataset)
# # Load distributions
# topic_word_df = load_topic_word_distribution(f"{path}/lda_topic_word_distribution.csv")
# doc_topic_df = load_doc_topic_distribution(f"{path}/lda_document_topic_distribution.csv")

# # Prepare data
# top_words_by_topic = get_top_words_for_topics(topic_word_df, top_n=8)
# bottom_words_by_topic = get_bottom_words_for_topics(topic_word_df, top_n=10)
# doc_topics = get_doc_topics(doc_topic_df, prob_threshold=0.1)

# # Run word intrusion task
# word_intrusion_results = word_intrusion(top_words_by_topic, model=llm_model, save_for_human_eval=True)

# # Run topic intrusion task
# topic_intrusion_results = topic_intrusion(doc_topics, top_words_by_topic, model=llm_model, save_for_human_eval=True)

# # Evaluate results
# evaluation_results = evaluate_intrusion_tasks(word_intrusion_results, topic_intrusion_results)

# # Save evaluation results to a file
# with open("evaluation_results.json", "w") as f:
#     json.dump(evaluation_results, f)


list_of_words = [['weapon', 'crime', 'rate', 'sickle', 'bill', 'license', 'control', 'carry', 'firearm'],
                 ['manager', 'window', 'problem', 'program', 'application', 'display', 'file', 'widget', 'sensor'],
                 ['peace', 'israel', 'israeli', 'taught', 'jewish', 'palestinian', 'muslim', 'arab', 'bosnia'],
                 ['medical', 'effect', 'slot', 'disease', 'cause', 'patient', 'treatment', 'food', 'doctor'],
                 ['hockey', 'team', 'baseball', 'game', 'information', 'season', 'player', 'play', 'league']]

intruders = ['sickle', 'sensor', 'taught', 'slot', 'information']

test_llm_performance(list_of_words, intruders)


