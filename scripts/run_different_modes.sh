#!/bin/bash

cd ~/TM || exit 1

# Define arrays of dataset names, models, and corresponding number of topics
modes=("ScaSE")  # Replace with your actual mode
datasets=("20NewsGroup" "BBC" "Trump'sTweets")
num_topics=("100" "10" "200")  # Replace with the actual number of topics for each dataset

# Path to your Python file
python_file="main_BERT.py"

# Loop through the models, datasets, and their corresponding topics
for mode in "${modes[@]}"; do
  for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    topics="${num_topics[$i]}"
    
    echo "Running $mode on $dataset with $topics topics..."
    
    # Run the Python script with the model, dataset, and num_of_topics arguments
    python "$python_file" --mode "$mode" --dataset "$dataset" --topics "$topics"
    
    # Check if the Python script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing $dataset with $mode. Exiting."
        exit 1
    fi
  done
done

echo "All models and datasets processed successfully!"

