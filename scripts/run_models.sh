#!/bin/bash

cd ~/TM || exit 1

# Define arrays of dataset names, models, and corresponding number of topics
datasets=("20NewsGroup")
models=("pmi")  # Replace with your actual model names
num_topics=("100")  # Replace with the actual number of topics for each dataset
current_dir=${1}

# Path to your Python file
python_file="test_graph_gensim.py"

# Loop through the models, datasets, and their corresponding topics
for model in "${models[@]}"; do
  for topic in "${num_topics[@]}"; do
    dataset="${datasets[@]}"
    topics="${topic}"
    
    echo "Running $model on $dataset with $topics topics in $current_dir directory..."
    
    # Run the Python script with the model, dataset, and num_of_topics arguments
    python "$python_file" --model_type "$model" --dataset "$dataset" --num_topics "$topics" --current_dir "$current_dir"
    
    # Check if the Python script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing $dataset with $model. Exiting."
        exit 1
    fi
  done
done

echo "All models and datasets processed successfully!"

