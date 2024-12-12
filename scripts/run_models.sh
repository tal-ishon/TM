#!/bin/bash

cd ~/TM || exit 1

# Define arrays of dataset names, models, and corresponding number of topics
datasets=("20NewsGroup" "BBC")
models=("lda" "prior_GMM" "prior_ScaSE")  # Replace with your actual model names
num_topics=("100" "10")  # Replace with the actual number of topics for each dataset

# Path to your Python file
python_file="test_gensim.py"

# Loop through the models, datasets, and their corresponding topics
for model in "${models[@]}"; do
  for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    topics="${num_topics[$i]}"
    
    echo "Running $model on $dataset with $topics topics..."
    
    # Run the Python script with the model, dataset, and num_of_topics arguments
    python "$python_file" --model_type "$model" --dataset "$dataset" --num_of_topics "$topics"
    
    # Check if the Python script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing $dataset with $model. Exiting."
        exit 1
    fi
  done
done

echo "All models and datasets processed successfully!"

