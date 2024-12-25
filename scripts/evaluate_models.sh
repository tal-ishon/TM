#!/bin/bash

cd ~/TM || exit 1

# Define arrays of dataset names, models, and corresponding number of topics
models=("200prior_GMM" "100GMM") 
datasets=("20NewsGroup")

# Path to your Python file
python_file="model_evaluation_HuggingFace.py"

# Loop through the models, datasets, and their corresponding topics
for model in "${models[@]}"; do
  for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    
    echo "Running $model on $dataset..."
    
    # Run the Python script with the model and dataset args
    python "$python_file" --model_type "$model" --dataset "$dataset" 
    
    # Check if the Python script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing $dataset with $model. Exiting."
        # Continue to the next iteration if in a loop, or just proceed without exiting
        continue    
    fi
  done
done

echo "All models and datasets processed successfully!"

