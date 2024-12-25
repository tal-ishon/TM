# Steps for full code running

## Run main.py
Main creates the priors for downstream task - LDA models.
In order to run main.py you need to choose which word embedding to use, dataset, number of topics and model of evaluation (GMM, ScaSE and so on).


## Run test_gensim.py
In order to run different models over your dataset for later on model's performance evaluation you can run the file with the right parameters, or you can just adjust the params in run_models.sh script and it'lll run your file.
#### For example:
```
./scripts/run_models.sh 20NewsGroup 100 GMM
```
Runnig this file will output a topic-word distribution that will be used for word intrusion task to evaluate the different models.

## Run python notebook create_intruder_files.ipynb
By running this file we convert our word-topic distributions into a suited file for word intrusion task. You need to choose the dataset and the model you want to create the file for.

The program outputs 2 files. 
> {model_name}_intruders_check.csv

> {model_name}_intruders_check.csv

first file contains a top_k words + intruder for each topic of the model that has been chosen in the program's parameters. The second file contains the intruders that had been assigned for each topic. Those files are input files for word-intrusion task that will be given to a LLM.

## Run evaluate_models_HuggingFace.py
Evaluates models performance by running word-intrusion task over LLM.