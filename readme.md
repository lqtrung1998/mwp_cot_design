# SFT
Modify parameters in ```run_train_sft_model.sh``` and run
```
bash run_train_sft_model.sh
```
The script will finetune a model (initialized from ```${model_name_or_path}```) on ```${train_file}``` dataset, save the checkpoint to ```${model_dir}``` directory and evaluate the solving accuracy on ```${test_file}``` dataset.

# Sampling
Modify parameters in ```run_sampling.sh``` and run
```
bash run_sampling.sh
```
This script will sample K=```${num_return_sequence}``` solutions from model init from ```${model_name}``` for each input/prefix from data in ```${input_path}```. The sampling result is stored in ```${save_dir}```.

Each correctness of the solution is determined by the extracted/executed answer and the ground-truth answer.

# Rerank model training
Sample reward model training and evaluation data (following the above section)
* Training set: sample completion on the training set using an early checkpoint saved during SFT fine-tuning process. (to ensure high sampling diversity)
* Test set: sample completion on the training set using best SFT checkpoint


Then modify parameters in ```run_train_reward_model.sh``` and train reward model by running
```
bash run_train_reward_model.sh
```
This script train the reward model (initialized from ```${model_name_or_path}```) on ```${train_file}``` dataset, save the checkpoint to ```${model_dir}``` directory and evaluate the re-ranking accuracy on the ```${test_file}``` dataset.
