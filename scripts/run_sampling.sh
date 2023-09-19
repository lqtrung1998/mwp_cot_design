#!/bin/bash

num_processes=6
main_process_port=8889


# Sample from train set
input_path="data/gsm8k_python_sdp.json" 
model_name="checkpoints/gsm8k_python_sdp/global_step_310_epoch_1" 
engine="python" 
batch_size='32' 
max_length='1024' 
num_return_sequences='2' 
temperature='1.0' 
do_sample='1'
save_dir="sampling_results/gsm8k_python_sdp/global_step_310_epoch_1/gsm8k_python_sdp_K_${num_return_sequences}_temp_${temperature}" 


accelerate launch --config_file ./default_config.yaml \
    --num_processes=${num_processes} --main_process_port=${main_process_port} \
    sampling.py \
    --model_name "${model_name}" \
    --input_path "${input_path}" \
    --save_dir "${save_dir}" \
    --engine "${engine}" \
    --batch_size "${batch_size}" \
    --max_length "${max_length}" \
    --num_return_sequences "${num_return_sequences}" \
    --temperature "${temperature}" \
    --do_sample "${do_sample}"


# Sample from test set
input_path="data/gsm8k_test_set.json" 
model_name="checkpoints/gsm8k_python_sdp/global_step_12090_epoch_39" 
engine="python" 
batch_size='32' 
max_length='1024' 
num_return_sequences='2' 
temperature='1.0' 
do_sample='1'
save_dir="sampling_results/gsm8k_python_sdp/global_step_12090_epoch_39/gsm8k_test_set_K_${num_return_sequences}_temp_${temperature}" 


accelerate launch --config_file ./default_config.yaml \
    --num_processes=${num_processes} --main_process_port=${main_process_port} \
    sampling.py \
    --model_name "${model_name}" \
    --input_path "${input_path}" \
    --save_dir "${save_dir}" \
    --engine "${engine}" \
    --batch_size "${batch_size}" \
    --max_length "${max_length}" \
    --num_return_sequences "${num_return_sequences}" \
    --temperature "${temperature}" \
    --do_sample "${do_sample}"


