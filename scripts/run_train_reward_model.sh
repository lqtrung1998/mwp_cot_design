#!/bin/bash

exp_name="gsm8k_python_sdp_reward"
model_dir="checkpoints/gsm8k_python_sdp_reward/"
train_file="sampling_results/gsm8k_python_sdp/global_step_310_epoch_1/gsm8k_python_sdp_K_2_temp_1.0.json"
test_file="sampling_results/gsm8k_python_sdp/global_step_12090_epoch_39/gsm8k_test_set_K_2_temp_1.0.json"

model_name_or_path="checkpoints/gsm8k_python_sdp/global_step_12090_epoch_39"
tokenizer_name_or_path="checkpoints/gsm8k_python_sdp/global_step_12090_epoch_39"
batch_size="3"
n_epochs="3"
num_workers="8"
learning_rate="1e-6"
weight_decay="1e-6"
warmup_step="0"
clip_grad_norm="1"
evaluating_epoch_freq="1"
logging_epoch_freq="1"
saving_epoch_freq="1"
evaluating_step_freq="-100"
logging_step_freq="10"
saving_step_freq="-100"
seed="42"
max_input_length="700"

num_processes='6'
main_process_port='8888'

CMD='train_reward_model.py \
    --model_name_or_path "${model_name_or_path}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --train_file "${train_file}" \
    --test_file "${test_file}" \
    --model_dir "${model_dir}" \
    --batch_size "${batch_size}" \
    --n_epochs "${n_epochs}" \
    --num_workers "${num_workers}" \
    --learning_rate "${learning_rate}" \
    --weight_decay "${weight_decay}" \
    --warmup_step "${warmup_step}" \
    --clip_grad_norm "${clip_grad_norm}" \
    --evaluating_epoch_freq "${evaluating_epoch_freq}" \
    --logging_epoch_freq "${logging_epoch_freq}" \
    --saving_epoch_freq "${saving_epoch_freq}" \
    --evaluating_step_freq "${evaluating_step_freq}" \
    --logging_step_freq "${logging_step_freq}" \
    --saving_step_freq "${saving_step_freq}" \
    --seed "${seed}" \
    --max_input_length "${max_input_length}" \
    1> >(tee "${model_dir}"/"${exp_name}".log) \
    2> >(tee "${model_dir}"/"${exp_name}".err >&2)'

mkdir -p "${model_dir}"
final_cmd="accelerate launch \
            --config_file ./default_config_deepspeed.yaml \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
            $CMD \
            "

eval "$final_cmd"