from accelerate import Accelerator, InitProcessGroupKwargs
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import numpy as np
import os
from src.python_engine import run_python_code
from src.wolfram_engine import run_wolfram_code
from src.utils import set_seed
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW
tqdm = partial(tqdm, ncols=0, leave=False)

# Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
instruction = 'Question:\n'
cot_trigger = '\nAnswer reasoning:\n'
answer_trigger = '\nTherefore, the answer is: '

post_process_final_answer_fn_mapper = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    'svamp': lambda x: float(x.replace(',','').strip()),
    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
}
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: float(run_python_code(code_gen=answer_cot.strip())),
    ('python', 'svamp'): lambda answer_cot: float(run_python_code(code_gen=answer_cot.strip())),
    ('python', 'mathqa'): lambda answer_cot: str(run_python_code(code_gen=answer_cot.strip())).lower().replace('"','').replace("'",'').strip(),

    ('wolfram', 'gsm8k'): lambda answer_cot: float(eval(run_wolfram_code(code_gen=answer_cot.strip()).replace('*^','* 10 ** ').strip())),
    ('wolfram', 'svamp'): lambda answer_cot: float(eval(run_wolfram_code(code_gen=answer_cot.strip()).replace('*^','* 10 ** ').strip())),
    ('wolfram', 'mathqa'): lambda answer_cot: run_wolfram_code(code_gen=answer_cot.strip()).lower().replace('"','').replace("'",'').strip(),

    ('nl', 'gsm8k'): lambda answer_cot: float(answer_cot.split(answer_trigger)[-1].strip()),
    ('nl', 'svamp'): lambda answer_cot: float(answer_cot.split(answer_trigger)[-1].strip()),
    ('nl', 'mathqa'): lambda answer_cot: answer_cot.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip(),
}
compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-3,
    'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-3,
    'mathqa': lambda extracted_ans, target_answer: extracted_ans == target_answer,
}

def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(args['train_file'],'r'))),
            'test': Dataset.from_list(json.load(open(args['test_file'],'r'))),
        })
        accelerator.print('Raw data:', raw_dataset)
        tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True, remove_columns=dataset.column_names, 
                num_proc=8, load_from_cache_file=False
            ) for mode, dataset in raw_dataset.items()})
        accelerator.print('Processed data:', tokenized_dataset)

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, 
                        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
                        
    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, 
                        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
                        
    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)

def do_checkpoint(args, model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    tokenizer.save_pretrained(save_path)

def train_one_epoch(args, model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, test_dataset, test_dataloader, 
                    prefix, epoch, best_eval_log_dict):

    max_epoch = args['n_epochs']
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq = args.get('logging_step_freq', None)
    saving_step_freq = args.get('saving_step_freq', None)

    model.train()
    epoch_result_dict = defaultdict(list)
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop'):
        output = model(**batch['forward_kwargs'])
        # Get some metrics
        loss = output[0]
        result_dict, extra = {}, None

        # Update
        accelerator.backward(loss)
        if clip_grad_norm is not None:
            accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1

        # Step update metric
        epoch_result_dict['loss'].append(loss.item()) 
        for k, v in result_dict.items():
            epoch_result_dict[k].append(v)

        # Step evaluating
        eval_log_dict = {}
        is_best = False
        if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
            evaluate_result_dict = {f'Eval.Rerank.{k}':  v for k, v in evaluate_rerank(args, model, test_dataset, test_dataloader, tokenizer).items()}
            eval_log_dict.update(evaluate_result_dict)

        # Step logging
        train_log_dict = {}
        if logging_step_freq is not None and global_step % logging_step_freq == 0:
            train_log_dict = {f'Train.{k}': sum(v)/len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}
        
        if eval_log_dict or train_log_dict:
            log_dict = {**train_log_dict, **eval_log_dict, **best_eval_log_dict}
            log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
            accelerator.print(f"{prefix}[Epoch={epoch}/{max_epoch}, Step={global_step}] LR={scheduler.get_last_lr()[0]:.5g}, Metric: {log_dict}")

        # Step saving
        if saving_step_freq is not None and global_step % saving_step_freq == 0:
            save_path = os.path.join(model_dir, f'global_step_{str(global_step)}{"_best" if is_best else ""}')
            do_checkpoint(args, model, tokenizer, save_path)

        # Keep only max_record items
        for k, v in epoch_result_dict.items():
            if len(v) > 50:
                epoch_result_dict[k] = v[-50:]

    # Metric summary:
    epoch_result_dict = {k:(sum(v)/len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step

def evaluate_rerank(args, model, dataset, dataloader, tokenizer):
    model.eval()
    epoch_result_dict = defaultdict(list)
    predictions = []
    probabilities = []
    targets = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process, desc='Evaluation Loop'):
        output = model(**batch['forward_kwargs'])
        
        # Get some metrics:
        loss = output[0]     

        # Step update metric
        loss = accelerator.gather(loss).mean()
        epoch_result_dict['loss'].append(loss.item())

        # Prediction 
        logits = output[1]
        labels = batch['forward_kwargs']['labels']

        # Gather
        logits, labels = accelerator.gather(logits), accelerator.gather(labels)
        probs = torch.softmax(logits, dim=-1)
        probs, labels = probs.cpu().float().numpy(), labels.cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        predictions.extend(preds.tolist())
        probabilities.extend(probs.tolist())
        targets.extend(labels.tolist())

    # Pred
    predictions = predictions[:len(dataset)]
    probabilities = probabilities[:len(dataset)]
    targets = targets[:len(dataset)]

    cls_acc = (np.array(predictions) == np.array(targets)).mean()

    # Gathering from multiple sample
    item_id_to_result = defaultdict(list)
    for pred, tar, prob, item in zip(predictions, targets, probabilities, dataset):
        item_id = item.get('item_id', None)
        item_id_to_result[item_id].append({
            'item_id':item_id,
            # 'question': item['question'],
            # 'answer_value': item['answer_value'],
            # 'prediction_cot': item['prediction_cot'].split('\n'),
            # 'prediction_value': item['prediction_value'],
            'vote_correctness': item['vote_correctness'],
            'prediction_correctness': item['prediction_correctness'],
            'cls_prob_tokens': prob,
            # 'cls_tar_tokens': tar,
            # 'cls_pred_tokens': pred,
        })

    rerank_acc = []
    rerank_ub = []
    vote_correctness = []
    for item_id, group in item_id_to_result.items():
        # Upper bound:
        upper_bound = 0
        if any([item['prediction_correctness'] for item in group]):
            upper_bound = 1
        rerank_ub.append(upper_bound)

        # Last score
        last_score = [item['cls_prob_tokens'][1] for item in group]
        last_score_pred = group[int(np.argmax(last_score))]
        rerank_acc.append(last_score_pred['prediction_correctness'])

        # Vote
        vote_correctness.append(last_score_pred['vote_correctness'])


    model.train()
    return {'cls_acc': cls_acc, 
            'vote_acc': sum(vote_correctness)/len(vote_correctness), 
            'rerank_acc': sum(rerank_acc)/len(rerank_acc), 
            'upper_bound': sum(rerank_ub)/len(rerank_ub)}

def tokenize_fn(batch, args, tokenizer):
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        item_id, question, answer_value, predictions, vote_correctness = \
                item['item_id'], \
                item['question'], \
                item['answer_value'], \
                item['predictions'], \
                item['is_correct']

        question, answer_value = question.strip(), answer_value.strip()
        for sample in predictions:
            prediction_cot, prediction_correctness, prediction_value = sample['completion'], sample['correctness'], sample['solving_res']

            input = f'{instruction}{question}{cot_trigger}{prediction_cot}'
            input_encode = tokenizer(input, add_special_tokens=False)
            input_ids = input_encode['input_ids']
            attention_mask = [1]* len(input_ids)
            labels = prediction_correctness 

            # Truncation and filtering 
            input_ids = input_ids[:args['max_input_length']]
            attention_mask = attention_mask[:args['max_input_length']]

            ##
            new_batch['input_ids'].append(input_ids)
            new_batch['labels'].append(labels)
            new_batch['attention_mask'].append(attention_mask)
            ##
            new_batch['item_id'].append(item_id)
            new_batch['question'].append(question)
            new_batch['prediction_cot'].append(prediction_cot)
            new_batch['prediction_correctness'].append(prediction_correctness)
            new_batch['prediction_value'].append(prediction_value)
            new_batch['answer_value'].append(answer_value)
            new_batch['vote_correctness'].append(vote_correctness)
        
    return new_batch

def collate_fn(batch, args, tokenizer):
    max_input_length = max([len(item['input_ids']) for item in batch])
    input_ids  = []
    attention_mask  = []
    labels  = []
    for item in batch:
        input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
        attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))
        labels.append(item['labels'])
    forward_kwargs = {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.BoolTensor(attention_mask),
        'labels': torch.LongTensor(labels),
    }
    return {
        'forward_kwargs': forward_kwargs,
    }

def main(args):
    set_seed(args['seed'] + accelerator.process_index)
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'], use_fast=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.unk_token_id is None:
        tokenizer.unk_token_id = 3
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = 3

    (train_dataset, train_dataloader), (test_dataset, test_dataloader) = prepare_datasets_and_data_loaders(args, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args['model_name_or_path'], num_labels=2, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    accelerator.print(f'[Vocab size]: {len(tokenizer)}')
    model.resize_token_embeddings(len(tokenizer))

    n_epochs = args['n_epochs']
    batch_size_per_device = len(train_dataloader) // accelerator.num_processes
    num_training_steps = batch_size_per_device * n_epochs
    warmup_step = args['warmup_step'] if args['warmup_step'] > 0 else int(0.1 * num_training_steps)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    accelerator.print(
        f"***** Running training *****\n"
        f"  Num examples = {len(train_dataset)}\n"
        f"  Num Epochs = {n_epochs}\n"
        f"  Instantaneous batch size per device = {args['batch_size']}\n"
        f"  Total train batch size (w. parallel, distributed) = {args['batch_size']*accelerator.num_processes}\n"
        f"  Total optimization steps = {num_training_steps}\n"
        f"  Warm up step: {warmup_step}\n"
        f"  Learning rate: {args['learning_rate']}\n"
    )   
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    global_step = 0
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    logging_epoch_freq = args['logging_epoch_freq']
    saving_epoch_freq = args['saving_epoch_freq']
    model_dir=args['model_dir']
    best_eval_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(1, n_epochs+1):
        kwargs = {
            'args': args,
            'model': model, 
            'train_dataset': train_dataset, 
            'train_dataloader': train_dataloader, 
            'test_dataset': test_dataset,
            'test_dataloader': test_dataloader,
            'optimizer': optimizer, 
            'scheduler': scheduler,
            'global_step': global_step, 
            'tokenizer': tokenizer,
            'prefix':'[Train-Step]', 
            'epoch': epoch,
            'best_eval_log_dict': best_eval_log_dict
        }
        train_epoch_result_dict, global_step = train_one_epoch(**kwargs)

        eval_log_dict = {}
        is_best = False
        if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
            evaluate_result_dict = {f'Eval.Rerank.{k}':  v for k, v in evaluate_rerank(args, model, test_dataset, test_dataloader, tokenizer).items()}
            eval_log_dict.update(evaluate_result_dict)

        train_log_dict = {}
        if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
            train_log_dict = {f'Train.{k}': sum(v)/len(v) if isinstance(v, list) else v for k, v in train_epoch_result_dict.items()}
        
        if eval_log_dict or train_log_dict:
            log_dict = {**train_log_dict, **eval_log_dict, **best_eval_log_dict}
            log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
            accelerator.print(f"[Epoch={epoch}/{args['n_epochs']}, Step={global_step}] LR={scheduler.get_last_lr()[0]:.5g}, Metric: {log_dict}")

        if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
            save_path=os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}{"_best" if is_best else ""}')
            do_checkpoint(args, model, tokenizer, save_path)

if __name__ == '__main__':
    from transformers import HfArgumentParser
    NONE_INT = -100 
    NONE_STR = 'None'
    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str 
        test_file: str
        batch_size: int = field(default=4)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        ###

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k,v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]) # wait for processing upto 5hrs
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
