from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
import random
from src.python_engine import run_python_code
from src.wolfram_engine import run_wolfram_code
from src.utils import set_seed, is_numeric, timeout
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW
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
            evaluate_result_dict = {f'Eval.Gen.{k}':  v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
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

def evaluate_generation(args, model, dataset, dataloader, tokenizer):
    model.eval()
    predictions = []
    targets = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process, desc='Evaluation Gen Loop'):
        output_ = model.module.generate(
                        **batch['generate_prefix_kwargs'], 
                        max_length=args['max_input_length'],
                        output_scores=True,
                        return_dict_in_generate=True,
                        num_beams=1,
                        use_cache=True,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)

        labels = batch['generate_prefix_kwargs']['labels']
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        labels[labels==-100]=tokenizer.pad_token_id

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in generated_ids]
        predictions.extend(preds)
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in labels]
        targets.extend(target)

    predictions = predictions[:len(dataset)]
    targets = targets[:len(dataset)]
    
    # Preprocess
    for i, (pred, tar) in enumerate(zip(predictions, targets)):
        new_pred = pred.split(cot_trigger)[-1]
        new_tar = tar.split(cot_trigger)[-1]
        predictions[i] = new_pred
        targets[i] = new_tar

    if accelerator.is_main_process and accelerator.is_local_main_process:
        results = [{
            'pred': pred, 
            'tar': tar, 
            'item_id': item.get('item_id', None),
            'answer_value': item.get('answer_value', None),
            'answer_type': item.get('answer_type', None),
        } for pred, tar, item in zip(predictions, targets, dataset)]

        corr_value = 0
        for cur_res in results:
            prediction, target, item_id = cur_res['pred'], cur_res['tar'], cur_res['item_id']
            src_name = item_id.split('_')[0]
            answer_value = cur_res['answer_value']

            ## Processing target
            target_cot = target.strip()
            target_value = post_process_final_answer_fn_mapper[src_name](answer_value)
            cur_res['target_cot'] = target_cot
            cur_res['target_value'] = target_value

            ## Processing prediction
            try:
                with timeout(seconds=10):
                    prediction_cot = prediction.strip()
                    prediction_value = post_process_answer_cot_fn_mapper[(args['engine'], src_name)](prediction_cot)
            except Exception as e:
                print(str(e))
                prediction_cot = None
                prediction_value = None
            cur_res['prediction_cot'] = prediction_cot
            cur_res['prediction_value'] = prediction_value

            # Compute correctness
            is_correct = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
            corr_value += is_correct
            cur_res['is_correct']  = is_correct

        value_accuracy = corr_value / len(predictions) * 100
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]

    # Metric summary:
    model.train()
    return {'value_accuracy': value_accuracy}

def tokenize_fn(batch, args, tokenizer):
    assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
    new_batch = defaultdict(list)
    all_keys = list(batch.keys())
    for item_values in zip(*(batch[k] for k in all_keys)):
        item = {k: item_values[i] for i, k in enumerate(all_keys)}
        item_id, question, answer_value, answer_cot = \
                item['item_id'], \
                item['question'], \
                item['answer_value'], \
                item.get('answer_cot', None), \

        question, answer_value = question.strip(), answer_value.strip()
        if answer_cot:
            answer_cot = answer_cot.strip()

        input = f'{instruction}{question}'
        output = f'{cot_trigger}{answer_cot}'
        prefix_text = f'{instruction}{question}{cot_trigger}'

        input_encode = tokenizer(input, add_special_tokens=False)
        output_encode = tokenizer(output, add_special_tokens=False)
        prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

        input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
        labels = [-100]*len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
        attention_mask = [1]* len(input_ids)
        prefix = prefix_encode['input_ids']
        prefix_attention_mask = prefix_encode['attention_mask']

        # Truncation
        input_ids = input_ids[:args['max_input_length']]
        labels = labels[:args['max_input_length']]
        attention_mask = attention_mask[:args['max_input_length']]
        prefix = prefix[:args['max_input_length']]
        prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]

        ##
        new_batch['input_ids'].append(input_ids)
        new_batch['labels'].append(labels)
        new_batch['attention_mask'].append(attention_mask)
        new_batch['prefix'].append(prefix)
        new_batch['prefix_attention_mask'].append(prefix_attention_mask)
        ##
        new_batch['item_id'].append(item_id)
        new_batch['question'].append(question)
        new_batch['answer_cot'].append(answer_cot)
        new_batch['answer_value'].append(answer_value)
        
    return new_batch

def collate_fn(batch, args, tokenizer):
    max_input_length = max([len(item['input_ids']) for item in batch])
    max_target_length = max([len(item['labels']) for item in batch])
    max_prefix_length = max([len(item['prefix']) for item in batch])

    input_ids  = []
    attention_mask  = []
    labels, labels_left_padded  = [], []
    prefix_left_padded  = []
    prefix_attention_mask_left_padded  = []
    for item in batch:
        input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
        attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))
        labels.append(item['labels'] + [-100]*(max_target_length - len(item['labels'])))

        labels_left_padded.append([-100]*(max_target_length - len(item['labels'])) + item['labels'])
        prefix_left_padded.append([tokenizer.pad_token_id]*(max_prefix_length - len(item['prefix'])) + item['prefix'])
        prefix_attention_mask_left_padded.append([0]*(max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])

    forward_kwargs = {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.BoolTensor(attention_mask),
        'labels': torch.LongTensor(labels)
    }
    generate_prefix_kwargs = {
        'input_ids': torch.LongTensor(prefix_left_padded),
        'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'labels': torch.LongTensor(labels_left_padded)
    }

    return {
        'forward_kwargs': forward_kwargs,
        'generate_prefix_kwargs': generate_prefix_kwargs,
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
    model = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'], low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
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
            evaluate_result_dict = {f'Eval.Gen.{k}':  v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
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
        engine: str = field(default='python')

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
