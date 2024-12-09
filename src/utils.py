import argparse
import json
import re
import string
import sys
from collections import Counter
import os  
import random

import torch 
import random
import datasets
import numpy as np  
from torch.utils.data import DataLoader
from collections import OrderedDict
from transformers import RobertaTokenizer
from transformers import logging
from torch.backends import cudnn

logging.set_verbosity_error()

task_text_field_map = {
            'cola': ['sentence'],
            'sst2': ['sentence'],
            'mrpc': ['sentence1', 'sentence2'],
            'qqp': ['question1', 'question2'],
            'stsb': ['sentence1', 'sentence2'],
            'mnli': ['premise', 'hypothesis'],
            'qnli': ['question', 'sentence'],
            'rte': ['sentence1', 'sentence2'],
            'wnli': ['sentence1', 'sentence2'],
            'ax': ['premise', 'hypothesis']
        }

glue_task_num_labels = {
            'cola': 2,
            'sst2': 2,
            'mrpc': 2,
            'qqp': 2,
            'stsb': 1,
            'mnli': 3,
            'qnli': 2,
            'rte': 2,
            'wnli': 2,
            'ax': 3
        }

loader_columns = [
            'input_ids',
            'label',
            'attention_mask'
        ]

task_dict = {
    'rte' : (2, 16),
    'cola' : (2, 16),
    'mrpc' : (2, 16),
    'stsb' : (1, 16),
    'wnli': (2, 16),
    'sst2' : (2, 32),
    'mnli' : (3, 32),
    'qnli' : (2, 32),
    'qqp' : (2, 32)
}

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def seed_worker(seed):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    
def make_dataloader(batch_size,
                    dataframe, num_workers,
                    task,
                    tokenizer, 
                    padding,
                    max_len,
                    dist=False
                    ):
       
    text_fields = task_text_field_map[task]

    def convert_to_features(example_batch, indices=None):
        
        if len(text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[text_fields[0]], 
                                               example_batch[text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[text_fields[0]]
                
        features = tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=max_len,
            truncation=True,
            padding=padding,
            add_special_tokens=True,
            return_token_type_ids=False
        ) 
            
        features['label'] = example_batch['label']
        return features
       
    for split in dataframe.keys():
        dataframe[split] = dataframe[split].map(
            convert_to_features,
            batched=True,
            )
        columns = [c for c in dataframe[split].column_names if c in loader_columns]
        dataframe[split].set_format(type="torch", columns=columns)

    eval_splits = [x for x in dataframe.keys() if 'validation' in x]
    
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataframe['train'])
        train_loader = DataLoader(
                dataframe['train'],
                batch_size = batch_size,
                num_workers = num_workers,
                sampler=train_sampler,
                worker_init_fn=seed_worker,
                shuffle=False,
                )
        
    else:
        train_loader = DataLoader(
                dataframe['train'],
                batch_size = batch_size,
                num_workers = num_workers,
                worker_init_fn=seed_worker,
                shuffle=False,
                drop_last=False
                )

    if len(eval_splits) == 1:
        valid_loader = DataLoader(
            dataframe['validation'],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
            )
            
    elif len(eval_splits) > 1:
        valid_loader = [DataLoader(
            dataframe[x],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False) for x in eval_splits]
           
    return train_loader, valid_loader

def remove_trash(state_dict, type=0, name='model_state_dict'):
    new_state_dict = OrderedDict()
    for k, v in state_dict[name].items():
        if (k == 'n_averaged'):
            continue
        if 'module' in k:
            if type == 0:
                name = k[7:]
            else:
                name = k[14:]
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
        
def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class AlphaWrapper(torch.nn.Module):
    def __init__(self, paramslist, model, names, valid_loader, device, metric):
        super(AlphaWrapper, self).__init__()
        self.paramslist = paramslist
        self.model = model 
        self.names = names 
        self.dataloader = valid_loader 
        self.device = device
        self.metric = metric
        
    def forward(self, train_x):
        
        params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, train_x.cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
            
        load_weights(self.model, self.names, params)
        
        valid_metrics = 0.0
        self.model.eval()
        for data in self.dataloader:
            
            input_ids = data['input_ids'].to(self.device)
            targets = data['label'].to(self.device)
            out = self.model(input_ids)
            _, valid_metric = self.metric.calculate(out['logits'], targets)
            valid_metrics += valid_metric
            
        return valid_metrics / len(self.dataloader)
    
def normalize(seq):
    v = 1/torch.sum(seq, dim=-1)
    seq = seq * v
    return seq

class metrics:
    def __init__(self, task_flag):
        self.metric_fn = datasets.load_metric('glue', task_flag)
        self.task_flag = task_flag

    def calculate(self, logits, targets):
        result = None 
        
        if self.task_flag != 'stsb':
            _, preds = torch.max(logits, dim=-1)
            result = preds.eq(targets).sum().item()
            result = (result / preds.size(0)) * 100
            metric = self.metric_fn.compute(predictions=preds, references=targets)
        else:
            metric = self.metric_fn.compute(predictions=logits, references=targets)
        
        if self.task_flag == 'cola':
            metric = metric["matthews_correlation"] * 100
        elif self.task_flag == 'stsb':
            metric = metric["pearson"] * 100
            result = metric
        elif self.task_flag in ['mrpc', 'qqp']:
            metric = metric["f1"] * 100
        else:
            metric = result

        return result, metric

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(seed):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)

def configure_cudnn(debug):
    cudnn.enabled = True
    cudnn.benchmark = True
    if debug:
        cudnn.deterministic = True
        cudnn.benchmark = False
        
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def squad_evaluate(prediction, gold_answers):
  em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
  f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
  
def generate_summary(model, tokenizer, source, target):
  
  output = model.module.generate(source,
                max_length=200, 
                num_beams=10,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True)
  

  output = output
  target[target[:,:]==-100] = 0

  machine_text = [tokenizer.decode(senten, skip_special_tokens=True) for senten in output]
  human_text = [tokenizer.decode(senten, skip_special_tokens=True) for senten in target]

  return {"machine_text": machine_text, "human_text":human_text}