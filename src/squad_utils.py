import os 
import torch
from tqdm import tqdm
from collections import Counter

import re
import string

import numpy as np


def generate_summary(model, tokenizer, source, source_mask, target):
  
  output = model.module.generate(source,
                max_length=16, 
                early_stopping=True)
  
  target[target[:,:]==-100] = 0
  
  machine_text = [tokenizer.decode(senten, skip_special_tokens=True) for senten in output]
  human_text = [tokenizer.decode(senten, skip_special_tokens=True) for senten in target]
  machine_text = ["replacement_word" if text == "" else text for text in machine_text]
  human_text = ["replacement_word" if text == "" else text for text in human_text]
  
  del source
  del output 
  del target

  return {"machine_text": machine_text, "human_text":human_text}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return metric_fn(prediction, ground_truths)


def qa_metrics(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
      total += 1
      exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'f1':f1, 'em':exact_match}


def evaluate(model,
             tokenizer,
             valid_loader,
             metric,
             device):
    
    em = 0.0
    f1 = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader)):
            
                source, source_mask, target, _ = torch.tensor(batch['input_ids']).to(device),\
                    batch['attention_mask'].to(device),\
                    batch['labels'], batch['decoder_attention_mask'].to(device)
                target = torch.tensor(target)
                target[target[: ,:] == 0 ] = -100
                target = target.to(device)
                
                all_text = generate_summary(model, tokenizer, source, source_mask, target)
                result = qa_metrics(all_text['machine_text'], all_text['human_text'])
                
                em += result['em']
                f1 += result['f1']

    print(f'Exact matching score: {em/len(valid_loader)}')
    print(f'F1 score: {f1/len(valid_loader)}')
    return (em + f1) / len(valid_loader)
 
 
def train_fn(model,
          tokenizer,
          optimizer,
          scheduler,
          train_loader, 
          valid_loader,
          metric,
          accumulation_steps=None,
          device=None):
        
    epoch_train_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader)):
        
        source, source_mask, target, _ = torch.tensor(batch['input_ids']).to(device),\
            batch['attention_mask'].to(device),\
            batch['labels'],\
            batch['decoder_attention_mask'].to(device)
            
        target = torch.tensor(target)
        target[target[: ,:] == 0 ] = -100 # to let the model skip computing the loss of the zeros (paddings)
        target = target.to(device)
        
        output = model(source, source_mask, labels = target, return_dict=True)
        loss = output['loss']
        loss = loss / accumulation_steps 
        loss.backward()
        
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        epoch_train_loss += loss.item()
           
    epoch_train_loss = epoch_train_loss / len(train_loader)
    epoch_rouge_f1 = evaluate(model, tokenizer, valid_loader, metric, device)
    
    return epoch_train_loss, epoch_rouge_f1