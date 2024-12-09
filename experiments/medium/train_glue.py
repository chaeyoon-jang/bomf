# -----------------------------------------------------------------------------
# Copyright (c) 2024 Chaeyun Jang
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------
import os
import copy     
import gc
import argparse
import warnings

import numpy as np
import datetime
import logging

import utils
import ipdb
import json
import time
import torch
from functools import partial

from transformers import RobertaModel
from transformers.optimization import AdamW

from transformers import get_constant_schedule_with_warmup
from silence_tensorflow import silence_tensorflow
from torch.utils.data import DataLoader, Dataset, Subset

import socket
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tqdm import tqdm 

warnings.filterwarnings(action='ignore')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
silence_tensorflow()

large_task = ['sst2','qnli','qqp','mnli']
super_large_task = ['qqp', 'mnli']

class RobertaGLUE_base(torch.nn.Module):
    def __init__(self, config):
        super(RobertaGLUE_base, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.model_type, add_pooling_layer=False, ignore_mismatched_sizes=True)
        
        #for param in self.roberta.parameters():
        #    param.requires_grad = False
        
        #for layer in self.roberta.encoder.layer[config.freeze_num:]:
        #    for param in layer.parameters():
        #        param.requires_grad = True
                
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.classifier_dropout)
        self.out = torch.nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask.float())

        sequence_output = outputs[0]
        logits = sequence_output[:, 0, :]  
        logits = self.dropout(logits)
        logits = self.dense(logits)
        logits = torch.tanh(logits)
        logits = self.dropout(logits)  
        logits = self.out(logits)

        return logits

def validate(model,
             criterion,
             metric,
             valid_loader,
             device,
             task):
    
    model.eval()
    with torch.no_grad():
        
        valid_loss = 0.0
        valid_metric = 0.0
        for data in valid_loader:
            
            input_ids = data['input_ids'].cuda(device)
            attention_mask = data['attention_mask'].cuda(device)
            logits = model(input_ids, attention_mask)
            del input_ids

            if task == 'stsb':
                targets = data['label'].to(torch.float32).cuda(device)
                logits = logits.to(torch.float32).cuda(device).squeeze()
            else:
                targets = data['label'].to(torch.int64).cuda(device)
                
            if criterion:
                loss = criterion(logits, targets)
                valid_loss += loss.item()
                
            _, add = metric.calculate(logits, targets)
            valid_metric += add  
            
            del add  
            del loss 
            del targets   
            del logits  
            gc.collect()
            torch.cuda.empty_cache()
    
    return valid_loss / len(valid_loader), valid_metric / len(valid_loader)

def train_iter(n_epochs,
               model,
               train_loader,
               valid_loader,
               task,
               criterion,
               metric,
               optimizer,
               lr_scheduler,
               gpu, 
               args):
    
    if gpu == 0:
        print('***** Start Baseline Training *****')
    
    best_metric = 0.0
    step = 0
    stop_counter = 0
    
    for epoch in range(n_epochs):
        
        if stop_counter > 4:
            break
        
        model.train()
        for data in train_loader:
            
            input_ids = data['input_ids'].cuda(gpu)
            attention_mask = data['attention_mask'].cuda(gpu)
            logits = model(input_ids, attention_mask)
            del input_ids
            del attention_mask
            
            if task == 'stsb':
                targets = data['label'].to(torch.float64).cuda(gpu)
                logits = logits.to(torch.float64).cuda(gpu).squeeze()
            
            else:
                targets = data['label'].to(torch.int64).cuda(gpu)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            lr_scheduler.step()
            
            del targets  
            del logits   
            del loss   
            
            gc.collect()
            torch.cuda.empty_cache()
            
            step += 1
        
        epoch_valid_loss, epoch_valid_metric = validate(model, criterion, metric, valid_loader, gpu, task)
        
        if gpu == 0:
            print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} | metric: {epoch_valid_metric:.4f}%")
        
        if epoch_valid_metric > best_metric:
            best_metric = epoch_valid_metric
            best_step = step
            stop_counter = 0
            
        else:
            stop_counter += 1
    
    return best_step
            

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) 
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  

def cleanup():
    dist.destroy_process_group()

def collect_ckpt(best_step,
                 train_loader,
                 valid_loader,
                 swa_optimizer,
                 swa_model,
                 swa_lr_scheduler,
                 criterion,
                 metric,
                 gpu,
                 task,
                 n_epochs,
                 args,
                 ):
    
    if gpu == 0:
        print('***** Collecting SWA Check Points *****')
    
    swa_scheduler = torch.optim.swa_utils.SWALR(swa_optimizer, anneal_strategy="cos", swa_lr=args.learning_rate)
    
    swa_start_step = int(best_step * 0.5)
    swa_end_step = int(best_step * 2.0)
    
    if gpu == 0:
        print(f'***** SWA start step & end step = {swa_start_step}, {swa_end_step}')
    
    step = 0
    dk = 0
    
    for epoch in range(n_epochs):
        
        if step > swa_end_step:
            break

        swa_model.train()
        for data in train_loader:
            
            if step > swa_end_step:
                break
            
            input_ids = data['input_ids'].cuda(gpu)
            attention_mask = data['attention_mask'].cuda(gpu)
            
            logits = swa_model(input_ids, attention_mask)
            
            del input_ids
            del attention_mask
            
            if task == 'stsb':
                targets = data['label'].to(torch.float64).cuda(gpu)
                logits = logits.to(torch.float64).cuda(gpu).squeeze()
            
            else:
                targets = data['label'].to(torch.int64).cuda(gpu)
            
            loss = criterion(logits, targets)
            
            swa_optimizer.zero_grad()
            loss.backward()
            swa_optimizer.step()    
            
            if step < swa_start_step:
                swa_lr_scheduler.step()
            
            if step >= swa_start_step:
                if step % 50 == 0:
                    torch.save({
                        'model_state_dict': swa_model.state_dict(),
                        }, os.path.join('./new_saves2', f'{args.task}_seed_{args.train_seed}_{args.learning_rate}_swa_member_{step}.pt'))
            
            del targets  
            del logits   
            del loss   
            
            gc.collect()
            torch.cuda.empty_cache()
            
            step += 1
        
        if step >= swa_start_step:
            swa_scheduler.step()
        
        epoch_valid_loss, epoch_valid_metric = validate(swa_model, criterion, metric, valid_loader, gpu, task)
        
        if gpu == 0:
            print(f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} | metric: {epoch_valid_metric:.4f}%")   
        
def main_worker(rank,
                world_size,
                config, 
                train_data,
                valid_data):
    
    utils.set_seed(config.train_seed)
    dist.init_process_group(backend='nccl', init_method=config.dist_url,
                            world_size=world_size, rank=rank) 
    
    model = RobertaGLUE_base(config)
    
    torch.cuda.set_device(rank)
    model.cuda(rank)
    
    config.batch_size = int(config.batch_size / world_size) #if ~(config.debug) else 1
    config.num_workers = int(config.num_workers / world_size) 

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = torch.nn.CrossEntropyLoss() if config.task != 'stsb' else torch.nn.MSELoss()
    metric = utils.metrics(config.task)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers,
                              sampler=train_sampler,
                              worker_init_fn=utils.seed_worker,
                              shuffle=False, 
                              drop_last=False)
    
    valid_loader = DataLoader(valid_data,
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers,
                              shuffle=False, 
                              drop_last=False)

    total_steps = int(len(train_loader)*config.n_epochs)
    warmup_steps = int(total_steps*config.warmup_ratio)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay
            },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
            }]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=config.learning_rate,
                        eps=1e-06,
                        betas=(0.9,0.98))
    
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps
        )
    
    best_step = train_iter(config.n_epochs,
                           model,
                           train_loader,
                           valid_loader,
                           config.task,
                           criterion,
                           metric,
                           optimizer,
                           lr_scheduler,
                           rank,
                           config)
                        
    del model
    del optimizer
    del lr_scheduler
    
    utils.set_seed(config.train_seed)
    
    swa_model = RobertaGLUE_base(config)
    
    torch.cuda.set_device(rank)
    swa_model.cuda(rank)

    swa_model = torch.nn.parallel.DistributedDataParallel(swa_model, device_ids=[rank])
    criterion = torch.nn.CrossEntropyLoss() if config.task != 'stsb' else torch.nn.MSELoss()
    metric = utils.metrics(config.task)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers,
                              sampler=train_sampler,
                              worker_init_fn=utils.seed_worker,
                              shuffle=False, 
                              drop_last=False)
    
    valid_loader = DataLoader(valid_data,
                              batch_size=16 if config.task not in large_task else 32, 
                              num_workers=config.num_workers,
                              shuffle=False, 
                              drop_last=False)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in swa_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay
            },
        {
            "params": [p for n, p in swa_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
            }]
    
    swa_optimizer = AdamW(optimizer_grouped_parameters,
                        lr=config.learning_rate,
                        eps=1e-06,
                        betas=(0.9,0.98))
    
    swa_lr_scheduler = get_constant_schedule_with_warmup(
        swa_optimizer,
        num_warmup_steps=warmup_steps
        )
    
    collect_ckpt(best_step,
                 train_loader,
                 valid_loader,
                 swa_optimizer,
                 swa_model,
                 swa_lr_scheduler,
                 criterion,
                 metric,
                 rank,
                 config.task,
                 config.n_epochs,
                 config
                 )
    
    
    
def train(config,
          train_data,
          valid_data,
          batch_size,
          learning_rate):          
    
    config.batch_size = int(batch_size)
    config.learning_rate = float(learning_rate)
    #config.learning_rate = learning_rate
    
    config.num_workers = 4
    config.world_size = torch.cuda.device_count()
    
    port = find_free_port()
    config.dist_url = f'tcp://localhost:{port}'
    
    mp.spawn(main_worker, nprocs=config.world_size, args=(config.world_size,
                                                          config,
                                                          train_data,
                                                          valid_data))


def get_arg_parser():

    parser = argparse.ArgumentParser(description="outer bo glue")
    
    parser.add_argument('--task', type=str, default='rte')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    
    parser.add_argument('--train_seed', type=int, default=0)
    
    parser.add_argument('--model_type', type=str, default='roberta-base')
    parser.add_argument('--classifier_dropout', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=768)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dist_url', type=str)
    parser.add_argument('--freeze_num', type=int, default=0)

    parser.add_argument('--debug', type=bool, default=False)
    
    return parser 

def main():
    
    parser = get_arg_parser()
    args = parser.parse_args()
    
    utils.set_seed(args.train_seed)
    utils.configure_cudnn(False)
    
    print("Loading datasets...")
    train_data = torch.load(f'./data/{args.task}_train.pt')
    valid_data = torch.load(f'./data/{args.task}_validation_matched.pt') if args.task == 'mnli' else\
        torch.load(f'./data/{args.task}_validation.pt')
    
    if args.task in large_task:
        args.n_epochs = 10
        
    if args.task == 'rte':
        learning_rate = 2e-05
        batch_size = 20
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
        
        #full
        #learning_rate = 5.424307528301142e-05
        #batch_size = 16.0
        #train(args, train_data, valid_data, 
        #      batch_size, learning_rate)
        
        #freeze
        #learning_rate = 2.7979966034763493e-05
        #batch_size = 9.533492088317871
        #train(args, train_data, valid_data, 
        #      batch_size, learning_rate)
        
    
    elif args.task == 'mrpc':
        #learning_rate = 1.5311752576963045e-05
        #batch_size = 9.37967300415039
        learning_rate = 3.892972017638385e-05
        batch_size = 14
        # full
        #learning_rate = 3.523879422573373e-05
        #batch_size = 11.705581665039062
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
        
        # freeze
        '''
        learning_rate = 8.895499922800809e-05
        batch_size = 15.76117992401123
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
        '''
        
    elif args.task == 'mnli':
        # full
        learning_rate = 3e-06
        batch_size = 16
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
        
        # freeze
        learning_rate = 4e-06
        batch_size = 16
        train(args, train_data, valid_data, 
              batch_size, learning_rate)

    elif args.task == 'sst2':
        # full
        #learning_rate = 3.999999989900971e-06
        #batch_size = 18
        learning_rate = 2.638844853208866e-05
        batch_size = 26.92575454711914
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
        
        # freeze
        '''
        learning_rate  = 3.081153772654943e-05
        batch_size = 52.29051971435547
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
        '''
        
    elif args.task == 'qnli':
        # full
        '''
        learning_rate = 8.938895007304382e-06
        batch_size = 36.2249755859375
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
        '''
        # freeze
        learning_rate = 3.2198706321651116e-05
        batch_size = 63.618507385253906
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
    
    elif args.task == 'qqp':
        # full
        learning_rate = 1e-05
        batch_size = 31
        train(args, train_data, valid_data, 
              batch_size, learning_rate)
          
        # freeze
        #learning_rate = 3.636758265201934e-05
        #batch_size = 52
        #train(args, train_data, valid_data, 
        #      batch_size, learning_rate)
        
        
if __name__ == '__main__':
    main()