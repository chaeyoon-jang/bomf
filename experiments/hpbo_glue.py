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
from src import utils, glue_utils

import gc
import os
import json
import time
import random
import socket
import logging
import warnings
import datetime
import argparse
import numpy as np
from functools import partial

import torch
from torch import cuda
from torch.backends import cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import Dataset,DataLoader

from transformers import (AdamW, get_constant_schedule_with_warmup,
                          RobertaTokenizer)

from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.logei import qLogExpectedImprovement

# If an error occurs, run the following code.  
warnings.filterwarnings(action='ignore')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def generate_initial_points(bounds, n_points):
    lower_bounds, upper_bounds = bounds[0], bounds[1]
    scale = upper_bounds - lower_bounds
    points = lower_bounds + torch.rand((n_points, bounds.size(1))) * scale
    return points  


def validate(
    model,
    criterion,
    metric,
    valid_loader,
    task,
    device
    ):
    
    model.eval()
    with torch.no_grad():
        valid_loss, valid_metric = 0.0, 0.0
        for data in valid_loader:
            
            input_ids = data['input_ids'].cuda(gpu)
            attention_mask = data['attention_mask'].cuda(gpu)
            
            logits = model(input_ids, attention_mask)
            targets = data['label'].to(torch.int64).cuda(device)
                
            if criterion:
                loss = criterion(logits, targets)
                valid_loss += loss.item()
                
            _, m = metric.calculate(logits, targets)
            valid_metric += m  
    
    return valid_loss / len(valid_loader), valid_metric / len(valid_loader)


def train_iter(
    model,
    optimizer,
    lr_scheduler,
    criterion,
    metric,
    n_epochs,
    task,
    train_loader,
    valid_loader,
    gpu
    ):

    best_metric, step = 0.0, 0
    for epoch in range(n_epochs):
        model.train()
        for data in train_loader:
            
            if step == 0:
                start_time = time.time()
                
            input_ids = data['input_ids'].cuda(gpu)
            attention_mask = data['attention_mask'].cuda(gpu)
            
            logits = model(input_ids, attention_mask)
            
            targets = data['label'].to(torch.int64).cuda(gpu)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            lr_scheduler.step()
                
            gc.collect()
            torch.cuda.empty_cache()
            step += 1
        
        epoch_valid_loss, epoch_valid_metric = validate(model,
                                                        criterion,
                                                        metric,
                                                        valid_loader,
                                                        task,
                                                        gpu)
        if gpu == 0:
            print(
                f"Epoch: {epoch+1} | valid loss: {epoch_valid_loss:.4f} | "
                f"metric: {epoch_valid_metric:.4f}%")
        
    return np.float32(best_metric)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) 
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  


def cleanup():
    dist.destroy_process_group()
    
    
def main_worker(
    rank,
    world_size, 
    conn,
    config, 
    train_data,
    valid_data
    ):
    
    utils.set_seed(config.train_seed)
    dist.init_process_group(backend='nccl', init_method=config.dist_url,
                            world_size=world_size, rank=rank)
    
    model = glue_utils.RobertaGLUE(config)
        
    torch.cuda.set_device(rank)
    model.cuda(rank)
    
    config.batch_size = int(config.batch_size / world_size) 
    config.num_workers = int(config.num_workers / world_size) 
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = torch.nn.CrossEntropyLoss()
    metric = glue_utils.metrics(config.task)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,\
        num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_data,
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers,
                              sampler=train_sampler,
                              worker_init_fn=utils.seed_worker,
                              shuffle=False, 
                              drop_last=False)
    
    valid_loader = DataLoader(valid_data,
                              batch_size=config.valid_batch_size, 
                              num_workers=config.num_workers,
                              shuffle=False, 
                              drop_last=False)

    total_steps = int(len(train_loader)*config.n_epochs)
    warmup_steps = int(total_steps*config.warmup_ratio)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()\
                if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay
            },
        {
            "params": [p for n, p in model.named_parameters()\
                if any(nd in n for nd in no_decay)],
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
    
    best_metric = train_iter(model,
                             optimizer,
                             lr_scheduler,
                             criterion,
                             metric,
                             config.n_epochs,
                             config.task,
                             train_loader,
                             valid_loader,
                             rank)
    conn.send(best_metric)
        
    del model
    del optimizer
    del lr_scheduler
    
    cleanup()
    
    
def train(
    learning_rate, 
    batch_size, 
    train_data,
    valid_data,
    config
    ):          
    
    config.batch_size = int(batch_size)
    config.learning_rate = float(learning_rate)

    config.world_size = torch.cuda.device_count()
    
    port = find_free_port()
    config.dist_url = f'tcp://localhost:{port}'
    
    parent_conn, child_conn = mp.Pipe()
    mp.spawn(main_worker, 
             nprocs=config.world_size, 
             args=(config.world_size,
                   child_conn,
                   config,
                   train_data,
                   valid_data), join=True)
    best_metric = []
    while parent_conn.poll():
        best_metric.append(parent_conn.recv())
    
    gc.collect()
    torch.cuda.empty_cache()
    best_metric = best_metric[0]
    utils.set_seed(config.bo_seed)
    return best_metric


def optimize_hyperparameters(bounds, evaluate_model, config):
    
    train_x = generate_initial_points(bounds, config.init_points)
    train_y = torch.tensor([evaluate_model(learning_rate=lr, batch_size=bs)\
        for lr, bs in train_x])

    for it in range(config.bo_iters):
        
        print(f">>>>>{it+1} iteration:")
                
        gp = SingleTaskGP(train_x, train_y.unsqueeze(-1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        
        acq_func = qLogExpectedImprovement(gp, best_f=train_y.max())

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=config.q,
            num_restarts=config.num_restarts,
            raw_samples=config.raw_samples,
        )

        new_x = candidates.detach()
        new_y = evaluate_model(learning_rate=new_x[..., 0],
                               batch_size=new_x[..., 1])
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, torch.tensor([new_y])])
        
        print(f'>>>>>> Selected learning rate: {train_x[-1][0]}')
        print(f'>>>>>> Selected batch size   : {int(train_x[-1][1])}')
        print(f'>>>>>> Validation Metric     : {train_y[-1]}')
        print('='*30)

    return train_x, train_y


def get_arg_parser():

    parser = argparse.ArgumentParser(description="HPBO")
    
    parser.add_argument('--task', type=str, default='rte')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--freeze_num', type=int, default=6)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--valid_batch_size', type=int, default=16)
    
    parser.add_argument('--bo_seed', type=int, default=0)
    parser.add_argument('--train_seed', type=int, default=0)
    
    parser.add_argument('--model_type', type=str, default='roberta-base')
    parser.add_argument('--classifier_dropout', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=768)
    
    parser.add_argument('--bo_iters', type=int, default=100)
    parser.add_argument('--init_points', type=int, default=1)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--num_restarts', type=int, default=5)
    parser.add_argument('--raw_samples', type=int, default=20)
    
    parser.add_argument('--lr_lower_bound', type=float, default=1e-07)
    parser.add_argument('--lr_upper_bound', type=float, default=1e-03)
    parser.add_argument('--bs_lower_bound', type=int, default=8)
    parser.add_argument('--bs_upper_bound', type=int, default=16)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dist_url', type=str)
    
    return parser 


def main():
    
    parser = get_arg_parser()
    args = parser.parse_args()
    
    utils.set_seed(args.bo_seed) 
    utils.configure_cudnn(False)
    
    bounds = torch.tensor([[args.lr_lower_bound, args.bs_lower_bound],
                           [args.lr_upper_bound, args.bs_upper_bound]],
                          dtype=torch.float32) 
    
    print("Loading datasets...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_type)
    train_data, valid_data = glue_utils.make_glue_data(args.task, tokenizer)
    evaluate_model = partial(train,
                             config=args,
                             train_data=train_data,
                             valid_data=valid_data)
    
    print(f"Starting HPBO with {args.task} task...")
    start_time = time.time()
    optimized_params = optimize_hyperparameters(bounds, evaluate_model, args)
    elapse_time = time.time() - start_time
    elapse_time = datetime.timedelta(seconds=elapse_time)
    
    print(f'Total time cost: {elapse_time}')
    hps, results = optimized_params
    max_value = torch.max(results)
    max_hps = hps[torch.argmax(results)].tolist()
    
    print(f'Max valid metric result: {max_value}')
    print(f'-> Hyperparameters: {max_hps}')
    
    log_data = {
        'cost':f'{elapse_time}',
        'learning_rate':hps[:,0].tolist(),
        'batch_size':hps[:,1].tolist(),
        'metric':results.tolist()
    }
    with open(f'./logs/freeze_{args.freeze_num}_{args.task}_outer_bo.json', 'w')\
        as file:
        json.dump(log_data, file)
    
    
if __name__ == '__main__':
    main()