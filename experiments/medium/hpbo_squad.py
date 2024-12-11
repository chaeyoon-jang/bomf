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
from src import utils, squad_utils 

import time
import datetime
import argparse

import gc
import os
import json
import torch
from torch import cuda

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import socket
import datasets
from datasets import load_metric
from dataset import Xsum_Dataset, make_dataloader

from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import logging
import warnings

import random
import os
import torch
from torch.backends import cudnn
import numpy as np
from functools import partial
from torch.utils.data import Dataset
import ipdb

warnings.filterwarnings(action='ignore')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def cleanup():
    dist.destroy_process_group()

    
def main_worker(rank, world_size, conn, args):

    utils.set_seed(args.train_seed)
    utils.configure_cudnn(False)

    os.makedirs(args.ckpt_path, exist_ok=True)
    
    print("Use GPU: {} for training...".format(rank))

    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=world_size, rank=rank)
    
    args.batch_size = int(args.batch_size / world_size)
    args.num_worker = int(args.num_workers / world_size)

    metric = load_metric(args.metric)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    
    if args.freeze_num > 0:
        top_layers = len(model.encoder.block)
        for layers in model.encoder.block[top_layers-6:]:
            for param in layers.parameters():
                param.requires_grad = False    
                
    cuda.set_device(rank)
    model.cuda(rank)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()\
                    if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in model.named_parameters()\
                    if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.lr)
    
    wikitokenizer = T5Tokenizer.from_pretrained(args.model, use_fast=True)

    class SquadV2Dataset(Dataset):
        def __init__(self, tokenizer, max_length, split):
            self.dataset = datasets.load_dataset('squad_v2', split=split)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            context = item['context']
            question = item['question']
            answers = item['answers']['text'][0] if item['answers']['text'] else ''

            input_text = f"question: {question} context: {context}"
            answer_text = answers

            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            answer_encoding = self.tokenizer(
                answer_text,
                max_length=32,  
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': answer_encoding['input_ids'].squeeze(),
                'decoder_attention_mask': answer_encoding['attention_mask'].squeeze()
            }
    
    train_ds = SquadV2Dataset(wikitokenizer, 512, split='train')
    valid_ds = SquadV2Dataset(wikitokenizer, 512, split='validation')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, 
                                               batch_size=args.batch_size, 
                                               sampler=train_sampler, 
                                               worker_init_fn=utils.seed_worker)
    valid_loader = torch.utils.data.DataLoader(dataset= valid_ds, batch_size=32)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * 0.1),
        num_training_steps=int(len(train_loader) * args.epoch))
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters of model is {}...".format(num_params))
        
    epoch_start = time.time()
    best_m = 0.0
    for ep in range(args.epoch):
        epoch_loss, epoch_m = squad_utils.train_fn(model,
                                                   wikitokenizer,  
                                                   optimizer,
                                                   scheduler,
                                                   train_loader, 
                                                   valid_loader,
                                                   metric,
                                                   2,
                                                   rank)
                      
        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        
        if rank == 0:   
            print(f"Epoch: {ep+1} | train loss: {epoch_loss:.4f} | "
                  f"valid metrics: {epoch_rouge:.4f}% | time: {elapse_time}")

        if epoch_m > best_m:
            best_m = epoch_m
 
    conn.send(best_m)
    
    del model
    del optimizer
    del scheduler
    
    cleanup()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) 
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  
    
    
def train(learning_rate,
          batch_size,
          args):
    
    args.learning_rate = learning_rate
    args.batch_size = int(batch_size)

    args.world_size = torch.cuda.device_count()
    
    port = find_free_port()
    args.dist_url = f'tcp://localhost:{port}'
    
    parent_conn, child_conn = mp.Pipe()
    mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, child_conn, args), join=True)
    
    best_metric = []
    while parent_conn.poll():
        best_metric.append(parent_conn.recv())
    
    gc.collect()
    torch.cuda.empty_cache()
    best_metric = best_metric[0]
    utils.set_seed(args.bo_seed)
    return torch.tensor(best_metric, dtype=torch.float32)


def generate_initial_points(bounds, n_points=3):
    lower_bounds, upper_bounds = bounds[0], bounds[1]
    scale = upper_bounds - lower_bounds
    points = lower_bounds + torch.rand((n_points, bounds.size(1))) * scale
    return points  


def optimize_hyperparameters(bounds, evaluate_model, config):
    
    train_x = generate_initial_points(bounds)
    train_y = torch.tensor([evaluate_model(learning_rate=lr, batch_size=bs) for lr, bs in train_x])

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
        new_y = evaluate_model(learning_rate=new_x[..., 0], batch_size=new_x[..., 1])
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, torch.tensor([new_y])])
        
        print(f'>>>>>> Selected learning rate: {train_x[-1][0]}')
        print(f'>>>>>> Selected batch size   : {int(train_x[-1][1])}')
        print(f'>>>>>> Validation Metric     : {train_y[-1]}')
        print('='*30)

    return train_x, train_y


def get_arg_parser():
    parser =  argparse.ArgumentParser(description="HPBO")
    
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--model", type=str, default='t5-base')
    
    parser.add_argument('--bo_seed', type=int, default=None)
    parser.add_argument('--train_seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://localhost:32465', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    
    parser.add_argument('--ckpt_path', type=str, default="./ckpt")
    parser.add_argument('--swa_start', type=int, default=None)
    parser.add_argument('--bo_iters', type=int, default=None)
    parser.add_argument('--q', type=int, default=None)
    parser.add_argument('--num_restarts', type=int, default=5)
    parser.add_argument('--raw_samples', type=int, default=20)

    parser.add_argument('--lr_lower_bound', type=float, default=1e-07)
    parser.add_argument('--lr_upper_bound', type=float, default=1e-03)
    parser.add_argument('--bs_lower_bound', type=int, default=8)
    parser.add_argument('--bs_upper_bound', type=int, default=16)
    
    parser.add_argument('--freeze_num', type=int, default=0)
    
    return parser


def main():
    
    parser = get_arg_parser()
    args = parser.parse_args()
    
    utils.set_seed(args.bo_seed)
    
    bounds = torch.tensor([[args.lr_lower_bound, args.bs_lower_bound],
                           [args.lr_upper_bound, args.bs_upper_bound]],
                          dtype=torch.float32) 
    evaluate_model = partial(train, args=args)
    
    print(f"Starting HPBO with SQuAD...")
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
    with open(f'./logs/freeze_{args.freeze_num}_squad_outer_bo.json', 'w') as file:
        json.dump(log_data, file)


if __name__ =="__main__":
    main()