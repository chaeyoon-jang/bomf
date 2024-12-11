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
from src import utils, squad_utils, glue_utils

import time
import datetime
import argparse
from collections import defaultdict

import os
import torch

import re 
import string  
import datasets 

import json
from collections import Counter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

import random

from torch.backends import cudnn
import numpy as np
from tqdm import tqdm 
from collections import OrderedDict
from torch.utils.data import Dataset


def generate_summary(model, tokenizer, source, target):
  
  output = model.generate(source,
                max_length=10, 
                early_stopping=True)
  
  output = output

  machine_text = [tokenizer.decode(senten, skip_special_tokens=True) for senten in output]
  machine_text = ["replacement" if text == "" else text for text in machine_text]
  return machine_text

device ="cuda" if torch.cuda.is_available() else "cpu"


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
    
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    
    return max(scores_for_ground_truths)


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

    return {'em': exact_match, 'f1': f1}


def evaluate(model,
             tokenizer,
             valid_loader,
             ground_truth,
             device):
    
    total_loss = 0.0
    all_text = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader)):

                source, source_mask, target, _ = torch.tensor(batch['input_ids']).to(device),\
                    batch['attention_mask'].to(device),\
                    batch['labels'], batch['decoder_attention_mask'].to(device) 
                
                target = torch.tensor(target)
                target[target[: ,:] == 0 ] = -100
                target = target.to(device)
                loss = model(source, source_mask, labels=target, return_dict=True)
                loss = loss['loss'].item()
                total_loss += loss
                
                all_text.extend(generate_summary(model, tokenizer, source, target))

    results = qa_metrics(ground_truth, all_text)
                
    return results['em'], results['f1'], total_loss / len(valid_loader)


def load_weights(model, params, names):

    new_s = OrderedDict()
    for idx, p in enumerate(params):
        new_s[names[idx]] = p
    model.load_state_dict(new_s, strict=False)


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


def function(train_X, alpha_model, args):
    
    def tn(seq):
        v = 1/torch.sum(seq, dim=-1)
        seq = seq * v
        return seq

    train_Y_loss = []
    train_Y_metric1 = []
    train_Y_metric2 = []
    
    loss_ = 0.0
    for xi in train_X:
        em, f1, loss = alpha_model(tn(xi.to('cpu')))
        
        print(em)
        print(f1)
        print(loss)
        
        loss_ += loss
        em = em * 0.01
        f1 = f1 * 0.01
        
        train_Y_loss.append(((args.max_loss - loss) - args.min_loss)/(args.max_loss - args.min_loss))
        train_Y_metric1.append((em - args.min_em)/(args.max_em - args.min_em))
        train_Y_metric2.append((f1 - args.min_f1)/(args.max_f1 - args.min_f1))
    
    train_Y_loss = torch.Tensor(train_Y_loss)
    train_Y_loss = train_Y_loss.reshape(1, -1)
    
    train_Y_metric1 = torch.Tensor(train_Y_metric1)
    train_Y_metric1 = train_Y_metric1.reshape(1, -1)

    train_Y_metric2 = torch.Tensor(train_Y_metric2)
    train_Y_metric2 = train_Y_metric2.reshape(1, -1)
    
    return torch.cat((train_Y_metric1, train_Y_metric2, train_Y_loss)).transpose(0,1), loss_/len(train_X)


def get_arg_parser():

    parser = argparse.ArgumentParser(description="BOMF")
    
    parser.add_argument('--seed', '-s', type=int, default=0)
        
    parser.add_argument('--min_em', type=float, required=True)
    parser.add_argument('--max_em', type=float, required=True)
    parser.add_argument('--min_f1', type=float, required=True)
    parser.add_argument('--max_f1', type=float, required=True)
    parser.add_argument('--max_loss', type=float, required=True)
    parser.add_argument('--max_loss', type=float, required=True)
    parser.add_argument('--niters', type=int, required=True)
    
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
        
    return parser
             
def main():

    parser = get_arg_parser()
    args = parser.parse_args()
    
    utils.set_seed(args.seed)
    utils.configure_cudnn(False)
    
    model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
    
    names = []
    for k, p in model.state_dict().items():
        names.append(k) 
        
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    valid_ds = SquadV2Dataset(tokenizer, 512, split='validation')
    valid_loader = torch.utils.data.DataLoader(dataset= valid_ds, batch_size=32, shuffle=False)
    data = datasets.load_dataset('squad_v2', split='validation')['answers']
    ground_truth = []
    for ro in data:
        if ro['text'] == []:
            temp = ["replacement"]
        else:
            temp = ro['text']
        ground_truth.append(temp)
        
    fl = os.listdir(args.base_path)
    nfl = []
    for f in fl:
        if f'{args.key}' in f:
            nfl.append(f)
    fl = nfl  
    
    ckpt_list = []
    for f in fl:
        t = torch.load(os.path.join(args.base_path, f), map_location='cpu')
        ckpt_list.append(t)
    
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    sds = [glue_utils.clean_ckpt(m) for m in ckpt_list]
    new_sds = [OrderedDict() for _ in range(len(sds))]
    for idx in range(len(sds)):
        for n, v in sds[idx].items():
            new_sds[idx][n] = v
        
    paramslist = [tuple(v.detach() for _, v in sd.items()) for i, sd in enumerate(new_sds)]
    
    class AlphaWrapper(torch.nn.Module):
        def __init__(self, paramslist, 
                    model, names, 
                    dataloader, device,ground):
            super(AlphaWrapper, self).__init__()
            self.paramslist = paramslist
            self.model = model 
            self.names = names 
            self.dataloader = dataloader
            self.device = device
            self.ground = ground  
            
        def forward(self, train_X):
            alph = train_X
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, alph.cpu())))\
                for j, p in enumerate(zip(*self.paramslist)))
            params = tuple(p.cuda(0) for p in params)
            load_weights(self.model, params, self.names)
            em, f1, loss = evaluate(
                self.model,
                tokenizer,
                self.dataloader,
                self.ground,
                self.device
            )
            return em, f1, loss
    
    alpha_model = AlphaWrapper(paramslist, model, names, valid_loader, device, ground_truth)

    def generate_initial_data(n=6):
        train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
        train_obj, _ = function(train_x, alpha_model, args)
        train_x = train_x.to(**tkwargs)
        train_obj = train_obj.to(**tkwargs)
        return train_x, train_obj

    def initialize_model(train_x, train_obj, tkwargs):
        train_x = normalize(train_x, bounds).type(torch.float32)
        train_x = train_x.to(**tkwargs)
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]
            models.append(
                SingleTaskGP(
                    train_x, train_y, outcome_transform=Standardize(m=1)
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    
    def tn(seq):
        v = 1/torch.sum(seq, dim=-1)
        seq = seq * v
        return seq
    
    set_seed(0)
    configure_cudnn(False)
    
    tkwargs = { 
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    SMOKE_TEST = os.environ.get("SMOKE_TEST")
    bounds = torch.Tensor([[0]*len(ckpt_list), [1]*len(ckpt_list)]).to(**tkwargs)
    ref_point = torch.tensor([0, 0, 0]).to(**tkwargs)
    
    BATCH_SIZE = 1
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4

    standard_bounds = torch.zeros(2, len(ckpt_list), **tkwargs)
    standard_bounds[0] = 0
    standard_bounds[1] = 1
    
    MC_SAMPLES = 128 if not SMOKE_TEST else 16

    hvs_qnehvi= []
    train_x, train_obj = generate_initial_data(n=5)
    mll_qnehvi, model_qnehvi = initialize_model(train_x, train_obj, tkwargs) 
    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
    volume = bd.compute_hypervolume().item()

    hvs_qnehvi.append(volume)
    
    def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),  
            X_baseline=normalize(train_x, bounds),
            prune_baseline=True,  
            sampler=sampler,
        )
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        new_x = unnormalize(candidates.detach(), bounds=bounds)
        new_obj, loss = function(new_x, alpha_model)
        new_obj = new_obj.to(**tkwargs)
        return new_x, new_obj, loss
    
    all_em = []
    all_f1 = []
    all_loss = []
    best_metric = 0.0
    for _ in tqdm(range(1, args.niters)):

        fit_gpytorch_mll(mll_qnehvi)
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        new_x, new_obj, loss_ = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x, train_obj, qnehvi_sampler
            )
        train_x = torch.cat([train_x, new_x])
        train_obj= torch.cat([train_obj, new_obj])

        mll_qnehvi, model_qnehvi = initialize_model(train_x, train_obj, tkwargs)
        
        cur_metric1 = train_obj[:,1].max()*(args.max_em - args.min_em) + args.min_em
        cur_metric2 = train_obj[:,2].max()*(args.max_f1 - args.min_f1) + args.min_f1
        total_metric = cur_metric1 + cur_metric2
        
        all_em.append(cur_metric1.item())
        all_f1.append(cur_metric2.item())
        all_loss.append(loss_)
        
        if best_metric < total_metric:
            best_metric = total_metric
    
    data = {
        'em':all_em,
        'f1':all_f1,
        'loss':all_loss
    }
    
    with open('squad_bomf_results.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
if __name__ =="__main__":
    main()