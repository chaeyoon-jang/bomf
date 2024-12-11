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

import os
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm 
from functools import partial
from collections import OrderedDict

import torch
from easydict import EasyDict
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from torch.utils.data import DataLoader

# If an error occurs, run the following code.  
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def simplex_normalize(seq):
    v = 1/torch.sum(seq, dim=-1)
    seq = seq * v
    return seq


def simplex_proj(Y):
    D = Y.shape[0]
    X = np.sort(Y)[::-1] 
    X_cumsum = np.cumsum(X)
    X_tmp = (X_cumsum - 1) / np.arange(1, D + 1)
    j = np.sum(X > X_tmp) - 1
    threshold = X_tmp[j]
    return np.maximum(Y - threshold, 0)


def tfunction(train_X, alpha_model, min_m1, max_m1, min_m2, max_m2):
        
    train_Y_metric1 = []
    train_Y_metric2 = []
    
    for xi in train_X:
        m1, m2 = alpha_model(simplex_normalize(xi.to('cpu')))
        train_Y_metric1.append((m1 - min_m1)/(max_m1 - min_m1))
        train_Y_metric2.append(((max_m2 - m2)-min_m2)/(max_m2 - min_m2))
    
    train_Y_metric1 = torch.Tensor(train_Y_metric1).reshape(1, -1)
    train_Y_metric2 = torch.Tensor(train_Y_metric2).reshape(1, -1)
    
    r = torch.cat((train_Y_metric1, train_Y_metric2)).transpose(0,1)
    return r


def make_env(task, seed):

    num_classes = 1 if task == 'stsb' else 2
    
    config = {}
    config['model_type'] = 'roberta-base'
    config['classifier_dropout'] = 0.1
    config['num_labels'] = num_classes
    config['freeze_num']  = 0
    
    config = EasyDict(config)
    model = glue_utils.RobertaGLUE(config)
    
    ckpt_list = []
    path = args.base_path
    file_list = os.listdir(path)
    
    new_file_list = []
    for fn in file_list:
        if args.key in fn:
            if args.task in fn:
                new_file_list.append(fn)
    
    for fn in new_file_list:
        ckpt_list.append(torch.load(os.path.join(path, fn), map_location='cpu'))
        
    print(len(ckpt_list))

    data = datasets.load_dataset('glue', task)
    data_loader = make_dataloader(
        32,
        data, num_classes,
        task, tokenizer,
        'max_length', 512
    )
    metric = metrics(task)
    
    return model, ckpt_list, data_loader, metric
    
    
def get_arg_parser():

    parser = argparse.ArgumentParser(description="BOMF")
    
    parser.add_argument('--task', '-t', type=str, default='rte')
    parser.add_argument('--seed', '-s', type=int, default=0)
    
    parser.add_argument('--max_m1', type=float, required=True)
    parser.add_argument('--min_m1', type=float, required=True)
    parser.add_argument('--max_m2', type=float, required=True)
    parser.add_argument('--min_m2', type=float, required=True)
    parser.add_argument('--n_iters', type=int, default=50, required=True)
    
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
        
    return parser


def main():
    
    parser = get_arg_parser()
    args = parser.parse_args()
    
    set_seed(args.seed)
    configure_cudnn(False)
    
    tkwargs = { 
               "dtype": torch.double,
               "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    SMOKE_TEST = os.environ.get("SMOKE_TEST")
    
    struct, ckpt_list, valid_loader, metric = make_env(args.task, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = [glue_utils.clean_ckpt(m, 0) for m in ckpt_list]

    new_sds = [OrderedDict() for _ in range(len(sds))]
    for idx in range(len(sds)):
        for n, v in sds[idx].items():
            if 'position_ids' not in n:
                new_sds[idx][n] = v

    paramslist = [tuple(v.detach() for _, v in sd.items()) for i, sd in enumerate(new_sds)]
    _, names = make_functional(struct)
    
    alpha_model = AlphaWrapper(paramslist, struct, names, valid_loader, device, metric)
    bounds = torch.Tensor([[0]*len(ckpt_list), [1]*len(ckpt_list)]).to(**tkwargs)
    
    ref_point = torch.tensor([0.0, 0.0]).to(**tkwargs)
        
    function = partial(tfunction, 
                       min_m1=args.min_m1, max_m1=args.max_m1,
                       min_m2=args.min_m2, max_m2=args.max_m2)
    
    def generate_initial_data(n=6):
        train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
        train_obj = function(train_x, alpha_model)
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
    
    BATCH_SIZE = 1
    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4

    standard_bounds = torch.zeros(2, len(ckpt_list), **tkwargs)
    standard_bounds[0] = 0
    standard_bounds[1] = 1
    
    MC_SAMPLES = 128 if not SMOKE_TEST else 16
    
    hvs_qnehvi= []
    print(len(ckpt_list))
    
    train_x, train_obj = generate_initial_data(n=len(ckpt_list))
    mll_qnehvi, model_qnehvi = initialize_model(train_x, train_obj, tkwargs) 
    
    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
    volume = bd.compute_hypervolume().item()
    hvs_qnehvi.append(volume)
    
    def optimize_qnehvi_and_get_observation(model, train_x, sampler):
        
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
        new_obj = function(new_x, alpha_model)
        new_obj = new_obj.to(**tkwargs)
        return new_x, new_obj

    best_metric = 0.0 
    for _ in tqdm(range(1, args.niter)):

        fit_gpytorch_mll(mll_qnehvi)
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        (
            new_x,
            new_obj
        ) = optimize_qnehvi_and_get_observation(
            model_qnehvi, train_x, qnehvi_sampler
        )
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        
        bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
        volume = bd.compute_hypervolume().item()
        hvs_qnehvi.append(volume)
                    
        mll_qnehvi, model_qnehvi = initialize_model(train_x, train_obj, tkwargs)
        
        cur_metric = train_obj[:,0].max()*(args.max_m1 - args.min_m1) + args.min_m1
        if best_metric < cur_metric:
            best_metric = cur_metric
            print(f"Current Best Metric: {best_metric:.4f}")
        
    print('='*30)
    print(f"Final Best Metric: {best_metric:.4f}") 


if __name__ == '__main__':
    main()