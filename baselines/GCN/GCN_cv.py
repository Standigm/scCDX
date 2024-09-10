import copy
import os
import random
import tomlkit
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from itertools import product
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from src.utils import load_data, norm_sys
from model import EMOGINet, MTGCNNet, HGDCNet


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')

def setup(method, input_dim, device, **kwargs): #FIXME
    if "MTGCN" in method:
        model = MTGCNNet(input_dim, hid_dim1=kwargs["hid_dim1"], hid_dim2=kwargs["hid_dim2"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["learning_rate"])#0.001)
        EPOCH = 2500
        
    elif "EMOGI" in method:
        model = EMOGINet(input_dim, hid_dim1=kwargs["hid_dim1"], hid_dim2=kwargs["hid_dim2"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["learning_rate"], weight_decay=kwargs["weight_decay"])#lr=0.001, wd=0.005
        EPOCH = 2000
        
    elif "HGDC" in method:
        model = HGDCNet(input_dim, kwargs["hid_dim"]).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.linear1.parameters(), weight_decay=kwargs["weight_decay"]),
            dict(params=model.linear_r0.parameters(), weight_decay=kwargs["weight_decay"]),
            dict(params=model.linear_r1.parameters(), weight_decay=kwargs["weight_decay"]),
            dict(params=model.linear_r2.parameters(), weight_decay=kwargs["weight_decay"]),
            dict(params=model.linear_r3.parameters(), weight_decay=kwargs["weight_decay"]),
            dict(params=model.weight_r0, lr=kwargs["learning_rate"] * 0.1),
            dict(params=model.weight_r1, lr=kwargs["learning_rate"] * 0.1),
            dict(params=model.weight_r2, lr=kwargs["learning_rate"] * 0.1),
            dict(params=model.weight_r3, lr=kwargs["learning_rate"] * 0.1)
        ], lr=kwargs["learning_rate"])
        EPOCH = 100
        
    return model, optimizer, EPOCH
        

def train_epoch(model, data, mask, optimizer, method="MTGCN"):
    model.train()
    optimizer.zero_grad()

    if "MTGCN" in method:
        pred, rl, c1, c2 = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(pred[mask], data.y[mask]) / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)
    
    elif "EMOGI" in method:
        pred = model(data)
        loss = F.binary_cross_entropy_with_logits(pred[mask], data.y[mask], weight=torch.tensor([45.]).to(data.x.device))
    
    elif "HGDC" in method:
        pred = model(data)
        loss = F.binary_cross_entropy_with_logits(pred[mask], data.y[mask].view(-1, 1))
    
    loss.backward()
    optimizer.step()


@torch.no_grad()
def eval(model, data, mask, method="MTGCN"):
    model.eval()
    if "MTGCN" in method:
        pred, _, _, _ = model(data.x, data.edge_index)
    else:
        pred = model(data)

    pred = torch.sigmoid(pred[mask]).cpu().detach().numpy()
    Yn = data.y[mask].cpu().numpy()
    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
    perf = {
        "AUC": metrics.roc_auc_score(Yn, pred),
        "AUPR": metrics.auc(recall, precision),
        "AveP": metrics.average_precision_score(Yn, pred),
    }
    
    return None, perf#metrics.roc_auc_score(Yn, pred), area, ap


@torch.no_grad()
def eval_independent(model, data, method="MTGCN"):
    model.eval()
    if "MTGCN" in method:
        out, _, _, _ = model(data.x, data.edge_index)
    else:
        out = model(data)

    return torch.sigmoid(out).cpu().detach().numpy()


def set_seed(seed: int | None):
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load(args, seed):
    if args.method in ["EMOGI", "MTGCN"]:
        data, genes, _ = load_data(omics=args.omics, 
                                   experiment=args.method + ("_selection_all" if "scGNN" in args.omics else ""), 
                                   seed=seed)
        if "SYS" in args.omics: 
            norm_sys(args, args.method + ("_selection_all" if "scGNN" in args.omics else ""), data)
            
    elif args.method == "HGDC":
        data, genes, _ = load_data(omics=args.omics, 
                                   experiment=args.method + ("_selection_all" if "scGNN" in args.omics else ""),
                                   seed=seed)
        data.x = torch.FloatTensor(StandardScaler().fit_transform(data.x.numpy()))
        
        gdc = T.GDC(self_loop_weight=None, normalization_in='sym', normalization_out='col',
                    diffusion_kwargs=dict(method="ppr", alpha=0.9, eps=0.0001),
                    sparsification_kwargs=dict(method='threshold', avg_degree=111),
                    exact=True)
        data_aux = gdc(data.clone())
        data.edge_index_aux = data_aux.edge_index
        
    print(data.x.shape, data.y.sum(), (data.train_mask | data.val_mask | data.test_mask).sum(), (data.independent_mask.sum()))
        
    return data, genes


def train(args, params_grid):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    i, j = args.seed // 10000, args.seed % 10000
    if os.path.exists(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/AUC.txt"):
        AUC = np.loadtxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/AUC.txt", delimiter="\t")
        if AUC[i][j] != 0:
            return None

    set_seed(args.seed)
    # seed_everything(seed)

    data, _ = load(args, args.seed)
    data = data.to(device)
    
    best_avep, best_params = 0., {}
    log_dict = {}
    for k_th, params in tqdm(enumerate(list(product(*list(params_grid.values()))))):
        param_dict = {param: params[i] for i, param in enumerate(params_grid.keys())}

        model, optimizer, EPOCH = setup(args.method, input_dim=data.x.shape[1], device=device, **param_dict)
        for epoch in trange(EPOCH):
            train_epoch(model, data, data.train_mask, optimizer, args.method)
        _, perf = eval(model, data, data.val_mask, args.method)

        if perf["AveP"] > best_avep:
            best_avep = perf["AveP"]
            best_params = copy.deepcopy(param_dict)

        log_dict.update({f"{k_th}-th hparams": param_dict | perf})

    with open(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/logs/log_{args.seed}.toml", "w") as f:
        tomlkit.dump(log_dict, f)


    model, optimizer, EPOCH = setup(args.method, input_dim=data.x.shape[1], device=device, **best_params)
    for epoch in trange(EPOCH):
        train_epoch(model, data, data.train_mask | data.val_mask, optimizer, args.method)
    _, perf = eval(model, data, data.test_mask, args.method)

    if os.path.exists(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/AUC.txt"):
        result = {
            metric: np.loadtxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/{metric}.txt", delimiter="\t")
            for metric in ["AUC", "AUPR", "AveP"]
        }
    else:
        result = {metric: np.zeros(shape=(10, 5)) for metric in ["AUC", "AUPR", "AveP"]}
    
    #save
    for metric in ["AUC", "AUPR", "AveP"]:
        result[metric][i, j] = perf[metric]
        np.savetxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/{metric}.txt", result[metric], delimiter="\t")

    if (result["AUC"] == 0).sum() == 0:
        result = {"Mean": {metric: result[metric].mean().round(6) for metric in result.keys()},
                  "Std": {metric: result[metric].std().round(6) for metric in result.keys()}}
        with open(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/performance.toml", "w") as f:
            tomlkit.dump(result, f)
    

def test(args, params_grid):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/preds/pred_{args.seed}.txt"):
        return None
    set_seed(args.seed)
    
    data, _ = load(args, args.seed)
    data.to(device)

    best_avep, best_params = 0., {}
    for params in tqdm(list(product(*list(params_grid.values())))):
        param_dict = {param: params[i] for i, param in enumerate(params_grid.keys())}

        model, optimizer, EPOCH = setup(args.method, input_dim=data.x.shape[1], device=device, **param_dict)
        for epoch in trange(EPOCH):
            train_epoch(model, data, data.train_mask | data.val_mask, optimizer, args.method)
        _, perf = eval(model, data, data.test_mask, args.method)

        if perf["AveP"] > best_avep:
            best_avep = perf["AveP"]
            best_params = copy.deepcopy(param_dict)
        

    # independent
    model, optimizer, EPOCH = setup(args.method, input_dim=data.x.shape[1], device=device, **best_params)
    for epoch in trange(EPOCH):
        train_epoch(model, data, data.train_mask | data.val_mask | data.test_mask, optimizer, args.method)

    pred = eval_independent(model, data, args.method)
    np.savetxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/preds/pred_{args.seed}.txt", pred, delimiter="\t")       



if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    parser = ArgumentParser()
    parser.add_argument("--ppi", default="CPDB_v34", type=str)
    parser.add_argument("--omics", default="MF+METH+GE+SYS+TOPO", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--method", default="MTGCN", type=str, choices=["EMOGI", "MTGCN", "HGDC"])
    args = parser.parse_args()
    
    os.makedirs(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/preds", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/logs", exist_ok=True)
    
    if args.method == "MTGCN":
        params_grid = {
            "hid_dim1": [300, 500] if "sc" in args.omics else [150, 300, 500],
            "hid_dim2": [100, 200] if "sc" in args.omics else [50, 100, 200],
            "learning_rate": [0.001, 0.002, 0.0005],
        }
    elif args.method == "EMOGI":
        params_grid = {
            "hid_dim1": [300, 500] if "sc" in args.omics else [150, 300, 500],
            "hid_dim2": [100, 200] if "sc" in args.omics else [50, 100, 200],
            "learning_rate": [0.001, 0.002, 0.0005],
            "weight_decay": [0.005, 0.001, 0.01],
        }
    elif args.method == "HGDC":
        params_grid = {
            "learning_rate": [0.001, 0.002, 0.0005],
            "hid_dim": [100, 200, 300] if "sc" in args.omics else [50, 100, 200, 300],
            "weight_decay": [0.00001, 0.0001, 0.001],
        }

    train(args, params_grid)
    test(args, params_grid)