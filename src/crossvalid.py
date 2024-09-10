import copy
import numpy as np
import os
import sys
import tomlkit
from argparse import ArgumentParser, Namespace
from itertools import product
from tqdm import tqdm, trange

import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data

from src.model import setup
from src.utils import load_data, set_seed, norm_sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_step(loader, model, device, optimizer, criterion):
    model.train()

    for batch in loader:
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss#, lr


@torch.no_grad()
def eval_step(loader, model, device, criterion=None):
    model.eval()

    out, ys = [], []
    for batch in loader:
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out.append(model(x))
        ys.append(y)
    out = torch.cat(out)
    ys = torch.cat(ys)
    
    loss = criterion(out, ys).item() if criterion is not None else None
        
    return out, loss


def fit_clf(clf, data, mask, param_dict):
    if args.model in ["rf", "svm"]:
        clf.fit(data.x[mask].numpy(), data.y[mask].numpy())
    elif args.model == "xgb":
        clf.fit(data.x[mask].numpy(), data.y[mask].numpy(), verbose=False)
    elif args.model == "tabnet":
        clf.fit(x[mask].numpy(), y[mask].numpy(), 
                max_epochs=param_dict["epoch"], patience=0, weights=1,
                loss_fn=nn.CrossEntropyLoss(), batch_size=512, num_workers=4)
        
    return clf


def eval_perf(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    perfs = {}
    perfs["AUPR"] = auc(recall, precision)
    perfs["AveP"] = average_precision_score(y_true, y_pred)
    perfs["AUC"] = roc_auc_score(y_true, y_pred)
    
    return perfs


def train_mlp(args):
    # sets seeds for numpy, torch and python.random
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "gpu" else "cpu")
    
    # setup data
    data, _, _ = load_data(args.omics, args.model, seed=args.seed)
    if "SYS" in args.omics:
        norm_sys(args, args.model, data)
    data.y_orig = data.y.detach().clone()
    
    if args.mode == "train":
        train_mask = data.train_mask
        val_mask = data.val_mask
    elif args.mode == "independent":
        train_mask = data.train_mask | data.val_mask
        val_mask = data.test_mask
    
    train_dataset = TensorDataset(data.x[train_mask], data.y[train_mask])
    train_val_dataset = TensorDataset(data.x[train_mask | val_mask], data.y[train_mask | val_mask])
    test_dataset = TensorDataset(data.x, data.y)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=5, pin_memory=True)
    train_val_loader = DataLoader(train_val_dataset, batch_size=512, shuffle=True, num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=14, pin_memory=True)
    
    params_grid = json.loads(f"{BASE_DIR}/src/hparams.json")[args.model]
    best_avep = 0
    for k_th, params in tqdm(enumerate(list(product(*list(params_grid.values()))))):
        param_dict= {"seed": args.seed}
        for i, param in enumerate(params_grid.keys()):
            param_dict[param] = params[i]
        
        # init the model
        model, optimizer, criterion = setup(param_dict, data, train_mask, device)
        
        for epoch in trange(1, 2000 + 1):
            loss_train = train_step(train_loader, model, device, optimizer, criterion)
        
            if epoch in [500, 1000, 2000]:
                param_dict["epoch"] = epoch
                out, loss_val = eval_step(test_loader, model, device, criterion)
                
                perf = eval_perf(data.y_orig[val_mask], out[val_mask])
                if perf["AveP"] > best_avep:
                    best_avep = perf["AveP"]
                    best_params = copy.deepcopy(param_dict)

    model, optimizer, criterion = setup(param_dict, data, train_mask | val_mask, device)
    for epoch in trange(1, best_params["epoch"] + 1):
        loss_train = train_step(train_val_loader, model, device, optimizer, criterion)
    out, loss_val = eval_step(test_loader, model, device, criterion)
    
    if args.mode == "train":
        return eval_perf(data.y_orig[data.test_mask], out[data.test_mask])

    elif args.mode == "independent":
        np.savetxt(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/preds/pred_{args.seed}.txt", out.detach().cpu().numpy(), delimiter="\t")


def train_clf(args):
    set_seed(args.seed)
    data, _, feature_names = load_data(args.omics, args.model, seed=args.seed)
    if args.model in ["svm", "tabnet"] and "SYS" in args.omics:
        norm_sys(args, args.model, data)
        
    data.y_orig = data.y.detach().clone()
    print(data.x.shape, (data.train_mask | data.val_mask | data.test_mask).sum(), data.y.sum(), data.independent_mask.sum())
    if args.mode == "train":
        train_mask = data.train_mask
        val_mask = data.val_mask
    elif args.mode == "independent":
        train_mask = data.train_mask | data.val_mask
        val_mask = data.test_mask
    
    params_grid = json.loads(f"{BASE_DIR}/src/hparams.json")[args.model]
    best_avep = 0
    pos_weight = ((len(data.y[train_mask]) - sum(data.y[train_mask])) / sum(data.y[train_mask])).item()
    for k_th, params in tqdm(enumerate(list(product(*list(params_grid.values()))))):
        param_dict = {"seed": args.seed}
        param_dict.update({param: params[i] for i, param in enumerate(params_grid.keys())})
        
        clf = fit_clf(args, param_dict, data.x, data.y, train_mask, pos_weight)
        out = torch.from_numpy(clf.predict_proba(data.x.numpy()))[:, 1]
        perf = eval_perf(data.y_orig[val_mask], out[val_mask])
        if perf["AveP"] > best_avep:
            best_avep = perf["AveP"]
            best_params = copy.deepcopy(param_dict)

    clf = fit_clf(args, best_params, data.x, data.y, train_mask | val_mask, pos_weight)
    out = torch.from_numpy(clf.predict_proba(data.x.numpy()))[:, 1]
    
    if args.mode == "train":
        return eval_perf(data.y_orig[data.test_mask], out[data.test_mask])

    elif args.mode == "independent":
        np.savetxt(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/preds/pred_{args.seed}.txt", out, delimiter="\t")
        with open(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/models/model_{args.seed}.pkl", "wb") as f:
            pkl.dump(clf, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--model", default="xgb", type=str, choices=["xgb", "mlp", "rf", "svm", "tabnet"])
    parser.add_argument("--omics", default="MF+METH+GE+SYS+TOPO+scRaw_all+nonzero_mean", type=str)
    parser.add_argument("--mode", default="train", type=str, choices=["train", "independent"])
    args = parser.parse_args()
    
    if args.mode == "train":
        os.makedirs(os.path.join(BASE_DIR, 'crossvalid', args.model, args.omics, 'results'), exist_ok=True)
        i, j = args.seed // 10000, args.seed % 10000
        if os.path.exists(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/results/AUC.txt"):
            AUC = np.loadtxt(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/results/AUC.txt", delimiter="\t")
            if AUC[i][j] != 0:
                sys.exit(0)
            
        if "mlp" in args.model:
            perf = train_mlp(args)
        else:
            perf = train_clf(args)

        if os.path.exists(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/results/AUC.txt"):
            result = {
                metric: np.loadtxt(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/results/{metric}.txt", delimiter="\t")
                for metric in ["AUC", "AUPR", "AveP"]
            }
        else:
            result = {metric: np.zeros(shape=(10, 5)) for metric in ["AUC", "AUPR", "AveP"]}

        #save
        for metric in ["AUC", "AUPR", "AveP"]:
            result[metric][i, j] = perf[metric]
            np.savetxt(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/results/{metric}.txt", result[metric], delimiter="\t")
            
        if np.count_nonzero(result["AveP"]) == 50:
            result_final = {"Mean": {metric: result[metric].mean().round(6) for metric in ["AUC", "AUPR", "AveP"]},
                            "Std": {metric: result[metric].std().round(6) for metric in ["AUC", "AUPR", "AveP"]}}
            with open(os.path.join(BASE_DIR, 'crossvalid', args.model, args.omics, "result.toml"), "w") as f:
                tomlkit.dump(result_final, f)
                
                
    elif args.mode == "independent":
        os.makedirs(os.path.join(BASE_DIR, 'crossvalid', args.model, args.omics, 'preds'), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, 'crossvalid', args.model, args.omics, 'models'), exist_ok=True)
        if os.path.exists(f"{BASE_DIR}/crossvalid/{args.model}/{args.omics}/preds/pred_{args.seed}.txt"):
            sys.exit(0)

        if "mlp" in args.model:
            train_mlp(args)
        else:
            train_clf(args)