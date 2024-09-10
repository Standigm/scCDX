import copy
import numpy as np
import pandas as pd
import os
import time
import tomlkit
from argparse import ArgumentParser
from itertools import product
from tqdm import tqdm, trange

import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn import linear_model
from torch_geometric.utils import from_scipy_sparse_matrix, add_remaining_self_loops

import src.utils
from model_all import Net
from get_BA import get_bilinear

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')


def load_data(path, omics, device):
    # load network
    network1 = []
    adj1 = sp.load_npz(path + "PP.adj.npz")      # gene-gene network
    adj2 = sp.load_npz(path + "PO.adj.npz")      # gene-outlying gene network
    adj3 = sp.load_npz(path + "PR.adj.npz")      # gene-miRNA network

    network1.append(adj1.tocsc())
    network1.append(adj2.tocsc())
    network1.append(adj3.tocsc())

    # netwroks for bilinear aggregation layer
    network2 = []
    adj4 = sp.load_npz(path + "O.adj_loop.npz")
    adj5 = sp.load_npz(path + "O.N_all.npz")

    network2.append(adj4.tocsc())
    network2.append(adj5.tocsc())

    # load node features
    l_feature = []      # gene
    feat1 = pd.read_csv(path + f"P{omics}.feat-final.csv", sep=",").values[:, 1:]
    feat1 = torch.Tensor(feat1).to(device)
    feat2 = pd.read_csv(path + f"P{omics}.feat-final.csv", sep=",").values[:, 1:]
    feat2 = torch.Tensor(feat2).to(device)
    feat3 = pd.read_csv(path + f"P{omics}.feat-pre.csv", sep=",").values[:, 1:]
    feat3 = torch.Tensor(feat3).to(device)

    l_feature.append(feat1)
    l_feature.append(feat2)
    l_feature.append(feat3)

    r_feature = []
    feat4 = pd.read_csv(path + f"P{omics}.feat-final.csv", sep=",").values[:, 1:]       # gene
    feat4 = torch.Tensor(feat4).to(device)
    feat5 = pd.read_csv(path + "O.feat-final.csv", sep=",").values[:, 1:]       # outlying gene
    feat5 = torch.Tensor(feat5).to(device)
    feat6 = pd.read_csv(path + f"R{omics}.feat-pre.csv", sep=",").values[:, 1:]         # miRNA
    feat6 = torch.Tensor(feat6).to(device)

    r_feature.append(feat4)
    r_feature.append(feat5)
    r_feature.append(feat6)

    # load edge
    pos_edge = np.array(np.loadtxt(path + "PP_pos.txt").transpose())
    pos_edge = torch.from_numpy(pos_edge).long()
    pos_edge1, _ = add_remaining_self_loops(pos_edge)

    return network1, network2, l_feature, r_feature, pos_edge, pos_edge1

def LR(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    pre = pre[:,1]

    return pre

def train_epoch(model, optimizer, mask, label):
    model.train()
    optimizer.zero_grad()

    pred, pred1, r_loss, _ = model()
    loss = F.binary_cross_entropy_with_logits(pred[mask].squeeze(), label[mask])
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask].squeeze(), label[mask])
    loss = loss + 0.1 * loss1 + 0.01 * r_loss

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) ###FIXME
    optimizer.step()


@torch.no_grad()
def eval(model, train_mask, test_mask, label):#train_label, test_label):
    model.eval()
    _, _, _, x = model()

    # logistic regression model
    train_x = torch.sigmoid(x[train_mask]).cpu().detach().numpy()
    train_y = label[train_mask].cpu().numpy()
    test_x = torch.sigmoid(x[test_mask]).cpu().detach().numpy()
    Yn = label[test_mask].cpu().numpy().reshape(-1)
    pred = LR(train_x, train_y, test_x)
    precision, recall, _ = metrics.precision_recall_curve(Yn, pred)
    perf = {
        "AUC": metrics.roc_auc_score(Yn, pred),
        "AUPR": metrics.auc(recall, precision),
        "AveP": metrics.average_precision_score(Yn, pred),
    }

    return perf


@torch.no_grad()
def eval_independent(model, train_mask, independent_mask, data):#train_label, independent_label):
    model.eval()
    _, _, _, x = model()

    # logistic regression model
    train_x = torch.sigmoid(x[train_mask]).cpu().detach().numpy()
    train_y = data.y[train_mask].cpu().numpy()
    test_x = torch.sigmoid(x[independent_mask]).cpu().detach().numpy()
    pred = LR(train_x, train_y, test_x)

    return pred


def split_data(seed: int=0):
    data, genes, _ = src.utils.load_data("MF+METH+GE+TOPO", "MRNGCN", seed=seed)
    node_gene = pd.read_csv('data/gene_names.txt', names=['ENSG', 'Hugosymbol'])["Hugosymbol"].tolist()
    gene2idx = {gene: i for i, gene in enumerate(genes)}
    mrngcn_idx = [gene2idx[gene] for gene in node_gene]
    
    for attr in ["x", "y", "train_mask", "val_mask", "test_mask"]:
        setattr(data, attr, getattr(data, attr)[mrngcn_idx])
    
    return data, node_gene


def setup(network, device, **params): #FIXME
    network1, network2, l_feature, r_feature, pos_edge, pos_edge1 = network #FIXME
    # print("network1", network1)
    # print("network2", network2)
    # print("l_feature", l_feature)
    # print("r_feature", r_feature)
    # model = Net(l_feature, r_feature, network1, network2, 1, 64, 256, 128, pos_edge, pos_edge1).to(device)    # hop=1
    # model = Net(l_feature, r_feature, network1, network2, 1, l_feature[0].shape[1], 256, 128, pos_edge, pos_edge1).to(device)    # hop=1
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
    model = Net(l_feature, r_feature, network1, network2, 1, l_feature[0].shape[1], params["hid_dim1"], params["hid_dim2"], pos_edge, pos_edge1).to(device)    # hop=1
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    # decayRate = 0.96
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
    
    return model, optimizer, scheduler


def train(args, network, device, params_grid):
    # 十次五倍交叉
    # pan-cancer
    i, j = args.seed // 10000, args.seed % 10000
    if os.path.exists(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/AUC.txt"):
        AUC = np.loadtxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/AUC.txt", delimiter="\t")
        if AUC[i][j] != 0:
            return None
    src.utils.set_seed(args.seed)
    
    data, genes = split_data(args.seed)
    # print(genes, data.y.tolist())
    data.to(device)
    
    # Tuning
    best_avep, best_params = 0., {}
    log_dict = {}
    for k_th, params in tqdm(enumerate(list(product(*list(params_grid.values()))))):
        param_dict = {param: params[i] for i, param in enumerate(params_grid.keys())}

        # network1, network2, l_feature, r_feature, pos_edge, pos_edge1 = network
        model, optimizer, scheduler = setup(network, device, **param_dict)
        
        for epoch in trange(1, args.epoch + 1):
            train_epoch(model, optimizer, data.train_mask, data.y)
            if epoch % 50 ==0:
                scheduler.step()
            
        perf = eval(model, data.train_mask, data.val_mask, data.y)
    
        if perf["AveP"] > best_avep:
            best_avep = perf["AveP"]
            best_params = copy.deepcopy(param_dict)
        log_dict.update({f"{k_th}-th hparams": param_dict | perf})
        

    with open(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/logs/log_{args.seed}.toml", "w") as f:
        tomlkit.dump(log_dict, f)


    # Test
    # network1, network2, l_feature, r_feature, pos_edge, pos_edge1 = network
    model, optimizer, scheduler = setup(network, device, **best_params)
    
    for epoch in trange(1, args.epoch + 1):
        train_epoch(model, optimizer, data.train_mask | data.val_mask, data.y)
        if epoch % 50 ==0:
            scheduler.step()
            
    perf = eval(model, data.train_mask | data.val_mask, data.test_mask, data.y)

    if os.path.exists(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/AUC.txt"):
        result = {
            metric: np.loadtxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/{metric}.txt", delimiter="\t")
            for metric in ["AUC", "AUPR", "AveP"]
        }
    else:
        result = {metric: np.zeros(shape=(10, 5)) for metric in ["AUC", "AUPR", "AveP"]}
    
    #save
    for metric in ["AUC", "AUPR", "AveP"]:
        result[metric][i, j] = perf[metric]
        np.savetxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/{metric}.txt", result[metric], delimiter="\t")


    if (result["AUC"] == 0).sum() == 0:
        result = {"Mean": {metric: result[metric].mean().round(6) for metric in result.keys()},
                  "Std": {metric: result[metric].std().round(6) for metric in result.keys()}}
        with open(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/performance.toml", "w") as f:
            tomlkit.dump(result, f)


def test(args, network, device, params_grid):
    if os.path.exists(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/preds/pred_{args.seed}.txt"):
        return None
    src.utils.set_seed(args.seed)

    data, genes = split_data(args.seed)
    data.unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
    data.to(device)
        
    # independent
    best_avep, best_params = 0., {}
    for params in tqdm(list(product(*list(params_grid.values())))):
        param_dict = {param: params[i] for i, param in enumerate(params_grid.keys())}

        # network1, network2, l_feature, r_feature, pos_edge, pos_edge1 = network
        model, optimizer, scheduler = setup(network, device, **param_dict)
        for epoch in trange(1, args.epoch + 1):
            train_epoch(model, optimizer, data.train_mask | data.val_mask, data.y)
            if epoch % 50 ==0:
                scheduler.step()
        perf = eval(model, data.train_mask | data.val_mask, data.test_mask, data.y)

        if perf["AveP"] > best_avep:
            best_avep = perf["AveP"]
            best_params = copy.deepcopy(param_dict)


    # network1, network2, l_feature, r_feature, pos_edge, pos_edge1 = network
    model, optimizer, scheduler = setup(network, device, **best_params)
    for epoch in trange(1, args.epoch + 1):
        train_epoch(model, optimizer, data.train_mask | data.val_mask | data.test_mask, data.y)
        if epoch % 50 ==0:
            scheduler.step()
            
    pred = eval_independent(model, data.train_mask | data.val_mask | data.test_mask, data.independent_mask & data.unlabeled_mask, data)
    np.savetxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/preds/pred_{args.seed}.txt", pred, delimiter="\t") 
            

if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    time_start = time.time()
    parser = ArgumentParser()
    parser.add_argument("--ppi", default="CPDB_v34", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--omics", default="MF+METH+GE+TOPO", type=str)
    parser.add_argument("--epoch", default=1065, type=int)
    args = parser.parse_args()
    
    os.makedirs(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/preds", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/logs", exist_ok=True)
    
    # load data
    path = "./data/"     # pan-cancer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.omics == "MF+METH+GE+TOPO":
        omics = ""
    # elif args.omics in ["MF+METH+GE+SYS+TOPO", "MF+METH+GE+SYS+TOPO+scGNN"]:
    else:
        omics = "." + args.omics
    network = load_data(path, omics, device)
    
    params_grid = {
        "lr": [0.001, 0.002, 0.005],
        "hid_dim1": [256, 512] if "sc" in args.omics else [128, 256, 512],
        "hid_dim2": [128, 256] if "sc" in args.omics else [64, 128, 256],
        "weight_decay": [0.0005, 0.0001, 0.001],
    }

    train(args, network, device, params_grid)
    test(args, network, device, params_grid)