import numpy as np
import pandas as pd
import os
import time
import tomlkit
from argparse import ArgumentParser
from glob import glob
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

    return perf#metrics.roc_auc_score(Yn, pred), aupr, ap


@torch.no_grad()
def eval_independent(model, train_mask, independent_mask, data):#train_label, independent_label):
    model.eval()
    _, _, _, x = model()

    # logistic regression model
    train_x = torch.sigmoid(x[train_mask]).cpu().detach().numpy()
    train_y = data.y[train_mask].cpu().numpy()
    test_x = torch.sigmoid(x[independent_mask]).cpu().detach().numpy()
    pred = LR(train_x, train_y, test_x)
    
    perf = {
        "OncoKB": metrics.average_precision_score(data.y_oncokb_latest[independent_mask].cpu().numpy(), pred),
        "ONGene": metrics.average_precision_score(data.y_ongene[independent_mask].cpu().numpy(), pred),
        "NCG": metrics.average_precision_score(data.y_ncg7[independent_mask].cpu().numpy(), pred),
        "Bailey": metrics.average_precision_score(data.y_bailey[independent_mask].cpu().numpy(), pred),
        "Independent_NCG7": metrics.average_precision_score(data.y_independent1[independent_mask].cpu().numpy(), pred),
        "Independent_CCGD_A": metrics.average_precision_score(data.y_independent2[independent_mask].cpu().numpy(), pred),
    }

    return pred, perf#ap


def split_data(cancer_type:str = "pancancer", ppi: str="CPDB_v34", seed: int=0):
    data, genes, _ = src.utils.load_data(cancer_type, ppi, "MF+METH+GE+TOPO", "MRNGCN", seed=seed)
    node_gene = pd.read_csv('data/gene_names.txt', names=['ENSG', 'Hugosymbol'])["Hugosymbol"].tolist()
    gene2idx = {gene: i for i, gene in enumerate(genes)}
    mrngcn_idx = [gene2idx[gene] for gene in node_gene]
    
    for attr in ["x", "y", "train_mask", "val_mask", "test_mask"]:
        setattr(data, attr, getattr(data, attr)[mrngcn_idx])
    
    return data, node_gene


def setup(l_feature, r_feature, network1, network2, pos_edge, pos_edge1, device):
    # model = Net(l_feature, r_feature, network1, network2, 1, 64, 256, 128, pos_edge, pos_edge1).to(device)    # hop=1
    model = Net(l_feature, r_feature, network1, network2, 1, l_feature[0].shape[1], 256, 128, pos_edge, pos_edge1).to(device)    # hop=1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0005)
    # decayRate = 0.96
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
    
    return model, optimizer, scheduler


def train(args, network, device):
    # 十次五倍交叉
    # pan-cancer
    for j in range(5):
        i = args.seed // 10000
        if os.path.exists(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/AUC.txt"):
            AUC = np.loadtxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/AUC.txt", delimiter="\t")
            if AUC[i][j] != 0:
                continue
        seed = args.seed + j
        src.utils.set_seed(seed)
        
        data, genes = split_data(args.cancer_type, "CPDB_v34", seed)
        # print(genes, data.y.tolist())
        data.to(device)
        
        # load model
        network1, network2, l_feature, r_feature, pos_edge, pos_edge1 = network
        model, optimizer, scheduler = setup(l_feature, r_feature, network1, network2, pos_edge, pos_edge1, device)
        
        for epoch in trange(1, args.epoch + 1):
            # train(model, optimizer, torch.cat([train_mask, val_mask]), torch.cat([train_label, val_label]))
            train_epoch(model, optimizer, data.train_mask | data.val_mask, data.y)
            if epoch % 50 ==0:
                scheduler.step()
                # print(eval(model, data.train_mask | data.val_mask, data.test_mask, data.y))
                
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



def test(args, network, device):
    # for j in trange(10):
    for j in trange(5):
        i = args.seed // 10000
        if os.path.exists(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/OncoKB.txt"):
            OncoKB = np.loadtxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/OncoKB.txt", delimiter="\t")
            if OncoKB[i][j] != 0:
                continue
        seed = args.seed + j
        src.utils.set_seed(seed)
        
        data, genes = split_data(args.cancer_type, "CPDB_v34", seed)
        data.unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
        data.y_independent1 = ((data.y_oncokb_latest + data.y_ongene + data.y_ncg7 + data.y_bailey) > 0).float()
        data.y_independent2 = ((data.y_oncokb_latest + data.y_ongene + data.y_ncg7 + data.y_bailey + data.y_cancermine + data.y_ccgda) > 0).float()
        data.to(device)
            
        # independent
        network1, network2, l_feature, r_feature, pos_edge, pos_edge1 = network
        model, optimizer, scheduler = setup(l_feature, r_feature, network1, network2, pos_edge, pos_edge1, device)

        for epoch in trange(1, args.epoch + 1):
            train_epoch(model, optimizer, data.train_mask | data.val_mask | data.test_mask, data.y)
            if epoch % 50 ==0:
                scheduler.step()
                
        pred, perf = eval_independent(model, data.train_mask | data.val_mask | data.test_mask, data.independent_mask & data.unlabeled_mask, data)

        if os.path.exists(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/OncoKB.txt"):
            result = {dataset: np.loadtxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/{dataset}.txt", delimiter="\t") 
                      for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]
                      }
        else:
            result = {dataset: np.zeros(shape=(10, 5))
                      for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]
                      }
            
        np.savetxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/pred_{seed}.txt", pred, delimiter="\t")
        for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]:
            result[dataset][i, j] = perf[dataset]
            np.savetxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/{dataset}.txt", result[dataset], delimiter="\t")
            
            
    pred_files = sorted(glob(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/pred_*.txt"))
    if len(pred_files) == 50:
        result = {dataset: np.loadtxt(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results/{dataset}.txt", delimiter="\t") 
                  for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]
                  }
        result = {"Mean": {dataset: result[dataset].mean().round(6) for dataset in result.keys()},
                  "Std": {dataset: result[dataset].std().round(6) for dataset in result.keys()}}
        with open(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/independence.toml", "w") as f:
            tomlkit.dump(result, f)
            
        preds = [np.loadtxt(file, delimiter="\t") for file in pred_files]
        preds.append(data.y_independent1.cpu().numpy())
        preds.append(data.y_independent2.cpu().numpy())
        preds = np.stack(preds, axis=1)
        preds_df = pd.DataFrame(preds[data.unlabeled_mask.cpu().numpy()], 
                            index=np.array(genes)[data.unlabeled_mask.cpu().numpy()], 
                            columns=[f"seed{i * 10000 + j}" for i in range(10) for j in range(5)] + ["Independent_NCG7", "Independent_CCGD_A"])
        preds_df.index.name = "Gene"
        preds_df.to_csv(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/pred.csv")
        preds_rank_df = pd.DataFrame(np.concatenate([np.argsort(-preds[data.unlabeled_mask.cpu().numpy(), :-5], axis=0).argsort(axis=0) + 1, preds[data.unlabeled_mask.cpu().numpy(), -5:]], axis=1), 
                            index=np.array(genes)[data.unlabeled_mask.cpu().numpy()], 
                            columns=[f"seed{i * 10000 + j}" for i in range(10) for j in range(5)] + ["Independent_NCG7", "Independent_CCGD_A"])
        preds_rank_df.index.name = "Gene"
        preds_rank_df.to_csv(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/pred_rank.csv")


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    time_start = time.time()
    parser = ArgumentParser()
    parser.add_argument("--ppi", default="CPDB_v34", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--omics", default="MF+METH+GE+TOPO", type=str)
    parser.add_argument("--epoch", default=1065, type=int)
    parser.add_argument("--cancer-type", default="pancancer", type=str, choices=["LUAD", "BRCA", "LIHC", "BLCA", "CESC", "COAD", "ESCA", 
                                                                                 "HNSC", "KIRC", "KIRP", "LUSC", "PRAD", "READ", "STAD",
                                                                                 "UCEC", "THCA", "pancancer"])
    args = parser.parse_args()
    os.makedirs(f"{BASE_DIR}/{args.cancer_type}/{args.omics}/results", exist_ok=True)
    
    # load data
    path = "./data/"     # pan-cancer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.omics == "MF+METH+GE+TOPO":
        omics = ""
    elif args.omics in ["MF+METH+GE+SYS+TOPO", "MF+METH+GE+SYS+TOPO+scGNN"]:
        omics = "." + args.omics
    network = load_data(path, omics, device)
    
    train(args, network, device)
    # test(args, network, device)