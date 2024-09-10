import os
import random
import tomlkit
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.profiler import profile, record_function, ProfilerActivity#

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from src.utils import load_data, norm_sys
from model import EMOGINet, MTGCNNet, HGDCNet


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')

def setup(method, input_dim, device):
    if "MTGCN" in method:
        model = MTGCNNet(input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        EPOCH = 2500
        
    elif "EMOGI" in method:
        model = EMOGINet(input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
        EPOCH = 2000
        
    elif "HGDC" in method:
        model = HGDCNet(input_dim).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.linear1.parameters(), weight_decay=0.00001),
            dict(params=model.linear_r0.parameters(), weight_decay=0.00001),
            dict(params=model.linear_r1.parameters(), weight_decay=0.00001),
            dict(params=model.linear_r2.parameters(), weight_decay=0.00001),
            dict(params=model.linear_r3.parameters(), weight_decay=0.00001),
            dict(params=model.weight_r0, lr=0.001 * 0.1),
            dict(params=model.weight_r1, lr=0.001 * 0.1),
            dict(params=model.weight_r2, lr=0.001 * 0.1),
            dict(params=model.weight_r3, lr=0.001 * 0.1)
        ], lr=0.001)
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
def eval_independent(model, data, mask, method="MTGCN"):
    model.eval()
    if "MTGCN" in method:
        out, _, _, _ = model(data.x, data.edge_index)
    else:
        out = model(data)
    pred = torch.sigmoid(out[mask]).cpu().detach().numpy()

    perf = {
        "OncoKB": metrics.average_precision_score(data.y_oncokb_latest[mask].cpu().numpy(), pred),
        "ONGene": metrics.average_precision_score(data.y_ongene[mask].cpu().numpy(), pred),
        "NCG": metrics.average_precision_score(data.y_ncg7[mask].cpu().numpy(), pred),
        "Bailey": metrics.average_precision_score(data.y_bailey[mask].cpu().numpy(), pred),
        "Independent_NCG7": metrics.average_precision_score(data.y_independent1[mask].cpu().numpy(), pred),
        "Independent_CCGD_A": metrics.average_precision_score(data.y_independent2[mask].cpu().numpy(), pred),
    }

    return torch.sigmoid(out).cpu().detach().numpy(), perf


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
        data, genes, _ = load_data(cancer=args.cancer_type,
                                   ppi=args.ppi, 
                                   omics=args.omics, 
                                   experiment=args.method + ("_selection_all" if "scGNN" in args.omics else ""), 
                                   seed=seed)
        if "SYS" in args.omics: 
            norm_sys(args, args.method + ("_selection_all" if "scGNN" in args.omics else ""), data)
            
    elif args.method == "HGDC":
        data, genes, _ = load_data(cancer=args.cancer_type,
                                   ppi=args.ppi, 
                                   omics=args.omics, 
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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for j in range(5):
        i = args.seed // 10000
        if os.path.exists(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/AUC.txt"):
            AUC = np.loadtxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/AUC.txt", delimiter="\t")
            if AUC[i][j] != 0:
                continue
    
        seed = args.seed + j
        set_seed(seed)
        # seed_everything(seed)

        data, _ = load(args, seed)
        data.unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
        data = data.to(device)
        
        model, optimizer, EPOCH = setup(args.method, input_dim=data.x.shape[1], device=device)
        # EPOCH=10
        # prof = torch.profiler.profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        #     record_shapes=True,
        #     with_stack=True,
        #     with_modules=True)
        # prof.start()

        for epoch in trange(EPOCH):
            train_epoch(model, data, data.train_mask | data.val_mask, optimizer, args.method)
        #     prof.step()
            
        # prof.stop()
        # return 0

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
    

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for j in trange(5):
        i = args.seed // 10000
        if os.path.exists(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/OncoKB.txt"):
            OncoKB = np.loadtxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/OncoKB.txt", delimiter="\t")
            if OncoKB[i][j] != 0:
                continue
        seed = args.seed + j
        set_seed(seed)
        
        data, _ = load(args, seed)
        data.unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
        data.y_independent1 = ((data.y_oncokb_latest + data.y_ongene + data.y_ncg7 + data.y_bailey) > 0).float()
        data.y_independent2 = ((data.y_oncokb_latest + data.y_ongene + data.y_ncg7 + data.y_bailey + data.y_cancermine + data.y_ccgda) > 0).float()
        data.to(device)

        model, optimizer, EPOCH = setup(args.method, input_dim=data.x.shape[1], device=device)
            
        # independent
        for epoch in trange(EPOCH):
            train_epoch(model, data, data.train_mask | data.val_mask | data.test_mask, optimizer, args.method)

        pred, perf = eval_independent(model, data, data.unlabeled_mask & data.independent_mask, args.method)

        if os.path.exists(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/OncoKB.txt"):
            result = {dataset: np.loadtxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/{dataset}.txt", delimiter="\t") 
                      for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]
                      }
        else:
            result = {dataset: np.zeros(shape=(10, 5))
                      for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]
                      }
            
        np.savetxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/pred_{seed}.txt", pred, delimiter="\t")
        for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]:
            result[dataset][i, j] = perf[dataset]
            np.savetxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/{dataset}.txt", result[dataset], delimiter="\t")
            
            
    pred_files = sorted(glob(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/pred_*.txt"))
    if len(pred_files) == 50:
        result = {dataset: np.loadtxt(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results/{dataset}.txt", delimiter="\t") 
                  for dataset in ["OncoKB", "ONGene", "NCG", "Bailey", "Independent_NCG7", "Independent_CCGD_A"]
                  }
        result = {"Mean": {dataset: result[dataset].mean().round(6) for dataset in result.keys()},
                  "Std": {dataset: result[dataset].std().round(6) for dataset in result.keys()}}
        with open(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/independence.toml", "w") as f:
            tomlkit.dump(result, f)
            
        preds = [np.loadtxt(file, delimiter="\t") for file in pred_files]
        preds.append(data.y_independent1.cpu().numpy())
        preds.append(data.y_independent2.cpu().numpy())
        preds = np.stack(preds, axis=1)
        preds_df = pd.DataFrame(preds[data.unlabeled_mask.cpu().numpy()], 
                            index=np.array(genes)[data.unlabeled_mask.cpu().numpy()], 
                            columns=[f"seed{i * 10000 + j}" for i in range(10) for j in range(5)] + ["Independent_NCG7", "Independent_CCGD_A"])
        preds_df.index.name = "Gene"
        preds_df.to_csv(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/pred.csv")
        preds_rank_df = pd.DataFrame(np.concatenate([np.argsort(-preds[data.unlabeled_mask.cpu().numpy(), :-5], axis=0).argsort(axis=0) + 1, preds[data.unlabeled_mask.cpu().numpy(), -5:]], axis=1), 
                            index=np.array(genes)[data.unlabeled_mask.cpu().numpy()], 
                            columns=[f"seed{i * 10000 + j}" for i in range(10) for j in range(5)] + ["Independent_NCG7", "Independent_CCGD_A"])
        preds_rank_df.index.name = "Gene"
        preds_rank_df.to_csv(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/pred_rank.csv")


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    parser = ArgumentParser()
    parser.add_argument("--ppi", default="CPDB_v34", type=str)
    parser.add_argument("--omics", default="MF+METH+GE+SYS+TOPO", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--method", default="MTGCN", type=str, choices=["EMOGI", "MTGCN", "HGDC"])
    parser.add_argument("--cancer-type", default="pancancer", type=str, choices=["LUAD", "BRCA", "LIHC", "BLCA", "CESC", "COAD", "ESCA", 
                                                                                 "HNSC", "KIRC", "KIRP", "LUSC", "PRAD", "READ", "STAD",
                                                                                 "UCEC", "THCA", "pancancer"])
    
    args = parser.parse_args()
    os.makedirs(f"{BASE_DIR}/{args.method}/{args.cancer_type}/{args.omics}/results", exist_ok=True)
    
    train(args)
    # test(args)