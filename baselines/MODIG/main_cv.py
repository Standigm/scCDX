import argparse
import copy
import os
import tomlkit
from glob import glob
from itertools import product
from tqdm import trange, tqdm

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim

from modig import MODIG
from modig_graph import ModigGraph
# from utils import *

cuda = torch.cuda.is_available()


def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MODIG with cross-validation and save model to file')
    parser.add_argument('-ppi', '--ppi', help='the chosen type of PPI',
                        dest='ppi',
                        default='CPDB_v34',
                        type=str
                        )
    parser.add_argument('-omics', '--omics', help='the chosen node attribute [multiomic, snv, cnv, mrna, dm]',
                        dest='omics',
                        default='multiomic',
                        type=str
                        )
    parser.add_argument('-cancer', '--cancer', help='the model on pancan or specific cancer type',
                        dest='cancer',
                        default='pancan',
                        type=str
                        )
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 1000)',
                        dest='epochs',
                        default=1000,
                        type=int
                        )
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.25)',
                        dest='dp',
                        default=0.25,
                        type=float
                        )
    parser.add_argument('-lr', '--learningrate', help='the learning rate (default: 0.001)',
                        dest='lr',
                        default=0.001,
                        type=float
                        )
    parser.add_argument('-wd', '--weightdecay', help='the weight decay (default: 0.0005)',
                        dest='wd',
                        default=0.0005,
                        type=float
                        )
    parser.add_argument('-hs1', '--hiddensize1', help='the hidden size of first convolution layer (default: 300)',
                        dest='hs1',
                        default=300,
                        type=int
                        )
    parser.add_argument('-hs2', '--hiddensize2', help='the hidden size of second convolution layer (default: 100)',
                        dest='hs2',
                        default=100,
                        type=int
                        )
    parser.add_argument('-thr_go', '--thr_go', help='the threshold for GO semantic similarity (default: 0.8)',
                        dest='thr_go',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_seq', '--thr_seq', help='the threshold for gene sequence similarity (default: 0.5)',
                        dest='thr_seq',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-thr_exp', '--thr_exp', help='the threshold for tissue co-expression pattern (default: 0.8)',
                        dest='thr_exp',
                        default=0.8,
                        type=float
                        )
    parser.add_argument('-thr_path', '--thr_path', help='the threshold of gene pathway co-occurrence (default: 0.5)',
                        dest='thr_path',
                        default=0.5,
                        type=float
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 0)',
                        dest='seed',
                        default=0,
                        type=int
                        )
    parser.add_argument('-scaler', '--scaler', help='whether to use scaler or not (default: False)',
                        dest='scaler',
                        default=False,
                        action="store_true",
                        )
    args = parser.parse_args()
    return args


def train_epoch(graphlist_adj, mask, label, model, optimizer, scaler):
    model.train()
    optimizer.zero_grad()
    if scaler is None:
        output = model(graphlist_adj).squeeze()
        loss = F.binary_cross_entropy_with_logits(
            output[mask], label[mask], pos_weight=torch.Tensor([2.7]).to(device))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(graphlist_adj).squeeze()
            loss = F.binary_cross_entropy_with_logits(
                output[mask], label[mask], pos_weight=torch.Tensor([2.7]).to(device))
            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) ###
            scaler.step(optimizer)
            scaler.update()

    return loss.item()#, acc


@torch.no_grad()
def eval(graphlist_adj, mask, label, model):
    model.eval()
    output = model(graphlist_adj).squeeze()
    loss = F.binary_cross_entropy_with_logits(
        output[mask], label[mask], pos_weight=torch.Tensor([2.7]).to(device))
    
    pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
    # auroc = metrics.roc_auc_score(label.to('cpu'), pred)
    Yn = label[mask].cpu().detach().numpy()
    pr, rec, _ = metrics.precision_recall_curve(Yn, pred)
    # aupr = metrics.auc(rec, pr)
    # ap = metrics.average_precision_score(Yn, pred)
    perf = {
        "AUC": metrics.roc_auc_score(Yn, pred),
        "AUPR": metrics.auc(rec, pr),
        "AveP": metrics.average_precision_score(Yn, pred),
    }

    return pred, loss.item(), perf#auroc, aupr, ap
    # return torch.sigmoid(output).cpu().detach().numpy(), loss.item(), perf#auroc, aupr, ap



def load_data(args, device):
    # load data
    graph_path = os.path.join('./Data/graph', args['ppi'])
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    modig_input = ModigGraph(graph_path, args['ppi'], args['cancer'])

    print('Network INFO')

    ppi_path = os.path.join(graph_path, 'ppi.pkl')
    go_path = os.path.join(
        graph_path, str(args['thr_go']) + '_go.pkl')
    exp_path = os.path.join(
        graph_path, str(args['thr_exp']) + '_exp.pkl')
    seq_path = os.path.join(
        graph_path, str(args['thr_seq']) + '_seq.pkl')
    path_path = os.path.join(
        graph_path, str(args['thr_path']) + '_path.pkl')

    omic_path = os.path.join(graph_path, 'omics.pkl')

    if os.path.exists(ppi_path) & os.path.exists(go_path) & os.path.exists(exp_path) & os.path.exists(seq_path) & os.path.exists(path_path) & os.path.exists(omic_path):
        print('The five gene similarity profiles and omic feature already exist!')
        ppi_network = pd.read_pickle(ppi_path)#, sep='\t', index_col=0)
        go_network = pd.read_pickle(go_path)#, sep='\t', index_col=0)
        exp_network = pd.read_pickle(exp_path)#, sep='\t', index_col=0)
        seq_network = pd.read_pickle(seq_path)#, sep='\t', index_col=0)
        path_network = pd.read_pickle(path_path)#, sep='\t', index_col=0)
        omicsfeature = pd.read_pickle(omic_path)#, sep='\t', index_col=0)
        final_gene_node = list(omicsfeature.index)

    else:
        omicsfeature, final_gene_node = modig_input.get_node_omicfeature()
        ppi_network, go_network, exp_network, seq_network, path_network = modig_input.generate_graph(
            args['thr_go'], args['thr_exp'], args['thr_seq'], args['thr_path'])
    data, _ = modig_input.split_data(omics=args["omics"], 
                                            exp="MODIG" + ("_selection_all" if "scGNN" in args["omics"] else ""), 
                                            seed=args["seed"])
    omicsfeature = data.x.numpy()
    
    print("==========================================================")
    print('Network INFO')
    name_of_network = ['PPI', 'GO', 'EXP', 'SEQ', 'PATH']
    graphlist = []
    for i, network in enumerate([ppi_network, go_network, exp_network, seq_network, path_network]):
        featured_graph = modig_input.load_featured_graph(network, omicsfeature)
        print(f'The {name_of_network[i]} graph: {featured_graph}')
        graphlist.append(featured_graph)
    
    n_fdim = graphlist[0].x.shape[1]  # n_gene = featured_gsn.x.shape[0]
    graphlist_adj = [graph.to(device) for graph in graphlist]
    print("==========================================================")
    
    return modig_input, graphlist_adj, n_fdim


def train(args, modig_input, graphlist_adj, n_fdim, save_path, device):
    i, j = args["seed"] // 10000, args["seed"] % 10000
    if os.path.exists(f"{save_path}/results/AUC.txt"):
        AUC = np.loadtxt(f"{save_path}/results/AUC.txt", delimiter="\t")
        if AUC[i][j] != 0:
            return None
        
    seed_torch(args["seed"])
    
    data, _ = modig_input.split_data(args["omics"], exp="MODIG", seed=args["seed"])
    # data.x, data.edge_index = torch.zeros(1), torch.zeros(1)
    data.to(device)
    
    best_avep, best_params = 0., {}
    log_dict = {}
    for k_th, params in tqdm(enumerate(list(product(*list(params_grid.values()))))):
        param_dict = {param: params[i] for i, param in enumerate(params_grid.keys())}
        
        while True:
            try:
                model = MODIG(
                    nfeat=n_fdim, hidden_size1=param_dict['hs1'], hidden_size2=param_dict['hs2'], dropout=param_dict['dp'])
                model.to(device)
                optimizer = optim.Adam(
                    model.parameters(), lr=param_dict['lr'], weight_decay=param_dict['wd'])
                
                scaler = torch.cuda.amp.GradScaler() if args["scaler"] else None

                for epoch in trange(1, args['epochs']+1):
                    _ = train_epoch(graphlist_adj, data.train_mask, data.y, model, optimizer, scaler)
                    
                _, _, perf = eval(graphlist_adj, data.val_mask, data.y, model)
                
                if perf["AveP"] > best_avep:
                    best_avep = perf["AveP"]
                    best_params = copy.deepcopy(param_dict)

                log_dict.update({f"{k_th}-th hparams": param_dict | perf})
                break
            except:
                pass
        
    with open(f"{save_path}/logs/log_{args['seed']}.toml", "w") as f:
        tomlkit.dump(log_dict, f)
        
    while True:
        try:
            model = MODIG(
                nfeat=n_fdim, hidden_size1=best_params['hs1'], hidden_size2=best_params['hs2'], dropout=best_params['dp'])
            model.to(device)
            optimizer = optim.Adam(
                model.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])
            
            scaler = torch.cuda.amp.GradScaler() if args["scaler"] else None

            for epoch in trange(1, args['epochs']+1):
                _ = train_epoch(graphlist_adj, data.train_mask | data.val_mask, data.y, model, optimizer, scaler)
                
            _, _, perf = eval(graphlist_adj, data.test_mask, data.y, model)
            
            if os.path.exists(f"{save_path}/results/AUC.txt"):
                result = {
                    metric: np.loadtxt(f"{save_path}/results/{metric}.txt", delimiter="\t")
                    for metric in ["AUC", "AUPR", "AveP"]
                }
            else:
                result = {metric: np.zeros(shape=(10, 5)) for metric in ["AUC", "AUPR", "AveP"]}
            
            #save
            for metric in ["AUC", "AUPR", "AveP"]:
                result[metric][i, j] = perf[metric]
                np.savetxt(f"{save_path}/results/{metric}.txt", result[metric], delimiter="\t")

            if (result["AUC"] == 0).sum() == 0:
                result = {"Mean": {metric: result[metric].mean().round(6) for metric in result.keys()},
                        "Std": {metric: result[metric].std().round(6) for metric in result.keys()}}
                with open(f"{save_path}/performance.toml", "w") as f:
                    tomlkit.dump(result, f)
            break
        except:
            pass

            
def test(args, modig_input, graphlist_adj, n_fdim, save_path, device):
    if os.path.exists(f"{save_path}/preds/pred_{args['seed']}.txt"):
        return None
        
    seed_torch(args["seed"])

    
    data, _ = modig_input.split_data(args["omics"], exp="MODIG", seed=args["seed"])
    # data.x, data.edge_index = torch.zeros(1), torch.zeros(1)
    data.unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
    data.to(device)
    
    best_avep, best_params = 0., {}
    for k_th, params in tqdm(enumerate(list(product(*list(params_grid.values()))))):
        param_dict = {param: params[i] for i, param in enumerate(params_grid.keys())}
        
        while True:
            try:
                model = MODIG(
                    nfeat=n_fdim, hidden_size1=param_dict['hs1'], hidden_size2=param_dict['hs2'], dropout=param_dict['dp'])
                model.to(device)
                optimizer = optim.Adam(
                    model.parameters(), lr=param_dict['lr'], weight_decay=param_dict['wd'])
                
                scaler = torch.cuda.amp.GradScaler() if args["scaler"] else None

                for epoch in trange(1, args['epochs']+1):
                    _ = train_epoch(graphlist_adj, data.train_mask | data.val_mask, data.y, model, optimizer, scaler)
                    
                _, _, perf = eval(graphlist_adj, data.test_mask, data.y, model)
                
                if perf["AveP"] > best_avep:
                    best_avep = perf["AveP"]
                    best_params = copy.deepcopy(param_dict)
                break
            except:
                pass
    
    while True:
        try:
            model = MODIG(
                nfeat=n_fdim, hidden_size1=best_params['hs1'], hidden_size2=best_params['hs2'], dropout=best_params['dp'])
            model.to(device)
            optimizer = optim.Adam(
                model.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])
            
            scaler = torch.cuda.amp.GradScaler() if args["scaler"] else None

            for epoch in trange(1, args['epochs']+1):
                _ = train_epoch(graphlist_adj, data.train_mask | data.val_mask | data.test_mask, data.y, model, optimizer, scaler)
            with torch.no_grad():
                model.eval()
                output = model(graphlist_adj).squeeze()
                pred = torch.sigmoid(output[data.independent_mask & data.unlabeled_mask]).cpu().detach().numpy()
            # pred, _, _ = eval(graphlist_adj, data.independent_mask & data.unlabeled_mask, data.y, model)
            np.savetxt(f"{save_path}/preds/pred_{args['seed']}.txt", pred, delimiter="\t") 
            break
        except:
            pass


if __name__ == '__main__':

    args = parse_args()
    args_dic = vars(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = os.path.join("log", args_dic['omics'])
    os.makedirs(save_path + '/results', exist_ok=True)
    os.makedirs(save_path + '/logs', exist_ok=True)
    os.makedirs(save_path + '/preds', exist_ok=True)
    modig_input, graphlist_adj, n_fdim = load_data(args_dic, device)

    params_grid = {
        "lr": [0.002, 0.001, 0.0005],
        "dp": [0.25, 0.5],
        "wd": [0.0005],
        "hs1": [200, 300, 400],
        "hs2": [50, 100, 200],
    }

    test(args_dic, modig_input, graphlist_adj, n_fdim, save_path, device)
    train(args_dic, modig_input, graphlist_adj, n_fdim, save_path, device)
    print('Training is finished!')
