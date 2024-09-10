import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from argparse import ArgumentParser
from tqdm import tqdm

import src.utils
from model_pretrain import pretrain

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
EPOCH = 1000        # pan-cancer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess(args):
    data, genes, _ = src.utils.load_data(
        cancer=args.cancer_type,
        ppi="CPDB_v34", 
        omics=args.omics, 
        experiment="MRNGCN" + ("_selection_all" if "scGNN" in args.omics else "")
    )
    if "SYS" in args.omics:
        src.utils.norm_sys(args, "MRNGCN" + ("_selection_all" if "scGNN" in args.omics else ""), data)
    
    final_gene_node = pd.read_csv(os.path.join(BASE_DIR, "gene_names.txt"), names=["ENSG", "gene"])["gene"].tolist()
    gene2idx = {gene: i for i, gene in enumerate(genes)}
    mrngcn_idx = [gene2idx[gene] for gene in final_gene_node]

    data.x = data.x[mrngcn_idx].clone()
    pd.DataFrame(data.x.numpy()).to_csv(os.path.join(BASE_DIR, f"P.{args.omics}.feat-final.csv"))
    
    return data.x.numpy()


def load_data(args):
    # load network
    adj = sp.load_npz(os.path.join(BASE_DIR, "PR.adj.npz"))
    network = adj.tocsc()
    
    # load node features
    if args.omics != "MF+METH+GE+TOPO":
        if os.path.exists(os.path.join(BASE_DIR, f"P.{args.omics}.feat-final.csv")):
            feat1 = pd.read_csv(os.path.join(BASE_DIR, f"P.{args.omics}.feat-final.csv"), sep=",").values[:, 1:]
        else:
            feat1 = preprocess(args)
    else:
        feat1 = pd.read_csv(os.path.join(BASE_DIR, "P.feat-final.csv"), sep=",").values[:, 1:]
    l_feature = torch.Tensor(feat1).to(device)
    feat2 = pd.read_csv(os.path.join(BASE_DIR, "R.feat-final.csv"), sep=",").values[:, 1:]
    r_feature = torch.Tensor(feat2).to(device)

    # load edge
    pos_edge = np.array(np.loadtxt(os.path.join(BASE_DIR, "PR_pos.txt")).transpose())
    pos_edge = torch.from_numpy(pos_edge).long()

    neg_edge = np.array(np.loadtxt(os.path.join(BASE_DIR, "PR_neg.txt")).transpose())# gene-miRNA网络
    neg_edge = torch.from_numpy(neg_edge).long()

    return network, l_feature, r_feature, pos_edge, neg_edge


def train():
    model.train()
    optimizer.zero_grad()
    loss, l_node, r_node = model()

    # print(loss)
    loss.backward()
    optimizer.step()

    # return l_node, r_node
    return l_node, r_node, loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ppi", default="CPDB_v34", type=str)
    parser.add_argument("--omics", default="MF+METH+GE+TOPO", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cancer-type", default="pancancer", type=str, choices=["LUAD", "BRCA", "LIHC", "BLCA", "CESC", "COAD", "ESCA", 
                                                                                 "HNSC", "KIRC", "KIRP", "LUSC", "PRAD", "READ", "STAD",
                                                                                 "UCEC", "THCA", "pancancer"])
    args = parser.parse_args()
    
    src.utils.set_seed(0)

    network, feature1, feature2, pos_edge, neg_edge = load_data(args)       # pan-cancer
    model = pretrain(feature1, feature2, network, 1, feature1.shape[1], 256, feature1.shape[1], pos_edge, neg_edge).to(device)  # hop = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    with tqdm(total=EPOCH) as pbar:
        for epoch in range(1, EPOCH + 1):
            l_node, r_node, loss = train()
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

    # Save node features
    P_feat = l_node.cpu().detach().numpy()
    R_feat = r_node.cpu().detach().numpy()
    if args.omics != "MF+METH+GE+TOPO":
        pd.DataFrame(P_feat).to_csv(os.path.join(BASE_DIR, f"P.{args.omics}.feat-pre.csv"))
        pd.DataFrame(R_feat).to_csv(os.path.join(BASE_DIR, f"R.{args.omics}.feat-pre.csv"))
    else:
        pd.DataFrame(P_feat).to_csv(os.path.join(BASE_DIR, f"P.feat-pre.csv"))
        pd.DataFrame(R_feat).to_csv(os.path.join(BASE_DIR, f"R.feat-pre.csv"))
