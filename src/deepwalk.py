import os
from argparse import ArgumentParser
from tqdm import tqdm

import scipy.sparse as sp
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_scipy_sparse_matrix, remove_self_loops

from src.utils import load_data

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def deepwalk(ppi: str, num_workers: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = sp.load_npz(os.path.join(BASE_DIR, "ppi", f"{ppi}_adj.npz"))
    edge_index, _ = remove_self_loops(from_scipy_sparse_matrix(adj)[0])
    print(adj.shape)    

    model =  Node2Vec(edge_index, embedding_dim=16, walk_length=80,
                        context_size=5,  walks_per_node=10,
                        num_negative_samples=1, p=1, q=1).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

    model.train()
    with tqdm(total=500, desc="Epoch") as pbar:
        for epoch in range(1, 501):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            pbar.set_postfix({"loss": total_loss / len(loader)})
            pbar.update(1)

    model.eval()
    str_fearures = model().detach().cpu()

    os.makedirs(os.path.join(BASE_DIR, "feature"), exist_ok=True)
    torch.save(str_fearures, os.path.join(BASE_DIR, "feature/TOPO.pkl"))
 
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-workers", default=4, type=int)
    
    args = parser.parse_args()
    
    deepwalk("CPDB_v34", args.num_workers)