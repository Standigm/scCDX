import gzip
import os
import numpy as np
import pandas as pd
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

import discotoolkit as dt
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

from src.utils import match_genes, load_symbol_df

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def preprocess_ppi(ppi: str):
    edges = pd.read_csv(os.path.join(BASE_DIR, f"ppi/{ppi}.tsv"), sep="\t", compression="gzip")
    nodes = sorted(set(edges['partner1']) | set(edges['partner2']))

    node2idx = {node: i for i, node in enumerate(nodes)}
    adj = sp.dok_matrix((len(nodes), len(nodes)), dtype=bool)
    for _, row in tqdm(edges.iterrows(), total=len(edges)):
        i, j = node2idx[row['partner1']], node2idx[row['partner2']]
        adj[i, j] = adj[j, i] = True

    sp.save_npz(os.path.join(BASE_DIR, f'ppi/{ppi}_adj.npz'), adj.tocsr())

  
def get_metadata(tumor: bool, num_cells: int):
    if os.path.exists("DISCOtmp/metadata_" + ("tumor" if tumor else "normal") + ".pkl"):
        with open("DISCOtmp/metadata_" + ("tumor" if tumor else "normal") + ".pkl", "rb") as f:
            metadata = pkl.load(f)
    else:
        ft = dt.Filter(sample_type=["Primary Tumor", "Cancer"] if tumor else ["Normal"])
        metadata = dt.filter_disco_metadata(ft)
        with open("DISCOtmp/metadata_" + ("tumor" if tumor else "normal") + ".pkl", "wb") as f:
            pkl.dump(metadata, f)
    
    sample_metadata = metadata.sample_metadata
    sample_metadata = sample_metadata[(sample_metadata["treatment"].isna()) & (sample_metadata["diseaseStage"].isna())]
    if tumor:
        sample_metadata = sample_metadata[~sample_metadata["diseaseSubtype"].isin(["pre-neoplastic", "metastatic breast cancer"])]
        sample_metadata = sample_metadata[~sample_metadata["disease"].isin(["COVID-19", "Juxta tumor", "intraductal papillary mucinous neoplasm", "lipoma"])]
        sample_metadata = sample_metadata[~sample_metadata["disease"].isna()]
    else:
        sample_metadata = sample_metadata[sample_metadata["disease"].isna()]
    
    cm = metadata.cell_type_metadata
    df_metadata = sample_metadata.merge(cm)
    df_metadata = df_metadata.replace({"esophagus squamous cell carcinoma": "esophageal squamous cell carcinoma",
                                       "nasopharyngeal tumor": "nasopharyngeal carcinoma",
                                       "pancreatic adenocarcenoma": "pancreatic ductal adenocarcinoma",
                                       "renal adenocarcinoma": "renal cell carcinoma"})
    
    if tumor:
        df_cellnum = df_metadata[["disease", "tissue", "cellType", "cellNumber"]].groupby(by=["disease", "tissue", "cellType"]).sum()
    else:
        df_cellnum = df_metadata[["tissue", "cellType", "cellNumber"]].groupby(by=["tissue", "cellType"]).sum()
    indices = df_cellnum[df_cellnum["cellNumber"] >= num_cells].drop("cellNumber", axis=1).reset_index()
    print(indices)
    
    return df_metadata, indices


def get_features(metadata, common_indices):
    os.makedirs("DISCOtmp/features", exist_ok=True)
    
    results = []
    adata = sc.read_h5ad(f"DISCOtmp/AML003_3p.h5ad")
    for _, row in tqdm(common_indices.iterrows(), total=len(common_indices)):
        row_values = [rv.replace("/", "-") for rv in row.values]
        if os.path.exists(f"DISCOtmp/features/{'_'.join(row_values)}.pkl"):
            with open(f"DISCOtmp/features/{'_'.join(row_values)}.pkl", "rb") as f:
                results.append(pkl.load(f))
        else:
            print(row)
            samples = metadata.loc[tuple(row.values), "sampleId"].unique()
            X = []
            for sample in samples:
                if os.path.exists(f"DISCOtmp/{sample}.h5ad"):
                    adata = sc.read_h5ad(f"DISCOtmp/{sample}.h5ad")
                    # sc.pp.normalize_total(adata, target_sum = 1e4)####
                    X.append(adata.X[adata.obs["cell.type"] == row["cellType"]].A)
            X = np.concatenate(X, axis=0)
            
            X[X == 0.] = np.nan
            nonzero_mean = np.nanmean(X, axis=0, keepdims=True)
            
            with open(f"DISCOtmp/features/{'_'.join(row_values)}.pkl", "wb") as f:
                pkl.dump(nonzero_mean, f)
            results.append(nonzero_mean)
    print(len(results), len(common_indices))
    
    return pd.DataFrame(np.concatenate([np.concatenate(results, axis=0), common_indices], axis=1), 
                        columns=adata.var.index.tolist() + common_indices.columns.tolist())



def preprocess_feature(num_cells: int, download: bool=False):
    tumor_metadata, tumor_indices = get_metadata(tumor=True, num_cells=num_cells)
    normal_metadata, normal_indices = get_metadata(tumor=False, num_cells=num_cells)
    
    tumor_common_indices = tumor_indices.merge(normal_indices)
    normal_common_indices = tumor_common_indices.drop("disease", axis=1).drop_duplicates()
    tumor_metadata = tumor_metadata.merge(tumor_common_indices).set_index(["disease", "tissue", "cellType"])
    normal_metadata = normal_metadata.merge(normal_common_indices).set_index(["tissue", "cellType"])
    print(len(tumor_common_indices), len(normal_common_indices), len(tumor_metadata), len(normal_metadata), len(tumor_metadata["sampleId"].unique()), len(normal_metadata["sampleId"].unique()))
    
    tumor_metadata["sampleId"].drop_duplicates().to_csv("supplementary/tumor_samples.csv", index=False)
    normal_metadata["sampleId"].drop_duplicates().to_csv("supplementary/normal_samples.csv", index=False)
    if download:
        tumor_filter = dt.filter_disco_metadata(dt.Filter(sample=tumor_metadata["sampleId"].unique().tolist()))
        dt.download_disco_data(tumor_filter)
        normal_filter = dt.filter_disco_metadata(dt.Filter(sample=normal_metadata["sampleId"].unique().tolist()))
        dt.download_disco_data(normal_filter)
        def download_disco(sample):
            return dt.download_disco_data(dt.filter_disco_metadata(dt.Filter(sample=sample)))
        with ThreadPoolExecutor(60) as executor:
            _ = list(tqdm(executor.map(download_disco, tumor_metadata["sampleId"].unique().tolist()), total=len(tumor_metadata["sampleId"].unique().tolist())))
        with ThreadPoolExecutor(60) as executor:
            _ = list(tqdm(executor.map(download_disco, normal_metadata["sampleId"].unique().tolist()), total=len(normal_metadata["sampleId"].unique().tolist())))
    
    if os.path.exists(f"{BASE_DIR}/preprocessed/tumor_{num_cells}.pkl.gz"):
        with gzip.open(f"{BASE_DIR}/preprocessed/tumor_{num_cells}.pkl.gz", "rb") as f:
            tumor_feature = pkl.load(f)
    else:
        tumor_feature = get_features(tumor_metadata, tumor_common_indices)
        with gzip.open(f"{BASE_DIR}/preprocessed/tumor_{num_cells}.pkl.gz", "wb") as f:
            pkl.dump(tumor_feature, f)
    if os.path.exists(f"{BASE_DIR}/preprocessed/normal_{num_cells}.pkl.gz"):
        with gzip.open(f"{BASE_DIR}/preprocessed/normal_{num_cells}.pkl.gz", "rb") as f:
            normal_feature = pkl.load(f)
    else:
        normal_feature = get_features(normal_metadata, normal_common_indices)
        with gzip.open(f"{BASE_DIR}/preprocessed/normal_{num_cells}.pkl.gz", "wb") as f:
            pkl.dump(normal_feature, f)
    
    
if __name__ == "__main__":
    preprocess_ppi("CPDB_v34")
    preprocess_feature(num_cells=500, download=True)