import gzip
import os
import random
import tomlkit
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, remove_self_loops

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_scrna(matched_genes: list[str], omics: str):
    with gzip.open(f"{BASE_DIR}/preprocessed/tumor_500_normalized_all.pkl.gz", "rb") as f:
        tumor_feature = pkl.load(f)
    with gzip.open(f"{BASE_DIR}/preprocessed/normal_500_normalized_all.pkl.gz", "rb") as f:
        normal_feature = pkl.load(f)

    feature = []
    for stat in tumor_feature.keys():
        if stat in omics.split("+"):
            tumor_feature[stat]['tissue_celltype'] = [row["tissue"] + "_" + row["cellType"] for _, row in tumor_feature[stat].iterrows()]
            tumor_feature[stat] = tumor_feature[stat].drop(columns=["tissue", "cellType"]).set_index(["disease", "tissue_celltype"])
            normal_feature[stat]['tissue_celltype'] = [row["tissue"] + "_" + row["cellType"] for _, row in normal_feature[stat].iterrows()]
            normal_feature[stat] = normal_feature[stat].drop(columns=["tissue", "cellType"]).set_index("tissue_celltype")
            if "mean" in stat:
                normal_feature[stat] = normal_feature[stat].replace(0, np.nan)
                array = normal_feature[stat].rdiv(tumor_feature[stat], level=1, axis="index").astype(float)
                array = np.log2(array).loc[:, matched_genes]
            else:
                array = (tumor_feature[stat].subtract(normal_feature[stat], level=1, axis="index")).loc[:, matched_genes]
            
            array["stat"] = [stat] * len(array)
            array = array.reset_index().set_index(["stat", "disease", "tissue_celltype"])
            feature.append(array)
    feature = pd.concat(feature, axis=0).replace([np.inf, np.nan, -np.inf], [0, 0, 0])
    feature = feature.loc[(feature!=0).any(axis=1)] ###

    return feature.to_numpy().T, feature.index


def load_ppi(ppi_genes: list[str], matched_genes: list[str]):
    node_idx = [i for i, gene in enumerate(ppi_genes) if gene in matched_genes]
    adj = sp.load_npz(os.path.join(BASE_DIR, f'ppi/CPDB_v34_adj.npz'))
    adj = adj[node_idx][:, node_idx]
    
    return adj
    

def load_label(matched_genes: list[str], symbol_df, symbols):
    label_df = pd.read_csv(os.path.join(BASE_DIR, "label/pancan_genelist_for_train.tsv"), sep='\t').replace({"1-Mar": "MARC1", "2-Mar": "MARC2"}).set_index('Hugosymbol')
    label_genes = set(label_df.index)
    matched_ppi_genes, matched_label_genes = match_genes(matched_genes, label_genes, symbol_df, symbols, "label.pkl")
    labels = label_df.loc[matched_label_genes, 'Label'].to_numpy()

    return np.array(matched_ppi_genes), labels


def load_independent_label(exp_ppi_genes: list[str], scbulk_ppi_genes: list[str], symbol_df, symbols):
    false_genes = set(exp_ppi_genes) - set(scbulk_ppi_genes)
    # print("false genes num", len(false_genes))
    mask_independent = [False if gene in false_genes else True for gene in exp_ppi_genes]

    oncokb_df = pd.read_csv(os.path.join(BASE_DIR, "independent", "oncokb_cancerGeneList.tsv"), sep="\t")
    oncokb_symbols = oncokb_df[oncokb_df['OncoKB Annotated'] == 'Yes']['Hugo Symbol'].tolist()
    oncokb_matched_ppi_genes, _ = match_genes(scbulk_ppi_genes, oncokb_symbols, symbol_df, symbols, "oncokb.pkl")

    ongene_df = pd.read_csv(os.path.join(BASE_DIR, "independent", "ongene_human.txt"), sep="\t")
    ongene_symbols = ongene_df['OncogeneName'].tolist()
    ongene_matched_ppi_genes, _ = match_genes(scbulk_ppi_genes, ongene_symbols, symbol_df, symbols, "ongene.pkl")
    
    tsgene_df = pd.read_csv(os.path.join(BASE_DIR, "independent", "TSGene2.0.txt"), sep="\t")
    tsgene_symbols = tsgene_df['GeneSymbol'].tolist()
    tsgene_matched_ppi_genes, _ = match_genes(scbulk_ppi_genes, tsgene_symbols, symbol_df, symbols, "tsgene.pkl")
    
    ncg_df = pd.read_csv(os.path.join(BASE_DIR, "independent", "NCG_candidate_drivers.txt"), names=["Gene"])
    ncg_symbols = ncg_df["Gene"].tolist()
    ncg_matched_ppi_genes, _ = match_genes(scbulk_ppi_genes, ncg_symbols, symbol_df, symbols, "ncg.pkl")
    
    cancermine_df = pd.read_csv(os.path.join(BASE_DIR, "independent", "cancermine_collated.tsv"), sep="\t")
    cancermine_symbols = cancermine_df["gene_normalized"].unique().tolist()
    cancermine_matched_ppi_genes, _ = match_genes(scbulk_ppi_genes, cancermine_symbols, symbol_df, symbols, "cancermine.pkl")

    ccgd_df = pd.read_csv(os.path.join(BASE_DIR, "independent", "ccgd.csv"))
    ccgda_symbols = ccgd_df[ccgd_df["Rank"] == "A"]["HumanName"].unique().tolist()
    ccgda_matched_ppi_genes, _ = match_genes(scbulk_ppi_genes, ccgda_symbols, symbol_df, symbols, "ccgda.pkl")

    y_oncokb, y_ongene, y_tsgene, y_ncg, y_cancermine, y_ccgda = [], [], [], [], [], []
    for symbol in exp_ppi_genes:
        y_oncokb.append(True if symbol in oncokb_matched_ppi_genes else False)
        y_ongene.append(True if symbol in ongene_matched_ppi_genes else False)
        y_tsgene.append(True if symbol in tsgene_matched_ppi_genes else False)
        y_ncg.append(True if symbol in ncg_matched_ppi_genes else False)
        y_cancermine.append(True if symbol in cancermine_matched_ppi_genes else False)
        y_ccgda.append(True if symbol in ccgda_matched_ppi_genes else False)
    
    return (torch.tensor(y_oncokb).float(), torch.tensor(y_ongene).float(), torch.tensor(y_tsgene).float(), 
            torch.tensor(y_ncg).float(), torch.tensor(y_cancermine).float(), torch.tensor(y_ccgda).float(),
            torch.tensor(mask_independent))


def integrate_data(ppi_genes, symbol_df, symbols, ppi: str, omics: str):
    with gzip.open(f"{BASE_DIR}/preprocessed/tumor_500.pkl.gz", "rb") as f:
        tumor_feature = pkl.load(f)
    sc_matched_ppi_genes, matched_sc_genes = match_genes(ppi_genes, tumor_feature.columns[:-3], symbol_df, symbols, "sc.pkl")
    bulk_df = pd.read_csv(os.path.join(BASE_DIR, "feature", "biological_features.tsv"), sep="\t").set_index("Hugosymbol").sort_index()
    bulk_matched_ppi_genes, matched_bulk_genes = match_genes(ppi_genes, bulk_df.index, symbol_df, symbols, "bulk.pkl")
    
    x = pd.DataFrame(index=ppi_genes)
    if "scRaw" in omics:
        sc_x, sc_celltypes = load_scrna(matched_sc_genes, omics) #FIXME
        x = x.join(pd.DataFrame(sc_x, index=sc_matched_ppi_genes, columns=sc_celltypes))
    
    if any(bulk in omics for bulk in ["MF", "METH", "GE"]):
        bulk_x = bulk_df.loc[matched_bulk_genes].to_numpy()
        bulk_x = np.concatenate([bulk_x[:, 16*i:16*(i+1)] for i, bulk in enumerate(["MF", "METH", "GE"]) if bulk in omics], axis=1)
        bulk_disease = ["KIRC", "BRCA", "READ", "PRAD", "STAD", "HNSC", "LUAD", "THCA", "BLCA", "ESCA", "LIHC", "UCEC", "COAD", "LUSC", "CESC", "KIRP"]
        bulk_celltypes = [(bulk, disease) for bulk in ["MF", "METH", "GE"] if bulk in omics for disease in bulk_disease]
        
        x = x.join(pd.DataFrame(bulk_x, index=bulk_matched_ppi_genes, columns=bulk_celltypes))
    
    if "TOPO" in omics:
        str_x = torch.load(os.path.join(BASE_DIR, "feature", f"TOPO.pkl")).numpy()
        str_x = pd.DataFrame(str_x, index=ppi_genes, columns=[("TOPO", f"dim {i}") for i in range(16)])
        x = x.join(str_x)

    if "SYS" in omics:
        sys_x, sys_matched_ppi_genes = load_systemlevel_features(ppi_genes)
        if "SYS" == omics: matched_ppi_genes = sys_matched_ppi_genes
        sys_x = pd.DataFrame(sys_x, 
                             index=sys_matched_ppi_genes, 
                             columns=[("SYS", f) for f in ["essentiality_percentage", "expressed_tissues_rnaseq", "ppin_degree", 
                                                            "ppin_betweenness", "ppin_clustering", "complexes", "mirna", 
                                                            "ohnolog", "essentiality_oneCellLine", "ppin_hub"]])
        x = x.join(sys_x)

    
    return x


def load_data(omics: str = "SC5", experiment: str = "FC", seed: int=0):
    symbol_df, symbols = load_symbol_df()
    ppi_df = pd.read_csv(os.path.join(BASE_DIR, f'ppi/CPDB_v34.tsv'), sep='\t', compression='gzip')
    ppi_genes = sorted(set(ppi_df['partner1']) | set(ppi_df['partner2']))   
  
    x = integrate_data(ppi_genes, symbol_df, symbols, omics)
    # if ppi == "CPDB_v34":
    matched_ppi_genes = ppi_genes
        
    feature_names = x.columns
    x = x.loc[matched_ppi_genes].replace(np.nan, 0)
    x = x.replace(0, np.nan).to_numpy() if "nan" in experiment else x.to_numpy()
    adj = load_ppi(ppi_genes, matched_ppi_genes)
    label_matched_ppi_genes, labels = load_label(ppi_genes, symbol_df, symbols)
    print(len(labels), sum(labels), len(labels)-sum(labels))

    edge_index, _ = remove_self_loops(from_scipy_sparse_matrix(adj)[0])
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor([1. if gene in label_matched_ppi_genes[labels == 1] else 0. for gene in matched_ppi_genes])
    data = Data(x = x, edge_index = edge_index, y = y)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed // 10)
    for i, (train_val_idx, test_idx) in enumerate(kf.split(label_matched_ppi_genes, labels)):
        if i == seed % 10:
            genes_tv, genes_test, labels_tv = label_matched_ppi_genes[train_val_idx], label_matched_ppi_genes[test_idx], labels[train_val_idx]
    genes_train, genes_val = train_test_split(genes_tv, test_size=.25, random_state=seed // 10, stratify=labels_tv)

    data.train_mask = torch.tensor([True if gene in genes_train else False for gene in matched_ppi_genes])
    data.val_mask = torch.tensor([True if gene in genes_val else False for gene in matched_ppi_genes])
    data.test_mask = torch.tensor([True if gene in genes_test else False for gene in matched_ppi_genes])
    
    data.y_oncokb_latest, data.y_ongene, data.y_tsgene, data.y_ncg7, data.y_cancermine, data.y_ccgda, data.independent_mask = \
        load_independent_label(matched_ppi_genes, ppi_genes, symbol_df, symbols)
    data = scale_data(data, experiment)
    
    return data, matched_ppi_genes, feature_names


def scale_data(data, experiment):
    if "minmax" in experiment:
        data_mask = (data.x != 0).float()
        data.x -= data.x.min(0, keepdim=True)[0]
        data.x /= data.x.max(0, keepdim=True)[0]
        data.x *= data_mask
    if "norm" in experiment:
        data_mask = (data.x != 0).float()
        data_mean = data.x.sum(0) / data_mask.sum(0)
        data_std = torch.sqrt((torch.pow(data.x - data_mean, 2) * data_mask).sum(0) / (data_mask.sum(0) - 1))
        data.x = (data.x - data_mean) / data_std * data_mask
        
    return data


def load_systemlevel_features(genes: list[str]):
    system_feat = pd.read_csv(os.path.join(BASE_DIR, "feature/systemsLevel_features_allGenes.tsv"), sep="\t", index_col="entrez")
    system_feat = system_feat[["essentiality_percentage", "expressed_tissues_rnaseq", "ppin_degree", 
                                "ppin_betweenness", "ppin_clustering", "complexes", "mirna", 
                                "ohnolog", "essentiality_oneCellLine", "ppin_hub"]]
    system_feat = system_feat.replace({"ohnolog": 0, "essentiality_oneCellLine": 0, "ppin_hub": 0}, -1)

    entrez2gene = pd.read_csv(os.path.join(BASE_DIR, "feature/entrez2gene.txt"), sep="\t")
    symb2entrez = entrez2gene[["Approved symbol", "NCBI gene ID"]].drop_duplicates().set_index("Approved symbol")
    aliasb2entrez = entrez2gene[["Alias symbol", "NCBI gene ID"]].drop_duplicates().set_index("Alias symbol")
    prev2entrez = entrez2gene[["Previous symbol", "NCBI gene ID"]].drop_duplicates().set_index("Previous symbol")

    symb = symb2entrez.index
    alias = aliasb2entrez.index
    prev = prev2entrez.index
    entrez_ids = system_feat.index

    feat, matched_genes = [], []
    for gene in genes:
        if gene in symb:
            entrez = symb2entrez.loc[gene, "NCBI gene ID"].tolist()
        elif gene in alias:
            entrez = aliasb2entrez.loc[gene, "NCBI gene ID"].tolist()
        elif gene in prev:
            entrez = prev2entrez.loc[gene, "NCBI gene ID"].tolist()
        else:
            entrez = -1
        
        if isinstance(entrez, list):
            for e in entrez:
                if e in entrez_ids:
                    feat.append(system_feat.loc[e].to_numpy())
                    matched_genes.append(gene)
                    break

        elif entrez in entrez_ids:
            feat.append(system_feat.loc[entrez].to_numpy())
            matched_genes.append(gene)

            
    return np.stack(feat, axis=0), matched_genes


def load_symbol_df():
    symbol_df = pd.read_csv(os.path.join(BASE_DIR, "HGNC_v2023.05.08.txt"), sep="\t")
    symbol_df["Approved symbol"] = symbol_df["Approved symbol"].str.upper()
    symbol_df["Alias symbol"] = symbol_df["Alias symbol"].str.upper()
    symbol_df["Previous symbol"] = symbol_df["Previous symbol"].str.upper()
    symb = symbol_df["Approved symbol"].unique().tolist()
    alias = symbol_df["Alias symbol"].unique().tolist()
    prev = symbol_df["Previous symbol"].unique().tolist()
    symbols = (symb, alias, prev)
    
    return symbol_df, symbols


def match_genes(
    ppi_genes: list[str], 
    feature_genes: list[str], 
    symbol_df: pd.DataFrame, 
    symbols: list[list[str]],
    # ppi: str,
    file_name: str
):
    os.makedirs(os.path.join(BASE_DIR, ".cache", "CPDB_v34"), exist_ok=True)
    if os.path.exists(os.path.join(BASE_DIR, ".cache", "CPDB_v34", file_name)):
        ppi_gs, feature_gs = torch.load(os.path.join(BASE_DIR, ".cache","CPDB_v34", file_name))
        
    else:
        symb, alias, prev = symbols
        # all_syms = set(symb + alias + prev)
        all_syms = set(symb + prev)
        feat_only_genes = set(feature_genes) - set(ppi_genes)
        
        gene_pairs = [[gene, gene] for gene in set(ppi_genes) & set(feature_genes)]
        for ppi_g in set(ppi_genes) - set(feature_genes):
            ppi_g_upper = ppi_g.upper()
            if ppi_g_upper in all_syms:
                if ppi_g_upper in symb:
                    sdf = symbol_df[symbol_df["Approved symbol"] == ppi_g_upper]
                # elif ppi_g_upper in alias:
                #     sdf = symbol_df[symbol_df["Alias symbol"] == ppi_g_upper]
                elif ppi_g_upper in prev:
                    sdf = symbol_df[symbol_df["Previous symbol"] == ppi_g_upper]
                ap = sdf["Approved symbol"].dropna().unique().tolist()
                # al = sdf["Alias symbol"].dropna().unique().tolist()
                pr = sdf["Previous symbol"].dropna().unique().tolist()
                g = set(ap + pr) & feat_only_genes
                for gg in g:
                    gene_pairs.append([ppi_g, gg])

        df = pd.DataFrame(np.array(sorted(gene_pairs)), columns=["ppi", "feature"])
        df = df.drop_duplicates(["ppi"], keep=False).drop_duplicates(["feature"], keep=False)
        ppi_gs, feature_gs = df["ppi"].tolist(), df["feature"].tolist()
    
        # torch.save((ppi_gs, feature_gs), os.path.join(BASE_DIR, ".cache", ppi, file_name))
        torch.save((ppi_gs, feature_gs), os.path.join(BASE_DIR, ".cache", "CPDB_v34", file_name))
    
    return ppi_gs, feature_gs


def set_seed(seed: int):
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def norm_sys(args, experiment, data):
    sys_feats = ["essentiality_percentage", "expressed_tissues_rnaseq", "ppin_degree", "ppin_betweenness", 
                     "ppin_clustering", "complexes", "mirna", "ohnolog", "essentiality_oneCellLine", "ppin_hub"]
    data_norm, _, feature_names = load_data(args.cancer_type, args.ppi, args.omics, experiment + "_norm", seed=args.seed)
    sys_idx = [i for i, feat in enumerate(feature_names) if feat[1] in sys_feats]
    # print(sys_idx)
    # non_sys_idx = [i for i, feat in enumerate(feature_names) if feat[1] not in sys_feats]
    # data.x = torch.cat([data.x[:, non_sys_idx], data_norm.x[:, sys_idx]], dim=1)
    
    nonzero_idx = [i for i, feat in enumerate(feature_names) if feat[0] == "nonzero_mean"]
    # non_nonzero_idx = [i for i, feat in enumerate(feature_names) if feat[0] != "nonzero_mean"]
    no_norm_idx = [i for i, feat in enumerate(feature_names) if (feat[1] not in sys_feats and feat[0] != "nonzero_mean")]
    
    data.x = torch.cat([data_norm.x[:, nonzero_idx], data.x[:, no_norm_idx], data_norm.x[:, sys_idx]], dim=1)