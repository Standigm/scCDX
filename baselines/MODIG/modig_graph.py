import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

from src.utils import load_data, norm_sys
    

class ModigGraph(object):

    def __init__(self, graph_path, ppi_type, cancer_type):

        self.graph_path = graph_path
        self.ppi_type = ppi_type
        self.cancer_type = cancer_type

    def split_data(self, omics: str="MF+METH+GE", exp: str="MODIG", seed: int=0):
        data, genes, _ = load_data("pancancer", ppi=self.ppi_type, omics=omics, experiment=exp, seed=seed)
        if "SYS" in omics:
            sys_feats = ["essentiality_percentage", "expressed_tissues_rnaseq", "ppin_degree", "ppin_betweenness", 
                        "ppin_clustering", "complexes", "mirna", "ohnolog", "essentiality_oneCellLine", "ppin_hub"]
            data_sys, _, feature_names = load_data("pancancer", self.ppi_type, omics, exp + "_norm", seed)
            sys_idx = [i for i, feat in enumerate(feature_names) if feat[1] in sys_feats]
            non_sys_idx = [i for i, feat in enumerate(feature_names) if feat[1] not in sys_feats]
            data.x = torch.cat([data.x[:, non_sys_idx], data_sys.x[:, sys_idx]], dim=1)
        
        final_gene_node, _ = self.get_node_genelist()
        # genes_match = pd.merge(pd.Series(sorted(final_gene_node), name='Hugosymbol'), label_df, 
        #                        on='Hugosymbol', how='left')
        gene2idx = {gene: i for i, gene in enumerate(final_gene_node)}
        modig_idx = [gene2idx[gene] for gene in genes]
        
        for attr in ["x", "y", "train_mask", "val_mask", "test_mask", "independent_mask"]:
            value = torch.zeros_like(getattr(data, attr).float())
            value = value[[0] * len(final_gene_node)]
            value[modig_idx] = getattr(data, attr).float()
            setattr(data, attr, value.bool() if "mask" in attr else value)
        
        return data, final_gene_node#, celltypes
        

    def get_node_genelist(self):
        print('Get gene list')
        gene = pd.read_csv("./Data/simmatrix/gene_info_for_GOSemSim.csv")
        gene_list = list(set(gene['Symbol']))

        ppi = pd.read_csv(os.path.join('./Data/ppi', self.ppi_type + '.tsv'), sep='\t',
                          compression='gzip', encoding='utf8', usecols=['partner1', 'partner2'])
        ppi.columns = ['source', 'target']
        ppi = ppi[ppi['source'] != ppi['target']]
        ppi.dropna(inplace=True)

        final_gene_node = sorted(
            list(set(gene_list) | set(ppi.source) | set(ppi.target)))

        return final_gene_node, ppi

    def get_node_omicfeature(self):
        print('Get node omic feature')
        final_gene_node, _ = self.get_node_genelist()

        # process the omic data
        omics_file = pd.read_csv(
            './Data/feature/biological_features.tsv', sep='\t', index_col=0)

        expendgene = sorted(list(set(omics_file.index) | set(final_gene_node)))
        temp = pd.DataFrame(index=expendgene, columns=omics_file.columns)
        omics_adj = temp.combine_first(omics_file)
        omics_adj.fillna(0, inplace=True)
        omics_adj = omics_adj.loc[final_gene_node]
        omics_adj.sort_index(inplace=True)

        if self.cancer_type != 'pancan':
            omics_data = omics_adj[omics_adj.columns[omics_adj.columns.str.contains(
                self.cancer_type)]]
        elif self.cancer_type == 'pancan':
            # chosen 16 cancer type
            chosen_project = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD',
                              'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
            omics_temp = [omics_adj[omics_adj.columns[omics_adj.columns.str.contains(
                cancer)]] for cancer in chosen_project]
            omics_data = pd.concat(omics_temp, axis=1)

        omics_data.to_pickle(os.path.join(self.graph_path, 'omics.pkl'))
        # omics_feature_vector = sp.csr_matrix(omics_data, dtype=np.float32)
        # omics_feature_vector = torch.FloatTensor(
        #     np.array(omics_feature_vector.todense()))
        # print(
        #     f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

        return omics_data, final_gene_node#omics_feature_vector, final_gene_node

    def generate_graph(self, thr_go, thr_exp, thr_seq, thr_path):
        """
        generate tri-graph: PPI GSN GO_network
        """
        print('generate graph')
        final_gene_node, ppi = self.get_node_genelist()

        path = pd.read_csv(os.path.join(
            './Data/simmatrix/pathsim_matrix.csv'), sep='\t', index_col=0)
        path_matrix = path.applymap(lambda x: 0 if x < thr_path else 1)
        np.fill_diagonal(path_matrix.values, 0)

        go = pd.read_csv('./Data/simmatrix/GOSemSim_matrix.csv',
                         sep='\t', index_col=0)
        go_matrix = go.applymap(lambda x: 0 if x < thr_go else 1)
        np.fill_diagonal(go_matrix.values, 0)

        exp = pd.read_csv(os.path.join(
            './Data/simmatrix/expsim_matrix.csv'), sep='\t', index_col=0)
        exp_matrix = exp.applymap(lambda x: 0 if x < thr_exp else 1)
        np.fill_diagonal(exp_matrix.values, 0)

        seq = pd.read_csv(os.path.join(
            './Data/simmatrix/seqsim_rrbs_matrix.csv'), sep='\t', index_col=0)
        seq_matrix = seq.applymap(lambda x: 0 if x < thr_seq else 1)
        np.fill_diagonal(seq_matrix.values, 0)

        networklist = []
        for matrix in [go_matrix, exp_matrix, seq_matrix, path_matrix]:
            temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)
            network = temp.combine_first(matrix)
            network.fillna(0, inplace=True)
            network_adj = network[final_gene_node].loc[final_gene_node].astype(pd.SparseDtype("int", 0))
            networklist.append(network_adj)
            print('The shape of network_adj:', network_adj.shape)

        # Save the processed graph data and omic data
        ppi.to_pickle(os.path.join(self.graph_path, 'ppi.pkl'))#,
                #sep='\t', index=False, compression='gzip')
        networklist[0].to_pickle(os.path.join(
            self.graph_path, str(thr_go) + '_go.pkl'))#, sep='\t')
        networklist[1].to_pickle(os.path.join(
            self.graph_path, str(thr_exp) + '_exp.pkl'))#, sep='\t')
        networklist[2].to_pickle(os.path.join(
            self.graph_path, str(thr_seq) + '_seq.pkl'))#, sep='\t')
        networklist[3].to_pickle(os.path.join(
            self.graph_path, str(thr_path) + '_path.pkl'))#, sep='\t')

        return ppi, networklist[0], networklist[1], networklist[2], networklist[3]


    def load_featured_graph(self, network, omicfeature):

        omics_feature_vector = sp.csr_matrix(omicfeature, dtype=np.float32)
        omics_feature_vector = torch.FloatTensor(
            np.array(omics_feature_vector.todense()))
        print(
            f'The shape of omics_feature_vector:{omics_feature_vector.shape}')

        if network.shape[0] == network.shape[1]:
            G = nx.from_pandas_adjacency(network)
        else:
            G = nx.from_pandas_edgelist(network)

        G_adj = nx.convert_node_labels_to_integers(
            G, ordering='sorted', label_attribute='label')

        print(f'If the graph is connected graph: {nx.is_connected(G_adj)}')
        print(
            f'The number of connected components: {nx.number_connected_components(G_adj)}')

        graph = from_networkx(G_adj)
        assert graph.is_undirected() == True
        print(f'The edge index is {graph.edge_index}')

        graph.x = omics_feature_vector

        return graph
