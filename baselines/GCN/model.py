import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops


class MTGCNNet(torch.nn.Module):
    def __init__(self, input_dim: int=58, hid_dim1: int=300, hid_dim2: int=100): #FIXME
        super(MTGCNNet, self).__init__()
        self.conv1 = ChebConv(input_dim, hid_dim1, K=2, normalization="sym")
        self.conv2 = ChebConv(hid_dim1, hid_dim2, K=2, normalization="sym")
        self.conv3 = ChebConv(hid_dim2, 1, K=2, normalization="sym")

        self.lin1 = Linear(input_dim, hid_dim2)
        self.lin2 = Linear(input_dim, hid_dim2)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x, edge_index):
        pos_edge_index, _ = dropout_edge(edge_index, p=0.5,
                                    force_undirected=True,
                                    # num_nodes=x.size()[0],
                                    training=self.training)

        x0 = F.dropout(x, training=self.training)
        x = torch.relu(self.conv1(x0, pos_edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, pos_edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        # neg_edge_index = negative_sampling(pb, 13627, 504378)
        neg_edge_index = negative_sampling(add_self_loops(edge_index)[0], num_nodes=x.shape[0], num_neg_samples=edge_index.shape[1])

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss


        x = F.dropout(x, training=self.training)
        # x = self.conv3(x, pos_edge_index)
        x = self.conv3(x, pos_edge_index).squeeze()

        return x, r_loss, self.c1, self.c2


class EMOGINet(torch.nn.Module):
    def __init__(self, input_dim: int=48, hid_dim1: int=300, hid_dim2: int=100):
        super(EMOGINet, self).__init__()
        self.conv1 = GCNConv(input_dim, hid_dim1)
        self.conv2 = GCNConv(hid_dim1, hid_dim2)
        self.conv3 = GCNConv(hid_dim2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x.squeeze()


class HGDCNet(torch.nn.Module):
    def __init__(self, input_dim: int=58, hid_dim: int=100, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()
        in_channels = input_dim
        hidden_channels = hid_dim
        self.linear1 = Linear(in_channels, hidden_channels)

        # 3 convolutional layers for the original network
        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops = False) #
        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        # 3 convolutional layers for the auxiliary network
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops = False) #
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops = False)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)

        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(2*hidden_channels, 1)
        self.linear_r2 = Linear(2*hidden_channels, 1)
        self.linear_r3 = Linear(2*hidden_channels, 1)

        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = data.edge_index_aux

        edge_index_1, _ = dropout_edge(edge_index_1, p=0.5,
                                      force_undirected=True,
                                    #   num_nodes=x_input.shape[0],
                                      training=self.training)
        edge_index_2, _ = dropout_edge(edge_index_2, p=0.5,
                                      force_undirected=True,
                                    #   num_nodes=x_input.shape[0],
                                      training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 =self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out
