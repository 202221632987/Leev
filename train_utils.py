import math
import sys

import torch
import torch.nn as nn
from cogdl.utils import edge_softmax, get_activation
from cogdl.utils.spmm_utils import MultiHeadSpMM


class myGATConv(nn.Module):

    def __init__(
        self,
        edge_feats,
        num_etypes,
        in_features,
        out_features,
        nhead,
        feat_drop=0.0,
        attn_drop=0.5,
        negative_slope=0.2,
        residual=False,
        activation=None,
        alpha=0.0,
    ):
        super(myGATConv, self).__init__()
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.edge_feats = edge_feats
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.edge_emb = nn.Parameter(torch.zeros(size=(num_etypes, edge_feats)))
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * nhead))
        self.W_e = nn.Parameter(torch.FloatTensor(edge_feats, edge_feats * nhead))

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features)))
        self.a_e = nn.Parameter(torch.zeros(size=(1, nhead, edge_feats)))

        self.mhspmm = MultiHeadSpMM()

        self.feat_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.act = None if activation is None else get_activation(activation)

        if residual:
            self.residual = nn.Linear(in_features, out_features * nhead)
        else:
            self.register_buffer("residual", None)
        self.reset_parameters()
        self.alpha = alpha
    # 给参数做一个初始化
    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.a_e)
        reset(self.W)
        reset(self.W_e)
        reset(self.edge_emb)

    def forward(self, graph, x, res_attn=None):
        x = self.feat_drop(x)
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_features)
        h[torch.isnan(h)] = 0.0
        e = torch.matmul(self.edge_emb, self.W_e).view(-1, self.nhead, self.edge_feats)

        row, col = graph.edge_index
        row = row.type(torch.long)
        col = col.type(torch.long)
        graph.edge_index = (row,col)
        tp = graph.edge_type
        h_l = (self.a_l * h).sum(dim=-1)[row]
        h_r = (self.a_r * h).sum(dim=-1)[col]
        h_e = (self.a_e * e).sum(dim=-1)[tp]
        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(graph, edge_attention)
        edge_attention = self.dropout(edge_attention)
        if res_attn is not None:
            edge_attention = edge_attention * (1 - self.alpha) + res_attn * self.alpha
        out = self.mhspmm(graph, edge_attention, h)
        if self.residual:
            res = self.residual(x)
            out += res
        if self.act is not None:
            out = self.act(out)
        return out, edge_attention.detach()


class ClassifyModel(nn.Module):
    def __init__(self, input_dim, hiden_dim1=64, hiden_dim2=16, output_dim=2):
        super(ClassifyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hiden_dim1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(hiden_dim1, hiden_dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(hiden_dim2, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hiden_dim1=4, output_dim=128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hiden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hiden_dim1, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x



def get_train_cli_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN model.")
    if "--model" in sys.argv:
        model_param_name, task_param_name, data_path_param_name = "--model", "--task", "--data_path"
    else:
        model_param_name, task_param_name, data_path_param_name = "model", "task", "data_path"

    parser.add_argument(
        model_param_name,
        type=str,
        default= "simpleHGN",
        help="GNN model type to train.",
    )
    parser.add_argument(
        task_param_name,
        type=str,
        default = "VDM",
        help="Task to train model for.",
    )
    parser.add_argument(data_path_param_name, type=str, help="Directory containing the task data.")
    parser.add_argument(
        "--in_dims",
        dest="in_dims",
        type=int,
        default=100,
        help="the nodes size",
    )
    parser.add_argument(
        "--num_etypes",
        dest="num_etypes",
        type=int,
        default=4,
        help="the types of egde",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default="trained_model",
        help="Path in which to store the trained model and log.",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=25,
        help="Maximal number of epochs to continue training without improvement.",
    )
    parser.add_argument(
        "--seed", dest="random_seed", type=int, default=0, help="Random seed to use.",
    )
    parser.add_argument(
        "--run-test",
        dest="run_test",
        action="store_true",
        default=True,
        help="Run on testset after training.",
    )
    return parser