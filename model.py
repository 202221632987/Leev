import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from train_utils import myGATConv,ClassifyModel,MLP
from cogdl.models import BaseModel
from cogdl.data import Graph
class LeeV(BaseModel):
    def __init__(
            self,
            in_dims=200,
            edge_dim=64,
            num_etypes=3,
            num_hidden=128,
            num_layers=2,
            heads=[8, 8, 1],
            feat_drop=0.5,
            attn_drop=0.5,
            negative_slope=0.05,
            residual=True,
            alpha=0.05,
            dim = 128,
    ):
        super(LeeV, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.g_ast = None
        self.g_cdfg = None
        self.num_layers = num_layers
        self.gat_layers_ast = nn.ModuleList()
        self.gat_layers_cdfg = nn.ModuleList()
        self.activation = F.elu
        self.dim = dim
        self.gat_layers_cdfg.append(
            myGATConv(
                edge_dim,
                num_etypes,
                in_dims,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )
        for l in range(1, num_layers):
            self.gat_layers_cdfg.append(
                myGATConv(
                    edge_dim,
                    num_etypes,
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    alpha=alpha,
                ).to("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        self.gat_layers_cdfg.append(
            myGATConv(
                edge_dim,
                num_etypes,
                num_hidden * heads[-2],
                self.dim,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                alpha=alpha,
            ).to("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        self.score_mlp = MLP(input_dim=128)
        self.transformation_mlp = MLP(input_dim=128)
        self.classfymodel = ClassifyModel(128)
        self.softmax = torch.nn.functional.softmax


    def build_g_feat(self, A):
        edge2type = {}
        edges = []
        for k, mat in enumerate(A):
            edges.append(mat.cpu().numpy())
            for u, v in zip(*edges[-1]):
                edge2type[(u, v)] = k
        edges = np.concatenate(edges, axis=1)
        edges = torch.tensor(edges).to(self.device)

        g = Graph(edge_index=edges,)
        g = g.to(self.device)
        e_feat = []
        for u, v in zip(*g.edge_index):
            u = u.cpu().item()
            v = v.cpu().item()
            e_feat.append(edge2type[(u, v)])
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(self.device)
        g.edge_type = e_feat
        return g

    def forward(self, data):
        logits_total = []
        length = len(data)
        for i in range(length):
            data_cpg = data[i]
            A_cdfg = data_cpg["adjacency_lists"]
            X_cdfg = data_cpg['node_features']
            h_cdfg = X_cdfg
            if self.g_cdfg is None:
                self.g_cdfg = self.build_g_feat(A_cdfg)
            res_attn = None
            for l in range(self.num_layers):
                h_cdfg, res_attn = self.gat_layers_cdfg[l](self.g_cdfg, h_cdfg,res_attn=res_attn)
                h_cdfg = h_cdfg.flatten(1)
            logits_cdfg, _ = self.gat_layers_cdfg[-1](self.g_cdfg,h_cdfg, res_attn=None)
            logits_cdfg = logits_cdfg / (torch.max(torch.norm(logits_cdfg, dim=1, keepdim=True), self.epsilon))
            logit_vector = logits_cdfg[0,:]
            logit = logit_vector
            self.g_ast = None
            self.g_cdfg = None

            logits_total.append(logit)
        logits_total = torch.stack(logits_total)
        logits = self.classfymodel(logits_total)
        return logits
