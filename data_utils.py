import numpy as np
import torch
from torch.utils.data import Dataset
class LeeVdataset(Dataset):
    def __init__(self,datapath):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.datapath = datapath
        self.g= self.data_to_g(datapath)


    def __len__(self):
        return len(self.g)


    def __getitem__(self, index):
        return self.g[index]


    def data_to_g(self,data_path):
        print(f"\nbegin to load data from", data_path)
        g_data = self.load_data(data_path)
        print("data is ready")
        return g_data

    def load_data(self,data_file):
        return [self.process_raw_datapoint_ast(datapoint) for datapoint in data_file.read_by_file_suffix()]

    def process_raw_datapoint_ast(self,datapoint):
        graph = datapoint["graph_cdfg"]
        Vitual_node = [0 for x in range(200)]
        node_features = graph["node_features"]
        node_features.insert(0,Vitual_node)
        tensor_node_features = torch.tensor(node_features)
        type_to_adj_list = self.process_raw_adjacency_lists(
            raw_adjacency_lists=graph["adjacency_lists"],node_class = graph["nodes_class"]
        )
        for i in range(len(type_to_adj_list)):
            type_to_adj_list[i] = torch.tensor(type_to_adj_list[i]).to(self.device)
            type_to_adj_list[i] = type_to_adj_list[i].t()
        target_value = float(datapoint["Property"])
        GraphWithPropertySample = {
            "adjacency_lists": type_to_adj_list,
            "node_features": tensor_node_features.to(self.device),
            "target_value": target_value,
            "node_types": graph["nodes_class"]
        }
        return GraphWithPropertySample

    def process_raw_adjacency_lists(self,raw_adjacency_lists,node_class):
        type_to_adj_list = [
            [] for _ in range(5)
        ]

        for step, edges in enumerate(raw_adjacency_lists):
            fwd_edge_type = edges[2]
            type_to_adj_list[fwd_edge_type].append((edges[0]+1, edges[1]+1))
            type_to_adj_list[fwd_edge_type].append((edges[1]+1, edges[0]+1))
        for i in range(len(node_class)):
            type_to_adj_list[node_class[i] + 3].append((0, i + 1))
            type_to_adj_list[node_class[i] + 3].append((i + 1, 0))

        type_to_adj_list = [
            np.array(adj_list, dtype=np.int32)
            if len(adj_list) > 0
            else np.zeros(shape=(0, 2), dtype=np.int32)
            for adj_list in type_to_adj_list
        ]

        return type_to_adj_list






