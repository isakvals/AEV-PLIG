import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.preprocessing import StandardScaler

def init_weights(layer):
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is True:
            torch.nn.init.zeros_(layer.bias)

class GraphDataset(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 ids=None, y=None, graphs_dict=None, y_scaler=None):

        super(GraphDataset, self).__init__(root)
        self.dataset = dataset
        #torch.serialization.add_safe_globals([GraphDataset])
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            #self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])

        else:
            self.process(ids, y, graphs_dict)
            #self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
        
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self._data.y, (self.__len__(),1)))
        self.y_scaler = y_scaler
        self._data.y = [torch.tensor(element[0]).float() for element in self.y_scaler.transform(np.reshape(self._data.y, (self.__len__(),1)))]
        

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, y, graphs_dict):
        assert (len(ids) == len(y)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            label = y[i]
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(x=torch.Tensor(np.array(features)),
                                   edge_index=torch.LongTensor(np.array(edge_index)).T,
                                   edge_attr=torch.Tensor(np.array(edge_features)),
                                   y=torch.FloatTensor(np.array([label])))
            
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        #self.data, self.slices = self.collate(data_list)
        self.save(data_list, self.processed_paths[0])
        #torch.save((self.data, self.slices), self.processed_paths[0])
        


class GraphDatasetPredict(InMemoryDataset):
    def __init__(self, root='data', dataset=None,
                 ids=None, graph_ids=None, graphs_dict=None):

        super(GraphDatasetPredict, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])

        else:
            self.process(ids, graph_ids, graphs_dict)
            self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, graph_ids, graphs_dict):
        assert (len(ids) == len(graph_ids)), 'Number of datapoints and labels must be the same'
        data_list = []
        data_len = len(ids)
        for i in range(data_len):
            pdbcode = ids[i]
            graph_id = graph_ids[i]
            c_size, features, edge_index, edge_features = graphs_dict[pdbcode]
            data_point = Data(x=torch.Tensor(np.array(features)),
                                   edge_index=torch.LongTensor(np.array(edge_index)).T,
                                   edge_attr=torch.Tensor(np.array(edge_features)),
                                   y=torch.IntTensor(np.array([graph_id])))
            
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        self.save(data_list, self.processed_paths[0])


class GraphDatasetInference(InMemoryDataset):
    def __init__(self, ids, graph_ids, graphs_dict, transform=None, pre_transform=None, log=False):
        # Use an empty string or dummy path for root since we don't use disk I/O.
        super(GraphDatasetInference, self).__init__(root='', transform=transform, pre_transform=pre_transform, log=log)
        
        # Process data in memory
        data_list = self._process_data(ids, graph_ids, graphs_dict)
        self.data, self.slices = self.collate(data_list)

    def _process_data(self, ids, graph_ids, graphs_dict):
        data_list = []
        for i in range(len(ids)):
            compound_name = ids[i]
            graph_id = graph_ids[i]
            c_size, features, edge_index, edge_features = graphs_dict[compound_name]
         
            data_point = Data(
                x=torch.tensor(np.array(features), dtype=torch.float),
                edge_index=torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(np.array(edge_features), dtype=torch.float),
                y=torch.tensor([graph_id], dtype=torch.int)
            )
            data_list.append(data_point)
        return data_list

    @property
    def raw_file_names(self):
        # Not used since we're processing in memory
        return []

    @property
    def processed_file_names(self):
        # No processed files will be saved on disk.
        return []

    def download(self):
        # Override to do nothing.
        pass

    def process(self):
        # Override to do nothing; processing is handled in __init__.
        pass