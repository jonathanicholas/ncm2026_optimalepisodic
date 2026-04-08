import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from modules import *





"""
Set environment
"""

# set random seed
seed = 15
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# parse args
parser = ArgParser()
args = parser.args

# set experiment path
exp_path = os.path.join(args.path, f'exp_{args.init_num_items}_{args.jobid}')





"""
Load data
"""

# load data
with open(os.path.join(exp_path, 'data_simulation.p'), 'rb') as f:
    data_pickle = pickle.load(f)
print(data_pickle.keys())

# initialize new data
data_json = {
    'pairs': [],
    'features': [],
    'values': [],
    'options': [],
    'relevances': [],
    'offer_values': [],
    'actions': [],
}





"""
Global variables
"""

feature_map = {
    0: [0, 0, 0, 0],
    1: [1, 0, 0, 0],
    2: [0, 1, 0, 0],
    3: [1, 1, 0, 0],
    4: [0, 0, 1, 0],
    5: [1, 0, 1, 0],
    6: [0, 1, 1, 0],
    7: [1, 1, 1, 0],
    8: [0, 0, 0, 1],
    9: [1, 0, 0, 1],
    10: [0, 1, 0, 1],
    11: [1, 1, 0, 1],
    12: [0, 0, 1, 1],
    13: [1, 0, 1, 1],
    14: [0, 1, 1, 1],
    15: [1, 1, 1, 1],
}

string_to_item_map = {
    "Object_Land_Solid_Small": 0,
    "Animal_Land_Solid_Small": 1,
    "Object_Sea_Solid_Small": 2,
    "Animal_Sea_Solid_Small": 3,
    "Object_Land_Pattern_Small": 4,
    "Animal_Land_Pattern_Small": 5,
    "Object_Sea_Pattern_Small": 6,
    "Animal_Sea_Pattern_Small": 7,
    "Object_Land_Solid_Large": 8,
    "Animal_Land_Solid_Large": 9,
    "Object_Sea_Solid_Large": 10,
    "Animal_Sea_Solid_Large": 11,
    "Object_Land_Pattern_Large": 12,
    "Animal_Land_Pattern_Large": 13,
    "Object_Sea_Pattern_Large": 14,
    "Animal_Sea_Pattern_Large": 15,
}





"""
Process data
"""

for i in range(len(data_pickle['action_seqs'])):

    if data_pickle['offer_values'][i] != 0:

        # items
        items_ep = data_pickle['items'][i]
        data_json['pairs'].append(items_ep)

        # features
        features_ep = [feature_map[item] for item in items_ep]
        data_json['features'].append(features_ep)

        # values
        values_ep = data_pickle['values'][i]
        data_json['values'].append(values_ep)

        # options
        query_feature_ep = data_pickle['query_features'][i]
        data_json['options'].append(query_feature_ep)

        # relevances
        relevance_ep = data_pickle['relevances'][i]
        data_json['relevances'].append(relevance_ep)

        # offer_values
        offer_value_ep = data_pickle['offer_values'][i]
        data_json['offer_values'].append(offer_value_ep)

        # actions
        action_seq_ep = data_pickle['action_seqs'][i]
        data_json['actions'].append(action_seq_ep)





"""
custom json encoder
"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.float64)):
            return obj.item()
        return super().default(obj)





"""
Save data
"""

# set ouput path
output_path = os.path.join(args.path, f'data_json')

# save data
with open(os.path.join(output_path, f'data_{args.init_num_items}_{args.jobid}.json'), 'w') as file:
    json.dump(data_json, file, cls = NumpyEncoder)