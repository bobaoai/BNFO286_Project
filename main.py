from NetworkPropagation import NetworkPropagation
import json
import torch
import torch.nn.functional as F

def store_json(address, dic):
    with open(address, 'w') as fp:
        json.dump(dic, fp)

def load_json(address):
    with open(address, 'r') as f:
        return json.load(f)
    
if __name__ == '__main__':
    p_list, source_list, target_list = load_json(r'preprocessed_data.json')
    netprop = NetworkPropagation(p_list, source_list, target_list, optimizer = torch.optim.Adam, gpu = True, adaptiveEdgeWeights = True)
    netprop.train(1000)
    output = netprop.evaluate()
    print(max(output), min(output))
    store_json(r'raw_output_data.json', output)