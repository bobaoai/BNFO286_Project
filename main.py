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
    p_list, source_list, target_list = load_json(r'C:\Users\hovea\Documents\GitHub\BNFO286\preprocessed_data.json')
    netprop = NetworkPropagation(p_list, source_list, target_list, lossFn = F.mse_loss, optimizer = torch.optim.Adam)
    #netprop.loadNetwork('test.pt')
    netprop.train(1000)
    netprop.saveNetwork('test.pt')