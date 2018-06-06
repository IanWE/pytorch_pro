import torch

torch.save(net1,'new.pkl') #torch.save(net.state_dict(),'net_params.pkl')
net = torch.load('new.pkl') #net.load_state_dict(torch.load('net_params.pkl'))


