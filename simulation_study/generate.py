import torch
from torch import distributions as D

def generate_data(true=False, test=False, return_noise=True, **kwargs):
    prior = kwargs['prior']
    if true:
        mb_size = kwargs['true_mb_size'] 
    elif test:
        mb_size = kwargs['test_mb_size'] 
    else:
        mb_size = kwargs['favi_mb_size']
    sigma = kwargs['sigma']
    device = kwargs['device']

    angles = prior.sample((mb_size,))
    #squared_angles = torch.square(angles) #optional
    noise = D.Normal(0, sigma).sample((mb_size,))
    data = torch.stack([torch.cos(angles+noise), torch.sin(angles+noise)]).T
    if return_noise:
        return angles.to(device), data.to(device), noise.to(device)
    else:
        return angles.to(device), data.to(device)
