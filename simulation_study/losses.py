import torch
from generate import generate_data

def favi_loss(**kwargs):
    device = kwargs['device']
    encoder = kwargs['encoder']

    z, x, _ = generate_data(**kwargs)
    lps = encoder.get_log_prob(z, x)

    return -1*lps.mean()
