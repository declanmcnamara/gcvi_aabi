import torch
from torch import distributions as D
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from copy import deepcopy
from utils import MyGetJacobian
from generate import generate_data
from modules.custom_lr_scheduler import CustomOptim
from modules.dense import DenseEncoder1Layer, LinearEncoder
import hydra

def setup(cfg):
    # CONFIGURE
    seed_value = cfg.seed
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    # HYPERPARAMETERS & INITIALIZATION
    sigma = cfg.data.sigma
    device = cfg.training.device
    epochs = cfg.training.epochs
    angle_min = cfg.data.angle_min*math.pi/180
    angle_max = cfg.data.angle_max*math.pi/180
    # prior = D.Uniform(angle_min, math.sqrt(angle_max))
    prior = D.Uniform(angle_min, 2*math.pi)
    true_mb_size = cfg.training.true_mb_size
    favi_mb_size = cfg.training.favi_mb_size
    test_mb_size = cfg.training.test_mb_size

    kwargs = {
        'sigma': sigma,
        'device': device,
        'epochs': epochs,
        'prior': prior,
        'true_mb_size': true_mb_size,
        'favi_mb_size': favi_mb_size,
        'test_mb_size': test_mb_size
    }

    # SET UP MODELS AND TRAINER(S)
    true_theta, true_x, true_noise = generate_data(true=True, return_noise=True, **kwargs)
    true_theta, true_x = true_theta.to(device), true_x.to(device)

    test_theta, test_x, test_noise = generate_data(test=True, return_noise=True, **kwargs)
    test_theta, test_x = test_theta.to(device), test_x.to(device)

    if cfg.encoder.n_hidden_layer == 1:
        encoder = DenseEncoder1Layer(true_x.shape[-1], 2, cfg.encoder.hidden_dim, cfg.encoder.n_hidden_layer)
    else:
        pass
    encoder = encoder.to(device)

    # SET UP LINEAR MODEL IN WEIGHTS -- LAZY TRAINING
    init_jacobian = MyGetJacobian(encoder, true_x)
    init_flat_param_vec = torch.cat([x.flatten() for x in list(encoder.parameters())]).clone().detach()
    init_model = deepcopy(encoder)
    init_model = init_model.to(device)
    lin_encoder = LinearEncoder(true_x.shape[-1], 2, cfg.encoder.hidden_dim, cfg.encoder.n_hidden_layer, init_model, init_jacobian, init_flat_param_vec)
    lin_encoder = lin_encoder.to(device)
    
    kwargs['encoder'] = encoder
    kwargs['lin_encoder'] = lin_encoder
    logger_string = 'loss={},sigma={},lr={},width={},nlayer={}'.format(cfg.training.loss, sigma, cfg.training.lr, cfg.encoder.hidden_dim, cfg.encoder.n_hidden_layer)
    encoder.to(device)
    lin_encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.training.lr)
    kwargs['optimizer'] = optimizer
    scheduler = CustomOptim(optimizer, start_lr=cfg.training.lr)
    lin_optimizer = torch.optim.Adam([lin_encoder.param_vec_flatten], lr=cfg.training.lr)
    kwargs['lin_optimizer'] = lin_optimizer
    lin_scheduler = CustomOptim(lin_optimizer, start_lr=cfg.training.lr)

    # Select loss function
    loss_name = cfg.training.loss
    kwargs['loss'] = loss_name

    return (true_theta, 
            true_x,
            true_noise,
            test_theta,
            test_x,
            test_noise, 
            logger_string,
            encoder,
            optimizer,
            scheduler,
            lin_encoder,
            lin_optimizer,
            lin_scheduler,
            kwargs)

    




