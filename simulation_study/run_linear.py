import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import torch.distributions as D
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import numpy as np 
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from os.path import exists
import torch.distributions as D
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import random
from setup import setup
from losses import favi_loss
import time
from utils import MyGetJacobian
from generate import generate_data

def loss_choice(loss_name, **kwargs):
    if loss_name == 'favi':
        return favi_loss(**kwargs)
    else:
        raise ValueError('Specify an appropriate loss name string.')


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg : DictConfig) -> None:
    # initialize(config_path="../config")
    # cfg = compose(config_name="canonical")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    # cfg.encoder.hidden_dim = 1024
    # cfg.training.favi_mb_size = 16
    # cfg.encoder.n_hidden_layer = 1
    # cfg.training.device = 'cuda:7'

    (true_theta, 
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
    kwargs) = setup(cfg)

    if not exists('{}/simulation_study'.format(cfg.log_dir, logger_string)):
        os.mkdir('{}/simulation_study'.format(cfg.log_dir, logger_string))
    if not exists('{}/simulation_study/{}'.format(cfg.log_dir, logger_string)):
        os.mkdir('{}/simulation_study/{}'.format(cfg.log_dir, logger_string))
    torch.save(lin_encoder.state_dict(), '{}/simulation_study/{}/init_weights_lin.pth'.format(cfg.log_dir, logger_string))

    kwargs['encoder'] = lin_encoder ##NECESSARY TO USE LINEARIZED ENCODER
    loss_name = kwargs['loss']

    # Logs
    training_losses = []
    test_losses = []
    angle_estimates = []
    jacobian_distances = []

    # Initial jacobian
    init_jacobian = lin_encoder.Jf0

    # Fit the encoder
    for j in range(kwargs['epochs']):
        lin_scheduler.zero_grad()
        loss = loss_choice(loss_name, **kwargs)
        print('Loss iter {} is {}'.format(j, loss))
        loss.backward()
        if lin_encoder.param_vec_flatten.grad.isnan().any():
            print('Found a nan in gradient')
            continue
        lin_scheduler.step_and_update_lr()

        # Logging
        training_losses.append(loss.item())

        if (j+1) % 25 == 0:
            eta = lin_encoder(true_x)
            kappa = torch.sqrt(torch.square(eta).sum(1))
            cosines = eta[:,0]/kappa
            sines = eta[:,1]/kappa
            curr_angle_estimates = torch.atan2(sines, cosines)
            angle_estimates.append(curr_angle_estimates.detach().cpu().numpy())
        if (j+1) % 500 == 0:
            lps = lin_encoder.get_log_prob(test_theta, test_x)
            test_losses.append((-1*lps.mean()).item())

        if (j+1) % 5000 == 0:
            np.save('{}/simulation_study/{}/losses_lin.npy'.format(cfg.log_dir, logger_string), np.array(training_losses))
            np.save('{}/simulation_study/{}/test_losses_lin.npy'.format(cfg.log_dir, logger_string), np.array(test_losses))
            np.save('{}/simulation_study/{}/angle_estimates_lin.npy'.format(cfg.log_dir, logger_string), np.stack(angle_estimates))
            np.save('{}/simulation_study/{}/true_angles_lin.npy'.format(cfg.log_dir, logger_string), true_theta.numpy())
            np.save('{}/simulation_study/{}/jacobians_lin.npy'.format(cfg.log_dir, logger_string), jacobian_distances)
            np.save('{}/simulation_study/{}/init_jacobian_lin.npy'.format(cfg.log_dir, logger_string), init_jacobian.numpy())
            torch.save(lin_encoder.state_dict(), '{}/simulation_study/{}/weights_lin.pth'.format(cfg.log_dir, logger_string))
    
if __name__ == "__main__":
   main()