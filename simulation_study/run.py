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
from utils import MyGetJacobian
from einops import rearrange
#torch.autograd.set_detect_anomaly(True)

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
    torch.save(encoder.state_dict(), '{}/simulation_study/{}/init_weights.pth'.format(cfg.log_dir, logger_string))

    loss_name = kwargs['loss']

    # Logs
    training_losses = []
    test_losses = []
    angle_estimates = []
    jacobian_distances = []
    ntk_distances = []
    evals = []

    # Initial jacobian
    init_jacobian = lin_encoder.Jf0
    init_ntk = torch.bmm(init_jacobian, rearrange(init_jacobian, 'n_obs out_dim n_param -> n_obs n_param out_dim'))

    # Fit the encoder
    for j in range(kwargs['epochs']):
        scheduler.zero_grad()
        loss = loss_choice(loss_name, **kwargs)
        print('Loss iter {} is {}'.format(j, loss))
        loss.backward()
        if torch.cat([x.grad.flatten() for x in list(encoder.parameters())]).isnan().any():
            continue
        scheduler.step_and_update_lr()

        # Logging
        training_losses.append(loss.item())

        if (j+1) % 25 == 0:
            eta = encoder(true_x)
            kappa = torch.sqrt(torch.square(eta).sum(1))
            cosines = eta[:,0]/kappa
            sines = eta[:,1]/kappa
            curr_angle_estimates = torch.atan2(sines, cosines)
            angle_estimates.append(curr_angle_estimates.detach().cpu().numpy())
        if (j+1) % 250 == 0:
            # Change in Jacobian
            current_jacobian = MyGetJacobian(encoder, true_x)
            diffs = current_jacobian-init_jacobian
            norms = torch.norm(diffs, 'fro', dim=(1,2))
            avg_norm = norms.mean(-1) #average over datapoints x we're tracking
            jacobian_distances.append(avg_norm.item())

            # Change in NTK
            ntk = torch.bmm(current_jacobian, rearrange(current_jacobian, 'n_obs out_dim n_param -> n_obs n_param out_dim'))
            diffs = ntk-init_ntk
            norms = torch.norm(diffs, 'fro', dim=(1,2))
            avg_norm = norms.mean(-1) #average over datapoints x we're tracking
            ntk_distances.append(avg_norm.item())

            # Spectrum, look at eigenvalues
            eigvals = torch.real(torch.linalg.eigvals(diffs)).abs().mean(0)
            evals.append(eigvals.detach().numpy())


        if (j+1) % 500 == 0:
            lps = encoder.get_log_prob(test_theta, test_x)
            test_losses.append((-1*lps.mean()).item())

        if (j+1) % 5000 == 0:
            np.save('{}/simulation_study/{}/losses.npy'.format(cfg.log_dir, logger_string), np.array(training_losses))
            np.save('{}/simulation_study/{}/test_losses.npy'.format(cfg.log_dir, logger_string), np.array(test_losses))
            np.save('{}/simulation_study/{}/angle_estimates.npy'.format(cfg.log_dir, logger_string), np.stack(angle_estimates))
            np.save('{}/simulation_study/{}/true_angles.npy'.format(cfg.log_dir, logger_string), true_theta.numpy())
            np.save('{}/simulation_study/{}/jacobians.npy'.format(cfg.log_dir, logger_string), jacobian_distances)
            np.save('{}/simulation_study/{}/ntks.npy'.format(cfg.log_dir, logger_string), ntk_distances)
            np.save('{}/simulation_study/{}/evals.npy'.format(cfg.log_dir, logger_string), np.stack(evals))
            np.save('{}/simulation_study/{}/init_jacobian.npy'.format(cfg.log_dir, logger_string), init_jacobian.numpy())
            torch.save(encoder.state_dict(), '{}/simulation_study/{}/weights.pth'.format(cfg.log_dir, logger_string))

    
if __name__ == "__main__":
   main()
