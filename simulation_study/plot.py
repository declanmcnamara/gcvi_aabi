import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import math
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("../")
import numpy as np 
import scienceplots
# -- plotting -- 
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import hydra
from omegaconf import DictConfig
import random
import pandas as pd
from setup import setup
from utils import log_pos_dens_at
plt.style.use('science')
matplotlib.rcParams.update({'font.size': 30}) 


def fixed_sigma_width_plot_lin(cfg, this_sigma, widths, theta_grid, bin_width, test_theta, test_x, **kwargs):
    all_losses = []
    all_test_losses = []
    all_angle_estimates = []
    all_true_angles = []
    for w in widths:
        access_string = 'loss={},sigma={},lr={},width={},nlayer=1'.format(cfg.training.loss, this_sigma, 1e-4, w)
        try:
            losses = np.load('{}/simulation_study/{}/losses.npy'.format(cfg.log_dir, access_string))
            all_losses.append(losses)

            test_losses = np.load('{}/simulation_study/{}/test_losses.npy'.format(cfg.log_dir, access_string))
            all_test_losses.append(test_losses)
        except:
            pass

        try:
            angle_estimates = np.load('{}/simulation_study/{}/angle_estimates.npy'.format(cfg.log_dir, access_string))
            all_angle_estimates.append(angle_estimates)
        except:
            pass

        true_angles = np.load('{}/simulation_study/{}/true_angles.npy'.format(cfg.log_dir, access_string))

        all_true_angles.append(true_angles)

    all_losses_lin = []
    all_test_losses_lin = []
    all_angle_estimates_lin = []
    all_true_angles_lin = []
    for w in widths:
        access_string = 'loss={},sigma={},lr={},width={},nlayer=1'.format(cfg.training.loss, this_sigma, 1e-4, w)
        try:
            losses = np.load('{}/simulation_study/{}/losses_lin.npy'.format(cfg.log_dir, access_string))
            all_losses_lin.append(losses)

            test_losses = np.load('{}/simulation_study/{}/test_losses_lin.npy'.format(cfg.log_dir, access_string))
            all_test_losses_lin.append(test_losses)
        except:
            pass

        try:
            angle_estimates = np.load('{}/simulation_study/{}/angle_estimates_lin.npy'.format(cfg.log_dir, access_string))
            all_angle_estimates_lin.append(angle_estimates)
        except:
            pass

        true_angles = np.load('{}/simulation_study/{}/true_angles_lin.npy'.format(cfg.log_dir, access_string))

        all_true_angles_lin.append(true_angles)

    # Get vline for true posterior
    log_posterior_ground_truths = log_pos_dens_at(test_theta, theta_grid, test_x, bin_width, **kwargs)
    mean_posterior_nll = -1*log_posterior_ground_truths.mean()

    # Plot training losses
    fig, ax = plt.subplots(figsize=(10,10))
    for j in range(len(all_losses)):
        ax.plot(np.arange(len(all_losses[j])), gaussian_filter1d(all_losses[j], 10.), label='{}'.format(widths[j]), linewidth=2.0)
        ax.plot(np.arange(len(all_losses_lin[j])), gaussian_filter1d(all_losses_lin[j], 10.), label='{}-lin'.format(widths[j]), linewidth=2.0)
    
    plt.legend(title='Width')
    plt.xlabel('Gradient Step')
    plt.ylabel('NLL')
    plt.ylim(0.70, 0.75)
    plt.savefig('./simulation_study/figures/sigma={},width={},losses_lin.png'.format(this_sigma, widths[0]))

    # Plot testing losses
    fig, ax = plt.subplots(figsize=(10,10))
    for j in range(len(all_losses)):
        if j == len(all_losses)-1:
            ax.plot(5000*np.arange(len(all_test_losses[j])), all_test_losses[j], label='{}'.format(widths[j]), linewidth=2.0)
        ax.plot(5000*np.arange(len(all_test_losses_lin[j])), all_test_losses_lin[j], label='{}-lin'.format(widths[j]), linewidth=2.0)
    ax.axhline(mean_posterior_nll.item(), c='r', linestyle='dashed')
    plt.legend(title='Width')
    plt.xlabel('Gradient Step')
    plt.ylabel('NLL')
    plt.savefig('./simulation_study/figures/sigma={},width={},testlosses_lin.png'.format(this_sigma, widths[0]))

    # Plot testing losses zoomed
    fig, ax = plt.subplots(figsize=(10,10))
    for j in range(len(all_losses)):
        if j == len(all_losses)-1:
            ax.plot(5000*np.arange(len(all_test_losses[j])), all_test_losses[j], label='{}'.format(widths[j]), linewidth=2.0)
        ax.plot(5000*np.arange(len(all_test_losses_lin[j])), all_test_losses_lin[j], label='{}-lin'.format(widths[j]), linewidth=2.0)
    ax.axhline(mean_posterior_nll.item(), c='r', linestyle='dashed')
    plt.legend(title='Width')
    plt.xlabel('Gradient Step')
    plt.ylabel('NLL')
    plt.ylim(0.705, 0.735)
    plt.savefig('./simulation_study/figures/sigma={},width={},testlosses_lin_zoom.png'.format(this_sigma, widths[0]))



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

    cfg.training.lr=1e-4

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

    sigmas = [5e-1]
    widths = [64,128,256,512,1024,2048,4096]
    log_dir = cfg.log_dir

    bin_width=1e-3
    theta_grid = torch.arange(1e-6, 2*math.pi, bin_width)

    for sigma in sigmas:
        for width in widths:
            try:
                fixed_sigma_width_plot_lin(cfg, sigma, [width], theta_grid, bin_width, test_theta, test_x, **kwargs)
            except:
                print('Exception on {}'.format(sigma))
                continue

if __name__ == "__main__":
   main()



