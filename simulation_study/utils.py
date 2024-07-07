import torch
from generate import generate_data
#from modules.dense import DenseEncoder, LinearEncoder
from einops import rearrange, pack, unpack, repeat, reduce
from torch.func import functional_call, vmap, jacrev
from torch import logsumexp
import math
import torch.distributions as D

def processor(outputs):
    '''Given outputs of set transformer, cast
    to parameters for encoder model.'''
    means = outputs[:,0].view(-1,1) 
    log_sds = outputs[:,1].view(-1,1).clamp(-10., 10.)
    return means, torch.exp(log_sds)

def processor_batch(outputs):
    means = outputs[..., 0].unsqueeze(-1)
    log_sds = outputs[..., 1].clamp(-10., 10.).unsqueeze(-1)
    return means, torch.exp(log_sds)


# Tools for computing the Jacobian for lazy training examination
def MyGetJacobian(encoder, fixed_x):
	'''
	My new wrapper for computing Jacobian of encoder w/r/t weights $\phi$.
	Returns a flattened Jacobian corresponding to flattened parameter vector $\phi$.
	'''

	#fixed_x.requires_grad_(True)
	curr_params = encoder.state_dict()
	def get_output(params, any_x):
		return functional_call(encoder, params, any_x)
	
	jac1 = vmap(jacrev(get_output, chunk_size=1), (None, 0))(curr_params, fixed_x)
	all_jacobians = list(jac1.values())
	for j in range(len(all_jacobians)):
		jac = all_jacobians[j]
		if len(jac.shape) > 3:
			jac = rearrange(jac, 'b out_dim width height -> b out_dim (width height)')
			all_jacobians[j] = jac
	jacobian, _ = pack(all_jacobians, 'b out_dim *')
	return jacobian

### Code for computing exact posterior
def unnormalized_posterior_log_density(theta, x, **kwargs):
	'''
	Given 1d theta in [0,2pi]
	and   2d x
	computes posterior p(theta | x).
	'''

	theta_plus_z = torch.atan2(x[...,1], x[...,0]) % (2*math.pi)
	log_p_theta = kwargs['prior'].log_prob(theta)

	# theta_plus_z should be at most pi radians away from theta
	# e.g. if theta_plus_z = 6 rad, and theta = 0 rad, then
	# D.Normal(theta, sigma).log_prob(theta_plus_z) should be large
	# We account for this in construction of the to_eval object

	to_eval = repeat(theta_plus_z, 'n_obs -> n_obs n_theta', n_theta=theta.shape[0])
	to_eval = (to_eval - theta) % (2*math.pi)
	to_eval = torch.where(to_eval > math.pi, -1*(2*math.pi-to_eval), to_eval)

	log_p_theta_plus_z_given_theta = D.Normal(0., kwargs['sigma']).log_prob(to_eval)
	log_p_theta = repeat(log_p_theta, 'n_theta -> n_obs n_theta', n_obs=theta_plus_z.shape[0])
	return log_p_theta + log_p_theta_plus_z_given_theta

def exact_posterior_log_density(theta, x, bin_width, **kwargs):
	log_joint = unnormalized_posterior_log_density(theta, x, **kwargs) #n_obs x n_grid_vals
	log_marginal = torch.log(torch.tensor(bin_width)) + reduce(log_joint, 'n_obs n_theta_grid -> n_obs',  logsumexp)
	log_pos = (log_joint.T - log_marginal).T
	return log_pos

def log_pos_dens_at(eval_thetas, theta_grid, eval_xs, bin_width, **kwargs):
	log_pos = exact_posterior_log_density(theta_grid, eval_xs, bin_width, **kwargs)
	rgrid = repeat(theta_grid, 'n_theta_grid -> n_theta_grid n_eval_theta', n_eval_theta = eval_thetas.shape[0])
	abs_diffs_argmin = torch.abs(eval_thetas - rgrid).argmin(0)
	out = torch.diag(log_pos[:, abs_diffs_argmin])
	return out

