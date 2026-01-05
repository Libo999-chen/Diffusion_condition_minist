import torch
import torch.nn as nn
import numpy as np
import functools
from generate_samples import *
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def marginal_prob_std(t, sigma): 
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    
    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
        y: The conditioning variable (e.g., weather information).
    
    Returns:
        The standard deviation.
    """
    t = torch.tensor(t, device=device)
    

    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE for the conditional case.
    
    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
        y: The conditioning variable (e.g., weather information).
    
    Returns:
        The vector of diffusion coefficients.
    """
     # Example: using the mean of y as a scaling factor
    return torch.tensor(sigma**t, device=device)

sigma = 25.0  

# Define partial functions for conditional SDE calculations
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)





def loss_fn(model, x, y, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models (Conditional).

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.
    y: The conditioning variable (e.g., labels, features).
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, y, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
  
  return loss



def samples_plt(score_model, data_loader, device, sampler, sample_batch_size, marginal_prob_std_fn, diffusion_coeff_fn,y,x):
    """
    Generates and visualizes samples using a specified sampler.

    Parameters:
    - score_model: The pre-trained model used for generating samples.
    - data_loader: The data loader used to fetch data.
    - device: The device to run the model (e.g., 'cuda' or 'cpu').
    - sampler: The sampler to use for generating samples (e.g., ode_sampler).
    - sample_batch_size: The batch size used for sampling.
    - marginal_prob_std_fn: The function used for marginal probability standard deviation.
    - diffusion_coeff_fn: The function used for diffusion coefficient.
    """
    # Define `y` as the masked version of the data
    # Generate samples using the specified sampler
    samples = sampler(score_model = score_model,
                      marginal_prob_std = marginal_prob_std_fn,
                      diffusion_coeff = diffusion_coeff_fn,
                      y = y,  # Pass the conditional input `y`
                      batch_size = sample_batch_size,
                      device = device)

    # Clamp the values to ensure they are within the range [0, 1]
    samples = samples.clamp(0.0, 1.0)

    # Create grids for visualization
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    y_sample_grid = make_grid(y, nrow=int(np.sqrt(sample_batch_size)))
    x_sample_grid = make_grid(x, nrow=int(np.sqrt(sample_batch_size)))

    # Create a 1x3 grid of plots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust the size as needed

    # Plot the generated samples
    axes[0].axis('off')
    axes[0].imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    axes[0].set_title('Generated Samples')

    # Plot the conditional samples (masked input `y`)
    axes[1].axis('off')
    axes[1].imshow(y_sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    axes[1].set_title('Conditional Samples (y)')

    # Plot the original samples (input `x`)
    axes[2].axis('off')
    axes[2].imshow(x_sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    axes[2].set_title('Original Samples (x)')

    # Display the figure
    plt.tight_layout()
    plt.show()
    return plt.gcf()









#@title Define the likelihood function (double click to expand or collapse)

def prior_likelihood(z, sigma):
  """The likelihood of a Gaussian distribution with mean zero and
      standard deviation sigma."""
  shape = z.shape
  N = np.prod(shape[1:])
  return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x,
                   y,
                   score_model,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=64,
                   device='cuda',
                   eps=1e-5):
  """Compute the likelihood with probability flow ODE.

  Args:
    x: Input data.
    score_model: A PyTorch model representing the score-based model.
    marginal_prob_std: A function that gives the standard deviation of the
      perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the
      forward SDE.
    batch_size: The batch size. Equals to the leading dimension of `x`.
    device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
    eps: A `float` number. The smallest time step for numerical stability.

  Returns:
    z: The latent code for `x`.
    bpd: The log-likelihoods in bits/dim.
  """

  # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
  #epsilon = torch.randn_like(x)
  epsilon = torch.randn_like(x.to(torch.float32))  # Convert x to float32


  def divergence_eval(sample, y, time_steps, epsilon):
    """Compute the divergence of the score-based model with Skilling-Hutchinson."""
    with torch.enable_grad():
      sample.requires_grad_(True)
      device = sample.device    
      epsilon = epsilon.to(device)
      score_e = torch.sum(score_model(sample, y, time_steps) * epsilon)
      grad_score_e = torch.autograd.grad(score_e, sample)[0]
    return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))

  shape = x.shape

  def score_eval_wrapper(sample,y, time_steps):
    """A wrapper for evaluating the score-based model for the black-box ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    with torch.no_grad():
      score = score_model(sample,y, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def divergence_eval_wrapper(sample, y,time_steps):
    """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
    with torch.no_grad():
      # Obtain x(t) by solving the probability flow ODE.
      sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
      time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
      # Compute likelihood.
      div = divergence_eval(sample,y, time_steps, epsilon)
      return div.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x, y):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((shape[0],)) * t
    sample = x[:-shape[0]]
    logp = x[-shape[0]:]
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, y,time_steps)
    logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, y,time_steps)
    return np.concatenate([sample_grad, logp_grad], axis=0)

  init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
  # Black-box ODE solver
  res = integrate.solve_ivp(ode_func, (eps, 1.), init, args=(y,), rtol=1e-5, atol=1e-5, method='RK45')
  zp = torch.tensor(res.y[:, -1], device=device)
  z = zp[:-shape[0]].reshape(shape)
  delta_logp = zp[-shape[0]:].reshape(shape[0])
  sigma_max = marginal_prob_std(1.)
  prior_logp = prior_likelihood(z, sigma_max)
  bpd = -(prior_logp + delta_logp) / np.log(2)
  N = np.prod(shape[1:])
  bpd = bpd / N + 8.
  return z, bpd
