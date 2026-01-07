import torch
import numpy as np
import tqdm
from scipy import integrate

num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           y,  # Conditional information (y0, fixed)
                           batch_size, 
                           num_steps=500, 
                           device='cuda', 
                           eps=1e-3):

  y = y.to(device)
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]

  
  y_t = torch.randn_like(y) * marginal_prob_std(t)[:, None, None, None]

  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x

  mask = torch.ones_like(x)
  mask[:, :, :, 16:] = 0.

  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)

      std = marginal_prob_std(batch_time_step)[:, None, None, None]
      std2 = std**2 + 1e-12

      #score_model [B,2,28,28]
      score_xy = score_model(x, y_t, batch_time_step)
      score_x = score_xy[:, 0:1, :, :]
      score_y = score_xy[:, 1:2, :, :]

      # x update：only score_x（
      x_mean = x + (g**2)[:, None, None, None] * score_x * step_size
      x = x_mean + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

      # y_t ）
      conditional_term = (y - y_t) * mask / std2
      y_mean = y_t + (g**2)[:, None, None, None] * (score_y + 0.576568555 * conditional_term) * step_size
      y_t = y_mean + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(y_t)

  
  return x_mean






signal_to_noise_ratio = 0.16
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y,  # Conditional information (y0, fixed)
               batch_size, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3):

  y = y.to(device)  # y0 (fixed observation)

  # Initialize variables
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]

  # ✅ 新增：y_t 从噪声开始（joint state）
  y_t = torch.randn_like(y) * marginal_prob_std(t)[:, None, None, None]

  time_steps = np.linspace(1., eps, num_steps)
  step_size = torch.tensor(time_steps[0] - time_steps[1], device=device, dtype=torch.float32)

  x = init_x
  mask = torch.ones_like(x)
  mask[:, :, :, 16:] = 0.

  w = 0.576568555  # 你原来的权重先保留（最小改动）

  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      std = marginal_prob_std(batch_time_step)[:, None, None, None]
      std2 = std**2 + 1e-12

      # =========================
      # Corrector step (Langevin)
      # =========================
      score_xy = score_model(x, y_t, batch_time_step)   # [B,2,H,W]
      score_x = score_xy[:, 0:1, :, :]
      score_y = score_xy[:, 1:2, :, :]

      # ✅ 条件项只加在 y 上：∇_{y_t} log p(y0 | y_t)
      conditional_term_y = (y - y_t) * mask / std2

      # x 的 Langevin 用 score_x
      grad_x = score_x
      grad_x_norm = torch.norm(grad_x.reshape(grad_x.shape[0], -1), dim=-1).mean()

      # y 的 Langevin 用 score_y + cond
      grad_y = score_y + w * conditional_term_y
      grad_y_norm = torch.norm(grad_y.reshape(grad_y.shape[0], -1), dim=-1).mean()

      noise_norm = np.sqrt(np.prod(x.shape[1:]))

      langevin_step_size_x = 2 * (snr * noise_norm / (grad_x_norm + 1e-12))**2
      langevin_step_size_y = 2 * (snr * noise_norm / (grad_y_norm + 1e-12))**2

      x = x + langevin_step_size_x * grad_x + torch.sqrt(2 * langevin_step_size_x) * torch.randn_like(x)
      y_t = y_t + langevin_step_size_y * grad_y + torch.sqrt(2 * langevin_step_size_y) * torch.randn_like(y_t)

      # =========================
      # Predictor step (EM)
      # =========================
      g = diffusion_coeff(batch_time_step)

      score_xy = score_model(x, y_t, batch_time_step)
      score_x = score_xy[:, 0:1, :, :]
      score_y = score_xy[:, 1:2, :, :]

      conditional_term_y = (y - y_t) * mask / std2

      x_mean = x + (g**2)[:, None, None, None] * score_x * step_size
      y_mean = y_t + (g**2)[:, None, None, None] * (score_y + w * conditional_term_y) * step_size

      x = x_mean + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
      y_t = y_mean + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(y_t)

  return x_mean



## The error tolerance for the black-box ODE solver

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                y,  # Conditional information
                batch_size, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                z=None,
                eps=1e-3):
  """Generate samples from conditional score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    y: The conditional input, e.g., a masked version of the data.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Ensure `y` is on the same device as the sampler
  y = y.to(device)
  
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the conditional score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, y, time_steps)  # Pass `y` to the score model
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x
