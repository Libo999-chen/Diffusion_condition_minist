import torch
import numpy as np
import tqdm
from scipy import integrate

num_steps =  500#@param {'type':'integer'}



def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           y,  # y0 fixed (masked image)
                           batch_size,
                           num_steps=500,
                           device='cuda',
                           eps=1e-3):

  beta = 0.3
  y = y.to(device)

  t1 = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t1)[:, None, None, None]

  # y_t init noise with std_y = beta * std_x
  #y_t = torch.randn_like(y) * (beta * marginal_prob_std(t1)[:, None, None, None])
  x = init_x

  mask = torch.ones_like(x)
  mask[:, :, :, 16:] = 0.
  y_t = (y + torch.randn_like(y) * (beta * marginal_prob_std(t1)[:, None, None, None]) * mask) * mask

  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]



  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step

      g_x = diffusion_coeff(batch_time_step)                 # [B]
      g_y = beta * g_x                                        # multi-speed

      std_x = marginal_prob_std(batch_time_step)[:, None, None, None]
      std_y = beta * std_x
      std_y2 = std_y**2 + 1e-12

      score_xy = score_model(x, y_t, batch_time_step)
      score_x = score_xy[:, 0:1]
      score_y = score_xy[:, 1:2]

      # x update
      x_mean = x + (g_x**2)[:, None, None, None] * score_x * step_size
      x = x_mean + torch.sqrt(step_size) * g_x[:, None, None, None] * torch.randn_like(x)

      # y update: plug-in term coefficient should be 1 for VE
      conditional_term = (y - y_t) * mask / std_y2
      y_mean = y_t + (g_y**2)[:, None, None, None] * (score_y + conditional_term) * step_size
      #y_t = y_mean + torch.sqrt(step_size) * g_y[:, None, None, None] * torch.randn_like(y_t)
      y_t = y_mean + torch.sqrt(step_size) * g_y[:, None, None, None] * torch.randn_like(y_t) * mask
      y_t = y_t * mask


  return x_mean




signal_to_noise_ratio = 0.16
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               y,  # y0 (fixed)
               batch_size, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3):

  beta = 0.3
  w = 1.0  # VE 下 analytic plug-in term 系数=1

  y = y.to(device)

  t = torch.ones(batch_size, device=device)
  std_x_1 = marginal_prob_std(t)[:, None, None, None]
  std_y_1 = beta * std_x_1

  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * std_x_1

  x = init_x
  mask = torch.ones_like(x)
  mask[:, :, :, 16:] = 0.

  # make y consistent with mask & init y_t from y + noise (only on mask) 
  y = y * mask
  y_t = (y + torch.randn_like(y) * std_y_1 * mask) * mask


  time_steps = np.linspace(1., eps, num_steps)
  step_size = torch.tensor(time_steps[0] - time_steps[1], device=device, dtype=torch.float32)

  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step

      std_x = marginal_prob_std(batch_time_step)[:, None, None, None]
      std_y = beta * std_x
      std_y2 = std_y**2 + 1e-12

      g_x = diffusion_coeff(batch_time_step)             # [B]
      g_y = beta * g_x


      # Corrector (Langevin)

      score_xy = score_model(x, y_t, batch_time_step)    # [B,2,H,W]
      score_x = score_xy[:, 0:1, :, :]
      score_y = score_xy[:, 1:2, :, :]

      conditional_term_y = (y - y_t) * mask / std_y2

      grad_x = score_x
      grad_y = score_y + w * conditional_term_y

      grad_x_norm = torch.norm(grad_x.reshape(grad_x.shape[0], -1), dim=-1).mean()
      grad_y_norm = torch.norm(grad_y.reshape(grad_y.shape[0], -1), dim=-1).mean()

      noise_norm = np.sqrt(np.prod(x.shape[1:]))

      langevin_step_size_x = 2 * (snr * noise_norm / (grad_x_norm + 1e-12))**2
      langevin_step_size_y = 2 * (snr * noise_norm / (grad_y_norm + 1e-12))**2

      x = x + langevin_step_size_x * grad_x + torch.sqrt(2 * langevin_step_size_x) * torch.randn_like(x)

      # Langevin noise only on mask + project back 
      y_t = y_t + langevin_step_size_y * grad_y + torch.sqrt(2 * langevin_step_size_y) * torch.randn_like(y_t) * mask
      y_t = y_t * mask
  

      # Predictor (EM)
      score_xy = score_model(x, y_t, batch_time_step)
      score_x = score_xy[:, 0:1, :, :]
      score_y = score_xy[:, 1:2, :, :]

      conditional_term_y = (y - y_t) * mask / std_y2

      x_mean = x + (g_x**2)[:, None, None, None] * score_x * step_size
      y_mean = y_t + (g_y**2)[:, None, None, None] * (score_y + w * conditional_term_y) * step_size

      x = x_mean + torch.sqrt(step_size) * g_x[:, None, None, None] * torch.randn_like(x)

      # EM noise only on mask + project back
      y_t = y_mean + torch.sqrt(step_size) * g_y[:, None, None, None] * torch.randn_like(y_t) * mask
      y_t = y_t * mask


  return x_mean


## The error tolerance for the black-box ODE solver

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                y,  # y0 (fixed)
                batch_size, 
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                z=None,
                eps=1e-3):

  t = torch.ones(batch_size, device=device)
  y = y.to(device)

  # init x at t=1
  if z is None:
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z

  beta = 0.3

  # same mask as your inpainting setup
  mask = torch.ones_like(init_x)
  mask[:, :, :, 16:] = 0.

  #  make y masked + init_y consistent with training
  y = y * mask
  init_y = (y + torch.randn_like(y) * (beta * marginal_prob_std(t)[:, None, None, None]) * mask) * mask


  init_xy = torch.cat([init_x, init_y], dim=1)
  shape = init_xy.shape

  def score_eval_wrapper(sample, time_steps):
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    x_t = sample[:, 0:1, :, :]
    y_t = sample[:, 1:2, :, :]

    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((shape[0],))

   
    y_t = y_t * mask


    with torch.no_grad():
      score_xy = score_model(x_t, y_t, time_steps)
      score_x = score_xy[:, 0:1, :, :]
      score_y = score_xy[:, 1:2, :, :]

      std = marginal_prob_std(time_steps)[:, None, None, None]
      std2 = std**2 + 1e-12

      cond_grad = (y - y_t) * mask / std2
      score_y = score_y + cond_grad

      out = torch.cat([score_x, score_y], dim=1)

    return out.detach().cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t_scalar, x_flat):
    """
    Probability flow ODE (multi-speed):
      dx/dt = -0.5 * g_x(t)^2 * score_x
      dy/dt = -0.5 * g_y(t)^2 * score_y_tilde
    """
    time_steps = np.ones((shape[0],), dtype=np.float64) * t_scalar

    # g_x scalar
    g_x = diffusion_coeff(torch.tensor(t_scalar, device=device, dtype=torch.float32))
    g_x2 = (g_x**2).detach().cpu().numpy().item()

    # multi-speed for y 
    g_y2 = (beta**2) * g_x2

    drift = score_eval_wrapper(x_flat, time_steps)  # flattened [B*2*H*W]
    drift = drift.reshape(shape[0], 2, shape[2], shape[3])  # [B,2,H,W]


    drift[:, 0:1, :, :] *= (-0.5 * g_x2)
    drift[:, 1:2, :, :] *= (-0.5 * g_y2)


    return drift.reshape((-1,)).astype(np.float64)

  res = integrate.solve_ivp(
    ode_func,
    (1., eps),
    init_xy.reshape(-1).detach().cpu().numpy(),
    rtol=rtol,
    atol=atol,
    method='RK45'
  )
  print(f"Number of function evaluations: {res.nfev}")

  xy = torch.tensor(res.y[:, -1], device=device, dtype=torch.float32).reshape(shape)

  # (optional but safe) project y channel back to mask after solve
  xy[:, 1:2, :, :] = xy[:, 1:2, :, :] * mask

  x = xy[:, 0:1, :, :]
  return x
