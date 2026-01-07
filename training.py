import functools
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from model import ScoreNet
from utils import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import random


from torch.utils.tensorboard import SummaryWriter
base_log_dir = "/home/lc2762/Diffusion_condition/runs/conditional_inpainting"

n_epochs =100  # Number of epochs
batch_size = 64  # Mini-batch size
lr = 1e-3  # Learning rate



run_name = f"lr_{lr:.0e}_{n_epochs}_epochs_batch_size_{batch_size}"
log_dir = f"{base_log_dir}/{run_name}"
writer = SummaryWriter(log_dir=log_dir)

def figure_to_numpy(fig):
    """Convert a matplotlib figure to a NumPy array."""
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.get_size_inches() * fig.get_dpi()
    return buf.reshape(int(height), int(width), 3)


device = 'cuda'  # Param for device type ('cuda' or 'cpu')
def marginal_prob_std(t, sigma): 
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    
    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
        y: The conditioning variable (e.g., weather information).
    
    Returns:
        The standard deviation.
    """
    #t = torch.tensor(t, device=device)
    t = torch.tensor(t, device=device).clone().detach()

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
    #return torch.tensor(sigma**t, device=device)
    return torch.tensor(sigma**t).clone().detach()

   
def generate_random_mask(x):
    _, _, height, width = x.shape  
    x1 = random.randint(0, width-1)
    y1 = random.randint(0, height-1)
    mask = torch.ones_like(x)
    mask[:, :, :y1, :x1] = 0

    return x * mask 


  
    

sigma = 25.0  

# Define partial functions for conditional SDE calculations
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

'''
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
'''

def loss_fn(model, x, y, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models (Conditional / Joint).

  Args:
    model: score model, now outputs 2 channels: [B,2,H,W] = (score_x, score_y)
    x: data image, shape [B,1,H,W]
    y: conditioning image (will be treated as y0), shape [B,1,H,W]
    marginal_prob_std: returns std(t), shape [B]
    eps: numerical stability
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  std = marginal_prob_std(random_t)  # [B]

  #  noise for x and y
  z_x = torch.randn_like(x)
  z_y = torch.randn_like(y)

  # perturb BOTH x and y to get (x_t, y_t)
  perturbed_x = x + z_x * std[:, None, None, None]
  perturbed_y = y + z_y * std[:, None, None, None]

  # model outputs joint score
  score_xy = model(perturbed_x, perturbed_y, random_t)   # [B,2,H,W]
  score_x = score_xy[:, 0:1, :, :]
  score_y = score_xy[:, 1:2, :, :]

  # VE DSM loss for both channels
  loss_x = torch.mean(torch.sum((score_x * std[:, None, None, None] + z_x)**2, dim=(1, 2, 3)))
  loss_y = torch.mean(torch.sum((score_y * std[:, None, None, None] + z_y)**2, dim=(1, 2, 3)))

  return loss_x + loss_y



# Initialize the score-based model
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)



# Dataset and DataLoader
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4) #the number of subprocesses used to load data.

val_data = dataset.data[:1].unsqueeze(1)

# Optimizer
optimizer = Adam(score_model.parameters(), lr=lr)

# Training loop
tqdm_epoch = tqdm.trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0

    
    for x, _ in data_loader:  # We ignore the original labels from MNIST
        x = x.to(device)
        #y = generate_random_mask(x)
        #y = y.to(device)
        
        
        # Define the mask
        
        mask = torch.ones_like(x)
        mask[:, :, :, 16:] = 0. 

       
        y = x * mask  
        
        y = y.to(device)
     

        loss = loss_fn(score_model, x, y, marginal_prob_std_fn)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]


    writer.add_scalar('Epoch Loss', avg_loss / num_items, epoch)


    #likeihood !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    all_bpds = 0.
    all_items = 0   

    x=val_data
    x = x.to(device)
  
    mask = torch.ones_like(x)
    mask[:, :, :, 16:] = 0. 
    y = x * mask 
 

    #y = generate_random_mask(x)
    '''
    _, bpd = ode_likelihood(x=x,
                            y=y,
                            score_model=score_model, 
                            marginal_prob_std=marginal_prob_std_fn,
                            diffusion_coeff=diffusion_coeff_fn,
                            batch_size=batch_size, 
                            device=device, 
                            eps=1e-5)
    all_bpds += bpd.sum()
    all_items += bpd.shape[0]
      
    writer.add_scalar("Average bits/dim", all_bpds / all_items, epoch)
    #likeihood !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''

    

    x = next(iter(data_loader))[0]

    x = x.to(device)
    mask = torch.ones_like(x)
    mask[:, :, :, 16:] = 0. 
    y = x * mask 
    
    #y = generate_random_mask(x)


    '''
   samplefig_ode = samples_plt(score_model = score_model, 
                                data_loader = data_loader, 
                               device = device, 
                                sampler = ode_sampler,
                                sample_batch_size = batch_size, 
                                marginal_prob_std_fn = marginal_prob_std_fn, 
                                diffusion_coeff_fn = diffusion_coeff_fn,
                                y=y,
                                x=x)
    '''
    samplefig_pc = samples_plt(score_model = score_model, 
                                data_loader = data_loader, 
                                device = device, 
                                sampler = pc_sampler,
                                sample_batch_size = batch_size, 
                                marginal_prob_std_fn = marginal_prob_std_fn, 
                                diffusion_coeff_fn = diffusion_coeff_fn,
                                y=y,
                                x=x)

    samplefig_EM = samples_plt(score_model = score_model, 
                                data_loader = data_loader, 
                                device = device, 
                                sampler = Euler_Maruyama_sampler,
                                sample_batch_size = batch_size, 
                                marginal_prob_std_fn = marginal_prob_std_fn, 
                                diffusion_coeff_fn = diffusion_coeff_fn,
                                y=y,
                                x=x)

    #numpy_ode = figure_to_numpy(samplefig_ode)
    numpy_pc = figure_to_numpy(samplefig_pc)
    numpy_em = figure_to_numpy(samplefig_EM)

   # Add to TensorBoard
    #writer.add_image('ode Images', numpy_ode, epoch, dataformats='HWC')
    writer.add_image('PC Images', numpy_pc, epoch, dataformats='HWC')
    writer.add_image('SDE Images', numpy_em, epoch, dataformats='HWC')
   
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))



torch.save(score_model.state_dict(), f'/home/lc2762/Diffusion_condition/run_model_history/lr_{lr:.0e}_n_epochs_{n_epochs}_batch_size_{batch_size}.pth')

writer.close()


