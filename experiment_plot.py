import torch
from torchvision.utils import make_grid
from model import ScoreNet
from training import *
import matplotlib.pyplot as plt


## Load the pre-trained checkpoint from disk.
device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)

sample_batch_size = 32 #@param {'type':'integer'}
sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

 # Define `y` as the masked version of the data
batch = next(iter(data_loader))[0]  #!!!


mask = torch.ones_like(x)
mask[:, :, :, 16:] = 0. 

y = x * mask  
## Generate samples using the specified sampler.
samples = sampler(score_model, 
                  marginal_prob_std_fn, 
                  diffusion_coeff_fn, 
                  y=y,  # Pass the conditional input `y`
                  batch_size=sample_batch_size, 
                  device=device)

## Sample visualization.
samples = samples.clamp(0.0, 1.0)
#matplotlib inline

sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()

#y_sample_grid = make_grid(y, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(y_sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()

x_sample_grid = make_grid(x, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(x_sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()