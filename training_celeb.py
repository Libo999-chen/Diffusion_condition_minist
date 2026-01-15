import functools
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from tqdm import tqdm as tqdm1
from model import ScoreNet
from model1 import RobustScoreNet
from utils import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import random
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset1 import CelebADataset
import torch.optim as optim

base_log_dir = "runs/conditional_inpainting"

n_epochs =2000  # Number of epochs
batch_size = 64  # Mini-batch size
lr = 2e-4  # Learning rate



run_name = f"lr_{lr:.0e}_{n_epochs}_epochs_batch_size_{batch_size}_celeba_newmask2"

log_dir = f"{base_log_dir}/{run_name}"
writer = SummaryWriter(log_dir=log_dir)
os.makedirs(f'ckpt/{run_name}', exist_ok=True)
# def figure_to_numpy(fig):
#     """Convert a matplotlib figure to a NumPy array."""
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
#     width, height = fig.get_size_inches() * fig.get_dpi()
#     return buf.reshape(int(height), int(width), 3)

def figure_to_numpy(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    # Drop alpha channel → RGBA → RGB
    buf = buf[..., :3]
    return buf

def generate_random_square_mask(x, mask_ratio=0.25):
    """
    Generate a random square mask covering mask_ratio of the image.
    
    Args:
        x: input tensor [B, C, H, W]
        mask_ratio: fraction of image area to mask (default 0.25 for 25%)
    
    Returns:
        mask: tensor where 1 = observed, 0 = to be inpainted
    """
    B, C, H, W = x.shape
    
    # Calculate square side length to cover mask_ratio of area
    side_length = int(np.sqrt(mask_ratio * H * W))
    
    # Create masks for entire batch
    masks = torch.ones_like(x)
    
    for i in range(B):
        # Random top-left corner position
        y1 = random.randint(0, H - side_length)
        x1 = random.randint(0, W - side_length)
        
        # Zero out the square region (this is what we'll inpaint)
        masks[i, :, y1:y1+side_length, x1:x1+side_length] = 0.
    
    return masks




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

# def generate_random_mask(x):
#     _, _, height, width = x.shape  
#     x1 = random.randint(0, width-1)
#     y1 = random.randint(0, height-1)
#     mask = torch.ones_like(x)
#     mask[:, :, :y1, :x1] = 0

#     return x * mask 

sigma = 25.0  

# Define partial functions for conditional SDE calculations
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

def loss_fn(model, x, y, mask, marginal_prob_std, eps=1e-5,
            beta=0.3, lam_x=1.0, lam_y=0.2):
    B = x.shape[0]
    device = x.device
    t = torch.rand(B, device=device) * (1. - eps) + eps

    std_x = marginal_prob_std(t)[:, None, None, None]          # [B,1,1,1]
    std_y = beta * std_x

    # mask: observed region = 1 (left half)
    # mask = torch.ones_like(x)
    # mask[:, :, :, 16:] = 0.

    y0 = y * mask

    # noises (make y noise consistent with mask)
    z_x = torch.randn_like(x)
    z_y = torch.randn_like(y0) * mask

    # forward perturb
    x_t = x  + z_x * std_x
    y_t = y0 + z_y * std_y

    score_xy = model(x_t, y_t, t)   # [B,2,H,W]
    score_x = score_xy[:, 0:3]
    score_y = score_xy[:, 3:6]

    r_x = score_x * std_x + z_x
    r_y = score_y * std_y + z_y


    loss_x = r_x.pow(2).sum(dim=(1,2,3)) 
    loss_y = (r_y.pow(2) * mask).sum(dim=(1,2,3)) 

    return (lam_x * loss_x + lam_y * loss_y).mean()




# Initialize the score-based model
score_model = torch.nn.DataParallel(RobustScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.crop(img, top=39, left=9, height=160, width=160)),
    transforms.Resize(128),  # Resize to 32x32 to match your model
    transforms.ToTensor(),  # Convert to [0, 1] range
])
celeba_root = '/ssd_scratch/cvit/souvik/CelebA/CelebA/Img/img_align_celeba'
split_file = "/ssd_scratch/cvit/souvik/CelebA/CelebA/Eval/list_eval_partition.txt"

train_dataset = CelebADataset(
    root_dir=celeba_root,
    split_file=split_file,
    split="train",
    transform=transform
)

val_dataset = CelebADataset(
    root_dir=celeba_root,
    split_file=split_file,
    split="test",
    transform=transform
)

data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=9,
    pin_memory=True
)

val_data_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=9,
    pin_memory=True
)

# Optimizer
# optimizer = Adam(score_model.parameters(), lr=lr)

optimizer = optim.AdamW(score_model.parameters(), lr=2e-4, weight_decay=1e-4)

def get_lr_lambda(warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda

warmup_steps = 5000
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(warmup_steps))

# Training loop
tqdm_epoch = tqdm.trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    score_model.train()
    
    for x, _ in tqdm1(data_loader):  # We ignore the original labels from MNIST
        x = x.to(device)        
        # mask = torch.ones_like(x)
        # mask[:, :, :, 16:] = 0. 
        mask = generate_random_square_mask(x, mask_ratio=0.25)
        y = x * mask      
        y = y.to(device)
  
        loss = loss_fn(score_model, x, y, mask, marginal_prob_std_fn)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    writer.add_scalar('Epoch Loss', avg_loss / num_items, epoch)

    #likeihood !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    all_bpds = 0.
    all_items = 0
    if epoch % 5 == 0:
        score_model.eval()
        with torch.no_grad():

            x = next(iter(val_data_loader))[0]
            x = x.to(device)
            mask = generate_random_square_mask(x, mask_ratio=0.25)
            y = x * mask 

            samplefig_ode = samples_plt(score_model = score_model, 
                                        data_loader = data_loader, 
                                    device = device, 
                                        sampler = ode_sampler,
                                        sample_batch_size = batch_size, 
                                        marginal_prob_std_fn = marginal_prob_std_fn, 
                                        diffusion_coeff_fn = diffusion_coeff_fn,
                                        y=y,
                                        x=x,
                                        mask = mask)

            samplefig_pc = samples_plt(score_model = score_model, 
                                        data_loader = data_loader, 
                                        device = device, 
                                        sampler = pc_sampler,
                                        sample_batch_size = batch_size, 
                                        marginal_prob_std_fn = marginal_prob_std_fn, 
                                        diffusion_coeff_fn = diffusion_coeff_fn,
                                        y=y,
                                        x=x,
                                        mask = mask)

            samplefig_EM = samples_plt(score_model = score_model, 
                                        data_loader = data_loader, 
                                        device = device, 
                                        sampler = Euler_Maruyama_sampler,
                                        sample_batch_size = batch_size, 
                                        marginal_prob_std_fn = marginal_prob_std_fn, 
                                        diffusion_coeff_fn = diffusion_coeff_fn,
                                        y=y,
                                        x=x,
                                        mask = mask)

            numpy_ode = figure_to_numpy(samplefig_ode)
            numpy_pc = figure_to_numpy(samplefig_pc)
            numpy_em = figure_to_numpy(samplefig_EM)

        # Add to TensorBoard
            writer.add_image('ode Images', numpy_ode, epoch, dataformats='HWC')
            writer.add_image('PC Images', numpy_pc, epoch, dataformats='HWC')
            writer.add_image('SDE Images', numpy_em, epoch, dataformats='HWC')
        
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            torch.save({
                'model_state_dict': score_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Saves the current step counter
                'epoch': epoch
            }, f'ckpt/{run_name}/epoch_{epoch}.pth')

writer.close()


