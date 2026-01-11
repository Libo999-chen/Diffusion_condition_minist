# training_with_close_gap.py
import os
import argparse
import functools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from model import ScoreNet  # 2-channel output: score_x, score_y


# ----------------------------
# Utilities: figure -> numpy for TensorBoard
# ----------------------------
def figure_to_numpy(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.get_size_inches() * fig.get_dpi()
    arr = buf.reshape(int(height), int(width), 3)
    return arr


# ----------------------------
# A) Sliced Wasserstein Distance (SWD)
# ----------------------------
@torch.no_grad()
def sliced_wasserstein_distance(x, y, num_projections=256, p=2, seed=None):
    """
    SWD between two empirical distributions.
    x,y: [N,1,28,28] or [N,D]
    p=2 gives sliced W2 (sqrt(mean squared)).
    """
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)

    n = min(x.shape[0], y.shape[0])
    x = x[:n].to(dtype=torch.float32)
    y = y[:n].to(dtype=torch.float32)

    d = x.shape[1]
    if seed is not None:
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed))
        proj = torch.randn(num_projections, d, device=x.device, generator=g)
    else:
        proj = torch.randn(num_projections, d, device=x.device)

    proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-12)

    x_proj = x @ proj.t()   # [N,P]
    y_proj = y @ proj.t()

    x_sort, _ = torch.sort(x_proj, dim=0)
    y_sort, _ = torch.sort(y_proj, dim=0)
    diff = x_sort - y_sort

    if p == 1:
        return diff.abs().mean()
    elif p == 2:
        return torch.sqrt((diff ** 2).mean() + 1e-12)
    else:
        raise ValueError("p must be 1 or 2")


# ----------------------------
# B) Plot comparison grid: y / GT / ODE / SDE
# ----------------------------
def plot_inpainting_comparison(y, x_gt, x_ode, x_sde, n_show=8):
    y = y[:n_show].detach().cpu()
    x_gt = x_gt[:n_show].detach().cpu()
    x_ode = x_ode[:n_show].detach().cpu()
    x_sde = x_sde[:n_show].detach().cpu()

    fig, axes = plt.subplots(4, n_show, figsize=(1.6 * n_show, 6))
    titles = ["Condition y0", "GT x0", "ODE (fixed RK4)", "SDE (EM)"]
    rows = [y, x_gt, x_ode, x_sde]

    for r in range(4):
        for c in range(n_show):
            ax = axes[r, c]
            ax.imshow(rows[r][c, 0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_title(titles[r], fontsize=10, loc="left")
    plt.tight_layout()
    return fig


# ----------------------------
# VE SDE utilities
# ----------------------------
def marginal_prob_std(t, sigma):
    log_sigma = torch.log(torch.tensor(sigma, device=t.device, dtype=t.dtype))
    return torch.sqrt((sigma ** (2.0 * t) - 1.0) / (2.0 * log_sigma))

def diffusion_coeff(t, sigma):
    return sigma ** t


# ----------------------------
# Hutchinson divergence estimator
# ----------------------------
def divergence_hutchinson(v, x, mask=None):
    eps = torch.randn_like(x)
    if mask is not None:
        eps = eps * mask
    inner = (v * eps).sum()
    grad_inner = torch.autograd.grad(inner, x, create_graph=True)[0]
    if mask is not None:
        grad_inner = grad_inner * mask
    div = (grad_inner * eps).sum(dim=(1, 2, 3))
    return div


# ----------------------------
# u-model: scalar potential u(t,x,y) ~ log p_t(x,y)
# ----------------------------
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim=128, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class UModel(nn.Module):
    def __init__(self, embed_dim=128, ch=64):
        super().__init__()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim, scale=30.0),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(ch, 2 * ch, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(2 * ch, 2 * ch, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * ch + embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x_t, y_t, t):
        h = torch.cat([x_t, y_t], dim=1)
        feat = self.conv(h).mean(dim=(2, 3))
        temb = self.embed(t)
        out = self.head(torch.cat([feat, temb], dim=1))
        return out.squeeze(1)


# ----------------------------
# Close-gap residual: forward log-FP (VE, multi-speed)
# ----------------------------
def logfp_residual_loss(u_model, score_x, score_y, x_t, y_t, t,
                        diffusion_coeff_fn, beta=0.3, mask_y=None):
    u = u_model(x_t, y_t, t)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

    div_x = divergence_hutchinson(score_x, x_t)
    div_y = divergence_hutchinson(score_y, y_t, mask=mask_y)

    sq_x = (score_x ** 2).sum(dim=(1, 2, 3))
    if mask_y is None:
        sq_y = (score_y ** 2).sum(dim=(1, 2, 3))
    else:
        sq_y = ((score_y ** 2) * mask_y).sum(dim=(1, 2, 3))

    g_x = diffusion_coeff_fn(t)
    g_x2 = g_x ** 2
    g_y2 = (beta ** 2) * g_x2

    r = u_t + 0.5 * g_x2 * (div_x + sq_x) + 0.5 * g_y2 * (div_y + sq_y)
    return (r ** 2).mean()


# ----------------------------
# Auto alpha controller (no manual alpha)
# ----------------------------
class AlphaController:
    """
    Make alpha*res approx target_ratio * dsm on residual-steps.
    If residual_every=K and you want GLOBAL 15%, set target_global=0.15,
    then on-step target becomes target_on = target_global * K.
    """
    def __init__(self, residual_every: int, target_global=0.15, ema=0.98,
                 alpha_min=1e-12, alpha_max=1e-2, warmup_steps=200):
        self.K = max(1, int(residual_every))
        self.target_on = float(target_global) * self.K
        self.ema = float(ema)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.warmup_steps = int(warmup_steps)
        self.alpha = None
        self._ema_alpha = None

    def update(self, step, dsm_loss, res_loss):
        # Only update when res_loss computed and after warmup
        if step < self.warmup_steps:
            if self.alpha is None:
                self.alpha = 0.0
            return self.alpha

        d = float(dsm_loss)
        r = float(res_loss)
        if r <= 0 or not np.isfinite(r) or not np.isfinite(d) or d <= 0:
            return self.alpha if self.alpha is not None else 0.0

        alpha_est = self.target_on * d / (r + 1e-12)
        alpha_est = float(np.clip(alpha_est, self.alpha_min, self.alpha_max))

        if self._ema_alpha is None:
            self._ema_alpha = alpha_est
        else:
            self._ema_alpha = self.ema * self._ema_alpha + (1.0 - self.ema) * alpha_est

        self.alpha = float(np.clip(self._ema_alpha, self.alpha_min, self.alpha_max))
        return self.alpha


# ----------------------------
# Total training loss: DSM + alpha_res * residual (sparse)
# ----------------------------
def loss_fn_total(score_model, u_model, x, y,
                  marginal_prob_std_fn, diffusion_coeff_fn,
                  step_idx,
                  eps=1e-5, beta=0.3, lam_x=1.0, lam_y=0.2,
                  alpha_res=0.0, residual_every=8, use_close=True):
    B = x.shape[0]
    dev = x.device

    t = torch.rand(B, device=dev) * (1.0 - eps) + eps
    t = t.detach().requires_grad_(True)

    std_x = marginal_prob_std_fn(t)[:, None, None, None]
    std_y = beta * std_x

    mask = torch.ones_like(x)
    mask[:, :, :, 16:] = 0.0
    y0 = y * mask

    z_x = torch.randn_like(x)
    z_y = torch.randn_like(y0) * mask

    x_t = x + z_x * std_x
    y_t = y0 + z_y * std_y

    x_t = x_t.detach().requires_grad_(True)
    y_t = y_t.detach().requires_grad_(True)

    score_xy = score_model(x_t, y_t, t)
    score_x = score_xy[:, 0:1]
    score_y = score_xy[:, 1:2]

    # DSM
    r_x = score_x * std_x + z_x
    r_y = score_y * std_y + z_y
    loss_x = r_x.pow(2).sum(dim=(1, 2, 3))
    loss_y = (r_y.pow(2) * mask).sum(dim=(1, 2, 3))
    dsm_loss = (lam_x * loss_x + lam_y * loss_y).mean()

    # residual
    do_res = use_close and (residual_every > 0) and (step_idx % residual_every == 0)
    if do_res:
        res_loss = logfp_residual_loss(
            u_model=u_model,
            score_x=score_x,
            score_y=score_y,
            x_t=x_t,
            y_t=y_t,
            t=t,
            diffusion_coeff_fn=diffusion_coeff_fn,
            beta=beta,
            mask_y=mask
        )
    else:
        res_loss = torch.zeros((), device=dev)

    total = dsm_loss + alpha_res * res_loss
    return total, dsm_loss.detach(), res_loss.detach(), do_res


# ============================================================
# Samplers
#   - SDE: EM (fixed steps)
#   - ODE: fixed-step RK4 (FAIR: fixed NFE)
# ============================================================
@torch.no_grad()
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std_fn,
                           diffusion_coeff_fn,
                           y,
                           batch_size,
                           num_steps=500,
                           device='cuda',
                           eps=1e-3,
                           beta=0.3,
                           z_init=None,
                           y_init_noise=None,
                           seed=None):

    if seed is not None:
        torch.manual_seed(int(seed))

    y = y.to(device)
    t1 = torch.ones(batch_size, device=device)

    if z_init is None:
        x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std_fn(t1)[:, None, None, None]
    else:
        x = z_init

    mask = torch.ones_like(x)
    mask[:, :, :, 16:] = 0.0

    if y_init_noise is None:
        y_t = (y + torch.randn_like(y) * (beta * marginal_prob_std_fn(t1)[:, None, None, None]) * mask) * mask
    else:
        y_t = (y + y_init_noise * (beta * marginal_prob_std_fn(t1)[:, None, None, None]) * mask) * mask

    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    for time_step in time_steps:
        batch_time_step = torch.ones(batch_size, device=device) * time_step

        g_x = diffusion_coeff_fn(batch_time_step)
        g_y = beta * g_x

        std_x = marginal_prob_std_fn(batch_time_step)[:, None, None, None]
        std_y = beta * std_x
        std_y2 = std_y ** 2 + 1e-12

        score_xy = score_model(x, y_t, batch_time_step)
        score_x = score_xy[:, 0:1]
        score_y = score_xy[:, 1:2]

        x_mean = x + (g_x ** 2)[:, None, None, None] * score_x * step_size
        x = x_mean + torch.sqrt(step_size) * g_x[:, None, None, None] * torch.randn_like(x)

        conditional_term = (y - y_t) * mask / std_y2
        y_mean = y_t + (g_y ** 2)[:, None, None, None] * (score_y + conditional_term) * step_size
        y_t = y_mean + torch.sqrt(step_size) * g_y[:, None, None, None] * torch.randn_like(y_t) * mask
        y_t = y_t * mask

    return x_mean


@torch.no_grad()
def ode_sampler_fixed_rk4(score_model,
                          marginal_prob_std_fn,
                          diffusion_coeff_fn,
                          y,
                          batch_size,
                          ode_steps,
                          device='cuda',
                          eps=1e-3,
                          beta=0.3,
                          z_init=None,
                          y_init_noise=None,
                          seed=None):
    """
    Fixed-step RK4 probability-flow ODE in joint (x, y_t).
    Fairness: NFE is FIXED = 4 * ode_steps (each RK4 step uses 4 score evals).
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    t1 = torch.ones(batch_size, device=device)
    y = y.to(device)

    if z_init is None:
        x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std_fn(t1)[:, None, None, None]
    else:
        x = z_init

    mask = torch.ones_like(x)
    mask[:, :, :, 16:] = 0.0
    y = y * mask

    if y_init_noise is None:
        y_t = (y + torch.randn_like(y) * (beta * marginal_prob_std_fn(t1)[:, None, None, None]) * mask) * mask
    else:
        y_t = (y + y_init_noise * (beta * marginal_prob_std_fn(t1)[:, None, None, None]) * mask) * mask

    # fixed grid
    ts = torch.linspace(1.0, eps, ode_steps + 1, device=device)
    # dt is negative
    for i in range(ode_steps):
        t0 = ts[i]
        t1_ = ts[i + 1]
        dt = (t1_ - t0)  # negative

        def f(x_in, y_in, t_scalar):
            tvec = torch.ones(batch_size, device=device) * t_scalar
            g_x = diffusion_coeff_fn(tvec)  # [B]
            g_x2 = (g_x ** 2)[:, None, None, None]
            g_y2 = (beta ** 2) * g_x2

            score_xy = score_model(x_in, y_in * mask, tvec)
            score_x = score_xy[:, 0:1]
            score_y = score_xy[:, 1:2]

            std_x = marginal_prob_std_fn(tvec)[:, None, None, None]
            std_y2 = (beta * std_x) ** 2 + 1e-12
            cond_grad = (y - y_in) * mask / std_y2

            # probability flow ODE drift
            dx = -0.5 * g_x2 * score_x
            dy = -0.5 * g_y2 * (score_y + cond_grad)
            return dx, dy

        k1x, k1y = f(x, y_t, t0)
        k2x, k2y = f(x + 0.5 * dt * k1x, y_t + 0.5 * dt * k1y, t0 + 0.5 * dt)
        k3x, k3y = f(x + 0.5 * dt * k2x, y_t + 0.5 * dt * k2y, t0 + 0.5 * dt)
        k4x, k4y = f(x + dt * k3x, y_t + dt * k3y, t0 + dt)

        x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y_t = y_t + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        y_t = y_t * mask

    nfe_fixed = 4 * ode_steps
    return x, nfe_fixed


# ============================================================
# Evaluation: fair samples + SWD + plot + TensorBoard
# ============================================================
@torch.no_grad()
def evaluate_and_log(score_model, data_loader, writer, epoch,
                     marginal_prob_std_fn, diffusion_coeff_fn,
                     device,
                     eval_batch_size=32,
                     sde_steps=500,
                     swd_projections=256,
                     plot_n_show=8,
                     beta=0.3,
                     eps=1e-3):

    score_model.eval()

    x_gt = next(iter(data_loader))[0].to(device)[:eval_batch_size]
    B = x_gt.shape[0]

    mask = torch.ones_like(x_gt)
    mask[:, :, :, 16:] = 0.0
    y0 = x_gt * mask

    # ---- FAIR SEED + SAME INIT ----
    # fix evaluation randomness across epochs (still different per epoch)
    seed = 20240 + epoch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    z_init = torch.randn(B, 1, 28, 28, device=device) * marginal_prob_std_fn(torch.ones(B, device=device))[:, None, None, None]
    y_init_noise = torch.randn_like(y0)

    # SDE(EM): NFE_SDE = sde_steps
    x_sde = Euler_Maruyama_sampler(
        score_model=score_model,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        y=y0,
        batch_size=B,
        num_steps=sde_steps,
        device=device,
        eps=eps,
        beta=beta,
        z_init=z_init,
        y_init_noise=y_init_noise,
        seed=seed
    )

    # ODE(RK4): choose ode_steps so that NFE_ODE == NFE_SDE approximately
    # RK4 uses 4 evals per step => ode_steps = sde_steps//4
    ode_steps = max(1, sde_steps // 4)
    x_ode, nfev = ode_sampler_fixed_rk4(
        score_model=score_model,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        y=y0,
        batch_size=B,
        ode_steps=ode_steps,
        device=device,
        eps=eps,
        beta=beta,
        z_init=z_init,
        y_init_noise=y_init_noise,
        seed=seed
    )

    # Clamp
    x_gt_c  = x_gt.clamp(0, 1)
    y0_c    = y0.clamp(0, 1)
    x_sde_c = x_sde.clamp(0, 1)
    x_ode_c = x_ode.clamp(0, 1)

    # SWD
    swd_seed = 1234 + epoch
    swd_ode_sde = sliced_wasserstein_distance(x_ode_c, x_sde_c, num_projections=swd_projections, p=2, seed=swd_seed)
    swd_ode_gt  = sliced_wasserstein_distance(x_ode_c, x_gt_c,  num_projections=swd_projections, p=2, seed=swd_seed)
    swd_sde_gt  = sliced_wasserstein_distance(x_sde_c, x_gt_c,  num_projections=swd_projections, p=2, seed=swd_seed)

    # Log compute FAIRNESS
    writer.add_scalar("Compute/SDE_steps", sde_steps, epoch)
    writer.add_scalar("Compute/ODE_steps", ode_steps, epoch)
    writer.add_scalar("Compute/ODE_NFE_fixed", nfev, epoch)  # fixed, not adaptive

    # Log distances
    writer.add_scalar("SWD/W2_ODE_vs_SDE", float(swd_ode_sde), epoch)
    writer.add_scalar("SWD/W2_ODE_vs_GT",  float(swd_ode_gt),  epoch)
    writer.add_scalar("SWD/W2_SDE_vs_GT",  float(swd_sde_gt),  epoch)

    # Plot
    fig = plot_inpainting_comparison(y0_c, x_gt_c, x_ode_c, x_sde_c, n_show=plot_n_show)
    img = figure_to_numpy(fig)
    writer.add_image("Compare/y_GT_ODE_SDE", img, epoch, dataformats="HWC")
    plt.close(fig)

    return {
        "nfev": int(nfev),
        "ode_steps": int(ode_steps),
        "swd_ode_sde": float(swd_ode_sde),
        "swd_ode_gt": float(swd_ode_gt),
        "swd_sde_gt": float(swd_sde_gt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="/home/lc2762/Diffusion_condition/runs/conditional_inpainting")
    parser.add_argument("--save_dir", type=str, default="/home/lc2762/Diffusion_condition/run_model_history")
    parser.add_argument("--n_epochs", type=int, default=110)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=25.0)

    # close on/off (baseline ablation)
    parser.add_argument("--use_close", type=int, default=1)  # 1=close, 0=baseline

    # residual scheduling
    parser.add_argument("--residual_every", type=int, default=8)

    # auto alpha target
    parser.add_argument("--target_global_ratio", type=float, default=0.15)  # global target ratio
    parser.add_argument("--alpha_ema", type=float, default=0.98)
    parser.add_argument("--alpha_max", type=float, default=1e-2)

    # evaluation
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--sde_steps", type=int, default=500)
    parser.add_argument("--swd_projections", type=int, default=256)
    parser.add_argument("--plot_n_show", type=int, default=8)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=args.sigma)

    run_tag = "closegap_autoalpha" if args.use_close == 1 else "baseline"
    run_name = (
        f"{run_tag}_lr_{args.lr:.0e}_ep{args.n_epochs}_bs{args.batch_size}"
        f"_re{args.residual_every}_sde{args.sde_steps}"
        f"_tg{args.target_global_ratio}"
    )
    log_dir = os.path.join(args.base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn)).to(device)
    u_model = UModel(embed_dim=128, ch=64).to(device)

    # if baseline: only optimize score_model
    params = list(score_model.parameters()) + (list(u_model.parameters()) if args.use_close == 1 else [])
    optimizer = Adam(params, lr=args.lr)

    dataset = MNIST(".", train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # alpha controller (only used in close mode)
    alpha_ctrl = AlphaController(
        residual_every=args.residual_every,
        target_global=args.target_global_ratio,
        ema=args.alpha_ema,
        alpha_min=1e-12,
        alpha_max=args.alpha_max,
        warmup_steps=200
    )
    alpha_res = 0.0

    global_step = 0
    tqdm_epoch = tqdm.trange(args.n_epochs)

    for epoch in tqdm_epoch:
        score_model.train()
        u_model.train()

        avg_total, avg_dsm, avg_res = 0.0, 0.0, 0.0
        num_items = 0

        # epoch-level stats for alpha contribution
        alpha_used_steps = 0
        ratio_on_steps = []

        for x, _ in data_loader:
            x = x.to(device)

            mask = torch.ones_like(x)
            mask[:, :, :, 16:] = 0.0
            y = x * mask

            # Use current alpha_res (auto updated only when res computed)
            loss, dsm_loss, res_loss, did_res = loss_fn_total(
                score_model=score_model,
                u_model=u_model,
                x=x, y=y,
                marginal_prob_std_fn=marginal_prob_std_fn,
                diffusion_coeff_fn=diffusion_coeff_fn,
                step_idx=global_step,
                beta=0.3,
                lam_x=1.0, lam_y=0.2,
                alpha_res=alpha_res,
                residual_every=args.residual_every,
                use_close=(args.use_close == 1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # update alpha after seeing res_loss (only when computed)
            if (args.use_close == 1) and did_res:
                alpha_res = alpha_ctrl.update(global_step, float(dsm_loss), float(res_loss))
                # track on-step contribution ratio
                ratio = (alpha_res * float(res_loss)) / (float(dsm_loss) + 1e-12)
                ratio_on_steps.append(ratio)
                alpha_used_steps += 1

            bs = x.shape[0]
            avg_total += loss.item() * bs
            avg_dsm += float(dsm_loss) * bs
            avg_res += float(res_loss) * bs
            num_items += bs
            global_step += 1

        # log losses
        writer.add_scalar("Loss/total", avg_total / num_items, epoch)
        writer.add_scalar("Loss/dsm", avg_dsm / num_items, epoch)
        writer.add_scalar("Loss/residual", avg_res / num_items, epoch)

        # log alpha stats
        if args.use_close == 1:
            writer.add_scalar("Close/alpha_res", float(alpha_res), epoch)
            if len(ratio_on_steps) > 0:
                writer.add_scalar("Close/onstep_ratio_alphaRes_over_dsm", float(np.mean(ratio_on_steps)), epoch)
                writer.add_scalar("Close/onstep_ratio_median", float(np.median(ratio_on_steps)), epoch)

        # evaluation
        if (epoch % args.eval_every) == 0:
            stats = evaluate_and_log(
                score_model=score_model,
                data_loader=data_loader,
                writer=writer,
                epoch=epoch,
                marginal_prob_std_fn=marginal_prob_std_fn,
                diffusion_coeff_fn=diffusion_coeff_fn,
                device=device,
                eval_batch_size=args.eval_batch_size,
                sde_steps=args.sde_steps,
                swd_projections=args.swd_projections,
                plot_n_show=args.plot_n_show,
                beta=0.3,
                eps=1e-3
            )
            tqdm_epoch.set_description(
                f"ep {epoch} | loss {avg_total/num_items:.3f} | "
                f"SWD(ODE,SDE) {stats['swd_ode_sde']:.4f} | "
                f"NFE_ODE {stats['nfev']} (fixed)"
            )
        else:
            tqdm_epoch.set_description(f"ep {epoch} | loss {avg_total/num_items:.3f}")

    # save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{run_name}.pth")
    payload = {
        "score_model": score_model.state_dict(),
        "sigma": args.sigma,
        "use_close": int(args.use_close),
        "residual_every": int(args.residual_every),
        "target_global_ratio": float(args.target_global_ratio),
        "final_alpha_res": float(alpha_res),
        "sde_steps": int(args.sde_steps),
    }
    if args.use_close == 1:
        payload["u_model"] = u_model.state_dict()

    torch.save(payload, save_path)
    writer.close()
    print("Saved:", save_path)
    print("TensorBoard logdir:", log_dir)


if __name__ == "__main__":
    main()
