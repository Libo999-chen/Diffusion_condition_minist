import torch

def divergence_hutchinson(v, x, mask=None):
    """
    Estimate div_x v(x) = sum_i d v_i / d x_i via Hutchinson:
      div v ≈ eps^T J_v eps, eps ~ N(0,I)

    v, x: [B,1,H,W]
    mask: optional [B,1,H,W] to restrict dims
    returns: [B]
    """
    eps = torch.randn_like(x)
    if mask is not None:
        eps = eps * mask

    inner = (v * eps).sum()
    grad_inner = torch.autograd.grad(inner, x, create_graph=True)[0]  # [B,1,H,W]
    if mask is not None:
        grad_inner = grad_inner * mask

    div = (grad_inner * eps).sum(dim=(1, 2, 3))  # [B]
    return div


def logfp_residual_loss(u_model, score_x, score_y, x_t, y_t, t,
                        diffusion_coeff, beta=0.3, mask_y=None):
    """
    Forward log-FP residual (VE, zero drift) in joint (x,y):
      u_t + 0.5 g_x^2 (div_x s_x + ||s_x||^2) + 0.5 g_y^2 (div_y s_y + ||s_y||^2) = 0
    with g_y = beta g_x.

    IMPORTANT: we normalize div and ||s||^2 terms per-dimension (per-pixel)
    so residual scale is comparable to DSM.
    """
    # u(t,x,y) -> [B]
    u = u_model(x_t, y_t, t)
    if u.dim() > 1:
        u = u.squeeze(-1)

    # u_t: [B]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

    # divergence terms (raw sums over pixels)
    div_x = divergence_hutchinson(score_x, x_t)              # [B]
    div_y = divergence_hutchinson(score_y, y_t, mask=mask_y) # [B]

    # ---------------------------
    # PER-DIM / PER-PIXEL NORMALIZATION
    # ---------------------------
    # x dims: fixed 1*H*W per sample
    nx = float(x_t[0].numel())  # e.g. 1*28*28 = 784
    div_x = div_x / nx
    sq_x  = (score_x ** 2).mean(dim=(1, 2, 3))  # mean instead of sum

    if mask_y is None:
        # y dims: full 1*H*W
        ny = float(y_t[0].numel())
        div_y = div_y / ny
        sq_y  = (score_y ** 2).mean(dim=(1, 2, 3))
    else:
        # masked y dims: use per-sample effective pixel count
        m = mask_y.to(dtype=y_t.dtype)
        ny = m.sum(dim=(1, 2, 3)).clamp(min=1.0)  # [B]
        div_y = div_y / ny
        sq_y  = ((score_y ** 2) * m).sum(dim=(1, 2, 3)) / ny  # masked mean

    # diffusion scales
    g_x = diffusion_coeff(t)   # [B]
    g_x2 = g_x ** 2
    g_y2 = (beta ** 2) * g_x2

    # residual
    r = u_t + 0.5 * g_x2 * (div_x + sq_x) + 0.5 * g_y2 * (div_y + sq_y)
    return (r ** 2).mean()


def loss_fn_total(score_model, u_model, x, y, marginal_prob_std, diffusion_coeff,
                  eps=1e-5, beta=0.3, lam_x=1.0, lam_y=0.2,
                  alpha_res=0.1):
    B = x.shape[0]
    device = x.device

    # t needs grad for u_t
    t = torch.rand(B, device=device) * (1. - eps) + eps
    t = t.detach().requires_grad_(True)

    std_x = marginal_prob_std(t)[:, None, None, None]
    std_y = beta * std_x

    # inpainting mask (left half observed)
    mask = torch.ones_like(x)
    mask[:, :, :, 16:] = 0.
    y0 = y * mask

    z_x = torch.randn_like(x)
    z_y = torch.randn_like(y0) * mask

    x_t = x  + z_x * std_x
    y_t = y0 + z_y * std_y

    # Make x_t, y_t require grad for divergence estimation
    x_t = x_t.detach().requires_grad_(True)
    y_t = y_t.detach().requires_grad_(True)

    score_xy = score_model(x_t, y_t, t)
    score_x = score_xy[:, 0:1]
    score_y = score_xy[:, 1:2]

    # ---- DSM loss ----
    r_x = score_x * std_x + z_x
    r_y = score_y * std_y + z_y
    loss_x = r_x.pow(2).sum(dim=(1, 2, 3))
    loss_y = (r_y.pow(2) * mask).sum(dim=(1, 2, 3))
    dsm_loss = (lam_x * loss_x + lam_y * loss_y).mean()

    # ---- close residual (normalized) ----
    res_loss = logfp_residual_loss(
        u_model=u_model,
        score_x=score_x,
        score_y=score_y,
        x_t=x_t,
        y_t=y_t,
        t=t,
        diffusion_coeff=diffusion_coeff,
        beta=beta,
        mask_y=mask
    )

    total = dsm_loss + alpha_res * res_loss
    return total, dsm_loss.detach(), res_loss.detach()
