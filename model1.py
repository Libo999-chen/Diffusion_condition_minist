import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Helper Functions ---

def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal positional encoding for time conditioning."""
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# --- Building Blocks ---

class ResnetBlockDDPM(nn.Module):
    """Residual block with time embedding projection and GroupNorm."""
    def __init__(self, in_ch, out_ch, temb_dim, dropout=0.1):
        super().__init__()
        self.gnorm0 = nn.GroupNorm(32, in_ch)
        self.conv0 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.gnorm1 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Zero-init the last conv for stable training start
        nn.init.zeros_(self.conv1.weight)
        
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, temb):
        h = self.conv0(F.silu(self.gnorm0(x)))
        # Time embedding added as bias
        h += self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv1(self.dropout(F.silu(self.gnorm1(h))))
        return self.shortcut(x) + h

class AttnBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""
    def __init__(self, channels):
        super().__init__()
        self.gnorm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, x):
        h_ = self.gnorm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)
        
        weights = torch.bmm(q, k) * (c**-0.5)
        weights = F.softmax(weights, dim=-1)
        
        h = torch.bmm(v, weights.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h)

class Upsample(nn.Module):
    """Nearest neighbor interpolation followed by a convolution."""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)

# --- Main Architecture ---

class RobustScoreNet(nn.Module):
    def __init__(self, marginal_prob_std):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        nf = 96
        ch_mult = (1, 1, 2, 2, 3, 3)
        attn_resolutions = (16, 8, 4)
        temb_dim = nf * 4

        # Time Embedding
        self.temb_net = nn.Sequential(
            nn.Linear(nf, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim)
        )

        # Initial Conv
        self.conv_in = nn.Conv2d(6, nf, 3, padding=1)

        # --- Encoder ---
        self.downs = nn.ModuleList()
        curr_ch = nf
        in_ch_list = [nf]
        res = 128
        for i, mult in enumerate(ch_mult):
            out_ch = nf * mult
            for _ in range(2):
                block = nn.ModuleList([ResnetBlockDDPM(curr_ch, out_ch, temb_dim)])
                curr_ch = out_ch
                if res in attn_resolutions:
                    block.append(AttnBlock(curr_ch))
                self.downs.append(block)
                in_ch_list.append(curr_ch)
            if i != len(ch_mult) - 1:
                self.downs.append(nn.Conv2d(curr_ch, curr_ch, 3, stride=2, padding=1))
                res //= 2
                in_ch_list.append(curr_ch)

        # --- Middle ---
        self.middle = nn.ModuleList([
            ResnetBlockDDPM(curr_ch, curr_ch, temb_dim),
            AttnBlock(curr_ch),
            ResnetBlockDDPM(curr_ch, curr_ch, temb_dim)
        ])

        # --- Decoder ---
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = nf * mult
            for _ in range(3): # 3 ResBlocks in decoder per level
                skip_ch = in_ch_list.pop()
                block = nn.ModuleList([ResnetBlockDDPM(curr_ch + skip_ch, out_ch, temb_dim)])
                curr_ch = out_ch
                if res in attn_resolutions:
                    block.append(AttnBlock(curr_ch))
                self.ups.append(block)
            if i != 0:
                self.ups.append(Upsample(curr_ch))
                res *= 2

        # --- Output ---
        self.final_gnorm = nn.GroupNorm(32, nf)
        self.final_conv = nn.Conv2d(nf, 6, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)

    def forward(self, x, y, t):
        # 1. Input Processing: [B, 6, 128, 128]
        # Channels 0-2: Masked, 3-5: Original
        h = torch.cat([x, y], dim=1) 
        temb = self.temb_net(get_timestep_embedding(t, 96))

        # 2. Initial Conv
        h = self.conv_in(h)
        hs = [h] # Skip connection storage

        # 3. Downsampling Path
        for layer in self.downs:
            if isinstance(layer, nn.Conv2d): # Downsample layer
                h = layer(h)
            else: # Resnet + optional Attention
                for sub_layer in layer:
                    h = sub_layer(h, temb) if isinstance(sub_layer, ResnetBlockDDPM) else sub_layer(h)
            hs.append(h)

        # 4. Bottleneck
        for layer in self.middle:
            h = layer(h, temb) if isinstance(layer, ResnetBlockDDPM) else layer(h)

        # 5. Upsampling Path
        for layer in self.ups:
            if isinstance(layer, Upsample):
                h = layer(h)
            else: # Resnet + optional Attention
                for sub_layer in layer:
                    if isinstance(sub_layer, ResnetBlockDDPM):
                        h = torch.cat([h, hs.pop()], dim=1) # Concat skip connection
                        h = sub_layer(h, temb)
                    else:
                        h = sub_layer(h)

        # 6. Final Head
        h = self.final_conv(F.silu(self.final_gnorm(h)))
        
        # Normalize by std as per score-based model theory
        return h / self.marginal_prob_std(t)[:, None, None, None]


def test_robust_score_net():
    # 1. Mock Marginal Probability Standard Deviation
    # In DDPM, this usually returns a value based on t.
    def mock_marginal_prob_std(t):
        return torch.ones_like(t)

    # 2. Initialize the Model
    # According to the architecture, input channels are 6 (3 masked + 3 original).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustScoreNet(marginal_prob_std=mock_marginal_prob_std).to(device)
    model.eval()

    # 3. Create Random Samples
    batch_size = 2
    img_size = 128 # The effective image size is 128x128
    
    # Masked image (3 channels)
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    # Original image or conditioning image (3 channels)
    y = torch.randn(batch_size, 3, img_size, img_size).to(device)
    # Random time steps (batch_size)
    t = torch.rand(batch_size).to(device)
    # print(t)

    print(f"Input shape (x): {x.shape}")
    print(f"Input shape (y): {y.shape}")
    print(f"Time steps (t): {t.shape}")

    # 4. Perform Forward Pass
    try:
        with torch.no_grad():
            output = model(x, y, t)
        
        print("\n--- Forward Pass Success ---")
        print(f"Output shape: {output.shape}") # Should be [B, 6, 128, 128]
        
        # Validation of Architecture Specifications
        expected_shape = (batch_size, 6, 128, 128)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print("✅ Output shape matches architecture specifications.")
        
    except Exception as e:
        print("\n--- Forward Pass Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_score_net()