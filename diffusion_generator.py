import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep encoding for diffusion"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionDenoiser(nn.Module):
    """
    U-Net style denoiser for time-series vital signs
    Input: [batch, seq_len=24, features=5] + timestep + condition_label
    Output: [batch, seq_len=24, features=5] (denoised prediction)
    """
    def __init__(self, seq_len=24, n_features=5, hidden_dim=128, time_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding (sepsis label)
        self.condition_embed = nn.Embedding(2, hidden_dim)  # 0=healthy, 1=sepsis
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.GELU()
        )
        
        # Bottleneck with time and condition
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, n_features, kernel_size=3, padding=1)
        )
    
    def forward(self, x_noisy, timestep, condition_label):
        """
        x_noisy: [batch, seq_len, features] - noisy vital signs
        timestep: [batch] - diffusion timestep
        condition_label: [batch] - 0 or 1 for healthy/sepsis
        """
        batch_size = x_noisy.shape[0]
        
        # Embeddings
        t_emb = self.time_mlp(timestep)  # [batch, hidden_dim]
        c_emb = self.condition_embed(condition_label)  # [batch, hidden_dim]
        
        # Encode: [batch, seq, feat] -> [batch, feat, seq]
        x = x_noisy.transpose(1, 2)
        encoded = self.encoder(x)  # [batch, hidden_dim*2, seq]
        
        # Pool for bottleneck: [batch, hidden_dim*2, seq] -> [batch, hidden_dim*2]
        pooled = encoded.mean(dim=2)
        
        # Concatenate time and condition
        combined = torch.cat([pooled, t_emb, c_emb], dim=1)
        bottleneck_out = self.bottleneck(combined)  # [batch, hidden_dim*2]
        
        # Reshape back: [batch, hidden_dim*2] -> [batch, hidden_dim*2, seq]
        bottleneck_out = bottleneck_out.unsqueeze(2).expand(-1, -1, self.seq_len)
        
        # Decode
        decoded = self.decoder(bottleneck_out)  # [batch, features, seq]
        decoded = decoded.transpose(1, 2)  # [batch, seq, features]
        
        return decoded


class TimeSeriesDDPM:
    """
    Denoising Diffusion Probabilistic Model for clinical time-series
    """
    def __init__(self, denoiser, n_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.denoiser = denoiser.to(device)
        self.n_timesteps = n_timesteps
        self.device = device
        
        # Variance schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0, t):
        """
        Forward diffusion: q(x_t | x_0)
        x_0: [batch, seq, feat] clean data
        t: [batch] timestep indices
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise
    
    def denoise_step(self, x_t, t, condition_label):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t)
        """
        # Predict noise
        predicted_noise = self.denoiser(x_t, t, condition_label)
        
        # Coefficients
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Mean of p(x_{t-1} | x_t)
        model_mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
        )
        
        # Add noise if not final step
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            model_mean += torch.sqrt(beta_t) * noise
        
        return model_mean
    
    @torch.no_grad()
    def sample(self, condition_label, batch_size=16, seq_len=24, n_features=5):
        """
        Generate synthetic vital signs from pure noise
        condition_label: tensor [batch] of 0s or 1s
        """
        self.denoiser.eval()
        
        # Start from pure noise
        x_t = torch.randn(batch_size, seq_len, n_features).to(self.device)
        
        # Reverse diffusion
        for t_idx in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), t_idx, dtype=torch.long).to(self.device)
            x_t = self.denoise_step(x_t, t, condition_label)
        
        return x_t  # [batch, seq, feat]