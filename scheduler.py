import torch
import torch.nn as nn

class NoiseScheduler(nn.Module):
    def __init__(self, T=1000, latent_dim=512, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        self.latent_dim = latent_dim
        
        # Register buffers for noise schedule parameters
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, T))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        
        # Time step embedding layer
        self.time_embedding = nn.Embedding(T, latent_dim)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.betas.device)

    def add_noise(self, x_0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def get_latent(self, t):
        return self.time_embedding(t)