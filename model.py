import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A simple residual block for 1D features, enabling the network to train much deeper
    without suffering from vanishing gradients. Crucial for massive datasets like NSD.
    """
    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return x + self.block(x)

class DeepResidualMapper(nn.Module):
    """
    Production-grade Neural Decoder.
    Maps high-dimensional fMRI voxel signals to 77x768 (or 1x1024) CLIP embedding space.
    Uses residual connections and LayerNorm for stable cluster training.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 4096, output_dim: int = 1024, num_blocks: int = 4, dropout_rate: float = 0.3):
        """
        Args:
            input_dim: Number of voxels (varies by NSD subject, e.g., 10,000+).
            hidden_dim: Width of the hidden layers.
            output_dim: Dimension of the CLIP embedding (1024 for SD v1.5 text embedding size equivalent, or pooled embedding).
            num_blocks: Number of residual blocks.
            dropout_rate: Regularization dropout.
        """
        super().__init__()
        
        # Projection from voxel space to hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Deep Residual Core
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # Projection to CLIP embedding space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim) # Normalize to help CLIP condition effectively
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_proj(x)
        return x
