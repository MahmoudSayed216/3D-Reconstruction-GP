import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from einops import rearrange

# --- 1. Feature Extractor (Swin Transformer) ---
class MultiViewFeatureExtractor(nn.Module):
    def __init__(self, backbone="swin_base_patch4_window7_224", embed_dim=256):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=False, features_only=True)
        self.proj = nn.Linear(self.backbone.feature_info.channels()[-1], embed_dim)  # Project to 256-dim
    def forward(self, images):
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)  # Merge batch & views

        features = self.backbone(images)[-1]  # Extract features
        print("Raw Backbone Features:", features.shape)

        features = features.flatten(2).permute(0, 2, 1)  # (B*V, seq_len, C)
        print("Flattened Features:", features.shape)

        self.proj = nn.Linear(features.shape[-1], 256)  # Ensure correct input size
        features = self.proj(features)  # Project features to embedding dim

        return features.view(B, V, -1, features.shape[-1])  # (B, V, seq_len, embed_dim)
# --- 2. Multi-View Fusion (Transformer Attention) ---
class MultiViewAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4),
            num_layers=num_layers
        )

    def forward(self, features):
        """
        features: (B, num_views, seq_len, embed_dim)
        Returns: (B, seq_len, embed_dim)  # Fused feature representation
        """
        B, V, S, D = features.shape
        features = rearrange(features, "B V S D -> B S (V D)")  # Merge view info
        return self.transformer(features)  # (B, seq_len, embed_dim)

# --- 3. Transformer Queries for Voxel Prediction ---
class VoxelTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_queries=1024):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4),
            num_layers=4
        )
        self.fc = nn.Linear(embed_dim, 32 * 32 * 32)  # Final voxel output

    def forward(self, fused_features):
        """
        fused_features: (B, seq_len, embed_dim)
        Returns: (B, 32, 32, 32) - 3D voxel grid
        """
        B = fused_features.shape[0]
        queries = self.query_embed.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, embed_dim)
        decoded = self.decoder(queries, fused_features.permute(1, 0, 2))  # Transformer decoding
        decoded = self.fc(decoded.mean(dim=0)).view(B, 32, 32, 32)  # Reshape to voxel grid
        return torch.sigmoid(decoded)  # Normalize output to (0,1)

# --- 4. Full Model Combining All Parts ---
class Mask2Former3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = MultiViewFeatureExtractor()
        self.multi_view_attention = MultiViewAttention()
        self.voxel_decoder = VoxelTransformerDecoder()

    def forward(self, images):
        """
        images: (batch_size, num_views, 3, H, W)
        Returns: (batch_size, 32, 32, 32) - voxel grid
        """
        features = self.feature_extractor(images)  # (B, num_views, seq_len, embed_dim)
        fused_features = self.multi_view_attention(features)  # (B, seq_len, embed_dim)
        voxel_grid = self.voxel_decoder(fused_features)  # (B, 32, 32, 32)
        return voxel_grid
