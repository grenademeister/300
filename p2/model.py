import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from ResNet.model import ResNetRegressor

class ChimneyRegressor(nn.Module):
    def __init__(
        self,
        dinov3_name="facebook/dinov3-vitl16-pretrain-sat493m",
        metadata_dim=5,
        fusion_type="baseline",
    ):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(dinov3_name)
        self.backbone = AutoModel.from_pretrained(dinov3_name)

        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size
        self.fusion_type = fusion_type

        if fusion_type == "baseline":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim + metadata_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
        elif fusion_type == "cross_attn":
            self.metadata_proj = nn.Linear(metadata_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
        elif fusion_type == "film":
            self.gamma = nn.Linear(metadata_dim, hidden_dim)
            self.beta = nn.Linear(metadata_dim, hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )

    def forward(self, pixel_values, metadata):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        if self.fusion_type == "baseline":
            cls_token = outputs.pooler_output
            combined = torch.cat([cls_token, metadata], dim=1)
            return self.head(combined).squeeze(-1)

        elif self.fusion_type == "cross_attn":
            tokens = outputs.last_hidden_state
            metadata_emb = self.metadata_proj(metadata).unsqueeze(1)
            attn_out, _ = self.cross_attn(metadata_emb, tokens, tokens)
            fused = self.norm(attn_out.squeeze(1))
            return self.head(fused).squeeze(-1)

        elif self.fusion_type == "film":
            tokens = outputs.last_hidden_state
            gamma = self.gamma(metadata).unsqueeze(1)
            beta = self.beta(metadata).unsqueeze(1)
            modulated = gamma * tokens + beta
            fused = modulated.mean(dim=1)
            return self.head(fused).squeeze(-1)


class ModelWrapper:
    def __init__(
        self,
        model_type = 'dino',
        dinov3_name="facebook/dinov3-vitl16-pretrain-sat493m",
        device=None,
        fusion_type="baseline",
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if(model_type =='resnet'):
            self.model = ResNetRegressor().to(self.device)
        else:
            self.model = ChimneyRegressor(dinov3_name, fusion_type=fusion_type).to(
                self.device
            )
        self.processor = self.model.processor
    
    def train_mode(self):
        self.model.train()
        self.model.backbone.eval()

    def eval_mode(self):
        self.model.eval()

    def predict(self, pixel_values, metadata):
        self.eval_mode()
        with torch.no_grad():
            return self.model(pixel_values.to(self.device), metadata.to(self.device))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device), strict=False
        )
