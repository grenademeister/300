import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


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
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        if fusion_type == "baseline_crop":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2 + metadata_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        elif fusion_type == "cross_attn":
            self.metadata_proj = nn.Linear(metadata_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        elif fusion_type == "film":
            self.gamma = nn.Linear(metadata_dim, hidden_dim)
            self.beta = nn.Linear(metadata_dim, hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, pixel_values, metadata):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        if self.fusion_type == "baseline":
            cls_token = outputs.pooler_output
            combined = torch.cat([cls_token, metadata], dim=1)
            return self.head(combined).squeeze(-1)

        if self.fusion_type == "baseline_crop":
            cls_token_whole = outputs.pooler_output
            cls_token_crop = self.backbone(
                pixel_values=self._crop_img(pixel_values, metadata)
            ).pooler_output

            combined = torch.cat([cls_token_whole, cls_token_crop, metadata], dim=1)
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

    def _crop_img(self, imgs, metadata):
        crops = []
        for i in range(imgs.size(0)):
            x1 = int(metadata[i, 2].item())
            x2 = int(metadata[i, 3].item())
            y1 = int(metadata[i, 4].item())
            y2 = int(metadata[i, 5].item())
            crop_x1, crop_y1, crop_x2, crop_y2 = self._get_enclosing_rect(
                (x1, y1, x2, y2)
            )
            crop = imgs[i:, :, crop_y1:crop_y2, crop_x1:crop_x2]
            crop = nn.functional.interpolate(
                crop, size=(224, 224), mode="bilinear", align_corners=False
            )
            crops.append(crop)
        return torch.cat(crops, dim=0)

    def _get_enclosing_rect(self, point_pairs, w=60, imgsz=511):
        """
        get minimal enclosing square for given point pairs
        Args:
            point_pairs: (x1, y1, x2, y2)
            w: padding width
            imgsz: image size
        Returns:
            (xmin, ymin, xmax, ymax)
        """
        x1, y1, x2, y2 = point_pairs
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        side = max(abs(x2 - x1), abs(y2 - y1)) + w
        half_side = side // 2
        point_pairs = (
            max(0, x_center - half_side),
            max(0, y_center - half_side),
            min(x_center + half_side, imgsz),
            min(y_center + half_side, imgsz),
        )
        return point_pairs


class ModelWrapper:
    def __init__(
        self,
        dinov3_name="facebook/dinov3-vitl16-pretrain-sat493m",
        device=None,
        fusion_type="baseline",
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
