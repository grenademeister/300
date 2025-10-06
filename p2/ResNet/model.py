import torch
import torch.nn as nn
from transformers import AutoImageProcessor , AutoModel

class ResNetRegressor(nn.Module):
    def __init__(self, checkpoint = 'microsoft/resnet-50',freeze_resnet = True ,head_layers = 3,head_dim = 512):
        super().__init__()
        self.image_encoder = AutoModel.from_pretrained(checkpoint)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        if freeze_resnet:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        num_image_features = self.image_encoder.config.hidden_sizes[-1]

        bbox_input_dim = 4
        bbox_embedded_dim = 64
        self.bbox_encoder = nn.Sequential(
            nn.Linear(bbox_input_dim,32),
            nn.ReLU(),
            nn.Linear(32,bbox_embedded_dim)
        )
        combined_dim = num_image_features + bbox_embedded_dim
        self.regressor = []
        in_dim = combined_dim
        for _ in range(head_layers -1):
            self.regressor.append(nn.Linear(in_dim , head_dim))
            self.regressor.append(nn.ReLU())
            self.regressor.append(nn.Dropout(0.3))
            in_dim = head_dim
        
        self.regressor.append(nn.Linear(head_dim,1))
        self.regressor = nn.Sequential(*self.regressor)
    def forward(self, image,bbox):
        image_feature = self.image_encoder(image).pooler_output    #[batch , num_image_features]
        bbox_feature = self.bbox_encoder(bbox) #[batch , 64]
        feature = torch.cat([image_feature,bbox_feature],dim = 1)
        height_prediction = self.regressor(feature)
        return height_prediction

   

