import torch
import torch.nn as nn
from torchvision import models

class ResNetRegressor(nn.Module):
    def __init__(self, freeze_resnet = True ,head_layers = 3,head_dim = 1024):
        super().__init__()
        resnet = models.resnet50(pretrained = True)
        if freeze_resnet:
            for param in resnet.parameters():
                param.requires_grad = False
        num_image_features = resnet.fc.in_features
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        bbox_input_dim = 4
        bbox_embedded_dim = 64
        self.bbox_encoder = nn.Sequential(
            nn.Linear(bbox_input_dim,32),
            nn.ReLU(),
            nn.Linear(32,bbox_embedded_dim)
        )
        combined_dim = num_image_features + bbox_embedded_dim
        self.regressor = nn.ModuleList()

        for _ in range(head_layers -1):
            if(_ == 0):
                in_dim = combined_dim
            else:
                in_dim = head_dim
            self.regressor.append(nn.Sequential(
                nn.Linear(in_dim , head_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ))
        self.regressor.append(nn.Linear(head_dim,1))
    
    def forward(self, image,bbox):
        image_feature = self.image_encoder(image) 
        image_feature = torch.flatten(image_feature) #[batch , num_image_features]
        bbox_feature = self.bbox_encoder(bbox) #[batch , 64]
        feature = torch.cat([image_feature,bbox_feature],dim = 1)
        height_prediction = self.regressor(feature)
        return height_prediction
