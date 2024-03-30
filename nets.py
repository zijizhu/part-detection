import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Softmax2d
from torchvision.models.resnet import ResNet

class IndividualLandmarkNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int = 8,
                 num_classes: int = 2000, landmark_dropout: float = 0.3) -> None:
        """
        Parameters
        ----------
        init_model: ResNet
            The pretrained ResNet model
        num_landmarks: int
            Number of landmarks to detect
        num_classes: int
            Number of classes for the classification
        landmark_dropout: float
            Probability of dropping out a given landmark
        """
        super().__init__()

        # The base model
        self.num_landmarks = num_landmarks
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.finalpool = torch.nn.AdaptiveAvgPool2d(1)

        # New part of the model
        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = BatchNorm2d(11)
        self.fc_landmarks = torch.nn.Conv2d(1024 + 2048, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(1024 + 2048, num_classes, bias=False)
        self.modulation = torch.nn.Parameter(torch.ones((1,1024 + 2048,num_landmarks + 1)))
        self.dropout = torch.nn.Dropout(landmark_dropout)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x: torch.Tensor
            Input image

        Returns
        -------
        all_features: torch.Tensor
            Features per landmark
        maps: torch.Tensor
            Attention maps per landmark
        scores: torch.Tensor
            Classification scores per landmark
        """
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        # x = torch.nn.functional.upsample_bilinear(x, size=(l3.shape[-2], l3.shape[-1]))
        x = torch.nn.functional.interpolate(x, size=(l3.shape[-2], l3.shape[-1]), mode='bilinear') # shape: [b, 2048, h, w], e.g. h=w=14
        x = torch.cat((x, l3), dim=1)

        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        batch_size = x.shape[0]
        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, x.shape[-2], x.shape[-1])
        a_sq = a_sq.permute(1, 0, 2, 3)
        maps = b_sq - 2 * ab + a_sq
        maps = -maps

        # Softmax so that the attention maps for each pixel add up to 1
        maps = self.softmax(maps)

        # Use maps to get weighted average features per landmark
        feature_tensor = x
        all_features = ((maps).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-1).mean(-1)

        # Classification based on the landmarks
        all_features_modulated = all_features * self.modulation
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0,2,1)).permute(0,2,1)
        scores = self.fc_class_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)

        return all_features, maps, scores


class IndividualLandmarkNetModified(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int = 8,
                 num_classes: int = 2000, landmark_dropout: float = 0.3) -> None:
        """
        Parameters
        ----------
        init_model: ResNet
            The pretrained ResNet model
        num_landmarks: int
            Number of landmarks to detect
        num_classes: int
            Number of classes for the classification
        landmark_dropout: float
            Probability of dropping out a given landmark
        """
        super().__init__()

        # The base model
        self.num_landmarks = num_landmarks
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.finalpool = torch.nn.AdaptiveAvgPool2d(1)

        # New part of the model
        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = BatchNorm2d(11)
        # self.fc_landmarks = torch.nn.Conv2d(1024 + 2048, num_landmarks + 1, 1, bias=False)
        self.landmarks = torch.nn.Parameter(torch.randn(num_landmarks + 1, 1024 + 2048))

        self.fc_class_landmarks = torch.nn.Linear(1024 + 2048, num_classes, bias=False)
        self.modulation = torch.nn.Parameter(torch.ones((1,1024 + 2048,num_landmarks + 1)))
        self.dropout = torch.nn.Dropout(landmark_dropout)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)

    def forward(self, x: torch.Tensor):
        """
        Modified with different attention calculation
        
        Parameters
        ----------
        x: torch.Tensor
            Input image

        Returns
        -------
        all_features: torch.Tensor
            Features per landmark
        maps: torch.Tensor
            Attention maps per landmark
        scores: torch.Tensor
            Classification scores per landmark
        """
        # Pretrained ResNet part of the model
        x = self.conv1(x) # shape: [b, 64, h1, w1], e.g. h1=w1=112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # shape: [b, 64, h2, w2], e.g. h2=w2=56
        x = self.layer1(x) # shape: [b, 256, h2, w2], e.g. h2=w2=56
        x = self.layer2(x) # shape: [b, 512, h3, w3], e.g. h2=w2=28
        l3 = self.layer3(x) # shape: [b, 1024, h3, w3], e.g. h2=w2=28
        x = self.layer4(l3) # shape: [b, 2048, h4, w4], e.g. h2=w2=7
        # x = torch.nn.functional.upsample_bilinear(x, size=(l3.shape[-2], l3.shape[-1]))
        x = torch.nn.functional.interpolate(x, size=(l3.shape[-2], l3.shape[-1]), mode='bilinear') # shape: [b, 2048, h, w], e.g. h=w=14
        x = torch.cat((x, l3), dim=1) # shape: [b, 2048 + 1024, h, w], e.g. h=w=14

        # Compute per landmark attention maps
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h*w).permute(0, 2, 1) # shape: [b, h*w, 2048 + 1024]
        maps = torch.cdist(x_flat, self.landmarks, p=2) # shape: [b, h*w, nlandmarks]
        maps = maps.permute(0, 2, 1).reshape(b, -1, h, w) # shape: [b, nlandmarks, h, w]
        # Softmax so that the attention maps for each pixel add up to 1
        maps = self.softmax(-maps) # shape: [b, nlandmarks, h, w]

        # Use maps to get weighted average features per landmark
        feature_tensor = x
        all_features = ((maps).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-1).mean(-1) # shape: [b, 2048 + 1024, nlandmarks]

        # Classification based on the landmarks
        all_features_modulated = all_features * self.modulation
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0,2,1)).permute(0,2,1)
        scores = self.fc_class_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)

        return all_features, maps, scores
    

class PartCEM(nn.Module):
    def __init__(self, backbone='resnet101', num_parts=8, num_classes=200, dropout=0.3) -> None:
        super(PartCEM, self).__init__()
        self.num_landmarks = num_parts
        self.k = num_parts + 1
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.dim = self.backbone.fc.weight.shape[-1]

        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        self.modulations = torch.nn.Parameter(torch.ones((1, self.k, self.dim)))

        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout1d(p=dropout)
        self.class_fc = nn.Linear(self.dim, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        b, c, h, w = x.shape

        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b,h*w,c]
        maps = torch.cdist(x_flat, self.prototypes, p=2) # shape: [b,h*w,k]
        maps = maps.permute(0, 2, 1).reshape(b, -1, h, w) # shape: [b,k,h,w]
        maps = self.softmax2d(-maps) # shape: [b,k,h,w]

        parts = torch.einsum('bkhw,bchw->bkchw', maps, x).mean((-1,-2)) # shape: [b,k,h,w], [b,c,h,w] -> [b,k,c]
        parts_modulated = parts * self.modulations # shape: [b,k,c]
        parts_modulated_dropped = self.dropout(parts_modulated) # shape: [b,k,c]
        class_logits = self.class_fc(parts_modulated_dropped) # shape: [b,k,|y|]

        return parts.permute(0, 2, 1), maps, class_logits.permute(0, 2, 1)