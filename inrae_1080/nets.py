import torch
from torch import Tensor
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, AdaptiveAvgPool2d, Linear, Softmax2d, Parameter
from torchvision.models.resnet import ResNet
from typing import Tuple, Optional
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

# Baseline model, a modified ResNet with reduced downsampling for a spatially larger feature tensor in the last layer
class Net(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_classes: int=2000) -> None:
        super().__init__()
        self.conv1: Conv2d = init_model.conv1
        self.bn1: BatchNorm2d = init_model.bn1
        self.relu: ReLU = init_model.relu
        self.maxpool: MaxPool2d = init_model.maxpool
        self.layer1: Sequential = init_model.layer1
        self.layer2: Sequential = init_model.layer2
        self.layer3: Sequential = init_model.layer3
        self.layer4: Sequential = init_model.layer4
        self.layer3[0].downsample[0].stride = (1, 1)
        self.layer3[0].conv1.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv1.stride = (1, 1)
        self.finalpool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc: Linear = torch.nn.Linear(512, 300, bias=False)
        self.fc_class: Linear = torch.nn.Linear(300, num_classes, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.finalpool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        y: Tensor = self.fc_class(x)
        return x, x, y, x

# Proposed landmark-based model, also based on ResNet
class LandmarkNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int=8, num_classes: int=2000) -> None:
        super().__init__()

        self.num_landmarks = num_landmarks
        self.conv1: Conv2d = init_model.conv1
        self.bn1: BatchNorm2d = init_model.bn1
        self.relu: ReLU = init_model.relu
        self.maxpool: MaxPool2d = init_model.maxpool
        self.layer1: Sequential = init_model.layer1
        self.layer2: Sequential = init_model.layer2
        self.layer3: Sequential = init_model.layer3
        self.layer4: Sequential = init_model.layer4
        self.layer3[0].downsample[0].stride = (1, 1)
        self.layer3[0].conv1.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv1.stride = (1, 1)
        self.fc: Conv2d = torch.nn.Conv2d(512, 300, 1, bias=False)
        self.pool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_landmarks: Conv2d = torch.nn.Conv2d(512, num_landmarks + 1, 1,
                                            bias=False)

        self.mha = torch.nn.MultiheadAttention(embed_dim=310, num_heads=1, bias=False)
        self.fc_class_attention: Linear = torch.nn.Linear(300 + self.num_landmarks, num_classes, bias=False)
        self.fc_class_landmarks: Linear = torch.nn.Linear(300, num_classes, bias=False)


        self.softmax: Softmax2d = torch.nn.Softmax2d()
        # self.avg_dist_pos = torch.nn.Parameter(torch.zeros([num_landmarks]), requires_grad=False)
        # self.avg_dist_neg: Parameter = torch.nn.Parameter(torch.zeros([num_landmarks]),
        #                                        requires_grad=False)



    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Compute per landmark attention maps
        maps: Tensor = self.fc_landmarks(x)
        maps = self.softmax(maps)

        # Use maps to get weighted average features per landmark
        x = self.fc(x)
        feature_tensor: Tensor = x
        x = (maps[:, 0:-1, :, :].unsqueeze(-1).permute(0, 4, 2, 3,1) * x.unsqueeze(-1)).mean(2).mean(2)

        identity = torch.eye(self.num_landmarks).repeat(x.size(dim=0), 1, 1).to(x.get_device())
        att_input = torch.permute(torch.cat((identity, x), dim=1), (0, 2, 1))
        att, _ = self.mha(att_input, att_input, att_input, need_weights=False)
        att = torch.mean(att, dim=1)

        y: Tensor = self.fc_class_landmarks(x.permute(0, 2, 1)).permute(0, 2, 1)

        # classification = torch.Tensor([0.])
        classification = self.fc_class_attention(att)

        # x: feature vectors. y: classification results
        # return x, maps, y, feature_tensor
        return x, maps, y, feature_tensor, classification

