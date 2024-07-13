from torch.nn import functional as F
from torch import nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        # Load the pre-trained ResNet available in torchvision
        resnet18 = torchvision.models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(512, num_classes)
        self.resnet18 = resnet18

    def forward(self, input):
        output = self.resnet18(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        return output
    
    
class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        in_features = None,
        norm_layer = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.fc1 = nn.Linear(in_features, in_features//2)
        self.bn1 = norm_layer(in_features//2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features//2, in_features//2)
        self.bn2 = norm_layer(in_features//2)
        self.downsample = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        identity = x
        batch_size = x.shape[0]

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = x.view(batch_size, 1, -1)
            identity = self.downsample(x)
            identity = identity.view(batch_size, -1)

        out += identity
        out = self.relu(out)

        return out
    

class resfcn18(nn.Module):
    def __init__(self, num_classes=100, img_size=224):
        super(resfcn18, self).__init__()
        self.img_size = img_size
        self.resize = 192
        target_size = 2048
        self.fc1 = nn.Linear((self.resize ** 2) * 3, target_size * 2)
        self.bn1 = nn.BatchNorm1d(target_size * 2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.block1 = nn.Sequential(BasicBlock(target_size//(2**0)),
                                    BasicBlock(target_size//(2**1)),
                                    )
        self.block2 = nn.Sequential(BasicBlock(target_size//(2**2)),
                                    BasicBlock(target_size//(2**3)),
                                    )
        self.block3 = nn.Sequential(BasicBlock(target_size//(2**4)),
                                    BasicBlock(target_size//(2**5)),
                                    )
        self.block4 = nn.Sequential(BasicBlock(target_size//(2**6)),
                                    BasicBlock(target_size//(2**7)),
                                    )
        self.fc2 = nn.Linear(target_size//(2**8), num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.interpolate(x, size=(self.resize, self.resize))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(batch_size, 1, -1)
        x = self.maxpool(x)
        x = x.view(batch_size, -1)
        
        x = self.block1(x)  # (N, 3, feature)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = x.view(batch_size, -1)
        logits = self.fc2(x)
        return logits

    
if __name__ == "__main__":
    resnet = resfcn18()
    print("test")