from torch import nn
import torchvision


# class ResNet18(nn.Module):
#     """
#     A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
#     before the i-th maxpooling layer within the VGG19 network', as defined in the paper.
#     Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
#     """

#     def __init__(self, num_cls=4):
#         """
#         :param i: the index i in the definition above
#         :param j: the index j in the definition above
#         """
#         super(ResNet18, self).__init__()

#         # Load the pre-trained VGG19 available in torchvision
#         resnet18 = torchvision.models.resnet18(pretrained=True)
#         resnet18.fc = nn.Linear(512, num_cls)
#         self.resnet18 = resnet18

#     def forward(self, input):
#         """
#         Forward propagation
#         :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
#         :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
#         """
#         output = self.resnet18(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

#         return output

class CustomCNN(nn.Module):
    def __init__(self, num_cls=100, img_size=32):
        super(CustomCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_cls)

    def forward(self, x):
        x = self.block1(x)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(-1, 512)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        logits = self.fc2(x)
        return logits
    
    
if __name__ == "__main__":
    resnet = CustomCNN()
    print("test")