import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class SatelliteResNet(nn.Module):
    def __init__(self, num_classes = 10):
       
        super(SatelliteResNet, self).__init__()
        
        # 1. Loading ImageNet weights for initialization
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. change original designn to a 3x3 kernel，stride=1，padding=1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 3. remove the first maxpool layer
        self.resnet.maxpool = nn.Identity()

        # 4. change original classification num from 1000 to 10
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# model quick test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_input = torch.randn(2, 3, 64, 64).to(device)
    
    model = SatelliteResNet(num_classes=5).to(device)
    output = model(dummy_input)
    
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == (2, 5), "Output shape error!"
    print("Test Passed: Architecture is ready for 64x64.")