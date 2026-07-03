import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone(x)
        #x = self.pool(x)
        #return x.view(x.size(0), -1)
        return x
    
    #test before heatmap
if __name__ == "__main__":
    model = FeatureExtractor()
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(output.shape)