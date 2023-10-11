import torch
import torchvision.models
from torch import nn


class TorchvisionResNet50(nn.Module):
    def __init__(self, output_size: int = 2, pretrained: bool = False) -> None:
        super().__init__()

        if pretrained:
            self.model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )
        else:
            self.model = torchvision.models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        return self.model(x)


if __name__ == "__main__":
    _ = TorchvisionResNet50()
