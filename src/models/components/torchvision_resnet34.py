import torch
import torchvision.models
from torch import nn


class TorchvisionResNet34(nn.Module):
    def __init__(
        self,
        output_size: int = 2,
        pretrained: bool = False,
        num_of_days: int = 2,
        embedding_dim: int = 16,
        frozen: bool = False,
    ) -> None:
        super().__init__()

        if pretrained:
            self.model = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            )
        else:
            self.model = torchvision.models.resnet34()

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.day_embedding = nn.Embedding(num_of_days, embedding_dim)

        self.model.fc = nn.Linear(self.model.fc.in_features, self.model.fc.in_features)

        self.cond_head = nn.Sequential(
            nn.Linear(self.model.fc.in_features + embedding_dim, self.model.fc.in_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.model.fc.in_features, output_size),
        )

    def forward(self, x: torch.Tensor, day_labels: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :param x: The labels of day, either 0 or 1.
        :return: A tensor of predictions.
        """

        features = self.model(x)
        day_emb = self.day_embedding(day_labels)
        x = torch.cat((features, day_emb), dim=1)

        return self.cond_head(x)


if __name__ == "__main__":
    _ = TorchvisionResNet34()
