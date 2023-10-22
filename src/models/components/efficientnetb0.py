import pretrained_microscopy_models as pmm
import torch
import torch.utils.model_zoo as model_zoo
from efficientnet_pytorch import EfficientNet
from torch import nn


class EfficientNetB0(nn.Module):
    def __init__(
        self,
        output_size: int = 2,
        pretrained: bool = False,
        num_of_days: int = 2,
        embedding_dim: int = 16,
        frozen: bool = False,
    ) -> None:
        super().__init__()

        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        if pretrained:
            url = pmm.util.get_pretrained_microscopynet_url("efficientnet-b0", "image-micronet")
            self.model.load_state_dict(
                model_zoo.load_url(
                    url, map_location="cuda" if torch.cuda.is_available() else "cpu"
                )
            )

        self.day_embedding = nn.Embedding(num_of_days, embedding_dim)
        self.model._fc = nn.Linear(self.model._fc.in_features, self.model._fc.in_features // 2)

        self.cond_head = nn.Sequential(
            nn.Linear(
                (self.model._fc.in_features // 2) + embedding_dim, self.model._fc.in_features // 4
            ),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.model._fc.in_features // 4, output_size),
        )

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model._conv_head.parameters():
                param.requires_grad = True

            for param in self.model._bn1.parameters():
                param.requires_grad = True

            for param in self.model._fc.parameters():
                param.requires_grad = True

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
    _ = EfficientNetB0()
