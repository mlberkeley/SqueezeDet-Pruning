import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from ttq import Quantize, TTQ

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, small_input=False, use_ttq=False):
        super(AlexNet, self).__init__()
        self.feature_output_size = 256 if small_input else 256 * 6 * 6
        if not use_ttq:
            Conv2d = nn.Conv2d
            Linear = nn.Linear
        else:
            Conv2d = Quantize(nn.Conv2d)
            Linear = Quantize(nn.Linear)
        feature_layers = [
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if not small_input:
            feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features = nn.Sequential(*feature_layers)
        fc_size = min(self.feature_output_size, 4096)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            Linear(self.feature_output_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            Linear(fc_size, fc_size),
            nn.ReLU(inplace=True),
            Linear(fc_size, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, TTQ):
                init.uniform(m.W_p.data, 0.05, 0.1)
                init.uniform(m.W_n.data, -0.1, -0.05)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.feature_output_size)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
