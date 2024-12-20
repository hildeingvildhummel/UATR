"""https://music-classification.github.io/tutorial/part3_supervised/tutorial.html"""
from torch import nn
import torchaudio
from torch import Tensor
import torch
from typing import Callable, Optional
from torchaudio.models import Conformer
from torchvision.models import ResNet


"""https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """This function creates a 3x3 convolution with padding.
    Input:
        - in_planes: the number of channels in the input
        - out_planes: the number of channels in the output
        - stride: the stride of the convolution
        - groups: the number of blocked connections from the input channel
        - dilation: the spacing between the kernel elements
    Output:
        2 Dimensional Convolutional layer"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 464,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2, dropout=0.1):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size= 3, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm2d(3)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        return out

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, out_features):
        super(MultiLayerPerceptron, self).__init__()
        self.linear = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.input_bn1 = nn.BatchNorm1d(in_features)
        self.input_bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        out = self.linear(x)
        out = self.input_bn1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.input_bn2(out)
        return out

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHead, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x):
        out = self.linear(x)
        return out

class Expander(nn.Module):
    def __init__(self, input_dim):
       super(Expander, self).__init__()
       self.layer1 = nn.Linear(input_dim, 8192)
       self.bn = nn.BatchNorm1d(8192)
       self.relu = nn.ReLU(True)
       self.layer2 = nn.Linear(8192, 8192, bias=False)

    def forward(self,x):
       out = self.layer1(x)
       out = self.bn(out)
       out = self.relu(out)
       out = self.layer2(out)
       return out

class Encoder(nn.Module):
    def __init__(self, num_channels=3,
                       sample_rate=16000,
                       n_fft=1024,
                       f_min=0.0,
                       f_max=8000,
                       num_mels=128,
                       num_classes=2,
                        Baseline=False,
                        transformed=False,
                        num_heads=4,
                        ffn_dim=128,
                        num_layers=4,
                        depthwise_conv_kernel_size=31):
        super(Encoder, self).__init__()

        # mel spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                               n_mfcc=13,
                                               melkwargs={"n_fft": 1024, "n_mels": 128, "f_min": f_min, "f_max":f_max})
        self.num_classes = num_classes
        self.Baseline = Baseline
        self.input_bn = nn.BatchNorm2d(1)
        self.transformed = transformed

        # convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))

        if self.Baseline:
            self.layer2 = ResNet(layers=[2, 2, 2, 2], num_classes=2048, block=BasicBlock)
            self.layer3 = MultiLayerPerceptron(2048, 128)

        else:
            self.layer2 = Conformer(input_dim=num_mels, num_heads=4, ffn_dim=128, num_layers=4,depthwise_conv_kernel_size=31)
            self.layer3 = Expander(2048)
            self.fc = nn.Linear(8064, 2048)

        # if self.Baseline:
        self.layer4 = LinearHead(128, 4)

    def forward(self, wav):
        # input Preprocessing
        out = self.melspec(wav)
        out = self.amplitude_to_db(out)

        out = self.input_bn(out)

        if self.Baseline:
            # convolutional layers
            out = self.layer1(out)
            out_repr = self.layer2(out)
            out_contrast = self.layer3(out_repr)

        else:
            # Conformer layer
            out, lengths = self.layer2(out.squeeze(1).transpose(1,2), torch.tensor([63]).repeat(out.shape[0]))
            out = torch.flatten(out, 1)
            out_repr = self.fc(out)
            out_contrast = self.layer3(out_repr)

        return out_contrast, out_repr