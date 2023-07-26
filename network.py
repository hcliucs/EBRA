import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Union,
    Tuple,
)

class Inpaintor(nn.Module):
    def __init__(self, in_channels: int = 4,
                       leaky_relu_slope : float = 0.2):
        super(Inpaintor, self).__init__()


        def gated_conv2d(inp: int, out: int, kern: int, strd: int, pad: int, dil: int = 1,
                         act = nn.LeakyReLU(0.2)):
            conv = GatedConv2d(in_channels=inp, out_channels=out, kernel_size=kern,
                        stride=strd, padding=pad, dilation=dil, activation=act)
            return conv

        def gated_upconv2d(inp: int, out: int, kern: int, strd: int, pad: int):
            return GatedUpConv2d(in_channels=inp, out_channels=out, kernel_size=kern,
                                    stride=strd, padding=pad)


        self.step1 = nn.Sequential(
            gated_conv2d(in_channels, 32, 5, 1, 2),     # layer 01 (4 x 256 x 256)  -> (32 x 256 x 256)
            gated_conv2d(32, 64, 3, 2, 1),              # layer 02 (32 x 256 x 256) -> (64 x 128 x 128)
            gated_conv2d(64, 64, 3, 1, 1),              # layer 03 (64 x 128 x 128) -> (64 x 128 x 128)
            gated_conv2d(64, 128, 3, 2, 1),             # layer 04 (64 x 128 x 128) -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 05 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 06 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 2, dil=2),     # layer 07 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 4, dil=4),     # layer 08 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 8, dil=8),     # layer 09 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 256, 3, 1, 1),            # layer 11 (128 x 64 x 64)  -> (128 x 64 x 64)
        )
        self.step2 = nn.Sequential(
            gated_conv2d(256+256*2, 256, 3, 1, 1),            # layer 12 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_upconv2d(256, 128, 3, 1, 1),           # layer 13 (128 x 64 x 64)  -> (64 x 128 x 128)
        ) 
        self.step3 = nn.Sequential(
            gated_conv2d(128+128*2, 128, 3, 1, 1),              # layer 14 (64 x 128 x 128) -> (64 x 128 x 128)
            gated_upconv2d(128, 64, 3, 1, 1),            # layer 15 (64 x 128 x 128) -> (32 x 256 x 256)
        )
        self.step4 = nn.Sequential(  
            gated_conv2d(64+64*2, 64, 3, 1, 1),              # layer 16 (32 x 256 x 256) -> (16 x 256 x 256)
        )
        self.step5 = nn.Sequential(   
            gated_conv2d(64+4, 3, 3, 1, 1, act=None),    # layer 17 (16 x 256 x 256) -> (3 x 256 x 256)
        )

    def forward(self,x,mask,edge_features,color_features):
        x=torch.cat([x,mask],dim=1)
        out=self.step1(x)
        out=torch.cat([out,color_features[0],edge_features[0]],dim=1)
        out=self.step2(out)
        out=torch.cat([out,color_features[1],edge_features[1]],dim=1)
        out=self.step3(out)
        out=torch.cat([out,color_features[2],edge_features[2]],dim=1)
        out=self.step4(out)
        out=torch.cat([out,color_features[3],edge_features[3]],dim=1)
        out=self.step5(out)
        return out

class GatedConv2d(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[Tuple[int], int],
                       stride: Union[Tuple[int], int] = 1,
                       padding: Union[Tuple[int], int] = 0,
                       dilation: Union[Tuple[int], int] = 1,
                       groups: int = 1,
                       bias: bool = True,
                       padding_mode: str = 'zeros',
                       activation: torch.nn.Module = nn.LeakyReLU(0.2)):

        super(GatedConv2d, self).__init__()

        self.conv_gating = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                padding_mode=padding_mode)

        self.conv_feature = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                padding_mode=padding_mode)

        self.gating_act = nn.Sigmoid()
        self.feature_act = activation
        self.b_norm = nn.BatchNorm2d(out_channels)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        gating = self.conv_gating(X)
        feature = self.conv_feature(X)

        if self.feature_act is None:
            output = feature * self.gating_act(gating)
        else:
            output = self.feature_act(feature) * self.gating_act(gating)

        output = self.b_norm(output)
        return output

class GatedUpConv2d(nn.Module):
    def __init__(self, *args, scale_factor: int = 2, **kwargs):
        """
        Gated convolution layer with scaling. For more information
        see `GatedConv2d` parameter description.

        Parameters
        ----------
        scale_factor : int
            Scaling factor.
        """

        super(GatedUpConv2d, self).__init__()
        self.conv = GatedConv2d(*args, **kwargs)
        self.scaling_factor = scale_factor


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.interpolate(X, scale_factor=self.scaling_factor)
        return self.conv(X)

class LocalDis(nn.Module):
    def __init__(self):
        super(LocalDis, self).__init__()
        self.input_dim = 3
        self.cnum = 64
        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return torch.sigmoid(x)

class GlobalDis(nn.Module):
    def __init__(self):
        super(GlobalDis, self).__init__()
        self.input_dim = 3
        self.cnum = 64
        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return torch.sigmoid(x)

def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[Tuple[int], int],
                       stride: Union[Tuple[int], int] = 1,
                       padding: Union[Tuple[int], int] = 0,
                       dilation: Union[Tuple[int], int] = 1,
                       groups: int = 1,
                       bias: bool = True,
                       padding_mode: str = 'zeros'):
        """
        Constructor for SpectralConv2d. For parameter explanation
        see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.
        """

        super(SpectralConv2d, self).__init__()

        self.conv = nn.utils.spectral_norm(nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                            padding_mode=padding_mode))


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv(X)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4,
                       leaky_relu_slope: float = 0.2):
        super(PatchGANDiscriminator, self).__init__()

        def spectral_conv2d(inp: int, out: int, kern: int, strd: int, pad: int):
            return nn.Sequential(
                    SpectralConv2d(in_channels=inp, out_channels=out,
                        kernel_size=kern, stride=strd, padding=pad),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope),
                )
        self.layers = nn.Sequential(
            spectral_conv2d(in_channels, 64, 5, 2, 2),  # layer 1 (5 x 256 x 256)  -> (64 x 128 x 128)
            spectral_conv2d(64, 128, 5, 2, 2),          # layer 2 (64 x 128 x 128) -> (128 x 64 x 64)
            spectral_conv2d(128, 256, 5, 2, 2),         # layer 3 (128 x 64 x 64)  -> (256 x 32 x 32)
            spectral_conv2d(256, 256, 5, 2, 2),         # layer 4 (256 x 32 x 32)  -> (256 x 16 x 16)
            spectral_conv2d(256, 256, 5, 2, 2),         # layer 5 (256 x 16 x 16)  -> (256 x 8 x 8)
            spectral_conv2d(256, 256, 5, 2, 2),         # layer 6 (256 x 8 x 8)    -> (256 x 4 x 4)
            nn.Flatten(),                               # layer 7 (256 x 4 x 4)    -> (4096)
        )
    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([images, masks], dim=1)
        output = self.layers(input_tensor)
        return output

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)

class Color_Extractor(BaseNetwork):
    def __init__(self, residual_blocks=3, use_spectral_norm=True, init_weights=True, outchannel=3):
        super(Color_Extractor, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_layer1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.decoder_layer2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )

        self.decoder_layer3=nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=outchannel, kernel_size=7, padding=0),
            nn.Tanh()
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        f1 = self.middle(x)
        f2 = self.decoder_layer1(f1)
        f3 = self.decoder_layer2(f2)
        f4 = (self.decoder_layer3(f3)+1)/2
        return f1, f2, f3, f4

class Edge_Extractor(BaseNetwork):
    def __init__(self, residual_blocks=3, use_spectral_norm=True, init_weights=True, outchannel=1):
        super(Edge_Extractor, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_layer1=nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.decoder_layer2=nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )

        self.decoder_layer3=nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=outchannel, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        f1 = self.middle(x)
        f2 = self.decoder_layer1(f1)
        f3 = self.decoder_layer2(f2)
        f4 = self.decoder_layer3(f3)
        return f1, f2, f3, f4

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

if __name__=='__main__':
    ext=Edge_Extractor()
    col=Color_Extractor()
    inp=Inpaintor()
    dis1=GlobalDis()
    dis2=LocalDis()
    a=torch.ones(1,3,256,256)
    b=torch.ones(1,1,256,256)
    ex=ext(a)
    co=col(a)
    out=inp(a,b,ex,co)
    c=dis1(a)
    # d=dis2(a)
    print(c.shape)