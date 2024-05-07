import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
import numpy as np

#############new import####
import logging
from ptflops import get_model_complexity_info
import math
import thop
import yaml
from tensorboardX import SummaryWriter

from torch.nn.parameter import Parameter
from torch.nn import init
from torch._jit_internal import Optional
from torch.nn.modules.module import Module

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k

    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k

        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(DWCONV, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=True
                                   )

    def forward(self, x):
        result = self.depthwise(x)
        return result


#######0822add DW卷积

class depth_separable(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(depth_separable, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))

class LIE(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LIE, self).__init__()
        self.DSConv = depth_separable(in_channels=in_channels, out_channels=out_channels)
        ########## add#########
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        dwconv_result = self.DSConv(x)  # 8,48,64,64
        dwconv_result = self.batchnorm(dwconv_result)
        dwconv_result = self.activation(dwconv_result)
        result = dwconv_result + x
        return result


##############################new###############################

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# dim ,head,bias,k=1,s=1
class DCMHSA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, kernel_size=1, stride=1):
        super(DCMHSA, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.d = dim // num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.padding = (kernel_size - 1) // 2
        ratio = 3

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(num_heads, dim, kernel_size=1, bias=bias)

        self.conv_attn_right = nn.Conv2d(self.d, 1, kernel_size=1, stride=stride, padding=0, bias=False)

        self.conv_v_right = nn.Conv2d(self.d, self.d // 2, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.d // 2, self.d // ratio * 2, kernel_size=1),

            nn.LayerNorm([self.d // ratio * 2, 8, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d // ratio * 2, self.d, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.conv_attn_left = nn.Conv2d(self.d, self.d // 2, kernel_size=1, stride=stride, padding=0,
                                        bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.d, self.d // 2, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_attn_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_attn_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_attn_right.inited = True
        self.conv_v_right.inited = True
        self.conv_attn_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1,8,6,4096
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)


        attn = torch.einsum('... i d, ... i d -> ... i d', q, k) * self.temperature  # 1,8,6,4096
        attn = attn.softmax(dim=-1)  # b,head,c，h*w

        v = v.permute(0, 2, 1, 3)
        input_x = self.conv_v_right(v)  # b,c,head,h*w 1,3,8,4096
        attn = attn.permute(0, 2, 1, 3)
        context_mask = self.conv_attn_right(attn)  # b，1，head，h*w,1,1,8,4096

        context_mask = self.softmax_right(context_mask)  # b，head，1，hw

        input_x = input_x.permute(0, 2, 1, 3)  # 1,8,3,4096
        context_mask = context_mask.permute(0, 2, 1, 3)  # 1.8.1.4096

        context = torch.einsum('... i d, ... j d -> ... j i', context_mask, input_x)  # 1,8,3,1

        context = context.permute(0, 2, 1, 3)  # 1,3,8,1
        context = self.conv_up(context)  # 1,6,8,1

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)  # 1,6,8,1
        d_b, d_c, d_h, d_s = mask_ch.shape
        mask_ch = mask_ch.view(d_b, d_h * d_c, d_s, d_s)  # 1,48,1,1

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 1,8,6,4096
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = torch.einsum('... i d, ... i d -> ... i d', q, k) * self.temperature
        attn = attn.softmax(dim=-1)  # 1,8,6,4096

        attn = attn.permute(0, 2, 1, 3)  # 1.6,8.4096
        # g_x = rearrange(attn, 'b h c hw -> b (h c) h w')
        g_x = self.conv_attn_left(attn)  # 1,3,8,4096

        g_x = g_x.permute(0, 2, 1, 3)  # 1,8,3,4096.

        batch, heads, channel, size = g_x.size()
        g_x = g_x.view(batch, heads, channel, size, -1)
        avg_x = self.avg_pool(g_x)  # b，head，3,1，1
        avg_x = avg_x.view(batch, heads, channel, -1)
        avg_x = avg_x.permute(0, 1, 3, 2)
        avg_x = self.softmax_left(avg_x)

        v = v.permute(0, 2, 1, 3)  # 1,6,8,4096
        theta_x = self.conv_v_left(v)  # 1,3,8,4096,b，head，c，hw
        # reshape
        theta_x = theta_x.permute(0, 2, 1, 3)  # 1,8

        context = torch.matmul(avg_x, theta_x)  # 1.8,1,4096 *#1,8,3,4096
        mask_sp = self.sigmoid(context)

        d_b, d_c, d_h, d_s = mask_sp.shape
        mask_sp = mask_sp.view(d_b, d_h * d_c, h, w)  # 1,8,64,64
        mask_sp = self.project_out(mask_sp)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out


#######0831 channel--dim
# dim,ex_pration
class CRFFN(nn.Module):

    def __init__(self, dim, expand_ratio):
        super(CRFFN, self).__init__()
        exp_channels = dim * expand_ratio
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, exp_channels, kernel_size=1),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels, stride=1),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_channels, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        result = x + self.conv2(self.dwconv(self.conv1(x)))
        return result


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# dim,heads,exp,in_channels
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, expand_ratio, in_channels, bias=False):
        super(TransformerBlock, self).__init__()
        self.dim = in_channels

        self.lie = LIE(in_channels=in_channels, out_channels=in_channels)
        ####20230821 add ############

        self.norm1 = nn.LayerNorm(self.dim)
        self.mhsa = DCMHSA(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(self.dim)

        # Inverted Residual FFN
        self.crffn = CRFFN(in_channels, expand_ratio)

    def forward(self, x):
        lie = self.lie(x)
        x = x + lie

        b, c, h, w = x.shape
        x_1 = rearrange(x, 'b c h w -> b ( h w ) c ')
        norm1 = self.norm1(x_1)
        norm1 = rearrange(norm1, 'b ( h w ) c -> b c h w', h=h, w=w)
        attn = self.mhsa(norm1)
        x = x + attn

        b, c, h, w = x.shape
        x_2 = rearrange(x, 'b c h w -> b ( h w ) c ')
        norm2 = self.norm2(x_2)
        norm2 = rearrange(norm2, 'b ( h w ) c -> b c h w', h=h, w=w)
        ffn = self.crffn(norm2)
        x = x + ffn

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


# dim,heads,exp,in_channels
class Conv_Transformer(nn.Module):

    def __init__(self, dim, num_heads, expand_ratio, in_channels):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=in_channels)
        self.Transformer = TransformerBlock(dim, num_heads, expand_ratio, in_channels)
        self.channel_reduce = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1,
                                        stride=1)
        self.Conv_out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1))

    def forward(self, x):
        conv = self.lrelu(self.doubleconv(x))
        trans = self.Transformer(x)
        x = torch.cat([conv, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x

class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class DOConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size=3, D_mul=None, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', simam=False):
        super(DOConv2d, self).__init__()

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.simam = simam
        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)
        ##################################################################################################
        if simam:
            self.simam_block = simam_module()
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.D_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            DoW = torch.reshape(self.W, DoW_shape)
        if self.simam:
            DoW_h1, DoW_h2 = torch.chunk(DoW, 2, dim=2)
            DoW = torch.cat([self.simam_block(DoW_h1), DoW_h2], dim=2)

        return self._conv_forward(input, DoW)


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                         groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SEAttention(nn.Module):

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.CA = SEAttention(channel=out_channel * 2, reduction=8)
        self.conv1 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1)
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        conv = self.main(x)

        ft = torch.cat([y, conv], 1)
        res = torch.cat([y, conv], 1)

        ft = self.CA(ft)
        ft = self.conv1(ft)

        res = self.conv2(res)
        return ft + x + res


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()

        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


class ADF(nn.Module):
    def __init__(self, in_channels):
        super(ADF, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)


    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        x_down = self.down(x)
        g_down = self.down(depth_map)

        g = F.max_pool2d(self.g(g_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)

        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)


        Ra = F.softmax(torch.bmm(theta, phi), 2)
        y = torch.bmm(Ra, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.interpolate(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


class FDCFormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_heads=8,
                 expand_ratio=4,
                 ):
        super(FDCFormer, self).__init__()

        self.pixelunshuffle = nn.PixelUnshuffle(2)  # 1,12,64,64
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

        self.net = nn.Conv2d(inp_channels * 4, dim, kernel_size=3, stride=1, padding=1)
        self.ft = ResBlock_do_fft_bench(out_channel=dim, )

        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, dim, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=dim, num_channels=dim),
            nn.SELU(inplace=True)
        )

        self.ft1 = ResBlock_do_fft_bench(out_channel=dim)
        self.down1_1 = Downsample(dim)
        self.ft2 = ResBlock_do_fft_bench(out_channel=dim * 2)
        self.dowm2_2 = Downsample(dim * 2)
        self.ft3 = ResBlock_do_fft_bench(out_channel=dim * 4)
        self.dowm3_3 = Downsample(dim * 4)
        self.ft4 = ResBlock_do_fft_bench(out_channel=dim * 8)
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1)

        self.ft5 = ResBlock_do_fft_bench(out_channel=dim * 8)
        self.u1 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.ft6 = ResBlock_do_fft_bench(out_channel=dim * 4)
        self.u2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.ft7 = ResBlock_do_fft_bench(out_channel=dim * 2)
        self.u3 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.ft8 = ResBlock_do_fft_bench(out_channel=dim)
        self.depth_pred = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            #     nn.Conv2d(dim, 1, kernel_size=1, stride=1),
            #     # nn.BatchNorm2d(1),
            #     # nn.GELU()
            #     nn.Sigmoid()
        )

        self.conv_tran1 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.down1 = Downsample(dim)
        self.conv_tran2 = Conv_Transformer(dim * 2, num_heads, expand_ratio, in_channels=dim * 2,
                                           )  # h/2
        self.down2 = Downsample(dim * 2)
        self.conv_tran3 = Conv_Transformer(dim * 4, num_heads, expand_ratio, in_channels=dim * 4,
                                           )  # h/4
        self.down3 = Downsample(dim * 4)
        self.conv_tran4 = Conv_Transformer(dim * 8, num_heads, expand_ratio, in_channels=dim * 8,
                                           )  # h/8

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.conv_tran5 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.up2 = nn.ConvTranspose2d(dim * 4, dim, kernel_size=4, stride=4)
        self.channel_reduce2 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.conv_tran6 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.up3 = nn.ConvTranspose2d(dim * 8, dim, kernel_size=8, stride=8)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.conv_tran7 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.conv_tran8 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.adf = ADF(dim)
        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, stride=1, padding=1)

        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x1 = self.pixelunshuffle(x)

        x1 = self.lrelu(self.net(x1))  # 8 48 64 64

        x = (x - self.mean) / self.std
        x = self.conv1(x)
        d_f1 = self.ft1(x)  # d

        d_f2 = self.down1_1(d_f1)  # 2d
        d_f2 = self.ft2(d_f2)

        d_f3 = self.dowm2_2(d_f2)  # 4d
        d_f3 = self.ft3(d_f3)

        d_f4 = self.dowm3_3(d_f3)  # 8d
        d_f4 = self.ft4(d_f4)

        d_f5 = self.ft5(d_f4)

        d_f6 = self.u1(d_f5)  # 4d
        d_f6 = self.ft6(d_f6 + d_f3)

        d_f7 = self.u2(d_f6)  # 2d
        d_f7 = self.ft7(d_f7 + d_f2)

        d_f8 = self.u3(d_f7)  # d

        depth_pred = self.depth_pred(d_f8 + d_f1)

        conv_tran1 = self.conv_tran1(x1)
        pool1 = self.down1(conv_tran1)
        conv_tran2 = self.conv_tran2(pool1)
        pool2 = self.down2(conv_tran2)

        conv_tran3 = self.conv_tran3(pool2)
        pool3 = self.down3(conv_tran3)

        conv_tran4 = self.conv_tran4(pool3)

        up1 = self.up1(conv_tran2)
        concat1 = torch.cat([up1, conv_tran1, ], 1)
        concat1_1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1_1)

        up2 = self.up2(conv_tran3)
        concat2 = torch.cat([up2, conv_tran5], 1)
        concat2_2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2_2)

        up3 = self.up3(conv_tran4)
        concat3 = torch.cat([up3, conv_tran6], 1)
        concat3_3 = self.channel_reduce2(concat3)
        conv_tran7 = self.conv_tran7(concat3_3)

        conv_tran8 = self.conv_tran8(conv_tran7)
        f = self.adf(conv_tran8, depth_pred)

        conv_out = self.lrelu(self.conv_out(f))

        out = self.pixelshuffle(conv_out)

        return out


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = FDCFormer(inp_channels=3, out_channels=3, dim=48)
    if torch.cuda.is_available():
        model.cuda()
    summary(model, (3, 128, 128))
    ops, params = get_model_complexity_info(model, (1, 3, 128, 128), as_strings=True, print_per_layer_stat=True,
                                            verbose=True)

    print(ops, params)
    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))

