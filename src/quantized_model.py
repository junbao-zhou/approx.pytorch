import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq
from torch.nn.parameter import Parameter
import torch.quantization as quantization
import torch.nn.functional as F
from torch import Tensor
from util import *
from model import *
import torchvision.models as models

IS_OUTPUT_VAR = False
IS_CUDA = True
IS_APPROX = True
if IS_CUDA:
    from torch.utils.cpp_extension import load
    tensor_compute = load(
        name="tensor_compute",
        sources=["cpp/tensor_compute.cu", "cpp/approx_mul_dict.cpp"],
        extra_include_paths=["cpp/include"],
        extra_cflags=['-O2'],
        verbose=True
    )


class ModelQuant(nn.Module):
    def __init__(self, module: nn.Module):
        super(ModelQuant, self).__init__()
        self.quant = quantization.QuantStub()
        self.layers = module
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x

    def print_state_dict(self):
        pass
    #     for key, val in self.state_dict().items():
    #         print(f"""{key} :
    # size : {val.shape} , dtype : {val.dtype, val.min()}""")


def fuse_model(model):
    FUSE_MODELS_DICT = {
        AlexNet: [['features.0', 'features.1'], ['features.3', 'features.4'], ['features.6', 'features.7'], ['features.8', 'features.9'], ['features.10', 'features.11'], ['classifier.1', 'classifier.2'], ['classifier.4', 'classifier.5']],
    }

    for m in model.modules():
        if type(m) == Conv2dBatchNormActivion:
            quantization.fuse_modules(
                m, ['conv', 'bn', 'act'], inplace=True)
        elif type(m) == Conv2dActivion:
            quantization.fuse_modules(
                m, ['conv', 'act'], inplace=True)
        elif type(m) == LinearActivion:
            quantization.fuse_modules(
                m, ['linear', 'act'], inplace=True)
        elif type(m) == AlexNet or type(m) == models.AlexNet:
            quantization.fuse_modules(
                m, FUSE_MODELS_DICT[type(m)], inplace=True)
            return
        elif type(m) == QResNet18 or type(m) == QResNet18_32x32:
            m.fuse_model()
            return
        elif type(m) == VGG16 or type(m) == VGG16_32x32 or type(m) == VGG11 or type(m) == VGG11_32x32:
            modules_names = [m_name for m_name, _ in m.named_modules()]
            modules_list = [m_child for m_child in m.modules()]
            for ind, m_child in enumerate(modules_list):
                if type(m_child) == nn.Conv2d:
                    if type(modules_list[ind+1]) == nn.BatchNorm2d:
                        if type(modules_list[ind+2]) == nn.ReLU:
                            torch.quantization.fuse_modules(
                                m, [modules_names[ind], modules_names[ind+1], modules_names[ind+2]], inplace=True)
                        else:
                            torch.quantization.fuse_modules(
                                m, [modules_names[ind], modules_names[ind+1]], inplace=True)
                    elif type(modules_list[ind+1]) == nn.ReLU:
                        torch.quantization.fuse_modules(
                            m, [modules_names[ind], modules_names[ind+1]], inplace=True)
                elif type(m_child) == nn.Linear and ind < (len(modules_list) - 1) and type(modules_list[ind+1]) == nn.ReLU:
                    torch.quantization.fuse_modules(
                        m, [modules_names[ind], modules_names[ind+1]], inplace=True)
            return


class _QConvLinearBase(nn.Module):
    def __init__(
            self,
            origin_model,
            is_cuda: bool = False):
        super(_QConvLinearBase, self).__init__()
        self.is_cuda = is_cuda
        self.scale = Parameter(
            torch.tensor(origin_model.scale), requires_grad=False)
        self.zero_point = Parameter(
            torch.tensor(origin_model.zero_point), requires_grad=False)
        self.dtype = torch.quint8

    def _param_to_cuda(self):
        if self.is_cuda:
            self.weight_prepared = Parameter(
                self.weight_prepared.to('cuda', torch.int8), requires_grad=False)
            self.bias = Parameter(
                self.bias.to('cuda'), requires_grad=False)
            self.weight_scale_prepared = Parameter(
                self.weight_scale_prepared.to('cuda'), requires_grad=False)
            self.bias_prepared = Parameter(
                self.bias_prepared.to('cuda'), requires_grad=False)


class _QConvBase(_QConvLinearBase):
    _global_token = 0

    def __init__(
            self,
            origin_model,
            is_cuda: bool = False):
        super(_QConvBase, self).__init__(origin_model, is_cuda)
        self._token = _QConvBase._global_token
        _QConvBase._global_token += 1

        self.stride = expand(origin_model.stride, 2)
        self.padding = expand(origin_model.padding, 2)
        tmp_weight = origin_model.weight()
        self.weight_prepared = Parameter(
            tmp_weight.int_repr().int() - tmp_weight.q_per_channel_zero_points().int().view(
                (tmp_weight.size(0), 1, 1, 1)), requires_grad=False)
        if IS_OUTPUT_VAR:
            print(
                Statistic(self.weight_prepared.to(torch.float), f'weight_int_layer{self._token}', is_fig=True))
        self.weight_scale_prepared = Parameter(
            tmp_weight.q_per_channel_scales().to(
                torch.float32).view((tmp_weight.size(0), 1, 1)), requires_grad=False)
        if origin_model.bias:
            self.bias = Parameter(
                origin_model.bias().clone().detach(), requires_grad=False)
            self.bias_prepared = Parameter(
                self.bias.view((1, self.bias.size(0), 1, 1)), requires_grad=False)
        else:
            self.bias = None

        self._param_to_cuda()

    def conv_forward(self, x: Tensor) -> Tensor:
        if self.is_cuda:
            conv_int = tensor_compute.qconv2d(
                x,
                self.weight_prepared,
                self.stride,
                self.padding,
                IS_APPROX
            )
        else:
            input_prepared = x.int_repr().int() - x.q_zero_point()
            if IS_OUTPUT_VAR:
                print(
                    Statistic(input_prepared.to(torch.float), f'input_int_layer{self._token}', is_fig=True, range=[0, 100]))
            conv_int = F.conv2d(
                input=input_prepared,
                weight=self.weight_prepared,
                stride=self.stride,
                padding=self.padding
            )
            if IS_OUTPUT_VAR:
                print(
                    Statistic(conv_int.to(torch.float), f'output_int_layer{self._token}', is_fig=True))
        conv_fp = conv_int.to(torch.float) * x.q_scale() * \
            self.weight_scale_prepared + self.bias_prepared
        return conv_fp


class QConv2d(_QConvBase):
    def __init__(
            self,
            origin_model: nnq.Conv2d,
            is_cuda: bool = False):
        assert type(origin_model) == nnq.Conv2d
        super(QConv2d, self).__init__(origin_model, is_cuda)

    def forward(self, x: Tensor) -> Tensor:
        return torch.quantize_per_tensor(
            self.conv_forward(x),
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=torch.quint8
        )


class QConv2dReLU(_QConvBase):
    def __init__(
            self,
            origin_model: nniq.ConvReLU2d,
            is_cuda: bool = False):
        assert type(origin_model) == nniq.ConvReLU2d
        super(QConv2dReLU, self).__init__(origin_model, is_cuda)

    def forward(self, x: Tensor) -> Tensor:
        return torch.quantize_per_tensor(
            F.relu(self.conv_forward(x)),
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=torch.quint8
        )


class QAdaptiveAvgPool2d(nn.Module):
    __constants__ = ['output_size']

    def __init__(
            self,
            origin_model: nn.AdaptiveAvgPool2d,
            is_cuda: bool = False) -> None:
        assert type(origin_model) == nn.AdaptiveAvgPool2d
        super(QAdaptiveAvgPool2d, self).__init__()
        self.output_size = origin_model.output_size

    def extra_repr(self) -> str:
        return 'output_size={}'.format(self.output_size)

    def forward(self, input: Tensor) -> Tensor:
        x = torch.dequantize(input)
        x = F.adaptive_avg_pool2d(x, self.output_size)
        return torch.quantize_per_tensor(
            x,
            scale=input.q_scale(),
            zero_point=input.q_zero_point(),
            dtype=torch.quint8
        )


class QMaxPool2d(nn.Module):

    def __init__(
            self,
            origin_model: nn.MaxPool2d,
            is_cuda: bool = False) -> None:
        assert type(origin_model) == nn.MaxPool2d
        super(QMaxPool2d, self).__init__()
        self.kernel_size = origin_model.kernel_size
        self.stride = origin_model.stride
        self.padding = origin_model.padding
        self.dilation = origin_model.dilation
        self.ceil_mode = origin_model.ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        x = torch.dequantize(input)
        x = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )
        return torch.quantize_per_tensor(
            x,
            scale=input.q_scale(),
            zero_point=input.q_zero_point(),
            dtype=torch.quint8
        )


class QFunctional(nn.Module):
    def __init__(
            self,
            origin_model: nnq.QFunctional,
            is_cuda: bool = False):
        assert type(origin_model) == nnq.QFunctional
        super(QFunctional, self).__init__()
        self.scale = origin_model.scale
        self.zero_point = origin_model.zero_point
        self.activation_post_process = torch.nn.Identity()

    def add_relu(self, x, y):
        # type: (Tensor, Tensor) -> Tensor
        r = torch.quantize_per_tensor(
            F.relu(x.dequantize() + y.dequantize()),
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=torch.quint8)
        r = self.activation_post_process(r)
        return r

# class QReLU(nn.Module):
#     def __init__(
#             self,
#             origin_model: nnq.ReLU,
#             is_cuda: bool = False):
#         assert type(origin_model) == nnq.ReLU
#         super(QReLU, self).__init__()

#     def forward(self, input):
#         x = torch.dequantize(input)
#         x = F.relu(x)
#         return torch.quantize_per_tensor(
#             x,
#             scale=input.q_scale(),
#             zero_point=input.q_zero_point(),
#             dtype=torch.quint8
#         )


class _QLinearBase(_QConvLinearBase):
    def __init__(
            self,
            origin_model: nnq.Linear,
            is_cuda: bool = False):
        super(_QLinearBase, self).__init__(origin_model, is_cuda)
        tmp_weight = origin_model.weight()
        self.weight_prepared = Parameter(
            tmp_weight.int_repr().int() - tmp_weight.q_per_channel_zero_points().int().view(
                (tmp_weight.size(0), 1)), requires_grad=False)
        self.weight_scale_prepared = Parameter(
            tmp_weight.q_per_channel_scales().to(
                torch.float32), requires_grad=False)
        if origin_model.bias:
            self.bias = Parameter(
                origin_model.bias().clone().detach(), requires_grad=False)
            self.bias_prepared = Parameter(
                self.bias.view((1, self.bias.size(0))), requires_grad=False)
        else:
            self.bias = None

        self._param_to_cuda()

    def linear_forward(self, x):
        if self.is_cuda:
            linear_int = tensor_compute.q_mat_mul(
                x,
                self.weight_prepared,
            )
        else:
            linear_int = F.linear(
                input=x.int_repr().int() - x.q_zero_point(),
                weight=self.weight_prepared
            )
        linear_fp = linear_int.to(torch.float) * x.q_scale() * self.weight_scale_prepared + \
            self.bias_prepared
        return linear_fp


class QLinear(_QLinearBase):
    def __init__(
            self,
            origin_model: nnq.Linear,
            is_cuda: bool = False):
        assert type(origin_model) == nnq.Linear
        super(QLinear, self).__init__(origin_model, is_cuda)

    def forward(self, x: Tensor) -> Tensor:
        return torch.quantize_per_tensor(
            self.linear_forward(x),
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=torch.quint8
        )


class QLinearReLU(_QLinearBase):
    def __init__(
            self,
            origin_model: nniq.LinearReLU,
            is_cuda: bool = False):
        assert type(origin_model) == nniq.LinearReLU
        super(QLinearReLU, self).__init__(origin_model, is_cuda)

    def forward(self, x: Tensor) -> Tensor:
        return torch.quantize_per_tensor(
            F.relu(self.linear_forward(x)),
            scale=self.scale,
            zero_point=self.zero_point,
            dtype=torch.quint8
        )


types_replace_dict = {
    nnq.Conv2d: QConv2d,
    nniq.ConvReLU2d: QConv2dReLU,
    nnq.Linear: QLinear,
    nniq.LinearReLU: QLinearReLU,
    nn.AdaptiveAvgPool2d: QAdaptiveAvgPool2d,
    nn.MaxPool2d: QMaxPool2d,
    nnq.QFunctional: QFunctional,
    # nnq.ReLU: QReLU,
}


def replace_layers(model, is_cuda=False):
    if is_cuda:
        model = model.to('cuda')
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            replace_layers(m, is_cuda)

        if type(m) in types_replace_dict.keys():
            # print(f'find type {type(m)} of {n}')
            setattr(model, n, types_replace_dict[type(m)](m, is_cuda))
    return model


if __name__ == '__main__':
    from data_loader import data_loader
    train_loader, valid_loader = data_loader('CIFAR10', 1)
    nnq.QFunctional()

# model_int8_quant_out = model_int8.quant(input)
# print(f"""\
# model_int8_quant_out :
# {model_int8_quant_out}""")

# exact = model_int8_quant_out.int_repr()
# mine = torch.round(input.float() / model_int8.quant.scale +
#                    model_int8.quant.zero_point).to(torch.uint8)
