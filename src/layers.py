import torchvision.models as models
from functools import reduce
import torchvision.models.quantization as models_q
from torchvision import datasets
from torchvision.transforms import ToTensor
from data_loader import data_loader
import torch
from model import VGG16, QResNet18, VGG11_32x32, VGG16_32x32
from util import *
from torch.utils.cpp_extension import load

model_dir = '../model/'
DATASET = 'ImageNet'

train_loader, valid_loader = data_loader(
    DATASET, batch_size=256, is_normalize=True, is_augment=False)


def get_module_by_name(module,
                       access_string: str):
    """Retrieve a module nested in another by its access string.
    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def plot_VGG16(net):
    for i in [0, 3, 7, 10]:
        print(Statistic(net.features[i].weight.data.flatten().numpy(),
              f"../{type(net).__name__}/Weight_fp_{i}", is_fig=True, range=[-0.2, 0.2]))
    X, y = next(iter(train_loader))

    check_output_list = [0, 3, 7, 10]
    for i in range(check_output_list[-1]+1):
        if i in check_output_list:
            print(Statistic(
                X.data.flatten().numpy(), f"../{type(net).__name__}/{i}_Input_fp", is_fig=True, range=[0, 2]))
        X = net.features[i](X)
        if i in check_output_list:
            print(Statistic(
                X.data.flatten().numpy(), f"../{type(net).__name__}/{i}_Output_fp", is_fig=True, range=[-5, 5]))


def plot_ResNet18(net):
    for layer_str in ['conv1', 'layer1.0.conv1', 'layer1.0.conv2']:
        weight = get_module_by_name(net, layer_str).weight
        print(Statistic(weight.data.flatten().numpy(),
              f"../{type(net).__name__}/Weight_fp_{layer_str}", is_fig=True, range=[-0.3, 0.3]))
    X, y = next(iter(train_loader))

    check_input_list = ['layer1.0.conv1', 'layer1.0.conv2',
                        'layer1.1.conv1', 'layer1.1.conv2']
    for name, m in net.named_modules():
        if not any(True for _ in m.children()):
            if name in check_input_list:
                print(Statistic(
                    X.data.flatten().numpy(), f"../{type(net).__name__}/{name}_Input_fp", is_fig=True, range=[0, 2]
                ))
            X = get_module_by_name(net, name)(X)


# net = VGG16(1000)
# MODEL_PATH = model_dir + type(net).__name__ + '-' + DATASET + '.model'
# net.load_state_dict(torch.load(MODEL_PATH))
# net.eval()
# print(net)
# plot_VGG16(net)
# plot_ResNet18(net)

net = VGG11_32x32(100)
print(net)



# torch.save(net.state_dict(), MODEL_PATH)
