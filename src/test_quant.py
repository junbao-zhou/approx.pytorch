from quantized_model import *
from model import *
from data_loader import data_loader
from model_run import train, validate
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.quantization as quantization
import torch.nn.functional as F
import random
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq
from util import *

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
# print(DEVICE)

BATCH_SIZE = 500
N_CLASSES = 10
DATASET_NAME = 'CIFAR10'
model_dir = '../model/'
IS_LOAD_MODEL = False

# torch.manual_seed(1234)
batch = random.randint(32, 200)
in_feature = random.randint(1, 128)
out_feature = random.randint(1, 128)
kernel_H = random.randint(3, 10)
kernel_W = random.randint(3, 10)
input_H = random.randint(kernel_H, 64)
input_W = random.randint(kernel_W, 64)
stride = (random.randint(1, 3), random.randint(1, 3))
padding = (random.randint(1, 2), random.randint(1, 2))

# batch = 128
# in_feature = 16
# out_feature = 3
# kernel_H = 4
# kernel_W = 4
# input_H = 7
# input_W = 7
# stride = (1, 1)
# padding = (0, 0)

print(f"""
batch = {batch}
in_feature = {in_feature}
out_feature = {out_feature}
input size : {input_H} X {input_W}
kernel size : {kernel_H} X {kernel_W}
stride = {stride}
padding = {padding}
""")

train_input = torch.rand((batch, in_feature, input_H, input_W))
# train_input = torch.rand((batch, in_feature))

model_fp32 = ModelQuant(
    # nn.Sequential(
    #     Conv2dActivion(
    #         in_feature, out_feature, (kernel_H, kernel_W), stride=stride, padding=padding),
    #     # Conv2dActivion(
    #     #     out_feature, out_feature, (kernel_H, kernel_W), stride=stride, padding=padding),
    # )
    nn.Conv2d(
        in_feature, out_feature, (kernel_H, kernel_W), stride=stride, padding=padding),
    # Conv2dBatchNormActivion(
    #     in_feature, out_feature, (kernel_H, kernel_W), stride=stride, padding=padding),
    # LinearActivion(in_feature, out_feature),
    # nn.Linear(in_feature, out_feature),
).to('cpu')

print(f"""\
model_fp32 :
{model_fp32}""")

model_fp32.eval()
model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')

IS_FUSE = True
if IS_FUSE:
    model_fp32.fuse_model()
    print(f"""\
    fused model_fp32 :
    {model_fp32}""")

model_fp32 = quantization.prepare(model_fp32)
print(f"""\
prepared model_fp32 :
{model_fp32}""")

model_fp32(train_input)

model_int8 = quantization.convert(model_fp32)
print(f"""\
model_int8 :
{model_int8}""")
# print(f"""\
# model_int8 state_dict :
# {model_int8.state_dict()}""")

eval_input = train_input[:2]

exact_model_int8_quant_out = model_int8.quant(eval_input)
exact_model_int8_layers_out = model_int8.layers(exact_model_int8_quant_out)

exact = model_int8(eval_input)

IS_CUDA = True
model_int8_replaced = replace_layers(model_int8, is_cuda=IS_CUDA)
if IS_CUDA:
    eval_input = eval_input.to('cuda')
# print(f"""\
# model_int8_replaced :
# {model_int8_replaced}""")

mine_model_int8_quant_out = model_int8_replaced.quant(eval_input)
mine_model_int8_layers_out = model_int8_replaced.layers(
    mine_model_int8_quant_out)

compare(exact_model_int8_layers_out.int_repr().int(
), mine_model_int8_layers_out.int_repr().int().to('cpu'), 'model_int8_layers_out')

# mine = model_int8_replaced(input)
# print(mine)

# mine = mine.to('cpu')
# compare(exact, mine, 'model_int8_layers_out')
