
import importlib
from data_loader import data_loader
from model_run import train, validate
import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.functional as F
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq
import quantized_model as q_model
from util import compare
import torchvision
import argparse
from torchvision.models import AlexNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
print(DEVICE)

criterion = nn.CrossEntropyLoss()


def validate_model(model, valid_loader, device='cpu', print_every=0):
    valid_loss, valid_acc = validate(
        valid_loader, model, criterion, device, print_every=print_every)


def model_fp32_to_int8(model_fp32, valid_loader, is_fuse=True, is_train=False):

    print(f"""\
    model_fp32 : 
    {model_fp32}""")
    model_fp32.eval()

    if is_fuse:
        q_model.fuse_model(model_fp32)
        # print(f"""\
        # fused model_fp32 :
        # {model_fp32}""")

    model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')

    quantization.prepare(model_fp32, inplace=True)
    print(f"""\
    prepared model_fp32 : 
    {model_fp32}""")

    if is_train:
        print("validate model fp32")
        validate_model(model_fp32, valid_loader, DEVICE, print_every=1)

    model_fp32 = model_fp32.to('cpu')
    model_int8 = quantization.convert(model_fp32)

    return model_int8


BATCH_SIZE = 64
model_dir = '../model/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Test quantized model'
    parser.add_argument('--model_name',
                        type=str)
    parser.add_argument('--dataset_name',
                        type=str)
    parser.add_argument('--n_classes',
                        type=int)
    parser.add_argument('--is_train_fp32',
                        action='store_true')
    parser.add_argument('--is_load_int8',
                        action='store_true')
    parser.add_argument('--is_store_int8',
                        action='store_true')
    parser.add_argument('--is_run_int8',
                        action='store_true')
    parser.add_argument('--is_approx',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = getattr(importlib.import_module("model"), args.model_name)
    _, valid_loader = data_loader(
        args.dataset_name, BATCH_SIZE, is_normalize=True, is_augment=False, image_net_out_size=(227 if args.dataset_name == 'ImageNet' else 224), is_shuffle_valid_loader=True)

    model_fp32 = q_model.ModelQuant(MODEL_NAME(args.n_classes)).to(DEVICE)
    MODEL_PATH = f'{model_dir}{type(model_fp32.layers).__name__}-{args.dataset_name}.model'

    model_fp32.layers.load_state_dict(torch.load(MODEL_PATH))

    model_int8 = model_fp32_to_int8(
        model_fp32, valid_loader, is_fuse=True, is_train=args.is_train_fp32)
    MODEL_PATH = f'{model_dir}{type(model_int8.layers).__name__}-int8-{args.dataset_name}.model'

    if args.is_load_int8:
        model_int8.load_state_dict(torch.load(MODEL_PATH))
    if args.is_store_int8:
        torch.save(model_int8.state_dict(), MODEL_PATH)
    print(f"""\
    model_int8 : 
    {model_int8}""")

    if args.is_run_int8:
        print("validate model int8")
        validate_model(model_int8, valid_loader, print_every=1)

    inputs, classes = next(iter(valid_loader))
    exact_out = model_int8(inputs)

    q_model.IS_APPROX = args.is_approx

    IS_CUDA = True
    model_int8_replaced = q_model.replace_layers(model_int8, is_cuda=IS_CUDA)
    print(f"""\
    model_int8_replaced : 
    {model_int8_replaced}""")
    if IS_CUDA:
        inputs = inputs.to('cuda')

    replaced_out = model_int8_replaced(inputs)

    replaced_out = replaced_out.to('cpu')
    compare(exact_out, replaced_out, "model_int8")

    print("validate mine model int8")
    validate_model(model_int8_replaced, valid_loader,
                   'cuda' if IS_CUDA else 'cpu', print_every=1)
