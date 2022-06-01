import random
from util import *
import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

# torch.manual_seed(1234)

torch_example = load(
    name="torch_example",
    sources=["cpp/torch_example.cpp"],
    extra_include_paths=["cpp/include"],
    verbose=True
)
a = torch_example.rand2d(3, 4)
print(a.device)

tensor_compute = load(
    name="tensor_compute",
    sources=["cpp/tensor_compute.cu"],
    extra_include_paths=["cpp/include"],
    verbose=True
)


def compare_cpu_gpu(tensor_sizes, cpu_func, gpu_func, dtype):
    input_tensors = []
    for size in tensor_sizes:
        if dtype in [torch.float, torch.float16, torch.float32, torch.float64]:
            t = torch.rand(size, dtype=dtype)
        elif dtype in [torch.int, torch.int16, torch.int32, torch.int64]:
            t = torch.randint(0, 4, size, dtype=dtype)
        input_tensors.append(t)

    # for t in input_tensors:
    #     print(f't = {t}')

    cpu_time, res_cpu = exec_time(lambda: cpu_func(input_tensors))
    print(f'cpu_time = {cpu_time}')
    # print(f'res_cpu = {res_cpu}')
    # print(f'res_cpu.max() = {res_cpu.max()}')
    # print(f'res_cpu.min() = {res_cpu.min()}')

    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].to('cuda')

    gpu_time, res_gpu = exec_time(lambda: gpu_func(input_tensors))
    print(f'gpu_time = {gpu_time}')
    # print(f'res_gpu = {res_gpu}')
    # print(f'res_gpu.max() = {res_gpu.max()}')
    # print(f'res_gpu.min() = {res_gpu.min()}')

    res_cpu = res_cpu.to('cuda')
    compare(res_cpu, res_gpu, 'res')


# batch = 128
# in_feature = 64
# out_feature = 128
# kernel_H = 3
# kernel_W = 3
# input_H = 7
# input_W = 7
# stride = (1, 1)
# padding = (0, 0)

batch = random.randint(32, 200)
in_feature = random.randint(1, 128)
out_feature = random.randint(1, 128)
kernel_H = random.randint(3, 10)
kernel_W = random.randint(3, 10)
input_H = random.randint(kernel_H, 64)
input_W = random.randint(kernel_W, 64)
stride = (random.randint(1, 3), random.randint(1, 3))
padding = (random.randint(1, 2), random.randint(1, 2))

print(f"""
batch = {batch}
in_feature = {in_feature}
out_feature = {out_feature}
input size : {input_H} X {input_W}
kernel size : {kernel_H} X {kernel_W}
stride = {stride}
padding = {padding}
""")


def test_conv():
    compare_cpu_gpu(
        [(batch, in_feature, input_H, input_W), (out_feature,
                                                 in_feature, kernel_H, kernel_W), (out_feature,)],
        lambda tensors: F.conv2d(
            tensors[0], tensors[1], tensors[2], stride=stride, padding=padding),
        lambda tensors: tensor_compute.conv2d(
            tensors[0], tensors[1], tensors[2], stride, padding),
        dtype=torch.int32
    )


def test_mat_mul():
    compare_cpu_gpu(
        [(batch, in_feature), (out_feature, in_feature), (out_feature, )],
        lambda tensors: F.linear(
            tensors[0], tensors[1], tensors[2]),
        lambda tensors: tensor_compute.mat_mul(
            tensors[0], tensors[1], tensors[2]),
        dtype=torch.int32
    )


def test_dot_mul():
    compare_cpu_gpu(
        [(batch, in_feature), (batch, in_feature)],
        lambda tensors: tensors[0] * tensors[1],
        lambda tensors: tensor_compute.dot_mul(
            tensors[0], tensors[1]),
        dtype=torch.int32
    )


test_conv()

test_mat_mul()