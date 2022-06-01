#ifndef TENSOR_COMPUTE_H
#define TENSOR_COMPUTE_H

#include <cuda.h>
#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor dot_mul(const torch::Tensor &a, const torch::Tensor &b);
torch::Tensor conv2d(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &bias, const at::IntArrayRef &stride, const at::IntArrayRef &padding);

#endif