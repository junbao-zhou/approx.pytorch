#ifndef TORCH_EXAMPLE_H
#define TORCH_EXAMPLE_H

#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor rand2d(int x, int y);

#endif