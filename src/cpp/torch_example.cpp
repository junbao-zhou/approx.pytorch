#include <pybind11/pybind11.h>
#include "torch_example.h"

auto options =
    torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

torch::Tensor rand2d(int x, int y)
{
    return torch::rand({x, y}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rand2d", &rand2d, "torch rand 2d array");
}