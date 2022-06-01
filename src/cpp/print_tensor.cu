#include <pybind11/pybind11.h>
#include "tensor_compute.cuh"
#include "util.h"

void print_tensor(torch::Tensor a)
{
    // auto p = a.data_ptr<at::quint8>();
    auto p = a.accessor<at::quint8, 2>();
    // OUTPUT_VAR(p);
    for (uint i = 0; i < a.size(0); ++i)
    {
        for (uint j = 0; j < a.size(1); ++j)
            std::cout << int(*(uint8_t *)(&p[i][j])) << "   ";
        std::cout << std::endl;
    }
    int8_t tmp = p[0][0].val_ - 126;
    std::cout << int(tmp) << std::endl;

    auto b = a.int_repr();
    OUTPUT_VAR(b);
    a = a.to(torch::kCUDA);
    auto q = a.packed_accessor32<at::quint8, 2>();
    // for (uint i = 0; i < a.size(0); ++i)
    // {
    //     for (uint j = 0; j < a.size(1); ++j)
    //         std::cout << q[i][j].val_ << "   ";
    //     std::cout << std::endl;
    // }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("print_tensor", &print_tensor, "");
}
