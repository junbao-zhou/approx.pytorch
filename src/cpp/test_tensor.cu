#include "tensor_compute.cuh"
#include "util.h"
#include <iostream>
#include <array>
#include <vector>

template <uint32_t input_num, at::ScalarType data_type>
std::array<double, 2> compare_cpu_gpu(
    std::array<at::IntArrayRef, input_num> tensor_sizes,
    std::function<torch::Tensor(std::array<torch::Tensor, input_num>)> cpu_func,
    std::function<torch::Tensor(std::array<torch::Tensor, input_num>)> gpu_func)
{
    auto options =
        torch::TensorOptions()
            .dtype(data_type);
    std::array<torch::Tensor, input_num> input_tensors;
    for (uint32_t i = 0; i < input_num; ++i)
    {
        switch (data_type)
        {
        case at::ScalarType::Double:
        case at::ScalarType::Float:
        case at::ScalarType::Half:
            input_tensors[i] = torch::rand(tensor_sizes[i], options);
            break;
        case at::ScalarType::Int:
        case at::ScalarType::Long:
        case at::ScalarType::Short:
            input_tensors[i] = torch::randint(10, tensor_sizes[i], options);
            break;
        default:
            break;
        }
    }
    torch::Tensor res_cpu = torch::Tensor();
    double ms_cpu = exec_time(
        [&res_cpu, &input_tensors, &cpu_func]()
        {
            // for (auto &t : input_tensors)
            //     OUTPUT_VAR(t);
            res_cpu = cpu_func(input_tensors);
            // OUTPUT_VAR(res_cpu);
        });
    OUTPUT_VAR(ms_cpu);

    double ms_to_cuda = exec_time(
        [&input_tensors]()
        {
            for (auto &t : input_tensors)
                t = t.to(torch::kCUDA);
        });
    OUTPUT_VAR(ms_to_cuda);

    torch::Tensor res_gpu = torch::Tensor();
    double ms_gpu = exec_time(
        [&res_gpu, &input_tensors, &gpu_func]()
        {
            // for (auto &t : input_tensors)
            //     std::cout << t.device() << std::endl;
            res_gpu = gpu_func(input_tensors);
            // for (auto &t : input_tensors)
            //     std::cout << t.dtype() << std::endl;
            // OUTPUT_VAR(res_gpu)
        });
    OUTPUT_VAR(ms_gpu);

    res_cpu = res_cpu.to(torch::kCUDA);
    // std::cout << res_cpu.device() << std::endl;
    bool is_equal = res_cpu.equal(res_gpu);
    OUTPUT_VAR(is_equal);
    if (!is_equal)
    {
        auto error = torch::sum(res_cpu - res_gpu);
        OUTPUT_VAR(error);
    }

    return {ms_cpu, ms_gpu};
}

// void test_1()
// {
//     compare_cpu_gpu<2, torch::kFloat64>(
//         {at::IntArrayRef{0xFFF, 0xFF}, at::IntArrayRef{0xFFF, 0xFF}},
//         [](std::array<torch::Tensor, 2> inputs) -> torch::Tensor
//         {
//             auto a_flattened = torch::flatten(inputs[0]);
//             auto b_flattened = torch::flatten(inputs[1]);

//             torch::Tensor res_flattened = a_flattened * b_flattened;
//             return res_flattened.view(inputs[0].sizes());
//         },
//         [](std::array<torch::Tensor, 2> inputs) -> torch::Tensor
//         {
//             return dot_mul(inputs[0], inputs[1]);
//         });
// }

// void test_2()
// {
//     compare_cpu_gpu<3, torch::kFloat>(
//         {at::IntArrayRef{500, 3, 4, 4}, at::IntArrayRef{64, 3, 2, 2}, at::IntArrayRef{64}},
//         [](std::array<torch::Tensor, 3> inputs) -> torch::Tensor
//         {
//             return torch::conv2d(inputs[0], inputs[1], inputs[2]);
//         },
//         [](std::array<torch::Tensor, 3> inputs) -> torch::Tensor
//         {
//             return conv2d(inputs[0], inputs[1], inputs[2], {1, 1}, {0, 0});
//         });
// }

int main()
{
    // test_2();

    torch::Tensor a = torch::randint(10, {3, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kQInt8));
    std::cout << a << std::endl;

}