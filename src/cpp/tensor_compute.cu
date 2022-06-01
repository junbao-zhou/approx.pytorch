#include <pybind11/pybind11.h>
#include "tensor_compute.cuh"
#include "util.h"
#include <stdio.h>
#include <initializer_list>
#include "approx_mul_dict.h"

template <typename cpp_type>
constexpr at::ScalarType cpp_to_scalar_type()
{
    return at::ScalarType::Undefined;
}
#define CPP_TO_SCALAR_TYPES(cpp_type, scalar_type)          \
    template <>                                             \
    constexpr at::ScalarType cpp_to_scalar_type<cpp_type>() \
    {                                                       \
        return at::ScalarType::scalar_type;                 \
    }

CPP_TO_SCALAR_TYPES(uint8_t, Byte)
CPP_TO_SCALAR_TYPES(int8_t, Char)
CPP_TO_SCALAR_TYPES(int16_t, Short)
CPP_TO_SCALAR_TYPES(int32_t, Int)
CPP_TO_SCALAR_TYPES(int64_t, Long)
CPP_TO_SCALAR_TYPES(float, Float)
CPP_TO_SCALAR_TYPES(double, Double)

#define AT_DISPATCH_CUSTOM_TYPES(TYPE, NAME, ...)                                \
    [&] {                                                                        \
        const auto &the_type = TYPE;                                             \
        /* don't use TYPE again in case it is an expensive or side-effect op  */ \
        at::ScalarType _st = ::detail::scalar_type(the_type);                    \
        switch (_st)                                                             \
        {                                                                        \
            FOR_EACH(AT_PRIVATE_CASE_TYPE_WARP, __VA_ARGS__)                     \
        default:                                                                 \
            AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
        }                                                                        \
    }()

template <typename scalar_t, int thread_compute_num>
__global__ void dot_mul_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 1> a,
    const torch::PackedTensorAccessor32<scalar_t, 1> b,
    torch::PackedTensorAccessor32<scalar_t, 1> result)
{
    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t offset = global_thread_id * thread_compute_num;
    const uint32_t boundary = min(offset + thread_compute_num, a.size(0));
    for (uint32_t i = offset; i < boundary; ++i)
        result[i] = a[i] * b[i];
}

torch::Tensor dot_mul(const torch::Tensor &a, const torch::Tensor &b)
{
    constexpr uint32_t threads_per_block = 1 << 6;
    constexpr uint32_t thread_compute_num = 32;
    constexpr uint32_t block_compute_num = threads_per_block * thread_compute_num;
    auto a_flattened = torch::flatten(a);
    auto b_flattened = torch::flatten(b);

    assert(a_flattened.sizes() == b_flattened.sizes());
    assert(a_flattened.dtype() == b_flattened.dtype());

    uint64_t N = a_flattened.size(0);
    uint32_t blocks_num = (N + block_compute_num - 1) / block_compute_num;

    if (a_flattened.device().type() == torch::kCPU)
        a_flattened = a_flattened.to(torch::kCUDA);
    if (b_flattened.device().type() == torch::kCPU)
        b_flattened = b_flattened.to(torch::kCUDA);

    auto options =
        torch::TensorOptions()
            .dtype(a_flattened.dtype())
            .device(torch::kCUDA);

    torch::Tensor result = torch::empty(a_flattened.sizes(), options);

#define AT_PRIVATE_CASE_TYPE_WARP(type)                                                    \
    AT_PRIVATE_CASE_TYPE(                                                                  \
        cpp_to_scalar_type<type>(), type,                                       \
        [&]()                                                                              \
        {                                                                                  \
            dot_mul_cuda<scalar_t, thread_compute_num><<<blocks_num, threads_per_block>>>( \
                a_flattened.packed_accessor32<scalar_t, 1>(),                              \
                b_flattened.packed_accessor32<scalar_t, 1>(),                              \
                result.packed_accessor32<scalar_t, 1>());                                  \
        })
    AT_DISPATCH_CUSTOM_TYPES(
        a_flattened.scalar_type(), "dot_mul", int32_t, float);
#undef AT_PRIVATE_CASE_TYPE_WARP

    cudaDeviceSynchronize();

    return result.view(a.sizes());
}

#define CHECK_CONV2D_INPUT_NO_BIAS(input, weight, stride, padding)                 \
    assert(input.device().type() == torch::kCUDA);                                 \
    assert(weight.device().type() == torch::kCUDA);                                \
    assert(input.dim() == 4);                                                      \
    assert(weight.dim() == 4);                                                     \
    const uint32_t N = input.size(0);                                              \
    const uint32_t C_in = input.size(1);                                           \
    const uint32_t H_image = input.size(2);                                        \
    const uint32_t W_image = input.size(3);                                        \
    assert(weight.size(1) == C_in);                                                \
    const uint32_t C_out = weight.size(0);                                         \
    const uint32_t H_kernel = weight.size(2);                                      \
    const uint32_t W_kernel = weight.size(3);                                      \
    assert(stride.size() == 2);                                                    \
    const uint32_t H_stride = stride[0];                                           \
    const uint32_t W_stride = stride[1];                                           \
    assert(padding.size() == 2);                                                   \
    const uint32_t H_padding = padding[0];                                         \
    const uint32_t W_padding = padding[1];                                         \
    assert(H_image + 2 * H_padding >= H_kernel);                                   \
    assert(W_image + 2 * W_padding >= W_kernel);                                   \
    const uint32_t H_output = (H_image + 2 * H_padding - H_kernel) / H_stride + 1; \
    const uint32_t W_output = (W_image + 2 * W_padding - W_kernel) / W_stride + 1

#define CHECK_CONV2D_INPUT(input, weight, bias, stride, padding) \
    CHECK_CONV2D_INPUT_NO_BIAS(input, weight, stride, padding);  \
    assert(bias.device().type() == torch::kCUDA);                \
    assert(bias.dim() == 1);                                     \
    assert(bias.size(0) == C_out)

template <typename scalar_t>
__global__ void conv2d_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 4> input,
    const torch::PackedTensorAccessor32<scalar_t, 4> weight,
    const torch::PackedTensorAccessor32<scalar_t, 1> bias,
    const uint32_t H_stride, const uint32_t W_stride,
    const uint32_t H_padding, const uint32_t W_padding,
    const uint32_t C_OUT,
    torch::PackedTensorAccessor32<scalar_t, 4> output)
{
    uint32_t c_in = threadIdx.x;
    uint32_t row_kernel = threadIdx.y;
    uint32_t col_kernel = threadIdx.z;

    bool is_kernel_first = (c_in == 0 && row_kernel == 0 && col_kernel == 0);
    __shared__ scalar_t output_local;
    if (is_kernel_first)
        output_local = 0;
    __syncthreads();

    uint32_t batch_id = blockIdx.x / C_OUT;
    uint32_t c_out = blockIdx.x % C_OUT;
    uint32_t row_output = blockIdx.y;
    uint32_t col_output = blockIdx.z;

    int32_t row_input = row_output * H_stride + row_kernel - H_padding;
    int32_t col_input = col_output * W_stride + col_kernel - W_padding;
    if (row_input >= 0 && row_input < input.size(2) && col_input >= 0 && col_input < input.size(3))
        atomicAdd(&output_local, input[batch_id][c_in][row_input][col_input] * weight[c_out][c_in][row_kernel][col_kernel]);
    __syncthreads();
    if (is_kernel_first)
    {
        output_local += bias[c_out];
        output[batch_id][c_out][row_output][col_output] = output_local;
    }
}

torch::Tensor conv2d_naive(torch::Tensor &input, torch::Tensor &weight, torch::Tensor &bias, at::IntArrayRef stride, at::IntArrayRef padding)
{

    CHECK_CONV2D_INPUT(input, weight, bias, stride, padding);

    dim3 threads_per_block(C_in, H_kernel, W_kernel);
    dim3 blocks(N * C_out, (H_image + 2 * H_padding - H_kernel) / H_stride + 1, (W_image + 2 * W_padding - W_kernel) / W_stride + 1);
    auto options =
        torch::TensorOptions()
            .dtype(input.dtype())
            .device(torch::kCUDA);
    auto output = torch::zeros({N, C_out, blocks.y, blocks.z}, options);

#define AT_PRIVATE_CASE_TYPE_WARP(type)                           \
    AT_PRIVATE_CASE_TYPE(                                         \
        cpp_to_scalar_type<type>(), type,         \
        [&]()                                                     \
        {                                                         \
            conv2d_cuda<scalar_t><<<blocks, threads_per_block>>>( \
                input.packed_accessor32<scalar_t, 4>(),           \
                weight.packed_accessor32<scalar_t, 4>(),          \
                bias.packed_accessor32<scalar_t, 1>(),            \
                H_stride, W_stride,                               \
                H_padding, W_padding,                             \
                C_out,                                            \
                output.packed_accessor32<scalar_t, 4>());         \
        })
    AT_DISPATCH_CUSTOM_TYPES(
        input.scalar_type(), "conv2d_naive", int32_t, float);
#undef AT_PRIVATE_CASE_TYPE_WARP

    cudaDeviceSynchronize();
    return output;
}

constexpr uint32_t MAX_INPUT_TILE_H = 20, MAX_INPUT_TILE_W = 20;
constexpr uint32_t MAX_OUTPUT_TILE_H = 8, MAX_OUTPUT_TILE_W = 8;
constexpr uint32_t MAX_KERNEL_H = 12, MAX_KERNEL_W = 12;
constexpr uint32_t MAX_CHANNEL_TILE = 16;

template <typename scalar_t>
__global__ void conv2d_cuda_tiled(
    const torch::PackedTensorAccessor32<scalar_t, 4> input,
    const torch::PackedTensorAccessor32<scalar_t, 4> weight,
    torch::PackedTensorAccessor32<scalar_t, 5> output,
    const uint32_t H_stride, const uint32_t W_stride,
    const uint32_t H_padding, const uint32_t W_padding,
    const uint32_t actual_input_tile_h, const uint32_t actual_input_tile_w,
    const uint32_t channel_tile_num)
{
    DIV(blockIdx.x, weight.size(0), batch_id, c_out, uint32_t);
    DIV(blockIdx.y, channel_tile_num, row_tile, channel_tile_id, uint32_t);
    const uint32_t col_tile = blockIdx.z;

    const uint32_t actual_output_tile_h = blockDim.x;
    const uint32_t actual_output_tile_w = blockDim.y;
    const uint32_t H_kernel = weight.size(2);
    const uint32_t W_kernel = weight.size(3);

    const uint32_t row_output_tile = threadIdx.x;
    const uint32_t col_output_tile = threadIdx.y;
    const uint32_t local_channel_id = threadIdx.z;

    const uint32_t c_in_id = channel_tile_id * MAX_CHANNEL_TILE + local_channel_id;

    // printf("thread_x : %u, thread_y : %u, col_output_tile : %u\n", threadIdx.x, threadIdx.y, col_output_tile);
    __shared__ scalar_t weight_shared[MAX_CHANNEL_TILE][MAX_KERNEL_H][MAX_KERNEL_W];
    for (uint32_t row = threadIdx.x; row < H_kernel; row += blockDim.x)
        for (uint32_t col = threadIdx.y; col < W_kernel; col += blockDim.y)
            if (c_in_id < weight.size(1))
                weight_shared[local_channel_id][row][col] = weight[c_out][c_in_id][row][col];

    __shared__ scalar_t input_tile_shared[MAX_CHANNEL_TILE][MAX_INPUT_TILE_H][MAX_INPUT_TILE_W];
    uint32_t input_tile_stride_h = actual_output_tile_h * H_stride;
    uint32_t input_tile_stride_w = actual_output_tile_w * W_stride;
    for (uint32_t row = threadIdx.x; row < actual_input_tile_h; row += blockDim.x)
        for (uint32_t col = threadIdx.y; col < actual_input_tile_w; col += blockDim.y)
        {
            int32_t row_input = row_tile * input_tile_stride_h + row - H_padding;
            int32_t col_input = col_tile * input_tile_stride_w + col - W_padding;
            if (row_input >= 0 && row_input < input.size(2) &&
                col_input >= 0 && col_input < input.size(3) &&
                c_in_id < input.size(1))
                input_tile_shared[local_channel_id][row][col] = input[batch_id][c_in_id][row_input][col_input];
            else
                input_tile_shared[local_channel_id][row][col] = 0;
        }
    __syncthreads();

    __shared__ scalar_t output_tile_shared[MAX_CHANNEL_TILE][MAX_OUTPUT_TILE_H][MAX_OUTPUT_TILE_W];
    output_tile_shared[local_channel_id][row_output_tile][col_output_tile] = 0;
    // scalar_t output_point = 0;
    for (uint32_t row_kernel = 0; row_kernel < H_kernel; ++row_kernel)
        for (uint32_t col_kernel = 0; col_kernel < W_kernel; ++col_kernel)
            output_tile_shared[local_channel_id][row_output_tile][col_output_tile] +=
                weight_shared[local_channel_id][row_kernel][col_kernel] * input_tile_shared[local_channel_id][row_output_tile * H_stride + row_kernel][col_output_tile * W_stride + col_kernel];

    Unroll_device::call_1(
        [&](uint stride)
        {
            __syncthreads();
            if (local_channel_id < stride)
                output_tile_shared[local_channel_id][row_output_tile][col_output_tile] +=
                    output_tile_shared[local_channel_id + stride][row_output_tile][col_output_tile];
        },
        8, 4, 2, 1);
    if (local_channel_id < 1)
    {
        const uint32_t row_output = row_tile * actual_output_tile_h + row_output_tile;
        const uint32_t col_output = col_tile * actual_output_tile_w + col_output_tile;

        if ((row_output < output.size(3)) && (col_output < output.size(4)))
            output[batch_id][c_out][channel_tile_id][row_output][col_output] = output_tile_shared[local_channel_id][row_output_tile][col_output_tile];
    }
}

torch::Tensor conv2d(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &bias, const at::IntArrayRef &stride, const at::IntArrayRef &padding)
{
    CHECK_CONV2D_INPUT(input, weight, bias, stride, padding);

    const uint32_t actual_output_tile_h = min((MAX_INPUT_TILE_H - H_kernel) / H_stride + 1, MAX_OUTPUT_TILE_H);
    const uint32_t actual_output_tile_w = min((MAX_INPUT_TILE_W - W_kernel) / W_stride + 1, MAX_OUTPUT_TILE_W);

    const uint32_t actual_input_tile_h = (actual_output_tile_h - 1) * H_stride + H_kernel;
    const uint32_t actual_input_tile_w = (actual_output_tile_w - 1) * W_stride + W_kernel;

    dim3 threads_per_block(actual_output_tile_h, actual_output_tile_w, MAX_CHANNEL_TILE);
    const uint32_t channel_tile_num = int_div_ceil(C_in, MAX_CHANNEL_TILE);

    const uint32_t H_tiles = int_div_ceil(H_output, actual_output_tile_h);
    const uint32_t W_tiles = int_div_ceil(W_output, actual_output_tile_w);

    dim3 blocks_grid(N * C_out, channel_tile_num * H_tiles, W_tiles);

    auto options =
        torch::TensorOptions()
            .dtype(input.dtype())
            .device(torch::kCUDA);
    auto output = torch::empty({N, C_out, channel_tile_num, H_output, W_output}, options);

#define AT_PRIVATE_CASE_TYPE_WARP(type)                                      \
    AT_PRIVATE_CASE_TYPE(                                                    \
        cpp_to_scalar_type<type>(), type,                          \
        [&]()                                                                \
        {                                                                    \
            conv2d_cuda_tiled<scalar_t><<<blocks_grid, threads_per_block>>>( \
                input.packed_accessor32<scalar_t, 4>(),                      \
                weight.packed_accessor32<scalar_t, 4>(),                     \
                output.packed_accessor32<scalar_t, 5>(),                     \
                H_stride, W_stride,                                          \
                H_padding, W_padding,                                        \
                actual_input_tile_h, actual_input_tile_w,                    \
                channel_tile_num);                                           \
        })
    AT_DISPATCH_CUSTOM_TYPES(
        input.scalar_type(), "conv2d", int32_t, float, int16_t, int8_t, uint8_t);
#undef AT_PRIVATE_CASE_TYPE_WARP

    cudaDeviceSynchronize();
    output = output.sum({2}, false, output.scalar_type());
    output.add_(bias.view({1, bias.size(0), 1, 1}));
    return output;
}

template <bool is_approx>
__device__ inline int32_t mul(const int16_t &a, const int16_t &b, const int16_t dict[0x10000])
{
}
template <>
__device__ inline int32_t mul<false>(const int16_t &a, const int16_t &b, const int16_t dict[0x10000])
{
    return a * b;
}
template <>
__device__ inline int32_t mul<true>(const int16_t &a, const int16_t &b, const int16_t dict[0x10000])
{
    return (int32_t)dict[uint8_t(b) | (uint8_t(a) << 8)];
}

__constant__ int input_zero_point;
template <bool is_approx>
__global__ void qconv2d_cuda_tiled(
    const torch::PackedTensorAccessor32<at::quint8, 4> input,
    const torch::PackedTensorAccessor32<int8_t, 4> weight,
    torch::PackedTensorAccessor32<int32_t, 5> output,
    const uint32_t H_stride, const uint32_t W_stride,
    const uint32_t H_padding, const uint32_t W_padding,
    const uint32_t actual_input_tile_h, const uint32_t actual_input_tile_w,
    const uint32_t channel_tile_num,
    const int16_t dict[0x10000])
{
    DIV(blockIdx.x, weight.size(0), batch_id, c_out, uint32_t);
    DIV(blockIdx.y, channel_tile_num, row_tile, channel_tile_id, uint32_t);
    const uint32_t col_tile = blockIdx.z;

    const uint32_t actual_output_tile_h = blockDim.x;
    const uint32_t actual_output_tile_w = blockDim.y;
    const uint32_t H_kernel = weight.size(2);
    const uint32_t W_kernel = weight.size(3);

    const uint32_t row_output_tile = threadIdx.x;
    const uint32_t col_output_tile = threadIdx.y;
    const uint32_t local_channel_id = threadIdx.z;

    const uint32_t c_in_id = channel_tile_id * MAX_CHANNEL_TILE + local_channel_id;

    // printf("thread_x : %u, thread_y : %u, col_output_tile : %u\n", threadIdx.x, threadIdx.y, col_output_tile);
    __shared__ int16_t weight_shared[MAX_CHANNEL_TILE][MAX_KERNEL_H][MAX_KERNEL_W];
    for (uint32_t row = threadIdx.x; row < H_kernel; row += blockDim.x)
        for (uint32_t col = threadIdx.y; col < W_kernel; col += blockDim.y)
            if (c_in_id < weight.size(1))
                weight_shared[local_channel_id][row][col] = weight[c_out][c_in_id][row][col];

    __shared__ int16_t input_tile_shared[MAX_CHANNEL_TILE][MAX_INPUT_TILE_H][MAX_INPUT_TILE_W];
    uint32_t input_tile_stride_h = actual_output_tile_h * H_stride;
    uint32_t input_tile_stride_w = actual_output_tile_w * W_stride;
    for (uint32_t row = threadIdx.x; row < actual_input_tile_h; row += blockDim.x)
        for (uint32_t col = threadIdx.y; col < actual_input_tile_w; col += blockDim.y)
        {
            int32_t row_input = row_tile * input_tile_stride_h + row - H_padding;
            int32_t col_input = col_tile * input_tile_stride_w + col - W_padding;
            if (row_input >= 0 && row_input < input.size(2) &&
                col_input >= 0 && col_input < input.size(3) &&
                c_in_id < input.size(1))
                input_tile_shared[local_channel_id][row][col] = input[batch_id][c_in_id][row_input][col_input].val_ - input_zero_point;
            else
                input_tile_shared[local_channel_id][row][col] = 0;
        }
    __syncthreads();

    __shared__ int32_t output_tile_shared[MAX_CHANNEL_TILE][MAX_OUTPUT_TILE_H][MAX_OUTPUT_TILE_W];
    output_tile_shared[local_channel_id][row_output_tile][col_output_tile] = 0;
    for (uint32_t row_kernel = 0; row_kernel < H_kernel; ++row_kernel)
        for (uint32_t col_kernel = 0; col_kernel < W_kernel; ++col_kernel)
            output_tile_shared[local_channel_id][row_output_tile][col_output_tile] +=
                mul<is_approx>(
                    weight_shared[local_channel_id][row_kernel][col_kernel],
                    input_tile_shared[local_channel_id][row_output_tile * H_stride + row_kernel][col_output_tile * W_stride + col_kernel],
                    dict);

    Unroll_device::call_1(
        [&](uint stride)
        {
            __syncthreads();
            if (local_channel_id < stride)
                output_tile_shared[local_channel_id][row_output_tile][col_output_tile] +=
                    output_tile_shared[local_channel_id + stride][row_output_tile][col_output_tile];
        },
        8, 4, 2, 1);
    if (local_channel_id < 1)
    {
        const uint32_t row_output = row_tile * actual_output_tile_h + row_output_tile;
        const uint32_t col_output = col_tile * actual_output_tile_w + col_output_tile;

        if ((row_output < output.size(3)) && (col_output < output.size(4)))
            output[batch_id][c_out][channel_tile_id][row_output][col_output] = output_tile_shared[local_channel_id][row_output_tile][col_output_tile];
    }
}

torch::Tensor qconv2d(const torch::Tensor &input, const torch::Tensor &weight, const at::IntArrayRef &stride, const at::IntArrayRef &padding, bool is_approx)
{
    CHECK_CONV2D_INPUT_NO_BIAS(input, weight, stride, padding);

    const uint32_t actual_output_tile_h = min((MAX_INPUT_TILE_H - H_kernel) / H_stride + 1, MAX_OUTPUT_TILE_H);
    const uint32_t actual_output_tile_w = min((MAX_INPUT_TILE_W - W_kernel) / W_stride + 1, MAX_OUTPUT_TILE_W);

    const uint32_t actual_input_tile_h = (actual_output_tile_h - 1) * H_stride + H_kernel;
    const uint32_t actual_input_tile_w = (actual_output_tile_w - 1) * W_stride + W_kernel;

    dim3 threads_per_block(actual_output_tile_h, actual_output_tile_w, MAX_CHANNEL_TILE);
    const uint32_t channel_tile_num = int_div_ceil(C_in, MAX_CHANNEL_TILE);

    const uint32_t H_tiles = int_div_ceil(H_output, actual_output_tile_h);
    const uint32_t W_tiles = int_div_ceil(W_output, actual_output_tile_w);

    dim3 blocks_grid(N * C_out, channel_tile_num * H_tiles, W_tiles);

    auto options =
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA);
    auto output = torch::empty({N, C_out, channel_tile_num, H_output, W_output}, options);

    {
        int i_zero_point = input.q_zero_point();
        cudaMemcpyToSymbol(input_zero_point, &i_zero_point, sizeof(int));
    }

    int16_t *cuda_mul_dict;
    cudaMalloc((void **)&cuda_mul_dict, sizeof(mul_dict));
    cudaMemcpy((void *)cuda_mul_dict, (void *)mul_dict, sizeof(mul_dict), cudaMemcpyHostToDevice);

    if (is_approx)
        qconv2d_cuda_tiled<true><<<blocks_grid, threads_per_block>>>(
            input.packed_accessor32<at::quint8, 4>(),
            weight.packed_accessor32<int8_t, 4>(),
            output.packed_accessor32<int32_t, 5>(),
            H_stride, W_stride,
            H_padding, W_padding,
            actual_input_tile_h, actual_input_tile_w,
            channel_tile_num,
            cuda_mul_dict);
    else
        qconv2d_cuda_tiled<false><<<blocks_grid, threads_per_block>>>(
            input.packed_accessor32<at::quint8, 4>(),
            weight.packed_accessor32<int8_t, 4>(),
            output.packed_accessor32<int32_t, 5>(),
            H_stride, W_stride,
            H_padding, W_padding,
            actual_input_tile_h, actual_input_tile_w,
            channel_tile_num,
            cuda_mul_dict);

    cudaDeviceSynchronize();
    output = output.sum({2}, false, output.scalar_type());
    return output;
}

constexpr uint32_t MAT_MUL_BLOCK_SIZE = 16;

template <typename scalar_t>
__global__ void mat_mul_cuda(
    const torch::PackedTensorAccessor32<scalar_t, 2> input,
    const torch::PackedTensorAccessor32<scalar_t, 2> weight,
    const torch::PackedTensorAccessor32<scalar_t, 1> bias,
    torch::PackedTensorAccessor32<scalar_t, 2> output,
    const uint32_t in_features_block_num)
{
    uint32_t i, j;
    scalar_t temp = 0;
    const uint32_t N_batch = input.size(0);
    const uint32_t in_features = input.size(1);
    const uint32_t out_features = weight.size(0);

    __shared__ scalar_t input_shared[MAT_MUL_BLOCK_SIZE][MAT_MUL_BLOCK_SIZE];
    __shared__ scalar_t weight_shared[MAT_MUL_BLOCK_SIZE][MAT_MUL_BLOCK_SIZE];

    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t tileNUM = 0; tileNUM < in_features_block_num; ++tileNUM)
    {
        j = tileNUM * MAT_MUL_BLOCK_SIZE + threadIdx.x;
        i = tileNUM * MAT_MUL_BLOCK_SIZE + threadIdx.y;

        if (row < N_batch && j < in_features)
            input_shared[threadIdx.y][threadIdx.x] = input[row][j];
        else
            input_shared[threadIdx.y][threadIdx.x] = 0;

        if (col < out_features && i < in_features)
            weight_shared[threadIdx.y][threadIdx.x] = weight[col][i];
        else
            weight_shared[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();
        for (uint32_t k = 0; k < MAT_MUL_BLOCK_SIZE; ++k)
        {

            temp += input_shared[threadIdx.y][k] * weight_shared[k][threadIdx.x]; // no shared memory bank conflict
        }
        __syncthreads();
    }
    if (row < N_batch && col < out_features)
        output[row][col] = temp + bias[col];
}

#define CHECK_MAT_MUL_INPUT_NO_BIAS(input, weight)  \
    assert(input.device().type() == torch::kCUDA);  \
    assert(weight.device().type() == torch::kCUDA); \
    assert(input.dim() == 2);                       \
    assert(weight.dim() == 2);                      \
    const uint32_t N_batch = input.size(0);         \
    const uint32_t in_features = input.size(1);     \
    assert(weight.size(1) == in_features);          \
    const uint32_t out_features = weight.size(0)

#define CHECK_MAT_MUL_INPUT(input, weight, bias)  \
    CHECK_MAT_MUL_INPUT_NO_BIAS(input, weight);   \
    assert(bias.device().type() == torch::kCUDA); \
    assert(bias.dim() == 1);                      \
    assert(bias.size(0) == out_features)

torch::Tensor mat_mul(const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &bias)
{
    CHECK_MAT_MUL_INPUT(input, weight, bias);

    dim3 Block_dim(MAT_MUL_BLOCK_SIZE, MAT_MUL_BLOCK_SIZE);
    // Grid dimension is found by dividing matrix dimension to block_size
    dim3 Grid_dim(int_div_ceil(out_features, MAT_MUL_BLOCK_SIZE), int_div_ceil(N_batch, MAT_MUL_BLOCK_SIZE));
    const uint32_t in_features_block_num = int_div_ceil(in_features, MAT_MUL_BLOCK_SIZE);
    auto options =
        torch::TensorOptions()
            .dtype(input.dtype())
            .device(torch::kCUDA);
    auto output = torch::empty({N_batch, out_features}, options);

#define AT_PRIVATE_CASE_TYPE_WARP(type)                      \
    AT_PRIVATE_CASE_TYPE(                                    \
        cpp_to_scalar_type<type>(), type,          \
        [&]()                                                \
        {                                                    \
            mat_mul_cuda<scalar_t><<<Grid_dim, Block_dim>>>( \
                input.packed_accessor32<scalar_t, 2>(),      \
                weight.packed_accessor32<scalar_t, 2>(),     \
                bias.packed_accessor32<scalar_t, 1>(),       \
                output.packed_accessor32<scalar_t, 2>(),     \
                in_features_block_num);                      \
        })
    AT_DISPATCH_CUSTOM_TYPES(
        input.scalar_type(), "conv2d", int64_t, int32_t, double, float, int16_t, int8_t);
#undef AT_PRIVATE_CASE_TYPE_WARP

    cudaDeviceSynchronize();
    return output;
}

__global__ void q_mat_mul_cuda(
    const torch::PackedTensorAccessor32<at::quint8, 2> input,
    const torch::PackedTensorAccessor32<int8_t, 2> weight,
    torch::PackedTensorAccessor32<int32_t, 2> output,
    const uint32_t in_features_block_num)
{
    uint32_t i, j;
    int32_t temp = 0;
    const uint32_t N_batch = input.size(0);
    const uint32_t in_features = input.size(1);
    const uint32_t out_features = weight.size(0);

    __shared__ int16_t input_shared[MAT_MUL_BLOCK_SIZE][MAT_MUL_BLOCK_SIZE];
    __shared__ int16_t weight_shared[MAT_MUL_BLOCK_SIZE][MAT_MUL_BLOCK_SIZE];

    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t tileNUM = 0; tileNUM < in_features_block_num; ++tileNUM)
    {
        j = tileNUM * MAT_MUL_BLOCK_SIZE + threadIdx.x;
        i = tileNUM * MAT_MUL_BLOCK_SIZE + threadIdx.y;

        if (row < N_batch && j < in_features)
            input_shared[threadIdx.y][threadIdx.x] = input[row][j].val_ - input_zero_point;
        else
            input_shared[threadIdx.y][threadIdx.x] = 0;

        if (col < out_features && i < in_features)
            weight_shared[threadIdx.y][threadIdx.x] = weight[col][i];
        else
            weight_shared[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();
        for (uint32_t k = 0; k < MAT_MUL_BLOCK_SIZE; ++k)
        {

            temp += input_shared[threadIdx.y][k] * weight_shared[k][threadIdx.x]; // no shared memory bank conflict
        }
        __syncthreads();
    }
    if (row < N_batch && col < out_features)
        output[row][col] = temp;
}

torch::Tensor q_mat_mul(const torch::Tensor &input, const torch::Tensor &weight)
{
    CHECK_MAT_MUL_INPUT_NO_BIAS(input, weight);

    dim3 Block_dim(MAT_MUL_BLOCK_SIZE, MAT_MUL_BLOCK_SIZE);
    dim3 Grid_dim(int_div_ceil(out_features, MAT_MUL_BLOCK_SIZE), int_div_ceil(N_batch, MAT_MUL_BLOCK_SIZE));
    const uint32_t in_features_block_num = int_div_ceil(in_features, MAT_MUL_BLOCK_SIZE);
    auto options =
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA);
    auto output = torch::empty({N_batch, out_features}, options);
    {
        int i_zero_point = input.q_zero_point();
        cudaMemcpyToSymbol(input_zero_point, &i_zero_point, sizeof(int));
    }

    q_mat_mul_cuda<<<Grid_dim, Block_dim>>>(
        input.packed_accessor32<at::quint8, 2>(),
        weight.packed_accessor32<int8_t, 2>(),
        output.packed_accessor32<int32_t, 2>(),
        in_features_block_num);
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dot_mul", &dot_mul, "Tensor dot multiplication");
    m.def("conv2d", &conv2d, "conv2d");
    m.def("qconv2d", &qconv2d, "qconv2d");
    m.def("mat_mul", &mat_mul, "mat_mul");
    m.def("q_mat_mul", &q_mat_mul, "q_mat_mul");
}

