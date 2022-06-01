#ifndef UTIL_H
#define UTIL_H

#include <chrono>
#include <functional>
#include <cuda.h>

template <unsigned N>
struct Unroll_N
{
    template <typename F>
    inline static void call(F const &f)
    {
        f();
        Unroll_N<N - 1>::call(f);
    }
};
template <>
struct Unroll_N<0u>
{
    template <typename F>
    inline static void call(F const &) {}
};

struct Unroll
{
    template <typename F, typename X, typename... Args>
    inline static void call_1(F const &f, X const &x, Args... args)
    {
        f(x);
        Unroll::call_1(f, args...);
    }
    template <typename F, typename X>
    inline static void call_1(F const &f, X const &x)
    {
        f(x);
    }
};

struct Unroll_device
{
    template <typename F, typename X, typename... Args>
    __device__ inline static void call_1(F const &f, X const &x, Args... args)
    {
        f(x);
        Unroll_device::call_1(f, args...);
    }
    template <typename F, typename X>
    __device__ inline static void call_1(F const &f, X const &x)
    {
        f(x);
    }
};

#define FOR_EACH_1(func, x) func(x)
#define FOR_EACH_2(func, x, ...) func(x) FOR_EACH_1(func, __VA_ARGS__)
#define FOR_EACH_3(func, x, ...) func(x) FOR_EACH_2(func, __VA_ARGS__)
#define FOR_EACH_4(func, x, ...) func(x) FOR_EACH_3(func, __VA_ARGS__)
#define FOR_EACH_5(func, x, ...) func(x) FOR_EACH_4(func, __VA_ARGS__)
#define FOR_EACH_6(func, x, ...) func(x) FOR_EACH_5(func, __VA_ARGS__)
#define FOR_EACH_7(func, x, ...) func(x) FOR_EACH_6(func, __VA_ARGS__)
#define FOR_EACH_8(func, x, ...) func(x) FOR_EACH_7(func, __VA_ARGS__)
#define FOR_EACH_9(func, x, ...) func(x) FOR_EACH_8(func, __VA_ARGS__)
#define FOR_EACH_10(func, x, ...) func(x) FOR_EACH_9(func, __VA_ARGS__)
#define FOR_EACH_11(func, x, ...) func(x) FOR_EACH_10(func, __VA_ARGS__)
#define FOR_EACH_12(func, x, ...) func(x) FOR_EACH_11(func, __VA_ARGS__)
#define FOR_EACH_13(func, x, ...) func(x) FOR_EACH_12(func, __VA_ARGS__)
#define FOR_EACH_14(func, x, ...) func(x) FOR_EACH_13(func, __VA_ARGS__)

#define _NUM_ARGS_OCCUPATION(X, X64, X63, X62, X61, X60, X59, X58, X57, X56, X55, X54, X53, X52, X51, X50, X49, X48, X47, X46, X45, X44, X43, X42, X41, X40, X39, X38, X37, X36, X35, X34, X33, X32, X31, X30, X29, X28, X27, X26, X25, X24, X23, X22, X21, X20, X19, X18, X17, X16, X15, X14, X13, X12, X11, X10, X9, X8, X7, X6, X5, X4, X3, X2, X1, N, ...) N
#define NUM_ARGS(...) _NUM_ARGS_OCCUPATION(0, __VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define FOR_EACH_N(N, func, ...) FOR_EACH_##N(func, __VA_ARGS__)
#define __FOR_EACH_N_INNER(N, func, ...) FOR_EACH_N(N, func, __VA_ARGS__)
#define FOR_EACH(func, ...) __FOR_EACH_N_INNER(NUM_ARGS(__VA_ARGS__), func, __VA_ARGS__)

#define OUTPUT_VAR(x) std::cout << #x << " : " << x << std::endl

#define DIV(dividend, divisor, quotient, remainder, scalar_t) \
    const scalar_t quotient = dividend / divisor;             \
    const scalar_t remainder = dividend - quotient * divisor

template <typename scalar_t>
inline scalar_t int_div_ceil(scalar_t a, scalar_t b)
{
    return (a + b - 1) / b;
}

double exec_time(std::function<void(void)> func);

#endif