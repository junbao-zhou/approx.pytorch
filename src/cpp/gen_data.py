import random
from Booth import to_signed, n_bits_1, booth_mul_approx_trunc, booth_mul_approx


def mul(x_x):
    a = to_signed(x_x & 0xFF, 8)
    b = to_signed((x_x & 0xFF00) >> 8, 8)
    ret = a * b
    return ret


def nand(a_b):
    a = a_b & 0x1
    b = (a_b & 0x2) >> 1
    return ~(a & b)


def xor(a_b):
    a = a_b & 0x1
    b = (a_b & 0x2) >> 1
    return a ^ b


def xnor(a_b):
    a = a_b & 0x1
    b = (a_b & 0x2) >> 1
    return ~(a ^ b)


def gen_data(input_datas, input_bits, output_bits, func):
    with open("x.txt", 'w') as f_in:
        with open("y.txt", 'w') as f_out:
            for x in input_datas:
                f_in.write(bin(x)[2:].zfill(input_bits))
                f_in.write('\n')
                f_out.write(bin(func(x) & n_bits_1(output_bits))
                            [2:].zfill(output_bits))
                f_out.write('\n')


def gen_data_cpp(input_datas, input_bits, output_bits, func):
    with open("approx_mul_dict.cpp", 'w') as f_out:
        f_out.write("""#include <stdint.h>
#include "approx_mul_dict.h"

const int16_t mul_dict[] = {""")
        for x in input_datas:
            f_out.write(
                str(to_signed(func(x) & n_bits_1(output_bits), output_bits)))
            f_out.write(',\n')
        f_out.write('};')


if __name__ == '__main__':
    # gen_data(0xFFFF, 3, 3, lambda num : booth_code(num).to_bits())
    # gen_data(range(0x10000), 16, 16, mul)
    # gen_data(range(0x10000), 16, 16,
    #          lambda a_b: booth_mul_approx_trunc((a_b & 0xFF00) >> 8, (a_b & 0xFF), 8))
    gen_data_cpp(range(0x10000), 16, 16,
                 lambda a_b: booth_mul_approx_trunc((a_b & 0xFF00) >> 8, (a_b & 0xFF), 8))
    # gen_data(0xFFF, 2, 1, nand)
    # gen_data(0x10000, 2, 1, xor)
    # print(n_bits_1(16))
