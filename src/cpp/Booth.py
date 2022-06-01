import random
import numpy as np
import matplotlib.pyplot as plt

is_debug = 0


def debug(*args):
    if is_debug:
        print(*args)
    else:
        pass


def n_bits_1(n):
    return (1 << n) - 1


def high_bits_1(n):
    return 1 << (n - 1)


def set_low_bits(target, bits_num, num):
    target &= ~n_bits_1(bits_num)
    target |= (num & n_bits_1(bits_num))
    return target


def get_bit(num, bit):
    return (num & (1 << (bit - 1))) >> (bit - 1)


def bin_num(num, bits):
    num = num & n_bits_1(bits)
    return bin(num)[2:].zfill(bits)


def to_signed(n, bits):
    n = n & n_bits_1(bits)
    return (n ^ high_bits_1(bits)) - high_bits_1(bits)


negs_list = 4*[0] + 3*[1] + [0]
twos_list = 3*[0] + 2*[1] + 3*[0]
zeros_list = [1] + 6*[0] + [1]


class booth_code:
    def __init__(self, num):
        num = num & n_bits_1(3)
        self.neg = negs_list[num]
        self.two = twos_list[num]
        self.zero = zeros_list[num]

    def to_bits(self):
        return (self.neg << 2) | (self.two << 1) | self.zero

    def __repr__(self):
        return f'( neg : {self.neg}, two : {self.two}, zero : {self.zero} )'


def plt_save(plot, name: str):
    plt.savefig(name)
    plt.show()
    plt.close()


def opposite_sign(a, b):
    return (a ^ b) < 0


class Error:
    def __init__(self):
        self.zero_deviated_approx = []
        self.non_zero_deviated_true = []
        self.non_zero_deviated_approx = []
        self.zero_cross_true = []
        self.zero_cross_approx = []

    def add(self, true, approx):
        if true == 0:
            self.zero_deviated_approx.append(approx)
        else:
            self.non_zero_deviated_true.append(true)
            self.non_zero_deviated_approx.append(approx)
        if opposite_sign(true, approx):
            self.zero_cross_true.append(true)
            self.zero_cross_approx.append(approx)

    class _error():
        def __init__(self, err_list, name: str) -> None:
            self.name = name
            self.error = np.array(err_list)
            self.err_num = (self.error != 0).sum()
            self.rate = self.err_num / self.error.__len__()
            self.mean = self.error.mean()
            self.std = self.error.std()
            self.var = self.error.var()
            self.abs = np.abs(self.error)
            self.abs_mean = self.abs.mean()
            self.abs_std = self.abs.std()
            self.abs_max = self.abs.max()
            self.abs_min = self.abs.min()

        def __repr__(self) -> str:
            plt_save(plt.hist(self.error, bins=100), f'{self.name}.png')
            return f"""
{self.name} : 
rate     = {self.rate:.3f}
mean     = {self.mean:.3f}
std      = {self.std:.3f}
var      = {self.var:.3f}
abs_mean = {self.abs_mean:.3f}
abs_std  = {self.abs_std:.3f}
abs_max  = {self.abs_max:.3f}
abs_min  = {self.abs_min:.3f}
"""

    def __repr__(self):
        self_vars = [attr for attr in self.__dir__() if not callable(
            self.__getattribute__(attr)) and not attr.startswith("__")]
        for v in self_vars:
            self.__setattr__(v, np.array(self.__getattribute__(v)))
        true_np = np.concatenate(
            [self.non_zero_deviated_true, np.zeros([self.zero_deviated_approx.__len__()])])
        approx_np = np.concatenate(
            [self.non_zero_deviated_approx, self.zero_deviated_approx])
        err = self._error(true_np - approx_np, 'Error')
        relative_err = self._error(
            np.abs(self.non_zero_deviated_approx-self.non_zero_deviated_true) /
            np.abs(self.non_zero_deviated_true),
            'Relativa_error')
        zero_deviate_err = self._error(
            self.zero_deviated_approx, 'Zero_deviate')

        plt_save(plt.scatter(
            true_np, approx_np, s=2, marker=","), 'total_error.png')
        plt_save(plt.scatter(
            self.non_zero_deviated_true, relative_err.error, s=2, marker=","), 'relative_error.png')
        return f"""
{err}
{zero_deviate_err}
{relative_err}
"""


def booth_encoder_radix_4(num, bits):
    assert (bits == 8) or (bits == 16)
    debug(f"encoding : {bin(num)}")
    num_shift = (num & n_bits_1(bits)) << 1
    three_bits_1 = n_bits_1(3)
    ret = []
    for i in range(0, bits, 2):
        code = booth_code((num_shift >> i) & three_bits_1)
        debug(f'code = {code}')
        ret.append(code)
    return ret


def partial_product(a, booth_code, bits):
    a = a & n_bits_1(bits)
    a += (a & high_bits_1(bits)) << 1
    if booth_code.zero:
        return 0
    if booth_code.two:
        a = a << 1
    if booth_code.neg:
        a = ~a
    a = a & n_bits_1(bits+1)
    debug(f'pp : {bin(a)}')
    return a


def partial_product_approx(a, booth_code, bits, approx_bits):
    assert approx_bits <= bits
    exact_pp = partial_product(a, booth_code, bits)
    if booth_code.zero:
        a = 0
    if booth_code.neg:
        a = ~a
    if approx_bits < 0:
        approx_bits = 0
    a = a & n_bits_1(approx_bits)
    res = a + (exact_pp & (~n_bits_1(approx_bits)))
    debug(f'approx pp : {bin(res)}')
    return res


def partial_products_gen(booth_codes, b, bits):
    assert (bits == 8) or (bits == 16)
    pps = []
    for i in range(len(booth_codes)):
        code = booth_codes[i]
        pps.append(partial_product(b, code, bits))
    return pps


def partial_products_approx_gen(booth_codes, b, bits, approx_bits):
    assert (bits == 8) or (bits == 16)
    pps = []
    for i in range(len(booth_codes)):
        code = booth_codes[i]
        pps.append(partial_product_approx(b, code, bits, approx_bits[i]-2*i))
    return pps


def trunc(b, trunc_bit, last_bit):
    return set_low_bits(
        b, trunc_bit, last_bit << (trunc_bit - 1))


def partial_sum_gen(partial_products, bits, booth_codes, is_neg=['add']*4):
    assert (bits == 8) or (bits == 16)
    p_sum = []
    for i in range(len(partial_products)):
        debug(f'partial_pro  = {bin(partial_products[i])}')
        neg_bits = booth_codes[i].neg
        if is_neg[i] == 'add':
            pp_neg = (partial_products[i] ^ (0x3 << bits)) + neg_bits
        elif is_neg[i] == 'or':
            pp_neg = (partial_products[i] ^ (0x3 << bits)) | neg_bits
        elif is_neg[i] == 'trunc':
            pp_neg = (partial_products[i] ^ (0x3 << bits))
        else:
            raise Exception('is_neg not regconized')
        debug(f'pp_neg      = {bin(pp_neg)}')
        shift_ps = pp_neg << (2*i)
        debug(f'shifted p_s = {bin(shift_ps)}')
        p_sum.append(shift_ps)
    return p_sum


def trunc_partial_product(partial_products, booth_codes, trunc_bits):
    for i in range(4):
        if trunc_bits[i] > 0:
            partial_products[i] = trunc(
                partial_products[i], trunc_bits[i]-2*i, (~booth_codes[i].zero))
    return partial_products


def partial_product_compress(partial_sums, bits):
    assert (bits == 8) or (bits == 16)
    res = 0
    for i in range(len(partial_sums)):
        debug(f'p_sum = {bin(partial_sums[i])}')
        res += partial_sums[i]
    res += (1 << bits)
    return res & n_bits_1(2*bits)


def four_2(a, b, c, d):
    if (not a) and (not b) and c and d:
        return 1
    elif a and b and c and d:
        return -1
    else:
        return 0


def four_2_err(partial_sums, err_bit):
    last_col = []
    for p_sum in partial_sums:
        last_col.append(get_bit(p_sum, err_bit))
    err = four_2(last_col[0], last_col[1], last_col[2], last_col[3])
    return (err << (err_bit-1))


def booth_mul(a, b, bits):
    assert (bits == 8) or (bits == 16)
    booth_codes = booth_encoder_radix_4(a, bits)
    pps = partial_products_gen(booth_codes, b, bits)
    p_sums = partial_sum_gen(pps, bits, booth_codes)
    return partial_product_compress(p_sums, bits)


def booth_mul_approx(a, b, bits):
    assert (bits == 8) or (bits == 16)
    booth_codes = booth_encoder_radix_4(a, bits)
    pps = partial_products_approx_gen(booth_codes, b, bits, [5, 5, 5, 5])
    p_sums = partial_sum_gen(
        pps, bits, booth_codes,
        is_neg=['or', 'or', 'or', 'add'])
    res = partial_product_compress(p_sums, bits)
    # if b == 0:
    #     res = 0
    return res


def booth_mul_approx_trunc(a, b, bits):
    assert (bits == 8)
    booth_codes = booth_encoder_radix_4(a, bits)
    trunc_bit = 7
    pps = partial_products_approx_gen(
        booth_codes, b, bits, [5, 5, 5, 0])
    pps = trunc_partial_product(pps, booth_codes, [5, 5, 0, 0])
    p_sums = partial_sum_gen(
        pps, bits, booth_codes,
        is_neg=['trunc', 'trunc', 'add', 'add'])
    res = partial_product_compress(p_sums, bits)
    # res += four_2_err(p_sums, 7)
    # if b == 0:
    #     res = 0
    return res


def test_booth_mul(mul_func):
    error = Error()
    for i in range(-2**7, 2**7):
        for j in range(-2**7, 2**7):
            exact_mul = i * j
            booth = to_signed(mul_func(i, j, 8), 16)
            print(f'exact : {i} * {j} = {exact_mul},   approx : {booth}')
            err = exact_mul - booth
            error.add(exact_mul, booth)

            if err != 0:
                print(
                    f"""\
    error = {err} {'zero deviate' if exact_mul == 0 else ''}
""")
    print(f"""\
{error}
""")


if __name__ == '__main__':
    test_booth_mul(booth_mul_approx_trunc)
    is_debug = 1
    # print(bin(64))
    # result = booth_mul_approx(2, 0, 8)
    # print(bin(result))
