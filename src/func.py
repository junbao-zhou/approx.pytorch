import torch
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(exp, x):
    return 1 / (1 + exp(-x))


def sigmoid_PLAN(x):
    return \
        (5 <= x) * 1 + (2.375 <= x) * (x < 5) * (x / 32 + 0.84375) + \
        (1 <= x) * (x < 2.375) * (x / 8 + 0.625) + (0 <= x) * (x < 1) * (x / 4 + 0.5) + \
        (x <= -5) * 0 + (-5 < x) * (x <= -2.375) * (1 + x / 32 - 0.84375) + \
        (-2.375 < x) * (x <= -1) * (1 + x / 8 - 0.625) + \
        (-1 < x) * (x < 0) * (1 + x / 4 - 0.5)


def sigmoid_2_mul(x):
    return \
        (4 <= x) * 1 + (0 <= x) * (x < 4) * (-0.03577 * x * x + 0.25908 * x + 0.5038) + \
        (x <= -4) * 0 + (x < 0) * (-4 < x) * \
        (1 + 0.03577 * x * x + 0.25908 * x - 0.5038)


def sigmoid_1_mul(x):
    return \
        (4 <= x) * 1 + (0 <= x) * (x < 4) * (1 - 0.5 * (x / 4 - 1) * (x / 4 - 1)) + \
        (x <= -4) * 0 + (x < 0) * (-4 < x) * (0.5 * (x / 4 + 1) * (x / 4 + 1))


def frac(x):
    return x + torch.abs(x.int())


def sigmoid_alippi(x):
    return \
        (x > 0) * (1 - (0.5 + frac(-x) / 4) / torch.pow(2, torch.abs(x.int()))) + \
        (x <= 0) * ((0.5 + frac(x) / 4) / torch.pow(2, torch.abs(x.int())))


def sigmoid_a_law(x):
    return \
        (x <= -8) * 0 + (x > -8) * (x <= -4) * (0.015625 * x + 0.125) + \
        (x > -4) * (x <= -2) * (0.03125 * x + 0.1875) + \
        (x > -2) * (x <= -1) * (0.125 * x + 0.375) + \
        (x > -1) * (x < 1) * (0.25 * x + 0.5) + \
        (x >= 1) * (x < 2) * (0.125 * x + 0.625) + \
        (x >= 2) * (x < 4) * (0.03125 * x + 0.8125) + \
        (x >= 4) * (x < 8) * (0.015625 * x + 0.875) + \
        (x >= 8) * 1


def tanh(sigmoid, exp, x):
    return 2 * sigmoid(exp, 2 * x) - 1


def calc_a_law():
    A_law_points = [(-8, 0), (-4, 0.0625), (-2, 0.125), (-1, 0.25),
                    (1, 0.75), (2, 0.875), (4, 0.9375), (8, 1)]
    for i in range(len(A_law_points)-1):
        slope = (A_law_points[i][1] - A_law_points[i+1][1]) / \
            (A_law_points[i][0] - A_law_points[i+1][0])
        cut = (0 - A_law_points[i][0]) * slope + A_law_points[i][1]
        print(f'slope = {slope}, cut = {cut}', end='')
        expo = 0
        while True:
            slope *= 2
            expo += 1
            if slope == 1:
                break
        print('  ', expo)

# from fxpmath import Fxp
def sigmoid_alippi_quantized(x):
    x_r = Fxp(x, dtype='S8.8')
    print("x_r")
    x_r.info()
    with open("x.txt", 'w') as f:
        f.write('\n'.join(x_r.bin()))

    x_r_int = Fxp(x_r, dtype='S8.0')
    print("x_r_int")
    x_r_int.info()
    # plt.plot(x, x_r_int)
    with open("x_int.txt", 'w') as f:
        f.write('\n'.join(x_r_int.bin()))

    abs_x_r_int = Fxp((x_r_int < 0) * (-x_r_int) + (x_r_int > 0) * (x_r_int), dtype='U7.0')
    print("abs_x_r_int")
    abs_x_r_int.info()
    with open("abs_x_int.txt", 'w') as f:
        f.write('\n'.join(abs_x_r_int.bin()))
    # plt.plot(x, x_r_int)

    frac = (x_r < 0) * (x_r + abs_x_r_int) + (x_r > 0) * (-x_r + abs_x_r_int)
    frac = Fxp(frac, dtype='S1.8')
    print("frac")
    frac.info()
    plt.plot(x, frac)

    frac = frac >> 2
    print("frac")
    frac.info()
    with open("frac.txt", 'w') as f:
        f.write('\n'.join(frac.bin()))

    middle = (0.5 + frac)
    print("middle")
    middle.info()
    plt.plot(x, middle)

    for i in range(len(middle)):
        middle[i] = middle[i] >> int(abs_x_r_int[i]())
        # middle[i].info()
    print("middle")
    middle.info()
    with open("middle.txt", 'w') as f:
        f.write('\n'.join(middle.bin()))

    y_r = (x_r < 0) * middle + (x_r >= 0) * (1 - middle)
    y_r = Fxp(y_r, dtype='S1.8')
    # y_r = Fxp(sigmoid_alippi(x_r), dtype='S8.8')
    y_r.info()
    with open("y.txt", 'w') as f:
        f.write('\n'.join(y_r.bin()))
    plt.plot(x, y_r)



if __name__ == "__main__":
    # calc_a_law()
    x = np.arange(-10, 10, 2**-8)
    # x = torch.linspace(-6, 6, 1000)
    

    # plt.plot(x, sigmoid(x))
    # plt.plot(x_r, y_r)
    plt.show()
